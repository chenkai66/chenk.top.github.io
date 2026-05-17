---
title: "Differential Geometry (5): The Gauss-Bonnet Theorem — Where Geometry Meets Topology"
date: 2021-11-09 09:00:00
tags:
  - differential-geometry
  - gauss-bonnet
  - topology
  - mathematics
categories: Mathematics
series: differential-geometry
translationKey: "differential-geometry-5-gauss-bonnet"
lang: en
mathjax: true
description: "The Gauss-Bonnet theorem connects total Gaussian curvature to the Euler characteristic — a stunning bridge between local differential geometry and global topology."
disableNunjucks: true
series_order: 5
series_total: 12
---

The Theorema Egregium of the previous chapter showed that Gaussian curvature is intrinsic — bend a surface without stretching and $K$ does not change. The Gauss-Bonnet theorem, which we develop here, says something equally remarkable in a different direction: integrate $K$ over a closed surface and you get a topological invariant. The total curvature of any sphere is $4\pi$. The total curvature of any torus is $0$. The total curvature of a double torus is $-4\pi$. These are facts about *topology*, blind to the specific geometry — bend, twist, or smoosh the surface as you like, the total curvature is the same.

Stated this way, Gauss-Bonnet sounds almost too good to be true. A *local* differential-geometric quantity (curvature, defined by second-order behavior of the metric) integrates to a *global* topological invariant (the Euler characteristic, defined combinatorially from a triangulation). Two completely different worlds — analysis of metrics and combinatorics of cell complexes — meet in the middle. This is one of the most beautiful and influential theorems in mathematics, and it is the conceptual ancestor of an entire industry of "index theorems" connecting analysis and topology, culminating in the Atiyah-Singer index theorem of the 1960s.

This chapter develops Gauss-Bonnet in two stages. First, the *local* version: an integral over a triangle bounded by curves. Second, the *global* version: the integral over a whole closed surface. The first is the engine; the second is the punchline.

---

## The Local Gauss-Bonnet Theorem

Set up: let $T \subset S$ be a region on a surface $S$, bounded by a simple closed piecewise-smooth curve $\partial T$. Suppose $\partial T$ has corners at points $p_1, \ldots, p_n$ where the tangent direction jumps by exterior angles $\theta_i$ (positive if turning left, in the chosen orientation). The smooth pieces of $\partial T$ are themselves curves with geodesic curvature $\kappa_g$.

**Theorem (Local Gauss-Bonnet).** Under these conditions,
$$\iint_T K\,dA + \int_{\partial T}\kappa_g\,ds + \sum_{i=1}^n\theta_i = 2\pi.$$

In words: the total Gaussian curvature inside, plus the total geodesic curvature along the smooth boundary, plus the total exterior angle at corners, equals $2\pi$.

![Local Gauss-Bonnet for a geodesic triangle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_1_local_gb.png)

This is the local version. Various special cases:

**Geodesic triangle.** A triangle whose three sides are geodesics ($\kappa_g \equiv 0$). The line integral vanishes; we get
$$\iint_T K\,dA + \theta_1 + \theta_2 + \theta_3 = 2\pi.$$
The exterior angles are $\theta_i = \pi - A_i$, where $A_i$ is the interior angle. So
$$\iint_T K\,dA = 2\pi - \sum(\pi - A_i) = \sum A_i - \pi.$$
The integral of $K$ over a geodesic triangle equals the interior angle sum minus $\pi$.

In Euclidean geometry, $K = 0$ and the angle sum is $\pi$ exactly. In spherical geometry, $K > 0$ and the angle sum exceeds $\pi$ — by an amount equal to the area times $K$. In hyperbolic geometry, $K < 0$ and the angle sum is less than $\pi$.

![Spherical geodesic triangle whose angle sum exceeds pi](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_2_geodesic_triangle_sphere.png)

**Why this matters.** The local Gauss-Bonnet theorem is the rigorous content of the slogan "curvature equals angle excess per unit area". It is also the engine of the global theorem: integrating around the boundaries of a triangulation, the line integrals telescope and what remains is the global statement.

### Worked example: spherical triangle

Take a triangle on the unit sphere with all three angles equal to $\pi/2$ — a right-angle triangle at the corners. Set the corners at the north pole and at two points on the equator $\pi/2$ apart. Each interior angle is $\pi/2$. So angle sum = $3\pi/2$, and
$$\iint K\,dA = 3\pi/2 - \pi = \pi/2.$$
On the unit sphere, $K = 1$, so the area of this triangle is $\pi/2$. Indeed: the triangle is one-eighth of the sphere ($S^2$ has area $4\pi$, divided by 8 is $\pi/2$). Numerically consistent.

### Worked example: hyperbolic triangle

Take the hyperbolic plane (constant $K = -1$) and a triangle with all three angles equal to zero — an "ideal triangle" with vertices at infinity (in the upper-half-plane model, three points on the real line). Angle sum = $0$. So
$$\iint K\,dA = 0 - \pi = -\pi.$$
With $K = -1$, the area is $\pi$. Reassuringly finite: ideal triangles in the hyperbolic plane have area $\pi$, even though they are unbounded.

This is the hyperbolic-geometric reason that ideal triangles "tile the hyperbolic plane" in a particular way (you can fit infinitely many of them, all of equal area $\pi$, into the disk model — yes, infinitely many, even though the total area of the disk is also $\infty$).

---

## Triangulations and the Euler Characteristic

To pass from local to global, we need a way to chop up a surface into triangles. This is the realm of *triangulations*.

**Definition.** A *triangulation* of a closed surface $S$ is a decomposition of $S$ into a finite collection of triangles such that any two triangles either share no points, share a single vertex, or share an entire edge.

Every closed surface admits a triangulation (Radó, 1925). A given surface admits many triangulations.

**Definition.** Given a triangulation, let $V, E, F$ be the number of vertices, edges, and faces. The *Euler characteristic* is
$$\chi(S) = V - E + F.$$

![Triangulating a closed surface for Gauss-Bonnet](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_5_triangulation.png)

**Theorem.** $\chi(S)$ is independent of the triangulation. It is a *topological invariant* — depends only on the topology of $S$, not on the geometry, not on the chosen triangulation.

Some values:
- Sphere: $\chi(S^2) = 2$.
- Torus: $\chi(T^2) = 0$.
- Double torus (genus 2): $\chi = -2$.
- Genus $g$ surface: $\chi = 2 - 2g$.
- Real projective plane $\mathbb{RP}^2$: $\chi = 1$.
- Klein bottle: $\chi = 0$.

For any closed orientable surface, $\chi = 2 - 2g$ where $g$ is the *genus* (number of "handles"). For non-orientable closed surfaces, $\chi = 2 - k$ where $k$ is the number of "crosscaps".

**Quick verification for the sphere.** Take a tetrahedron, drawn on the sphere (project from interior point). It has 4 vertices, 6 edges, 4 faces. $\chi = 4 - 6 + 4 = 2$. Now take a cube on the sphere: 8 vertices, 12 edges, 6 faces. $\chi = 8 - 12 + 6 = 2$. Same. Same as any other triangulation of the sphere. The number is invariant.

**Quick verification for the torus.** A standard triangulation of the torus has 9 vertices, 27 edges, 18 faces (you can draw it from the unit square with opposite sides identified, divided into a 3-by-3 grid of squares each subdivided into two triangles). $\chi = 9 - 27 + 18 = 0$. As expected.

![Euler characteristic chi for sphere = 2, torus = 0, double torus = -2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_4_euler_char.png)

---

## The Global Gauss-Bonnet Theorem

**Theorem (Global Gauss-Bonnet).** For any closed (compact, no boundary) orientable surface $S$,
$$\iint_S K\,dA = 2\pi\chi(S).$$

That's it. The total Gaussian curvature is $2\pi$ times the Euler characteristic.

![Global Gauss-Bonnet: integral of K equals 2pi times Euler characteristic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_3_global_gb.png)

**Proof sketch.** Triangulate $S$ with a triangulation $\mathcal{T}$ where each triangle has geodesic edges (this can always be done by refinement). Apply local Gauss-Bonnet to each triangle. The interior integrals of $K$ sum to $\iint_S K\,dA$. The line integrals of $\kappa_g$ sum to zero, because each edge is shared by two triangles and the contributions cancel (one orientation each way). The exterior-angle sums combine, with each interior angle around a vertex contributing.

For each interior triangle $T_i$:
$$\iint_{T_i}K\,dA + \sum_j(\pi - A_{ij}) = 2\pi$$
(where $A_{ij}$ are the interior angles of $T_i$). Summing over all triangles:
$$\iint_S K\,dA + \sum_i\sum_j(\pi - A_{ij}) = 2\pi F.$$
The double sum is $3\pi F - \sum_{\text{angles}}A$. At each vertex, the interior angles sum to $2\pi$ (since they fill the tangent plane). So $\sum_{\text{angles}}A = 2\pi V$. Then
$$\iint_S K\,dA + 3\pi F - 2\pi V = 2\pi F$$
$$\iint_S K\,dA = 2\pi V - \pi F = 2\pi V - \pi F.$$
Wait, that does not match. Let me redo: $3\pi F - 2\pi V$, plus integral, equals $2\pi F$. So integral $= 2\pi F - 3\pi F + 2\pi V = 2\pi V - \pi F$. We need to relate this to $\chi = V - E + F$.

A triangulation of a surface satisfies $3F = 2E$ (each face has 3 edges, each edge bounds 2 faces). So $E = 3F/2$, and $\chi = V - 3F/2 + F = V - F/2$, giving $F = 2(V - \chi)$. Substitute:
$$\iint_S K\,dA = 2\pi V - \pi\cdot 2(V - \chi) = 2\pi V - 2\pi V + 2\pi\chi = 2\pi\chi.$$
$\square$

That telescoping is the entire proof: the boundary contributions of geodesic curvature cancel because the triangulation has interior edges shared by two triangles in opposite orientations; the angle sums collect into a count of vertices; the count of vertices, edges, faces is exactly the Euler characteristic.

**Consequences.**

*Sphere.* $\iint K\,dA = 4\pi$. For the unit sphere, $K = 1$, area $= 4\pi$, so $\iint K\,dA = 4\pi$. Consistent. For an ellipsoid, $K$ varies, but the total is still $4\pi$. For *any* topological sphere with *any* metric, the total curvature is $4\pi$.

*Torus.* $\iint K\,dA = 0$. The standard donut torus in $\mathbb{R}^3$ has $K > 0$ on the outer rim, $K < 0$ on the inner rim, $K = 0$ on the top and bottom curves, and the integrals balance to zero. We computed this directly in chapter 3. Now we see it is forced by topology: any topological torus, in any metric, has total curvature zero.

*Genus-$g$ surface.* $\iint K\,dA = 4\pi(1 - g)$. So a double torus has total curvature $-4\pi$. A pretzel-shape (genus 3) has total curvature $-8\pi$. The more "handles", the more negative the total curvature.

This forces some metric facts. For example, no metric on the torus can have $K > 0$ everywhere (the integral would be positive, but it must equal zero). No metric on the sphere can have $K < 0$ everywhere (the integral would be negative, but it must equal $4\pi$). And so on.

---

## Topology Constrains Geometry

This is the headline of Gauss-Bonnet, written large. The topology of a surface — its Euler characteristic, captured by counting vertices, edges, faces of any triangulation — *constrains* what its geometry can be. Curvature can be redistributed in any way, but the total is fixed.

![Topology constrains geometry: chi controls total curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_6_topology_constrains.png)

A few striking corollaries.

**Hairy ball theorem.** Equivalent to: no continuous nowhere-vanishing tangent vector field on $S^2$. Proof idea (one of several): if such a field existed, we could "comb" the sphere flat, which would contradict positive total curvature. Better: relate the existence of such a field to $\chi$, which for $S^2$ is $2 \neq 0$. The full statement is the *Poincaré-Hopf theorem*: the sum of indices of a vector field's zeros equals $\chi(S)$. We will see this in chapter 7.

**Brouwer fixed point on $S^2$.** Every continuous map $f: S^2\to S^2$ has either a fixed point or maps a point to its antipode. (Restricted to maps near the identity, certainly.) The exact statement uses degree theory, but its spirit is "topology forces the existence of certain points".

**Hilbert's theorem.** No complete embedded surface in $\mathbb{R}^3$ has $K \equiv -1$ everywhere. The pseudosphere is incomplete; a hypothetical complete surface with $K = -1$ would have to have infinite total curvature ($\int K\,dA = -\infty$ for an unbounded region with negative density), but it cannot embed in $\mathbb{R}^3$ — the proof is non-trivial and uses the geodesic equation directly. The connection to Gauss-Bonnet: total curvature constrains the geometry, and there is no consistent way to fit infinite negative curvature into a complete embedded surface in $\mathbb{R}^3$.

**Cohn-Vossen rigidity.** Any closed convex surface in $\mathbb{R}^3$ is rigid: if you isometrically deform it, the deformation is a rigid motion of $\mathbb{R}^3$. This is partly a consequence of the integral constraint and partly extrinsic geometry.

---

## Worked Example: Triangulating a Sphere with All-Geodesic Triangles

Take the unit sphere, place an octahedron's vertices at $(\pm 1, 0, 0)$, $(0, \pm 1, 0)$, $(0, 0, \pm 1)$. Connect each pair of adjacent vertices by a great-circle arc. The result is a triangulation of $S^2$ with 6 vertices, 12 edges, 8 faces (each a spherical "right-angle" triangle with three angles of $\pi/2$). 

$\chi = 6 - 12 + 8 = 2$. By Gauss-Bonnet, $\iint K\,dA = 4\pi$. Each spherical triangle has area $\pi/2$ (angle excess $\pi/2$ over $\pi = $ area times $K = $ area times 1; so area = $\pi/2$). Eight triangles times $\pi/2$ = $4\pi$. The sphere has total area $4\pi$. Consistent.

Local Gauss-Bonnet on one of these triangles: $\iint K\,dA = \pi/2$, $\int\kappa_g\,ds = 0$ (geodesic edges), $\sum\theta_i = \sum(\pi - \pi/2) = 3\pi/2$. Total: $\pi/2 + 0 + 3\pi/2 = 2\pi$. Consistent with the local theorem.

---

## Worked Example: Total Curvature of an Ellipsoid

Take the ellipsoid $x^2/a^2 + y^2/b^2 + z^2/c^2 = 1$ with $a > b > c > 0$. Computing the integral $\int K\,dA$ directly is unpleasant — $K$ is a complicated function. But Gauss-Bonnet tells us, without any computation, that the answer is $4\pi$.

The Gauss map of the ellipsoid covers $S^2$ exactly once (the ellipsoid is convex, so this is a degree-1 map). The Jacobian of the Gauss map is $K$ (in the "area density" sense), so $\int K\,dA$ equals the area of $S^2$, which is $4\pi$. Same answer, different proof.

---

## Surfaces with Boundary

If $S$ has a boundary $\partial S$ — say, a hemisphere — there is a boundary version of Gauss-Bonnet:
$$\iint_S K\,dA + \int_{\partial S}\kappa_g\,ds + \sum\theta_i = 2\pi\chi(S),$$
where $\chi$ is computed using a triangulation that respects the boundary.

For a disk (topological disk), $\chi = 1$, so the right side is $2\pi$. For an annulus, $\chi = 0$, so $0$. For a Möbius strip, $\chi = 0$, so $0$ (although orientation issues require care).

The local Gauss-Bonnet theorem we stated is the case of $T = $ topological disk, $\chi = 1$, giving $2\pi$.

---

## Examples of Topological Constraints

A few more illustrations of the Gauss-Bonnet philosophy.

![Worked Gauss-Bonnet examples on classical surfaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/05-gauss-bonnet/dg_v2_05_7_examples.png)

**Sphere minus a disk.** $\chi = 1$. A small disk removed from a sphere leaves a region homeomorphic to a disk (topologically). The integral of $K$ over the remaining surface plus the geodesic-curvature integral over the boundary equals $2\pi$. If the remaining region is "almost the whole sphere" — large region — the curvature integral is close to $4\pi$, and the boundary integral must be close to $-2\pi$. Indeed, the boundary of a small disk traversed in the right orientation has $\int\kappa_g\,ds$ close to $2\pi$ for small disks (the curvature of the boundary is $\approx 1/$ small radius), but with the opposite sign convention for the outer region's boundary, it comes out to $-2\pi$. Gauss-Bonnet balances.

**Cylinder.** $\chi = 0$. A cylinder is a surface with boundary, topologically equivalent to an annulus. $\iint K\,dA = 0$ (the cylinder is intrinsically flat); $\int_{\partial}\kappa_g\,ds = 0$ (the boundary circles are geodesics on the cylinder, which we verified in chapter 4). Total = $0 = 2\pi\chi$. Consistent.

**Pair of pants (sphere with three holes).** $\chi = 2 - 3 = -1$. So $\iint K\,dA + \int_{\partial}\kappa_g = -2\pi$.

Each piece of the apparatus — total curvature, boundary integral, corner angles, Euler characteristic — is doing its part to make the equation balance.

---

## Why This Matters Beyond Surfaces

Gauss-Bonnet is the prototype of an *index theorem*: a deep equality between an analytic quantity (an integral involving curvature, an analytical object) and a topological quantity (the Euler characteristic, a combinatorial invariant). The pattern repeats:

- **Riemann-Roch theorem.** For complex curves: degree + 1 = Euler characteristic of holomorphic line bundle, in some sense. The 1-dimensional ancestor of Gauss-Bonnet in algebraic geometry.
- **Hirzebruch-Riemann-Roch.** Generalizes Riemann-Roch to higher-dimensional complex manifolds; the index of an elliptic operator equals an integral of characteristic classes.
- **Atiyah-Singer index theorem.** Generalizes to elliptic operators on arbitrary smooth manifolds; the analytical index equals the topological index. Gauss-Bonnet is a special case.
- **Chern-Gauss-Bonnet for higher dimensions.** $\int_M \mathrm{Pf}(R/2\pi) = \chi(M)$ for even-dimensional Riemannian $M$, where $\mathrm{Pf}$ is a Pfaffian and $R$ is the curvature 2-form. Gauss-Bonnet is the 2D case.

The whole structure — local-to-global integration of curvature giving topological invariants — generalizes far beyond surfaces. The 2D Gauss-Bonnet we proved is a window into a much larger landscape.

---

## Limitations and Subtleties

A few honest caveats.

**Compactness.** Gauss-Bonnet as stated requires compactness. For non-compact surfaces, the integral may diverge or be undefined, and there are subtler "Gauss-Bonnet with boundary at infinity" statements (using cohomology with compact support).

**Orientability.** I have stated the theorem for orientable surfaces. The non-orientable version exists (Möbius strip, projective plane, Klein bottle), with the integral being half the Euler characteristic times $2\pi$, or similar — the precise statement requires care with the "double cover".

**Higher dimensions.** As mentioned, Gauss-Bonnet generalizes via Chern's formula. In odd dimensions the result is trivial ($\chi = 0$ for any odd-dimensional closed manifold). In even dimensions, it is highly non-trivial.

**Smoothness.** I have implicitly assumed everything is smooth. Gauss-Bonnet has versions for piecewise-smooth and Lipschitz surfaces, but the statements need adjustment.

**Ricci flow and the proof of Poincaré.** Hamilton's Ricci flow on surfaces (1980s) showed that any metric on a compact surface flows under the Ricci flow to a metric of constant Gaussian curvature, and the value of that constant is determined by Gauss-Bonnet ($K = 2\pi\chi/\mathrm{Area}$). This is the "easy" 2D case of the program that Perelman completed in the 3D Poincaré conjecture (2003). The connection between Gauss-Bonnet and Ricci flow runs deep.

---

## Summary

We have proved one of the most beautiful theorems in mathematics:
$$\iint_S K\,dA = 2\pi\chi(S),$$
along with its local version
$$\iint_T K\,dA + \int_{\partial T}\kappa_g\,ds + \sum\theta_i = 2\pi\chi(T).$$

The integral of $K$ over a closed surface — a *local* differential-geometric quantity, defined by second derivatives of the metric — equals $2\pi$ times the Euler characteristic — a *global* topological invariant, defined combinatorially. The two conceptual worlds — analysis of metrics and topology of triangulations — are bound together by this equation.

Its consequences are wide-ranging: topology constrains geometry (no positive-curvature metric on a torus, no negative-curvature metric on a sphere); the hairy ball theorem; the Poincaré-Hopf index theorem; the basis of cobordism, characteristic classes, and the index theorems of the 20th century. All of this descends, more or less, from the formula above.

This concludes the classical theory of surfaces. The next chapter, *Smooth Manifolds*, abstracts away the embedding into $\mathbb{R}^3$ and develops the framework of manifolds, charts, atlases, and tangent spaces in pure intrinsic form. Everything we have learned about surfaces — first fundamental form, geodesics, curvature, Gauss-Bonnet — has a direct generalization to higher-dimensional manifolds with Riemannian metrics. We will spend chapters 6 through 12 building that generalization, culminating in the theory of connections and the Riemann curvature tensor.

For now, savor the result. The total curvature of any sphere is $4\pi$. The total curvature of any torus is $0$. Bend, twist, deform the metric — these numbers do not change. Geometry knows about topology, and topology knows about geometry. They are two views of the same object.

---

## Appendix: A Direct Proof for the Sphere

For the unit sphere, here is a direct verification not using triangulations. The Gauss map is $N: S^2\to S^2$, the identity. The Jacobian of $N$ is exactly $K$ (a fact we used in chapter 3). So
$$\iint_{S^2}K\,dA = \iint_{S^2}|\det dN|\,dA = \mathrm{Area}(N(S^2)) = \mathrm{Area}(S^2) = 4\pi.$$
The Gauss map covers $S^2$ exactly once, so its image area equals the unit sphere's area, which is $4\pi$. By Gauss-Bonnet, $\chi = 2$. Done.

For an ellipsoid, the same argument works: the Gauss map is still a degree-1 map onto $S^2$ (because the ellipsoid is convex), so the integral of $|K|$ is again $4\pi$. Since $K > 0$ everywhere (convexity), $\int K = \int|K| = 4\pi$. Same answer.

For a torus, the Gauss map is more interesting: it covers the sphere with degree zero (every regular value has equal numbers of positive and negative orientations in the preimage). The signed integral is therefore zero. Gauss-Bonnet again recovers $\chi = 0$.

This "Gauss map degree" formulation gives an alternative perspective: the total curvature is $2\pi$ times the degree of the Gauss map, and the degree equals half the Euler characteristic. (For surfaces in $\mathbb{R}^3$. The general Chern-Gauss-Bonnet theorem packages this without the embedding.)

---

## Appendix: The Gauss-Bonnet Theorem and the Polyhedral Case

For polyhedra (convex polyhedra in $\mathbb{R}^3$), Gauss-Bonnet specializes to a classical theorem.

The total Gaussian curvature of a smooth surface, as we have seen, is concentrated where curvature is non-zero. For a polyhedron — which is "flat" everywhere except at edges and vertices — curvature is concentrated at the *vertices*. Precisely: at each vertex $v$, the *angle defect* is $2\pi - \sum A_i$, where $A_i$ are the face angles meeting at $v$. This angle defect is the analog of $\int_{\text{small region around }v}K\,dA$ for the smooth case.

**Descartes's theorem (1630s).** For any convex polyhedron,
$$\sum_{\text{vertices}}(2\pi - \sum A_i) = 4\pi.$$

This is Gauss-Bonnet for the topological sphere ($\chi = 2$), specialized to polyhedra. The angle-defect at each vertex sums to $4\pi$, regardless of the polyhedron's shape.

**Cube.** Eight vertices, three squares meeting at each. Angle defect at each vertex: $2\pi - 3\cdot\pi/2 = \pi/2$. Eight times $\pi/2 = 4\pi$. Verified.

**Tetrahedron.** Four vertices, three triangles meeting at each. Angle defect: $2\pi - 3\cdot\pi/3 = \pi$. Four times $\pi = 4\pi$. Verified.

**Octahedron.** Six vertices, four triangles meeting at each. Angle defect: $2\pi - 4\cdot\pi/3 = 2\pi/3$. Six times $2\pi/3 = 4\pi$. Verified.

This gives a hands-on, paper-and-tape way to verify Gauss-Bonnet without doing any calculus. Take a paper polyhedron, measure the angles at each vertex, sum the defects. You always get $4\pi$ for a topological sphere, $0$ for a topological torus, and so on.

Descartes's theorem predates Gauss-Bonnet by 200 years. It is a surprising historical fact that the polyhedral case was known long before the smooth case, and that Gauss's proof did not specifically reference polyhedra (he proved the smooth case directly). The realization that they are the same theorem is essentially Bonnet's contribution from 1848.

---

## Appendix: Hopf-Rinow and Completeness

A useful technical theorem governs when geodesics can be extended.

**Hopf-Rinow theorem.** For a connected Riemannian manifold $(M, g)$, the following are equivalent:
1. $M$ is complete as a metric space (every Cauchy sequence converges).
2. Every geodesic can be extended for all time (geodesic completeness).
3. Closed bounded subsets of $M$ are compact.

Moreover, if any of these holds, every two points of $M$ can be joined by a length-minimizing geodesic.

The relevance to Gauss-Bonnet: the theorem applies to a closed (compact) surface, so geodesics extend forever and length-minimizers exist. Triangulations with geodesic edges therefore exist, and the proof of global Gauss-Bonnet goes through.

For non-compact incomplete surfaces (e.g. the pseudosphere), some geodesics escape to "infinity" in finite time, and Gauss-Bonnet's compactness assumption is essential.

---

## Appendix: A Proof of the Gauss-Bonnet Theorem via Differential Forms

The proof I gave (telescoping local Gauss-Bonnet over a triangulation) is the historical proof, attributable to Gauss and Bonnet. There is a more modern proof using differential forms, which generalizes more readily.

The intrinsic statement is: $K\,dA$ is the curvature 2-form of the Levi-Civita connection on the tangent bundle of $S$. This 2-form, integrated over $S$, gives the "Euler class" of the tangent bundle, which equals the Euler characteristic.

This proof uses the formalism of vector bundles, connections, and characteristic classes. It is shorter once the formalism is in place, and it generalizes immediately to the Chern-Gauss-Bonnet theorem in higher dimensions:
$$\int_M e(TM) = \chi(M),$$
where $e(TM)$ is the Euler class of the tangent bundle, expressed via the curvature of any chosen connection.

We will not develop this formalism in detail until later chapters. The point for now is that there are deeper proofs lurking, and the elementary proof we gave is just the most accessible window into the theorem.

---

## Appendix: Surfaces of Constant Curvature

Combining Gauss-Bonnet with the model surfaces of chapter 4, we get a beautiful classification.

**Sphere theorem.** A simply connected, complete, surface with constant $K = 1$ is isometric to the unit sphere.

**Hyperbolic plane theorem.** A simply connected, complete, surface with constant $K = -1$ is isometric to the hyperbolic plane.

**Plane theorem.** A simply connected, complete, surface with constant $K = 0$ is isometric to the Euclidean plane.

For non-simply-connected surfaces, this generalizes: any complete surface of constant curvature is a quotient of one of the three model spaces by a discrete group of isometries acting freely. This is the *uniformization theorem* in the smooth setting (the original is for Riemann surfaces / complex curves, due to Koebe and Poincaré). Every smooth surface admits a metric of constant curvature, and the sign of that curvature is determined by the Euler characteristic via Gauss-Bonnet:

- $\chi > 0$: must have positive constant curvature. Only $S^2$ and $\mathbb{RP}^2$.
- $\chi = 0$: zero constant curvature. Torus or Klein bottle.
- $\chi < 0$: negative constant curvature. Higher genus surfaces.

Gauss-Bonnet forces the sign; uniformization provides the metric. Together they classify all closed surfaces by their geometry.

---

## Appendix: Worked Example, the Octahedron-on-Sphere

Take the octahedral triangulation of $S^2$ once more: 6 vertices on the coordinate axes, 12 great-circle edges, 8 right-angle spherical triangles. Each face is a geodesic triangle with three angles of $\pi/2$.

Verify the local theorem on one face: $\sum A_i = 3\pi/2$. By the geodesic-triangle case of local Gauss-Bonnet, $\iint K\,dA = \sum A_i - \pi = \pi/2$. With $K = 1$ on the unit sphere, area $= \pi/2$. Each face has area $\pi/2$; eight faces times $\pi/2 = 4\pi$. Total area of $S^2 = 4\pi$. Reassuringly consistent.

Verify the global theorem: $\iint_{S^2}K\,dA = 4\pi$. Triangulation Euler char: $V - E + F = 6 - 12 + 8 = 2$. Right-hand side: $2\pi\cdot 2 = 4\pi$. Match.

This is an example where every piece is computable by hand. Worth doing at least once in your geometric life.

---

## Appendix: A Subtle Point About Orientation

Throughout, I have assumed orientable surfaces. Let me briefly mention what happens for non-orientable ones.

The Gauss map and Gaussian curvature both depend on a choice of unit normal. For non-orientable surfaces (Möbius strip, Klein bottle), there is no global continuous choice of unit normal. The orientation flips as you traverse certain loops.

**Workaround.** $K$ as a function of point — i.e. the *unsigned* Gaussian curvature, which equals $\det S$ regardless of normal orientation — is still well-defined. The integral $\iint |K|\,dA$ makes sense.

**Klein bottle.** $\chi = 0$, and indeed the Klein bottle admits a flat metric with $K \equiv 0$ globally. Total curvature is zero either way. Consistent.

**Möbius strip with boundary.** $\chi = 0$ (it deformation-retracts to a circle). The boundary version of Gauss-Bonnet applies, with appropriate orientation conventions on the boundary. $\iint K\,dA + \int_{\partial}\kappa_g\,ds = 0$.

**Real projective plane $\mathbb{RP}^2$.** $\chi = 1$. So $\iint K\,dA = 2\pi$. Indeed, $\mathbb{RP}^2$ admits a constant positive curvature metric — namely, a quotient of the unit sphere $S^2$ by the antipodal map. In that metric, $K = 1$, area $= 2\pi$ (half the sphere's area), total $= 2\pi$. Matches Gauss-Bonnet.

The non-orientable case is a routine modification, mostly involving keeping track of signs. The conceptual content is the same: total curvature equals $2\pi$ times Euler characteristic.

---

## Recap

This article was about a single equation:
$$\iint_S K\,dA = 2\pi\chi(S).$$

Everything else was setup or consequence.

- *Setup:* triangulations, Euler characteristic, the local Gauss-Bonnet theorem (the version with corners and a non-geodesic boundary), Christoffel symbols and geodesic curvature inherited from chapter 4.
- *Consequence:* topology constrains geometry; the hairy ball theorem; the Descartes angle-defect formula for polyhedra; the link to Ricci flow; the foundation of the Atiyah-Singer index theorem.

The proof by triangulation-and-telescoping is the clearest one to see. The proof via differential forms / Euler class is the one that scales up to higher dimensions.

The theorem itself is one of the founding insights of modern mathematics: that "differential geometry" and "topology" are not separate subjects but two views of the same thing. Curvature, an analytic local quantity, has a global topological meaning. After Gauss-Bonnet, mathematics could no longer pretend that geometry and topology lived in separate buildings. The 20th century — Hodge theory, characteristic classes, K-theory, the Atiyah-Singer theorem, Donaldson theory, Seiberg-Witten theory — is in some sense a vast elaboration of the basic insight that local geometric data (curvature, fields, operators) integrates to global topological invariants.

Now, with the classical theory of surfaces complete, we shift gears. Chapter 6 introduces the abstract framework of *smooth manifolds*: spaces that locally look like $\mathbb{R}^n$ but need not embed in any ambient space. This is where the modern subject begins.

A final reflection on the depth of Gauss-Bonnet. There are very few theorems in mathematics that genuinely connect two seemingly unrelated worlds. Gauss-Bonnet is one of them. The integral $\int K\,dA$ is the kind of thing a student would compute in a calculus class — a smooth function over a smooth surface, integrated by the usual machinery of multivariable calculus. The Euler characteristic $V - E + F$ is the kind of thing a student would compute in a topology class — a combinatorial count over a triangulation, justified by the algebraic topology of cell complexes. The two computations have nothing in common: one is analysis, the other is combinatorics. And yet they are equal, with a simple multiplicative constant.

When you first encounter this theorem, the response should be incredulity. Why on earth should an integral of a smooth quantity equal a combinatorial count? The proof, telescoping local Gauss-Bonnet over a triangulation, makes the equality plausible. But the meaning of the equality — that any deformation of the metric leaves the total integral fixed, because it can be redistributed but not changed — runs deeper than any one proof. It is a hint that there is some unifying structure linking metrics to topology that we have not yet fully named.

That structure has names — *Euler class*, *Pontryagin classes*, *Chern classes*, characteristic classes of vector bundles in general — and was developed in the 1930s through 1950s by Whitney, Stiefel, Pontryagin, Chern, and others. It culminates in Atiyah-Singer's index theorem of 1963, which subsumes Gauss-Bonnet, Riemann-Roch, and the Hirzebruch signature theorem as special cases. All of these are local-to-global theorems: an integral of curvature-like quantities equals a topological invariant.

This is the deepest reason to study Gauss-Bonnet carefully even though it concerns "only" 2D surfaces. It is the prototype of the general theory, the cleanest example, the one where every step can be carried out by hand. Master it, and the more general theorems are easier to swallow when they arrive. Skip it, and the abstract index theorems will feel like magic.

So: study the surface. Compute on the sphere, the torus, the saddle. Verify Gauss-Bonnet by hand. The investment pays off.

A small computational addendum for readers who like numerical reassurance. For the unit sphere, $K = 1$, area $= 4\pi$, total $= 4\pi$. For a sphere of radius $r$, $K = 1/r^2$, area $= 4\pi r^2$, total $= 4\pi$. Same. For an ellipsoid with $a = 3, b = 2, c = 1$, the integral cannot be done in closed form, but Gauss-Bonnet promises $4\pi$. A numerical computation (Monte Carlo or careful surface integration) confirms this to whatever precision you bother with — try it as a sanity check that your code is correct. Topology is unforgiving: $4\pi$ exactly, no rounding error, no choice of metric. The integral always lands on $4\pi$.

For a torus with $R = 2, r = 1$ (a "fat donut"), the formula $K = \cos v / (r(R + r\cos v))$ from chapter 3 gives a function ranging from positive on the outer rim to negative on the inner rim. A numerical integration over $u, v\in[0, 2\pi)\times[0, 2\pi)$ with the area element $r(R + r\cos v)$ gives exactly zero, regardless of the values of $R, r$ (provided $R > r$). Topology again: the torus has $\chi = 0$, total curvature is zero. The donut radius and tube radius do not change the topology, so they do not change the total curvature.

These two examples — sphere always $4\pi$, torus always $0$ — are good ones to keep in your back pocket. They are the cleanest illustrations of "topology constrains geometry", and they are the gateway to the rest of differential geometry's relationship with topology.

---

*This is Part 5 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 4 — Intrinsic Geometry](/en/differential-geometry/04-intrinsic-geometry/)*

*Next: [Part 6 — Smooth Manifolds](/en/differential-geometry/06-smooth-manifolds/)*
