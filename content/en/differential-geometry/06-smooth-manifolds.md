---
title: "Smooth Manifolds: Geometry Beyond Embedded Surfaces"
date: 2021-11-11 09:00:00
tags:
  - differential-geometry
  - manifolds
  - topology
  - mathematics
categories: Mathematics
series: differential-geometry
translationKey: "differential-geometry-6-smooth-manifolds"
lang: en
mathjax: true
description: "Manifolds free geometry from ambient space — charts, atlases, and smooth structure let us do calculus on spaces that don't live in R^n."
disableNunjucks: true
series_order: 6
series_total: 12
---

The first five chapters of this series lived inside $\mathbb{R}^3$. We had curves and surfaces, parametrized explicitly, with all the geometric data — first and second fundamental forms, principal curvatures, Christoffel symbols, the Theorema Egregium, Gauss-Bonnet — built up from coordinates we could write down. The Theorema Egregium revealed that the intrinsic story can be told without reference to the embedding. But "without reference to the embedding" still meant "the embedding exists; we just choose not to use it". This chapter takes the next step: we cut the surface loose from $\mathbb{R}^3$ entirely. Smooth manifolds are spaces that locally look like $\mathbb{R}^n$ but need not be subsets of any larger space. They are the natural setting for general relativity, gauge theory, the topology of moduli spaces, and most of modern geometry. The price is some upfront axiomatics; the payoff is a framework that scales to arbitrary dimensions and arbitrary topologies.

Once you accept the abstract framework, everything we have done previously becomes a special case. Curves are 1-manifolds. Surfaces are 2-manifolds embedded in $\mathbb{R}^3$. The intrinsic apparatus — metric, Christoffel symbols, geodesics, curvature — generalizes seamlessly. The extrinsic apparatus does not, because there is no "outside" to compare to.

This chapter develops the language: charts, atlases, smooth structure, smooth maps, tangent spaces, the differential. From here, the rest of the series builds: vector fields and flows (chapter 7), differential forms (chapter 8), integration and Stokes' theorem (chapter 9), Riemannian metrics (chapter 10), curvature on manifolds (chapter 11), and bundles and physics (chapter 12).

---

## What is a Manifold?

The intuition: a manifold is a space that, locally, looks like $\mathbb{R}^n$. Globally, it can have non-trivial topology — it can wrap around itself, have handles, be compact or non-compact. The "looking like $\mathbb{R}^n$" condition is captured by the existence of *coordinate charts*.

**Definition (Topological manifold).** A *topological $n$-manifold* is a topological space $M$ such that:
1. $M$ is Hausdorff (any two distinct points have disjoint neighborhoods);
2. $M$ is second-countable (has a countable basis for its topology);
3. for every $p\in M$, there is an open neighborhood $U\ni p$ and a homeomorphism $\varphi: U\to V$ where $V\subseteq\mathbb{R}^n$ is an open set.

The pair $(U, \varphi)$ is a *coordinate chart*. Hausdorff and second-countable are technical conditions excluding pathologies (line with two origins; long line); locally Euclidean is the geometric content.

**Definition (Smooth manifold).** A *smooth atlas* on a topological manifold $M$ is a collection $\{(U_\alpha, \varphi_\alpha)\}$ of charts whose domains cover $M$ and whose *transition maps*
$$\varphi_\beta\circ\varphi_\alpha^{-1}: \varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta)$$
are smooth (where defined). A *smooth structure* on $M$ is a maximal smooth atlas. A *smooth manifold* is a topological manifold equipped with a smooth structure.

![Charts and atlas making a topological space into a smooth manifold](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_1_chart_atlas.png)

The transition maps are diffeomorphisms of open subsets of $\mathbb{R}^n$ (since their inverses are also smooth). They tell us how to consistently move between different coordinate descriptions of the same region.

![Transition map between two overlapping charts](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_2_transition.png)

**Why this matters.** This definition packages exactly what is needed to do calculus on $M$: locally, every point has $\mathbb{R}^n$-coordinates, and these coordinates are smoothly compatible with each other. We can define smooth functions $f: M\to\mathbb{R}$ as those whose composition $f\circ\varphi^{-1}: V\to\mathbb{R}$ is smooth for every chart. The axioms guarantee that this notion of smoothness is well-defined (independent of which compatible chart we use).

---

## Examples

**$\mathbb{R}^n$ itself.** A single chart $\varphi = \mathrm{id}$ gives a smooth structure. The trivial example.

**Open subsets of $\mathbb{R}^n$.** Same as above; inherit the smooth structure. Examples include $\mathrm{GL}(n,\mathbb{R}) = \{n\times n\text{ invertible matrices}\}$, an open subset of $\mathbb{R}^{n^2}$.

**The sphere $S^n$.** As we saw for $S^2$, no single chart covers the sphere. The standard atlas uses two stereographic projections (from the north pole and from the south pole), giving two charts whose union covers $S^n$.

![Two-chart atlas for the 2-sphere from stereographic projection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_3_sphere_charts.png)

In coordinates, stereographic projection from the north pole $\mathbf{N} = (0, \ldots, 0, 1)$ sends a point $\mathbf{p}\in S^n\setminus\{\mathbf{N}\}$ to the intersection of the line through $\mathbf{N}$ and $\mathbf{p}$ with the equatorial plane:
$$\varphi_\mathbf{N}(\mathbf{p}) = \frac{1}{1 - p_{n+1}}(p_1, \ldots, p_n).$$
Similarly $\varphi_\mathbf{S}$ from the south pole. The transition map between these two charts is the inversion $\mathbf{x}\mapsto \mathbf{x}/|\mathbf{x}|^2$, which is smooth on $\mathbb{R}^n\setminus\{0\}$.

**The torus $T^n$.** $T^n = \mathbb{R}^n/\mathbb{Z}^n$, the quotient of $\mathbb{R}^n$ by the integer lattice. Charts are obtained by taking small enough neighborhoods that the projection $\mathbb{R}^n\to T^n$ is a homeomorphism. Transition maps are translations by integer vectors. The 2-torus needs four charts to cover it (corresponding to the four corners of the fundamental domain).

![An atlas of four charts on the torus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_4_torus_charts.png)

**Real projective space $\mathbb{RP}^n$.** The space of lines through the origin in $\mathbb{R}^{n+1}$. Equivalently, $S^n/\{\pm 1\}$, the sphere with antipodal points identified. A standard atlas has $n+1$ charts $U_i = \{[x_0: \ldots: x_n] : x_i\neq 0\}$ with coordinates $\varphi_i([x]) = (x_0/x_i, \ldots, \hat{x_i/x_i}, \ldots, x_n/x_i)\in\mathbb{R}^n$.

**Lie groups.** $\mathrm{GL}(n,\mathbb{R})$ is an open subset of $\mathbb{R}^{n^2}$, hence a manifold. $O(n)$, $SO(n)$, $U(n)$ are smooth manifolds defined by polynomial equations (level sets of smooth maps; see below). Lie groups carry both smooth and group structure, compatibly. They include rotation groups, unitary groups, symplectic groups — the "infinitesimal symmetries" of geometry and physics.

**Grassmannians, flag manifolds, moduli spaces.** All examples of smooth manifolds (with appropriate care). They will appear in advanced applications.

---

## Smooth Maps

**Definition.** Let $F: M\to N$ be a continuous map between smooth manifolds. $F$ is *smooth* if for every $p\in M$, every chart $(U, \varphi)$ around $p$, and every chart $(V, \psi)$ around $F(p)$ with $F(U)\subseteq V$, the composition
$$\psi\circ F\circ\varphi^{-1}: \varphi(U)\to\psi(V)$$
is smooth as a map of open subsets of Euclidean spaces.

In other words, in any chart-pair, $F$ has a smooth coordinate representation.

**Diffeomorphism.** A smooth map $F: M\to N$ with a smooth inverse. Diffeomorphic manifolds are "the same" in differential geometry.

A natural question: are diffeomorphism classes the same as homeomorphism classes? In low dimensions, yes (up to dimension 3). In dimension 4, *no*: $\mathbb{R}^4$ admits uncountably many distinct smooth structures (Donaldson, 1980s), all homeomorphic to standard $\mathbb{R}^4$ but pairwise non-diffeomorphic. This is a profound and beautiful result in the geometric topology of 4-manifolds, exclusive to dimension 4 — in dimensions $\geq 5$, exotic smooth structures exist but are countable, and in dimension $\leq 3$, smooth and topological classification coincide.

For most of our concerns, we will work with the obvious smooth structures on the standard manifolds and not worry about exotic ones.

---

## The Tangent Space

Tangent vectors on a manifold are the most subtle of the basic concepts. On a surface in $\mathbb{R}^3$, a tangent vector at $p$ is just a vector in $\mathbb{R}^3$ that is tangent to the surface — easy. On an abstract manifold, there is no $\mathbb{R}^3$. We need an intrinsic definition.

There are three equivalent definitions of $T_pM$, the tangent space at $p$. Each has its uses.

### Definition 1: Tangent vectors as derivations

A *derivation at $p$* is a linear map $X: C^\infty(M)\to\mathbb{R}$ satisfying the Leibniz rule
$$X(fg) = X(f)g(p) + f(p)X(g)\quad\text{for all }f, g\in C^\infty(M).$$

The set of all derivations at $p$ is a real vector space, which we define to be $T_pM$.

Why this works: in $\mathbb{R}^n$, the directional derivative $f\mapsto (\partial_v f)(p)$ at a point in direction $v$ is a derivation. Conversely, every derivation at $p$ has this form for some $v$. In fact, the directional derivatives form a complete description of tangent vectors. So generalizing "tangent vector" as "directional derivative operator" gives an intrinsic definition.

![Tangent space T_p M as the space of derivations at p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_5_tangent_space.png)

### Definition 2: Tangent vectors as equivalence classes of curves

A *tangent vector at $p$* is an equivalence class of smooth curves $\gamma: (-\epsilon, \epsilon)\to M$ with $\gamma(0) = p$, where two curves $\gamma_1, \gamma_2$ are equivalent if $(\varphi\circ\gamma_1)'(0) = (\varphi\circ\gamma_2)'(0)$ for every chart $\varphi$ around $p$.

Why this works: equivalent curves have the same "velocity" at $p$, so they represent the same direction. The set of equivalence classes is a vector space (after choosing a chart, the tangent space becomes $\mathbb{R}^n$). The definition does not depend on the chart used to check equivalence.

### Definition 3: Tangent vectors as $n$-tuples in a chart, modulo coordinate change

In a chart $\varphi: U\to V\subseteq\mathbb{R}^n$, $T_pM\cong \mathbb{R}^n$. Different charts give different identifications, related by the Jacobian of the transition map. A tangent vector is then an equivalence class of $n$-tuples $\{(\varphi, v_\varphi) : \varphi\text{ chart at }p, v_\varphi\in\mathbb{R}^n\}$ subject to: if $\tilde\varphi = \psi\circ\varphi$, then $v_{\tilde\varphi} = J_\psi v_\varphi$.

This is a "change of coordinates" definition. It is the most operational and the most opaque.

All three definitions give the same vector space $T_pM$, of dimension $n = \dim M$. In a chart $\varphi = (x^1, \ldots, x^n)$, a basis for $T_pM$ is $\{\partial/\partial x^1\big|_p, \ldots, \partial/\partial x^n\big|_p\}$ — the *coordinate vector fields*. Each $\partial/\partial x^i$ is the derivation that, applied to $f$, gives $\partial(f\circ\varphi^{-1})/\partial x^i$ evaluated at $\varphi(p)$.

The tangent bundle $TM = \bigsqcup_{p\in M}T_pM$ is itself a smooth manifold of dimension $2n$, with charts induced from those of $M$.

---

## The Differential of a Smooth Map

Given a smooth map $F: M\to N$ and a point $p\in M$, there is an induced linear map
$$dF_p: T_pM\to T_{F(p)}N$$
called the *differential* (or pushforward) of $F$ at $p$. In the derivation picture: if $X\in T_pM$ is a derivation, then $dF_p(X)$ is the derivation on $C^\infty(N)$ defined by
$$(dF_p(X))(g) = X(g\circ F)\quad\text{for all }g\in C^\infty(N).$$
In coordinates, $dF_p$ is the Jacobian matrix of the coordinate representation of $F$.

![Smooth map between manifolds and its differential](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_6_smooth_map.png)

**Chain rule.** $d(G\circ F)_p = dG_{F(p)}\circ dF_p$. Linear-algebraically: composition of differentials.

**Diffeomorphism.** $F$ is a diffeomorphism iff $dF_p$ is invertible at every $p$ (and $F$ is bijective).

**Submersion / immersion.** $F$ is a *submersion* if $dF_p$ is surjective at every $p$. A *submersion* gives, locally, a smooth quotient: every fiber $F^{-1}(q)$ is a smooth submanifold of dimension $\dim M - \dim N$. $F$ is an *immersion* if $dF_p$ is injective at every $p$. An immersion realizes $M$ locally as a smooth submanifold of $N$ of dimension $\dim M$.

**Embedding.** An injective immersion $F: M\to N$ that is also a homeomorphism onto its image. An embedding realizes $M$ as a smooth submanifold of $N$. Whitney's theorem (1936) says every smooth $n$-manifold can be embedded in $\mathbb{R}^{2n+1}$, and immersed in $\mathbb{R}^{2n}$.

---

## Submanifolds

A *submanifold* of $N$ is a subset $M\subseteq N$ that is itself a manifold, with the smooth structure inherited from $N$. Equivalently: $M$ is the image of an embedding.

**Examples.** $S^n\subset\mathbb{R}^{n+1}$ is a submanifold of dimension $n$. The unit circle $S^1\subset\mathbb{C}^*$ (the multiplicative group of nonzero complex numbers) is a submanifold of dimension 1. The orthogonal group $O(n)\subset\mathrm{GL}(n,\mathbb{R})$ is a submanifold of dimension $n(n-1)/2$.

**Regular value theorem.** If $F: N\to\mathbb{R}^k$ is a smooth map with $0$ a *regular value* (i.e. $dF_p$ is surjective at every $p\in F^{-1}(0)$), then $F^{-1}(0)$ is a smooth submanifold of $N$ of dimension $\dim N - k$.

This is how most submanifolds in practice are constructed: as level sets of regular maps. The sphere is $\{(x_1, \ldots, x_{n+1}) : x_1^2 + \ldots + x_{n+1}^2 = 1\}$, the level set of the regular function $\|\mathbf{x}\|^2 - 1 = 0$. The orthogonal group is $\{A : A A^T = I\}$, a level set of the map $A\mapsto AA^T - I$ (with values in symmetric matrices).

---

## Manifolds with Boundary

A small generalization: a *manifold with boundary* is a Hausdorff second-countable space locally homeomorphic to $\mathbb{R}^n_{\geq 0} = \{x : x_n \geq 0\}$. The *boundary* $\partial M$ is the set of points mapped to $\partial\mathbb{R}^n_{\geq 0} = \{x_n = 0\}$ in some (and hence every) chart. The boundary is itself an $(n-1)$-manifold.

Closed disks, cylinders with capped ends, hemispheres — all are manifolds with boundary. Stokes' theorem (chapter 9) is naturally formulated for compact oriented manifolds with boundary.

---

## Orientability

A smooth manifold is *orientable* if it admits an atlas whose transition maps all have positive Jacobian determinant. Equivalently: there is a continuous nowhere-zero $n$-form on $M$.

**Orientable.** $S^n$, $T^n$, $\mathbb{R}^n$, $\mathbb{CP}^n$. Most "natural" manifolds.

**Non-orientable.** Möbius strip, Klein bottle, $\mathbb{RP}^n$ for $n$ even.

The orientation issue matters for integration: $\int_M\omega$ is well-defined for an oriented manifold and an $n$-form $\omega$, but not for a non-orientable manifold. Stokes' theorem requires orientation.

---

## A Tour of Classical Smooth Manifolds

A few celebrated examples to have in mind throughout the rest of the series.

![Classical smooth manifolds: sphere, torus, projective space, Klein bottle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_v2_06_7_classical_manifolds.png)

**Spheres $S^n$.** Compact, orientable, simply connected for $n\geq 2$. $S^1$ is the circle. $S^2$ is the surface of the ball. $S^3$ is the unit sphere in $\mathbb{R}^4$, the underlying manifold of $SU(2)$ and a fundamental object in 3-manifold theory (Poincaré conjecture).

**Tori $T^n$.** $T^1 = S^1$. $T^2$ is the donut shape (a square with opposite sides identified). $T^n$ is a compact abelian Lie group, isomorphic to $(\mathbb{R}/\mathbb{Z})^n$.

**Real projective spaces $\mathbb{RP}^n$.** Spheres with antipodal points identified. $\mathbb{RP}^1\cong S^1$. $\mathbb{RP}^2$ is non-orientable, compact. $\mathbb{RP}^3\cong SO(3)$.

**Complex projective spaces $\mathbb{CP}^n$.** Lines through the origin in $\mathbb{C}^{n+1}$. $\mathbb{CP}^1\cong S^2$ (the Riemann sphere). $\mathbb{CP}^2$, $\mathbb{CP}^3$ are basic objects of complex algebraic geometry.

**Klein bottle $K$.** A non-orientable closed surface. $\chi(K) = 0$. Cannot be embedded in $\mathbb{R}^3$ (it self-intersects), but embeds in $\mathbb{R}^4$.

**Lie groups.** $GL(n, \mathbb{R})$, $SL(n, \mathbb{R})$, $O(n)$, $SO(n)$, $U(n)$, $SU(n)$, $Sp(2n)$. Classical Lie groups, each a smooth manifold with a compatible group structure.

**Grassmannians $\mathrm{Gr}(k, n)$.** The space of $k$-dimensional subspaces of $\mathbb{R}^n$. Smooth manifold of dimension $k(n-k)$.

**Stiefel manifolds.** $V_k(\mathbb{R}^n) = \{\text{orthonormal $k$-frames in }\mathbb{R}^n\}$. Smooth manifold of dimension $kn - k(k+1)/2$.

These are the workhorses of modern geometry. We will not need their detailed structure for this series, but it is helpful to know they exist.

---

## Lie Groups: A Brief Aside

A *Lie group* is a smooth manifold with a compatible group structure: multiplication $G\times G\to G$ and inversion $G\to G$ are smooth.

Examples: $\mathbb{R}^n$ (additive), $S^1$ (multiplicative), $T^n$, $GL(n)$, $SL(n)$, $O(n)$, $SO(n)$, $U(n)$, $SU(n)$.

The tangent space of a Lie group at the identity is special. It carries a natural bracket operation $[\cdot, \cdot]$ (the Lie bracket) inherited from the group structure, making it a *Lie algebra*. The exponential map $\exp: \mathfrak{g}\to G$ sends Lie algebra elements to group elements and locally parametrizes the group near the identity. For matrix Lie groups, this is literally the matrix exponential $\exp(X) = \sum_{k=0}^\infty X^k/k!$.

The interplay between the global structure of the Lie group and the local (linear) structure of its Lie algebra is one of the most powerful ideas in modern mathematics and physics.

---

## How Many Manifolds Are There?

The classification of smooth manifolds is incredibly rich.

**Dimension 1.** The only connected smooth 1-manifolds are $\mathbb{R}$ and $S^1$. Trivially classified.

**Dimension 2.** The classification theorem for surfaces says every compact connected orientable surface is a sphere with $g$ handles (genus $g$) — i.e. $S^2$, $T^2$, double torus, etc. Adding non-orientability gives $\mathbb{RP}^2$, Klein bottle, etc. The Euler characteristic plus orientability completely classifies them.

**Dimension 3.** The Poincaré conjecture (proved by Perelman in 2003) says the only simply connected compact 3-manifold is $S^3$. The full classification uses Thurston's geometrization theorem: every closed 3-manifold can be cut along spheres and tori into pieces, each of which is *one of eight* model geometries. This is the deepest result of modern 3-manifold topology.

**Dimension 4.** The most mysterious. Gauge theory (Donaldson invariants in the 1980s, Seiberg-Witten invariants in the 1990s) reveals phenomena that have no analog in other dimensions: $\mathbb{R}^4$ admits uncountably many smooth structures (Donaldson, Freedman); the 4-dimensional Poincaré conjecture is open in the smooth category (it is known in the topological category, by Freedman's 1981 theorem).

**Dimensions $\geq 5$.** Surgery theory provides systematic tools (Smale, Wall). The high-dimensional Poincaré conjecture (smooth and topological) is true (Smale 1961, Newman, Stallings).

This is a vast subject — entire textbooks are written on each dimension's manifold theory. For this series, we will work mostly in the abstract framework and focus on geometric structure (metrics, connections, curvature) rather than topological classification.

---

## What's Next

We now have the stage: smooth manifolds, smooth maps, tangent spaces. But geometry requires more structure. We need to talk about vector fields — smooth assignments of tangent vectors — and the flows they generate. The next article develops *vector fields, integral curves, and the Lie bracket*, the machinery that captures how infinitesimal symmetries act on a manifold. This sets the stage for differential forms, integration on manifolds, and eventually the full apparatus of Riemannian geometry in the intrinsic setting.

**Summary of the key ideas.** Let us recapitulate the conceptual progression:

1. A *topological manifold* is a space that is locally homeomorphic to $\mathbb{R}^n$ — it has local coordinates, but no notion of smoothness.
2. A *smooth structure* (a maximal atlas of $C^\infty$-compatible charts) lets us define smooth functions, smooth maps, and do calculus.
3. *Tangent vectors* are derivations — they differentiate functions, and their definition is intrinsic (no ambient space needed).
4. The *differential* $dF_p$ of a smooth map linearizes the map at a point, sending tangent vectors to tangent vectors.
5. The *tangent bundle* $TM$ assembles all tangent spaces into a single manifold of double the dimension.

These five ideas form the foundation on which all of differential geometry rests. Every subsequent construction — vector fields, differential forms, connections, curvature — is built from these building blocks.

---

## Appendix: Working in Charts

Let me ground the formalism with one extended example: $S^2$ with stereographic projection.

Stereographic projection from the north pole $\mathbf{N} = (0, 0, 1)$ sends $\mathbf{p} = (p_1, p_2, p_3)\in S^2\setminus\{\mathbf{N}\}$ to the intersection of the line through $\mathbf{N}$ and $\mathbf{p}$ with the equatorial plane $z = 0$:
$$\varphi_N(p_1, p_2, p_3) = \biggl(\frac{p_1}{1 - p_3}, \frac{p_2}{1 - p_3}\biggr) =: (u, v).$$

Inverse: starting from $(u, v)\in\mathbb{R}^2$, recover $\mathbf{p}\in S^2$ via
$$\mathbf{p} = \biggl(\frac{2u}{1+u^2+v^2}, \frac{2v}{1+u^2+v^2}, \frac{u^2+v^2 - 1}{1+u^2+v^2}\biggr).$$

This is one chart, covering everything except the north pole. The other chart $\varphi_S$ covers everything except the south pole. The transition map (computed by composing one with the inverse of the other) is the inversion $(u, v)\mapsto (u, v)/(u^2+v^2)$, smooth on $\mathbb{R}^2\setminus\{0\}$.

Now consider a smooth function $f: S^2\to\mathbb{R}$ — say, $f(p_1, p_2, p_3) = p_3$, the height. In the chart $\varphi_N$:
$$f\circ\varphi_N^{-1}(u, v) = \frac{u^2+v^2-1}{u^2+v^2+1}.$$
Smooth on all of $\mathbb{R}^2$. Compute the partial derivatives of $f$ in this chart:
$$\partial_u(f\circ\varphi_N^{-1}) = \frac{2u(u^2+v^2+1) - (u^2+v^2-1)\cdot 2u}{(u^2+v^2+1)^2} = \frac{4u}{(u^2+v^2+1)^2}.$$

So at the point $\varphi_N^{-1}(u, v)$ in $S^2$, the directional derivative of $f$ along $\partial/\partial u$ (the coordinate vector field of the $u$-coordinate) is $4u/(u^2+v^2+1)^2$. This is a well-defined geometric object, even though it is computed in coordinates.

To see what this means in $\mathbb{R}^3$: the derivation $\partial/\partial u$ at a point $\mathbf{p}\in S^2$ corresponds, in $\mathbb{R}^3$, to a tangent vector at $\mathbf{p}$ — namely, the velocity of a curve through $\mathbf{p}$ obtained by varying $u$ and holding $v$ fixed. Differentiating $\varphi_N^{-1}$ with respect to $u$ gives this vector explicitly. The directional derivative of $f = p_3$ along this vector is the Euclidean directional derivative — which the chain rule confirms must equal what we computed in chart coordinates.

This is the essential operational content of "doing calculus on a manifold". Pick a chart; compute as in $\mathbb{R}^n$; the answer is geometrically meaningful provided the construction is invariant under change of chart. The smooth structure is exactly what guarantees this.

---

## Appendix: Why the Hausdorff and Second-Countable Conditions

The two technical conditions in the definition of a topological manifold (Hausdorff, second-countable) are not arbitrary; they are exactly what is needed to exclude pathological examples while keeping all the "geometric" examples in.

**Without Hausdorff: the line with two origins.** Take two copies of $\mathbb{R}$, glue them along $\mathbb{R}\setminus\{0\}$ but leave the two zeros distinct. Result: locally Euclidean (every point has an $\mathbb{R}$ neighborhood), but the two zeros cannot be separated by disjoint open sets. Definitely not what we want.

**Without second-countable: the long line.** Take an uncountable well-ordered set and use it as the index for laying down copies of $[0, 1)$. Result: locally Euclidean, Hausdorff, but uncountably "long" — a 1-manifold that does not embed in $\mathbb{R}^n$ for any $n$. Pathological.

The two conditions are exactly enough to exclude these pathologies and ensure that every manifold can be realized as a subset of some $\mathbb{R}^n$ (Whitney embedding theorem). They are technical, but minimal.

---

## Appendix: Submersions, Immersions, Submanifolds

The three flavors of "well-behaved" smooth maps deserve a closer look.

**Immersion.** $F: M\to N$ with $dF_p$ injective at every $p$. Locally, $F$ looks like the inclusion $\mathbb{R}^m\hookrightarrow\mathbb{R}^n$ (with $m = \dim M, n = \dim N, m \leq n$). The image of an immersion can self-intersect and is not always a submanifold.

**Embedding.** An immersion that is also a homeomorphism onto its image (with the subspace topology). The image of an embedding *is* a submanifold.

**Submersion.** $F: M\to N$ with $dF_p$ surjective at every $p$. Locally, $F$ looks like a projection $\mathbb{R}^m\to\mathbb{R}^n$ (with $m \geq n$). Submersions are fibrations: the preimage $F^{-1}(q)$ of any value is a submanifold of dimension $m - n$.

A surjective submersion that is "nice" (locally trivial) is a *fiber bundle*. Examples: trivial bundle $M\times F\to M$, tangent bundle $TM\to M$, sphere bundles, etc. Bundles will be central in chapter 12.

The *implicit function theorem* underlies all of this: given a submersion $F: M\to N$, the level sets are submanifolds, and locally we can choose coordinates "horizontally + vertically" to express $F$ as a projection. Given an immersion, the image looks locally like a flat slice of a product. The local structure of smooth maps is much more rigid than for continuous maps.

---

## Appendix: Sard's Theorem

A foundational result about smooth maps: the set of critical values has measure zero.

**Sard's theorem.** Let $F: M^m\to N^n$ be smooth (with sufficient differentiability — specifically, $F$ should be at least $C^k$ where $k = \max(m - n + 1, 1)$). Then the set $\{F(p) : dF_p$ is not surjective$\}\subseteq N$ has Lebesgue measure zero.

Practical consequence: regular values are *generic*. So the regular value theorem applies to "most" values, in a precise sense. If you pick a value $c\in N$ at random, $F^{-1}(c)$ is almost surely a submanifold.

This theorem is the hidden engine behind much of differential topology. It guarantees that "transverse" intersections are generic, that submanifolds are abundant, that the obstructions to making constructions smooth are usually absent. Without Sard, the theory would be much less robust.

---

## Appendix: The Tangent Bundle

The tangent bundle $TM = \bigsqcup_{p\in M}T_pM$ is itself a smooth manifold of dimension $2n$.

In a chart $(U, \varphi)$ on $M$, with $\varphi = (x^1, \ldots, x^n)$, a chart on $TM\big|_U$ is given by
$$(p, \mathbf{v})\mapsto (x^1(p), \ldots, x^n(p), v^1, \ldots, v^n),$$
where $\mathbf{v} = \sum v^i\partial/\partial x^i\big|_p$. So $T U$ inherits $\mathbb{R}^{2n}$ coordinates from $U$'s $\mathbb{R}^n$ coordinates.

The transition maps on $TM$ are determined by those on $M$ together with the Jacobian of the transition: $(x, v)\mapsto (\tilde x(x), J\tilde x\cdot v)$ where $J\tilde x$ is the Jacobian. Smooth.

The tangent bundle has a natural projection $\pi: TM\to M$ sending $(p, \mathbf{v})\mapsto p$. This is a submersion, and the fibers are exactly the tangent spaces — a vector bundle of rank $n$.

A *vector field* on $M$ is a smooth section of $\pi$: a smooth map $X: M\to TM$ with $\pi\circ X = \mathrm{id}_M$. Equivalently, a smooth assignment of a tangent vector to each point of $M$.

The set of all vector fields is denoted $\mathfrak{X}(M)$, and it is an infinite-dimensional Lie algebra under the Lie bracket (which we will define in chapter 7). It is the algebraic embodiment of "infinitesimal symmetries" of the manifold.

---

## Appendix: Cotangent Bundle and Differential Forms

Dual to the tangent bundle is the cotangent bundle $T^*M = \bigsqcup_p T_p^*M$. A point of $T^*M$ is a pair $(p, \omega)$ where $\omega\in T_p^*M$ is a linear functional on $T_pM$.

A *one-form* on $M$ is a smooth section of the cotangent bundle. In coordinates, a one-form is $\omega = \sum_i\omega_i\,dx^i$, where $dx^i$ is the dual basis to $\partial/\partial x^i$.

Higher-degree differential forms (sections of $\Lambda^k T^*M$) generalize this. The exterior derivative $d$ is an antiderivation $d: \Omega^k(M)\to\Omega^{k+1}(M)$ satisfying $d^2 = 0$. The de Rham cohomology $H^k(M; \mathbb{R}) = \ker d/\mathrm{im}\,d$ is a topological invariant of $M$. We will develop this in chapter 8.

The pair (tangent bundle, cotangent bundle) is the basic data of differential geometry on a smooth manifold. Everything else — connections, curvature, integration — is built from these two bundles and operations on them.

---

## Appendix: Partition of Unity

A technical tool I should mention: every smooth manifold admits *partitions of unity*. A partition of unity subordinate to an open cover $\{U_\alpha\}$ is a collection of smooth functions $\{\rho_\alpha\}$ with $\rho_\alpha\geq 0$, $\mathrm{supp}\,\rho_\alpha\subseteq U_\alpha$, $\sum_\alpha\rho_\alpha = 1$ (with the sum locally finite).

Partitions of unity are how local constructions (Riemannian metrics defined chart-by-chart, integration on non-compact manifolds, etc.) get glued into global ones. They depend on the second-countability axiom: without it, partitions of unity may not exist.

For example: if you want a Riemannian metric on a manifold, define a metric on each chart (the standard $\mathbb{R}^n$ metric, say) and use a partition of unity to glue: $g = \sum_\alpha\rho_\alpha g_\alpha$. The result is a smooth Riemannian metric on all of $M$. So *every* smooth manifold admits a Riemannian metric (hence the metric apparatus of chapter 10 will apply universally).

This is a uniquely smooth-manifold trick. Topological manifolds may fail to have analogous "topological partitions of unity" in the same useful way; algebraic varieties and complex manifolds have related but more restrictive constructions.

---

## A Final Word on Abstraction

The shift from "surfaces in $\mathbb{R}^3$" to "abstract smooth manifolds" is non-trivial. It takes some adjustment to think of geometry without an ambient space. The reward is a framework that can handle:

- Spacetimes in general relativity (4-manifolds with Lorentzian metrics).
- Phase spaces in classical mechanics (symplectic manifolds, often $T^*M$ for some configuration space $M$).
- Moduli spaces (parameter spaces of geometric structures, which can be highly complicated).
- The total spaces of fiber bundles in physics (gauge theory, the Standard Model).
- Group manifolds (Lie groups), where the manifold structure interacts with the group operation.

None of these can be cleanly handled in the framework of "embedded surfaces in $\mathbb{R}^n$". The abstraction is necessary.

If the formalism feels heavy at first, that is normal. The way through is to compute. Pick a manifold (say $S^2$ with stereographic charts, or $T^2$ with the obvious charts), pick a smooth function or smooth map, and write everything out in coordinates. Verify that the answer is independent of which chart you use. After a few exercises of this kind, the formalism becomes second nature, and the conceptual benefits become clear.

The next chapter introduces vector fields and their integral curves, a major step in the operational toolkit. Then differential forms (chapter 8), integration and Stokes (chapter 9), and Riemannian metrics (chapter 10), where we finally regain the apparatus of chapter 4 — geodesics, curvature, parallel transport — but now in the abstract setting. The classical theory of surfaces becomes a special case of the general theory, and we can now do geometry on spaces of any dimension and any topology.

---

## Appendix: A Worked Example with Two Charts on $S^2$

To cement the chart-based formalism, here is one more extended example. Take $S^2\subset\mathbb{R}^3$ with two charts: stereographic from the north pole ($\varphi_N$) and from the south pole ($\varphi_S$). Their inverses are
$$\varphi_N^{-1}(u, v) = \frac{1}{1+u^2+v^2}(2u, 2v, u^2+v^2-1),$$
$$\varphi_S^{-1}(\tilde u, \tilde v) = \frac{1}{1+\tilde u^2+\tilde v^2}(2\tilde u, 2\tilde v, 1 - \tilde u^2 - \tilde v^2).$$

The transition map: take a point $(u, v)\in\mathbb{R}^2\setminus\{0\}$, apply $\varphi_N^{-1}$ to get $\mathbf{p}\in S^2$, then apply $\varphi_S$ to get $(\tilde u, \tilde v)$. After algebra:
$$\tilde u = u/(u^2+v^2),\quad \tilde v = v/(u^2+v^2).$$

This is the "inversion" $\mathbf{x}\mapsto \mathbf{x}/|\mathbf{x}|^2$ in $\mathbb{R}^2$. Smooth on $\mathbb{R}^2\setminus\{0\}$, with smooth inverse (itself the same map). So the two charts are smoothly compatible, and together they make $S^2$ a smooth 2-manifold.

Now consider a smooth function $f: S^2\to\mathbb{R}$ defined globally by $f(p_1, p_2, p_3) = p_1$ (the $x$-coordinate). In the north chart:
$$f\circ\varphi_N^{-1}(u, v) = \frac{2u}{1+u^2+v^2}.$$
Smooth on $\mathbb{R}^2$. In the south chart:
$$f\circ\varphi_S^{-1}(\tilde u, \tilde v) = \frac{2\tilde u}{1+\tilde u^2+\tilde v^2}.$$
Same form (by symmetry). On the overlap, the two coordinate representations are related by the transition map: $(\tilde u, \tilde v) = (u, v)/(u^2+v^2)$. Plugging in:
$$\frac{2\tilde u}{1+\tilde u^2+\tilde v^2} = \frac{2u/(u^2+v^2)}{1 + (u^2+v^2)/(u^2+v^2)^2} = \frac{2u/(u^2+v^2)}{(u^2+v^2+1)/(u^2+v^2)} = \frac{2u}{u^2+v^2+1}.$$
Same as $f\circ\varphi_N^{-1}(u, v)$. The two coordinate representations agree on the overlap, as they must. The function $f$ is well-defined globally.

This calculation is the kind of bookkeeping that justifies the formalism. We can confidently say "$f$ is a smooth function on $S^2$" because in any chart, its representation is smooth, and on overlaps the representations agree under the transition map. The intrinsic notion of "smooth function on $S^2$" is well-defined.

A similar calculation defines smooth maps $S^2\to S^2$, smooth vector fields on $S^2$, the differential of any smooth map, and so on. The general principle: anything you want to define on $S^2$ should have a coordinate representation in each chart that transforms covariantly under change of chart. The transition maps automate this. After enough exercises you stop checking and just trust the framework — but it is good to have done the bookkeeping at least once, on at least one example, to know it works.

---

## Recap

Smooth manifolds free geometry from the constraints of ambient embedding. The basic data is:

- A topological space $M$ that is locally Euclidean.
- A smooth structure (maximal atlas of $C^\infty$-compatible charts).
- For each $p\in M$, a tangent space $T_pM$ — defined intrinsically as derivations, or as equivalence classes of curves, or as $n$-tuples in a chart with appropriate transformation rules.
- For smooth maps $F: M\to N$, a differential $dF_p: T_pM\to T_{F(p)}N$.

From here, we can:
- Define smooth functions and check smoothness chart-by-chart.
- Identify submanifolds via the regular value theorem.
- Build the tangent bundle $TM$, a 2$n$-dimensional manifold with natural projection to $M$.
- Think about orientability, manifolds with boundary, partitions of unity, all routinely.

The classification of manifolds in dimensions 1 and 2 is complete; in dimension 3, settled by Perelman; in dimension 4, mysterious and rich; in dimensions $\geq 5$, accessible via surgery theory. We have a vast world to do geometry in.

The series continues from here without ever returning to embedded surfaces in $\mathbb{R}^3$. Vector fields, differential forms, integration, Riemannian metrics, connections, curvature, bundles — all built on top of the smooth-manifold foundation we have just laid. Welcome to modern differential geometry.

A philosophical closing thought. The decision to abstract away from $\mathbb{R}^3$ was not made lightly in the history of mathematics. The classical theory of surfaces, developed by Gauss, Riemann, Bonnet, and others throughout the 19th century, was concrete and powerful. The abstract manifold concept (Riemann's *Habilitationsvortrag* of 1854, fully formalized by Whitney and Veblen in the 1930s) was driven by the need to handle examples that did not naturally embed: complex curves, fundamental polygons of group quotients, phase spaces of mechanics. By the 1910s, Einstein had pushed the abstraction further with general relativity, where spacetime itself is an abstract 4-manifold, not given by any embedding. The manifold concept is the language in which modern geometry is written.

For a beginner, the cost of the abstraction is real. The intuition of "thing in space" is gone. Instead, you have charts and atlases and transition maps and an abstract topology, and you have to trust the formalism to do the right thing. The good news: the formalism *does* do the right thing. Every concrete computation you might want to do — find a geodesic, compute curvature, integrate a form — reduces, in any chart, to a computation in $\mathbb{R}^n$. The chart is the bridge from the abstract to the concrete. Coordinate-bound computations are still valid, even on abstract manifolds. The abstraction does not eliminate computation; it provides a clean framework for what computations mean.

So while the formalism is upfront, the payoff is permanent: a geometric vocabulary that scales to any dimension, any topology, any application from quantum field theory to data science. That is the gift of the manifold concept, and it is what we will spend the rest of this series exploiting.

One last thing worth noting before we move on. This chapter has been heavy on definitions and light on theorems. That is intentional. The job of chapter 6 is to set up the language; the next several chapters will use it to prove things. The concepts here are meant to be absorbed, not memorized. If you find yourself looking up the definition of "smooth structure" two articles from now, that is fine. The formalism becomes second nature with use. The conceptual map — manifold, chart, atlas, smooth function, tangent space, differential — is what matters. The technical details fall into place in the next several chapters.

---

*This is Part 6 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 5 — The Gauss-Bonnet Theorem](/en/differential-geometry/05-gauss-bonnet/)*

*Next: [Part 7 — Vector Fields and Flows](/en/differential-geometry/07-vector-fields-flows/)*
