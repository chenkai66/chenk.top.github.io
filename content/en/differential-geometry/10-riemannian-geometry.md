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

Up to this point in the series we have studied smooth manifolds with their differentiable structure: charts, tangent vectors, differential forms, exterior calculus, integration. None of this required a notion of *distance*. We could differentiate functions, integrate top-degree forms, decide whether a distribution is integrable — all without ever measuring how long a curve is or what angle two tangent vectors make. The smooth structure is purely topological-with-derivatives. To do *geometry* in the classical sense — to measure, to compare, to speak of curvature, to recover the everyday meanings of "length" and "angle" — we need additional structure.

That structure is a **Riemannian metric**: a smooth choice of inner product on each tangent space. Add this single piece of data to a smooth manifold and the entire classical apparatus comes online. You can compute lengths of curves, angles between tangent vectors, volumes of regions. You get a canonical way to differentiate vector fields along curves (the Levi-Civita connection). You get **geodesics** — the curves that locally minimize length, the natural generalization of straight lines. You get parallel transport along curves, holonomy around loops, and ultimately the curvature tensor that distinguishes a sphere from a torus.

The story is motivated by a concrete question: on a curved surface like the Earth, what is the "straightest" path between two cities? The answer — a great-circle arc — is a geodesic, and constructing it requires the metric. Every statement in general relativity about the motion of planets, the bending of light, or the expansion of the universe rests on the Riemannian (or, more precisely, Lorentzian — same machinery, different signature) framework we develop here. This article is the bridge between abstract differential geometry and the geometry of physical space.

The plan: define the metric, derive its computational consequences (lengths, volumes, the gradient operator). Introduce affine connections in general, then specialize to the unique torsion-free metric-compatible one (Levi-Civita). Define parallel transport, geodesics, and exponential map. Touch on holonomy and the Hopf-Rinow theorem (completeness). Work concrete examples on the sphere and the hyperbolic plane.

A historical orientation, since the names accumulate quickly. Riemann introduced the metric concept in his 1854 habilitation lecture as an extension of Gauss's intrinsic geometry of surfaces to arbitrary dimensions. Christoffel (1869) and Ricci-Curbastro (1880s) developed the index calculus. Levi-Civita (1917) introduced parallel transport, which clarified what Riemann's curvature tensor measured (the failure of parallel transport to be path-independent). Cartan (1920s-30s) recast the entire framework using moving frames and differential forms, the language modern differential geometers prefer. Hopf and Rinow (1931) settled completeness. Berger (1953) classified holonomy. Each of these contributions answered a question raised by the previous generation, and all of it is what we mean today by "Riemannian geometry."

---

## 1. Riemannian Metrics

A **Riemannian metric** on a smooth manifold $M$ is a smooth assignment
$$g: M \to T^*M \otimes T^*M, \qquad p \mapsto g_p$$
where each $g_p$ is a symmetric, positive-definite bilinear form on $T_pM$. In words: at every point you have an inner product on the tangent space, and the inner product varies smoothly.

In coordinates, $g$ is encoded by an $n \times n$ symmetric positive-definite matrix of smooth functions:
$$g = g_{ij}(x)\,dx^i \otimes dx^j, \qquad g_{ij} = g_{ji}, \qquad (g_{ij}) \succ 0.$$

![A Riemannian metric assigning an inner product to each tangent space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_1_riem_metric.png)

The metric immediately produces the standard geometric quantities:
- **Length of a tangent vector:** $\|v\|_g = \sqrt{g_p(v, v)}$.
- **Angle between tangent vectors:** $\cos\theta = \frac{g_p(u, v)}{\|u\|_g\|v\|_g}$.
- **Length of a curve $\gamma: [a, b] \to M$:** $L(\gamma) = \int_a^b \|\dot\gamma(t)\|_g\,dt$.
- **Riemannian volume form** (on an oriented manifold): $\mathrm{vol}_g = \sqrt{\det(g_{ij})}\,dx^1\wedge\dots\wedge dx^n$.
- **Riemannian distance:** $d_g(p, q) = \inf\{L(\gamma) : \gamma \text{ joins } p, q\}$.

The Riemannian distance turns $(M, g)$ into a metric space (in the topology-textbook sense), and the metric topology agrees with the manifold topology.

**Examples.**
1. **Euclidean space** $(\mathbb{R}^n, g_{\mathrm{Eucl}})$: $g_{ij} = \delta_{ij}$, the identity matrix. The standard inner product. $L(\gamma) = \int |\dot\gamma|$, distances are the usual Euclidean ones.

2. **Sphere $S^2$ with round metric.** In spherical coordinates $(\theta, \phi)$ ($\theta$ polar, $\phi$ azimuthal):
$$g = d\theta^2 + \sin^2\theta\,d\phi^2.$$
This is the metric inherited from the standard embedding $S^2 \subset \mathbb{R}^3$. The volume form is $\sin\theta\,d\theta\wedge d\phi$, and the surface area $\int_{S^2} \mathrm{vol}_g = 4\pi$ as expected.

3. **Hyperbolic plane $\mathbb{H}^2$.** Upper half-plane $\{(x, y) : y > 0\}$ with metric
$$g = \frac{dx^2 + dy^2}{y^2}.$$
This metric has constant Gaussian curvature $-1$. Geodesics are vertical lines and semicircles meeting the $x$-axis at right angles. The distance between $(0, 1)$ and $(0, e)$ is $1$ (for the geodesic going up); the distance "tends to infinity" as $y \to 0$ along any path approaching the boundary. The hyperbolic plane is the prototypical model of negative curvature.

4. **Pullback / induced metric.** If $f: M \to (N, h)$ is an immersion (injective differential), the **pullback metric** $f^*h$ on $M$ is defined by $(f^*h)_p(u, v) = h_{f(p)}(df_p u, df_p v)$. This is how submanifolds inherit metrics from their ambient spaces. Every smoothly embedded surface in $\mathbb{R}^3$ becomes Riemannian this way — and the result is always positive-definite if $f$ is an immersion.

**Existence of Riemannian metrics.** Every paracompact smooth manifold admits a Riemannian metric. Proof: cover by charts, take the Euclidean metric in each chart, and glue with a partition of unity — convex combinations of inner products are inner products, so positive-definiteness is preserved.

**Why this matters.** A Riemannian metric is the minimal extra data that turns a smooth manifold into something one can do classical geometry on. Smoothness gives derivatives; the metric gives lengths. Every formula one writes in classical geometry — the law of cosines, the cross product, the area of a triangle, the volume of a region, the gradient of a function, the divergence of a vector field — eventually traces back to either the metric or a derivative of it. Without the metric, those operations are not defined.

A useful mental model: a smooth manifold is like a piece of fabric you can fold and unfold smoothly; a Riemannian manifold is the same fabric with a built-in tape measure. The fabric and the tape measure are independent — you can put many different metrics on the same smooth manifold, and they give different geometries. Compare $S^2$ with the round metric (constant positive curvature) versus an ellipsoid (varying curvature) versus a smoothed-out cube (almost-flat regions joined by high-curvature ridges). All three are diffeomorphic as smooth manifolds; their geometries are wildly different.

**Numerical example of distance on $S^2$.** Compute the distance from the north pole $(\theta = 0)$ to the point $(\theta_0, \phi_0)$ at latitude $\pi/2 - \theta_0$. The geodesic is the meridian along longitude $\phi_0$, parameterised as $\gamma(t) = (t\theta_0/T, \phi_0)$ for $t \in [0, T]$. Then $\|\dot\gamma\|^2 = (\theta_0/T)^2 + 0 = \theta_0^2/T^2$, and $L(\gamma) = \int_0^T \theta_0/T\,dt = \theta_0$. So the great-circle distance from the north pole to a point at colatitude $\theta_0$ is exactly $\theta_0$ — a unit sphere has equator at distance $\pi/2$ and antipode at distance $\pi$, as expected.

---

## 2. Musical Isomorphisms

A Riemannian metric gives a canonical identification between tangent and cotangent vectors at each point. Given $v \in T_pM$, define the covector $v^\flat \in T_p^*M$ by $v^\flat(w) = g_p(v, w)$. The map $\flat: T_pM \to T_p^*M$, $v \mapsto v^\flat$ ("flat") is a linear isomorphism (positive-definiteness ensures injectivity; finite dimension gives surjectivity). Its inverse $\sharp: T_p^*M \to T_pM$ ("sharp") raises an index.

In coordinates: if $v = v^i\partial_i$, then $v^\flat = g_{ij}v^j\,dx^i$. The matrix of $\flat$ is the metric matrix $g_{ij}$; the matrix of $\sharp$ is its inverse $g^{ij}$. These are the constructions that physicists call "raising and lowering indices."

**Gradient.** The gradient of a smooth $f: M \to \mathbb{R}$ is the metric-dependent vector field $\nabla f = (df)^\sharp$. In coordinates $(\nabla f)^i = g^{ij}\partial_j f$. Without a metric, $df$ is a covector field; with a metric, you can convert it to a vector field that "points in the direction of steepest ascent" with magnitude equal to the rate of ascent. The gradient depends on the metric and is the right notion for variational and gradient-flow problems.

---

## 3. Affine Connections

To differentiate vector fields, we need extra data. An **affine connection** on $M$ is an $\mathbb{R}$-bilinear map $\nabla: \Gamma(TM) \times \Gamma(TM) \to \Gamma(TM)$, written $(X, Y) \mapsto \nabla_X Y$, such that

1. $\nabla_{fX} Y = f \nabla_X Y$ (linear in $X$ over smooth functions),
2. $\nabla_X (fY) = X(f) Y + f \nabla_X Y$ (Leibniz rule in $Y$).

In a coordinate chart, $\nabla$ is determined by the **Christoffel symbols** $\Gamma^k_{ij}$ via $\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\partial_k$. The components of $\nabla_X Y$ for $X = X^i\partial_i$, $Y = Y^j\partial_j$ are then $(\nabla_X Y)^k = X^i(\partial_i Y^k + \Gamma^k_{ij}Y^j)$.

**Torsion** of a connection: $T(X, Y) = \nabla_X Y - \nabla_Y X - [X, Y]$. The connection is **torsion-free** iff $\Gamma^k_{ij} = \Gamma^k_{ji}$. **Metric compatibility**: $X(g(Y, Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$, equivalently $\nabla g = 0$.

---

## 4. The Levi-Civita Connection

**Theorem (Fundamental theorem of Riemannian geometry).** On any Riemannian manifold $(M, g)$, there is a *unique* affine connection $\nabla$ that is both torsion-free and metric-compatible. This is the **Levi-Civita connection**.

*Proof sketch.* Imposing torsion-freeness and metric compatibility produces the **Koszul formula**:

$$2g(\nabla_X Y, Z) = X g(Y,Z) + Y g(Z,X) - Z g(X,Y) + g([X,Y], Z) - g([Y,Z], X) + g([Z,X], Y).$$

The right-hand side is determined by $g$ and Lie brackets, so $\nabla_X Y$ is uniquely determined. $\square$

In coordinates,

$$\Gamma^k_{ij} = \tfrac{1}{2} g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right).$$

Every covariant derivative, geodesic equation, Riemann tensor calculation traces back to this master formula.

![The Levi-Civita connection as the unique torsion-free metric connection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_2_levi_civita.png)

**Worked example: Christoffel symbols on $S^2$.** With $g = d\theta^2 + \sin^2\theta\,d\phi^2$, $g^{ij} = \mathrm{diag}(1, 1/\sin^2\theta)$. Computing yields $\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta$, $\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta$, all others zero. The geodesic equations $\ddot\theta - \sin\theta\cos\theta\,\dot\phi^2 = 0$ and $\ddot\phi + 2\cot\theta\,\dot\theta\dot\phi = 0$ have great circles as solutions: a curve with constant $\phi$ reduces to $\ddot\theta = 0$, so meridians are geodesics. Other great circles arise by symmetry under $SO(3)$.

**A slightly subtle point.** Note that $\Gamma^\phi_{\theta\phi}$ blows up at the poles ($\theta = 0, \pi$). This is not a singularity of the geometry — the round sphere is perfectly smooth — but a singularity of the spherical-coordinate chart. The Christoffel symbols are not tensors and depend on the chart; choosing a different chart at the poles (e.g. stereographic projection) gives bounded Christoffel symbols there. The lesson: Christoffel symbol blowups always tell you about the chart, not about the manifold. A coordinate-invariant sanity check is needed before believing the geometry is singular — usually the Riemann tensor (next article) is the right diagnostic.

---

## 5. Parallel Transport

Given a curve $\gamma$ and a vector $v$ at $\gamma(0)$, the **parallel transport** of $v$ along $\gamma$ is the unique vector field $V$ along $\gamma$ with $V(0) = v$ and $\nabla_{\dot\gamma} V = 0$. In coordinates, this is a linear ODE $\dot V^k + \Gamma^k_{ij}\dot\gamma^i V^j = 0$. By Picard-Lindelof, parallel transport exists and is unique. The transport map $P_\gamma : T_{\gamma(0)} M \to T_{\gamma(1)} M$ is a linear isomorphism, and for the Levi-Civita connection an isometry — it preserves the metric.

![Parallel transport along a closed loop on the sphere returns a rotated vector](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_3_parallel_transport.png)

**The key point.** On Euclidean space, parallel transport is path-independent: walking from $p$ to $q$ along any path, you carry vectors the same way (just translate them). On a curved manifold, parallel transport is generally **path-dependent**: a vector carried from $p$ to $p$ around a closed loop returns rotated. This is the geometric heart of curvature.

**Worked example: parallel transport on $S^2$ around a triangle.** Take the spherical triangle with vertices at the north pole, $(\theta, \phi) = (\pi/2, 0)$ (equator on prime meridian), and $(\theta, \phi) = (\pi/2, \pi/2)$ (equator at $90^\circ$E). All three sides are great-circle arcs (geodesics). Start at the north pole with a vector pointing south along the prime meridian. Parallel-transport down the prime meridian: the vector remains pointing south (along the geodesic — a tangent of a geodesic stays parallel to itself). At the equator on the prime meridian, the vector points south, i.e., away from the equator into the southern hemisphere — but our tangent space is along the equator, so let me set up clearly: at the north pole, choose a vector pointing toward the equator at longitude $0$. After transport down to the equator at longitude $0$, that vector now lies in the equatorial tangent plane and points southward (perpendicular to the equator). Now transport along the equator from longitude $0$ to longitude $90^\circ$: a vector perpendicular to the equator stays perpendicular (parallel transport along a geodesic preserves angles with the geodesic). So at $(\pi/2, \pi/2)$, the vector still points south. Now transport back up along the meridian at longitude $90^\circ$ to the north pole: the vector points outward (along that meridian, opposite to the meridian's tangent direction). It originally pointed along the prime meridian's direction at the north pole; now it points along the $90^\circ$ meridian's direction. The angle between these is $90^\circ$.

So parallel transport around this geodesic triangle rotates tangent vectors by $90^\circ$. The integrated holonomy equals the **angle excess** of the triangle: a spherical triangle has angle sum $> \pi$, with excess $= \int K\,dA$ — exactly what the Gauss-Bonnet theorem (article 5) said. For our triangle, angles are all $90^\circ$, sum is $3\pi/2$, excess is $\pi/2$, area is $\pi/2$ on a unit sphere ($K = 1$), and the holonomy rotation is also $\pi/2$. All three numbers agree, as Gauss-Bonnet demands.

**Numerical detail of holonomy on $S^2$.** Consider parallel transport around a circle of latitude $\theta_0$ on $S^2$ (not a geodesic — circles of latitude are not geodesics except for the equator). The holonomy angle around this circle is $2\pi(1 - \cos\theta_0)$. As $\theta_0 \to 0$ (small circle near the pole), holonomy $\to 0$ (the circle bounds a tiny region, small total curvature). As $\theta_0 \to \pi/2$ (equator, which *is* a geodesic), holonomy $\to 2\pi$. The continuous variation matches the area enclosed times the curvature — a direct numerical confirmation that holonomy = $\int K\,dA$ over the enclosed region.

**Why this matters.** Parallel transport is the *geometric* manifestation of a connection. Connection, curvature, holonomy — all of these are facets of the same phenomenon, viewed as differential operator, tensor field, and integrated transport. In gauge theory, parallel transport along a Wilson loop is a fundamental observable; in general relativity, parallel transport along a geodesic determines the experienced gravitational field via tidal forces.

**Tidal forces and geodesic deviation.** Two nearby geodesics $\gamma_1, \gamma_2$ in a Riemannian manifold drift apart at a rate determined by curvature. The **Jacobi equation** $\nabla_{\dot\gamma}^2 J + R(J, \dot\gamma)\dot\gamma = 0$ (where $J$ is the separation vector and $R$ is the Riemann tensor — see article 11) governs this drift. In general relativity, this is exactly the equation of *tidal gravitational forces*: two nearby freely-falling objects approach or recede from each other based on the local curvature of spacetime. Newton's "force of gravity" is, in Einstein's reformulation, the curvature-induced geodesic deviation. Riemannian geometry is the rigorous expression of this idea.

**A concrete tidal computation.** Two satellites in slightly different orbits around the Earth drift apart at a rate proportional to $\sqrt{R}$ where $R$ is a curvature component. For Earth-orbit altitudes, the drift rate is microscopic per orbit — but for objects falling toward a black hole, the tidal force can be lethal long before crossing the event horizon. The "spaghettification" famous from popular science is the Jacobi equation with large $R$, exactly the same mathematical phenomenon as parallel transport rotation around a small loop. The deep unification — of tidal forces, parallel transport, and the Riemann tensor — is one of the great triumphs of geometric thinking applied to physics.

---

## 6. Geodesics and the Exponential Map

A **geodesic** is a curve $\gamma$ that is its own parallel transport: $\nabla_{\dot\gamma}\dot\gamma = 0$. In coordinates,
$$\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i\dot\gamma^j = 0.$$
This is a second-order ODE in the curve's coordinates, with initial data $(\gamma(0), \dot\gamma(0)) \in TM$. By Picard-Lindelof, geodesics exist and are locally unique; they may not extend for all time (the manifold could be incomplete).

Geodesics are **locally length-minimizing**: among nearby curves with the same endpoints, the geodesic has the shortest length. They are *not always* globally minimizing — on the sphere, the great-circle arc from the north pole to a point near the south pole going "the long way" is a geodesic but not the shortest path.

![Geodesic completeness vs incompleteness](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_5_geodesic_complete.png)

**Examples.**
- **Euclidean space:** geodesics are straight lines.
- **Sphere $S^2$:** geodesics are great circles. From any starting point and direction, the geodesic is the great circle in that direction; geodesics close up after distance $2\pi$.
- **Hyperbolic plane (upper half-plane model):** geodesics are vertical lines and semicircles meeting the $x$-axis at right angles.
- **Cylinder $S^1 \times \mathbb{R}$:** geodesics are images of straight lines under the universal cover. They wind helically around the cylinder.

**Exponential map.** The **exponential map** $\exp_p: T_pM \to M$ is defined by $\exp_p(v) = \gamma_v(1)$, where $\gamma_v$ is the geodesic with $\gamma_v(0) = p$ and $\dot\gamma_v(0) = v$ (when this exists for $t = 1$). On a small neighborhood of $0 \in T_pM$, $\exp_p$ is a diffeomorphism onto its image, and $(\exp_p)^{-1}$ provides **normal coordinates** at $p$ — coordinates in which the metric looks Euclidean to first order at $p$ and the Christoffel symbols vanish at $p$.

**Worked example.** On $S^2$, $\exp_{NP}$ from the north pole sends $v \in T_{NP}S^2 \cong \mathbb{R}^2$ (with $|v| = r$) to the point at latitude $\pi/2 - r$ in the direction of $v$ (mod $2\pi$). The exponential map is well-defined as a smooth map $\mathbb{R}^2 \to S^2$, but it is not a diffeomorphism — it folds onto the south pole at $|v| = \pi$ and then comes back to itself periodically. Inside the disk $|v| < \pi$ in $T_{NP}S^2$, $\exp_{NP}$ is a diffeomorphism onto $S^2 \setminus \{SP\}$.

**Cut locus.** The **cut locus** of $p$ is the set of points where $\exp_p$ stops being a diffeomorphism — equivalently, points beyond which the geodesic from $p$ stops being globally minimizing. On $S^2$, the cut locus of any point is its antipode (a single point). On a compact Riemannian manifold, the cut locus is non-empty; its structure encodes the global geometry. On a flat torus, the cut locus is a more complicated 1-complex; on a generic surface, it is a piecewise-smooth network. The cut locus is the geometric obstruction to geodesics being globally minimizing, and it appears in everything from optimal transport theory to robotics.

**A computational cautionary tale.** When numerically integrating geodesics on a Riemannian manifold, the choice of coordinates matters enormously. Spherical coordinates on $S^2$ have singular Christoffel symbols at the poles, so a numerical integrator using those coordinates will produce garbage near $\theta = 0, \pi$ even though the physical geodesic is perfectly behaved. The fix is to switch charts when one starts approaching the singularity, or to embed $S^2$ into $\mathbb{R}^3$ and use the constraint $|x|^2 = 1$ together with a Lagrange multiplier. The latter is robust at the cost of higher dimensionality; the former is efficient at the cost of code complexity. There is no universally best approach, and the trade-off is one I have seen graduate students misjudge with predictable consequences.

**First and second variation of arc length.** Geodesics arise as critical points of the length functional. If $\gamma_s(t)$ is a one-parameter family of curves with $\gamma_0 = \gamma$, the **first variation** of $L(\gamma_s)$ at $s = 0$ vanishes iff $\gamma$ is a geodesic (with appropriate boundary conditions). The **second variation** involves the Riemann tensor and decides whether the geodesic is a *minimum* or merely a critical point. This calculus-of-variations perspective is the bridge between Riemannian geometry and Morse theory — and it is how Bott deduced topological information about Lie groups from their geodesics.

---

## 7. Holonomy and Hopf-Rinow

For a closed loop $\gamma$ at $p$, parallel transport defines a linear isometry $P_\gamma \in \mathrm{O}(T_pM, g_p)$. The set of all such isometries forms a subgroup, the **holonomy group** $\mathrm{Hol}(p)$.

![Holonomy group as the angular defect of parallel transport around closed loops](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_4_holonomy.png)

For a Riemannian manifold of dimension $n$, $\mathrm{Hol}(p) \subseteq \mathrm{O}(n)$. For an oriented manifold, $\mathrm{Hol}(p) \subseteq \mathrm{SO}(n)$. Berger classified the irreducible holonomy groups of simply-connected Riemannian manifolds — they are $\mathrm{SO}(n)$, $\mathrm{U}(n/2)$, $\mathrm{SU}(n/2)$, $\mathrm{Sp}(n/4)$, $\mathrm{Sp}(n/4)\mathrm{Sp}(1)$, $G_2$, $\mathrm{Spin}(7)$. Each holonomy class corresponds to a special geometric structure: Kahler ($\mathrm{U}$), Calabi-Yau ($\mathrm{SU}$), hyperkahler ($\mathrm{Sp}$), and the exceptional cases.

**Examples.**
- $\mathbb{R}^n$: $\mathrm{Hol} = \{e\}$ (trivial). Parallel transport is path-independent.
- $S^2$: $\mathrm{Hol} = \mathrm{SO}(2)$ (the full rotation group of the tangent plane).
- Flat torus: $\mathrm{Hol} = \{e\}$ (locally Euclidean, no curvature).
- Calabi-Yau 3-fold: $\mathrm{Hol} = \mathrm{SU}(3)$. These are the manifolds string theorists compactify on.

**Hopf-Rinow theorem.** For a connected Riemannian manifold $M$, the following are equivalent:
1. $(M, d_g)$ is a complete metric space.
2. Every geodesic extends to all of $\mathbb{R}$ (geodesic completeness).
3. The exponential map $\exp_p$ is defined on all of $T_pM$ for some (equivalently, every) $p$.
4. Closed and bounded subsets of $M$ are compact.

Moreover, in this case any two points are joined by a minimizing geodesic.

![Hopf-Rinow theorem: completeness, geodesics, and minimizing curves](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_6_hopf_rinow.png)

This is the Riemannian analog of the Heine-Borel theorem (closed and bounded $\Leftrightarrow$ compact). It tells you that "nice" Riemannian manifolds — those with no missing points or finite-time blowups — admit geodesic-based geometry that behaves the way you'd expect.

**Examples.**
- $\mathbb{R}^n$, $S^n$, $T^n$, $\mathbb{H}^n$: all complete.
- $\mathbb{R}^2 \setminus \{0\}$ with Euclidean metric: not complete (you can approach the missing origin).
- Open ball with Euclidean metric: not complete (geodesics hit the boundary in finite time).
- $\mathbb{H}^2$ in the upper half-plane model: complete despite "looking" bounded — the metric blows up near the $x$-axis, so you cannot reach the $x$-axis in finite proper time.
- Closed manifolds (compact, without boundary) are *always* complete (Hopf-Rinow item 4 holds trivially).

**Why this matters.** Most theorems in Riemannian geometry assume completeness. Without it, geodesics may "fall off the edge" of the manifold, the exponential map is not globally defined, and minimizing geodesics may not exist. Hopf-Rinow is the cleanest summary of when these pathologies are absent.

**Comparison theorems.** Once completeness is assumed, the curvature of the manifold places strong constraints on the geometry. The **Bonnet-Myers theorem** says: a complete Riemannian manifold with Ricci curvature bounded below by a positive constant is compact, with diameter bounded above. The **Cartan-Hadamard theorem** says: a complete simply-connected Riemannian manifold of non-positive sectional curvature is diffeomorphic to $\mathbb{R}^n$ via the exponential map (this is what makes $\mathbb{H}^n$ topologically trivial despite its negative curvature). These comparison theorems propagate local pointwise curvature information into global topological consequences — exactly the kind of result Riemannian geometry was invented to prove.

**Synge's theorem.** A compact orientable even-dimensional manifold with positive sectional curvature must be simply connected. So $S^2$ is simply connected (true), $\mathbb{RP}^2$ does not admit such a metric (true — $\mathbb{RP}^2$ is non-orientable). Synge's theorem and its kin are the kind of statement Riemannian geometry buys you: positivity of curvature constrains the topology in deep ways.

**Volume and curvature: Bishop-Gromov.** Another comparison principle: in a complete Riemannian manifold of Ricci curvature bounded below by $(n-1)k$, the volume of a ball of radius $r$ is bounded above by the volume of the corresponding ball in the model space of constant curvature $k$. As a corollary, if Ricci curvature is nonnegative, volumes grow at most polynomially with radius. This is a quantitative formulation of "positive curvature makes space wrap around itself." The Bishop-Gromov inequality is a workhorse of geometric analysis and underlies Cheeger-Colding theory of metric measure space limits.

---

## 8. Examples on Sphere, Hyperbolic Plane, and Torus

To consolidate, the three model 2-dimensional Riemannian manifolds.

![Classical Riemannian metrics on the sphere, hyperbolic plane, and torus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/10-riemannian-geometry/dg_v2_10_7_examples.png)

**Sphere $S^2$ (constant curvature $+1$).** Metric $g = d\theta^2 + \sin^2\theta\,d\phi^2$. Geodesics are great circles, of length $2\pi$. Parallel transport around a triangle picks up an angle equal to the angular excess. Holonomy $\mathrm{SO}(2)$. Complete. Compact.

**Hyperbolic plane $\mathbb{H}^2$ (constant curvature $-1$).** Upper half-plane metric $g = (dx^2 + dy^2)/y^2$. Geodesics are vertical lines and semicircles meeting the boundary $\{y = 0\}$ at right angles. Triangles have angle sum $< \pi$, with deficit $= -\int K\,dA$. Holonomy $\mathrm{SO}(2)$. Complete (the boundary is "at infinity" — at infinite distance from any interior point). Non-compact.

A useful numerical fact: the area of a hyperbolic triangle with angles $\alpha, \beta, \gamma$ is exactly $\pi - (\alpha + \beta + \gamma)$ — the angle deficit. So all hyperbolic triangles have area at most $\pi$, and ideal triangles (vertices on the boundary, all angles zero) have area exactly $\pi$. This is a striking phenomenon: in negative curvature, the triangle area is a function only of its angles, never its side lengths. Compare to Euclidean geometry, where similar triangles can have any area.

**Flat torus $T^2$ (curvature $0$).** Quotient of $\mathbb{R}^2$ by a lattice, e.g., $\mathbb{Z}^2$. Metric inherited from Euclidean. Geodesics are straight lines mod the lattice — they may close up (rational slope) or be dense (irrational slope). Holonomy $\{e\}$ (trivial). Complete. Compact.

These three are the only simply-connected, 2-dimensional, complete Riemannian manifolds of constant curvature — up to scale. The classification of constant-curvature spaces (Killing-Hopf) extends this trichotomy to higher dimensions: the universal covers are $S^n$, $\mathbb{R}^n$, $\mathbb{H}^n$.

**Why this matters.** These three models occupy a special role in mathematics: every Riemann surface of genus $\geq 2$ has a hyperbolic metric (uniformization theorem); the sphere is the only positively-curved compact orientable surface; and the torus is the unique flat orientable surface of higher genus. For 3-manifolds, Thurston's geometrization (now a theorem of Perelman) decomposes any 3-manifold into pieces modeled on eight homogeneous geometries, of which three are constant-curvature and the rest are products and twisted variations.

**Killing vector fields.** A vector field $X$ on $(M, g)$ is **Killing** if $\mathcal{L}_X g = 0$ — its flow preserves the metric. On the sphere, the rotational vector fields $L_x, L_y, L_z$ are Killing — they generate the isometry group $\mathrm{SO}(3)$. On Euclidean space, translations and rotations are Killing — they generate $\mathrm{Iso}(\mathbb{R}^n) = \mathbb{R}^n \rtimes \mathrm{O}(n)$. The Lie algebra of Killing fields is finite-dimensional (at most $\binom{n+1}{2}$, achieved by maximally symmetric spaces). Computing the Killing fields of a metric is one of the main tools in classifying it; in physics, Killing fields are the conserved-quantity generators of Noether's theorem on a curved space.

**Numerical example: Killing field on $S^2$.** Check that $X = \partial_\phi$ (rotation around the $z$-axis) is Killing on $(S^2, g)$. Compute $\mathcal{L}_X g$: $\mathcal{L}_X(d\theta^2) = 2\,d\theta\,\mathcal{L}_X d\theta = 0$ since $X = \partial_\phi$ doesn't change $\theta$. $\mathcal{L}_X(\sin^2\theta\,d\phi^2) = X(\sin^2\theta)\,d\phi^2 + \sin^2\theta\,\mathcal{L}_X d\phi^2 = 0 + 0 = 0$ (since $\sin^2\theta$ doesn't depend on $\phi$). So $\mathcal{L}_X g = 0$. Confirmed: rotation around $z$ is an isometry. The other rotational fields $\partial_\theta$-related ones can be checked similarly by writing them in coordinates.

**Why these examples matter.** Sphere, hyperbolic plane, flat torus are not just illustrative; they are the universal local models of Riemannian geometry. By the uniformisation theorem, every Riemann surface is locally isometric to one of them (after rescaling). By Thurston's geometrisation, every closed 3-manifold decomposes into pieces locally modelled on eight homogeneous geometries, of which these three are the constant-curvature core and the others are products and twisted variations. Understanding the three model 2-geometries fluently is the prerequisite for understanding their higher-dimensional generalisations.

**A unifying perspective: homogeneous spaces.** All three model surfaces are homogeneous: their isometry groups act transitively. $S^2 = SO(3)/SO(2)$, $\mathbb{R}^2 = E(2)/SO(2)$, $\mathbb{H}^2 = PSL(2,\mathbb{R})/SO(2)$. In each case the geometry is rigid: any local isometry extends globally. This rigidity is what makes constant-curvature geometry computable in closed form. Most Riemannian manifolds are not homogeneous and their geometry is not closed-form — but the constant-curvature cases provide the comparison standards (Bonnet-Myers, Cartan-Hadamard, Bishop-Gromov) against which general Riemannian manifolds are measured.

**A computational example tying it all together.** On hyperbolic space $\mathbb{H}^2 = \{(x,y): y > 0\}$ with metric $g = (dx^2 + dy^2)/y^2$, let me compute the area of an ideal triangle: vertices at $-1$, $+1$, $\infty$ on the boundary. The "triangle" is bounded by the vertical lines $x = -1$ and $x = +1$ (geodesics going to infinity) and the semicircle $x^2 + y^2 = 1$ from $-1$ to $+1$. The hyperbolic area element is $dx\,dy/y^2$. Integrating:

$$\text{Area} = \int_{-1}^{1}\int_{\sqrt{1-x^2}}^{\infty}\frac{dy\,dx}{y^2} = \int_{-1}^{1}\frac{dx}{\sqrt{1-x^2}} = \pi.$$

The ideal triangle has area exactly $\pi$, matching the angle-deficit formula since all three angles are zero. This is the kind of clean closed-form result that makes the constant-curvature models so pleasant to compute on, and it provides a sanity benchmark for any numerical scheme on hyperbolic geometry.

---

## What's Next

We have built the apparatus of Riemannian geometry: metrics, the Levi-Civita connection, parallel transport, geodesics, the exponential map, holonomy, Hopf-Rinow. The next article studies **curvature** in detail: the Riemann tensor, sectional, Ricci, and scalar curvatures, and the model spaces of constant curvature. Curvature is the obstruction to a manifold being locally Euclidean, and it is the geometric quantity that enters Einstein's equations.

**Summary of the key ideas.**

1. A **Riemannian metric** is a smooth choice of inner product on each tangent space; it produces lengths, angles, areas, gradients, and Riemannian distance.
2. **Musical isomorphisms** $\flat$ and $\sharp$ identify tangent and cotangent bundles, yielding the gradient as a metric-dependent companion to the differential.
3. An **affine connection** lets you differentiate vector fields along curves; the **Levi-Civita connection** is the unique torsion-free metric-compatible one.
4. **Parallel transport** is the integral of the connection along curves — generally path-dependent, with path-dependence equal to curvature.
5. **Geodesics** are self-parallel curves, locally length-minimizing; the **exponential map** turns tangent vectors into geodesic endpoints.
6. **Holonomy** is the parallel-transport-induced isometry group around closed loops; **Hopf-Rinow** characterizes complete Riemannian manifolds.
7. The three constant-curvature 2-manifolds — sphere, hyperbolic plane, flat torus — are the universal models of geometry.

**One last reflection** on the conceptual layering. Smooth manifold (article 6) gives differentiability without geometry. Vector field (article 7) gives flow without measurement. Differential form (article 8) gives integration without metric. The Riemannian metric is what fuses all of these into classical geometry — the place where flows preserve length, where forms have norms, where derivations integrate to lengths. Each previous layer was chosen to be metric-free precisely so the metric can be added on top as the final ingredient. That layering is not pedagogical convenience but mathematical reality: the same smooth manifold supports infinitely many Riemannian metrics, and choosing one is choosing a geometry. Every theorem to come — sectional curvature, Ricci flow, Einstein equations — depends on which metric was chosen, and varying the metric is the central activity of Riemannian geometry.

**Practical advice for working with metrics.** Three habits I have found pay off when computing in Riemannian geometry. First: always work in normal coordinates near a point of interest. Christoffel symbols vanish there, the metric is Euclidean to first order, and most identities are easier to verify. Second: when computing on a homogeneous space like $S^n$ or $\mathbb{H}^n$, use the symmetry. The Levi-Civita connection commutes with isometries; computing at one base point and transporting by the isometry group gives the global picture. Third: when in doubt, compute the Riemann tensor (next article) directly from $g_{ij}$ via Mathematica or SymPy. Hand calculation of curvature is tedious enough that automated computer algebra is a productivity multiplier. The conceptual understanding has to be human; the index gymnastics can be machine.

The next article puts curvature centre stage. Riemann tensor, sectional curvature, Ricci, scalar — the hierarchy of curvature invariants — together with the model spaces of constant sectional curvature and the Einstein manifolds that interpolate between them. The metric machinery developed here is exactly what those constructions need.

A final observation on what metrics give you. The fact that we get a metric space structure (Riemannian distance) from infinitesimal data (the inner product on tangent spaces) is genuinely remarkable. It is the Riemannian incarnation of the principle that local data, integrated, gives global data — the same principle that makes integration on smooth manifolds work, that makes Stokes' theorem identify boundary and bulk, and that ultimately makes Gauss-Bonnet identify local curvature with global topology. Every step in the conceptual hierarchy of differential geometry is a variation on this theme.

---

*This is Part 10 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 9 — Integration and Stokes' Theorem](/en/differential-geometry/09-integration-stokes/)*

*Next: [Part 11 — Curvature on Manifolds](/en/differential-geometry/11-curvature-on-manifolds/)*
