---
title: "Differential Geometry (3): The Shape Operator — Curvature of Surfaces"
date: 2021-11-05 09:00:00
tags:
  - differential-geometry
  - curvature
  - surfaces
  - mathematics
categories: Mathematics
series: differential-geometry
translationKey: "differential-geometry-3-shape-operator-curvature-of-surfaces"
lang: en
mathjax: true
description: "The Gauss map and shape operator capture how a surface bends in space — principal, Gaussian, and mean curvatures classify every point as elliptic, hyperbolic, or parabolic."
disableNunjucks: true
series_order: 3
series_total: 12
---

The previous article gave us the intrinsic apparatus: the first fundamental form $\mathrm{I}$, encoded as the symmetric matrix $\begin{pmatrix}E & F \\ F & G\end{pmatrix}$. With it, an ant on the surface can measure lengths, angles, and areas without ever leaving. What an ant on a cylinder cannot do, equipped only with $\mathrm{I}$, is detect that the cylinder is bent. The cylinder has the same first fundamental form as the plane, yet sits very differently in $\mathbb{R}^3$.

This chapter develops the apparatus for that distinction: the *second fundamental form* $\mathrm{II}$, the *Gauss map*, and the *shape operator*. These are extrinsic quantities — they depend on how the surface bends in the surrounding $\mathbb{R}^3$. Out of them we will read off principal, Gaussian, and mean curvatures. Two scalars per point. The story will rhyme deliberately with chapter 1: there we had $\kappa$ and $\tau$, the two scalars that characterize a curve. Here we have $K$ and $H$ (or equivalently $k_1, k_2$), the two scalars that characterize a surface — but only its extrinsic shape, not its intrinsic metric.

A spoiler for the next chapter: $K$ — the Gaussian curvature, defined extrinsically — is in fact intrinsic. That is the *Theorema Egregium*. The mean curvature $H$ is genuinely extrinsic and does change under bending. Watch for the asymmetry as it appears.

---

## The Gauss Map and the Shape Operator

![Gaussian curvature classification: sphere K>0, cylinder K=0, saddle K<0](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_curvature_classification.png)

![Gauss map: surface normals mapped to the unit sphere](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_gauss_map.png)

Given a regular oriented surface $S\subset\mathbb{R}^3$, every point $p$ has a well-defined unit normal $\mathbf{n}(p)\in S^2$ (the unit sphere of directions). The assignment $p\mapsto \mathbf{n}(p)$ is the *Gauss map*:
$$N: S\to S^2,\qquad p\mapsto \mathbf{n}(p).$$

![Gauss map N: S to S^2 sending each point to its unit normal](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_1_gauss_map.png)

In a chart $\mathbf{x}(u, v)$, the Gauss map has the explicit form $N(\mathbf{x}(u,v)) = \frac{\mathbf{x}_u\times\mathbf{x}_v}{|\mathbf{x}_u\times\mathbf{x}_v|}$. Its image lies in $S^2$, and its behavior encodes everything about the surface's bending. Imagine placing your hand flat on a surface and watching which direction "up" points as you slide along. On a flat plane, "up" never changes — the Gauss map is constant, a single point on $S^2$. On a sphere of radius $r$, the normal at any point is $\mathbf{n}(p) = p/r$, so the Gauss map is essentially the identity: every point maps to a distinct direction, reflecting the fact that the sphere curves in every direction equally. On a cylinder of radius $r$ around the $z$-axis, the normal depends only on the angle $\theta$ around the axis, tracing a great circle on $S^2$ — a one-dimensional image, because the cylinder bends in one direction but is flat along its axis.

The saddle surface $z = x^2 - y^2$ gives a more subtle picture: the normal tilts one way along the $x$-direction and the opposite way along the $y$-direction, reflecting the "opposite-bend" character of a saddle point. The Gauss map's image spreads out in two dimensions but with a characteristic "pinch" at the saddle point where the two bending directions fight.

The key insight, and the one that makes the entire theory work: the *differential* $dN_p: T_pS\to T_{N(p)}S^2$ is a linear map between tangent planes. Since $T_{N(p)}S^2$ is the orthogonal complement of $\mathbf{n}(p)$ in $\mathbb{R}^3$ — which is precisely $T_pS$ — we can view $dN_p$ as a linear endomorphism of the tangent plane $T_pS$. Its eigenvalues will be the principal curvatures; its determinant will be the Gaussian curvature; its trace will be twice the mean curvature. Everything flows from this single linear map.

The *shape operator* (or Weingarten map) at $p$ is defined as $S_p = -dN_p: T_pS \to T_pS$. The minus sign is convention: with it, a surface curving "toward" the normal (like the inside of a bowl when the normal points up) has positive principal curvatures. The shape operator is self-adjoint with respect to the first fundamental form: for any $\mathbf{v}, \mathbf{w}\in T_pS$, $\langle S_p\mathbf{v}, \mathbf{w}\rangle = \langle\mathbf{v}, S_p\mathbf{w}\rangle$.

The proof of self-adjointness is short and illuminating. Since $\mathbf{n}\cdot\mathbf{x}_u = 0$ identically (the normal is perpendicular to the surface), differentiating with respect to $v$ gives $\mathbf{n}_v\cdot\mathbf{x}_u + \mathbf{n}\cdot\mathbf{x}_{uv} = 0$. Similarly, differentiating $\mathbf{n}\cdot\mathbf{x}_v = 0$ with respect to $u$ gives $\mathbf{n}_u\cdot\mathbf{x}_v + \mathbf{n}\cdot\mathbf{x}_{vu} = 0$. Since $\mathbf{x}_{uv} = \mathbf{x}_{vu}$ (mixed partials commute), we get $\mathbf{n}_v\cdot\mathbf{x}_u = \mathbf{n}_u\cdot\mathbf{x}_v$. This is exactly the symmetry condition for $S_p = -dN_p$ to be self-adjoint. The proof reveals *why* the shape operator is symmetric: it is because the surface is twice-differentiable and partial derivatives commute. No deeper structure is needed.

![Shape operator and its eigenvectors as principal directions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_2_shape_operator.png)

Self-adjoint operators on a real inner product space are diagonalizable with real eigenvalues. The eigenvalues of $S_p$ are the *principal curvatures* $k_1(p)$ and $k_2(p)$, and the corresponding eigenvectors (orthogonal in $T_pS$) are the *principal directions*. Imagine standing at a point on a surface and rotating through all tangent directions: the normal curvature — how much the surface bends in that direction — oscillates sinusoidally between $k_1$ and $k_2$. This is Euler's theorem (1760): if $\mathbf{w}$ makes angle $\alpha$ with the first principal direction, then $\kappa_n(\mathbf{w}) = k_1\cos^2\alpha + k_2\sin^2\alpha$.

The principal directions are the directions of maximum and minimum bending. Picture a potato chip (a hyperbolic paraboloid): there is one direction where it curves up and another where it curves down. Those are the principal directions. At a point where $k_1 = k_2$ (called an *umbilic* point), every direction is equally curved — think of a perfect sphere, where the surface looks the same in every direction from any point. The sphere is entirely umbilic. The famous Caratheodory conjecture asks whether every smooth convex surface has at least two umbilic points; it remains unresolved in full generality.

Gauss introduced this map around 1825 during his work on geodesy and map projections. He was measuring the surface of the Kingdom of Hanover, and he needed to understand how curvature affects map-making. The Gauss map was his tool for quantifying "how bent" a surface is at each point, and it led directly to the Theorema Egregium — the realization that some curvature information is intrinsic to the surface and cannot be changed by bending.

---

## The Second Fundamental Form and Its Coefficients

The shape operator is a linear map; to compute with it, we need its matrix representation. In a chart $\mathbf{x}(u, v)$, define three coefficients:
$$L = -\mathbf{n}_u\cdot\mathbf{x}_u = \mathbf{n}\cdot\mathbf{x}_{uu}, \quad M = -\mathbf{n}_u\cdot\mathbf{x}_v = \mathbf{n}\cdot\mathbf{x}_{uv}, \quad N = -\mathbf{n}_v\cdot\mathbf{x}_v = \mathbf{n}\cdot\mathbf{x}_{vv}.$$

![Shape operator: how normals change along the surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_shape_operator.png)

The second equality in each line comes from differentiating $\mathbf{n}\cdot\mathbf{x}_u = 0$: since $\partial_u(\mathbf{n}\cdot\mathbf{x}_u) = \mathbf{n}_u\cdot\mathbf{x}_u + \mathbf{n}\cdot\mathbf{x}_{uu} = 0$, we get $\mathbf{n}\cdot\mathbf{x}_{uu} = -\mathbf{n}_u\cdot\mathbf{x}_u = L$. These three numbers form the *second fundamental form* $\mathrm{II} = \begin{pmatrix}L & M \\ M & N\end{pmatrix}$, a symmetric bilinear form on each tangent plane.

The geometric meaning: for a tangent vector $\mathbf{w} = a\,\mathbf{x}_u + b\,\mathbf{x}_v$, the *normal curvature* in the direction of $\mathbf{w}$ is $\kappa_n(\mathbf{w}) = \mathrm{II}(\mathbf{w}, \mathbf{w})/\mathrm{I}(\mathbf{w}, \mathbf{w}) = (La^2 + 2Mab + Nb^2)/(Ea^2 + 2Fab + Gb^2)$. This is the curvature of the normal section — the curve you get by slicing the surface with a plane containing $\mathbf{n}$ and $\mathbf{w}$. It measures how the surface bends away from its tangent plane in a specific direction.

The relationship between forms and the shape operator: $S_p = \mathrm{I}^{-1}\mathrm{II}$, viewing both as $2\times 2$ matrices. The eigenvalues of $S_p$ — the principal curvatures — satisfy $\det(\mathrm{II} - k\,\mathrm{I}) = 0$. They are the solutions of the characteristic equation $(L - kE)(N - kG) - (M - kF)^2 = 0$.

Why have both $\mathrm{I}$ and $\mathrm{II}$ rather than just the shape operator? Because they play different geometric roles. The first fundamental form is the metric: it knows about distances and angles within the surface, blind to how the surface sits in space. The second fundamental form measures extrinsic bending: how fast the tangent plane tilts as you move. Together, they determine the surface up to rigid motion — this is the Fundamental Theorem of Surfaces (Bonnet, 1867): given smooth functions $E, F, G, L, M, N$ satisfying the Gauss equation and the Codazzi-Mainardi compatibility conditions, there exists a unique (up to rigid motion) surface realizing them.

A worked computation for the unit sphere with outward normal: parametrize $\mathbf{x}(\theta, \varphi) = (\sin\varphi\cos\theta, \sin\varphi\sin\theta, \cos\varphi)$. The outward normal is $\mathbf{n} = \mathbf{x}/|\mathbf{x}| = \mathbf{x}$ (unit sphere, so the position vector itself is the outward normal). Then $\mathbf{x}_{uu} = \partial_\varphi^2\mathbf{x}$ gives second derivatives whose dot product with $\mathbf{n} = \mathbf{x}$ yields $L = -1$ (with outward normal convention; effectively $\mathrm{II} = -\mathrm{I}$ on the unit sphere). The shape operator is $S = \mathrm{I}^{-1}\mathrm{II} = -I_2$, with both eigenvalues equal to $-1$ (or $+1$ with inward normal). Both principal curvatures are $1/r = 1$, every point is umbilic, and the Gaussian curvature is $K = 1$.

For a cylinder of radius $r$ with axis along the $z$-direction, parametrize $\mathbf{x}(\theta, z) = (r\cos\theta, r\sin\theta, z)$. The inward normal is $\mathbf{n} = -(\cos\theta, \sin\theta, 0)$. Computing: $L = \mathbf{n}\cdot\mathbf{x}_{\theta\theta} = -(\cos\theta, \sin\theta, 0)\cdot(-r\cos\theta, -r\sin\theta, 0) = r$, $M = 0$, $N = 0$ (since $\mathbf{x}_{zz} = 0$ and $\mathbf{x}_{\theta z} = 0$). With $E = r^2$, $G = 1$, $F = 0$: the shape operator has eigenvalues $k_1 = L/E = 1/r$ and $k_2 = N/G = 0$. One principal curvature is $1/r$ (bending around the cylinder), the other is $0$ (no bending along the axis). Gaussian curvature $K = 0$, mean curvature $H = 1/(2r)$.

---

## Gaussian and Mean Curvature: Classification of Surface Points

From the principal curvatures $k_1, k_2$, we extract two scalar invariants:
$$K = k_1 k_2 = \det S = \frac{LN - M^2}{EG - F^2}, \qquad H = \frac{k_1 + k_2}{2} = \frac{1}{2}\mathrm{tr}\,S = \frac{EN - 2FM + GL}{2(EG - F^2)}.$$

![Animation: normal curvature as direction rotates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_normal_rotation.gif)

![Principal curvatures: maximum and minimum normal curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_normal_curvature.png)

The Gaussian curvature $K$ is the product of principal curvatures; the mean curvature $H$ is their average. Together they determine $k_1$ and $k_2$ via $k_{1,2} = H \pm \sqrt{H^2 - K}$, so no information is lost.

The sign of $K$ provides a three-way classification of every point on a surface:

**Elliptic points ($K > 0$).** Both principal curvatures have the same sign — the surface bends like a bowl. The second fundamental form is definite (either positive or negative definite), meaning the surface lies entirely on one side of its tangent plane (locally). Examples: every point on a sphere, the vertex of a paraboloid $z = x^2 + y^2$, the outer rim of a torus. Imagine holding a bowl: water poured on it rolls toward the center from every direction. That is positive curvature.

**Hyperbolic points ($K < 0$).** The principal curvatures have opposite signs — the surface bends like a saddle. The second fundamental form is indefinite, and the surface crosses its tangent plane along two curves (the asymptotic directions). Examples: every point on a one-sheeted hyperboloid, the inner rim of a torus, the center of a saddle $z = x^2 - y^2$. Imagine sitting in a horse saddle: you curve downward in the left-right direction and upward in the front-back direction. That is negative curvature.

**Parabolic points ($K = 0$, with at least one $k_i \neq 0$).** One principal curvature vanishes — the surface bends in one direction but is flat in the other. The second fundamental form is semi-definite. Examples: every point on a cylinder, every point on a cone (away from the apex). Imagine rolling a piece of paper into a tube: it bends around the tube but is perfectly straight along the tube's length. That is zero Gaussian curvature — the paper remains "intrinsically flat" even though it appears bent from outside.

There is also the *planar* or *flat* point where $k_1 = k_2 = 0$ (both curvatures vanish). The surface is locally approximated by its tangent plane to second order. These are typically isolated on generic surfaces.

The torus beautifully exhibits all three regimes. Parametrize with major radius $R$ and minor radius $r$: the Gaussian curvature is $K = \cos v / (r(R + r\cos v))$. The outer half (where $\cos v > 0$) is elliptic — it curves like a sphere. The inner half (where $\cos v < 0$) is hyperbolic — it curves like a saddle. The top and bottom circles (where $\cos v = 0$) are parabolic — transition zones between the two regimes. And the total curvature $\int K\,dA = 0$, because the positive and negative contributions cancel exactly. This cancellation is forced by topology: the torus has Euler characteristic $\chi = 0$, and Gauss-Bonnet says $\int K\,dA = 2\pi\chi = 0$.

A physical thought experiment cements the intuition. Take a sheet of paper ($K = 0$ everywhere). You can roll it into a cylinder ($K = 0$), a cone ($K = 0$), or any other *developable surface* (ruled surface with $K = 0$). But you cannot shape it into a sphere ($K > 0$) or a saddle ($K < 0$) without crumpling or stretching. This is because bending without stretching preserves the first fundamental form, and the Theorema Egregium (next chapter) says $K$ depends only on the first form. So if you start with $K = 0$ and only bend (no stretching), you stay at $K = 0$. To reach positive or negative $K$, you must stretch or compress, which distorts distances. This is why cartographers cannot make a perfectly accurate flat map of the Earth: the sphere has $K > 0$ and paper has $K = 0$, and no isometry exists between them.

---

## Worked Examples: From Spheres to Saddles

Theory needs grounding. Here are four detailed computations that cover the main cases.

![Normal curvature varies with direction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_normal_curvature.png)

**The saddle surface $z = uv$ (hyperbolic paraboloid).** Chart $\mathbf{x}(u,v) = (u, v, uv)$. Compute: $\mathbf{x}_u = (1, 0, v)$, $\mathbf{x}_v = (0, 1, u)$. First form: $E = 1+v^2$, $F = uv$, $G = 1+u^2$, and $EG-F^2 = 1+u^2+v^2$. Cross product: $\mathbf{x}_u\times\mathbf{x}_v = (-v, -u, 1)$, with magnitude $\sqrt{1+u^2+v^2}$. Unit normal: $\mathbf{n} = (-v, -u, 1)/\sqrt{1+u^2+v^2}$. Second derivatives: $\mathbf{x}_{uu} = 0$, $\mathbf{x}_{uv} = (0,0,1)$, $\mathbf{x}_{vv} = 0$. So $L = \mathbf{n}\cdot\mathbf{x}_{uu} = 0$, $M = \mathbf{n}\cdot\mathbf{x}_{uv} = 1/\sqrt{1+u^2+v^2}$, $N = 0$.

Gaussian curvature: $K = (LN - M^2)/(EG-F^2) = -M^2/(1+u^2+v^2) = -1/(1+u^2+v^2)^2$. Negative everywhere — the surface is entirely hyperbolic. At the origin, $K = -1$, and $K \to 0$ as we move away from the origin. The saddle is sharpest at its center.

Mean curvature: $H = (EN - 2FM + GL)/(2(EG-F^2)) = -2uv \cdot M/(2(1+u^2+v^2)) = -uv/((1+u^2+v^2)^{3/2})$. At the origin, $H = 0$. The origin is a *minimal point* — not just a saddle point, but one where the mean curvature vanishes. This is physically meaningful: a soap film stretched over the saddle-shaped boundary would pass through the origin without any net "pressure" from curvature.

**The paraboloid $z = u^2 + v^2$ (elliptic paraboloid).** Chart $\mathbf{x}(u,v) = (u, v, u^2+v^2)$. Set $w = 1 + 4u^2 + 4v^2$ for convenience. After computation: $E = 1+4u^2$, $F = 4uv$, $G = 1+4v^2$, $EG - F^2 = w$. Normal: $\mathbf{n} = (-2u, -2v, 1)/\sqrt{w}$. Second derivatives: $\mathbf{x}_{uu} = (0,0,2)$, $\mathbf{x}_{uv} = 0$, $\mathbf{x}_{vv} = (0,0,2)$. So $L = 2/\sqrt{w}$, $M = 0$, $N = 2/\sqrt{w}$.

Gaussian curvature: $K = (LN - M^2)/(EG - F^2) = (4/w)/w = 4/w^2 = 4/(1+4u^2+4v^2)^2$. Positive everywhere — entirely elliptic. At the origin, $K = 4$; at distance $d$ from the origin (with $4d^2 \gg 1$), $K \approx 4/(4d^2)^2 = 1/(4d^4)$, decaying rapidly. The paraboloid is sharpest at its vertex and flattens toward its rim.

At the origin: $H = (E\cdot N + G\cdot L - 2FM)/(2w) = (2/\sqrt{1} + 2/\sqrt{1})/(2\cdot 1) = 2$. With $K = 4 = H^2$, we get $k_1 = k_2 = 2$ — the origin is umbilic. This makes sense from the rotational symmetry of $z = u^2 + v^2$ about the $z$-axis: at the symmetric center, all directions must have equal curvature.

**The torus $(R + r\cos v)\cos u, (R + r\cos v)\sin u, r\sin v)$.** The computation gives $K = \cos v/(r(R + r\cos v))$, which transitions from positive (outer, $v$ near $0$) through zero (top/bottom, $v = \pm\pi/2$) to negative (inner, $v$ near $\pi$). At the outermost point ($v = 0$): $K = 1/(r(R+r))$. At the innermost point ($v = \pi$): $K = -1/(r(R-r))$. The inner rim has stronger negative curvature than the outer rim has positive curvature — but the inner rim has smaller area, and the total integral still vanishes.

**The pseudosphere (tractrix of revolution).** This is the surface of revolution of the tractrix curve, and its distinguishing property is $K = -1$ everywhere — constant negative Gaussian curvature. It is the analog of the sphere (constant positive curvature) but in the hyperbolic world. The pseudosphere realizes a piece of the hyperbolic plane in $\mathbb{R}^3$, and its geodesics exhibit the non-Euclidean behavior of Lobachevsky geometry: through a point not on a given geodesic, multiple "parallel" geodesics exist. Hilbert proved in 1901 that no *complete* surface in $\mathbb{R}^3$ can have $K = -1$ everywhere — the pseudosphere necessarily has a singular edge — but as a local model of hyperbolic geometry, it is invaluable.

---

## Mean Curvature, Minimal Surfaces, and Physical Applications

Mean curvature has a variational characterization that gives it direct physical significance. Consider a compact surface patch $S$ with boundary $\partial S$ held fixed. Perturb the surface by $\mathbf{x} \mapsto \mathbf{x} + tf\mathbf{n}$ where $f$ vanishes on $\partial S$. The first variation of area is:
$$\frac{d}{dt}\bigg|_{t=0}\mathrm{Area}(S_t) = -2\int_S fH\,dA.$$

![Mean curvature and minimal surfaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_mean_curvature.png)

This vanishes for all variations $f$ if and only if $H \equiv 0$. Surfaces with $H = 0$ everywhere are *minimal surfaces* — critical points of the area functional. They are not necessarily area-minimizing globally (saddle points of the functional are also critical), but they are locally stationary.

Soap films are physical minimal surfaces. Surface tension drives the film to minimize area subject to boundary constraints, which forces $H = 0$ at every interior point. The classical minimal surfaces are:
- The plane (trivially $H = 0$, $K = 0$).
- The catenoid: the surface of revolution of a catenary $y = a\cosh(x/a)$. It is the shape of a soap film spanning two parallel circular rings. Its Gaussian curvature is $K = -1/(a^2\cosh^4(x/a))$ — negative, as expected for a saddle-like minimal surface.
- The helicoid: a helicoidal ramp $\mathbf{x}(u,v) = (v\cos u, v\sin u, au)$. This is the only ruled minimal surface (apart from the plane). It has $K < 0$ everywhere except along the axis.
- Enneper's surface: a self-intersecting minimal surface with interesting topology, parametrized by $\mathbf{x}(u,v) = (u - u^3/3 + uv^2, v - v^3/3 + vu^2, u^2 - v^2)$.

The catenoid and helicoid are related by a remarkable deformation: there is a one-parameter family of isometric minimal surfaces interpolating between them. You can physically demonstrate this by dipping an appropriate wire frame in soap solution and slowly twisting it. The intermediate surfaces maintain $H = 0$ throughout the deformation.

In biology, the Helfrich model (1973) describes cell membranes via the bending energy $\mathcal{E} = \int (H - H_0)^2\,dA + \int K\,dA$, where $H_0$ is a spontaneous curvature determined by the lipid composition. By Gauss-Bonnet, the $\int K\,dA$ term is topological (constant for a given topology), so the shape is controlled by minimizing $\int(H - H_0)^2\,dA$. Red blood cells have their characteristic biconcave discoid shape because it minimizes this energy for their specific $H_0$. The physics of cell shape is literally the differential geometry of mean curvature.

In architecture, doubly curved surfaces (those with $K \neq 0$) are prized for structural efficiency. Felix Candela's hyperbolic paraboloid shells ($K < 0$) are remarkably stiff despite their thinness, because the saddle geometry distributes forces along both families of asymptotic lines. The Sydney Opera House shells have $K > 0$ (pieces of a sphere), chosen so that all shells have the same curvature — simplifying construction by using identical formwork.

In general relativity, the mean curvature of spacelike hypersurfaces in a Lorentzian 4-manifold appears in the Hamiltonian constraint equations of the initial value formulation. Maximal slices ($H = 0$) are the gravitational analog of minimal surfaces, and they play a role in singularity theorems and numerical relativity.

In computer graphics, mean and Gaussian curvature guide mesh processing algorithms. Curvature-based mesh simplification preserves high-curvature regions (which carry visual information) while decimating low-curvature regions (which are perceptually flat). Surface reconstruction algorithms use curvature to guide interpolation: the reconstructed surface should have curvature consistent with the measured data points. Discrete differential geometry — the study of curvature on triangulated meshes — has become a major field bridging pure mathematics and computer science.

A deep connection to partial differential equations: the equation $H = 0$ (characterizing minimal surfaces) is a second-order quasilinear elliptic PDE in the graph case $z = f(x,y)$:
$$\frac{(1 + f_y^2)f_{xx} - 2f_xf_yf_{xy} + (1 + f_x^2)f_{yy}}{(1 + f_x^2 + f_y^2)^{3/2}} = 0.$$
This is the *minimal surface equation*. Its solutions include $f = 0$ (plane), Scherk's surface $f = \log(\cos x/\cos y)$, and many others. The theory of this PDE (existence, regularity, boundary behavior) is a major chapter of geometric analysis. The Plateau problem — finding a minimal surface spanning a given boundary curve — was solved by Douglas and Rado (independently, 1931), earning Douglas the first Fields Medal in 1936.

---

## Normal Curvature, Asymptotic Lines, and Ruled Surfaces

Beyond the principal curvatures, the variation of normal curvature $\kappa_n$ across directions in $T_pS$ reveals additional geometric structure.

![Gaussian curvature coloring on a torus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/03_torus_curvature.png)

Euler's theorem states: if $\mathbf{w}$ makes angle $\alpha$ with the first principal direction, then $\kappa_n(\mathbf{w}) = k_1\cos^2\alpha + k_2\sin^2\alpha$. The normal curvature varies sinusoidally between $k_1$ and $k_2$ as we rotate through the tangent plane. This gives a clean geometric picture: the two principal directions are where the surface bends the most and the least, and all other directions are intermediate.

*Lines of curvature* are curves whose tangent at every point is a principal direction. On the sphere, every curve is a line of curvature (every direction is principal at an umbilic). On the cylinder, the two families are axial lines and circular cross-sections. On a generic surface, lines of curvature form an orthogonal net covering the surface, and in line-of-curvature coordinates we have $F = M = 0$ simultaneously — both forms are diagonal. This simplifies many computations dramatically.

*Asymptotic lines* are curves along which the normal curvature vanishes: $\mathrm{II}(\gamma', \gamma') = La'^2 + 2Ma'b' + Nb'^2 = 0$. They exist only where $K \leq 0$. At a hyperbolic point, the discriminant $M^2 - LN > 0$ (since $K = (LN-M^2)/(EG-F^2) < 0$), giving two real solutions for $a':b'$ — two asymptotic directions. At a parabolic point, exactly one asymptotic direction. At an elliptic point, none (the second form is definite).

A beautiful geometric fact: on a *ruled surface* (one swept out by a one-parameter family of straight lines), the rulings are always asymptotic lines. A straight line in $\mathbb{R}^3$ has zero curvature, so its normal curvature on any surface it lies on must be zero. The hyperboloid of one sheet $x^2 + y^2 - z^2 = 1$ is doubly ruled — two families of lines cover it — and both families are asymptotic lines. The hyperbolic paraboloid $z = xy$ is also doubly ruled (lines $t \mapsto (t, c, ct)$ and $t \mapsto (c, t, ct)$). This is why these surfaces appear in architecture: they can be built from straight beams, and the ruled structure aligns with directions of zero normal curvature, providing structural efficiency.

Asymptotic lines also have a physical interpretation for thin shells. Along an asymptotic direction, the surface offers no bending resistance from normal curvature — it can "flex" freely in that direction. Engineers designing shells must know the asymptotic directions because these are potential failure modes under bending loads.

For a more exotic example, consider the Mobius strip parametrized by $\mathbf{x}(u,v) = ((1 + v\cos(u/2))\cos u, (1 + v\cos(u/2))\sin u, v\sin(u/2))$ for $u \in [0, 2\pi)$ and $v$ small. This is a non-orientable surface — the normal vector flips sign after one circuit around the strip. The principal curvatures and lines of curvature on the Mobius strip exhibit interesting topology: following a line of curvature around the strip, you return to a different principal direction than the one you started with (because the orientation has reversed). The Mobius strip does not admit a continuous global choice of principal directions, reflecting its non-orientability.

On surfaces of revolution, the lines of curvature are always the meridians and the parallels (circles of latitude). This follows from symmetry: the meridians are planes of reflection symmetry, and the principal curvatures must be along and perpendicular to the axis of symmetry. Working in these natural coordinates diagonalizes both $\mathrm{I}$ and $\mathrm{II}$ ($F = M = 0$), simplifying all curvature computations. This is why surfaces of revolution — spheres, cylinders, cones, tori, catenoids, pseudospheres — serve as the primary computational testing ground for surface theory.

---

## The Fundamental Theorem and the Road Ahead

We now have two symmetric bilinear forms on each tangent plane. The natural question: do they determine the surface?

**Theorem (Bonnet, 1867).** Given smooth functions $E, F, G, L, M, N$ on an open set $U \subset \mathbb{R}^2$ with $EG - F^2 > 0$, satisfying the *Gauss equation* and the *Codazzi-Mainardi equations*, there exists a surface $\mathbf{x}: U \to \mathbb{R}^3$ with these as its first and second fundamental form coefficients. This surface is unique up to rigid motion in $\mathbb{R}^3$.

The compatibility conditions are non-trivial: the Gauss equation expresses $K$ in terms of the Christoffel symbols (derived from $\mathrm{I}$ alone), and the Codazzi-Mainardi equations relate derivatives of $L, M, N$ to Christoffel symbols. The Gauss equation *is* the Theorema Egregium in disguise: it forces $K$ to be computable from $\mathrm{I}$, regardless of the specific $\mathrm{II}$.

The operational vocabulary: two surfaces are *isometric* if a diffeomorphism preserves $\mathrm{I}$ (same distances). They are *congruent* if a rigid motion takes one to the other (same $\mathrm{I}$ *and* $\mathrm{II}$). Bending preserves $\mathrm{I}$ but not $\mathrm{II}$: cylinder and plane are isometric but not congruent. A surface is *rigid* if every isometry to another surface in $\mathbb{R}^3$ must be a rigid motion. Cohn-Vossen (1927) proved spheres are rigid — you cannot bend a closed sphere without stretching it. This explains why a ping-pong ball resists deformation: any visible bending must involve stretching, and stretching requires force.

What bending preserves: $K$ (Theorema Egregium). What bending does not preserve: $H$, $k_1$, $k_2$, $\mathrm{II}$, lines of curvature, asymptotic lines. The cylinder has $K = 0$ and so does the plane — rolling paper into a tube does not change $K$. But $H$ changes from $0$ (plane) to $1/(2r)$ (cylinder). The Theorema Egregium is the surprising discovery that one extrinsic-looking quantity ($K$) is actually intrinsic.

A sign convention warning before we close this section. The signs of $L, M, N$ — and hence of $H$ — depend on the choice of normal direction. Flipping $\mathbf{n}$ to $-\mathbf{n}$ flips the sign of all three, hence flips $H$. But $K = (LN - M^2)/(EG - F^2)$ is invariant: two sign flips in the numerator cancel. Always check conventions before comparing formulas between different sources.

A further subtlety: the second fundamental form $\mathrm{II}(\mathbf{w}, \mathbf{w})$ can also be interpreted as the second-order deviation of the surface from its tangent plane. If you write the surface locally as a graph $z = f(x, y)$ over the tangent plane at $p$ (with $p$ at the origin and the tangent plane as $z = 0$), then $f(x,y) = \frac{1}{2}(Lx^2 + 2Mxy + Ny^2) + O(|(x,y)|^3)$. The second fundamental form is literally the Hessian of this graph function, evaluated in surface coordinates. This gives the most direct geometric picture: $\mathrm{II}$ measures how the surface "pulls away" from its tangent plane, quadratically, in each direction. Elliptic points have a definite Hessian (the surface lies locally on one side of the tangent plane); hyperbolic points have an indefinite Hessian (the surface crosses the tangent plane along two curves — the asymptotic directions); parabolic points have a degenerate Hessian.

The Gauss map provides one more beautiful perspective before we close. The *area distortion* of the Gauss map at a point equals $|K|$. For a convex surface, $N$ maps the surface onto $S^2$ with degree 1, so $\int_S K\,dA = 4\pi$. For the torus, the degree is 0, giving $\int K\,dA = 0$. This is the Gauss-Bonnet theorem in preview: total curvature is a topological invariant. The proof comes in chapter 5, but the geometric picture — curvature as the "stretching factor" of the Gauss map, whose total integral counts how many times $S^2$ is covered — is already here.

---

## What's next

The next chapter proves the Theorema Egregium and develops the intrinsic apparatus: Christoffel symbols, geodesics, and parallel transport. The central revelation is that $K$ depends only on $\mathrm{I}$, opening the door to intrinsic geometry — geometry without an ambient space.

---
