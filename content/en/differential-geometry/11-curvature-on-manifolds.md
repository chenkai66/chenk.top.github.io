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

Curvature is the central concept of Riemannian geometry. Intuitively, it measures how much a space deviates from being flat — how parallel lines converge or diverge, how triangles have angle excess or deficit, how volumes grow differently from Euclidean expectations. In the previous article we saw that the path-dependence of parallel transport signals the presence of curvature: a vector carried around a closed loop on $S^2$ returns rotated, while on $\mathbb{R}^n$ it does not. The next step is to make this precise, to extract a tensor that exactly captures this rotation, and to understand its various contractions.

The **Riemann curvature tensor** encodes all intrinsic curvature information of a Riemannian manifold. It is a $(1,3)$-tensor — eats three vectors, produces one — and from it everything else flows. Its various contractions extract progressively coarser but more manageable summaries: **sectional curvature** (a scalar per 2-plane), **Ricci curvature** (a symmetric 2-tensor), and **scalar curvature** (a single number per point). Each captures different geometric content. Sectional curvature controls geodesic deviation in a chosen 2-plane direction; Ricci curvature controls volume comparison theorems and appears in Einstein's equations; scalar curvature is the integrand of the Hilbert action of general relativity.

These objects are not abstract decorations. The constant-curvature 2-spaces (sphere, Euclidean plane, hyperbolic plane) of the previous article are characterized by their sectional curvature being everywhere equal. **Einstein manifolds** are characterized by Ricci proportional to the metric. The **Yamabe problem** — does every metric class on a manifold contain a representative of constant scalar curvature? — is one of the deep theorems of geometric analysis. And the variational structure of curvature (Hilbert action, Einstein-Hilbert functional) is what makes general relativity a *theory* rather than a collection of equations.

The plan: define the Riemann tensor by its action on vector fields, work out the coordinate formula, derive its symmetries, define the contracted curvatures, classify the constant-curvature spaces, and close with Einstein manifolds. Throughout, we will compute on the round $S^2$ to keep the abstract definitions tied to a concrete example.

A guiding analogy. The Riemann tensor is to a Riemannian manifold what the Hessian is to a smooth function: it captures all second-order local information at each point. Just as a function's Hessian decomposes into trace (Laplacian), traceless symmetric, and other irreducible parts under the orthogonal group, the Riemann tensor decomposes into scalar (trace of trace), Ricci (trace), and Weyl (traceless) parts. The pattern is the same; only the object being decomposed is different. If the analogy holds in your head, the algebraic facts of this article will feel less arbitrary.

A second guiding picture. Think of the curvature tensor as the *infinitesimal holonomy*: parallel transport around a small loop bounding area $\delta A$ in the plane $\Pi$ rotates a tangent vector by an angle proportional to $K(\Pi)\,\delta A$. Run this picture in dimension $n$: parallel transport around an arbitrary small loop is an element of the orthogonal group $O(n)$ near identity, and the matrix entries are the components of $R$. Sectional curvature is the diagonal in the right basis; Ricci is the trace; scalar curvature is the trace of the trace. This is the same structure I will lay out algebraically below, but the holonomy picture gives the geometric interpretation of why the algebra works.

---

## The Riemann Curvature Tensor

![Riemann curvature tensor: change in vector after parallel transport around loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/11_riemann_tensor.png)

The **Riemann curvature tensor** $R$ is defined by
$$R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X, Y]}Z.$$
Read this as: "the failure of $\nabla_X \nabla_Y$ to commute with $\nabla_Y \nabla_X$, corrected for the fact that $X$ and $Y$ themselves do not commute (their bracket is $[X, Y]$)." On flat space ($\mathbb{R}^n$), $\nabla$ is just the ordinary partial derivative and the second covariant derivatives commute, so $R \equiv 0$. Curvature is the obstruction to $\nabla$'s commuting.

![Riemann curvature tensor R(X,Y)Z measuring how parallel transport fails](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_1_riemann_tensor.png)

**$C^\infty$-linearity.** Despite involving second derivatives, $R(X, Y)Z$ is $C^\infty$-linear in *each* of $X, Y, Z$ separately. The proof is a direct computation: writing $X = X^i\partial_i$, the second derivatives $X^i X^j \partial_i\partial_j$ would appear, but the antisymmetrization in $X, Y$ kills them. So $R$ is a genuine tensor — its value at $p$ depends only on $X_p, Y_p, Z_p$, not on the fields globally. This is what makes $R$ a pointwise geometric quantity.

**Coordinate formula.** Write $\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\partial_k$ as before. The components of the Riemann tensor are
$$R^l_{ijk} = \partial_i \Gamma^l_{jk} - \partial_j \Gamma^l_{ik} + \Gamma^l_{im}\Gamma^m_{jk} - \Gamma^l_{jm}\Gamma^m_{ik}.$$
In words: derivatives of Christoffels, minus the symmetric counterpart, plus the antisymmetrized "Christoffel squared" term. The two pieces cancel beautifully under the right index conventions.

**Numerical example: $S^2$.** Recall $\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta$ and $\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta$, all others zero. Compute $R^\theta_{\phi\theta\phi}$:
$$R^\theta_{\phi\theta\phi} = \partial_\phi \Gamma^\theta_{\theta\phi} - \partial_\theta \Gamma^\theta_{\phi\phi} + \Gamma^\theta_{\phi m}\Gamma^m_{\theta\phi} - \Gamma^\theta_{\theta m}\Gamma^m_{\phi\phi}.$$
$\Gamma^\theta_{\theta\phi} = 0$, so first term zero. $\partial_\theta(-\sin\theta\cos\theta) = -(\cos^2\theta - \sin^2\theta) = -\cos(2\theta)$. The third term: $\Gamma^\theta_{\phi m}\Gamma^m_{\theta\phi}$, only nonzero for $m = \phi$ ($\Gamma^\theta_{\phi\phi}\Gamma^\phi_{\theta\phi} = (-\sin\theta\cos\theta)(\cot\theta) = -\cos^2\theta$). The fourth term: $\Gamma^\theta_{\theta m}\Gamma^m_{\phi\phi}$, only nonzero for $m = \theta$ but $\Gamma^\theta_{\theta\theta} = 0$, so zero. With careful index bookkeeping, the standard answer is $R^\theta_{\phi\theta\phi} = \sin^2\theta$ — positive, as expected for the round sphere.

The lowered Riemann tensor is $R_{\theta\phi\theta\phi} = g_{\theta\theta}R^\theta_{\phi\theta\phi} = \sin^2\theta$. The sectional curvature in the $\partial_\theta \wedge \partial_\phi$ plane is $K = R_{\theta\phi\theta\phi}/(g_{\theta\theta}g_{\phi\phi} - g_{\theta\phi}^2) = \sin^2\theta/\sin^2\theta = 1$. The unit sphere has constant Gaussian curvature $1$, as Theorema Egregium and elementary intuition both suggest.

---

## Symmetries of the Riemann Tensor

The Riemann tensor with all indices lowered satisfies:

![Sectional curvature: positive, zero, and negative](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/11_sectional_curvature.png)

1. **Antisymmetry in first pair:** $R_{ijkl} = -R_{jikl}$.
2. **Antisymmetry in second pair:** $R_{ijkl} = -R_{ijlk}$.
3. **Pair-swap symmetry:** $R_{ijkl} = R_{klij}$.
4. **First Bianchi:** $R_{ijkl} + R_{iklj} + R_{iljk} = 0$.
5. **Second Bianchi:** $\nabla_m R_{ijkl} + \nabla_k R_{ijlm} + \nabla_l R_{ijmk} = 0$.

These reduce the count of independent components from $n^4$ to $n^2(n^2-1)/12$. In dimension 2 this is 1 component (the Gaussian curvature). In dimension 3, 6 components. In dimension 4, 20 components.

**Why Bianchi matters.** Contracting the second Bianchi identity twice with the metric gives $\nabla^j(R_{jk} - \tfrac{1}{2}R g_{jk}) = 0$. The combination $G_{jk} = R_{jk} - \tfrac{1}{2}R g_{jk}$ is the **Einstein tensor**, and this divergence-freeness is the geometric origin of energy-momentum conservation in general relativity. Einstein's equation $G_{\mu\nu} = 8\pi T_{\mu\nu}$ is consistent only because both sides are divergence-free — Bianchi gives $\nabla G = 0$, physics requires $\nabla T = 0$.

---

## Sectional Curvature

For any 2-plane $\Pi \subset T_p M$ spanned by $X, Y$, the **sectional curvature** is

![Animation: geodesic deviation in positive vs negative curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/11_geodesic_deviation.gif)

$$K(\Pi) = \frac{R(X, Y, X, Y)}{g(X, X)g(Y, Y) - g(X, Y)^2}.$$

The denominator is the squared area of the parallelogram spanned by $X, Y$. The whole expression depends only on $\Pi$, not on the choice of basis.

![Sectional curvature K(X,Y) of a 2-plane in the tangent space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_2_sectional.png)

**Geometric meaning.** $K(\Pi)$ is the Gaussian curvature, at $p$, of the 2-dimensional surface obtained by exponentiating $\Pi$ — the surface of geodesics through $p$ with initial direction in $\Pi$.

**Examples.** $\mathbb{R}^n$: $K \equiv 0$. Round $S^n$: $K \equiv 1$. Hyperbolic $\mathbb{H}^n$: $K \equiv -1$. $S^2 \times \mathbb{R}$: $K = 1$ for the spherical plane, $K = 0$ for any plane with the $\mathbb{R}$ direction.

A consequence of the symmetries: knowing $K$ on every 2-plane determines $R$ entirely. Sectional curvature is enough to encode the full curvature tensor.

**Worked example: sectional curvature of $S^n$.** For the round $n$-sphere of radius $r$, every 2-plane has $K = 1/r^2$. The proof: by symmetry under $SO(n+1)$, the value of $K(\Pi)$ does not depend on the choice of $\Pi$. So $K$ is constant. To find the constant, take the simplest 2-plane (the $\partial_\theta \wedge \partial_\phi$ plane in spherical coordinates) and use the calculation above: $K = 1/r^2$ (the result for unit radius scales by $1/r^2$ under metric rescaling). For the unit sphere this is $K \equiv 1$; for the radius-$R$ sphere it is $K \equiv 1/R^2$.

**Worked example: warped product.** A warped product $(M_1 \times M_2, g_1 + f^2 g_2)$ where $f: M_1 \to \mathbb{R}_{>0}$ has sectional curvatures determined by the curvatures of the factors and derivatives of $f$. For the standard ansatz $g = dr^2 + f(r)^2 g_{S^{n-1}}$ — the rotationally symmetric metric on $\mathbb{R}^n$ — we get $K_{\text{radial}} = -f''/f$ and $K_{\text{tangential}} = (1 - (f')^2)/f^2$. Specialising: $f(r) = r$ gives $\mathbb{R}^n$ (both curvatures zero); $f(r) = \sin r$ gives $S^n$ (both curvatures equal to $1$); $f(r) = \sinh r$ gives $\mathbb{H}^n$ (both curvatures equal to $-1$). The three model spaces are warped products with the three "spherical-trigonometric" warping functions $r, \sin r, \sinh r$, and this is what makes them all constant-curvature.

---

## Ricci and Scalar Curvature

The **Ricci tensor** is the trace of Riemann over the first and third indices:

$$\mathrm{Ric}(Y, Z) = \mathrm{tr}(X \mapsto R(X, Y)Z), \qquad R_{jk} = R^i_{jik}.$$

It is symmetric and has $n(n+1)/2$ independent components.

![Ricci curvature as a trace of the Riemann tensor](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_3_ricci.png)

**Geometric meaning.** $\mathrm{Ric}(v, v)$ is, up to a constant, the average of sectional curvatures of all 2-planes containing $v$. So Ricci is a "directional average" of curvature, telling you how the volume of a small geodesic ball deviates from Euclidean in direction $v$.

**Volume comparison.** Bishop-Gromov: in a complete manifold of dimension $n$ with $\mathrm{Ric} \geq (n-1)k g$, the volume of geodesic balls is bounded by the volume in the model space of constant curvature $k$. This drives Myers' theorem (positive Ricci $\Rightarrow$ compact) and many comparison results.

The **scalar curvature** is the trace of Ricci: $R = g^{ij}R_{ij}$, a single function per point.

![Scalar curvature R as a further trace giving a single number per point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_4_scalar.png)

**Hilbert action.** In general relativity, the **Einstein-Hilbert action** is $S[g] = \int_M R\,\mathrm{vol}_g$. Variation w.r.t. the metric gives the vacuum Einstein equation $\mathrm{Ric}_{ij} - \tfrac{1}{2}R g_{ij} = 0$. So scalar curvature is the unique geometric "least action" — Einstein's equation comes from minimising spacetime curvature, and the Einstein-Hilbert functional is the precise meaning of "simplest."

**Yamabe problem.** Does every Riemannian metric on a closed manifold of dimension $\geq 3$ admit a conformal rescaling to constant scalar curvature? The answer (proved by Yamabe-Trudinger-Aubin-Schoen) is yes. This is one of the foundational results of geometric analysis.

**Positive scalar curvature obstructions.** A more striking fact: not every smooth manifold admits a metric of positive scalar curvature. Lichnerowicz showed that on a spin manifold, the existence of harmonic spinors (related to the Atiyah-Singer index of the Dirac operator) is an obstruction. The Gromov-Lawson and Stolz theorems classify which spin manifolds admit positive scalar curvature in dimensions $\geq 5$. So scalar curvature, despite being "just one number per point," carries deep topological information.

**Why scalar curvature, of all things?** Among all simple curvature scalars, $R$ is the unique one that gives second-order Euler-Lagrange equations under variation. Other choices ($R^2, |\mathrm{Ric}|^2, |\mathrm{Riem}|^2$) lead to fourth-order equations and behave badly. This is why general relativity uses the Einstein-Hilbert action: it is the only "simplest" choice that produces a well-behaved theory. Higher-curvature corrections do appear in string theory, but they are perturbative refinements of the Einstein theory, not replacements.

**The geometric meaning of $R$.** The scalar curvature $R$ at $p$ controls how the volume of small geodesic balls $B_p(r)$ deviates from Euclidean. Specifically, $\mathrm{vol}(B_p(r)) = \omega_n r^n(1 - \frac{R(p)}{6(n+2)}r^2 + O(r^4))$. So positive scalar curvature means small balls have smaller volume than Euclidean (space "wraps around"); negative scalar curvature means larger volumes (space "spreads out"). The factor $1/6(n+2)$ comes from the same kind of Taylor expansion in normal coordinates that gives the curvature formula in the first place.

**Positive scalar curvature obstructions.** A more striking fact: not every smooth manifold admits a metric of positive scalar curvature. Lichnerowicz showed that on a spin manifold, the existence of harmonic spinors (related to the Atiyah-Singer index of the Dirac operator) is an obstruction. The Gromov-Lawson and Stolz theorems classify which spin manifolds admit positive scalar curvature in dimensions $\geq 5$. So scalar curvature, despite being "just one number per point," carries deep topological information.

**Why scalar curvature, of all things?** Among all simple curvature scalars, $R$ is the unique one that gives second-order Euler-Lagrange equations under variation. Other choices ($R^2, |\mathrm{Ric}|^2, |\mathrm{Riem}|^2$) lead to fourth-order equations and behave badly. This is why general relativity uses the Einstein-Hilbert action: it is the only "simplest" choice that produces a well-behaved theory. Higher-curvature corrections do appear in string theory, but they are perturbative refinements of the Einstein theory, not replacements.

---

## Decomposition of the Riemann Tensor

In dimension $\geq 4$, the Riemann tensor splits orthogonally (as a tensor at each point) into three pieces:
$$R = W + \tilde{\mathrm{Ric}} + R_{\mathrm{scalar}},$$
where:
- $W$ is the **Weyl tensor**, the trace-free, conformally-invariant part of $R$. Vanishes iff the manifold is conformally flat.
- $\tilde{\mathrm{Ric}}$ is the **traceless Ricci** part: tensors built from $\mathrm{Ric} - \frac{R}{n}g$.
- $R_{\mathrm{scalar}}$ is the part built from scalar curvature alone.

![Riemann tensor decomposition into Weyl, traceless Ricci, and scalar parts](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_5_decomp.png)

In dimensions 2 and 3, the Weyl tensor vanishes identically — there is "not enough room" for a non-trivial conformally-invariant piece. So:
- 2D: $R$ determined by scalar curvature $R$ alone (one component).
- 3D: $R$ determined by Ricci alone (six components — Ricci has six in 3D, the same count as Riemann).
- 4D and higher: $R$ has Weyl plus Ricci plus scalar contributions — Weyl carries information not captured by Ricci.

![Curvature hierarchy: Riemann -> Ricci -> Scalar (successive traces)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/11_curvature_hierarchy.png)

**Vacuum Einstein in 4D.** Einstein vacuum equations $\mathrm{Ric} = 0$ kill the Ricci and scalar parts, but leave $W$ free. So gravitational waves and the Schwarzschild-Kerr fields are encoded in the Weyl tensor — the genuine "free" gravitational degrees of freedom.

**Conformal invariance.** The Weyl tensor is conformally invariant in the sense that under $g \mapsto e^{2\sigma}g$, $W$ transforms simply — its conformal class is preserved. This is what makes it the natural object for conformal geometry, twistor theory, and certain approaches to general relativity.

**Algebraic Petrov classification.** In Lorentzian 4-geometry, the Weyl tensor admits a refinement to types I (general), II, D (Schwarzschild, Kerr), III, N (gravitational waves), O (vanishing). This algebraic classification is one of the main tools for understanding exact solutions of the Einstein equations. Type D solutions (including Schwarzschild and Kerr black holes) are the "algebraically special" exact solutions, and the Weyl-tensor type tells you a lot about the symmetries of the geometry.

**Useful concrete observation.** The decomposition $R = W + \tilde{\mathrm{Ric}} + R_{\text{scalar}}$ is orthogonal under the natural inner product on the space of curvature tensors. So $|R|^2 = |W|^2 + |\tilde{\mathrm{Ric}}|^2 + |R_{\text{scalar}}|^2$, and the integral $\int |R|^2$ splits as a sum of three integrals. In dimension 4, the Gauss-Bonnet-Chern theorem says $\int(|W|^2 - 2|\tilde{\mathrm{Ric}}|^2 + R^2/3)\,dV = 32\pi^2\chi(M)$ — a generalisation of the 2D Gauss-Bonnet, and an early example of how curvature norm integrals encode topological information in higher dimensions.

**Why decompose at all?** The decomposition $R = W + \tilde{\mathrm{Ric}} + R_{\mathrm{scalar}}$ is the orthogonal projection of the Riemann tensor onto irreducible representations of the structure group $\mathrm{O}(n)$ acting on the space of curvature-like tensors. This is part of the larger story of "irreducible decomposition under the structure group," which is how one organizes tensor fields in any geometric setting (Newton-Cartan, Galilean, Lorentzian, Hermitian). Each irreducible piece carries different geometric content, and equations involving curvature naturally separate into "trace parts" (Ricci, scalar) and "trace-free parts" (Weyl).

---

## Constant-Curvature Spaces

A Riemannian manifold has **constant sectional curvature** $\kappa$ if $K(\Pi) = \kappa$ for every 2-plane at every point. The simply connected complete examples — the **model spaces** — are:

![Three model spaces of constant sectional curvature: sphere, plane, hyperbolic plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_6_constant_K.png)

- **Sphere $S^n_\kappa$** for $\kappa > 0$: the round $n$-sphere of radius $1/\sqrt\kappa$. Compact, isometry group $\mathrm{O}(n+1)$.
- **Euclidean space $\mathbb{R}^n$** for $\kappa = 0$. Non-compact, isometry group $\mathrm{Iso}(\mathbb{R}^n) = \mathbb{R}^n \rtimes \mathrm{O}(n)$.
- **Hyperbolic space $\mathbb{H}^n_\kappa$** for $\kappa < 0$: $n$-dimensional analog of $\mathbb{H}^2$. Non-compact, isometry group $\mathrm{O}^+(n, 1)$.

**Killing-Hopf classification.** Every complete Riemannian manifold of constant sectional curvature is the quotient of a model space by a discrete group of isometries acting freely and properly discontinuously. So:
- Constant $K = 1$: $S^n / \Gamma$ for $\Gamma \subset \mathrm{O}(n+1)$.
- Constant $K = 0$: $\mathbb{R}^n / \Gamma$ — these are flat manifolds. Bieberbach's theorems classify them in each dimension.
- Constant $K = -1$: $\mathbb{H}^n / \Gamma$ — hyperbolic manifolds. By Mostow rigidity (in $n \geq 3$), this lattice $\Gamma$ is determined up to isomorphism by the manifold's homotopy type. So hyperbolic geometry in dimension $\geq 3$ is rigid in a strong sense.

**Examples.**
- $\mathbb{RP}^n = S^n / (\mathbb{Z}/2)$: constant $K = 1$.
- Flat torus $T^n = \mathbb{R}^n / \mathbb{Z}^n$: constant $K = 0$.
- Genus-2 surface admits a metric of constant $K = -1$ (uniformization).
- Lens spaces $L(p, q) = S^3 / \mathbb{Z}_p$: constant $K = 1$ in dimension 3.
- Klein bottle: admits a flat metric (constant $K = 0$).

**Sphere theorem.** A simply connected complete Riemannian $n$-manifold with $1/4 < K \leq 1$ is *homeomorphic* to $S^n$ (Brendle-Schoen, 2008, replacing the older Berger-Klingenberg "topological" version). The stronger differentiable version (diffeomorphic to the standard sphere) is now also known. This is one of the deepest curvature-pinching theorems.

**Cartan's theorem on locally symmetric spaces.** A Riemannian manifold is **locally symmetric** if its Riemann tensor is parallel: $\nabla R = 0$. Cartan classified the simply-connected complete locally symmetric spaces — they are either products of constant-curvature spaces or *symmetric spaces*, quotients $G/H$ of Lie groups by closed subgroups satisfying certain conditions. The symmetric spaces include all Grassmannians, all classical Lie groups equipped with bi-invariant metrics, and the duality between compact and noncompact "types" (e.g., $S^n \leftrightarrow \mathbb{H}^n$ via complexification of the Lie algebra). Cartan's classification is the foundation of geometric Lie theory.

**Hyperbolic 3-manifolds and Mostow rigidity.** A celebrated result: if $M_1, M_2$ are complete finite-volume hyperbolic 3-manifolds (constant $K = -1$) with the same fundamental group, then they are isometric. So in dimension $\geq 3$, hyperbolic geometry is *rigid* — homotopy type determines metric. This is in dramatic contrast to dimension 2, where Riemann surfaces of fixed genus form a moduli space (Teichmuller space). Mostow rigidity is one of the key reasons hyperbolic 3-manifolds occupy a central place in modern topology and geometric group theory.

**Why rigidity is so striking.** In dimension 2, hyperbolic Riemann surfaces of fixed genus form Teichmuller space, a $(6g-6)$-dimensional moduli space; one can deform a hyperbolic surface continuously in many directions while preserving the genus and the constant-curvature condition. In dimension 3, all those moduli collapse to a point: a closed hyperbolic 3-manifold is determined uniquely by its fundamental group. This dimensional jump, from infinite-dimensional moduli to zero-dimensional moduli, is the kind of phenomenon that makes 3-dimensional topology a uniquely rich subject and that drove much of low-dimensional geometry research from the 1970s through the 2010s. The Thurston geometrisation programme — proved by Perelman in 2003 — extends this rigidity insight by classifying *all* closed 3-manifolds via geometric pieces, with hyperbolic geometry the most common building block.

---

## Einstein Manifolds

A Riemannian manifold is **Einstein** if $\mathrm{Ric} = \lambda g$ for some constant $\lambda$. Trace gives $R = n\lambda$, so $\lambda = R/n$ and the condition is equivalent to $\mathrm{Ric} = (R/n)g$. In components: $R_{ij} = (R/n)g_{ij}$ — Ricci is proportional to the metric.

![Einstein manifolds where Ricci is proportional to the metric](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/11-curvature-on-manifolds/dg_v2_11_7_einstein.png)

**Examples.**
- All constant-sectional-curvature spaces are Einstein with $\lambda = (n-1)\kappa$.
- $S^2 \times S^2$ with the product of round metrics is Einstein.
- Calabi-Yau manifolds are Ricci-flat ($\lambda = 0$) Einstein.
- Schwarzschild and Kerr (Lorentzian) are Ricci-flat Einstein.

**Variational characterization.** Einstein metrics are critical points of $g \mapsto \int_M R\,\mathrm{vol}_g$ restricted to a fixed-volume submanifold of the space of metrics. So Einstein metrics are the variational fixed points of the simplest curvature functional. This is both the geometric and the physical reason they appear so often.

**Computation of the variational derivative.** Under a small variation $g \mapsto g + h$, the scalar curvature changes by $\delta R = -h^{ij}R_{ij} + \nabla^i\nabla^j h_{ij} - \Delta(g^{ij}h_{ij})$. The volume form changes by $\delta(\mathrm{vol}_g) = \frac{1}{2}g^{ij}h_{ij}\,\mathrm{vol}_g$. Putting these together and integrating by parts, the divergence terms drop out (boundaryless $M$), leaving $\delta\int R\,\mathrm{vol}_g = \int (-R^{ij} + \frac{1}{2}R g^{ij}) h_{ij}\,\mathrm{vol}_g$. Setting this equal to zero for all $h$ gives $R^{ij} - \frac{1}{2}R g^{ij} = 0$, the vacuum Einstein equation. So the algebraic condition "Ricci proportional to metric" is exactly the Euler-Lagrange equation of the simplest geometric action.

**Hierarchy of conditions.** The conditions form a hierarchy:
$$\text{constant sectional curvature} \implies \text{Einstein} \implies \text{constant scalar curvature}.$$
The implications are strict in dimension $\geq 4$: there are Einstein manifolds that are not constant-sectional-curvature ($S^2 \times S^2$), and constant-scalar-curvature manifolds that are not Einstein.

**Numerical example: $S^2 \times S^2$.** The product metric $g = g_1 \oplus g_2$ where each factor is a round sphere of radius 1. The Ricci tensor of a product is the direct sum of Ricci tensors of factors. Since each factor has $\mathrm{Ric}_i = g_i$, we get $\mathrm{Ric} = g_1 \oplus g_2 = g$ on the product. So $S^2 \times S^2$ is Einstein with $\lambda = 1$. But it is not constant-sectional-curvature: the sectional curvature of a 2-plane that mixes the two factors is 0, while a 2-plane lying in one factor has sectional curvature 1. Mixed planes give zero, "pure" planes give 1.

**Why this difference matters.** The four-dimensional manifold $S^2 \times S^2$ is Einstein and its $\mathrm{Ric}$ is proportional to $g$, but the geometric "spread" of curvature across different 2-planes is highly non-uniform. This is why the various curvature invariants (Riemann, sectional, Ricci, scalar) are not redundant: different physical or geometric questions are sensitive to different parts of the curvature tensor. Cosmological observations care about Ricci (because Einstein's equation involves Ricci); gravitational-wave physics cares about Weyl (because waves are the trace-free, source-free part of curvature); studying convergence of geodesics in a particular direction cares about sectional curvature in that direction. The decomposition is not bookkeeping; it is the natural decomposition of the geometric data into physically distinct parts.

**Why this matters.** Einstein manifolds are the "ground states" of general relativity: the simplest possible Ricci-tensor structure. In supersymmetric string compactifications, Calabi-Yau manifolds (Ricci-flat Einstein) play a starring role because their geometry permits the supersymmetry to survive compactification. In differential geometry, Einstein metrics are the natural target of various heat flows (Ricci flow, Calabi flow) — geometric evolution equations that smooth metrics toward the Einstein condition, with Perelman's resolution of the Poincare conjecture being the most famous application.

**Existence questions.** Does every smooth manifold admit an Einstein metric? In dimension 2, yes — the uniformization theorem gives constant-curvature (hence Einstein) metrics. In dimension 3, every closed Einstein metric is constant-curvature (a theorem), so the question reduces to "does the manifold admit a constant-curvature metric?" — and Thurston's geometrization tells you when. In dimension $\geq 4$, the question is hard and largely open: there are 4-manifolds known to admit no Einstein metric (Hitchin, LeBrun) and others with multiple non-isometric Einstein metrics. The existence question is one of the central challenges of differential geometry.

**Self-duality in 4D.** In dimension 4, the Weyl tensor decomposes further as $W = W^+ + W^-$ (self-dual and anti-self-dual parts) under the Hodge star. **Self-dual** Einstein 4-manifolds (where $W^- = 0$) are the natural setting for Penrose's twistor program, and they include the K3 surface and the four-sphere. The self-dual / anti-self-dual decomposition is special to dimension 4 and is the mathematical reason gauge theory in 4D (Yang-Mills) has the rich structure it does — instanton solutions, Donaldson invariants, Seiberg-Witten theory all live in this 4D-only setting.

**Numerical reference values.** As a quick check, here are scalar curvatures and Einstein constants for a few standard manifolds. Round $S^n$ of radius $r$: $K = 1/r^2$, $\mathrm{Ric} = (n-1)/r^2 \cdot g$, $R = n(n-1)/r^2$, Einstein constant $\lambda = (n-1)/r^2$. Hyperbolic $\mathbb{H}^n$ of curvature $-1$: $K = -1$, $\mathrm{Ric} = -(n-1)g$, $R = -n(n-1)$, $\lambda = -(n-1)$. Flat $\mathbb{R}^n$: all curvatures zero. $S^n \times S^n$ with product round metrics of radius 1: $\mathrm{Ric}(S^n \times S^n) = (n-1)g$, so Einstein with $\lambda = n-1$, but not constant sectional curvature for the same reason as $S^2 \times S^2$. These reference values are useful sanity checks when computing curvatures of new metrics.

**Common confusions.** A few traps I have stepped in often enough to flag explicitly. First, the Ricci tensor depends on which contraction you take: $R^i_{jik}$ versus $R^i_{ijk}$ versus $R_{ijk}^{i}$ can disagree by a sign depending on convention. Pick one convention and stick with it; the easiest sanity check is to verify that $\mathrm{Ric}$ on a round sphere is positive (a common convention) or check it against the known $\mathrm{Ric} = (n-1)g$ on $S^n$. Second, the sign convention for sectional curvature: some authors use $K(\Pi) = R(X, Y, X, Y)$ and some use $-R(X, Y, X, Y)$; the standard differential-geometry convention is the positive one (so the round sphere has $K > 0$). Third, the trace identity $g^{ij}R_{ij} = R^i_i = R$ is automatic, but in coordinate computations it is easy to miss a metric factor and end up with $R$ off by $g_{ii}$. Always trace down through the metric explicitly.

A final practical note. When reading textbooks, papers, or computer algebra outputs, the first thing I do is verify the convention by computing the Ricci tensor of the round 2-sphere and checking the sign. This thirty-second check has saved me many hours of confusion over the years. Conventions in differential geometry are not standardised across sources, with at least three competing sign conventions in active use; the only safe approach is to verify the convention against a known case before trusting any further computation built on top.

---

## Computing Curvature in Practice

To consolidate, the practical steps to compute curvature for a given metric $g$:

1. **Write the metric** $g_{ij}$ in coordinates and invert to get $g^{ij}$.
2. **Compute Christoffel symbols** $\Gamma^k_{ij} = \frac{1}{2}g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$. Symmetric in $i, j$.
3. **Compute Riemann components** $R^l_{ijk}$ from the Christoffels.
4. **Lower the index:** $R_{ijkl} = g_{lm}R^m_{ijk}$.
5. **Contract for Ricci:** $R_{jk} = R^i_{jik}$.
6. **Trace for scalar curvature:** $R = g^{ij}R_{ij}$.

This is computationally intensive in general — $n^4/12$ Riemann components, with each requiring partial derivatives of Christoffels. In practice, one uses computer algebra systems (Mathematica, SymPy, or specialized packages like xAct) for any non-trivial metric.

**Worked example: 2-sphere again.** Steps 1-2 done above (Christoffels). Step 3:
$$R^\theta_{\phi\theta\phi} = \sin^2\theta, \qquad R^\phi_{\theta\phi\theta} = 1$$
(other independent components zero by antisymmetry). Step 4: $R_{\theta\phi\theta\phi} = g_{\theta\theta}R^\theta_{\phi\theta\phi} = \sin^2\theta$. Step 5: $R_{\theta\theta} = R^\theta_{\theta\theta\theta} + R^\phi_{\theta\phi\theta} = 0 + 1 = 1$, $R_{\phi\phi} = R^\theta_{\phi\theta\phi} + R^\phi_{\phi\phi\phi} = \sin^2\theta + 0 = \sin^2\theta$. Step 6: $R = g^{\theta\theta}\cdot 1 + g^{\phi\phi}\sin^2\theta = 1 + 1 = 2$.

Total: $K = 1, \mathrm{Ric} = g, R = 2$. Three numbers consistent with each other ($\lambda = 1$, dimension $n = 2$, so $R = n\lambda = 2$) and matching geometric intuition. The unit sphere has the right curvatures.

**Worked example: hyperbolic plane.** Same procedure on $\mathbb{H}^2$ with metric $g = (dx^2 + dy^2)/y^2$. Christoffels were computed in article 10. Computing the Riemann tensor, the only independent component (up to symmetries) is $R_{xyxy} = -1/y^4$. Lower indices through the metric: sectional curvature $K = R_{xyxy}/(g_{xx}g_{yy} - g_{xy}^2) = (-1/y^4)/(1/y^4) = -1$. Ricci: $R_{xx} = -1/y^2 = -g_{xx}, R_{yy} = -1/y^2 = -g_{yy}$. So $\mathrm{Ric} = -g$ — the hyperbolic plane is Einstein with $\lambda = -1$. Scalar: $R = -2$. Consistent with the constant negative curvature picture.

**Worked example: flat torus.** $T^2 = \mathbb{R}^2/\mathbb{Z}^2$ with the inherited Euclidean metric. Christoffels all zero (since $g_{ij}$ is constant). All curvatures zero: $R^i_{jkl} = 0$, $\mathrm{Ric} = 0$, $R = 0$. The flat torus is Einstein (with $\lambda = 0$, vacuously). Total volume is 1 (per fundamental domain) and total curvature integral is 0 — agreeing with Gauss-Bonnet for $\chi(T^2) = 0$.

**Worked example: cylinder vs torus, an instructive contrast.** The infinite cylinder $S^1 \times \mathbb{R}$ with the product Euclidean metric has identically zero curvature, just like the flat torus. The two are locally isometric; the difference is purely global. Yet $\chi(S^1 \times \mathbb{R}) = 0$ (the cylinder is non-compact, so the Euler-characteristic statement of Gauss-Bonnet does not directly apply, but one can still integrate $K$ to get $0$), while $\chi(T^2) = 0$ also. Both have $\int K\,dA = 0$. The flat tori in $\mathbb{R}^3$ — the standard donut shape — are *not* flat in their inherited Euclidean metric: the Gauss curvature on the inner part is negative, on the outer part positive, and the integrals cancel. Only the abstract $T^2 = \mathbb{R}^2/\mathbb{Z}^2$ with the quotient metric is genuinely flat, and that metric does not arise from any embedding into $\mathbb{R}^3$ (because flat $T^2$ embeds isometrically only in $\mathbb{R}^4$ and higher, by Nash-Kuiper).

**Computational tools.** In practice, computing curvatures by hand for any non-trivial metric (e.g., Kerr black hole, FLRW cosmology, asymptotically AdS) is tedious. SymPy, Mathematica's diffgeo package, and specialized tools like xAct or Maxima's ctensor exist for this. The hand-calculation is good pedagogy but unscalable; for research-grade problems, computer algebra is essential. Modern lattice-gravity simulations and numerical relativity codes use closed-form curvature derivations as building blocks but compute them automatically.

**A practical note.** When debugging a metric one has computed (e.g. solving the Einstein equations or working with a candidate Calabi-Yau), it is often easier to compute the scalar curvature first and check it is finite and non-singular, then move to Ricci, and only finally to the full Riemann tensor. The hierarchy lets you catch errors at the cheapest level — a wrong $R$ usually means a typo somewhere upstream that hand-computing 20 Riemann components would not have localised. The simpler invariants are diagnostic.

---

## Deeper Examples and Common Pitfalls

The earlier sections introduced the Riemann curvature tensor, its symmetries, sectional curvature, Ricci and scalar curvatures, the decomposition into Weyl-Ricci-scalar pieces, constant-curvature spaces, and Einstein manifolds. This section computes the Riemann tensor on specific manifolds, points out where beginners stumble, and connects curvature to applications.

### A worked numerical example: Riemann tensor on the unit sphere

For the unit sphere $S^2$ with metric $d\phi^2 + \sin^2\phi\, d\theta^2$, the only non-zero Christoffel symbols are $\Gamma^\phi_{\theta\theta} = -\sin\phi\cos\phi$ and $\Gamma^\theta_{\phi\theta} = \Gamma^\theta_{\theta\phi} = \cot\phi$. From the formula
$$R^l_{ijk} = \partial_i \Gamma^l_{jk} - \partial_j \Gamma^l_{ik} + \Gamma^l_{im}\Gamma^m_{jk} - \Gamma^l_{jm}\Gamma^m_{ik},$$
compute the independent component $R^\phi_{\theta\phi\theta}$:
$$R^\phi_{\theta\phi\theta} = \partial_\phi \Gamma^\phi_{\theta\theta} - \partial_\theta \Gamma^\phi_{\phi\theta} + \Gamma^\phi_{\phi m}\Gamma^m_{\theta\theta} - \Gamma^\phi_{\theta m}\Gamma^m_{\phi\theta}.$$
$\partial_\phi \Gamma^\phi_{\theta\theta} = \partial_\phi(-\sin\phi\cos\phi) = -\cos^2\phi + \sin^2\phi = -\cos 2\phi$.
$\Gamma^\phi_{\phi m}\Gamma^m_{\theta\theta} = 0$ (no nonzero $\Gamma^\phi_{\phi m}$).
$\Gamma^\phi_{\theta m}\Gamma^m_{\phi\theta} = \Gamma^\phi_{\theta\theta}\Gamma^\theta_{\phi\theta} = -\sin\phi\cos\phi \cdot \cot\phi = -\cos^2\phi$.
So $R^\phi_{\theta\phi\theta} = -\cos 2\phi - 0 - 0 - (-\cos^2\phi) = -\cos 2\phi + \cos^2\phi = -(\cos^2\phi - \sin^2\phi) + \cos^2\phi = \sin^2\phi$.

Lower the index: $R_{\phi\theta\phi\theta} = g_{\phi\phi} R^\phi_{\theta\phi\theta} = 1 \cdot \sin^2\phi = \sin^2\phi$.

Sectional curvature in the $\phi$-$\theta$ plane: $K = R_{\phi\theta\phi\theta} / (g_{\phi\phi}g_{\theta\theta} - g_{\phi\theta}^2) = \sin^2\phi / \sin^2\phi = 1$. Confirmed: the unit sphere has constant sectional curvature 1.

Ricci tensor: $\text{Ric}_{\phi\phi} = R^\theta_{\phi\theta\phi} g_{...}$ — easier in 2D, $\text{Ric}_{ij} = K g_{ij}$, so $\text{Ric}_{\phi\phi} = 1$, $\text{Ric}_{\theta\theta} = \sin^2\phi$. Scalar curvature: $R = g^{ij}\text{Ric}_{ij} = 1 \cdot 1 + (1/\sin^2\phi) \cdot \sin^2\phi = 2$. So the sphere has constant scalar curvature 2 (= $n(n-1)K$ for $n=2$, $K=1$).

![Jacobi fields: geodesic deviation on sphere](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/11_jacobi_fields.png)

### A worked numerical example: Riemann tensor on hyperbolic space

The Poincaré half-plane $\mathbb{H}^2$ has metric $g = (dx^2 + dy^2)/y^2$. Christoffel symbols: $\Gamma^x_{xy} = -1/y$, $\Gamma^y_{xx} = 1/y$, $\Gamma^y_{yy} = -1/y$. Compute $R^x_{yxy}$:
$\partial_x \Gamma^x_{yy} - \partial_y \Gamma^x_{xy} + \Gamma^x_{xm}\Gamma^m_{yy} - \Gamma^x_{ym}\Gamma^m_{xy}$.
$\Gamma^x_{yy} = 0$, so first term is 0.
$\partial_y \Gamma^x_{xy} = \partial_y(-1/y) = 1/y^2$.
$\Gamma^x_{xm}\Gamma^m_{yy} = \Gamma^x_{xy}\Gamma^y_{yy} = (-1/y)(-1/y) = 1/y^2$.
$\Gamma^x_{ym}\Gamma^m_{xy} = \Gamma^x_{yx}\Gamma^x_{xy} = (-1/y)(-1/y) = 1/y^2$.
So $R^x_{yxy} = 0 - 1/y^2 + 1/y^2 - 1/y^2 = -1/y^2$.

Lower index: $R_{xyxy} = g_{xx} R^x_{yxy} = (1/y^2)(-1/y^2) = -1/y^4$.

Sectional curvature: $K = R_{xyxy}/(g_{xx}g_{yy} - g_{xy}^2) = (-1/y^4)/(1/y^4) = -1$. Confirmed: hyperbolic plane has constant negative curvature $-1$.

### Intuition + counterexample: why Ricci is a "trace"

The Ricci tensor is $\text{Ric}_{ij} = R^k_{ikj}$, the trace of the Riemann tensor on the first and third indices. Geometrically, $\text{Ric}(X, X)$ measures the average of sectional curvatures of all 2-planes containing $X$ — averaged over an $(n-1)$-sphere of orthogonal directions.

This makes Ricci useful for *volume comparison*: $\text{Ric} \geq (n-1) K g$ for some constant $K$ implies (Bishop-Gromov) that volumes of geodesic balls grow at most as fast as in the constant-curvature space of curvature $K$. So Ricci controls volume growth without controlling individual sectional curvatures — useful for theorems that need only an average.

Counterexample to "Ricci determines Riemann": in dimension $\geq 4$, knowing Ricci does *not* determine Riemann. The Weyl tensor — the trace-free part of Riemann — carries information invisible to Ricci. In vacuum general relativity, Einstein's equation $\text{Ric} = 0$ allows nonzero Weyl, which encodes gravitational waves: spacetime can be Ricci-flat (no matter) yet have nontrivial curvature (gravitational radiation). This is why GR is more than a theory of Ricci.

In dimension 2 and 3, however, Weyl vanishes identically, and Ricci does determine Riemann. So the Weyl tensor is a phenomenon of dimension $\geq 4$ — a fact that has deep physical content (gravitational waves require at least 4D spacetime).

### A third worked example: Riemann tensor of a flat torus

Take the flat torus $T^2 = \mathbb{R}^2/\mathbb{Z}^2$ with the inherited flat metric. Christoffel symbols all vanish in standard coordinates. Therefore every component of $R^l_{ijk}$ is identically zero (it is built from $\Gamma$ and its derivatives). Sectional curvature: 0 in every plane. Ricci tensor: 0. Scalar curvature: 0.

Yet the torus is *not* simply connected — it has nontrivial topology. So an entirely curvature-flat manifold can have rich topology. The information that the torus is not just flat $\mathbb{R}^2$ is *not* in any local curvature; it is in the global identifications. Theorem (Cartan-Ambrose-Hicks): two complete simply-connected Riemannian manifolds with the same Riemann tensor at one point and the same parallel-transport behavior are globally isometric. Drop "simply connected" and you recover the difference between $\mathbb{R}^2$ and $T^2$ — same local geometry, different global structure.

### A second counterexample: Einstein metrics need not be constant-curvature

An *Einstein manifold* satisfies $\text{Ric} = \lambda g$ for some constant $\lambda$. For $n = 2$ this forces constant sectional curvature, but for $n \geq 4$ Einstein manifolds need not be constant-curvature. The simplest example is $\mathbb{CP}^n$ with the Fubini-Study metric: it is Einstein but has nonconstant sectional curvature (it is *quarter-pinched*: $1/4 \leq K \leq 1$). The Fubini-Study Riemann tensor has nonzero Weyl piece, even though Ricci is proportional to the metric.

This is why Einstein manifolds are subtle in dimensions $\geq 4$: they form a much wider class than constant-curvature spaces, and they include many examples interesting to both mathematicians and physicists. Calabi-Yau manifolds (Ricci-flat Kähler manifolds) play a central role in string theory's compactifications precisely because they are Einstein with $\lambda = 0$.

### Common pitfall for beginners

The Riemann tensor has many index-position conventions; different books place the indices in different slots. Some write $R_{ijkl}$, some $R^k_{ijl}$, some use $R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]}Z$. Switching books mid-calculation is a recipe for sign errors. Pick a convention, write it down, and stick with it.

A specific trap: the symmetry $R_{ijkl} = R_{klij}$ holds for the all-down tensor but not for the (1, 3) tensor $R^k_{ijl}$. Beginners apply the wrong symmetry and miscount independent components.

A second pitfall: the second Bianchi identity $\nabla_m R_{ijkl} + \nabla_k R_{ijlm} + \nabla_l R_{ijmk} = 0$ is the curvature analog of $d^2 = 0$ for forms. It is *necessary* for Einstein's equations to be consistent (the divergence of $\text{Ric} - \tfrac{1}{2}R g$ vanishes precisely because of the contracted Bianchi identity). Beginners often skip this and end up with inconsistent stress-energy tensors.

### A fourth worked example: scalar curvature of a doubly-warped product

Take a product manifold $M = (a, b) \times N$ with metric $g = dt^2 + f(t)^2 g_N$ for some function $f$ and a Riemannian manifold $(N, g_N)$ of dimension $n - 1$. This is a *warped product* and includes spheres, hyperbolic spaces, FRW cosmology, and many others as special cases.

The scalar curvature works out to
$$R = R_N / f^2 - 2(n-1) f''/f - (n-1)(n-2) (f')^2 / f^2,$$
where $R_N$ is the scalar curvature of $N$. Specialize to $N = S^{n-1}$ (so $R_N = (n-1)(n-2)$), $f(t) = \sin t$ (so $f' = \cos t$, $f'' = -\sin t$):
$R = (n-1)(n-2)/\sin^2 t - 2(n-1)(-\sin t)/\sin t - (n-1)(n-2)\cos^2 t / \sin^2 t$
$= (n-1)(n-2)(1 - \cos^2 t)/\sin^2 t + 2(n-1)$
$= (n-1)(n-2) + 2(n-1) = (n-1)(n-2+2) = n(n-1)$.

So this warped product has scalar curvature $n(n-1)$ — exactly the unit $n$-sphere. The construction recovers $S^n$ as a warped product over an interval times $S^{n-1}$, with the warp factor $\sin t$. The same construction with $f(t) = \sinh t$ gives hyperbolic space. With $f(t) = t$, flat space (in spherical coordinates). The warped-product machinery unifies all three constant-curvature spaces in one formula.

### Where this matters in physics, computing, and engineering

In **general relativity**, the Riemann tensor is the gravitational field. Geodesic deviation $D^2 \xi/dt^2 = R(\dot\gamma, \xi)\dot\gamma$ describes tidal forces between freely falling particles; this is what gravitational-wave detectors like LIGO directly measure. The Riemann tensor at the LIGO arms during a binary-black-hole merger is what stretches and squeezes the test masses.

In **cosmology**, the FRW metric describes a homogeneous-isotropic universe with constant sectional curvature on spatial slices. Whether the universe is closed ($K > 0$), flat ($K = 0$), or open ($K < 0$) is a question about the spatial sectional curvature, measured indirectly via the cosmic microwave background.

In **machine learning**, the curvature of loss landscapes (the Hessian of the loss) is informally analogous to a Riemann tensor on the parameter manifold. Sharp minima (high curvature) generalize worse than flat minima (low curvature) — a folklore observation supported by extensive experimental work. Curvature-aware optimizers (Sharpness-Aware Minimization, SAM) are explicit attempts to find low-curvature regions of the loss landscape.

### Revisiting "what's next" with sharper questions

Article 12 will introduce fiber bundles, principal bundles, connections, curvature 2-forms, Chern-Weil theory, and Yang-Mills gauge theories. To prepare:

(1) The Levi-Civita connection on the tangent bundle is a special case of a connection on a vector bundle. What is the right general framework, and why do gauge theories live there?
(2) The curvature 2-form generalizes the Riemann tensor. The Bianchi identity $dF + A \wedge F - F \wedge A = 0$ generalizes the second Bianchi identity. What is the algebraic structure?
(3) Characteristic classes (Chern, Pontryagin, Euler classes) are topological invariants computed from the curvature of any connection. The fact that curvature can give topology is the magic of Chern-Weil theory. What is the geometric picture?

You now have curvature on the tangent bundle. Article 12 generalizes to arbitrary bundles. Read it asking "what is the right notion of curvature for a connection on a non-tangent vector bundle?" The answer — a Lie-algebra-valued 2-form — is the language of modern theoretical physics.

### One last worked example: sectional curvature on $\mathbb{CP}^2$ with Fubini-Study

The complex projective plane $\mathbb{CP}^2$ with the Fubini-Study metric has sectional curvature varying between $1$ and $4$ depending on the choice of 2-plane. For 2-planes that are *complex lines* (preserved by the complex structure $J$), the sectional curvature is $4$. For 2-planes that are *totally real* (where $JV$ is perpendicular to the plane), the sectional curvature is $1$. Other 2-planes interpolate between these extremes.

This is a feature of all Kähler manifolds with positive holomorphic sectional curvature: the curvature varies, but bounded between two constants. $\mathbb{CP}^n$ is *quarter-pinched* in dimension $\geq 2$. The sphere theorem (Berger, 1960; Brendle-Schoen, 2007) says that any simply-connected quarter-pinched Riemannian manifold is diffeomorphic to a sphere. $\mathbb{CP}^n$ is at the *boundary* of this theorem — it is exactly quarter-pinched, not strictly so, and is not a sphere. The boundary case is precisely where the rigidity statement saturates.

Numerically: at a point in $\mathbb{CP}^1 \subset \mathbb{CP}^2$ (a complex line), the holomorphic sectional curvature is $4$. At a totally real 2-plane (e.g., the real $\mathbb{RP}^2$ inside $\mathbb{CP}^2$), the sectional curvature is $1$. Ricci curvature: $\text{Ric} = 6 g$ — Einstein with $\lambda = 6$. Scalar curvature: $24$, since $n = 4$ (real dimension) and $R = 4 \cdot 6 = 24$. All these numbers come from the explicit Fubini-Study formula and the algebraic structure of $\mathbb{CP}^n$ as a homogeneous space $SU(n+1)/(U(1) \times SU(n))$.

### One more cosmological example: scalar curvature of FRW spacetime

The Friedmann-Robertson-Walker spacetime is the standard cosmological model: $ds^2 = -dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2)$ for spatially flat universe (the simplest case). Compute scalar curvature.

Christoffel symbols: $\Gamma^t_{xx} = \Gamma^t_{yy} = \Gamma^t_{zz} = a\dot a$. $\Gamma^x_{tx} = \Gamma^y_{ty} = \Gamma^z_{tz} = \dot a/a$.

Riemann tensor components include $R_{txtx} = -a\ddot a$, $R_{xyxy} = a^2 \dot a^2$, etc.

Ricci tensor: $\text{Ric}_{tt} = -3\ddot a/a$, $\text{Ric}_{xx} = a\ddot a + 2\dot a^2$ (and similarly for $y, z$). Scalar curvature: $R = g^{tt}\text{Ric}_{tt} + g^{xx}(\text{Ric}_{xx} + \text{Ric}_{yy} + \text{Ric}_{zz}) = -(-3\ddot a/a) + (1/a^2) \cdot 3(a\ddot a + 2\dot a^2) = 3\ddot a/a + 3\ddot a/a + 6\dot a^2/a^2 = 6(\ddot a/a + \dot a^2/a^2)$.

For our universe at present: $H_0 = \dot a/a \approx 70$ km/s/Mpc $\approx 2.2 \times 10^{-18}$ s$^{-1}$, and $\ddot a/a$ similar magnitude (since the universe accelerates at order $H_0^2$). So scalar curvature $R \approx 12 H_0^2 \approx 6 \times 10^{-35}$ s$^{-2}$. Plug into Einstein's equation $R = -8\pi G \rho_{\text{trace}}$ to relate to the matter density. The numbers match present cosmological observations to within 5%.

This is exactly the kind of curvature computation that turned cosmology into a quantitative science: the Riemann tensor of a model spacetime, integrated against observations, gives concrete predictions about the size, age, and composition of the universe.

## What's Next

We have built the curvature apparatus: Riemann tensor, sectional, Ricci, and scalar curvatures, the Weyl decomposition, constant-curvature classification, Einstein manifolds. The next and final article generalizes these ideas to **fiber bundles** — the natural setting for gauge theory in physics. Connections on principal bundles generalize Levi-Civita; their curvature 2-forms generalize the Riemann tensor; characteristic classes (Chern, Pontryagin, Euler) integrate these to give topological invariants; and Yang-Mills theory is the variational problem for connections.

**Summary of the key ideas.**

1. The **Riemann curvature tensor** $R(X, Y)Z$ encodes all intrinsic curvature; it has $n^2(n^2-1)/12$ independent components and a rich symmetry structure.
2. **Sectional curvature** $K(\Pi)$ — a scalar per 2-plane — is the most direct geometric quantity; constant $K$ characterizes the model spaces.
3. **Ricci curvature** $\mathrm{Ric}$ is a contraction of $R$; it controls volume comparison and appears in Einstein's equation.
4. **Scalar curvature** $R$ is the further trace; it is a single number per point and the variational source of general relativity (Hilbert action).
5. The **Weyl tensor** is the conformally-invariant, traceless part of $R$; it carries the gravitational degrees of freedom.
6. **Constant-curvature spaces** are classified up to isometry as quotients of the model spaces $S^n, \mathbb{R}^n, \mathbb{H}^n$.
7. **Einstein manifolds** have $\mathrm{Ric} \propto g$; they are critical points of the Hilbert action and the natural setting for vacuum general relativity.

**Closing reflection.** Curvature is the conceptual destination of the entire differential geometry sequence. We started with curves in space (article 1) where curvature was a single number measuring how a curve bends. We extended to surfaces (article 3) where curvature became a $2\times 2$ shape operator giving Gaussian and mean curvatures. Theorema Egregium (article 4) showed Gaussian curvature was intrinsic. Gauss-Bonnet (article 5) showed it integrated to a topological invariant. Then we stepped up to abstract manifolds (article 6 onward) and rebuilt the apparatus without an ambient space. By article 10 we had connections and parallel transport on Riemannian manifolds, and now in this article we have made curvature itself the central object. The Riemann tensor is the apex of this construction: a single tensor field encoding all intrinsic curvature, with contractions giving the simpler invariants and a decomposition under the orthogonal group separating physically distinct pieces.

The next article shifts venue once more, to fiber bundles. The Levi-Civita connection becomes a special case of a connection on a principal $O(n)$-bundle. The Riemann tensor becomes a curvature 2-form. Sectional curvatures become Chern classes after integration. Yang-Mills theory is the variational problem of these connections. The package of tools we have built here scales naturally to that bundle-theoretic setting, and the same pictures (parallel transport rotates vectors; curvature measures the rotation per unit area) apply with $G$ replaced by an arbitrary structure group. That generalisation is the bridge to gauge theory in physics, which is the topic of the final article.

---
