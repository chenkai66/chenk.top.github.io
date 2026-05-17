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
lang: en
mathjax: true
description: "The Gauss map and shape operator capture how a surface bends in space — principal, Gaussian, and mean curvatures classify every point as elliptic, hyperbolic, or parabolic."
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "differential-geometry-3"
---

A curve bends in one direction at a time — its curvature is a single number. A surface, by contrast, can bend differently in every tangential direction simultaneously. To quantify this richer phenomenon, we need a linear map that encodes the rate at which the surface normal rotates as we slide along the surface. That map is the **shape operator**, and its eigenvalues — the principal curvatures — give us the complete local bending picture.

This article develops the shape operator from the Gauss map, extracts principal, Gaussian, and mean curvatures, and classifies surface points by the sign of Gaussian curvature. Throughout, we work with regular parametrized surfaces $\mathbf{r}(u,v)$ in $\mathbb{R}^3$ and assume sufficient differentiability.

---

## How Surfaces Bend: The Gauss Map

For a curve in the plane, the unit normal $\mathbf{n}$ rotates as we travel along the curve, and the speed of that rotation is the curvature $\kappa$. The same idea generalizes to surfaces, but now the normal lives on the unit sphere rather than the unit circle.

**Definition.** Let $S$ be a regular oriented surface in $\mathbb{R}^3$. The **Gauss map** is the smooth function

![Gaussian curvature: positive (sphere), zero (cylinder), negative (saddle)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_fig3_curvature.png)


$$N : S \to S^2, \qquad p \mapsto N(p),$$

where $N(p)$ is the unit normal to $S$ at $p$, and $S^2$ is the unit sphere in $\mathbb{R}^3$.

For a parametrized surface $\mathbf{r}(u,v)$, the unit normal is

$$N = \frac{\mathbf{r}_u \times \mathbf{r}_v}{|\mathbf{r}_u \times \mathbf{r}_v|}.$$

The Gauss map sends each point of $S$ to the corresponding point on the sphere. The derivative $dN_p$ of the Gauss map at $p$ is a linear map from the tangent plane $T_pS$ to itself (since $T_{N(p)}S^2$ is parallel to $T_pS$ — both are perpendicular to $N(p)$).

**Example: the sphere.** For $S^2(R)$ of radius $R$ centered at the origin, $N(p) = p/R$. Then

$$dN_p(v) = v/R$$

for every tangent vector $v \in T_pS$. The Gauss map simply scales every tangent direction uniformly by $1/R$.

**Example: the cylinder.** For the cylinder $x^2 + y^2 = R^2$, the unit normal is $N = (x/R, y/R, 0)$. Moving in the $z$-direction does not change $N$ at all, while moving around the cylinder changes $N$ at rate $1/R$. The Gauss map compresses the cylinder onto a great circle of $S^2$, collapsing an entire dimension of the surface.

**Example: the plane.** $N$ is constant, so $dN_p = 0$ everywhere. The plane has zero curvature in every direction — exactly what we expect. The Gauss map sends the entire plane to a single point on $S^2$.

The key insight is that the derivative $dN_p$ measures how fast the normal turns as we move on the surface. Directions where $dN_p$ is large correspond to directions of high curvature; directions where it vanishes correspond to flat directions.

**Area distortion.** The Gauss map sends small patches of $S$ to small patches of $S^2$. The ratio of the image area to the source area (with sign, accounting for orientation) converges to $\det(dN_p)$ as the patch shrinks to a point. For the sphere, every patch is mapped to a patch of equal angular size, scaled by $1/R^2$. For the cylinder, an entire strip of finite area collapses onto a curve of zero area on $S^2$ — the cylinder's Gauss image is one-dimensional, reflecting the fact that one principal curvature vanishes.

This area ratio interpretation will reappear when we define Gaussian curvature: $K = \det(L_p) = \det(-dN_p)$, which is exactly the signed area magnification factor of the Gauss map.

---

## The Shape Operator (Weingarten Map)

The derivative of the Gauss map has a sign convention that makes it convenient to work with the negative.

**Definition.** The **shape operator** (or **Weingarten map**) at $p \in S$ is the linear map

$$L_p = -dN_p : T_pS \to T_pS.$$

The minus sign is chosen so that the shape operator of a sphere of radius $R$ is $(1/R) \cdot \text{Id}$, matching the intuition that convex surfaces have positive curvature.

### Self-adjointness

The shape operator is self-adjoint with respect to the first fundamental form. That is, for all $v, w \in T_pS$,

$$\langle L_p(v), w \rangle = \langle v, L_p(w) \rangle.$$

**Proof.** Write $v = a\mathbf{r}_u + b\mathbf{r}_v$ and $w = c\mathbf{r}_u + d\mathbf{r}_v$. Since $\langle N, \mathbf{r}_u \rangle = 0$ and $\langle N, \mathbf{r}_v \rangle = 0$, differentiating these identities gives

$$\langle N_u, \mathbf{r}_v \rangle + \langle N, \mathbf{r}_{uv} \rangle = 0, \qquad \langle N_v, \mathbf{r}_u \rangle + \langle N, \mathbf{r}_{uv} \rangle = 0.$$

Therefore $\langle N_u, \mathbf{r}_v \rangle = \langle N_v, \mathbf{r}_u \rangle$, which is exactly the symmetry condition $\langle dN(\mathbf{r}_u), \mathbf{r}_v \rangle = \langle dN(\mathbf{r}_v), \mathbf{r}_u \rangle$. Since $L = -dN$, the same symmetry holds for $L$. $\square$

This self-adjointness is crucial: it guarantees that the shape operator has real eigenvalues and orthogonal eigenvectors (by the spectral theorem for symmetric linear maps on a finite-dimensional inner product space). Without this property, the entire theory of principal curvatures would collapse — we would have no guarantee that the extremal curvature directions exist or that they are perpendicular.

**Remark on sign conventions.** Different textbooks use different conventions. Some define $L = dN$ (without the minus sign), which makes the shape operator of a sphere equal to $-(1/R)\,\text{Id}$. Others define the second fundamental form with an opposite sign. The key structural fact — self-adjointness — is independent of sign convention. In this series we follow the convention where convex surfaces have positive principal curvatures.

### The second fundamental form

The **second fundamental form** $\mathrm{II}$ is the bilinear form associated with the shape operator:

$$\mathrm{II}(v, w) = \langle L_p(v), w \rangle = -\langle dN_p(v), w \rangle.$$

In coordinates, if $v = du\,\mathbf{r}_u + dv\,\mathbf{r}_v$, then

$$\mathrm{II} = e\,du^2 + 2f\,du\,dv + g\,dv^2,$$

where the coefficients are

$$e = -\langle N_u, \mathbf{r}_u \rangle = \langle N, \mathbf{r}_{uu} \rangle, \quad f = -\langle N_u, \mathbf{r}_v \rangle = \langle N, \mathbf{r}_{uv} \rangle, \quad g = -\langle N_v, \mathbf{r}_v \rangle = \langle N, \mathbf{r}_{vv} \rangle.$$

The second fundamental form measures how the surface deviates from its tangent plane — it is the quadratic approximation of the signed distance from the tangent plane as we move in the tangent direction.

**Taylor expansion interpretation.** If we expand the surface near a point $p$ in normal coordinates (where $p$ is at the origin, the tangent plane is the $xy$-plane, and the normal is the $z$-axis), then

$$z(x,y) = \frac{1}{2}(ex^2 + 2fxy + gy^2) + O(|(x,y)|^3).$$

The second fundamental form captures the leading-order deviation of the surface from its tangent plane. At an elliptic point, this quadratic form is definite (positive or negative), so the surface lies entirely on one side of the tangent plane near $p$. At a hyperbolic point, the form is indefinite, so the surface crosses the tangent plane — creating a saddle shape. At a parabolic point, the form is degenerate, and the surface "touches" the tangent plane along a line direction.

### Matrix representation

In the basis $\{\mathbf{r}_u, \mathbf{r}_v\}$, the shape operator has matrix representation

$$[L] = \begin{pmatrix} E & F \\ F & G \end{pmatrix}^{-1} \begin{pmatrix} e & f \\ f & g \end{pmatrix},$$

where $E, F, G$ are the first fundamental form coefficients and $e, f, g$ are the second fundamental form coefficients. This follows from the relation $\mathrm{II}(v, w) = \mathrm{I}(Lv, w)$.

**Explicit computation for a graph surface.** Let $z = f(x,y)$, parametrized by $\mathbf{r}(x,y) = (x, y, f(x,y))$. Then

$$E = 1+f_x^2, \quad F = f_xf_y, \quad G = 1+f_y^2,$$

and the unit normal is $N = (-f_x, -f_y, 1)/\sqrt{1+f_x^2+f_y^2}$. The second form coefficients are

$$e = \frac{f_{xx}}{\sqrt{1+f_x^2+f_y^2}}, \quad f = \frac{f_{xy}}{\sqrt{1+f_x^2+f_y^2}}, \quad g = \frac{f_{yy}}{\sqrt{1+f_x^2+f_y^2}}.$$

At a critical point ($f_x = f_y = 0$), this simplifies dramatically: $E = G = 1$, $F = 0$, and $e = f_{xx}$, $f = f_{xy}$, $g = f_{yy}$. The shape operator matrix is just the Hessian matrix of $f$, and the principal curvatures are the eigenvalues of the Hessian. This makes the local geometry of a graph surface particularly transparent at critical points.

---

## Principal Curvatures and Principal Directions

Since $L_p$ is self-adjoint, the spectral theorem guarantees real eigenvalues and (when distinct) orthogonal eigenvectors.

**Definition.** The eigenvalues $\kappa_1, \kappa_2$ of the shape operator $L_p$ are the **principal curvatures** at $p$. The corresponding eigenvectors $\mathbf{e}_1, \mathbf{e}_2$ are the **principal directions**.

The principal curvatures are the maximum and minimum normal curvatures over all tangential directions. More precisely, the normal curvature in direction $v \in T_pS$ (with $|v| = 1$) is

$$\kappa_n(v) = \mathrm{II}(v, v) = \langle L(v), v \rangle,$$

and by the min-max characterization of eigenvalues of a symmetric operator,

$$\kappa_1 = \min_{|v|=1} \kappa_n(v), \qquad \kappa_2 = \max_{|v|=1} \kappa_n(v),$$

(or vice versa, depending on convention). The extrema are achieved precisely along the principal directions.

**Euler's formula.** If $\theta$ is the angle between a unit tangent vector $v$ and the first principal direction $\mathbf{e}_1$, then

$$\kappa_n(v) = \kappa_1 \cos^2\theta + \kappa_2 \sin^2\theta.$$

This is immediate from expanding $v = \cos\theta\,\mathbf{e}_1 + \sin\theta\,\mathbf{e}_2$ and using the orthogonality of eigenvectors.

Euler's formula has a beautiful geometric interpretation: as we rotate the tangent direction from $\mathbf{e}_1$ to $\mathbf{e}_2$, the normal curvature interpolates smoothly between $\kappa_1$ and $\kappa_2$ via the $\cos^2/\sin^2$ weighting. The curve $\theta \mapsto \kappa_n(\theta)$ traces out an ellipse (or a circle when $\kappa_1 = \kappa_2$) known as **Dupin's indicatrix**. At an elliptic point, the indicatrix is an ellipse; at a hyperbolic point, it consists of two conjugate hyperbolas.

**Computing principal curvatures.** Given the matrix representation of $L$, the principal curvatures are the eigenvalues:

$$\kappa_{1,2} = \frac{\text{tr}(L) \pm \sqrt{\text{tr}(L)^2 - 4\det(L)}}{2} = H \pm \sqrt{H^2 - K},$$

where $H = (\kappa_1+\kappa_2)/2$ is the mean curvature and $K = \kappa_1\kappa_2$ is the Gaussian curvature. This formula shows that $H$ and $K$ together determine the principal curvatures (up to labeling).

**Example: the sphere** $S^2(R)$. We computed $L = (1/R)\,\text{Id}$, so both principal curvatures equal $1/R$. Every direction is a principal direction — the sphere is the most symmetric surface.

**Example: the cylinder** $x^2 + y^2 = R^2$. The shape operator has eigenvalue $1/R$ in the circumferential direction and $0$ in the axial direction. One principal curvature is $1/R$, the other is $0$.

**Example: the saddle surface** $z = xy$. At the origin, the shape operator has eigenvalues $+1$ and $-1$ (after suitable normalization). The surface curves upward in one direction and downward in the orthogonal direction.

---

## Gaussian Curvature $K = \kappa_1 \kappa_2$

**Definition.** The **Gaussian curvature** at $p$ is the product of the principal curvatures:

$$K = \kappa_1 \kappa_2 = \det(L_p).$$

Since $K = \det(L)$, we can compute it from the first and second fundamental forms:

$$K = \frac{eg - f^2}{EG - F^2}.$$

This formula is extremely useful for computations: we never need to find the principal curvatures individually.

### Geometric interpretation via the Gauss map

Gaussian curvature has a beautiful geometric interpretation. The Gauss map $N: S \to S^2$ sends a small patch of area $dA$ on $S$ to a patch of area $dA'$ on $S^2$. The ratio of these areas (with sign) is

$$K = \lim_{\text{patch} \to p} \frac{\text{signed area of } N(\text{patch})}{\text{area of patch}}.$$

This is because $\det(dN_p) = \det(-L_p) = \det(L_p) = K$ (the sign works out since $L$ is $2 \times 2$).

### Sign interpretation

The sign of $K$ carries deep geometric information:

- **$K > 0$** (elliptic point): $\kappa_1$ and $\kappa_2$ have the same sign. The surface is locally convex — it lies entirely on one side of its tangent plane. The Gauss map preserves orientation locally. Examples: sphere, ellipsoid.

- **$K < 0$** (hyperbolic point): $\kappa_1$ and $\kappa_2$ have opposite signs. The surface is saddle-shaped — it crosses its tangent plane. The Gauss map reverses orientation. Examples: saddle surface $z = xy$, hyperboloid of one sheet.

- **$K = 0$** (parabolic or flat point): at least one principal curvature vanishes. The surface is locally like a cylinder (one flat direction) or a plane (both flat). Examples: cylinder, cone (away from apex), plane.

### Explicit computations

**Sphere** $S^2(R)$: $\kappa_1 = \kappa_2 = 1/R$, so $K = 1/R^2$. The curvature is constant and positive, as expected for a uniformly convex surface.

**Cylinder** $x^2+y^2=R^2$: $\kappa_1 = 1/R$, $\kappa_2 = 0$, so $K = 0$. This makes geometric sense: a cylinder can be unrolled onto a plane without stretching.

**Saddle surface** $z = x^2 - y^2$: parametrize as $\mathbf{r}(x,y) = (x, y, x^2 - y^2)$. At the origin,

$$\mathbf{r}_x = (1,0,0), \quad \mathbf{r}_y = (0,1,0), \quad N = (0,0,1),$$
$$\mathbf{r}_{xx} = (0,0,2), \quad \mathbf{r}_{xy} = (0,0,0), \quad \mathbf{r}_{yy} = (0,0,-2).$$

So $E=1, F=0, G=1$ and $e = 2, f = 0, g = -2$. Thus

$$K = \frac{eg - f^2}{EG - F^2} = \frac{(2)(-2) - 0}{1 \cdot 1 - 0} = -4.$$

The negative Gaussian curvature confirms the saddle-point geometry.

**Paraboloid** $z = x^2 + y^2$: parametrize as $\mathbf{r}(x,y) = (x, y, x^2+y^2)$. At the origin,

$$E = 1, \; F = 0, \; G = 1, \quad e = 2, \; f = 0, \; g = 2.$$

Thus $K = (2)(2)/1 = 4 > 0$. Both principal curvatures are positive — this is an elliptic point, and the surface is bowl-shaped.

Away from the origin, the computation is more involved because the first fundamental form coefficients are no longer trivial: $E = 1 + 4x^2$, $F = 4xy$, $G = 1 + 4y^2$. The general formula still applies but yields a position-dependent $K$ that decreases as we move away from the origin — the paraboloid becomes flatter far from the vertex.

**Torus.** Parametrize the torus with major radius $R$ and minor radius $r$ as

$$\mathbf{r}(u,v) = \big((R + r\cos v)\cos u,\; (R + r\cos v)\sin u,\; r\sin v\big).$$

Computing the fundamental form coefficients:

$$E = (R + r\cos v)^2, \quad F = 0, \quad G = r^2,$$
$$e = (R + r\cos v)\cos v, \quad f = 0, \quad g = r.$$

Therefore

$$K = \frac{eg - f^2}{EG - F^2} = \frac{r(R + r\cos v)\cos v}{r^2(R + r\cos v)^2} = \frac{\cos v}{r(R + r\cos v)}.$$

On the outer equator ($v = 0$): $K = 1/[r(R+r)] > 0$ — elliptic. On the inner equator ($v = \pi$): $K = -1/[r(R-r)] < 0$ — hyperbolic. On the top and bottom circles ($v = \pi/2, 3\pi/2$): $K = 0$ — parabolic. The torus exhibits all three types of points.

**Total curvature of the torus.** Integrating $K$ over the entire torus:

$$\iint K\,dA = \int_0^{2\pi}\int_0^{2\pi} \frac{\cos v}{r(R+r\cos v)} \cdot r(R+r\cos v)\,du\,dv = \int_0^{2\pi}\int_0^{2\pi} \cos v\,du\,dv = 2\pi \cdot 0 = 0.$$

The total Gaussian curvature vanishes. This is not a coincidence — the Gauss-Bonnet theorem (Article 5) tells us that $\int K\,dA = 2\pi\chi(T^2) = 0$ since the Euler characteristic of the torus is zero. The positive curvature on the outside exactly cancels the negative curvature on the inside.

---

## Mean Curvature $H$ and Minimal Surfaces

**Definition.** The **mean curvature** at $p$ is the average of the principal curvatures:

$$H = \frac{\kappa_1 + \kappa_2}{2} = \frac{1}{2}\operatorname{tr}(L_p).$$

In terms of the fundamental form coefficients:

$$H = \frac{eG - 2fF + gE}{2(EG - F^2)}.$$

While Gaussian curvature is intrinsic (as we'll see in Article 4), mean curvature is extrinsic — it depends on how the surface is embedded in space.

### Minimal surfaces

A **minimal surface** is one with $H = 0$ everywhere. The name comes from the calculus of variations: minimal surfaces are critical points of the area functional.

**Euler-Lagrange connection.** Consider a surface $z = f(x,y)$ over a domain $D \subset \mathbb{R}^2$. The area functional is

$$A[f] = \iint_D \sqrt{1 + f_x^2 + f_y^2}\,dx\,dy.$$

The Euler-Lagrange equation for critical points of $A$ is

$$(1 + f_y^2)f_{xx} - 2f_xf_yf_{xy} + (1 + f_x^2)f_{yy} = 0,$$

which is precisely the condition $H = 0$ for a graph surface.

**Examples of minimal surfaces:**

1. **The plane** — trivially $H = 0$ since both principal curvatures are zero.

2. **The catenoid** — the surface of revolution generated by the catenary $y = a\cosh(z/a)$. It is the only minimal surface of revolution (besides the plane).

3. **The helicoid** — parametrized by $\mathbf{r}(u,v) = (v\cos u, v\sin u, au)$. It is the only ruled minimal surface (besides the plane).

The catenoid and helicoid are in fact related by a continuous one-parameter family of isometric minimal surfaces — a beautiful result in the theory. Explicitly, the family is given by

$$\mathbf{r}_\theta(u,v) = \cos\theta\,\mathbf{r}_{\text{cat}}(u,v) + \sin\theta\,\mathbf{r}_{\text{hel}}(u,v),$$

where $\theta$ ranges from $0$ (catenoid) to $\pi/2$ (helicoid). Every member of this family is a minimal surface, and the induced metric is independent of $\theta$ — the deformation is an isometry at every stage.

4. **Scherk's surface** — given implicitly by $e^z = \cos x/\cos y$, or equivalently $z = \log(\cos y) - \log(\cos x)$. This is a doubly periodic minimal surface, the first non-trivial example discovered after the catenoid and helicoid.

**Physical realization.** Minimal surfaces arise as soap films spanning wire frames. The soap film minimizes its area (due to surface tension), so it satisfies $H = 0$ at every point not touching the wire.

Mean curvature also arises in the **Young-Laplace equation** from fluid mechanics: the pressure difference across a thin membrane is proportional to the mean curvature:

$$\Delta p = 2\gamma H,$$

where $\gamma$ is the surface tension. This is why soap bubbles (constant nonzero mean curvature $H = 1/R$) are spherical — the sphere is the unique compact surface with constant mean curvature (by Alexandrov's theorem). For minimal surfaces ($H = 0$), the pressure is equal on both sides, which is why soap films are flat in the absence of a pressure differential.

**Constant mean curvature surfaces.** Surfaces with $H = \text{const} \neq 0$ are called **CMC surfaces**. Besides the sphere, the most famous example is Delaunay's surfaces of revolution — the unduloids and nodoids — which arise as the shapes of liquid bridges and capillary surfaces. The study of CMC surfaces connects differential geometry to the calculus of variations, elliptic PDE theory, and mathematical physics.

---

## Classification of Surface Points

Combining the signs of Gaussian and mean curvature, we obtain a complete local classification of surface points.

| Type | Condition | Principal curvatures | Geometry |
|:-----|:----------|:--------------------|:---------|
| **Elliptic** | $K > 0$ | same sign | locally convex, bowl-shaped |
| **Hyperbolic** | $K < 0$ | opposite signs | saddle-shaped |
| **Parabolic** | $K = 0$, $H \neq 0$ | one is zero | locally cylindrical |
| **Flat (planar)** | $K = 0$, $H = 0$ | both zero | locally planar |
| **Umbilical** | $\kappa_1 = \kappa_2$ | equal | same curvature in all directions |

### Umbilical points

An **umbilical point** (or **umbilic**) is a point where $\kappa_1 = \kappa_2$. At such a point, the normal curvature is the same in every tangent direction, and the shape operator is a scalar multiple of the identity.

**Theorem.** If every point of a connected surface $S$ is umbilical, then $S$ is contained in a plane or a sphere.

**Proof sketch.** At an umbilic, $L = \kappa \cdot \text{Id}$, so $dN = -\kappa \cdot \text{Id}$. If $\kappa = 0$ everywhere, then $N$ is constant and $S$ lies in a plane. If $\kappa \neq 0$, consider the map $p \mapsto p + (1/\kappa(p))N(p)$. Differentiating and using $dN = -\kappa\,\text{Id}$ plus the constancy of $\kappa$ (which follows from the Codazzi equations), one shows this map is constant — so $S$ lies on a sphere centered at that constant point with radius $1/\kappa$. $\square$

This theorem shows that spheres and planes are the only totally umbilical surfaces in $\mathbb{R}^3$ — they are geometrically rigid.

**Umbilics on common surfaces.** On an ellipsoid with three distinct semi-axes, there are exactly four umbilical points. On the sphere, every point is an umbilic. On the torus, there are no umbilics (the inner and outer equators have different principal curvatures, and this disparity persists at every point). The existence and distribution of umbilical points on a surface is a delicate interplay between the surface's geometry and topology.

**Hilbert's theorem on umbilics.** A famous conjecture (due to Caratheodory, and partially resolved by various authors) states that every smooth convex surface homeomorphic to $S^2$ has at least two umbilical points. This is known to hold for analytic surfaces and surfaces sufficiently close to a sphere, but the general $C^\infty$ case remains open.

### Lines of curvature

The integral curves of the principal directions form a net called the **lines of curvature**. Away from umbilical points, the two families of lines of curvature are orthogonal (since the principal directions are orthogonal). At umbilical points, the net has singularities — the index theory of these singularities connects to the topology of the surface (a preview of the Gauss-Bonnet story in Article 5).

**Rodrigues' theorem.** A curve $\alpha(t)$ on $S$ is a line of curvature if and only if

$$dN(\alpha'(t)) = -\kappa(t)\,\alpha'(t)$$

for some scalar function $\kappa(t)$. That is, along a line of curvature, the normal turns at a rate proportional to the speed, with proportionality constant equal to the corresponding principal curvature.

### Asymptotic directions and curves

A tangent direction $v$ is **asymptotic** if $\mathrm{II}(v,v) = 0$ — the normal curvature vanishes in that direction. Asymptotic directions exist if and only if $K \leq 0$:

- At a hyperbolic point ($K < 0$), there are exactly two asymptotic directions, bisecting the principal directions.
- At a parabolic point ($K = 0$, one $\kappa_i = 0$), there is exactly one asymptotic direction.
- At an elliptic point ($K > 0$), there are no asymptotic directions.

Integral curves of asymptotic directions are called **asymptotic curves**. On ruled surfaces (which have $K \leq 0$), the rulings are always asymptotic curves.

**The third fundamental form.** For completeness, we mention the **third fundamental form** $\mathrm{III}(v,w) = \langle dN(v), dN(w) \rangle = \langle L(v), L(w) \rangle$. It satisfies the identity

$$\mathrm{III} - 2H\,\mathrm{II} + K\,\mathrm{I} = 0,$$

which follows from the Cayley-Hamilton theorem applied to the $2\times 2$ shape operator matrix (every $2\times 2$ matrix satisfies its own characteristic equation $L^2 - \text{tr}(L)\,L + \det(L)\,I = 0$). This relation shows that the third fundamental form carries no information beyond what $\mathrm{I}$ and $\mathrm{II}$ (or equivalently, $H$ and $K$) already provide.

**Normal curvature via the second and first forms.** A useful computational identity: the normal curvature in direction $v$ is

$$\kappa_n(v) = \frac{\mathrm{II}(v,v)}{\mathrm{I}(v,v)}.$$

For $v = du\,\mathbf{r}_u + dv\,\mathbf{r}_v$, this becomes

$$\kappa_n = \frac{e\,du^2 + 2f\,du\,dv + g\,dv^2}{E\,du^2 + 2F\,du\,dv + G\,dv^2}.$$

This is the formula most often used in practice to compute normal curvature in a specified direction.

---

## What's Next

We have now assembled the complete toolkit for studying how a surface bends in $\mathbb{R}^3$: the first fundamental form (Article 2) measures intrinsic geometry, and the second fundamental form / shape operator (this article) measures extrinsic bending.

A remarkable surprise awaits: **Gaussian curvature, despite being defined extrinsically as the product of principal curvatures, turns out to be an intrinsic invariant** — it depends only on the first fundamental form and its derivatives. This is Gauss's **Theorema Egregium**, the centerpiece of the next article. Alongside it, we'll develop the theory of geodesics — the "straight lines" of curved surfaces — completing the intrinsic geometry toolkit before moving to the global Gauss-Bonnet theorem.

**Summary of key formulas.** For reference, here are the principal computational formulas from this article:

| Quantity | Formula |
|:---------|:--------|
| Gaussian curvature | $K = \kappa_1\kappa_2 = \dfrac{eg - f^2}{EG - F^2}$ |
| Mean curvature | $H = \dfrac{\kappa_1+\kappa_2}{2} = \dfrac{eG-2fF+gE}{2(EG-F^2)}$ |
| Normal curvature | $\kappa_n = \dfrac{e\,du^2 + 2f\,du\,dv + g\,dv^2}{E\,du^2 + 2F\,du\,dv + G\,dv^2}$ |
| Euler's formula | $\kappa_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta$ |
| Principal curvatures | $\kappa_{1,2} = H \pm \sqrt{H^2 - K}$ |

These formulas, together with the first and second fundamental form computations from Article 2, provide a complete computational toolkit for analyzing the curvature of any explicitly parametrized surface.

---

*This is Part 3 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 2 — Surfaces and the First Fundamental Form](/en/differential-geometry/02-surfaces-first-form/)*

*Next: [Part 4 — Intrinsic Geometry](/en/differential-geometry/04-intrinsic-geometry/)*
