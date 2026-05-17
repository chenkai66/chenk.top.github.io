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

## The Gauss Map

Given a regular oriented surface $S\subset\mathbb{R}^3$, every point $p$ has a well-defined unit normal $\mathbf{n}(p)\in S^2$ (where $S^2$ is the unit sphere). The assignment $p\mapsto \mathbf{n}(p)$ is the *Gauss map*:
$$N: S\to S^2,\qquad p\mapsto \mathbf{n}(p).$$

![Gauss map N: S to S^2 sending each point to its unit normal](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_1_gauss_map.png)

In a chart $\mathbf{x}(u, v)$, $N$ has the explicit form
$$N(\mathbf{x}(u,v)) = \frac{\mathbf{x}_u\times\mathbf{x}_v}{|\mathbf{x}_u\times\mathbf{x}_v|}.$$

The Gauss map is smooth (when $S$ is) and its image lies in $S^2$. Some examples:

- **Plane.** $N \equiv $ const (a single point on $S^2$). The plane has no bending — its normal never changes.
- **Sphere of radius $r$.** $N(p) = p/r$. The Gauss map is essentially the identity (up to scaling). Every point on the sphere maps to a different point on $S^2$.
- **Cylinder of radius $r$ around the $z$-axis.** $N$ depends only on the angle around the $z$-axis: it traces out a great circle on $S^2$ (specifically, the equator if the cylinder axis is vertical). The image is 1D.
- **Saddle $z = u^2 - v^2$.** $N$ varies in two directions, but with a *flipping* character: as you move one way the normal tilts toward one pole, the other way it tilts toward the opposite pole.

The intuitive role of the Gauss map: it records how the unit normal varies as you move on $S$. A surface that bends sharply will have a Gauss map that moves quickly; a flat surface has a constant Gauss map.

**Why this matters.** The *differential* $dN_p: T_pS\to T_{N(p)}S^2$ — the linearization of the Gauss map — is the central object of extrinsic surface geometry. Its eigenvalues are the principal curvatures; its determinant is the Gaussian curvature; its trace is twice the mean curvature. From this single linear map, we will read off everything.

A small but useful fact: $T_{N(p)}S^2$ is the orthogonal complement of $\mathbf{n}(p)$ in $\mathbb{R}^3$ — which is exactly $T_pS$. So we can identify $T_{N(p)}S^2 = T_pS$ and view $dN_p$ as an endomorphism of $T_pS$.

---

## The Shape Operator

**Definition.** The *shape operator* (or Weingarten map) at $p$ is
$$S_p = -dN_p: T_pS \to T_pS.$$

The minus sign is convention; with it, a surface curving "toward" the normal has positive principal curvatures.

The shape operator is a self-adjoint linear map with respect to the first fundamental form: for any $\mathbf{v}, \mathbf{w}\in T_pS$,
$$\langle S_p\mathbf{v}, \mathbf{w}\rangle = \langle\mathbf{v}, S_p\mathbf{w}\rangle.$$

**Proof of self-adjointness.** Use a chart $\mathbf{x}(u, v)$ with basis $\{\mathbf{x}_u, \mathbf{x}_v\}$ for $T_pS$. The relation $\mathbf{n}\cdot\mathbf{x}_u = 0$ (normal is perpendicular to tangent) differentiates to $\mathbf{n}_u\cdot\mathbf{x}_u + \mathbf{n}\cdot\mathbf{x}_{uu} = 0$, so $\mathbf{n}_u\cdot\mathbf{x}_u = -\mathbf{n}\cdot\mathbf{x}_{uu}$. Similarly $\mathbf{n}_u\cdot\mathbf{x}_v = -\mathbf{n}\cdot\mathbf{x}_{uv}$ and $\mathbf{n}_v\cdot\mathbf{x}_u = -\mathbf{n}\cdot\mathbf{x}_{vu}$. Since mixed partials commute, $\mathbf{x}_{uv} = \mathbf{x}_{vu}$, so $\mathbf{n}_u\cdot\mathbf{x}_v = \mathbf{n}_v\cdot\mathbf{x}_u$. This is exactly the symmetry condition. $\square$

Self-adjoint operators on a real inner product space are diagonalizable with real eigenvalues. The eigenvalues of $S_p$ are the *principal curvatures* $k_1(p), k_2(p)$, and the corresponding eigenvectors (orthogonal in $T_pS$) are the *principal directions*.

![Shape operator and its eigenvectors as principal directions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_2_shape_operator.png)

---

## The Second Fundamental Form

Define three coefficients:
$$L = -\mathbf{n}_u\cdot\mathbf{x}_u = \mathbf{n}\cdot\mathbf{x}_{uu},$$
$$M = -\mathbf{n}_u\cdot\mathbf{x}_v = \mathbf{n}\cdot\mathbf{x}_{uv} = -\mathbf{n}_v\cdot\mathbf{x}_u,$$
$$N = -\mathbf{n}_v\cdot\mathbf{x}_v = \mathbf{n}\cdot\mathbf{x}_{vv}.$$

(I will write the second fundamental form coefficient as $N$ in italic and the Gauss map as $N$ in roman in this article, to match common conventions; in context the meaning is clear.)

The matrix
$$\mathrm{II} = \begin{pmatrix}L & M \\ M & N\end{pmatrix}$$
is the *second fundamental form*. It is symmetric (we proved this above) but not necessarily positive-definite — it can be indefinite, semidefinite, or zero, depending on the surface.

The shape operator in the basis $\{\mathbf{x}_u, \mathbf{x}_v\}$ has matrix
$$[S_p] = \mathrm{I}^{-1}\mathrm{II},$$
which is the standard relation between the matrix of a self-adjoint operator and the bilinear forms (one defining the inner product, one defining the operator).

A useful explicit formula:
$$[S_p] = \frac{1}{EG - F^2}\begin{pmatrix}G & -F \\ -F & E\end{pmatrix}\begin{pmatrix}L & M \\ M & N\end{pmatrix}.$$

Determinant and trace:
$$\det[S_p] = \frac{LN - M^2}{EG - F^2},\qquad \mathrm{tr}[S_p] = \frac{EN - 2FM + GL}{EG - F^2}.$$

**Definitions.**
- **Gaussian curvature.** $K(p) = \det[S_p] = (LN - M^2)/(EG - F^2)$.
- **Mean curvature.** $H(p) = \frac{1}{2}\mathrm{tr}[S_p] = (EN - 2FM + GL)/(2(EG-F^2))$.
- **Principal curvatures.** $k_1, k_2 = $ eigenvalues of $S_p$. They satisfy $K = k_1 k_2$ and $H = (k_1+k_2)/2$.

Equivalently, $k_1$ and $k_2$ are the roots of the characteristic polynomial $\lambda^2 - 2H\lambda + K = 0$, which gives
$$k_{1,2} = H\pm\sqrt{H^2 - K}.$$

The discriminant $H^2 - K$ is non-negative (since the eigenvalues are real); equality $H^2 = K$ characterizes *umbilic points*, where the shape operator is a multiple of the identity and every direction is principal.

![Principal curvatures k_1 and k_2 at a surface point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_3_principal_curvatures.png)

---

## Geometric Interpretation: Normal Curvature

Take a unit tangent vector $\mathbf{w}\in T_pS$. The plane through $p$ spanned by $\mathbf{w}$ and $\mathbf{n}(p)$ — the *normal section* — intersects $S$ in a curve. This curve has its own curvature in $\mathbb{R}^3$, and signed by the orientation of $\mathbf{n}$, this signed curvature is the *normal curvature in direction $\mathbf{w}$*:
$$\kappa_n(\mathbf{w}) = \mathrm{II}(\mathbf{w}, \mathbf{w}) = \langle S_p\mathbf{w}, \mathbf{w}\rangle.$$

Equivalently, $\kappa_n(\mathbf{w}) = (L a^2 + 2 M a b + N b^2)/(E a^2 + 2 F a b + G b^2)$ if $\mathbf{w} = a\mathbf{x}_u + b\mathbf{x}_v$.

**Euler's theorem (1760).** The principal curvatures $k_1, k_2$ are the maximum and minimum normal curvatures over all unit tangent directions. If $\theta$ is the angle between $\mathbf{w}$ and the principal direction for $k_1$, then
$$\kappa_n(\mathbf{w}) = k_1\cos^2\theta + k_2\sin^2\theta.$$

Beautiful and old. The normal curvature in any direction is a convex combination (sometimes literally, when both $k_i$ have the same sign; sometimes with sign-changing weights, when they differ) of the two extremes.

---

## Three Regimes of Gaussian Curvature

Based on the sign of $K = k_1 k_2$, every (non-umbilic) point on a surface falls into one of three categories.

![Three regimes of Gaussian curvature: positive, zero, negative](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_4_gauss_K_three.png)

**Elliptic point ($K > 0$).** Both principal curvatures have the same sign. The surface curves to one side of the tangent plane in *every* direction. Locally, $S$ looks like a piece of an ellipsoid. Examples: every point on a sphere ($k_1 = k_2 = 1/r$), every point inside the rim of an egg, the convex parts of a torus.

**Hyperbolic point ($K < 0$).** Principal curvatures have opposite signs. The surface curves up in some directions and down in others — saddle-like. Examples: every point on a saddle, every point on the inside of a torus's rim. There are *asymptotic directions* where $\kappa_n = 0$ (the surface is locally flat in those directions).

**Parabolic point ($K = 0$, $H \neq 0$).** Exactly one principal curvature vanishes; the other is non-zero. The surface is flat in one direction and curved in the perpendicular one. Examples: every point on a cylinder, every point on a cone.

**Planar point ($K = H = 0$).** Both principal curvatures vanish; locally to second order, $S$ is the plane. Either the surface really is flat (e.g. $S$ is a plane), or there is a higher-order tangency. Examples: the umbilic points of a sphere are not planar (they have $K \neq 0$ since $k_1 = k_2 \neq 0$); the saddle $z = u^4 - v^4$ has a planar point at the origin.

The three pictures — elliptic, parabolic, hyperbolic — are worth memorizing. They give an immediate answer to "what does this surface look like near here?"

---

## Worked Examples

### Sphere of radius $r$

Use spherical coordinates $\mathbf{x}(\theta,\varphi) = r(\sin\varphi\cos\theta, \sin\varphi\sin\theta, \cos\varphi)$. We computed earlier $\mathrm{I} = \begin{pmatrix}r^2\sin^2\varphi & 0 \\ 0 & r^2\end{pmatrix}$.

Unit normal: $\mathbf{n} = \mathbf{x}/r$, the outward radial direction.

Second fundamental form: $\mathbf{x}_{\theta\theta} = r(-\sin\varphi\cos\theta, -\sin\varphi\sin\theta, 0)$, $\mathbf{n}\cdot\mathbf{x}_{\theta\theta} = -r\sin^2\varphi$, so $L = -r\sin^2\varphi$. Similarly $M = 0$ (orthogonal coordinates) and $N = -r$. Wait — sign convention: with the outward normal, $L = \mathbf{n}\cdot\mathbf{x}_{\theta\theta} = -r\sin^2\varphi$ is *negative*. If we instead use the inward normal $\mathbf{n}_{\mathrm{in}} = -\mathbf{x}/r$, both $L$ and $N$ become positive. Different texts choose differently. I will stick with the outward normal; then a sphere has *negative* coefficients.

$\mathrm{II} = \begin{pmatrix}-r\sin^2\varphi & 0 \\ 0 & -r\end{pmatrix}$. Determinant: $r^2\sin^2\varphi$. So
$$K = \frac{LN - M^2}{EG - F^2} = \frac{r^2\sin^2\varphi}{r^4\sin^2\varphi} = \frac{1}{r^2}.$$
Constant positive, as expected. The sphere has Gaussian curvature $1/r^2$ everywhere.

Mean curvature: $H = \frac{EN + GL}{2(EG-F^2)} = \frac{r^2(-r) + r^2(-r\sin^2\varphi)/\sin^2\varphi}{2r^4\sin^2\varphi}\cdot\sin^2\varphi = -1/r$. Constant.

Principal curvatures: $k_1 = k_2 = -1/r$ (or $+1/r$ with the inward normal). Every point on the sphere is *umbilic*: every direction is principal, and there is no "preferred" curvature direction.

### Cylinder of radius $r$

$\mathbf{x}(u, v) = (r\cos u, r\sin u, v)$. We computed $\mathrm{I} = \mathrm{diag}(r^2, 1)$.

Outward normal: $\mathbf{n} = (\cos u, \sin u, 0)$.

Second derivatives: $\mathbf{x}_{uu} = (-r\cos u, -r\sin u, 0)$, $\mathbf{n}\cdot\mathbf{x}_{uu} = -r$. So $L = -r$. $\mathbf{x}_{uv} = \mathbf{x}_{vv} = 0$, so $M = N = 0$.

$\mathrm{II} = \begin{pmatrix}-r & 0 \\ 0 & 0\end{pmatrix}$. Determinant zero, so $K = 0$. The cylinder has zero Gaussian curvature — it is intrinsically flat, just as we predicted from the equality of its first form with the plane's.

Mean curvature: $H = (EN + GL)/(2(EG-F^2)) = (0 + r^2\cdot(-r))/(2 r^2) = -r/2$ (with the outward normal; sign flip with inward). Non-zero. So the cylinder is intrinsically flat ($K = 0$) but extrinsically bent ($H \neq 0$). That is the signature of a parabolic surface.

Principal curvatures: eigenvalues of $\mathrm{I}^{-1}\mathrm{II} = \mathrm{diag}(1/r^2, 1)\mathrm{diag}(-r, 0) = \mathrm{diag}(-1/r, 0)$. So $k_1 = -1/r$ (in the angular direction) and $k_2 = 0$ (in the axial direction). The cylinder bends in one direction and not the other — exactly the picture of a parabolic surface.

### Saddle $z = u^2 - v^2$

Chart: $\mathbf{x}(u, v) = (u, v, u^2 - v^2)$.

$\mathbf{x}_u = (1, 0, 2u)$, $\mathbf{x}_v = (0, 1, -2v)$. So $E = 1 + 4u^2$, $G = 1 + 4v^2$, $F = -4uv$.

At the origin $(0, 0)$: $E = G = 1$, $F = 0$, so $\mathrm{I} = I_2$ at that point.

Cross product: $\mathbf{x}_u\times\mathbf{x}_v = (-2u, 2v, 1)$. Magnitude: $\sqrt{1 + 4u^2 + 4v^2}$.

At the origin, $\mathbf{n} = (0, 0, 1)$.

Second derivatives at the origin: $\mathbf{x}_{uu} = (0, 0, 2)$, $\mathbf{x}_{vv} = (0, 0, -2)$, $\mathbf{x}_{uv} = 0$. So $L = 2$, $N = -2$, $M = 0$.

$\mathrm{II} = \mathrm{diag}(2, -2)$ at the origin. $K = (2)(-2)/1 = -4 < 0$. The saddle has negative Gaussian curvature at the origin, as a saddle should.

$H = (1\cdot(-2) + 1\cdot 2)/(2\cdot 1) = 0$. Mean curvature zero. The saddle is a *minimal surface* at this point (and turns out to be minimal everywhere if we use the standard saddle $z = uv$ with appropriate setup, but $z = u^2 - v^2$ is *not* minimal globally — the off-origin computation gives $H \neq 0$).

Principal curvatures at the origin: eigenvalues of $\mathrm{II} = \mathrm{diag}(2, -2)$ — just $\pm 2$. So $k_1 = 2$, $k_2 = -2$, with principal directions along the $u$- and $v$-axes. The surface curves up by 2 in the $u$-direction and curves down by 2 in the $v$-direction.

### Torus

For the torus $\mathbf{x}(u, v) = ((R+r\cos v)\cos u, (R+r\cos v)\sin u, r\sin v)$:

After computation (which I will not reproduce in detail), the Gaussian curvature comes out to
$$K(u, v) = \frac{\cos v}{r(R + r\cos v)}.$$

Sign analysis:
- Outer rim ($v = 0$): $K = 1/(r(R+r)) > 0$. Elliptic.
- Top and bottom ($v = \pm\pi/2$): $K = 0$. Parabolic.
- Inner rim ($v = \pi$): $K = -1/(r(R-r)) < 0$. Hyperbolic.

The torus has all three regimes! And the parabolic curves $v = \pm\pi/2$ separate the elliptic outer band from the hyperbolic inner band. This is a beautifully structured example.

The total integral of $K$ over the torus:
$$\int_S K\,dS = \int_0^{2\pi}\int_0^{2\pi}\frac{\cos v}{r(R+r\cos v)}\cdot r(R+r\cos v)\,du\,dv = \int_0^{2\pi}\int_0^{2\pi}\cos v\,du\,dv = 0.$$

Total Gaussian curvature on a torus is zero. This is not an accident; it is the Gauss-Bonnet theorem at work, which we will see in chapter 5: $\int K\,dS = 2\pi\chi(S)$, where $\chi$ is the Euler characteristic, and $\chi(\text{torus}) = 0$.

---

## Mean Curvature and Minimal Surfaces

Mean curvature has a different geometric flavour from Gaussian curvature. Where $K$ measures "how the surface fails to look flat in any direction", $H$ is more like "the average bending".

![Mean curvature H = (k_1 + k_2)/2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_5_mean_curvature.png)

**Definition (Minimal surface).** A surface with $H \equiv 0$ is *minimal*.

A foundational result: among all surfaces with a fixed boundary curve, the one minimizing area has $H \equiv 0$ in its interior. This is the variational characterization, and it explains the name "minimal". Soap films are minimal surfaces — they minimize surface tension energy, which is proportional to area.

**Examples of minimal surfaces.**

*Catenoid.* Surface of revolution with profile curve $\rho(v) = a\cosh(v/a)$. The catenoid is the only non-planar minimal surface of revolution.

*Helicoid.* $\mathbf{x}(u, v) = (v\cos u, v\sin u, c u)$. Mean curvature is zero everywhere. It is also ruled (a one-parameter family of straight lines), and in fact the helicoid and catenoid form a famous *isometric pair*: they are intrinsically the same surface, as seen from chapter 2. But they have different shape operators because the embedding is different.

*Scherk's surfaces.* A two-parameter family of doubly periodic minimal surfaces, $z = \log(\cos u/\cos v)$. The first non-trivial minimal surface families discovered (1834).

*Costa's surface.* A 1982 discovery by Celso Costa: a complete embedded minimal surface of finite topology (genus 1, three ends), refuting a long-standing conjecture that the catenoid, helicoid, and plane were the only such examples. A turning point in modern minimal surface theory.

![A minimal surface where H vanishes everywhere](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_7_minimal_surfaces.png)

The variational and PDE-theoretic study of minimal surfaces is a vast subject in itself (Plateau's problem, Bernstein's theorem, the Weierstrass-Enneper representation, and so on). I am not going to do it justice here; the takeaway for this article is that $H = 0$ is a meaningful geometric and physical condition.

---

## Side-by-Side Comparison: Sphere, Cylinder, Saddle

Let me put the three signature surfaces in a single table.

| Surface | $K$ | $H$ | $k_1$ | $k_2$ | Regime |
|---|---|---|---|---|---|
| Sphere (radius $r$) | $1/r^2$ | $-1/r$ | $-1/r$ | $-1/r$ | Elliptic, umbilic |
| Cylinder (radius $r$) | $0$ | $-1/(2r)$ | $-1/r$ | $0$ | Parabolic |
| Saddle (at origin) | $-4$ | $0$ | $2$ | $-2$ | Hyperbolic, minimal |

(Signs depend on normal orientation; I am using outward / upward normals.)

Three different geometric flavors in three lines. Every surface point you will ever encounter falls somewhere in this taxonomy (modulo umbilic and planar special cases).

![Sphere, cylinder, saddle: K and H side by side](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-second-form-curvature/dg_v2_03_6_curvature_compare.png)

---

## Why This Matters: Physics, Engineering, Computer Graphics

A few words on why anyone outside the math department should care.

**General relativity.** The Einstein field equations relate the curvature of spacetime (a 4-dimensional Lorentzian manifold) to the energy-momentum tensor. The relevant curvature is the Riemann tensor, which generalizes Gaussian curvature to higher dimensions. Surface curvature is the entry-level model.

**Computer graphics.** Mean and Gaussian curvature are used for shading, decimation (mesh simplification), and stylized rendering. A surface with high $|K|$ has visually distinctive features that should be preserved when the mesh is reduced.

**Architecture and structural engineering.** The Sydney Opera House shells, Felix Candela's hyperbolic paraboloids, Frei Otto's tensile structures — all are designed using surface curvature. Doubly curved (hyperbolic / saddle) surfaces are particularly stiff, which is why they appear in shells and roofs.

**Biology.** Cell membranes are described by Helfrich's bending energy, $\int (H - H_0)^2\,dS$, where $H_0$ is a "spontaneous curvature" depending on the lipid composition. Vesicle shapes, red blood cell biconcave morphology, and so on are predicted by minimizing this energy. The connection between $H$ and physical bending is direct.

**Geodesy.** The Earth is roughly an oblate ellipsoid; for fine work, even more complicated geoidal shapes are used. The curvature of the Earth's surface enters into mapping, surveying, and GPS calculations.

---

## Limitations and What Comes Next

A few caveats.

**Sign conventions.** The signs of $L, M, N, H$ depend on the choice of normal orientation. Different texts make different choices. Always check the convention before using a formula.

**Umbilic points.** Where $k_1 = k_2$, principal directions are not unique. For most surfaces, umbilics are isolated points (Hilbert-Cohn-Vossen for the sphere is the famous exception: every point of a sphere is umbilic, and a famous open problem — the Caratheodory conjecture — asks whether *every* closed convex surface has at least two umbilics; still unresolved in full generality).

**Higher-order behavior.** $\mathrm{I}$ encodes second-order intrinsic data; $\mathrm{II}$ encodes second-order extrinsic data. Together they almost determine the surface up to rigid motion (the full statement involves the integrability conditions of the next chapter). Higher-order behavior — third derivatives and beyond — is rarely useful in classical surface theory but matters when studying singular surfaces, cusps, or limit behavior.

**The next chapter.** Will reveal that $K$ — although defined via $\mathrm{II}$ — depends only on $\mathrm{I}$. This is the *Theorema Egregium*. One immediate consequence: an isometry between two surfaces preserves $K$ (since it preserves $\mathrm{I}$). So even though we computed $K$ from extrinsic data, the answer was secretly intrinsic all along. Mean curvature, by contrast, is genuinely extrinsic and changes under bending. Watch for this in chapter 4.

**Beyond surfaces.** The shape operator generalizes to embedded submanifolds in any ambient Riemannian manifold; you get the *second fundamental form* of the embedding, which is a tensor. The eigenvalues are still principal curvatures. The story scales up.

---

## Summary

We now have, in addition to the first fundamental form, a second tensor:

| Quantity | Symbol | Formula |
|---|---|---|
| Second fundamental form | $\mathrm{II}$ | $\begin{pmatrix}L & M\\M & N\end{pmatrix}$ with $L = \mathbf{n}\cdot\mathbf{x}_{uu}$, etc. |
| Shape operator | $S$ | $\mathrm{I}^{-1}\mathrm{II}$ |
| Gaussian curvature | $K$ | $\det S = (LN - M^2)/(EG - F^2)$ |
| Mean curvature | $H$ | $\frac{1}{2}\mathrm{tr}\,S = (EN - 2FM + GL)/(2(EG-F^2))$ |
| Principal curvatures | $k_1, k_2$ | eigenvalues of $S$ |

These quantities classify surface points into elliptic, parabolic, hyperbolic, planar regimes. Together with the first fundamental form, they will eventually let us state the Fundamental Theorem of Surfaces: $\mathrm{I}$ and $\mathrm{II}$ (subject to compatibility conditions) determine a surface up to rigid motion.

The next chapter takes us back inside the surface, to ask: which of all this data is actually intrinsic? The answer — that $K$ is — is one of the most important results of classical differential geometry. From it spring geodesics, the Theorema Egregium, the Gauss-Bonnet theorem, and ultimately the entire intrinsic apparatus of Riemannian geometry that powers general relativity. We are about to leave the safety of $\mathbb{R}^3$ behind.

---

## Appendix: Three More Worked Computations

For practice, three more computations that drive the formulas home.

### Paraboloid $z = u^2 + v^2$

Chart $\mathbf{x}(u, v) = (u, v, u^2 + v^2)$.

$\mathbf{x}_u = (1, 0, 2u)$, $\mathbf{x}_v = (0, 1, 2v)$, so $E = 1 + 4u^2$, $F = 4uv$, $G = 1 + 4v^2$, $EG-F^2 = 1 + 4u^2 + 4v^2$.

$\mathbf{x}_u\times\mathbf{x}_v = (-2u, -2v, 1)$, $|\cdot| = \sqrt{1+4u^2+4v^2}$, so $\mathbf{n} = (-2u, -2v, 1)/\sqrt{1+4u^2+4v^2}$.

$\mathbf{x}_{uu} = (0, 0, 2)$, $\mathbf{x}_{uv} = 0$, $\mathbf{x}_{vv} = (0, 0, 2)$.

$L = \mathbf{n}\cdot\mathbf{x}_{uu} = 2/\sqrt{1+4u^2+4v^2}$. Similarly $N = 2/\sqrt{1+4u^2+4v^2}$, $M = 0$.

$\mathrm{II} = \frac{2}{\sqrt{1+4u^2+4v^2}}I_2$.

Gaussian curvature: $K = (LN - M^2)/(EG-F^2) = \frac{4}{(1+4u^2+4v^2)^2}$.

At the origin: $K = 4$. As $(u,v)\to\infty$: $K\to 0$. The paraboloid has positive Gaussian curvature everywhere (elliptic), peaking at the vertex and decaying as you move away. Reasonable; the paraboloid is "most curved" at its vertex and flattens out on its sides.

Mean curvature: $H = \frac{(1+4u^2+4v^2)\cdot 2 + (1+4u^2+4v^2)\cdot 2 - 0}{2(1+4u^2+4v^2)\sqrt{1+4u^2+4v^2}} = \frac{2}{\sqrt{1+4u^2+4v^2}}$. Wait, that simplifies wrongly; let me redo. Use the formula $H = (EN - 2FM + GL)/(2(EG-F^2))$. With $L = N = 2/\sqrt{w}$, $M = 0$, $F = 4uv$, $E = 1+4u^2$, $G = 1+4v^2$, $EG-F^2 = w := 1+4u^2+4v^2$:
$$H = \frac{(1+4u^2)(2/\sqrt{w}) + (1+4v^2)(2/\sqrt{w})}{2w} = \frac{2(2 + 4u^2 + 4v^2)/\sqrt{w}}{2w} = \frac{2+4u^2+4v^2}{w^{3/2}}.$$
At the origin: $H = 2$. So $k_{1,2} = 2 \pm \sqrt{4 - 4} = 2$, both equal — the origin is umbilic (every direction is principal). Makes sense by rotational symmetry of $z = u^2+v^2$.

### Hyperbolic paraboloid $z = uv$

The "Pringles chip" surface. Chart $\mathbf{x}(u,v) = (u, v, uv)$.

$\mathbf{x}_u = (1, 0, v)$, $\mathbf{x}_v = (0, 1, u)$, so $E = 1+v^2$, $F = uv$, $G = 1+u^2$, $EG-F^2 = 1+u^2+v^2$.

$\mathbf{x}_u\times\mathbf{x}_v = (-v, -u, 1)$, $|\cdot| = \sqrt{1+u^2+v^2}$, so $\mathbf{n} = (-v, -u, 1)/\sqrt{1+u^2+v^2}$.

$\mathbf{x}_{uu} = 0$, $\mathbf{x}_{vv} = 0$, $\mathbf{x}_{uv} = (0, 0, 1)$. So $L = N = 0$, $M = 1/\sqrt{1+u^2+v^2}$.

$K = (0 - M^2)/(EG-F^2) = -1/(1+u^2+v^2)^2$. Always negative. Hyperbolic everywhere. At the origin $K = -1$.

$H = (E\cdot 0 - 2F M + G\cdot 0)/(2(EG-F^2)) = -F M/(EG-F^2) = -uv/((1+u^2+v^2)^{3/2})$.

At the origin: $H = 0$. Minimal there. But $H \neq 0$ off-center, so $z = uv$ is not a minimal surface globally.

This pair of examples (paraboloid, hyperbolic paraboloid) illustrates the same name applied to opposite curvatures. The "elliptic paraboloid" $z = u^2+v^2$ is bowl-shaped and elliptic; the "hyperbolic paraboloid" $z = uv$ is saddle-shaped and hyperbolic.

### Tractrix surface (pseudosphere)

Revolve the tractrix $\rho(v) = \sin v$, $z(v) = \cos v + \log\tan(v/2)$ around the $z$-axis, for $v\in (0, \pi)$.

Compute (after some work): $\rho'(v) = \cos v$, $z'(v) = -\sin v + \csc v = (1 - \sin^2 v)/\sin v = \cos^2 v/\sin v$. So $(\rho')^2 + (z')^2 = \cos^2 v + \cos^4 v/\sin^2 v = \cos^2 v(\sin^2 v + \cos^2 v)/\sin^2 v = \cos^2 v/\sin^2 v$. Then with the formulas for surfaces of revolution:
$$E = \rho^2 = \sin^2 v,\quad G = (\rho')^2+(z')^2 = \cot^2 v,\quad F = 0.$$

After computing $\mathrm{II}$ (which I will skip), one finds $K = -1$ identically. This is the *pseudosphere*, the classical surface of constant negative Gaussian curvature. Hilbert proved in 1901 that no complete surface in $\mathbb{R}^3$ has $K = -1$ everywhere; the pseudosphere is incomplete (it has a singular boundary at $v = \pi/2$).

Why this matters: the pseudosphere realizes a piece of the *hyperbolic plane*, the model of non-Euclidean geometry. Two pseudosphere geodesics drawn through a point and not through another given line are the "Lobachevsky lines" of non-Euclidean geometry. We will revisit this in chapter 4 when discussing constant curvature surfaces.

---

## Appendix: Symmetry, Isometry, Bending

Time to fix some operational vocabulary.

**Two surfaces are *isometric*** if there is a diffeomorphism between them preserving the first fundamental form. Equivalently (chapter 2): same $E, F, G$ in matching charts.

**Two surfaces are *applicable to each other*** if one can be bent (without stretching) into the other. This is a slightly more permissive notion than "isometric", in that it asks for the bending to be realized by a continuous family of isometric embeddings.

**A surface is *rigid*** if any isometry of it onto another surface in $\mathbb{R}^3$ is a rigid motion. The sphere is rigid (Cohn-Vossen, 1927): you cannot bend a closed sphere without stretching it. This is the rigorous reason a ping-pong ball is hard to deform without breaking.

**Bending preserves $K$** (Theorema Egregium, next chapter). It does *not* preserve $H$, $k_1$, $k_2$, or the second fundamental form.

A canonical example: helicoid and catenoid. Both have $K = -1/(c^2 + r^2)^2$ for the corresponding parameters; they are isometric. But their $H$ values are different (helicoid has $H = 0$, catenoid has $H = 0$ — they are *both* minimal, so this particular comparison is uninstructive). A better comparison: cylinder and plane. Both have $K = 0$, but cylinder has $H = -1/(2r) \neq 0$ and plane has $H = 0$. The cylinder is rolled paper; the plane is unrolled paper. Bending the paper changes $H$ but not $K$.

This is the concrete content of "intrinsic vs extrinsic". $K$ survives bending; $H$ does not.

---

## Appendix: Asymptotic Lines and Lines of Curvature

Two natural families of curves on a surface deserve brief mention.

**Lines of curvature.** A curve $\gamma(t)$ on $S$ is a *line of curvature* if at every point of $\gamma$, the tangent vector $\gamma'(t)$ is a principal direction. Equivalently, the Gauss map sends the tangent to the tangent: $dN(\gamma') \parallel \gamma'$.

On the sphere, every great circle is a line of curvature (because every direction is principal). On the cylinder, axial lines (along the $v$-axis) and circular cross-sections are the two families of lines of curvature. On a generic surface, lines of curvature form an *orthogonal net* — two families crossing at right angles. They give the surface a natural coordinate system that diagonalizes both $\mathrm{I}$ and $\mathrm{II}$. Working in line-of-curvature coordinates can simplify computations dramatically.

**Asymptotic lines.** A curve $\gamma(t)$ is *asymptotic* if at every point $\kappa_n(\gamma') = 0$ — the normal curvature vanishes in the tangent direction. Equivalently, $\mathrm{II}(\gamma', \gamma') = 0$.

Asymptotic lines exist only at points where $K \leq 0$ (so that $\mathrm{II}$ is indefinite or semi-definite). On a hyperbolic point, exactly two asymptotic directions exist; they are the directions of the cone $\mathrm{II}(\mathbf{w}, \mathbf{w}) = 0$. On a parabolic point, exactly one asymptotic direction. On an elliptic point, none.

A nice fact: a *ruled surface* (one made up of straight lines) has its rulings as asymptotic curves (since a straight line in $\mathbb{R}^3$ has zero curvature, so $\kappa_n = 0$). The hyperboloid of one sheet, the helicoid, the saddle: these all have explicit rulings, and those rulings are asymptotic lines.

---

## Appendix: The Fundamental Theorem of Surfaces

A natural question is the analogue of the Fundamental Theorem of Curves: given the data $\mathrm{I}$ and $\mathrm{II}$ as functions on $U\subset\mathbb{R}^2$, does there exist a surface realizing them? And if so, is it unique?

The answer (Bonnet, 1867): *yes, locally and up to rigid motion*, provided $\mathrm{I}$ and $\mathrm{II}$ satisfy the *Gauss equation* and the *Codazzi-Mainardi equations*. These are integrability conditions, and they are non-trivial. The Gauss equation in particular gives a formula for $K$ in terms of $\mathrm{I}$ alone — which is the Theorema Egregium! We will derive it explicitly in chapter 4.

The structure here is the classical instantiation of a more general theorem: a Riemannian metric and a "second fundamental form" on a $k$-manifold define an embedding into $\mathbb{R}^n$ ($k < n$) iff certain compatibility equations hold (Gauss, Codazzi, and Ricci equations in the general case).

This theorem is the reason classical differential geometry feels closed: knowing $\mathrm{I}$ and $\mathrm{II}$ is knowing the surface (locally, up to rigid motion). Two surfaces with the same forms are the same surface. Two surfaces with different forms are different. The forms are the genuine invariants.

---

## Recap

We have now built the full first-order extrinsic apparatus.

- Gauss map $N: S\to S^2$.
- Shape operator $S_p = -dN_p$, a self-adjoint linear map on $T_pS$.
- Second fundamental form $\mathrm{II}$, the bilinear form encoding $S_p$.
- Principal curvatures $k_1, k_2$, the eigenvalues of $S_p$.
- Gaussian curvature $K = k_1 k_2$ and mean curvature $H = (k_1+k_2)/2$.
- Three regimes — elliptic, parabolic, hyperbolic — distinguished by sign of $K$.
- Worked examples: sphere ($K = 1/r^2$, umbilic), cylinder ($K = 0$, parabolic), saddle ($K < 0$, hyperbolic, minimal at origin), torus (all three regimes), paraboloid, hyperbolic paraboloid, pseudosphere ($K = -1$).
- Lines of curvature and asymptotic lines as natural curves on a surface.
- Mean curvature ties to area minimization; minimal surfaces are $H \equiv 0$.
- The Fundamental Theorem of Surfaces: $\mathrm{I}$ and $\mathrm{II}$ (with compatibility) determine $S$ up to rigid motion.

The single most important open question is: which of the quantities computed above survive bending of the surface? The answer comes next chapter: $K$ does, $H$ does not, and the proof — Gauss's Theorema Egregium — is one of the milestones of mathematics.

A pedagogical note before we close. I have spent a lot of this chapter pushing through computations on specific surfaces — sphere, cylinder, saddle, torus, paraboloid, helicoid, pseudosphere. The reason is that the formulas for $K$ and $H$ in arbitrary parametrization are unwieldy and easy to misuse. Working with concrete surfaces builds the intuition that the formulas are doing what they ought to: assigning positive $K$ to bowl-like points, negative $K$ to saddle-like points, zero $K$ to "rolled paper" points. Once you have the intuition, the formulas become a bookkeeping device rather than a hurdle. If you have not yet computed $K$ and $H$ for a surface other than the ones I have done here, do one as an exercise — perhaps the catenoid, or the cone, or a Möbius strip parametrization. The first time you do it, the algebra is mechanical but slow; the second time, much faster; the third time, you can read off the curvature from the parametrization on inspection. That is the level of fluency that opens the rest of the subject.

A further note on conventions. The signs of $L, M, N$ — and hence of $H$, but not of $K$ — depend on the choice of unit normal. For closed orientable surfaces (sphere, torus, etc.), one usually takes the outward normal. For surfaces of revolution, "outward" usually means away from the axis. For graphs $z = f(u,v)$, one usually takes the upward normal. Different texts make different choices; what matters is consistency within a calculation. The Gaussian curvature $K = \det S = (LN - M^2)/(EG-F^2)$ is invariant under sign flip of $\mathbf{n}$ (because both $L$ and $N$ flip, and so does $M$ in the off-diagonal calculation). The mean curvature $H = \mathrm{tr}\,S/2$ flips sign under normal-flip. When in doubt, recompute, and check the sign on a test surface like the unit sphere where the answer is unambiguous.

With those caveats noted, on to the most beautiful theorem in classical differential geometry.

A final extended example: the Gauss map of an ellipsoid. Take the ellipsoid $x^2/a^2 + y^2/b^2 + z^2/c^2 = 1$ with $a > b > c > 0$. The Gauss map sends each point to its outward normal. Because the principal axes are different, the Gauss map is not a uniform scaling: it stretches and squishes parts of the ellipsoid differently as it maps to the sphere. The image of the ellipsoid under the Gauss map covers the entire sphere exactly once; the *Jacobian* of the Gauss map, by definition, is the Gaussian curvature $K$ (this is one common alternative definition of $K$, and it gives an immediate proof that the integral of $K$ over a closed convex surface equals $4\pi$, the area of $S^2$).

For the sphere, $K = 1/r^2$ everywhere, and indeed $\int_S K\,dS = (1/r^2)\cdot 4\pi r^2 = 4\pi$. For an ellipsoid, $K$ varies — peaking at the "ends" of the long axis and reaching its minimum at the "ends" of the short axis — but the integral is still $4\pi$. This is the Gauss-Bonnet theorem in advance: total curvature is a topological invariant. We will spend chapter 5 on this.

For an explicit ellipsoid computation: at the point $(a, 0, 0)$ on the long axis, the principal curvatures are $k_1 = a/c^2$ and $k_2 = a/b^2$, so $K = a^2/(b^2 c^2)$. At the point $(0, 0, c)$ on the short axis, $k_1 = c/a^2$ and $k_2 = c/b^2$, so $K = c^2/(a^2 b^2)$. For $a > b > c$, the "long-axis" curvature $a^2/(b^2 c^2)$ is larger than the "short-axis" curvature $c^2/(a^2 b^2)$. Geometrically: the ellipsoid is more sharply curved at its tips than at its waist. Numerically with $a = 3, b = 2, c = 1$: long-axis $K = 9/4 = 2.25$, short-axis $K = 1/36 \approx 0.028$. Two orders of magnitude. The ellipsoid is dramatically anisotropic.

This concrete picture — Gauss map as a "curvature-weighted" mapping to the sphere — is one of the deepest intuitions in surface theory. It reframes Gaussian curvature as the area density of the Gauss map's image, and it makes the Gauss-Bonnet theorem nearly visible: integrate the Gauss map's Jacobian, get the area covered, get a topological invariant.

---

*This is Part 3 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 2 — Surfaces and the First Fundamental Form](/en/differential-geometry/02-surfaces-first-form/)*

*Next: [Part 4 — Intrinsic Geometry](/en/differential-geometry/04-intrinsic-geometry/)*
