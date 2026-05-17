---
title: "Gaussian Curvature and the Second Fundamental Form"
date: 2021-05-15 09:00:00
tags:
  - differential-geometry
  - gaussian-curvature
  - second-fundamental-form
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 3
series_total: 6
translationKey: "differential-geometry-3"
description: "The second fundamental form and how Gaussian curvature captures the intrinsic shape of surfaces."
---

The first fundamental form measures distances on a surface. The second fundamental form measures how the surface bends away from its tangent plane. Together they determine the shape of a surface completely, up to rigid motion — paralleling how curvature and torsion determine a space curve. This chapter introduces the shape operator, principal curvatures, and the two most important curvature invariants: Gaussian curvature $K$ and mean curvature $H$.

## The Gauss map and the shape operator

Let $S$ be a regular oriented surface with unit normal field $\mathbf{N}: S \to S^2$. This is the **Gauss map** — it sends each point of the surface to the corresponding point on the unit sphere.

The **shape operator** (or Weingarten map) at $p$ is the differential of the Gauss map:

$$dN_p: T_pS \to T_pS$$

It sends a tangent vector $\mathbf{v}$ to the rate of change of the normal in the direction $\mathbf{v}$. The fact that $dN_p$ maps the tangent plane to itself (not into $\mathbb{R}^3$) follows from $\mathbf{N}\cdot\mathbf{N} = 1$: differentiating gives $\mathbf{N}\cdot d\mathbf{N} = 0$, so $dN_p(\mathbf{v})$ is perpendicular to $\mathbf{N}$, hence tangent.

**Theorem.** The shape operator is self-adjoint: $\langle dN_p(\mathbf{v}), \mathbf{w}\rangle = \langle \mathbf{v}, dN_p(\mathbf{w})\rangle$.

*Proof sketch.* Compute both sides in coordinates using $\mathbf{N}\cdot\mathbf{x}_u = 0$ and $\mathbf{N}\cdot\mathbf{x}_v = 0$ differentiated. The symmetry of mixed partial derivatives ($\mathbf{x}_{uv} = \mathbf{x}_{vu}$) gives the result.

Self-adjointness means the shape operator has real eigenvalues and orthogonal eigenvectors — a crucial fact for what follows.

## The shape operator in matrix form

In the coordinate basis $\{\mathbf{x}_u, \mathbf{x}_v\}$, the shape operator is represented by a $2\times 2$ matrix. Since $dN_p(\mathbf{x}_u) = \mathbf{N}_u$ and $dN_p(\mathbf{x}_v) = \mathbf{N}_v$, we can write:

$$\mathbf{N}_u = a_{11}\mathbf{x}_u + a_{21}\mathbf{x}_v, \quad \mathbf{N}_v = a_{12}\mathbf{x}_u + a_{22}\mathbf{x}_v.$$

The matrix of the shape operator (with the sign convention from $II$) is:

$$-\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22}\end{pmatrix} = \begin{pmatrix} E & F \\ F & G \end{pmatrix}^{-1}\begin{pmatrix} e & f \\ f & g \end{pmatrix}.$$

That is, $-dN = I^{-1}\cdot II$ in matrix form. The principal curvatures are the eigenvalues of this matrix, and the principal directions are its eigenvectors (expressed in the $\{\mathbf{x}_u, \mathbf{x}_v\}$ basis).

For an orthogonal parametrization with $F = 0$, the shape operator matrix simplifies to:

$$\begin{pmatrix} e/E & f/E \\ f/G & g/G \end{pmatrix}.$$

If additionally $f = 0$ (the coordinate curves are lines of curvature), the matrix is diagonal with entries $\kappa_1 = e/E$ and $\kappa_2 = g/G$.


![Gaussian curvature: positive (sphere), zero (cylinder), negative (saddle)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/03-curvature-of-surfaces/dg_fig3_curvature.png)

## The second fundamental form

The **second fundamental form** is the bilinear form associated to the shape operator:

$$II(\mathbf{v}, \mathbf{w}) = -\langle dN_p(\mathbf{v}), \mathbf{w}\rangle.$$

The minus sign is a convention that makes curvature positive for convex surfaces (normals point outward). In coordinates $(u,v)$, write $II = e\,du^2 + 2f\,du\,dv + g\,dv^2$ where:

$$e = -\mathbf{N}_u \cdot \mathbf{x}_u = \mathbf{N} \cdot \mathbf{x}_{uu}, \quad f = \mathbf{N} \cdot \mathbf{x}_{uv}, \quad g = \mathbf{N} \cdot \mathbf{x}_{vv}.$$

The second equality uses $\mathbf{N} \cdot \mathbf{x}_u = 0$ differentiated: $\mathbf{N}_u \cdot \mathbf{x}_u + \mathbf{N}\cdot\mathbf{x}_{uu} = 0$.

While the first fundamental form is always positive definite (it is an inner product), the second fundamental form can be positive definite, negative definite, or indefinite — and this sign pattern tells us the type of curvature at each point.

## Principal curvatures

The eigenvalues $\kappa_1, \kappa_2$ of $dN_p$ (with the sign convention from $II$) are the **principal curvatures**. The corresponding eigenvectors give the **principal directions** — the directions in which the surface bends the most and the least.

Geometrically: slice the surface with a normal plane containing direction $\mathbf{v}$. The resulting cross-section is a plane curve, and its curvature at $p$ is the **normal curvature**:

$$\kappa_n(\mathbf{v}) = \frac{II(\mathbf{v}, \mathbf{v})}{I(\mathbf{v}, \mathbf{v})}.$$

As $\mathbf{v}$ rotates through all tangent directions, $\kappa_n$ varies continuously. Its maximum and minimum values are the principal curvatures $\kappa_1$ and $\kappa_2$.

**Theorem (Euler, 1760).** If $\mathbf{v}$ makes angle $\theta$ with the first principal direction, then:

$$\kappa_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta.$$

*Proof.* In principal coordinates (where the shape operator is diagonal with eigenvalues $\kappa_1, \kappa_2$ and the metric is the identity), we have $II(\mathbf{v},\mathbf{v}) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta$ and $I(\mathbf{v},\mathbf{v}) = 1$.

Euler's formula tells us the normal curvature varies sinusoidally between $\kappa_1$ and $\kappa_2$. The average normal curvature over all directions is the mean curvature $H$. More precisely, integrating $\kappa_n(\theta)$ over $\theta \in [0, 2\pi]$ and dividing by $2\pi$ gives $(\kappa_1 + \kappa_2)/2 = H$. This provides an alternative definition of mean curvature: it is the average bending in all tangent directions.

A direction $\mathbf{v}$ with $\kappa_n(\mathbf{v}) = 0$ is called an **asymptotic direction**. At a hyperbolic point ($K < 0$), there are exactly two asymptotic directions — the directions in which the surface does not bend away from the tangent plane at all. At an elliptic point ($K > 0$), no asymptotic directions exist.

## Gaussian and mean curvature

From the principal curvatures, define two invariants:

$$K = \kappa_1\kappa_2 \quad \text{(Gaussian curvature)}$$

$$H = \frac{\kappa_1 + \kappa_2}{2} \quad \text{(mean curvature)}$$

In terms of the shape operator: $K = \det(dN_p)$ and $H = \frac{1}{2}\text{tr}(dN_p)$.

In coordinates:

$$K = \frac{eg - f^2}{EG - F^2}, \qquad H = \frac{eG - 2fF + gE}{2(EG - F^2)}.$$

These formulas let us compute curvature from a parametrization without finding principal directions explicitly.

## Geometric meaning of Gaussian curvature

The sign of $K$ classifies the local shape:

- **$K > 0$ (elliptic point):** Both principal curvatures have the same sign. The surface is locally convex — it curves the same way in all directions. Think of a sphere, the top of a hill, or the bowl of a spoon.
- **$K < 0$ (hyperbolic point):** Principal curvatures have opposite signs. The surface is saddle-shaped — it curves up in one direction and down in the perpendicular direction. Think of a mountain pass, a Pringles chip, or the inner ring of a torus.
- **$K = 0$ (parabolic point):** At least one principal curvature is zero. The surface is flat in one direction. Cylinders and cones have $K = 0$ everywhere — they can be "unrolled" without stretching.

There is another way to visualize $K$: it measures how the Gauss map distorts area. If a small region of area $\Delta A$ on the surface maps to a region of area $\Delta A'$ on the unit sphere, then $|K| = \lim \Delta A'/\Delta A$. Positive $K$ means the Gauss map preserves orientation; negative $K$ means it reverses it.

## Example: the sphere

For a sphere of radius $R$ with outward normal $\mathbf{N} = \mathbf{x}/R$, the Gauss map is $\mathbf{N}(p) = p/R$. Its differential is $dN_p = (1/R)\,\text{Id}$, giving $\kappa_1 = \kappa_2 = 1/R$.

$$K = \frac{1}{R^2}, \qquad H = \frac{1}{R}.$$

Every point is **umbilic** ($\kappa_1 = \kappa_2$). The sphere curves equally in all directions — it has no preferred bending direction. In fact, a connected umbilic surface (all points umbilic) must be a sphere or a plane.

## Example: the elliptic paraboloid

The surface $z = x^2 + y^2$ parametrized by $\mathbf{x}(u,v) = (u, v, u^2+v^2)$ has normal $\mathbf{N} = (-2u, -2v, 1)/\sqrt{1+4u^2+4v^2}$. At the origin: $e = 2/1 = 2$, $f = 0$, $g = 2$, $E = G = 1$, $F = 0$. The shape operator matrix is $\text{diag}(2, 2)$, giving $\kappa_1 = \kappa_2 = 2$, $K = 4$, $H = 2$. The origin is umbilic — the paraboloid is "sphere-like" there.

Away from the origin, the principal curvatures diverge ($\kappa_1 \neq \kappa_2$) and the surface becomes less symmetric. This illustrates a general principle: umbilic points on generic surfaces are isolated — the sphere's property of being entirely umbilic is exceptional.

## Example: the torus

Parametrize the torus by $\mathbf{x}(u,v) = ((R+r\cos u)\cos v,\, (R+r\cos u)\sin v,\, r\sin u)$ where $R > r > 0$. After computing the second fundamental form coefficients ($e = r$, $f = 0$, $g = (R+r\cos u)\cos u$), we get:

$$K = \frac{\cos u}{r(R + r\cos u)}.$$

- Outer equator ($u = 0$): $K = 1/(r(R+r)) > 0$ — elliptic, like a sphere.
- Inner equator ($u = \pi$): $K = -1/(r(R-r)) < 0$ — hyperbolic, saddle-shaped.
- Top and bottom circles ($u = \pi/2, 3\pi/2$): $K = 0$ — parabolic.

The total curvature of the torus is $\int K\,dA = 0$. The positive-curvature outer region and the negative-curvature inner region cancel exactly. This is not a coincidence — it is a consequence of Gauss-Bonnet (Chapter 6).

## Example: the catenoid (a minimal surface)

A surface with $H = 0$ everywhere is called a **minimal surface** — it locally minimizes area among surfaces with the same boundary. The name comes from soap films: a soap film spanning a wire frame adopts a shape with $H = 0$ (assuming no pressure difference).

The catenoid, $\mathbf{x}(u,v) = (\cosh u\cos v,\, \cosh u\sin v,\, u)$, is a minimal surface. Its principal curvatures are $\kappa_1 = 1/\cosh^2 u$ and $\kappa_2 = -1/\cosh^2 u$ — equal in magnitude but opposite in sign at every point. Therefore $H = 0$ and $K = -1/\cosh^4 u < 0$ everywhere.

Visually, the catenoid looks like two flared trumpets joined at a narrow waist. The saddle shape ($K < 0$) is visible: horizontal cross-sections are circles (curving one way), while vertical cross-sections curve the other way.

## Example: the saddle surface

The hyperbolic paraboloid $z = xy$ (a standard saddle) parametrized by $\mathbf{x}(u,v) = (u, v, uv)$ has $\mathbf{N} = (-v, -u, 1)/\sqrt{1+u^2+v^2}$. At the origin: $\kappa_1 = 1$, $\kappa_2 = -1$, so $K = -1$ and $H = 0$. The origin is simultaneously a saddle point and a point of zero mean curvature — this particular saddle is locally a minimal surface at the origin.

## The fundamental theorem of surfaces

**Theorem (Bonnet).** Let $E, F, G$ and $e, f, g$ be smooth functions on an open set $U \subset \mathbb{R}^2$ with $EG - F^2 > 0$, satisfying the Gauss equation and the Codazzi-Mainardi equations. Then there exists a surface patch $\mathbf{x}: U \to \mathbb{R}^3$ with these as its first and second fundamental form coefficients. The surface is unique up to rigid motion.

The Gauss equation and Codazzi-Mainardi equations are compatibility conditions — they ensure the system of PDEs for $\mathbf{x}_u$, $\mathbf{x}_v$, $\mathbf{N}$ is consistent. The Gauss equation relates $K$ to the Christoffel symbols (intrinsic data); the Codazzi-Mainardi equations constrain how the second fundamental form varies relative to the first.

*Proof sketch.* By the Frobenius theorem, integrability conditions (Gauss + Codazzi-Mainardi) guarantee existence of a solution to the overdetermined PDE system that determines $\mathbf{x}$. Uniqueness up to rigid motion follows from the uniqueness of solutions to ODEs with fixed initial data.

This parallels the fundamental theorem of curves: prescribe the curvature data, get the shape. The difference is that for surfaces, there are compatibility conditions — you cannot freely choose the fundamental forms independently.

## What's next

A remarkable fact awaits: Gaussian curvature, despite being defined using the normal vector and the shape operator (extrinsic data), turns out to depend only on the first fundamental form (intrinsic data). This is Gauss's Theorema Egregium — the gateway to intrinsic geometry. It means a two-dimensional creature can detect $K$ without ever leaving the surface. The proof and its consequences occupy the next chapter.

---

*This is Part 3 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Previous: [Part 2 — Surfaces and the First Fundamental Form](/en/differential-geometry/02-surfaces-first-form/) · Next: [Part 4 — Intrinsic Geometry](/en/differential-geometry/04-intrinsic-geometry/)*
