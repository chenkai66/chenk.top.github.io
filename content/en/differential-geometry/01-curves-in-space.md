---
title: "Curves in Space — Curvature, Torsion, and the Frenet Frame"
date: 2021-05-01 09:00:00
tags:
  - differential-geometry
  - curves
  - curvature
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 6
translationKey: "differential-geometry-1"
description: "How curves bend and twist in three-dimensional space, captured by curvature and torsion."
---

A curve is the simplest object in differential geometry: a one-dimensional thing living in three-dimensional space. Yet even here the mathematics is rich enough to occupy us for a while. Two scalar functions — curvature and torsion — encode the entire shape of a space curve up to rigid motion. That is the punchline of this chapter, but the path there involves building a moving coordinate frame and learning to read the differential equations it satisfies.

## Parametrized curves

A smooth curve in $\mathbb{R}^3$ is a map $\alpha: I \to \mathbb{R}^3$ where $I$ is an open interval and $\alpha$ is infinitely differentiable. We write $\alpha(t) = (x(t), y(t), z(t))$.

The velocity vector is $\alpha'(t)$. A curve is **regular** if $\alpha'(t) \neq 0$ for all $t$. Regularity means the curve never stops moving — no cusps, no backtracking. It ensures that the curve has a well-defined tangent direction at every point.

**Definition.** A **smooth regular curve** is a $C^\infty$ map $\alpha: I \to \mathbb{R}^3$ satisfying $\alpha'(t) \neq 0$ for all $t \in I$.

**Example (circular helix).** The circular helix $\alpha(t) = (a\cos t,\, a\sin t,\, bt)$ with $a > 0$ has velocity $\alpha'(t) = (-a\sin t,\, a\cos t,\, b)$ and speed $|\alpha'(t)| = \sqrt{a^2 + b^2}$. It is regular for all $t$. Visually, it spirals upward along the $z$-axis like a spring or a DNA double helix, with radius $a$ and pitch $2\pi b$.

**Example (plane curves).** The parabola $\alpha(t) = (t, t^2, 0)$ is a regular plane curve. The astroid $\alpha(t) = (\cos^3 t, \sin^3 t, 0)$ fails regularity at $t = 0, \pi/2, \pi, 3\pi/2$ — it has cusps at those points where the velocity vanishes.

The parametrization of a curve is not unique. The same geometric path can be traced at different speeds, forwards or backwards. What matters geometrically is the image $\alpha(I) \subset \mathbb{R}^3$ together with its orientation.


![Frenet frame (T, N, B) moving along a helix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_fig1_frenet.png)

## Arc length and reparametrization

The arc length from $t_0$ to $t$ is

$$s(t) = \int_{t_0}^{t} |\alpha'(u)|\, du.$$

Since $ds/dt = |\alpha'(t)| > 0$ for a regular curve, the function $s(t)$ is strictly increasing, hence invertible. We can reparametrize by arc length: define $\beta(s) = \alpha(t(s))$. A curve parametrized by arc length satisfies $|\beta'(s)| = 1$ — the unit-speed condition.

Arc-length parametrization strips away the "how fast" and leaves only the "which way." All intrinsic geometry of the curve depends only on this reparametrized version. It is the canonical parametrization.

**Example.** For the helix with speed $\sqrt{a^2+b^2}$, the arc-length parameter is $s = t\sqrt{a^2+b^2}$, so the unit-speed version is $\beta(s) = (a\cos(s/c),\, a\sin(s/c),\, bs/c)$ where $c = \sqrt{a^2+b^2}$.

From here on, I will work with unit-speed curves unless stated otherwise — it simplifies every formula.

## Curvature

For a unit-speed curve $\alpha(s)$, the **curvature** is defined as:

$$\kappa(s) = |\alpha''(s)|.$$

Curvature measures how sharply the curve bends. A straight line has $\kappa = 0$ everywhere (its tangent never changes direction). A circle of radius $r$ has $\kappa = 1/r$ — smaller circles bend harder, which matches intuition: turning a tight corner requires more steering than a gentle arc.

Geometrically, $1/\kappa$ is the radius of the **osculating circle** — the circle that best approximates the curve at that point. The osculating circle kisses the curve to second order: it shares the same position, first derivative, and second derivative.

For a curve not parametrized by arc length, the formula becomes:

$$\kappa = \frac{|\alpha' \times \alpha''|}{|\alpha'|^3}.$$

This expression is coordinate-free and reparametrization-invariant — it gives the same number regardless of how you trace the curve.

**Example (helix curvature).** For $\alpha(t) = (a\cos t, a\sin t, bt)$: compute $\alpha'' = (-a\cos t, -a\sin t, 0)$, then $\alpha' \times \alpha'' = (ab\sin t,\, -ab\cos t,\, a^2)$. So $|\alpha' \times \alpha''| = a\sqrt{a^2+b^2}$ and $|\alpha'|^3 = (a^2+b^2)^{3/2}$, giving:

$$\kappa = \frac{a}{a^2 + b^2}.$$

Constant curvature. The helix bends the same amount everywhere. When $b = 0$ this reduces to $\kappa = 1/a$, the curvature of a circle of radius $a$.

**Example (curvature of an ellipse).** Consider $\alpha(t) = (a\cos t,\, b\sin t,\, 0)$ with $a > b > 0$. The curvature works out to $\kappa(t) = ab/(a^2\sin^2 t + b^2\cos^2 t)^{3/2}$. At $t = 0$ (the end of the major axis): $\kappa = a/b^2$. At $t = \pi/2$ (end of minor axis): $\kappa = b/a^2$. The ellipse curves more sharply at the ends of its major axis — the "pointy" ends — which you can see by drawing one.

## The Frenet frame

Assume $\kappa(s) > 0$ (the curve is actually bending — not a straight line). Define three orthonormal vectors at each point:

$$\mathbf{T}(s) = \alpha'(s), \quad \mathbf{N}(s) = \frac{\alpha''(s)}{|\alpha''(s)|}, \quad \mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s).$$

Here $\mathbf{T}$ is the **unit tangent** (the direction of travel), $\mathbf{N}$ is the **principal normal** (pointing toward the center of curvature — the direction the curve is turning toward), and $\mathbf{B}$ is the **binormal** (perpendicular to both, completing a right-handed frame). Together $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$ form the **Frenet frame** — a moving orthonormal basis attached to the curve.

Picture it: the Frenet frame is a coordinate system that rides along the curve like a roller coaster car. $\mathbf{T}$ points along the track, $\mathbf{N}$ points toward the center of the turn, and $\mathbf{B}$ points "up" relative to the osculating plane. As the car moves, the frame continuously rotates.

The **osculating plane** is spanned by $\mathbf{T}$ and $\mathbf{N}$. It is the plane that best approximates the curve locally — a plane curve stays entirely in its osculating plane.

## Torsion

The **torsion** $\tau(s)$ measures how the osculating plane rotates about the tangent line:

$$\tau(s) = -\mathbf{B}'(s) \cdot \mathbf{N}(s).$$

A plane curve has $\tau = 0$ everywhere: its osculating plane never changes. Nonzero torsion means the curve is genuinely three-dimensional — it twists out of any fixed plane.

Positive torsion means the curve twists like a right-handed screw (the binormal rotates toward $-\mathbf{N}$). Negative torsion means left-handed twisting.

For the helix: $\tau = b/(a^2 + b^2)$. Constant torsion, matching the constant curvature. The helix twists uniformly as it climbs.

**Example (torsion of a non-planar curve).** Consider $\alpha(t) = (t, t^2, t^3)$. At $t = 0$: $\alpha' = (1,0,0)$, $\alpha'' = (0,2,0)$, $\alpha''' = (0,0,6)$. The torsion formula gives $\tau = (\alpha' \times \alpha'') \cdot \alpha''' / |\alpha' \times \alpha''|^2 = (0,0,2)\cdot(0,0,6)/4 = 3$. The cubic space curve twists sharply near the origin.

## The Frenet-Serret formulas

The derivatives of the frame vectors are governed by:

$$\begin{aligned}
\mathbf{T}' &= \kappa \mathbf{N}, \\
\mathbf{N}' &= -\kappa \mathbf{T} + \tau \mathbf{B}, \\
\mathbf{B}' &= -\tau \mathbf{N}.
\end{aligned}$$

These three equations encode everything about the local geometry of the curve. They say: the tangent turns toward the normal at rate $\kappa$ (curvature); the normal is pulled back by the tangent and pushed toward the binormal (both curvature and torsion contribute); and the binormal drifts toward the normal at rate $\tau$ (torsion).

Written as a matrix equation:

$$\begin{pmatrix} \mathbf{T}' \\ \mathbf{N}' \\ \mathbf{B}' \end{pmatrix} = \begin{pmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{pmatrix} \begin{pmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{pmatrix}$$

The matrix is skew-symmetric — a consequence of the frame being orthonormal (differentiating $\mathbf{T}\cdot\mathbf{T} = 1$ gives $\mathbf{T}'\cdot\mathbf{T} = 0$, etc.). Skew-symmetry means the frame rotates without scaling, as it must.

## The fundamental theorem of curves

**Theorem.** Given smooth functions $\kappa(s) > 0$ and $\tau(s)$ defined on an interval $I$, there exists a unit-speed curve $\alpha: I \to \mathbb{R}^3$ with curvature $\kappa$ and torsion $\tau$. This curve is unique up to rigid motion (rotation and translation).

*Proof sketch.* The Frenet-Serret equations form a linear ODE system for the frame $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$. By the Picard-Lindelof existence-uniqueness theorem for ODEs, given initial conditions (a starting frame at $s = 0$), there is a unique solution. The curve is recovered by integrating: $\alpha(s) = \alpha(0) + \int_0^s \mathbf{T}(u)\,du$. Different initial conditions (position and frame orientation) give curves related by rigid motion.

This is remarkable: two scalar functions completely determine the shape of a space curve. A helix is characterized by $\kappa = \text{const}$, $\tau = \text{const}$. A circle has $\kappa = \text{const}$, $\tau = 0$. A straight line has $\kappa = 0$. Every other curve corresponds to some pair $(\kappa(s), \tau(s))$.

## Worked example: characterizing helices

**Theorem (Lancret, 1802).** A curve with $\kappa > 0$ is a **generalized helix** (its tangent makes a constant angle with a fixed direction) if and only if $\tau/\kappa$ is constant.

*Proof.* Suppose $\mathbf{T} \cdot \mathbf{v} = \cos\theta$ for a fixed unit vector $\mathbf{v}$ and constant $\theta$. Differentiate with respect to $s$: $\mathbf{T}' \cdot \mathbf{v} = 0$, i.e., $\kappa\mathbf{N}\cdot\mathbf{v} = 0$. Since $\kappa > 0$, we get $\mathbf{N}\cdot\mathbf{v} = 0$, so $\mathbf{v}$ lies in the $\mathbf{T}$-$\mathbf{B}$ plane. Write $\mathbf{v} = \cos\theta\,\mathbf{T} + \sin\theta\,\mathbf{B}$. Differentiate again: $0 = \cos\theta\,\mathbf{T}' + \sin\theta\,\mathbf{B}' = \cos\theta\,\kappa\mathbf{N} - \sin\theta\,\tau\mathbf{N} = (\kappa\cos\theta - \tau\sin\theta)\mathbf{N}$. Since $\mathbf{N} \neq 0$, we get $\tau/\kappa = \cos\theta/\sin\theta = \cot\theta = \text{const}$.

Conversely, if $\tau/\kappa = c$ is constant, define $\theta = \text{arccot}(c)$ and $\mathbf{v} = \cos\theta\,\mathbf{T} + \sin\theta\,\mathbf{B}$. Then $\mathbf{v}' = \cos\theta\,\kappa\mathbf{N} - \sin\theta\,\tau\mathbf{N} = (\kappa\cos\theta - \tau\sin\theta)\mathbf{N} = 0$, so $\mathbf{v}$ is constant, and $\mathbf{T}\cdot\mathbf{v} = \cos\theta$ is constant.

Our circular helix satisfies $\tau/\kappa = b/a$, confirming Lancret's theorem. The fixed direction is the $z$-axis.

## Worked example: plane curves and signed curvature

For a curve $\alpha(s) = (x(s), y(s), 0)$ confined to the $xy$-plane, the Frenet apparatus simplifies. The binormal is always $\mathbf{B} = (0,0,1)$ (or $-1$), and torsion vanishes. We can define a **signed curvature** $\kappa_s$ by:

$$\alpha''(s) = \kappa_s(s)\,\mathbf{n}(s)$$

where $\mathbf{n}$ is the unit normal obtained by rotating $\mathbf{T}$ counterclockwise by $90°$. The sign carries information: $\kappa_s > 0$ means the curve bends left, $\kappa_s < 0$ means it bends right.

**Example.** A circle of radius $r$ traversed counterclockwise has $\kappa_s = +1/r$. Traversed clockwise: $\kappa_s = -1/r$. A figure-eight has $\kappa_s$ that changes sign at the crossing point.

## Worked example: total curvature and the Fenchel theorem

The **total curvature** of a closed curve is $\int_0^L \kappa(s)\,ds$.

**Theorem (Fenchel, 1929).** The total curvature of any closed curve in $\mathbb{R}^3$ satisfies $\int_0^L \kappa\,ds \geq 2\pi$, with equality if and only if the curve is a convex plane curve.

The tangent indicatrix $\mathbf{T}(s)$ traces a path on the unit sphere $S^2$, and its length equals $\int \kappa\,ds$. A closed curve has a closed tangent indicatrix, and any closed curve on $S^2$ has length at least $2\pi$ (it must go "all the way around" in some sense). Equality holds when the indicatrix is a great circle — which happens exactly for convex plane curves.

For a helix making $n$ full turns before closing, the total curvature exceeds $2\pi$ by an amount depending on the pitch. For a knotted curve (e.g., a trefoil knot), the Fary-Milnor theorem strengthens this to $\int \kappa\,ds > 4\pi$.

## What's next

We move from one-dimensional curves to two-dimensional surfaces. The first fundamental form will tell us how to measure lengths and angles on a surface — the beginning of intrinsic geometry. The idea of "curvature as the rate of change of direction" will reappear, but now there are infinitely many directions to choose from at each point, making the story considerably richer.

---

*This is Part 1 of [Differential Geometry](/en/series/differential-geometry/) (6 parts).
Next: [Part 2 — Surfaces and the First Fundamental Form](/en/differential-geometry/02-surfaces-first-form/)*
