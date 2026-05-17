---
title: "Curves in Space: Curvature, Torsion, and the Frenet Frame"
date: 2021-11-01 09:00:00
tags:
  - differential-geometry
  - curves
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 12
translationKey: "differential-geometry-1"
description: "Parametrized curves, arc length, curvature, torsion, and the Frenet-Serret apparatus — the complete local theory of space curves."
---

Differential geometry begins with curves. Before we can study the curvature of surfaces or the abstract machinery of manifolds, we need a precise language for describing how a one-dimensional object — a curve — bends and twists as it moves through Euclidean space. This chapter develops that language from scratch. By the end, we will have the Frenet-Serret formulas, a compact system of ordinary differential equations that encodes all the local geometry of a space curve in two scalar functions: curvature and torsion.

---

## Curves as Paths through Space

A curve in $\mathbb{R}^3$ is, informally, a path traced out by a moving point. But there is already a subtlety lurking here: should we think of a curve as the *image* — the set of points the path passes through — or as the *parametrization* — the specific function describing the motion?

**Definition (Parametrized curve).** A *parametrized curve* in $\mathbb{R}^3$ is a smooth map $\alpha: I \to \mathbb{R}^3$, where $I \subseteq \mathbb{R}$ is an open interval. In coordinates we write

![Frenet frame (T, N, B) moving along a helix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_fig1_frenet.png)


$$\alpha(t) = \bigl(x(t),\, y(t),\, z(t)\bigr),$$

where each component function is $C^\infty$ (infinitely differentiable).

The velocity vector at $t$ is $\alpha'(t) = (x'(t), y'(t), z'(t))$, and the speed is $|\alpha'(t)|$.

**Definition (Regular curve).** A parametrized curve $\alpha$ is *regular* if $\alpha'(t) \neq 0$ for all $t \in I$.

Regularity is a minimal nondegeneracy condition: it ensures that the curve has a well-defined tangent direction at every point, and that the image of $\alpha$ is genuinely one-dimensional. A curve that fails regularity at a point $t_0$ can develop a cusp there (consider $\alpha(t) = (t^2, t^3, 0)$ at $t = 0$, where $\alpha'(0) = (0,0,0)$).

Two parametrized curves $\alpha: I \to \mathbb{R}^3$ and $\beta: J \to \mathbb{R}^3$ are *reparametrizations* of each other if there exists a smooth bijection $\phi: J \to I$ with $\phi' > 0$ everywhere such that $\beta = \alpha \circ \phi$. The condition $\phi' > 0$ preserves orientation — the direction of travel. A reparametrization changes the speed at which we traverse the curve but not the geometric shape of the image.

**Example 1 (Circle).** The map $\alpha(t) = (\cos t, \sin t, 0)$ for $t \in (0, 2\pi)$ is a parametrized curve whose image is the unit circle in the $xy$-plane. We have $\alpha'(t) = (-\sin t, \cos t, 0)$ and $|\alpha'(t)| = 1$, so the speed is constant and the curve is certainly regular. The reparametrization $\beta(s) = (\cos 2s, \sin 2s, 0)$ for $s \in (0, \pi)$ traces the same circle twice as fast.

**Example 2 (Circular helix).** The curve $\alpha(t) = (a\cos t,\, a\sin t,\, bt)$ for constants $a > 0$, $b \neq 0$ traces a helix of radius $a$ and pitch $2\pi b$. The velocity is $\alpha'(t) = (-a\sin t, a\cos t, b)$, so $|\alpha'(t)| = \sqrt{a^2 + b^2}$, which is constant. The helix is regular, and it provides a fundamental test case for everything that follows.

The distinction between a parametrized curve and its image matters throughout differential geometry. The image of $\alpha(t) = (t, t^2, 0)$ and $\beta(t) = (t^3, t^6, 0)$ is the same parabola, but they are different parametrized curves with different velocity fields. Most of our constructions (curvature, torsion, the Frenet frame) depend on the parametrization, although the most important quantities turn out to be invariant under orientation-preserving reparametrization.

---

## Arc Length and the Natural Parameter

Among all parametrizations of a given curve, there is a canonical one: parametrization by arc length. It is the differential-geometric analogue of choosing "natural" units.

**Definition (Arc length).** Let $\alpha: I \to \mathbb{R}^3$ be a regular curve. The *arc length function* starting from a fixed point $t_0 \in I$ is

$$s(t) = \int_{t_0}^{t} |\alpha'(u)|\, du.$$

Since $\alpha$ is regular, $|\alpha'(u)| > 0$ for all $u$, so $s'(t) = |\alpha'(t)| > 0$, and $s$ is a strictly increasing function. By the inverse function theorem, $s$ has a smooth inverse $t = t(s)$, and we can reparametrize the curve as $\beta(s) = \alpha(t(s))$.

**Proposition.** A parametrized curve $\beta$ is parametrized by arc length if and only if $|\beta'(s)| = 1$ for all $s$.

*Proof.* By the chain rule,

$$\beta'(s) = \alpha'(t(s)) \cdot t'(s) = \alpha'(t(s)) \cdot \frac{1}{s'(t(s))} = \frac{\alpha'(t(s))}{|\alpha'(t(s))|},$$

so $|\beta'(s)| = 1$. Conversely, if $|\beta'(s)| = 1$, then the arc length computed from any starting point $s_0$ is $\int_{s_0}^{s} 1\, du = s - s_0$, which is the identity map up to a constant shift. $\square$

Arc length parametrization is elegant but rarely practical for computation: the integral $\int |\alpha'(u)|\, du$ seldom has a closed form. For instance, the arc length of an ellipse leads to an elliptic integral. This is why we develop formulas that work for *arbitrary* parametrizations, not just unit-speed ones.

**Example 3 (Arc length of the helix).** For $\alpha(t) = (a\cos t, a\sin t, bt)$ we already computed $|\alpha'(t)| = \sqrt{a^2 + b^2}$. Taking $t_0 = 0$:

$$s(t) = \int_0^t \sqrt{a^2 + b^2}\, du = t\sqrt{a^2 + b^2}.$$

So $t = s / \sqrt{a^2+b^2}$, and the arc-length reparametrization is

$$\beta(s) = \left(a\cos\frac{s}{\sqrt{a^2+b^2}},\; a\sin\frac{s}{\sqrt{a^2+b^2}},\; \frac{bs}{\sqrt{a^2+b^2}}\right).$$

One can verify directly that $|\beta'(s)| = 1$.

The arc length between two points $\alpha(t_1)$ and $\alpha(t_2)$ is independent of the parametrization: if $\beta = \alpha \circ \phi$, then

$$\int_{s_1}^{s_2} |\beta'(u)|\, du = \int_{\phi(s_1)}^{\phi(s_2)} |\alpha'(v)|\, dv$$

by the substitution $v = \phi(u)$. This confirms that arc length is a geometric — not parametric — quantity. It is the first *intrinsic* measurement associated with a curve.

---

## Curvature: Measuring How a Curve Bends

Once we have a curve parametrized by arc length, the tangent vector $\mathbf{T}(s) = \beta'(s)$ is a unit vector. The rate at which this unit tangent rotates as we move along the curve is a measure of bending.

**Definition (Curvature, unit-speed case).** Let $\beta: J \to \mathbb{R}^3$ be a unit-speed curve. The *curvature* at $s$ is

$$\kappa(s) = |\mathbf{T}'(s)| = |\beta''(s)|.$$

Since $\mathbf{T}$ has constant length 1, we have $\mathbf{T} \cdot \mathbf{T} = 1$, so differentiating gives $\mathbf{T}' \cdot \mathbf{T} = 0$. The acceleration $\mathbf{T}'$ is always perpendicular to $\mathbf{T}$: it measures how much the direction changes, not the speed. This is the geometric content of curvature.

A straight line has $\mathbf{T}$ constant, so $\kappa \equiv 0$. A circle of radius $R$ traversed at unit speed has $\kappa = 1/R$. The quantity $R_\kappa = 1/\kappa$ is the *radius of curvature*: it is the radius of the circle that best approximates the curve at a given point (the *osculating circle*).

For a curve with arbitrary parametrization $\alpha(t)$, we need a formula that does not require computing arc length explicitly.

**Proposition (Curvature formula for arbitrary parameter).** If $\alpha: I \to \mathbb{R}^3$ is a regular curve, then

$$\kappa(t) = \frac{|\alpha'(t) \times \alpha''(t)|}{|\alpha'(t)|^3}.$$

*Proof.* Write $\alpha(t) = \beta(s(t))$ where $\beta$ is the arc-length reparametrization and $s' = |\alpha'|$. Then:

$$\alpha' = \beta' s' = \mathbf{T}\, s',$$

$$\alpha'' = \mathbf{T}' (s')^2 + \mathbf{T}\, s'' = \kappa\,\mathbf{N}\,(s')^2 + \mathbf{T}\, s'',$$

where $\mathbf{N}$ is the unit principal normal (defined in the next section). Now compute the cross product:

$$\alpha' \times \alpha'' = \mathbf{T}\,s' \times \bigl[\kappa\,\mathbf{N}\,(s')^2 + \mathbf{T}\,s''\bigr] = \kappa\,(s')^3\,(\mathbf{T} \times \mathbf{N}) + 0.$$

The second term vanishes because $\mathbf{T} \times \mathbf{T} = 0$. Since $\mathbf{T} \perp \mathbf{N}$ and both are unit vectors, $|\mathbf{T} \times \mathbf{N}| = 1$, giving $|\alpha' \times \alpha''| = \kappa\, (s')^3 = \kappa\, |\alpha'|^3$. $\square$

This formula is the workhorse for computation. It requires only the first and second derivatives of $\alpha$ and avoids the arc-length integral entirely.

**Example 4 (Curvature of a circle of radius $R$).** Take $\alpha(t) = (R\cos t, R\sin t, 0)$. Then $\alpha' = (-R\sin t, R\cos t, 0)$, $\alpha'' = (-R\cos t, -R\sin t, 0)$, and

$$\alpha' \times \alpha'' = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ -R\sin t & R\cos t & 0 \\ -R\cos t & -R\sin t & 0 \end{vmatrix} = (0,\, 0,\, R^2\sin^2 t + R^2\cos^2 t) = (0,\, 0,\, R^2).$$

So $|\alpha' \times \alpha''| = R^2$, $|\alpha'| = R$, and $\kappa = R^2 / R^3 = 1/R$. The curvature of a circle of radius $R$ is $1/R$ — exactly as intuition suggests. A tighter circle bends more.

**Example 5 (Curvature of the circular helix).** For $\alpha(t) = (a\cos t, a\sin t, bt)$:

$$\alpha' = (-a\sin t, a\cos t, b), \quad \alpha'' = (-a\cos t, -a\sin t, 0).$$

The cross product:

$$\alpha' \times \alpha'' = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ -a\sin t & a\cos t & b \\ -a\cos t & -a\sin t & 0 \end{vmatrix} = (ab\sin t,\; -ab\cos t,\; a^2).$$

Thus $|\alpha' \times \alpha''| = \sqrt{a^2 b^2 + a^4} = a\sqrt{a^2 + b^2}$ and $|\alpha'|^3 = (a^2 + b^2)^{3/2}$, so

$$\kappa = \frac{a\sqrt{a^2 + b^2}}{(a^2 + b^2)^{3/2}} = \frac{a}{a^2 + b^2}.$$

The curvature of a circular helix is constant. When $b = 0$ this reduces to $1/a$ (a circle of radius $a$), and as $b \to \infty$ (the helix stretches out toward a vertical line) the curvature tends to zero. Both limiting cases match geometric intuition perfectly.

For a plane curve $\alpha(t) = (x(t), y(t))$, the general formula simplifies to:

$$\kappa = \frac{|x'y'' - y'x''|}{(x'^2 + y'^2)^{3/2}}.$$

One can also define a *signed curvature* $\kappa_s = (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2}$ for oriented plane curves, which is positive when the curve turns counterclockwise and negative when it turns clockwise. The signed curvature completely determines a plane curve up to rigid motion — this is the Fundamental Theorem of Plane Curves, a two-dimensional precursor of the theorem we will state for space curves below.

---

## The Frenet-Serret Frame: T, N, B

At each point of a regular curve with nonvanishing curvature, we can construct an orthonormal basis of $\mathbb{R}^3$ that moves with the curve. This is the Frenet-Serret frame, and it is the fundamental tool for analyzing the local geometry of space curves.

**Definition (Unit tangent).** For a unit-speed curve $\beta(s)$, the *unit tangent vector* is

$$\mathbf{T}(s) = \beta'(s).$$

We have $|\mathbf{T}| = 1$ by the unit-speed assumption.

**Definition (Principal normal).** If $\kappa(s) \neq 0$, the *principal normal vector* is

$$\mathbf{N}(s) = \frac{\mathbf{T}'(s)}{|\mathbf{T}'(s)|} = \frac{\mathbf{T}'(s)}{\kappa(s)}.$$

We showed that $\mathbf{T}' \perp \mathbf{T}$, so $\mathbf{N}$ is a unit vector perpendicular to $\mathbf{T}$. The principal normal points in the direction the curve is turning — toward the center of the osculating circle. The osculating circle has radius $1/\kappa$ and its center lies at $\beta(s) + (1/\kappa)\,\mathbf{N}(s)$.

**Definition (Binormal).** The *binormal vector* is

$$\mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s).$$

Since $\mathbf{T}$ and $\mathbf{N}$ are orthogonal unit vectors, $\mathbf{B}$ is also a unit vector, perpendicular to both, and $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$ forms a positively oriented orthonormal basis of $\mathbb{R}^3$ at each point.

The triple $(\mathbf{T}, \mathbf{N}, \mathbf{B})$ is the *Frenet-Serret frame* (also called the *Frenet frame* or *moving trihedron*). It depends on the point $\beta(s)$ and rotates as $s$ varies. The entire content of the Frenet-Serret theory is to describe *how* this frame rotates.

**Geometric meaning of the associated planes.** The plane spanned by $\mathbf{T}$ and $\mathbf{N}$ is the *osculating plane* — the plane in which the curve "most nearly lies" at that point. If you freeze the curve's bending at a given instant, the resulting circle lies in the osculating plane. The plane spanned by $\mathbf{N}$ and $\mathbf{B}$ is the *normal plane*, perpendicular to the direction of travel. The plane spanned by $\mathbf{T}$ and $\mathbf{B}$ is the *rectifying plane*.

For a plane curve (one that lies entirely in a fixed plane), the binormal $\mathbf{B}$ is a constant vector (the unit normal to that plane), and the osculating plane equals the plane of the curve at every point.

**Example 6 (Frenet frame of the helix).** We computed the arc-length parametrization of the helix: set $c = \sqrt{a^2 + b^2}$, then $s = ct$ and

$$\beta(s) = \left(a\cos\frac{s}{c},\; a\sin\frac{s}{c},\; \frac{bs}{c}\right).$$

Differentiating:

$$\mathbf{T}(s) = \beta'(s) = \left(-\frac{a}{c}\sin\frac{s}{c},\; \frac{a}{c}\cos\frac{s}{c},\; \frac{b}{c}\right).$$

$$\mathbf{T}'(s) = \left(-\frac{a}{c^2}\cos\frac{s}{c},\; -\frac{a}{c^2}\sin\frac{s}{c},\; 0\right).$$

So $|\mathbf{T}'| = a/c^2 = \kappa$ (consistent with $\kappa = a/(a^2+b^2) = a/c^2$). The principal normal:

$$\mathbf{N}(s) = \frac{\mathbf{T}'}{\kappa} = \left(-\cos\frac{s}{c},\; -\sin\frac{s}{c},\; 0\right).$$

The principal normal of the helix always points horizontally inward, toward the axis of the cylinder — confirming the geometric picture of a curve winding around a cylinder and being pulled inward at each point.

The binormal:

$$\mathbf{B}(s) = \mathbf{T} \times \mathbf{N} = \left(\frac{b}{c}\sin\frac{s}{c},\; -\frac{b}{c}\cos\frac{s}{c},\; \frac{a}{c}\right).$$

One can verify directly that $|\mathbf{B}| = 1$ and $\mathbf{B} \cdot \mathbf{T} = \mathbf{B} \cdot \mathbf{N} = 0$. Notice that $\mathbf{B}$ has a constant vertical component $a/c$, so the osculating plane of the helix makes a constant angle with the horizontal.

**Computing the Frenet frame for an arbitrary parametrization.** Given $\alpha(t)$ (not necessarily unit-speed), the Frenet frame can be computed directly:

$$\mathbf{T} = \frac{\alpha'}{|\alpha'|}, \quad \mathbf{B} = \frac{\alpha' \times \alpha''}{|\alpha' \times \alpha''|}, \quad \mathbf{N} = \mathbf{B} \times \mathbf{T}.$$

The formula for $\mathbf{B}$ works because we showed that $\alpha' \times \alpha''$ is parallel to $\mathbf{T} \times \mathbf{N} = \mathbf{B}$. This is the practical recipe: compute derivatives, take cross products, normalize.

---

## Torsion: Measuring How a Curve Twists Out of a Plane

Curvature measures how rapidly the tangent vector $\mathbf{T}$ rotates — equivalently, how much the curve deviates from a straight line. Torsion measures how rapidly the osculating plane rotates — equivalently, how much the curve deviates from being planar.

**Definition (Torsion).** For a unit-speed curve with $\kappa > 0$, the *torsion* $\tau$ is defined by

$$\tau(s) = -\mathbf{B}'(s) \cdot \mathbf{N}(s),$$

or equivalently by the equation $\mathbf{B}' = -\tau\, \mathbf{N}$.

The negative sign is a convention that makes right-handed helices have positive torsion.

**Why is $\mathbf{B}'$ proportional to $\mathbf{N}$?** Since $|\mathbf{B}| = 1$, we have $\mathbf{B}' \perp \mathbf{B}$. Differentiating $\mathbf{B} \cdot \mathbf{T} = 0$ gives:

$$\mathbf{B}' \cdot \mathbf{T} + \mathbf{B} \cdot \mathbf{T}' = \mathbf{B}' \cdot \mathbf{T} + \kappa\,(\mathbf{B} \cdot \mathbf{N}) = \mathbf{B}' \cdot \mathbf{T} + 0 = 0.$$

So $\mathbf{B}' \perp \mathbf{T}$ as well. Being perpendicular to both $\mathbf{B}$ and $\mathbf{T}$, it must be a scalar multiple of $\mathbf{N}$. We write this scalar as $-\tau$.

**Proposition (Planar curves have zero torsion).** A regular curve with $\kappa > 0$ lies in a plane if and only if $\tau \equiv 0$.

*Proof.* ($\Rightarrow$) If $\alpha$ lies in a plane with unit normal $\mathbf{n}$, then $\alpha' \cdot \mathbf{n} = 0$, so $\mathbf{T} \cdot \mathbf{n} = 0$. Since $\mathbf{T}' = \kappa\mathbf{N}$ lies in the same plane (it is a derivative of a vector tangent to the plane, hence tangent to the plane), $\mathbf{N} \cdot \mathbf{n} = 0$ as well. Therefore $\mathbf{B} = \mathbf{T} \times \mathbf{N}$ is parallel to $\mathbf{n}$, hence constant, and $\mathbf{B}' = 0$ gives $\tau = 0$.

($\Leftarrow$) If $\tau = 0$ then $\mathbf{B}' = 0$, so $\mathbf{B}$ is a constant vector $\mathbf{B}_0$. Consider $f(s) = (\beta(s) - \beta(s_0)) \cdot \mathbf{B}_0$. Then $f'(s) = \mathbf{T}(s) \cdot \mathbf{B}_0 = 0$ for all $s$ (since $\mathbf{T} \perp \mathbf{B}$ at every point), so $f \equiv 0$. This means $\beta(s) - \beta(s_0)$ is always perpendicular to $\mathbf{B}_0$, i.e., $\beta(s)$ lies in the plane through $\beta(s_0)$ with normal $\mathbf{B}_0$. $\square$

This theorem gives a clean criterion: a space curve is actually planar if and only if its torsion vanishes identically. The geometric content is that $\tau = 0$ means the osculating plane is not rotating at all — it stays the same plane.

**Torsion formula for arbitrary parametrization.** If $\alpha(t)$ is a regular curve with $\kappa > 0$:

$$\tau = \frac{(\alpha' \times \alpha'') \cdot \alpha'''}{|\alpha' \times \alpha''|^2}.$$

*Derivation outline.* Starting from $\alpha' = s'\mathbf{T}$ and $\alpha'' = s''\mathbf{T} + \kappa(s')^2 \mathbf{N}$, differentiate once more. After considerable bookkeeping with the Frenet formulas, one obtains:

$$\alpha''' = [s''' - \kappa^2(s')^3]\,\mathbf{T} + [3\kappa\, s'\, s'' + \kappa'(s')^2]\,\mathbf{N} + \kappa\,\tau\,(s')^3\,\mathbf{B}.$$

Taking the dot product with $\alpha' \times \alpha'' = \kappa(s')^3\,\mathbf{B}$:

$$(\alpha' \times \alpha'') \cdot \alpha''' = \kappa^2\,\tau\,(s')^6.$$

Since $|\alpha' \times \alpha''|^2 = \kappa^2(s')^6$, dividing gives $\tau$.

The numerator $(\alpha' \times \alpha'') \cdot \alpha'''$ is the scalar triple product of the first three derivatives. It has a pleasant geometric interpretation: it measures the volume of the parallelepiped spanned by $\alpha'$, $\alpha''$, $\alpha'''$. This volume is zero precisely when the three derivatives are coplanar — which happens exactly when the curve lies in a plane (consistent with $\tau = 0$).

**Example 7 (Torsion of the helix).** For $\alpha(t) = (a\cos t, a\sin t, bt)$:

$$\alpha' = (-a\sin t, a\cos t, b), \quad \alpha'' = (-a\cos t, -a\sin t, 0), \quad \alpha''' = (a\sin t, -a\cos t, 0).$$

We computed $\alpha' \times \alpha'' = (ab\sin t, -ab\cos t, a^2)$, so $|\alpha' \times \alpha''|^2 = a^2 b^2 + a^4 = a^2(a^2 + b^2)$.

The scalar triple product:

$$(\alpha' \times \alpha'') \cdot \alpha''' = (ab\sin t)(a\sin t) + (-ab\cos t)(-a\cos t) + a^2 \cdot 0 = a^2 b\sin^2 t + a^2 b\cos^2 t = a^2 b.$$

Therefore $\tau = a^2 b / [a^2(a^2 + b^2)] = b/(a^2 + b^2)$.

The torsion of the circular helix is constant, just like its curvature. When $b > 0$ (right-handed helix), $\tau > 0$; when $b < 0$ (left-handed), $\tau < 0$. When $b = 0$, the helix degenerates to a circle and $\tau = 0$.

Note the symmetry: $\kappa = a/(a^2+b^2)$ and $\tau = b/(a^2+b^2)$. The ratio $\tau/\kappa = b/a = \tan\theta$, where $\theta$ is the pitch angle of the helix. This is a purely geometric relationship.

**Example 8 (Twisted cubic).** Consider $\alpha(t) = (t, t^2, t^3)$. At $t = 0$:

$$\alpha'(0) = (1, 0, 0), \quad \alpha''(0) = (0, 2, 0), \quad \alpha'''(0) = (0, 0, 6).$$

$$\alpha'(0) \times \alpha''(0) = (0, 0, 2), \quad |\alpha'(0) \times \alpha''(0)|^2 = 4.$$

$$(\alpha'(0) \times \alpha''(0)) \cdot \alpha'''(0) = 12, \quad \tau(0) = 12/4 = 3.$$

Also $|\alpha'(0)| = 1$, so $\kappa(0) = 2/1 = 2$. The twisted cubic at the origin has curvature 2 and torsion 3. The Frenet frame there: $\mathbf{T} = (1,0,0)$, $\mathbf{B} = (0,0,1)$, $\mathbf{N} = \mathbf{B} \times \mathbf{T} = (0,1,0)$. The osculating plane at the origin is the $xy$-plane.

For general $t$, the calculation is more involved. We have:

$$\alpha'(t) = (1, 2t, 3t^2), \quad \alpha''(t) = (0, 2, 6t), \quad \alpha'''(t) = (0, 0, 6).$$

$$\alpha' \times \alpha'' = (12t^2 - 6t^2,\, -6t,\, 2) = (6t^2,\, -6t,\, 2).$$

$$|\alpha' \times \alpha''|^2 = 36t^4 + 36t^2 + 4.$$

$$(\alpha' \times \alpha'') \cdot \alpha''' = 12.$$

So $\tau(t) = 12/(36t^4 + 36t^2 + 4)$, which decreases from $\tau(0) = 3$ as $|t|$ increases. The torsion is everywhere positive, confirming that the twisted cubic consistently twists in the same direction.

---

## The Frenet-Serret Formulas and the Fundamental Theorem

We have defined $\kappa$ and $\tau$ by specifying how $\mathbf{T}'$ and $\mathbf{B}'$ relate to the frame vectors. The derivative of $\mathbf{N}$ can be deduced from these. The result is a clean system of ODEs that encodes the complete local geometry of a space curve.

**Theorem (Frenet-Serret formulas).** Let $\beta(s)$ be a unit-speed curve with $\kappa(s) > 0$. Then:

$$\begin{aligned}
\mathbf{T}' &= \kappa\,\mathbf{N}, \\
\mathbf{N}' &= -\kappa\,\mathbf{T} + \tau\,\mathbf{B}, \\
\mathbf{B}' &= -\tau\,\mathbf{N}.
\end{aligned}$$

In matrix form:

$$\frac{d}{ds}\begin{pmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{pmatrix} = \begin{pmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{pmatrix} \begin{pmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{pmatrix}.$$

**The skew-symmetric structure.** The coefficient matrix is skew-symmetric (antisymmetric): $\Omega^T = -\Omega$. This is not a coincidence but a necessity. The Frenet frame is orthonormal, meaning the matrix $F = (\mathbf{T}\ \mathbf{N}\ \mathbf{B})$ satisfies $F^T F = I$. Differentiating: $F'^T F + F^T F' = 0$, so $F^T F' = -(F^T F')^T$, i.e., $F^T F' = \Omega$ is skew-symmetric. Physically, a skew-symmetric matrix generates rotations: the Frenet frame rotates rigidly as we move along the curve, with instantaneous angular velocity given by the *Darboux vector* $\boldsymbol{\omega} = \tau\,\mathbf{T} + \kappa\,\mathbf{B}$.

*Derivation of $\mathbf{N}' = -\kappa\,\mathbf{T} + \tau\,\mathbf{B}$.* We derive the middle equation, the only one not given by definition. Expand $\mathbf{N}' = a\,\mathbf{T} + b\,\mathbf{N} + c\,\mathbf{B}$ and determine the coefficients using orthonormality:

- $b = \mathbf{N}' \cdot \mathbf{N} = \frac{1}{2}\frac{d}{ds}|\mathbf{N}|^2 = 0$.
- $a = \mathbf{N}' \cdot \mathbf{T}$. Differentiating $\mathbf{N} \cdot \mathbf{T} = 0$: $\mathbf{N}' \cdot \mathbf{T} + \mathbf{N} \cdot \mathbf{T}' = 0$, so $a = -\mathbf{N} \cdot \kappa\mathbf{N} = -\kappa$.
- $c = \mathbf{N}' \cdot \mathbf{B}$. Differentiating $\mathbf{N} \cdot \mathbf{B} = 0$: $\mathbf{N}' \cdot \mathbf{B} + \mathbf{N} \cdot \mathbf{B}' = 0$, so $c = -\mathbf{N} \cdot (-\tau\mathbf{N}) = \tau$.

Therefore $\mathbf{N}' = -\kappa\,\mathbf{T} + \tau\,\mathbf{B}$. $\square$

Alternatively, differentiate $\mathbf{N} = \mathbf{B} \times \mathbf{T}$ using the product rule:

$$\mathbf{N}' = \mathbf{B}' \times \mathbf{T} + \mathbf{B} \times \mathbf{T}' = (-\tau\mathbf{N}) \times \mathbf{T} + \mathbf{B} \times (\kappa\mathbf{N}).$$

Using $\mathbf{N} \times \mathbf{T} = -\mathbf{B}$ and $\mathbf{B} \times \mathbf{N} = -\mathbf{T}$:

$$\mathbf{N}' = \tau\mathbf{B} - \kappa\mathbf{T}.$$

Both approaches give the same answer, as they must.

**Example 9 (Verifying the Frenet-Serret formulas for the helix).** We computed the Frenet frame of the helix in Example 6. Let us verify the formula $\mathbf{N}' = -\kappa\mathbf{T} + \tau\mathbf{B}$. The principal normal is $\mathbf{N}(s) = (-\cos(s/c), -\sin(s/c), 0)$, so:

$$\mathbf{N}'(s) = \left(\frac{1}{c}\sin\frac{s}{c},\; -\frac{1}{c}\cos\frac{s}{c},\; 0\right).$$

On the other hand:

$$-\kappa\,\mathbf{T} + \tau\,\mathbf{B} = -\frac{a}{c^2}\left(-\frac{a}{c}\sin\frac{s}{c},\; \frac{a}{c}\cos\frac{s}{c},\; \frac{b}{c}\right) + \frac{b}{c^2}\left(\frac{b}{c}\sin\frac{s}{c},\; -\frac{b}{c}\cos\frac{s}{c},\; \frac{a}{c}\right).$$

The first component: $\frac{a^2}{c^3}\sin\frac{s}{c} + \frac{b^2}{c^3}\sin\frac{s}{c} = \frac{a^2+b^2}{c^3}\sin\frac{s}{c} = \frac{1}{c}\sin\frac{s}{c}$. Similarly for the other components. The third component: $-\frac{ab}{c^3} + \frac{ab}{c^3} = 0$. Everything checks out.

The Frenet-Serret formulas show that $\kappa(s)$ and $\tau(s)$ completely determine the evolution of the moving frame along the curve. The next theorem says they also determine the curve itself.

**Theorem (Fundamental Theorem of Space Curves).** Let $\kappa: I \to \mathbb{R}_{>0}$ and $\tau: I \to \mathbb{R}$ be smooth functions on an interval $I$. Then:

1. **(Existence)** There exists a unit-speed curve $\beta: I \to \mathbb{R}^3$ whose curvature is $\kappa$ and whose torsion is $\tau$.
2. **(Uniqueness)** If $\gamma: I \to \mathbb{R}^3$ is another unit-speed curve with the same $\kappa$ and $\tau$, then $\gamma$ differs from $\beta$ by a rigid motion of $\mathbb{R}^3$: there exist $A \in SO(3)$ and $\mathbf{b} \in \mathbb{R}^3$ such that $\gamma(s) = A\,\beta(s) + \mathbf{b}$ for all $s \in I$.

*Proof sketch.* **Existence.** The Frenet-Serret formulas form a $9 \times 9$ linear ODE system (three vector equations, each with three components). Given an initial point $p_0 \in \mathbb{R}^3$ and an initial orthonormal frame $(\mathbf{T}_0, \mathbf{N}_0, \mathbf{B}_0)$, the Picard-Lindelof theorem guarantees a unique global solution $(\mathbf{T}(s), \mathbf{N}(s), \mathbf{B}(s))$ on all of $I$.

We need to verify that the solution remains orthonormal. Define the $6$ inner products $f_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$ where $\mathbf{e}_1 = \mathbf{T}, \mathbf{e}_2 = \mathbf{N}, \mathbf{e}_3 = \mathbf{B}$. One can show that the $f_{ij}$ satisfy a linear ODE system with initial conditions $f_{ij}(s_0) = \delta_{ij}$, and the constant solution $f_{ij} \equiv \delta_{ij}$ also satisfies this system. By uniqueness, $f_{ij} = \delta_{ij}$ for all $s$, so the frame remains orthonormal. (This is the key step — it uses the skew-symmetry of the coefficient matrix.)

Once $\mathbf{T}(s)$ is known, define $\beta(s) = p_0 + \int_{s_0}^{s} \mathbf{T}(u)\, du$. Then $\beta'(s) = \mathbf{T}(s)$, so $\beta$ is unit-speed, and by construction its curvature and torsion are $\kappa$ and $\tau$.

**Uniqueness.** Suppose $\beta$ and $\gamma$ have the same $\kappa, \tau$. Choose $s_0$ and let $A \in SO(3)$ be the rotation mapping the Frenet frame of $\beta$ at $s_0$ to that of $\gamma$, and let $\mathbf{b} = \gamma(s_0) - A\beta(s_0)$. Set $\tilde\beta(s) = A\beta(s) + \mathbf{b}$. The curve $\tilde\beta$ has the same curvature and torsion as $\beta$ (rigid motions preserve both), and at $s_0$ its Frenet frame matches $\gamma$'s. Both $\tilde\beta$ and $\gamma$ solve the Frenet-Serret ODE with identical initial conditions, so by uniqueness of ODE solutions, $\tilde\beta = \gamma$. $\square$

The Fundamental Theorem tells us that a space curve is completely characterized, up to its position and orientation in ambient space, by the two functions $\kappa(s)$ and $\tau(s)$. These are the *natural invariants* of the curve. No matter how complicated a space curve looks, its shape is encoded in just two functions of one variable. This is both a compression result and a classification result.

**Corollary.** A curve with constant $\kappa > 0$ and constant $\tau$ is (a portion of) a circular helix. If additionally $\tau = 0$, it is a circle.

*Proof.* We computed that the circular helix $\alpha(t) = (a\cos t, a\sin t, bt)$ with $a = \kappa/(\kappa^2+\tau^2)$ and $b = \tau/(\kappa^2+\tau^2)$ has the given curvature and torsion. By the Fundamental Theorem, any other curve with the same constants must be a rigid motion of this helix. $\square$

**Remark (Curves with vanishing curvature).** The Frenet frame requires $\kappa > 0$ to define $\mathbf{N}$. At a point where $\kappa = 0$ (an inflection point), the curve instantaneously travels in a straight line and there is no preferred normal direction. The theory can be extended using the *Bishop frame* (or *relatively parallel adapted frame*), which replaces $\mathbf{N}$ and $\mathbf{B}$ with two normal vectors that are parallel-transported along the curve rather than determined by the curvature direction. The Bishop frame is always defined (even when $\kappa = 0$) and is widely used in computer graphics and engineering, where the Frenet frame's tendency to spin rapidly near inflection points is undesirable.

**Summary of formulas.** For a regular curve $\alpha(t)$ in $\mathbb{R}^3$ with arbitrary parametrization:

| Quantity | Formula |
|---|---|
| Speed | $v = |\alpha'|$ |
| Unit tangent | $\mathbf{T} = \alpha'/|\alpha'|$ |
| Curvature | $\kappa = |\alpha' \times \alpha''| / |\alpha'|^3$ |
| Principal normal | $\mathbf{N} = \mathbf{B} \times \mathbf{T}$ |
| Binormal | $\mathbf{B} = (\alpha' \times \alpha'') / |\alpha' \times \alpha''|$ |
| Torsion | $\tau = (\alpha' \times \alpha'') \cdot \alpha''' / |\alpha' \times \alpha''|^2$ |

---

## What's Next

We have developed the complete local theory of curves in $\mathbb{R}^3$: the Frenet-Serret apparatus gives us a moving orthonormal frame, two scalar invariants ($\kappa$ and $\tau$), and a fundamental theorem asserting that these invariants determine the curve up to rigid motion. This is, in a sense, the warm-up act: curves are one-dimensional objects, and their geometry is governed by ODEs.

The next chapter moves to surfaces — two-dimensional objects in $\mathbb{R}^3$. The jump from one dimension to two is qualitatively different. Instead of a single tangent vector, we will have a tangent *plane* at each point. Instead of curvature and torsion (two functions of one variable), we will encounter the *first fundamental form* — a $2 \times 2$ matrix of functions that encodes how to measure lengths, angles, and areas on the surface. The theory will shift from ODEs to bilinear algebra and eventually to PDEs, and the distinction between *intrinsic* and *extrinsic* geometry will become central.

In the curve setting, there is no real intrinsic geometry: a curve "from the inside" is just a copy of $\mathbb{R}$ (or an interval), and all the interesting geometry — curvature, torsion — comes from how the curve sits in the ambient space. For surfaces, the situation is fundamentally different: a surface has its own internal metric, its own notion of distance and angle, which can be measured by a being living on the surface with no knowledge of the surrounding $\mathbb{R}^3$. Understanding this intrinsic geometry begins with the first fundamental form, and it will eventually lead us to Gauss's *Theorema Egregium* — the remarkable theorem that Gaussian curvature is an intrinsic invariant, detectable from within the surface itself.

---

*This is Part 1 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Next: [Part 2 — Surfaces and the First Fundamental Form](/en/differential-geometry/02-surfaces-first-form/)*
