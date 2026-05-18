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

I am going to start this series the way every honest course on differential geometry starts: with a single moving particle in $\mathbb{R}^3$. Not a manifold, not a fiber bundle, not even a surface — just a dot tracing a path. Everything later — Gauss maps, second fundamental forms, connections, Riemannian curvature tensors — is in some sense an effort to do for higher-dimensional objects what we are about to do, very thoroughly, for a one-dimensional one. So bear with the warm-up. The pay-off is that by the end of this article we will own a complete local theory: two scalar functions ($\kappa$, $\tau$) and one orthonormal frame ($\mathbf{T}, \mathbf{N}, \mathbf{B}$) that together pin down a curve up to rigid motion. The slogan is "curve = two numbers per point", and that slogan deserves a proof.

If you want one image to keep in your head while reading, keep this one.

![Frenet frame T, N, B moving along a helix in 3D](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_1_helix_frenet.png)

The triple of arrows you see — tangent, normal, binormal — slides along the helix as a rigid little gyroscope. Curvature says how fast the tangent is turning; torsion says how fast that gyroscope is twisting around the tangent. Two numbers per point. That is the whole story.

A note on tone before we start. This series is going to favour computation over decoration. I will derive things carefully but I will also push numbers through, because differential geometry has a long tradition of beautiful theorems that go fuzzy in the reader's hands the moment a real example is required. The intuition lives in the numbers. So expect a lot of helices, paraboloids, and spheres with explicit coordinates and explicit answers. Where the answer is messy, I will say so.

---

## Curves as Paths through Space

![Frenet frame on a helix: tangent T (red), normal N (green), binormal B (purple)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_frenet_serret.png)

Informally, a curve is a path traced out by a moving point. There is one subtlety to flag right away: do we mean the *image* (the set of points on the path) or the *parametrization* (the specific function describing the motion)? Differential geometers care about both, and in fact a lot of the early grunt work is making sure our definitions distinguish properties of the image from properties of the particular speedometer reading.

**Definition (Parametrized curve).** A *parametrized curve* in $\mathbb{R}^3$ is a smooth map $\alpha: I \to \mathbb{R}^3$, where $I \subseteq \mathbb{R}$ is an open interval. In coordinates,
$$\alpha(t) = \bigl(x(t),\, y(t),\, z(t)\bigr),$$
with each component $C^\infty$.

The velocity at $t$ is $\alpha'(t) = (x'(t), y'(t), z'(t))$, and the speed is $|\alpha'(t)|$.

**Definition (Regular curve).** $\alpha$ is *regular* if $\alpha'(t) \neq 0$ for every $t$.

Regularity is the smallest non-degeneracy condition we can ask for. It guarantees a well-defined tangent direction at every point and prevents collapse onto a single value of $t$. Drop it and you can engineer cusps without effort: $\alpha(t) = (t^2, t^3, 0)$ has $\alpha'(0) = 0$ and the image has a sharp corner at the origin. The curve technically still exists as a map, but the geometry has a hiccup at $t = 0$.

**Definition (Reparametrization).** A *reparametrization* of $\alpha$ is $\beta = \alpha \circ \phi$ where $\phi: J \to I$ is a diffeomorphism of intervals. Reparametrizations come in two flavours: orientation-preserving ($\phi' > 0$) and orientation-reversing ($\phi' < 0$). The image of $\beta$ is the same as the image of $\alpha$; only the labelling of points has changed.

**Why this matters.** Throughout the series I will quietly assume regularity and consider only orientation-preserving reparametrizations equivalent. Whenever you see "curve" without further qualification, read "regular smooth curve up to orientation-preserving reparametrization". The reason is purely operational: every formula involving $\mathbf{T}$, $\mathbf{N}$, or $\kappa$ requires dividing by $|\alpha'|$ at some stage.

There is also a deeper reason. The whole project of defining curvature is to extract numbers that depend on the *image* and not on the *speed*. We will see this play out concretely: $\kappa$ has a clean formula in arc-length parametrization (just $|\alpha''|$), but the formula in arbitrary parametrization is uglier and involves a cross product to "kill" the parametrization-dependent part.

### A numerical warm-up

Take the helix
$$\alpha(t) = (\cos t,\, \sin t,\, 0.4\, t).$$
Then $\alpha'(t) = (-\sin t, \cos t, 0.4)$, and $|\alpha'(t)| = \sqrt{1 + 0.16} = \sqrt{1.16} \approx 1.0770$. The speed is constant — a pleasant accident of this particular helix that makes the rest of the calculations clean.

If we instead reparametrize as $\beta(s) = \alpha(s / \sqrt{1.16})$, then $|\beta'(s)| = 1$. We have not changed the image (still the same helix); we have only relabelled the points so that traversing the curve at unit speed corresponds to advancing $s$ by one unit. This is the *arc-length parametrization*, and it is going to make life much easier in a moment.

Aside on the helix's universality. Among all curves with constant curvature *and* constant torsion, the helix is the unique example up to rigid motion. We will prove this at the end of the chapter (the Fundamental Theorem of Curves). It is a small marvel of differential geometry: from two arbitrary positive constants you can integrate up a helix, and that helix is essentially the only curve those two constants describe.

---

## Arc Length and Reparametrization

The total length of a curve from $t = a$ to $t = b$ is
$$L(\alpha; a, b) = \int_a^b |\alpha'(t)|\, dt.$$
This formula is invariant under orientation-preserving reparametrization: if $\beta = \alpha \circ \phi$ with $\phi'>0$, then by the change-of-variables formula,
$$\int_a^b |\beta'(s)|\,ds = \int_a^b |\alpha'(\phi(s))|\phi'(s)\,ds = \int_{\phi(a)}^{\phi(b)} |\alpha'(t)|\,dt.$$
Length is therefore a property of the image, not the parametrization.

![Curvature comparison: circle, helix, and general curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_curves_gallery.png)

Define the *arc-length function* with reference point $t_0$:
$$s(t) = \int_{t_0}^{t} |\alpha'(\tau)|\, d\tau.$$
Then $s'(t) = |\alpha'(t)| > 0$, so $s$ is a strictly increasing smooth function — invertible by the inverse function theorem. We can therefore reparametrize: write $t = t(s)$, and define
$$\tilde\alpha(s) = \alpha\bigl(t(s)\bigr).$$
The chain rule gives
$$\tilde\alpha'(s) = \alpha'(t)\cdot \frac{dt}{ds} = \frac{\alpha'(t)}{|\alpha'(t)|},$$
which has unit length. The curve has been relabelled to "unit speed".

![Arc length parameterization of a curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_4_arc_length.png)

**Why this matters.** Arc length is the only parameter that is *intrinsic* to the curve as a subset of $\mathbb{R}^3$ — it does not depend on the speed at which we happen to be tracing it. Curvature and torsion, defined intrinsically below, are most naturally computed in arc-length parametrization. Of course, when we have to do calculations on a specific curve we usually do not want to invert the integral $s(t)$ explicitly (and often cannot — the integrand for an ellipse already involves elliptic integrals, the namesake of those special functions). So we will derive working formulas for arbitrary parametrization and use them for actual computation, while reasoning about the theory in arc length.

### Worked example: arc length of a helix

For $\alpha(t) = (\cos t, \sin t, 0.4 t)$ on $[0, 2\pi]$:
$$L = \int_0^{2\pi}\sqrt{1.16}\,dt = 2\pi\sqrt{1.16} \approx 6.7298.$$
One full turn of the helix has length about $6.73$, slightly more than $2\pi$ (the length of the projected circle), as one would expect because the helix is "slanted upward". The arc-length parameter is $s = t\sqrt{1.16}$, so the helix at unit speed is
$$\tilde\alpha(s) = \bigl(\cos(s/\sqrt{1.16}),\, \sin(s/\sqrt{1.16}),\, 0.4\, s/\sqrt{1.16}\bigr).$$

### Worked example: ellipse and elliptic integrals

For $\alpha(t) = (a\cos t, b\sin t, 0)$ with $a > b > 0$, the speed is $|\alpha'(t)| = \sqrt{a^2\sin^2 t + b^2\cos^2 t}$. Write $a^2\sin^2 t + b^2\cos^2 t = b^2 + (a^2 - b^2)\sin^2 t = a^2(1 - e^2 \cos^2 t)$ where $e = \sqrt{1 - b^2/a^2}$ is the eccentricity. The total perimeter is
$$L = 4 a \int_0^{\pi/2}\sqrt{1 - e^2\cos^2 t}\,dt = 4 a\, E(e),$$
the complete elliptic integral of the second kind. This is one of the simplest curves whose arc length cannot be expressed in elementary functions, and it is the historical reason "elliptic integrals" got their name. So when we say "use arc-length parametrization where you can", we mean "where you can"; in practice, almost no analytic curve has a closed-form $s^{-1}$.

The moral: arc length is a beautiful theoretical parameter and a frustrating computational one. Working geometers reach for it for proofs and formulas; they reach for the parametrization-friendly cross-product formulas for examples.

---

## The Tangent Vector and Curvature

Once we are in arc-length parametrization, $\mathbf{T}(s) = \alpha'(s)$ is a unit vector — the unit tangent. Differentiating the identity $\mathbf{T}\cdot\mathbf{T} = 1$ gives
$$2\mathbf{T}\cdot \mathbf{T}' = 0,$$
so $\mathbf{T}'(s)$ is orthogonal to $\mathbf{T}(s)$. The magnitude of $\mathbf{T}'$ measures how fast the unit tangent rotates as we walk along the curve.

![Torsion measures how the curve twists out of its osculating plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_torsion.png)

**Definition (Curvature).** $\kappa(s) = |\mathbf{T}'(s)| = |\alpha''(s)|$.

A straight line has $\mathbf{T}' = 0$, hence $\kappa = 0$ identically. A circle of radius $r$ has $\kappa = 1/r$ — sharp curves (small $r$) give large $\kappa$. The "sharp turn" intuition is correct, but pinned to a precise quantity: the rate of change of the tangent direction with respect to arc length.

![Curvature kappa as the rate at which the tangent direction rotates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_2_curvature_geom.png)

If $\mathbf{T}'(s) \neq 0$, the *principal normal* is
$$\mathbf{N}(s) = \frac{\mathbf{T}'(s)}{|\mathbf{T}'(s)|} = \frac{\mathbf{T}'(s)}{\kappa(s)},$$
so that
$$\mathbf{T}'(s) = \kappa(s)\mathbf{N}(s).$$
$\mathbf{N}$ points toward the center of curvature: it is the direction the curve is "trying to bend toward". For a circle in the plane, $\mathbf{N}$ always points to the centre. For a helix, $\mathbf{N}$ points horizontally inward, toward the axis of the helix.

### Computing $\kappa$ for an arbitrary parametrization

We rarely have $|\alpha'| = 1$ in practice, so we need a chain-rule formula. With $s' = |\alpha'|$ and $\mathbf{T} = \alpha'/s'$, two derivatives of $\alpha$ give
$$\alpha'' = s''\mathbf{T} + (s')^2 \kappa \mathbf{N}.$$
Cross with $\alpha' = s'\mathbf{T}$:
$$\alpha'\times\alpha'' = (s')^3\kappa\,(\mathbf{T}\times\mathbf{N}),$$
and $|\mathbf{T}\times\mathbf{N}| = 1$ since $\mathbf{T}\perp\mathbf{N}$. Hence
$$\boxed{\kappa = \frac{|\alpha'\times\alpha''|}{|\alpha'|^3}}.$$

The cross product elegantly cancels the parametrization-dependent component $s''\mathbf{T}$ (since $\mathbf{T}\times\mathbf{T} = 0$), leaving only the parametrization-invariant geometric information. Worth pausing over: this is one of the classic "tricks of the trade" in classical differential geometry, and it shows up again and again.

**Numerical check on the helix.** For $\alpha(t) = (\cos t, \sin t, 0.4 t)$, we have $\alpha'\times\alpha'' = (0.4\sin t, -0.4\cos t, 1)$, so $|\alpha'\times\alpha''| = \sqrt{0.16 + 1} = \sqrt{1.16}$. Then
$$\kappa = \frac{\sqrt{1.16}}{1.16^{3/2}} = \frac{1}{1.16} \approx 0.8621.$$
Constant curvature, as one would expect from a uniformly tilted screw motion. Compare with a circle of unit radius: $\kappa = 1$. The helix curves slightly less because the upward drift "uses up" some of the motion that would otherwise go into bending. Physically: if you projected the helix onto the $xy$-plane you would get a unit circle ($\kappa = 1$); lifting it into 3D dilutes the curvature.

![Curvature comparison: a circle with constant kappa and a helix with kappa and tau](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_3_circle_helix.png)

### A second worked example: parabola

$\alpha(t) = (t, t^2, 0)$. Then $\alpha' = (1, 2t, 0)$, $\alpha'' = (0, 2, 0)$, $\alpha'\times\alpha'' = (0,0,2)$. So
$$\kappa(t) = \frac{2}{(1+4t^2)^{3/2}}.$$
At $t = 0$ (the vertex), $\kappa = 2$. As $|t|\to\infty$, $\kappa\to 0$ — the parabola straightens out. The radius of curvature at the vertex is $1/2$; if you placed a circle of radius $1/2$ tangent to the parabola at the origin, it would match the parabola to second order. This is the *osculating circle* — literally "kissing circle" in Latin. It is the geometric content of curvature.

As the parameter sweeps along the curve, the centre of the osculating circle traces out a new curve called the *evolute*. The evolute encodes the locus of all curvature centres and is itself a rich geometric object — for an ellipse, the evolute is a four-cusped astroid; for a parabola, it is a semicubical parabola.

![Evolute: the locus of centers of curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_evolute.png)

**Why this matters.** Curvature is the first geometric invariant of a curve. It is independent of orientation-preserving reparametrization and rigid motion. Two curves with the same $\kappa(s)$ are "almost" the same — they could still differ by how much they twist out of any plane. That residual twist is what we capture next.

---

## The Binormal and Torsion

We have two orthogonal unit vectors at every point where $\kappa > 0$: $\mathbf{T}$ and $\mathbf{N}$. The third leg of the right-handed frame is the *binormal*:
$$\mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s).$$

![Helix: constant curvature and torsion](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_arc_length.png)

The plane spanned by $\mathbf{T}$ and $\mathbf{N}$ (the *osculating plane*) contains the curve to second order at the point of evaluation. Geometrically, it is the plane that "best fits" the curve at that point.

![Osculating plane to a curve at a point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_5_osculating.png)

The plane spanned by $\mathbf{N}$ and $\mathbf{B}$ is called the *normal plane* (perpendicular to the direction of motion); the plane spanned by $\mathbf{T}$ and $\mathbf{B}$ is the *rectifying plane* (the unique plane onto which the curve, when projected, has zero curvature at the projected point). These three planes — osculating, normal, rectifying — are sometimes drawn as a little orthogonal "cross" sliding along the curve.

For a planar curve, the osculating plane is just the plane of the curve, and $\mathbf{B}$ is constant. For a genuinely 3D curve, $\mathbf{B}$ rotates — and that rotation is what torsion measures.

**Definition (Torsion).** Differentiating $\mathbf{B}\cdot\mathbf{B} = 1$ shows $\mathbf{B}'\perp\mathbf{B}$. Differentiating $\mathbf{B}\cdot\mathbf{T} = 0$ and using $\mathbf{T}' = \kappa\mathbf{N}$ shows $\mathbf{B}'\perp\mathbf{T}$ (because $\mathbf{B}'\cdot\mathbf{T} = -\mathbf{B}\cdot\mathbf{T}' = -\kappa\mathbf{B}\cdot\mathbf{N} = 0$). So $\mathbf{B}'$ is parallel to $\mathbf{N}$, and we define $\tau$ by
$$\mathbf{B}'(s) = -\tau(s)\,\mathbf{N}(s).$$
The minus sign is the standard convention; with it, a right-handed helix winding upward will have $\tau > 0$.

For an arbitrary parametrization,
$$\boxed{\tau = \frac{(\alpha'\times\alpha'')\cdot\alpha'''}{|\alpha'\times\alpha''|^2}}.$$

The derivation is similar to that of the curvature formula. Express $\alpha'''$ in the Frenet basis and dot against $\alpha'\times\alpha'' = (s')^3\kappa\mathbf{B}$; the only surviving component is the $\mathbf{B}$-component of $\alpha'''$, which involves $\tau$.

**Numerical check on the helix.** With $\alpha = (\cos t, \sin t, 0.4 t)$ we get $\alpha''' = (\sin t, -\cos t, 0)$, and $(\alpha'\times\alpha'')\cdot\alpha''' = 0.4$. So
$$\tau = \frac{0.4}{1.16} \approx 0.3448.$$
Constant. The helix has constant curvature *and* constant torsion. In fact, this is essentially a characterization (see the Fundamental Theorem below).

### Sign of torsion: a small but useful sanity check

A right-handed helix (the standard one above with positive vertical drift) has $\tau > 0$. Replacing the helix by its mirror image — say, swapping $t\mapsto -t$ in the third component, $\alpha(t) = (\cos t, \sin t, -0.4 t)$ — flips the sign of $\tau$ but leaves $\kappa$ unchanged. Torsion is therefore a *signed* quantity, sensitive to the orientation of $\mathbb{R}^3$, while curvature is unsigned. The asymmetry is a direct consequence of using the cross product, which in turn relies on the right-hand rule.

**Why this matters.** Torsion is the second geometric invariant. A planar curve has $\tau \equiv 0$; conversely, if $\tau \equiv 0$ then the curve lies in a fixed plane (proof: $\mathbf{B}$ is constant, so the function $f(s) = (\alpha(s) - \alpha(s_0))\cdot \mathbf{B}$ has zero derivative, hence is identically zero — the curve stays in the plane through $\alpha(s_0)$ orthogonal to $\mathbf{B}$). Curvature and torsion together distinguish planar wiggling from genuine 3D twisting. In the language of invariants: $\kappa$ is a $SO(3)$-invariant, $\tau$ is an $O(3)$-pseudo-invariant (it picks up a sign under orientation reversal).

---

## The Frenet-Serret Formulas

Putting the three derivatives together — $\mathbf{T}'$, $\mathbf{N}'$, $\mathbf{B}'$ — gives a tidy linear ODE for the moving frame. Since the frame is orthonormal at every point, the matrix governing its evolution must be skew-symmetric. Working out
$$\mathbf{N}' = (\mathbf{B}\times\mathbf{T})' = \mathbf{B}'\times\mathbf{T} + \mathbf{B}\times\mathbf{T}' = -\tau\mathbf{N}\times\mathbf{T} + \kappa\mathbf{B}\times\mathbf{N} = -\kappa\mathbf{T} + \tau\mathbf{B},$$
we land on:

![Osculating circle at points of varying curvature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_curvature_osculating.png)

$$\frac{d}{ds}\begin{pmatrix}\mathbf{T}\\ \mathbf{N}\\ \mathbf{B}\end{pmatrix} = \begin{pmatrix}0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0\end{pmatrix}\begin{pmatrix}\mathbf{T}\\ \mathbf{N}\\ \mathbf{B}\end{pmatrix}.$$

These are the *Frenet-Serret formulas*. The coefficient matrix is skew-symmetric, exactly as required for orthogonal frames evolving in time.

![Frenet-Serret formulas as a system of ODEs for the moving frame](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_6_frenet_serret.png)

A linear-algebra remark. The skew-symmetric matrix
$$\Omega = \begin{pmatrix}0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0\end{pmatrix}$$
has eigenvalues $0$ and $\pm i\sqrt{\kappa^2 + \tau^2}$, with the zero eigenvalue corresponding to the *Darboux vector* $\boldsymbol{\omega} = \tau\mathbf{T} + \kappa\mathbf{B}$. Geometrically, $\boldsymbol{\omega}$ is the instantaneous axis of rotation of the Frenet frame. The frame at time $s + ds$ is obtained from the frame at time $s$ by an infinitesimal rotation about $\boldsymbol{\omega}$ by an angle $|\boldsymbol{\omega}|\,ds = \sqrt{\kappa^2+\tau^2}\,ds$. So the *total angular speed* of the frame is $\sqrt{\kappa^2+\tau^2}$, decomposed into a "twisting" part ($\tau\mathbf{T}$, around the tangent) and a "bending" part ($\kappa\mathbf{B}$, around the binormal).

**Why this matters.** The Frenet-Serret system is a closed ODE for the frame. Given $\kappa(s)$, $\tau(s)$, and an initial frame at $s = 0$, we can integrate forward and recover the entire moving frame, and then recover $\alpha$ itself by integrating $\mathbf{T}$. This is the engine behind the next theorem.

![Animation: Frenet frame moving along a curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/01_frenet_moving.gif)

### Fundamental Theorem of Curves

**Theorem.** Let $\kappa: I \to \mathbb{R}_{>0}$ and $\tau: I \to \mathbb{R}$ be smooth. There exists a regular curve $\alpha: I \to \mathbb{R}^3$, parametrized by arc length, with curvature $\kappa$ and torsion $\tau$. Moreover, this curve is unique up to a rigid motion of $\mathbb{R}^3$.

*Sketch of existence.* Solve the Frenet-Serret ODE
$$F'(s) = \Omega(s)F(s),\qquad F(0) = I_3,$$
where $F$ is the $3\times 3$ matrix whose rows are $\mathbf{T}$, $\mathbf{N}$, $\mathbf{B}$ and $\Omega(s)$ is built from $\kappa(s), \tau(s)$. Skew-symmetry of $\Omega$ forces $F(s)$ to remain orthogonal: $(FF^T)' = F'F^T + F (F')^T = F\Omega^T F^T + F\Omega F^T = F(\Omega + \Omega^T)F^T = 0$, so $F(s)F(s)^T = F(0)F(0)^T = I$. Then $\alpha(s) = \int_0^s \mathbf{T}(u)\,du$ is the desired curve.

*Sketch of uniqueness.* If $\beta$ is another such curve with the same $\kappa, \tau$, apply a rigid motion to align $\beta(0)$ with $\alpha(0)$ and the Frenet frame of $\beta$ at $0$ with that of $\alpha$. Both frames now satisfy the same ODE with the same initial condition; they coincide. Integrating, $\alpha = \beta$. $\square$

This is the precise sense in which $(\kappa, \tau)$ "determine" the curve. They are the analog of the metric for 1D objects: knowing them is knowing the curve.

### Worked corollary: helix from $(\kappa, \tau)$

Suppose $\kappa$ and $\tau$ are both constant. Solve the Frenet-Serret ODE: the system has constant coefficients, so the solution is a matrix exponential. One obtains
$$\alpha(s) = \biggl(\frac{\kappa}{\kappa^2+\tau^2}\cos(\omega s),\, \frac{\kappa}{\kappa^2+\tau^2}\sin(\omega s),\, \frac{\tau}{\kappa^2+\tau^2}\,\omega s\biggr)$$
where $\omega = \sqrt{\kappa^2+\tau^2}$. This is a helix with radius $r = \kappa/(\kappa^2+\tau^2)$ and pitch (vertical advance per turn) $h = 2\pi\tau/(\kappa^2+\tau^2)$. Check: $\kappa^2+\tau^2 = \kappa^2 + \tau^2$, and $r^2 + (h/2\pi)^2 = (\kappa^2+\tau^2)/(\kappa^2+\tau^2)^2 = 1/(\kappa^2+\tau^2) = 1/\omega^2$, so $r$ and $h/2\pi$ live on a circle of radius $1/\omega$ in the $(r, h/2\pi)$-plane. As $\tau\to 0$ the helix flattens to a circle (radius $1/\kappa$, no pitch); as $\kappa\to 0$ it straightens to a line.

Set $\kappa = 1/1.16$, $\tau = 0.4/1.16$, and you recover (after rigid motion) the original $\alpha = (\cos t, \sin t, 0.4 t)$.

In other words: the helix is, up to rigid motion, *the* curve with constant curvature and constant torsion. This is the differential-geometric reason every screw, every spring, every DNA double-strand looks the same up close.

---

## A Tour of Classical Curves

Numerical examples solidify the apparatus. I will take the time to compute $\kappa$ and $\tau$ on a few standards.

![Classical curves: cardioid, lemniscate, logarithmic spiral](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/01-curves-in-space/dg_v2_01_7_classical_curves.png)

**Plane circle.** $\alpha(t) = (r\cos t, r\sin t, 0)$. Then $|\alpha'| = r$, $\alpha'\times\alpha'' = (0,0,r^2)$, $|\alpha'\times\alpha''| = r^2$, so $\kappa = r^2/r^3 = 1/r$. And $\alpha''' = (r\sin t, -r\cos t, 0)$ has zero dot product with $\alpha'\times\alpha''$ (it lies in the $xy$-plane), so $\tau = 0$. As expected.

**Cardioid.** $\alpha(t) = (a(2\cos t - \cos 2t), a(2\sin t - \sin 2t), 0)$ traces a heart shape with a single cusp at $t = 0$. The curve is regular except at the cusp; away from there, $\kappa$ can be computed and turns out to be $\kappa(t) = 3/(8 a |\sin(t/2)|)$. Notice: $\kappa\to\infty$ as $t\to 0$. The cusp is where curvature blows up — a polite way of saying the curve "violently changes direction".

**Lemniscate of Bernoulli.** $\alpha(t) = \bigl(\cos t/(1+\sin^2 t), \sin t \cos t/(1+\sin^2 t), 0\bigr)$ is the figure-eight. It is planar ($\tau = 0$) but has a self-intersection at the origin. The curve is regular as a parametrized curve but the image is not an embedded submanifold.

**Logarithmic spiral.** $\alpha(t) = (e^{kt}\cos t, e^{kt}\sin t, 0)$ for $k > 0$. A computation gives $\kappa(t) = e^{-kt}/\sqrt{1+k^2}$. So curvature decays exponentially: the spiral becomes "straighter" as it unwinds. This explains why log spirals show up in nautilus shells, pinecones, and Romanesco broccoli — biological growth patterns that scale geometrically have constant *logarithmic* derivative, exactly the property of $e^{kt}$.

The logarithmic spiral has another property worth noting: the angle between the tangent $\mathbf{T}$ and the position vector $\alpha$ is constant. (Compute $\alpha\cdot\alpha'/|\alpha||\alpha'|$.) This is the *equiangular* property, and it characterizes the log spiral among all plane curves.

**Viviani's curve.** $\alpha(t) = (1+\cos t, \sin t, 2\sin(t/2))$ is the intersection of the cylinder $(x-1)^2+y^2 = 1$ with the sphere $x^2+y^2+z^2 = 4$. Both $\kappa$ and $\tau$ are non-trivial here; you will learn more from computing them yourself than from me writing it out. The shape is a figure-eight on the sphere, sometimes called a "spherical lemniscate".

**Astroid.** $\alpha(t) = (\cos^3 t, \sin^3 t, 0)$ has cusps at $t = 0, \pi/2, \pi, 3\pi/2$. Between cusps the curvature is finite, but at cusps it blows up. The astroid is planar but full of corners; it is the trace of a point on a small circle rolling inside a larger one (a *hypocycloid*).

**Toroidal knot.** $\alpha(t) = ((R + r\cos pt)\cos qt, (R + r\cos pt)\sin qt, r\sin pt)$ with coprime $p, q$ traces a closed curve on a torus that wraps $p$ times the small way and $q$ times the long way. The $(2,3)$-knot is the trefoil; $(3,2)$ is also a trefoil (mirror image). The curvature and torsion oscillate but remain bounded. Knots are an entry point to topology, and the connection between geometry and topology will become explicit in our Gauss-Bonnet chapter.

---

## Limits, Generalizations, and What Comes Next

A few honest caveats about the local theory.

**The Frenet frame fails when $\kappa = 0$.** At an inflection point the principal normal $\mathbf{N}$ is undefined. This is not a defect of the curve but of the frame. The *Bishop frame* (a "relatively parallel" alternative) replaces $\mathbf{N}$ and $\mathbf{B}$ by two normal vectors that are parallel-transported along the curve, and exists everywhere. It is the standard tool in computer graphics and tubing/extrusion algorithms, where the Frenet frame's tendency to spin near inflection points causes textures and twisting cross-sections to flicker. The Bishop frame is unique up to a choice of initial rotation, so it has one continuous degree of freedom; the Frenet frame is canonical wherever $\kappa > 0$.

**Higher dimensions.** In $\mathbb{R}^n$, a curve has a generalized Frenet frame $(\mathbf{e}_1, \ldots, \mathbf{e}_n)$ at each point and $n - 1$ curvature functions $\kappa_1, \ldots, \kappa_{n-1}$, related by an $n\times n$ skew-symmetric matrix of derivatives. The Fundamental Theorem extends. We will not need this in the series, but it is good to know the apparatus generalizes cleanly.

**Smoothness.** I have assumed $C^\infty$ everywhere. For the basic theorems we only need $C^3$ (so that $\alpha'''$ exists and is continuous). For most engineering applications $C^2$ is enough to talk about $\kappa$ but not $\tau$. Splines, piecewise-polynomial curves common in CAD, are typically only $C^2$ at knot points, and torsion can jump discontinuously there even if curvature is continuous.

**Closed curves.** A closed curve is one with $\alpha(s + L) = \alpha(s)$ for all $s$ (where $L$ is the period). For closed planar curves, the *total curvature* $\int_0^L \kappa\,ds$ is an integer multiple of $2\pi$ (the winding number times $2\pi$); this is the Whitney-Graustein theorem in disguise. We will encounter the analog for surfaces in the Gauss-Bonnet chapter; the connection is not a coincidence.

**Knot energy.** For a closed curve, the integral $\int_0^L \kappa^2\,ds$ is called the *bending energy*. Minimizing it among closed curves of fixed length and topological type leads to *elastica* (in the planar case) and to the more elaborate world of knot energies. These are not idle curiosities: they appear in modelling DNA supercoiling, plant stems, and anything else that treats the bending of a 1D filament as a physical quantity.

### A longer worked example: where $\kappa$ alone fails to distinguish curves

To drive home the point that curvature alone is not enough in 3D, consider two curves with identical $\kappa(s)$ but different $\tau(s)$.

Curve A: a circle of radius $1$ in the $xy$-plane, $\alpha_A(s) = (\cos s, \sin s, 0)$. Constant $\kappa_A = 1$, $\tau_A = 0$.

Curve B: the helix with constant $\kappa_B = 1$ and constant $\tau_B = 0.5$. By the Fundamental Theorem this is a uniquely determined helix (up to rigid motion); a quick computation gives radius $r = \kappa/(\kappa^2+\tau^2) = 1/1.25 = 0.8$ and pitch $h = 2\pi\tau/(\kappa^2+\tau^2) = \pi$. So $\alpha_B(s)$ traces a helix of radius $0.8$ around the $z$-axis, climbing $\pi$ per revolution.

Both have $\kappa \equiv 1$. They look nothing alike. The difference is entirely in the torsion: zero for the planar circle, $0.5$ for the helix. If you only had access to a curvature function, you would mistake them for the same curve.

A reverse experiment: fix $\tau \equiv 0$ and choose any smooth $\kappa$. The Fundamental Theorem yields a unique curve up to rigid motion; this curve is *planar* (we proved this above). So planar curves are exactly those for which $\tau$ vanishes identically, and within the planar world, $\kappa$ alone determines the curve up to rigid motion. The full 3D world is genuinely two-functional.

### Curvature in coordinates: a Maple-style sanity script

For sanity, here is the kind of pencil-and-paper computation worth running through once:

Take $\alpha(t) = (t, t^3, t^2)$. Then $\alpha' = (1, 3t^2, 2t)$, $\alpha'' = (0, 6t, 2)$, $\alpha''' = (0, 6, 0)$.

Cross product: $\alpha'\times\alpha'' = \det\begin{pmatrix}\mathbf{i}&\mathbf{j}&\mathbf{k}\\1&3t^2&2t\\0&6t&2\end{pmatrix} = (6t^2 - 12t^2, -2 + 0, 6t - 0) = (-6t^2, -2, 6t)$.

Magnitudes: $|\alpha'|^2 = 1 + 9t^4 + 4t^2$, $|\alpha'\times\alpha''|^2 = 36t^4 + 4 + 36t^2 = 4(9t^4 + 9t^2 + 1)$.

Curvature: $\kappa(t) = 2\sqrt{9t^4 + 9t^2 + 1}/(1 + 4t^2 + 9t^4)^{3/2}$.

At $t = 0$: $\kappa = 2$. At $t = 1$: $\kappa = 2\sqrt{19}/14^{3/2} \approx 2(4.359)/52.38 \approx 0.166$. So the curve bends sharply near the origin and straightens out as $t$ grows, much like the parabola but in 3D.

Triple product for torsion: $(\alpha'\times\alpha'')\cdot\alpha''' = (-6t^2)\cdot 0 + (-2)\cdot 6 + 6t\cdot 0 = -12$. Constant. So
$$\tau(t) = \frac{-12}{4(9t^4 + 9t^2 + 1)} = \frac{-3}{9t^4 + 9t^2 + 1}.$$
At $t = 0$, $\tau = -3$. At $t = 1$, $\tau = -3/19 \approx -0.158$. Negative torsion: the curve is "left-handed" under the standard right-hand convention.

This is what real computation of $\kappa$ and $\tau$ looks like. The formulas are mechanical; the algebra is finicky. Most working differential geometers verify their hand computations with a CAS for anything more complicated than a helix, and so should you.

### Curves on surfaces: foreshadowing the next chapter

A curve does not have to live in $\mathbb{R}^3$ — it can live on a surface, like a great circle on a sphere or a contour line on a hillside. When a curve lies on a surface, its curvature decomposes into two pieces: the *normal curvature* (how much the curve bends because the surface itself is bending) and the *geodesic curvature* (how much the curve "turns within" the surface). A geodesic is a curve with zero geodesic curvature — it bends only as much as the surface forces it to. We will spend a lot of time on geodesics in chapter 4.

The decomposition is striking: the same curve, viewed in $\mathbb{R}^3$, has one curvature; viewed on the surface, it has two. The second piece, $\kappa_g$, is *intrinsic* — it can be computed from the surface's metric alone, without reference to how the surface is embedded in space. The first piece, $\kappa_n$, is *extrinsic*. This distinction will dominate the rest of the series.

### Why I belabour the helix

I have come back to the helix three times in this article, and I will come back to it again. The reason is pedagogical and unapologetic: the helix is to differential geometry what the simple harmonic oscillator is to physics. It is the unique non-trivial curve with both invariants constant, it admits closed-form expressions for everything, it makes a striking 3D picture, and it shows up in nature whenever something with translational and rotational symmetry has to fit into a tube. Spring, screw, bolt, drill bit, double helix of DNA, alpha helix in a protein, the curl of a fern, the spiral staircase. Anywhere you see a long thin object that has to make the same handed turn over and over, you are seeing a helix, and locally that helix obeys the formulas in this chapter.

If at the end of the series you remember nothing else, remember this: the helix is the geometry of "constant rate of turn plus constant rate of twist". Curvature is the rate of turn; torsion is the rate of twist. Two numbers, computed in coordinates by simple cross-product formulas, generate every helix you will ever see. And then everything we do for surfaces and manifolds is a careful generalization of that idea — find the right invariants, express them in coordinates, prove they determine the geometry up to symmetry. The story scales up but the structure is already in your hands.

A final philosophical aside. The Frenet-Serret apparatus is sometimes criticized as "old-fashioned" — too tied to $\mathbb{R}^3$, too coordinate-dependent. There is some truth to that. Modern differential geometry uses connections and Lie algebras of frame bundles to talk about moving frames in greater generality, and many graduate texts skip the classical theory entirely. But there is value in seeing the machinery from the ground up: the Frenet-Serret system is a genuine mini-laboratory for the entire subject. Its skew-symmetric matrix becomes a connection one-form, its Darboux vector becomes the curvature of that connection, its Fundamental Theorem becomes the integrability statement of a flat connection on the trivial bundle. Once you have seen it concretely, the abstract version is much easier to swallow.

There is one more loose thread worth pulling. We have made a careful distinction between the curve as a parametrization (a map $\alpha: I\to \mathbb{R}^3$) and the curve as an image (a subset of $\mathbb{R}^3$). The image is what physically exists; the parametrization is a labelling we impose. The geometric invariants $\kappa$ and $\tau$ are functions of arc length, which is itself a property of the image. So when we say "this curve has constant curvature $1$", we mean that the function $\kappa$, viewed as a function on the image (parametrized however you like), takes the value $1$ at every point. That is a statement about the image, not the labelling. Conflating these two perspectives is one of the most common sources of confusion in introductory differential geometry, and resolving it pays dividends throughout the series. We will encounter the same distinction for surfaces and again for manifolds: the underlying geometric object versus a particular parametrization or chart. The invariants that survive change of parametrization or chart are the geometrically meaningful ones; everything else is bookkeeping.

With those bookkeeping conventions in place, we are ready to move up a dimension.

**Summary table.** For a regular curve $\alpha(t)$ in $\mathbb{R}^3$ with arbitrary parametrization:

| Quantity | Formula |
|---|---|
| Speed | $v = \|\alpha'\|$ |
| Unit tangent | $\mathbf{T} = \alpha'/\|\alpha'\|$ |
| Curvature | $\kappa = \|\alpha' \times \alpha''\| / \|\alpha'\|^3$ |
| Binormal | $\mathbf{B} = (\alpha' \times \alpha'') / \|\alpha' \times \alpha''\|$ |
| Principal normal | $\mathbf{N} = \mathbf{B} \times \mathbf{T}$ |
| Torsion | $\tau = (\alpha' \times \alpha'') \cdot \alpha''' / \|\alpha' \times \alpha''\|^2$ |
| Darboux vector | $\boldsymbol{\omega} = \tau\mathbf{T} + \kappa\mathbf{B}$ |
| Total bending energy | $\int \kappa^2\,ds$ |

Keep this at hand. We will not derive these from scratch again.

---

## Deeper Examples and Common Pitfalls

This section is the load-bearing one. The earlier sections gave the definitions; the goal here is to compute hard enough to feel the definitions push back, to point out the places where beginners slip, and to connect each abstract piece to a setting where it actually pays for itself.

### A worked example for arc length and reparametrization

Take the cubic curve $\gamma(t) = (t, t^2, \tfrac{2}{3} t^3)$ on $t \in [0, 1]$. Its velocity is $\gamma'(t) = (1, 2t, 2t^2)$ and the speed simplifies dramatically:
$$|\gamma'(t)|^2 = 1 + 4t^2 + 4t^4 = (1 + 2t^2)^2,$$
so $|\gamma'(t)| = 1 + 2t^2$. The arc length up to parameter $t$ is then
$$s(t) = \int_0^t (1 + 2u^2)\, du = t + \tfrac{2}{3} t^3.$$
At $t=1$ we get $s = 5/3$. Solving $s = t + \tfrac{2}{3} t^3$ for $t$ in closed form requires Cardano, but for any specific value of $s$ you can find $t$ numerically. The point is that the existence of the unit-speed reparametrization $\tilde{\gamma}(s) = \gamma(t(s))$ is guaranteed by $|\gamma'| > 0$, even when the explicit formula is ugly. This is exactly why every theorem in this chapter is stated for unit-speed curves: assume the inverse exists, work without it, and the bookkeeping disappears.

### A worked example for curvature and torsion

The helix $\gamma(t) = (\cos 2t, \sin 2t, t)$ is the canonical sanity check. Compute:
$\gamma'(t) = (-2\sin 2t, 2\cos 2t, 1)$, so $|\gamma'| = \sqrt{4 + 1} = \sqrt{5}$, constant.
$\gamma''(t) = (-4\cos 2t, -4\sin 2t, 0)$, so $|\gamma''| = 4$.
$\gamma'''(t) = (8\sin 2t, -8\cos 2t, 0)$.

Curvature: $\kappa = |\gamma' \times \gamma''| / |\gamma'|^3$. The cross product is $(4\sin 2t, -4\cos 2t, 8)$, magnitude $\sqrt{16 + 64} = \sqrt{80} = 4\sqrt{5}$. So $\kappa = 4\sqrt{5} / (\sqrt{5})^3 = 4\sqrt{5} / 5\sqrt{5} = 4/5$. Constant.

Torsion: $\tau = \det[\gamma', \gamma'', \gamma'''] / |\gamma' \times \gamma''|^2$. The triple product expands to $\det\begin{pmatrix} -2\sin 2t & 2\cos 2t & 1 \\ -4\cos 2t & -4\sin 2t & 0 \\ 8\sin 2t & -8\cos 2t & 0 \end{pmatrix} = 1 \cdot (32\cos^2 2t + 32 \sin^2 2t) = 32$. So $\tau = 32/80 = 2/5$. Also constant.

Both invariants are constant, which is the defining property of a generalized helix. At $t=0$: $T = \gamma' / |\gamma'| = (0, 2, 1)/\sqrt{5}$, $N = \gamma'' / |\gamma''| = (-1, 0, 0)$, $B = T \times N = (0, -1, 2)/\sqrt{5}$. Verify orthonormality: $T \cdot N = 0$, $T \cdot B = (0 \cdot 0 + 2 \cdot (-1) + 1 \cdot 2)/5 = 0$, $N \cdot B = 0$, $|T|^2 = (0 + 4 + 1)/5 = 1$, $|B|^2 = (0 + 1 + 4)/5 = 1$. The frame is orthonormal, as the Frenet construction guarantees.

### Counterexample: when the Frenet frame fails

The Frenet construction needs $\kappa > 0$ to define $N = T'/|T'|$. At a point where $\kappa = 0$, the principal normal is undefined. Take $\gamma(t) = (t, 0, t^3)$. Compute $\gamma'' = (0, 0, 6t)$, which vanishes at $t = 0$. The curve passes through a momentary inflection where it locally looks like a straight line, and the binormal $B$ can flip discontinuously across $t=0$ even though $\gamma$ is $C^\infty$.

The cure is the **Bishop frame** (rotation-minimizing frame), which uses parallel transport instead of the second derivative to propagate $N$ along the curve. The lesson: the Frenet frame is a *coordinate system attached to a curve*, and like all coordinate systems it has singularities. The geometry of the curve is real, but the frame is sometimes a bad chart for it.

### Common pitfall for beginners

Beginners frequently confuse the **signed curvature** of a planar curve with the **unsigned curvature** of a space curve. The signed version takes values in $\mathbb{R}$ and tells you which way the curve is turning relative to a chosen normal; the unsigned version takes values in $[0, \infty)$ and only measures how sharply. When you flatten a space curve into the plane, the magnitudes match, but the signed version carries an extra bit of information that depends on orientation. Consequence: $\int \kappa\, ds$ for a closed planar curve gives $2\pi \cdot (\text{turning number})$ when signed and a strictly larger number when unsigned. The Whitney-Graustein theorem and the Gauss-Bonnet theorem both depend on the signed version. If you ever see a textbook claim "$\int \kappa\, ds = 2\pi$ for a simple closed curve," check whether they mean signed; for a peanut-shaped curve the unsigned integral is strictly larger than $2\pi$.

A second pitfall: confusing arc length with chord length. For a space curve from $a$ to $b$, the chord $|\gamma(b) - \gamma(a)|$ is bounded above by the arc length, with equality only when the curve is a straight line. This is the metric statement of the triangle inequality applied to the integral, and it is also why polygonal approximations to a curve give a *lower* bound on its arc length, never an upper bound.

### Where this matters in physics and engineering

In computer graphics, when you sweep a tube along a curve to render a cable or a strand of hair, you must propagate a frame along the curve to orient the cross section. Using the Frenet frame produces visible flips wherever $\kappa = 0$ (think of straight segments in a piecewise curve). Production renderers therefore use the Bishop frame, which is the parallel-transported version. The trade-off is that the Bishop frame has a global twist depending on the curve's torsion integral, which is fine for graphics but matters for ribbons in DNA modeling.

In aerospace, the trajectory of a maneuvering aircraft is described by Frenet-like equations where the analog of curvature is the *load factor* (lateral acceleration divided by gravity) and the analog of torsion is the *roll rate*. The Frenet-Serret equations, written in terms of these quantities, become the kinematic equations of the aircraft. Pilots are trained to think in $T$, $N$, $B$ without ever using those names.

In molecular biology, the writhe and twist of a closed DNA loop are integral invariants directly built from the torsion and the framing of the curve. The Calugareanu-Fuller-White theorem (linking number = twist + writhe) is a topological constraint on closed framed curves, and it has biological consequences: enzymes called topoisomerases manage exactly these integer invariants when DNA replicates.

### A second worked example: closed planar curves and the turning number

To see the signed/unsigned distinction concretely, parametrize a cardioid: $\gamma(\theta) = ((1-\cos\theta)\cos\theta, (1-\cos\theta)\sin\theta)$ for $\theta \in [0, 2\pi]$. The cusp at $\theta=0$ makes $\gamma'(0) = 0$, so technically this is not a regular curve. Replace it by an ellipse $\gamma(\theta) = (a\cos\theta, b\sin\theta)$ with $a \neq b$. Compute $\gamma'(\theta) = (-a\sin\theta, b\cos\theta)$, $\gamma''(\theta) = (-a\cos\theta, -b\sin\theta)$. The signed planar curvature is
$$\kappa(\theta) = \frac{x' y'' - y' x''}{(x'^2 + y'^2)^{3/2}} = \frac{ab}{(a^2 \sin^2\theta + b^2 \cos^2\theta)^{3/2}}.$$
For $a = 2, b = 1$, $\kappa$ ranges from $\kappa_{\min} = 1/4$ (at the long axis tips) to $\kappa_{\max} = 2$ (at the short axis tips). The total turning is $\int_0^{2\pi} \kappa(\theta) |\gamma'(\theta)|\, d\theta = 2\pi$ exactly, by the rotation index theorem; the unsigned integral over the same interval gives the same number for a convex curve and strictly more for a non-convex one. This is the cleanest place to feel that "total signed curvature" is a *topological* invariant — it does not change when you smoothly deform the ellipse — while "total unsigned curvature" is geometric and does. The fact that $\int \kappa\, ds$ over a simple closed curve always equals $2\pi$ regardless of shape is your first taste of Gauss-Bonnet, which we will spend article 5 proving.

### Revisiting "what's next" with sharper questions

The next article moves from one-dimensional curves to two-dimensional surfaces. The transition is bigger than it looks because surfaces no longer have a canonical parametrization — there is no natural arc length, only an *area form*. To prepare for that, three questions to keep in mind:

(1) On a curve, "speed" is a scalar; on a surface, the analog is a *bilinear form* (the first fundamental form) eating two tangent vectors. Why is one number not enough?
(2) On a curve, curvature was defined by how fast $T$ rotates. On a surface, there is no single $T$; the analog is the *shape operator*, a linear map on the tangent plane. How do you collapse a linear map into invariants the same way curvature collapsed a derivative?
(3) On a curve, intrinsic and extrinsic geometry coincide trivially (arc length is intrinsic, curvature is extrinsic, end of story). On a surface, the two come apart, and the gap between them is what the entire rest of the series is about. What is the right way to keep them separate?

The Frenet machinery you now have is exactly the right preparation: it taught you to read a curve through a *frame* attached to it. The next article does the same for surfaces, but the frame becomes two-dimensional and the equations governing how it changes are matrix-valued. Read the next article asking "how do these formulas reduce to Frenet-Serret if I restrict the surface to a curve on it?" — they all do, and tracking the reduction is the cleanest way to ground the new abstractions.

### One last worked example: the cycloid and total turning

The cycloid $\gamma(t) = (t - \sin t, 1 - \cos t)$ traced by a point on a rolling unit circle is a great test case. Compute $\gamma'(t) = (1 - \cos t, \sin t)$ and $|\gamma'(t)|^2 = (1 - \cos t)^2 + \sin^2 t = 2(1 - \cos t) = 4\sin^2(t/2)$, so $|\gamma'(t)| = 2|\sin(t/2)|$. The arc length over one full arch ($t \in [0, 2\pi]$) is $\int_0^{2\pi} 2 \sin(t/2)\, dt = -4\cos(t/2)|_0^{2\pi} = 8$. So one arch has length exactly $8$, a clean integer despite the curve being transcendental. This is a famous result of Wren (1659).

The signed planar curvature: $\kappa = (x'y'' - y'x'')/|\gamma'|^3$. Compute $x'' = \sin t$, $y'' = \cos t$, so $x'y'' - y'x'' = (1-\cos t)\cos t - \sin t \cdot \sin t = \cos t - \cos^2 t - \sin^2 t = \cos t - 1$. So $\kappa = (\cos t - 1)/(2\sin(t/2))^3 = -2\sin^2(t/2)/(8\sin^3(t/2)) = -1/(4\sin(t/2))$. The curvature blows up at the cusps ($t = 0, 2\pi$) and equals $-1/4$ at the top of the arch ($t = \pi$). The cusps are precisely the points where $|\gamma'| = 0$, where the regularity hypothesis of the Frenet-Serret machinery fails. Total signed turning over one arch: $\int_0^{2\pi} \kappa(t) |\gamma'(t)|\, dt = \int_0^{2\pi} -2\sin(t/2) /(4\sin(t/2))\, dt = -\pi$. So the cycloid arch turns through $\pi$ radians — a half-turn — consistent with the cusps adding the missing turning to make the rolling-circle picture coherent.

## What's Next

We now have the complete local theory of curves. The Frenet-Serret apparatus provides a moving orthonormal frame and two scalar invariants ($\kappa, \tau$) which determine the curve up to rigid motion. The whole story is governed by ODEs.

The next chapter moves to surfaces — two-dimensional objects in $\mathbb{R}^3$. The jump from one dimension to two is qualitatively different. Instead of a single tangent vector we have a tangent *plane*; instead of two scalar invariants we have the *first fundamental form* — a $2\times 2$ matrix-valued function — and later a second one. The theory will shift from ODEs to bilinear algebra and PDEs, and the distinction between *intrinsic* and *extrinsic* geometry, which has barely surfaced for curves (whose intrinsic geometry is trivially that of an interval), will become the entire game.

The deepest result waiting for us in that direction is Gauss's *Theorema Egregium*: even though Gaussian curvature is defined extrinsically (in terms of how the surface bends in space), it can in fact be computed entirely from the intrinsic metric. A flatlander could measure it from inside the surface. That is the kind of theorem that gets named "egregious".

---
