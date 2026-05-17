---
title: "Differential Geometry (9): Integration on Manifolds and Stokes' Theorem"
date: 2021-11-17 09:00:00
tags:
  - differential-geometry
  - stokes-theorem
  - integration
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
description: "Stokes' theorem — the fundamental theorem of calculus on manifolds — unifies Green's, Gauss's, and the classical Stokes' theorems into one elegant statement."
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "differential-geometry-9"
---

In single-variable calculus, the fundamental theorem says that integrating a derivative over an interval equals the boundary difference: $\int_a^b f'(x)\,dx = f(b) - f(a)$. Every "fundamental theorem" you have ever met — Green's theorem, the divergence theorem, the classical Stokes' theorem — is a higher-dimensional version of this same idea. The goal of this article is to state and understand the single result that unifies them all: **Stokes' theorem on manifolds**.

To get there we need two things we do not yet have: a notion of **integration** on manifolds, and a notion of **boundary**. Both require care — a manifold has no ambient coordinates to lean on, so we must build integration intrinsically from differential forms and orientations.

Before diving in, let us recall where we stand in the series. We have defined smooth manifolds, tangent and cotangent bundles, differential forms, and the exterior derivative $d$. We know that $d^2 = 0$ and that $d$ satisfies the graded Leibniz rule. The exterior derivative is a *local* operation. What we need now is a *global* operation — integration — that converts differential forms into numbers. The interplay between the local $d$ and the global $\int$ is the content of Stokes' theorem.

---

## Integration needs orientation

### Why orientation matters

Consider trying to integrate a 2-form $\omega$ over a surface $S$ sitting in $\mathbb{R}^3$. At each point we pick a local parametrization, pull $\omega$ back to $\mathbb{R}^2$, and integrate. But a parametrization comes with a choice: do we traverse the surface with the normal pointing "up" or "down"? Reversing the parametrization flips the sign of the integral. For integration to be well-defined, we need a **consistent global choice** — an orientation.

![Stokes theorem unifies all classical integral theorems](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_fig9_stokes.png)


To make this concrete, consider the unit sphere $S^2$ parametrized in two different ways. Using the standard spherical coordinates $(\theta, \varphi) \mapsto (\sin\theta\cos\varphi, \sin\theta\sin\varphi, \cos\theta)$, the outward normal points away from the origin. If we instead use the map $(\theta, \varphi) \mapsto (\sin\theta\cos\varphi, -\sin\theta\sin\varphi, \cos\theta)$ — note the sign flip in the second component — the induced normal flips direction, and any integral of a 2-form over this parametrization acquires a minus sign. Both parametrizations cover the same surface, but they induce opposite orientations. Integration demands that we pick one and stick with it globally.

### Orientable manifolds

An $n$-dimensional smooth manifold $M$ is **orientable** if there exists a nowhere-vanishing $n$-form $\Omega \in \Omega^n(M)$. Such a form is called a **volume form**. Two volume forms $\Omega$ and $\Omega'$ define the same orientation if $\Omega' = f\Omega$ for some everywhere-positive smooth function $f > 0$. An **oriented manifold** is a manifold together with a choice of equivalence class of volume forms.

**Examples.**

- $\mathbb{R}^n$ is orientable: $\Omega = dx^1 \wedge \cdots \wedge dx^n$ is a global volume form.
- The sphere $S^n$ is orientable for every $n$. On $S^2 \subset \mathbb{R}^3$, the area form $\omega = x\,dy \wedge dz + y\,dz \wedge dx + z\,dx \wedge dy$ restricted to the sphere is a volume form.
- The Mobius band is **not** orientable: any attempt to define a consistent normal direction fails when you go around the band once.

**Equivalent characterization.** $M$ is orientable if and only if we can choose an atlas $\{(U_\alpha, \varphi_\alpha)\}$ such that all transition functions $\varphi_\beta \circ \varphi_\alpha^{-1}$ have positive Jacobian determinant everywhere. This is the condition that local coordinate orientations are globally compatible.

**The orientation double cover.** Every connected non-orientable manifold $M$ has a canonical double cover $\tilde{M}$ that *is* orientable. For the Mobius band, the double cover is a cylinder. For the Klein bottle, the double cover is a torus. The orientation double cover is connected if and only if $M$ is non-orientable; if $M$ is already orientable, the double cover is two disjoint copies of $M$.

### Volume forms on Riemannian manifolds

If $M$ carries a Riemannian metric $g$, then an orientation determines a canonical volume form: in local oriented coordinates,

$$\text{vol}_g = \sqrt{\det(g_{ij})}\, dx^1 \wedge \cdots \wedge dx^n.$$

This is the form that, when integrated, gives the Riemannian volume. On the unit sphere $S^2$ with the round metric, $\text{vol}_g = \sin\theta\, d\theta \wedge d\varphi$, which integrates to $4\pi$.

For the hyperbolic plane $\mathbb{H}^2$ with the Poincare upper half-plane metric $g = \frac{dx^2 + dy^2}{y^2}$, the volume form is $\text{vol}_g = \frac{dx \wedge dy}{y^2}$. The "volume" (area) of the region $\{(x,y) : 0 \le x \le 1, y \ge 1\}$ is $\int_0^1 \int_1^\infty \frac{1}{y^2}\,dy\,dx = 1$, a finite number despite the region extending to infinity — a characteristic feature of hyperbolic geometry.

---

## Integration of $n$-forms on oriented manifolds

### Integration on $\mathbb{R}^n$

We start with the base case. If $\omega = f(x)\, dx^1 \wedge \cdots \wedge dx^n$ is a compactly supported $n$-form on an open subset $U \subseteq \mathbb{R}^n$, we define

$$\int_U \omega = \int_U f(x)\, dx^1 \cdots dx^n,$$

where the right-hand side is the ordinary Lebesgue (or Riemann) integral.

### Integration on a manifold via charts

For a compactly supported $n$-form $\omega$ on an oriented $n$-manifold $M$, if $\text{supp}(\omega)$ lies inside a single oriented chart $(U, \varphi)$, we define

$$\int_M \omega = \int_{\varphi(U)} (\varphi^{-1})^*\omega.$$

The pullback $(\varphi^{-1})^*\omega$ is a compactly supported $n$-form on an open subset of $\mathbb{R}^n$, so we know how to integrate it.

**Key fact.** If $(V, \psi)$ is another oriented chart containing $\text{supp}(\omega)$, the change-of-variables formula for integrals (with positive Jacobian, thanks to the orientation) guarantees the same answer. This is precisely why we need orientation: without the positive-Jacobian condition, chart changes could introduce sign flips.

### Partition of unity

In general, $\text{supp}(\omega)$ may not fit inside a single chart. We handle this with a **partition of unity**: a collection of smooth functions $\{\rho_\alpha\}$ subordinate to a locally finite open cover $\{U_\alpha\}$ such that

1. $0 \le \rho_\alpha \le 1$ and $\text{supp}(\rho_\alpha) \subseteq U_\alpha$,
2. $\sum_\alpha \rho_\alpha = 1$ everywhere on $M$.

We then define

$$\int_M \omega = \sum_\alpha \int_M \rho_\alpha\, \omega,$$

where each $\rho_\alpha\, \omega$ is supported inside $U_\alpha$ and can be integrated via a single chart. A standard argument shows the result is independent of the choice of partition of unity and cover.

**Remark.** The existence of smooth partitions of unity is a fundamental property of smooth manifolds — it is what makes the passage from local to global possible in differential geometry.

### Properties of integration

Several important properties follow directly from the definition:

1. **Linearity:** $\int_M (a\omega + b\eta) = a\int_M \omega + b\int_M \eta$.
2. **Orientation dependence:** If $\bar{M}$ denotes $M$ with the opposite orientation, then $\int_{\bar{M}} \omega = -\int_M \omega$.
3. **Diffeomorphism invariance:** If $\phi : N \to M$ is an orientation-preserving diffeomorphism, then $\int_M \omega = \int_N \phi^*\omega$. This is the abstract change-of-variables formula.
4. **Positivity:** If $\omega$ is a volume form (compatible with the orientation), then $\int_M \omega > 0$.

These properties show that the integral is a well-defined, coordinate-independent operation that respects the geometry of the manifold.

---

## Manifolds with boundary

### Definition

A **manifold with boundary** $M$ is a topological space locally modeled on the upper half-space

$$\mathbb{H}^n = \{(x^1, \ldots, x^n) \in \mathbb{R}^n : x^n \ge 0\}.$$

Points that map to the interior of $\mathbb{H}^n$ are interior points of $M$; points that map to the hyperplane $\{x^n = 0\}$ form the **boundary** $\partial M$. The boundary $\partial M$ is itself a smooth $(n-1)$-manifold without boundary.

**Examples.**

- The closed unit disk $\bar{D}^2 = \{(x,y) : x^2 + y^2 \le 1\}$ is a 2-manifold with boundary $\partial \bar{D}^2 = S^1$.
- The closed unit ball $\bar{B}^n$ has boundary $\partial \bar{B}^n = S^{n-1}$.
- A compact surface with a hole punched out, like a cylinder $S^1 \times [0,1]$, has boundary consisting of two circles.
- A closed interval $[a,b]$ is a 1-manifold with boundary $\partial [a,b] = \{a, b\}$ — just two points.

### Induced orientation on the boundary

If $M$ is an oriented $n$-manifold with boundary, the boundary $\partial M$ inherits a natural orientation. The convention (which makes Stokes' theorem work with the correct sign) is the **outward-normal-first** convention: at a boundary point $p \in \partial M$, let $\nu$ be an outward-pointing vector (not tangent to $\partial M$). We say a basis $(e_1, \ldots, e_{n-1})$ of $T_p(\partial M)$ is positively oriented if $(\nu, e_1, \ldots, e_{n-1})$ is a positively oriented basis of $T_pM$.

**Example.** For $M = [a,b]$, the outward normal at $b$ points to the right (positive direction), so the induced orientation at $b$ is $+1$; at $a$ it points left, giving $-1$. Hence $\int_{\partial [a,b]} f = f(b) - f(a)$, recovering the boundary term in the fundamental theorem of calculus.

**Example.** For the closed unit disk $\bar{D}^2$ with the standard orientation (counterclockwise), the outward normal along $\partial \bar{D}^2 = S^1$ points radially outward. The induced orientation on $S^1$ is counterclockwise — the same direction as the standard parametrization $t \mapsto (\cos t, \sin t)$. This is consistent with the convention in Green's theorem: the boundary curve is traversed so that the interior is on the left.

**Example.** For the closed ball $\bar{B}^3$ with the standard orientation of $\mathbb{R}^3$, the outward normal on $\partial \bar{B}^3 = S^2$ points radially outward, inducing the standard orientation of $S^2$. This is why the divergence theorem involves the *outward* flux through the boundary surface.

### The collar neighborhood theorem

A useful structural result is the **collar neighborhood theorem**: if $M$ is a manifold with boundary $\partial M$, then $\partial M$ has a neighborhood in $M$ diffeomorphic to $\partial M \times [0, 1)$. This means the boundary is not "pinched" or pathological — it always looks locally like a product. This theorem is essential for constructing the partitions of unity needed in the proof of Stokes' theorem, ensuring that boundary charts behave well.

---

## Stokes' theorem: statement and proof outline

We now have all the ingredients: differential forms, exterior derivative, oriented manifolds with boundary, and integration.

### The theorem

> **Stokes' Theorem.** Let $M$ be a compact oriented $n$-dimensional smooth manifold with boundary $\partial M$ (given the induced orientation), and let $\omega$ be a smooth $(n-1)$-form on $M$. Then
>
> $$\int_M d\omega = \int_{\partial M} \omega.$$

The beauty of this statement is its simplicity: the integral of $d\omega$ over the "bulk" equals the integral of $\omega$ over the boundary. No vector fields, no dot products, no cross products — just forms and their exterior derivatives.

### Proof outline

The proof proceeds in three steps:

**Step 1: Reduce to charts.** Using a partition of unity $\{\rho_\alpha\}$, write $\omega = \sum_\alpha \rho_\alpha \omega$. By linearity of both integration and the exterior derivative, it suffices to prove the theorem for each $\rho_\alpha \omega$, which is compactly supported in a single chart.

**Step 2: Prove it on $\mathbb{R}^n$ (interior chart).** If $\text{supp}(\omega)$ lies entirely in the interior of $M$, we must show $\int_{\mathbb{R}^n} d\omega = 0$ (since there is no boundary contribution). Write $\omega = \sum_i f_i\, dx^1 \wedge \cdots \wedge \widehat{dx^i} \wedge \cdots \wedge dx^n$, where $\widehat{dx^i}$ means that factor is omitted. Then $d\omega = \sum_i (-1)^{i-1} \frac{\partial f_i}{\partial x^i} dx^1 \wedge \cdots \wedge dx^n$. Since each $f_i$ is compactly supported, $\int_{-\infty}^{\infty} \frac{\partial f_i}{\partial x^i} dx^i = 0$ by the fundamental theorem of calculus (the function vanishes at $\pm \infty$). So $\int_{\mathbb{R}^n} d\omega = 0$.

**Step 3: Prove it on $\mathbb{H}^n$ (boundary chart).** If $\text{supp}(\omega)$ meets the boundary $\{x^n = 0\}$, the same computation shows all terms vanish except the one involving $\frac{\partial f_n}{\partial x^n}$. For that term, $\int_0^\infty \frac{\partial f_n}{\partial x^n} dx^n = -f_n(x^1, \ldots, x^{n-1}, 0)$ (the function vanishes at $+\infty$ but not at $0$). The resulting integral over $\{x^n = 0\}$ is precisely $\int_{\partial \mathbb{H}^n} \omega$, with the sign working out correctly thanks to the outward-normal-first orientation convention.

Summing over all charts completes the proof. The key insight is that the entire argument reduces to the one-dimensional fundamental theorem of calculus, applied one variable at a time.

### Immediate consequences

Several important results follow directly from Stokes' theorem:

**Closed forms on closed manifolds.** If $M$ is a compact manifold *without* boundary (a "closed" manifold), then for any $(n-1)$-form $\omega$, $\int_M d\omega = \int_{\partial M} \omega = \int_\emptyset \omega = 0$. In other words, exact $n$-forms integrate to zero on closed manifolds. This is the starting point for de Rham cohomology.

**Conservation laws.** If $d\omega = 0$ (the form is closed), then $\int_{\partial M} \omega = 0$ for any manifold $M$ over which $\omega$ extends. This is the abstract form of a conservation law: the total flux of a conserved quantity through a closed surface vanishes.

**Homotopy invariance.** If two $(n-1)$-forms $\omega_0$ and $\omega_1$ differ by an exact form ($\omega_1 - \omega_0 = d\eta$), their integrals over any $(n-1)$-cycle (closed submanifold without boundary) agree. This is because $\int_C (\omega_1 - \omega_0) = \int_C d\eta = \int_{\partial C} \eta = 0$.

---

## Recovering the classical theorems

Stokes' theorem on manifolds is powerful because all the classical integral theorems of vector calculus are special cases. Let us see how.

### The fundamental theorem of calculus

Take $M = [a,b]$, a 1-manifold with boundary $\{a, b\}$. Let $\omega = f$ be a 0-form (function). Then $d\omega = f'\,dx$ and

$$\int_{[a,b]} f'\,dx = \int_{\partial [a,b]} f = f(b) - f(a).$$

### Green's theorem

Let $M = D$ be a compact region in $\mathbb{R}^2$ with boundary curve $\partial D$. Let $\omega = P\,dx + Q\,dy$ be a 1-form. Then $d\omega = \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx \wedge dy$, and Stokes' theorem gives

$$\iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx\,dy = \oint_{\partial D} P\,dx + Q\,dy.$$

This is Green's theorem.

**Concrete application.** Take $P = -y$ and $Q = x$, so $\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 2$. Then $\oint_{\partial D} (-y\,dx + x\,dy) = 2 \cdot \text{Area}(D)$. This gives the well-known formula for computing area via a line integral: $\text{Area}(D) = \frac{1}{2}\oint_{\partial D}(x\,dy - y\,dx)$. For the unit circle $(\cos t, \sin t)$, this gives $\frac{1}{2}\int_0^{2\pi}(\cos^2 t + \sin^2 t)\,dt = \pi$.

### The divergence theorem (Gauss's theorem)

Let $M = \Omega$ be a compact region in $\mathbb{R}^3$ with boundary surface $\partial \Omega$. Given a vector field $\mathbf{F} = (F_1, F_2, F_3)$, define the 2-form

$$\omega = F_1\, dy \wedge dz + F_2\, dz \wedge dx + F_3\, dx \wedge dy.$$

Then $d\omega = \left(\frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z}\right) dx \wedge dy \wedge dz = (\nabla \cdot \mathbf{F})\, dV$, so Stokes gives

$$\iiint_\Omega \nabla \cdot \mathbf{F}\, dV = \oiint_{\partial \Omega} \mathbf{F} \cdot d\mathbf{S}.$$

This is the divergence theorem. The 2-form $\omega$ is precisely the flux form associated to $\mathbf{F}$.

**Concrete application.** Take $\mathbf{F} = (x, y, z)$, so $\nabla \cdot \mathbf{F} = 3$. For the unit ball $\Omega = B^3$, the divergence theorem gives $\oiint_{S^2} \mathbf{F} \cdot d\mathbf{S} = 3 \cdot \text{Vol}(B^3) = 3 \cdot \frac{4\pi}{3} = 4\pi$. You can verify this directly: on the unit sphere, $\mathbf{F} \cdot \hat{n} = x^2 + y^2 + z^2 = 1$, so $\oiint_{S^2} 1 \cdot dS = \text{Area}(S^2) = 4\pi$.

### The classical Stokes' theorem (curl theorem)

Let $M = S$ be an oriented surface in $\mathbb{R}^3$ with boundary curve $\partial S$. Given a vector field $\mathbf{F}$, define the 1-form $\omega = F_1\,dx + F_2\,dy + F_3\,dz$. Then

$$d\omega = \left(\frac{\partial F_3}{\partial y} - \frac{\partial F_2}{\partial z}\right) dy \wedge dz + \left(\frac{\partial F_1}{\partial z} - \frac{\partial F_3}{\partial x}\right) dz \wedge dx + \left(\frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y}\right) dx \wedge dy,$$

which is the curl form. Stokes' theorem gives

$$\iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S} = \oint_{\partial S} \mathbf{F} \cdot d\mathbf{r}.$$

**The unification is complete.** All four theorems — fundamental theorem of calculus, Green's, divergence, and classical Stokes' — are the single statement $\int_M d\omega = \int_{\partial M} \omega$ applied to manifolds of dimension 1, 2, 2, and 3, respectively, with appropriate choices of differential form.

### A summary table

| Classical theorem | $\dim M$ | Form $\omega$ | $d\omega$ involves | Boundary $\partial M$ |
|---|---|---|---|---|
| Fundamental theorem of calculus | 1 | 0-form $f$ | $f'dx$ | Two points $\{a, b\}$ |
| Green's theorem | 2 | 1-form $Pdx + Qdy$ | $(\partial_x Q - \partial_y P)\,dx \wedge dy$ | Curve $\partial D$ |
| Divergence theorem | 3 | 2-form (flux of $\mathbf{F}$) | $(\nabla \cdot \mathbf{F})\,dV$ | Surface $\partial \Omega$ |
| Classical Stokes' | 2 (in $\mathbb{R}^3$) | 1-form $\mathbf{F} \cdot d\mathbf{r}$ | $(\nabla \times \mathbf{F}) \cdot d\mathbf{S}$ | Curve $\partial S$ |

The power of the manifold formulation is that it works in *any* dimension, on *any* oriented manifold with boundary — not just subsets of $\mathbb{R}^2$ or $\mathbb{R}^3$.

---

## De Rham cohomology: first look

Stokes' theorem has a profound topological consequence. If $\omega$ is a closed $(n-1)$-form ($d\omega = 0$) on a compact manifold $M$ without boundary, then $\int_M d\omega = 0$ trivially. But what if $\omega$ is closed but **not exact** — what if there is no $(n-2)$-form $\eta$ with $d\eta = \omega$? The obstruction to exactness carries topological information.

### Closed and exact forms

A differential $k$-form $\omega$ on $M$ is:

- **Closed** if $d\omega = 0$.
- **Exact** if $\omega = d\eta$ for some $(k-1)$-form $\eta$.

Since $d^2 = 0$, every exact form is closed. The converse fails in general, and the failure is measured by cohomology.

### De Rham cohomology groups

The **$k$-th de Rham cohomology group** of $M$ is the quotient vector space

$$H^k_{\text{dR}}(M) = \frac{\ker(d : \Omega^k(M) \to \Omega^{k+1}(M))}{\text{im}(d : \Omega^{k-1}(M) \to \Omega^k(M))} = \frac{\{\text{closed } k\text{-forms}\}}{\{\text{exact } k\text{-forms}\}}.$$

Two closed forms represent the same cohomology class if they differ by an exact form. The dimension $b_k = \dim H^k_{\text{dR}}(M)$ is the **$k$-th Betti number**.

### Examples

- **$\mathbb{R}^n$:** By the Poincare lemma, every closed form on $\mathbb{R}^n$ is exact, so $H^k_{\text{dR}}(\mathbb{R}^n) = 0$ for all $k > 0$, and $H^0 = \mathbb{R}$ (the constant functions). All Betti numbers are zero except $b_0 = 1$.

- **$S^1$:** The 1-form $d\theta$ is closed but not exact (its integral around $S^1$ is $2\pi \ne 0$). So $H^1_{\text{dR}}(S^1) \cong \mathbb{R}$, with $b_1 = 1$. This detects the "hole" in the circle.

- **$S^2$:** We have $H^0 = H^2 = \mathbb{R}$ and $H^1 = 0$, reflecting that $S^2$ is connected, has no 1-dimensional holes, and encloses a 2-dimensional void.

- **Torus $T^2$:** $H^0 = \mathbb{R}$, $H^1 = \mathbb{R}^2$, $H^2 = \mathbb{R}$. The two generators of $H^1$ correspond to the two independent loops on the torus.

- **Klein bottle $K$:** $H^0 = \mathbb{R}$, $H^1 = \mathbb{R}$, $H^2 = 0$. The vanishing of $H^2$ reflects the fact that the Klein bottle is non-orientable (it has no volume form). The single generator of $H^1$ corresponds to the "longitudinal" loop; the "transverse" loop is trivial in de Rham cohomology over $\mathbb{R}$ (though it survives in $\mathbb{Z}/2$ cohomology).

- **Genus-$g$ surface $\Sigma_g$:** $H^0 = H^2 = \mathbb{R}$ and $H^1 = \mathbb{R}^{2g}$. The $2g$ generators of $H^1$ correspond to the $g$ handles, each contributing one "meridional" and one "longitudinal" loop. The Euler characteristic is $\chi = 2 - 2g$.

### Relation to topology

The remarkable **de Rham theorem** states that de Rham cohomology is isomorphic to singular cohomology with real coefficients:

$$H^k_{\text{dR}}(M) \cong H^k(M; \mathbb{R}).$$

This means that purely analytic objects (differential forms and the exterior derivative) capture purely topological invariants (cohomology classes and Betti numbers). Stokes' theorem is the bridge: it shows that integration of closed forms over cycles depends only on the cohomology class of the form and the homology class of the cycle.

**The period map.** More precisely, Stokes' theorem shows that for a closed $k$-form $\omega$ and a $k$-cycle $C$ (a compact $k$-dimensional submanifold without boundary), the integral $\int_C \omega$ depends only on $[\omega] \in H^k_{\text{dR}}(M)$ and $[C] \in H_k(M)$. This defines a bilinear pairing

$$H^k_{\text{dR}}(M) \times H_k(M; \mathbb{R}) \to \mathbb{R}, \qquad ([\omega], [C]) \mapsto \int_C \omega.$$

De Rham's theorem says this pairing is non-degenerate — it is a perfect pairing between cohomology and homology. The numbers $\int_C \omega$ for various cycles $C$ are called the **periods** of $\omega$ and carry deep arithmetic and geometric information. In complex geometry, the periods of holomorphic forms on algebraic varieties are the subject of Hodge theory.

The Euler characteristic can be computed from Betti numbers: $\chi(M) = \sum_{k=0}^n (-1)^k b_k$. For $S^2$, $\chi = 1 - 0 + 1 = 2$. For $T^2$, $\chi = 1 - 2 + 1 = 0$. These agree with the classical values.

**Homotopy invariance of de Rham cohomology.** If two smooth manifolds $M$ and $N$ are homotopy equivalent (there exist smooth maps $f : M \to N$ and $g : N \to M$ with $g \circ f \simeq \text{id}_M$ and $f \circ g \simeq \text{id}_N$), then $H^k_{\text{dR}}(M) \cong H^k_{\text{dR}}(N)$ for all $k$. In particular, $H^k_{\text{dR}}(M)$ is a topological invariant — it does not depend on the smooth structure, only on the homotopy type. This is why cohomology can detect holes: a punctured plane $\mathbb{R}^2 \setminus \{0\}$ is homotopy equivalent to $S^1$, so $H^1(\mathbb{R}^2 \setminus \{0\}) \cong H^1(S^1) \cong \mathbb{R}$.

De Rham cohomology provides one of the most elegant connections between analysis and topology, and it will reappear when we discuss characteristic classes in later articles.

### Poincare duality

On a closed oriented $n$-manifold $M$, the wedge product and integration define a non-degenerate pairing

$$H^k_{\text{dR}}(M) \times H^{n-k}_{\text{dR}}(M) \to \mathbb{R}, \qquad ([\alpha], [\beta]) \mapsto \int_M \alpha \wedge \beta.$$

**Poincare duality** asserts that this pairing is a perfect pairing, giving an isomorphism $H^k_{\text{dR}}(M) \cong H^{n-k}_{\text{dR}}(M)^*$. In particular, $b_k = b_{n-k}$: the Betti numbers are symmetric. For a closed oriented surface ($n = 2$), this says $b_0 = b_2$, which we have already seen.

Poincare duality is one of the most fundamental results in algebraic topology, and the de Rham framework makes it particularly transparent: the pairing is just "wedge and integrate," and non-degeneracy follows from the Hodge theorem (that every cohomology class has a unique harmonic representative with respect to any Riemannian metric).

### The Mayer-Vietoris sequence

A powerful computational tool for de Rham cohomology is the **Mayer-Vietoris sequence**. If $M = U \cup V$ where $U$ and $V$ are open sets, there is a long exact sequence

$$\cdots \to H^{k-1}(U \cap V) \xrightarrow{\delta} H^k(M) \to H^k(U) \oplus H^k(V) \to H^k(U \cap V) \to \cdots$$

This allows us to compute the cohomology of $M$ from the cohomology of simpler pieces. For instance, decomposing the circle $S^1$ as two overlapping arcs gives $H^0(S^1) = \mathbb{R}$ and $H^1(S^1) = \mathbb{R}$, confirming our earlier result.

### Application: the winding number

De Rham cohomology gives a clean formulation of the **winding number**. Consider a closed curve $\gamma : S^1 \to \mathbb{R}^2 \setminus \{0\}$. The 1-form $\omega = \frac{-y\,dx + x\,dy}{x^2 + y^2}$ is closed on $\mathbb{R}^2 \setminus \{0\}$ but not exact (it generates $H^1(\mathbb{R}^2 \setminus \{0\}) \cong \mathbb{R}$). The winding number of $\gamma$ around the origin is

$$n = \frac{1}{2\pi}\oint_\gamma \omega \in \mathbb{Z}.$$

The fact that this is always an integer follows from the de Rham theorem: it is the evaluation of the cohomology class $[\omega]$ on the homology class $[\gamma]$, and this pairing takes integer values. This example illustrates how cohomology detects topological features (the "hole" at the origin) that are invisible to local analysis.

---

## What's next

With integration and Stokes' theorem, we have completed the core machinery of differential forms on manifolds. But we have been studying **smooth** structure — we have not yet asked how to measure lengths, angles, or curvature. In the next article, we introduce **Riemannian metrics**: the additional structure that turns a smooth manifold into a geometric space where distances, geodesics, and curvature are defined. This is Riemannian geometry, and it is the language of general relativity and much of modern geometry.

### A historical note

The story of Stokes' theorem is itself worth a moment. The "classical" Stokes' theorem (for surfaces in $\mathbb{R}^3$) was stated by Lord Kelvin in a letter to Stokes in 1850, and Stokes set it as an examination problem at Cambridge in 1854. The general version for manifolds emerged gradually through the work of Elie Cartan (who developed the exterior calculus), Georges de Rham (who connected it to topology), and was given its modern form in textbooks by Spivak, Warner, and others in the mid-20th century. The theorem stands as one of the supreme achievements of mathematics: a single equation that encompasses all the fundamental theorems of calculus in every dimension.

---

*This is Part 9 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 8 — Differential Forms](/en/differential-geometry/08-differential-forms/)*

*Next: [Part 10 — Riemannian Geometry](/en/differential-geometry/10-riemannian-geometry/)*
