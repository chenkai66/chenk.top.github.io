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

In single-variable calculus, the fundamental theorem says that integrating a derivative over an interval equals the boundary difference: $\int_a^b f'(x)\,dx = f(b) - f(a)$. The "boundary" of $[a, b]$ is the two-point set $\{a, b\}$, with $b$ counted positively and $a$ negatively. The right-hand side is the integral of $f$ over this signed boundary. The left-hand side is the integral of the derivative over the interval. This is, in essence, every "fundamental theorem" you have ever met — Green's theorem in the plane, the divergence theorem in three dimensions, the classical Stokes' theorem on surfaces. They are all instances of one statement on manifolds: **the integral of $d\omega$ over $M$ equals the integral of $\omega$ over $\partial M$**.

The goal of this article is to prove and digest this single equation. To get there, we need three things. First, a coherent notion of **orientation** — without it, integrals do not even have signs. Second, a notion of **boundary** with its **induced orientation** — without that, the right-hand side is meaningless. Third, a notion of **integration** of a differential form on a $k$-dimensional submanifold — this requires a careful construction using charts and partitions of unity. With those in hand, Stokes' theorem follows from one local computation plus a partition-of-unity argument.

The reason this article matters: Stokes' theorem is the *single* result of differential calculus on manifolds. Every other integral theorem is a corollary. Once you understand Stokes, you understand why electromagnetic flux is conserved, why winding numbers are integers, why the Gauss-Bonnet theorem of article 5 holds, and why de Rham cohomology pairs with singular homology. It is the mountain peak of the local theory and the gateway to every global result.

---

## 1. Orientation

A **tangent space** $T_pM$ is an $n$-dimensional real vector space, and like any such space it admits two equivalence classes of ordered bases (related by orientation-preserving vs. orientation-reversing linear maps). A choice of one class is an **orientation** of $T_pM$. A manifold $M$ is **orientable** if such choices can be made smoothly across the manifold, agreeing on overlaps. An orientation, when it exists, is a global topological choice — there are exactly two orientations on a connected orientable manifold.

![Orientation of a manifold via consistent ordered bases](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_1_orientation.png)

**Three equivalent definitions.** A manifold $M$ is orientable if any of the following holds:
1. There is an atlas of charts whose transition maps all have positive Jacobian determinant.
2. There is a nowhere-vanishing $n$-form (a "volume form") on $M$.
3. The frame bundle of $M$ admits a section into the $\mathrm{GL}^+(n)$ subbundle.

These are equivalent, and each is useful in different contexts. The form-theoretic version (definition 2) is what we will use to define integrals.

**Examples.**
- Every open subset of $\mathbb{R}^n$ is orientable; the standard volume form $dx^1\wedge\dots\wedge dx^n$ gives a canonical orientation.
- The sphere $S^n$ is orientable — the outward unit normal contracted with $dx^0\wedge\dots\wedge dx^n$ gives a volume form.
- The torus $T^n$ is orientable.
- The **Mobius strip** is non-orientable. Walk around the strip once with a chosen ordered basis; you return with the basis reversed. Equivalently, no nowhere-vanishing 2-form exists on the Mobius strip.
- The **real projective plane** $\mathbb{RP}^2$ is non-orientable for the same reason. In fact, $\mathbb{RP}^n$ is orientable iff $n$ is odd.

**A concrete check.** Consider $\mathbb{RP}^2$ as the quotient of $S^2$ by the antipodal map $A(x) = -x$. The pullback $A^* (dx\wedge dy\wedge dz) = -dx\wedge dy\wedge dz$ in $\mathbb{R}^3$ — so $A$ reverses orientation on the ambient space, but on the *sphere* the calculation is more subtle. Working with local charts, you find that the antipodal map reverses orientation on $S^2$, so the quotient cannot inherit an orientation. Concrete, mechanical, and fundamentally a topological fact about the equivalence class of bases.

**Why this matters.** Orientation is not a redundant decoration — it is what makes integrals signed. The integral $\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx$ in 1D is the simplest manifestation: reversing the orientation of the interval flips the sign of the integral. On surfaces and higher-dimensional manifolds, the same phenomenon controls flux signs, charge conservation, and the consistency of Stokes' theorem. Without orientation, you would not even know which side of a surface counts as "outward."

**Non-orientable integration: densities.** When $M$ is non-orientable, you cannot integrate top-degree forms (the sign is ambiguous), but you can integrate **densities** — objects that pick up the *absolute value* of the Jacobian under change of coordinates rather than the signed Jacobian. Densities are how you do integration on non-orientable manifolds. In physics they are usually invisible because spacetime is taken to be orientable, but in mathematical biology and certain topology problems (counting orbits on a Klein bottle, for instance) they are unavoidable.

**Orientability and double covers.** Every connected non-orientable manifold $M$ has a connected orientable double cover $\tilde M \to M$ — the orientable cover. Computations on $M$ can often be lifted to $\tilde M$ where signs make sense. For example, the Mobius strip's orientable double cover is the cylinder; the projective plane's is the sphere. This trick reduces many non-orientable problems to orientable ones at the cost of doubling the data.

---

## 2. Manifolds with Boundary; Induced Orientation

A **manifold with boundary** is a topological space locally modeled on the upper half-space $\mathbb{H}^n = \{x \in \mathbb{R}^n : x^n \geq 0\}$. Charts come in two flavors: interior charts (whose images miss the boundary $\{x^n = 0\}$) and boundary charts (whose images touch $\{x^n = 0\}$). The **boundary** $\partial M$ is the set of points sitting on $\{x^n = 0\}$ in some boundary chart. It is an $(n-1)$-dimensional manifold (without boundary).

![Boundary of an oriented manifold with the induced orientation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_2_boundary.png)

**Examples.**
- Closed disk $\bar D^2 = \{|x| \leq 1\} \subset \mathbb{R}^2$: boundary is $S^1$.
- Closed ball $\bar B^3$: boundary is $S^2$.
- Cylinder $S^1 \times [0, 1]$: boundary is $S^1 \times \{0\} \sqcup S^1 \times \{1\}$ — two circles.
- Mobius strip: boundary is a single circle (which double-covers the central circle).
- Genus-$g$ handlebody: boundary is a genus-$g$ surface. The 3-dimensional region "inside" a doubly-handled solid has a 2-dimensional surface as its boundary; many constructions in 3-manifold topology start here.

**Boundary commutes with itself trivially.** The boundary of a manifold-with-boundary has no boundary itself — $\partial(\partial M) = \emptyset$. This corresponds to $d^2 = 0$ on the form side. The two facts ($\partial^2 = 0$ on chains, $d^2 = 0$ on forms) are mirror images of each other, and they fit together via Stokes: $\int_C d^2\eta = \int_{\partial^2 C}\eta = \int_\emptyset \eta = 0$, and the converse pairing argument also works. Geometric "boundary of boundary is empty" and analytic "differential squared is zero" are two views of one structural fact.

**Induced orientation.** If $M$ is oriented, $\partial M$ inherits a canonical orientation. The rule: a basis $(v_1, \dots, v_{n-1})$ of $T_p \partial M$ is positively oriented iff $(N, v_1, \dots, v_{n-1})$ is positively oriented in $T_p M$, where $N$ is an outward-pointing normal vector. Equivalently: "outward normal first."

**1D example.** $M = [a, b]$ with the standard orientation (increasing $x$). The boundary is $\{a, b\}$. At $b$, the outward normal points right ($+\partial_x$), so the orientation rule gives "no $v$'s, just the empty basis," conventionally meaning "positive sign at $b$." At $a$, the outward normal points left ($-\partial_x$), giving "negative sign at $a$." This is exactly the sign convention $f(b) - f(a)$ in the 1D fundamental theorem.

**2D example.** $M = \bar D^2$ with the standard orientation $dx\wedge dy$. The boundary is $S^1$. At a point, the outward normal $N$ is the radial direction. The induced orientation on $S^1$ is then "counter-clockwise" — the standard mathematical convention. This is why Green's theorem in the plane requires a counter-clockwise boundary.

**3D example.** $M = \bar B^3$ with $dx\wedge dy\wedge dz$. Boundary is $S^2$. Outward normal at a point of the sphere is the radial direction; the induced orientation on $S^2$ is then the orientation that, with the radial vector first, reproduces the standard volume form. This is the orientation under which the surface area is positive — the natural "outside-looking-out" orientation.

**Why this matters.** The induced orientation is the geometric content of the sign in Stokes' theorem. If you choose the wrong orientation on $\partial M$, your formula picks up a global minus sign — and the theorem will appear false. The "outward normal first" convention is not arbitrary; it is the unique choice that makes Stokes' theorem hold without ad hoc sign corrections.

**Corners and Lipschitz boundaries.** Real-life manifolds with boundary often have *corners*: a square in the plane, a cube in 3-space. Stokes' theorem still holds, but the boundary $\partial M$ is now a piecewise smooth manifold and the induced orientation breaks naturally at corners. The integration $\int_{\partial M}\omega$ is just the sum over the smooth pieces. There is also a generalization to manifolds with corners (Joyce, Melrose) where the corner stratification is part of the data; this matters in singular perturbation theory and in moduli spaces of stable curves.

**Boundaries with multiple components.** Stokes also handles disconnected boundaries naturally. The annulus $\{1 \leq r \leq 2\}$ has $\partial M = S^1_{r=1} \sqcup S^1_{r=2}$, with the inner circle oriented clockwise (because the outward normal points inward) and the outer counter-clockwise. The integral $\int_{\partial M}\omega$ is the sum, with these signs. Confusing the signs gives wrong answers; getting them right is the whole point of the induced orientation rule.

---

## 3. Integration of Top-Degree Forms

The natural objects to integrate on a manifold are top-degree forms: $n$-forms on an $n$-dimensional manifold. Why? Because pullback of a top-degree form by a positive diffeomorphism is well-defined, and the resulting integral is invariant under change of coordinates. (Lower-degree forms are integrated only over corresponding submanifolds.)

**Local definition.** On $\mathbb{R}^n$ with the standard orientation, an $n$-form $\omega = f(x)\,dx^1\wedge\dots\wedge dx^n$ has integral
$$\int_{\mathbb{R}^n} \omega = \int_{\mathbb{R}^n} f(x)\,dx^1\dots dx^n,$$
where the right-hand side is the ordinary Lebesgue integral. Notice the implicit ordering: the wedge product with positive sign matches the standard ordering of variables in the iterated integral.

**Change of variables.** If $\varphi: U \to V$ is an orientation-preserving diffeomorphism of open subsets of $\mathbb{R}^n$ and $\omega$ is an $n$-form on $V$, then
$$\int_U \varphi^* \omega = \int_V \omega.$$
This is the change-of-variables formula. The key point: the wedge product *automatically* handles the Jacobian determinant. Recall from article 8: $\varphi^*(dx^1\wedge\dots\wedge dx^n) = \det(D\varphi)\,du^1\wedge\dots\wedge du^n$. The signed determinant matches the orientation-preserving condition.

**Globally on a manifold.** To integrate an $n$-form $\omega$ on an oriented manifold $M$:
1. Cover $M$ by oriented charts $U_\alpha$.
2. Choose a partition of unity $\{\rho_\alpha\}$ subordinate to the cover.
3. Define $\int_M \omega = \sum_\alpha \int_{U_\alpha} \rho_\alpha \omega$.

The result is independent of the choice of charts and partition of unity. This is the entire definition.

**Numerical example.** Compute $\int_{S^2} \omega$ where $\omega = x\,dy\wedge dz + y\,dz\wedge dx + z\,dx\wedge dy$ is the "spherical volume form" coming from contracting the radial vector with $dx\wedge dy\wedge dz$. On the upper hemisphere parametrized by $\varphi(u, v) = (u, v, \sqrt{1 - u^2 - v^2})$ with $u^2 + v^2 < 1$, pullback gives (after computation) $\frac{1}{\sqrt{1-u^2-v^2}}du\wedge dv$, and integrating over the unit disk gives $2\pi$. Adding the lower hemisphere (with appropriate orientation) gives $4\pi$, which is exactly the surface area of $S^2$. Sanity check: $\omega = \iota_R(dx\wedge dy\wedge dz)$, where $R = x\partial_x + y\partial_y + z\partial_z$ is the radial vector. Pull back to $S^2$, where $R$ is the unit normal: the result is the area form. So $\int_{S^2}\omega$ is the area, and the area of $S^2$ is $4\pi$. Confirmed.

**Why this matters.** Integration on manifolds is the bridge between local geometry (forms, derivatives) and global quantities (total flux, total volume, total charge). Without partition-of-unity arguments, you cannot define integrals on manifolds covered by multiple charts; with them, the definition is unambiguous and the theorems work.

**Why partition of unity?** A partition of unity $\{\rho_\alpha\}$ subordinate to a cover $\{U_\alpha\}$ is a collection of smooth nonnegative functions, with $\rho_\alpha$ supported in $U_\alpha$, summing to 1 pointwise (locally finitely). Such a partition exists on any paracompact Hausdorff manifold (which is essentially every manifold appearing in practice). The function $\rho_\alpha\omega$ is supported in $U_\alpha$ and can be integrated using the chart of $U_\alpha$. The independence of choices follows from change-of-variables formula plus a routine check.

**Pseudo-numerical example: integrating over $S^2$.** Cover $S^2$ by two charts, the upper and lower hemispheres (each diffeomorphic to a disk, with overlap an equator strip). Pick a partition of unity $\rho_+ + \rho_- = 1$ with $\rho_\pm$ supported in the corresponding hemisphere. Then $\int_{S^2}\omega = \int_{\text{upper}}\rho_+\omega + \int_{\text{lower}}\rho_-\omega$. In practice, choices like spherical coordinates make these calculations explicit, and the sphere's area $4\pi$ can be confirmed by direct integration. The partition-of-unity formalism is the *theoretical* device that makes this rigorous; in practice you usually just compute.

---

## 4. Stokes' Theorem

**Theorem (Stokes).** Let $M$ be an oriented compact $n$-dimensional manifold with boundary $\partial M$, equipped with the induced orientation. Let $\omega$ be a smooth $(n-1)$-form on $M$. Then
$$\int_M d\omega = \int_{\partial M} \omega.$$

That is the entire theorem, and it is the most important formula in differential calculus.

![Stokes' theorem: integral of d-omega over M equals integral of omega over the boundary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_3_stokes.png)

**Proof sketch.** Two steps.

*Step 1: local case in upper half-space.* Take $M = \mathbb{H}^n$ and $\omega$ supported in a compact subset. Write $\omega = \sum_i (-1)^{i-1} f_i\,dx^1\wedge\dots\wedge \hat{dx^i}\wedge\dots\wedge dx^n$. Then $d\omega = \sum_i \partial_i f_i\,dx^1\wedge\dots\wedge dx^n$. Integrate term by term. For $i < n$, the integral $\int \partial_i f_i$ is zero by the fundamental theorem (compactly supported $f_i$). For $i = n$, the integral $\int_{x^n \geq 0} \partial_n f_n = -\int_{\mathbb{R}^{n-1}} f_n(x^1, \dots, x^{n-1}, 0)$, which is exactly the boundary integral with the sign of the induced orientation. So Stokes holds locally.

*Step 2: globalize.* Use a partition of unity $\{\rho_\alpha\}$ subordinate to a cover of $M$ by oriented charts. Then $\omega = \sum_\alpha \rho_\alpha \omega$, and applying step 1 chart by chart gives $\int_M d(\rho_\alpha\omega) = \int_{\partial M}\rho_\alpha\omega$. Summing over $\alpha$ uses $\sum_\alpha d(\rho_\alpha\omega) = d(\sum_\alpha \rho_\alpha\omega) - 0 = d\omega$ (the cross-term $\sum_\alpha d\rho_\alpha\wedge\omega$ vanishes because $\sum_\alpha \rho_\alpha = 1$ implies $\sum_\alpha d\rho_\alpha = 0$). Done.

That's it. The whole theorem is the 1D fundamental theorem applied chart by chart, glued by a partition of unity.

**Trivial case: closed manifolds.** If $M$ has no boundary ($\partial M = \emptyset$), then $\int_M d\omega = 0$ for every $(n-1)$-form $\omega$. In other words, exact $n$-forms integrate to zero on closed manifolds. The contrapositive: an $n$-form whose integral over $M$ is nonzero cannot be exact, hence represents a nonzero class in $H^n_{dR}(M)$.

**Worked example: integrating a winding form.** On $\mathbb{R}^2 \setminus \{0\}$, the angle form $\omega = \frac{-y\,dx + x\,dy}{x^2+y^2}$ is closed but not exact. Take $M$ to be the annulus $\{1 \leq r \leq 2\}$. Stokes' theorem says
$$\int_M d\omega = \int_{\partial M}\omega.$$
The left side is zero ($d\omega = 0$). The boundary is the inner circle (oriented clockwise — opposite of the standard counter-clockwise) plus the outer circle (counter-clockwise). The integral around each circle is $2\pi$, so the boundary integral is $-2\pi + 2\pi = 0$. Consistent. The form contributes $2\pi$ from each circle, but with opposite orientations they cancel.

**Worked example: a non-trivial calculation.** Compute $\int_{\partial \bar B^3} \omega$ for $\omega = (x^2 + y)\,dy\wedge dz + (xy + z)\,dz\wedge dx + (xz - y)\,dx\wedge dy$, where $\bar B^3$ is the closed unit ball. Use Stokes:
$$d\omega = (\partial_x(x^2+y) + \partial_y(xy + z) + \partial_z(xz - y))\,dx\wedge dy\wedge dz = (2x + x + x)\,dx\wedge dy\wedge dz = 4x\,dx\wedge dy\wedge dz.$$
By symmetry $\int_{\bar B^3} 4x\,dV = 0$ (odd function over a symmetric domain). So $\int_{S^2}\omega = 0$. A direct surface integral in spherical coordinates would have been agonizing; Stokes makes it instant.

**Why Stokes always works.** The proof has only two ingredients: the 1D fundamental theorem of calculus (a chart-by-chart fact) and the gluing power of partition of unity (a global structural fact). No exotic analysis, no special hypotheses beyond compactness and orientability. Stokes is as basic as differentiation itself; it is essentially "differentiating a form on $M$ and integrating the result equals integrating the form along $\partial M$" — the same statement at every dimension, the same proof.

---

## 5. Classical Theorems Recovered

All three classical "integral theorems" of vector calculus are special cases of Stokes' theorem.

![Classical theorems unified: gradient, Stokes, Green, divergence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_4_classical_unify.png)

**Fundamental theorem of line integrals.** $M = \gamma$ (a curve), $\omega = f$ (a 0-form):
$$\int_\gamma df = f(\gamma(b)) - f(\gamma(a)).$$
This is just Stokes with $n = 1$. The "boundary" is the two endpoints with appropriate signs.

**Green's theorem.** $M$ a region in $\mathbb{R}^2$, $\omega = P\,dx + Q\,dy$ a 1-form. Then $d\omega = (\partial_x Q - \partial_y P)\,dx\wedge dy$, and Stokes gives
$$\iint_M (\partial_x Q - \partial_y P)\,dA = \oint_{\partial M}(P\,dx + Q\,dy).$$
The classical Green's theorem.

**Classical Stokes' theorem (on surfaces).** $M$ a surface with boundary in $\mathbb{R}^3$, $F$ a vector field, $\omega = F^\flat$ (the corresponding 1-form). Then $d\omega$ is the curl 2-form, and integrating gives
$$\iint_M (\nabla\times F)\cdot dA = \oint_{\partial M} F\cdot dr.$$
The classical "curl theorem."

**Divergence theorem.** $M$ a region in $\mathbb{R}^3$, $F$ a vector field, $\omega$ the corresponding flux 2-form. Then $d\omega = (\nabla\cdot F)\,dx\wedge dy\wedge dz$, and Stokes gives
$$\iiint_M \nabla\cdot F\,dV = \iint_{\partial M} F\cdot dA.$$
The divergence theorem of Gauss.

**Why this matters.** All four classical theorems are corollaries of one statement on manifolds. Memorizing them as four separate results is a coordinate-bound provincialism; a working differential geometer uses Stokes once and recovers them on demand.

**A subtler corollary: Cauchy's integral theorem.** On $\mathbb{C} = \mathbb{R}^2$, a holomorphic function $f$ has $df = f'(z)\,dz$ where $dz = dx + i\,dy$. So the 1-form $f(z)\,dz$ is closed (the Cauchy-Riemann equations), and Stokes gives $\oint_{\partial M} f(z)\,dz = 0$ for any region $M$ on which $f$ is holomorphic. This is Cauchy's theorem of complex analysis. The whole subject of complex analysis is differential forms in two real dimensions, viewed through the lens of holomorphicity.

**Cauchy's integral formula via Stokes.** Take $f$ holomorphic in a region containing the closed disk of radius $R$ around $z_0$. Then $\frac{f(z)}{z - z_0}$ has a simple pole at $z_0$. Apply Stokes (or the residue theorem) to a small annulus around $z_0$:
$$\oint_{|z - z_0| = R}\frac{f(z)}{z - z_0}dz = 2\pi i\,f(z_0).$$
Cauchy's integral formula. From this, all of complex analysis (Liouville's theorem, the maximum modulus principle, the residue theorem) cascades. The miraculous rigidity of holomorphic functions — that knowing $f$ on a circle determines $f$ inside — is, at root, Stokes' theorem applied to closed forms with poles.

**Hodge theorem and harmonic forms.** On a compact oriented Riemannian manifold, every de Rham cohomology class has a unique harmonic representative ($\Delta\omega = 0$ where $\Delta = d\delta + \delta d$). The proof uses Stokes' theorem to set up the inner product $\langle\alpha, \beta\rangle = \int_M \alpha \wedge *\beta$, then PDE theory to find harmonic representatives. Hodge theory gives a vast generalization of "every closed form is exact mod $\ker\Delta$" and is the analytic foundation of Kahler geometry, the index theorem, and elliptic regularity on manifolds.

---

## 6. de Rham Cohomology and Stokes

Stokes' theorem implies a fundamental fact: integration of closed forms over closed manifolds depends only on the cohomology class of the form.

**Claim.** If $\omega_1, \omega_2$ are closed $k$-forms on $M$ (no boundary) with $\omega_1 - \omega_2 = d\eta$ exact, then for any closed $k$-cycle $C$,
$$\int_C \omega_1 = \int_C \omega_2.$$

*Proof.* $\int_C(\omega_1 - \omega_2) = \int_C d\eta = \int_{\partial C} \eta = 0$ (since $\partial C = \emptyset$).

So the integral $\int_C \omega$ depends only on $[\omega] \in H^k_{dR}(M)$. Similarly, replacing $C$ by a homologous cycle $C'$ (i.e. $C - C' = \partial D$ for some chain $D$) does not change the integral, because $\int_C\omega - \int_{C'}\omega = \int_{\partial D}\omega = \int_D d\omega = 0$ (since $\omega$ is closed).

![de Rham cohomology and Poincare duality](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_5_de_rham_coh.png)

**de Rham's theorem.** The integration pairing
$$H^k_{dR}(M) \times H_k(M; \mathbb{R}) \to \mathbb{R}, \qquad ([\omega], [C]) \mapsto \int_C \omega$$
is a perfect pairing. So $H^k_{dR}(M) \cong H_k(M; \mathbb{R})^*$ — the de Rham cohomology is the dual of singular homology with real coefficients. Equivalently, $H^k_{dR}(M) \cong H^k(M; \mathbb{R})$ via the universal coefficient theorem.

**Poincare duality.** On a compact oriented $n$-manifold without boundary, the pairing
$$H^k_{dR}(M) \times H^{n-k}_{dR}(M) \to \mathbb{R}, \qquad ([\alpha], [\beta]) \mapsto \int_M \alpha\wedge\beta$$
is also a perfect pairing. So $H^k_{dR}(M) \cong H^{n-k}_{dR}(M)^*$. For $M$ closed and oriented, this implies the Betti numbers satisfy $b_k = b_{n-k}$ — the symmetry of cohomology dimensions seen in the torus example of article 8.

**Why this matters.** Cohomology classes can be computed *by integration*. Given a closed form, you can detect its non-triviality by integrating over cycles; given a cycle, you can detect its non-triviality by integrating closed forms over it. This is the analytic foundation of topology: the Betti numbers, the Euler characteristic, the genus of a surface — all are accessible through differential forms.

**Periods and arithmetic.** When the manifold has extra structure (e.g., a complex algebraic variety), the integrals $\int_C\omega$ for distinguished forms $\omega$ and cycles $C$ are called **periods**. Periods of algebraic varieties are deep arithmetic invariants — they include $2\pi i$, $\log 2$, values of zeta functions, and more exotic transcendental numbers. The conjectures of Grothendieck, Kontsevich-Zagier, and the modern theory of motivic cohomology revolve around understanding periods. So Stokes' theorem connects to one of the deepest open problems in mathematics: the structure of transcendental numbers obtained as integrals of algebraic forms over algebraic cycles.

**Index theorem foreshadowing.** Many topological invariants of a manifold can be computed as integrals of curvature-like forms. The Euler characteristic equals $\int_M e(TM)$ (Chern-Gauss-Bonnet). The signature equals $\int_M L(TM)$ (Hirzebruch). The index of an elliptic operator equals $\int_M \mathrm{ch}(\sigma) \mathrm{Td}(TM)$ (Atiyah-Singer). Each of these is a Stokes-type computation in disguise: a topological invariant emerges as an integral of a closed form. Article 12 will develop the relevant characteristic classes and explain why these formulas hold.

---

## 7. Integration on Chains and Cycles

We have been integrating forms over manifolds with boundary. The right setting for full generality is integration over **chains**: formal $\mathbb{Z}$-linear combinations of smooth singular simplices.

A **smooth $k$-simplex** in $M$ is a smooth map $\sigma: \Delta^k \to M$ from the standard $k$-simplex. A **smooth $k$-chain** is a finite formal sum $C = \sum_i a_i \sigma_i$ with integer coefficients. The boundary operator $\partial$ maps $k$-chains to $(k-1)$-chains by the alternating sum of restrictions to faces.

![Integrating a form along a chain of cells](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_6_chain_integration.png)

Integration on chains is defined linearly: $\int_{\sum a_i\sigma_i} \omega = \sum a_i \int_{\sigma_i^* \omega}$. Stokes' theorem extends:
$$\int_C d\omega = \int_{\partial C} \omega.$$
This is just the chain-level statement.

**Why chains?** They are more flexible than submanifolds. A chain need not be embedded, need not be a manifold, can have multiplicities. This flexibility is what makes singular homology work — you can subdivide, refine, and compute. Triangulations and their boundaries are chains. The proof of de Rham's theorem in either direction (constructing a closed form representing a cocycle, or constructing a cycle integrating to give a cohomology class) lives at the chain level.

**Worked example: winding number.** On $\mathbb{R}^2 \setminus \{0\}$ with the angle form $\omega = \frac{-y\,dx + x\,dy}{x^2+y^2}$, and a closed curve $\gamma: S^1 \to \mathbb{R}^2 \setminus \{0\}$, the integer
$$n(\gamma) = \frac{1}{2\pi}\int_\gamma \omega$$
is the **winding number** of $\gamma$ around the origin. It is integer-valued by topology and computable analytically. By de Rham, $H^1_{dR}(\mathbb{R}^2\setminus\{0\}) = \mathbb{R}$ with $[\omega/2\pi]$ a generator; the winding number is just the cohomology pairing.

**Worked example: linking number.** For two disjoint smooth loops $\gamma_1, \gamma_2$ in $\mathbb{R}^3$, the **linking number** $\mathrm{lk}(\gamma_1, \gamma_2)$ is an integer measuring how often they wind around each other. There is an integral formula (Gauss):
$$\mathrm{lk}(\gamma_1,\gamma_2) = \frac{1}{4\pi}\oint_{\gamma_1}\oint_{\gamma_2}\frac{(\vec r_1 - \vec r_2)\cdot(d\vec r_1 \times d\vec r_2)}{|\vec r_1 - \vec r_2|^3}.$$
This is again a winding-number-type integral, and integer-valuedness comes from de Rham cohomology of $\mathbb{R}^3 \setminus \gamma_2$. The linking number is the simplest knot-theoretic invariant; higher-order analogs (Massey products, finite-type invariants) generalize the same idea.

**The chain-level statement.** The full power of Stokes appears most clearly at the chain level: $\partial$ on chains and $d$ on forms are *adjoint* under the integration pairing. This adjunction is what makes the singular-de Rham comparison work, and it is the seed of every "duality theorem" in algebraic topology. The signs in chain complexes — which can look daunting — are just the signs the induced orientation forces on you.

---

## 8. Examples on Sphere and Torus

To consolidate, two classical applications of Stokes.

![Stokes' theorem applied to a sphere and a torus](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/09-integration-stokes/dg_v2_09_7_examples.png)

**Surface area of $S^2$ via Stokes.** Take $M = \bar B^3 \subset \mathbb{R}^3$ with $\omega = x\,dy\wedge dz + y\,dz\wedge dx + z\,dx\wedge dy$. Then $d\omega = 3\,dx\wedge dy\wedge dz$. Stokes gives
$$\int_{S^2}\omega = \int_{\bar B^3} 3\,dV = 3 \cdot \frac{4}{3}\pi = 4\pi.$$
And $\omega$ on $S^2$ equals the area form, so the surface area is $4\pi$. Two computations, one identity. The factor of 3 is exactly the dimension; in $\mathbb{R}^n$, the analogous identity gives the area of $S^{n-1}$ as $n$ times the volume of $B^n$.

**Volume of $B^3$ from $S^2$.** The reverse direction is also useful: knowing the surface area, integrate radially to get the volume. $\mathrm{vol}(B^3) = \int_0^1 4\pi r^2\,dr = \frac{4\pi}{3}$. The factor of 3 appears again — it is the dimension. This kind of dimensional bookkeeping is implicit in every computation in differential geometry, and Stokes gives it a clean statement.

**Solid angle and Gauss's law.** The 2-form $\omega = \frac{x\,dy\wedge dz + y\,dz\wedge dx + z\,dx\wedge dy}{(x^2+y^2+z^2)^{3/2}}$ on $\mathbb{R}^3 \setminus \{0\}$ is closed, with $\oint_{S^2_R} \omega = 4\pi$ for every sphere of radius $R$ around the origin. So $\omega/(4\pi)$ generates $H^2_{dR}(\mathbb{R}^3\setminus\{0\}) = \mathbb{R}$. Physically, this is exactly the electric field of a point charge — the integral of the field flux around any closed surface enclosing the charge is $4\pi$ (in Gaussian units), independent of the surface. This is Gauss's law, and its content is purely topological: it counts the charges enclosed.

**Gauss-Bonnet via Stokes.** Recall from article 5 that on a compact oriented surface,
$$\int_M K\,dA = 2\pi \chi(M).$$
In modern language, this is Stokes applied to the curvature form (with some Lie-algebraic dressing). The proof in article 5 used a triangulation and the angle excess at each triangle. The deeper proof, via Chern-Weil theory, will be sketched in article 12.

**Brouwer fixed-point theorem.** Suppose $f: \bar B^n \to \bar B^n$ has no fixed point. Then the map $r: \bar B^n \to S^{n-1}$ defined by sending each $x$ to the unique boundary point in the ray from $f(x)$ through $x$ is a smooth retraction (with $r|_{S^{n-1}} = \mathrm{id}$). Now choose any $(n-1)$-form $\omega$ on $S^{n-1}$ with $\int_{S^{n-1}}\omega = 1$. Then $r^*\omega$ is a closed $(n-1)$-form on $\bar B^n$, and Stokes gives $\int_{\bar B^n}d(r^*\omega) = \int_{S^{n-1}}r^*\omega = \int_{S^{n-1}}\omega = 1$. But $d(r^*\omega) = r^*(d\omega) = 0$ since $\omega$ is top-degree on $S^{n-1}$ — contradiction. So no such $r$ exists, hence $f$ has a fixed point. Stokes proves topology.

**Hairy ball theorem.** A nowhere-vanishing tangent vector field on $S^{2n}$ does not exist. Sketch: if $X$ were such a field, the homotopy $X_t(p) = (\cos t)\,p + (\sin t)\,X(p)/|X(p)|$ on $S^{2n}$ (with appropriate normalization) gives a homotopy from $\mathrm{id}$ to the antipodal map. But the antipodal map on $S^{2n}$ has degree $(-1)^{2n+1} = -1$, and the identity has degree $1$ — degree is a homotopy invariant computable as a Stokes-type integral, contradiction. So no such field. Sphere combing fails on even-dimensional spheres. Topology again, via integrals of forms.

**Why this matters.** Stokes is not merely a calculation tool. It is how *topological invariants are computed analytically*. Once you internalize that integers can be obtained as integrals of differential forms, you start seeing topology hiding in physics: charge quantization in Maxwell theory, instanton numbers in Yang-Mills, anomalies in quantum field theory, indices of Dirac operators. Each of these is a Stokes calculation in a slightly more elaborate setting. The pattern is universal.

**Torus periods.** On the 2-torus $T^2$ with coordinates $(\theta_1, \theta_2)$, the closed forms $d\theta_1, d\theta_2$ are both not exact. Their integrals over the two basic 1-cycles (the "meridian" and "longitude") give the period matrix $\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$ (up to factors of $2\pi$). The cohomology classes $[d\theta_1], [d\theta_2]$ form a basis of $H^1_{dR}(T^2) \cong \mathbb{R}^2$, dual to the cycle basis. This is the simplest nontrivial example of de Rham theory and the model for elliptic curve theory in algebraic geometry.

**Genus-$g$ surface.** A compact orientable surface of genus $g$ has $H^0 = H^2 = \mathbb{R}$ and $H^1 = \mathbb{R}^{2g}$. The $2g$ generators of $H^1$ pair with the $2g$ generators of $H_1$ (the $a$- and $b$-cycles of a standard handle decomposition) via integration. The intersection form on $H_1$ corresponds via Poincare duality to the wedge product on $H^1$, and the diagonal symplectic form $\bigoplus\begin{pmatrix}0 & 1 \\ -1 & 0\end{pmatrix}$ encodes the genus. This algebraic structure is the geometry of Riemann surfaces in nuce.

**Volume form on $S^n$.** By Stokes applied to $\bar B^{n+1}$ with $\omega = \iota_R(dx^0\wedge\dots\wedge dx^n)$ (radial contraction), the surface area of $S^n$ is computed inductively: $\mathrm{vol}(S^n) = (n+1)\,\mathrm{vol}(B^{n+1})$. Combined with the iterative formula $\mathrm{vol}(B^n) = \pi^{n/2}/\Gamma(n/2 + 1)$, this gives all the classical formulas. The $4\pi$ for $S^2$, the $2\pi^2$ for $S^3$, the $\frac{8\pi^2}{3}$ for $S^4$ — all corollaries of one Stokes computation.

---

## What's Next

We now have integration, Stokes, and the algebraic-topological consequences. The next article introduces **Riemannian metrics** — the structure that lets us measure lengths and angles, defines geodesics ("straight lines on curved spaces"), and motivates parallel transport. Combined with Stokes, the metric will give us the curvature 2-forms whose integrals — Chern classes, Pontryagin classes, the Euler class — are the topological invariants computable by the techniques of this article.

**Summary of the key ideas.**

1. **Orientation** makes integrals signed and is a global topological choice — there are exactly two on a connected orientable manifold.
2. **Manifolds with boundary** carry an **induced orientation** on $\partial M$ via "outward normal first."
3. **Integration** of top-degree forms is defined locally via Lebesgue, globalized by partition of unity, and unambiguous.
4. **Stokes' theorem** $\int_M d\omega = \int_{\partial M}\omega$ unifies all classical integral theorems.
5. The classical theorems (gradient, Green, Stokes, divergence) are all Stokes in different dimensions and form-degrees.
6. **de Rham's theorem** identifies $H^k_{dR}(M) \cong H^k(M;\mathbb{R})$ via the integration pairing; **Poincare duality** gives $H^k \cong H^{n-k}$ for closed orientable manifolds.
7. Topological invariants (winding number, Euler characteristic, Brouwer fixed point) are computed by Stokes-type arguments.

---

*This is Part 9 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 8 — Differential Forms](/en/differential-geometry/08-differential-forms/)*

*Next: [Part 10 — Riemannian Geometry](/en/differential-geometry/10-riemannian-geometry/)*
