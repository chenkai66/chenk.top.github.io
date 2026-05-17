---
title: "Vector Fields, Flows, and the Lie Bracket"
date: 2021-11-13 09:00:00
tags:
  - differential-geometry
  - vector-fields
  - lie-bracket
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
description: "Vector fields generate flows — one-parameter families of diffeomorphisms. The Lie bracket measures the failure of flows to commute, leading to Frobenius integrability."
disableNunjucks: true
series_order: 7
series_total: 12
translationKey: "differential-geometry-7"
---

A single tangent vector lives at one point — it tells you a direction and speed at that instant. A **vector field** assigns a tangent vector to every point of a manifold, smoothly. Where a tangent vector is an instantaneous velocity, a vector field is a velocity prescription: at each point of the manifold, it tells a particle which way to move and how fast. Following this prescription produces **integral curves** — the trajectories of the system — and assembling all trajectories together gives a **flow**, a one-parameter family of diffeomorphisms that moves the entire manifold.

This is not merely kinematic imagery. Vector fields are the infinitesimal generators of symmetry. In Hamiltonian mechanics, every conserved quantity corresponds to a vector field whose flow preserves the system. In Lie theory, the exponential map sends a Lie algebra element (a vector field on the group) to a group element (a diffeomorphism). And the **Lie bracket** $[X, Y]$ of two vector fields — measuring the failure of their flows to commute — encodes the algebraic structure of infinitesimal symmetries.

---

## Vector Fields on Manifolds

**Definition.** A **smooth vector field** $X$ on a smooth manifold $M$ is a smooth section of the tangent bundle $TM$. That is, $X$ assigns to each point $p \in M$ a tangent vector $X_p \in T_pM$, and this assignment varies smoothly with $p$.

In local coordinates $(x^1, \ldots, x^n)$ on an open set $U \subseteq M$, a vector field can be written:

![Vector field and its integral flow lines](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_fig7_vector_field.png)


$$X = X^i(x) \frac{\partial}{\partial x^i}$$

where the component functions $X^i: U \to \mathbb{R}$ are smooth. The notation uses the Einstein summation convention: repeated upper and lower indices are summed over. The set of all smooth vector fields on $M$ is denoted $\mathfrak{X}(M)$. It is a vector space over $\mathbb{R}$ and a module over $C^\infty(M)$: if $f \in C^\infty(M)$ and $X \in \mathfrak{X}(M)$, then $(fX)_p = f(p) X_p$ is again a smooth vector field. Note that $\mathfrak{X}(M)$ is infinite-dimensional (as a vector space over $\mathbb{R}$) unless $M$ is a finite set of points.

**Vector fields as derivations on $C^\infty(M)$.** Since each $X_p$ is a derivation at $p$, a vector field $X$ acts on smooth functions by $(Xf)(p) = X_p(f)$. This gives a map $X: C^\infty(M) \to C^\infty(M)$ that is $\mathbb{R}$-linear and satisfies the Leibniz rule:

$$X(fg) = (Xf)g + f(Xg).$$

In coordinates, $Xf = X^i \frac{\partial f}{\partial x^i}$. The map $f \mapsto Xf$ is a first-order linear differential operator. Conversely, every derivation $C^\infty(M) \to C^\infty(M)$ arises from a unique smooth vector field. This gives an equivalent characterization: $\mathfrak{X}(M) = \text{Der}(C^\infty(M))$.

**Global existence of vector fields.** On $\mathbb{R}^n$ or any parallelizable manifold, nonvanishing vector fields always exist. But on a general compact manifold, the **Euler characteristic** $\chi(M)$ provides an obstruction: by the Poincare-Hopf theorem, any smooth vector field on a compact manifold $M$ must have zeros, and the sum of the indices at the zeros equals $\chi(M)$. For instance, $\chi(S^2) = 2$, so every vector field on $S^2$ must vanish somewhere — this is the "hairy ball theorem." In contrast, $\chi(T^2) = 0$, and indeed the torus admits a nowhere-vanishing vector field (the constant field $\frac{\partial}{\partial \theta}$ in the angular coordinates).

**Example: the rotation field on $S^2$.** In spherical coordinates $(\theta, \phi)$ with $\theta \in (0, \pi)$ (polar angle) and $\phi \in (0, 2\pi)$ (azimuthal angle), the vector field $X = \frac{\partial}{\partial \phi}$ generates rotation about the $z$-axis. At each point of the sphere, $X$ points "eastward" along the latitude circle. At the poles, the spherical coordinate chart degenerates, but the vector field extends smoothly (it vanishes at the north and south poles).

In Cartesian coordinates of $\mathbb{R}^3$ restricted to $S^2$, this field is $X = -y \frac{\partial}{\partial x} + x \frac{\partial}{\partial y}$, which is visibly smooth everywhere on $S^2$ and vanishes at $(0, 0, \pm 1)$.

---

## Integral Curves and the Existence/Uniqueness Theorem

**Definition.** An **integral curve** of a vector field $X$ through a point $p \in M$ is a smooth curve $\gamma: I \to M$ (where $I$ is an open interval containing 0) such that:

$$\gamma(0) = p, \quad \gamma'(t) = X_{\gamma(t)} \quad \text{for all } t \in I.$$

The condition $\gamma'(t) = X_{\gamma(t)}$ says that the velocity of the curve at each moment equals the value of the vector field at the curve's current position. In local coordinates, if $\gamma(t) = (\gamma^1(t), \ldots, \gamma^n(t))$, this becomes a system of ODEs:

$$\frac{d\gamma^i}{dt} = X^i(\gamma^1(t), \ldots, \gamma^n(t)), \quad i = 1, \ldots, n.$$

**Theorem (Existence and Uniqueness).** Let $X$ be a smooth vector field on $M$ and $p \in M$. Then there exists $\varepsilon > 0$ and a unique smooth curve $\gamma: (-\varepsilon, \varepsilon) \to M$ satisfying $\gamma(0) = p$ and $\gamma'(t) = X_{\gamma(t)}$.

This follows from the Picard-Lindelof theorem for ODEs in local coordinates, together with the observation that smooth implies Lipschitz on compact sets. Moreover, the solution depends smoothly on the initial condition $p$ — a crucial fact for defining flows.

**Maximal integral curves.** The integral curve through $p$ can be extended to a maximal domain $I_p = (\alpha_p, \omega_p)$, the largest open interval on which the solution exists. The integral curve is unique on its maximal domain. If $M$ is compact, or more generally if $X$ has compact support, then every maximal integral curve is defined on all of $\mathbb{R}$.

**Example.** For the rotation field $X = \frac{\partial}{\partial \phi}$ on $S^2$, the integral curve through a point $(\theta_0, \phi_0)$ is $\gamma(t) = (\theta_0, \phi_0 + t)$ — uniform rotation along the latitude circle. The curve is defined for all $t \in \mathbb{R}$ and is periodic with period $2\pi$.

**Example: linear vector fields on $\mathbb{R}^n$.** Let $A$ be an $n \times n$ real matrix and define $X_A(x) = Ax$ (viewing $T_x\mathbb{R}^n \cong \mathbb{R}^n$). The integral curve through $p$ satisfies $\dot{\gamma}(t) = A\gamma(t)$, with solution $\gamma(t) = e^{tA}p$. This is defined for all $t \in \mathbb{R}$ — linear vector fields on $\mathbb{R}^n$ are always complete. When $A$ is skew-symmetric ($A^T = -A$), the flow $e^{tA}$ is a one-parameter family of orthogonal transformations (rotations); when $A$ is symmetric, the flow stretches or compresses along the eigendirections; in general, the Jordan normal form of $A$ determines the qualitative behavior of the flow (spirals, saddle points, nodes, etc.).

For the vector field $Y = x \frac{\partial}{\partial x}$ on $\mathbb{R}$, the integral curve through $p \in \mathbb{R}$ satisfies $\dot{\gamma} = \gamma$, giving $\gamma(t) = p \cdot e^t$. This is defined for all $t$ — the curve "escapes to infinity" but never reaches it in finite time. However, for $Z = x^2 \frac{\partial}{\partial x}$ on $\mathbb{R}$, the ODE $\dot{\gamma} = \gamma^2$ gives $\gamma(t) = p/(1 - pt)$, which blows up at $t = 1/p$ for $p > 0$. Not all vector fields on non-compact manifolds have complete flows. The distinction between $x$ and $x^2$ is growth rate: linear growth produces exponential solutions (which live forever), while quadratic growth produces rational solutions (which blow up in finite time).

---

## Flows: One-Parameter Groups of Diffeomorphisms

**Definition.** The **flow** of a vector field $X$ on $M$ is the map $\theta: \mathcal{D} \to M$, where $\mathcal{D} = \{(t, p) \in \mathbb{R} \times M : t \in I_p\}$, defined by $\theta(t, p) = \gamma_p(t)$, the integral curve through $p$ evaluated at time $t$. We write $\theta_t(p) = \theta(t, p)$.

The key properties are:

1. **Group law:** $\theta_0 = \text{id}_M$ and $\theta_s \circ \theta_t = \theta_{s+t}$ wherever both sides are defined.
2. **Smoothness:** $\theta: \mathcal{D} \to M$ is smooth (by smooth dependence on initial conditions and parameters).
3. **Each $\theta_t$ is a diffeomorphism** from its domain onto its image, with inverse $\theta_{-t}$.

If $X$ is **complete** — every integral curve is defined for all $t \in \mathbb{R}$ — then $\theta_t: M \to M$ is a diffeomorphism for each $t$, and the map $t \mapsto \theta_t$ is a group homomorphism $(\mathbb{R}, +) \to \text{Diff}(M)$. This is a **one-parameter group of diffeomorphisms**.

**Theorem.** Every smooth vector field on a compact manifold is complete.

*Proof sketch.* On a compact manifold, integral curves cannot "escape to infinity" (there is no infinity to escape to). The maximal interval of existence can only be finite if the curve leaves every compact set, which cannot happen when the manifold is compact. $\square$

**Example: rotation flow on $S^2$.** The flow of $X = \frac{\partial}{\partial \phi}$ is $\theta_t(\theta_0, \phi_0) = (\theta_0, \phi_0 + t)$, which in Cartesian coordinates is:

$$\theta_t(x, y, z) = (x \cos t - y \sin t, x \sin t + y \cos t, z).$$

This is rotation about the $z$-axis by angle $t$. Each $\theta_t$ is an isometry of $S^2$, and $t \mapsto \theta_t$ gives the one-parameter subgroup of $SO(3)$ corresponding to the $z$-axis.

**Example: the Hopf flow on $S^3$.** The 3-sphere $S^3 \subseteq \mathbb{C}^2$ carries a natural vector field defined by $X_{(z_1, z_2)} = (iz_1, iz_2)$, where $i = \sqrt{-1}$ denotes complex multiplication. The flow is $\theta_t(z_1, z_2) = (e^{it}z_1, e^{it}z_2)$ — rotation by the same angle in both complex coordinates. Each orbit is a great circle in $S^3$, and the space of orbits is $S^2$ (the **Hopf fibration** $S^1 \to S^3 \to S^2$). This is a deep example connecting dynamics, topology, and geometry: the flow decomposes $S^3$ into a family of linked circles, parametrized by points of $S^2$.

**Example: gradient flow.** On a Riemannian manifold $(M, g)$, the gradient of a smooth function $f: M \to \mathbb{R}$ is the vector field $\text{grad}\, f$ defined by $g(\text{grad}\, f, Y) = Yf$ for all vector fields $Y$. The flow of $-\text{grad}\, f$ is the **gradient descent flow**: integral curves move in the direction of steepest decrease of $f$. This is the continuous-time version of the gradient descent algorithm. On compact $M$, this flow exists for all time and converges to critical points of $f$ (under mild conditions — this is the content of Morse theory).

**The flow-box theorem.** If $X_p \neq 0$, there exist local coordinates $(y^1, \ldots, y^n)$ around $p$ such that $X = \frac{\partial}{\partial y^1}$ in these coordinates. This means that near any non-singular point, a vector field looks like a constant field — the integral curves are parallel straight lines. The flow-box theorem reduces the local study of vector fields to the trivial case; all the interesting behavior is global (how integral curves wrap around the manifold, where they accumulate, where they diverge). This is why the qualitative theory of dynamical systems — studying limit cycles, attractors, and chaotic behavior — is fundamentally a global enterprise.

**Fixed points and their classification.** A point $p$ where $X_p = 0$ is called a **fixed point** (or singular point, or equilibrium) of the flow. Near a fixed point, the flow-box theorem fails, and the local behavior is governed by the linearization $dX_p: T_pM \to T_pM$. The eigenvalues of $dX_p$ classify the fixed point: if all eigenvalues have negative real part, $p$ is a sink (stable node); if all have positive real part, a source (unstable node); if some are positive and some negative, a saddle. This is the Hartman-Grobman theorem, which states that near a hyperbolic fixed point (all eigenvalues have nonzero real part), the flow is topologically conjugate to its linearization.

---

## The Lie Bracket $[X, Y]$

Given two vector fields $X, Y \in \mathfrak{X}(M)$, each generates a flow. What happens if we flow along $X$ for a short time $\varepsilon$, then along $Y$ for time $\varepsilon$, then back along $X$ for time $\varepsilon$, then back along $Y$ for time $\varepsilon$? If the flows commute ($\theta^\varepsilon_X \circ \theta^\varepsilon_Y = \theta^\varepsilon_Y \circ \theta^\varepsilon_X$), we return to the starting point. If they don't, the displacement is (to leading order) proportional to $\varepsilon^2$, and the direction of this displacement defines a new vector field: the **Lie bracket**.

More precisely, denote the flow of $X$ by $\theta^X$ and the flow of $Y$ by $\theta^Y$. Then the "commutator curve" starting at $p$ is:

$$c(\varepsilon) = \theta^Y_{-\varepsilon} \circ \theta^X_{-\varepsilon} \circ \theta^Y_\varepsilon \circ \theta^X_\varepsilon(p).$$

A Taylor expansion shows $c(\varepsilon) = p + \varepsilon^2 [X, Y]_p + O(\varepsilon^3)$. The Lie bracket is the infinitesimal commutator of the two flows. This geometric interpretation is crucial: $[X, Y] = 0$ if and only if the flows of $X$ and $Y$ commute.

**Definition (algebraic).** The **Lie bracket** of $X, Y \in \mathfrak{X}(M)$ is the vector field $[X, Y]$ defined by:

$$[X, Y](f) = X(Yf) - Y(Xf) \quad \text{for all } f \in C^\infty(M).$$

This is well-defined: even though $X \circ Y$ (as a composition of derivations) is a second-order operator, the commutator $[X, Y] = X \circ Y - Y \circ X$ is first-order, hence a vector field.

**In local coordinates:** if $X = X^i \frac{\partial}{\partial x^i}$ and $Y = Y^j \frac{\partial}{\partial x^j}$, then:

$$[X, Y] = \left(X^j \frac{\partial Y^i}{\partial x^j} - Y^j \frac{\partial X^i}{\partial x^j}\right) \frac{\partial}{\partial x^i}.$$

**Verification that $[X, Y]$ is first-order.** Let us check this directly. Applying $X \circ Y$ to a function $f$:

$$(X \circ Y)(f) = X^i \frac{\partial}{\partial x^i}\left(Y^j \frac{\partial f}{\partial x^j}\right) = X^i \frac{\partial Y^j}{\partial x^i} \frac{\partial f}{\partial x^j} + X^i Y^j \frac{\partial^2 f}{\partial x^i \partial x^j}.$$

The second term contains second derivatives of $f$. But in $Y \circ X$ we get the same second-derivative term $Y^j X^i \frac{\partial^2 f}{\partial x^j \partial x^i}$, which equals $X^i Y^j \frac{\partial^2 f}{\partial x^i \partial x^j}$ by symmetry of mixed partials. Subtracting, the second derivatives cancel, and we are left with first-order terms only. This cancellation is the algebraic miracle that makes the Lie bracket a derivation rather than a second-order operator.

**Properties of the Lie bracket:**
1. **Bilinearity:** $[aX + bY, Z] = a[X, Z] + b[Y, Z]$ and similarly in the second argument.
2. **Antisymmetry:** $[X, Y] = -[Y, X]$.
3. **Jacobi identity:** $[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$.
4. **Leibniz over $C^\infty(M)$ multiplication:** $[fX, Y] = f[X, Y] - (Yf)X$.

Properties (1)-(3) say that $\mathfrak{X}(M)$ is a **Lie algebra** over $\mathbb{R}$. Property (4) shows the bracket is *not* $C^\infty(M)$-linear — it involves differentiation, not just pointwise operations.

**Definition (via flows).** Equivalently, the Lie bracket can be defined using flows:

$$[X, Y]_p = \lim_{t \to 0} \frac{(\theta^X_{-t})_* Y_{\theta^X_t(p)} - Y_p}{t} = \frac{d}{dt}\bigg|_{t=0} (\theta^X_{-t})_* Y_{\theta^X_t(p)}.$$

This formula says: flow along $X$ for time $t$, evaluate $Y$ at the new point, pull back to the original point, and differentiate at $t = 0$. The Lie bracket measures the infinitesimal rate of change of $Y$ along the flow of $X$.

**Example.** On $\mathbb{R}^2$, let $X = \frac{\partial}{\partial x}$ (translation in $x$) and $Y = x \frac{\partial}{\partial y}$ (vertical shear). Then:

$$[X, Y](f) = X(Yf) - Y(Xf) = \frac{\partial}{\partial x}\left(x \frac{\partial f}{\partial y}\right) - x \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right) = \frac{\partial f}{\partial y}.$$

So $[X, Y] = \frac{\partial}{\partial y}$. The flow of $X$ is horizontal translation, the flow of $Y$ is vertical shear proportional to $x$. They don't commute: translating right increases the shear, producing a net vertical displacement. The Lie bracket captures this non-commutativity.

**Example: rotation fields on $S^2$.** Define vector fields generating rotation about the three coordinate axes. In Cartesian coordinates on $\mathbb{R}^3$ restricted to $S^2$:

$$R_x = -z\frac{\partial}{\partial y} + y\frac{\partial}{\partial z}, \quad R_y = z\frac{\partial}{\partial x} - x\frac{\partial}{\partial z}, \quad R_z = -y\frac{\partial}{\partial x} + x\frac{\partial}{\partial y}.$$

Computing the brackets: $[R_x, R_y] = R_z$, $[R_y, R_z] = R_x$, $[R_z, R_x] = R_y$. These are the structure equations of the Lie algebra $\mathfrak{so}(3)$. The non-vanishing brackets reflect the fact that rotations about different axes don't commute — a fact with profound consequences in quantum mechanics (where $R_x, R_y, R_z$ become the angular momentum operators $L_x, L_y, L_z$ satisfying $[L_x, L_y] = i\hbar L_z$).

---

## The Lie Derivative of Vector Fields and Tensors

The Lie bracket is a special case of a more general operation: the **Lie derivative** $\mathcal{L}_X$, which measures the rate of change of any tensor field along the flow of $X$.

**Lie derivative of a function.** For $f \in C^\infty(M)$:

$$\mathcal{L}_X f = Xf = \frac{d}{dt}\bigg|_{t=0} (\theta_t^X)^* f = \frac{d}{dt}\bigg|_{t=0} f \circ \theta_t^X.$$

**Lie derivative of a vector field.** For $Y \in \mathfrak{X}(M)$:

$$\mathcal{L}_X Y = [X, Y] = \frac{d}{dt}\bigg|_{t=0} (\theta_{-t}^X)_* Y.$$

Note the pushforward $(\theta_{-t})_*$ rather than pullback — vector fields push forward, not pull back.

**Lie derivative of a 1-form.** For a smooth 1-form $\omega$ (a smooth section of $T^*M$, which we will study in detail next article):

$$(\mathcal{L}_X \omega)(Y) = X(\omega(Y)) - \omega([X, Y]).$$

**Example: Lie derivative computation.** On $\mathbb{R}^2$, let $X = x\frac{\partial}{\partial x} + y\frac{\partial}{\partial y}$ (the radial vector field generating scaling), and let $\omega = dx \wedge dy$ (the standard area form). Then $\mathcal{L}_X \omega = d(\iota_X \omega) + \iota_X(d\omega)$. Since $d\omega = 0$ (it is a 2-form on $\mathbb{R}^2$, hence top-degree and automatically closed), we get $\mathcal{L}_X \omega = d(\iota_X \omega)$. Now $\iota_X(dx \wedge dy) = x\,dy - y\,dx$, so $d(\iota_X \omega) = dx \wedge dy + dx \wedge dy = 2\,dx \wedge dy$. Thus $\mathcal{L}_X \omega = 2\omega$: the scaling field stretches the area form by a factor of 2 (at rate 2, since the flow is $\theta_t(x,y) = (e^t x, e^t y)$ and the area scales as $e^{2t}$).

**Lie derivative of a general tensor.** The Lie derivative extends to arbitrary tensor fields by requiring the Leibniz rule with respect to the tensor product. For a $(r, s)$-tensor field $T$:

$$\mathcal{L}_X T = \frac{d}{dt}\bigg|_{t=0} (\theta_t^X)^* T.$$

The pullback $(\theta_t)^*$ acts on a $(0, s)$-tensor by precomposition with the pushforward; the general case uses the pushforward for contravariant indices and pullback for covariant indices.

**Cartan's magic formula** (for differential forms, previewing the next article):

$$\mathcal{L}_X \omega = d(\iota_X \omega) + \iota_X(d\omega)$$

where $\iota_X$ is interior multiplication (contraction with $X$) and $d$ is the exterior derivative. This elegant formula reduces the computation of Lie derivatives of forms to two simpler operations.

**Physical significance.** A tensor field $T$ is **invariant** under the flow of $X$ if and only if $\mathcal{L}_X T = 0$. This is the infinitesimal version of a symmetry condition. In general relativity, a Killing vector field is a vector field $X$ satisfying $\mathcal{L}_X g = 0$ — the metric is invariant under the flow of $X$, meaning $X$ generates an isometry. The existence of Killing fields constrains the geometry and leads to conservation laws via Noether's theorem.

---

## Frobenius Theorem: Integrability of Distributions

Given a single vector field $X$, its integral curves foliate $M$ into one-dimensional curves (on the open set where $X \neq 0$). What if we have $k$ linearly independent vector fields $X_1, \ldots, X_k$? Can we find $k$-dimensional submanifolds $N$ through each point such that $T_p N = \text{span}(X_1|_p, \ldots, X_k|_p)$ for all $p \in N$?

**Definition.** A **$k$-dimensional distribution** $\mathcal{D}$ on $M$ is a smooth assignment $p \mapsto \mathcal{D}_p \subseteq T_pM$ of a $k$-dimensional subspace of each tangent space. Locally, $\mathcal{D}$ can be spanned by $k$ smooth vector fields.

**Definition.** A distribution $\mathcal{D}$ is **integrable** if through each point $p \in M$, there exists a connected submanifold $N$ (called an **integral manifold** or **leaf**) such that $T_q N = \mathcal{D}_q$ for all $q \in N$. The manifold $M$ is then foliated by these integral manifolds.

**Definition.** A distribution $\mathcal{D}$ is **involutive** if whenever $X, Y$ are smooth vector fields lying in $\mathcal{D}$ (i.e., $X_p, Y_p \in \mathcal{D}_p$ for all $p$), the Lie bracket $[X, Y]$ also lies in $\mathcal{D}$.

**Theorem (Frobenius).** A smooth distribution is integrable if and only if it is involutive.

The forward direction is elementary: if $\mathcal{D}$ is integrable and $X, Y$ are tangent to the integral manifolds, then $[X, Y]$ (being the Lie derivative of $Y$ along $X$) is also tangent to the integral manifolds, hence lies in $\mathcal{D}$.

The converse — involutivity implies integrability — is the deep content. The proof uses the flow-box theorem and induction on $k$, constructing the integral manifold as an intersection of level sets of suitable functions.

**Example: integrability.** On $\mathbb{R}^3$, let $X = \frac{\partial}{\partial x}$ and $Y = \frac{\partial}{\partial y}$. Then $[X, Y] = 0 \in \text{span}(X, Y)$, so the distribution $\text{span}(X, Y)$ is involutive. The integral manifolds are the horizontal planes $\{z = c\}$ — exactly what you'd expect.

**Example: non-integrability.** On $\mathbb{R}^3$, let $X = \frac{\partial}{\partial x}$ and $Y = \frac{\partial}{\partial y} + x \frac{\partial}{\partial z}$. Then:

$$[X, Y] = \left[\frac{\partial}{\partial x}, \frac{\partial}{\partial y} + x \frac{\partial}{\partial z}\right] = \frac{\partial}{\partial z}.$$

Since $\frac{\partial}{\partial z} \notin \text{span}(X, Y)$ at any point, the distribution is not involutive, hence not integrable. There is no 2-dimensional surface in $\mathbb{R}^3$ that is everywhere tangent to $\text{span}(X, Y)$. This distribution is a **contact structure** on $\mathbb{R}^3$ — a maximally non-integrable distribution, central to contact geometry and symplectic topology.

Geometrically, imagine trying to walk in the $(x, y)$-plane while staying tangent to the distribution. Moving in the $x$-direction doesn't change $z$, but moving in the $y$-direction at position $x$ forces you upward at rate $x$. By making a small square (right, forward, left, backward), you end up at a different $z$-height than where you started. This "holonomy" is precisely measured by $[X, Y] = \frac{\partial}{\partial z}$.

**Physical application.** In thermodynamics, the 1-form $\omega = dE - T\,dS + P\,dV$ defines a distribution $\ker \omega$ (the contact structure of thermodynamic state space). The non-integrability of this distribution is equivalent to the impossibility of a "potential function" for heat exchange — the physical content of the second law of thermodynamics.

**Application to Lie groups.** For a Lie group $G$, the Lie algebra $\mathfrak{g}$ can be identified with the space of **left-invariant** vector fields — vector fields $X$ satisfying $(L_g)_* X = X$ for all $g \in G$, where $L_g: G \to G$ is left multiplication. These vector fields are determined by their value at the identity, giving the isomorphism $\mathfrak{g} \cong T_e G$. The Lie bracket of left-invariant vector fields is again left-invariant, and this bracket is precisely the Lie bracket of $\mathfrak{g}$.

The Frobenius theorem for Lie groups has a striking consequence: every **Lie subalgebra** $\mathfrak{h} \subseteq \mathfrak{g}$ (a subspace closed under the bracket) determines a unique connected Lie subgroup $H \subseteq G$ with $T_e H = \mathfrak{h}$. The left-invariant vector fields spanning $\mathfrak{h}$ form an involutive distribution (since $\mathfrak{h}$ is closed under brackets), and the integral manifold through the identity is the subgroup $H$. This is Lie's fundamental theorem: the passage between Lie groups and Lie algebras.

---

## What's Next

We now have the tools to study how tangent vectors move and interact: vector fields, flows, the Lie bracket, the Lie derivative, and the Frobenius integrability theorem. The next article turns to the dual side: **differential forms** — the objects that live in the cotangent bundle $T^*M$ rather than the tangent bundle $TM$. Forms are the natural objects to integrate on manifolds (you can integrate a $k$-form over a $k$-dimensional submanifold), and the exterior derivative $d$ unifies gradient, curl, and divergence into a single operator. Together with the Lie derivative and interior product, forms provide the most elegant computational framework for differential geometry.

**Summary of the key ideas.** The conceptual arc of this article:

1. A **vector field** assigns a tangent vector to each point, smoothly. Algebraically, it is a derivation of $C^\infty(M)$; geometrically, it is a velocity prescription on $M$.
2. **Integral curves** are the trajectories of the velocity field, given by a system of ODEs. Existence and uniqueness are guaranteed locally by the Picard-Lindelof theorem; on compact manifolds, solutions exist for all time.
3. The **flow** of a vector field is the one-parameter family of diffeomorphisms generated by following all integral curves simultaneously. It is the exponentiation of an infinitesimal symmetry.
4. The **Lie bracket** $[X, Y]$ is the infinitesimal commutator of two flows. It encodes the non-commutativity of infinitesimal symmetries and makes $\mathfrak{X}(M)$ a Lie algebra.
5. The **Lie derivative** $\mathcal{L}_X$ generalizes the Lie bracket to arbitrary tensor fields, measuring rates of change along a flow.
6. The **Frobenius theorem** provides a criterion (involutivity) for when a family of vector fields can be simultaneously "integrated" into submanifolds — the bridge between local and global integrability.

---

*This is Part 7 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 6 — Smooth Manifolds](/en/differential-geometry/06-smooth-manifolds/)*

*Next: [Part 8 — Differential Forms](/en/differential-geometry/08-differential-forms/)*
