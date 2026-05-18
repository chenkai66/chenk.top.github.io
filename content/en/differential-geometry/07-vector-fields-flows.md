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

A tangent vector lives at one point. It tells you "this direction, this speed, right here, right now." It is fundamentally local — pluck it off the manifold and it remembers nothing about its neighbors. A **vector field**, by contrast, is what you get when you let one tangent vector at every point conspire smoothly. It is a velocity prescription on the entire manifold: stand anywhere, and the field tells you where to go. Follow the prescription, and you trace out an **integral curve**. Follow it from every starting point at once, and you get a **flow** — a one-parameter family of diffeomorphisms that drags the whole manifold along itself like a slow river.

I want to stress that this picture is not just kinematic decoration. Vector fields are how differential geometry encodes infinitesimal symmetry. Every conserved quantity in Hamiltonian mechanics arises from a vector field whose flow preserves the Hamiltonian. Every one-parameter subgroup of a Lie group is the flow of a left-invariant vector field. Every gauge transformation in Yang-Mills theory is, infinitesimally, a vector field on the configuration space of connections. And the **Lie bracket** $[X, Y]$ — the gadget that measures how badly the flows of $X$ and $Y$ fail to commute — is exactly the algebraic structure that turns the tangent space at the identity of a Lie group into its Lie algebra. Understanding vector fields well is, by a substantial margin, the prerequisite to understanding everything that follows in this series.

The plan for this article: define vector fields three different ways (geometric, algebraic, sectional), turn them into integral curves via ODE theory, package the curves into flows, define the bracket and the Lie derivative, work through enough numerical examples to make the formulas reflexive, then close with the Frobenius theorem on integrability — the deep result that ties everything together. Throughout, I will resist the temptation to prove things from scratch (Lee's textbook does that very well) and instead focus on what the formulas mean and how to compute with them.

---

## 1. Vector Fields: Smooth Sections of the Tangent Bundle


![Vector field with integral curves (spiral source)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/dg07_vector_fields_2d.png)

![Vector fields on the plane: sources, sinks, and vortices](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_vector_field_2d.png)


A **vector field** $X$ on a smooth manifold $M$ is a smooth assignment $p \mapsto X_p \in T_p M$. In bundle language, it is a smooth section of the tangent bundle $TM \to M$: pick a tangent vector at every point, and require that the choice vary smoothly as you move. The space of all vector fields on $M$ is denoted $\mathfrak{X}(M)$. It is simultaneously a real vector space (you can add fields and scale by constants), a module over $C^\infty(M)$ (you can multiply by smooth functions pointwise), and — as we will see in section 4 — a Lie algebra under the bracket.

In a coordinate chart $(x^1, \dots, x^n)$, every vector field can be written
$$X = X^i(x) \frac{\partial}{\partial x^i},$$
where the components $X^i$ are smooth functions of the coordinates. The intuition is simple: at each point, the vector $X_p$ has $n$ components in the coordinate basis $\{\partial_i\}$, and those components vary smoothly with $p$. Smoothness of $X$ as a section is equivalent to smoothness of all the component functions $X^i$ in any chart; you only need to check this on an atlas.

![Planar vector field shown as a quiver plot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_1_vector_field_2d.png)

There is a second, less geometric but operationally crucial way to think of a vector field: as a **derivation** of the algebra $C^\infty(M)$. Given $X$ and a smooth function $f$, define
$$X(f)(p) = X_p(f),$$
where on the right, $X_p$ acts on $f$ as a tangent vector (a directional derivative). This produces a new smooth function $X(f)$. The map $X: C^\infty(M) \to C^\infty(M)$ is $\mathbb{R}$-linear and satisfies the Leibniz rule
$$X(fg) = f \cdot X(g) + g \cdot X(f).$$
A theorem of Hadamard says that *every* such derivation arises from a unique vector field. So we have two equivalent definitions: geometric (a section of $TM$) and algebraic (a derivation of $C^\infty(M)$). Use whichever is more convenient — and one of the basic skills of differential geometry is knowing instantly which is more convenient for any given problem.

**A first numerical example.** On $\mathbb{R}^2$, let $X = -y\,\partial_x + x\,\partial_y$ — the standard rotation field. At the point $(1, 0)$, $X = (0, 1)$: pointing straight up. At $(0, 1)$, $X = (-1, 0)$: pointing left. At $(1, 1)$, $X = (-1, 1)$: northwest, with speed $\sqrt{2}$. If we apply $X$ to the function $f(x, y) = x^2 + y^2$, we get
$$X(f) = -y \cdot 2x + x \cdot 2y = 0.$$
The function $f = x^2 + y^2$ is **invariant** under $X$. Geometrically, this is obvious: $X$ generates rotations, and rotations preserve distance from the origin. Algebraically, the derivation killed the radial function. Both perspectives give the same answer, which is comforting.

Apply $X$ to $g(x, y) = x$: we get $X(g) = -y$. To $h(x, y) = xy$: we get $X(h) = -y \cdot y + x \cdot x = x^2 - y^2$. Notice that $h$ is *not* invariant (the field is rotational but $xy$ rotates into something different). This kind of computation is what occupies the first hour of any practical work with vector fields, and being able to read $X = -y\partial_x + x\partial_y$ as "rotation" without computing should be your first goal.

**Why this matters.** The derivation perspective is what makes vector fields composable. If $X, Y \in \mathfrak{X}(M)$, you cannot meaningfully "add" them as sections of $TM$ in any nontrivial way beyond pointwise addition. But as derivations, you can compose them: $(XY)(f) = X(Y(f))$. This composition is *not* itself a derivation (it fails Leibniz — try it on $f = g = x$ and watch the second derivatives appear), but a clever combination of $XY$ and $YX$ *is* a derivation, and that combination is the Lie bracket. So the algebraic viewpoint is what unlocks the bracket — and through the bracket, the entire theory of Lie groups and infinitesimal symmetry.

---

## 2. Integral Curves: Following the Field

An **integral curve** of $X$ through $p \in M$ is a smooth curve $\gamma: I \to M$ defined on an open interval $I \ni 0$ such that
$$\gamma(0) = p, \qquad \dot\gamma(t) = X_{\gamma(t)} \text{ for all } t \in I.$$
At every instant, the curve's velocity is exactly the value of the vector field at the curve's current location. In coordinates, this is a system of ordinary differential equations:
$$\frac{d x^i}{dt} = X^i(x^1(t), \dots, x^n(t)), \qquad x^i(0) = p^i.$$
This is a first-order autonomous system in $n$ variables.

![Animation: flow of a vector field over time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_flow.gif)


![Flow of a vector field: integral curves](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_flow_map.png)


![Integral curves of a vector field via streamplot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_2_integral_curves.png)

The **Picard-Lindelof theorem** guarantees that for any smooth $X$ and any starting point $p$, an integral curve exists and is unique on some open interval around $0$. The interval might be small (the integral curve could escape to infinity in finite time — think of $\dot x = x^2$ on $\mathbb{R}$, which blows up at $t = 1/x_0$), but locally everything is fine. On compact manifolds, the interval is always all of $\mathbb{R}$ — there is nowhere to escape to — and the field is called **complete**.

**Worked example 1 — rotation.** Take $X = -y\,\partial_x + x\,\partial_y$ on $\mathbb{R}^2$. The ODE system is
$$\dot x = -y, \qquad \dot y = x.$$
Differentiating once more, $\ddot x = -\dot y = -x$. So $x(t) = A\cos t + B\sin t$. With $x(0) = p^1$, $y(0) = p^2$, the solution is
$$\gamma(t) = (p^1 \cos t - p^2 \sin t,\; p^1 \sin t + p^2 \cos t).$$
This is exactly rotation by angle $t$ around the origin. The integral curves are circles, parametrized by angle. The fixed point — where $X$ vanishes — is the origin.

**Worked example 2 — exponential growth.** Take $X = x\,\partial_x$ on $\mathbb{R}$. The ODE is $\dot x = x$, so $x(t) = p e^t$. Integral curves are exponentially expanding rays. Starting at $p > 0$, you blow up to $+\infty$ as $t \to +\infty$ and shrink to $0$ as $t \to -\infty$. The origin is a fixed point, but now an unstable one. Notice that $X$ is complete on $\mathbb{R}$: the integral curve $x(t) = p e^t$ is defined for all $t$.

**Worked example 3 — finite-time blowup.** Take $X = x^2 \partial_x$ on $\mathbb{R}$. The ODE $\dot x = x^2$ has solution $x(t) = p / (1 - pt)$, which blows up at $t = 1/p$ if $p > 0$. So $X$ is *not* complete on $\mathbb{R}$. This is the local-vs-global distinction in action: existence is always local, but global existence requires a growth bound on $X$ (intuitively: the integral curve must not run off to infinity in finite time).

**Why this matters.** Integral curves are the trajectories of physical systems. Newton's law $\ddot{q} = -\nabla V(q)$ becomes a first-order system on the phase space $(q, p)$ via $\dot q = p, \dot p = -\nabla V$. The integral curves of this Hamiltonian vector field are precisely the trajectories of the mechanical system. ODE existence and uniqueness translate into the deterministic evolution of physics. When physicists say "given initial conditions, the future is determined," they are invoking Picard-Lindelof on the manifold of states.

---

## 3. Flows: One-Parameter Families of Diffeomorphisms

If $X$ is complete, we can stitch all integral curves together. Define
$$\varphi_t: M \to M, \qquad \varphi_t(p) = \gamma_p(t),$$
where $\gamma_p$ is the integral curve starting at $p$. The map $\varphi_t$ pushes every point along its integral curve for time $t$. The collection $\{\varphi_t\}_{t \in \mathbb{R}}$ is the **flow** of $X$.

![Lie bracket measures non-commutativity of flows](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_lie_bracket.png)


![Flow phi_t advancing each point along its integral curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_3_flow_phi_t.png)

The flow satisfies three properties that I want to highlight, because they are the entire reason flows are useful:

1. **Identity:** $\varphi_0 = \mathrm{id}_M$.
2. **Group law:** $\varphi_{s+t} = \varphi_s \circ \varphi_t$.
3. **Smoothness:** the map $(t, p) \mapsto \varphi_t(p)$ is smooth, and each $\varphi_t$ is a diffeomorphism with inverse $\varphi_{-t}$.

The group law is the punchline. It says the flow is a homomorphism from the additive group $(\mathbb{R}, +)$ into the diffeomorphism group $\mathrm{Diff}(M)$. In other words, every complete vector field $X$ generates a **one-parameter subgroup** of diffeomorphisms. The vector field is the *infinitesimal generator*; the flow is the *exponentiation*. We sometimes write $\varphi_t = \exp(tX)$ to emphasize this — and on Lie groups, this notation is literally a matrix exponential.

**Continuing the rotation example.** For $X = -y\,\partial_x + x\,\partial_y$, the flow is
$$\varphi_t(x, y) = (x\cos t - y\sin t,\; x\sin t + y\cos t),$$
which is exactly the rotation matrix $R_t$ acting on $\mathbb{R}^2$. The group law $\varphi_s \circ \varphi_t = \varphi_{s+t}$ becomes $R_s R_t = R_{s+t}$, the angle-addition formula. We have just rediscovered $\mathrm{SO}(2) \cong S^1$ as the flow of a vector field on $\mathbb{R}^2$, and the angle-addition identity for sine and cosine has been recast as a group homomorphism property of the flow.

**Continuing the dilation example.** For $X = x\,\partial_x$ on $\mathbb{R}$, the flow is $\varphi_t(x) = e^t x$. The group law is $e^s e^t = e^{s+t}$, and we have rediscovered $(\mathbb{R}_{>0}, \times) \cong (\mathbb{R}, +)$ as a flow. The exponential map on this Lie group is literally the exponential function from calculus.

**Pushforward of a vector field.** If $\varphi_t$ is the flow of $X$ and $Y$ is another vector field, we can ask: how does $Y$ change as we drag it along $X$'s flow? The answer is the *pushforward* $(\varphi_t)_* Y$, defined by $((\varphi_t)_* Y)_q = (d\varphi_t)_p Y_p$ where $q = \varphi_t(p)$. This will be central in section 4 when we define the Lie bracket as a derivative of pushforwards.

**Aside: incomplete fields and local flows.** When $X$ is not complete — e.g. $X = x^2 \partial_x$ on $\mathbb{R}$ — the flow is only defined on an open subset of $\mathbb{R} \times M$ called the *flow domain*. For each $p$, there is a maximal interval $(a_p, b_p)$ on which $\varphi_t(p)$ is defined; the integral curve "escapes" if $b_p < \infty$. Most theorems involving flows still work locally, by restricting to small enough $t$. In practice, complete vector fields are common (compactly supported $X$ on any manifold; bounded $X$ on $\mathbb{R}^n$; left-invariant fields on Lie groups), so the distinction is more often a technical aside than a serious obstacle.

**Why this matters.** Every continuous symmetry in physics — translation, rotation, time evolution, gauge transformation — is the flow of a vector field. Noether's theorem in its modern form says: a symmetry of the Lagrangian (a flow that preserves the action) gives rise to a conserved quantity (a function constant along the flow). The dictionary between vector fields and one-parameter symmetry groups is the geometric core of mechanics. Without it, you cannot do classical field theory in any serious way.

---

## 4. The Lie Bracket: Failure of Flows to Commute

Given two vector fields $X, Y$, what happens if we flow along $X$ for a small time $\epsilon$, then along $Y$, then back along $-X$, then back along $-Y$? Naively, we should return to where we started. We don't. The discrepancy is a vector at $p$, and to second order in $\epsilon$ it is exactly $\epsilon^2 [X, Y]_p$. The bracket is a quantitative measure of how non-commuting the flows are.

![Vector fields on the sphere](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_vector_field_sphere.png)


![Lie bracket [X,Y] measuring the failure of flows to commute](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_4_lie_bracket.png)

**Algebraic definition.** The **Lie bracket** of $X, Y \in \mathfrak{X}(M)$ is the vector field $[X, Y]$ defined by
$$[X, Y](f) = X(Y(f)) - Y(X(f)).$$
This is a derivation of $C^\infty(M)$ — the failures of Leibniz from $XY$ and $YX$ exactly cancel — so it corresponds to a vector field by Hadamard's theorem. Notice that $XY$ alone is a *second-order* differential operator; the antisymmetrization knocks the second-order part out and leaves a first-order operator.

**Coordinate formula.** If $X = X^i \partial_i$ and $Y = Y^j \partial_j$, then
$$[X, Y]^k = X^i \frac{\partial Y^k}{\partial x^i} - Y^i \frac{\partial X^k}{\partial x^i}.$$
The second-order terms cancel because mixed partials commute; only first-order corrections survive. This is the formula you compute with.

**Geometric definition.** Equivalently,
$$[X, Y]_p = \lim_{t \to 0} \frac{Y_p - ((\varphi_t^X)_* Y)_p}{t} = \left.\frac{d}{dt}\right|_{t=0} (\varphi_{-t}^X)_* Y_p.$$
The bracket is the rate at which $Y$ changes when dragged backward by the flow of $X$. Antisymmetric in $X$ and $Y$ because the roles of "drag" and "compare" are swapped.

**Worked numerical example 1.** On $\mathbb{R}^2$, take $X = \partial_x$ and $Y = x\,\partial_y$. Then
$$[X, Y](f) = \partial_x(x \partial_y f) - x \partial_y(\partial_x f) = \partial_y f + x \partial_x \partial_y f - x \partial_y \partial_x f = \partial_y f.$$
So $[X, Y] = \partial_y$. Geometrically: $X$ translates in the $x$-direction, $Y$ translates in $y$ but with strength proportional to $x$. The two flows fail to commute precisely because $Y$ "depends on" $x$. This pair $(X, Y)$ is the canonical example of a non-integrable contact distribution (we revisit in section 8).

**Worked numerical example 2.** Angular momentum on $\mathbb{R}^3$. Define
$$L_x = -z\partial_y + y\partial_z, \quad L_y = -x\partial_z + z\partial_x, \quad L_z = -y\partial_x + x\partial_y.$$
Compute $[L_x, L_y]$. The $z$-component:
$$[L_x, L_y]^z = L_x^i \partial_i L_y^z - L_y^i \partial_i L_x^z = (-z)(0) + y(\partial_z(-x)) - (-x)(\partial_x(y)) - 0 \cdot (\dots) - z(\partial_z(0)) + (\dots)$$
Doing this carefully (it is a standard exercise): $[L_x, L_y] = L_z$. Cyclic permutations give the full $\mathfrak{so}(3)$ commutation relations
$$[L_x, L_y] = L_z, \quad [L_y, L_z] = L_x, \quad [L_z, L_x] = L_y.$$
This is the Lie algebra of rotations, written in vector-field form on $\mathbb{R}^3$.

**Properties.** The bracket on $\mathfrak{X}(M)$ satisfies:

1. **Bilinearity:** linear in each argument over $\mathbb{R}$.
2. **Antisymmetry:** $[X, Y] = -[Y, X]$.
3. **Jacobi identity:** $[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0$.

These are exactly the axioms of a **Lie algebra**. So $\mathfrak{X}(M)$ is an infinite-dimensional Lie algebra, and finite-dimensional Lie algebras (like $\mathfrak{so}(3)$ or $\mathfrak{su}(2)$) appear as subalgebras coming from group actions.

**Note on $C^\infty$-linearity.** Although $X, Y$ are $C^\infty$-linear in their arguments (you can multiply by smooth functions pointwise), the bracket $[fX, gY]$ is *not* simply $fg[X, Y]$. You get
$$[fX, gY] = fg[X, Y] + f X(g) Y - g Y(f) X.$$
Extra terms appear because the bracket differentiates the coefficient functions. This $C^\infty$-non-bilinearity will reappear in the next article when we contrast Lie derivatives (no extra structure needed) with covariant derivatives (which require a connection).

**Why this matters.** The Jacobi identity is not arbitrary — it falls out of the associativity of composition $XYZ$. The bracket is the algebraic shadow of associativity, just translated to first-order terms. When physicists write $[\hat L_x, \hat L_y] = i\hbar \hat L_z$ for angular momentum operators, they are computing a Lie bracket in the Lie algebra $\mathfrak{so}(3)$ realized as vector fields on $\mathbb{R}^3$. Quantum mechanics inherits its non-commutativity directly from the geometry. The factor of $i\hbar$ is just a unit choice; the algebra is geometric.

---

## 5. The Lie Derivative: Differentiating Anything Along a Flow

The bracket only differentiates vector fields. But we want to differentiate functions, forms, tensors — anything — along a flow. The general operation is the **Lie derivative**. For a tensor field $T$ and a vector field $X$ with flow $\varphi_t$, define
$$\mathcal{L}_X T = \lim_{t \to 0} \frac{\varphi_t^* T - T}{t} = \left.\frac{d}{dt}\right|_{t=0} \varphi_t^* T,$$
where $\varphi_t^*$ is the pullback by $\varphi_t$. Geometrically: drag $T$ along the flow for time $t$, compare to its original value, divide by $t$, take the limit.

![Integral curves of a nonlinear vector field](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_integral_curves.png)


![Lie derivative L_X T as the rate of change of T along the flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_5_lie_derivative.png)

**Special cases.**
- For a function $f$: $\mathcal{L}_X f = X(f)$, the directional derivative.
- For a vector field $Y$: $\mathcal{L}_X Y = [X, Y]$, the Lie bracket.
- For a 1-form $\omega$: by the Leibniz rule, $(\mathcal{L}_X \omega)(Y) = X(\omega(Y)) - \omega([X, Y])$.
- For a $k$-form, **Cartan's magic formula**: $\mathcal{L}_X = d \iota_X + \iota_X d$, where $\iota_X$ is interior product.

The Lie derivative satisfies Leibniz over tensor products and commutes with contractions. It is the canonical way to differentiate any object on a manifold along a vector field, *without needing extra structure* (no metric, no connection — just the smooth structure). This minimality is its defining virtue.

**Worked example.** On $\mathbb{R}^2$, take $X = -y\,\partial_x + x\,\partial_y$ (rotation) and $\omega = x\,dy - y\,dx$ (the angular form). Compute $\mathcal{L}_X \omega$ using Cartan's magic formula $\mathcal{L}_X = d \iota_X + \iota_X d$:
$$\iota_X \omega = \omega(X) = x \cdot x - y \cdot (-y) = x^2 + y^2.$$
$$d \iota_X \omega = d(x^2 + y^2) = 2x\,dx + 2y\,dy.$$
$$d\omega = 2\,dx \wedge dy, \qquad \iota_X d\omega = 2 X^i \partial_i \,\lrcorner\, dx\wedge dy = 2(-y\,dy - x\,dx) = -2x\,dx - 2y\,dy.$$
$$\mathcal{L}_X \omega = (2x\,dx + 2y\,dy) + (-2x\,dx - 2y\,dy) = 0.$$
The angular form is invariant under rotation — as it should be, since rotations preserve angle. The computation took six lines, but every step was mechanical.

**Worked example 2.** Same $X$, but now compute $\mathcal{L}_X g$ where $g = dx \otimes dx + dy \otimes dy$ is the Euclidean metric. The flow $\varphi_t$ is rotation by angle $t$, and rotation preserves Euclidean distance — so we expect $\mathcal{L}_X g = 0$. Indeed, computing in coordinates with the formula
$$(\mathcal{L}_X g)_{ij} = X^k \partial_k g_{ij} + g_{kj}\partial_i X^k + g_{ik}\partial_j X^k$$
(here $g_{ij} = \delta_{ij}$ so the first term vanishes):
$(\mathcal{L}_X g)_{xx} = \partial_x X^x + \partial_x X^x = 0 + 0 = 0$. (Since $X^x = -y$.)
$(\mathcal{L}_X g)_{xy} = \partial_x X^y + \partial_y X^x = 1 + (-1) = 0$.
$(\mathcal{L}_X g)_{yy} = \partial_y X^y + \partial_y X^y = 0$.
Confirmed: $\mathcal{L}_X g = 0$. The vector field $X$ is a **Killing vector field** for the Euclidean metric.

**Worked example 3.** Compute $\mathcal{L}_X (x^2 + y^2) dx\wedge dy$ for the same rotation $X$. The function $x^2 + y^2$ is invariant under $X$ (computed in section 1), so $\mathcal{L}_X(x^2 + y^2) = 0$. The form $dx \wedge dy$ is the volume form on $\mathbb{R}^2$, and $\mathcal{L}_X(dx\wedge dy) = (\nabla \cdot X) dx \wedge dy = 0$ since $X$ is divergence-free. By the Leibniz rule, $\mathcal{L}_X((x^2+y^2)dx\wedge dy) = 0$. Three different "preservation" facts about rotation, all encoded in single Lie-derivative computations.

**Why this matters.** Symmetries of geometric structures are exactly the vector fields with $\mathcal{L}_X(\text{structure}) = 0$. Killing fields preserve the metric ($\mathcal{L}_X g = 0$); these are the infinitesimal isometries. Symplectic vector fields preserve the symplectic form ($\mathcal{L}_X \omega = 0$); these generate Hamiltonian flows. Conformal vector fields satisfy $\mathcal{L}_X g = \lambda g$; these generate angle-preserving (but not length-preserving) transformations. The Lie derivative is the universal language of "this flow is a symmetry of that object" — and it is the same operator regardless of what kind of tensor that object is.

---

## 6. Worked Examples: Rotation, Source, Sink

To consolidate, here are three classical fields on $\mathbb{R}^2$ that you should learn to recognize on sight. Each illustrates a different qualitative behavior, and each sits inside a larger family with physical significance.

![Lie derivative: rate of change along a flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/figures/07_lie_derivative.png)


![Three classical vector fields: rotation, source, sink](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_6_field_examples.png)

**Rotation field:** $X_R = -y\,\partial_x + x\,\partial_y$. Integral curves are circles of constant radius, flow is rotation by angle $t$. Divergence $\nabla \cdot X_R = -\partial_x y + \partial_y x = 0$, curl (in 2D, $\partial_x X^y - \partial_y X^x$) equals $1 - (-1) = 2$. Generates $\mathrm{SO}(2)$. Preserves area (because divergence-free).

**Radial source:** $X_S = x\,\partial_x + y\,\partial_y$. Integral curves are rays from the origin, flow is dilation $\varphi_t(p) = e^t p$. Divergence $2$, curl $0$. Generates the multiplicative group of positive reals (in each direction). Multiplies area by $e^{2t}$ in time $t$ — so volume grows exponentially under the flow.

**Sink (negative source):** $X_K = -x\,\partial_x - y\,\partial_y$. Integral curves are rays *into* the origin, flow is contraction $\varphi_t(p) = e^{-t} p$. Divergence $-2$, curl $0$. Same one-parameter group as the source, run in reverse. Volume contracts by $e^{-2t}$.

Computing brackets:
- $[X_R, X_S]$: Let $X_R = -y\partial_x + x\partial_y$, $X_S = x\partial_x + y\partial_y$.
  $[X_R, X_S]^x = X_R^i \partial_i X_S^x - X_S^i \partial_i X_R^x = (-y)(1) + x(0) - x(0) - y(-1) = -y + y = 0$.
  $[X_R, X_S]^y = (-y)(0) + x(1) - x(1) - y(0) = 0$.
  So $[X_R, X_S] = 0$. The flows commute. Indeed, rotation and dilation commute as transformations of $\mathbb{R}^2$ — rotating then dilating gives the same point as dilating then rotating.
- $[X_S, X_K] = -2 X_S$? Wait, $X_K = -X_S$, so $[X_S, X_K] = [X_S, -X_S] = 0$ trivially. Sink and source generate the *same* one-parameter subgroup.
- More interesting: take $X = -y\partial_x + x\partial_y$ (rotation) and $Y = \partial_x$ (translation in $x$). Then $[X, Y]^x = (-y)(0) - 1\cdot 0 - 1\cdot(-1) = $ ... let me be careful: $[X, Y] = XY - YX$. $X(Y^x) = X(1) = 0$. $Y(X^x) = \partial_x(-y) = 0$. $X(Y^y) = X(0) = 0$. $Y(X^y) = \partial_x(x) = 1$. So $[X, Y]^x = 0 - 0 = 0$, $[X, Y]^y = 0 - 1 = -1$. Thus $[X, Y] = -\partial_y$. Rotation and $x$-translation do *not* commute, and the discrepancy is a $-y$-translation. Consistent with $\mathfrak{so}(2) \ltimes \mathbb{R}^2$, the Lie algebra of the Euclidean group.

This is a healthy sanity check: when geometric intuition says flows commute, the bracket should vanish. When it says they do not commute, the bracket should produce a sensible field — typically lying in the same Lie algebra you started with.

---

## 7. Phase Portraits: Geometry of ODEs

A **phase portrait** is the picture of all integral curves of a vector field, drawn in the manifold (or its phase space). Phase portraits are the qualitative theory of ODEs in geometric form. Where 19th-century mathematicians struggled to write down explicit solutions, Poincare realized that for most vector fields you cannot — but you can still describe the *qualitative behavior* of trajectories, and that is often what physics actually wants to know.

![Phase portrait of a planar dynamical system](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/07-vector-fields-flows/dg_v2_07_7_phase_portrait.png)

The structure of a phase portrait is dictated by:

1. **Fixed points** (zeros of $X$) — where flow stagnates. Linearize $X$ at a fixed point $p$ to get a matrix $A = (\partial X^i / \partial x^j)|_p$; the eigenvalues of $A$ classify the fixed point as source (both eigenvalues positive real), sink (both negative real), saddle (one of each sign), center (purely imaginary), spiral source/sink (complex with positive/negative real part).
2. **Periodic orbits** — closed integral curves. Generic systems may have isolated **limit cycles** that attract nearby trajectories. The Poincare-Bendixson theorem says that on an annulus in $\mathbb{R}^2$ with no fixed points, every trajectory tends to a limit cycle.
3. **Separatrices** — integral curves connecting fixed points (heteroclinic) or returning to the same fixed point (homoclinic); they separate qualitatively distinct regions and govern the topology of the portrait.

**Example: damped pendulum.** $\ddot\theta + \gamma\dot\theta + \sin\theta = 0$ becomes the planar system
$$X = \omega\,\partial_\theta + (-\gamma\omega - \sin\theta)\,\partial_\omega$$
on the cylinder $S^1 \times \mathbb{R}$. Fixed points at $(\theta, \omega) = (n\pi, 0)$. At $\theta = 0$ (downward equilibrium), the linearization has eigenvalues with negative real part (because of friction $\gamma > 0$) — a stable spiral sink. At $\theta = \pi$ (inverted pendulum), the linearization has one positive and one negative real eigenvalue — a saddle. The phase portrait shows trajectories spiraling into the sinks, being deflected by the saddles, and the separatrices form the boundary between trajectories that "swing back and forth" and trajectories that "go over the top."

**Example: Lotka-Volterra predator-prey.** $\dot x = ax - bxy$, $\dot y = -cy + dxy$. Fixed points at $(0, 0)$ (saddle) and $(c/d, a/b)$ (center). The portrait shows closed orbits around the interior fixed point, oscillating populations of predators and prey 90 degrees out of phase. This is a Hamiltonian system (with an unusual Hamiltonian); the closed orbits reflect a conserved quantity.

**Example: Van der Pol oscillator.** $\ddot x - \mu(1 - x^2)\dot x + x = 0$ becomes $\dot x = y$, $\dot y = \mu(1 - x^2)y - x$. For $\mu > 0$ there is one fixed point at the origin (an unstable spiral) and a unique stable limit cycle that all nonzero trajectories converge to. This is the prototypical relaxation oscillator — it appears in heart rhythms, electronic circuits, and any system with a slow buildup followed by a fast release. The geometric content (existence of limit cycle + global attraction) is invisible in the analytic formula but immediate in the phase portrait.

**Why this matters.** Engineers and physicists almost never care about *exact* solutions to ODEs — they care about qualitative behavior: stability, oscillation, escape, bifurcation. The geometric language of vector fields and flows turns ODE theory from analytic drudgery into a study of pictures. The whole field of dynamical systems lives in this picture, and modern stability theory (Lyapunov, Floquet, KAM) is essentially the deep version of the elementary observations above. Even the modern theory of bifurcations — how phase portraits change as a parameter is varied — is essentially the study of how vector fields fail to be structurally stable, and that failure is a geometric statement about the field.

---

## 8. Frobenius Integrability: When Distributions Integrate

A **distribution** $\mathcal{D}$ on $M$ is a smooth assignment $p \mapsto \mathcal{D}_p \subseteq T_p M$ of a $k$-dimensional subspace at each point. (Think of a distribution as a "field of $k$-planes," generalizing a "field of lines" which is a 1-dimensional distribution given by a non-vanishing vector field up to scale.) An **integral submanifold** is a $k$-dimensional submanifold $N \subseteq M$ with $T_q N = \mathcal{D}_q$ for all $q \in N$.

A distribution is **involutive** if it is closed under brackets: $X, Y \in \mathcal{D}$ implies $[X, Y] \in \mathcal{D}$. The **Frobenius theorem** states:

> A distribution is integrable (admits integral submanifolds through every point) if and only if it is involutive.

The "only if" direction is geometric: if you can move along $\mathcal{D}$ in two directions $X, Y$ and stay on a submanifold $N$, then the bracket flow must also keep you on $N$, so $[X, Y]$ must lie in $\mathcal{D}$. The "if" direction is the deep content — given involutivity, you can construct foliation charts in which the integral submanifolds are coordinate slices.

**A non-integrable example.** On $\mathbb{R}^3$, take the distribution spanned by
$$X = \partial_x, \qquad Y = \partial_y + x\,\partial_z.$$
These are linearly independent at every point, so $\mathcal{D}$ is a 2-plane field. Compute
$$[X, Y] = \partial_x \cdot 0 + \partial_x(x)\partial_z - 0 = \partial_z.$$
But $\partial_z$ is *not* in $\mathcal{D}$: $\mathcal{D}_p = \mathrm{span}(X_p, Y_p) = \mathrm{span}(\partial_x, \partial_y + x\partial_z)$, which equals the kernel of the 1-form $\alpha = dz - x\,dy$ — and $\alpha(\partial_z) = 1 \neq 0$. So $\mathcal{D}$ is **not** involutive, and Frobenius says no integral surfaces exist.

This is the **standard contact structure** on $\mathbb{R}^3$. Try to walk while staying in the plane field: move in $x$, then in $y$, then back in $x$, then back in $y$ — you do not return to where you started, you end up shifted in the $z$-direction. This is "holonomy" in the most elementary form, and physically it is the geometric content of the second law of thermodynamics: heat $\delta Q = TdS$ defines a non-integrable distribution on the thermodynamic state space, which is why heat is not a state function.

**Application to Lie groups.** For a Lie group $G$, the Lie algebra $\mathfrak{g}$ is identified with left-invariant vector fields. The Frobenius theorem then says: every Lie subalgebra $\mathfrak{h} \subseteq \mathfrak{g}$ corresponds to a unique connected Lie subgroup $H \subseteq G$. Subalgebras integrate to subgroups. This is Lie's third theorem in geometric form, and it is the reason group theory and infinitesimal calculus are the same subject.

**A foliation example.** On $\mathbb{R}^3$, consider $\mathcal{D} = \ker(dz)$. This is the distribution of horizontal planes; integral surfaces are the planes $z = \mathrm{const}$. It is involutive (a basis $\partial_x, \partial_y$ has $[\partial_x, \partial_y] = 0 \in \mathcal{D}$), and the integral submanifolds foliate $\mathbb{R}^3$. Frobenius confirms what is geometrically obvious.

**Codimension-1 case.** When $\mathcal{D}$ has codimension 1, it is the kernel of a 1-form $\alpha$ (locally, up to nonzero scalar multiplication). The involutivity criterion becomes $\alpha \wedge d\alpha = 0$. For our contact distribution above, $\alpha = dz - x\,dy$ has $d\alpha = -dx \wedge dy$ and $\alpha \wedge d\alpha = (dz - x\,dy)\wedge(-dx\wedge dy) = -dz \wedge dx \wedge dy \neq 0$, confirming non-integrability via the form calculus. This is a preview of how the language of forms (next article) makes Frobenius computations almost automatic.

**Why this matters.** Frobenius is the precise statement of when first-order PDE systems are solvable. Many problems in physics and engineering reduce to "find a function whose differential equals a given 1-form" or "find a submanifold tangent to a given plane field" — and Frobenius tells you exactly when these problems have solutions. Without it, you would not know when a thermodynamic potential exists, when a connection is flat, or when a control system is reachable.

---

## What's Next

We now have the kinematic toolkit: vector fields, flows, brackets, Lie derivatives, integrability. The next article moves from $TM$ to its dual $T^*M$ and studies **differential forms** — the objects you integrate. The exterior derivative $d$ unifies gradient, curl, and divergence, and Cartan's magic formula $\mathcal{L}_X = d\iota_X + \iota_X d$ ties everything in this article to the form calculus we are about to develop.

**Summary of the key ideas.**

1. A **vector field** assigns a tangent vector to every point smoothly. Equivalently, it is a derivation of $C^\infty(M)$.
2. **Integral curves** are trajectories of the field, governed by an ODE; Picard-Lindelof gives local existence and uniqueness.
3. The **flow** $\varphi_t$ assembles all integral curves into a one-parameter group of diffeomorphisms — the exponentiation of an infinitesimal symmetry.
4. The **Lie bracket** $[X, Y]$ measures the failure of two flows to commute and makes $\mathfrak{X}(M)$ a Lie algebra.
5. The **Lie derivative** $\mathcal{L}_X$ extends the bracket to all tensor fields, providing the universal "rate of change along a flow."
6. The **Frobenius theorem** characterizes integrable distributions by involutivity — the bridge between infinitesimal data (brackets) and global structure (submanifolds, subgroups).

---

*This is Part 7 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 6 — Smooth Manifolds](/en/differential-geometry/06-smooth-manifolds/)*

*Next: [Part 8 — Differential Forms](/en/differential-geometry/08-differential-forms/)*
