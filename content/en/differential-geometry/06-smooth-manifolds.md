---
title: "Smooth Manifolds: Geometry Beyond Embedded Surfaces"
date: 2021-11-11 09:00:00
tags:
  - differential-geometry
  - manifolds
  - topology
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
description: "Manifolds free geometry from ambient space — charts, atlases, and smooth structure let us do calculus on spaces that don't live in R^n."
disableNunjucks: true
series_order: 6
series_total: 12
translationKey: "differential-geometry-6"
---

Everything we have done so far — curvature, geodesics, the Gauss-Bonnet theorem — has treated surfaces as subsets of $\mathbb{R}^3$. The surface inherits its metric from the ambient Euclidean space, tangent vectors are literally arrows in $\mathbb{R}^3$, and normal vectors give us an extrinsic handle on curvature. This perspective is concrete and geometric, but it hides a deeper truth: most of the interesting geometry depends only on measurements made *within* the surface, not on how the surface sits inside $\mathbb{R}^3$.

Gauss himself glimpsed this with the Theorema Egregium — Gaussian curvature is intrinsic. Riemann took the next step: geometry can be formulated on abstract spaces that carry no ambient environment at all. The language for making this precise is the theory of **smooth manifolds**.

---

## Why Leave $\mathbb{R}^3$?

There are compelling reasons, both physical and mathematical, to develop geometry without an ambient space.

**Configuration spaces.** Consider a double pendulum: two rigid rods connected end-to-end, free to rotate in a plane. Each rod has an angular position $\theta_i \in [0, 2\pi)$. The configuration space — the space of all possible states — is the torus $T^2 = S^1 \times S^1$. This torus does not naturally sit in $\mathbb{R}^2$ (it is not even a subset); its geometry is intrinsic to the problem. Describing the dynamics of the pendulum requires calculus on the torus, not on Euclidean space.

For a more dramatic example, consider the configuration space of a rigid body free to rotate about a fixed point. The state is described by three Euler angles, and the configuration space is the rotation group $SO(3)$ — a 3-dimensional manifold that is topologically different from any subset of $\mathbb{R}^3$ (for one thing, it is compact but not simply connected: a loop that rotates by $2\pi$ about any axis is not contractible, while a loop that rotates by $4\pi$ is). Robotics, spacecraft attitude control, and molecular dynamics all require doing calculus on $SO(3)$ directly.

**Spacetime.** General relativity models spacetime as a 4-dimensional manifold $M$ equipped with a Lorentzian metric $g$. There is no 5-dimensional ambient space in which spacetime is "embedded." The Einstein field equations $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi T_{\mu\nu}$ are formulated entirely in terms of the intrinsic geometry of $M$. Any attempt to force an embedding would be physically meaningless and mathematically cumbersome — the Nash embedding theorem guarantees existence but requires absurdly high ambient dimensions (up to 231 for a generic 4-manifold).

**Abstract Riemannian manifolds.** In pure mathematics, many spaces of interest have no natural ambient space: the space of positive-definite matrices $\text{Sym}^+(n)$ with the affine-invariant metric, the Grassmannian $\text{Gr}(k, n)$ of $k$-planes in $\mathbb{R}^n$, the moduli space of Riemann surfaces. These are manifolds with rich geometric structure, but trying to embed them before studying them would be putting the cart before the horse.

The common thread: we need calculus on spaces that are "locally Euclidean" but globally may have nontrivial topology. This is precisely what manifold theory provides.

It is worth pausing to reflect on how radical this shift is. In the classical theory of curves and surfaces, "geometry" meant the study of shapes embedded in a known space. Now "geometry" means the study of abstract spaces themselves — spaces that may have been assembled from purely algebraic or topological data, with no pre-existing Euclidean room in which to draw pictures. The entire machinery of charts, atlases, and smooth structures is designed to make this leap possible: we bring $\mathbb{R}^n$ in locally (through coordinate charts) but never assume it exists globally.

---


![Charts, overlap regions, and transition maps on a manifold](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/differential-geometry/06-smooth-manifolds/dg_fig6_manifold.png)

## Topological Manifolds: Local Euclidean Structure

The starting point is purely topological — no smoothness yet.

**Definition.** A **topological manifold** of dimension $n$ is a topological space $M$ that is:
1. **Hausdorff:** distinct points have disjoint open neighborhoods.
2. **Second-countable:** the topology has a countable basis.
3. **Locally Euclidean of dimension $n$:** every point $p \in M$ has an open neighborhood $U$ that is homeomorphic to an open subset of $\mathbb{R}^n$.

Conditions (1) and (2) are technical regularity conditions that exclude pathologies (the line with two origins is locally Euclidean but not Hausdorff; the long line is Hausdorff and locally Euclidean but not second-countable). Condition (3) is the heart of the definition: locally, $M$ looks like $\mathbb{R}^n$, even though globally it may have very different topology.

A pair $(U, \varphi)$ where $U \subseteq M$ is open and $\varphi: U \to \varphi(U) \subseteq \mathbb{R}^n$ is a homeomorphism is called a **chart** (or coordinate chart). The component functions of $\varphi(p) = (x^1(p), \ldots, x^n(p))$ are **local coordinates** on $U$.

**Example: The $n$-sphere $S^n$.** Define $S^n = \{x \in \mathbb{R}^{n+1} : |x| = 1\}$. Stereographic projection from the north pole $N = (0, \ldots, 0, 1)$ gives a chart $\varphi_N: S^n \setminus \{N\} \to \mathbb{R}^n$:

$$\varphi_N(x^1, \ldots, x^{n+1}) = \frac{1}{1 - x^{n+1}}(x^1, \ldots, x^n).$$

Similarly, projection from the south pole $S = (0, \ldots, 0, -1)$ gives $\varphi_S: S^n \setminus \{S\} \to \mathbb{R}^n$. Together, $\{(S^n \setminus \{N\}, \varphi_N), (S^n \setminus \{S\}, \varphi_S)\}$ covers all of $S^n$.

**Example: Real projective space $\mathbb{R}P^n$.** Define $\mathbb{R}P^n$ as the set of lines through the origin in $\mathbb{R}^{n+1}$, equivalently the quotient $(\mathbb{R}^{n+1} \setminus \{0\}) / \sim$ where $x \sim \lambda x$ for $\lambda \neq 0$. A point is represented by homogeneous coordinates $[x^0 : x^1 : \cdots : x^n]$. For each $i$, the set $U_i = \{[x^0 : \cdots : x^n] : x^i \neq 0\}$ is open, and the map

$$\varphi_i([x^0 : \cdots : x^n]) = \left(\frac{x^0}{x^i}, \ldots, \widehat{\frac{x^i}{x^i}}, \ldots, \frac{x^n}{x^i}\right) \in \mathbb{R}^n$$

(omitting the $i$-th entry) is a homeomorphism onto $\mathbb{R}^n$. The $n+1$ charts $\{(U_i, \varphi_i)\}_{i=0}^n$ cover $\mathbb{R}P^n$. For $\mathbb{R}P^2$, this gives three charts, each covering the complement of a "line at infinity."

**Example: The torus $T^2$.** As a product $S^1 \times S^1$, the torus inherits charts from $S^1$. If $\theta$ is a local angular coordinate on the first factor and $\phi$ on the second, then $(\theta, \phi)$ gives a chart on the corresponding open set. Four such charts suffice to cover $T^2$.

**Non-example: the cone.** The cone $\{(x,y,z) : x^2 + y^2 = z^2, z \geq 0\}$ is *not* a topological manifold. At the vertex $(0,0,0)$, no neighborhood is homeomorphic to an open subset of $\mathbb{R}^2$ — removing the vertex disconnects any small neighborhood into two components, which cannot happen for an open disc. The cone has a singularity at the tip, and manifold theory deliberately excludes such points. Handling singularities requires additional machinery (orbifolds, stratified spaces, algebraic geometry).

---

## Smooth Structure: Atlases and Transition Maps

A topological manifold admits local coordinates, but we cannot yet do calculus: the notion "differentiable function on $M$" requires more structure. The key observation is that if two charts $(U_\alpha, \varphi_\alpha)$ and $(U_\beta, \varphi_\beta)$ overlap on $U_\alpha \cap U_\beta \neq \emptyset$, we can form the **transition map** (or change-of-coordinates map):

$$\varphi_\beta \circ \varphi_\alpha^{-1}: \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta).$$

This is a map between open subsets of $\mathbb{R}^n$, so we can ask whether it is smooth.

**Definition.** Two charts $(U_\alpha, \varphi_\alpha)$ and $(U_\beta, \varphi_\beta)$ are **$C^\infty$-compatible** if either $U_\alpha \cap U_\beta = \emptyset$ or the transition map $\varphi_\beta \circ \varphi_\alpha^{-1}$ (and its inverse $\varphi_\alpha \circ \varphi_\beta^{-1}$) are $C^\infty$ as maps between open subsets of $\mathbb{R}^n$.

**Definition.** A **smooth atlas** on a topological manifold $M$ is a collection $\mathcal{A} = \{(U_\alpha, \varphi_\alpha)\}_{\alpha \in A}$ of pairwise $C^\infty$-compatible charts whose domains cover $M$: $\bigcup_\alpha U_\alpha = M$.

**Definition.** A smooth atlas $\mathcal{A}$ is **maximal** if it contains every chart that is $C^\infty$-compatible with every chart in $\mathcal{A}$. A **smooth structure** on $M$ is a maximal smooth atlas. A **smooth manifold** is a topological manifold equipped with a smooth structure.

In practice, we never write down a maximal atlas. We specify a (non-maximal) smooth atlas and invoke the fact that every smooth atlas is contained in a unique maximal one (obtained by throwing in all compatible charts).

**Back to $S^n$:** the two stereographic charts are $C^\infty$-compatible. On $S^n \setminus \{N, S\}$, the transition map $\varphi_S \circ \varphi_N^{-1}$ sends $y \in \mathbb{R}^n \setminus \{0\}$ to $y / |y|^2$ — the inversion map — which is smooth (even real-analytic) away from the origin. So $S^n$ with these two charts is a smooth manifold.

**Back to $\mathbb{R}P^n$:** on $U_i \cap U_j$, the transition map sends $(y^0, \ldots, \widehat{y^i}, \ldots, y^n) \mapsto (y^0/y^j, \ldots, \widehat{y^j/y^j}, \ldots, y^n/y^j)$ (where $y^j \neq 0$ on the overlap). These are rational functions with nonvanishing denominators, hence smooth.

**Concrete computation for $\mathbb{R}P^2$.** Let us spell out the transition maps for $\mathbb{R}P^2$ completely, since this is an important example. The three charts are:

- $U_0 = \{[x^0:x^1:x^2] : x^0 \neq 0\}$ with $\varphi_0([x^0:x^1:x^2]) = (x^1/x^0, x^2/x^0) = (u_0, v_0)$.
- $U_1 = \{[x^0:x^1:x^2] : x^1 \neq 0\}$ with $\varphi_1([x^0:x^1:x^2]) = (x^0/x^1, x^2/x^1) = (u_1, v_1)$.
- $U_2 = \{[x^0:x^1:x^2] : x^2 \neq 0\}$ with $\varphi_2([x^0:x^1:x^2]) = (x^0/x^2, x^1/x^2) = (u_2, v_2)$.

The transition map $\varphi_1 \circ \varphi_0^{-1}$: a point $(u_0, v_0)$ with $u_0 \neq 0$ (needed for $x^1 \neq 0$) represents $[1:u_0:v_0]$, which in chart $U_1$ is $(1/u_0, v_0/u_0)$. So $\varphi_1 \circ \varphi_0^{-1}(u_0, v_0) = (1/u_0, v_0/u_0)$, defined on $\{u_0 \neq 0\}$. This is smooth (even real-analytic).

Each chart covers $\mathbb{R}P^2$ minus a copy of $\mathbb{R}P^1$ (a "line at infinity"). Together, they cover all of $\mathbb{R}P^2$. The topology of $\mathbb{R}P^2$ is nontrivial: it is non-orientable (it contains a Mobius band) and has fundamental group $\mathbb{Z}/2$.

**A subtlety.** Different smooth atlases on the same topological manifold can give genuinely different smooth structures. The most dramatic example: $\mathbb{R}^4$ admits uncountably many distinct smooth structures (Donaldson, 1983), while $\mathbb{R}^n$ for $n \neq 4$ admits exactly one. The 7-sphere $S^7$ admits exactly 28 distinct smooth structures (Milnor, 1956). These "exotic" smooth structures are a deep phenomenon in topology.

**Constructing manifolds in practice.** Besides the explicit examples above, there are several systematic ways to build manifolds:

- **Products.** If $M^m$ and $N^n$ are smooth manifolds, then $M \times N$ is a smooth manifold of dimension $m + n$, with charts of the form $(\varphi_\alpha \times \psi_\beta, U_\alpha \times V_\beta)$.
- **Submanifolds via the preimage theorem.** If $F: M \to N$ is a smooth map and $q \in N$ is a **regular value** (meaning $dF_p$ is surjective for every $p \in F^{-1}(q)$), then $F^{-1}(q)$ is a smooth submanifold of $M$ of dimension $\dim M - \dim N$. This is the manifold version of the implicit function theorem. For instance, $S^n = F^{-1}(1)$ where $F: \mathbb{R}^{n+1} \to \mathbb{R}$ is $F(x) = |x|^2$; since $dF_p = 2p \neq 0$ for $|p| = 1$, the value 1 is regular and $S^n$ is a smooth submanifold.
- **Quotient manifolds.** If a group $G$ acts freely and properly on a manifold $M$, the quotient $M/G$ is a smooth manifold. This is how $\mathbb{R}P^n$ arises ($\mathbb{Z}/2$ acting on $S^n$ by the antipodal map) and how the torus arises ($\mathbb{Z}^n$ acting on $\mathbb{R}^n$ by translations).

---

## Smooth Maps Between Manifolds

With smooth structure in place, we can define smooth maps.

**Definition.** Let $M$ and $N$ be smooth manifolds of dimensions $m$ and $n$ respectively. A continuous map $F: M \to N$ is **smooth** if for every $p \in M$, there exist charts $(U, \varphi)$ around $p$ and $(V, \psi)$ around $F(p)$ such that the **coordinate representation**

$$\hat{F} = \psi \circ F \circ \varphi^{-1}: \varphi(U \cap F^{-1}(V)) \to \psi(V)$$

is a smooth map between open subsets of Euclidean spaces ($\mathbb{R}^m \to \mathbb{R}^n$).

The definition is independent of the choice of charts because transition maps are smooth (the composition of smooth maps is smooth).

**Special cases:**
- A smooth map $f: M \to \mathbb{R}$ is a **smooth function** on $M$. The set of all smooth functions $C^\infty(M)$ is a real algebra (closed under addition, scalar multiplication, and pointwise multiplication).
- A **diffeomorphism** is a smooth bijection $F: M \to N$ whose inverse $F^{-1}: N \to M$ is also smooth. Diffeomorphic manifolds are "the same" from the standpoint of smooth geometry.
- An **immersion** at $p$ is a smooth map $F: M \to N$ whose differential $dF_p$ is injective ($\text{rank} = m$). An **embedding** is an injective immersion that is also a homeomorphism onto its image. A **submersion** at $p$ is a smooth map whose differential $dF_p$ is surjective ($\text{rank} = n$).

**Example.** The inclusion $\iota: S^2 \hookrightarrow \mathbb{R}^3$ is a smooth embedding. The projection $\pi: \mathbb{R}^3 \setminus \{0\} \to S^2$ given by $\pi(x) = x/|x|$ is a smooth submersion. The map $\gamma: \mathbb{R} \to T^2$ defined by $\gamma(t) = (e^{2\pi i t}, e^{2\pi i \alpha t})$ for irrational $\alpha$ is a smooth immersion whose image is dense in $T^2$ — a classic example of an immersion that is not an embedding.

**Example: smooth functions on $\mathbb{R}P^2$.** A smooth function $f: \mathbb{R}P^2 \to \mathbb{R}$ is equivalently a smooth function $\tilde{f}: S^2 \to \mathbb{R}$ satisfying $\tilde{f}(-p) = \tilde{f}(p)$ for all $p \in S^2$. In spherical coordinates $(\theta, \phi)$, the antipodal map sends $(\theta, \phi) \mapsto (\pi - \theta, \phi + \pi)$, so $\tilde{f}(\theta, \phi) = \tilde{f}(\pi - \theta, \phi + \pi)$. Only the even spherical harmonics $Y_\ell^m$ with $\ell$ even satisfy this condition. This shows that $C^\infty(\mathbb{R}P^2)$ is a strictly smaller algebra than $C^\infty(S^2)$: the topology of the manifold constrains which functions can live on it.

**The partition of unity.** A powerful technical tool on smooth manifolds is the existence of **partitions of unity** subordinate to any open cover. Given an open cover $\{U_\alpha\}$ of $M$, there exist smooth functions $\rho_\alpha: M \to [0, 1]$ with $\text{supp}(\rho_\alpha) \subseteq U_\alpha$ and $\sum_\alpha \rho_\alpha = 1$ (the sum being locally finite). Partitions of unity allow us to "glue" local constructions into global ones — they are the reason smooth manifolds are much more flexible than analytic or algebraic varieties. For instance, Riemannian metrics exist on every smooth manifold precisely because we can build one locally (using any chart) and then patch the local metrics together using a partition of unity.

---

## Tangent Vectors as Derivations

In the classical setting, a tangent vector to a surface $S \subseteq \mathbb{R}^3$ at a point $p$ is a velocity vector $\gamma'(0)$ of some curve $\gamma$ on $S$ with $\gamma(0) = p$. This definition relies on the ambient space $\mathbb{R}^3$. On an abstract manifold, there is no ambient space, so we need an intrinsic definition.

The key insight: a tangent vector $v$ at $p$ determines a way to differentiate smooth functions at $p$. If $\gamma$ is a curve with $\gamma(0) = p$ and $\gamma'(0) = v$, then for any $f \in C^\infty(M)$:

$$v(f) = \frac{d}{dt}\bigg|_{t=0} f(\gamma(t)).$$

This operation $v: C^\infty(M) \to \mathbb{R}$ satisfies two properties: it is $\mathbb{R}$-linear and satisfies the **Leibniz rule** $v(fg) = f(p) \cdot v(g) + g(p) \cdot v(f)$. We take these properties as the definition.

**Definition.** A **derivation** at $p \in M$ is an $\mathbb{R}$-linear map $v: C^\infty(M) \to \mathbb{R}$ satisfying the Leibniz rule:

$$v(fg) = f(p) \cdot v(g) + g(p) \cdot v(f) \quad \text{for all } f, g \in C^\infty(M).$$

The **tangent space** $T_pM$ is the set of all derivations at $p$.

This is a vector space: $(v + w)(f) = v(f) + w(f)$ and $(cv)(f) = c \cdot v(f)$. Its dimension equals $\dim M$.

**Why derivations?** This definition may seem abstract compared to the geometric picture of arrows tangent to curves. The advantage is that it is purely algebraic — it refers only to the algebra $C^\infty(M)$, not to any ambient space or to curves. It generalizes effortlessly to infinite-dimensional settings (Banach and Frechet manifolds) and to algebraic geometry (where the Zariski tangent space is defined analogously for commutative rings). The equivalence between the three approaches — velocity vectors of curves, equivalence classes of curves, and derivations — is a theorem that works specifically because we are in the $C^\infty$ category.

**Proof that derivations kill constants.** If $v$ is a derivation at $p$ and $c \in \mathbb{R}$ is a constant function, then $v(c) = v(c \cdot 1) = c \cdot v(1)$. But also $v(1) = v(1 \cdot 1) = 1 \cdot v(1) + 1 \cdot v(1) = 2v(1)$, so $v(1) = 0$, and thus $v(c) = 0$. Derivations annihilate constant functions — they detect only the infinitesimal variation of $f$ at $p$.

**Basis from coordinates.** Given a chart $(U, \varphi)$ with $\varphi(p) = (a^1, \ldots, a^n)$, define partial derivative operators at $p$:

$$\frac{\partial}{\partial x^i}\bigg|_p (f) = \frac{\partial (f \circ \varphi^{-1})}{\partial r^i}\bigg|_{\varphi(p)}$$

where $r^i$ are standard coordinates on $\mathbb{R}^n$. These $n$ derivations form a basis for $T_pM$. Every $v \in T_pM$ can be written uniquely as:

$$v = v^i \frac{\partial}{\partial x^i}\bigg|_p$$

where $v^i = v(x^i)$ and we use the Einstein summation convention.

**The differential of a smooth map.** Given $F: M \to N$, the **differential** (or pushforward) at $p$ is the linear map $dF_p: T_pM \to T_{F(p)}N$ defined by:

$$(dF_p(v))(g) = v(g \circ F) \quad \text{for all } g \in C^\infty(N).$$

In local coordinates, if $\varphi = (x^1, \ldots, x^m)$ on $M$ and $\psi = (y^1, \ldots, y^n)$ on $N$, then $dF_p$ is represented by the **Jacobian matrix**:

$$\left(\frac{\partial \hat{F}^j}{\partial x^i}\right)$$

where $\hat{F} = \psi \circ F \circ \varphi^{-1}$. This connects the abstract definition back to familiar multivariable calculus.

**Chain rule on manifolds.** Given smooth maps $F: M \to N$ and $G: N \to P$, the chain rule takes the elegant form:

$$d(G \circ F)_p = dG_{F(p)} \circ dF_p.$$

In matrix terms, the Jacobian of the composition is the product of the Jacobians. This is the functoriality of the tangent space construction: $T$ sends manifolds to vector spaces and smooth maps to linear maps, respecting composition and identities. In category theory language, $T_p$ is a functor from the category of pointed smooth manifolds to the category of vector spaces.

**The tangent bundle.** The collection of all tangent spaces assembles into the **tangent bundle** $TM = \bigsqcup_{p \in M} T_pM$, which is itself a smooth manifold of dimension $2n$ (where $n = \dim M$). A point of $TM$ is a pair $(p, v)$ with $p \in M$ and $v \in T_pM$. The projection $\pi: TM \to M$ sending $(p, v) \mapsto p$ is a smooth map, and the fiber $\pi^{-1}(p) = T_pM$ is a vector space. This makes $TM$ a **vector bundle** over $M$ — the prototype of all the bundles (cotangent, tensor, exterior power) that we will encounter throughout the rest of this series.

---

## Examples: Tangent Spaces of Spheres, Lie Groups as Manifolds

**Tangent space of $S^2$ at the north pole.** Using stereographic coordinates from the south pole, $\varphi_S: S^2 \setminus \{S\} \to \mathbb{R}^2$, the north pole $N = (0,0,1)$ maps to the origin. The tangent space $T_N S^2$ is 2-dimensional, spanned by $\frac{\partial}{\partial u}\big|_N$ and $\frac{\partial}{\partial v}\big|_N$ where $(u,v) = \varphi_S$. As a subspace of $\mathbb{R}^3$ (via the embedding), this corresponds to the horizontal plane $\{(x,y,0) : x,y \in \mathbb{R}\}$ — the plane tangent to the sphere at the north pole, as expected.

More generally, for any $p \in S^n \subseteq \mathbb{R}^{n+1}$, the tangent space $T_p S^n$ (viewed extrinsically) is the orthogonal complement of $p$ in $\mathbb{R}^{n+1}$:

$$T_p S^n \cong \{v \in \mathbb{R}^{n+1} : \langle v, p \rangle = 0\}.$$

This is an $n$-dimensional subspace, consistent with $\dim S^n = n$.

**Lie groups as smooth manifolds.** A **Lie group** is a group $G$ that is simultaneously a smooth manifold, with the property that the group operations (multiplication $G \times G \to G$ and inversion $G \to G$) are smooth maps. Lie groups are among the most important examples of manifolds in both mathematics and physics.

**Example: The general linear group $GL(n, \mathbb{R})$.** This is the group of invertible $n \times n$ real matrices. As a subset of $\mathbb{R}^{n^2}$ (identifying a matrix with its $n^2$ entries), it is the open set $\{\det A \neq 0\}$ — open because the determinant is a continuous function. So $GL(n, \mathbb{R})$ is an $n^2$-dimensional smooth manifold (a single chart, the identity, suffices). Its tangent space at the identity is the Lie algebra $\mathfrak{gl}(n, \mathbb{R})$ — the space of all $n \times n$ real matrices with the commutator bracket $[A, B] = AB - BA$.

**Example: The orthogonal group $O(n)$.** This is the subgroup of $GL(n, \mathbb{R})$ satisfying $A^T A = I$. By the preimage theorem (a consequence of the implicit function theorem on manifolds), $O(n)$ is a smooth submanifold of dimension $n(n-1)/2$. Its tangent space at the identity is the space of skew-symmetric matrices $\mathfrak{o}(n) = \{X \in \mathbb{R}^{n \times n} : X^T + X = 0\}$.

To see why: if $A(t)$ is a curve in $O(n)$ with $A(0) = I$, then $A(t)^T A(t) = I$ for all $t$. Differentiating at $t = 0$: $A'(0)^T + A'(0) = 0$, so $A'(0)$ is skew-symmetric.

**Example: The special unitary group $SU(2)$.** This is the group of $2 \times 2$ unitary matrices with determinant 1. Every element can be written as

$$\begin{pmatrix} \alpha & -\bar{\beta} \\ \beta & \bar{\alpha} \end{pmatrix}, \quad |\alpha|^2 + |\beta|^2 = 1$$

with $\alpha, \beta \in \mathbb{C}$. Writing $\alpha = a + bi$ and $\beta = c + di$, the condition $a^2 + b^2 + c^2 + d^2 = 1$ shows that $SU(2) \cong S^3$ as a smooth manifold. This is a 3-dimensional Lie group — the simplest simply-connected non-abelian Lie group, fundamental in quantum mechanics as the double cover of $SO(3)$.

**The tangent space of a Lie group at the identity is special.** It carries a natural bracket operation $[\cdot, \cdot]$ (the Lie bracket) inherited from the group structure, making it a **Lie algebra**. The exponential map $\exp: \mathfrak{g} \to G$ sends Lie algebra elements to group elements and locally parametrizes the group near the identity. For matrix Lie groups, this is literally the matrix exponential $\exp(X) = \sum_{k=0}^\infty X^k / k!$.

The interplay between the global structure of the Lie group and the local (linear) structure of its Lie algebra is one of the most powerful ideas in modern mathematics and physics.

**How many manifolds are there?** The classification of smooth manifolds is an incredibly rich subject. In dimension 1, the only connected manifolds are $\mathbb{R}$ and $S^1$. In dimension 2, the classification theorem for surfaces says every compact connected orientable surface is a sphere with $g$ handles (genus $g$); adding non-orientability gives the projective plane, Klein bottle, etc. In dimension 3, the Poincare conjecture (proved by Perelman in 2003) says the only simply connected compact 3-manifold is $S^3$; the full classification uses Thurston's geometrization. In dimensions $\geq 5$, surgery theory provides systematic tools. Dimension 4 remains the most mysterious: gauge theory (Donaldson invariants, Seiberg-Witten invariants) reveals phenomena that have no analog in other dimensions.

---

## What's Next

We now have the stage: smooth manifolds, smooth maps, tangent spaces. But geometry requires more structure. We need to talk about vector fields — smooth assignments of tangent vectors — and the flows they generate. The next article develops **vector fields, integral curves, and the Lie bracket**, the machinery that captures how infinitesimal symmetries act on a manifold. This sets the stage for differential forms, integration on manifolds, and eventually the full apparatus of Riemannian geometry in the intrinsic setting.

**Summary of the key ideas.** Let us recapitulate the conceptual progression:

1. A **topological manifold** is a space that is locally homeomorphic to $\mathbb{R}^n$ — it has local coordinates, but no notion of smoothness.
2. A **smooth structure** (a maximal atlas of $C^\infty$-compatible charts) lets us define smooth functions, smooth maps, and do calculus.
3. **Tangent vectors** are derivations — they differentiate functions, and their definition is intrinsic (no ambient space needed).
4. The **differential** $dF_p$ of a smooth map linearizes the map at a point, sending tangent vectors to tangent vectors.
5. The **tangent bundle** $TM$ assembles all tangent spaces into a single manifold of double the dimension.

These five ideas form the foundation on which all of differential geometry rests. Every subsequent construction — vector fields, differential forms, connections, curvature — is built from these building blocks.

---

*This is Part 6 of the [Differential Geometry](/en/series/differential-geometry/) series (12 articles).*

*Previous: [Part 5 — The Gauss-Bonnet Theorem](/en/differential-geometry/05-gauss-bonnet/)*

*Next: [Part 7 — Vector Fields and Flows](/en/differential-geometry/07-vector-fields-flows/)*
