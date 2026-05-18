---
title: "Group Actions: How Groups Move Things Around"
date: 2021-09-03 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 12
translationKey: "abstract-algebra-2"
description: "We formalize how groups act on sets, prove the orbit-stabilizer theorem, derive Burnside's lemma, and count necklaces."
---

![Counting orbits with Burnside](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_orbit_counting.png)

## From Abstract Groups to Concrete Actions

Mental picture before any definitions: a group acting on a set is a collection of moves, and the set is the playground those moves permute. Pick up an object, do a move from the group, set it down. The orbit of an object is everywhere you can take it. The stabilizer is every move that puts it back exactly where it started.

In the previous article we defined groups and proved Lagrange's theorem. The groups themselves --- $\mathbb{Z}/n\mathbb{Z}$, $S_n$, $D_n$ --- were interesting, but we studied them in isolation. The real power of group theory emerges when a group does something: when it permutes the elements of a set, rotates the vertices of a polygon, or rearranges the colors of a necklace.

This is the idea of a *group action*. Historically, groups arose precisely as collections of symmetries acting on geometric objects. Galois studied permutation groups acting on roots of polynomials. Klein's Erlangen program (1872) proposed defining geometry itself as the study of invariants under a group action. In modern algebra, the action viewpoint is indispensable: it converts abstract group-theoretic questions into concrete combinatorial ones, and vice versa.

The central results of this article are the orbit-stabilizer theorem and Burnside's lemma. The orbit-stabilizer theorem relates the size of a group to the sizes of orbits and stabilizers under an action. Burnside's lemma uses this to count distinct objects up to symmetry --- the kind of problem that arises whenever we ask "how many essentially different colorings are there?" The answer is always a weighted average of fixed-point counts, and the proof is a clean application of the orbit-stabilizer machinery.

We also introduce conjugation as a group action, which leads to the class equation --- a tool that will be crucial in later articles when we prove the Sylow theorems.

![A group action G x X to X as an arrow diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_4_action_arrow.png)

## Formal Definition and Examples

Mental picture: a group action is a homomorphism from $G$ into the symmetric group of $X$. Each group element gets a personal permutation of the set, and composing group elements is the same as composing those permutations.

**Definition.** Let $G$ be a group and $X$ a set. A *(left) group action* of $G$ on $X$ is a function $G \times X \to X$, written $(g, x) \mapsto g \cdot x$, satisfying:

1. **Identity.** $e \cdot x = x$ for all $x \in X$.
2. **Compatibility.** $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$ and $x \in X$.

We say $G$ *acts on* $X$, or that $X$ is a *$G$-set*.

**Equivalent formulation.** A group action is the same as a group homomorphism $\varphi : G \to \text{Sym}(X)$, where $\text{Sym}(X)$ is the group of all bijections $X \to X$. Given an action, define $\varphi(g)(x) = g \cdot x$; the compatibility axiom ensures $\varphi(gh) = \varphi(g) \circ \varphi(h)$, and the identity axiom ensures $\varphi(e) = \text{id}_X$. Conversely, any such homomorphism defines an action.

This reformulation is important: the *kernel* of $\varphi$ is $\{g \in G : g \cdot x = x \text{ for all } x \in X\}$, the set of group elements that act trivially on every point of $X$.

**Why this matters.** The homomorphism reformulation is the bridge between abstract groups and concrete permutations. Cayley's theorem (every group embeds into some $S_n$) is just the observation that the regular action is faithful. This is what makes finite group theory amenable to brute-force computer enumeration.

**Definition.** An action is *faithful* (or *effective*) if $\ker\varphi = \{e\}$, i.e., only the identity fixes every element of $X$. Equivalently, the homomorphism $\varphi : G \to \text{Sym}(X)$ is injective, so $G$ embeds into $\text{Sym}(X)$.

![Regular action: Cayley's theorem embedding](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_regular_action.png)

**Cayley's theorem** follows immediately: every group acts on itself by left multiplication, and this action is faithful. Hence every group is isomorphic to a subgroup of some symmetric group. Specifically, if $|G| = n$, then $G$ embeds into $S_n$.

**Right actions.** A *right action* is a map $X \times G \to X$, written $(x, g) \mapsto x \cdot g$, satisfying $x \cdot e = x$ and $x \cdot (gh) = (x \cdot g) \cdot h$. Any right action becomes a left action via $g \cdot x = x \cdot g^{-1}$. We focus on left actions throughout.

**Example 1: $S_n$ acting on $\{1, \ldots, n\}$.** The natural action: $\sigma \cdot i = \sigma(i)$. This is faithful (distinct permutations move at least one element differently).

**Example 2: $D_n$ acting on the vertices of a regular $n$-gon.** Label the vertices $1, \ldots, n$. Each symmetry (rotation or reflection) permutes the vertices. This defines a faithful action $D_n \to S_n$.

**Example 3: $G$ acting on itself by left multiplication.** Define $g \cdot x = gx$ for $g, x \in G$. The axioms are satisfied: $e \cdot x = ex = x$ and $(gh) \cdot x = (gh)x = g(hx) = g \cdot (h \cdot x)$. This is the *left regular action*. It is always faithful.

**Example 4: $G$ acting on itself by conjugation.** Define $g \cdot x = gxg^{-1}$. Check: $e \cdot x = exe^{-1} = x$, and $(gh) \cdot x = (gh)x(gh)^{-1} = g(hxh^{-1})g^{-1} = g \cdot (h \cdot x)$. This action is typically not faithful: $g$ acts trivially on all of $G$ if and only if $gxg^{-1} = x$ for all $x$, i.e., $g \in Z(G)$ (the center of $G$). The kernel is precisely $Z(G)$.

**Example 5: $G$ acting on its subsets by conjugation.** For a subset $S \subseteq G$, define $g \cdot S = gSg^{-1} = \{gsg^{-1} : s \in S\}$. If $H \leq G$, then $g \cdot H = gHg^{-1}$, which is again a subgroup. This gives an action of $G$ on the set of all subgroups of $G$.

**Example 6: $G$ acting on left cosets of $H$.** Let $G/H = \{gH : g \in G\}$ denote the set of left cosets. Define $g \cdot (aH) = (ga)H$. This is well-defined: if $aH = bH$, then $b^{-1}a \in H$, so $(gb)^{-1}(ga) = b^{-1}a \in H$, hence $(ga)H = (gb)H$. The kernel of this action is $\bigcap_{g \in G} gHg^{-1}$, the largest normal subgroup of $G$ contained in $H$.

**Numerical example: $\mathbb{Z}/4\mathbb{Z}$ acting on the four corners of a square.** Identify the corners with $\{0, 1, 2, 3\}$ and let $k \cdot i = (i + k) \bmod 4$. Then $1$ rotates each corner one step around, $2$ takes opposite corners to each other, $3$ rotates the other way. Faithful, transitive, and the simplest possible cyclic action.

**Numerical example: $S_3$ acting on the six elements of $S_3$ by left multiplication.** This is Cayley's embedding $S_3 \hookrightarrow S_6$. Each element of $S_3$ becomes a permutation of $S_3$'s six elements. For example, the element $(1\ 2) \in S_3$ acts on the list $\{e, (1\ 2), (1\ 3), (2\ 3), (1\ 2\ 3), (1\ 3\ 2)\}$ by left multiplication, sending the first to the second, the second to the first, and so on. The result is a permutation of six labels, an element of $S_6$. Doing this for all six elements of $S_3$ gives an injective homomorphism $S_3 \to S_6$. Cayley's theorem at work, in the smallest non-abelian case.

**Counting actions versus identifying actions.** A common confusion: writing down two actions of the same group on the same set may give isomorphic actions (in the sense that there is a $G$-equivariant bijection between them) or genuinely different ones. The trivial action ($g \cdot x = x$ for all $g, x$) is always available, and is typically not the most informative. When one says "the action of $S_n$ on $\{1, \ldots, n\}$" one almost always means the natural permutation action.

**Transitive actions and subgroups, a structural correspondence.** Up to equivariant isomorphism, transitive actions of $G$ are in bijection with subgroups of $G$ up to conjugacy. The transitive action of $G$ on $G/H$ corresponds to the conjugacy class of $H$. This is one of the cleanest structural facts in elementary group theory and is the bridge from "action thinking" to "subgroup thinking." We will develop the bijection more carefully when we discuss normal subgroups in the next article.

## Orbits, Stabilizers, and the Orbit-Stabilizer Theorem

Mental picture: the orbit of a point is its trajectory under the group --- "everywhere it can travel." The stabilizer is the set of moves that hold the point fixed. These two notions are inversely proportional in a precise sense.

![Orbit-stabilizer theorem: |G| = |Orb(x)| * |Stab(x)|](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_orbit_stabilizer.png)

**Definition.** Let $G$ act on $X$ and $x \in X$.

- The *orbit* of $x$ is $\text{Orb}(x) = G \cdot x = \{g \cdot x : g \in G\}$.
- The *stabilizer* of $x$ is $\text{Stab}(x) = G_x = \{g \in G : g \cdot x = x\}$.

The orbit is a subset of $X$; the stabilizer is a subset of $G$.

**Proposition.** $\text{Stab}(x)$ is a subgroup of $G$.

*Proof.* $e \cdot x = x$, so $e \in \text{Stab}(x)$. If $g, h \in \text{Stab}(x)$, then $h \cdot x = x$ implies (applying $h^{-1}$) $h^{-1} \cdot x = x$, hence $(gh^{-1}) \cdot x = g \cdot (h^{-1} \cdot x) = g \cdot x = x$. By the subgroup criterion, $\text{Stab}(x) \leq G$. $\square$

![How orbits partition the set X into disjoint pieces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_5_orbit_partition.png)

**The orbits partition $X$.** Define $x \sim y$ if $y = g \cdot x$ for some $g \in G$. This is an equivalence relation: reflexive ($x = e \cdot x$), symmetric (if $y = g \cdot x$ then $x = g^{-1} \cdot y$), transitive (if $y = g \cdot x$ and $z = h \cdot y$ then $z = (hg) \cdot x$). The equivalence classes are exactly the orbits. When there is a single orbit, we say the action is *transitive*.

**Theorem (Orbit-Stabilizer).** Let $G$ be a finite group acting on a set $X$, and let $x \in X$. Then

$$|G| = |\text{Orb}(x)| \cdot |\text{Stab}(x)|$$

or equivalently,

$$|\text{Orb}(x)| = [G : \text{Stab}(x)] = \frac{|G|}{|\text{Stab}(x)|}$$

*Proof.* We construct a bijection $\Phi : G/\text{Stab}(x) \to \text{Orb}(x)$ by $\Phi(g \cdot \text{Stab}(x)) = g \cdot x$.

**Well-defined:** If $g \cdot \text{Stab}(x) = h \cdot \text{Stab}(x)$, then $h^{-1}g \in \text{Stab}(x)$, so $g \cdot x = h \cdot x$.

**Surjective:** Every element of $\text{Orb}(x)$ has the form $g \cdot x$.

**Injective:** If $g \cdot x = h \cdot x$, then $h^{-1}g \in \text{Stab}(x)$, hence $g \cdot \text{Stab}(x) = h \cdot \text{Stab}(x)$.

So $|\text{Orb}(x)| = [G : \text{Stab}(x)]$, and multiplying by $|\text{Stab}(x)|$ gives $|G| = |\text{Orb}(x)| \cdot |\text{Stab}(x)|$ (using Lagrange). $\square$

![Orbit-Stabilizer theorem: orbit size times stabilizer size equals group order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_1_orbit_stabilizer.png)

**Why this matters.** Orbit-Stabilizer is the source of nearly every counting argument in finite group theory. It shows that orbit sizes are constrained to divisors of $|G|$, which is a powerful arithmetic limitation that rules out impossible orbit decompositions before any computation is done.

A historical note: the orbit-stabilizer theorem is often attributed to Lagrange in spirit, although it was not formulated in modern language until the late 19th century. Lagrange's original theorem on subgroup orders is essentially the special case of orbit-stabilizer applied to the regular action of $G$ on itself. Many "named theorems" in group theory are special cases of orbit-stabilizer in disguise; recognizing this is a major step in understanding the subject.

**Worked Example: $D_4$ acting on the vertices of a square.** Label the vertices $\{1, 2, 3, 4\}$. The group $D_4$ (order $8$) acts on this set. Consider $x = 1$.

The orbit: every vertex can be reached from vertex $1$ by some rotation, so $\text{Orb}(1) = \{1, 2, 3, 4\}$, $|\text{Orb}(1)| = 4$.

The stabilizer: identity fixes $1$, and the reflection through the diagonal containing vertex $1$ also fixes $1$. So $\text{Stab}(1) = \{e, s\}$ for that diagonal reflection.

Check: $|D_4| = 8 = 4 \times 2 = |\text{Orb}(1)| \times |\text{Stab}(1)|$. Confirmed.

**Worked Example: $S_4$ acting on $\{1,2,3,4\}$ --- stabilizer of $1$.** $\text{Stab}(1) = \{\sigma \in S_4 : \sigma(1) = 1\}$ permutes $\{2,3,4\}$, so $\text{Stab}(1) \cong S_3$ and $|\text{Stab}(1)| = 6$. Orbit: all of $\{1,2,3,4\}$. Check: $24 = 4 \times 6$. $\checkmark$

**Worked Example: rotational symmetries of the cube acting on vertices.** A cube has $8$ vertices, and its rotation group has $24$ elements. The action on vertices is transitive (any vertex can be rotated to any other), so $|\text{Orb}(v)| = 8$. By orbit-stabilizer, $|\text{Stab}(v)| = 24/8 = 3$. Indeed, the three rotations fixing a chosen vertex are the rotations by $0$, $120$, $240$ degrees about the body diagonal through that vertex.

**Worked Example: $S_5$ acting on $2$-element subsets.** Let $X = \binom{\{1,\ldots,5\}}{2}$, the set of $2$-element subsets, $|X| = 10$. $S_5$ acts on $X$ by $\sigma \cdot \{a,b\} = \{\sigma(a), \sigma(b)\}$. The action is transitive (any 2-subset can be mapped to any other), so $|\text{Orb}| = 10$. By orbit-stabilizer, the stabilizer of $\{1, 2\}$ has order $|S_5|/10 = 120/10 = 12$. Concretely, the stabilizer consists of permutations that either fix both $1, 2$ (giving $S_3$ on the remaining three, $6$ elements) or swap $1$ and $2$ (giving another $6$). Total: $12$, matching the prediction.

**Counting principle.** Whenever a finite group acts transitively on a finite set $X$, the size $|X|$ must divide $|G|$. This is one of those quietly powerful constraints that immediately rule out many imagined actions before any computation. For instance, $S_5$ cannot act transitively on a set of size $7$, because $7 \nmid 120$.

**Worked Example: $A_4$ acting on cosets.** $A_4$ has order $12$ and contains a unique normal subgroup of order $4$ (the Klein four-group $V_4$), but no subgroup of order $6$. Suppose $A_4$ acted transitively on a set of size $6$. By orbit-stabilizer the stabilizer has order $12/6 = 2$, and the action gives a homomorphism $A_4 \to S_6$ whose kernel is a normal subgroup of $A_4$ contained in the stabilizer. The only normal subgroups of $A_4$ are $\{e\}, V_4, A_4$. None has order $\le 2$ except $\{e\}$, so the action would have to be faithful. But then $A_4 \hookrightarrow S_6$ has image of order $12$, with stabilizer of size $2$ at each of the $6$ points. A direct check shows this is incompatible with the cycle structure of $A_4$. So $A_4$ has no transitive action on a $6$-set --- equivalent to saying it has no subgroup of order $6$, as we knew.

This is a paradigmatic application: the existence of a transitive action of $G$ on an $n$-set is equivalent to the existence of a subgroup of index $n$ in $G$.

**Worked example: orbit decomposition of $S_4$ on $2$-subsets.** Let $X = \binom{\{1,2,3,4\}}{2}$, $|X| = 6$. $S_4$ acts on $X$ transitively. Stabilizer of $\{1, 2\}$ consists of permutations preserving the partition $\{1,2\}|\{3,4\}$: either fixing both blocks (a $\mathbb{Z}/2 \times \mathbb{Z}/2$, four elements) or swapping them (also four elements, but with constraint). Actually the stabilizer of $\{1, 2\}$ as a set consists of $\sigma \in S_4$ with $\sigma(\{1,2\}) = \{1,2\}$, namely $\sigma$ permutes $\{1,2\}$ and permutes $\{3,4\}$. Count: $2 \cdot 2 = 4$. By orbit-stabilizer $|X| = 24/4 = 6$. $\checkmark$.

![The 24 rotational symmetries of the cube classified by orbit type](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_2_cube_rotations.png)

## Fixed Points and Burnside's Lemma

Mental picture: to count things up to symmetry, average the number of things each symmetry leaves alone. This is the entire content of Burnside's lemma, modulo a couple of double-counting tricks.

![Burnside's lemma: counting distinct colorings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_burnside.png)

For a group element $g \in G$ acting on $X$, define the *fixed-point set* of $g$:

$$X^g = \text{Fix}(g) = \{x \in X : g \cdot x = x\}$$

This is the set of elements of $X$ left unchanged by $g$.

![Fixed points of group actions on a set](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_fixed_points.png)

**Theorem (Burnside's Lemma).** Let $G$ be a finite group acting on a finite set $X$. The number of distinct orbits is

$$|\text{Orbits}| = \frac{1}{|G|} \sum_{g \in G} |X^g|$$

That is, the number of orbits equals the average number of fixed points.

*Proof.* Count the set $S = \{(g, x) \in G \times X : g \cdot x = x\}$ in two ways.

**By $x$:** $|S| = \sum_{x \in X} |\text{Stab}(x)|$.

**By $g$:** $|S| = \sum_{g \in G} |X^g|$.

By orbit-stabilizer, $|\text{Stab}(x)| = |G|/|\text{Orb}(x)|$. Grouping by orbits $O_1, \ldots, O_k$:

$$\sum_{x \in X} \frac{1}{|\text{Orb}(x)|} = \sum_{i=1}^k \sum_{x \in O_i} \frac{1}{|O_i|} = \sum_{i=1}^k 1 = k$$

So $|S| = |G| \cdot k$, giving $k = \frac{1}{|G|} \sum_{g \in G} |X^g|$. $\square$

The lemma is often attributed to Burnside, though it was known earlier to Cauchy and Frobenius. Some authors call it the Cauchy-Frobenius lemma.

**Why this matters.** Burnside's lemma turns "count distinct objects up to symmetry" into a finite, mechanical computation: list group elements, count fixed points of each, take the average. It scales easily from necklaces with four beads to the rotation classes of a Rubik's cube.

Burnside's lemma is also one of the rare results in pure algebra that has direct industrial applications: chemists use it to count distinct isomers of molecules, computer scientists use it to count graph isomorphism classes, and combinatorialists use it to count tilings, codes, and designs. Whenever a problem asks "how many objects up to rotation/reflection/relabeling," Burnside is the first tool to reach for.

![Burnside's lemma counting distinct 2-color necklaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_3_burnside_necklace.png)

**Worked Example: Counting necklaces with 4 beads and 2 colors.**

We want to count distinct necklaces of $4$ beads, each colored black or white, where two necklaces are "the same" if one rotates into the other. The symmetry group is $C_4 = \{e, r, r^2, r^3\}$.

The set $X$ of all colorings has $|X| = 2^4 = 16$. Compute fixed points:

- **$g = e$:** all $16$ colorings.
- **$g = r$ (90°):** all four beads must match. $|X^r| = 2$.
- **$g = r^2$ (180°):** bead $1$ = bead $3$, bead $2$ = bead $4$. $|X^{r^2}| = 4$.
- **$g = r^3$ (270°):** all four beads must match. $|X^{r^3}| = 2$.

Burnside: $|\text{Orbits}| = (16 + 2 + 4 + 2)/4 = 24/4 = 6$.

The six necklaces: BBBB, BBBW, BBWW, BWBW, BWWW, WWWW. Verified by hand.

**Extended example: necklaces under $D_4$.** Including reflections (group of order $8$). Reflections come in two types:

- **Through opposite vertices ($2$ reflections):** fixes 2 beads, swaps the other 2. $|X^s| = 2^3 = 8$.
- **Through opposite edges ($2$ reflections):** swaps two pairs. $|X^s| = 2^2 = 4$.

Total: $(16 + 2 + 4 + 2 + 8 + 8 + 4 + 4)/8 = 48/8 = 6$. Same answer here, by coincidence.

**Worked Example: 4-color necklaces of length 4 under $C_4$.** Use the same fixed-point structure but with $4$ colors instead of $2$:

$$N = \frac{1}{4}(4^4 + 4 + 4^2 + 4) = \frac{1}{4}(256 + 4 + 16 + 4) = \frac{280}{4} = 70$$

So there are $70$ distinct $4$-color length-$4$ necklaces under rotation.

**Worked Example: 3-color colorings of an equilateral triangle's vertices under $D_3$.** $|X| = 3^3 = 27$. The group $D_3$ has order $6$:

- Identity: $|X^e| = 27$.
- Rotation by $120°$ or $240°$ (two elements): all three vertices must match. $|X^g| = 3$ each, total $6$.
- Three reflections: each fixes one vertex and swaps the other two; that pair must match. $|X^s| = 3 \cdot 3 = 9$ each, total $27$.

Burnside: $(27 + 6 + 27)/6 = 60/6 = 10$. There are $10$ inequivalent $3$-colorings.

Sanity check by enumeration: $3$ monochromatic (RRR, GGG, BBB), $6$ "two-and-one" types up to rotation but reflection collapses them further, and $1$ all-different (R, G, B) coloring counted once. Direct count gives $3 + 6 + 1 = 10$. Matches.

![Polya enumeration: 4-colorings of a square modulo rotation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_7_polya_square.png)

## Conjugation and the Class Equation

Mental picture: conjugation is the group's way of looking at itself in a mirror. Two elements are conjugate if some change of coordinates (the conjugating element) turns one into the other. The conjugacy classes are the "structural species" of the group.

![Conjugacy classes partition the group](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_conjugacy_classes.png)

The action of a group on itself by conjugation: $g \cdot x = gxg^{-1}$. The orbits are *conjugacy classes*, and the stabilizer of $x$ is the *centralizer* $C_G(x) = \{g \in G : gx = xg\}$.

By orbit-stabilizer, the size of the conjugacy class of $x$ is $[G : C_G(x)]$.

A useful sanity check: in an abelian group, every element commutes with everything, so $C_G(x) = G$ for all $x$. Each conjugacy class has size $1$. The class equation degenerates to $|G| = |G|$. Conjugacy is therefore an interesting invariant only for non-abelian groups; in commutative settings, every element is its own structural species.

**The center.** The center of $G$ is $Z(G) = \{z \in G : zg = gz \text{ for all } g \in G\}$. An element $x \in Z(G)$ has $C_G(x) = G$, so its conjugacy class is $\{x\}$. Conversely, a singleton class $\{x\}$ implies $x \in Z(G)$.

**The class equation.** Let $G$ be a finite group. Partition $G$ into conjugacy classes. Singleton classes correspond to elements of $Z(G)$. Let non-singleton classes have representatives $x_1, \ldots, x_r$. Then:

$$|G| = |Z(G)| + \sum_{i=1}^{r} [G : C_G(x_i)]$$

Each term $[G : C_G(x_i)]$ is a divisor of $|G|$ greater than $1$.

![The class equation: conjugacy class sizes summing to group order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_v2_02_6_class_equation.png)

**Worked Example: Conjugacy classes of $S_3$.** Cycle types determine classes:
- $(1,1,1)$: $\{e\}$ --- $1$ element
- $(2,1)$: $\{(1\ 2), (1\ 3), (2\ 3)\}$ --- $3$ elements
- $(3)$: $\{(1\ 2\ 3), (1\ 3\ 2)\}$ --- $2$ elements

$1 + 3 + 2 = 6 = |S_3|$. $Z(S_3) = \{e\}$. Class equation: $6 = 1 + 3 + 2$.

**Worked Example: Conjugacy classes of $D_4$.** Order $8$ with elements $\{e, r, r^2, r^3, s, rs, r^2s, r^3s\}$. Compute $rsr^{-1} = r s r^3 = r (sr^3) = r (r^{-3} s) = r^{-2} s = r^2 s$. So $s$ and $r^2 s$ are conjugate. Working through, the classes are:

- $\{e\}$
- $\{r^2\}$
- $\{r, r^3\}$
- $\{s, r^2 s\}$
- $\{rs, r^3 s\}$

Total: $1 + 1 + 2 + 2 + 2 = 8$. Center: $Z(D_4) = \{e, r^2\}$, of order $2$. Class equation: $8 = 2 + 2 + 2 + 2$.

**Worked Example: Conjugacy classes of $S_4$.** $|S_4| = 24$. Cycle types and their sizes:

| Cycle type | Representative | Class size |
|------------|----------------|-----------|
| $(1,1,1,1)$ | $e$ | $1$ |
| $(2,1,1)$ | $(1\ 2)$ | $6$ |
| $(2,2)$ | $(1\ 2)(3\ 4)$ | $3$ |
| $(3,1)$ | $(1\ 2\ 3)$ | $8$ |
| $(4)$ | $(1\ 2\ 3\ 4)$ | $6$ |

Total: $1 + 6 + 3 + 8 + 6 = 24$. $\checkmark$. The class size formula for $S_n$ is $n! / (\prod_i i^{c_i} \cdot c_i!)$ where the cycle type is $(1^{c_1} 2^{c_2} \cdots)$. For type $(2,1,1)$: $24/(2 \cdot 2!) = 6$.

**Why conjugation matters.** Two group elements are conjugate exactly when they "do the same thing in different coordinates." In $S_n$, conjugate permutations have the same cycle structure --- conjugation by $\tau$ relabels the symbols a permutation acts on. The conjugacy classes are therefore the natural "species" of group elements. Almost every classification result in finite group theory is stated in terms of conjugacy classes.

**Proposition.** If $|G| = p^n$ for some prime $p$ and $n \geq 1$ (a *$p$-group*), then $Z(G) \neq \{e\}$.

*Proof.* Each non-singleton class has size $[G : C_G(x_i)]$ dividing $p^n$ and greater than $1$, so divisible by $p$. The class equation gives $|Z(G)| \equiv |G| \equiv 0 \pmod p$, so $|Z(G)| \geq p$. $\square$

**Why this matters.** This seemingly modest result is the engine driving Sylow theory. It implies, for instance, that every group of order $p^2$ is abelian: if $|Z(G)| = p$, then $G/Z(G)$ has order $p$, hence is cyclic, which forces $G$ abelian --- a contradiction. So $|Z(G)| = p^2$, meaning $G = Z(G)$.

The fact that $p$-groups always have a non-trivial center makes them dramatically easier to analyze than arbitrary finite groups: every $p$-group has a non-trivial normal subgroup (any subgroup of $Z(G)$ is normal), so we can always factor and induct. This is in stark contrast to *simple* groups, which by definition have no proper non-trivial normal subgroups, and which power the harder parts of finite group theory.

## Applications: Coloring Problems and Rubik's Cube Symmetries

**Application 1: Coloring the faces of a cube with $k$ colors.**

A cube has $6$ faces. The rotation group of the cube has $24$ elements (isomorphic to $S_4$). The $24$ rotations:

- $1$ identity
- $6$ face rotations by $\pm 90°$
- $3$ face rotations by $180°$
- $8$ vertex rotations by $\pm 120°$
- $6$ edge rotations by $180°$

Total $1 + 6 + 3 + 8 + 6 = 24$. Compute $|X^g|$ for $X = $ all $k$-colorings of $6$ faces:

- **Identity:** $k^6$.
- **Face $\pm 90°$:** top, bottom free; four sides cycle and must all match. $k^3$.
- **Face $180°$:** top, bottom free; sides split into two opposite pairs. $k^4$.
- **Vertex $\pm 120°$:** two triples of faces, each must be uniform. $k^2$.
- **Edge $180°$:** three pairs of faces, each must match. $k^3$.

Burnside:

$$N = \frac{1}{24}\left(k^6 + 6k^3 + 3k^4 + 8k^2 + 6k^3\right) = \frac{1}{24}\left(k^6 + 3k^4 + 12k^3 + 8k^2\right)$$

For $k = 2$: $N = \frac{1}{24}(64 + 48 + 96 + 32) = 240/24 = 10$.

For $k = 3$: $N = \frac{1}{24}(729 + 243 + 324 + 72) = 1368/24 = 57$.

For $k = 6$ (one color per face): $N = \frac{1}{24}(46656 + 3888 + 2592 + 288) = 53424/24 = 2226$. Since you must use each color exactly once for a Rubik's cube setup, the count of distinct standard cubes is $6!/24 = 30$.

A subtle confusion worth flagging: $N = 2226$ counts colorings allowing repetition with $6$ colors, while the problem of "label the $6$ faces with $6$ distinct colors" is a strict-bijection variant whose count is $720/24 = 30$. Both calculations use orbit-stabilizer, but on different sets.

**Application 2: Rubik's cube symmetries.**

The Rubik's cube group $\mathcal{R}$ acts on the set of $48$ non-center facelets and has order

$$|\mathcal{R}| = \frac{8! \cdot 3^8 \cdot 12! \cdot 2^{12}}{12} = 43{,}252{,}003{,}274{,}489{,}856{,}000 \approx 4.3 \times 10^{19}$$

The factors: $8!$ corner positions, $3^8$ corner orientations (over $3$ for total-twist constraint), $12!$ edge positions, $2^{12}$ edge orientations (over $2$ for parity), times $1/2$ for matching corner-edge parity.

Action perspective: $\mathcal{R}$ acts transitively on the $4.3 \times 10^{19}$ solvable configurations, with trivial stabilizer at the solved state. Orbit-stabilizer confirms $|\text{Orb}| = |\mathcal{R}|$.

A practical consequence: the diameter of the cube graph (the maximum number of moves needed to solve any configuration, "God's number") was proved to be exactly $20$ in 2010, using a combination of group theory and large-scale computer search. The fact that $20$ moves suffice is a statement about the geometry of $\mathcal{R}$ as a Cayley graph generated by the six face turns.

**Application 3: Why are there exactly $2$ groups of order $6$?**

Let $|G| = 6$. By Lagrange, possible element orders are $1, 2, 3, 6$. If $G$ has an element of order $6$, $G \cong \mathbb{Z}/6\mathbb{Z}$. Otherwise, $G$ has elements $a$ of order $3$ and $b$ of order $2$ (Cauchy's theorem). The subgroup $\langle a \rangle$ has index $2$, hence is normal. Conjugation: $bab^{-1} \in \langle a \rangle$, so $bab^{-1} \in \{a, a^2\}$. The case $bab^{-1} = a$ forces $G$ abelian, contradiction. So $bab^{-1} = a^{-1}$, giving $G \cong S_3$.

Two groups of order $6$: $\mathbb{Z}/6\mathbb{Z}$ and $S_3$. The action viewpoint pinpoints the structural dichotomy.

**Why this matters.** The same kind of argument --- "find a normal subgroup, classify the conjugation action of the quotient on it" --- is the engine behind the classification of groups of every small order. Up to order $15$, this can be done by hand; modern classification of finite simple groups extended this idea to a 10000-page proof.

**Application 4: counting bracelets of length $5$ with $2$ colors under $D_5$.** $D_5$ has order $10$ ($5$ rotations, $5$ reflections). Fixed-point counts:

- Identity: $2^5 = 32$.
- Rotations by $72°, 144°, 216°, 288°$: each has order $5$, all beads must match. $|X^g| = 2$ each, total $4 \cdot 2 = 8$.
- Each of the $5$ reflections has axis through one vertex and the opposite edge midpoint. Fixes the on-axis vertex and pairs the other $4$ beads into $2$ pairs. $|X^s| = 2 \cdot 2^2 = 8$. Total $5 \cdot 8 = 40$.

Burnside: $(32 + 8 + 40)/10 = 80/10 = 8$. So there are $8$ bracelets.

If we use only rotations (group $C_5$, order $5$), the count becomes $(32 + 4 \cdot 2)/5 = 40/5 = 8$. Same answer here: for $5$-bead $2$-color bracelets, reflections do not produce additional identifications because no $2$-color length-$5$ pattern is its own reflection except those already symmetric under rotation. This is one of those numerical coincidences that initially looks meaningful and turns out to be small-case noise.

**Application 5: counting orbits of $S_n$ on functions $\{1,\ldots,n\} \to \{1,\ldots,k\}$.** A function from $n$ to $k$ is the same as a multiset of size $n$ chosen from $k$ symbols, once we quotient by relabeling of the input set. Burnside gives:

$$N = \frac{1}{n!} \sum_{\sigma \in S_n} k^{c(\sigma)}$$

where $c(\sigma)$ is the number of cycles of $\sigma$ (including fixed points). The right-hand side, after grouping by cycle type, is the standard formula $\binom{n+k-1}{n}$ for multisets. The fact that this combinatorial identity drops out of Burnside is one of the quiet pleasures of the subject.

**Numerical instance.** Take $n = 3, k = 2$. Then $\binom{4}{3} = 4$. Check via Burnside: $S_3$ has cycle types and cycle counts $(e: 3, (1\ 2): 2, (1\ 3): 2, (2\ 3): 2, (1\ 2\ 3): 1, (1\ 3\ 2): 1)$, so the sum is $2^3 + 2^2 + 2^2 + 2^2 + 2 + 2 = 8 + 12 + 4 = 24$, and $24/6 = 4$. Match. The four multisets of size $3$ from $\{0, 1\}$ are $\{0,0,0\}, \{0,0,1\}, \{0,1,1\}, \{1,1,1\}$.

**Numerical instance, larger.** Take $n = 4, k = 3$. Multiset count: $\binom{6}{4} = 15$. Burnside on $S_4$: cycle counts per element type are $(e: 4, (a\ b): 3, (a\ b)(c\ d): 2, (a\ b\ c): 2, (a\ b\ c\ d): 1)$ with class sizes $(1, 6, 3, 8, 6)$. Sum: $1 \cdot 3^4 + 6 \cdot 3^3 + 3 \cdot 3^2 + 8 \cdot 3^2 + 6 \cdot 3 = 81 + 162 + 27 + 72 + 18 = 360$. Divide by $|S_4| = 24$: $360/24 = 15$. Match.

**A note on Polya enumeration.** Burnside's lemma generalizes to a *cycle-index polynomial* construction due to Polya, which lets us count colorings stratified by color usage. The cycle index of a permutation group $G$ acting on $\{1,\ldots,n\}$ is

$$Z_G(z_1, \ldots, z_n) = \frac{1}{|G|} \sum_{\sigma \in G} z_1^{c_1(\sigma)} z_2^{c_2(\sigma)} \cdots z_n^{c_n(\sigma)}$$

where $c_i(\sigma)$ is the number of $i$-cycles in $\sigma$. Substituting $z_i \to k$ for all $i$ recovers Burnside; substituting $z_i \to x_1^i + x_2^i + \cdots + x_k^i$ gives a generating function tracking color usage. We will not develop this fully here, but the punchline is: cycle structure of permutations is the data Burnside needs.

## What Comes Next

Group actions are a lens through which abstract group structure becomes visible. The orbit-stabilizer theorem converts questions about group size into questions about orbits and fixed points. Burnside's lemma converts counting-up-to-symmetry into an average over fixed-point counts. The class equation, derived from the conjugation action, reveals structural constraints on finite groups that lead to deep theorems.

![Animation: group acting on colored objects](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/02_group_action.gif)

In the next article, we turn to *normal subgroups and quotient groups*. A normal subgroup $N \trianglelefteq G$ is precisely a subgroup invariant under the conjugation action. The quotient $G/N$ is a new group whose elements are the cosets of $N$, and the natural map $G \to G/N$ is the prototype of a group homomorphism. This leads to the isomorphism theorems --- the structural backbone of the entire subject.

The tools developed here --- orbits, stabilizers, conjugacy classes --- will reappear throughout the series. They are not optional machinery for special problems; they are the standard vocabulary for analyzing any finite group. Every theorem from the Sylow theorems to the classification of finite simple groups relies on counting arguments rooted in the orbit-stabilizer framework.

**A summary in one sentence.** Group actions translate group theory into combinatorics; orbit-stabilizer is the dictionary, Burnside is the bilingual word frequency table, and the class equation is the structural fingerprint that controls everything.

**Reading recommendations.** For a deep dive into actions, the chapter on group actions in Dummit and Foote covers everything we touched on and more, including double cosets, $G$-equivariant maps, and the orbit-counting interpretation of double cosets. For a more combinatorial bent, Stanley's *Enumerative Combinatorics* (vol. 2) develops Polya theory and applications to enumeration with great care. Tom Leinster's *Basic Category Theory* gives the modern abstract perspective: a $G$-set is the same as a functor from the one-object category $BG$ to $\mathbf{Set}$.

## Deeper Dive: Worked Computations on Actions

Group actions are easiest to absorb in coordinates. Five problems I worked when this material was fresh:

**Computation A: $S_4$ on the four corners of a tetrahedron.** Label the corners $\{1, 2, 3, 4\}$ and let $S_4$ act by permuting labels — every rigid motion of the tetrahedron permutes corners, and conversely every corner permutation lifts to a rigid motion. Pick corner $1$. Its orbit is all four corners (the action is transitive: any corner can be moved to any other). So $|\mathrm{Orb}(1)| = 4$. The stabilizer of corner $1$ is the set of permutations fixing $1$ — namely the symmetric group on $\{2, 3, 4\}$, which is $S_3$, of size $6$. Orbit-stabilizer: $|S_4| = 4 \cdot 6 = 24$. ✓

Now take the same group and look at the action on the *edges*. There are $\binom{4}{2} = 6$ edges. Pick the edge $\{1, 2\}$. Its orbit under $S_4$ is again all six edges, so $|\mathrm{Orb}(\{1,2\})| = 6$. The stabilizer is the set of $\sigma \in S_4$ with $\sigma\{1,2\} = \{1,2\}$ — namely $\langle (1\ 2), (3\ 4)\rangle$, the Klein four-group of order $4$. Check: $24 = 6 \cdot 4$. ✓ Same group, different action, different orbit/stabilizer sizes; the product is always $|G|$.

**Computation B: conjugation in $S_5$.** Take $\sigma = (1\ 2\ 3)$ and conjugate by $\tau = (1\ 4\ 5\ 2)$. The rule: $\tau \sigma \tau^{-1}$ is obtained by relabelling each entry of $\sigma$ via $\tau$. So $\tau (1\ 2\ 3) \tau^{-1} = (\tau(1)\ \tau(2)\ \tau(3)) = (4\ 1\ 3)$. Direct verification: $\tau^{-1} = (1\ 2\ 5\ 4)$, then compute $\tau\sigma\tau^{-1}(4) = \tau\sigma(1) = \tau(2) = 5$. Hmm that's wrong; let me recompute. $\tau(4) = 5$, $\tau(1) = 4$, $\tau(2) = 1$, $\tau(3) = 3$, $\tau(5) = 2$. So $\tau^{-1}(4) = 1$, $\sigma(1) = 2$, $\tau(2) = 1$. So $\tau\sigma\tau^{-1}(4) = 1$. The cycle therefore contains $4 \to 1$. Continuing: $\tau^{-1}(1) = 2$, $\sigma(2) = 3$, $\tau(3) = 3$. So $\tau\sigma\tau^{-1}(1) = 3$. Then $\tau^{-1}(3) = 3$, $\sigma(3) = 1$, $\tau(1) = 4$. So $\tau\sigma\tau^{-1}(3) = 4$. The cycle is $(4\ 1\ 3) = (1\ 3\ 4)$. The relabelling rule produces $(\tau(1)\ \tau(2)\ \tau(3)) = (4\ 1\ 3)$, same cycle. ✓ Conjugation in $S_n$ is "rename the entries"; cycle type is preserved.

**Computation C: Burnside on a necklace.** Count two-coloured necklaces with $6$ beads up to rotation, where the rotation group is $C_6 = \{e, r, r^2, r^3, r^4, r^5\}$. For each $r^k$, count colourings fixed by $r^k$: a colouring is fixed iff the colours are constant on each cycle of $r^k$. The cycle type of $r^k$ on $\{1, \dots, 6\}$ depends on $\gcd(k, 6)$: it has $\gcd(k, 6)$ cycles each of length $6/\gcd(k, 6)$. So the fixed-colouring count is $2^{\gcd(k,6)}$. Sum: $2^6 + 2^1 + 2^2 + 2^3 + 2^2 + 2^1 = 64 + 2 + 4 + 8 + 4 + 2 = 84$. Divide by $|G| = 6$: $84 / 6 = 14$. So there are $14$ distinct necklaces. (The full dihedral version would also account for flips and gives $13$ — but that uses $D_6$, not $C_6$.)

**Computation D: the conjugacy class equation of $S_4$.** Cycle types in $S_4$ and their counts: identity $(1^4)$ → $1$ element; transpositions $(2, 1^2)$ → $\binom{4}{2} = 6$; double transpositions $(2^2)$ → $3$; $3$-cycles $(3, 1)$ → $\frac{4!}{3} = 8$; $4$-cycles $(4)$ → $\frac{4!}{4} = 6$. Total: $1 + 6 + 3 + 8 + 6 = 24$. ✓ Each cycle type forms one conjugacy class, so $S_4$ has exactly $5$ conjugacy classes — the same as the number of partitions of $4$.

**Computation E: a faithful action and Cayley's theorem.** Take $G = \mathbb{Z}/3\mathbb{Z} = \{0, 1, 2\}$. Embed it into $S_3$ via the regular representation $g \mapsto L_g$ where $L_g(h) = g + h$. Then $L_1$ sends $0 \mapsto 1, 1 \mapsto 2, 2 \mapsto 0$, which is the $3$-cycle $(0\ 1\ 2)$ — equivalently $(1\ 2\ 3)$ if you re-index. $L_2 = L_1^2 = (0\ 2\ 1)$. The image in $S_3$ is the subgroup generated by a $3$-cycle, $\{e, (1\ 2\ 3), (1\ 3\ 2)\}$, which is $A_3 \cong \mathbb{Z}/3\mathbb{Z}$. Cayley's theorem checks out: the abstract cyclic group of order $3$ is faithfully realized as a subgroup of $S_3$.

## Counterexamples That Sharpen the Action Definitions

**Action vs. permutation representation.** A group action $G \times X \to X$ is the same data as a homomorphism $\rho: G \to \mathrm{Sym}(X)$, but the homomorphism perspective makes one thing visible that the action perspective hides: the *kernel* of the action, $\{g \in G : g \cdot x = x \text{ for all } x \in X\}$. The action is *faithful* iff the kernel is trivial. The action of $S_4$ on the corners of a tetrahedron is faithful; the action of $S_4$ on the *pairs of opposite edges* (there are three such pairs) has kernel the Klein four-group $V_4 = \{e, (1\ 2)(3\ 4), (1\ 3)(2\ 4), (1\ 4)(2\ 3)\}$, and the induced map $S_4 \to S_3$ is the famous quotient $S_4/V_4 \cong S_3$. Same group, two different actions, totally different kernels.

**Stabilizers along an orbit are conjugate, not equal.** A common slip: assuming $\mathrm{Stab}(x) = \mathrm{Stab}(y)$ for $y$ in the orbit of $x$. They are not. If $y = g \cdot x$, then $\mathrm{Stab}(y) = g \, \mathrm{Stab}(x) \, g^{-1}$. Concrete check in $S_4$ acting on corners: $\mathrm{Stab}(1) = S_{\{2,3,4\}}$, $\mathrm{Stab}(2) = S_{\{1,3,4\}}$. These are different subgroups (different element sets), but conjugate (relabelling $1 \leftrightarrow 2$ moves one to the other). The intersection $\mathrm{Stab}(1) \cap \mathrm{Stab}(2) = S_{\{3,4\}}$, of order $2$, is the pointwise stabilizer of the *pair* $\{1, 2\}$ — distinct from the setwise stabilizer.

## Common Pitfalls for Beginners

The first pitfall: thinking of an "orbit" as an attribute of $G$ rather than of the action. The same group acts on many sets, and orbits live in the set being acted upon, not in the group. A finite group has finitely many orbit-types but infinitely many actions; do not confuse the structural fact with the specific computation.

The second pitfall: the orbit-stabilizer formula $|G| = |\mathrm{Orb}(x)| \cdot |\mathrm{Stab}(x)|$ is not "obvious" without the bijection $G/\mathrm{Stab}(x) \to \mathrm{Orb}(x)$. Beginners often try to count orbits directly and end up with the wrong answer because they forget that two coset representatives can give the same orbit element only if they differ by something in the stabilizer.

The third pitfall: conflating the conjugation action with the left multiplication action. They are both actions of $G$ on itself, but they have wildly different orbit structures. Left multiplication is transitive (one orbit, all of $G$). Conjugation has orbits = conjugacy classes (many of them, including singletons for central elements). Burnside applied to the conjugation action gives the class equation; applied to left multiplication, it just gives $|G| = |G| \cdot 1$.

## Where This Shows Up

*Galois theory.* The Galois group of a polynomial acts on the roots. Orbit structure under this action is exactly what determines whether a polynomial is irreducible, whether it factors over a sub-extension, and whether the polynomial is solvable by radicals. We will see this in Part 8.

*Counting under symmetry.* Combinatorial enumeration in chemistry, physics, and combinatorics is dominated by Burnside-style arguments. The number of distinct chemical isomers of a hydrocarbon, the number of distinct $n \times n$ binary matrices up to row and column permutation, the number of distinct Rubik's cube positions up to whole-cube rotation — all are Burnside or Polya computations.

*Algebraic topology and covering spaces.* The deck transformation group of a covering space acts freely on the cover, and the quotient is the base space. Classification of covering spaces of a topological space corresponds to classification of subgroups of the fundamental group. The dictionary between group actions and topology is essentially perfect for nice spaces, and it is the gateway to higher topology.

## What I Want You to Carry Forward

Four questions for the road into Part 3:

1. *When can we form the quotient $G/H$ as a group, not just as a set of cosets?* The condition will be that $H$ is *normal*, i.e., that left and right cosets coincide, equivalently that $H$ is fixed by conjugation.
2. *What does the kernel of a homomorphism look like, and why is it always a normal subgroup?* This is the algebraic content of "the homomorphism is well-defined on the quotient."
3. *How does the first isomorphism theorem encode the universal property of quotients?* It says that any homomorphism $\varphi: G \to H$ factors uniquely through $G/\ker\varphi$, and the factor is injective. This is the universal property that makes the quotient construction "the right one."
4. *What is the relation between conjugacy classes of $G$ and orbits under the conjugation action?* Identical, as already noted, but the language of normal subgroups will let us go further: a normal subgroup is a union of conjugacy classes, which gives a fast computational test.

If the orbit-stabilizer computations above feel routine, you are ready. If not, redo Computation A with $S_5$ acting on something — say, the $5$ vertices of a $5$-pointed star, or the $10$ pairs from $\{1, 2, 3, 4, 5\}$. Group actions are a muscle, and the only way to develop it is to flex it on examples that you have not seen worked out for you.

---
