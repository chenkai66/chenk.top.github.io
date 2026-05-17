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

## From Abstract Groups to Concrete Actions

In the previous article we defined groups and proved Lagrange's theorem. The groups themselves --- $\mathbb{Z}/n\mathbb{Z}$, $S_n$, $D_n$ --- were interesting, but we studied them in isolation. The real power of group theory emerges when a group *does something*: when it permutes the elements of a set, rotates the vertices of a polygon, or rearranges the colors of a necklace.

This is the idea of a *group action*. Historically, groups arose precisely as collections of symmetries acting on geometric objects. Galois studied permutation groups acting on roots of polynomials. Klein's Erlangen program (1872) proposed defining geometry itself as the study of invariants under a group action. In modern algebra, the action viewpoint is indispensable: it converts abstract group-theoretic questions into concrete combinatorial ones, and vice versa.

The central results of this article are the orbit-stabilizer theorem and Burnside's lemma. The orbit-stabilizer theorem relates the size of a group to the sizes of orbits and stabilizers under an action. Burnside's lemma uses this to count distinct objects up to symmetry --- the kind of problem that arises whenever we ask "how many essentially different colorings are there?" The answer is always a weighted average of fixed-point counts, and the proof is a clean application of the orbit-stabilizer machinery.

We also introduce conjugation as a group action, which leads to the class equation --- a tool that will be crucial in later articles when we prove the Sylow theorems.


![Orbit-Stabilizer theorem diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-group-actions-and-symmetry/aa_fig2_orbit_stabilizer.png)

## Formal Definition and Examples

**Definition.** Let $G$ be a group and $X$ a set. A *(left) group action* of $G$ on $X$ is a function $G \times X \to X$, written $(g, x) \mapsto g \cdot x$, satisfying:

1. **Identity.** $e \cdot x = x$ for all $x \in X$.
2. **Compatibility.** $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$ and $x \in X$.

We say $G$ *acts on* $X$, or that $X$ is a *$G$-set*.

**Equivalent formulation.** A group action is the same as a group homomorphism $\varphi : G \to \text{Sym}(X)$, where $\text{Sym}(X)$ is the group of all bijections $X \to X$. Given an action, define $\varphi(g)(x) = g \cdot x$; the compatibility axiom ensures $\varphi(gh) = \varphi(g) \circ \varphi(h)$, and the identity axiom ensures $\varphi(e) = \text{id}_X$. Conversely, any such homomorphism defines an action.

This reformulation is important: the *kernel* of $\varphi$ is $\{g \in G : g \cdot x = x \text{ for all } x \in X\}$, the set of group elements that act trivially on every point of $X$.

**Definition.** An action is *faithful* (or *effective*) if $\ker\varphi = \{e\}$, i.e., only the identity fixes every element of $X$. Equivalently, the homomorphism $\varphi : G \to \text{Sym}(X)$ is injective, so $G$ embeds into $\text{Sym}(X)$.

**Cayley's theorem** follows immediately: every group acts on itself by left multiplication, and this action is faithful. Hence every group is isomorphic to a subgroup of some symmetric group. Specifically, if $|G| = n$, then $G$ embeds into $S_n$.

**Right actions.** A *right action* is a map $X \times G \to X$, written $(x, g) \mapsto x \cdot g$, satisfying $x \cdot e = x$ and $x \cdot (gh) = (x \cdot g) \cdot h$. Any right action becomes a left action via $g \cdot x = x \cdot g^{-1}$. We focus on left actions throughout.

**Example 1: $S_n$ acting on $\{1, \ldots, n\}$.** The natural action: $\sigma \cdot i = \sigma(i)$. This is faithful (distinct permutations move at least one element differently).

**Example 2: $D_n$ acting on the vertices of a regular $n$-gon.** Label the vertices $1, \ldots, n$. Each symmetry (rotation or reflection) permutes the vertices. This defines a faithful action $D_n \to S_n$, which is the embedding we used in the previous article.

**Example 3: $G$ acting on itself by left multiplication.** Define $g \cdot x = gx$ for $g, x \in G$. The axioms are satisfied: $e \cdot x = ex = x$ and $(gh) \cdot x = (gh)x = g(hx) = g \cdot (h \cdot x)$. This is the *left regular action*. It is always faithful.

**Example 4: $G$ acting on itself by conjugation.** Define $g \cdot x = gxg^{-1}$. Check: $e \cdot x = exe^{-1} = x$, and $(gh) \cdot x = (gh)x(gh)^{-1} = g(hxh^{-1})g^{-1} = g \cdot (h \cdot x)$. This action is typically not faithful: $g$ acts trivially on all of $G$ if and only if $gxg^{-1} = x$ for all $x$, i.e., $g \in Z(G)$ (the center of $G$). The kernel is precisely $Z(G)$.

**Example 5: $G$ acting on its subsets by conjugation.** For a subset $S \subseteq G$, define $g \cdot S = gSg^{-1} = \{gsg^{-1} : s \in S\}$. If $H \leq G$, then $g \cdot H = gHg^{-1}$, which is again a subgroup. This gives an action of $G$ on the set of all subgroups of $G$.

**Example 6: $G$ acting on left cosets of $H$.** Let $G/H = \{gH : g \in G\}$ denote the set of left cosets. Define $g \cdot (aH) = (ga)H$. This is well-defined: if $aH = bH$, then $b^{-1}a \in H$, so $(gb)^{-1}(ga) = b^{-1}a \in H$, hence $(ga)H = (gb)H$. The axioms are immediate. The kernel of this action is $\bigcap_{g \in G} gHg^{-1}$, the largest normal subgroup of $G$ contained in $H$.

## Orbits, Stabilizers, and the Orbit-Stabilizer Theorem

**Definition.** Let $G$ act on $X$ and $x \in X$.

- The *orbit* of $x$ is $\text{Orb}(x) = G \cdot x = \{g \cdot x : g \in G\}$.
- The *stabilizer* of $x$ is $\text{Stab}(x) = G_x = \{g \in G : g \cdot x = x\}$.

The orbit is a subset of $X$; the stabilizer is a subset of $G$.

**Proposition.** $\text{Stab}(x)$ is a subgroup of $G$.

*Proof.* $e \cdot x = x$, so $e \in \text{Stab}(x)$. If $g, h \in \text{Stab}(x)$, then $(gh^{-1}) \cdot x = g \cdot (h^{-1} \cdot x) = g \cdot x = x$ (using $h^{-1} \cdot x = x$, which follows from $h \cdot (h^{-1} \cdot x) = (hh^{-1}) \cdot x = x = h \cdot x$, so $h^{-1} \cdot x = x$ by cancellation in $X$... more precisely: from $h \cdot x = x$, apply $h^{-1}$ to get $h^{-1} \cdot (h \cdot x) = h^{-1} \cdot x$, i.e., $(h^{-1}h) \cdot x = h^{-1} \cdot x$, i.e., $x = h^{-1} \cdot x$). By the subgroup criterion, $\text{Stab}(x) \leq G$. $\square$

**The orbits partition $X$.** Define $x \sim y$ if $y = g \cdot x$ for some $g \in G$. This is an equivalence relation: reflexive ($x = e \cdot x$), symmetric (if $y = g \cdot x$ then $x = g^{-1} \cdot y$), transitive (if $y = g \cdot x$ and $z = h \cdot y$ then $z = (hg) \cdot x$). The equivalence classes are exactly the orbits. When there is a single orbit ($G \cdot x = X$ for some --- equivalently every --- $x \in X$), we say the action is *transitive*.

**Theorem (Orbit-Stabilizer).** Let $G$ be a finite group acting on a set $X$, and let $x \in X$. Then

$$|G| = |\text{Orb}(x)| \cdot |\text{Stab}(x)|$$

or equivalently,

$$|\text{Orb}(x)| = [G : \text{Stab}(x)] = \frac{|G|}{|\text{Stab}(x)|}$$

*Proof.* We construct a bijection between $\text{Orb}(x)$ and the set of left cosets $G / \text{Stab}(x)$. Define $\Phi : G/\text{Stab}(x) \to \text{Orb}(x)$ by $\Phi(g \cdot \text{Stab}(x)) = g \cdot x$.

**Well-defined:** If $g \cdot \text{Stab}(x) = h \cdot \text{Stab}(x)$, then $h^{-1}g \in \text{Stab}(x)$, so $(h^{-1}g) \cdot x = x$, hence $g \cdot x = h \cdot x$.

**Surjective:** Every element of $\text{Orb}(x)$ has the form $g \cdot x$ for some $g \in G$.

**Injective:** If $g \cdot x = h \cdot x$, then $h^{-1} \cdot (g \cdot x) = h^{-1} \cdot (h \cdot x)$, so $(h^{-1}g) \cdot x = x$, meaning $h^{-1}g \in \text{Stab}(x)$, hence $g \cdot \text{Stab}(x) = h \cdot \text{Stab}(x)$.

So $\Phi$ is a bijection, giving $|\text{Orb}(x)| = |G/\text{Stab}(x)| = [G : \text{Stab}(x)]$. Multiply both sides by $|\text{Stab}(x)|$ and use Lagrange's theorem: $|G| = [G : \text{Stab}(x)] \cdot |\text{Stab}(x)| = |\text{Orb}(x)| \cdot |\text{Stab}(x)|$. $\square$

**Worked Example: $D_4$ acting on the vertices of a square.** Label the vertices $\{1, 2, 3, 4\}$. The group $D_4$ (order $8$) acts on this set. Consider $x = 1$.

The orbit: every vertex can be reached from vertex $1$ by some symmetry (e.g., rotation by $90°$ sends $1 \to 2$, by $180°$ sends $1 \to 3$, etc.), so $\text{Orb}(1) = \{1, 2, 3, 4\}$ and $|\text{Orb}(1)| = 4$.

The stabilizer: $\text{Stab}(1) = \{g \in D_4 : g \cdot 1 = 1\}$. The identity fixes $1$. The reflection $s$ through the axis passing through vertex $1$ and the midpoint of the opposite side fixes $1$. So $\text{Stab}(1) = \{e, s\}$ (one can verify that no rotation except $e$ fixes vertex $1$, and only one of the four reflections does).

Check: $|D_4| = 8 = 4 \times 2 = |\text{Orb}(1)| \times |\text{Stab}(1)|$. Confirmed.

**Worked Example: $S_4$ acting on $\{1,2,3,4\}$ --- stabilizer of $1$.** We have $\text{Stab}(1) = \{\sigma \in S_4 : \sigma(1) = 1\}$, which is the set of permutations that fix $1$. These are exactly the permutations of $\{2, 3, 4\}$, so $\text{Stab}(1) \cong S_3$ and $|\text{Stab}(1)| = 6$. The orbit of $1$ is $\{1,2,3,4\}$ since $S_4$ acts transitively. Check: $24 = 4 \times 6$. $\checkmark$

## Fixed Points and Burnside's Lemma

For a group element $g \in G$ acting on $X$, define the *fixed-point set* of $g$:

$$X^g = \text{Fix}(g) = \{x \in X : g \cdot x = x\}$$

This is the set of elements of $X$ left unchanged by $g$.

**Theorem (Burnside's Lemma).** Let $G$ be a finite group acting on a finite set $X$. The number of distinct orbits is

$$|\text{Orbits}| = \frac{1}{|G|} \sum_{g \in G} |X^g|$$

That is, the number of orbits equals the average number of fixed points.

*Proof.* Count the set $S = \{(g, x) \in G \times X : g \cdot x = x\}$ in two ways.

**Counting by $x$:** For each $x \in X$, the number of $g$ with $g \cdot x = x$ is $|\text{Stab}(x)|$. So

$$|S| = \sum_{x \in X} |\text{Stab}(x)|$$

**Counting by $g$:** For each $g \in G$, the number of $x$ with $g \cdot x = x$ is $|X^g|$. So

$$|S| = \sum_{g \in G} |X^g|$$

From the orbit-stabilizer theorem, $|\text{Stab}(x)| = |G| / |\text{Orb}(x)|$. Therefore:

$$\sum_{x \in X} |\text{Stab}(x)| = \sum_{x \in X} \frac{|G|}{|\text{Orb}(x)|} = |G| \sum_{x \in X} \frac{1}{|\text{Orb}(x)|}$$

The inner sum, grouped by orbits: if the orbits are $O_1, \ldots, O_k$, then

$$\sum_{x \in X} \frac{1}{|\text{Orb}(x)|} = \sum_{i=1}^{k} \sum_{x \in O_i} \frac{1}{|O_i|} = \sum_{i=1}^{k} 1 = k$$

So $|S| = |G| \cdot k$. But also $|S| = \sum_{g \in G} |X^g|$. Equating: $k = \frac{1}{|G|} \sum_{g \in G} |X^g|$. $\square$

The lemma is often attributed to Burnside, though it was known earlier to Cauchy and Frobenius. Some authors call it the Cauchy-Frobenius lemma.

**Worked Example: Counting necklaces with 4 beads and 2 colors.**

We want to count the number of distinct necklaces made of $4$ beads, each colored black or white, where two necklaces are "the same" if one can be rotated into the other. The symmetry group is the cyclic group $C_4 = \{e, r, r^2, r^3\}$ (rotations only; if we also allow flipping the necklace over, we'd use $D_4$).

The set $X$ of all colorings has $|X| = 2^4 = 16$ elements. We need $|X^g|$ for each $g \in C_4$:

- **$g = e$ (identity):** Every coloring is fixed. $|X^e| = 16$.
- **$g = r$ (rotation by $90°$):** A coloring is fixed if and only if all four beads have the same color. $|X^r| = 2$ (all black or all white).
- **$g = r^2$ (rotation by $180°$):** A coloring is fixed if bead $1$ = bead $3$ and bead $2$ = bead $4$. Two free choices. $|X^{r^2}| = 2^2 = 4$.
- **$g = r^3$ (rotation by $270°$):** Same constraint as $r$ (all beads equal). $|X^{r^3}| = 2$.

Burnside:

$$|\text{Orbits}| = \frac{1}{4}(16 + 2 + 4 + 2) = \frac{24}{4} = 6$$

The six distinct necklaces are: BBBB, BBBW, BBWW (adjacent), BWBW (alternating), BWWW, WWWW.

Let us verify by listing them. With beads labeled $1,2,3,4$ going clockwise, up to rotation:

1. $\{BBBB\}$
2. $\{BBBW, BWBB, WBBB, BBWB\}$ --- one orbit of size $4$, but wait: if we rotate BBBW by $90°$ we get WBBB, by $180°$ we get BWBB, by $270°$ we get BBWB. So this is one orbit of size $4$.
3. $\{BBWW, WBBW, WWBB, BWWB\}$ --- one orbit (adjacent pair).
4. $\{BWBW, WBWB\}$ --- one orbit of size $2$.
5. $\{BWWW, WBWW, WWBW, WWWB\}$ --- one orbit of size $4$.
6. $\{WWWW\}$

Total orbits: $6$. Confirmed.

**Extended example: necklaces under $D_4$.** If we allow reflections (flipping the necklace), the group is $D_4$ of order $8$. We need fixed-point counts for all $8$ elements. The four rotations contribute $16 + 2 + 4 + 2 = 24$ as before. Now the reflections:

- **Reflection through two opposite vertices (2 such reflections):** Beads at the axis are free, the other two must match. $3$ free binary choices? Let me think carefully. Say the reflection swaps beads $2 \leftrightarrow 4$ and fixes beads $1, 3$. Then we need bead $2$ = bead $4$. Three free choices (beads $1, 3, 2$), so $|X^s| = 2^3 = 8$. Same for the other such reflection.
- **Reflection through midpoints of opposite edges (2 such reflections):** Swaps bead $1 \leftrightarrow 2$ and bead $3 \leftrightarrow 4$. Need bead $1$ = bead $2$ and bead $3$ = bead $4$. Two free choices. $|X^s| = 2^2 = 4$. Same for the other.

Total: $\frac{1}{8}(16 + 2 + 4 + 2 + 8 + 8 + 4 + 4) = \frac{48}{8} = 6$.

Interestingly, the answer is the same! This is a coincidence for this particular $(n, k)$; in general, allowing reflections reduces the count.

## Conjugation and the Class Equation

One of the most important group actions is the action of a group on itself by conjugation: $g \cdot x = gxg^{-1}$. The orbits under this action are called *conjugacy classes*, and the stabilizer of $x$ is the *centralizer* $C_G(x) = \{g \in G : gxg^{-1} = x\} = \{g \in G : gx = xg\}$.

By orbit-stabilizer, the size of the conjugacy class of $x$ is $[G : C_G(x)]$.

**The center.** The center of $G$ is $Z(G) = \{z \in G : zg = gz \text{ for all } g \in G\}$. An element $x \in Z(G)$ has $C_G(x) = G$, so its conjugacy class is $\{x\}$ (a singleton). Conversely, if the conjugacy class of $x$ is $\{x\}$, then $gxg^{-1} = x$ for all $g$, so $x \in Z(G)$.

**The class equation.** Let $G$ be a finite group. Partition $G$ into conjugacy classes. The classes of size $1$ correspond to elements of $Z(G)$. Let the non-singleton classes have representatives $x_1, \ldots, x_r$. Then:

$$|G| = |Z(G)| + \sum_{i=1}^{r} [G : C_G(x_i)]$$

Each term $[G : C_G(x_i)]$ is a divisor of $|G|$ greater than $1$.

**Worked Example: Conjugacy classes of $S_3$.** The elements of $S_3$ are $\{e, (1\ 2), (1\ 3), (2\ 3), (1\ 2\ 3), (1\ 3\ 2)\}$.

Conjugacy classes in $S_n$ correspond to cycle types. In $S_3$:
- Cycle type $(1,1,1)$: $\{e\}$ --- $1$ element
- Cycle type $(2,1)$: $\{(1\ 2), (1\ 3), (2\ 3)\}$ --- $3$ elements
- Cycle type $(3)$: $\{(1\ 2\ 3), (1\ 3\ 2)\}$ --- $2$ elements

Verify: $1 + 3 + 2 = 6 = |S_3|$. $\checkmark$

The center: $Z(S_3) = \{e\}$ (only the identity commutes with all elements, since $S_3$ is non-abelian and too small for a non-trivial center). Class equation: $6 = 1 + 3 + 2$.

**Proposition.** If $|G| = p^n$ for some prime $p$ and $n \geq 1$ (a *$p$-group*), then $Z(G) \neq \{e\}$.

*Proof.* Apply the class equation: $|G| = |Z(G)| + \sum [G : C_G(x_i)]$. Each term $[G : C_G(x_i)]$ divides $|G| = p^n$ and is greater than $1$, so each is divisible by $p$. Since $|G| = p^n$ is divisible by $p$, and the sum is divisible by $p$, it follows that $|Z(G)|$ is divisible by $p$. Since $e \in Z(G)$, we have $|Z(G)| \geq 1$, but actually $|Z(G)| \geq p$. In particular $Z(G) \neq \{e\}$. $\square$

This seemingly modest result has powerful consequences. For instance, it implies that every group of order $p^2$ is abelian (proof: if $Z(G) \neq G$, then $|Z(G)| = p$, so $G/Z(G)$ has order $p$, hence is cyclic, which forces $G$ to be abelian --- a contradiction).

## Applications: Coloring Problems and Rubik's Cube Symmetries

**Application 1: Coloring the faces of a cube with $k$ colors.**

A cube has $6$ faces. The rotation group of the cube has $24$ elements (isomorphic to $S_4$, the symmetric group on $4$ body diagonals). We want to count colorings of the $6$ faces using $k$ colors, up to rotation.

The $24$ rotations, grouped by type:

| Rotation type | Count | Fixed colorings |
|--------------|-------|-----------------|
| Identity | $1$ | $k^6$ |
| Face rotations ($90°$ and $270°$) | $6$ | $k^3$ (the top/bottom faces are fixed separately, the $4$ side faces must all match) |
| Face rotations ($180°$) | $3$ | $k^4$ (top free, bottom free, side faces split into $2$ pairs) |
| Vertex rotations ($120°$ and $240°$) | $8$ | $k^2$ (faces split into $2$ triples, each triple must be uniform) |
| Edge rotations ($180°$) | $6$ | $k^3$ (faces split into $3$ pairs, each pair must match) |

Wait, let me recount. The rotation group of a cube consists of:
- $1$ identity
- $6$ face rotations by $\pm 90°$ (3 axes, 2 rotations each)
- $3$ face rotations by $180°$
- $8$ vertex rotations by $\pm 120°$ (4 body diagonals, 2 rotations each)
- $6$ edge rotations by $180°$ (6 axes through midpoints of opposite edges)

Total: $1 + 6 + 3 + 8 + 6 = 24$. $\checkmark$

Now compute $|X^g|$ for each type (where $X$ is the set of all $k$-colorings of $6$ faces):

- **Identity:** All $k^6$ colorings are fixed.
- **Face rotation by $90°$ or $270°$:** The axis passes through the centers of two opposite faces. Those two faces are each fixed individually (but may have different colors). The four side faces are cyclically permuted, so they must all be the same color. Free choices: $k$ (top) $\times$ $k$ (bottom) $\times$ $k$ (sides) $= k^3$.
- **Face rotation by $180°$:** Same axis. Top face fixed, bottom face fixed. Four side faces split into two pairs of opposite faces; each pair must match. Free choices: $k \times k \times k \times k = k^4$. Wait: top ($k$), bottom ($k$), pair 1 ($k$), pair 2 ($k$) = $k^4$.
- **Vertex rotation by $120°$ or $240°$:** The axis passes through two opposite vertices. The three faces meeting at one vertex are cyclically permuted (must all match), and the three faces meeting at the other vertex are cyclically permuted (must all match). Free choices: $k \times k = k^2$.
- **Edge rotation by $180°$:** The axis passes through midpoints of two opposite edges. No face is fixed; the six faces are paired into three pairs, each swapped. Each pair must match. Free choices: $k^3$.

Burnside:

$$N = \frac{1}{24}\left(k^6 + 6k^3 + 3k^4 + 8k^2 + 6k^3\right) = \frac{1}{24}\left(k^6 + 3k^4 + 12k^3 + 8k^2\right)$$

For $k = 2$: $N = \frac{1}{24}(64 + 48 + 96 + 32) = \frac{240}{24} = 10$.

So there are $10$ distinct ways to color the faces of a cube with $2$ colors, up to rotation.

For $k = 3$: $N = \frac{1}{24}(729 + 243 + 324 + 72) = \frac{1368}{24} = 57$.

**Application 2: Understanding Rubik's cube symmetries.**

The Rubik's cube is a physical puzzle, but its mathematical structure is a group-theoretic object. The group of all possible moves (sequences of face rotations) acts on the set of configurations. This group --- call it $\mathcal{R}$ --- is a subgroup of $S_{48}$ (since there are $48$ colored facelets that can be permuted, excluding the $6$ center facelets which are fixed by construction). Its order is:

$$|\mathcal{R}| = \frac{8! \cdot 3^8 \cdot 12! \cdot 2^{12}}{12} = 43{,}252{,}003{,}274{,}489{,}856{,}000$$

approximately $4.3 \times 10^{19}$. The factors account for: $8!$ positions of corner cubies, $3^8$ orientations of corners (divided by $3$ for the constraint that total twist is $0 \mod 3$), $12!$ positions of edge cubies, $2^{12}$ orientations of edges (divided by $2$ for the parity constraint), and an additional factor of $1/2$ because the overall permutation parity of corners and edges must match.

This is not a number one derives casually; it requires understanding the group structure. The Rubik's cube group is generated by $6$ elements (the six face rotations $U, D, L, R, F, B$) and has a rich internal structure involving commutator subgroups, normal series, and eventually an expression as a semidirect product.

The key insight from the action perspective: the group $\mathcal{R}$ acts on the $4.3 \times 10^{19}$ configurations transitively (any solvable configuration can reach the solved state). The orbit-stabilizer theorem tells us that the stabilizer of any particular configuration (say, the solved state) has size $1$ --- because the only move sequence that takes the solved state to the solved state is the trivial one (do nothing). This means $|\text{Orb}| = |\mathcal{R}|/1 = |\mathcal{R}|$, confirming that every configuration is reachable.

**Application 3: Why are there exactly $2$ nonisomorphic groups of order $6$?**

Consider a group $G$ with $|G| = 6 = 2 \times 3$. By Lagrange, the possible element orders are $1, 2, 3, 6$. If $G$ has an element of order $6$, then $G \cong \mathbb{Z}/6\mathbb{Z}$, which is abelian. Otherwise, $G$ has no element of order $6$.

By Cauchy's theorem (which follows from the class equation applied to $p$-groups, extended to general groups), $G$ has an element $a$ of order $3$ and an element $b$ of order $2$. The subgroup $\langle a \rangle = \{e, a, a^2\}$ has index $2$ in $G$, hence is normal. The element $b$ is not in $\langle a \rangle$ (since $b$ has order $2$, not $1$ or $3$). The six elements of $G$ are $\{e, a, a^2, b, ab, a^2 b\}$.

Conjugation: $bab^{-1} = ba b \in \langle a \rangle$ (since $\langle a \rangle$ is normal), so $bab = a$ or $bab = a^2$. If $bab = a$, then $ba = ab$, so $G$ is abelian and $ab$ has order $6$, contradicting our assumption. So $bab = a^2 = a^{-1}$. This relation, together with $a^3 = e$ and $b^2 = e$, determines the multiplication table completely, giving $G \cong S_3 \cong D_3$.

So the only groups of order $6$ are $\mathbb{Z}/6\mathbb{Z}$ and $S_3$, and the group action viewpoint (specifically, the normality forced by index $2$ and the conjugation structure) is what pins this down.

## What Comes Next

Group actions are a lens through which abstract group structure becomes visible. The orbit-stabilizer theorem converts questions about group size into questions about orbits and fixed points. Burnside's lemma converts counting-up-to-symmetry into an average over fixed-point counts. The class equation, derived from the conjugation action, reveals structural constraints on finite groups that lead to deep theorems.

In the next article, we turn to *normal subgroups and quotient groups*. A normal subgroup $N \trianglelefteq G$ is precisely a subgroup whose left and right cosets coincide, or equivalently, a subgroup invariant under the conjugation action. The quotient $G/N$ is a new group whose elements are the cosets of $N$, and the natural map $G \to G/N$ is the prototype of a group homomorphism. This leads to the isomorphism theorems --- the structural backbone of the entire subject.

The tools developed here --- orbits, stabilizers, conjugacy classes --- will reappear throughout the series. They are not optional machinery for special problems; they are the standard vocabulary for analyzing any finite group. Every theorem from the Sylow theorems to the classification of finite simple groups relies on counting arguments rooted in the orbit-stabilizer framework.

---

*This is Part 2 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 1 — Groups: Your First Encounter](/en/abstract-algebra/01-groups-first-encounter/)*

*Next: [Part 3 — Quotient Groups and Homomorphisms](/en/abstract-algebra/03-quotient-groups-and-homomorphisms/)*
