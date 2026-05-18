---
title: "Quotient Groups and Homomorphisms: The Art of Collapsing Structure"
date: 2021-09-05 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - homomorphisms
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "Normal subgroups, quotient constructions, and the isomorphism theorems — how to systematically simplify groups while preserving their essence."
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "abstract-algebra-3"
---

A group can be enormous --- millions of elements, intricate multiplication tables, symmetries that take pages to describe. Yet hidden inside every group are natural compression points where you can collapse entire chunks of the group into single elements, producing a smaller group that still remembers something essential about the original. This article develops the machinery for doing that: normal subgroups, quotient groups, homomorphisms, and the isomorphism theorems that tie everything together.

If groups are the atoms of symmetry, quotient groups are what you get when you deliberately blur your vision --- ignoring certain details so the remaining structure becomes visible. The metaphor has teeth: every theorem about quotients is, at root, a precise statement about what gets lost and what survives.

## Why Collapse Structure?

Mental picture: imagine a globe with many cities marked on it. If you only care about which country each city is in, you "collapse" cities into countries. The set of countries inherits some natural structure from the cities (you can group countries by continent, by hemisphere, etc.). Quotient groups are this idea, applied to algebra.

Consider the integers $\mathbb{Z}$ under addition. This is an infinite group, and reasoning about its full structure all at once is unwieldy. But we have a familiar trick: work modulo $n$. When we compute mod 5, we declare any two integers differing by a multiple of 5 to be the same. The set $\{0, 1, 2, 3, 4\}$ with addition mod 5 forms a perfectly well-behaved group $\mathbb{Z}/5\mathbb{Z}$.

What just happened? We took an infinite group and produced a finite one by identifying elements that differ by something from a subgroup ($5\mathbb{Z}$). The result is smaller, simpler, and still a group. This is the quotient construction, and it is one of the most powerful ideas in all of algebra.

But not every subgroup works. If you try to mod out by an arbitrary subgroup, the resulting set of cosets might not form a group at all --- the operation might not be well-defined. The subgroups that do work have a special property: they are *normal*. Understanding what normality means is our first task.

**Motivation from geometry.** In the symmetry group of a square ($D_4$, eight elements), the four rotations form a subgroup $R = \{e, r, r^2, r^3\}$. If we collapse all rotations to a single element and all reflections to another, we get a group of order $2$ --- essentially $\mathbb{Z}/2\mathbb{Z}$. This captures the fact that reflections and rotations are "categorically different," while the specific rotation or reflection does not matter at this coarse level. The rotation subgroup happens to be normal in $D_4$, which is why this works.

**Motivation from linear algebra.** Given a linear map $T: V \to W$, the kernel $\ker T$ is a subspace of $V$. The quotient space $V / \ker T$ is isomorphic to the image $\text{im}(T)$. This is the rank-nullity theorem in disguise. The same idea generalizes to groups: the kernel of a group homomorphism is always normal, and the quotient by the kernel is isomorphic to the image.

![Quotient group G/N construction by collapsing cosets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_3_quotient_construction.png)

## Normal Subgroups

Mental picture: a normal subgroup is a subgroup that "looks the same from every conjugation viewpoint." If you stir the group around by conjugation, the subgroup is unchanged as a set. This invariance is exactly what is needed to make the coset multiplication well-defined.

![Normal subgroup: conjugation-invariant](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_normal_subgroup.png)

**Definition.** A subgroup $N$ of a group $G$ is *normal* (written $N \trianglelefteq G$) if for every $g \in G$ and every $n \in N$, the conjugate $gng^{-1}$ belongs to $N$. Equivalently, $gNg^{-1} = N$ for all $g \in G$.

![Short exact sequence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_short_exact.png)

This says that $N$ is invariant under conjugation: you cannot escape $N$ by conjugating its elements.

**Equivalent characterizations.** The following are equivalent for a subgroup $N \leq G$:

1. $N \trianglelefteq G$ ($gNg^{-1} = N$ for all $g$).
2. $gN = Ng$ for all $g \in G$ (left cosets equal right cosets).
3. $N$ is the kernel of some group homomorphism from $G$.
4. Every left coset of $N$ is also a right coset.

Condition (2) is often most practical for checking normality. Note: $gN = Ng$ does *not* mean $gn = ng$ for every $n$; it means the *sets* are equal.

![Normal subgroup vs. arbitrary subgroup: cosets compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_1_normal_vs_subgroup.png)

**Example 1: $n\mathbb{Z}$ in $\mathbb{Z}$.** Since $\mathbb{Z}$ is abelian, every subgroup is automatically normal. So $5\mathbb{Z} \trianglelefteq \mathbb{Z}$, and indeed every $n\mathbb{Z}$ is normal.

**Example 2: $A_n$ in $S_n$.** The alternating group is normal in $S_n$. For any $\sigma \in S_n$ and $\alpha \in A_n$: $\text{sgn}(\sigma \alpha \sigma^{-1}) = \text{sgn}(\sigma) \text{sgn}(\alpha) \text{sgn}(\sigma^{-1}) = \text{sgn}(\alpha)$, so $\sigma\alpha\sigma^{-1} \in A_n$.

**Example 3: A non-normal subgroup.** In $S_3$, consider $H = \{e, (1\ 2)\}$. The left coset $(1\ 3)H = \{(1\ 3), (1\ 2\ 3)\}$ while the right coset $H(1\ 3) = \{(1\ 3), (1\ 3\ 2)\}$. The cosets differ, so $H$ is not normal.

**Numerical example: cosets of $\langle (1\ 2\ 3) \rangle$ in $S_3$.** Let $N = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$. Left coset $(1\ 2)N = \{(1\ 2), (1\ 2)(1\ 2\ 3), (1\ 2)(1\ 3\ 2)\} = \{(1\ 2), (2\ 3), (1\ 3)\}$. Right coset $N(1\ 2) = \{(1\ 2), (1\ 3), (2\ 3)\}$. Same set, so $N$ is normal in $S_3$. Index is $2$, so this fits the index-$2$ rule.

**Numerical example: a non-normal subgroup made visible.** Take $H = \{e, (1\ 2)\} \le S_3$. Then $(1\ 3)H = \{(1\ 3), (1\ 2\ 3)\}$, $H(1\ 3) = \{(1\ 3), (1\ 3\ 2)\}$. The two cosets contain three distinct elements between them: $(1\ 3), (1\ 2\ 3), (1\ 3\ 2)$. The conjugate $(1\ 3)H(1\ 3)^{-1} = (1\ 3)\{e, (1\ 2)\}(1\ 3) = \{e, (2\ 3)\} \neq H$. So conjugation moves $H$ to a different subgroup of $S_3$. There are three conjugate subgroups of order 2 in $S_3$, none normal.

**Tests for normality:**

- **Index 2 test:** If $[G : N] = 2$, then $N$ is automatically normal.
- **Center test:** $Z(G) = \{z : zg = gz \text{ for all } g\}$ is always normal.
- **Kernel test:** Show $N = \ker \varphi$ for some homomorphism $\varphi$.
- **Commutator test:** $[G, G] = \langle g^{-1}h^{-1}gh \rangle$ is always normal, and any subgroup containing it is normal.

**Worked check of the index-2 test.** $A_n$ has index $2$ in $S_n$, so $A_n \trianglelefteq S_n$. The rotation subgroup of $D_n$ has index $2$, so it is normal in $D_n$. The orientation-preserving isometries of $\mathbb{R}^n$ form an index-$2$ subgroup of all isometries, hence are normal. The pattern repeats: any "even-vs-odd" split is automatically a normal subgroup.

![Commutator subgroup and abelianization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_commutator.png)

**A worked non-test.** Consider the permutation $\sigma = (1\ 2)$ in $S_4$. The cyclic subgroup $\langle \sigma \rangle = \{e, (1\ 2)\}$ has order $2$, but its index in $S_4$ is $12$, not $2$. So the index-$2$ test does not apply. In fact $\langle \sigma \rangle$ is not normal: $(1\ 3)(1\ 2)(1\ 3)^{-1} = (2\ 3) \notin \langle \sigma \rangle$.

**Why normality matters beyond quotients.** Normal subgroups appear everywhere in algebra. A group is *simple* if it has no proper non-trivial normal subgroups. Simple groups are the building blocks of finite group theory in the same way primes are the building blocks of integers. The classification of finite simple groups (completed in the 2000s after $\sim 10000$ pages of proofs) lists every simple group, and the Jordan-Hölder theorem says every finite group is built from simple groups via successive normal extensions.

A second reason normality matters: the *normalizer* $N_G(H) = \{g \in G : gHg^{-1} = H\}$ is the largest subgroup of $G$ in which $H$ is normal. Every subgroup is normal in its normalizer, so the question "is $H$ normal in $G$?" is the same as "is $N_G(H) = G$?" The normalizer concept will be central to Sylow theory in the next article.

**Concrete count: number of normal subgroups.** A small group can have very few. $S_3$ has only three normal subgroups: $\{e\}, A_3, S_3$. $S_4$ has four: $\{e\}, V_4, A_4, S_4$. $A_5$ has only two: $\{e\}, A_5$ --- this is the smallest non-abelian simple group. The dramatic shrinkage of the list of normal subgroups as $n$ grows is one of the structural surprises of finite group theory.

## The Quotient Group Construction

Mental picture: pick a normal subgroup $N$. Collapse each coset $gN$ to a single point. The collapsed points form a new group $G/N$, with operation inherited from $G$. The points of $G/N$ are equivalence classes, where $g_1 \sim g_2$ iff $g_1 g_2^{-1} \in N$.

![Quotient group: collapsing a normal subgroup](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_quotient_group.png)

**Construction.** Let $N \trianglelefteq G$. Define $G/N$ to be the set of left cosets of $N$:

$$G/N = \{gN : g \in G\}$$

with operation $(g_1 N)(g_2 N) = (g_1 g_2)N$.

**This is well-defined precisely because $N$ is normal.** Suppose $g_1 N = g_1' N$ and $g_2 N = g_2' N$. We need $(g_1 g_2)N = (g_1' g_2')N$. Write $g_1' = g_1 n_1$ and $g_2' = g_2 n_2$ with $n_1, n_2 \in N$. Then $g_1' g_2' = g_1 n_1 g_2 n_2$. The trick: $n_1 g_2 = g_2 (g_2^{-1} n_1 g_2)$, and $g_2^{-1} n_1 g_2 \in N$ by normality. So $g_1' g_2' = g_1 g_2 \cdot (g_2^{-1} n_1 g_2) n_2 \in g_1 g_2 N$. $\checkmark$

If $N$ were not normal, this argument would fail at the step $g_2^{-1} n_1 g_2 \in N$.

**Group axioms in $G/N$.** Identity: $eN = N$. Inverse of $gN$: $(gN)^{-1} = g^{-1}N$. Associativity: inherited from $G$.

**Order:** $|G/N| = [G:N] = |G|/|N|$ when $G$ is finite.

**A useful counting reflex.** When $G/N$ has small order, every element of $G/N$ has order dividing $|G/N|$. So if $|G/N| = 2$, every coset squared is the identity coset, meaning $g^2 \in N$ for every $g \in G$. This kind of "easy upper bound from the quotient" is one of the most-used tricks in elementary group theory.

![Cosets of 2Z inside Z and the resulting quotient](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_2_cosets_z4.png)

**Worked example: $\mathbb{Z}/4\mathbb{Z}$ as $\mathbb{Z}/4\mathbb{Z}$.** Take $G = \mathbb{Z}$, $N = 4\mathbb{Z}$. The cosets are $0 + 4\mathbb{Z}, 1 + 4\mathbb{Z}, 2 + 4\mathbb{Z}, 3 + 4\mathbb{Z}$. Operation: $(2 + 4\mathbb{Z}) + (3 + 4\mathbb{Z}) = 5 + 4\mathbb{Z} = 1 + 4\mathbb{Z}$. The quotient has $4$ elements with the standard addition mod $4$.

**Worked example: $D_4 / \langle r^2 \rangle$.** $\langle r^2 \rangle = \{e, r^2\}$ is the center of $D_4$, and the center is always normal. The quotient has order $8/2 = 4$. Cosets: $\{e, r^2\}, \{r, r^3\}, \{s, r^2 s\}, \{rs, r^3 s\}$. Multiplication: $\{r, r^3\} \cdot \{r, r^3\} = \{r^2, e, e, r^2\} = \{e, r^2\}$, the identity coset. Each non-identity coset squares to the identity, so $D_4/\langle r^2 \rangle \cong V_4$ (Klein four).

**Worked example: $S_3/A_3$.** $A_3 = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$ is normal in $S_3$ (index 2). The two cosets are $A_3$ (the even permutations) and $(1\ 2)A_3$ (the odd permutations). Coset multiplication: even $\cdot$ even = even, even $\cdot$ odd = odd, odd $\cdot$ odd = even. This is exactly $\mathbb{Z}/2\mathbb{Z}$ as a group.

**Worked example: $\mathbb{Z}^2/\langle (1, 1) \rangle$.** Take $G = \mathbb{Z}^2$ and let $N$ be the cyclic subgroup generated by $(1, 1)$, namely $\{(k, k) : k \in \mathbb{Z}\}$. The quotient is $\mathbb{Z}^2/N$. Each coset $(a, b) + N$ corresponds to the value $a - b$, which can be any integer. So $\mathbb{Z}^2/N \cong \mathbb{Z}$. We will redo this computation more carefully below.

**Worked example: $\mathbb{Z}^2/\langle (2, 0), (0, 3) \rangle$.** Take $N = 2\mathbb{Z} \times 3\mathbb{Z}$. The quotient is $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/3\mathbb{Z}$, of order $6$. Since $\gcd(2,3) = 1$, this is also $\cong \mathbb{Z}/6\mathbb{Z}$ by CRT. Numerical reading: classes are $(\bar{0}, \bar{0}), (\bar{1}, \bar{0}), (\bar{0}, \bar{1}), (\bar{1}, \bar{1}), (\bar{0}, \bar{2}), (\bar{1}, \bar{2})$. Six classes.

**Why this matters.** The quotient construction is the algebraic analogue of taking a quotient space in topology. It is also the technical foundation for "modding out" in any algebraic setting: rings, modules, vector spaces. The normality requirement is the algebraic version of the requirement that a subspace be closed under the relevant equivalence relation.

A pedagogical note: students often struggle with quotient groups because the elements ("cosets") are themselves sets, not points. Once you internalize that $gN$ is a single object in the quotient (despite being a set of objects in $G$), most of the conceptual difficulty evaporates. A useful exercise: write out the multiplication table of $\mathbb{Z}/4\mathbb{Z}$ as cosets of $4\mathbb{Z}$ in $\mathbb{Z}$, with each cell being an explicit coset like $\{1, 5, -3, 9, \ldots\}$. After two or three rows, the abstraction crystallizes.

## Group Homomorphisms

Mental picture: a homomorphism is a function from one group to another that respects the multiplication. It is the right notion of "structure-preserving map" between groups, and the key tool for comparing groups.

![Homomorphism preserves group structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_homomorphism.png)

**Definition.** A *homomorphism* is a function $\varphi: G \to H$ between groups satisfying $\varphi(g_1 g_2) = \varphi(g_1)\varphi(g_2)$ for all $g_1, g_2 \in G$.

**Basic properties.** $\varphi(e_G) = e_H$ (proof: $\varphi(e_G) = \varphi(e_G \cdot e_G) = \varphi(e_G)\varphi(e_G)$, then cancel). And $\varphi(g^{-1}) = \varphi(g)^{-1}$ (proof: $\varphi(g)\varphi(g^{-1}) = \varphi(g g^{-1}) = \varphi(e) = e$).

A consequence worth noting: a homomorphism preserves orders in the divisor sense. If $g \in G$ has order $n$, then $\varphi(g)$ has order dividing $n$ in $H$. (Reason: $\varphi(g)^n = \varphi(g^n) = \varphi(e) = e$.) But the order can drop: in a non-injective homomorphism, $\varphi(g)$ might have strictly smaller order. Concretely, the projection $\mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$ sends $1$ (infinite order) to $1$ (order $4$).

**Kernel and image.**

- $\ker \varphi = \{g \in G : \varphi(g) = e_H\}$ --- a subgroup of $G$, in fact normal.
- $\text{im}(\varphi) = \{\varphi(g) : g \in G\}$ --- a subgroup of $H$ (not normal in general).

![Kernel as preimage of identity, image as range of homomorphism](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_5_kernel_image.png)

**Example 1: Natural projection.** For $N \trianglelefteq G$, the map $\pi: G \to G/N$ with $\pi(g) = gN$ is a surjective homomorphism with kernel $N$.

**Example 2: Sign homomorphism.** $\text{sgn}: S_n \to \{\pm 1\}$. Kernel: $A_n$.

**Example 3: Exponentiation.** $\varphi: \mathbb{Z} \to G$, $\varphi(n) = g^n$ for fixed $g \in G$. Image: $\langle g \rangle$. Kernel: $\{0\}$ if $|g| = \infty$, else $k\mathbb{Z}$ where $k = |g|$.

**Numerical example.** $\varphi: \mathbb{Z}/12\mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$ by $\varphi(\bar{k}) = k \bmod 4$. Kernel: $\{0, 4, 8\}$. Image: all of $\mathbb{Z}/4\mathbb{Z}$. Check: $\varphi(7 + 5) = \varphi(0) = 0$, $\varphi(7) + \varphi(5) = 3 + 1 = 4 \equiv 0$. Match.

**Numerical example: a non-trivial endomorphism.** Define $\varphi: \mathbb{Z}/6\mathbb{Z} \to \mathbb{Z}/6\mathbb{Z}$ by $\varphi(k) = 2k \bmod 6$. Check it is a homomorphism: $\varphi(a + b) = 2(a+b) = 2a + 2b = \varphi(a) + \varphi(b)$. Yes. Kernel: $\{k : 2k \equiv 0 \pmod 6\} = \{0, 3\}$. Image: $\{0, 2, 4\}$, an order-$3$ subgroup. By First Isomorphism: $\mathbb{Z}/6\mathbb{Z}/\{0, 3\} \cong \{0, 2, 4\}$, both groups of order $3$.

![Injective vs surjective vs isomorphism comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_6_homo_types.png)

**Types of homomorphisms.**

- *Monomorphism* (injective): $\ker \varphi = \{e\}$.
- *Epimorphism* (surjective): $\text{im}(\varphi) = H$.
- *Isomorphism* (bijective): both.
- *Automorphism*: isomorphism $G \to G$.
- *Endomorphism*: homomorphism $G \to G$.

**Why this matters.** Homomorphisms are the morphisms in the category of groups. Almost every theorem about groups is naturally stated as a property of homomorphisms (not of groups in isolation). The category-theoretic perspective will become explicit in Article 11.

**A useful lemma: composition.** The composition of two homomorphisms is a homomorphism. If $\varphi: G \to H$ and $\psi: H \to K$ are homomorphisms, then $\psi \circ \varphi: G \to K$ is too. Kernels behave functorially: $\ker(\psi \circ \varphi) \supseteq \ker\varphi$, with equality iff $\psi$ is injective on $\text{im}(\varphi)$. This is the algebraic ancestor of the chain rule for derivatives (which is itself a statement about composition of linear maps, in turn a special case of compositions of homomorphisms in the additive category).

**Numerical example: composing.** Let $\varphi: \mathbb{Z} \to \mathbb{Z}/12\mathbb{Z}$ and $\psi: \mathbb{Z}/12\mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$, both reduction maps. Compose: $\psi \circ \varphi: \mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$ is reduction mod $4$. Kernels: $\ker\varphi = 12\mathbb{Z}$, $\ker(\psi\circ\varphi) = 4\mathbb{Z}$. As expected, $12\mathbb{Z} \subseteq 4\mathbb{Z}$.

## The First Isomorphism Theorem

Mental picture: collapsing a group by its kernel gives exactly the image. A homomorphism $\varphi$ "factors" canonically as a surjective projection $G \to G/\ker\varphi$ followed by an isomorphism $G/\ker\varphi \to \text{im}(\varphi)$.

![First isomorphism theorem: G/ker(f) ≅ im(f)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_first_isomorphism.png)

**Theorem (First Isomorphism Theorem).** Let $\varphi: G \to H$ be a group homomorphism. Then

$$G / \ker \varphi \cong \text{im}(\varphi).$$

The map $\bar{\varphi}: G/\ker\varphi \to \text{im}(\varphi)$ defined by $\bar{\varphi}(g \ker\varphi) = \varphi(g)$ is a well-defined isomorphism.

**Proof.** Let $K = \ker \varphi$.

*Well-defined:* If $g_1 K = g_2 K$, then $g_1^{-1} g_2 \in K$, so $\varphi(g_1)^{-1}\varphi(g_2) = e$, hence $\varphi(g_1) = \varphi(g_2)$.

*Homomorphism:* $\bar{\varphi}(g_1 K \cdot g_2 K) = \bar{\varphi}(g_1 g_2 K) = \varphi(g_1 g_2) = \varphi(g_1)\varphi(g_2) = \bar{\varphi}(g_1 K)\bar{\varphi}(g_2 K)$.

*Injective:* $\bar{\varphi}(gK) = e \implies \varphi(g) = e \implies g \in K \implies gK = K$.

*Surjective:* For any $h \in \text{im}(\varphi)$, choose $g$ with $\varphi(g) = h$; then $\bar{\varphi}(gK) = h$. $\square$

![First Isomorphism Theorem as a commutative triangle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_4_first_iso_diagram.png)

**Why this matters.** This theorem is the structural backbone of the subject. Every quotient group is the image of some homomorphism; every image is a quotient. This duality lets us turn questions about quotients into questions about homomorphisms (and vice versa), and it underlies the rank-nullity theorem in linear algebra and many other "structural division" results.

In categorical terms, this theorem says that every morphism factors uniquely as an epi followed by a mono. For groups, that factorization is $G \twoheadrightarrow G/\ker\varphi \hookrightarrow H$. The same structure shows up in vector spaces (rank-nullity), modules (the snake lemma), and topological spaces (orbit decompositions). When you see "first isomorphism theorem" anywhere in algebra, it is the same theorem in a different category.

**Application 1: $GL_n(\mathbb{R})/SL_n(\mathbb{R}) \cong \mathbb{R}^*$.** The determinant $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ is a surjective homomorphism with kernel $SL_n(\mathbb{R})$.

**Application 2: $\mathbb{R}/\mathbb{Z} \cong S^1$.** Define $\varphi: \mathbb{R} \to \mathbb{C}^*$ by $\varphi(t) = e^{2\pi i t}$. Surjective onto the unit circle $S^1$ with kernel $\mathbb{Z}$. So $\mathbb{R}/\mathbb{Z} \cong S^1$. The quotient of the real line by the integers wraps around to form a circle, both algebraically and topologically.

**Application 3: $(\mathbb{Z}/12\mathbb{Z})/(3\mathbb{Z}/12\mathbb{Z}) \cong \mathbb{Z}/3\mathbb{Z}$.** With $\varphi: \mathbb{Z}/12\mathbb{Z} \to \mathbb{Z}/3\mathbb{Z}$ by $\varphi(\bar{k}) = k \bmod 3$: kernel is $\{0, 3, 6, 9\}$, which is $3\mathbb{Z}/12\mathbb{Z}$, image is $\mathbb{Z}/3\mathbb{Z}$.

**Application 4: classifying cyclic quotients.** For cyclic groups, every quotient is cyclic. Specifically, $\mathbb{Z}/n\mathbb{Z}$ has a quotient isomorphic to $\mathbb{Z}/m\mathbb{Z}$ exactly when $m \mid n$. The relevant homomorphism is $\bar{k} \mapsto k \bmod m$ with kernel $\langle \bar{m} \rangle$ of order $n/m$. Concretely: quotients of $\mathbb{Z}/12\mathbb{Z}$ are $\mathbb{Z}/12\mathbb{Z}, \mathbb{Z}/6\mathbb{Z}, \mathbb{Z}/4\mathbb{Z}, \mathbb{Z}/3\mathbb{Z}, \mathbb{Z}/2\mathbb{Z}, \{0\}$, mirroring the divisors $12, 6, 4, 3, 2, 1$ of $12$.

**Worked Example: proving non-simplicity.** A group is *simple* if it has no proper non-trivial normal subgroups. The First Isomorphism Theorem provides a non-simplicity test: find any non-trivial homomorphism $\varphi: G \to H$ with $|H| < |G|$. Then $\ker\varphi$ is a non-trivial proper normal subgroup. Application: any group of order $6$ with a normal subgroup of order $3$ (which exists by Sylow theory in the next article) gives a homomorphism $G \to S_3$ via the conjugation action on cosets, and analyzing kernel/image shows $G$ is either $\mathbb{Z}/6\mathbb{Z}$ or $S_3$.

## Second and Third Isomorphism Theorems

The First Isomorphism Theorem has two important companions describing how subgroups and quotients interact.

**Second Isomorphism Theorem (Diamond).** Let $G$ be a group, $H \leq G$, $N \trianglelefteq G$. Then:

1. $HN = \{hn : h \in H, n \in N\}$ is a subgroup of $G$.
2. $H \cap N \trianglelefteq H$.
3. $HN / N \cong H / (H \cap N)$.

**Proof sketch.** Define $\varphi: H \to G/N$ by $\varphi(h) = hN$. Kernel: $H \cap N$. Image: $HN/N$. Apply First Isomorphism. $\square$

**Intuition.** Think of $H$ and $N$ as overlapping lenses on $G$. What $H$ sees in the quotient $G/N$ ($= HN/N$) equals what $H$ sees when it removes its overlap with $N$ ($= H/(H \cap N)$).

**Third Isomorphism Theorem.** Let $N \trianglelefteq K \trianglelefteq G$ (both normal in $G$). Then:

1. $K/N \trianglelefteq G/N$.
2. $(G/N)/(K/N) \cong G/K$.

**Proof sketch.** $\varphi: G/N \to G/K$ by $\varphi(gN) = gK$ (well-defined since $N \subseteq K$). Surjective. Kernel: $K/N$. Apply First Isomorphism. $\square$

**Intuition.** Quotienting out a quotient is the same as quotienting out all at once.

**The Lattice Correspondence Theorem (Fourth Isomorphism Theorem).** Let $N \trianglelefteq G$. There is a bijection between:

- subgroups of $G/N$, and
- subgroups of $G$ containing $N$,

via $H/N \leftrightarrow H$ (for $N \leq H \leq G$). The bijection preserves containment, normality, and index.

![Correspondence theorem: subgroups containing N match subgroups of G/N](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_v2_03_7_correspondence.png)

This is extraordinarily useful. Taking a quotient does not destroy subgroup structure --- it merely "forgets" everything below $N$. The subgroup lattice of $G/N$ is the portion of the lattice of $G$ lying above $N$.

**Numerical example: subgroup lattice of $\mathbb{Z}/12\mathbb{Z}$ above $\langle 4 \rangle$.** $\langle 4 \rangle = \{0, 4, 8\}$. Subgroups of $\mathbb{Z}/12\mathbb{Z}$ containing $\langle 4 \rangle$: $\langle 4 \rangle$, $\langle 2 \rangle = \{0, 2, 4, 6, 8, 10\}$, $\mathbb{Z}/12\mathbb{Z}$. The quotient $\mathbb{Z}/12\mathbb{Z} / \langle 4 \rangle \cong \mathbb{Z}/4\mathbb{Z}$ has subgroups $\{0\}$, $\{0, 2\}$, $\mathbb{Z}/4\mathbb{Z}$. Three above, three below. Match.

**A second example to illustrate the correspondence.** Take $G = D_4$, $N = \langle r^2 \rangle = Z(D_4)$. Subgroups of $D_4$ containing $N$: $N$ itself (order 2), $\langle r \rangle$ (order 4), $\{e, r^2, s, r^2 s\}$ (order 4), $\{e, r^2, rs, r^3 s\}$ (order 4), $D_4$ (order 8). Five subgroups. The quotient $D_4/N \cong V_4$ has exactly $5$ subgroups: $\{e\}$, three order-2 subgroups, and $V_4$. Match.

**Worked Example.** $G = \mathbb{Z}, K = 6\mathbb{Z}, N = 30\mathbb{Z}$. Then $N \subseteq K$ both normal. Third Isomorphism: $(\mathbb{Z}/30\mathbb{Z})/(6\mathbb{Z}/30\mathbb{Z}) \cong \mathbb{Z}/6\mathbb{Z}$. Here $6\mathbb{Z}/30\mathbb{Z}$ has $5$ elements ($\{0, 6, 12, 18, 24\} \bmod 30$), so we get $\mathbb{Z}_{30}$ modded by a copy of $\mathbb{Z}_5$ giving $\mathbb{Z}_6$. Numerically: $30/5 = 6$. $\checkmark$

**Worked Example: $S_4$ has chain $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$.** Sizes $1, 4, 12, 24$. Successive quotients: $V_4/\{e\} \cong V_4$, $A_4/V_4 \cong \mathbb{Z}/3\mathbb{Z}$, $S_4/A_4 \cong \mathbb{Z}/2\mathbb{Z}$. Composition factors: $\mathbb{Z}/2\mathbb{Z}, \mathbb{Z}/2\mathbb{Z}, \mathbb{Z}/3\mathbb{Z}, \mathbb{Z}/2\mathbb{Z}$. Their product of orders: $2 \cdot 2 \cdot 3 \cdot 2 = 24 = |S_4|$. $\checkmark$ This is the kind of analysis the Jordan-Hölder theorem makes systematic.

**Worked Example: a non-trivial use of the Second Isomorphism Theorem.** In $S_4$, take $H = \langle (1\ 2\ 3\ 4) \rangle$ (cyclic of order 4) and $N = V_4 = \{e, (1\ 2)(3\ 4), (1\ 3)(2\ 4), (1\ 4)(2\ 3)\}$. Then $H \cap N = \{e, (1\ 3)(2\ 4)\}$ (the unique non-identity element of $H$ that is also in $V_4$ is $(1\ 2\ 3\ 4)^2 = (1\ 3)(2\ 4)$). $|H| = 4, |N| = 4, |H \cap N| = 2$, so $|HN| = |H| \cdot |N| / |H \cap N| = 16/2 = 8$. Therefore $HN$ has order $8$, and $HN/N$ has order $2$. By the Second Isomorphism, $H/(H \cap N) \cong HN/N$, both of order $2$. Match. The group $HN$ turns out to be a copy of $D_4$ inside $S_4$.

**Worked Example: kernel of a matrix homomorphism.** Define $\varphi: GL_2(\mathbb{R}) \to \mathbb{R}^*$ by $\varphi(A) = \det(A)$. Surjective with kernel $SL_2(\mathbb{R})$. So $GL_2(\mathbb{R})/SL_2(\mathbb{R}) \cong \mathbb{R}^*$. Now consider the chain $\{I\} \subset \{\pm I\} \subset SL_2(\mathbb{R}) \subset GL_2(\mathbb{R})$. Successive quotients: $\{\pm I\}/\{I\} \cong \mathbb{Z}/2\mathbb{Z}$, $SL_2/\{\pm I\} = PSL_2(\mathbb{R})$, $GL_2/SL_2 \cong \mathbb{R}^*$. The middle quotient $PSL_2(\mathbb{R})$ is the projective special linear group, fundamental in hyperbolic geometry.

**Worked Example: counting normal subgroups of $D_4$.** $D_4$ has $10$ subgroups (see Article 1). Among them, $5$ are normal: $\{e\}, \langle r^2 \rangle, \langle r \rangle, \{e, r^2, s, r^2 s\}, \{e, r^2, rs, r^3 s\}, D_4$. Wait that is $6$. Let me recount: $\{e\}$ trivial, $\langle r^2 \rangle$ is the center (normal), $\langle r \rangle$ has index $2$ (normal), the two Klein four subgroups containing $r^2$ have index $2$ (normal), $D_4$ itself. Six normal subgroups. The five reflections-only subgroups of order 2 are not normal. By the lattice correspondence, the $6$ normal subgroups of $D_4$ correspond to the $6$ subgroups of $D_4/\{e\} = D_4$ that contain $\{e\}$ --- i.e., all subgroups, which is $10$. So the lattice correspondence does not separate "normal" from "not normal" in the obvious way. The correct statement: normal subgroups of $G/N$ correspond to normal subgroups of $G$ containing $N$. So $D_4/\{e\}$ has the same normal subgroups as $D_4$ does. Both have $6$.

**Why the isomorphism theorems matter in practice.** The Chinese Remainder Theorem is a quick consequence: if $\gcd(m, n) = 1$, then $\mathbb{Z}/mn\mathbb{Z} \cong \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$. Apply First Isomorphism to $\varphi: \mathbb{Z} \to \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$ by $\varphi(k) = (k \bmod m, k \bmod n)$. Surjective when $\gcd(m,n) = 1$, kernel $mn\mathbb{Z}$. The general form of CRT, which generalizes to ring theory, follows the same template.

**Numerical CRT.** Take $m = 4, n = 9$, $\gcd = 1$. $\mathbb{Z}/36\mathbb{Z} \cong \mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/9\mathbb{Z}$. The element $13 \in \mathbb{Z}/36\mathbb{Z}$ corresponds to $(13 \bmod 4, 13 \bmod 9) = (1, 4)$. Adding $13 + 25 = 38 \equiv 2 \pmod{36}$, which corresponds to $(2 \bmod 4, 2 \bmod 9) = (2, 2)$. On the right side: $(1, 4) + (1, 7) = (2, 11) = (2, 2)$ since $11 \bmod 9 = 2$. Match.

The CRT is one of the oldest theorems in number theory, going back to Sun Tzu's *Sunzi Suanjing* in the 3rd century. The modern phrasing as an isomorphism of rings is a 20th-century reformulation that makes its structural content immediate.

In algebraic topology, the fundamental group of a quotient space is often a quotient of the original. Covering spaces of a torus are classified by subgroups of $\pi_1(T^2) = \mathbb{Z}^2$, organized via the Lattice Correspondence.

**A practical computation: solvability of $S_4$.** A group is *solvable* if it has a chain $G = G_0 \trianglerighteq G_1 \trianglerighteq \cdots \trianglerighteq G_n = \{e\}$ with abelian successive quotients. For $S_4$, take $S_4 \trianglerighteq A_4 \trianglerighteq V_4 \trianglerighteq \{e\}$. Quotients: $S_4/A_4 \cong \mathbb{Z}/2$, $A_4/V_4 \cong \mathbb{Z}/3$, $V_4/\{e\} \cong V_4$ (which is abelian). So $S_4$ is solvable. By contrast, $S_5$ is not solvable: $A_5$ is simple non-abelian, so any subnormal chain reaches $A_5$ as a composition factor, and $A_5$ is not abelian. This is the heart of why the general quintic is unsolvable by radicals.

**Why solvability matters.** Solvability is the bridge between group theory and Galois theory. A polynomial is solvable by radicals iff its Galois group is solvable. The unsolvability of the general quintic comes from $A_5$ being simple non-abelian, which makes $S_5$ non-solvable. We will return to this in Articles 7 and 8.

## A Worked Example: $(\mathbb{Z} \times \mathbb{Z})/H$

Take $G = \mathbb{Z} \times \mathbb{Z}$ and $H = \{(a, a) : a \in \mathbb{Z}\}$, the diagonal subgroup. Both groups are abelian, so $H$ is automatically normal. We compute the quotient.

**Step 1: identify cosets.** A coset is $(m, n) + H = \{(m+a, n+a) : a \in \mathbb{Z}\}$. Two cosets $(m_1, n_1) + H$ and $(m_2, n_2) + H$ are equal iff their difference $(m_1 - m_2, n_1 - n_2)$ is in $H$, i.e., iff $m_1 - m_2 = n_1 - n_2$, i.e., iff $m_1 - n_1 = m_2 - n_2$.

So each coset is uniquely determined by the integer $m - n$.

**Step 2: identify the quotient with $\mathbb{Z}$.** The map $\Phi: G/H \to \mathbb{Z}$ given by $(m, n) + H \mapsto m - n$ is a bijection. Check it is a homomorphism: $\Phi((m_1, n_1) + (m_2, n_2) + H) = (m_1 + m_2) - (n_1 + n_2) = (m_1 - n_1) + (m_2 - n_2) = \Phi(\cdot) + \Phi(\cdot)$. Yes.

**Conclusion.** $(\mathbb{Z} \times \mathbb{Z})/H \cong \mathbb{Z}$.

**Sanity check via the First Isomorphism Theorem.** Define $\varphi: \mathbb{Z} \times \mathbb{Z} \to \mathbb{Z}$ by $\varphi(m, n) = m - n$. Surjective. Kernel: $\{(m, n) : m = n\} = H$. By the theorem, $(\mathbb{Z} \times \mathbb{Z})/H \cong \mathbb{Z}$. Identical answer, two-line proof.

**Numerical illustration.** Coset of $(7, 3)$: corresponds to $7 - 3 = 4$. Coset of $(2, -2)$: corresponds to $2 - (-2) = 4$. So $(7, 3)$ and $(2, -2)$ live in the same coset. Adding cosets: $(7, 3) + (1, 5) + H$ has class $7 + 1 - 3 - 5 = 0$, the identity coset, which is $H$ itself. Match: $(8, 8) \in H$. $\checkmark$

**Geometric picture.** Visualize $\mathbb{Z}^2$ as the integer lattice in the plane. The subgroup $H$ is the points on the line $y = x$. Cosets are lines parallel to $y = x$ at integer offsets, indexed by $m - n$. Quotienting by $H$ collapses each diagonal to a point, and the resulting space is the integer lattice on the orthogonal axis $y = -x$, which is just $\mathbb{Z}$.

**Variant: $(\mathbb{Z} \times \mathbb{Z})/\langle (1, 0) \rangle$.** $\langle (1, 0) \rangle = \mathbb{Z} \times \{0\}$. Quotient: $(\mathbb{Z}^2)/(\mathbb{Z} \times \{0\}) \cong \mathbb{Z}$, this time via the projection $(m, n) \mapsto n$.

**Variant: $(\mathbb{Z} \times \mathbb{Z})/\langle (2, 3) \rangle$.** Now $H$ is generated by a single element of "infinite order" inside the lattice. Its index in $\mathbb{Z}^2$ is infinite. The quotient $\mathbb{Z}^2/H$ is isomorphic to $\mathbb{Z}$, since by Bezout (or Smith normal form) we can find a complementary direction. Specifically, since $\gcd(2, 3) = 1$, there exist $a, b$ with $2a + 3b = 1$ (e.g., $a = -1, b = 1$). The map $(m, n) \mapsto am - bn = -m - n$... let me redo. The quotient computation via Smith normal form is the cleanest approach: row-reduce the relation matrix $(2, 3)$ to $(1)$, giving quotient $\mathbb{Z}$. Whatever specific isomorphism is chosen, the answer is $\mathbb{Z}$.

## Historical Notes on Galois and Noether

The concept of quotient groups, though not formulated in modern language, has roots in the work of Galois and Noether.

Évariste Galois (1811-1832) introduced the idea of a normal subgroup implicitly when studying solvability of polynomial equations by radicals. Galois showed that solvability is governed by the existence of a chain of subgroups $G \trianglerighteq G_1 \trianglerighteq G_2 \trianglerighteq \cdots \trianglerighteq \{e\}$ with abelian successive quotients --- exactly what we now call a *solvable group*. He did not use the words "normal subgroup" or "quotient," but the geometry of his correspondence between intermediate fields and subgroups of the Galois group makes them implicit.

Galois died at age 20 in a duel, leaving his major work in a hastily written letter on the night before. The mathematical community took decades to digest his ideas. Liouville published Galois's papers in 1846, fourteen years after his death. Jordan's *Traité des substitutions* (1870) was the first systematic exposition.

Emmy Noether (1882-1935) gave the abstract definitions we use today and proved the isomorphism theorems in their modern form. Her textbook style of "definition-theorem-proof" reorganized algebra around homomorphisms and quotients as primary objects, displacing the older calculation-heavy approach. Modern algebra textbooks (Lang, Dummit-Foote, Hungerford) descend directly from Noether's reorganization. She also formulated Noether's theorem in physics, which connects continuous symmetries to conservation laws --- a different but equally fundamental way that quotient structures arise.

Noether's career ran into substantial institutional barriers: she was unpaid for years at Göttingen because the German university system did not allow women to be professors. She lectured under Hilbert's name on his offer ("I do not see that the sex of the candidate is an argument against her admission"). After 1933 she was dismissed by the Nazi regime and emigrated to Bryn Mawr in the US, where she died of complications from surgery in 1935.

## What's Next

We now have a powerful toolkit: normal subgroups let us build quotients, homomorphisms let us compare groups, and the isomorphism theorems reveal deep structural connections. The next article takes on a different challenge: given a finite group $G$, how can we find subgroups of specific orders? The *Sylow theorems* provide the answer, giving existence, conjugacy, and counting results for subgroups of prime-power order. They are the sharpest tool available for classifying finite groups, and they build directly on the normal subgroup machinery developed here.

![Animation: cosets collapsing into quotient elements](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/03_coset_collapse.gif)

A summary worth keeping: quotient groups are about deliberate forgetting; homomorphisms are about systematic comparison; the isomorphism theorems are the precise statement that forgetting and comparing are two sides of the same coin.

**Reading recommendations.** For a thorough treatment of the isomorphism theorems with many examples, see Dummit and Foote chapter 3. For a categorical perspective, see Mac Lane's *Categories for the Working Mathematician* or Awodey's *Category Theory*. For the historical development, Kiernan's article "The development of Galois theory from Lagrange to Artin" (1971) traces the evolution of the normal subgroup concept across a century.

**One last reflection.** The isomorphism theorems are sometimes presented as "abstract nonsense," but they are anything but: every concrete computation in finite group theory --- from listing groups of order $100$ to deciding whether two given matrices generate the same group --- depends on them. They are the syntax that makes group theory writable; learn to use them fluently and you will find that most "advanced" results are surprisingly direct consequences.

## Deeper Dive: Quotients and Homomorphisms in Practice

The slogan of this article is that quotients and homomorphisms are two sides of one coin: every quotient $G/N$ comes with a canonical surjection $G \to G/N$, and every homomorphism $\varphi: G \to H$ factors as $G \twoheadrightarrow G/\ker\varphi \hookrightarrow H$. Below are computations that make the slogan concrete.

**Computation A: a non-normal subgroup in $S_3$.** Take $G = S_3$ and $H = \{e, (1\ 2)\}$, of order $2$. Is $H$ normal? Compute $g H g^{-1}$ for $g = (1\ 2\ 3)$: $g e g^{-1} = e$, and $g(1\ 2)g^{-1} = (g(1)\ g(2)) = (2\ 3)$. So $g H g^{-1} = \{e, (2\ 3)\}$, which is *not* $H$. Therefore $H$ is not normal in $S_3$. The left coset $g H = \{(1\ 2\ 3), (1\ 2\ 3)(1\ 2)\} = \{(1\ 2\ 3), (1\ 3)\}$, but the right coset $H g = \{(1\ 2\ 3), (1\ 2)(1\ 2\ 3)\} = \{(1\ 2\ 3), (2\ 3)\}$. Different sets. Multiplication of cosets is not well-defined.

This is exactly why the definition of normal subgroup says "$g N g^{-1} = N$ for *every* $g$." If we only required "for some $g$" or "for $g$ in a generating set," counterexamples like the above would slip through. The formulation has been honed to be exactly strong enough to make $G/N$ a group.

**Computation B: $A_n$ is normal in $S_n$.** The sign map $\text{sgn}: S_n \to \{\pm 1\}$ is a homomorphism. Its kernel is $A_n$ (the even permutations). Kernels are always normal, so $A_n \trianglelefteq S_n$. The quotient $S_n / A_n$ has order $2$ (by the index formula: $|S_n|/|A_n| = 2$), and it is isomorphic to $\mathbb{Z}/2\mathbb{Z}$ — concretely, $\{A_n, (1\ 2) A_n\}$ with multiplication "even·even=even, even·odd=odd, odd·odd=even." The first isomorphism theorem says $S_n / A_n \cong \mathrm{im}(\text{sgn}) = \{\pm 1\}$.

**Computation C: a quotient of $\mathbb{Z}$.** Take the homomorphism $\varphi: \mathbb{Z} \to \mathbb{Z}/12\mathbb{Z} \times \mathbb{Z}/8\mathbb{Z}$ defined by $\varphi(n) = (n \bmod 12, n \bmod 8)$. The kernel is $\{n : 12 \mid n \text{ and } 8 \mid n\} = \mathrm{lcm}(12, 8) \mathbb{Z} = 24\mathbb{Z}$. So by the first isomorphism theorem, $\mathbb{Z}/24\mathbb{Z} \cong \mathrm{im}(\varphi) \subseteq \mathbb{Z}/12\mathbb{Z} \times \mathbb{Z}/8\mathbb{Z}$. The image has order $24$ and the codomain has order $96$, so the image is a strict subgroup. (The image misses pairs $(a, b)$ with $a \not\equiv b \pmod{\gcd(12, 8)} = \pmod 4$.)

**Computation D: $D_4 / Z(D_4)$ is the Klein four-group.** $Z(D_4) = \{e, r^2\}$ (computed in Part 1). The quotient $D_4 / Z(D_4)$ has order $8/2 = 4$. The four cosets are $\{e, r^2\}$, $\{r, r^3\}$, $\{s, r^2 s\}$, $\{rs, r^3 s\}$. Pick representatives $\bar e, \bar r, \bar s, \overline{rs}$. Compute: $\bar r^2 = \overline{r^2} = \bar e$ (since $r^2 \in Z(D_4)$); $\bar s^2 = \bar e$; $\overline{rs}^2 = \overline{rsrs} = \overline{rr^{-1}s^2} = \bar e$. All non-identity elements have order $2$, so the quotient is $\mathbb{Z}/2 \times \mathbb{Z}/2$, the Klein four-group. The general theorem behind this — "$G/Z(G)$ cyclic implies $G$ abelian" — has an elegant proof: if $G/Z(G) = \langle \bar g \rangle$, then every element of $G$ is $g^k z$ for some $z \in Z(G)$, and elements of that form commute with each other.

**Computation E: the second isomorphism theorem on $\mathbb{Z}$.** Take $G = \mathbb{Z}$, $H = 6\mathbb{Z}$, $N = 4\mathbb{Z}$. Both are normal (everything is in an abelian group). Then $H + N = 6\mathbb{Z} + 4\mathbb{Z} = \gcd(6,4)\mathbb{Z} = 2\mathbb{Z}$, $H \cap N = \mathrm{lcm}(6,4)\mathbb{Z} = 12\mathbb{Z}$. The second isomorphism theorem says $H/(H \cap N) \cong (H + N)/N$. Check: $6\mathbb{Z}/12\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z}$ (left side, order $2$); $2\mathbb{Z}/4\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z}$ (right side, order $2$). ✓

## Why the Definition of "Normal" Is Tuned the Way It Is

A normal subgroup $N \trianglelefteq G$ is, by definition, fixed by all inner automorphisms: $gNg^{-1} = N$ for every $g \in G$. Equivalently, $g N = N g$ for every $g$ (left and right cosets coincide). Equivalently again, $N$ is a union of conjugacy classes.

The intuition I keep coming back to: a subgroup is normal iff "modding out by it" is well-defined as an operation on cosets, i.e., $(gN)(hN) := (gh)N$ does not depend on the representatives. The check is short. Suppose $g_1 N = g_2 N$ and $h_1 N = h_2 N$, so $g_2 = g_1 n_1$ and $h_2 = h_1 n_2$ for some $n_i \in N$. Then $g_2 h_2 = g_1 n_1 h_1 n_2 = g_1 h_1 (h_1^{-1} n_1 h_1) n_2$. For this to lie in $g_1 h_1 N$, we need $h_1^{-1} n_1 h_1 \in N$, which is exactly normality. The definition is the *minimal* condition under which the quotient is a group. Anything weaker fails for some choice of representatives; anything stronger throws away information.

A crisp counterexample of failure: in $S_3$ with the non-normal $H = \{e, (1\ 2)\}$, attempt the multiplication $(1\ 3)H \cdot H = ?$. With representative $(1\ 3)$, we get $(1\ 3) \cdot e = (1\ 3) \in (1\ 3)H$. With representative $(1\ 3)(1\ 2) = (1\ 2\ 3)$, we get $(1\ 2\ 3) \cdot e = (1\ 2\ 3) \notin (1\ 3)H = \{(1\ 3), (1\ 2\ 3)\}$ — actually $(1\ 2\ 3)$ *is* in $(1\ 3)H$, let me recompute. $(1\ 3)H = \{(1\ 3) \cdot e, (1\ 3) \cdot (1\ 2)\} = \{(1\ 3), (1\ 3)(1\ 2)\}$, and $(1\ 3)(1\ 2)$ sends $1 \to 2, 2 \to 3, 3 \to 1$ wait let me just multiply: applied to $1$: $(1\ 2)$ sends $1 \to 2$, then $(1\ 3)$ fixes $2$. So $(1\ 3)(1\ 2)(1) = 2$. Applied to $2$: $(1\ 2)$ sends $2 \to 1$, then $(1\ 3)$ sends $1 \to 3$. So $2 \to 3$. Applied to $3$: $(1\ 2)$ fixes $3$, $(1\ 3)$ sends $3 \to 1$. So $3 \to 1$. The product is the cycle $(1\ 2\ 3)$. So $(1\ 3)H = \{(1\ 3), (1\ 2\ 3)\}$. Now try multiplying coset $(1\ 3)H$ by coset $(2\ 3)H$ using two different representatives: $(1\ 3)(2\ 3) = (1\ 3\ 2)$ vs $(1\ 2\ 3)(2\ 3) = (1\ 2)$ — different cosets. Quotient ill-defined.

## Common Pitfalls for Beginners

The single most common pitfall: equating "normal" with "abelian." A subgroup of an abelian group is automatically normal (every conjugation is trivial), but a normal subgroup need not be abelian, nor must its containing group be. $A_4$ is non-abelian, normal in $S_4$, and contains the non-abelian subgroups it contains. Conflating the two concepts will scramble half of group theory.

A second pitfall: thinking that if $H \trianglelefteq G$ and $K \trianglelefteq H$, then $K \trianglelefteq G$. False. Normality is not transitive. Counterexample: in $S_4$, the subgroup $V_4 = \{e, (1\ 2)(3\ 4), (1\ 3)(2\ 4), (1\ 4)(2\ 3)\}$ is normal in $A_4$ (and in $S_4$). Inside $V_4$, every subgroup is normal because $V_4$ is abelian — say $K = \{e, (1\ 2)(3\ 4)\}$. But $K$ is *not* normal in $S_4$: conjugating $(1\ 2)(3\ 4)$ by $(1\ 2\ 3)$ yields $(2\ 3)(1\ 4)$, which is not in $K$. Normality is a relative notion, and the chain $K \trianglelefteq V_4 \trianglelefteq S_4$ does not give $K \trianglelefteq S_4$.

A third pitfall: the surjection in the first isomorphism theorem is not optional. Beginners sometimes write "$G/\ker\varphi \cong H$" without checking that $\varphi$ is surjective. If $\varphi$ is not surjective, you only get $G/\ker\varphi \cong \mathrm{im}(\varphi)$, a *subgroup* of $H$. The image bit matters whenever you are using the theorem to *identify* a known quotient with a known group.

## Where This Shows Up

*Composition series and the Jordan–Hölder theorem.* Every finite group admits a composition series, a maximal chain of normal subgroups, and the simple quotients in any two such chains are the same up to permutation and isomorphism. This is a structure theorem about how groups are "made of" simple ones, and it is one of the foundational facts on which the classification of finite simple groups rests. It is also the reason solvability (as defined in Part 8) is so well-behaved.

*Group cohomology and extensions.* The data needed to reconstruct $G$ from a normal subgroup $N$ and the quotient $G/N$ is captured by an element of $H^2(G/N, N)$, the second cohomology group. This is the language that lets number theorists classify central simple algebras, that lets topologists classify principal bundles, and that lets physicists analyse anomalies. All of it begins with quotients and the first isomorphism theorem.

*Splitting of polynomials.* In Galois theory, normal subgroups of the Galois group correspond to intermediate fields fixed by the subgroup, and the quotient $\mathrm{Gal}(L/K) / \mathrm{Gal}(L/F)$ gives $\mathrm{Gal}(F/K)$ when $F/K$ is itself Galois. The whole correspondence is built on the bijection between normal subgroups and "well-behaved" intermediate fields.

## What I Want You to Carry Forward

Three questions for Part 4:

1. *Given a finite group $G$, how many subgroups of order $p^k$ does it have, and how are they related?* The Sylow theorems will give precise existence, conjugacy, and counting statements.
2. *Why is every group of order $p$ cyclic, every group of order $p^2$ abelian, and every group of order $pq$ either cyclic or a single specific non-abelian construction?* All three results follow from Sylow plus the class equation.
3. *How do you classify groups of small order, say up to $30$?* The combination of Sylow, the structure of cyclic groups, and the semidirect product construction does almost all the work.

If the homomorphism / quotient computations feel like reflex by now, you have the right footing for Sylow theory, where the techniques get more delicate but the underlying logic is the same: count cosets, identify normal subgroups, factor through quotients.

---
