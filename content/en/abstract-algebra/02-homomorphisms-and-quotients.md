---
title: "Homomorphisms, Normal Subgroups, and Quotient Groups"
date: 2021-01-17 09:00:00
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
series_total: 6
translationKey: "abstract-algebra-2"
description: "Homomorphisms are structure-preserving maps. Normal subgroups let you build quotient groups. The First Isomorphism Theorem ties it all together."
---

## Maps that respect structure

A function between groups that preserves the operation is called a **homomorphism**. This is the right notion of "map" in group theory — it tells you when two groups are structurally related, even if their elements look completely different.

In [Part 1](/en/abstract-algebra/01-groups-and-subgroups/) we built groups and proved Lagrange's theorem. But isolated groups are only half the story. The real power of algebra comes from understanding how groups relate to each other — and that requires maps. Not arbitrary maps, but maps that carry the algebraic structure along for the ride.


![First Isomorphism Theorem: the commutative diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/02-homomorphisms-and-quotients/aa_fig2_isomorphism_thm.png)

## The definition

A **group homomorphism** $\varphi: G \to H$ satisfies:

$$
\varphi(ab) = \varphi(a)\varphi(b) \quad \forall\, a, b \in G
$$

Consequences: $\varphi(e_G) = e_H$ (apply the property to $\varphi(e_G) = \varphi(e_G \cdot e_G) = \varphi(e_G)\varphi(e_G)$ and cancel) and $\varphi(a^{-1}) = \varphi(a)^{-1}$ (since $\varphi(a)\varphi(a^{-1}) = \varphi(aa^{-1}) = \varphi(e_G) = e_H$).

The **kernel** and **image** are:

$$
\ker\varphi = \{g \in G : \varphi(g) = e_H\}, \qquad \text{im}\,\varphi = \{\varphi(g) : g \in G\}
$$

A homomorphism is injective iff $\ker\varphi = \{e_G\}$. A bijective homomorphism is an **isomorphism** — the groups are "the same" up to relabeling. An isomorphism from a group to itself is an **automorphism**.

**Non-example.** The map $f: \mathbb{Z} \to \mathbb{Z}$ defined by $f(n) = n + 1$ is a bijection but NOT a homomorphism: $f(a + b) = a + b + 1 \neq f(a) + f(b) = a + b + 2$. Bijectivity alone does not make an isomorphism — structure preservation is the essential requirement.

### Example 1: The sign homomorphism

Define $\text{sgn}: S_n \to \{+1, -1\}$ by mapping even permutations to $+1$ and odd permutations to $-1$, where $\{+1, -1\}$ is a group under multiplication. This is a homomorphism: the product of two even permutations is even ($+1 \cdot +1 = +1$), and so on for all four cases. We have $\ker(\text{sgn}) = A_n$ (the alternating group).

### Example 2: Determinant as homomorphism

The map $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ is a homomorphism from the general linear group to the multiplicative group of nonzero reals. The key property $\det(AB) = \det(A)\det(B)$ is precisely the homomorphism condition. Its kernel is $SL_n(\mathbb{R})$, the matrices with determinant 1.

### Example 3: Exponentiation

Fix a group $G$ and an integer $n$. The map $\varphi: G \to G$ defined by $\varphi(g) = g^n$ is a homomorphism if and only if $G$ is abelian. When $G = \mathbb{Z}/m\mathbb{Z}$ (written additively), this becomes the map $k \mapsto nk \pmod{m}$, which is always a homomorphism since $\mathbb{Z}/m\mathbb{Z}$ is abelian. Its kernel is the set of $k$ with $nk \equiv 0 \pmod{m}$, i.e., the subgroup of order $\gcd(n, m)$.

## The kernel-image relationship

The kernel and image of a homomorphism are not independent — they partition the information in $G$ into "what gets collapsed" and "what survives."

**Proposition.** For $\varphi: G \to H$: (a) $\ker\varphi \leq G$, (b) $\text{im}\,\varphi \leq H$, (c) $\varphi$ is injective iff $\ker\varphi = \{e\}$, and (d) $\varphi$ is surjective iff $\text{im}\,\varphi = H$.

Moreover, the fibers of $\varphi$ (the preimages of individual elements) are exactly the cosets of $\ker\varphi$:

$$
\varphi^{-1}(\{h\}) = g \cdot \ker\varphi \quad \text{where } \varphi(g) = h
$$

This means $\varphi$ is a "constant-to-one" map: every element of $\text{im}\,\varphi$ has exactly $|\ker\varphi|$ preimages. So if $G$ is finite:

$$
|G| = |\ker\varphi| \cdot |\text{im}\,\varphi|
$$

This is the "rank-nullity theorem" of group theory — and indeed, when $G$ and $H$ are vector spaces (which are abelian groups under addition), this literally becomes the rank-nullity theorem from linear algebra.

## Normal subgroups

A subgroup $N \leq G$ is **normal** (written $N \trianglelefteq G$) if it is invariant under conjugation:

$$
N \trianglelefteq G \iff gNg^{-1} = N \quad \forall\, g \in G
$$

Equivalently: the left coset $gN$ equals the right coset $Ng$ for every $g$.

Why does this matter? Because normality is exactly the condition that makes the set of cosets into a group. If $N$ is not normal, then coset multiplication is not well-defined — different representatives of the same coset can give different products.

**Key fact.** The kernel of any homomorphism is a normal subgroup. Conversely, every normal subgroup is the kernel of some homomorphism (namely, the projection onto the quotient).

*Proof that kernels are normal.* Let $n \in \ker\varphi$ and $g \in G$. Then $\varphi(gng^{-1}) = \varphi(g)\varphi(n)\varphi(g^{-1}) = \varphi(g) \cdot e_H \cdot \varphi(g)^{-1} = e_H$. So $gng^{-1} \in \ker\varphi$. $\square$

### Example 4: $A_n \trianglelefteq S_n$

Since $A_n = \ker(\text{sgn})$, it is automatically normal in $S_n$. The index is $[S_n : A_n] = 2$. In fact, any subgroup of index 2 is normal: there are only two cosets ($H$ and the single other coset $G \setminus H$), so left and right cosets must coincide since there is no room for them to differ.

**Non-example.** In $S_3$, the subgroup $H = \{e, (1\;2)\}$ has index 3 and is NOT normal. We can check: $(1\;2\;3)(1\;2)(1\;2\;3)^{-1} = (2\;3) \notin H$. So $H$ is not closed under conjugation. The coset $(1\;2\;3)H = \{(1\;2\;3), (1\;3)\}$ is not equal to $H(1\;2\;3) = \{(1\;2\;3), (2\;3)\}$. This is why we cannot form a quotient $S_3/H$.

### Example 5: The center

The **center** of a group, $Z(G) = \{z \in G : zg = gz \;\forall g \in G\}$, is always a normal subgroup. For abelian groups, $Z(G) = G$. For $S_n$ with $n \geq 3$, $Z(S_n) = \{e\}$ — only the identity commutes with every permutation.

## Quotient groups

Given $N \trianglelefteq G$, the **quotient group** $G/N$ has elements that are cosets $\{gN : g \in G\}$ with the operation:

$$
(g_1 N)(g_2 N) = (g_1 g_2)N
$$

This is well-defined precisely because $N$ is normal. The order is $|G/N| = [G:N] = |G|/|N|$.

The **canonical projection** $\pi: G \to G/N$ defined by $\pi(g) = gN$ is a surjective homomorphism with kernel $N$. This confirms that normal subgroups and kernels are two descriptions of the same phenomenon.

### Example 6: $\mathbb{Z}/n\mathbb{Z}$ as quotient

Take $G = \mathbb{Z}$ (under addition) and $N = n\mathbb{Z} = \{0, \pm n, \pm 2n, \ldots\}$. Since $\mathbb{Z}$ is abelian, every subgroup is normal. The quotient $\mathbb{Z}/n\mathbb{Z}$ is the group of remainders mod $n$ — the same object we met in Part 1, now constructed as a quotient.

### Example 7: $GL_n / SL_n$

Since $SL_n(\mathbb{R}) = \ker(\det)$, we have $SL_n(\mathbb{R}) \trianglelefteq GL_n(\mathbb{R})$ and the quotient $GL_n(\mathbb{R})/SL_n(\mathbb{R})$ is isomorphic to $\mathbb{R}^*$ (the multiplicative group of nonzero reals). Informally: if you quotient out "volume-preserving" transformations, what remains is the "scaling factor."

## The First Isomorphism Theorem

This is the central theorem connecting homomorphisms, kernels, and quotients.

**Theorem (First Isomorphism Theorem).** If $\varphi: G \to H$ is a homomorphism, then:

$$
G / \ker\varphi \;\cong\; \text{im}\,\varphi
$$

*Proof.* Define $\bar\varphi: G/\ker\varphi \to \text{im}\,\varphi$ by $\bar\varphi(g\ker\varphi) = \varphi(g)$.

- **Well-defined:** If $g_1\ker\varphi = g_2\ker\varphi$, then $g_1^{-1}g_2 \in \ker\varphi$, so $\varphi(g_1^{-1}g_2) = e_H$, giving $\varphi(g_1) = \varphi(g_2)$.
- **Homomorphism:** $\bar\varphi(g_1\ker\varphi \cdot g_2\ker\varphi) = \bar\varphi(g_1g_2\ker\varphi) = \varphi(g_1g_2) = \varphi(g_1)\varphi(g_2) = \bar\varphi(g_1\ker\varphi)\bar\varphi(g_2\ker\varphi)$.
- **Surjective:** Every $h \in \text{im}\,\varphi$ is $\varphi(g)$ for some $g$, and $\bar\varphi(g\ker\varphi) = h$.
- **Injective:** If $\bar\varphi(g\ker\varphi) = e_H$, then $\varphi(g) = e_H$, so $g \in \ker\varphi$, meaning $g\ker\varphi$ is the identity coset. $\square$

**Worked example.** Consider $\varphi: \mathbb{Z} \to \mathbb{Z}/6\mathbb{Z}$ by $\varphi(n) = n \bmod 6$. Then $\ker\varphi = 6\mathbb{Z}$ and $\text{im}\,\varphi = \mathbb{Z}/6\mathbb{Z}$. The theorem says $\mathbb{Z}/6\mathbb{Z} \cong \mathbb{Z}/6\mathbb{Z}$ — trivially true but illustrating the mechanism. More interestingly: restrict $\varphi$ to $3\mathbb{Z}$ (multiples of 3). Then $\text{im}(\varphi|_{3\mathbb{Z}}) = \{0, 3\}$ and $\ker(\varphi|_{3\mathbb{Z}}) = 6\mathbb{Z}$. The theorem gives $3\mathbb{Z}/6\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z}$.

## The correspondence theorem

The First Isomorphism Theorem has a powerful companion that describes the subgroup structure of quotients.

**Theorem (Correspondence/Lattice Theorem).** Let $N \trianglelefteq G$ and let $\pi: G \to G/N$ be the canonical projection. There is a bijection:

$$
\{\text{subgroups of } G \text{ containing } N\} \;\longleftrightarrow\; \{\text{subgroups of } G/N\}
$$

given by $H \mapsto H/N$ (and conversely $\bar H \mapsto \pi^{-1}(\bar H)$). Moreover, this bijection preserves normality, indices, and containment.

*Example.* Take $G = \mathbb{Z}$ and $N = 12\mathbb{Z}$. The subgroups of $\mathbb{Z}$ containing $12\mathbb{Z}$ are $\mathbb{Z}, 2\mathbb{Z}, 3\mathbb{Z}, 4\mathbb{Z}, 6\mathbb{Z}, 12\mathbb{Z}$. These correspond to the subgroups of $\mathbb{Z}/12\mathbb{Z}$: $\mathbb{Z}/12\mathbb{Z}, \langle 2 \rangle, \langle 3 \rangle, \langle 4 \rangle, \langle 6 \rangle, \{0\}$. The lattice structure is identical.

This theorem is indispensable for understanding quotients of quotients, and it reappears in ring theory (as the correspondence for ideals, see [Part 3](/en/abstract-algebra/03-rings-and-ideals/)) and field theory (as the Galois correspondence, see [Part 5](/en/abstract-algebra/05-galois-theory/)).

## The other isomorphism theorems

**Second Isomorphism Theorem (Diamond Theorem).** If $H \leq G$ and $N \trianglelefteq G$, then $HN \leq G$, $H \cap N \trianglelefteq H$, and:

$$
HN/N \cong H/(H \cap N)
$$

*Proof sketch.* Define $\varphi: H \to HN/N$ by $\varphi(h) = hN$. This is a homomorphism (restriction of the canonical projection). It is surjective since every element of $HN$ is $hn$ for some $h \in H, n \in N$, and $hnN = hN$. Its kernel is $\{h \in H : hN = N\} = H \cap N$. Apply the First Isomorphism Theorem. $\square$

**Third Isomorphism Theorem.** If $N \subseteq M$ are both normal in $G$, then $M/N \trianglelefteq G/N$ and:

$$
(G/N)/(M/N) \cong G/M
$$

Informally: "quotienting by a quotient is the same as quotienting directly." These theorems are bookkeeping, but they simplify many inductive arguments on group structure.

## Direct products and semidirect products

Given groups $G$ and $H$, the **direct product** $G \times H$ has elements $(g, h)$ with component-wise operation. If $N \trianglelefteq G$ and $H \leq G$ with $G = NH$ and $N \cap H = \{e\}$, then $G \cong N \rtimes H$ — a **semidirect product** where $H$ acts on $N$ by conjugation. When the action is trivial, this reduces to the direct product.

**Example.** $S_3 \cong A_3 \rtimes \mathbb{Z}/2\mathbb{Z}$: the normal subgroup $A_3 = \{e, (1\;2\;3), (1\;3\;2)\} \cong \mathbb{Z}/3\mathbb{Z}$ is complemented by $H = \{e, (1\;2)\} \cong \mathbb{Z}/2\mathbb{Z}$. The action is non-trivial (conjugation by $(1\;2)$ sends $(1\;2\;3)$ to $(1\;3\;2)$), so this is genuinely a semidirect product, not a direct product. Indeed $S_3 \not\cong \mathbb{Z}/3\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z} \cong \mathbb{Z}/6\mathbb{Z}$ since $S_3$ is non-abelian.

The dihedral group $D_n$ from Part 1 is always a semidirect product: $D_n \cong \mathbb{Z}/n\mathbb{Z} \rtimes \mathbb{Z}/2\mathbb{Z}$ where $\mathbb{Z}/2\mathbb{Z}$ acts on $\mathbb{Z}/n\mathbb{Z}$ by inversion ($k \mapsto -k$).

## Simple groups and composition series

A group $G$ is **simple** if its only normal subgroups are $\{e\}$ and $G$ itself. Simple groups are the "atoms" of group theory — they cannot be broken down further by taking quotients.

Examples: $\mathbb{Z}/p\mathbb{Z}$ for $p$ prime (having no nontrivial subgroups at all), and $A_n$ for $n \geq 5$ (a non-trivial theorem — it is the key ingredient in proving the unsolvability of the quintic, discussed in [Part 5](/en/abstract-algebra/05-galois-theory/)).

A **composition series** for $G$ is a chain $\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_n = G$ where each quotient $G_{i+1}/G_i$ is simple. The **Jordan-Holder Theorem** states that the multiset of composition factors $\{G_{i+1}/G_i\}$ is unique up to permutation — just as every integer has a unique prime factorization.

For $S_4$: one composition series is $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$, with composition factors $\mathbb{Z}/2\mathbb{Z}, \mathbb{Z}/3\mathbb{Z}, \mathbb{Z}/2\mathbb{Z}$ — all abelian. The fact that all composition factors of $S_4$ are abelian (equivalently, that $S_4$ is **solvable**) is why the quartic polynomial is solvable by radicals.

## What's next

Groups capture one operation. But integers have two — addition and multiplication — and so do polynomials. When you have two operations that interact via distributivity, you get a **ring**. Next: rings, ideals (the ring-theoretic analog of normal subgroups), and why polynomial division works.

---

*This is Part 2 of the [Abstract Algebra](/en/series/abstract-algebra/) series (6 articles).*

*Previous: [Part 1 — Groups and Subgroups](/en/abstract-algebra/01-groups-and-subgroups/)*

*Next: [Part 3 — Rings and Ideals](/en/abstract-algebra/03-rings-and-ideals/)*
