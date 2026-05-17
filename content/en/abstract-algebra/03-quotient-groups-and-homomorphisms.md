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

A group can be enormous — millions of elements, intricate multiplication tables, symmetries that take pages to describe. Yet hidden inside every group are natural "compression points" where you can collapse entire chunks of the group into single elements, producing a smaller group that still remembers something essential about the original. This article develops the machinery for doing that: normal subgroups, quotient groups, homomorphisms, and the isomorphism theorems that tie everything together.

If groups are the atoms of symmetry, quotient groups are what you get when you deliberately blur your vision — ignoring certain details so that the remaining structure becomes visible.

---

## Why Collapse Structure?

Consider the integers $\mathbb{Z}$ under addition. This is an infinite group, and reasoning about its full structure all at once is unwieldy. But we have a familiar trick: work modulo $n$. When we compute "mod 5," we declare that any two integers differing by a multiple of 5 are "the same." The set $\{0, 1, 2, 3, 4\}$ with addition mod 5 forms a perfectly well-behaved group $\mathbb{Z}/5\mathbb{Z}$.

What just happened? We took an infinite group and produced a finite one by identifying elements that differ by something from a subgroup ($5\mathbb{Z}$). The result is smaller, simpler, and still a group. This is the quotient construction, and it is one of the most powerful ideas in all of algebra.

But not every subgroup works. If you try to "mod out" by an arbitrary subgroup, the resulting set of cosets might not form a group at all — the operation might not be well-defined. The subgroups that do work have a special property: they are **normal**. Understanding what normality means and why it matters is our first task.

**Motivation from geometry.** In the symmetry group of a square ($D_4$, eight elements), the four rotations form a subgroup $R = \{e, r, r^2, r^3\}$. If we collapse all rotations to a single element and all reflections to another, we get a group of order 2 — essentially $\mathbb{Z}/2\mathbb{Z}$. This captures the fact that reflections and rotations are "categorically different," while the specific rotation or reflection doesn't matter at this coarse level. The rotation subgroup happens to be normal in $D_4$, which is why this works.

**Motivation from linear algebra.** Given a linear map $T: V \to W$, the kernel $\ker T$ is a subspace of $V$. The quotient space $V / \ker T$ is isomorphic to the image $\text{im}(T)$. This is the rank-nullity theorem in disguise. The same idea generalizes to groups: the kernel of a group homomorphism is always normal, and the quotient by the kernel is isomorphic to the image.

---


![First Isomorphism Theorem commutative diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/03-quotient-groups-and-homomorphisms/aa_fig3_isomorphism_thm.png)

## Normal Subgroups

**Definition.** A subgroup $N$ of a group $G$ is called **normal** (written $N \trianglelefteq G$) if for every $g \in G$ and every $n \in N$, the conjugate $gng^{-1}$ belongs to $N$. Equivalently, $gNg^{-1} = N$ for all $g \in G$.

This condition says that $N$ is "invariant under conjugation" — you can't escape $N$ by conjugating its elements.

**Equivalent characterizations.** The following are all equivalent for a subgroup $N \leq G$:

1. $N \trianglelefteq G$ (i.e., $gNg^{-1} = N$ for all $g \in G$).
2. $gN = Ng$ for all $g \in G$ (left cosets equal right cosets).
3. $N$ is the kernel of some group homomorphism from $G$.
4. Every left coset of $N$ is also a right coset.

Condition (2) is often the most practical for checking normality. Note that $gN = Ng$ does **not** mean $gn = ng$ for every $n$; it means the *sets* are equal.

**Example 1: $n\mathbb{Z}$ in $\mathbb{Z}$.** Since $\mathbb{Z}$ is abelian, every subgroup is automatically normal (conjugation does nothing: $g + n - g = n$). So $5\mathbb{Z} \trianglelefteq \mathbb{Z}$, and indeed every $n\mathbb{Z}$ is normal.

**Example 2: $A_n$ in $S_n$.** The alternating group $A_n$ (even permutations) is normal in $S_n$. To verify: for any $\sigma \in S_n$ and $\alpha \in A_n$, the conjugate $\sigma \alpha \sigma^{-1}$ has the same parity as $\alpha$ (since $\text{sign}(\sigma \alpha \sigma^{-1}) = \text{sign}(\sigma) \cdot \text{sign}(\alpha) \cdot \text{sign}(\sigma^{-1}) = \text{sign}(\alpha)$). So $\sigma \alpha \sigma^{-1} \in A_n$.

**Example 3: A non-normal subgroup.** In $S_3$, consider $H = \{e, (1\ 2)\}$. The left coset $(1\ 3)H = \{(1\ 3), (1\ 2\ 3)\}$ while the right coset $H(1\ 3) = \{(1\ 3), (1\ 3\ 2)\}$. Since $(1\ 2\ 3) \neq (1\ 3\ 2)$, the cosets differ: $H$ is not normal in $S_3$.

**Tests for normality:**
- **Index 2 test:** If $[G : N] = 2$, then $N$ is automatically normal. (There are only two cosets: $N$ and its complement. Both left and right cosets must partition $G$ into the same two pieces.)
- **Center test:** The center $Z(G) = \{z \in G : zg = gz \text{ for all } g\}$ is always normal.
- **Kernel test:** Show $N = \ker \varphi$ for some homomorphism $\varphi$.
- **Commutator test:** The commutator subgroup $[G, G] = \langle g^{-1}h^{-1}gh : g, h \in G \rangle$ is always normal. More generally, any subgroup containing $[G, G]$ is normal.

**Why normality matters beyond quotients.** Normal subgroups appear everywhere in algebra, not just in quotient constructions. A group is **simple** if it has no proper non-trivial normal subgroups. Simple groups are the "atoms" of group theory: every finite group can be built from simple groups via a composition series (the Jordan-Holder theorem). The classification of finite simple groups — completed in the 1980s across tens of thousands of pages — is one of the greatest achievements of modern mathematics.

The search for normal subgroups is therefore the search for ways to decompose a group into simpler pieces. The quotient $G/N$ is one piece; $N$ itself is another. Together, they "almost" determine $G$ (the ambiguity is captured by the extension problem, which we'll address in later articles).

---

## The Quotient Group Construction

Given $N \trianglelefteq G$, we build a new group whose elements are the cosets of $N$ in $G$. The idea is simple but the details require care — especially the question of well-definedness, which is where the normality condition earns its keep.

**Definition.** The **quotient group** (or factor group) $G/N$ is the set of all left cosets:
$$G/N = \{gN : g \in G\}$$
with the operation $(g_1 N)(g_2 N) = (g_1 g_2)N$.

**Why normality is needed.** The operation must be well-defined: if $g_1 N = g_1' N$ and $g_2 N = g_2' N$, we need $(g_1 g_2)N = (g_1' g_2')N$. Write $g_1' = g_1 n_1$ and $g_2' = g_2 n_2$ for some $n_1, n_2 \in N$. Then:
$$g_1' g_2' = g_1 n_1 g_2 n_2 = g_1 g_2 (g_2^{-1} n_1 g_2) n_2$$

We need $g_2^{-1} n_1 g_2 \in N$. This is exactly the normality condition! Without normality, the product of cosets depends on which representatives you choose, and the construction collapses.

**Verification of group axioms:**
- *Closure:* $(g_1 N)(g_2 N) = (g_1 g_2)N \in G/N$. This holds since $g_1 g_2 \in G$.
- *Identity:* The coset $eN = N$ is the identity since $(gN)(eN) = (ge)N = gN$.
- *Inverses:* $(gN)^{-1} = g^{-1}N$ since $(gN)(g^{-1}N) = (gg^{-1})N = N$.
- *Associativity:* Inherited from $G$.

**Example: $\mathbb{Z}/n\mathbb{Z}$.** The cosets of $n\mathbb{Z}$ in $\mathbb{Z}$ are $\{0 + n\mathbb{Z}, 1 + n\mathbb{Z}, \ldots, (n-1) + n\mathbb{Z}\}$. The group operation is addition of representatives mod $n$. This is precisely the cyclic group of order $n$.

**Example: $S_3 / A_3$.** The alternating group $A_3 = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$ has index 2 in $S_3$, so it is normal. The quotient $S_3/A_3$ has two elements: $A_3$ (the even permutations) and $(1\ 2)A_3$ (the odd permutations). The multiplication table is that of $\mathbb{Z}/2\mathbb{Z}$: even $\times$ even $=$ even, even $\times$ odd $=$ odd, odd $\times$ odd $=$ even.

**The natural projection.** There is a canonical surjective homomorphism $\pi: G \to G/N$ defined by $\pi(g) = gN$. Its kernel is exactly $N$. This map is the bridge between $G$ and its quotient. Every property of the quotient group can be understood through this map: $G/N$ is abelian if and only if $[G, G] \leq N$; $G/N$ is cyclic if and only if $G = \langle g \rangle N$ for some $g$; and more generally, properties of $G/N$ reflect "coarse" properties of $G$ that are invisible at the scale of $N$.

**Order of elements in $G/N$.** The order of the coset $gN$ in $G/N$ divides the order of $g$ in $G$ (since $(gN)^k = g^k N$, and $g^k N = N$ whenever $g^k \in N$). But the order of $gN$ can be strictly smaller than the order of $g$. For instance, in $\mathbb{Z}/6\mathbb{Z}$ the element $2$ has order 3, but in $(\mathbb{Z}/6\mathbb{Z}) / (3\mathbb{Z}/6\mathbb{Z}) \cong \mathbb{Z}/3\mathbb{Z}$, the image of $2$ has order 3 as well — in this case the orders happen to match. But in $\mathbb{Z}/6\mathbb{Z}$ modulo $\langle 2 \rangle = \{0, 2, 4\}$, the element $1$ has order 6 in the original group but order 2 in the quotient (since $1 + 1 = 2 \in \langle 2 \rangle$).

---

## Homomorphisms: Structure-Preserving Maps

**Definition.** A **group homomorphism** is a map $\varphi: G \to H$ between groups satisfying:
$$\varphi(g_1 g_2) = \varphi(g_1) \varphi(g_2) \quad \text{for all } g_1, g_2 \in G.$$

This single condition forces many consequences:

- $\varphi(e_G) = e_H$ (map identity to identity).
- $\varphi(g^{-1}) = \varphi(g)^{-1}$ (map inverses to inverses).
- $\varphi(g^n) = \varphi(g)^n$ for all $n \in \mathbb{Z}$.

**The kernel and image.**
- $\ker \varphi = \{g \in G : \varphi(g) = e_H\}$ is a **normal** subgroup of $G$.
- $\text{im}(\varphi) = \{\varphi(g) : g \in G\}$ is a subgroup of $H$.

The kernel measures how far $\varphi$ is from being injective: $\varphi$ is injective if and only if $\ker \varphi = \{e\}$.

**Example 1: The determinant map.** Define $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ (the nonzero reals under multiplication). Since $\det(AB) = \det(A)\det(B)$, this is a homomorphism. Its kernel is $SL_n(\mathbb{R})$ (matrices with determinant 1), which is therefore normal in $GL_n(\mathbb{R})$. Its image is all of $\mathbb{R}^*$.

**Example 2: The sign homomorphism.** Define $\text{sign}: S_n \to \{+1, -1\}$ by mapping even permutations to $+1$ and odd to $-1$. Since $\text{sign}(\sigma \tau) = \text{sign}(\sigma) \cdot \text{sign}(\tau)$, this is a homomorphism. Its kernel is $A_n$.

**Example 3: Exponentiation.** Define $\varphi: \mathbb{Z} \to G$ by $\varphi(n) = g^n$ for a fixed element $g$ in a group $G$. This is a homomorphism from $(\mathbb{Z}, +)$ to $(G, \cdot)$. Its image is the cyclic subgroup $\langle g \rangle$. Its kernel is $\{n \in \mathbb{Z} : g^n = e\}$, which is either $\{0\}$ (if $g$ has infinite order) or $k\mathbb{Z}$ (if $g$ has order $k$).

**Types of homomorphisms:**
- **Monomorphism** (injective homomorphism): $\ker \varphi = \{e\}$.
- **Epimorphism** (surjective homomorphism): $\text{im}(\varphi) = H$.
- **Isomorphism** (bijective homomorphism): both injective and surjective.
- **Automorphism**: isomorphism from $G$ to itself.
- **Endomorphism**: homomorphism from $G$ to itself.

---

## The First Isomorphism Theorem

This is arguably the single most important theorem in group theory. It says that quotient groups and images of homomorphisms are the same thing.

**Theorem (First Isomorphism Theorem).** Let $\varphi: G \to H$ be a group homomorphism. Then:
$$G / \ker \varphi \cong \text{im}(\varphi).$$

More precisely, the map $\bar{\varphi}: G/\ker\varphi \to \text{im}(\varphi)$ defined by $\bar{\varphi}(g \ker\varphi) = \varphi(g)$ is a well-defined isomorphism.

**Proof.** Let $K = \ker \varphi$.

*Well-defined:* If $g_1 K = g_2 K$, then $g_1^{-1} g_2 \in K$, so $\varphi(g_1^{-1} g_2) = e_H$, hence $\varphi(g_1) = \varphi(g_2)$. So $\bar{\varphi}$ does not depend on the choice of coset representative.

*Homomorphism:* $\bar{\varphi}(g_1 K \cdot g_2 K) = \bar{\varphi}(g_1 g_2 K) = \varphi(g_1 g_2) = \varphi(g_1)\varphi(g_2) = \bar{\varphi}(g_1 K) \cdot \bar{\varphi}(g_2 K)$.

*Injective:* If $\bar{\varphi}(gK) = e_H$, then $\varphi(g) = e_H$, so $g \in K$, meaning $gK = K$ is the identity in $G/K$. Thus $\ker \bar{\varphi} = \{K\}$.

*Surjective:* For any $h \in \text{im}(\varphi)$, there exists $g \in G$ with $\varphi(g) = h$, so $\bar{\varphi}(gK) = h$.

Therefore $\bar{\varphi}$ is a bijective homomorphism, i.e., an isomorphism. $\blacksquare$

**The diamond diagram.** The theorem is often visualized as a commutative triangle:

$$G \xrightarrow{\varphi} H$$
$$\downarrow \pi \qquad \nearrow \bar{\varphi}$$
$$G/K$$

where $\pi$ is the natural projection, $\varphi = \bar{\varphi} \circ \pi$, and $\bar{\varphi}$ is an isomorphism onto $\text{im}(\varphi)$.

**Application 1.** The determinant map $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ has kernel $SL_n(\mathbb{R})$ and is surjective. By the First Isomorphism Theorem:
$$GL_n(\mathbb{R}) / SL_n(\mathbb{R}) \cong \mathbb{R}^*.$$

**Application 2.** Consider $\varphi: \mathbb{Z} \to \mathbb{Z}/n\mathbb{Z}$ defined by $\varphi(k) = k \bmod n$. This is surjective with kernel $n\mathbb{Z}$. The theorem confirms $\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}/n\mathbb{Z}$ — a tautology here, but it validates that the modular arithmetic group is exactly the quotient.

**Application 3.** Define $\varphi: \mathbb{R} \to \mathbb{C}^*$ by $\varphi(t) = e^{2\pi i t}$. This maps the reals (under addition) to the unit circle (under multiplication). It is a surjective homomorphism onto the unit circle group $S^1$, with kernel $\mathbb{Z}$ (those $t$ for which $e^{2\pi i t} = 1$). Therefore:
$$\mathbb{R}/\mathbb{Z} \cong S^1.$$

This is a beautiful example: the quotient of the real line by the integers wraps around to form a circle. The topology here is not a coincidence — the algebraic quotient and the topological identification give the same object.

**Application 4: Classifying cyclic quotients.** Let $G = \mathbb{Z}/12\mathbb{Z}$ and define $\varphi: \mathbb{Z}/12\mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$ by $\varphi(\bar{k}) = k \bmod 4$. This is a well-defined surjective homomorphism. Its kernel consists of elements $\bar{k}$ where $4 \mid k$, which is $\{0, 4, 8\} = \langle 4 \rangle \cong \mathbb{Z}/3\mathbb{Z}$. The First Isomorphism Theorem confirms:
$$(\mathbb{Z}/12\mathbb{Z}) / (\mathbb{Z}/3\mathbb{Z}) \cong \mathbb{Z}/4\mathbb{Z}.$$

This illustrates a general pattern: for cyclic groups, every quotient is again cyclic, and the quotient $\mathbb{Z}/m\mathbb{Z}$ of $\mathbb{Z}/n\mathbb{Z}$ exists precisely when $m \mid n$.

**Worked Example: Proving a group is not simple.** A group $G$ is **simple** if its only normal subgroups are $\{e\}$ and $G$ itself. The First Isomorphism Theorem provides a strategy for proving non-simplicity: find a non-trivial homomorphism $\varphi: G \to H$ where $|H| < |G|$. Then $\ker \varphi$ is a non-trivial normal subgroup (it can't be all of $G$ since $\varphi$ is non-trivial, and it can't be $\{e\}$ since that would make $\varphi$ injective, contradicting $|H| < |G|$).

For instance, consider any group $G$ of order 6. Define the action of $G$ on itself by left multiplication, giving a homomorphism $\varphi: G \to S_6$. But we can be more clever: $G$ acts on the 3 left cosets of a subgroup of index 3 (which exists by Sylow theory — we'll see this in the next article). This gives $\varphi: G \to S_3$. If $G$ is non-abelian, then $|G| = |S_3| = 6$ and $\varphi$ is injective, so $G \cong S_3$. If $G$ is abelian, the kernel analysis shows $G \cong \mathbb{Z}/6\mathbb{Z}$.

---

## Second and Third Isomorphism Theorems

The First Isomorphism Theorem has two important companions that describe how subgroups and quotients interact.

**Second Isomorphism Theorem (Diamond Isomorphism Theorem).** Let $G$ be a group, $H \leq G$ a subgroup, and $N \trianglelefteq G$ a normal subgroup. Then:

1. $HN = \{hn : h \in H, n \in N\}$ is a subgroup of $G$.
2. $H \cap N \trianglelefteq H$.
3. $HN / N \cong H / (H \cap N)$.

**Proof sketch.** Define $\varphi: H \to G/N$ by $\varphi(h) = hN$. This is a homomorphism (as the restriction of the natural projection $\pi: G \to G/N$ to $H$). Its kernel is $\{h \in H : hN = N\} = H \cap N$. Its image is $\{hN : h \in H\} = HN/N$ (since any element of $HN$ is of the form $hn$, and $(hn)N = hN$). By the First Isomorphism Theorem, $H/(H \cap N) \cong HN/N$. $\blacksquare$

**Intuition.** Think of $H$ and $N$ as two overlapping "lenses" on $G$. The theorem says that what $H$ "sees" in the quotient $G/N$ (which is $HN/N$) is the same as what $H$ "sees" when it removes its own overlap with $N$ (which is $H/(H \cap N)$).

**Third Isomorphism Theorem.** Let $G$ be a group and $N \trianglelefteq K \trianglelefteq G$ (both normal in $G$, with $N \subseteq K$). Then:

1. $K/N \trianglelefteq G/N$.
2. $(G/N) / (K/N) \cong G/K$.

**Proof sketch.** Define $\varphi: G/N \to G/K$ by $\varphi(gN) = gK$. This is well-defined because $N \subseteq K$: if $g_1 N = g_2 N$, then $g_1^{-1} g_2 \in N \subseteq K$, so $g_1 K = g_2 K$. It is clearly a surjective homomorphism, and its kernel is $\{gN : gK = K\} = \{gN : g \in K\} = K/N$. The First Isomorphism Theorem gives $(G/N)/(K/N) \cong G/K$. $\blacksquare$

**Intuition.** Quotienting out a quotient is the same as quotienting out all at once. If you blur your vision by $N$ and then blur again by $K/N$, the result is the same as blurring by $K$ from the start.

**The Lattice Correspondence Theorem (Fourth Isomorphism Theorem).** Let $N \trianglelefteq G$. There is a bijection between:
- Subgroups of $G/N$, and
- Subgroups of $G$ that contain $N$,

given by $H/N \leftrightarrow H$ (for $N \leq H \leq G$). This bijection preserves:
- **Containment:** $H_1/N \leq H_2/N$ if and only if $H_1 \leq H_2$.
- **Normality:** $H/N \trianglelefteq K/N$ if and only if $H \trianglelefteq K$.
- **Index:** $[K/N : H/N] = [K : H]$.

This is an extraordinarily useful theorem. It says that taking a quotient doesn't destroy the subgroup structure — it merely "forgets" everything below $N$. The subgroup lattice of $G/N$ is literally the portion of the subgroup lattice of $G$ that lies above $N$.

**Concrete illustration.** Consider $G = \mathbb{Z}/12\mathbb{Z}$ and $N = \langle 4 \rangle = \{0, 4, 8\}$. The subgroups of $\mathbb{Z}/12\mathbb{Z}$ containing $N$ are: $\langle 4 \rangle$ itself, $\langle 2 \rangle = \{0, 2, 4, 6, 8, 10\}$, and $\mathbb{Z}/12\mathbb{Z}$. The correspondence gives three subgroups of $G/N \cong \mathbb{Z}/4\mathbb{Z}$, which are exactly $\{0\}$, $\{0, 2\}$, and $\mathbb{Z}/4\mathbb{Z}$ — precisely the subgroup lattice of $\mathbb{Z}/4\mathbb{Z}$.

**Worked Example.** Consider $G = \mathbb{Z}$, $K = 6\mathbb{Z}$, $N = 30\mathbb{Z}$. Then $N \subseteq K$ (since $30\mathbb{Z} \subseteq 6\mathbb{Z}$), and both are normal in $\mathbb{Z}$. The Third Isomorphism Theorem gives:
$$(\mathbb{Z}/30\mathbb{Z}) / (6\mathbb{Z}/30\mathbb{Z}) \cong \mathbb{Z}/6\mathbb{Z}.$$

Now $6\mathbb{Z}/30\mathbb{Z}$ has elements $\{0 + 30\mathbb{Z}, 6 + 30\mathbb{Z}, 12 + 30\mathbb{Z}, 18 + 30\mathbb{Z}, 24 + 30\mathbb{Z}\}$, which is a cyclic group of order 5. So we're saying that $\mathbb{Z}_{30}$ modulo a copy of $\mathbb{Z}_5$ gives $\mathbb{Z}_6$. This checks out: $30 / 5 = 6$.

**Worked Example: A non-abelian application.** Let $G = S_4$, and consider the chain $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$, where $V_4 = \{e, (1\ 2)(3\ 4), (1\ 3)(2\ 4), (1\ 4)(2\ 3)\}$ is the Klein four-group. By the Third Isomorphism Theorem:
$$(S_4 / V_4) / (A_4 / V_4) \cong S_4 / A_4 \cong \mathbb{Z}/2\mathbb{Z}.$$

Since $|A_4/V_4| = 12/4 = 3$ and $|S_4/V_4| = 24/4 = 6$, we have a group of order 6 with a normal subgroup of order 3 and quotient $\mathbb{Z}/2\mathbb{Z}$. In fact $S_4/V_4 \cong S_3$ (the quotient is isomorphic to $S_3$ acting on the three "double transposition pairs"), and $A_4/V_4 \cong \mathbb{Z}/3\mathbb{Z}$.

**Worked Example: Kernel of a matrix homomorphism.** Define $\varphi: GL_2(\mathbb{R}) \to \mathbb{R}^* / \{+1, -1\}$ by $\varphi(A) = |\det(A)|$ (the absolute value of the determinant, mapping to the positive reals under multiplication). This is a homomorphism. Its kernel is the set of matrices with $|\det(A)| = 1$, i.e., matrices with determinant $\pm 1$. This is the orthogonal-like group $O_2'(\mathbb{R}) = \{A \in GL_2(\mathbb{R}) : \det(A) = \pm 1\}$. The First Isomorphism Theorem gives $GL_2(\mathbb{R})/O_2'(\mathbb{R}) \cong \mathbb{R}^+$, the positive reals under multiplication. Meanwhile, if we use the ordinary determinant, $GL_2(\mathbb{R})/SL_2(\mathbb{R}) \cong \mathbb{R}^*$. The Second Isomorphism Theorem connects these: $SL_2(\mathbb{R}) \cdot O_2'(\mathbb{R}) = GL_2(\mathbb{R})$ (every matrix can be written as a product of a determinant-1 matrix and a $\pm 1$ determinant matrix, after appropriate scaling), and $SL_2(\mathbb{R}) \cap O_2'(\mathbb{R}) = SL_2(\mathbb{R})$.

**Why the isomorphism theorems matter in practice.** In number theory, the Chinese Remainder Theorem can be expressed as: if $\gcd(m, n) = 1$, then $\mathbb{Z}/mn\mathbb{Z} \cong \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$. This follows from considering the homomorphism $\varphi: \mathbb{Z} \to \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$ defined by $\varphi(k) = (k \bmod m, k \bmod n)$. It is surjective (by CRT) with kernel $mn\mathbb{Z}$, and the First Isomorphism Theorem does the rest.

In algebraic topology, the fundamental group of a quotient space is often a quotient of the fundamental group of the original space. The covering space theory of a torus $T^2$, for instance, relies on the isomorphism $\pi_1(T^2) \cong \mathbb{Z}^2$, and the classification of covering spaces corresponds to subgroups of $\mathbb{Z}^2$ — which the Lattice Correspondence Theorem organizes completely.

---


## A Detailed Worked Example: Constructing the Quotient Group \((\mathbb{Z} \times \mathbb{Z})/H\)

To construct the quotient group \((\mathbb{Z} \times \mathbb{Z})/H\) where \(H = \{(a, a) : a \in \mathbb{Z}\}\), we need to understand the structure of the cosets of \(H\) in \(\mathbb{Z} \times \mathbb{Z}\). 

### Step 1: Understanding the Subgroup \(H\)
The subgroup \(H\) consists of all pairs \((a, a)\) where \(a\) is an integer. This means that \(H\) is the set of all points on the line \(y = x\) in the \(\mathbb{Z} \times \mathbb{Z}\) plane.

### Step 2: Finding Cosets of \(H\)
A coset of \(H\) in \(\mathbb{Z} \times \mathbb{Z}\) is a set of the form \((m, n) + H\), where \((m, n) \in \mathbb{Z} \times \mathbb{Z}\). Explicitly, this coset is given by:
\[
(m, n) + H = \{(m + a, n + a) : a \in \mathbb{Z}\}
\]

Let's consider some specific examples of cosets:

- For \((0, 0)\):
  \[
  (0, 0) + H = \{(a, a) : a \in \mathbb{Z}\} = H
  \]
  This is just the subgroup \(H\) itself.

- For \((1, 0)\):
  \[
  (1, 0) + H = \{(1 + a, a) : a \in \mathbb{Z}\} = \{(1, 0), (2, 1), (3, 2), \ldots, (0, -1), (-1, -2), \ldots\}
  \]
  This coset consists of all points \((x, y)\) such that \(x - y = 1\).

- For \((0, 1)\):
  \[
  (0, 1) + H = \{(a, 1 + a) : a \in \mathbb{Z}\} = \{(0, 1), (1, 2), (2, 3), \ldots, (-1, 0), (-2, -1), \ldots\}
  \]
  This coset consists of all points \((x, y)\) such that \(y - x = 1\).

In general, for any \((m, n) \in \mathbb{Z} \times \mathbb{Z}\), the coset \((m, n) + H\) can be described as the set of all points \((x, y)\) such that \(x - y = m - n\). Therefore, each coset corresponds to a line parallel to \(y = x\) with a fixed difference \(m - n\).

### Step 3: Describing the Quotient Group
The quotient group \((\mathbb{Z} \times \mathbb{Z})/H\) consists of all distinct cosets of \(H\) in \(\mathbb{Z} \times \mathbb{Z}\). From the above, we see that each coset is determined by the difference \(m - n\). Thus, the quotient group can be identified with the set of integers \(\mathbb{Z}\), where the coset \((m, n) + H\) corresponds to the integer \(m - n\).

Formally, we have:
\[
(\mathbb{Z} \times \mathbb{Z})/H \cong \mathbb{Z}
\]

### Step 4: Group Operation in the Quotient Group
The group operation in \((\mathbb{Z} \times \mathbb{Z})/H\) is defined by:
\[
((m_1, n_1) + H) + ((m_2, n_2) + H) = (m_1 + m_2, n_1 + n_2) + H
\]
This corresponds to the addition of the integers \(m_1 - n_1\) and \(m_2 - n_2\):
\[
(m_1 - n_1) + (m_2 - n_2) = (m_1 + m_2) - (n_1 + n_2)
\]

Thus, the quotient group \((\mathbb{Z} \times \mathbb{Z})/H\) is isomorphic to \(\mathbb{Z}\) with the usual addition of integers.

## Historical Notes on Galois and Noether

The concept of quotient groups, though not explicitly formulated in the modern sense, has its roots in the work of Évariste Galois and Emmy Noether. 

### Évariste Galois (1811-1832)
Galois, a French mathematician, is best known for his work on the solvability of polynomial equations by radicals. In his work, he introduced the idea of a group, which he used to study the symmetries of the roots of polynomials. Although Galois did not use the term "quotient group," his work laid the foundation for the development of group theory and, by extension, the concept of quotient groups. His ideas were revolutionary and provided a new way to understand the structure of algebraic equations.

### Emmy Noether (1882-1935)
Emmy Noether, a German mathematician, made significant contributions to abstract algebra, including the formalization of quotient groups. Noether's work on ring theory and her famous Noether's theorems in physics highlighted the importance of symmetry and invariants. She introduced the concept of a quotient group in the context of abstract algebra, providing a rigorous framework for understanding the structure of groups and their subgroups. Noether's insights have had a profound impact on the development of modern algebra and continue to influence mathematical research today.

## The Correspondence Theorem with a Concrete Example

The correspondence theorem, also known as the lattice isomorphism theorem, establishes a one-to-one correspondence between the subgroups of a quotient group \(G/N\) and the subgroups of \(G\) that contain the normal subgroup \(N\). Formally, if \(N\) is a normal subgroup of \(G\), then there is a bijection between the set of subgroups of \(G/N\) and the set of subgroups of \(G\) that contain \(N\).

### Concrete Example: Subgroups of \((\mathbb{Z} \times \mathbb{Z})/H\)

Consider the quotient group \((\mathbb{Z} \times \mathbb{Z})/H\) where \(H = \{(a, a) : a \in \mathbb{Z}\}\). We have already established that \((\mathbb{Z} \times \mathbb{Z})/H \cong \mathbb{Z}\).

#### Subgroups of \(\mathbb{Z}\)
The subgroups of \(\mathbb{Z}\) are of the form \(k\mathbb{Z}\) for \(k \in \mathbb{Z}\), where \(k\mathbb{Z} = \{kn : n \in \mathbb{Z}\}\).

#### Corresponding Subgroups of \(\mathbb{Z} \times \mathbb{Z}\)
For each \(k \in \mathbb{Z}\), the corresponding subgroup of \(\mathbb{Z} \times \mathbb{Z}\) that contains \(H\) is:
\[
K_k = \{(m, n) \in \mathbb{Z} \times \mathbb{Z} : m - n \in k\mathbb{Z}\}
\]
This means that \(K_k\) consists of all pairs \((m, n)\) such that \(m - n\) is a multiple of \(k\).

For example:
- For \(k = 1\), \(K_1 = \mathbb{Z} \times \mathbb{Z}\) (since \(m - n\) is always an integer).
- For \(k = 2\), \(K_2 = \{(m, n) : m - n \text{ is even}\}\).

### Verification
To verify the correspondence, note that:
- The subgroup \(k\mathbb{Z}\) of \(\mathbb{Z}\) corresponds to the subgroup \(K_k\) of \(\mathbb{Z} \times \mathbb{Z}\) containing \(H\).
- The quotient group \((\mathbb{Z} \times \mathbb{Z})/K_k\) is isomorphic to \(\mathbb{Z}/(k\mathbb{Z}) \cong \mathbb{Z}_k\), the cyclic group of order \(k\).

Thus, the correspondence theorem provides a clear and systematic way to understand the structure of subgroups in quotient groups, as demonstrated in this concrete example.

## What's Next

We now have a powerful toolkit: normal subgroups let us build quotients, homomorphisms let us compare groups, and the isomorphism theorems reveal deep structural connections. The next article takes on a different challenge: given a finite group $G$, how can we find subgroups of specific orders? The **Sylow theorems** provide the answer, giving us existence, conjugacy, and counting results for subgroups of prime-power order. These theorems are the sharpest tool available for classifying finite groups, and they build directly on the normal subgroup machinery we've developed here.

---

*This is Part 3 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 2 — Group Actions and Symmetry](/en/abstract-algebra/02-group-actions-and-symmetry/)*

*Next: [Part 4 — Sylow Theorems](/en/abstract-algebra/04-sylow-theorems/)*
