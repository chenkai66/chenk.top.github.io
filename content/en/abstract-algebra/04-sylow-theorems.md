---
title: "Sylow Theorems: Dissecting Finite Groups"
date: 2021-09-07 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - sylow-theorems
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "The Sylow theorems give us a systematic way to find and count subgroups of prime-power order — the sharpest tool for classifying finite groups."
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "abstract-algebra-4"
---

The structure theory of finite groups rests on one fundamental observation: the prime factorization of $|G|$ constrains what subgroups $G$ can have. Lagrange's theorem tells us that the order of any subgroup divides $|G|$, but the converse is false in general — not every divisor of $|G|$ corresponds to a subgroup. The Sylow theorems dramatically sharpen the picture: for each prime power dividing $|G|$, a subgroup of that order not only exists but has strong structural properties. These results, proved by Ludwig Sylow in 1872, remain the single most effective tool for analyzing finite groups.

---

## The Classification Problem for Finite Groups

Given an integer $n$, how many groups of order $n$ are there, up to isomorphism? For $n = 1$, there is exactly one (the trivial group). For $n = p$ (prime), there is exactly one (the cyclic group $\mathbb{Z}/p\mathbb{Z}$). But for composite $n$, the situation explodes: there are 2 groups of order 4, 2 of order 6, 5 of order 8, and 267 groups of order 64.

A systematic approach requires tools for:
1. **Existence:** Proving that subgroups of certain orders must exist.
2. **Uniqueness/Conjugacy:** Determining when such subgroups are essentially the same.
3. **Counting:** Bounding or determining the exact number of such subgroups.

Lagrange's theorem provides a necessary condition (subgroup orders divide $|G|$) but not a sufficient one. Cauchy's theorem provides a partial converse for prime divisors. The Sylow theorems complete the picture for prime-power divisors, giving us all three tools above.

---


![Subgroup lattice of S_3 showing Sylow subgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_fig4_sylow_lattice.png)

## p-Groups and Cauchy's Theorem

**Definition.** Let $p$ be a prime. A **$p$-group** is a group in which every element has order a power of $p$. For finite groups, this is equivalent to saying $|G| = p^k$ for some $k \geq 0$.

$p$-groups are the building blocks from which all finite groups are assembled, much as prime numbers are the building blocks of integers.

**Key property of $p$-groups.** If $G$ is a finite $p$-group, then its center $Z(G)$ is nontrivial: $|Z(G)| > 1$.

**Proof via the class equation.** Recall the class equation for a finite group $G$:
$$|G| = |Z(G)| + \sum_{i} [G : C_G(g_i)]$$
where the sum runs over one representative $g_i$ from each conjugacy class of size $> 1$, and $C_G(g_i)$ is the centralizer of $g_i$.

Each term $[G : C_G(g_i)]$ divides $|G| = p^k$ and is $> 1$, so each term is divisible by $p$. The left side $|G| = p^k$ is divisible by $p$. Therefore $|Z(G)|$ is divisible by $p$, which means $|Z(G)| \geq p > 1$. $\blacksquare$

This simple argument has enormous consequences. It immediately implies:

**Corollary.** Every group of order $p^2$ is abelian.

*Proof.* Let $|G| = p^2$. Then $|Z(G)| \in \{p, p^2\}$ (since $Z(G)$ is nontrivial and $|Z(G)|$ divides $|G|$). If $|Z(G)| = p^2$, then $G = Z(G)$ is abelian. If $|Z(G)| = p$, then $|G/Z(G)| = p$, so $G/Z(G)$ is cyclic. But if $G/Z(G)$ is cyclic, then $G$ is abelian (a standard exercise), contradicting $|Z(G)| = p < p^2$. $\blacksquare$

**Cauchy's Theorem.** If a prime $p$ divides $|G|$, then $G$ contains an element of order $p$.

**Proof (McKay's elegant argument via group actions).** Consider the set
$$S = \{(g_1, g_2, \ldots, g_p) \in G^p : g_1 g_2 \cdots g_p = e\}.$$

Any choice of $g_1, \ldots, g_{p-1}$ determines $g_p = (g_1 \cdots g_{p-1})^{-1}$, so $|S| = |G|^{p-1}$.

The cyclic group $\mathbb{Z}/p\mathbb{Z}$ acts on $S$ by cyclic permutation: $k \cdot (g_1, \ldots, g_p) = (g_{k+1}, \ldots, g_p, g_1, \ldots, g_k)$. This action is well-defined because if $g_1 \cdots g_p = e$, then any cyclic rearrangement also has product $e$ (since $g_2 \cdots g_p g_1 = g_1^{-1}(g_1 \cdots g_p)g_1 = g_1^{-1} e g_1 = e$).

Every orbit has size 1 or $p$ (since $p$ is prime). The fixed points are exactly the tuples $(g, g, \ldots, g)$ where $g^p = e$. The identity $(e, e, \ldots, e)$ is one such fixed point.

By the orbit-counting formula: $|S| = (\text{number of fixed points}) + p \cdot (\text{number of orbits of size } p)$. Since $|S| = |G|^{p-1}$ and $p \mid |G|$, we have $p \mid |S|$. Therefore $p$ divides the number of fixed points. Since there's at least one fixed point (the identity tuple), there must be at least $p$ fixed points, meaning there exists $g \neq e$ with $g^p = e$. This $g$ has order $p$. $\blacksquare$

This proof is a beautiful example of the "action counting" technique that pervades the Sylow theory.

**Further consequences of Cauchy's theorem.** Combined with Lagrange's theorem, Cauchy's theorem immediately shows that a finite group $G$ has an element of order $p$ for every prime $p$ dividing $|G|$. This is a partial converse to Lagrange: while not every divisor of $|G|$ is the order of a subgroup, every *prime* divisor is. The Sylow theorems push this further: every *prime power* divisor of the form $p^a$ (where $p^a$ is the full power of $p$ in $|G|$) is the order of a subgroup.

**Cauchy's theorem for abelian groups (alternative proof).** If $G$ is abelian and $p \mid |G|$, we can give a simpler proof by induction on $|G|$. Pick any $g \neq e$. If $p \mid |g|$, then $g^{|g|/p}$ has order $p$ and we're done. If $p \nmid |g|$, consider the quotient $G/\langle g \rangle$. Since $p \mid |G|$ and $p \nmid |\langle g \rangle|$, we have $p \mid |G/\langle g \rangle|$. By induction, $G/\langle g \rangle$ has an element $\bar{h}$ of order $p$. Then $h^p \in \langle g \rangle$, say $h^p = g^k$. The element $h^p$ has order dividing $|\langle g \rangle|/\gcd(k, |\langle g \rangle|)$, and careful analysis yields an element of order $p$ in $G$.

---

## Sylow's First Theorem: Existence

**Definition.** Let $|G| = p^a m$ where $p \nmid m$ (i.e., $p^a$ is the largest power of $p$ dividing $|G|$). A **Sylow $p$-subgroup** of $G$ is a subgroup of order $p^a$.

Sylow's First Theorem guarantees these maximal $p$-subgroups always exist.

**Theorem (Sylow I).** Let $G$ be a finite group and $p$ a prime dividing $|G|$. Write $|G| = p^a m$ with $\gcd(p, m) = 1$ and $a \geq 1$. Then $G$ contains a subgroup of order $p^a$ (a Sylow $p$-subgroup). More generally, $G$ contains a subgroup of order $p^k$ for every $0 \leq k \leq a$.

**Proof (via group actions on coset spaces).** We prove the stronger statement by induction on $|G|$.

*Base case:* $|G| = 1$ is trivial.

*Inductive step:* Assume the theorem holds for all groups of order less than $|G|$. Consider the class equation:
$$|G| = |Z(G)| + \sum_i [G : C_G(g_i)]$$

**Case 1:** Some $[G : C_G(g_i)]$ is not divisible by $p^a$. Then $p^a$ divides $|C_G(g_i)|$ (since $|G| = [G : C_G(g_i)] \cdot |C_G(g_i)|$). Since $|C_G(g_i)| < |G|$ (because $g_i \notin Z(G)$), the inductive hypothesis gives a Sylow $p$-subgroup of $C_G(g_i)$, which is also a subgroup of $G$ of order $p^a$.

**Case 2:** Every $[G : C_G(g_i)]$ is divisible by $p$. Then $p$ divides $|Z(G)|$ (from the class equation, since $p$ divides $|G|$ and every non-central conjugacy class size). By Cauchy's theorem applied to the abelian group $Z(G)$, there exists $z \in Z(G)$ of order $p$. The subgroup $N = \langle z \rangle$ is normal in $G$ (since $z$ is central). Consider $G/N$, which has order $|G|/p = p^{a-1}m$. By induction, $G/N$ has a subgroup of order $p^{a-1}$. By the Lattice Correspondence Theorem, this corresponds to a subgroup $H$ of $G$ containing $N$ with $|H/N| = p^{a-1}$, so $|H| = p \cdot p^{a-1} = p^a$. $\blacksquare$

**The "more generally" part** (subgroups of every prime-power order up to $p^a$) follows from the same induction, or alternatively from the fact that a $p$-group of order $p^a$ has subgroups of every order $p^k$ for $0 \leq k \leq a$ (proved by induction using the nontrivial center property).

---

## Sylow's Second and Third Theorems: Conjugacy and Counting

**Theorem (Sylow II).** Any two Sylow $p$-subgroups of $G$ are conjugate. That is, if $P$ and $Q$ are both Sylow $p$-subgroups, then there exists $g \in G$ with $gPg^{-1} = Q$.

**Proof sketch.** Let $P$ act on the set of left cosets $G/Q$ by left multiplication. We count fixed points. The coset $gQ$ is fixed by all of $P$ if and only if $P \subseteq gQg^{-1}$. The number of fixed points is congruent to $|G/Q| = |G|/|Q| = m \pmod{p}$ (by a counting argument using the fact that non-fixed orbits have size divisible by $p$). Since $\gcd(m, p) = 1$, there is at least one fixed point, say $gQ$. Then $P \subseteq gQg^{-1}$. Since both $P$ and $gQg^{-1}$ have order $p^a$, we get $P = gQg^{-1}$. $\blacksquare$

**Theorem (Sylow III).** Let $n_p$ denote the number of Sylow $p$-subgroups of $G$. Then:
1. $n_p \mid m$ (where $|G| = p^a m$, $\gcd(p, m) = 1$).
2. $n_p \equiv 1 \pmod{p}$.

**Proof sketch.** Let $\text{Syl}_p(G)$ denote the set of all Sylow $p$-subgroups. By Sylow II, $G$ acts transitively on $\text{Syl}_p(G)$ by conjugation. Fix $P \in \text{Syl}_p(G)$. The stabilizer of $P$ under this action is $N_G(P)$ (the normalizer). By the orbit-stabilizer theorem, $n_p = [G : N_G(P)]$, which divides $|G|$. Since $P \leq N_G(P)$, we have $p^a \mid |N_G(P)|$, so $n_p = |G|/|N_G(P)|$ divides $|G|/p^a = m$. This gives condition (1).

For condition (2), let $P$ act on $\text{Syl}_p(G)$ by conjugation. The fixed points are those $Q \in \text{Syl}_p(G)$ with $P \subseteq N_G(Q)$, which means $PQ$ is a subgroup of $G$. If $Q$ is a fixed point, then $PQ$ is a group with $|PQ| = |P||Q|/|P \cap Q|$. Since both $|P|$ and $|Q|$ are $p^a$, and $|PQ| \leq |G|$, the only way this works when $P \neq Q$ would require $|PQ| > p^a$, but then $p^a$ doesn't divide $|PQ|/p^a$... the careful analysis shows the only fixed point is $P$ itself. So the number of fixed points is 1, and since non-fixed orbits have size divisible by $p$, we get $n_p \equiv 1 \pmod{p}$. $\blacksquare$

**Summary.** Writing $|G| = p^a m$ with $\gcd(p, m) = 1$:
- **(Sylow I):** Sylow $p$-subgroups exist (subgroups of order $p^a$).
- **(Sylow II):** All Sylow $p$-subgroups are conjugate to each other.
- **(Sylow III):** The number $n_p$ of Sylow $p$-subgroups satisfies $n_p \mid m$ and $n_p \equiv 1 \pmod{p}$.

**Key corollary.** A Sylow $p$-subgroup $P$ is normal in $G$ if and only if $n_p = 1$. (Since all Sylow $p$-subgroups are conjugate, $P$ is normal iff it is the only one.)

**Normalizer growth.** An important consequence of Sylow II is that $N_G(N_G(P)) = N_G(P)$ for any Sylow $p$-subgroup $P$. This "normalizer doesn't grow" property is proved as follows: if $g \in N_G(N_G(P))$, then $gPg^{-1}$ is a Sylow $p$-subgroup of $N_G(P)$. But $P$ is the unique Sylow $p$-subgroup of $N_G(P)$ (it is normal there by definition), so $gPg^{-1} = P$, meaning $g \in N_G(P)$.

This property is used in transfer theory and in proving Burnside's normal $p$-complement theorem, which states that if a Sylow $p$-subgroup $P$ lies in the center of its normalizer ($P \leq Z(N_G(P))$), then $G$ has a normal subgroup $N$ with $G/N \cong P$.

---

## Classifying Groups of Small Order

The Sylow theorems become a powerful classification tool when combined with semidirect products and other structural results. Let us work through several examples.

### Groups of order 6

$|G| = 6 = 2 \cdot 3$. By Sylow III:
- $n_3 \mid 2$ and $n_3 \equiv 1 \pmod{3}$, so $n_3 \in \{1, 2\} \cap \{1, 4, 7, \ldots\} = \{1\}$.
- $n_2 \mid 3$ and $n_2 \equiv 1 \pmod{2}$, so $n_2 \in \{1, 3\}$.

Since $n_3 = 1$, the unique Sylow 3-subgroup $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ is normal.

**Case $n_2 = 1$:** Both $P_2$ and $P_3$ are normal. Since $|P_2| = 2$, $|P_3| = 3$, and $\gcd(2, 3) = 1$, we get $P_2 \cap P_3 = \{e\}$ and $|P_2 P_3| = 6 = |G|$, so $G = P_2 \times P_3 \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}/6\mathbb{Z}$.

**Case $n_2 = 3$:** There are 3 Sylow 2-subgroups, each of order 2. The group is non-abelian (otherwise $n_2 = 1$). We have a normal subgroup $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ and elements of order 2 acting on it by conjugation. The only non-trivial automorphism of $\mathbb{Z}/3\mathbb{Z}$ is inversion ($x \mapsto x^{-1}$). This gives $G \cong S_3$ (the unique non-abelian group of order 6).

**Conclusion:** Up to isomorphism, the groups of order 6 are $\mathbb{Z}/6\mathbb{Z}$ and $S_3$.

### Groups of order 8

$|G| = 8 = 2^3$. This is a $p$-group, so the Sylow theorems don't directly constrain internal structure (there's only one prime). Instead, we use the fact that $|Z(G)| > 1$, and specifically $|Z(G)| \in \{2, 4, 8\}$.

- If $|Z(G)| = 8$: $G$ is abelian. The abelian groups of order 8 are $\mathbb{Z}/8\mathbb{Z}$, $\mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, and $(\mathbb{Z}/2\mathbb{Z})^3$ (by the classification of finitely generated abelian groups).
- If $|Z(G)| = 4$: $G/Z(G)$ has order 2, which is cyclic, forcing $G$ to be abelian — contradiction. So this case doesn't occur.
- If $|Z(G)| = 2$: $G/Z(G)$ has order 4, which is either $\mathbb{Z}/4\mathbb{Z}$ or $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$.
  - If $G/Z(G) \cong \mathbb{Z}/4\mathbb{Z}$: this is cyclic, again forcing $G$ abelian — contradiction.
  - If $G/Z(G) \cong (\mathbb{Z}/2\mathbb{Z})^2$: this is consistent. The two non-abelian groups that arise are the **dihedral group** $D_4$ (symmetries of a square) and the **quaternion group** $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$.

**Conclusion:** There are exactly 5 groups of order 8: three abelian ($\mathbb{Z}/8\mathbb{Z}$, $\mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, $(\mathbb{Z}/2\mathbb{Z})^3$) and two non-abelian ($D_4$, $Q_8$).

### Groups of order 12

$|G| = 12 = 2^2 \cdot 3$. By Sylow III:
- $n_3 \mid 4$ and $n_3 \equiv 1 \pmod{3}$, so $n_3 \in \{1, 4\}$.
- $n_2 \mid 3$ and $n_2 \equiv 1 \pmod{2}$, so $n_2 \in \{1, 3\}$.

**Case $n_3 = 1$:** The unique Sylow 3-subgroup $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ is normal. The Sylow 2-subgroup $P_2$ has order 4, so $P_2 \cong \mathbb{Z}/4\mathbb{Z}$ or $P_2 \cong (\mathbb{Z}/2\mathbb{Z})^2$. The group $G$ is a semidirect product $P_3 \rtimes P_2$, classified by the action of $P_2$ on $P_3$.

- Trivial action gives direct products: $\mathbb{Z}/3\mathbb{Z} \times \mathbb{Z}/4\mathbb{Z} \cong \mathbb{Z}/12\mathbb{Z}$, or $\mathbb{Z}/3\mathbb{Z} \times (\mathbb{Z}/2\mathbb{Z})^2 \cong \mathbb{Z}/6\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$.
- Non-trivial action with $P_2 \cong \mathbb{Z}/4\mathbb{Z}$: the element of order 4 in $P_2$ acts on $P_3$ by an automorphism of order dividing 4. But $\text{Aut}(\mathbb{Z}/3\mathbb{Z}) \cong \mathbb{Z}/2\mathbb{Z}$ has no element of order 4. The only non-trivial action factors through the quotient $\mathbb{Z}/4\mathbb{Z} \to \mathbb{Z}/2\mathbb{Z}$, which yields the dicyclic group $\text{Dic}_3$ of order 12.
- Non-trivial action with $P_2 \cong (\mathbb{Z}/2\mathbb{Z})^2$: the non-trivial homomorphism $(\mathbb{Z}/2\mathbb{Z})^2 \to \text{Aut}(\mathbb{Z}/3\mathbb{Z}) \cong \mathbb{Z}/2\mathbb{Z}$ has kernel of order 2, giving $D_6$ (the dihedral group of order 12, also written $D_6$ since it has 6 rotational symmetries).

**Case $n_3 = 4$:** There are 4 Sylow 3-subgroups. Each has order 3 and pairwise intersection $\{e\}$, contributing $4 \times 2 = 8$ elements of order 3. That leaves $12 - 8 = 4$ remaining elements, which must form the unique Sylow 2-subgroup (so $n_2 = 1$). The group $G$ acts on its 4 Sylow 3-subgroups by conjugation, giving a homomorphism $G \to S_4$. With careful analysis, this case yields $A_4$ (the alternating group on 4 letters).

**Conclusion:** There are exactly 5 groups of order 12: $\mathbb{Z}/12\mathbb{Z}$, $\mathbb{Z}/6\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, $D_6$, $\text{Dic}_3$, and $A_4$.

### The general strategy

The examples above illustrate a recurring pattern for classifying groups of order $n$:

1. **Factor** $n = p_1^{a_1} \cdots p_k^{a_k}$.
2. **Apply Sylow III** to each prime: compute the allowed values of $n_{p_i}$.
3. **Identify forced normal subgroups:** If $n_{p_i} = 1$ for some $i$, the unique Sylow $p_i$-subgroup is normal.
4. **Count elements** to detect contradictions: elements of prime order from different Sylow subgroups are distinct (since their orders are coprime), which often forces $n_{p_i} = 1$ for at least one prime.
5. **Classify extensions:** Once normal subgroups are identified, the group is a semidirect product, classified by homomorphisms into automorphism groups.
6. **Handle the remaining cases** by embedding into symmetric groups (via the conjugation action on Sylow subgroups).

This strategy becomes increasingly powerful as you build intuition for which orders are "easy" (like $pq$ with $q \not\equiv 1 \pmod p$) and which require more work (like $p^3$ or $p^2 q$).

---

## Applications and Non-Abelian Examples

**Application 1: Groups of order $pq$ ($p < q$ primes).** By Sylow III, $n_q \mid p$ and $n_q \equiv 1 \pmod{q}$. Since $p < q$, the only possibility is $n_q = 1$. So the Sylow $q$-subgroup is unique and normal. If additionally $q \not\equiv 1 \pmod{p}$, then $n_p = 1$ as well, and $G \cong \mathbb{Z}/pq\mathbb{Z}$. If $q \equiv 1 \pmod{p}$, there is also a non-abelian semidirect product.

**Example:** Groups of order 15 ($= 3 \times 5$). We have $n_5 \mid 3$ and $n_5 \equiv 1 \pmod{5}$, so $n_5 = 1$. Also $n_3 \mid 5$ and $n_3 \equiv 1 \pmod{3}$, so $n_3 = 1$. Both Sylow subgroups are normal, so $G \cong \mathbb{Z}/3\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z} \cong \mathbb{Z}/15\mathbb{Z}$. There is only one group of order 15.

**Application 2: No simple group of order 12.** Suppose $G$ is simple with $|G| = 12$. Then $n_3 = 4$ (since $n_3 = 1$ would give a normal Sylow 3-subgroup). The conjugation action on the four Sylow 3-subgroups gives $\varphi: G \to S_4$. Since $G$ is simple, $\ker \varphi$ is trivial or all of $G$. It can't be all of $G$ (the action is non-trivial since the Sylow 3-subgroups are conjugate). So $\varphi$ is injective, embedding $G$ into $S_4$. But then $G$ is a subgroup of $S_4$ of order 12, which must be $A_4$. However $A_4$ has a normal subgroup $V_4$, so $A_4$ is not simple — contradiction. Therefore no simple group of order 12 exists.

**Application 3: $|G| = 30$ implies $G$ has a normal Sylow 5-subgroup.** We have $|G| = 30 = 2 \cdot 3 \cdot 5$. By Sylow III: $n_5 \mid 6$ and $n_5 \equiv 1 \pmod{5}$, so $n_5 \in \{1, 6\}$. Also $n_3 \mid 10$ and $n_3 \equiv 1 \pmod{3}$, so $n_3 \in \{1, 10\}$.

If $n_5 = 6$: there are $6 \times 4 = 24$ elements of order 5. If also $n_3 = 10$: there are $10 \times 2 = 20$ elements of order 3. Total elements of order 3 or 5: at least $24 + 20 = 44 > 30$. Contradiction. So $n_3 = 1$ when $n_5 = 6$.

With $n_3 = 1$, the normal Sylow 3-subgroup $P_3$ gives a quotient $G/P_3$ of order 10. In $G/P_3$, the image of any Sylow 5-subgroup is a subgroup of order 5, and $n_5(G/P_3) \mid 2$ and $n_5(G/P_3) \equiv 1 \pmod{5}$, so $n_5(G/P_3) = 1$. Lifting back, the preimage is a normal subgroup of $G$ of order 15. Since 15 = 3 $\times$ 5, any group of order 15 is cyclic (as shown above), so this subgroup has a characteristic (hence normal in $G$) Sylow 5-subgroup. This gives $n_5 = 1$ — contradicting our assumption.

Therefore $n_5 = 1$: every group of order 30 has a normal Sylow 5-subgroup.

**Application 4: The Sylow theorems detect non-isomorphic groups.** Consider two groups of order 20: $\mathbb{Z}/20\mathbb{Z}$ and the general affine group $GA_1(\mathbb{F}_5) = \{x \mapsto ax + b : a \in \mathbb{F}_5^*, b \in \mathbb{F}_5\}$. Both have order 20 = $2^2 \cdot 5$.

For $\mathbb{Z}/20\mathbb{Z}$: it is abelian, so every subgroup is normal. We have $n_5 = 1$ and $n_2 = 1$.

For $GA_1(\mathbb{F}_5)$: the translations $\{x \mapsto x + b\}$ form the unique normal Sylow 5-subgroup ($n_5 = 1$). The Sylow 2-subgroups are the groups $\{x \mapsto ax : a \in \langle g \rangle\}$ where $g$ has order 4 in $\mathbb{F}_5^*$ (there's only one such subgroup since $\mathbb{F}_5^* \cong \mathbb{Z}/4\mathbb{Z}$). But these Sylow 2-subgroups are conjugate under translation, and one can verify $n_2 = 5$. Since $n_2$ differs between the two groups, they are not isomorphic.

**Application 5: Simplicity of $A_5$.** The alternating group $A_5$ (order 60) is the smallest non-abelian simple group. We can verify this using Sylow theory. The Sylow subgroup counts are: $n_2 \in \{1, 3, 5, 15\}$, $n_3 \in \{1, 4, 10\}$, $n_5 \in \{1, 6\}$.

If $n_5 = 1$, there'd be a normal Sylow 5-subgroup. But $A_5$ acts transitively on 5 elements, and one can verify directly that conjugating any 5-cycle by different permutations produces different Sylow 5-subgroups. In fact $n_5 = 6$, $n_3 = 10$, and $n_2 = 5$. The element count: $6 \times 4 = 24$ elements of order 5, $10 \times 2 = 20$ elements of order 3, and the Sylow 2-subgroups (each isomorphic to $V_4$, the Klein four-group) contribute additional elements. No Sylow subgroup is unique, so none is normal. A more careful argument (checking all possible orders of normal subgroups) confirms $A_5$ is simple.

---

## What's Next

The Sylow theorems give us existence, conjugacy, and counting for prime-power subgroups. Combined with the quotient group and homomorphism machinery from the previous article, we can now dissect finite groups with real precision. The next article moves from subgroups to the internal structure of groups themselves: **group actions**, the orbit-stabilizer theorem, Burnside's lemma, and their applications to combinatorics and geometry. Group actions are the mechanism by which abstract groups connect to concrete mathematics — counting colorings, analyzing symmetry, and proving theorems about groups themselves.

---

*This is Part 4 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 3 — Quotient Groups and Homomorphisms](/en/abstract-algebra/03-quotient-groups-and-homomorphisms/)*

*Next: [Part 5 — Rings and Ideals](/en/abstract-algebra/05-rings-and-ideals/)*
