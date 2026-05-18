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

![Classification of groups of small order](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_group_classification.png)

Lagrange's theorem tells you the order of any subgroup must divide $|G|$. That is a necessary condition, and it is famously *not* sufficient — the alternating group $A_4$ has order $12$, but no subgroup of order $6$. So the moment you start asking "given $|G| = n$, what does $G$ actually look like?", Lagrange leaves you holding an empty bag.

The Sylow theorems are what go inside that bag. They say: for every maximal prime power $p^a$ dividing $|G|$, a subgroup of order $p^a$ exists, all such subgroups are conjugate, and their count $n_p$ is sharply constrained ($n_p \equiv 1 \pmod p$, $n_p \mid [G:P]$). Ludwig Sylow proved this in 1872, and 150 years later it is still the first thing you reach for when somebody hands you a finite group of unknown order and asks what it is.

The whole point of this article is to make those three statements feel inevitable rather than magical, and then to push them as far as they go: classifying small groups, ruling out simple groups at certain orders, and getting to the threshold where $A_5$ — order $60$ — finally cracks open the door to nonabelian simple groups.

![Conjugation action and Sylow counting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_conjugation_action.png)

I should be upfront about the dependency graph here. To make Sylow do real work you need three preliminaries glued in your head: the class equation, the orbit-stabilizer formula, and the basic theory of group actions. Sylow is essentially a clever choice of action ($G$ acting on a carefully chosen set of $p$-power-sized subsets, or on the set of its own Sylow subgroups by conjugation) followed by orbit-stabilizer, then a residue calculation modulo $p$. Once you internalize that pattern, the three theorems stop being three separate results and start looking like one technique applied three times.

---

## The Atoms: p-Groups, Cauchy, and the Class Equation

![Cauchy's theorem: p divides |G| implies element of order p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_cauchy_theorem.png)

Before the Sylow theorems proper, we need the atoms they decompose things into. A finite **$p$-group** is a group of order $p^k$. Equivalently (by Cauchy's theorem, which we prove below), every element has order a power of $p$. These are the building blocks; the whole Sylow program is a strategy for cutting an arbitrary finite group along its $p$-group pieces and reading off the structure.

The single most useful fact about a finite $p$-group $G$ is that its center $Z(G)$ is nontrivial. The proof is the class equation in its purest application:

$$|G| \;=\; |Z(G)| \;+\; \sum_i [G : C_G(g_i)],$$

where the sum runs over conjugacy classes of size $> 1$, with representatives $g_i$. Each index $[G : C_G(g_i)]$ divides $|G| = p^k$ and is bigger than $1$, so each is divisible by $p$. The left side is divisible by $p$. Therefore $|Z(G)|$ is divisible by $p$, hence at least $p$. End of argument.

That one observation immediately implies a chain of structural facts I will use repeatedly. Every group of order $p^2$ is abelian: if $|Z(G)| = p$, then $G/Z(G)$ has order $p$, hence is cyclic, hence $G$ is abelian by the "cyclic quotient by center implies abelian" lemma — contradiction, so $Z(G) = G$. Every $p$-group has a normal subgroup of every possible order $p^j$ for $0 \le j \le k$; this follows by induction on $k$ using the nontriviality of the center (quotient by a central element of order $p$, apply induction, lift back). Every nontrivial $p$-group has a chain of normal subgroups $\{e\} = N_0 \triangleleft N_1 \triangleleft \cdots \triangleleft N_k = G$ with $|N_j| = p^j$ — this is what people mean when they say $p$-groups are "as solvable as possible."

![Nested chain of subgroups in a p-group](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_1_p_group_chain.png)

The chain says you can build any $p$-group by repeatedly extending by $\mathbb{Z}/p\mathbb{Z}$. If you accept that, then knowing the abelian $p$-groups (which the fundamental theorem of finitely generated abelian groups hands you) plus extension data ($H^2$ in cohomological language) tells you everything — at least in principle. In practice, the extension data gets out of hand quickly ($267$ groups of order $64$, $2328$ of order $128$, over $10$ billion of order $512$), but the point is that the *framework* is complete.

Now Cauchy's theorem: if $p$ divides $|G|$, then $G$ has an element of order $p$. The cleanest proof is McKay's, and it is worth seeing because the same action-counting trick reappears throughout Sylow. Define

$$S \;=\; \{(g_1, g_2, \ldots, g_p) \in G^p : g_1 g_2 \cdots g_p = e\}.$$

The first $p-1$ entries are free, the last is forced: $g_p = (g_1 \cdots g_{p-1})^{-1}$. So $|S| = |G|^{p-1}$, which is divisible by $p$ (since $p \mid |G|$). Now let $\mathbb{Z}/p\mathbb{Z}$ act on $S$ by cyclic permutation of entries: $(g_1, \ldots, g_p) \mapsto (g_2, \ldots, g_p, g_1)$. This is well-defined because the product $g_1 \cdots g_p = e$ is invariant under cyclic permutation (to see this, note that $g_2 \cdots g_p g_1 = g_1^{-1}(g_1 g_2 \cdots g_p)g_1 = g_1^{-1} e \, g_1 = e$). Every orbit has size $1$ or $p$ (orbit-stabilizer for $\mathbb{Z}/p$, which has no intermediate subgroups). Orbits of size $1$ are tuples $(g, g, \ldots, g)$ with $g^p = e$. The identity $(e, e, \ldots, e)$ is one such orbit. Since $|S| \equiv 0 \pmod p$ and the number of fixed points is congruent to $|S| \pmod p$, there are at least $p$ fixed points total — so at least $p - 1$ nontrivial elements of order $p$.

The reason I presented Cauchy via McKay is that Sylow I uses the same philosophy: cook up a set, let something act on it, count orbits modulo $p$, and extract a subgroup from the arithmetic. The jump from Cauchy to Sylow is the jump from finding one element of order $p$ to finding an entire subgroup of order $p^a$. The technique is identical; only the choice of set on which you act changes. This conceptual unity — "Sylow is just Cauchy applied to a more carefully chosen set" — is what makes the proofs feel inevitable once you have seen them.

There is an intermediate result worth mentioning: **Cauchy-Sylow for $p$-subgroups**. If $P$ is a $p$-subgroup of $G$ with $|P| = p^j < p^a$ (where $p^a \| |G|$), then $P$ is contained in a strictly larger $p$-subgroup. The proof uses the same orbit-counting approach: let $P$ act on $G/P$ by left multiplication; $|G/P| = p^{a-j}m$ is divisible by $p$; the fixed points correspond to elements of $N_G(P)/P$; since the number of non-fixed-point orbits each have size divisible by $p$, we get $|N_G(P)/P| \equiv 0 \pmod p$; by Cauchy applied to $N_G(P)/P$, there is an element of order $p$ in this quotient, whose preimage gives a subgroup of order $p^{j+1}$ containing $P$. Iterating this until $j = a$ gives Sylow I. This version of the proof has the virtue of constructing the Sylow subgroup step-by-step, adding one factor of $p$ at a time, which mirrors how you often find Sylow subgroups in practice — start with a $p$-element, build up.

---

## The Three Sylow Theorems: Statement, Proof, and Structural Consequences

Fix a finite group $G$ with $|G| = p^a m$, where $p \nmid m$ (so $p^a$ is the exact power of $p$ dividing $|G|$). A **Sylow $p$-subgroup** is a subgroup of order $p^a$.

![Sylow p-subgroups: maximal p-power subgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_sylow_subgroups.png)

**Sylow I (Existence).** $G$ has at least one Sylow $p$-subgroup.

**Sylow II (Conjugacy).** Any two Sylow $p$-subgroups of $G$ are conjugate in $G$. Consequently, any $p$-subgroup of $G$ is contained in some Sylow $p$-subgroup.

**Sylow III (Counting).** The number $n_p$ of Sylow $p$-subgroups satisfies $n_p \equiv 1 \pmod p$ and $n_p \mid m$.

The second constraint in Sylow III — $n_p \mid m$ — is often written as $n_p \mid [G : P]$ for any Sylow $p$-subgroup $P$. They are the same since $[G : P] = |G|/p^a = m$.

**Proof of Sylow I.** Let $G$ act by left multiplication on $\Omega = \binom{G}{p^a}$, the set of all subsets of $G$ of size $p^a$. The key number-theoretic input: $|\Omega| = \binom{p^a m}{p^a}$, and this binomial coefficient is *not* divisible by $p$. (The cleanest way to see this: write $\binom{p^a m}{p^a} = \prod_{j=0}^{p^a - 1} \frac{p^a m - j}{p^a - j}$, and observe that $\nu_p(p^a m - j) = \nu_p(j)$ for $0 \leq j < p^a$ since $\nu_p(p^a m - j) = \nu_p(j)$ when $j < p^a$, so numerator and denominator have the same $p$-adic valuation factor by factor.) Since $|\Omega| \not\equiv 0 \pmod p$, not every orbit under the $G$-action has size divisible by $p$. Pick an orbit $\mathcal{O}$ with $p \nmid |\mathcal{O}|$. Fix $T \in \mathcal{O}$, and let $P = \mathrm{Stab}_G(T) = \{g \in G : gT = T\}$. By orbit-stabilizer, $|\mathcal{O}| = [G:P]$, so $p \nmid [G:P]$, meaning $p^a \mid |P|$. But $P$ acts freely on $T$ by left multiplication (if $gT = T$ and $gt = t$ for some $t \in T$, then $g = e$), so $|P| \leq |T| = p^a$. Thus $|P| = p^a$, and $P$ is a Sylow $p$-subgroup.

**Proof of Sylow II.** Let $P$ be a fixed Sylow $p$-subgroup and $Q$ any $p$-subgroup of $G$. Consider $Q$ acting by left multiplication on the set of left cosets $G/P$. The number of cosets is $[G:P] = m$, coprime to $p$. By the orbit-counting formula, the number of $Q$-fixed points is $\equiv m \not\equiv 0 \pmod p$, so there is at least one fixed coset $gP$. A fixed coset means $q(gP) = gP$ for all $q \in Q$, i.e., $g^{-1}qg \in P$ for all $q$, i.e., $g^{-1}Qg \subseteq P$. If $Q$ is itself Sylow (same order as $P$), then $g^{-1}Qg = P$, establishing conjugacy.

**Proof of Sylow III.** Let $\Sigma = \mathrm{Syl}_p(G)$ be the set of all Sylow $p$-subgroups, and let $G$ act on $\Sigma$ by conjugation. By Sylow II, this action is transitive, so $n_p = |\Sigma| = [G : N_G(P)]$ for any $P \in \Sigma$. Since $P \leq N_G(P) \leq G$, we have $n_p = [G:N_G(P)] \mid [G:P] = m$. For the congruence, let $P$ act on $\Sigma$ by conjugation. The fixed points are those $Q \in \Sigma$ with $P \subseteq N_G(Q)$, meaning $P$ and $Q$ are both Sylow $p$-subgroups of $N_G(Q)$. But $Q \trianglelefteq N_G(Q)$ (by definition of normalizer), so $Q$ is the unique Sylow $p$-subgroup of $N_G(Q)$ (a normal Sylow is unique). Hence the only fixed point is $P$ itself. Every other $P$-orbit has size divisible by $p$ (orbit-stabilizer for the $p$-group $P$). So $n_p = 1 + (\text{multiples of } p)$, i.e., $n_p \equiv 1 \pmod p$.

Two structural facts that ride alongside and get used constantly:

**Self-normalizing property.** $N_G(N_G(P)) = N_G(P)$ for any Sylow $p$-subgroup $P$. Proof: $P$ is the unique Sylow $p$-subgroup of $N_G(P)$ (it is normal there), so any $g$ normalizing $N_G(P)$ must send $P$ to itself (as the unique Sylow), hence $g \in N_G(P)$.

**Frattini argument.** If $N \trianglelefteq G$ and $P$ is a Sylow $p$-subgroup of $N$, then $G = N \cdot N_G(P)$. Proof: for any $g \in G$, $gPg^{-1}$ is also a Sylow $p$-subgroup of $N$ (since $N$ is normal, $gPg^{-1} \subseteq N$). By Sylow II applied within $N$, $gPg^{-1} = nPn^{-1}$ for some $n \in N$. Then $n^{-1}g \in N_G(P)$, so $g = n(n^{-1}g) \in N \cdot N_G(P)$.

![Normalizer N_G(P) and Sylow count](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_6_normalizer.png)

The Frattini argument is the workhorse for inductive arguments: if you can identify a normal subgroup $N$ and understand its Sylow structure, Frattini lets you factor the entire group as a product involving the normalizer of a Sylow subgroup of $N$. This is the tool that makes "Sylow + induction on $|G|$" a viable proof strategy for general structure theorems about finite groups.

Let me work a concrete example to show Sylow II in action. Consider $S_4$ and $p = 2$. The Sylow $2$-subgroups have order $8$ (since $|S_4| = 24 = 2^3 \cdot 3$). One such is the dihedral group $D_4 = \langle (1234), (13) \rangle$ (symmetries of a square, embedded via the action on vertices). Another is $\langle (1234), (24) \rangle$. Are they conjugate? Yes — by Sylow II they must be. Explicitly: $(23) D_4 (23)^{-1} = \langle (1324), (12) \rangle$... actually it is easier to just count. $n_2 = [S_4 : N_{S_4}(D_4)]$. Since $|D_4| = 8$ and $D_4 \leq N_{S_4}(D_4) \leq S_4$, and $|N_{S_4}(D_4)|$ divides $24$ and is a multiple of $8$, we get $|N_{S_4}(D_4)| \in \{8, 24\}$. If $|N_{S_4}(D_4)| = 24$, then $D_4 \trianglelefteq S_4$, which is false (the conjugate $(23)(1234)(23) = (1324) \notin D_4$). So $|N_{S_4}(D_4)| = 8$, meaning $N_{S_4}(D_4) = D_4$ itself, and $n_2 = 24/8 = 3$. Indeed, $n_2 \equiv 1 \pmod 2$ and $n_2 \mid 3$. ✓

The three Sylow $2$-subgroups of $S_4$ correspond to the three ways to partition $\{1,2,3,4\}$ into two pairs: $\{\{1,2\},\{3,4\}\}$, $\{\{1,3\},\{2,4\}\}$, $\{\{1,4\},\{2,3\}\}$. Each such partition determines a copy of $D_4$ — the symmetries that either preserve or swap the two pairs. This geometric picture is characteristic of how Sylow subgroups sit inside symmetric and alternating groups.

---

## The Congruence at Work: Forcing Normality and Uniqueness

The congruence $n_p \equiv 1 \pmod p$ combined with the divisibility $n_p \mid m$ is the sharpest tool in the kit. In many cases, the intersection of these two constraints is the singleton $\{1\}$, which means the Sylow $p$-subgroup is unique (hence normal). Let me systematically demonstrate how this works.

**The general $pq$ theorem.** Let $|G| = pq$ with primes $p < q$. Then $n_q \mid p$ and $n_q \equiv 1 \pmod q$. The divisors of $p$ are $1$ and $p$. Since $p < q$, we have $p \not\equiv 1 \pmod q$ (because $p < q$ means $p \leq q - 1$, and $p = 1$ would mean $p$ is not prime). So $n_q = 1$ — the Sylow $q$-subgroup $Q$ is always normal. For the Sylow $p$-subgroup: $n_p \mid q$ and $n_p \equiv 1 \pmod p$. Divisors of $q$: $1$ and $q$. If $q \equiv 1 \pmod p$ (i.e., $p \mid q - 1$), then $n_p \in \{1, q\}$, and both are possible. If $p \nmid q - 1$, then $n_p = 1$, both Sylows are normal, and $G \cong \mathbb{Z}/pq$.

Concrete instances: $|G| = 15 = 3 \cdot 5$: $3 \nmid 4$, so $G \cong \mathbb{Z}/15$. $|G| = 35 = 5 \cdot 7$: $5 \nmid 6$, so $G \cong \mathbb{Z}/35$. $|G| = 33 = 3 \cdot 11$: $3 \nmid 10$, so $G \cong \mathbb{Z}/33$. $|G| = 77 = 7 \cdot 11$: $7 \nmid 10$, so $G \cong \mathbb{Z}/77$. All settled by one line of modular arithmetic.

When $p \mid q - 1$, you get exactly two groups: the cyclic $\mathbb{Z}/pq$ (when $n_p = 1$) and a nonabelian semidirect product $\mathbb{Z}/q \rtimes \mathbb{Z}/p$ (when $n_p = q$). Example: $|G| = 6 = 2 \cdot 3$, and $2 \mid 3 - 1$, giving $\mathbb{Z}/6$ and $S_3$. Another: $|G| = 21 = 3 \cdot 7$, and $3 \mid 7 - 1 = 6$, giving $\mathbb{Z}/21$ and a nonabelian group of order $21$ (the unique one, which embeds as a subgroup of $\mathrm{Aff}(\mathbb{F}_7)$ via $x \mapsto ax + b$ with $a^3 = 1$).

The element-counting technique mentioned above is the second-most-useful tool after the congruence itself. Here is the key observation: if $P$ and $Q$ are distinct Sylow $p$-subgroups of prime order $p$, then $P \cap Q = \{e\}$ (since any nontrivial subgroup of a group of order $p$ is the whole group). So distinct Sylow $p$-subgroups of order $p$ share only the identity, meaning each contributes $p - 1$ elements of order $p$ that no other Sylow $p$-subgroup contains. If $n_p$ Sylow $p$-subgroups exist, they collectively account for $n_p(p-1)$ non-identity elements. When $p^a > p$ (i.e., the Sylow subgroup is not of prime order), the intersection arithmetic is more delicate — two Sylow subgroups can overlap in a proper non-trivial subgroup — but for prime-order Sylows the counting is clean and powerful.

![Semidirect product construction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_semi_direct.png)

**Order $12 = 2^2 \cdot 3$.** Sylow III: $n_3 \mid 4$ and $n_3 \equiv 1 \pmod 3$, so $n_3 \in \{1, 4\}$. $n_2 \mid 3$ and $n_2 \equiv 1 \pmod 2$, so $n_2 \in \{1, 3\}$. The congruence alone does not force uniqueness here. But element counting does: if $n_3 = 4$, that accounts for $4 \cdot 2 = 8$ elements of order $3$ (each Sylow $3$-subgroup has $2$ non-identity elements, and distinct Sylow $3$-subgroups intersect trivially since they have prime order). This leaves only $12 - 8 = 4$ non-order-$3$ elements, which must form the unique Sylow $2$-subgroup. So if $n_3 \neq 1$, then $n_2 = 1$. Either way, at least one Sylow subgroup is normal, and we can classify using semidirect products. The result: five groups of order $12$ ($\mathbb{Z}/12$, $\mathbb{Z}/6 \times \mathbb{Z}/2$, $A_4$, $D_6$, $\mathrm{Dic}_3$).

**Order $20 = 2^2 \cdot 5$.** $n_5 \mid 4$ and $n_5 \equiv 1 \pmod 5$. Divisors of $4$: $\{1, 2, 4\}$. Numbers $\equiv 1 \pmod 5$: $\{1, 6, 11, \ldots\}$. Intersection: $\{1\}$. So $n_5 = 1$ and the Sylow $5$-subgroup is unique and normal. This immediately gives $G = P_5 \rtimes P_2$ as a semidirect product, and the classification reduces to understanding homomorphisms $P_2 \to \mathrm{Aut}(\mathbb{Z}/5) \cong \mathbb{Z}/4$.

Since $P_2$ has order $4$ and is either $\mathbb{Z}/4$ or $V_4 = (\mathbb{Z}/2)^2$, we enumerate:

- $P_2 = \mathbb{Z}/4$, trivial action: $G \cong \mathbb{Z}/20$.
- $P_2 = \mathbb{Z}/4$, faithful action (generator maps to the generator $2 \in \mathbb{Z}/4 \cong \mathrm{Aut}(\mathbb{Z}/5)$): $G = F_{20}$, the Frobenius group of order $20$.
- $P_2 = \mathbb{Z}/4$, action with kernel $\mathbb{Z}/2$ (generator maps to the element of order $2$ in $\mathrm{Aut}(\mathbb{Z}/5)$, i.e., inversion $a \mapsto a^{-1}$): $G = \mathrm{Dic}_5$, the dicyclic group.
- $P_2 = V_4$, trivial action: $G \cong \mathbb{Z}/10 \times \mathbb{Z}/2$.
- $P_2 = V_4$, action with image $\mathbb{Z}/2$ (one generator acts by inversion, the other trivially): $G \cong D_{10}$.

Five groups of order $20$. The whole classification took one application of Sylow III (to force $n_5 = 1$) followed by routine semidirect product enumeration. This is the standard workflow: Sylow gives you a normal subgroup, then the rest is homomorphism-counting.

The exercise illustrates a general principle. Once Sylow forces a normal subgroup $N$ with complement $H$ (meaning $G = NH$ and $N \cap H = \{e\}$), the group is a semidirect product $N \rtimes H$, and the isomorphism type is determined by the conjugation action $\varphi : H \to \mathrm{Aut}(N)$, up to the equivalence: $\varphi$ and $\varphi'$ give isomorphic groups iff there exist $\alpha \in \mathrm{Aut}(N)$ and $\beta \in \mathrm{Aut}(H)$ with $\varphi' = \alpha \circ \varphi \circ \beta^{-1}$ (conjugating in $\mathrm{Aut}(N)$ by $\alpha$ and precomposing with $\beta^{-1}$). For order $20$, $\mathrm{Aut}(\mathbb{Z}/5) \cong \mathbb{Z}/4$ is abelian, so the $\alpha$-conjugation is trivial, and the classification reduces to orbits of the $\mathrm{Aut}(H)$-action on $\mathrm{Hom}(H, \mathbb{Z}/4)$. For $H = \mathbb{Z}/4$, $\mathrm{Aut}(H) \cong \mathbb{Z}/2$ acts by inversion, giving three orbits ($\varphi = 0$; $\varphi$ of order $2$, i.e., image $\{0,2\}$; $\varphi$ of order $4$, i.e., injective — noting that $\varphi$ and $-\varphi$ give the same orbit). For $H = V_4$, $\mathrm{Aut}(H) \cong S_3$, and we count orbits of $\mathrm{Hom}(V_4, \mathbb{Z}/4) = \mathrm{Hom}(V_4, \mathbb{Z}/2)$ (since $V_4$ has exponent $2$, images land in the unique subgroup of order $2$), giving two orbits (trivial; one generator maps nontrivially).

There is a subtlety worth flagging: for the $F_{20}$ case (faithful $\mathbb{Z}/4 \to \mathrm{Aut}(\mathbb{Z}/5)$), the generator $b$ acts by $bab^{-1} = a^2$ (since $2$ generates $(\mathbb{Z}/5)^\times$). This group is the **affine group** $\mathrm{GA}_1(\mathbb{F}_5) = \{x \mapsto ax + b : a \in \mathbb{F}_5^\times, b \in \mathbb{F}_5\}$, which is also the holomorph $\mathrm{Hol}(\mathbb{Z}/5)$. It is a Frobenius group: it acts transitively on a set of $5$ elements (the points of $\mathbb{F}_5$) with the property that every non-identity element fixes at most one point. Frobenius groups are a rich class in finite group theory, and $F_{20}$ is the simplest nontrivial example beyond the dihedral groups $D_p$ for odd primes $p$.

---

## Ruling Out Simple Groups and the Simplicity of $A_5$

Sylow earns its keep most dramatically when proving that certain orders cannot support simple groups. The two main techniques are the **embedding method** (conjugation action on Sylow subgroups embeds $G$ into a symmetric group) and **element counting** (too many Sylow subgroups of different primes would require more elements than $|G|$).

**No simple group of order $12$.** Suppose $G$ simple with $|G| = 12$. Then $n_3 \neq 1$ (a unique Sylow $3$-subgroup would be normal), so $n_3 = 4$. Conjugation on these four Sylow $3$-subgroups gives a homomorphism $\varphi : G \to S_4$. The kernel $\ker \varphi$ is normal in $G$; since $G$ is simple, $\ker \varphi \in \{e, G\}$. If $\ker \varphi = G$, the action is trivial, but the Sylow $3$-subgroups are conjugate to each other (by Sylow II), contradiction. So $\varphi$ is injective, embedding $G$ as a subgroup of order $12$ in $S_4$. The only subgroup of order $12$ in $S_4$ is $A_4$ (it is the unique subgroup of index $2$). But $A_4$ has the Klein four-group $V_4$ as a normal subgroup — contradicting our assumption that $G$ is simple.

**No simple group of order $30$.** $|G| = 30 = 2 \cdot 3 \cdot 5$. Sylow: $n_5 \in \{1, 6\}$, $n_3 \in \{1, 10\}$, $n_2 \in \{1, 3, 5, 15\}$. If $n_5 = 6$: six Sylow $5$-subgroups contribute $6 \cdot 4 = 24$ elements of order $5$. If simultaneously $n_3 = 10$: ten Sylow $3$-subgroups contribute $10 \cdot 2 = 20$ elements of order $3$. Total: $24 + 20 = 44 > 30$. Contradiction. So at least one of $n_5, n_3$ must equal $1$, giving a normal Sylow subgroup.

**No simple group of order $36$.** $n_3 \mid 4$, $n_3 \equiv 1 \pmod 3$, so $n_3 \in \{1, 4\}$. If $n_3 = 4$, conjugation gives $\varphi : G \to S_4$. But $|G| = 36 > |S_4| = 24$, so $\varphi$ cannot be injective — $\ker \varphi$ is a nontrivial normal subgroup, contradicting simplicity.

The general lesson: conjugation on $n_p$ Sylow subgroups gives $G \to S_{n_p}$. If $|G| > n_p!$, the kernel is nontrivial and we are done. Combined with element counting, this eliminates most composite orders below $60$.

**$A_5$ is simple.** $|A_5| = 60 = 2^2 \cdot 3 \cdot 5$. The conjugacy classes of $A_5$: $\{e\}$ ($1$), $20$ three-cycles, $15$ double transpositions, and two classes of $12$ five-cycles each (split because the $S_5$-conjugator uniting them is odd). Total: $1 + 20 + 15 + 12 + 12 = 60$.

Sylow counts: $n_5 = 24/4 = 6$ (from $24$ five-cycles, $4$ per Sylow $5$-subgroup). $n_3 = 20/2 = 10$ (from $20$ three-cycles, $2$ per Sylow $3$-subgroup). $n_2 = 5$ (from $15$ double transpositions distributed among $5$ copies of $V_4$). None equals $1$, so no Sylow is normal.

![Sylow argument for simple group of order 60](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_7_simple_60.png)

To show no *other* normal subgroup exists, case-split on $|N|$ for $N \trianglelefteq A_5$ proper and nontrivial. Key arguments: $|N| = 5$ contradicts $n_5 = 6$. $|N| = 4$ contradicts $n_2 = 5$. $|N| = 3$ contradicts $n_3 = 10$. $|N| = 15$: a group of order $15$ is cyclic with a characteristic Sylow $5$-subgroup, which would be normal in $A_5$, contradicting $n_5 \neq 1$. $|N| = 30$: index $2$, so $N$ is the kernel of a map $A_5 \to \mathbb{Z}/2$. But $A_5$ is generated by $3$-cycles (odd order, hence map to $0$), so the map is trivial — contradiction. The remaining cases ($|N| \in \{6, 10, 12, 20\}$) all contain Sylow subgroups that would be characteristic in $N$ hence normal in $A_5$, contradicting the Sylow counts.

Every case fails. $A_5$ is simple — the smallest nonabelian simple group, the obstruction to solving the quintic by radicals, the irreducible building block at the bottom of the classification of finite simple groups. The reason $60$ is special is that it is the smallest order where Sylow stops forcing normality on at least one Sylow subgroup *and* all backup techniques (element counting, symmetric group embedding) simultaneously fail to produce a normal subgroup.

The connection to the quintic deserves a sentence. The Galois group of the general quintic polynomial over $\mathbb{Q}$ is $S_5$, whose composition series has factors $\mathbb{Z}/2$ and $A_5$. Since $A_5$ is simple and nonabelian, the composition series has a non-cyclic simple factor, which by the Abel-Ruffini theorem means the quintic is not solvable by radicals. The chain of logic is: Sylow analysis shows $A_5$ has no normal subgroups $\Rightarrow$ $A_5$ is simple $\Rightarrow$ the composition series of $S_5$ has a non-abelian factor $\Rightarrow$ the quintic is unsolvable. The Sylow theorems are not just a classification tool; they are the mechanism by which an impossibility result about polynomial equations is proved.

---

## The Sylow Toolkit: Recipe, Boundaries, and the Local-Global Principle

The pattern of Sylow analysis is almost always the same five steps:

1. Factor $|G| = \prod p_i^{a_i}$.
2. For each prime $p_i$, compute the allowed values of $n_{p_i}$ from Sylow III: intersect $\{d : d \mid m_i\}$ (where $m_i = |G|/p_i^{a_i}$) with $\{k : k \equiv 1 \pmod{p_i}\}$.
3. If any $n_{p_i}$ is forced to be $1$, you have a normal subgroup. Use semidirect product theory to classify.
4. If no $n_{p_i}$ is forced to be $1$, count elements: each Sylow $p$-subgroup of order $p^a$ contributes elements of order dividing $p^a$ that are not in any other Sylow $p$-subgroup. If the total exceeds $|G|$, you have a contradiction — at least one Sylow count must be $1$.
5. If counting fails, embed $G$ into $S_{n_p}$ via conjugation on Sylow subgroups and use known structure of symmetric groups to derive contradictions.

Five steps, applied with discipline, classify all groups up to order $60$ by hand and prove non-simplicity for most composite orders in that range. Beyond $60$, the same techniques generalize but the bookkeeping gets heavier, and you eventually hand off to structure theorems for solvable groups and the classification of finite simple groups — both of which still rest on Sylow as foundational infrastructure.

A common beginner mistake: using $n_p \mid |G|$ instead of the sharper $n_p \mid m = |G|/p^a$. The weaker bound gives more candidate values, leading to unnecessary casework. Always use $m$.

One more piece of intuition worth internalizing. Sylow theory is the first place where the *interplay* between several primes inside one group becomes a structural tool rather than a numerical coincidence. If $|G| = p^a q^b$, you study the Sylow $p$- and Sylow $q$-subgroups separately, but the *interaction* between them — captured by the conjugation action of one on the other, or by which one is normal — controls the entire group up to extension data. Orders like $pq$, $p^2 q$, $pqr$ are tractable because small numbers of primes mean small degrees of freedom in the interaction. Order $60 = 2^2 \cdot 3 \cdot 5$ is the smallest three-prime case where every Sylow can fail to be normal *and* the interactions can avoid forcing a normal subgroup elsewhere.

This local-global pattern — "fix a prime, analyze the $p$-local structure, then assemble globally" — reappears throughout mathematics. In number theory, it is the Hasse principle: understand a Diophantine equation $p$-adically for each prime, then deduce global information. In algebraic geometry, it is localization at primes of a ring. In topology, it is $p$-local homotopy theory. The Sylow theorems are the entry point to all of this: they are the simplest example of a mathematical situation where prime-by-prime analysis, glued together with interaction data, gives you the whole picture.

**Why none of this works for infinite groups.** The proofs rely on orbit-stabilizer, divisibility of indices, and counting modulo $p$ — all requiring finiteness. In an infinite group, you can have torsion-free groups (no elements of finite order), non-conjugate maximal $p$-subgroups, or no maximal $p$-subgroup at all ($\mathbb{Z}[1/p]/\mathbb{Z}$ has $p$-subgroups of every finite $p$-power order but no maximal one). Partial replacements exist (Hall subgroups for solvable groups, profinite Sylow theory), but the clean three-theorem package is unique to finite groups.

One partial replacement deserves mention: **Hall's theorem** for solvable groups. Philip Hall proved (1928) that if $G$ is a finite *solvable* group and $|G| = mn$ with $\gcd(m, n) = 1$, then $G$ has a subgroup of order $m$ (a "Hall subgroup"), and all Hall subgroups of a given order are conjugate. This is a vast generalization of Sylow (which handles only prime-power orders) but requires solvability. For non-solvable groups, Hall subgroups need not exist: $A_5$ has order $60 = 4 \cdot 15$, and $A_5$ has no subgroup of order $15$ (since any such subgroup would be cyclic with a characteristic Sylow $5$-subgroup normal in $A_5$, contradicting simplicity). Hall's theorem is what makes solvable groups so much more tractable than general finite groups — and it is the mechanism behind the Burnside $p^a q^b$ theorem and Feit-Thompson.

A historical remark. Sylow's original 1872 paper proved all three theorems for permutation groups (not a restriction, by Cayley's theorem). The modern proofs via group actions on cosets and subsets are cleaner than Sylow's original inductive argument. The slickest approach — Wielandt's 1959 proof of Sylow I using the action on $p^a$-element subsets — is the one I gave above, requiring nothing beyond orbit-stabilizer and a binomial coefficient computation. The conceptual message: mere divisibility arithmetic of $|G|$ forces specific structural consequences, and this forcing is what makes finite group theory a subject with sharp theorems rather than loose heuristics.

The Sylow theorems also open the door to the *transfer homomorphism* and related tools. If $P$ is a Sylow $p$-subgroup of $G$ with $P \leq H \leq N_G(P)$, the transfer map $\mathrm{Ver} : G^{\mathrm{ab}} \to H^{\mathrm{ab}}$ provides information about how elements of $G$ project into abelian quotients of $H$. Burnside's normal $p$-complement theorem uses transfer: if a Sylow $p$-subgroup $P$ is central in its normalizer ($P \leq Z(N_G(P))$), then $G$ has a normal $p$-complement (a normal subgroup $N$ with $G = N \rtimes P$). This is a condition you can often verify directly from the Sylow data, and it gives structural information (a semidirect product decomposition) that goes beyond just counting Sylow subgroups.

---

## What's next

The next article moves from internal subgroup structure to the relationship between groups: **rings**, where two operations interact, and the ideals/quotients machinery that grows out of paying attention to that interaction. The bridge: when you classify groups via semidirect products $N \rtimes H$, the action $H \to \mathrm{Aut}(N)$ is a ring-like structure in disguise (the endomorphism ring of $N$ as a $\mathbb{Z}$-module). Rings make this explicit.

![Animation: counting Sylow subgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/04_sylow_counting.gif)

## Deeper Dive: Sylow in Concrete Numbers

Sylow theorems become much less mysterious once you have done two or three classifications by hand. The recipe is always the same: count Sylow $p$-subgroups, use the constraints, eliminate cases.

**Computation A: groups of order $15$ are cyclic.** Let $|G| = 15 = 3 \cdot 5$. By Sylow III, the number $n_5$ of Sylow $5$-subgroups satisfies $n_5 \equiv 1 \pmod 5$ and $n_5 \mid 3$. So $n_5 \in \{1\}$ — only $1$ works. Similarly $n_3 \equiv 1 \pmod 3$ and $n_3 \mid 5$, giving $n_3 \in \{1\}$. Both Sylow subgroups are unique, hence normal. Call them $P_5$ (order $5$) and $P_3$ (order $3$). They intersect trivially (different prime orders), so $P_3 P_5$ has order $15$, equal to $|G|$. Thus $G = P_3 \times P_5 \cong \mathbb{Z}/3 \times \mathbb{Z}/5 \cong \mathbb{Z}/15$. The cyclic group of order $15$ is the *only* group of order $15$, up to isomorphism.

**Computation B: groups of order $21$.** $|G| = 21 = 3 \cdot 7$. Sylow: $n_7 \equiv 1 \pmod 7$ and $n_7 \mid 3$, so $n_7 = 1$. The Sylow $7$-subgroup is normal. But $n_3 \equiv 1 \pmod 3$ and $n_3 \mid 7$, so $n_3 \in \{1, 7\}$. Two cases. If $n_3 = 1$, both are normal and we get $\mathbb{Z}/21 \cong \mathbb{Z}/3 \times \mathbb{Z}/7$. If $n_3 = 7$, the Sylow $3$-subgroups are not normal, and the structure is a non-trivial semidirect product $\mathbb{Z}/7 \rtimes \mathbb{Z}/3$ with the generator of $\mathbb{Z}/3$ acting on $\mathbb{Z}/7$ by multiplication by some element of order $3$ in $(\mathbb{Z}/7)^*$. Since $(\mathbb{Z}/7)^* \cong \mathbb{Z}/6$ has elements of order $3$ (namely $\{2, 4\}$), this action exists. So there are exactly *two* groups of order $21$: the cyclic and a Frobenius-type non-abelian.

This is the key point that separates $|G| = 15$ from $|G| = 21$: when the smaller prime $p$ does *not* divide $q - 1$ (here $3 \nmid 5 - 1 = 4$), only the cyclic group exists. When $p \mid q - 1$ (here $3 \mid 7 - 1 = 6$), a non-abelian semidirect product also exists.

**Computation C: $A_4$ has no subgroup of order $6$.** $A_4$ has order $12 = 2^2 \cdot 3$. By Lagrange, a subgroup of order $6$ is conceivable. But suppose $H \leq A_4$ with $|H| = 6$. Then $[A_4 : H] = 2$, so $H$ is normal (any index-$2$ subgroup is normal). The quotient $A_4 / H$ has order $2$, so for any $g \in A_4$, $g^2 \in H$. In particular, every $3$-cycle, squared, lies in $H$. The square of $(1\ 2\ 3)$ is $(1\ 3\ 2)$, which is also a $3$-cycle, and $A_4$ has eight $3$-cycles in total. So $H$ would contain at least eight $3$-cycles — but $|H| = 6$. Contradiction. So $A_4$ has no subgroup of order $6$, despite $6 \mid 12$. *Lagrange does not have a converse.* The example is the canonical demonstration of why "divides" is a necessary but not sufficient condition.

**Computation D: counting Sylow $2$-subgroups in $S_4$.** $|S_4| = 24 = 2^3 \cdot 3$. Sylow III says $n_2 \equiv 1 \pmod 2$ (odd) and $n_2 \mid 3$. So $n_2 \in \{1, 3\}$. Are the Sylow $2$-subgroups normal (i.e. is $n_2 = 1$)? A Sylow $2$-subgroup of $S_4$ has order $8$ and is isomorphic to $D_4$. The dihedral subgroups $D_4 \subset S_4$ are: the symmetries of a square with corners labelled $1, 2, 3, 4$. There are three ways to label the corners up to the symmetric group $S_4$ — wait, more precisely, three distinct $D_4$ subgroups, corresponding to the three ways to pair up $\{1, 2, 3, 4\}$ into two pairs of opposite corners: $\{1,3\}|\{2,4\}$, $\{1,2\}|\{3,4\}$, $\{1,4\}|\{2,3\}$. So $n_2 = 3$, none of them is normal, and they are all conjugate (by Sylow II). The intersection of all three Sylow $2$-subgroups is the Klein four-group $V_4$, which *is* normal in $S_4$.

**Computation E: groups of order $p^2$ are abelian.** Let $|G| = p^2$. By the class equation, $|G| = |Z(G)| + \sum [G : C_G(g)]$ where the sum is over conjugacy classes of size $> 1$. Each $[G : C_G(g)]$ for non-central $g$ is a non-trivial divisor of $p^2$, hence $p$ or $p^2$. Since $|Z(G)|$ and $|G|$ are both divisible by $p$, the sum of $[G : C_G(g)]$ must be divisible by $p$, which means $|Z(G)| \equiv 0 \pmod p$. So $|Z(G)| \in \{p, p^2\}$. If $|Z(G)| = p$, then $G/Z(G)$ is cyclic of order $p$, but "$G/Z(G)$ cyclic implies $G$ abelian" forces $|Z(G)| = |G| = p^2$, contradicting $|Z(G)| = p$. So $|Z(G)| = p^2$, i.e., $G$ is abelian. The two abelian groups of order $p^2$ are $\mathbb{Z}/p^2$ and $\mathbb{Z}/p \times \mathbb{Z}/p$. (Concretely for $p = 2$: $\mathbb{Z}/4$ and the Klein four-group.)

## Counterexamples That Sharpen the Sylow Picture

**Sylow does not classify; it constrains.** Even in cases where Sylow forces both Sylow subgroups to be normal, you still need to check whether the resulting product is a direct product or a semidirect product. For order $pq$ with $p < q$ and $p \mid q - 1$, the constraint $n_p \in \{1, q\}$ leaves two possibilities, and *both* can occur. Sylow gives the menu; you still need to read it.

**The converse of Lagrange is selective.** While groups of *prime power* order have subgroups of every dividing order (Cauchy + iteration), general groups do not. $A_4$ at order $12$ has no subgroup of order $6$. $A_5$ at order $60$ has no subgroup of order $30$ or $20$ or $15$ — only $1, 2, 3, 4, 5, 6, 10, 12, 60$. The lattice of subgroups of $A_5$ is one of the prettiest small lattices in algebra, exactly because the simplicity of $A_5$ forbids most divisor orders.

## Common Pitfalls for Beginners

The first pitfall: forgetting that Sylow's number-of-subgroups is *exact*, not "at most." Sylow III says $n_p \equiv 1 \pmod p$ and $n_p \mid [G : P]$, where the second condition is the index of any Sylow $p$-subgroup. Both must hold, simultaneously, with $n_p$ a positive integer. Cases like $|G| = 30$ become tractable only because $n_5 \in \{1, 6\}$ and $n_3 \in \{1, 10\}$, and one can rule out $n_5 = 6$ by counting elements: six Sylow $5$-subgroups would contribute $6 \cdot 4 = 24$ elements of order $5$, leaving $|G| - 24 = 6$ elements for everything else, including ten elements of order $3$ if $n_3 = 10$. Contradiction.

The second pitfall: confusing "Sylow $p$-subgroup" with "subgroup of order $p$." A Sylow $p$-subgroup has order $p^k$ where $p^k$ is the largest power of $p$ dividing $|G|$. For $|G| = 24 = 2^3 \cdot 3$, a Sylow $2$-subgroup has order $8$, not $2$.

The third pitfall: assuming that every group with a non-trivial centre is abelian. Having $|Z(G)| > 1$ is a useful structural fact (often forced by being a $p$-group), but it does not preclude non-trivial conjugation outside the centre. $D_4$ has $|Z(D_4)| = 2$ but is non-abelian.

## Where This Shows Up

*Classification of small groups.* The number of groups of each order up to $30$ is famously: $1, 1, 1, 2, 1, 2, 1, 5, 2, 2, 1, 5, 1, 2, 1, 14, 1, 5, 1, 5, 2, 2, 1, 15, 2, 2, 5, 4, 1, 4$. These counts come almost entirely from Sylow analysis plus a finite combinatorial check on which semidirect products are isomorphic. Sylow is the engine that turns a vague "find all groups of order $n$" question into a finite tractable problem.

*Cryptanalysis of Diffie-Hellman.* Pohlig–Hellman reduces the discrete logarithm problem in a cyclic group of order $n = \prod p_i^{e_i}$ to discrete logarithms in the Sylow subgroups (orders $p_i^{e_i}$) via the Chinese remainder theorem. This is exactly why cryptographic protocols insist on $n$ having a large prime factor: otherwise Sylow decomposition cracks the problem in time proportional to the largest prime factor.

*Burnside's $p^a q^b$ theorem.* Every group of order $p^a q^b$ (two prime factors) is solvable. The proof, due to Burnside (1904), uses character theory but ultimately rests on Sylow-style induction. The result was the original motivation for representation theory of finite groups, which we will see in Part 10.

## What I Want You to Carry Forward

Four questions for Part 5, on rings and ideals:

1. *What is the analogue of "normal subgroup" for rings?* It will be the *ideal*, which absorbs multiplication on both sides — exactly the condition that makes the quotient ring well-defined.
2. *What is the analogue of "Sylow theorem" for rings?* There is no clean analogue at the level of ideals, but for modules over a PID we will get a structure theorem (Part 9) that plays a similar role.
3. *Why does $\mathbb{Z}/p\mathbb{Z}$ have such different behaviour from $\mathbb{Z}/4\mathbb{Z}$ — one is a field, the other has zero divisors?* This is exactly the prime ideal versus general ideal distinction.
4. *Are there finite simple rings, the way there are finite simple groups?* Yes: matrix algebras over division rings, classified by Wedderburn.

If the Sylow computations above feel mechanical, you have the technique. The next time you see a group of order $\leq 100$, your first move should be: factor $|G|$, write down the Sylow constraints, see which Sylow subgroups are forced normal, and read off the structure. This is the bread and butter of finite group theory, and you have just earned the right to call it that.

---
