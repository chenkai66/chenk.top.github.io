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

Lagrange's theorem tells you the order of any subgroup must divide $|G|$. That is a necessary condition, and it is famously *not* sufficient — the alternating group $A_4$ has order $12$, but no subgroup of order $6$. So the moment you start asking "given $|G| = n$, what does $G$ actually look like?", Lagrange leaves you holding an empty bag.

The Sylow theorems are what go inside that bag. They say: for every maximal prime power $p^a$ dividing $|G|$, a subgroup of order $p^a$ exists, all such subgroups are conjugate, and their count $n_p$ is sharply constrained ($n_p \equiv 1 \pmod p$, $n_p \mid [G:P]$). Ludwig Sylow proved this in 1872, and 150 years later it is still the first thing you reach for when somebody hands you a finite group of unknown order and asks what it is.

The whole point of this article is to make those three statements feel inevitable rather than magical, and then to push them as far as they go: classifying small groups, ruling out simple groups at certain orders, and getting to the threshold where $A_5$ — order $60$ — finally cracks open the door to nonabelian simple groups.

I should be upfront about the dependency graph here. To make Sylow do real work you need three preliminaries glued in your head: the class equation, the orbit-stabilizer formula, and the basic theory of group actions. Sylow is essentially a clever choice of action ($G$ acting on a carefully chosen set of $p$-power-sized subsets, or on the set of its own Sylow subgroups by conjugation) followed by orbit-stabilizer, then a residue calculation modulo $p$. Once you internalize that pattern, the three theorems stop being three separate results and start looking like one technique applied three times.

---

## What Sylow Buys You That Lagrange Doesn't

For $n = 1$ there is one group. For $n = p$ prime there is one group, $\mathbb{Z}/p\mathbb{Z}$. For composite $n$ the count of isomorphism classes immediately gets weird: $2$ groups of order $4$, $5$ groups of order $8$, $14$ groups of order $16$, and a startling $267$ groups of order $64$ — and these are just the $2$-groups, where Sylow has nothing internal to say. The actual structure-theoretic problem of "list all groups of order $n$" gets unmanageable fast.

What we want from a structure theorem is three things:

1. **Existence** — given a number that "should" be the order of a subgroup, prove a subgroup of that size exists.
2. **Uniqueness up to conjugacy** — when several such subgroups exist, show they are essentially the same.
3. **Counting** — restrict the number of such subgroups tightly enough to derive contradictions or force normality.

Lagrange handles none of these. Cauchy's theorem (every prime $p \mid |G|$ produces an element of order $p$) does (1) but only for $p^1$. Sylow does all three, for the maximal prime power $p^a$ where $p^a \| |G|$ (the notation means $p^a \mid |G|$ but $p^{a+1} \nmid |G|$). That single jump from $p$ to $p^a$ is most of the difference between "I know a few elements of $G$" and "I can write down all groups of order $|G|$."

---

## p-Groups: The Atoms

A finite **$p$-group** is a group of order $p^k$. Equivalently (Cauchy), every element has order a power of $p$. These are the atoms; the whole Sylow program is a strategy for cutting an arbitrary finite group along its $p$-group atoms and reading off the structure of the pieces.

The single most useful fact about a finite $p$-group $G$ is that $Z(G)$ is nontrivial. Here is why, in one paragraph. Apply the class equation:

$$|G| \;=\; |Z(G)| \;+\; \sum_i [G : C_G(g_i)],$$

where the sum runs over conjugacy classes of size $> 1$, with representatives $g_i$. Each index $[G : C_G(g_i)]$ divides $|G| = p^k$ and is bigger than $1$, so each is divisible by $p$. The left side is divisible by $p$. Therefore $|Z(G)|$ is divisible by $p$, hence at least $p$. End of argument.

That one observation immediately implies a chain of structural facts I will use repeatedly:

- Every group of order $p^2$ is abelian. (If $|Z(G)| = p$, then $G/Z(G)$ has order $p$, hence is cyclic, hence $G$ is abelian — contradiction. So $Z(G) = G$.)
- Every $p$-group has a normal subgroup of every order $p^j$ for $0 \le j \le k$.
- Every nontrivial $p$-group has a chain of normal subgroups $\{e\} = N_0 \triangleleft N_1 \triangleleft \cdots \triangleleft N_k = G$ with $|N_j| = p^j$.

The last point is what people mean when they say $p$-groups are "as solvable as possible."

![Nested chain of subgroups in a p-group](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_1_p_group_chain.png)

**Why this matters.** The chain says you can build any $p$-group by repeatedly extending by $\mathbb{Z}/p\mathbb{Z}$. If you accept that, then knowing the abelian $p$-groups (which the structure theorem hands you) plus the $H^2$ extension data tells you everything. This is also the reason Sylow theory is so effective on $|G| = p^a m$ with $\gcd(p, m) = 1$: it reduces the local prime-$p$ structure to a problem we already control.

---

## Cauchy's Theorem via McKay's Action

Before Sylow, there is Cauchy: if $p$ divides $|G|$, then $G$ has an element of order $p$. The cleanest proof is McKay's, and it is worth seeing because the same action-counting trick reappears throughout Sylow.

Define

$$S \;=\; \{(g_1, g_2, \ldots, g_p) \in G^p : g_1 g_2 \cdots g_p = e\}.$$

The first $p-1$ entries are free, the last is forced: $g_p = (g_1 \cdots g_{p-1})^{-1}$. So $|S| = |G|^{p-1}$.

Let $\mathbb{Z}/p\mathbb{Z}$ act on $S$ by cyclic shifts: $(g_1, \ldots, g_p) \mapsto (g_2, \ldots, g_p, g_1)$. This is well-defined because $g_2 \cdots g_p g_1 = g_1^{-1}(g_1 \cdots g_p)g_1 = e$. Every orbit has size $1$ or $p$ (a cyclic group of prime order has no other orbit sizes). The fixed points are exactly the constant tuples $(g, g, \ldots, g)$ with $g^p = e$.

By orbit counting, $|S| = (\text{fixed points}) + p \cdot (\text{orbits of size } p)$. The left side $|G|^{p-1}$ is divisible by $p$. So the number of fixed points is divisible by $p$. The identity tuple is one fixed point, so there must be at least $p$ of them, meaning at least one $g \neq e$ with $g^p = e$. That $g$ has order $p$.

The proof is shorter than the proof of Lagrange and considerably more useful, which is the kind of thing I find genuinely satisfying.

---

## The Three Sylow Theorems, Stated Once

Fix a finite group $G$, a prime $p$, and write $|G| = p^a m$ with $\gcd(p, m) = 1$. A **Sylow $p$-subgroup** is a subgroup of order $p^a$ — i.e., the largest possible $p$-group inside $G$.

**Sylow I (Existence).** A Sylow $p$-subgroup of $G$ exists.

**Sylow II (Conjugacy).** Any two Sylow $p$-subgroups are conjugate. More generally, every $p$-subgroup of $G$ is contained in some Sylow $p$-subgroup.

**Sylow III (Counting).** Let $n_p$ be the number of Sylow $p$-subgroups. Then

$$n_p \equiv 1 \pmod{p} \qquad \text{and} \qquad n_p \mid m.$$

The combination is what makes Sylow III useful: $n_p$ has to land in the (usually short) list of divisors of $m$ that are $\equiv 1 \pmod p$, and that list is often $\{1\}$ — forcing the Sylow $p$-subgroup to be unique and therefore normal.

I will not redo all three proofs from scratch; the first uses an action of $G$ on $p$-element subsets and shows a fixed orbit gives a Sylow, the second comes from acting one Sylow on the cosets of another, and the third is essentially the orbit-stabilizer formula together with a class equation argument inside the action of a Sylow $P$ on the set of Sylow subgroups.

A quick sketch of Sylow I, since it is the existence theorem and skipping it feels rude. Write $|G| = p^a m$. Consider the set $\Omega$ of all $p^a$-element subsets of $G$. The size is $\binom{p^a m}{p^a}$, and a careful look (Kummer's theorem, or the lifting-the-exponent lemma) shows $p \nmid \binom{p^a m}{p^a}$. Now $G$ acts on $\Omega$ by left multiplication. By orbit-stabilizer, each orbit size divides $|G|$. If every orbit had size divisible by $p$, then $|\Omega|$ would be divisible by $p$ — but it isn't. So some orbit $\mathcal{O}$ has size coprime to $p$. The stabilizer $H$ of any element of $\mathcal{O}$ then has $|H| = |G|/|\mathcal{O}|$, which is divisible by $p^a$. On the other hand, if $X \in \mathcal{O}$ is one of the $p^a$-element subsets, then $H$ acts freely on $X$ by left multiplication (since $hx_1 = x_2$ in $X$ means $h$ takes one element of $X$ to another), so $|H|$ divides $|X| = p^a$. Combined, $|H| = p^a$. So $H$ is a Sylow $p$-subgroup. Done.

The proof is *almost* unfair. You declare a set, count it, observe the count is coprime to $p$, and a Sylow falls out by orbit-stabilizer. The cleverness is entirely in the choice of $\Omega$.

What I want to emphasize is the *use*, so I will reproduce the conjugacy proof in the form most students forget.

---

## Proof Sketch of Sylow II

Let $P$ be a Sylow $p$-subgroup, $Q$ any other $p$-subgroup. Act $Q$ on the coset space $G/P$ by left multiplication. Orbits have size dividing $|Q| = p^j$, so each orbit size is a power of $p$. The total $|G/P| = m$ is coprime to $p$, so at least one orbit has size $1$ — i.e., there is a coset $gP$ fixed by all of $Q$. That means $QgP = gP$, hence $g^{-1}Qg \subseteq P$, i.e., $Q$ is contained in the conjugate $gPg^{-1}$.

If $Q$ itself is a Sylow $p$-subgroup, then $|Q| = |P|$ forces $Q = gPg^{-1}$. Otherwise $Q \subsetneq gPg^{-1}$, which proves the second sentence of Sylow II: every $p$-subgroup is contained in a Sylow $p$-subgroup.

The argument is unreasonably short for what it delivers. The trick — acting on coset spaces by orbit size mod $p$ — is the same trick used six different ways in finite group theory, and it is the cleanest single instance of it.

![Constraints on the number of Sylow p-subgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_2_sylow_count.png)

---

## A First Worked Example: $|G| = 15$

The cleanest place to see Sylow do real work is $n = 15 = 3 \cdot 5$.

- $n_5$ divides $3$ and is $\equiv 1 \pmod 5$, so $n_5 \in \{1, 3, 6, 11, \ldots\} \cap \{1, 3\} = \{1\}$. The Sylow $5$-subgroup is unique and normal.
- $n_3$ divides $5$ and is $\equiv 1 \pmod 3$, so $n_3 \in \{1, 4, 7, \ldots\} \cap \{1, 5\} = \{1\}$. The Sylow $3$-subgroup is unique and normal.

So $G$ has a normal $\mathbb{Z}/5\mathbb{Z}$ and a normal $\mathbb{Z}/3\mathbb{Z}$, intersecting trivially, of coprime orders. By the recognition theorem for direct products, $G \cong \mathbb{Z}/3\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z} \cong \mathbb{Z}/15\mathbb{Z}$.

There is exactly one group of order $15$.

This is the kind of result that looks underwhelming until you realize you just classified an entire isomorphism class with two arithmetic checks. No element-pushing, no GAP, no enumeration — Sylow III did the work. To appreciate how much work that is, try to prove "every group of order $15$ is cyclic" without Sylow. You can do it — Cauchy gives you elements of orders $3$ and $5$, and you can argue about how they interact — but it takes most of a page, and the argument does not generalize to $|G| = 35$ or $|G| = 77$ without rewriting. Sylow III gives you the same conclusion in two lines for any of those orders.

![Classification of groups of order 15 via Sylow analysis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_5_classify_15.png)

---

## $S_4$ in Color: Where Are the Sylow Subgroups?

For a more textured example, take $G = S_4$, $|G| = 24 = 2^3 \cdot 3$.

**Sylow $2$-subgroups** have order $8$. By Sylow III, $n_2 \mid 3$ and $n_2 \equiv 1 \pmod 2$, so $n_2 \in \{1, 3\}$. It is not $1$ (no normal subgroup of order $8$ in $S_4$), so $n_2 = 3$. The three Sylow $2$-subgroups are isomorphic to $D_4$ (dihedral of order $8$); concretely, each is the symmetry group of the square that you get by labeling the four points so that opposite vertices match. For instance one of them is

$$P_1 = \langle (1234), (13) \rangle.$$

The other two are obtained by relabeling.

**Sylow $3$-subgroups** have order $3$. $n_3 \mid 8$ and $n_3 \equiv 1 \pmod 3$, so $n_3 \in \{1, 4\}$. There is no normal $3$-subgroup in $S_4$, so $n_3 = 4$. The four Sylow $3$-subgroups are the four cyclic groups $\langle (ijk) \rangle$ generated by $3$-cycles, one for each $3$-element subset of $\{1,2,3,4\}$. There are $\binom{4}{3} = 4$ such subsets, matching $n_3 = 4$.

This counts checks out beautifully: $4 \times (3-1) = 8$ elements of order $3$ in $S_4$, which agrees with the direct count of $3$-cycles ($8$ of them — two for each $3$-element subset). Sylow is consistent with the combinatorics.

![Sylow 2- and 3-subgroups of S_4 highlighted](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_3_s4_sylow.png)

---

## The Conjugation Action Is the Whole Engine

Sylow II says all Sylow $p$-subgroups are conjugate. Equivalently, $G$ acts transitively on the set $\mathrm{Syl}_p(G)$ by conjugation. This is not a side remark — it is the engine.

Let $P$ be any Sylow $p$-subgroup. Its stabilizer under the conjugation action is the **normalizer** $N_G(P) = \{g : gPg^{-1} = P\}$. By orbit-stabilizer,

$$n_p = |\mathrm{Syl}_p(G)| = [G : N_G(P)].$$

So Sylow III's "$n_p \mid m$" is just the statement that $[G:N_G(P)]$ divides $m$, i.e., $P \subseteq N_G(P)$ has $p$-power index in $N_G(P)$ which is then coprime to $m$ in the right way. All of this is bookkeeping inside one orbit-stabilizer calculation.

The "$n_p \equiv 1 \pmod p$" piece comes from then having $P$ act on $\mathrm{Syl}_p(G)$ (still by conjugation). Each orbit of $P$ has $p$-power size. The orbit $\{P\}$ has size $1$. Any other fixed point $Q$ would satisfy $P \subseteq N_G(Q)$, and then by the second part of Sylow II (applied inside $N_G(Q)$ where $Q$ is normal), $P = Q$. So $\{P\}$ is the unique orbit of size $1$, and all other orbits have size divisible by $p$. Therefore $|\mathrm{Syl}_p(G)| \equiv 1 \pmod p$.

This is one of those proofs where, once you see the action diagram, you cannot un-see it. The whole counting half of Sylow is a pair of orbit-stabilizer calculations.

![Conjugation action on the set of Sylow subgroups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_4_conjugacy_action.png)

---

## Normalizers and the Frattini Argument

The normalizer $N_G(P)$ keeps coming up, so it is worth knowing two facts.

**Fact 1.** $N_G(N_G(P)) = N_G(P)$. (The normalizer of the normalizer is the normalizer.) This is special to Sylow $p$-subgroups and is what people sometimes call the "self-normalizing" property — it means you cannot keep climbing.

**Fact 2 (Frattini argument).** If $N \triangleleft G$ and $P$ is a Sylow $p$-subgroup of $N$, then $G = N \cdot N_G(P)$.

Frattini is the workhorse for inductive arguments: if you can find a normal subgroup $N$ and analyze $N$'s Sylow subgroups, Frattini lets you extend the analysis to all of $G$. I will use it implicitly in the simple-group-of-order-60 argument below.

![Normalizer N_G(P) and Sylow count](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_6_normalizer.png)

---

## Classifying Groups of Small Order

Here is the standard recipe, applied to a few cases.

### Order 6

$|G| = 6 = 2 \cdot 3$. Sylow III: $n_3 \mid 2$ and $n_3 \equiv 1 \pmod 3$, so $n_3 = 1$. The Sylow $3$-subgroup is normal. $n_2 \in \{1, 3\}$.

- $n_2 = 1$: both Sylows normal, $G \cong \mathbb{Z}/2 \times \mathbb{Z}/3 \cong \mathbb{Z}/6$.
- $n_2 = 3$: the Sylow $2$-subgroups (each of order $2$) act on $P_3 \cong \mathbb{Z}/3$ by conjugation. The only nontrivial automorphism of $\mathbb{Z}/3$ is inversion. This gives $G \cong S_3$.

Two groups of order $6$.

### Order 8

This is a $p$-group; Sylow has nothing to say about internal structure (only one prime). Use $|Z(G)| > 1$ instead. Cases on $|Z(G)|$:

- $|Z(G)| = 8$: $G$ abelian, three options ($\mathbb{Z}/8$, $\mathbb{Z}/4 \times \mathbb{Z}/2$, $(\mathbb{Z}/2)^3$).
- $|Z(G)| = 4$: $G/Z(G)$ has order $2$, hence cyclic, hence $G$ abelian — contradiction.
- $|Z(G)| = 2$: $G/Z(G)$ has order $4$. Cyclic case forces abelian — contradiction. So $G/Z(G) \cong (\mathbb{Z}/2)^2$, giving the dihedral group $D_4$ and the quaternion group $Q_8$.

Five groups of order $8$.

### Order 12

$|G| = 12 = 2^2 \cdot 3$. Sylow III: $n_3 \in \{1, 4\}$, $n_2 \in \{1, 3\}$.

A counting trick handles the case split: if $n_3 = 4$, that is $4 \cdot 2 = 8$ elements of order $3$, leaving $12 - 8 = 4$ elements that must form the unique Sylow $2$-subgroup (so $n_2 = 1$). Conversely if $n_3 = 1$, the normal $P_3 \cong \mathbb{Z}/3$ leaves $P_2$ free to be either $\mathbb{Z}/4$ or $V_4 = (\mathbb{Z}/2)^2$, and $G$ is a semidirect product determined by a homomorphism $P_2 \to \mathrm{Aut}(\mathbb{Z}/3) \cong \mathbb{Z}/2$.

Working through the four (action, $P_2$-shape) combinations, we get exactly $\mathbb{Z}/{12}$, $\mathbb{Z}/6 \times \mathbb{Z}/2$, $D_6$, $\mathrm{Dic}_3$, and $A_4$. Five groups of order $12$.

The thing I want to call attention to is how the *element count* short-circuits the analysis. Once you know "$8$ elements of order $3$ implies the rest forms a normal Sylow $2$," you do not even need Sylow III's congruence — pigeonhole did it. This element-counting move is the second-most-useful tool after Sylow III itself.

---

## Why $A_5$ Is Simple, and No Group of Order 12, 24, 30, 36 Is

Sylow really earns its keep when we use it to prove non-existence and simplicity results.

**No simple group of order $12$.** Suppose $G$ simple with $|G| = 12$. Then $n_3 = 4$ (since $n_3 = 1$ would give a normal subgroup). Conjugation on the four Sylow $3$-subgroups gives $\varphi : G \to S_4$. The kernel is normal, so $\ker \varphi \in \{1, G\}$. If $\ker \varphi = G$ the action is trivial — but the Sylows are conjugate to each other, contradiction. So $\varphi$ is injective and embeds $G$ into $S_4$ as a subgroup of order $12$, which must be $A_4$. But $A_4$ has $V_4$ as a normal subgroup, contradicting simplicity.

The pattern — *use the conjugation action on Sylow subgroups to embed $G$ into a small symmetric group, then derive a contradiction from the structure of that symmetric group* — generalizes immediately. It is how you rule out simple groups at orders $24$, $36$, $48$, and so on.

**No simple group of order $30$.** $|G| = 30 = 2 \cdot 3 \cdot 5$. Sylow III: $n_5 \in \{1, 6\}$, $n_3 \in \{1, 10\}$. If $n_5 = 6$, that is $6 \cdot 4 = 24$ elements of order $5$. If also $n_3 = 10$, that is $10 \cdot 2 = 20$ elements of order $3$. But $24 + 20 = 44 > 30$, contradiction. So at most one of $n_5, n_3$ can be $> 1$ — meaning at least one of the Sylows is normal, so $G$ is not simple.

**$A_5$ is simple.** $|A_5| = 60 = 2^2 \cdot 3 \cdot 5$. Sylow III: $n_2 \in \{1, 3, 5, 15\}$, $n_3 \in \{1, 4, 10\}$, $n_5 \in \{1, 6\}$.

Direct calculation (counting $5$-cycles modulo conjugacy in $A_5$) shows $n_5 = 6$, $n_3 = 10$, $n_2 = 5$. So no Sylow is normal. To upgrade "no Sylow is normal" to "no normal subgroup of any order," we case-split on $|N|$ for a hypothetical proper normal $N \triangleleft A_5$:

$|N| \in \{2, 3, 4, 5, 6, 10, 12, 15, 20, 30\}$. For each one, Sylow + counting gives a contradiction. $|N| = 15$ is the cleanest: any group of order $15$ is cyclic, with characteristic Sylow $5$-subgroup, which would then be normal in $A_5$ — but $n_5(A_5) = 6$, contradiction. All other cases follow similar patterns.

So $A_5$ is simple. The smallest nonabelian simple group, the obstruction to solving the quintic by radicals, the irreducible building block at the bottom of the classification of finite simple groups — all delivered by Sylow + element counting.

![Sylow argument for simple group of order 60](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-sylow-theorems/aa_v2_04_7_simple_60.png)

---

## A Detour: Groups of Order 20 by Hand

Let me work an order-$20$ classification in painful detail, because it shows every Sylow technique in one place.

$|G| = 20 = 2^2 \cdot 5$. Sylow III: $n_5 \mid 4$ and $n_5 \equiv 1 \pmod 5$, so $n_5 \in \{1\} \cap \{1, 4\} = \{1\}$. The Sylow $5$-subgroup $P_5 \cong \mathbb{Z}/5$ is unique and normal. $n_2 \mid 5$ and $n_2 \equiv 1 \pmod 2$, so $n_2 \in \{1, 5\}$. The Sylow $2$-subgroup $P_2$ has order $4$, hence is either $\mathbb{Z}/4$ or $V_4$.

So $G = P_5 \rtimes P_2$ for some action $\varphi : P_2 \to \mathrm{Aut}(P_5) \cong (\mathbb{Z}/5)^\times \cong \mathbb{Z}/4$.

Case work on $P_2$ and $\varphi$:

- $P_2 = \mathbb{Z}/4$, $\varphi$ trivial: $G \cong \mathbb{Z}/5 \times \mathbb{Z}/4 \cong \mathbb{Z}/20$.
- $P_2 = \mathbb{Z}/4$, $\varphi$ injective onto $\mathrm{Aut}(\mathbb{Z}/5) \cong \mathbb{Z}/4$: a generator $b \in P_2$ acts on a generator $a \in P_5$ by $bab^{-1} = a^2$ (or $a^3$, equivalent). The result is $G = \langle a, b : a^5 = b^4 = e, bab^{-1} = a^2\rangle$, sometimes written $F_{20}$ or $\mathrm{Hol}(\mathbb{Z}/5)$ or $\mathrm{GA}_1(\mathbb{F}_5)$ — the affine group of the line over $\mathbb{F}_5$. It is the most non-abelian group of order $20$.
- $P_2 = \mathbb{Z}/4$, $\varphi$ has image $\mathbb{Z}/2$: $b$ acts on $a$ by $bab^{-1} = a^{-1}$, but $b^2$ acts trivially. This gives the dicyclic group $\mathrm{Dic}_5$ (also $Q_{20}$ in some sources).
- $P_2 = V_4$, $\varphi$ trivial: $G \cong \mathbb{Z}/5 \times V_4 \cong \mathbb{Z}/{10} \times \mathbb{Z}/2$.
- $P_2 = V_4$, $\varphi$ has image $\mathbb{Z}/2$: $G \cong D_{10}$, the dihedral group of order $20$.

Five groups of order $20$:

$$\mathbb{Z}/20, \quad \mathbb{Z}/{10} \times \mathbb{Z}/2, \quad D_{10}, \quad \mathrm{Dic}_5, \quad F_{20}.$$

The exercise illustrates two things. First, once Sylow forces a normal subgroup ($P_5$ here), the rest of the work is *not* Sylow — it is semidirect-product theory, classifying homomorphisms $P_2 \to \mathrm{Aut}(P_5)$ up to the appropriate equivalence. Second, Sylow gives you the skeleton, but to fill in the muscles you need supplementary facts about automorphism groups of small abelian groups: $\mathrm{Aut}(\mathbb{Z}/p) \cong \mathbb{Z}/(p-1)$, $\mathrm{Aut}((\mathbb{Z}/p)^2) \cong \mathrm{GL}_2(\mathbb{F}_p)$, and so on. These are facts you accumulate over time.

---

## Why $n_p \equiv 1 \pmod p$ Has Real Teeth

The congruence is what most students underuse. Let me walk through three cases where it is the *only* thing that closes the argument.

**Order $33 = 3 \cdot 11$.** $n_{11} \mid 3$ and $n_{11} \equiv 1 \pmod{11}$. Divisors of $3$: $\{1, 3\}$. Numbers $\equiv 1 \pmod{11}$: $\{1, 12, 23, \ldots\}$. Intersection: $\{1\}$. So $n_{11} = 1$. Similarly $n_3 \mid 11$ with $n_3 \equiv 1 \pmod 3$, intersection of $\{1, 11\}$ and $\{1, 4, 7, 10, 13, \ldots\}$ — wait, $11 \not\equiv 1 \pmod 3$ (since $11 = 9 + 2$). So $n_3 = 1$. Both Sylows normal, $G \cong \mathbb{Z}/{33}$.

**Order $35 = 5 \cdot 7$.** $n_7 \mid 5$, $n_7 \equiv 1 \pmod 7$: only $n_7 = 1$. $n_5 \mid 7$, $n_5 \equiv 1 \pmod 5$: only $n_5 = 1$. Unique group, $\mathbb{Z}/{35}$.

**Order $77 = 7 \cdot 11$.** $n_{11} \mid 7$, $n_{11} \equiv 1 \pmod{11}$: $n_{11} = 1$. $n_7 \mid 11$, $n_7 \equiv 1 \pmod 7$: $\{1, 11\} \cap \{1, 8, 15, \ldots\} = \{1\}$. Both Sylows normal, $G \cong \mathbb{Z}/{77}$.

The general fact behind these examples: if $|G| = pq$ with primes $p < q$ and $p \nmid q - 1$, then $G \cong \mathbb{Z}/{pq}$. The constraint "$p \nmid q-1$" is exactly what kills the nonabelian semidirect product. (When $p \mid q - 1$, you get a second group: e.g., $|G| = 6 = 2 \cdot 3$ has $S_3$ as well, because $2 \mid 3-1$.)

This is a strong structural statement and it costs you two arithmetic checks. I keep bringing this up because the gap between "Sylow tells me $n_p$ is constrained" and "Sylow classifies all groups of this order" is often a single line of mod-$p$ arithmetic.

---

## Order 60: $A_5$ Is Simple, In Detail

Let me redo the simplicity of $A_5$ carefully, because it is the cornerstone.

$|A_5| = 60 = 2^2 \cdot 3 \cdot 5$. The conjugacy classes of $A_5$ are $\{e\}$ (size $1$), the $20$ three-cycles, the $15$ products of two disjoint transpositions, and *two* classes of $5$-cycles of size $12$ each — the $5$-cycles split into two $A_5$-conjugacy classes because the $S_5$-conjugator that would unite them is odd. Total: $1 + 20 + 15 + 12 + 12 = 60$. Good.

Sylow III: $n_5 \in \{1, 6\}$, $n_3 \in \{1, 4, 10\}$, $n_2 \in \{1, 3, 5, 15\}$.

A Sylow $5$-subgroup is $\langle (12345) \rangle$ — cyclic of order $5$. There are $24$ five-cycles total, four per Sylow $5$-subgroup, so $n_5 = 24/4 = 6$. A Sylow $3$-subgroup is $\langle (123) \rangle$. There are $20$ three-cycles, two per subgroup, so $n_3 = 20/2 = 10$. A Sylow $2$-subgroup is $V_4 = \{e, (12)(34), (13)(24), (14)(23)\}$ — actually one such subgroup, plus its conjugates. There are $15$ double transpositions plus identity, partitioned into $5$ Sylow $2$-subgroups of size $4$ each (the identity is shared), so $n_2 = 5$.

None is $1$, so no Sylow subgroup is normal in $A_5$.

Now suppose for contradiction $N$ is a proper nontrivial normal subgroup. By Lagrange, $|N| \mid 60$, so $|N| \in \{2, 3, 4, 5, 6, 10, 12, 15, 20, 30\}$.

- $|N| \in \{2, 5, 10, 30\}$: any subgroup of order $5$ has the form of a Sylow $5$-subgroup. If $|N| = 5$, then $N$ is *the* unique normal Sylow $5$, contradicting $n_5 = 6$. If $|N| \in \{10, 30\}$, then $N$ contains a Sylow $5$-subgroup; since $N$ is normal and Sylow $5$-subgroups are conjugate, $N$ contains *all* Sylow $5$-subgroups, hence $\geq 24$ elements of order $5$ — manageable but the case $|N| = 30$ specifically would force $N$ to contain all $24$ five-cycles plus the identity ($25$ elements), needing $5$ more, which can only be double transpositions. Then $N$ is a normal subgroup of index $2$; but $A_5$ would have a homomorphism to $\mathbb{Z}/2$ with kernel $N$, and $A_5$ is generated by $3$-cycles which all map to $0$, contradiction. The case $|N| = 10$ similarly fails. $|N| = 2$ would mean $N = \{e, x\}$ with $x$ of order $2$, but conjugates of $x$ are also in $N$ (normal), and $x$ has $14$ other conjugates ($15$-class of double transpositions), too many to fit.
- $|N| \in \{3, 6, 12\}$: by similar Sylow-and-conjugacy-class arguments. $|N| = 3$ contradicts $n_3 = 10$. $|N| = 6$ would have a unique Sylow $3$, hence normal in $N$ hence characteristic-in-$N$ hence normal in $A_5$, contradicting $n_3 \neq 1$. $|N| = 12$: by the order-$12$ analysis, $N \cong A_4$ or $D_6$ or $\mathrm{Dic}_3$, all of which have characteristic subgroups that would lift to give us a contradiction with the Sylow numbers.
- $|N| = 4$: $N$ is a Sylow $2$-subgroup, contradicting $n_2 = 5$.
- $|N| = 15$: any group of order $15$ is cyclic with a characteristic Sylow $5$, normal in $A_5$, contradicting $n_5 = 6$.
- $|N| = 20$: index $3$, so $A_5$ acts on cosets $A_5/N$ giving a map $A_5 \to S_3$. The kernel is normal of order $\geq 60/6 = 10$. So either the map is trivial (impossible, action is transitive) or the kernel has order $10$ — back to a previous case.

Every case fails. Therefore $A_5$ has no proper nontrivial normal subgroup. $A_5$ is simple.

This proof is long but every line is just Sylow plus a counting argument or a case split. There is no clever trick beyond patience. The reason $A_5$ is "the" first nonabelian simple group is that $|A_5| = 60$ is the smallest order where Sylow stops forcing normality on at least one Sylow subgroup *and* the residual element-counting and embedding-into-$S_n$ arguments fail to find a normal subgroup elsewhere.

---

## Why None of This Survives in the Infinite Setting

Sylow is a *finite* theorem. The proofs use orbit-stabilizer, divisibility of indices, and the class equation — all of which require finiteness in essential ways. In an infinite group, you can have:

- A torsion-free group (no elements of finite order), so no $p$-subgroups of any kind exist for any $p$. Example: $(\mathbb{Q}, +)$, $(\mathbb{R}, +)$, $\mathrm{GL}_n(\mathbb{Q})$.
- A group where "Sylow $p$-subgroups" can be defined as maximal $p$-subgroups, but there are infinitely many, none conjugate to each other, and the counting congruences have no meaning.
- A group with no maximal $p$-subgroup at all — in $(\mathbb{Q}/\mathbb{Z}, +)$ the $p$-primary component is $\mathbb{Z}[1/p]/\mathbb{Z}$, which is its own (unique) "Sylow," but it is infinite, so the existence theorem says nothing nontrivial.

There are partial replacements: Hall subgroups in solvable groups, profinite Sylow theory, $p$-adic local-global principles. But the punchline is that Sylow's three statements are tightly bundled to finite group theory, and that bundling is not an accident — it is the source of the power.

---

## What I Take Away

The Sylow theorems are not a single result; they are a *toolkit*, and the toolkit has three pieces that you reach for in different proportions depending on the problem. Existence (Sylow I) lets you assert "such a subgroup is there." Conjugacy (Sylow II) collapses many subgroups into one orbit. Counting (Sylow III) gives the arithmetic constraint that closes the proof.

Before I summarize the recipe, one more piece of intuition worth carrying. Sylow theory is the first place where the *interplay* between several primes inside one group becomes a structural tool rather than a coincidence. If $|G| = p^a q^b$, you study the Sylow $p$- and Sylow $q$-subgroups separately, but the *interaction* between them — captured by the conjugation action of one on the other, or by which is normal — controls the entire group up to extension data. This is why orders like $pq$, $p^2 q$, $p q r$ are tractable: small numbers of primes mean small numbers of degrees of freedom in the interaction. Order $60 = 2^2 \cdot 3 \cdot 5$ is the smallest "three primes" case where every Sylow can fail to be normal *and* the interactions can avoid forcing a normal subgroup elsewhere — which is exactly why $A_5$ is the first nonabelian simple group, and exactly why "find more nonabelian simple groups" took the next ninety years.

The pattern of use is almost always the same:

1. Factor $|G| = \prod p_i^{a_i}$.
2. Compute the allowed values of $n_{p_i}$ for each prime.
3. If any $n_{p_i} = 1$ is forced, you have a normal subgroup. Use it.
4. If no $n_{p_i} = 1$ is forced, count elements: a Sylow $p$-subgroup of order $p^a$ contributes $p^a - 1$ elements not in any other Sylow $p$-subgroup. Sum and look for $> |G|$.
5. If counting still does not finish the job, embed $G$ into a symmetric group via the conjugation action on $\mathrm{Syl}_p(G)$ and use facts about $S_{n_p}$.

Five steps, applied with discipline, classify groups up to order $60$ or so by hand and rule out simple groups at most small composite orders. Beyond that the techniques generalize but the bookkeeping gets heavier — and eventually you hand off to the structure theorems for solvable groups and the classification of finite simple groups, both of which still rest on Sylow as a foundational ingredient.

A small comment on what *not* to do. Sylow III says $n_p \mid m$ where $m = [G:P]$ is coprime to $p$. It does *not* say $n_p \mid |G|$ in any useful way (of course it does, but the bound from "$n_p \mid m$" is much sharper). Beginners sometimes apply the weaker bound, get more cases, and either give up or grind through extra casework. Always use $m$, the part of $|G|$ coprime to $p$. Similarly the congruence is mod $p$, not mod $p^a$ — the bound is on the count of Sylow subgroups, not on the size of any individual one.

The next article moves from internal subgroup structure to the relationship between groups: **rings**, where two operations interact, and the ideals/quotients machinery that grows out of paying attention to that interaction.

One last thing worth saying. Sylow theory is the first example you encounter where local-global thinking pays off in finite group theory. "Local" here means $p$-local: focus on one prime at a time. "Global" means the whole group $G$. The Sylow theorems say the global structure is constrained — sometimes determined — by the local prime-by-prime data: which Sylow $p$-subgroups exist, how they sit inside their normalizers, how many copies there are. This local-global pattern reappears everywhere in algebra: Hasse principles in number theory, sheaf cohomology in algebraic geometry, $p$-adic analysis. The Sylow theorems are the entry point, and once you have the habit of thinking "fix a prime, then think globally," a lot of subsequent algebra becomes more natural to navigate.

---

*This is Part 4 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 3 — Quotient Groups and Homomorphisms](/en/abstract-algebra/03-quotient-groups-and-homomorphisms/)*

*Next: [Part 5 — Rings and Ideals](/en/abstract-algebra/05-rings-and-ideals/)*
