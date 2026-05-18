---
title: "Abstract Algebra (1): Groups — Your First Encounter with Algebraic Structure"
date: 2021-09-01 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 12
translationKey: "abstract-algebra-1"
description: "From integers to symmetries, we build the formal definition of a group, prove Lagrange's theorem, and compute our first subgroup lattice."
---

## Why Algebraic Structure Matters

![Dihedral group D4: all 8 symmetries of a square](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/aa01_dihedral_d4.png)

Before any definitions, here is the picture I want you to keep in mind. A group is a set in which you can combine any two elements to get a third, undo any element you have produced, and rearrange parentheses without consequence. That is the entire idea, dressed up. The rest of this article is a slow unpacking of that one sentence.

Most of undergraduate mathematics concerns itself with specific objects: the real numbers, continuous functions on $[0,1]$, the vector space $\mathbb{R}^n$. At some point a pattern emerges. The integers under addition and the nonzero rationals under multiplication share a structural resemblance that has nothing to do with the nature of their elements. Both carry a binary operation that is associative, possesses an identity, and admits inverses. Abstract algebra is the study of this structural resemblance, stripped of all incidental detail.

The concept of a *group* is the simplest and most pervasive algebraic structure. It appears in every branch of mathematics: in number theory (the multiplicative group of units modulo $n$), in geometry (the isometry group of a figure), in topology (the fundamental group of a space), in physics (Lie groups governing particle symmetries), and in combinatorics (permutation groups acting on finite sets). I have taught this section more times than I'd like to admit, and the students who eventually become comfortable algebraists are the ones who, on first reading, immediately try to verify the axioms on three or four examples of their own.

Historically, the notion crystallized in the early nineteenth century through the work of Galois, who used permutation groups to prove that the general quintic equation has no solution by radicals. Abel had arrived at essentially the same conclusion independently. Cayley later gave the abstract definition we use today. The path from solving polynomial equations to the four axioms of a group is one of the great compressions in mathematical thought.

In this article we define groups, build a library of examples, introduce the notion of a subgroup, and prove Lagrange's theorem --- the first genuinely nontrivial result in the subject. Every claim will be either proved or demonstrated by explicit computation. Worked-out numerical examples appear in roughly every section; if you skip them on first reading you will miss most of the value of an article like this.

![Cayley table of Z/4Z showing the four group axioms in action](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_1_cayley_z4.png)

## The Formal Definition: Four Axioms

Mental picture: a group is a set with a "combine" button. Push the button on any two elements and you get back a third, always inside the set. Some element does nothing. Every element has an undo.

![The four group axioms: closure, associativity, identity, inverse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/01_group_axioms.png)

**Definition.** A *group* is a set $G$ together with a binary operation $\cdot : G \times G \to G$ satisfying:

1. **Closure.** For all $a, b \in G$, the element $a \cdot b$ belongs to $G$.
2. **Associativity.** For all $a, b, c \in G$, $(a \cdot b) \cdot c = a \cdot (b \cdot c)$.
3. **Identity.** There exists an element $e \in G$ such that $e \cdot a = a \cdot e = a$ for all $a \in G$.
4. **Inverses.** For every $a \in G$, there exists $a^{-1} \in G$ such that $a \cdot a^{-1} = a^{-1} \cdot a = e$.

When the operation is commutative ($a \cdot b = b \cdot a$ for all $a, b$), we call $G$ an *abelian* group (after Abel). We often write the operation multiplicatively ($ab$ instead of $a \cdot b$) or additively ($a + b$, with identity $0$ and inverse $-a$) depending on context.

**Why this matters.** Stripping these four axioms out of any system that satisfies them is the move that lets one theorem cover thousands of cases. Lagrange's theorem, which we will prove below, is true for the integers mod 12, for symmetries of a hexagon, and for permutations of a deck of cards, all from a single proof.

**Uniqueness of identity and inverses.** Suppose $e$ and $e'$ are both identities. Then $e = e \cdot e' = e'$, where the first equality uses the fact that $e'$ is an identity and the second uses the fact that $e$ is. Similarly, if $b$ and $c$ are both inverses of $a$, then $b = b \cdot e = b \cdot (a \cdot c) = (b \cdot a) \cdot c = e \cdot c = c$. These two-line proofs are worth internalizing; they are the prototype for many arguments in algebra.

**Example 1: $(\mathbb{Z}, +)$.** The integers under addition form an abelian group. The identity is $0$, and the inverse of $n$ is $-n$. Closure and associativity are inherited from the arithmetic of the integers.

**Example 2: $(\mathbb{Q}^*, \times)$.** The nonzero rationals under multiplication form an abelian group. The identity is $1$, and the inverse of $q$ is $1/q$. We must exclude $0$ because it has no multiplicative inverse.

**Example 3: The trivial group.** The set $\{e\}$ with the operation $e \cdot e = e$ is a group. It is the unique group (up to isomorphism) with one element.

**Numerical example: verifying the axioms for $\mathbb{Z}/4\mathbb{Z}$.** Take the four elements $\{0, 1, 2, 3\}$ under addition mod 4. Closure: $2 + 3 = 5 \equiv 1 \pmod 4$, still in the set. Identity: $0$. Inverses: $-1 \equiv 3$, so the inverse of $1$ is $3$; the inverse of $2$ is $2$ itself, since $2 + 2 = 4 \equiv 0$. Associativity is inherited from $\mathbb{Z}$. All four axioms check out, on four lines of arithmetic.

**Non-example: $(\mathbb{Z}, \times)$.** The integers under multiplication fail to form a group because most elements lack inverses. The integer $2$ has no multiplicative inverse in $\mathbb{Z}$.

**Non-example: $(\mathbb{N}, +)$.** The natural numbers $\{0, 1, 2, \ldots\}$ under addition have an identity ($0$) but no inverses: there is no natural number $n$ with $1 + n = 0$. This is a *monoid*, not a group.

**Cancellation laws.** In any group, if $ab = ac$ then $b = c$ (left cancellation), and if $ba = ca$ then $b = c$ (right cancellation). Proof of left cancellation: multiply both sides on the left by $a^{-1}$ to obtain $a^{-1}(ab) = a^{-1}(ac)$, hence $(a^{-1}a)b = (a^{-1}a)c$ by associativity, hence $eb = ec$, hence $b = c$. Right cancellation is analogous.

A direct consequence worth noting: in the Cayley table of a finite group, every row and every column is a permutation of the group elements. If a row repeated an element, it would violate left cancellation; if a column did, it would violate right cancellation. This is one of the easiest ways to spot a non-group: write down the proposed table and check that every row is a permutation. Many "groups" invented by undergraduates fail this check on the second row.

**Notation.** For $a \in G$ and $n \in \mathbb{Z}$, define $a^n$ inductively: $a^0 = e$, $a^n = a \cdot a^{n-1}$ for $n > 0$, and $a^n = (a^{-1})^{-n}$ for $n < 0$. One verifies that $a^{m+n} = a^m a^n$ and $(a^m)^n = a^{mn}$ for all $m, n \in \mathbb{Z}$.

![Decision flow for verifying the four group axioms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_7_axioms.png)

## Cyclic Groups and the Integers Mod $n$

Mental picture: a cyclic group is a hand on a clock that ticks forward by one position each time you press the button. Some clocks have finitely many positions, some go on forever. That is essentially all there is to the structure.

![Cyclic groups: Z/nZ as clock arithmetic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/01_cyclic_groups.png)

**Definition.** A group $G$ is *cyclic* if there exists $g \in G$ such that every element of $G$ is a power of $g$: $G = \{g^n : n \in \mathbb{Z}\}$. We write $G = \langle g \rangle$ and call $g$ a *generator*.

The integers $(\mathbb{Z}, +)$ are cyclic, generated by $1$ (or by $-1$). This is the infinite cyclic group.

**The integers modulo $n$.** Fix a positive integer $n$. Define $\mathbb{Z}/n\mathbb{Z} = \{0, 1, 2, \ldots, n-1\}$ with the operation of addition modulo $n$. This is a cyclic group of order $n$ generated by $\bar{1}$.

Let us verify the axioms for $\mathbb{Z}/6\mathbb{Z}$. The elements are $\{0,1,2,3,4,5\}$. Addition modulo $6$: $3 + 5 = 2$ (since $8 \equiv 2 \pmod{6}$), $4 + 4 = 2$, etc. The identity is $0$. The inverse of $1$ is $5$, of $2$ is $4$, and of $3$ is $3$ (since $3 + 3 = 6 \equiv 0$). Associativity is inherited from $\mathbb{Z}$.

![The cyclic group Z/6Z visualized as rotations of a hexagon](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_5_cyclic_z6.png)

**Worked Example: Generators of $\mathbb{Z}/12\mathbb{Z}$.** The element $\bar{k}$ generates $\mathbb{Z}/12\mathbb{Z}$ if and only if $\gcd(k, 12) = 1$. The integers coprime to $12$ in $\{0, \ldots, 11\}$ are $1, 5, 7, 11$. So $\mathbb{Z}/12\mathbb{Z}$ has exactly four generators. Verify for $k = 5$: the successive multiples of $5$ modulo $12$ are $5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 0$, hitting all twelve elements. Try $k = 4$ for contrast: $4, 8, 0, 4, 8, 0, \ldots$ cycles after three steps because $\gcd(4, 12) = 4$.

**Theorem.** Every subgroup of a cyclic group is cyclic. Moreover, if $G = \langle g \rangle$ has order $n$, then for each divisor $d$ of $n$, there is exactly one subgroup of order $d$, namely $\langle g^{n/d} \rangle$.

*Proof sketch.* Let $H \leq G = \langle g \rangle$. If $H = \{e\}$, it is cyclic. Otherwise, let $m$ be the smallest positive integer with $g^m \in H$. We claim $H = \langle g^m \rangle$. Take any $g^k \in H$. Write $k = qm + r$ with $0 \leq r < m$. Then $g^r = g^k (g^m)^{-q} \in H$, and minimality of $m$ forces $r = 0$. Hence $g^k = (g^m)^q \in \langle g^m \rangle$. When $|G| = n$, the condition $\langle g^m \rangle \leq G$ forces $m \mid n$, and $|\langle g^m \rangle| = n/m$. Setting $d = n/m$ gives the result. $\square$

**Why this matters.** Cyclic groups are the building blocks of every finite abelian group, by the structure theorem we will see in a later article. So understanding their subgroup structure is the bottom rung of the ladder for all abelian group theory. It is also the bridge to elementary number theory: subgroups of $\mathbb{Z}$ are exactly the sets $n\mathbb{Z}$, which is the same data as the divisibility lattice on positive integers.

**Infinite versus finite cyclic groups.** The infinite cyclic group $\mathbb{Z}$ has exactly one subgroup for each nonnegative integer $n$: namely $n\mathbb{Z}$. The subgroup $n\mathbb{Z}$ has index $n$ in $\mathbb{Z}$ (for $n \geq 1$), and the quotient $\mathbb{Z}/n\mathbb{Z}$ is the cyclic group of order $n$. Every cyclic group is therefore either isomorphic to $\mathbb{Z}$ or to $\mathbb{Z}/n\mathbb{Z}$ for some $n \geq 1$.

A natural question: given two cyclic groups $\mathbb{Z}/m\mathbb{Z}$ and $\mathbb{Z}/n\mathbb{Z}$, when is their direct product $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$ again cyclic? The answer: if and only if $\gcd(m, n) = 1$. When this holds, $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}/mn\mathbb{Z}$ (the Chinese Remainder Theorem). Concrete check: $\mathbb{Z}/3\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z}$ has order $15$, and the element $(1, 1)$ has additive order $\text{lcm}(3,5) = 15$, so it generates the whole group. Contrast: in $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, every element has order at most $2$, so no element generates all four positions.

**The group of units $(\mathbb{Z}/n\mathbb{Z})^*$.** Define $(\mathbb{Z}/n\mathbb{Z})^* = \{k \in \mathbb{Z}/n\mathbb{Z} : \gcd(k, n) = 1\}$ with multiplication modulo $n$. This is an abelian group of order $\varphi(n)$, where $\varphi$ is Euler's totient function. For example, $(\mathbb{Z}/8\mathbb{Z})^* = \{1, 3, 5, 7\}$ under multiplication mod $8$:

| $\times$ | 1 | 3 | 5 | 7 |
|-----------|---|---|---|---|
| **1** | 1 | 3 | 5 | 7 |
| **3** | 3 | 1 | 7 | 5 |
| **5** | 5 | 7 | 1 | 3 |
| **7** | 7 | 5 | 3 | 1 |

Every element squares to $1$, so this group is isomorphic to $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ (the Klein four-group), not to $\mathbb{Z}/4\mathbb{Z}$. Notice how the diagonal of the table is all ones; that is the structural fingerprint of the Klein four-group.

**A small enumeration: $(\mathbb{Z}/n\mathbb{Z})^*$ for $n \le 10$.** Compute the order $\varphi(n)$ for each $n$ and decide whether the group is cyclic:

- $n = 2$: $\{1\}$, order $1$, trivial.
- $n = 3$: $\{1, 2\}$, order $2$, cyclic generator $2$.
- $n = 4$: $\{1, 3\}$, order $2$, cyclic generator $3$.
- $n = 5$: $\{1,2,3,4\}$, order $4$, cyclic with generator $2$ (since $2^1 = 2, 2^2 = 4, 2^3 = 3, 2^4 = 1$).
- $n = 7$: $\{1,2,3,4,5,6\}$, order $6$, cyclic with generator $3$.
- $n = 8$: order $4$, but as we saw above it is the Klein four-group, not cyclic.
- $n = 9$: order $6$, cyclic with generator $2$ (powers $2, 4, 8, 7, 5, 1$).

A theorem of Gauss says $(\mathbb{Z}/n\mathbb{Z})^*$ is cyclic if and only if $n \in \{1, 2, 4, p^k, 2p^k\}$ for an odd prime $p$. Note $n = 8 = 2^3$ is the smallest non-cyclic case. This gives concrete texture to the otherwise abstract structure of unit groups.

## Symmetry Groups: Dihedral and Symmetric Groups

Mental picture: take a physical object, write down everything you can do to it that leaves it looking the same, and compose those operations like rigid motions in your hand. That set of operations is a group.

![Symmetry group of a square: rotations and reflections](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/01_symmetry_group.png)

**The symmetric group $S_n$.** Let $X = \{1, 2, \ldots, n\}$. The set of all bijections $\sigma : X \to X$ forms a group under composition, called the *symmetric group* $S_n$. Its order is $n!$.

We write permutations in cycle notation. The permutation $\sigma \in S_4$ defined by $\sigma(1) = 2, \sigma(2) = 4, \sigma(3) = 3, \sigma(4) = 1$ is written $(1\ 2\ 4)$, a $3$-cycle fixing $3$. Every permutation decomposes uniquely (up to ordering) into disjoint cycles, and disjoint cycles commute.

**Worked Example: Computing in $S_4$.** Let $\sigma = (1\ 2\ 3)$ and $\tau = (1\ 3)(2\ 4)$. Compute $\sigma \tau$ (first apply $\tau$, then $\sigma$):

$$\sigma\tau(1) = \sigma(\tau(1)) = \sigma(3) = 1$$
$$\sigma\tau(2) = \sigma(\tau(2)) = \sigma(4) = 4$$
$$\sigma\tau(3) = \sigma(\tau(3)) = \sigma(1) = 2$$
$$\sigma\tau(4) = \sigma(\tau(4)) = \sigma(2) = 3$$

So $\sigma\tau = (2\ 4\ 3)$. Now compute $\tau\sigma$:

$$\tau\sigma(1) = \tau(2) = 4, \quad \tau\sigma(2) = \tau(3) = 1, \quad \tau\sigma(3) = \tau(1) = 3, \quad \tau\sigma(4) = \tau(4) = 2$$

So $\tau\sigma = (1\ 4\ 2)$. Since $\sigma\tau \neq \tau\sigma$, the group $S_4$ is non-abelian. In fact, $S_n$ is non-abelian for all $n \geq 3$.

**The sign of a permutation.** Every permutation can be written as a product of transpositions (2-cycles). The parity of the number of transpositions is an invariant: a permutation is *even* if it can be written as a product of an even number of transpositions, and *odd* otherwise. The map $\text{sgn} : S_n \to \{+1, -1\}$ defined by $\text{sgn}(\sigma) = (-1)^k$ where $\sigma$ is a product of $k$ transpositions is a well-defined group homomorphism. Its kernel is the *alternating group* $A_n$, which has order $n!/2$ for $n \geq 2$.

**Why this matters.** Sign is the simplest nontrivial homomorphism out of $S_n$, and it shows up everywhere: in the definition of the determinant, in solvability of polynomial equations (Galois theory), and in physics, where it is the difference between bosons and fermions.

**The dihedral group $D_n$.** The symmetry group of a regular $n$-gon is the *dihedral group* $D_n$, which has order $2n$. It consists of $n$ rotations $r^0, r^1, \ldots, r^{n-1}$ (where $r$ is rotation by $2\pi/n$) and $n$ reflections $s, rs, r^2s, \ldots, r^{n-1}s$ (where $s$ is a fixed reflection). The group is generated by $r$ and $s$ subject to the relations:

$$r^n = e, \quad s^2 = e, \quad srs = r^{-1}$$

The last relation (equivalently, $sr = r^{-1}s$) is the key: it says that conjugating a rotation by a reflection reverses the rotation. This makes $D_n$ non-abelian for $n \geq 3$.

![The 8 symmetries of the square forming the dihedral group D_4](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_2_dihedral_d4.png)

**Worked Example: The group $D_3$.** The symmetries of an equilateral triangle. Identifying $D_3$ with a subgroup of $S_3$: let $r = (1\ 2\ 3)$ and $s = (1\ 2)$ (reflection swapping vertices $1$ and $2$, fixing vertex $3$). Then:

- $e, r = (1\ 2\ 3), r^2 = (1\ 3\ 2)$ (rotations)
- $s = (1\ 2), rs = (1\ 2\ 3)(1\ 2) = (1\ 3), r^2 s = (1\ 3\ 2)(1\ 2) = (2\ 3)$ (reflections)

Verify: $sr = (1\ 2)(1\ 2\ 3) = (2\ 3) = r^2 s$, which confirms $sr = r^{-1}s = r^2 s$, consistent with the dihedral relation. The six elements of $D_3$ are exactly the six elements of $S_3$, so $D_3 \cong S_3$. This isomorphism is specific to $n = 3$; for $n \geq 4$, $D_n$ is a proper subgroup of $S_n$.

**Numerical sanity check on $D_4$.** Rotations of the square sit at angles $0, 90, 180, 270$ degrees; the four reflections sit on two diagonals and two edge-midpoint axes. Total: $4 + 4 = 8$ elements, matching $|D_4| = 2 \cdot 4 = 8$. Stack two reflections in succession and you always get a rotation (an even number of mirror flips), which matches the algebraic fact that the rotations form an index-2 subgroup.

**An important non-example: the quaternion group $Q_8$.** Not every group of order $8$ is isomorphic to $D_4$ or to a product of cyclic groups. The quaternion group $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$ has the multiplication rules $i^2 = j^2 = k^2 = -1$ and $ij = k$, $ji = -k$, $jk = i$, $kj = -i$, $ki = j$, $ik = -j$. This is a non-abelian group of order $8$ that is *not* isomorphic to $D_4$: in $D_4$, every element has order dividing $4$ and there are five elements of order $2$, while in $Q_8$ there is exactly one element of order $2$ (namely $-1$). The quaternion group will reappear when we classify groups of small order.

A useful invariant for distinguishing groups of the same order is the *order profile* --- the count of elements of each order. For $D_4$ the profile is $(1, 5, 2, 0)$ for orders $(1, 2, 4, 8)$ respectively, while $Q_8$ has profile $(1, 1, 6, 0)$. Different profiles rule out any possible isomorphism. This kind of bookkeeping is the cheapest and often the fastest way to tell two groups apart.

## Subgroups and Lagrange's Theorem

Mental picture: a subgroup is a smaller group living inside a bigger one, sharing the same operation. Lagrange's theorem says these inside-groups can only have certain sizes --- they have to evenly divide the host.

![Subgroup lattice of Z/12Z](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/01_subgroup_lattice.png)

**Definition.** A *subgroup* of $(G, \cdot)$ is a nonempty subset $H \subseteq G$ that is itself a group under the same operation. We write $H \leq G$.

**Subgroup criterion.** A nonempty subset $H \subseteq G$ is a subgroup if and only if for all $a, b \in H$, $ab^{-1} \in H$. (This single condition implies closure, identity, and inverses.)

*Proof.* Suppose $H \neq \emptyset$ and $ab^{-1} \in H$ for all $a, b \in H$. Pick $a \in H$; then $e = aa^{-1} \in H$. For any $b \in H$, $b^{-1} = eb^{-1} \in H$. For any $a, b \in H$, since $b^{-1} \in H$, $ab = a(b^{-1})^{-1} \in H$. Associativity is inherited from $G$. $\square$

**Examples of subgroups.**

- $n\mathbb{Z} = \{nk : k \in \mathbb{Z}\}$ is a subgroup of $(\mathbb{Z}, +)$ for every $n \geq 0$.
- The rotations $\{e, r, r^2, \ldots, r^{n-1}\}$ form a subgroup of $D_n$ isomorphic to $\mathbb{Z}/n\mathbb{Z}$.
- $A_n \leq S_n$ (the even permutations).
- For any $g \in G$, $\langle g \rangle = \{g^n : n \in \mathbb{Z}\}$ is a subgroup of $G$.

![Subgroup lattice (Hasse diagram) of D_4](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_3_subgroup_lattice.png)

**The order of an element.** The *order* of $g \in G$, written $|g|$ or $\text{ord}(g)$, is the smallest positive integer $n$ such that $g^n = e$, or $\infty$ if no such integer exists. Equivalently, $|g| = |\langle g \rangle|$.

**Proposition.** If $|g| = n$, then $g^k = e$ if and only if $n \mid k$.

*Proof.* If $n \mid k$, write $k = qn$ and $g^k = (g^n)^q = e^q = e$. Conversely, if $g^k = e$, write $k = qn + r$ with $0 \leq r < n$. Then $g^r = g^{k - qn} = g^k (g^n)^{-q} = e \cdot e = e$. By minimality of $n$, $r = 0$. $\square$

**Lagrange's Theorem.** If $G$ is a finite group and $H \leq G$, then $|H|$ divides $|G|$.

The proof uses the concept of cosets, which split the group into equal-sized translated copies of $H$.

**Definition.** For $g \in G$ and $H \leq G$, the *left coset* of $H$ containing $g$ is $gH = \{gh : h \in H\}$. The *right coset* is $Hg = \{hg : h \in H\}$.

**Key properties of left cosets:**

(i) Every element of $G$ belongs to some left coset: $g \in gH$ (since $e \in H$).

(ii) Two left cosets are either identical or disjoint. *Proof:* Suppose $aH \cap bH \neq \emptyset$. Then $ah_1 = bh_2$ for some $h_1, h_2 \in H$, so $a = bh_2 h_1^{-1} \in bH$. For any $ah \in aH$, $ah = bh_2 h_1^{-1} h \in bH$, so $aH \subseteq bH$. By symmetry, $bH \subseteq aH$. $\square$

(iii) Every left coset has the same cardinality as $H$. *Proof:* The map $\varphi : H \to gH$ defined by $\varphi(h) = gh$ is a bijection (surjective by definition, and injective by left cancellation). $\square$

![How a subgroup partitions a group of order 12 into cosets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_4_lagrange_partition.png)

**Proof of Lagrange's Theorem.** The distinct left cosets of $H$ in $G$ partition $G$ (by (i) and (ii)). Let $[G:H]$ denote the number of distinct left cosets (the *index* of $H$ in $G$). Each coset has $|H|$ elements (by (iii)). Since the cosets are disjoint and their union is $G$:

$$|G| = [G:H] \cdot |H|$$

In particular, $|H|$ divides $|G|$. $\square$

**Why this matters.** Lagrange's theorem is the first powerful divisibility constraint in group theory. It rules out, for example, a subgroup of order $5$ inside a group of order $12$ before you do any computation. It also gives a non-trivial proof of Fermat's little theorem in two lines (Corollary 2 below), which is one of those moments where abstraction earns its keep.

**Corollary 1.** The order of any element divides the order of the group: if $G$ is finite and $g \in G$, then $|g|$ divides $|G|$.

*Proof.* $|g| = |\langle g \rangle|$, and $\langle g \rangle \leq G$, so $|\langle g \rangle|$ divides $|G|$ by Lagrange. $\square$

**Corollary 2 (Fermat--Euler).** If $\gcd(a, n) = 1$, then $a^{\varphi(n)} \equiv 1 \pmod{n}$.

*Proof.* The residue class $\bar{a}$ belongs to $(\mathbb{Z}/n\mathbb{Z})^*$, which has order $\varphi(n)$. By Corollary 1, $\bar{a}^{\varphi(n)} = \bar{1}$, i.e., $a^{\varphi(n)} \equiv 1 \pmod{n}$. $\square$

**Numerical check of Fermat--Euler.** Take $n = 9$, so $\varphi(9) = 6$. Pick $a = 2$. Then $2^6 = 64 = 7 \cdot 9 + 1$, so $64 \equiv 1 \pmod 9$. The theorem predicts this without you having to compute the exponent.

**Corollary 3.** A group of prime order $p$ is cyclic (isomorphic to $\mathbb{Z}/p\mathbb{Z}$).

*Proof.* Let $|G| = p$. Pick any $g \neq e$ in $G$. Then $|\langle g \rangle|$ divides $p$ and is greater than $1$, so $|\langle g \rangle| = p = |G|$. Hence $\langle g \rangle = G$. $\square$

**Caution.** The converse of Lagrange's theorem is false in general: if $d$ divides $|G|$, there need not exist a subgroup of order $d$. The alternating group $A_4$ has order $12$, and $6 \mid 12$, but $A_4$ has no subgroup of order $6$. (This can be verified by exhaustive enumeration of the subgroups of $A_4$.)

The Sylow theorems, which we will reach in Article 4, restore a partial converse: for every prime power $p^k$ dividing $|G|$, a subgroup of order $p^k$ does exist. The failure for non-prime-power divisors (like $6 = 2 \cdot 3$ in $A_4$) is genuine, not a gap in technique.

## Orders and Cosets: Worked Examples

![A group homomorphism f: Z to Z/4Z and its kernel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_v2_01_6_homomorphism.png)

**Worked Example 1: Subgroups of $\mathbb{Z}/12\mathbb{Z}$.**

The divisors of $12$ are $1, 2, 3, 4, 6, 12$. By the subgroup theorem for cyclic groups, there is exactly one subgroup for each divisor:

| Divisor $d$ | Generator | Subgroup |
|-------------|-----------|----------|
| 1 | $\bar{0}$ | $\{0\}$ |
| 2 | $\bar{6}$ | $\{0, 6\}$ |
| 3 | $\bar{4}$ | $\{0, 4, 8\}$ |
| 4 | $\bar{3}$ | $\{0, 3, 6, 9\}$ |
| 6 | $\bar{2}$ | $\{0, 2, 4, 6, 8, 10\}$ |
| 12 | $\bar{1}$ | $\mathbb{Z}/12\mathbb{Z}$ |

Concrete check on inclusion: does $\{0,4,8\}$ live inside $\{0,3,6,9\}$? Try $4 \in \{0,3,6,9\}$ --- no. So these two subgroups are incomparable in the lattice. The Hasse diagram has $\{0\}$ at the bottom, $\{0,6\}$ and $\{0,4,8\}$ above it (incomparable), then $\{0,3,6,9\}$ and $\{0,2,4,6,8,10\}$ above those, and $\mathbb{Z}/12\mathbb{Z}$ at the top. The lattice mirrors the divisor lattice of $12$.

**Worked Example 2: Cosets of a subgroup of $S_3$.**

Let $H = \langle (1\ 2\ 3) \rangle = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$, a subgroup of $S_3$ of order $3$. The index $[S_3 : H] = 6/3 = 2$, so there are exactly two left cosets:

- $eH = H = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$
- $(1\ 2)H = \{(1\ 2), (1\ 2)(1\ 2\ 3), (1\ 2)(1\ 3\ 2)\} = \{(1\ 2), (1\ 3), (2\ 3)\}$

The two cosets partition $S_3$ into the set of even permutations $A_3 = H$ and the set of odd permutations. Right coset: $H(1\ 2) = \{(1\ 2), (2\ 3), (1\ 3)\}$. Same set. This always happens when the index is $2$: a subgroup of index $2$ is automatically normal.

**Worked Example 3: Element orders in $D_4$.**

The dihedral group $D_4$ has order $8$. Elements: $\{e, r, r^2, r^3, s, rs, r^2s, r^3s\}$.

- $|e| = 1$
- $|r| = 4$, $|r^2| = 2$, $|r^3| = 4$
- $|s| = |rs| = |r^2s| = |r^3s| = 2$

Verification of $|rs| = 2$: $(rs)(rs) = r(sr)s = r(r^{-1}s)s = (rr^{-1})(ss) = e$.

Possible subgroup orders by Lagrange: $1, 2, 4, 8$. The subgroups:
- Order $1$: $\{e\}$
- Order $2$: $\{e, r^2\}$, $\{e, s\}$, $\{e, rs\}$, $\{e, r^2s\}$, $\{e, r^3s\}$ (five of them)
- Order $4$: $\{e, r, r^2, r^3\} \cong \mathbb{Z}/4\mathbb{Z}$; $\{e, r^2, s, r^2s\} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$; $\{e, r^2, rs, r^3s\} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$
- Order $8$: $D_4$ itself

Total: $1 + 5 + 3 + 1 = 10$ subgroups --- a number that cannot be predicted from Lagrange alone but requires explicit computation.

**Worked Example 4: An element-order census of $\mathbb{Z}/15\mathbb{Z}$.** Order $15 = 3 \cdot 5$, so possible element orders are $1, 3, 5, 15$. By the cyclic subgroup theorem, the number of elements of order exactly $d$ (for $d \mid 15$) equals $\varphi(d)$. Counts: $\varphi(1) = 1, \varphi(3) = 2, \varphi(5) = 4, \varphi(15) = 8$. Total: $1 + 2 + 4 + 8 = 15$, matching the group order. The eight generators of $\mathbb{Z}/15\mathbb{Z}$ are the eight integers in $\{1, \ldots, 14\}$ coprime to $15$: namely $1, 2, 4, 7, 8, 11, 13, 14$.

**Worked Example 5: A Cayley table for the Klein four-group $V_4$.** Take $V_4 = \{e, a, b, c\}$ with the relations $a^2 = b^2 = c^2 = e$ and $ab = c$, $ba = c$. The full multiplication table:

| $\cdot$ | $e$ | $a$ | $b$ | $c$ |
|---------|-----|-----|-----|-----|
| $e$ | $e$ | $a$ | $b$ | $c$ |
| $a$ | $a$ | $e$ | $c$ | $b$ |
| $b$ | $b$ | $c$ | $e$ | $a$ |
| $c$ | $c$ | $b$ | $a$ | $e$ |

Notice that the table is symmetric across the main diagonal, confirming that $V_4$ is abelian. Each row and column is a permutation of $\{e, a, b, c\}$, which is a feature of every Cayley table (a consequence of cancellation). The diagonal is all $e$, meaning every non-identity element has order exactly $2$.

## Group Homomorphisms: A First Look

Mental picture: a homomorphism is a function between two groups that respects the operation. Think of it as a structure-preserving translation, much like a continuous map respects topology or a linear map respects vector addition.

**Definition.** A *group homomorphism* $\varphi: G \to H$ is a map satisfying $\varphi(ab) = \varphi(a)\varphi(b)$ for all $a, b \in G$.

A homomorphism automatically sends identity to identity ($\varphi(e_G) = e_H$) and inverses to inverses ($\varphi(a^{-1}) = \varphi(a)^{-1}$). We will prove these basic facts and develop homomorphisms thoroughly in Article 3, but it is useful to meet them here.

**Numerical example.** The map $\varphi: \mathbb{Z} \to \mathbb{Z}/4\mathbb{Z}$ given by $\varphi(n) = n \bmod 4$ is a homomorphism. Check: $\varphi(7 + 9) = \varphi(16) = 0$, and $\varphi(7) + \varphi(9) = 3 + 1 = 4 \equiv 0$. Match. The kernel of $\varphi$ --- the elements that map to $0$ --- is $4\mathbb{Z}$, the multiples of four. The image is all of $\mathbb{Z}/4\mathbb{Z}$.

**Sign as a homomorphism.** The sign map $\text{sgn}: S_n \to \{+1, -1\}$ is a homomorphism into the multiplicative group of two elements. Numerical check on $S_3$: $\text{sgn}((1\ 2)) = -1$, $\text{sgn}((1\ 2\ 3)) = +1$ (since a 3-cycle is two transpositions: $(1\ 2\ 3) = (1\ 3)(1\ 2)$), and $\text{sgn}((1\ 2)(1\ 2\ 3)) = \text{sgn}((1\ 3)) = -1 = (-1)(+1)$. Compatible.

**Isomorphism.** A bijective homomorphism is an *isomorphism*. Two groups related by an isomorphism are essentially the same as algebraic objects, possibly with different labels on their elements. Recognizing when two superficially different groups are isomorphic is a recurring theme. A short list of isomorphisms we have already encountered or implied:

- $D_3 \cong S_3$ (both are the symmetry group of a triangle).
- $(\mathbb{Z}/8\mathbb{Z})^* \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ (verified above by inspecting the multiplication table).
- $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}/mn\mathbb{Z}$ when $\gcd(m, n) = 1$.

**Why this matters.** Almost every result in group theory is stated up to isomorphism. The classification of finite simple groups, for instance, is a list of isomorphism classes, not of specific concrete groups. The point of abstract algebra is precisely that the labels do not matter --- only the structure does.

**A computational reflex.** When someone hands you a finite group via a multiplication table or generators-and-relations, your default move should be: list the elements, compute their orders, count cosets of plausible subgroups, and check the orbit structure under conjugation. Each of these is mechanical, and together they pin down the group up to isomorphism in almost every case under, say, order $30$. The exercises in any standard text (Dummit and Foote chapter 1, for instance) drill this reflex by example.

## What Comes Next

We have established the language of groups, built a working inventory of examples, and proved the first structural theorem. But Lagrange's theorem is only the beginning. It tells us that the order of a subgroup divides the order of the group, but it says nothing about how groups act on external objects.

![Animation: group operation on symmetries](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/01_group_operation.gif)

In the next article, we develop the theory of *group actions*: the formalism for how a group can move things around in a set. This leads to the orbit-stabilizer theorem, Burnside's counting lemma, and the conjugacy class equation --- tools that will let us prove results about the internal structure of groups by studying their external behavior.

Beyond that, the series will continue through normal subgroups and quotient groups (the correct notion of dividing one group by another), homomorphisms and isomorphism theorems, direct and semidirect products, the Sylow theorems (which provide a partial converse to Lagrange), and eventually the classification of finitely generated abelian groups. Each of these builds directly on the foundation laid here.

The reader who has worked through the examples in this article --- computing products in $S_n$, enumerating cosets, cataloguing subgroups --- is ready for what follows. The habit of checking abstract claims against concrete cases is the single most valuable practice in learning algebra. Keep it.

**A remark on notation.** Throughout this series, $|G|$ denotes the order (number of elements) of a finite group $G$, $|g|$ or $\text{ord}(g)$ denotes the order of an element, $H \leq G$ means $H$ is a subgroup of $G$, $[G:H]$ is the index, and $\langle g \rangle$ is the cyclic subgroup generated by $g$. We write $\cong$ for "is isomorphic to." These conventions are standard in algebra textbooks (Lang, Dummit-Foote, Hungerford) and will be used without further comment.

**A remark on the road ahead.** The next eleven articles will revisit every concept introduced here from a more powerful angle. Group actions will subsume the Cayley-table viewpoint. Quotient groups will replace ad hoc coset arguments. Sylow theory will give us the divisibility tools we need to actually classify groups of small order. Field and Galois theory will explain why all of this was originally invented. None of that will make sense, however, unless the foundations laid down here --- four axioms, cosets, Lagrange --- are absolutely solid. Spend the time.

## Deeper Dive: Worked Computations

The fastest way to build intuition for groups is to compute inside several small ones until the patterns stop surprising you. Here are five computations I keep returning to whenever a definition starts to feel hollow.

**Computation A: $S_4$ acting on the four corners of a tetrahedron.** Take $G = S_4$, the symmetric group on $\{1, 2, 3, 4\}$. As a sanity check that the group law is doing what I think it is doing, pick $\sigma = (1\ 2\ 3)$ and $\tau = (1\ 4)$. Composing right-to-left: $\sigma\tau$ sends $1 \mapsto \tau(1) = 4 \mapsto \sigma(4) = 4$, $2 \mapsto 2 \mapsto 3$, $3 \mapsto 3 \mapsto 1$, $4 \mapsto 1 \mapsto 2$. So $\sigma\tau = (1\ 4\ 2\ 3)$, a $4$-cycle. Now $\tau\sigma$: $1 \mapsto 2 \mapsto 2$, $2 \mapsto 3 \mapsto 3$, $3 \mapsto 1 \mapsto 4$, $4 \mapsto 4 \mapsto 1$. So $\tau\sigma = (1\ 2\ 3\ 4)$, also a $4$-cycle but a different one. The two products differ — non-abelian, on a single computation. Cycle types match (both are $(4)$), which they must, because conjugation preserves cycle type and $\tau\sigma = \tau \cdot \sigma\tau \cdot \tau^{-1}$ when $\tau$ is an involution.

**Computation B: powers of an element in $(\mathbb{Z}/13\mathbb{Z})^*$.** The group has order $12$. Take $a = 2$ and compute powers mod $13$: $2, 4, 8, 16 \equiv 3, 6, 12, 24 \equiv 11, 22 \equiv 9, 18 \equiv 5, 10, 20 \equiv 7, 14 \equiv 1$. That is $12$ distinct residues, so $2$ has order $12$ and generates the group. Now take $a = 3$: $3, 9, 27 \equiv 1$. Order $3$. By Lagrange the order of an element divides $12$, and indeed $3 \mid 12$. The element-order census of $(\mathbb{Z}/13\mathbb{Z})^*$, by $\varphi(d)$ count for each $d \mid 12$: orders $1, 2, 3, 4, 6, 12$ appear with multiplicities $1, 1, 2, 2, 2, 4$, summing to $12$. ✓

**Computation C: the centre of $D_4$.** $D_4 = \{e, r, r^2, r^3, s, rs, r^2 s, r^3 s\}$ with $r^4 = e$, $s^2 = e$, $srs = r^{-1}$. Which elements commute with everything? Clearly $e$ does. Does $r$? $rs = r \cdot s$ but $s \cdot r = sr = r^{-1} s = r^3 s$, so $rs \neq sr$ — $r$ is not central. By the same argument, $r^3$ is not central. What about $r^2$? Compute $sr^2 = (sr)r = r^{-1}sr = r^{-1}(sr) = r^{-1} r^{-1} s = r^{-2} s = r^2 s$. So $sr^2 = r^2 s$ and $r^2$ commutes with $s$; it also commutes with all powers of $r$ (the rotation subgroup is abelian). So $r^2$ is central. Reflections like $s$: $rs \neq sr$ already shows $s$ is not central. Conclusion: $Z(D_4) = \{e, r^2\}$, a subgroup of order $2$.

**Computation D: the dihedral relation in matrices.** Realize $D_4$ as $2\times 2$ real matrices: $r = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ and $s = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$. Then $r^2 = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix} = -I$, $s^2 = I$, and $srs = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} = r^{-1}$. The abstract relations check out against an explicit faithful representation. This is also a foretaste of representation theory: every finite group can be realized as matrices.

**Computation E: order of a product is not the product of orders.** Inside $S_5$, the element $\sigma = (1\ 2)$ has order $2$ and $\tau = (3\ 4\ 5)$ has order $3$. They are disjoint and commute, so $\sigma\tau$ has order $\text{lcm}(2, 3) = 6$. But take instead $\sigma = (1\ 2)$ and $\tau = (1\ 2\ 3)$ in $S_3$: both have orders $2$ and $3$ respectively, but $\sigma\tau = (1\ 3)$, which has order $2$, not $6$. The slogan "$|gh|$ divides $\text{lcm}(|g|, |h|)$" is *false* in general; commutativity is essential.

## Counterexamples That Sharpen the Definitions

A cleaner way to internalize the group axioms is to look at structures that satisfy three out of four and watch what breaks.

**Without inverses: monoid.** Take $(\mathbb{N}, +)$, the natural numbers under addition. Associative, identity ($0$), but only $0$ has an additive inverse. Cancellation still works ($a + b = a + c \Rightarrow b = c$), so the structure is well-behaved as a *monoid*, but you cannot solve $a + x = b$ for arbitrary $a, b$. The whole machinery of cosets, quotients, and homomorphism theorems collapses without inverses, because cosets stop partitioning and the kernel-image diagram develops holes.

**Without associativity: quasi-group.** Loosely, a Latin square gives a binary operation in which every row and column is a permutation, but associativity may fail. The smallest non-trivial example is $\mathbb{Z}/5\mathbb{Z}$ under the operation $a \star b = 2a + 3b \pmod 5$. Check $(1 \star 1) \star 1$ vs $1 \star (1 \star 1)$: $1 \star 1 = 2 + 3 = 5 \equiv 0$; $0 \star 1 = 0 + 3 = 3$. And $1 \star (1 \star 1) = 1 \star 0 = 2 + 0 = 2$. So $3 \neq 2$, associativity fails. Without associativity, you cannot even unambiguously write $abc$, and the whole edifice of group theory dissolves.

**Without symmetry of inverses: skew structures.** If you only require *left* inverses ($\exists a': a'a = e$), you do not automatically get right inverses unless you also have associativity. The two-axiom shortcut "$ea = a$ and $a'a = e$ imply group" is a classic exam question — it does work, but only with associativity holding the argument together.

## Common Pitfalls for Beginners

The single most common error I see is conflating the abstract group with one of its representations. The cyclic group of order $4$ has many presentations: $\mathbb{Z}/4\mathbb{Z}$, the fourth roots of unity in $\mathbb{C}$, the rotations of a square, the powers of $i = \sqrt{-1}$, the matrices $\{I, R, R^2, R^3\}$ for $R$ a $90°$ rotation. All five are *the same group*, isomorphic by relabelling. A beginner who has only seen $\mathbb{Z}/4\mathbb{Z}$ as integers mod $4$ may insist that "the group operation is addition" and refuse to recognize the same group when written multiplicatively. Train yourself out of this: a group is its multiplication table up to relabelling, nothing more.

A second pitfall is the assumption that $g^n = e$ implies $|g| = n$. It only implies that $|g|$ *divides* $n$. In $\mathbb{Z}/12\mathbb{Z}$, the element $4$ satisfies $4 \cdot 6 = 24 \equiv 0$, so $6 \cdot 4 = 0$ in additive notation. But also $4 \cdot 3 = 12 \equiv 0$, so the order of $4$ is $3$, not $6$. The order is the *smallest* positive $n$ with $g^n = e$.

A third pitfall: writing $g^{-1} h g$ and thinking of it as "the same as $h$ with $g$'s in front and behind." It is not the same as $h$ unless $g$ commutes with $h$. The whole point of conjugation is that it is a non-trivial operation that captures internal symmetry.

## Where This Shows Up

Group theory is the language of every kind of symmetry, from molecular vibration modes to the Standard Model gauge group $SU(3) \times SU(2) \times U(1)$. Three concrete payoffs:

*Crystallography.* The $230$ space groups classify all possible crystal symmetries in three dimensions. Every textbook table of crystal types (cubic, tetragonal, orthorhombic, ...) is a coarsening of this group-theoretic classification. The reason X-ray diffraction patterns have the symmetries they do is that the underlying atomic arrangement is invariant under one of these groups.

*Public-key cryptography.* RSA, Diffie-Hellman, and elliptic-curve schemes are built on the difficulty of inverting an exponentiation map in a finite cyclic group. The security parameter $|G|$ is exactly what makes these schemes work: the discrete logarithm in $(\mathbb{Z}/p\mathbb{Z})^*$ for $p$ a $2048$-bit prime is computationally hard precisely because the group is huge and structureless from a "shortcut" point of view. We will return to this in Part 12.

*Rubik's cube.* The cube group has order $43{,}252{,}003{,}274{,}489{,}856{,}000$, and every move you can make is a group element. The mathematical theory of solving the cube — including "God's number," the diameter of the Cayley graph — is pure group theory. A solver is, formally, an algorithm that expresses an arbitrary element as a short product of generators.

## What I Want You to Carry Forward

Three questions to keep in your head as you move to Part 2:

1. *If a group $G$ acts on a set $X$, what does the orbit decomposition look like, and how is it constrained by $|G|$?* The orbit-stabilizer theorem will give a clean answer that subsumes Lagrange.
2. *Why is the conjugacy structure of $S_n$ so much richer than that of cyclic groups?* The class equation will quantify exactly how the centre interacts with the orbit decomposition under conjugation.
3. *How can we count colourings of a necklace under rotation?* Burnside's lemma will produce the answer in one line, and the proof is a direct application of orbit-stabilizer.

If you have worked the five computations above and traced through the counterexamples, you are ready. If any of those still feel like incantations, pick the one that is shakiest and do it again with different numbers — a different prime $p$, a different $n$ for $S_n$, a different cyclic group. The reflex of "compute, do not just read" is the only thing that will get you through the next eleven articles.

---
