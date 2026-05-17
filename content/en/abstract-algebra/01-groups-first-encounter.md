---
title: "Groups: Your First Encounter with Algebraic Structure"
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

Most of undergraduate mathematics concerns itself with *specific* objects: the real numbers, continuous functions on $[0,1]$, the vector space $\mathbb{R}^n$. At some point a pattern emerges. The integers under addition and the nonzero rationals under multiplication share a structural resemblance that has nothing to do with the nature of their elements. Both carry a binary operation that is associative, possesses an identity, and admits inverses. Abstract algebra is the study of this structural resemblance, stripped of all incidental detail.

The concept of a *group* is the simplest and most pervasive algebraic structure. It appears in every branch of mathematics: in number theory (the multiplicative group of units modulo $n$), in geometry (the isometry group of a figure), in topology (the fundamental group of a space), in physics (Lie groups governing particle symmetries), and in combinatorics (permutation groups acting on finite sets). Understanding groups is not optional equipment for a working mathematician; it is the baseline.

Historically, the notion crystallized in the early nineteenth century through the work of Galois, who used permutation groups to prove that the general quintic equation has no solution by radicals. Abel had arrived at essentially the same conclusion independently. Cayley later gave the abstract definition we use today. The path from solving polynomial equations to the four axioms of a group is one of the great compressions in mathematical thought: a concrete problem about roots of polynomials led to a framework that now organizes vast tracts of algebra, geometry, and physics.

In this article we define groups, build a library of examples, introduce the notion of a subgroup, and prove Lagrange's theorem --- the first genuinely nontrivial result in the subject. Along the way, every claim will be either proved or demonstrated by explicit computation.


![Cayley table of Z/4Z showing the four group axioms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/01-groups-first-encounter/aa_fig1_cayley_table.png)

## The Formal Definition: Four Axioms

**Definition.** A *group* is a set $G$ together with a binary operation $\cdot : G \times G \to G$ satisfying:

1. **Closure.** For all $a, b \in G$, the element $a \cdot b$ belongs to $G$.
2. **Associativity.** For all $a, b, c \in G$, $(a \cdot b) \cdot c = a \cdot (b \cdot c)$.
3. **Identity.** There exists an element $e \in G$ such that $e \cdot a = a \cdot e = a$ for all $a \in G$.
4. **Inverses.** For every $a \in G$, there exists $a^{-1} \in G$ such that $a \cdot a^{-1} = a^{-1} \cdot a = e$.

When the operation is commutative ($a \cdot b = b \cdot a$ for all $a, b$), we call $G$ an *abelian* group (after Abel). We often write the operation multiplicatively ($ab$ instead of $a \cdot b$) or additively ($a + b$, with identity $0$ and inverse $-a$) depending on context.

**Uniqueness of identity and inverses.** Suppose $e$ and $e'$ are both identities. Then $e = e \cdot e' = e'$, where the first equality uses the fact that $e'$ is an identity and the second uses the fact that $e$ is. Similarly, if $b$ and $c$ are both inverses of $a$, then $b = b \cdot e = b \cdot (a \cdot c) = (b \cdot a) \cdot c = e \cdot c = c$. These two-line proofs are worth internalizing; they are the prototype for many arguments in algebra.

**Example 1: $(\mathbb{Z}, +)$.** The integers under addition form an abelian group. The identity is $0$, and the inverse of $n$ is $-n$. Closure and associativity are inherited from the arithmetic of the integers.

**Example 2: $(\mathbb{Q}^*, \times)$.** The nonzero rationals under multiplication form an abelian group. The identity is $1$, and the inverse of $q$ is $1/q$. We must exclude $0$ because it has no multiplicative inverse.

**Example 3: The trivial group.** The set $\{e\}$ with the operation $e \cdot e = e$ is a group. It is the unique group (up to isomorphism) with one element.

**Non-example: $(\mathbb{Z}, \times)$.** The integers under multiplication fail to form a group because most elements lack inverses. The integer $2$ has no multiplicative inverse in $\mathbb{Z}$.

**Non-example: $(\mathbb{N}, +)$.** The natural numbers $\{0, 1, 2, \ldots\}$ under addition have an identity ($0$) but no inverses: there is no natural number $n$ with $1 + n = 0$. This is a *monoid*, not a group.

**Cancellation laws.** In any group, if $ab = ac$ then $b = c$ (left cancellation), and if $ba = ca$ then $b = c$ (right cancellation). Proof of left cancellation: multiply both sides on the left by $a^{-1}$ to obtain $a^{-1}(ab) = a^{-1}(ac)$, hence $(a^{-1}a)b = (a^{-1}a)c$ by associativity, hence $eb = ec$, hence $b = c$. Right cancellation is analogous.

**Notation.** For $a \in G$ and $n \in \mathbb{Z}$, define $a^n$ inductively: $a^0 = e$, $a^n = a \cdot a^{n-1}$ for $n > 0$, and $a^n = (a^{-1})^{-n}$ for $n < 0$. One verifies that $a^{m+n} = a^m a^n$ and $(a^m)^n = a^{mn}$ for all $m, n \in \mathbb{Z}$.

## Cyclic Groups and the Integers Mod $n$

**Definition.** A group $G$ is *cyclic* if there exists $g \in G$ such that every element of $G$ is a power of $g$: $G = \{g^n : n \in \mathbb{Z}\}$. We write $G = \langle g \rangle$ and call $g$ a *generator*.

The integers $(\mathbb{Z}, +)$ are cyclic, generated by $1$ (or by $-1$). This is the infinite cyclic group.

**The integers modulo $n$.** Fix a positive integer $n$. Define $\mathbb{Z}/n\mathbb{Z} = \{0, 1, 2, \ldots, n-1\}$ with the operation of addition modulo $n$. This is a cyclic group of order $n$ generated by $\bar{1}$ (the residue class of $1$).

Let us verify the axioms for $\mathbb{Z}/6\mathbb{Z}$. The elements are $\{0,1,2,3,4,5\}$. Addition modulo $6$: $3 + 5 = 2$ (since $8 \equiv 2 \pmod{6}$), $4 + 4 = 2$, etc. The identity is $0$. The inverse of $1$ is $5$, of $2$ is $4$, and of $3$ is $3$ (since $3 + 3 = 6 \equiv 0$). Associativity is inherited from $\mathbb{Z}$.

**Worked Example: Generators of $\mathbb{Z}/12\mathbb{Z}$.** The element $\bar{k}$ generates $\mathbb{Z}/12\mathbb{Z}$ if and only if $\gcd(k, 12) = 1$. The integers coprime to $12$ in $\{0, \ldots, 11\}$ are $1, 5, 7, 11$. So $\mathbb{Z}/12\mathbb{Z}$ has exactly four generators. To verify for $k = 5$: the successive multiples of $5$ modulo $12$ are $5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7, 0$, which is all twelve elements.

**Theorem.** Every subgroup of a cyclic group is cyclic. Moreover, if $G = \langle g \rangle$ has order $n$, then for each divisor $d$ of $n$, there is exactly one subgroup of order $d$, namely $\langle g^{n/d} \rangle$.

*Proof sketch.* Let $H \leq G = \langle g \rangle$. If $H = \{e\}$, it is cyclic. Otherwise, let $m$ be the smallest positive integer with $g^m \in H$. We claim $H = \langle g^m \rangle$. Take any $g^k \in H$. Write $k = qm + r$ with $0 \leq r < m$. Then $g^r = g^k (g^m)^{-q} \in H$, and minimality of $m$ forces $r = 0$. Hence $g^k = (g^m)^q \in \langle g^m \rangle$. When $|G| = n$, the condition $\langle g^m \rangle \leq G$ forces $m \mid n$ (since $g^n = e$ implies $n \in \langle m \rangle$ in $\mathbb{Z}$), and $|\langle g^m \rangle| = n/m$. Setting $d = n/m$ gives the result. $\square$

**Infinite versus finite cyclic groups.** The structure theorem above concerns finite cyclic groups, but it is worth noting the infinite case explicitly. The infinite cyclic group $\mathbb{Z}$ has exactly one subgroup for each nonnegative integer $n$: namely $n\mathbb{Z}$. The subgroup $n\mathbb{Z}$ has index $n$ in $\mathbb{Z}$ (for $n \geq 1$), and the quotient $\mathbb{Z}/n\mathbb{Z}$ is the cyclic group of order $n$. This connection between infinite and finite cyclic groups is fundamental: every cyclic group is either isomorphic to $\mathbb{Z}$ or to $\mathbb{Z}/n\mathbb{Z}$ for some $n \geq 1$.

A natural question: given two cyclic groups $\mathbb{Z}/m\mathbb{Z}$ and $\mathbb{Z}/n\mathbb{Z}$, when is their direct product $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$ again cyclic? The answer: if and only if $\gcd(m, n) = 1$. When this holds, $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}/mn\mathbb{Z}$ (the Chinese Remainder Theorem). When $\gcd(m, n) > 1$, the product is abelian but not cyclic: no single element generates the entire group. For example, $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ has four elements, each of order at most $2$, so it cannot be cyclic (a cyclic group of order $4$ has an element of order $4$).

**The group of units $(\mathbb{Z}/n\mathbb{Z})^*$.** Define $(\mathbb{Z}/n\mathbb{Z})^* = \{k \in \mathbb{Z}/n\mathbb{Z} : \gcd(k, n) = 1\}$ with multiplication modulo $n$. This is an abelian group of order $\varphi(n)$, where $\varphi$ is Euler's totient function. For example, $(\mathbb{Z}/8\mathbb{Z})^* = \{1, 3, 5, 7\}$ under multiplication mod $8$, which has order $4$. Its multiplication table:

| $\times$ | 1 | 3 | 5 | 7 |
|-----------|---|---|---|---|
| **1** | 1 | 3 | 5 | 7 |
| **3** | 3 | 1 | 7 | 5 |
| **5** | 5 | 7 | 1 | 3 |
| **7** | 7 | 5 | 3 | 1 |

Every element squares to $1$, so this group is isomorphic to $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ (the Klein four-group), not to $\mathbb{Z}/4\mathbb{Z}$.

## Symmetry Groups: Dihedral and Symmetric Groups

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

**The dihedral group $D_n$.** The symmetry group of a regular $n$-gon is the *dihedral group* $D_n$, which has order $2n$. It consists of $n$ rotations $r^0, r^1, \ldots, r^{n-1}$ (where $r$ is rotation by $2\pi/n$) and $n$ reflections $s, rs, r^2s, \ldots, r^{n-1}s$ (where $s$ is a fixed reflection). The group is generated by $r$ and $s$ subject to the relations:

$$r^n = e, \quad s^2 = e, \quad srs = r^{-1}$$

The last relation (equivalently, $sr = r^{-1}s$) is the key: it says that conjugating a rotation by a reflection reverses the rotation. This makes $D_n$ non-abelian for $n \geq 3$.

**Worked Example: The group $D_3$.** The symmetries of an equilateral triangle. Label the vertices $1, 2, 3$. Then:

- $e$: identity
- $r = (1\ 2\ 3)$: rotation by $120°$
- $r^2 = (1\ 3\ 2)$: rotation by $240°$
- $s = (2\ 3)$: reflection fixing vertex $1$
- $rs = (1\ 2)$: reflection fixing vertex $3$ (this is actually $(1\ 3\ 2)(2\ 3) = (1\ 2)$... let us compute carefully)

Wait --- let me be precise. Identifying $D_3$ with a subgroup of $S_3$: let $r = (1\ 2\ 3)$ and $s = (1\ 2)$ (reflection swapping vertices $1$ and $2$, fixing vertex $3$). Then:

- $e, r = (1\ 2\ 3), r^2 = (1\ 3\ 2)$ (rotations)
- $s = (1\ 2), rs = (1\ 2\ 3)(1\ 2) = (1\ 3), r^2 s = (1\ 3\ 2)(1\ 2) = (2\ 3)$ (reflections)

Verify: $sr = (1\ 2)(1\ 2\ 3) = (2\ 3) = r^2 s$, which confirms $sr = r^{-1}s = r^2 s$, consistent with the dihedral relation. The six elements of $D_3$ are exactly the six elements of $S_3$, so $D_3 \cong S_3$. This isomorphism is specific to $n = 3$; for $n \geq 4$, $D_n$ is a proper subgroup of $S_n$.

**An important non-example: the quaternion group $Q_8$.** Not every group of order $8$ is isomorphic to $D_4$ or to a product of cyclic groups. The quaternion group $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$ has the multiplication rules $i^2 = j^2 = k^2 = -1$ and $ij = k$, $ji = -k$, $jk = i$, $kj = -i$, $ki = j$, $ik = -j$. This is a non-abelian group of order $8$ that is *not* isomorphic to $D_4$: in $D_4$, every element has order dividing $4$ and there are five elements of order $2$, while in $Q_8$ there is exactly one element of order $2$ (namely $-1$). The quaternion group will reappear when we classify groups of small order.

## Subgroups and Lagrange's Theorem

**Definition.** A *subgroup* of $(G, \cdot)$ is a nonempty subset $H \subseteq G$ that is itself a group under the same operation. We write $H \leq G$.

**Subgroup criterion.** A nonempty subset $H \subseteq G$ is a subgroup if and only if for all $a, b \in H$, $ab^{-1} \in H$. (This single condition implies closure, identity, and inverses.)

*Proof.* Suppose $H \neq \emptyset$ and $ab^{-1} \in H$ for all $a, b \in H$. Pick $a \in H$; then $e = aa^{-1} \in H$. For any $b \in H$, $b^{-1} = eb^{-1} \in H$. For any $a, b \in H$, since $b^{-1} \in H$, $ab = a(b^{-1})^{-1} \in H$. Associativity is inherited from $G$. $\square$

**Examples of subgroups.**

- $n\mathbb{Z} = \{nk : k \in \mathbb{Z}\}$ is a subgroup of $(\mathbb{Z}, +)$ for every $n \geq 0$.
- The rotations $\{e, r, r^2, \ldots, r^{n-1}\}$ form a subgroup of $D_n$ isomorphic to $\mathbb{Z}/n\mathbb{Z}$.
- $A_n \leq S_n$ (the even permutations).
- For any $g \in G$, $\langle g \rangle = \{g^n : n \in \mathbb{Z}\}$ is a subgroup of $G$ (the cyclic subgroup generated by $g$).

**The order of an element.** The *order* of $g \in G$, written $|g|$ or $\text{ord}(g)$, is the smallest positive integer $n$ such that $g^n = e$, or $\infty$ if no such integer exists. Equivalently, $|g| = |\langle g \rangle|$.

**Proposition.** If $|g| = n$, then $g^k = e$ if and only if $n \mid k$.

*Proof.* If $n \mid k$, write $k = qn$ and $g^k = (g^n)^q = e^q = e$. Conversely, if $g^k = e$, write $k = qn + r$ with $0 \leq r < n$. Then $g^r = g^{k - qn} = g^k (g^n)^{-q} = e \cdot e = e$. By minimality of $n$, $r = 0$. $\square$

**Lagrange's Theorem.** If $G$ is a finite group and $H \leq G$, then $|H|$ divides $|G|$.

This is one of the cornerstones of finite group theory. The proof uses the concept of cosets.

**Definition.** For $g \in G$ and $H \leq G$, the *left coset* of $H$ containing $g$ is $gH = \{gh : h \in H\}$. The *right coset* is $Hg = \{hg : h \in H\}$.

**Key properties of left cosets:**

(i) Every element of $G$ belongs to some left coset: $g \in gH$ (since $e \in H$).

(ii) Two left cosets are either identical or disjoint. *Proof:* Suppose $aH \cap bH \neq \emptyset$. Then $ah_1 = bh_2$ for some $h_1, h_2 \in H$, so $a = bh_2 h_1^{-1} \in bH$. For any $ah \in aH$, $ah = bh_2 h_1^{-1} h \in bH$, so $aH \subseteq bH$. By symmetry, $bH \subseteq aH$. $\square$

(iii) Every left coset has the same cardinality as $H$. *Proof:* The map $\varphi : H \to gH$ defined by $\varphi(h) = gh$ is a bijection (it is clearly surjective, and injective by left cancellation). $\square$

**Proof of Lagrange's Theorem.** The distinct left cosets of $H$ in $G$ partition $G$ (by (i) and (ii)). Let $[G:H]$ denote the number of distinct left cosets (the *index* of $H$ in $G$). Each coset has $|H|$ elements (by (iii)). Since the cosets are disjoint and their union is $G$:

$$|G| = [G:H] \cdot |H|$$

In particular, $|H|$ divides $|G|$. $\square$

**Corollary 1.** The order of any element divides the order of the group: if $G$ is finite and $g \in G$, then $|g|$ divides $|G|$.

*Proof.* $|g| = |\langle g \rangle|$, and $\langle g \rangle \leq G$, so $|\langle g \rangle|$ divides $|G|$ by Lagrange. $\square$

**Corollary 2 (Fermat--Euler).** If $\gcd(a, n) = 1$, then $a^{\varphi(n)} \equiv 1 \pmod{n}$.

*Proof.* The residue class $\bar{a}$ belongs to $(\mathbb{Z}/n\mathbb{Z})^*$, which has order $\varphi(n)$. By Corollary 1, $\bar{a}^{\varphi(n)} = \bar{1}$, i.e., $a^{\varphi(n)} \equiv 1 \pmod{n}$. $\square$

**Corollary 3.** A group of prime order $p$ is cyclic (isomorphic to $\mathbb{Z}/p\mathbb{Z}$).

*Proof.* Let $|G| = p$. Pick any $g \neq e$ in $G$. Then $|\langle g \rangle|$ divides $p$ and is greater than $1$, so $|\langle g \rangle| = p = |G|$. Hence $\langle g \rangle = G$. $\square$

**Caution.** The converse of Lagrange's theorem is false in general: if $d$ divides $|G|$, there need not exist a subgroup of order $d$. The alternating group $A_4$ has order $12$, and $6 \mid 12$, but $A_4$ has no subgroup of order $6$. (This can be verified by exhaustive enumeration of the subgroups of $A_4$.)

## Orders and Cosets: Worked Examples

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

The *subgroup lattice* (ordered by inclusion) is:

$$\{0\} \subset \{0,6\} \subset \{0,3,6,9\} \subset \mathbb{Z}/12\mathbb{Z}$$
$$\{0\} \subset \{0,6\} \subset \{0,2,4,6,8,10\} \subset \mathbb{Z}/12\mathbb{Z}$$
$$\{0\} \subset \{0,4,8\} \subset \{0,2,4,6,8,10\} \subset \mathbb{Z}/12\mathbb{Z}$$
$$\{0\} \subset \{0,4,8\} \subset \{0,3,6,9\}? \text{ --- No! } \{0,4,8\} \not\subseteq \{0,3,6,9\}.$$

So the lattice has the following Hasse diagram shape: $\{0\}$ at the bottom, $\{0,6\}$ and $\{0,4,8\}$ above it (incomparable), then $\{0,3,6,9\}$ and $\{0,2,4,6,8,10\}$ above those (each containing $\{0,6\}$, and the latter containing $\{0,4,8\}$), and $\mathbb{Z}/12\mathbb{Z}$ at the top. This is a concrete instance of the general principle that the subgroup lattice of $\mathbb{Z}/n\mathbb{Z}$ mirrors the divisor lattice of $n$.

**Worked Example 2: Cosets of a subgroup of $S_3$.**

Let $H = \langle (1\ 2\ 3) \rangle = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$, a subgroup of $S_3$ of order $3$. The index $[S_3 : H] = 6/3 = 2$, so there are exactly two left cosets:

- $eH = H = \{e, (1\ 2\ 3), (1\ 3\ 2)\}$
- $(1\ 2)H = \{(1\ 2), (1\ 2)(1\ 2\ 3), (1\ 2)(1\ 3\ 2)\} = \{(1\ 2), (1\ 3), (2\ 3)\}$ (after computing the products: $(1\ 2)(1\ 2\ 3) = (1\ 3)$ and $(1\ 2)(1\ 3\ 2) = (2\ 3)$)

The two cosets partition $S_3$ into the set of even permutations $A_3 = H$ and the set of odd permutations. This is not a coincidence: $A_n$ always has index $2$ in $S_n$.

Now let us check right cosets: $H(1\ 2) = \{(1\ 2), (1\ 2\ 3)(1\ 2), (1\ 3\ 2)(1\ 2)\} = \{(1\ 2), (2\ 3), (1\ 3)\}$. The right coset equals the left coset as a set. This always happens when the index is $2$ (a subgroup of index $2$ is always normal). Normality is a concept we will develop fully in a later article.

**Worked Example 3: Element orders in $D_4$.**

The dihedral group $D_4$ (symmetries of the square) has order $8$. With generators $r$ (rotation by $90°$) and $s$ (a reflection), the elements are $\{e, r, r^2, r^3, s, rs, r^2s, r^3s\}$.

Orders:
- $|e| = 1$
- $|r| = 4$ (since $r^4 = e$ and $r^2 \neq e$)
- $|r^2| = 2$ (since $(r^2)^2 = r^4 = e$)
- $|r^3| = 4$ (since $r^3$ generates the same cyclic group as $r$)
- $|s| = 2$ (since $s^2 = e$)
- $|rs| = 2$ (compute: $(rs)^2 = rs \cdot rs = r \cdot s \cdot r \cdot s = r \cdot r^{-1} \cdot s \cdot s = e \cdot e = e$, using $sr = r^{-1}s$, so $srs = r^{-1}s^2 = r^{-1}$, hence $rs \cdot rs = r(sr)s = r(r^{-1}s)s = ss = e$... let me redo this. We have $srs^{-1} = r^{-1}$, i.e., $sr = r^{-1}s = r^3 s$. So $(rs)(rs) = r(sr)s = r(r^3 s)s = r^4 s^2 = e$. Yes, $|rs| = 2$.)
- Similarly $|r^2 s| = 2$ and $|r^3 s| = 2$.

By Lagrange's theorem, the possible orders of subgroups of $D_4$ are $1, 2, 4, 8$. The subgroups:
- Order $1$: $\{e\}$
- Order $2$: $\{e, r^2\}$, $\{e, s\}$, $\{e, rs\}$, $\{e, r^2s\}$, $\{e, r^3s\}$ (five subgroups of order $2$... but wait, we need $\{e, r^2\}$ which is correct since $|r^2|=2$; and each reflection generates its own order-$2$ subgroup)
- Order $4$: $\{e, r, r^2, r^3\} \cong \mathbb{Z}/4\mathbb{Z}$; $\{e, r^2, s, r^2s\} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$; $\{e, r^2, rs, r^3s\} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$
- Order $8$: $D_4$ itself

This gives $1 + 5 + 3 + 1 = 10$ subgroups total --- a number that cannot be predicted from Lagrange alone but requires explicit computation.

## What Comes Next

We have established the language of groups, built a working inventory of examples, and proved the first structural theorem. But Lagrange's theorem is only the beginning. It tells us that the order of a subgroup divides the order of the group, but it says nothing about how groups act on external objects.

In the next article, we develop the theory of *group actions*: the formalism for how a group can "move things around" in a set. This leads to the orbit-stabilizer theorem, Burnside's counting lemma, and the conjugacy class equation --- tools that will let us prove results about the internal structure of groups by studying their external behavior.

Beyond that, the series will continue through normal subgroups and quotient groups (the correct notion of "dividing" one group by another), homomorphisms and isomorphism theorems, direct and semidirect products, the Sylow theorems (which provide a partial converse to Lagrange), and eventually the classification of finitely generated abelian groups. Each of these builds directly on the foundation laid here.

The reader who has worked through the examples in this article --- computing products in $S_n$, enumerating cosets, cataloguing subgroups --- is ready for what follows. The habit of checking abstract claims against concrete cases is the single most valuable practice in learning algebra. Keep it.

**A remark on notation.** Throughout this series, $|G|$ denotes the order (number of elements) of a finite group $G$, $|g|$ or $\text{ord}(g)$ denotes the order of an element, $H \leq G$ means $H$ is a subgroup of $G$, $[G:H]$ is the index, and $\langle g \rangle$ is the cyclic subgroup generated by $g$. We write $\cong$ for "is isomorphic to." These conventions are standard in algebra textbooks (Lang, Dummit-Foote, Hungerford) and will be used without further comment.

---

*This is Part 1 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Next: [Part 2 — Group Actions and Symmetry](/en/abstract-algebra/02-group-actions-and-symmetry/)*
