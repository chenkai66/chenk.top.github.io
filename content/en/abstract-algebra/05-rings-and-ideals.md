---
title: "Abstract Algebra (5): Rings and Ideals — When Multiplication Enters the Picture"
date: 2021-09-09 09:00:00
tags:
  - abstract-algebra
  - ring-theory
  - ideals
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "Adding multiplication to the mix: rings, integral domains, ideals, and quotient rings — the algebraic structures behind number theory and polynomial arithmetic."
disableNunjucks: true
series_order: 5
series_total: 12
translationKey: "abstract-algebra-5"
---

Groups capture symmetry through a single operation. But most of the number systems we actually compute with --- integers, polynomials, matrices --- carry two operations that interact: addition and multiplication. The moment you want to talk about divisibility, factorization, or solving equations, one operation is not enough. You need a *ring*.

This article develops ring theory from scratch: the axioms, the key examples, the pathologies that make ring theory richer (and harder) than group theory, and the central concept of an *ideal* --- the ring-theoretic analogue of a normal subgroup. By the end you will have the language to state the First Isomorphism Theorem for rings and to understand why "modding out by an ideal" is the right way to build new rings from old ones.

## From Groups to Rings: Why Two Operations?

Mental picture: a ring is a number system. It has an additive structure (you can add and subtract), a multiplicative structure (you can multiply, possibly without dividing), and the two interact via distributivity. The integers $\mathbb{Z}$ are the prototypical example; everything else is a generalization.

Consider $\mathbb{Z}$. As a group under addition, $(\mathbb{Z}, +)$ is infinite cyclic --- completely understood. But the interesting number theory of $\mathbb{Z}$ involves multiplication: primes, divisibility, the Fundamental Theorem of Arithmetic. Addition alone cannot see any of this structure.

Similarly, consider the set $\mathbb{R}[x]$ of polynomials with real coefficients. As an additive group it is just a vector space, but the ability to *multiply* polynomials is what makes factorization, roots, and algebraic geometry possible.

The pattern repeats:

- $\mathbb{Z}/n\mathbb{Z}$: modular arithmetic uses both operations.
- $M_n(\mathbb{R})$: matrix algebra needs both, and multiplication is not commutative.
- Function spaces: pointwise addition and multiplication of functions $f: X \to \mathbb{R}$.

In each case, addition gives you an abelian group, multiplication gives an associative operation, and the two are linked by distributivity. Abstracting yields the notion of a ring.

Historically, the concept of a ring crystallized in the late 19th century from two sources. *Algebraic number theory*: Dedekind studied rings of algebraic integers like $\mathbb{Z}[\sqrt{-5}]$ to understand when unique factorization fails and how to restore it using ideals. *Invariant theory*: Hilbert proved that rings of polynomial invariants are finitely generated. The formal axiomatization was completed by Emmy Noether in the 1920s.

![Ring hierarchy: rings, integral domains, UFDs, PIDs, Euclidean domains, fields](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_1_ring_hierarchy.png)

## Ring Axioms and the Zoo of Examples

Mental picture: a ring is two operations joined at the hip. Distributivity is the glue: it forces the two operations to play nicely together, ruling out monstrosities like $1 \cdot 0 \neq 0$.

**Definition.** A *ring* is a set $R$ with two binary operations $+$ and $\cdot$ such that:

1. $(R, +)$ is an abelian group with identity $0$.
2. Multiplication is associative.
3. Distributivity: $a(b+c) = ab + ac$ and $(a+b)c = ac + bc$.

A ring is *unital* if there exists $1 \in R$ with $1 \cdot a = a \cdot 1 = a$. *Commutative* if $ab = ba$. Throughout, "ring" means commutative unital unless stated otherwise (we will be explicit when we need to drop assumptions).

The two-sided distributivity is needed in non-commutative rings; in commutative rings, $a(b+c) = ab + ac$ implies $(a+b)c = ac + bc$ automatically. Note that in some textbook traditions, "ring" includes commutativity by default; in others (e.g., Lang), it does not. In contemporary research, "commutative ring" and "(non-commutative) ring" are the two main camps, both important.

**Immediate consequences:**

- $0 \cdot a = 0$. (Proof: $0 \cdot a = (0+0) \cdot a = 0 \cdot a + 0 \cdot a$, cancel.)
- $(-1) \cdot a = -a$.
- $(-a)(-b) = ab$.

### A Catalog of Rings

| Ring | Commutative? | Unity? | Notes |
|------|--------------|--------|-------|
| $\mathbb{Z}$ | yes | $1$ | the prototype |
| $\mathbb{Z}/n\mathbb{Z}$ | yes | $\bar 1$ | field iff $n$ is prime |
| $\mathbb{Q}, \mathbb{R}, \mathbb{C}$ | yes | $1$ | fields |
| $\mathbb{Z}[i] = \{a+bi : a, b \in \mathbb{Z}\}$ | yes | $1$ | Gaussian integers |
| $\mathbb{R}[x]$ | yes | $1$ | polynomial ring |
| $M_n(\mathbb{R})$ | no ($n \geq 2$) | $I_n$ | matrix ring |
| $\mathbb{H}$ (quaternions) | no | $1$ | division ring |
| $2\mathbb{Z}$ (even integers) | yes | no | non-unital |
| $C([0,1], \mathbb{R})$ | yes | $f \equiv 1$ | continuous functions |

![Catalog of classical rings with their key properties](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_7_ring_examples.png)

A *field* is a commutative unital ring in which every nonzero element has a multiplicative inverse. A *division ring* drops commutativity but keeps inverses. The quaternions $\mathbb{H}$: $i^2 = j^2 = k^2 = ijk = -1$, with $ij = k \neq -k = ji$.

**Units.** $u \in R$ is a *unit* if it has a two-sided inverse. The set $R^\times$ of units forms a group under multiplication.

- $\mathbb{Z}^\times = \{\pm 1\}$.
- $(\mathbb{Z}/n\mathbb{Z})^\times$: integers coprime to $n$, order $\varphi(n)$.
- $M_n(\mathbb{R})^\times = GL_n(\mathbb{R})$.
- For a field $F$, $F^\times = F \setminus \{0\}$.

**Numerical example: $(\mathbb{Z}/12\mathbb{Z})^\times$.** Elements coprime to $12$: $\{1, 5, 7, 11\}$. So $|(\mathbb{Z}/12\mathbb{Z})^\times| = 4 = \varphi(12)$. Each element squares to $1$ ($5^2 = 25 \equiv 1, 7^2 = 49 \equiv 1, 11^2 = 121 \equiv 1$), so this group is $\mathbb{Z}/2 \times \mathbb{Z}/2$.

**Numerical example: $(\mathbb{Z}/15\mathbb{Z})^\times$.** Elements coprime to $15$: $\{1, 2, 4, 7, 8, 11, 13, 14\}$, eight elements, $\varphi(15) = 8$. By CRT this group is $(\mathbb{Z}/3)^\times \times (\mathbb{Z}/5)^\times = \mathbb{Z}/2 \times \mathbb{Z}/4 \cong \mathbb{Z}/4 \times \mathbb{Z}/2$, an abelian group of order $8$ but not cyclic ($\mathbb{Z}/8 \neq \mathbb{Z}/4 \times \mathbb{Z}/2$).

**Numerical example: $\mathbb{Z}[i]^\times$.** Units of $\mathbb{Z}[i]$ have norm $1$, so $a^2 + b^2 = 1$ in $\mathbb{Z}$, giving $\{1, -1, i, -i\}$. Four units, forming a cyclic group of order $4$.

**Why this matters.** The ring axioms define a structure that sits one level above groups: groups have one operation, rings have two. Almost every concrete number system you compute with is a ring, and ring theory provides a unified framework for analyzing all of them.

Stratification by additional properties: commutative or not, with unity or not, finite or not, integral domain or not. Each refinement excludes some pathologies and admits stronger theorems. The bulk of classical algebra works with commutative integral domains, which strike a balance between richness and tractability.

**A quirk: zero divisors.** In $\mathbb{Z}/6\mathbb{Z}$, we have $\bar 2 \cdot \bar 3 = \bar 6 = \bar 0$, even though neither factor is zero. This phenomenon does not happen in $\mathbb{Z}$ or in any field. Such rings are called integral domains; the others are not. We will return to this distinction.

A practical implication of zero divisors: in a ring with zero divisors, you cannot meaningfully "divide by nonzero elements." The cancellation law fails. This rules out building a field of fractions, which means the ring is fundamentally less arithmetic than the integers.

**Subrings.** $S \subseteq R$ is a *subring* if it is a ring with the same operations and contains $1_R$. Examples: $\mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$. The Gaussian integers $\mathbb{Z}[i] \subset \mathbb{C}$.

The subring test: a non-empty subset $S$ is a subring iff $1 \in S$ and $S$ is closed under subtraction and multiplication. (Subtraction takes care of additive inverses; closure under subtraction implies closure under addition since $a + b = a - (-b) = a - (0 - b)$.)

**More examples of subrings.** $\mathbb{Z}[\sqrt 2] = \{a + b\sqrt 2 : a, b \in \mathbb{Z}\}$ in $\mathbb{R}$. $\mathbb{Z}[\omega]$ where $\omega = e^{2\pi i/3}$ (Eisenstein integers) in $\mathbb{C}$. The center $Z(M_n(R))$ in any ring of $n \times n$ matrices is a subring. The constants in $\mathbb{R}[x]$ are a subring isomorphic to $\mathbb{R}$.

## Integral Domains and Zero Divisors

Mental picture: an integral domain is a ring where the cancellation law $ab = ac \Rightarrow b = c$ holds whenever $a \neq 0$. This is the algebraic statement that "you can divide by nonzero elements without losing information."

**Definition.** A nonzero $a \in R$ is a *zero divisor* if there exists nonzero $b$ with $ab = 0$. A commutative unital ring with no zero divisors is an *integral domain*.

**Examples.**

- $\mathbb{Z}$: integral domain.
- Every field: integral domain (multiply by inverse).
- $\mathbb{Z}/p\mathbb{Z}$ for prime $p$: field, hence integral domain.
- $\mathbb{Z}/6\mathbb{Z}$: not an integral domain ($\bar 2 \cdot \bar 3 = 0$).
- $M_2(\mathbb{R})$: not an integral domain (and not commutative).
- $\mathbb{Z}[x]$: integral domain. $\mathbb{Z}[i]$: integral domain.
- $\mathbb{R}[x]/(x^2 - 1)$: not an integral domain. $(x-1)(x+1) = 0$ in this ring.

**Numerical example of zero divisors.** In $\mathbb{Z}/12\mathbb{Z}$: the zero divisors are $\{2, 3, 4, 6, 8, 9, 10\}$. The non-zero non-zero-divisors are $\{1, 5, 7, 11\}$, which is exactly the set of units. This is a special phenomenon for $\mathbb{Z}/n\mathbb{Z}$: every nonzero non-zero-divisor in a finite commutative ring is a unit. (The proof is the same as for finite integral domains.)

**Cancellation law.** In an integral domain, $ab = ac$ with $a \neq 0$ implies $b = c$. This is the property that lets us think of integral domains as "generalized integers."

The cancellation law is exactly the key property used in the construction of the field of fractions: it lets us declare that the equivalence relation $(a, b) \sim (c, d) \iff ad = bc$ is well-behaved. In a ring with zero divisors (like $\mathbb{Z}/6\mathbb{Z}$), the natural fraction construction collapses --- $\bar 1 / \bar 2$ does not have a well-defined value because $\bar 2$ is a zero divisor.

**The field of fractions.** Any integral domain $R$ embeds in a smallest field $\text{Frac}(R)$, constructed as equivalence classes of pairs $(a, b)$ with $b \neq 0$, modulo $(a, b) \sim (c, d) \iff ad = bc$.

Examples: $\text{Frac}(\mathbb{Z}) = \mathbb{Q}$, $\text{Frac}(\mathbb{Z}[i]) = \mathbb{Q}(i)$, $\text{Frac}(F[x]) = F(x)$.

Operations: $a/b + c/d = (ad + bc)/(bd)$ and $(a/b)(c/d) = (ac)/(bd)$. These are well-defined precisely because the cancellation law holds. The map $R \to \text{Frac}(R), a \mapsto a/1$ is an injective ring homomorphism, and $\text{Frac}(R)$ is *universal* among fields containing $R$: every injective ring map from $R$ into a field factors uniquely through $\text{Frac}(R)$.

**Proposition.** Every finite integral domain is a field.

*Proof.* Let $a \neq 0$. The map $r \mapsto ar$ is injective (cancellation), so surjective on a finite set. So $ar = 1$ for some $r$. $\square$

**Why this matters.** The proposition explains why $\mathbb{Z}/p\mathbb{Z}$ is a field for prime $p$: it is a finite integral domain, hence automatically a field. This is the algebraic foundation of finite field arithmetic, which underlies modern coding theory, cryptography, and computer algebra.

For prime $p$ and any $n \geq 1$, there is a unique field of order $p^n$ up to isomorphism, denoted $\mathbb{F}_{p^n}$. Constructing it: take $\mathbb{F}_p[x]/(f(x))$ for any irreducible $f$ of degree $n$. The resulting field has $p^n$ elements, and is independent of the choice of $f$. We will explore this in Article 7 on field extensions.

**Worked Example.** $\mathbb{Z}/n\mathbb{Z}$ is an integral domain $\iff$ $n$ is prime. If $n = ab$ ($1 < a, b < n$), then $\bar a \bar b = \bar 0$ with neither factor zero. Conversely, if $p$ prime and $\bar a \bar b = 0$, then $p \mid ab$, so by Euclid's lemma $p \mid a$ or $p \mid b$.

**Worked Example: Gaussian integers $\mathbb{Z}[i]$.** Define norm $N(a + bi) = a^2 + b^2$. Multiplicative: $N(\alpha\beta) = N(\alpha)N(\beta)$. If $\alpha\beta = 0$ then $N(\alpha)N(\beta) = 0$, so $N(\alpha) = 0$ or $N(\beta) = 0$, i.e., one of them is zero. So $\mathbb{Z}[i]$ is an integral domain.

The norm provides a Euclidean function: for any $\alpha, \beta \in \mathbb{Z}[i]$ with $\beta \neq 0$, there exist $q, r \in \mathbb{Z}[i]$ with $\alpha = q\beta + r$ and $N(r) < N(\beta)$. (Geometrically: round $\alpha/\beta \in \mathbb{Q}(i)$ to the nearest Gaussian integer.) This makes $\mathbb{Z}[i]$ a *Euclidean domain*, hence a PID, hence a UFD --- one of the rare cases where all the good factorization properties hold simultaneously.

**Numerical example in $\mathbb{Z}[i]$.** $(2 + i)(2 - i) = 4 - i^2 = 5$. So $2 \pm i$ are a factorization of $5$ in $\mathbb{Z}[i]$. The norm $N(2 + i) = 5$ is prime in $\mathbb{Z}$, which forces $2 + i$ to be irreducible in $\mathbb{Z}[i]$ (as we will see). The prime $5$ in $\mathbb{Z}$ "splits" in $\mathbb{Z}[i]$ as $(2+i)(2-i)$, a phenomenon that prefigures the deep theory of prime ideal splitting in algebraic number rings.

**Numerical example: which integer primes are sums of two squares.** $2 = 1^2 + 1^2$, $5 = 1^2 + 2^2$, $13 = 2^2 + 3^2$, $17 = 1^2 + 4^2$, $29 = 2^2 + 5^2$. By contrast, $3, 7, 11, 19, 23$ are not sums of two squares. Theorem (Fermat): an odd prime $p$ is a sum of two squares iff $p \equiv 1 \pmod 4$. The proof uses the structure of $\mathbb{Z}[i]$: primes $p \equiv 1 \pmod 4$ split as $p = \pi \bar\pi$ with $\pi \in \mathbb{Z}[i]$, while $p \equiv 3 \pmod 4$ remain prime. This is the simplest interesting case of "splitting of primes in number rings," a central theme in algebraic number theory.

## Ideals

Mental picture: an ideal is a subset closed under addition and "absorbing" multiplication by everything in the ring. It is the right notion of "kernel" for ring homomorphisms, and the right thing to mod out by.

**Definition.** A ring homomorphism $\varphi: R \to S$ satisfies:

1. $\varphi(a + b) = \varphi(a) + \varphi(b)$,
2. $\varphi(ab) = \varphi(a)\varphi(b)$,
3. $\varphi(1_R) = 1_S$.

Note the third axiom is independent of the first two: a "ring homomorphism" that does not preserve $1$ is a strange object, not a homomorphism in the proper sense. The map $\mathbb{Z} \to \mathbb{Z} \times \mathbb{Z}$ by $a \mapsto (a, 0)$ satisfies $(1) + (2)$ but not $(3)$, so it is not a ring homomorphism in the modern convention.

The kernel $\ker\varphi = \{r : \varphi(r) = 0\}$ is closed under addition (subgroup) and absorbs multiplication: if $a \in \ker\varphi$ and $r \in R$, then $\varphi(ra) = \varphi(r) \cdot 0 = 0$, so $ra \in \ker\varphi$.

**Definition.** $I \subseteq R$ is an *ideal* if:

1. $(I, +) \le (R, +)$.
2. $a \in I, r \in R \Rightarrow ra \in I$ and $ar \in I$.

Asymmetry note: in non-commutative rings we distinguish *left ideals* ($ra \in I$ only), *right ideals*, and *two-sided ideals*. In a commutative ring, the three notions coincide.

### Principal Ideals

In a commutative ring, the *principal ideal* $(a) = aR = \{ar : r \in R\}$.

![A principal ideal (a) inside the ring Z[x]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_3_principal_ideal.png)

**Theorem.** Every ideal of $\mathbb{Z}$ is principal.

*Proof.* Let $I \neq \{0\}$. Pick the smallest positive element $d \in I$. Then $(d) \subseteq I$. For $a \in I$, write $a = qd + r$ with $0 \le r < d$. Then $r = a - qd \in I$, so by minimality $r = 0$. So $I = (d)$. $\square$

This makes $\mathbb{Z}$ a *principal ideal domain* (PID).

![The ideals (n) of Z displayed as a divisibility lattice](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_2_z_ideals.png)

**Numerical example: ideals of $\mathbb{Z}/12\mathbb{Z}$.** The ideals correspond to divisors of $12$: $(1) = $ everything, $(2) = \{0, 2, 4, 6, 8, 10\}$, $(3) = \{0, 3, 6, 9\}$, $(4) = \{0, 4, 8\}$, $(6) = \{0, 6\}$, $(12) = \{0\}$. Six ideals total, matching the six divisors of $12$. The lattice of ideals mirrors the divisor lattice.

**Numerical example: ideals of $\mathbb{Z}[i]$.** Ideals of $\mathbb{Z}[i]$ are also principal (it is a PID), generated by Gaussian integers. The norm of the generator gives the index of the ideal in $\mathbb{Z}[i]$: $|\mathbb{Z}[i]/(\alpha)| = N(\alpha)$. So $\mathbb{Z}[i]/(2 + i)$ has $5$ elements, $\mathbb{Z}[i]/(3)$ has $9$ elements, etc.

### Maximal and Prime Ideals

**Definition.** A proper ideal $\mathfrak{m} \subsetneq R$ is *maximal* if no ideal $I$ has $\mathfrak{m} \subsetneq I \subsetneq R$.

**Definition.** A proper ideal $\mathfrak{p} \subsetneq R$ is *prime* if $ab \in \mathfrak{p}$ implies $a \in \mathfrak{p}$ or $b \in \mathfrak{p}$.

**Theorem.** In a commutative unital ring:

- $R/\mathfrak{m}$ is a field $\iff$ $\mathfrak{m}$ is maximal.
- $R/\mathfrak{p}$ is an integral domain $\iff$ $\mathfrak{p}$ is prime.
- Every maximal ideal is prime.

*Proof of "maximal $\Rightarrow$ field."* Let $\bar a \neq 0$ in $R/\mathfrak{m}$, so $a \notin \mathfrak{m}$. The ideal $\mathfrak{m} + (a)$ strictly contains $\mathfrak{m}$, so equals $R$. Hence $1 = m + ra$ for some $m \in \mathfrak{m}, r \in R$, giving $\bar r \bar a = \bar 1$. $\square$

![Prime ideals vs maximal ideals as a Venn diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_5_prime_maximal.png)

**Why this matters.** Prime and maximal ideals are the algebraic analogues of points in geometry. The set of prime ideals of $R$ (called $\text{Spec}(R)$) carries a topology that makes it a geometric space: this is the foundation of algebraic geometry. Maximal ideals correspond to "closed points," prime ideals to "subvarieties."

For $R = \mathbb{C}[x]$, the maximal ideals are exactly $(x - a)$ for $a \in \mathbb{C}$, in bijection with the points of $\mathbb{C}$. The corresponding quotient $\mathbb{C}[x]/(x - a) \cong \mathbb{C}$ is the "function value at $a$." This is the simplest case of the Hilbert Nullstellensatz: maximal ideals of $\mathbb{C}[x_1, \ldots, x_n]$ correspond to points of $\mathbb{C}^n$.

**Example: $\mathbb{Z}$.** Prime ideals: $(0)$ and $(p)$ for primes $p$. Maximal ideals: $(p)$. $(0)$ is prime (since $\mathbb{Z}/(0) = \mathbb{Z}$ is an integral domain) but not maximal (since $\mathbb{Z}$ is not a field).

**Worked Example: $(x)$ in $\mathbb{Z}[x]$.** $\mathbb{Z}[x]/(x) \cong \mathbb{Z}$ (evaluate at $0$). $\mathbb{Z}$ is an integral domain but not a field, so $(x)$ is prime but not maximal. $(x, 2) \supsetneq (x)$, with $\mathbb{Z}[x]/(x, 2) \cong \mathbb{Z}/2$, a field, so $(x, 2)$ is maximal.

**Worked Example: $(x, y)$ in $k[x, y]$.** $k[x, y]/(x, y) \cong k$ (evaluate at $(0, 0)$), so $(x, y)$ is maximal. By contrast, $(x)$ in $k[x, y]$ has quotient $k[y]$, which is an integral domain but not a field, so $(x)$ is prime but not maximal. The chain $(0) \subsetneq (x) \subsetneq (x, y)$ in $k[x, y]$ has length $2$, reflecting the fact that $\mathbb{A}^2$ is two-dimensional.

**Worked Example: prime ideals of $\mathbb{Z}[x]$.** Three flavors:
1. $(0)$ --- the zero ideal, corresponding to "the generic point."
2. $(p)$ for primes $p \in \mathbb{Z}$ --- corresponding to reduction mod $p$.
3. $(f(x))$ for $f$ irreducible in $\mathbb{Q}[x]$ with content $1$ --- corresponding to algebraic numbers.
4. $(p, f)$ where $p$ is prime and $f$ is irreducible mod $p$ --- the maximal ideals.

Only the type-4 ideals are maximal. This stratification underlies the geometry of $\text{Spec}(\mathbb{Z}[x])$, sometimes called the "arithmetic plane."

**Operations on ideals.**

- Sum: $I + J = \{a + b : a \in I, b \in J\}$.
- Product: $IJ = \{\sum a_k b_k : a_k \in I, b_k \in J\}$ (finite sums).
- Intersection: $I \cap J$.

When $I + J = R$ (comaximal), the Chinese Remainder Theorem for rings gives $R/(I \cap J) \cong R/I \times R/J$.

**Numerical example.** In $\mathbb{Z}$: $(4) + (6) = (\gcd(4,6)) = (2)$, $(4) \cap (6) = (\text{lcm}(4,6)) = (12)$, $(4)(6) = (24)$. Note $(4) + (6) \neq R$, so they are not comaximal. By contrast $(4) + (9) = (1) = \mathbb{Z}$, and $\mathbb{Z}/(4 \cap 9) = \mathbb{Z}/36 \cong \mathbb{Z}/4 \times \mathbb{Z}/9$ (CRT).

**A useful identity.** For ideals $I, J, K$ in a commutative ring: $I(J + K) = IJ + IK$, $I \cap (J + K) \supseteq (I \cap J) + (I \cap K)$, with equality not guaranteed. The lattice of ideals is *modular* (in the lattice-theoretic sense) but generally not distributive.

## Quotient Rings and the First Isomorphism Theorem

Mental picture: just as for groups, you mod out by an ideal to get a quotient ring. The ideal axioms guarantee the multiplication is well-defined.

For $I \trianglelefteq R$ a two-sided ideal, $R/I$ has elements $\{a + I\}$ and operations:

$$(a + I) + (b + I) = (a+b) + I, \quad (a + I)(b + I) = ab + I$$

**Multiplication is well-defined precisely because $I$ is an ideal.** If $a' = a + i$, $b' = b + j$ with $i, j \in I$:

$$a'b' = ab + aj + ib + ij$$

Each of $aj, ib, ij$ is in $I$ by absorption. So $a'b' - ab \in I$.

![Z[x]/(x^2+1) is isomorphic to the Gaussian integers Z[i]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_4_quotient_ring.png)

**First Isomorphism Theorem for Rings.** If $\varphi: R \to S$ is a ring homomorphism, then $\ker\varphi \trianglelefteq R$, $\text{im}\,\varphi$ is a subring of $S$, and

$$R/\ker\varphi \cong \text{im}\,\varphi.$$

![A ring homomorphism preserving addition and multiplication](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_v2_05_6_ring_homo.png)

**Worked Example: $\mathbb{R}[x]/(x^2+1) \cong \mathbb{C}$.** Define $\varphi: \mathbb{R}[x] \to \mathbb{C}$ by $\varphi(f) = f(i)$. Surjective. Kernel: polynomials vanishing at $i$, which is $(x^2 + 1)$ (the minimal polynomial of $i$). By the theorem, $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$.

This shows the power of the construction: we built $\mathbb{C}$ purely algebraically, without geometric intuition. Cosets of $a + bx$ correspond to $a + bi$, and the relation $\bar x^2 + 1 = 0$ enforces $\bar x^2 = -1$.

**Worked Example: $\mathbb{R}[x]/(x^2 - 1)$.** $x^2 - 1 = (x-1)(x+1)$ in $\mathbb{R}[x]$. The two factors are comaximal (since $\gcd = 1$). By CRT, $\mathbb{R}[x]/(x^2 - 1) \cong \mathbb{R}[x]/(x-1) \times \mathbb{R}[x]/(x+1) \cong \mathbb{R} \times \mathbb{R}$. So this quotient is not a field --- it has zero divisors $(\bar x - 1)(\bar x + 1) = 0$. Compare with $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$, which is a field because $x^2 + 1$ is irreducible.

The pattern: a quotient $F[x]/(f(x))$ is a field iff $f$ is irreducible over $F$. If $f$ factors, the quotient is a product of smaller quotients (CRT), with zero divisors.

**Worked Example: $\mathbb{Z}[x]/(x^2 + 1, 5)$.** $\cong (\mathbb{Z}/5\mathbb{Z})[x]/(x^2 + 1)$. In $\mathbb{F}_5$, $2^2 + 1 = 5 = 0$, so $x = 2$ is a root, and $x^2 + 1 = (x - 2)(x - 3)$. By CRT for rings, the quotient is $\mathbb{F}_5 \times \mathbb{F}_5$. So $\mathbb{Z}[x]/(x^2 + 1, 5) \cong \mathbb{F}_5 \times \mathbb{F}_5$.

**Worked Example: $\mathbb{Z}[i]$ as a quotient.** $\mathbb{Z}[i] \cong \mathbb{Z}[x]/(x^2 + 1)$. The map $f(x) \mapsto f(i)$ is a ring homomorphism $\mathbb{Z}[x] \to \mathbb{Z}[i]$, surjective, with kernel $(x^2 + 1)$. So $\mathbb{Z}[i]$ is naturally the quotient of the free polynomial ring by the relation $i^2 + 1 = 0$.

**Numerical example.** In $\mathbb{Z}[i] / (2 + i)$: the element $\bar 5 = (2+i)(2-i)$ is in the ideal, so $\bar 5 = 0$ in the quotient. Also $\bar i = -2$ in the quotient (from $2 + i = 0$). So every element has a representative $a + bi$ with $i \to -2$, giving $a - 2b$, and $a - 2b$ ranges over $\mathbb{Z}/5\mathbb{Z}$. Hence $\mathbb{Z}[i]/(2+i) \cong \mathbb{Z}/5\mathbb{Z}$.

**Worked Example: $\mathbb{Z}[\sqrt{2}]$ via quotient.** $\mathbb{Z}[\sqrt{2}] = \{a + b\sqrt{2} : a, b \in \mathbb{Z}\} \cong \mathbb{Z}[x]/(x^2 - 2)$, the quotient by the minimal polynomial of $\sqrt{2}$ over $\mathbb{Z}$. The norm $N(a + b\sqrt 2) = a^2 - 2b^2$ is multiplicative.

**Worked Example: a finite field via quotient.** $\mathbb{F}_2[x]/(x^2 + x + 1)$ is a field of order $4$, since $x^2 + x + 1$ is irreducible over $\mathbb{F}_2$ (no roots: $0^2 + 0 + 1 = 1, 1^2 + 1 + 1 = 1$). Its elements are $\{0, 1, \alpha, \alpha + 1\}$ where $\alpha^2 = \alpha + 1$. This is the field $\mathbb{F}_4$, and it is an example of how to build finite fields of any prime power order via quotients.

**Why this matters.** Quotient ring constructions let us build new rings (and especially new fields) from old ones with surgical precision. Field extensions, splitting fields, $p$-adic numbers, and most of algebraic number theory are built by repeated quotient ring constructions.

A useful slogan: "to add a relation $r = 0$ to a ring, mod out by the ideal generated by $r$." Want to add a square root of 2 to $\mathbb{Q}$? Take $\mathbb{Q}[x]/(x^2 - 2)$. Want to make $5 = 0$ in $\mathbb{Z}$? Take $\mathbb{Z}/(5) = \mathbb{F}_5$. Want both? Take $\mathbb{Z}[x]/(x^2 - 2, 5) = \mathbb{F}_5[x]/(x^2 - 2)$. Want to invert $2$? Take $\mathbb{Z}[1/2]$, which is a localization (a slightly different construction, but in the same spirit).

## PIDs and the Ascending Chain Condition

Mental picture: a PID is an integral domain with the simplest possible ideal structure --- every ideal is generated by one element. PIDs have particularly clean factorization theory.

**Definition.** A PID is an integral domain in which every ideal is principal.

**Examples:** $\mathbb{Z}$, $F[x]$ for $F$ a field, $\mathbb{Z}[i]$.

**Why this matters.** PIDs strike a balance between concreteness and structure. They have unique factorization (we will prove this in Article 6), they support the full toolkit of gcd's and Bezout's identity, and their ideals form a tractable lattice (the divisibility lattice of generators). Most "well-behaved" rings encountered in elementary algebra are PIDs.

**Non-example: $\mathbb{Z}[x]$.** The ideal $(2, x)$ is not principal. If $(2, x) = (g)$, then $g \mid 2$ and $g \mid x$. From $g \mid 2$ in $\mathbb{Z}[x]$, $g$ is a constant $\pm 1$ or $\pm 2$. From $g \mid x$: $g = \pm 2$ would mean $2 \mid x$, impossible. So $g = \pm 1$, but $(g) = \mathbb{Z}[x]$ contradicts $1 \notin (2, x)$ (since $1$ is odd and not divisible by $x$).

### The Ascending Chain Condition

A ring is *Noetherian* if every ascending chain $I_1 \subseteq I_2 \subseteq \cdots$ of ideals stabilizes.

**Proposition.** Every PID is Noetherian.

*Proof.* Let $I_1 \subseteq I_2 \subseteq \cdots$ be ascending. The union $I = \bigcup I_n$ is an ideal. Since PID, $I = (d)$ for some $d$. Then $d \in I_N$ for some $N$, so $I_n = (d)$ for $n \geq N$. $\square$

**Why this matters.** Noetherian rings are the workhorses of commutative algebra. The condition rules out infinite ascending chains, which is exactly what is needed to make many existence proofs go through. Hilbert Basis Theorem, primary decomposition, dimension theory --- all rest on Noether's condition.

A typical use: in a Noetherian ring, every ideal is finitely generated. Suppose $I$ has no finite generating set. Pick $a_1 \in I$, then $a_2 \in I \setminus (a_1)$, then $a_3 \in I \setminus (a_1, a_2)$, etc. The chain $(a_1) \subsetneq (a_1, a_2) \subsetneq (a_1, a_2, a_3) \subsetneq \cdots$ is strictly increasing, contradicting Noetherian. So $I$ is finitely generated.

**Hilbert Basis Theorem.** If $R$ is Noetherian, then $R[x]$ is Noetherian. By induction, $R[x_1, \ldots, x_n]$ is Noetherian for $R$ Noetherian. Every ideal in $\mathbb{Z}[x_1, \ldots, x_n]$ or $k[x_1, \ldots, x_n]$ is finitely generated --- not obvious for many variables. Hilbert's original proof was existential; Gordan reportedly said "this is not mathematics, this is theology."

The proof of HBT proceeds by contradiction: assume $I \trianglelefteq R[x]$ is not finitely generated. Pick $f_1$ of minimal degree in $I$, then $f_2$ of minimal degree in $I \setminus (f_1)$, etc. The leading coefficients form an ascending chain of ideals in $R$, which (by Noetherian) stabilizes. The stabilization point gives a contradiction with the construction. So $I$ is finitely generated. The argument is short but the idea is foundational.

**Modern significance.** Most rings encountered in practice are Noetherian: $\mathbb{Z}$, polynomial rings over fields, finitely generated $\mathbb{Z}$-algebras, formal power series rings, etc. Non-Noetherian examples exist (e.g., $k[x_1, x_2, \ldots]$ in countably many variables) but are rarer in the wild.

**Hierarchy:**

$$\text{Fields} \subsetneq \text{Euclidean Domains} \subsetneq \text{PIDs} \subsetneq \text{UFDs} \subsetneq \text{Integral Domains} \subsetneq \text{Commutative Rings}$$

Witnesses for strictness:

- $\mathbb{Z}$: Euclidean domain, not field.
- $\mathbb{Z}[(1+\sqrt{-19})/2]$: PID, not Euclidean.
- $\mathbb{Z}[x]$: UFD, not PID.
- $\mathbb{Z}[\sqrt{-5}]$: integral domain, not UFD ($6 = 2 \cdot 3 = (1+\sqrt{-5})(1-\sqrt{-5})$).

The UFD box will be developed in the next article on polynomial rings. The point: even within "ring with no zero divisors," there is a rich hierarchy of how well factorization behaves.

**Numerical example: divisor counts in PIDs.** In $\mathbb{Z}$, the number of divisors of $n = p_1^{a_1} \cdots p_k^{a_k}$ is $\prod(a_i + 1)$. In a general PID, this same formula holds for divisors of an element with the corresponding prime factorization. So in $\mathbb{Z}[i]$, the element $5 = (2+i)(2-i)$ has $4$ divisors up to units: $1, 2+i, 2-i, 5$. Compare with $5 \in \mathbb{Z}$ which has $2$ divisors: a different prime factorization gives a different divisor count, even though the "size" is the same.

**Numerical example: failure of unique factorization in $\mathbb{Z}[\sqrt{-5}]$.** $6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5})$. Both factorizations involve irreducibles, but the factorizations differ. The norm $N(a + b\sqrt{-5}) = a^2 + 5b^2$. Norm of $2$ is $4$, of $3$ is $9$, of $1 \pm \sqrt{-5}$ is $6$. None of these can be further decomposed nontrivially using the norm, so all four are irreducible. Yet $6$ has two essentially different factorizations. Dedekind's solution: pass to *ideals* rather than elements. The ideals $(2), (3), (1 + \sqrt{-5}), (1 - \sqrt{-5})$ are not all prime, and the unique factorization is restored by considering products of prime ideals.

**Numerical instance.** In $\mathbb{Z}[\sqrt{-5}]$: $(2) = \mathfrak{p}^2$ where $\mathfrak{p} = (2, 1 + \sqrt{-5})$, and $(3) = \mathfrak{q}_1 \mathfrak{q}_2$ where $\mathfrak{q}_1 = (3, 1 + \sqrt{-5}), \mathfrak{q}_2 = (3, 1 - \sqrt{-5})$. So $(6) = \mathfrak{p}^2 \mathfrak{q}_1 \mathfrak{q}_2$ as a product of prime ideals. Both factorizations $2 \cdot 3$ and $(1 + \sqrt{-5})(1 - \sqrt{-5})$ correspond to the same prime ideal factorization, just regrouped. This is the trick: ideals factor uniquely in any "Dedekind domain," even when elements do not.

**Why this matters.** The failure of unique factorization in $\mathbb{Z}[\sqrt{-5}]$ historically motivated Dedekind to invent ideals. The slogan: *ideals of $\mathbb{Z}[\sqrt{-5}]$ factor uniquely as products of prime ideals*. This idea generalizes massively: it underpins all of algebraic number theory, the proof of Fermat's Last Theorem, and the modern theory of $L$-functions.

A sociological point: the realization that "elements lie, but ideals tell the truth" was a major conceptual breakthrough of 19th century mathematics. Kummer originally introduced "ideal numbers" as fictitious elements that would restore unique factorization in cyclotomic rings (for the proof of Fermat's last theorem in special cases). Dedekind reformulated these as actual *subsets* (ideals) of the ring, removing the metaphysical sleight of hand. The modern definition we use is Dedekind's.

## What's Next

We have built the basic language of ring theory: rings, homomorphisms, ideals, quotient rings, integral domains, PIDs, and the Noetherian property. In the next article we focus on *polynomial rings* $R[x]$: the division algorithm, irreducibility criteria, Gauss's lemma, and the theory of unique factorization. Polynomial rings are the testing ground for everything we have developed here, and they connect ring theory directly to the classical problems of solving equations and understanding algebraic numbers.

A summary worth keeping: groups have one operation, rings have two; ideals are the kernel notion for rings; quotient rings let us construct new rings with prescribed relations; PIDs are the cleanest setting where ideals are generated by single elements.

A second summary, this time historical. The conceptual leap "from numbers to ideals" reframes classical questions about divisibility into questions about subsets of rings. The leap was made independently by Kummer (with his "ideal numbers"), Dedekind (with the modern set-theoretic definition), and Kronecker (with a constructive but more cumbersome formulation). Dedekind's version is what we use today. The whole subsequent history of algebraic number theory and algebraic geometry is the unfolding of this conceptual move.

**One more thought to close on.** Ring theory is a sprawling subject, and this article has only set up the basic vocabulary. The richness comes when one starts to combine the constructions: localization (inverting a multiplicative subset), tensor products (gluing rings together), modules (vector-space-like objects over rings), and homological methods (resolving modules to extract finer invariants). Each of these is the subject of an entire course. But none of them makes sense without the foundation we have built here: rings, ideals, quotient rings, integral domains.**Reading recommendations.** Atiyah and Macdonald's *Introduction to Commutative Algebra* is the classic short reference. Eisenbud's *Commutative Algebra with a View Toward Algebraic Geometry* is the modern comprehensive text. Dummit and Foote chapters 7-9 cover ring theory in depth.

**A reflection on the conceptual jump from groups to rings.** Group theory is single-operation algebra: every theorem about groups is a statement about composition. Ring theory is double-operation algebra: theorems involve the *interaction* of addition and multiplication. The interaction is encoded in distributivity, but its consequences are surprisingly rich. Number theory, polynomial algebra, algebraic geometry, and most of modern algebra live in this two-operation setting.

A second reflection: ideals are simultaneously the right substitute for normal subgroups (when modding out) and the right substitute for elements (when factoring). The "factor uniquely as primes" property of $\mathbb{Z}$ generalizes to many rings only at the level of ideals, not elements. This is why algebraic number theorists work with ideals more than with numbers: they are the units of structural arithmetic.

---

*This is Part 5 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 4 — Sylow Theorems](/en/abstract-algebra/04-sylow-theorems/)*

*Next: [Part 6 — Polynomial Rings](/en/abstract-algebra/06-polynomial-rings/)*
