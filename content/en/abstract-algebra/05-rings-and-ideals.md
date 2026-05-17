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

Groups capture symmetry through a single operation. But most of the number systems we actually compute with — integers, polynomials, matrices — carry two operations that interact: addition and multiplication. The moment you want to talk about divisibility, factorization, or solving equations, one operation is not enough. You need a **ring**.

This article develops ring theory from scratch: the axioms, the key examples, the pathologies that make ring theory richer (and harder) than group theory, and the central concept of an **ideal** — the ring-theoretic analogue of a normal subgroup. By the end, you will have the language to state the First Isomorphism Theorem for rings and to understand why "modding out by an ideal" is the right way to build new rings from old ones.

---

## From Groups to Rings: Why Two Operations?

Consider the integers $\mathbb{Z}$. As a group under addition, $(\mathbb{Z}, +)$ is infinite cyclic — completely understood. But the interesting number theory of $\mathbb{Z}$ involves multiplication: primes, divisibility, the Fundamental Theorem of Arithmetic. Addition alone cannot see any of this structure.

Similarly, consider the set $\mathbb{R}[x]$ of polynomials with real coefficients. As an additive group it is just a vector space, but the ability to *multiply* polynomials is what makes factorization, roots, and algebraic geometry possible.

The pattern repeats everywhere:

- **$\mathbb{Z}/n\mathbb{Z}$**: modular arithmetic uses both addition and multiplication mod $n$.
- **$M_n(\mathbb{R})$**: matrix algebra needs both operations; multiplication is not even commutative.
- **Function spaces**: pointwise addition and multiplication of functions $f: X \to \mathbb{R}$.

In each case, addition gives you an abelian group, multiplication gives you an associative operation, and the two are linked by **distributivity**. Abstracting this pattern yields the notion of a ring.

Historically, the concept of a ring crystallized in the late 19th century from two sources. The first was **algebraic number theory**: Dedekind studied rings of algebraic integers like $\mathbb{Z}[\sqrt{-5}]$ to understand when unique factorization fails and how to restore it using ideals. The second was **invariant theory**: Hilbert proved that rings of polynomial invariants are finitely generated, a result that only makes sense once you have a clear notion of "ring" and "ideal." The formal axiomatization was completed by Emmy Noether in the 1920s, and her framework — rings, ideals, modules, the ascending chain condition — remains the language of modern algebra.

---


![Hierarchy of ring structures from general rings to fields](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-rings-and-ideals/aa_fig5_ring_hierarchy.png)

## Ring Axioms and the Zoo of Examples

**Definition.** A **ring** is a set $R$ equipped with two binary operations $+$ and $\cdot$ such that:

1. $(R, +)$ is an abelian group (with identity $0$).
2. Multiplication is associative: $a \cdot (b \cdot c) = (a \cdot b) \cdot c$ for all $a, b, c \in R$.
3. Distributivity holds on both sides:
$$a \cdot (b + c) = a \cdot b + a \cdot c, \qquad (a + b) \cdot c = a \cdot c + b \cdot c.$$

We say $R$ is a **ring with unity** (or **unital ring**) if there exists $1 \in R$ with $1 \cdot a = a \cdot 1 = a$ for all $a$. We say $R$ is **commutative** if $a \cdot b = b \cdot a$ for all $a, b$. Throughout this article, "ring" means unital ring unless stated otherwise.

**Immediate consequences of the axioms.** For any ring $R$:
- $0 \cdot a = a \cdot 0 = 0$ for all $a \in R$. *Proof:* $0 \cdot a = (0 + 0) \cdot a = 0 \cdot a + 0 \cdot a$, so $0 \cdot a = 0$ by cancellation in $(R, +)$.
- $(-1) \cdot a = -a$. *Proof:* $a + (-1) \cdot a = 1 \cdot a + (-1) \cdot a = (1 + (-1)) \cdot a = 0 \cdot a = 0$.
- $(-a)(-b) = ab$. Apply the previous result twice.

### The Zoo of Examples

| Ring | Commutative? | Unity? | Notes |
|---|---|---|---|
| $\mathbb{Z}$ | Yes | $1$ | The prototypical ring |
| $\mathbb{Z}/n\mathbb{Z}$ | Yes | $\bar{1}$ | A field iff $n$ is prime |
| $\mathbb{Q}, \mathbb{R}, \mathbb{C}$ | Yes | $1$ | Fields (every nonzero element invertible) |
| $\mathbb{Z}[i] = \{a + bi : a, b \in \mathbb{Z}\}$ | Yes | $1$ | Gaussian integers |
| $\mathbb{R}[x]$ | Yes | $1$ | Polynomial ring |
| $M_n(\mathbb{R})$ | **No** ($n \geq 2$) | $I_n$ | Matrix ring |
| $\mathbb{H}$ (quaternions) | **No** | $1$ | Division ring (skew field) |
| $2\mathbb{Z}$ (even integers) | Yes | **No** | Ring without unity |
| $C(X, \mathbb{R})$ (continuous functions $X \to \mathbb{R}$) | Yes | $f \equiv 1$ | Infinite-dimensional |

A **field** is a commutative ring with unity in which every nonzero element has a multiplicative inverse. A **division ring** (or skew field) drops commutativity but keeps inverses. The quaternions $\mathbb{H}$ are the standard example: $i^2 = j^2 = k^2 = ijk = -1$, and $ij = k \neq -k = ji$.

**Units and the group of units.** An element $u \in R$ is a **unit** if it has a two-sided multiplicative inverse: there exists $v \in R$ with $uv = vu = 1$. The set of all units $R^{\times}$ forms a group under multiplication (the **group of units** of $R$). For example:
- $\mathbb{Z}^{\times} = \{1, -1\}$.
- $(\mathbb{Z}/n\mathbb{Z})^{\times} = \{\bar{a} : \gcd(a, n) = 1\}$, a group of order $\varphi(n)$.
- $(M_n(\mathbb{R}))^{\times} = GL_n(\mathbb{R})$, the general linear group.
- For a field $F$, $F^{\times} = F \setminus \{0\}$.

The notion of units is essential for discussing factorization: two elements $a, b$ that differ by a unit ($a = ub$ for some unit $u$) are called **associates**, and they should be considered "the same" for factorization purposes — just as $3$ and $-3$ are associates in $\mathbb{Z}$.

**Example: The ring $\mathbb{Z}/6\mathbb{Z}$.** This commutative ring with unity has six elements $\{\bar{0}, \bar{1}, \bar{2}, \bar{3}, \bar{4}, \bar{5}\}$. Its group of units is $\{\bar{1}, \bar{5}\}$ (the elements coprime to $6$). Observe that $\bar{2} \cdot \bar{3} = \bar{6} = \bar{0}$, even though neither $\bar{2}$ nor $\bar{3}$ is zero. This phenomenon — a product of nonzero elements being zero — does not happen in $\mathbb{Z}$ or in any field. It leads to our next concept.

**Subrings.** A subset $S \subseteq R$ is a **subring** if it is itself a ring under the same operations and contains $1_R$. For instance, $\mathbb{Z}$ is a subring of $\mathbb{Q}$, which is a subring of $\mathbb{R}$. The Gaussian integers $\mathbb{Z}[i]$ are a subring of $\mathbb{C}$. The **subring test**: a nonempty subset $S \subseteq R$ is a subring iff $1 \in S$ and $S$ is closed under subtraction and multiplication.

---

## Integral Domains and Zero Divisors

**Definition.** Let $R$ be a ring. A nonzero element $a \in R$ is a **zero divisor** if there exists a nonzero $b \in R$ with $ab = 0$ (or $ba = 0$).

**Definition.** A commutative ring with unity $R$ is an **integral domain** if it has no zero divisors: whenever $ab = 0$, either $a = 0$ or $b = 0$.

**Examples.**
- $\mathbb{Z}$ is an integral domain (this is essentially the content of Euclid's lemma).
- Every field is an integral domain: if $ab = 0$ and $a \neq 0$, multiply both sides by $a^{-1}$ to get $b = 0$.
- $\mathbb{Z}/p\mathbb{Z}$ for $p$ prime is a field, hence an integral domain.
- $\mathbb{Z}/6\mathbb{Z}$ is **not** an integral domain: $\bar{2} \cdot \bar{3} = \bar{0}$.
- $M_2(\mathbb{R})$ is not an integral domain (not commutative, and has zero divisors:
$\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$).

**Cancellation law.** In an integral domain, $ab = ac$ with $a \neq 0$ implies $b = c$. *Proof:* $a(b - c) = 0$ and $a \neq 0$, so $b - c = 0$.

This is exactly the property that makes integral domains behave like "generalized integers" — you can cancel common factors, which is the starting point for any theory of factorization.

**The field of fractions.** Just as we build $\mathbb{Q}$ from $\mathbb{Z}$ by "allowing division," any integral domain $R$ embeds into its **field of fractions** $\text{Frac}(R)$. The construction mirrors the one for $\mathbb{Q}$: elements are equivalence classes of pairs $(a, b)$ with $b \neq 0$, where $(a, b) \sim (c, d)$ iff $ad = bc$. Addition and multiplication are defined as $\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}$ and $\frac{a}{b} \cdot \frac{c}{d} = \frac{ac}{bd}$. The fact that $R$ is an integral domain is essential: without it, $bd$ could be zero and the construction collapses.

Examples: $\text{Frac}(\mathbb{Z}) = \mathbb{Q}$, $\text{Frac}(\mathbb{Z}[i]) = \mathbb{Q}(i) = \{a + bi : a, b \in \mathbb{Q}\}$, $\text{Frac}(F[x]) = F(x)$ (the field of rational functions). The field of fractions is the *smallest* field containing $R$, in the sense that any injective homomorphism from $R$ into a field factors through $\text{Frac}(R)$.

**Proposition.** Every finite integral domain is a field.

*Proof sketch.* Let $R$ be a finite integral domain and let $a \in R$ be nonzero. Consider the map $\varphi_a: R \to R$ defined by $\varphi_a(r) = ar$. This is injective: if $ar = as$, then $a(r - s) = 0$, so $r = s$ since $R$ is an integral domain. A finite set with an injective map to itself is surjective, so there exists $r$ with $ar = 1$. Thus $a$ is a unit. $\square$

**Worked Example 1.** *Show that $\mathbb{Z}/n\mathbb{Z}$ is an integral domain if and only if $n$ is prime.*

*Solution.* If $n = ab$ with $1 < a, b < n$, then $\bar{a} \neq \bar{0}$ and $\bar{b} \neq \bar{0}$ in $\mathbb{Z}/n\mathbb{Z}$, but $\bar{a} \cdot \bar{b} = \overline{ab} = \bar{n} = \bar{0}$. So $\mathbb{Z}/n\mathbb{Z}$ has zero divisors and is not an integral domain. Conversely, if $n = p$ is prime and $\bar{a} \cdot \bar{b} = \bar{0}$ in $\mathbb{Z}/p\mathbb{Z}$, then $p \mid ab$. By Euclid's lemma, $p \mid a$ or $p \mid b$, i.e., $\bar{a} = \bar{0}$ or $\bar{b} = \bar{0}$. So $\mathbb{Z}/p\mathbb{Z}$ is an integral domain. Since it is finite, the proposition above tells us it is actually a field. $\square$

**Worked Example 2.** *Show that the Gaussian integers $\mathbb{Z}[i]$ form an integral domain.*

*Solution.* Define the **norm** $N(a + bi) = a^2 + b^2$. This is a non-negative integer, and $N(\alpha) = 0$ iff $\alpha = 0$. The key property is multiplicativity: $N(\alpha \beta) = N(\alpha) N(\beta)$. (This follows from $|zw| = |z||w|$ for complex numbers, or by direct computation.) Now if $\alpha \beta = 0$, then $N(\alpha) N(\beta) = N(\alpha \beta) = N(0) = 0$. Since $N(\alpha), N(\beta) \in \mathbb{Z}_{\geq 0}$, either $N(\alpha) = 0$ or $N(\beta) = 0$, hence $\alpha = 0$ or $\beta = 0$. $\square$

---

## Ideals: The Right Notion of "Kernel" for Rings

In group theory, normal subgroups are exactly the kernels of homomorphisms, and they are exactly the subgroups you can quotient by. What is the ring-theoretic analogue?

**Definition.** A **ring homomorphism** is a map $\varphi: R \to S$ between rings satisfying:
1. $\varphi(a + b) = \varphi(a) + \varphi(b)$,
2. $\varphi(ab) = \varphi(a)\varphi(b)$,
3. $\varphi(1_R) = 1_S$.

The **kernel** is $\ker \varphi = \{r \in R : \varphi(r) = 0\}$.

What properties does $\ker \varphi$ have? Certainly it is an additive subgroup of $R$ (since $\varphi$ is a group homomorphism on $(R, +)$). But it has an extra *absorption* property: if $a \in \ker \varphi$ and $r \in R$, then

$$\varphi(ra) = \varphi(r)\varphi(a) = \varphi(r) \cdot 0 = 0,$$

so $ra \in \ker \varphi$. Similarly, $ar \in \ker \varphi$.

**Definition.** An **ideal** of a ring $R$ is a subset $I \subseteq R$ such that:
1. $(I, +)$ is a subgroup of $(R, +)$.
2. For all $r \in R$ and $a \in I$: $ra \in I$ and $ar \in I$.

Condition (2) is called **absorption** — the ideal "absorbs" multiplication by arbitrary ring elements. Note the asymmetry: we require both $ra \in I$ and $ar \in I$ because $R$ may not be commutative. (In a commutative ring, $ra = ar$, so one condition suffices.)

If only $ra \in I$ is required, $I$ is a **left ideal**; if only $ar \in I$, a **right ideal**; if both, a **two-sided ideal** (or simply an ideal).

### Principal Ideals

The simplest ideals are generated by a single element. In a commutative ring $R$, the **principal ideal generated by $a$** is

$$(a) = aR = \{ar : r \in R\}.$$

**Example.** In $\mathbb{Z}$, every ideal is principal: the ideal generated by $n$ is $n\mathbb{Z} = \{\ldots, -2n, -n, 0, n, 2n, \ldots\}$. This is a theorem, not a definition — it says $\mathbb{Z}$ is a **principal ideal domain** (PID).

*Proof that every ideal of $\mathbb{Z}$ is principal.* Let $I$ be an ideal of $\mathbb{Z}$. If $I = \{0\}$, then $I = (0)$. Otherwise, $I$ contains some nonzero element, hence some positive element (if $a \in I$, then $-a \in I$). Let $d$ be the smallest positive element of $I$. We claim $I = (d) = d\mathbb{Z}$. Certainly $d\mathbb{Z} \subseteq I$ by absorption. Conversely, for any $a \in I$, write $a = qd + r$ with $0 \leq r < d$ by the division algorithm. Then $r = a - qd \in I$ (since $I$ is closed under subtraction and absorption). By minimality of $d$, we must have $r = 0$, so $a \in d\mathbb{Z}$. $\square$

**Example.** In $\mathbb{R}[x]$, the ideal $(x^2 + 1)$ consists of all polynomials divisible by $x^2 + 1$. The quotient $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$ — this is a clean algebraic construction of the complex numbers.

### Maximal and Prime Ideals

Not all ideals are created equal. Two classes play starring roles:

**Definition.** A proper ideal $\mathfrak{m} \subsetneq R$ is **maximal** if there is no ideal $I$ with $\mathfrak{m} \subsetneq I \subsetneq R$.

**Definition.** A proper ideal $\mathfrak{p} \subsetneq R$ is **prime** if whenever $ab \in \mathfrak{p}$, either $a \in \mathfrak{p}$ or $b \in \mathfrak{p}$.

**Theorem.** In a commutative ring with unity:
- $R/\mathfrak{m}$ is a field if and only if $\mathfrak{m}$ is maximal.
- $R/\mathfrak{p}$ is an integral domain if and only if $\mathfrak{p}$ is prime.
- Every maximal ideal is prime (since every field is an integral domain).

*Proof sketch (maximal $\Leftrightarrow$ field).* If $\mathfrak{m}$ is maximal and $\bar{a} \neq \bar{0}$ in $R/\mathfrak{m}$, then $a \notin \mathfrak{m}$, so $\mathfrak{m} + (a) = R$ by maximality. Thus $1 = m + ra$ for some $m \in \mathfrak{m}$, $r \in R$, giving $\bar{r}\bar{a} = \bar{1}$. Conversely, if $R/\mathfrak{m}$ is a field and $\mathfrak{m} \subsetneq I$, pick $a \in I \setminus \mathfrak{m}$. Then $\bar{a}$ is a unit in $R/\mathfrak{m}$, so $ra - 1 \in \mathfrak{m} \subseteq I$ for some $r$. Hence $1 = ra - (ra - 1) \in I$, so $I = R$. $\square$

**Example.** In $\mathbb{Z}$, the prime ideals are $(0)$ and $(p)$ for primes $p$. The maximal ideals are exactly the $(p)$. Note $(0)$ is prime but not maximal (since $\mathbb{Z}/(0) \cong \mathbb{Z}$ is an integral domain but not a field).

**Worked Example (bonus).** *In $\mathbb{Z}[x]$, show that $(x)$ is prime but not maximal.*

*Solution.* The quotient $\mathbb{Z}[x]/(x) \cong \mathbb{Z}$ (the map $f(x) \mapsto f(0)$ has kernel $(x)$ and image $\mathbb{Z}$). Since $\mathbb{Z}$ is an integral domain, $(x)$ is prime. Since $\mathbb{Z}$ is not a field, $(x)$ is not maximal. Indeed, $(x) \subsetneq (x, 2) \subsetneq \mathbb{Z}[x]$, so $(x, 2)$ is a strictly larger ideal. And $\mathbb{Z}[x]/(x, 2) \cong \mathbb{Z}/2\mathbb{Z} = \mathbb{F}_2$, which is a field, so $(x, 2)$ is maximal. $\square$

**Operations on ideals.** Given ideals $I, J$ of $R$, we can form:
- **Sum:** $I + J = \{a + b : a \in I, b \in J\}$, the smallest ideal containing both.
- **Product:** $IJ = \{\sum_{k=1}^n a_k b_k : a_k \in I, b_k \in J\}$, not just pairwise products but finite sums of them.
- **Intersection:** $I \cap J$, which is always an ideal.

These operations obey many identities reminiscent of arithmetic: $I(J + K) = IJ + IK$, and $I \cap (J + K) \supseteq I \cap J + I \cap K$ (though equality may fail in general). When $I + J = R$, we say $I$ and $J$ are **comaximal**, and the Chinese Remainder Theorem gives $R/(I \cap J) \cong R/I \times R/J$.

---

## Quotient Rings and the First Isomorphism Theorem

Given a ring $R$ and a two-sided ideal $I$, we build the **quotient ring** $R/I$ exactly as we built quotient groups: the elements are cosets $a + I$, and the operations are

$$(a + I) + (b + I) = (a + b) + I, \qquad (a + I)(b + I) = ab + I.$$

The key check is **well-definedness** of multiplication. If $a + I = a' + I$ and $b + I = b' + I$, then $a' = a + i$ and $b' = b + j$ for some $i, j \in I$. So

$$a'b' = (a + i)(b + j) = ab + aj + ib + ij.$$

We need $a'b' - ab = aj + ib + ij \in I$. This holds because $I$ is an ideal: $aj \in I$ (absorption by $a$), $ib \in I$ (absorption by $b$), and $ij \in I$ (product of two elements of $I$). This is precisely why we need ideals — not just any additive subgroup will do.

**First Isomorphism Theorem for Rings.** If $\varphi: R \to S$ is a ring homomorphism, then $\ker \varphi$ is an ideal of $R$, $\operatorname{im} \varphi$ is a subring of $S$, and

$$R / \ker \varphi \cong \operatorname{im} \varphi.$$

The proof is essentially the same as for groups: define $\bar{\varphi}(a + \ker \varphi) = \varphi(a)$, check well-definedness, and verify it is a ring isomorphism.

**Worked Example 3.** *Show that $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$.*

*Solution.* Define $\varphi: \mathbb{R}[x] \to \mathbb{C}$ by $\varphi(f) = f(i)$ (evaluation at $i$). This is a ring homomorphism (evaluation maps always are). It is surjective: for any $a + bi \in \mathbb{C}$, the polynomial $a + bx$ maps to $a + bi$. The kernel consists of all polynomials vanishing at $i$. Since $x^2 + 1$ is the minimal polynomial of $i$ over $\mathbb{R}$, we have $\ker \varphi = (x^2 + 1)$. By the First Isomorphism Theorem, $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$. $\square$

This example shows the power of quotient rings: **we can construct new number systems purely algebraically**, without appealing to geometric intuition or "inventing" $\sqrt{-1}$. The quotient ring $\mathbb{R}[x]/(x^2 + 1)$ consists of cosets of the form $a + bx + (x^2 + 1)$, which we can identify with $a + b\bar{x}$ where $\bar{x}^2 = -1$. In other words, $\bar{x}$ plays the role of $i$, and the ring structure forces $\bar{x}^2 + 1 = 0$. We did not "assume" that $\sqrt{-1}$ exists — we *constructed* it by algebraic quotient.

This construction generalizes: for any irreducible polynomial $f(x) \in F[x]$ over a field $F$, the quotient $F[x]/(f(x))$ is a field extension of $F$ containing a root of $f$. This is the algebraic mechanism behind building splitting fields and algebraic closures — topics for a later article.

**Worked Example 4.** *Describe the quotient ring $\mathbb{Z}[x]/(x^2 + 1, 5)$.*

*Solution.* We have $\mathbb{Z}[x]/(x^2 + 1, 5) \cong (\mathbb{Z}/5\mathbb{Z})[x]/(x^2 + 1)$. In $\mathbb{F}_5 = \mathbb{Z}/5\mathbb{Z}$, does $x^2 + 1$ have a root? We check: $0^2 + 1 = 1$, $1^2 + 1 = 2$, $2^2 + 1 = 0$. So $x^2 + 1 = (x - 2)(x - 3)$ in $\mathbb{F}_5[x]$. By the Chinese Remainder Theorem for rings, $\mathbb{F}_5[x]/(x^2 + 1) \cong \mathbb{F}_5[x]/(x - 2) \times \mathbb{F}_5[x]/(x - 3) \cong \mathbb{F}_5 \times \mathbb{F}_5$. So $\mathbb{Z}[x]/(x^2 + 1, 5) \cong \mathbb{F}_5 \times \mathbb{F}_5$, a product of two fields. $\square$

---

## PIDs and the Ascending Chain Condition

We saw that $\mathbb{Z}$ is a principal ideal domain: every ideal is generated by a single element. This turns out to be a remarkably strong condition.

**Definition.** A commutative ring with unity is a **principal ideal domain (PID)** if it is an integral domain and every ideal is principal.

**Examples of PIDs:**
- $\mathbb{Z}$ (proved above).
- $F[x]$ for any field $F$ (the proof uses the division algorithm for polynomials — more on this in the next article).
- $\mathbb{Z}[i]$ (the Gaussian integers — the proof uses the norm $N(a + bi) = a^2 + b^2$ and a Euclidean division).

**Non-examples:**
- $\mathbb{Z}[x]$ is **not** a PID. The ideal $(2, x) = \{f \in \mathbb{Z}[x] : f(0) \text{ is even}\}$ is not principal. If $(2, x) = (g)$, then $g \mid 2$ and $g \mid x$ in $\mathbb{Z}[x]$. From $g \mid 2$, $g$ is a constant ($\pm 1$ or $\pm 2$). From $g \mid x$, if $g = \pm 2$, then $2 \mid x$ in $\mathbb{Z}[x]$, impossible. So $g = \pm 1$ and $(g) = \mathbb{Z}[x]$, but $(2, x) \neq \mathbb{Z}[x]$ since $1 \notin (2, x)$ (as $1$ is odd and not divisible by $x$). Contradiction.

### The Ascending Chain Condition

A ring $R$ satisfies the **ascending chain condition (ACC) on ideals** if every ascending chain of ideals

$$I_1 \subseteq I_2 \subseteq I_3 \subseteq \cdots$$

eventually stabilizes: there exists $N$ such that $I_n = I_N$ for all $n \geq N$. A ring satisfying the ACC on ideals is called **Noetherian** (after Emmy Noether).

**Proposition.** Every PID is Noetherian.

*Proof.* Let $I_1 \subseteq I_2 \subseteq \cdots$ be an ascending chain of ideals in a PID $R$. Set $I = \bigcup_{n=1}^{\infty} I_n$. One checks that $I$ is an ideal of $R$. Since $R$ is a PID, $I = (d)$ for some $d$. Since $d \in I = \bigcup I_n$, we have $d \in I_N$ for some $N$. Then $(d) \subseteq I_N \subseteq I \subseteq (d)$, so $I_n = (d)$ for all $n \geq N$. $\square$

The Noetherian condition is one of the most important finiteness conditions in algebra. In a Noetherian ring, you cannot build an infinite strictly ascending chain of ideals — every "construction process" that generates bigger and bigger ideals must eventually stop. This is the engine behind many existence proofs in commutative algebra: the Hilbert Basis Theorem, primary decomposition, and the theory of Noetherian modules all rest on the ACC.

**Hilbert Basis Theorem.** If $R$ is Noetherian, then $R[x]$ is Noetherian. This is one of the most consequential theorems in algebra. It implies that every ideal of $\mathbb{Z}[x_1, \ldots, x_n]$ or $k[x_1, \ldots, x_n]$ (for a field $k$) is finitely generated — a fact that is by no means obvious a priori for ideals in a polynomial ring with many variables. The original proof by Hilbert was existential (it showed the generators *exist* without constructing them), which was so controversial at the time that Gordan reportedly said: "This is not mathematics; this is theology." Today the Hilbert Basis Theorem is central to both commutative algebra and algebraic geometry (it guarantees that every algebraic variety is defined by finitely many equations).

**The connection to group theory.** It is worth noting how much richer the ideal structure of rings is compared to the subgroup structure of groups. In $\mathbb{Z}$ (viewed as a ring), the ideals are exactly $n\mathbb{Z}$ — they form a chain indexed by divisibility. But in a ring like $k[x, y]$, ideals can have complicated geometric shapes (corresponding to curves, points, unions of subvarieties). The jump from one-dimensional to two-dimensional ideal theory is enormous, and it is what makes commutative algebra a deep and active field of research.

**The hierarchy so far:**

$$\text{Fields} \subsetneq \text{Euclidean Domains} \subsetneq \text{PIDs} \subsetneq \text{UFDs} \subsetneq \text{Integral Domains} \subsetneq \text{Commutative Rings}$$

We will flesh out the UFD (Unique Factorization Domain) box in the next article. The strict inclusions are witnessed by:
- $\mathbb{Z}$ is a Euclidean domain (hence PID, hence UFD) but not a field.
- $\mathbb{Z}\left[\frac{1+\sqrt{-19}}{2}\right]$ is a PID but not a Euclidean domain.
- $\mathbb{Z}[x]$ is a UFD but not a PID.
- $\mathbb{Z}[\sqrt{-5}]$ is an integral domain but not a UFD (since $6 = 2 \cdot 3 = (1+\sqrt{-5})(1-\sqrt{-5})$).

---

## What's Next

We have built the basic language of ring theory: rings, homomorphisms, ideals, quotient rings, integral domains, PIDs, and the Noetherian property. In the next article, we focus on **polynomial rings** $R[x]$: the division algorithm, irreducibility criteria, Gauss's lemma, and the theory of unique factorization. Polynomial rings are the testing ground for everything we have developed here, and they connect ring theory directly to the classical problems of solving equations and understanding algebraic numbers.

---

*Abstract Algebra Series — Article 5 of 12*

---

*This is Part 5 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 4 — Sylow Theorems](/en/abstract-algebra/04-sylow-theorems/)*

*Next: [Part 6 — Polynomial Rings](/en/abstract-algebra/06-polynomial-rings/)*
