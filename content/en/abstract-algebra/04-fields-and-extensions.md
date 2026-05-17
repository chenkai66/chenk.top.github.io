---
title: "Fields and Field Extensions"
date: 2021-01-31 09:00:00
tags:
  - abstract-algebra
  - field-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 4
series_total: 6
translationKey: "abstract-algebra-4"
description: "Field extensions arise from adjoining roots. Degree, algebraic elements, and splitting fields set the stage for Galois theory."
---

## Fields: rings where everything divides

A **field** is a commutative ring with unity in which every nonzero element has a multiplicative inverse. Equivalently, a field has exactly two ideals: $(0)$ and the whole ring. If you can divide by anything nonzero, there is no room for interesting ideal structure — the ring is as "simple" as possible from the ideal-theoretic perspective.

$$
F \text{ is a field} \iff \forall\, a \neq 0,\; \exists\, a^{-1} : aa^{-1} = 1
$$

The standard examples: $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$, and the finite fields $\mathbb{F}_p = \mathbb{Z}/p\mathbb{Z}$ for prime $p$. But the story gets interesting when you ask: what fields sit *between* $\mathbb{Q}$ and $\mathbb{C}$? That question leads to field extensions, which are the setup for Galois theory in [Part 5](/en/abstract-algebra/05-galois-theory/).

I find it useful to think of fields as the "atoms" of ring theory. You cannot quotient a field further (no nontrivial ideals), and every integral domain embeds in a field (its field of fractions, as we saw in [Part 3](/en/abstract-algebra/03-rings-and-ideals/)). Fields are where the ring-theoretic story ends and the field-theoretic story begins.

**The characteristic.** Every field has a **characteristic**: the smallest positive integer $p$ such that $\underbrace{1 + 1 + \cdots + 1}_p = 0$, or 0 if no such $p$ exists. If positive, the characteristic must be prime (if $\text{char}(F) = mn$, then $0 = \underbrace{1+\cdots+1}_{mn}$, and since $F$ has no zero divisors, either the sum of $m$ ones or the sum of $n$ ones is zero). We have $\text{char}(\mathbb{Q}) = \text{char}(\mathbb{R}) = 0$ and $\text{char}(\mathbb{F}_p) = p$.

## Field extensions and degree

A **field extension** $F \subseteq K$ (written $K/F$) is simply a pair of fields where $F$ is a subfield of $K$. The key insight: we can view $K$ as a **vector space** over $F$. The dimension $[K:F] = \dim_F K$ is the **degree** of the extension.

$$
[K:F] = \dim_F K
$$

If $[K:F]$ is finite, the extension is **finite**; otherwise **infinite** (for instance, $[\mathbb{R}:\mathbb{Q}]$ is uncountably infinite).

**Tower law.** If $F \subseteq L \subseteq K$, then $[K:F] = [K:L] \cdot [L:F]$.

*Proof.* If $\{v_1, \ldots, v_m\}$ is a basis for $L/F$ and $\{w_1, \ldots, w_n\}$ is a basis for $K/L$, then $\{v_i w_j : 1 \leq i \leq m, 1 \leq j \leq n\}$ is a basis for $K/F$, giving $mn$ elements. (Spanning: any $\alpha \in K$ is $\sum_j c_j w_j$ with $c_j \in L$, and each $c_j = \sum_i a_{ij} v_i$ with $a_{ij} \in F$. Linear independence: if $\sum_{i,j} a_{ij} v_i w_j = 0$, group by $j$ to get $\sum_j (\sum_i a_{ij} v_i) w_j = 0$; by independence of $\{w_j\}$ over $L$, each inner sum is zero; by independence of $\{v_i\}$ over $F$, each $a_{ij} = 0$.) $\square$

The tower law is surprisingly powerful. It immediately tells you: if $[K:F]$ is prime, there are no intermediate fields between $F$ and $K$.

![Field extension tower](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/04-fields-and-extensions/fig_concept.png)

### Example 1: $\mathbb{Q}(\sqrt{2})/\mathbb{Q}$

The field $\mathbb{Q}(\sqrt{2}) = \{a + b\sqrt{2} : a, b \in \mathbb{Q}\}$ is a degree-2 extension of $\mathbb{Q}$. A basis as a $\mathbb{Q}$-vector space is $\{1, \sqrt{2}\}$.

To verify it is a field: given $a + b\sqrt{2} \neq 0$, its inverse is $\frac{a - b\sqrt{2}}{a^2 - 2b^2}$. The denominator $a^2 - 2b^2 \neq 0$ because $\sqrt{2}$ is irrational — if $a^2 = 2b^2$ with $a, b \in \mathbb{Q}$, we would have $\sqrt{2} = a/b \in \mathbb{Q}$, contradiction.

### Example 2: A degree-4 extension

$\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$ has degree 4. A basis is $\{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$. The tower: $[\mathbb{Q}(\sqrt{2}):\mathbb{Q}] = 2$ and $[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}(\sqrt{2})] = 2$ (because $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$; if it were, $\sqrt{3} = a + b\sqrt{2}$ implies $3 = a^2 + 2b^2 + 2ab\sqrt{2}$, forcing $ab = 0$, then either $a^2 = 3$ or $2b^2 = 3$, both impossible in $\mathbb{Q}$). So degree = $2 \times 2 = 4$.

## Algebraic and transcendental elements

An element $\alpha \in K$ is **algebraic** over $F$ if it satisfies some nonzero polynomial with coefficients in $F$. Otherwise, $\alpha$ is **transcendental** over $F$.

For algebraic $\alpha$, the **minimal polynomial** $m_\alpha(x) \in F[x]$ is the unique monic irreducible polynomial with $m_\alpha(\alpha) = 0$. It exists because the set of polynomials vanishing at $\alpha$ forms a nonzero ideal in the PID $F[x]$, hence is generated by a single (necessarily irreducible) element.

**Theorem.** If $\alpha$ is algebraic over $F$ with minimal polynomial of degree $n$, then:

$$
F(\alpha) \cong F[x]/(m_\alpha(x)), \quad [F(\alpha):F] = n
$$

A basis for $F(\alpha)$ over $F$ is $\{1, \alpha, \alpha^2, \ldots, \alpha^{n-1}\}$.

*Proof.* The evaluation map $\text{ev}_\alpha: F[x] \to K$ sending $f(x) \mapsto f(\alpha)$ is a ring homomorphism with kernel $(m_\alpha)$. Since $m_\alpha$ is irreducible, $(m_\alpha)$ is maximal in the PID $F[x]$, so $F[x]/(m_\alpha) \cong \text{im}(\text{ev}_\alpha) = F(\alpha)$ is a field. The images of $1, x, \ldots, x^{n-1}$ form a basis. $\square$

**Non-example.** $\pi$ is transcendental over $\mathbb{Q}$ (Lindemann, 1882). So $\mathbb{Q}(\pi) \cong \mathbb{Q}(x)$ — the field of rational functions — and $[\mathbb{Q}(\pi):\mathbb{Q}]$ is infinite.

### Example 3: $\mathbb{Q}(\sqrt[3]{2})$

The minimal polynomial of $\sqrt[3]{2}$ over $\mathbb{Q}$ is $x^3 - 2$ (irreducible by Eisenstein at $p=2$). So $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$ and a basis is $\{1, \sqrt[3]{2}, \sqrt[3]{4}\}$.

Multiplication in this field: set $\beta = \sqrt[3]{2}$. Then $(a + b\beta + c\beta^2)(d + e\beta + f\beta^2)$ expands and reduces using $\beta^3 = 2$. For instance $\beta^2 \cdot \beta^2 = \beta^4 = \beta \cdot \beta^3 = 2\beta$.

## Splitting fields

A polynomial $f \in F[x]$ **splits** over $K$ if it factors into linear factors in $K[x]$. The **splitting field** of $f$ over $F$ is the smallest extension of $F$ over which $f$ splits completely.

$$
\text{Split}_F(f) = F(\alpha_1, \ldots, \alpha_n) \text{ where } f(x) = c(x-\alpha_1)\cdots(x-\alpha_n)
$$

**Theorem.** The splitting field exists and is unique up to isomorphism (fixing $F$).

The existence is clear (adjoin roots one at a time). The uniqueness is the deep part: different orderings of root adjunction lead to isomorphic fields. This requires an inductive argument using the fact that if $F \cong F'$ and $p(x) \in F[x]$ corresponds to $p'(x) \in F'[x]$ under this isomorphism, then $F[x]/(p) \cong F'[x]/(p')$.

### Example 4: Splitting field of $x^3 - 2$ over $\mathbb{Q}$

The roots of $x^3 - 2$ are $\sqrt[3]{2}$, $\omega\sqrt[3]{2}$, and $\omega^2\sqrt[3]{2}$, where $\omega = e^{2\pi i/3}$ is a primitive cube root of unity satisfying $x^2 + x + 1 = 0$. The splitting field is $\mathbb{Q}(\sqrt[3]{2}, \omega)$.

Degree computation: $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$. Now $\omega$ satisfies $x^2 + x + 1$ over $\mathbb{Q}$, and this polynomial remains irreducible over $\mathbb{Q}(\sqrt[3]{2})$ (if it had a root in $\mathbb{Q}(\sqrt[3]{2})$, the tower law would give $[\mathbb{Q}(\sqrt[3]{2}, \omega):\mathbb{Q}] = 3$, but that field must contain all three cube roots of 2 and $\omega$, which is impossible in degree 3 over $\mathbb{Q}$). So $[\mathbb{Q}(\sqrt[3]{2}, \omega):\mathbb{Q}] = 6$.

This "6" equals $|S_3|$ — the Galois group of this extension is $S_3$, as we will see in [Part 5](/en/abstract-algebra/05-galois-theory/).

## Ruler-and-compass constructibility

One of the most elegant applications of field extensions is the resolution of classical Greek construction problems. A length $\alpha > 0$ is **constructible** (with ruler and compass, starting from a segment of length 1) if and only if $\alpha$ lies in a field obtainable from $\mathbb{Q}$ by a tower of degree-2 extensions.

**Theorem.** If $\alpha$ is constructible, then $[\mathbb{Q}(\alpha):\mathbb{Q}]$ is a power of 2.

*Proof sketch.* Each ruler-and-compass step (intersecting lines and circles) involves solving at most a quadratic equation. So each step extends the current field by degree at most 2. By the tower law, the total degree is a power of 2. $\square$

**Corollaries (impossibility results):**

1. **Doubling the cube** is impossible: we need $\sqrt[3]{2}$, which has $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$. Since 3 is not a power of 2, it is not constructible.

2. **Trisecting a general angle** is impossible: trisecting $60°$ requires constructing $\cos 20°$, which satisfies $8x^3 - 6x - 1 = 0$ (irreducible over $\mathbb{Q}$, so degree 3 over $\mathbb{Q}$).

3. **Squaring the circle** is impossible: we need $\sqrt{\pi}$, but $\pi$ is transcendental, so $[\mathbb{Q}(\sqrt{\pi}):\mathbb{Q}]$ is infinite.

These problems baffled mathematicians for over two thousand years. The proof of impossibility requires field theory — no amount of geometric ingenuity could have settled them.

**Non-example (what IS constructible).** Regular $n$-gons are constructible if and only if $n = 2^k p_1 p_2 \cdots p_r$ where the $p_i$ are distinct Fermat primes ($3, 5, 17, 257, 65537, \ldots$). This was proved by Gauss at age 19.

## Finite fields

**Theorem.** For every prime power $q = p^n$, there exists a unique (up to isomorphism) field with $q$ elements, denoted $\mathbb{F}_q$ or $GF(q)$. No other cardinalities of finite fields exist.

*Existence:* $\mathbb{F}_{p^n}$ is the splitting field of $x^{p^n} - x$ over $\mathbb{F}_p$. This polynomial has exactly $p^n$ distinct roots (its derivative is $p^n x^{p^n - 1} - 1 = -1$ in characteristic $p$, which is never zero, so no repeated roots).

*Uniqueness:* If $|F| = p^n$, then $F^* = F \setminus \{0\}$ is a group of order $p^n - 1$, so every element of $F^*$ satisfies $x^{p^n-1} = 1$, and every element of $F$ satisfies $x^{p^n} = x$. So $F$ consists exactly of the roots of $x^{p^n} - x$, making $F$ the splitting field of that polynomial over its prime subfield $\mathbb{F}_p$.

### Example 5: $\mathbb{F}_4$

Take $\mathbb{F}_2[x]/(x^2 + x + 1)$ (irreducible over $\mathbb{F}_2$ since $f(0) = 1$ and $f(1) = 1$). Let $\alpha$ denote the image of $x$. Then $\mathbb{F}_4 = \{0, 1, \alpha, \alpha+1\}$ with $\alpha^2 = \alpha + 1$.

The multiplicative group $\mathbb{F}_4^* = \{1, \alpha, \alpha+1\}$ is cyclic of order 3. Every finite field has a cyclic multiplicative group — this is essential for applications in coding theory and cryptography ([Part 6](/en/abstract-algebra/06-applications/)).

### Example 6: $\mathbb{F}_8$

We need a degree-3 irreducible over $\mathbb{F}_2$. The polynomial $x^3 + x + 1$ works ($f(0) = 1$, $f(1) = 1$, so no roots in $\mathbb{F}_2$). Then $\mathbb{F}_8 = \mathbb{F}_2[x]/(x^3 + x + 1)$, where $\alpha^3 = \alpha + 1$. The multiplicative group $\mathbb{F}_8^*$ is cyclic of order 7, generated by $\alpha$.

## Algebraic closure

A field $F$ is **algebraically closed** if every non-constant polynomial in $F[x]$ has a root in $F$ (equivalently, splits completely over $F$). The **algebraic closure** $\overline{F}$ is the smallest algebraically closed field containing $F$.

$\mathbb{C}$ is algebraically closed (the Fundamental Theorem of Algebra — a theorem whose natural proof uses topology or complex analysis, not algebra). $\overline{\mathbb{Q}}$ — the algebraic numbers — is a countable, algebraically closed subfield of $\mathbb{C}$.

The algebraic closure $\overline{\mathbb{F}_p}$ is the union $\bigcup_{n=1}^\infty \mathbb{F}_{p^n}$ (with suitable embeddings). It is a countably infinite field of characteristic $p$.

The existence of algebraic closures for arbitrary fields requires Zorn's lemma. Uniqueness (up to isomorphism fixing $F$) also uses choice. These are among the few points in algebra where set-theoretic foundations matter.

## Separability

A polynomial is **separable** if it has no repeated roots in its splitting field. An algebraic extension $K/F$ is **separable** if the minimal polynomial of every element of $K$ over $F$ is separable.

In characteristic 0 (like $\mathbb{Q}$, $\mathbb{R}$), every irreducible polynomial is automatically separable. Proof: if $f$ is irreducible and has a repeated root $\alpha$, then $\alpha$ is a common root of $f$ and $f'$, so $\gcd(f, f') \neq 1$. But $\deg f' < \deg f$ and $f$ is irreducible, so we must have $f' = 0$. In characteristic 0, $f' = 0$ implies $f$ is constant — contradiction.

In characteristic $p > 0$, inseparability can occur. The polynomial $x^p - t \in \mathbb{F}_p(t)[x]$ is irreducible (by Eisenstein in the ring $\mathbb{F}_p[t]$) but has the single root $t^{1/p}$ with multiplicity $p$ in the splitting field.

**Why separability matters:** it is a hypothesis of the Fundamental Theorem of Galois Theory. A finite extension $K/F$ is Galois if and only if it is both normal (a splitting field) and separable. In characteristic 0, separability is automatic, so "Galois" = "normal" = "splitting field of some polynomial."

## What's next

We have all the ingredients: field extensions, splitting fields, the degree formula, and separability. Next: the Galois group of an extension, the fundamental theorem connecting subgroups to intermediate fields, and the proof that there is no formula for quintic roots.

---

*This is Part 4 of [Abstract Algebra](/en/series/abstract-algebra/) (6 parts).
Previous: [Part 3 — Rings, Ideals, and Polynomial Arithmetic](/en/abstract-algebra/03-rings-and-ideals/) · Next: [Part 5 — Galois Theory](/en/abstract-algebra/05-galois-theory/)*
