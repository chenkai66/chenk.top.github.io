---
title: "Abstract Algebra (7): Field Extensions — Building Bigger Number Systems"
date: 2021-09-13 09:00:00
tags:
  - abstract-algebra
  - field-theory
  - galois-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "Algebraic and transcendental extensions, the tower law, minimal polynomials, and splitting fields — the machinery that makes Galois theory possible."
disableNunjucks: true
series_order: 7
series_total: 12
translationKey: "abstract-algebra-7"
---

Every mathematician, at some point, encounters a polynomial that refuses to be solved within the number system at hand. The ancient Greeks discovered that $\sqrt{2}$ is irrational — that is, $x^2 - 2$ has no solution in $\mathbb{Q}$. The resolution was not to abandon the polynomial, but to enlarge the field. Field extensions formalize this enlargement and provide the structural scaffolding on which Galois theory is built.

This article develops the theory of field extensions from the ground up: degrees and bases, simple extensions and minimal polynomials, the tower law, splitting fields, and separability. By the end, we will have the full toolkit needed to state and prove the Fundamental Theorem of Galois Theory in the next article.

---

## Motivation: Solving Polynomials Requires Bigger Fields

Consider the polynomial $f(x) = x^2 + 1$ over $\mathbb{R}$. It has no real roots, since $x^2 \geq 0$ for all $x \in \mathbb{R}$. But if we pass to the larger field $\mathbb{C} = \mathbb{R}(i)$, the polynomial factors as $(x - i)(x + i)$.

This situation is ubiquitous in algebra:

- $x^2 - 2$ has no root in $\mathbb{Q}$, but has roots $\pm\sqrt{2}$ in $\mathbb{Q}(\sqrt{2})$.
- $x^2 - 5$ has no root in $\mathbb{Q}(\sqrt{2})$, but does in $\mathbb{Q}(\sqrt{2}, \sqrt{5})$.
- $x^3 - 2$ has no root in $\mathbb{Q}$, but has a real root $\sqrt[3]{2}$ in $\mathbb{Q}(\sqrt[3]{2})$ and all three roots in $\mathbb{Q}(\sqrt[3]{2}, \omega)$, where $\omega = e^{2\pi i/3}$ is a primitive cube root of unity.

The pattern is always the same: given a polynomial over a field $K$ that we cannot factor completely, we build a bigger field $L \supseteq K$ in which the polynomial does factor. The theory of field extensions makes this process precise, answering three fundamental questions: how do we construct these larger fields, how "big" are they relative to the base field, and when does a minimal such extension exist?

Historically, this line of thinking emerged from centuries of attempts to find root formulas for polynomials. The quadratic formula works in degree 2. Cardano's formula handles degree 3. Ferrari's method extends to degree 4. But degree 5 resisted all attacks. Understanding *why* required a completely new perspective — not on the roots themselves, but on the symmetries of the field extensions they generate. Field extensions are thus not merely a technical convenience; they are the language in which the deepest structural results of algebra are expressed.

---


![Tower of field extensions over Q](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_fig7_field_tower.png)

## Field Extensions and Degree

**Definition.** A *field extension* is a pair of fields $K \subseteq L$ (equivalently, an injective field homomorphism $K \hookrightarrow L$). We write $L/K$ and call $K$ the *base field* (or *ground field*) and $L$ the *extension field*. The notation $L/K$ does not mean a quotient — it is simply a conventional way to indicate that $L$ extends $K$.

Since $L$ is a field containing $K$, it carries the structure of a vector space over $K$: addition in $L$ is the vector addition, and scalar multiplication by elements of $K$ is given by the field multiplication in $L$. The *degree* of the extension is

$$[L : K] = \dim_K L,$$

the dimension of $L$ as a $K$-vector space. If $[L:K]$ is finite, we say $L/K$ is a *finite extension*; otherwise, it is an *infinite extension*.

**Example 1 ($\mathbb{C}/\mathbb{R}$, degree 2).** Every complex number can be written as $a + bi$ with $a, b \in \mathbb{R}$. The set $\{1, i\}$ is linearly independent over $\mathbb{R}$ (since $a + bi = 0$ with $a,b$ real forces $a = b = 0$) and spans $\mathbb{C}$. So $[\mathbb{C}:\mathbb{R}] = 2$.

**Example 2 ($\mathbb{Q}(\sqrt{2})/\mathbb{Q}$, degree 2).** Define $\mathbb{Q}(\sqrt{2}) = \{a + b\sqrt{2} : a, b \in \mathbb{Q}\}$. This is indeed a field: it is closed under addition and multiplication (using $(\sqrt{2})^2 = 2$), and inverses exist because $1/(a + b\sqrt{2}) = (a - b\sqrt{2})/(a^2 - 2b^2)$, and the denominator $a^2 - 2b^2$ is nonzero when $(a,b) \neq (0,0)$ since $\sqrt{2}$ is irrational. A basis over $\mathbb{Q}$ is $\{1, \sqrt{2}\}$, so $[\mathbb{Q}(\sqrt{2}):\mathbb{Q}] = 2$.

**Example 3 ($\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$, degree 3).** We claim $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$ with basis $\{1, \sqrt[3]{2}, \sqrt[3]{4}\}$. Every element of $\mathbb{Q}(\sqrt[3]{2})$ can be written as $a + b\sqrt[3]{2} + c\sqrt[3]{4}$ with $a,b,c \in \mathbb{Q}$. To verify linear independence: suppose $a + b\sqrt[3]{2} + c\sqrt[3]{4} = 0$ with $a,b,c \in \mathbb{Q}$. If any coefficient is nonzero, then $\sqrt[3]{2}$ satisfies a polynomial of degree at most 2 over $\mathbb{Q}$. But the minimal polynomial of $\sqrt[3]{2}$ over $\mathbb{Q}$ is $x^3 - 2$, which is irreducible by the rational root theorem (the only candidates $\pm 1, \pm 2$ fail). A root of an irreducible cubic cannot satisfy a quadratic. Contradiction.

**Example 4 ($\mathbb{R}/\mathbb{Q}$, infinite).** The extension $\mathbb{R}/\mathbb{Q}$ has infinite degree. One way to see this: $\mathbb{R}$ is uncountable, while any finite-dimensional vector space over $\mathbb{Q}$ is countable (being a countable union of countable sets). More concretely, the elements $1, \sqrt{2}, \sqrt{3}, \sqrt{5}, \sqrt{7}, \ldots$ (square roots of distinct primes) are linearly independent over $\mathbb{Q}$, which already shows $[\mathbb{R}:\mathbb{Q}] \geq \aleph_0$. In fact, $[\mathbb{R}:\mathbb{Q}]$ has the cardinality of the continuum.

**Algebraic vs. transcendental elements.** An element $\alpha \in L$ is *algebraic over $K$* if there exists a nonzero polynomial $f(x) \in K[x]$ with $f(\alpha) = 0$. Otherwise, $\alpha$ is *transcendental over $K$*. A field extension $L/K$ is *algebraic* if every element of $L$ is algebraic over $K$.

**Proposition.** Every finite extension is algebraic.

*Proof.* If $[L:K] = n$ and $\alpha \in L$, then the $n+1$ elements $1, \alpha, \alpha^2, \ldots, \alpha^n$ are linearly dependent over $K$, so there exist $a_0, \ldots, a_n \in K$, not all zero, with $a_0 + a_1\alpha + \cdots + a_n\alpha^n = 0$. This is a polynomial relation, so $\alpha$ is algebraic over $K$. $\blacksquare$

The converse is false: $\overline{\mathbb{Q}}/\mathbb{Q}$ (the field of all algebraic numbers) is an algebraic extension of infinite degree.

---

## Simple Extensions and Minimal Polynomials

**Definition.** Given a field extension $L/K$ and an element $\alpha \in L$, the *simple extension* $K(\alpha)$ is the smallest subfield of $L$ containing both $K$ and $\alpha$. Concretely, $K(\alpha)$ consists of all elements of $L$ that can be expressed as rational functions of $\alpha$ with coefficients in $K$:

$$K(\alpha) = \left\{ \frac{f(\alpha)}{g(\alpha)} : f, g \in K[x], \ g(\alpha) \neq 0 \right\}.$$

There are two fundamentally different cases, depending on whether $\alpha$ is algebraic or transcendental over $K$.

### The Algebraic Case

Suppose $\alpha$ is algebraic over $K$. Among all nonzero polynomials in $K[x]$ having $\alpha$ as a root, there is a unique monic polynomial of smallest degree.

**Definition.** The *minimal polynomial* of $\alpha$ over $K$, denoted $\min_K(\alpha)$ or $m_\alpha(x)$, is the unique monic polynomial of smallest degree in $K[x]$ that vanishes at $\alpha$.

**Theorem (Properties of the minimal polynomial).** Let $\alpha$ be algebraic over $K$ with minimal polynomial $m(x)$ of degree $n$. Then:

1. $m(x)$ is irreducible over $K$.
2. If $f(x) \in K[x]$ satisfies $f(\alpha) = 0$, then $m(x) \mid f(x)$ in $K[x]$.
3. $K(\alpha) \cong K[x]/(m(x))$ as $K$-algebras.
4. $[K(\alpha) : K] = n$, and $\{1, \alpha, \alpha^2, \ldots, \alpha^{n-1}\}$ is a basis for $K(\alpha)$ over $K$.

*Proof.*

(1) Suppose $m(x) = g(x)h(x)$ with $1 \leq \deg g, \deg h < \deg m$. Then $0 = m(\alpha) = g(\alpha)h(\alpha)$. Since $L$ is a field (hence an integral domain), either $g(\alpha) = 0$ or $h(\alpha) = 0$. Either way, we have a monic polynomial of degree less than $\deg m$ vanishing at $\alpha$ (after dividing by the leading coefficient), contradicting the minimality of $m$.

(2) By the division algorithm, $f(x) = q(x)m(x) + r(x)$ with $\deg r < \deg m$. Evaluating at $\alpha$: $0 = f(\alpha) = q(\alpha) \cdot 0 + r(\alpha) = r(\alpha)$. If $r \neq 0$, dividing by its leading coefficient gives a monic polynomial of degree less than $\deg m$ vanishing at $\alpha$ — contradiction. So $r = 0$ and $m \mid f$.

(3) Consider the evaluation homomorphism $\operatorname{ev}_\alpha : K[x] \to L$ defined by $f(x) \mapsto f(\alpha)$. Its image is $K[\alpha]$, the ring of polynomial expressions in $\alpha$. Its kernel is $\{f \in K[x] : f(\alpha) = 0\}$, which equals $(m(x))$ by part (2). Since $K[x]$ is a PID and $m(x)$ is irreducible, $(m(x))$ is a maximal ideal, so $K[x]/(m(x)) \cong K[\alpha]$ is a field. But $K[\alpha]$ is a subfield of $L$ containing $K$ and $\alpha$, and it is contained in every such subfield, so $K[\alpha] = K(\alpha)$.

(4) In $K[x]/(m(x))$, every coset is represented by a unique polynomial of degree $< n$ (by the division algorithm). So $\{\overline{1}, \overline{x}, \ldots, \overline{x^{n-1}}\}$ is a basis over $K$. Under the isomorphism $K[x]/(m(x)) \cong K(\alpha)$, these correspond to $\{1, \alpha, \ldots, \alpha^{n-1}\}$. $\blacksquare$

**Remark.** Part (3) reveals something important: in the algebraic case, $K(\alpha) = K[\alpha]$ — every rational expression in $\alpha$ can be reduced to a polynomial expression. The point is that inverses come for free: if $f(\alpha) \neq 0$, then $\gcd(f(x), m(x)) = 1$ (since $m$ is irreducible and $m \nmid f$), so by Bezout's identity there exist $s, t \in K[x]$ with $s(x)f(x) + t(x)m(x) = 1$. Evaluating at $\alpha$: $s(\alpha)f(\alpha) = 1$, so $f(\alpha)^{-1} = s(\alpha)$ is a polynomial in $\alpha$.

### Worked Example: Arithmetic in $\mathbb{Q}(\sqrt{2})$

The minimal polynomial of $\sqrt{2}$ over $\mathbb{Q}$ is $m(x) = x^2 - 2$, which is irreducible over $\mathbb{Q}$ by Eisenstein's criterion at $p = 2$ (or by the rational root theorem). Therefore $[\mathbb{Q}(\sqrt{2}) : \mathbb{Q}] = 2$ with basis $\{1, \sqrt{2}\}$.

**Multiplication.** $(3 + 5\sqrt{2})(1 - 2\sqrt{2}) = 3 - 6\sqrt{2} + 5\sqrt{2} - 10(\sqrt{2})^2 = 3 - \sqrt{2} - 20 = -17 - \sqrt{2}$.

**Inversion.** To find $(3 + 5\sqrt{2})^{-1}$, multiply numerator and denominator by the conjugate:

$$\frac{1}{3 + 5\sqrt{2}} = \frac{3 - 5\sqrt{2}}{(3)^2 - 2(5)^2} = \frac{3 - 5\sqrt{2}}{9 - 50} = \frac{3 - 5\sqrt{2}}{-41} = -\frac{3}{41} + \frac{5}{41}\sqrt{2}.$$

### Worked Example: Arithmetic in $\mathbb{Q}(\sqrt[3]{2})$

The minimal polynomial of $\sqrt[3]{2}$ over $\mathbb{Q}$ is $x^3 - 2$. Let $\alpha = \sqrt[3]{2}$. Then $[\mathbb{Q}(\alpha):\mathbb{Q}] = 3$ with basis $\{1, \alpha, \alpha^2\}$.

**Multiplication.** $(1 + \alpha)(2 - \alpha + \alpha^2) = 2 - \alpha + \alpha^2 + 2\alpha - \alpha^2 + \alpha^3 = 2 + \alpha + \alpha^3 = 2 + \alpha + 2 = 4 + \alpha$, using $\alpha^3 = 2$.

**Inversion.** To find $(1 + \alpha)^{-1}$, use the extended Euclidean algorithm on $1 + x$ and $x^3 - 2$ in $\mathbb{Q}[x]$. We need $s(x)(1+x) + t(x)(x^3-2) = 1$. Performing polynomial long division: $x^3 - 2 = (x^2 - x + 1)(x + 1) + (-3)$, so $-3 = (x^3 - 2) - (x^2 - x + 1)(x+1)$, giving $1 = -\frac{1}{3}(x^3-2) + \frac{1}{3}(x^2-x+1)(x+1)$. Setting $x = \alpha$: $(1+\alpha)^{-1} = \frac{1}{3}(\alpha^2 - \alpha + 1)$.

Check: $\frac{1}{3}(\alpha^2 - \alpha + 1)(1 + \alpha) = \frac{1}{3}(\alpha^2 - \alpha + 1 + \alpha^3 - \alpha^2 + \alpha) = \frac{1}{3}(1 + 2) = 1$. Correct.

### The Transcendental Case

If $\alpha$ is transcendental over $K$, then $\operatorname{ev}_\alpha : K[x] \to K[\alpha]$ is injective (no nonzero polynomial vanishes at $\alpha$). So $K[\alpha] \cong K[x]$, a polynomial ring, which is *not* a field. In this case $K(\alpha) \cong K(x)$, the field of rational functions, and $[K(\alpha):K] = \infty$.

**Example.** Since $\pi$ is transcendental over $\mathbb{Q}$ (Lindemann, 1882), we have $\mathbb{Q}(\pi) \cong \mathbb{Q}(x)$ and $[\mathbb{Q}(\pi):\mathbb{Q}] = \infty$.

---

## The Tower Law and Its Consequences

The tower law is the multiplicativity of degrees in a tower of field extensions. It is the single most frequently used tool in computing degrees.

**Theorem (Tower Law).** If $K \subseteq M \subseteq L$ are fields with $[M:K]$ and $[L:M]$ both finite, then $[L:K]$ is finite and

$$[L : K] = [L : M] \cdot [M : K].$$

*Proof.* Let $m = [M:K]$ and $n = [L:M]$. Choose a basis $\{e_1, \ldots, e_m\}$ for $M$ over $K$ and a basis $\{f_1, \ldots, f_n\}$ for $L$ over $M$. We claim that the set

$$\mathcal{B} = \{e_i f_j : 1 \leq i \leq m, \ 1 \leq j \leq n\}$$

is a basis for $L$ over $K$. Since $|\mathcal{B}| = mn$, this gives $[L:K] = mn$.

*Spanning.* Let $\ell \in L$. Since $\{f_j\}$ is a basis for $L/M$, write $\ell = \sum_{j=1}^{n} \mu_j f_j$ with $\mu_j \in M$. Since $\{e_i\}$ is a basis for $M/K$, write each $\mu_j = \sum_{i=1}^{m} a_{ij} e_i$ with $a_{ij} \in K$. Substituting:

$$\ell = \sum_{j=1}^{n} \left(\sum_{i=1}^{m} a_{ij} e_i\right) f_j = \sum_{i,j} a_{ij} (e_i f_j).$$

So $\mathcal{B}$ spans $L$ over $K$.

*Linear independence.* Suppose $\sum_{i,j} a_{ij} (e_i f_j) = 0$ with $a_{ij} \in K$. Rearranging:

$$\sum_{j=1}^{n} \underbrace{\left(\sum_{i=1}^{m} a_{ij} e_i\right)}_{\mu_j \in M} f_j = 0.$$

Since $\{f_j\}$ is linearly independent over $M$, each $\mu_j = \sum_{i} a_{ij} e_i = 0$. Since $\{e_i\}$ is linearly independent over $K$, each $a_{ij} = 0$. $\blacksquare$

### Application 1: Degree Divisibility

If $K \subseteq M \subseteq L$ with $[L:K]$ finite, then $[M:K]$ divides $[L:K]$. In particular, if $\alpha$ is algebraic over $K$ and belongs to some finite extension $L/K$ of degree $n$, then $\deg \min_K(\alpha)$ divides $n$.

### Application 2: $\sqrt{2} + \sqrt{3}$ Generates $\mathbb{Q}(\sqrt{2}, \sqrt{3})$

We compute $[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}]$ via the tower:

$$K = \mathbb{Q} \subset \mathbb{Q}(\sqrt{2}) \subset \mathbb{Q}(\sqrt{2}, \sqrt{3}).$$

First, $[\mathbb{Q}(\sqrt{2}):\mathbb{Q}] = 2$. Next, we need $[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}(\sqrt{2})]$. The polynomial $x^2 - 3$ has $\sqrt{3}$ as a root, so this degree is at most 2. It equals exactly 2 provided $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$.

**Claim:** $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$.

*Proof.* Suppose $\sqrt{3} = a + b\sqrt{2}$ with $a, b \in \mathbb{Q}$. Squaring: $3 = a^2 + 2b^2 + 2ab\sqrt{2}$. Since $\sqrt{2}$ is irrational and $a^2 + 2b^2, 2ab$ are rational, we need $2ab = 0$ and $a^2 + 2b^2 = 3$. If $a = 0$: $2b^2 = 3$, so $b^2 = 3/2$, impossible for $b \in \mathbb{Q}$ (since $\sqrt{6}/2 \notin \mathbb{Q}$). If $b = 0$: $a^2 = 3$, impossible for $a \in \mathbb{Q}$. $\blacksquare$

So $[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}] = 2 \times 2 = 4$, with basis $\{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$.

Now let $\alpha = \sqrt{2} + \sqrt{3}$. We show $\mathbb{Q}(\alpha) = \mathbb{Q}(\sqrt{2},\sqrt{3})$.

- $\alpha^2 = 5 + 2\sqrt{6}$, so $\sqrt{6} = (\alpha^2 - 5)/2 \in \mathbb{Q}(\alpha)$.
- $\alpha \sqrt{6} = \sqrt{12} + \sqrt{18} = 2\sqrt{3} + 3\sqrt{2}$.
- From $\alpha = \sqrt{2} + \sqrt{3}$ and $\alpha\sqrt{6} = 3\sqrt{2} + 2\sqrt{3}$, subtract $2\alpha$ from $\alpha\sqrt{6}$:

$$\alpha\sqrt{6} - 2\alpha = 3\sqrt{2} + 2\sqrt{3} - 2\sqrt{2} - 2\sqrt{3} = \sqrt{2}.$$

So $\sqrt{2} = \alpha(\sqrt{6} - 2) \in \mathbb{Q}(\alpha)$, and $\sqrt{3} = \alpha - \sqrt{2} \in \mathbb{Q}(\alpha)$.

Since $\mathbb{Q}(\alpha) \supseteq \mathbb{Q}(\sqrt{2},\sqrt{3})$ and the reverse inclusion is trivial ($\alpha \in \mathbb{Q}(\sqrt{2},\sqrt{3})$), we have equality.

The minimal polynomial of $\alpha$ over $\mathbb{Q}$ has degree 4. We find it explicitly: $\alpha^2 = 5 + 2\sqrt{6}$, so $\alpha^2 - 5 = 2\sqrt{6}$, and $(\alpha^2 - 5)^2 = 24$, giving:

$$\alpha^4 - 10\alpha^2 + 25 = 24, \quad \text{i.e.,} \quad \alpha^4 - 10\alpha^2 + 1 = 0.$$

Therefore $\min_\mathbb{Q}(\sqrt{2}+\sqrt{3}) = x^4 - 10x^2 + 1$.

### Application 3: Classical Impossibility Results

The tower law connects to the classical Greek construction problems through the following observation: a real number $\alpha$ is *constructible* (by straightedge and compass) from a unit segment if and only if $\alpha$ lies in a field obtained from $\mathbb{Q}$ by a sequence of quadratic extensions. In other words, $[\mathbb{Q}(\alpha):\mathbb{Q}]$ must be a power of 2.

- **Doubling the cube** requires constructing $\sqrt[3]{2}$. But $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$, which is not a power of 2. Impossible.
- **Trisecting a $60°$ angle** requires constructing $\cos(20°)$, which satisfies $8x^3 - 6x - 1 = 0$. This cubic is irreducible over $\mathbb{Q}$ (rational root theorem), so $[\mathbb{Q}(\cos 20°):\mathbb{Q}] = 3$. Impossible.
- **Squaring the circle** requires constructing $\sqrt{\pi}$, which is transcendental (since $\pi$ is). So $[\mathbb{Q}(\sqrt{\pi}):\mathbb{Q}] = \infty$. Impossible.

---

## Splitting Fields and Algebraic Closures

So far, we have adjoined individual roots. But many questions — especially in Galois theory — require us to adjoin *all* roots of a polynomial at once.

**Definition.** Let $f(x) \in K[x]$ be a polynomial of degree $n$. A *splitting field* for $f$ over $K$ is an extension $L/K$ such that:

1. $f$ factors completely in $L[x]$: $f(x) = c(x - \alpha_1)(x - \alpha_2)\cdots(x - \alpha_n)$ with each $\alpha_i \in L$.
2. $L = K(\alpha_1, \ldots, \alpha_n)$ — that is, $L$ is generated over $K$ by the roots of $f$.

Condition (2) ensures minimality: $L$ is the smallest field over $K$ in which $f$ splits completely.

**Theorem (Existence of splitting fields).** Every nonconstant polynomial $f(x) \in K[x]$ has a splitting field.

*Proof.* By induction on $n = \deg f$.

*Base case:* $n = 1$. Then $f(x) = c(x - a)$ already splits in $K$. Take $L = K$.

*Inductive step:* Suppose $n > 1$ and the result holds for polynomials of degree $< n$. Let $p(x)$ be an irreducible factor of $f(x)$ in $K[x]$. The quotient $K_1 = K[x]/(p(x))$ is a field extension of $K$ containing a root $\alpha_1 = \overline{x}$ of $p$ (and hence of $f$). Write $f(x) = (x - \alpha_1) g(x)$ in $K_1[x]$, where $\deg g = n - 1$. By the inductive hypothesis, $g$ has a splitting field $L$ over $K_1$. Then $f$ splits completely in $L$, and $L$ is generated over $K$ by all roots of $f$ (the roots of $g$ together with $\alpha_1$). So $L$ is a splitting field for $f$ over $K$. $\blacksquare$

**Theorem (Uniqueness of splitting fields).** If $L_1$ and $L_2$ are both splitting fields for $f(x) \in K[x]$ over $K$, then there exists a $K$-isomorphism $\sigma : L_1 \xrightarrow{\sim} L_2$ (i.e., $\sigma|_K = \operatorname{id}_K$).

*Proof outline.* Induct on $[L_1:K]$. If $f$ splits in $K$ already, both $L_1 = L_2 = K$ and the identity works. Otherwise, pick an irreducible factor $p(x)$ of $f$ over $K$ with a root $\alpha \in L_1$ and a root $\beta \in L_2$. The map $\overline{x} \mapsto \beta$ gives an isomorphism $\sigma_0 : K(\alpha) = K[x]/(p(x)) \xrightarrow{\sim} K(\beta)$. Now $L_1$ is a splitting field for $f/(x-\alpha)$ over $K(\alpha)$, and $L_2$ is a splitting field for $f/(x-\beta)$ over $K(\beta)$. By induction, $\sigma_0$ extends to an isomorphism $L_1 \to L_2$. $\blacksquare$

This uniqueness is crucial: it allows us to speak of "*the* splitting field" of a polynomial, up to isomorphism.

### Worked Example: Splitting Field of $x^4 - 2$ over $\mathbb{Q}$

The roots of $x^4 - 2$ in $\mathbb{C}$ are $\sqrt[4]{2}, \ i\sqrt[4]{2}, \ -\sqrt[4]{2}, \ -i\sqrt[4]{2}$. These can be written as $\sqrt[4]{2} \cdot i^k$ for $k = 0,1,2,3$. The splitting field is therefore

$$L = \mathbb{Q}(\sqrt[4]{2}, \ i).$$

We compute $[L:\mathbb{Q}]$ using the tower $\mathbb{Q} \subset \mathbb{Q}(\sqrt[4]{2}) \subset L$.

- $[\mathbb{Q}(\sqrt[4]{2}) : \mathbb{Q}] = 4$, since $x^4 - 2$ is irreducible over $\mathbb{Q}$ by Eisenstein at $p = 2$.
- $[L : \mathbb{Q}(\sqrt[4]{2})] = 2$, since $i \notin \mathbb{Q}(\sqrt[4]{2}) \subset \mathbb{R}$, and $i$ satisfies the quadratic $x^2 + 1$.

By the tower law: $[L:\mathbb{Q}] = 4 \times 2 = 8$.

A basis for $L$ over $\mathbb{Q}$ is $\{1, \sqrt[4]{2}, \sqrt{2}, \sqrt[4]{8}, i, i\sqrt[4]{2}, i\sqrt{2}, i\sqrt[4]{8}\}$.

Note that $[L:\mathbb{Q}] = 8$ while $\deg(x^4-2) = 4$. In general, the degree of a splitting field can be much larger than the degree of the polynomial — or equal to it (as for $x^2 - 2$ over $\mathbb{Q}$, where $[\mathbb{Q}(\sqrt{2}):\mathbb{Q}] = 2$).

### Algebraic Closures

**Definition.** A field $K$ is *algebraically closed* if every nonconstant polynomial in $K[x]$ has a root in $K$. Equivalently, every polynomial in $K[x]$ splits completely in $K[x]$.

**Definition.** An *algebraic closure* of $K$, denoted $\overline{K}$, is a field extension of $K$ that is both algebraically closed and algebraic over $K$.

**Theorem.** Every field $K$ has an algebraic closure, and any two algebraic closures of $K$ are isomorphic over $K$.

The existence proof requires Zorn's lemma (or an equivalent set-theoretic axiom). The most familiar example is $\overline{\mathbb{Q}}$, the field of algebraic numbers, sitting inside $\mathbb{C}$. By contrast, $\mathbb{C}$ itself is $\overline{\mathbb{R}}$ (the Fundamental Theorem of Algebra).

A less familiar example: $\overline{\mathbb{F}_p}$, the algebraic closure of the finite field with $p$ elements, is the union $\bigcup_{n \geq 1} \mathbb{F}_{p^n}$.

---

## Separability and Perfect Fields

In characteristic zero, all algebraic extensions behave nicely: minimal polynomials have distinct roots. In positive characteristic, pathologies can arise. The concept of separability identifies and excludes these pathologies.

**Definition.** A polynomial $f(x) \in K[x]$ is *separable* if it has no repeated roots in its splitting field (i.e., all roots are distinct). An algebraic element $\alpha$ over $K$ is *separable* if its minimal polynomial is separable. An algebraic extension $L/K$ is *separable* if every element of $L$ is separable over $K$.

**Proposition (Derivative criterion).** A polynomial $f(x)$ has a repeated root in its splitting field if and only if $\gcd(f, f') \neq 1$, where $f'$ is the formal derivative.

*Proof.* If $\alpha$ is a repeated root, write $f(x) = (x - \alpha)^2 g(x)$. Then $f'(x) = 2(x - \alpha)g(x) + (x - \alpha)^2 g'(x)$, so $(x - \alpha) \mid \gcd(f, f')$. Conversely, if $\alpha$ is a root of $f$ but not a repeated root, write $f(x) = (x - \alpha)h(x)$ with $h(\alpha) \neq 0$. Then $f'(\alpha) = h(\alpha) \neq 0$, so $\alpha$ is not a root of $f'$, and $\gcd(f, f')$ is not divisible by $(x - \alpha)$. $\blacksquare$

**Corollary.** An irreducible polynomial $p(x) \in K[x]$ is inseparable if and only if $p'(x) = 0$.

*Proof.* Since $p$ is irreducible and $\deg p' < \deg p$, we have $\gcd(p, p') \neq 1$ iff $p \mid p'$ iff $p' = 0$ (by degree). $\blacksquare$

Now, $p'(x) = 0$ means every monomial $a_k x^k$ with $a_k \neq 0$ satisfies $k \cdot a_k = 0$ in $K$. In characteristic 0, this forces $k = 0$ for all such terms, meaning $p$ is constant — contradicting irreducibility. Therefore:

**In characteristic 0, every irreducible polynomial is separable.** This is why separability rarely appears in a first algebra course that stays in characteristic 0.

In characteristic $p > 0$, the condition $p' = 0$ means $p(x)$ is a polynomial in $x^p$: $p(x) = a_0 + a_1 x^p + a_2 x^{2p} + \cdots$.

**Definition.** A field $K$ is *perfect* if every irreducible polynomial over $K$ is separable.

All fields of characteristic 0 are perfect. All finite fields $\mathbb{F}_q$ are perfect (the Frobenius $a \mapsto a^p$ is injective on a finite set, hence surjective). In general, a field of characteristic $p$ is perfect if and only if every element has a $p$-th root: $K^p = K$.

**Counterexample.** The field $\mathbb{F}_p(t)$ of rational functions over $\mathbb{F}_p$ is not perfect. The polynomial $x^p - t$ is irreducible over $\mathbb{F}_p(t)$ (by Eisenstein's criterion applied with the prime element $t$ in $\mathbb{F}_p[t]$), yet in its splitting field $x^p - t = (x - t^{1/p})^p$, so it has a single root with multiplicity $p$.

**Why separability matters for Galois theory.** A finite extension $L/K$ is a *Galois extension* if it is both *normal* (i.e., $L$ is a splitting field for some polynomial over $K$) and *separable*. The Galois group $\operatorname{Gal}(L/K)$ of such an extension has order exactly $[L:K]$. Without separability, this equality breaks down: for an inseparable extension, $|\operatorname{Aut}_K(L)| < [L:K]$, and the Galois correspondence fails. For extensions of perfect fields (which covers $\mathbb{Q}$ and all finite fields), separability is automatic, so the only condition to check is normality.

---

### Normal Extensions

Splitting fields lead naturally to the concept of normality, which will be central in the next article.

**Definition.** An algebraic extension $L/K$ is *normal* if every irreducible polynomial in $K[x]$ that has at least one root in $L$ splits completely in $L[x]$.

**Theorem.** A finite extension $L/K$ is normal if and only if $L$ is a splitting field for some polynomial $f(x) \in K[x]$.

**Example.** $\mathbb{Q}(\sqrt{2})/\mathbb{Q}$ is normal: it is the splitting field of $x^2 - 2$. However, $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ is *not* normal: the polynomial $x^3 - 2$ has one root $\sqrt[3]{2}$ in $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$, but its other two roots $\sqrt[3]{2}\omega$ and $\sqrt[3]{2}\omega^2$ (where $\omega = e^{2\pi i/3}$) are non-real and therefore not in $\mathbb{Q}(\sqrt[3]{2})$. This is a key distinction: adjoining one root of an irreducible polynomial does not always give you a normal extension; you may need to adjoin all the roots.

**Example.** The splitting field of $x^3 - 2$ over $\mathbb{Q}$ is $\mathbb{Q}(\sqrt[3]{2}, \omega)$, which *is* normal (being a splitting field). Its degree over $\mathbb{Q}$ is $[\mathbb{Q}(\sqrt[3]{2},\omega):\mathbb{Q}] = 6$: we have $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$ and $[\mathbb{Q}(\sqrt[3]{2},\omega):\mathbb{Q}(\sqrt[3]{2})] = 2$ since $\omega$ satisfies $x^2 + x + 1 = 0$, which is irreducible over $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$ (it has no real roots).

---

## What's Next

We have assembled all the ingredients: field extensions and their degrees, the tower law for computing degrees in chains, minimal polynomials that describe simple extensions, splitting fields that give us "complete" factorizations, normality, and separability to prevent degenerate behavior. In the next article, we combine these tools into Galois theory proper: the group of automorphisms of a field extension and its remarkable correspondence with the lattice of intermediate fields. This correspondence will ultimately explain why the general quintic cannot be solved by radicals, settling a question that puzzled mathematicians for three centuries.

---

*This is Part 7 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 6 — Polynomial Rings](/en/abstract-algebra/06-polynomial-rings/)*

*Next: [Part 8 — Galois Theory](/en/abstract-algebra/08-galois-theory/)*
