---
title: "Abstract Algebra (6): Polynomial Rings — Factorization and Unique Decomposition"
date: 2021-09-11 09:00:00
tags:
  - abstract-algebra
  - ring-theory
  - polynomials
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "The division algorithm, irreducibility tests, and the climb from Z to Z[x] to Q[x] — understanding when and why unique factorization holds."
disableNunjucks: true
series_order: 6
series_total: 12
translationKey: "abstract-algebra-6"
---

Polynomials are the laboratory of algebra. Nearly every concept in ring theory — ideals, quotients, factorization, irreducibility — was first understood through polynomial examples before being abstracted. This is no coincidence: polynomial rings are rich enough to exhibit all the interesting phenomena yet structured enough to permit explicit computation.

In this article, we study the ring $R[x]$ of polynomials over a ring $R$. We develop the division algorithm, establish irreducibility criteria (Eisenstein, reduction mod $p$, the rational root test), define Unique Factorization Domains, and prove Gauss's Lemma — the bridge between factorization over $\mathbb{Z}$ and factorization over $\mathbb{Q}$. The payoff is a clear picture of *when* and *why* unique factorization holds, and what goes wrong when it fails.

---

## Why Polynomials Are Central to Algebra

Three reasons polynomials deserve a dedicated study:

**1. Polynomials generate all algebraic extensions.** If $\alpha$ is an algebraic number — a root of some polynomial with rational coefficients — then the field $\mathbb{Q}(\alpha)$ is isomorphic to $\mathbb{Q}[x]/(p(x))$ where $p$ is the minimal polynomial of $\alpha$. Understanding polynomial factorization is thus equivalent to understanding the structure of algebraic number fields.

**2. Polynomials provide a universal testing ground.** The ring $\mathbb{Z}[x]$ is simultaneously:
- A UFD but not a PID (so we can study the gap between these notions).
- The "generic" ring: by the Hilbert Basis Theorem, polynomial rings over Noetherian rings are Noetherian.
- The home of classical algebra: every high-school factoring problem is a question about $\mathbb{Z}[x]$ or $\mathbb{Q}[x]$.

**3. Polynomials connect algebra to geometry.** The zeros of a polynomial $f(x_1, \ldots, x_n)$ form an algebraic variety — this is the starting point of algebraic geometry. The ring-theoretic properties of the polynomial ring (its ideals, its quotients) translate directly into geometric properties of the variety.

---


![Factorization tree in Z[x]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_fig6_factorization.png)

## $R[x]$: Definition, Degree, and the Division Algorithm

**Definition.** Let $R$ be a commutative ring with unity. The **polynomial ring** $R[x]$ consists of all formal expressions

$$f(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0$$

where $a_i \in R$ and $a_n \neq 0$ (for the zero polynomial, we leave the degree undefined or set it to $-\infty$ by convention). The **degree** of $f$ is $\deg f = n$. The **leading coefficient** is $a_n$; if $a_n = 1$, the polynomial is **monic**.

Addition is coefficientwise. Multiplication uses the convolution formula: if $f = \sum a_i x^i$ and $g = \sum b_j x^j$, then

$$fg = \sum_k \left(\sum_{i+j=k} a_i b_j\right) x^k.$$

**Degree and multiplication.** If $R$ is an integral domain, then $\deg(fg) = \deg f + \deg g$. (This can fail if $R$ has zero divisors: in $\mathbb{Z}/4\mathbb{Z}[x]$, $(2x)(2x) = 4x^2 = 0$.)

**Consequence.** If $R$ is an integral domain, so is $R[x]$. *Proof:* If $fg = 0$ and $f, g \neq 0$, then $\deg(fg) = \deg f + \deg g \geq 0$, but $fg = 0$ has degree $-\infty$. Contradiction.

**Multivariate polynomial rings.** We can iterate the construction: $R[x][y]$, also written $R[x, y]$, is the ring of polynomials in two variables. More generally, $R[x_1, \ldots, x_n]$ is defined inductively as $R[x_1, \ldots, x_{n-1}][x_n]$. These rings are fundamental in algebraic geometry, where ideals of $k[x_1, \ldots, x_n]$ correspond to algebraic varieties in $k^n$.

**The evaluation homomorphism.** For any element $\alpha$ in a ring $S$ containing $R$, the map $\text{ev}_\alpha: R[x] \to S$ defined by $f(x) \mapsto f(\alpha)$ is a ring homomorphism. This connects the formal algebra of polynomials with their function-theoretic behavior. Two polynomials can define the same function without being equal as ring elements — for instance, in $\mathbb{F}_2[x]$, the polynomials $x$ and $x^2$ define the same function ($0 \mapsto 0$, $1 \mapsto 1$) but are distinct elements of the polynomial ring. The polynomial ring captures more structure than the function ring.

### The Division Algorithm

**Theorem (Division Algorithm).** Let $F$ be a field and let $f, g \in F[x]$ with $g \neq 0$. Then there exist unique $q, r \in F[x]$ such that

$$f = qg + r, \qquad \deg r < \deg g.$$

*Proof.* **Existence** by strong induction on $\deg f$. If $\deg f < \deg g$, take $q = 0$, $r = f$. Otherwise, let $f = a_n x^n + \cdots$ and $g = b_m x^m + \cdots$ with $n \geq m$. Set $f_1 = f - \frac{a_n}{b_m} x^{n-m} g$. Then $\deg f_1 < n$, so by induction $f_1 = q_1 g + r$ with $\deg r < \deg g$. Thus $f = \left(\frac{a_n}{b_m} x^{n-m} + q_1\right) g + r$.

**Uniqueness.** If $f = qg + r = q'g + r'$, then $(q - q')g = r' - r$. If $q \neq q'$, then $\deg((q-q')g) = \deg(q-q') + \deg g \geq \deg g > \deg(r' - r)$, contradiction. So $q = q'$ and $r = r'$. $\square$

**Crucial point:** the division algorithm requires dividing by the leading coefficient of $g$, which is why we need $F$ to be a *field* (not just an integral domain). Over $\mathbb{Z}$, we cannot always divide $x^2 + 1$ by $2x + 1$ and stay in $\mathbb{Z}[x]$.

**Corollary.** $F[x]$ is a Euclidean domain (with the Euclidean function being the degree), hence a PID, hence a UFD. This is the polynomial analogue of the Fundamental Theorem of Arithmetic.

**Corollary.** Every ideal of $F[x]$ is principal, generated by a single polynomial. The monic generator is unique.

**Roots and the Factor Theorem.** An element $\alpha \in F$ is a **root** of $f \in F[x]$ if $f(\alpha) = 0$. The Factor Theorem states: $\alpha$ is a root of $f$ if and only if $(x - \alpha)$ divides $f$ in $F[x]$. *Proof:* By the division algorithm, $f(x) = q(x)(x - \alpha) + r$ where $r \in F$ (since $\deg r < \deg(x - \alpha) = 1$). Evaluating at $\alpha$: $f(\alpha) = 0 + r = r$. So $r = 0$ iff $\alpha$ is a root.

**Corollary.** A polynomial of degree $n$ over a field has at most $n$ roots. (Each root accounts for one linear factor, and there are at most $n$ factors.) This innocent-looking statement fails over non-integral-domains: in $\mathbb{Z}/8\mathbb{Z}$, the polynomial $x^2 - 1$ has four roots: $1, 3, 5, 7$.

**The GCD in $F[x]$.** Since $F[x]$ is a Euclidean domain, the Euclidean algorithm computes $\gcd(f, g)$ for any $f, g \in F[x]$. This is both a theoretical tool (proving that $F[x]$ is a PID) and a practical algorithm (used in symbolic computation, partial fraction decomposition, and coding theory).

**Worked Example 1.** *Divide $f(x) = x^4 + 2x^3 - x + 3$ by $g(x) = x^2 + x - 1$ in $\mathbb{Q}[x]$.*

*Step 1:* $\frac{x^4}{x^2} = x^2$. Compute $f - x^2 g = f - x^2(x^2 + x - 1) = f - x^4 - x^3 + x^2 = x^3 + x^2 - x + 3$.

*Step 2:* $\frac{x^3}{x^2} = x$. Compute $(x^3 + x^2 - x + 3) - x(x^2 + x - 1) = x^3 + x^2 - x + 3 - x^3 - x^2 + x = 3$.

*Step 3:* $\deg(3) = 0 < 2 = \deg g$, so we stop.

Result: $q(x) = x^2 + x$, $r(x) = 3$. Check: $(x^2 + x)(x^2 + x - 1) + 3 = x^4 + x^3 - x^2 + x^3 + x^2 - x + 3 = x^4 + 2x^3 - x + 3$. $\checkmark$ $\square$

---

## Irreducibility Criteria

**Definition.** A polynomial $f \in R[x]$ of degree $\geq 1$ is **irreducible** (over $R$) if whenever $f = gh$ in $R[x]$, either $g$ or $h$ is a unit.

Over a field $F$, the units of $F[x]$ are exactly the nonzero constants, so irreducibility means: $f$ cannot be written as a product of two polynomials of strictly lower degree.

**Warning.** Irreducibility depends on the ring. The polynomial $x^2 + 1$ is irreducible over $\mathbb{R}$ but factors as $(x+i)(x-i)$ over $\mathbb{C}$, and factors as $(x+2)(x+3)$ over $\mathbb{F}_5$.

This dependence on the base ring is not a nuisance — it is the whole point. The question "over which fields does a given polynomial factor?" is the beginning of Galois theory. The splitting behavior of a polynomial over different fields reveals deep arithmetic information: for instance, the factorization pattern of a polynomial modulo different primes $p$ is intimately connected to the structure of its Galois group (this is the Chebotarev density theorem, one of the jewels of algebraic number theory).

### The Rational Root Test

**Theorem.** If $f(x) = a_n x^n + \cdots + a_0 \in \mathbb{Z}[x]$ has a rational root $p/q$ in lowest terms, then $p \mid a_0$ and $q \mid a_n$.

*Proof.* From $a_n(p/q)^n + \cdots + a_0 = 0$, multiply through by $q^n$: $a_n p^n + a_{n-1} p^{n-1} q + \cdots + a_0 q^n = 0$. Since $p$ divides every term except possibly $a_0 q^n$, we get $p \mid a_0 q^n$. Since $\gcd(p, q) = 1$, Euclid's lemma gives $p \mid a_0$. The argument for $q \mid a_n$ is symmetric. $\square$

**Worked Example 2.** *Show that $f(x) = x^3 - 3x + 1$ is irreducible over $\mathbb{Q}$.*

*Solution.* By the rational root test, any rational root must be $\pm 1$. Check: $f(1) = 1 - 3 + 1 = -1 \neq 0$ and $f(-1) = -1 + 3 + 1 = 3 \neq 0$. Since $\deg f = 3$, if $f$ were reducible over $\mathbb{Q}$, it would have a linear factor, hence a rational root. Since it has none, $f$ is irreducible over $\mathbb{Q}$. $\square$

### Eisenstein's Criterion

**Theorem (Eisenstein).** Let $f(x) = a_n x^n + \cdots + a_1 x + a_0 \in \mathbb{Z}[x]$ with $n \geq 1$. If there exists a prime $p$ such that:
1. $p \nmid a_n$ (the leading coefficient),
2. $p \mid a_i$ for all $0 \leq i < n$,
3. $p^2 \nmid a_0$,

then $f$ is irreducible over $\mathbb{Q}$.

*Proof sketch.* Suppose $f = gh$ in $\mathbb{Z}[x]$ (by Gauss's Lemma, which we prove below, irreducibility over $\mathbb{Z}$ is equivalent to irreducibility over $\mathbb{Q}$ for primitive polynomials). Reduce modulo $p$: $\bar{f} = \bar{a}_n x^n$, $\bar{g} \bar{h} = \bar{a}_n x^n$. In $\mathbb{F}_p[x]$, unique factorization forces $\bar{g} = \bar{b}_r x^r$ and $\bar{h} = \bar{c}_s x^s$ with $r + s = n$. So $p$ divides the constant terms of both $g$ and $h$, hence $p^2$ divides $a_0 = g(0)h(0)$. This contradicts condition (3). $\square$

**Example.** The **cyclotomic polynomial** $\Phi_p(x) = x^{p-1} + x^{p-2} + \cdots + x + 1$ (for $p$ prime) is irreducible over $\mathbb{Q}$. The trick is to substitute $x \mapsto x + 1$:

$$\Phi_p(x+1) = \frac{(x+1)^p - 1}{x} = x^{p-1} + \binom{p}{1} x^{p-2} + \cdots + \binom{p}{p-1}.$$

Now Eisenstein applies with the prime $p$: $\binom{p}{k}$ is divisible by $p$ for $1 \leq k \leq p-1$, the leading coefficient is $1$, and the constant term is $\binom{p}{p-1} = p$, which is not divisible by $p^2$. Since irreducibility is preserved under the invertible substitution $x \mapsto x+1$, $\Phi_p(x)$ is irreducible.

### Reduction Modulo $p$

**Theorem.** Let $f \in \mathbb{Z}[x]$ be a monic polynomial of degree $n$. If $\bar{f} \in \mathbb{F}_p[x]$ (the reduction mod $p$) is irreducible for some prime $p$, then $f$ is irreducible over $\mathbb{Q}$.

*Proof sketch.* If $f = gh$ in $\mathbb{Z}[x]$ with $\deg g, \deg h \geq 1$, then $\bar{f} = \bar{g}\bar{h}$ in $\mathbb{F}_p[x]$ with $\deg \bar{g} = \deg g$ and $\deg \bar{h} = \deg h$ (since $f$ is monic, so are $g$ and $h$, so leading coefficients are not killed by reduction). This contradicts irreducibility of $\bar{f}$.

**Warning:** The converse fails. $f$ can be irreducible over $\mathbb{Q}$ even if $\bar{f}$ is reducible mod every prime $p$. Example: $x^4 + 1$ is irreducible over $\mathbb{Q}$ but reducible mod $p$ for every prime $p$.

**A practical strategy.** When trying to prove a polynomial irreducible over $\mathbb{Q}$, try several methods in order:
1. **Rational root test** (quick, handles degree $\leq 3$ completely).
2. **Eisenstein's criterion** (check small primes; often works after a substitution $x \mapsto x + a$).
3. **Reduction mod $p$** (try small primes; if the reduced polynomial is irreducible over $\mathbb{F}_p$, you are done).
4. **Direct argument** (assume a factorization and derive a contradiction by comparing coefficients).

No single method works universally, but together they handle the vast majority of textbook problems.

**Worked Example 3.** *Show that $f(x) = x^4 + x + 1$ is irreducible over $\mathbb{Q}$.*

*Reduction mod 2:* $\bar{f} = x^4 + x + 1$ in $\mathbb{F}_2[x]$. Check: $\bar{f}(0) = 1 \neq 0$, $\bar{f}(1) = 1 + 1 + 1 = 1 \neq 0$, so no roots. If $\bar{f}$ is reducible, it must factor as a product of two quadratics: $\bar{f} = (x^2 + ax + b)(x^2 + cx + d)$ in $\mathbb{F}_2[x]$. The only irreducible quadratic in $\mathbb{F}_2[x]$ is $x^2 + x + 1$ (since $x^2$, $x^2 + 1 = (x+1)^2$, $x^2 + x = x(x+1)$ are all reducible). So the only possibility is $\bar{f} = (x^2 + x + 1)^2 = x^4 + x^2 + 1 \neq x^4 + x + 1$. Contradiction, so $\bar{f}$ is irreducible over $\mathbb{F}_2$, hence $f$ is irreducible over $\mathbb{Q}$. $\square$

---

## Unique Factorization Domains

**Definition.** An integral domain $R$ is a **Unique Factorization Domain (UFD)** if:
1. **Existence:** Every nonzero non-unit $a \in R$ can be written as a product of irreducible elements: $a = p_1 p_2 \cdots p_k$.
2. **Uniqueness:** If $a = p_1 \cdots p_k = q_1 \cdots q_m$, then $k = m$ and (after reordering) each $p_i$ is an associate of $q_i$ (i.e., $p_i = u_i q_i$ for some unit $u_i$).

**Examples of UFDs:**
- Every PID is a UFD (this takes real work to prove — the key is that irreducible elements in a PID are prime, and prime factorization is always essentially unique).
- $\mathbb{Z}$, $F[x]$ for a field $F$, $\mathbb{Z}[i]$ are all UFDs (as PIDs).
- $\mathbb{Z}[x]$ is a UFD (but not a PID — proved by Gauss's Lemma below).

### The Failure of Unique Factorization: $\mathbb{Z}[\sqrt{-5}]$

The ring $\mathbb{Z}[\sqrt{-5}] = \{a + b\sqrt{-5} : a, b \in \mathbb{Z}\}$ is the standard example of an integral domain where unique factorization fails.

Consider the equation:

$$6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5}).$$

**Claim:** All four factors — $2$, $3$, $1 + \sqrt{-5}$, $1 - \sqrt{-5}$ — are irreducible in $\mathbb{Z}[\sqrt{-5}]$.

*Proof for $2$:* Suppose $2 = \alpha \beta$ with $\alpha, \beta \in \mathbb{Z}[\sqrt{-5}]$. Taking norms (where $N(a + b\sqrt{-5}) = a^2 + 5b^2$), we get $N(2) = 4 = N(\alpha) N(\beta)$. If neither $\alpha$ nor $\beta$ is a unit, then $N(\alpha), N(\beta) > 1$, so $N(\alpha) = N(\beta) = 2$. But $a^2 + 5b^2 = 2$ has no integer solutions (if $b \neq 0$, then $a^2 + 5b^2 \geq 5$; if $b = 0$, then $a^2 = 2$, impossible). Contradiction, so $2$ is irreducible.

Similar arguments show $3$, $1 + \sqrt{-5}$, and $1 - \sqrt{-5}$ are irreducible (their norms are $9$, $6$, $6$ respectively, and in each case the possible factorizations of the norm admit no solution of $a^2 + 5b^2 = $ intermediate value).

Since $2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5})$ and no factor on one side is an associate of a factor on the other, unique factorization fails. This failure motivates the introduction of **ideals** as "ideal numbers" (Kummer's original motivation!) — restoring unique factorization at the level of ideals even when it fails for elements.

**Why does UFD fail in $\mathbb{Z}[\sqrt{-5}]$?** The deeper reason is that $\mathbb{Z}[\sqrt{-5}]$ is not a PID (the ideal $(2, 1 + \sqrt{-5})$ is not principal), and in a PID every irreducible element is prime. An element $p$ is **prime** if $p \mid ab$ implies $p \mid a$ or $p \mid b$. In a UFD, irreducible and prime are equivalent; in a general integral domain they are not. In $\mathbb{Z}[\sqrt{-5}]$, the element $2$ is irreducible but not prime: $2 \mid (1+\sqrt{-5})(1-\sqrt{-5}) = 6$ but $2 \nmid (1+\sqrt{-5})$ and $2 \nmid (1-\sqrt{-5})$ in $\mathbb{Z}[\sqrt{-5}]$. This gap between "irreducible" and "prime" is exactly where unique factorization breaks down.

**The class group** (a topic for a more advanced course) measures how far a ring of algebraic integers is from being a PID. For $\mathbb{Z}[\sqrt{-5}]$, the class number is $2$, reflecting the existence of non-principal ideals like $(2, 1+\sqrt{-5})$. When the class number is $1$, the ring is a PID and unique factorization holds.

---

## From $\mathbb{Z}$ to $\mathbb{Z}[x]$ to $\mathbb{Q}[x]$: Gauss's Lemma and Content

The key question connecting $\mathbb{Z}[x]$ and $\mathbb{Q}[x]$ is: if a polynomial with integer coefficients factors over the rationals, does it factor over the integers? Gauss's Lemma says yes (up to constants).

**Definition.** The **content** of a polynomial $f = a_n x^n + \cdots + a_0 \in \mathbb{Z}[x]$ is $c(f) = \gcd(a_n, \ldots, a_0)$. A polynomial is **primitive** if $c(f) = 1$.

**Gauss's Lemma.** The product of two primitive polynomials is primitive.

*Proof.* Let $f = \sum a_i x^i$ and $g = \sum b_j x^j$ be primitive, and suppose their product $fg$ is not primitive. Then some prime $p$ divides all coefficients of $fg$. Reduce modulo $p$: $\bar{f} \bar{g} = 0$ in $\mathbb{F}_p[x]$. Since $\mathbb{F}_p[x]$ is an integral domain, $\bar{f} = 0$ or $\bar{g} = 0$. But $\bar{f} = 0$ means $p$ divides all coefficients of $f$, contradicting $c(f) = 1$. Similarly for $g$. $\square$

**Corollary (Gauss).** If $f \in \mathbb{Z}[x]$ is primitive and $f = gh$ in $\mathbb{Q}[x]$, then there exist $g^*, h^* \in \mathbb{Z}[x]$ with $f = g^* h^*$ and $\deg g^* = \deg g$, $\deg h^* = \deg h$.

*Proof.* Clear denominators: write $g = \frac{a}{b} g^*$ and $h = \frac{c}{d} h^*$ where $g^*, h^* \in \mathbb{Z}[x]$ are primitive and $a, b, c, d \in \mathbb{Z}$. Then $f = \frac{ac}{bd} g^* h^*$, so $bd \cdot f = ac \cdot g^* h^*$. By Gauss's Lemma, $g^* h^*$ is primitive, and $f$ is primitive, so comparing contents gives $bd = \pm ac$, i.e., $f = \pm g^* h^*$. $\square$

**Theorem.** $\mathbb{Z}[x]$ is a UFD.

*Proof sketch.* We use two facts: (1) $\mathbb{Z}$ is a UFD, and (2) $\mathbb{Q}[x]$ is a UFD (since $\mathbb{Q}$ is a field). For any nonzero $f \in \mathbb{Z}[x]$, write $f = c(f) \cdot f^*$ where $f^*$ is primitive. Factor $c(f)$ into primes in $\mathbb{Z}$ (unique by the Fundamental Theorem of Arithmetic). Factor $f^*$ into irreducibles in $\mathbb{Q}[x]$ (unique since $\mathbb{Q}[x]$ is a PID). By Gauss's Corollary, each factor can be taken in $\mathbb{Z}[x]$ and primitive. Uniqueness follows from the uniqueness in $\mathbb{Z}$ and $\mathbb{Q}[x]$ plus Gauss's Lemma.

More generally: **if $R$ is a UFD, then $R[x]$ is a UFD.** This is the key inductive step that shows $\mathbb{Z}[x_1, \ldots, x_n]$ and $F[x_1, \ldots, x_n]$ are UFDs.

**The full hierarchy, revisited.** We now have concrete witnesses for every strict inclusion:

| Level | Example | Why it is at this level |
|---|---|---|
| Field | $\mathbb{Q}$, $\mathbb{F}_p$ | Every nonzero element invertible |
| Euclidean domain, not a field | $\mathbb{Z}$, $F[x]$ | Division algorithm exists, but not every element invertible |
| PID, not Euclidean | $\mathbb{Z}\left[\frac{1+\sqrt{-19}}{2}\right]$ | Every ideal principal, but no Euclidean function exists |
| UFD, not PID | $\mathbb{Z}[x]$ | Unique factorization holds, but $(2, x)$ is not principal |
| Integral domain, not UFD | $\mathbb{Z}[\sqrt{-5}]$ | No zero divisors, but $6 = 2 \cdot 3 = (1+\sqrt{-5})(1-\sqrt{-5})$ |

Each level carries strictly stronger factorization guarantees than the one below it.

---

## Factorization in Practice: Worked Examples

**Worked Example 4.** *Factor $f(x) = 2x^4 + 3x^3 - 4x^2 - 3x + 2$ over $\mathbb{Q}$.*

*Step 1 (Rational roots):* Possible rational roots are $\pm 1, \pm 2, \pm 1/2$. Check:
- $f(1) = 2 + 3 - 4 - 3 + 2 = 0$. So $(x - 1)$ is a factor.
- $f(-2) = 32 - 24 - 16 + 6 + 2 = 0$. So $(x + 2)$ is a factor.

*Step 2 (Polynomial division):* Divide out $(x-1)(x+2) = x^2 + x - 2$:

$$2x^4 + 3x^3 - 4x^2 - 3x + 2 = (x^2 + x - 2)(2x^2 + x - 1).$$

*Step 3 (Factor the quadratic):* $2x^2 + x - 1 = (2x - 1)(x + 1)$.

*Result:* $f(x) = (x - 1)(x + 2)(2x - 1)(x + 1)$. Over $\mathbb{Z}$, we can also write $f(x) = (x - 1)(x + 2)(x + 1)(2x - 1)$ where all factors are irreducible in $\mathbb{Z}[x]$, and the leading coefficient $2$ can be absorbed into $(2x - 1)$. $\square$

**Worked Example 5.** *Show that $f(x) = x^4 - 10x^2 + 1$ is irreducible over $\mathbb{Q}$.*

*Rational root test:* Possible rational roots are $\pm 1$. $f(1) = 1 - 10 + 1 = -8 \neq 0$, $f(-1) = -8 \neq 0$. So no linear factors over $\mathbb{Q}$.

*Check for quadratic factors over $\mathbb{Q}$:* Suppose $f = (x^2 + ax + b)(x^2 + cx + d)$. Expanding and comparing coefficients:
- $x^3$: $a + c = 0$, so $c = -a$.
- $x^2$: $b + d + ac = b + d - a^2 = -10$.
- $x^1$: $ad + bc = a(d - b) = 0$.
- $x^0$: $bd = 1$.

From $a(d - b) = 0$: either $a = 0$ or $d = b$.

**Case 1:** $a = 0$. Then $b + d = -10$ and $bd = 1$. So $b, d$ are roots of $t^2 + 10t + 1 = 0$, giving $t = \frac{-10 \pm \sqrt{96}}{2} = -5 \pm 2\sqrt{6}$. These are irrational, so no factorization with rational coefficients in this case.

**Case 2:** $d = b$. Then $b^2 = 1$, so $b = \pm 1$. If $b = 1$, then $2 - a^2 = -10$, so $a^2 = 12$, irrational. If $b = -1$, then $-2 - a^2 = -10$, so $a^2 = 8$, irrational.

Both cases fail, so $f$ is irreducible over $\mathbb{Q}$. $\square$

**Remark.** The polynomial $x^4 - 10x^2 + 1$ arises naturally: it is the minimal polynomial of $\sqrt{2} + \sqrt{3}$ over $\mathbb{Q}$. To see this, let $\alpha = \sqrt{2} + \sqrt{3}$. Then $\alpha^2 = 5 + 2\sqrt{6}$, so $\alpha^2 - 5 = 2\sqrt{6}$, hence $(\alpha^2 - 5)^2 = 24$, giving $\alpha^4 - 10\alpha^2 + 1 = 0$. Since $[\mathbb{Q}(\sqrt{2}, \sqrt{3}) : \mathbb{Q}] = 4$, the minimal polynomial has degree $4$, confirming that $f$ is indeed irreducible.

**Worked Example 6.** *Use Eisenstein's criterion to show $f(x) = x^5 + 6x^4 + 9x^3 + 12x + 3$ is irreducible over $\mathbb{Q}$.*

*Solution.* Take $p = 3$. The leading coefficient is $1$, not divisible by $3$. The remaining coefficients: $6 = 3 \cdot 2$, $9 = 3 \cdot 3$, $0$ (the $x^2$ coefficient), $12 = 3 \cdot 4$, $3 = 3 \cdot 1$ — all divisible by $3$. The constant term $3$ is not divisible by $3^2 = 9$. All three conditions of Eisenstein are satisfied, so $f$ is irreducible over $\mathbb{Q}$. $\square$

---

## What's Next

We have now covered the two pillars of elementary ring theory: the general theory of rings, ideals, and quotients (Article 5), and the specific theory of polynomial rings and factorization (this article). The hierarchy

$$\text{Fields} \subset \text{Euclidean Domains} \subset \text{PIDs} \subset \text{UFDs} \subset \text{Integral Domains}$$

is fully in place, with each inclusion strict and each level carrying stronger factorization guarantees.

In the next article, we turn to **field theory and extensions**: given a field $F$, how do we build larger fields? The answer involves quotients of polynomial rings — $F[x]/(f(x))$ — and leads directly to the theory of algebraic extensions, splitting fields, and eventually Galois theory. Everything in these two articles will serve as the algebraic foundation for that journey.

**A preview of what polynomial rings can tell us.** Consider the polynomial $x^2 + 1 \in \mathbb{R}[x]$. Since it is irreducible over $\mathbb{R}$, the quotient $\mathbb{R}[x]/(x^2 + 1)$ is a field — and that field is $\mathbb{C}$. Now consider $x^2 + 1 \in \mathbb{Q}[x]$, also irreducible. The quotient $\mathbb{Q}[x]/(x^2 + 1) \cong \mathbb{Q}(i)$ is a degree-2 extension of $\mathbb{Q}$. In both cases, factorization in $F[x]$ — specifically, which polynomials are irreducible — controls which field extensions we can build. This is the bridge from ring theory to field theory, and from factorization to Galois theory.

---

*Abstract Algebra Series — Article 6 of 12*

---

*This is Part 6 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 5 — Rings and Ideals](/en/abstract-algebra/05-rings-and-ideals/)*

*Next: [Part 7 — Field Extensions](/en/abstract-algebra/07-field-extensions/)*
