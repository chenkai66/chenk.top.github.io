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

Polynomials are the laboratory of algebra. Nearly every concept in ring theory --- ideals, quotients, factorization, irreducibility --- was first understood through polynomial examples before being abstracted. This is no coincidence: polynomial rings are rich enough to exhibit all the interesting phenomena yet structured enough to permit explicit computation.

In this article we study the ring $R[x]$ of polynomials over a ring $R$. We develop the division algorithm, establish irreducibility criteria (Eisenstein, reduction mod $p$, the rational root test), define Unique Factorization Domains, and prove Gauss's Lemma --- the bridge between factorization over $\mathbb{Z}$ and factorization over $\mathbb{Q}$. The payoff is a clear picture of *when* and *why* unique factorization holds, and what goes wrong when it fails.

## Why Polynomials Are Central to Algebra

Mental picture: a polynomial $f(x) \in R[x]$ is a "formal expression" with coefficients in $R$, but it is also a function (when evaluated at points), a generator of an ideal, and the witness to algebraic relations. Each lens reveals a different facet of the same object.

Three reasons polynomials deserve a dedicated study:

1. **Polynomials generate all algebraic extensions.** If $\alpha$ is a root of some polynomial with rational coefficients, then $\mathbb{Q}(\alpha) \cong \mathbb{Q}[x]/(p(x))$ where $p$ is the minimal polynomial of $\alpha$. Polynomial factorization controls field extensions.

2. **Polynomials are a universal testing ground.** $\mathbb{Z}[x]$ is a UFD but not a PID, so we can study the gap between these notions. By Hilbert's Basis Theorem, polynomial rings over Noetherian rings are Noetherian.

3. **Polynomials connect algebra to geometry.** Zeros of polynomials form algebraic varieties --- the starting point of algebraic geometry. Ring-theoretic properties (ideals, quotients) correspond to geometric properties (subvarieties, intersections).

![Factorization of x^4 - 1 over Q displayed as a tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_2_factor_tree.png)

## $R[x]$: Definition, Degree, and the Division Algorithm

Mental picture: addition is coefficientwise; multiplication is "convolution" of coefficient sequences. The degree behaves like an additive valuation on the multiplicative monoid.

**Definition.** Let $R$ be a commutative unital ring. The *polynomial ring* $R[x]$ consists of formal expressions

$$f(x) = a_n x^n + \cdots + a_1 x + a_0$$

with $a_i \in R$. Operations: pointwise addition and convolutional multiplication.

The convolutional multiplication: $(f \cdot g)_k = \sum_{i + j = k} f_i g_j$. So $(2x + 3)(x^2 + x + 1) = 2x^3 + 2x^2 + 2x + 3x^2 + 3x + 3 = 2x^3 + 5x^2 + 5x + 3$. The convolution structure makes polynomial multiplication an associative, distributive operation that mirrors the multiplication of formal power series.

The *degree* $\deg f$ is the largest $n$ with $a_n \neq 0$. *Leading coefficient* is $a_n$; if $a_n = 1$, $f$ is *monic*.

**Degree and multiplication.** If $R$ is an integral domain, $\deg(fg) = \deg f + \deg g$. (Fails for rings with zero divisors: $(2x)(2x) = 0$ in $(\mathbb{Z}/4)[x]$.)

**Consequence.** $R$ integral domain $\Rightarrow$ $R[x]$ integral domain.

**Multivariate.** $R[x_1, \ldots, x_n] := R[x_1, \ldots, x_{n-1}][x_n]$, defined inductively. Fundamental in algebraic geometry.

A monomial in $R[x_1, \ldots, x_n]$ is $x_1^{e_1} \cdots x_n^{e_n}$ with $e_i \in \mathbb{Z}_{\geq 0}$. The set of monomials is in bijection with $\mathbb{Z}_{\geq 0}^n$, and $R[x_1, \ldots, x_n]$ is the free $R$-module on this set. So an element is a finite $R$-linear combination of monomials. The total degree of a monomial is $e_1 + \cdots + e_n$. Multivariate polynomials are the home of such constructions as Gröbner bases (algorithmic basis for ideals) and resultants (eliminating variables from systems).

**Evaluation homomorphism.** For $\alpha \in S \supseteq R$, the map $\text{ev}_\alpha: R[x] \to S, f \mapsto f(\alpha)$ is a ring homomorphism. This connects formal polynomials to functions. Two polynomials may give the same function but be unequal as polynomials: in $\mathbb{F}_2[x]$, $x$ and $x^2$ both define the function $0 \mapsto 0, 1 \mapsto 1$, but $x \neq x^2$.

The kernel of $\text{ev}_\alpha$ is the ideal of polynomials vanishing at $\alpha$. If $S$ is an integral domain and $\alpha$ is a root of some nonzero polynomial in $R[x]$, the kernel is generated by the *minimal polynomial* of $\alpha$ (the monic polynomial of smallest degree with $\alpha$ as a root). For instance, the minimal polynomial of $\sqrt 2$ over $\mathbb{Q}$ is $x^2 - 2$, so $\ker(\text{ev}_{\sqrt 2}) = (x^2 - 2)$ in $\mathbb{Q}[x]$. The quotient $\mathbb{Q}[x]/(x^2 - 2) \cong \mathbb{Q}(\sqrt 2)$ by First Isomorphism. This is the algebraic foundation of the next article on field extensions.

### The Division Algorithm

**Theorem.** Let $F$ be a field, $f, g \in F[x]$, $g \neq 0$. There exist unique $q, r \in F[x]$ with

$$f = qg + r, \quad \deg r < \deg g.$$

*Proof (existence).* Induct on $\deg f$. If $\deg f < \deg g$, take $q = 0, r = f$. Else: let leading terms be $a_n x^n$ for $f$ and $b_m x^m$ for $g$ with $n \geq m$. Set $f_1 = f - (a_n/b_m) x^{n-m} g$. Then $\deg f_1 < n$, apply induction. (Uniqueness: if two decompositions exist, subtract and use degree.) $\square$

The division algorithm requires dividing by leading coefficients, which is why $F$ must be a field.

![Polynomial long division step by step](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_1_division.png)

**Why this matters.** The division algorithm makes $F[x]$ a Euclidean domain (Euclidean function: degree). This is the polynomial analogue of the Fundamental Theorem of Arithmetic and is the technical foundation for all of polynomial UFD theory.

Algorithmic consequence: the Euclidean algorithm computes the GCD of two polynomials in time linear in the maximum degree (with constant-size arithmetic). This is the basis of polynomial GCD computation in computer algebra systems, and underlies many further algorithms: partial fraction decomposition, Sturm chains for real root counting, Berlekamp's factorization algorithm over finite fields.

**Corollary.** $F[x]$ is a Euclidean domain $\Rightarrow$ PID $\Rightarrow$ UFD. Every ideal in $F[x]$ is principal, generated by a unique monic polynomial.

**Factor Theorem.** $\alpha \in F$ is a root of $f \iff (x - \alpha) \mid f$. (By dividing $f$ by $x - \alpha$ and evaluating remainder at $\alpha$.)

**Corollary.** A degree-$n$ polynomial over a field has at most $n$ roots. Fails over $\mathbb{Z}/8$: $x^2 - 1$ has roots $1, 3, 5, 7$.

**Worked Example 1.** Divide $f = x^4 + 2x^3 - x + 3$ by $g = x^2 + x - 1$ in $\mathbb{Q}[x]$.

Step 1: $x^4/x^2 = x^2$. $f - x^2 g = x^3 + x^2 - x + 3$.

Step 2: $x^3/x^2 = x$. Remainder: $3$.

Result: $q = x^2 + x, r = 3$. Verify: $(x^2 + x)(x^2 + x - 1) + 3 = x^4 + 2x^3 - x + 3$. $\checkmark$

**Worked Example 1b.** Divide $f = x^3 + 1$ by $g = x + 1$ in $\mathbb{Q}[x]$. Step 1: $x^3/x = x^2$, $f - x^2 g = -x^2 + 1$. Step 2: $-x^2/x = -x$, remainder $x + 1$. Step 3: $(x+1)/x = 1$ wait that gives a non-polynomial. Let me redo. Step 2: $-x^2 - (-x)(x+1) = -x^2 + x^2 + x = x$. Remainder so far: $x + 1$. Step 3: $x/x = 1$, $(x + 1) - 1 \cdot (x + 1) = 0$. So $q = x^2 - x + 1, r = 0$, confirming $x^3 + 1 = (x + 1)(x^2 - x + 1)$.

**Numerical example: GCD via Euclidean algorithm.** In $\mathbb{Q}[x]$, find $\gcd(x^4 - 1, x^3 - 1)$. Divide: $x^4 - 1 = x \cdot (x^3 - 1) + (x - 1)$. Then $\gcd(x^3 - 1, x - 1)$: $x^3 - 1 = (x^2 + x + 1)(x - 1) + 0$. So $\gcd = x - 1$. Confirms $x^4 - 1 = (x-1)(x+1)(x^2+1)$ and $x^3 - 1 = (x-1)(x^2+x+1)$ share the factor $x-1$.

**Worked Example: Bezout in $\mathbb{Q}[x]$.** From the Euclidean algorithm above, $x - 1 = (x^4 - 1) - x \cdot (x^3 - 1)$. So $1 \cdot (x^4 - 1) + (-x)(x^3 - 1) = x - 1$, expressing the gcd as a linear combination. This is Bezout's identity in the polynomial ring, the engine behind partial fraction decomposition and many tricks in calculus and combinatorics.

**Numerical example: ideals of $\mathbb{Q}[x]$ and divisibility.** $(f) \subseteq (g)$ iff $g \mid f$. So the ideal lattice of $\mathbb{Q}[x]$ is the divisibility lattice of polynomials, dualized. Given two polynomials, their gcd generates $(f) + (g)$ and their lcm generates $(f) \cap (g)$. This is the polynomial analogue of the integer relations $\gcd(m, n)\mathbb{Z} = m\mathbb{Z} + n\mathbb{Z}$ and $\text{lcm}(m, n)\mathbb{Z} = m\mathbb{Z} \cap n\mathbb{Z}$.

## Irreducibility Criteria

Mental picture: an irreducible polynomial is the polynomial analogue of a prime number. Whether a polynomial is irreducible depends crucially on the base ring: $x^2 + 1$ is irreducible over $\mathbb{R}$, factors as $(x+i)(x-i)$ over $\mathbb{C}$, factors over $\mathbb{F}_5$ as $(x+2)(x+3)$.

**Definition.** $f \in R[x]$ of positive degree is *irreducible* over $R$ if $f = gh$ in $R[x]$ implies $g$ or $h$ is a unit.

Over a field $F$, units of $F[x]$ are nonzero constants, so irreducibility means: $f$ does not factor as a product of polynomials of strictly smaller degree.

The dependence on the base ring is not a nuisance: it is the whole point. The question "over which fields does $f$ factor?" is the seed of Galois theory. The factorization pattern of a polynomial mod different primes connects to its Galois group via the Chebotarev density theorem.

![Decision tree for irreducibility tests over Q](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_3_irreducible_test.png)

### The Rational Root Test

**Theorem.** If $f = a_n x^n + \cdots + a_0 \in \mathbb{Z}[x]$ has a rational root $p/q$ in lowest terms, then $p \mid a_0$ and $q \mid a_n$.

*Proof.* Multiply $f(p/q) = 0$ by $q^n$, use coprime $p, q$.

**Worked Example 2.** $f(x) = x^3 - 3x + 1$ irreducible over $\mathbb{Q}$? Possible rational roots: $\pm 1$. $f(1) = -1$, $f(-1) = 3$. No rational roots. Since $\deg f = 3$, reducibility would imply a linear factor, hence a rational root. None exists. So $f$ is irreducible.

**Numerical example: $f(x) = 2x^3 + x^2 - 7x - 6$.** Possible rational roots: $\pm 1, \pm 2, \pm 3, \pm 6, \pm 1/2, \pm 3/2$. Check: $f(2) = 16 + 4 - 14 - 6 = 0$. So $(x - 2)$ is a factor. Divide: $f = (x - 2)(2x^2 + 5x + 3) = (x-2)(2x+3)(x+1)$. Three linear factors over $\mathbb{Z}$.

**Numerical example: rational root in $f = 6x^3 - 7x^2 + 1$.** Possible rational roots: $\pm 1, \pm 1/2, \pm 1/3, \pm 1/6$. $f(1) = 0$. So $(x - 1)$ is a factor. Divide: $f = (x - 1)(6x^2 - x - 1)$. The quadratic factors as $(3x + 1)(2x - 1)$. So $f = (x-1)(3x+1)(2x-1)$.

**Why this matters.** It is the cheapest, fastest test. For a cubic in $\mathbb{Q}[x]$, the rational root test alone determines reducibility (since reducibility forces a linear factor). For higher-degree polynomials, it rules out linear factors but leaves open the possibility of irreducible factors of higher degree.

It also has a meta-lesson: rationality of roots is constrained by integrality of coefficients. A polynomial with integer coefficients lives in a discrete world, and its rational roots must lie on a finite list determined by the coefficients. This finiteness is what makes the test computable.

### Eisenstein's Criterion

**Theorem.** $f = a_n x^n + \cdots + a_0 \in \mathbb{Z}[x]$ ($n \geq 1$). If a prime $p$ satisfies:
1. $p \nmid a_n$,
2. $p \mid a_i$ for $i < n$,
3. $p^2 \nmid a_0$,

then $f$ is irreducible over $\mathbb{Q}$.

*Proof.* If $f = gh$ in $\mathbb{Z}[x]$ (Gauss's Lemma below), reduce mod $p$: $\bar f = \bar a_n x^n$. By unique factorization in $\mathbb{F}_p[x]$, $\bar g$ and $\bar h$ are powers of $x$. So $p \mid g(0), h(0)$, hence $p^2 \mid a_0 = g(0)h(0)$, contradicting (3). $\square$

![Eisenstein criterion applied to a worked polynomial](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_4_eisenstein.png)

**Why this matters.** Eisenstein gives a quick sufficient (not necessary) test. When it works, irreducibility is immediate; when it does not, try a substitution like $x \to x + 1$ or use reduction mod $p$.

Eisenstein in disguise via translation: a polynomial $f(x)$ is irreducible iff $f(x + a)$ is irreducible for any constant $a$. So if Eisenstein fails on $f(x)$, try $f(x + 1), f(x - 1), f(x + 2), \ldots$. Sometimes the right shift unveils an Eisenstein structure that was hiding.

**Example: cyclotomic polynomial.** $\Phi_p(x) = x^{p-1} + \cdots + 1$ is irreducible over $\mathbb{Q}$. Substitute $x \to x + 1$:

$$\Phi_p(x + 1) = \frac{(x+1)^p - 1}{x} = x^{p-1} + \binom{p}{1} x^{p-2} + \cdots + \binom{p}{p-1}.$$

Eisenstein applies with prime $p$: $\binom{p}{k}$ divisible by $p$ for $1 \leq k \leq p-1$, leading coeff $1$, constant term $p$ (not $p^2$).

### Reduction Modulo $p$

**Theorem.** $f \in \mathbb{Z}[x]$ monic, degree $n$. If $\bar f \in \mathbb{F}_p[x]$ is irreducible for some prime $p$, then $f$ is irreducible over $\mathbb{Q}$.

*Proof.* If $f = gh$ in $\mathbb{Z}[x]$ with $\deg g, \deg h \geq 1$ (and monic since $f$ is), then $\bar f = \bar g \bar h$ with same degrees. Contradicts irreducibility of $\bar f$.

Converse fails: $x^4 + 1$ is irreducible over $\mathbb{Q}$ but reducible mod every prime.

**Why this matters.** Reduction mod $p$ is one of the most powerful techniques in number theory: it turns infinite, hard problems (factorization over $\mathbb{Z}$) into finite, mechanical problems (factorization over $\mathbb{F}_p$). When it works, it gives a quick proof of irreducibility. The strategy is also the prototype for modern arithmetic geometry: study global problems by reducing modulo each prime $p$ and assembling the local data.

**Strategy for irreducibility:**
1. Rational root test (handles degree $\leq 3$ completely).
2. Eisenstein (try small primes; substitute $x \to x + a$ if needed).
3. Reduction mod small primes.
4. Direct coefficient comparison.

For higher-degree polynomials, sometimes none of these suffice. In that case, the structure of the polynomial may give clues: cyclotomic polynomials, polynomials of the form $f(x) g(x) + 1$, or polynomials related to linear recurrences. Specialized techniques (Newton polygons, $p$-adic analysis) handle the harder cases. Computer algebra systems like Pari/GP, SageMath, and Magma can factor large-degree polynomials over many rings algorithmically; for "is this irreducible" questions in research, these tools are indispensable.

**Worked Example 3.** $f = x^4 + x + 1$ irreducible over $\mathbb{Q}$?

Mod 2: $\bar f = x^4 + x + 1$ has no roots in $\mathbb{F}_2$ ($\bar f(0) = 1, \bar f(1) = 1$). If it factors, must be product of two quadratics. Only irreducible quadratic in $\mathbb{F}_2[x]$ is $x^2 + x + 1$. So would need $\bar f = (x^2 + x + 1)^2 = x^4 + x^2 + 1 \neq x^4 + x + 1$. Contradiction. So $\bar f$ irreducible over $\mathbb{F}_2$, hence $f$ irreducible over $\mathbb{Q}$.

## Unique Factorization Domains

Mental picture: a UFD is an integral domain where every element has a "prime factorization" unique up to units and reordering. The arithmetic of UFDs is essentially as well-behaved as that of $\mathbb{Z}$.

**Definition.** Integral domain $R$ is a UFD if every nonzero non-unit has a factorization into irreducibles, unique up to associates and order.

**Examples:** $\mathbb{Z}$, $F[x]$ for $F$ a field, $\mathbb{Z}[i]$, $\mathbb{Z}[x]$.

For polynomial rings: $\mathbb{Q}[x], \mathbb{R}[x], \mathbb{C}[x], \mathbb{F}_p[x]$ are all UFDs (and PIDs, since the base is a field). $\mathbb{Z}[x]$ is a UFD but not a PID. $\mathbb{Q}[x, y]$ is a UFD but not a PID. The hierarchy of "good" rings stratifies into infinitely many levels as soon as we go past one variable.

**Theorem.** Every PID is a UFD.

The proof has two parts: existence (every nonzero non-unit factors) uses the ascending chain condition (no infinite chain of proper divisors). Uniqueness uses the fact that in a PID, irreducible elements are prime ($p \mid ab \Rightarrow p \mid a$ or $p \mid b$).

**Sketch of existence.** Suppose $a$ has no factorization. Then $a = a_1 b_1$ with $a_1, b_1$ non-units, and (WLOG) $a_1$ has no factorization. Iterate to get $a, a_1, a_2, \ldots$ with $(a) \subsetneq (a_1) \subsetneq (a_2) \subsetneq \cdots$, an ascending chain. PID is Noetherian, contradiction.

**Sketch of uniqueness.** In a PID, given an irreducible $p$, the ideal $(p)$ is maximal (any larger principal ideal $(a)$ with $p \in (a)$ has $p = ax$, so by irreducibility $a$ or $x$ is a unit, giving $(a) = (p)$ or $(a) = R$). Maximal $\Rightarrow$ prime. So in a PID, irreducible $\Rightarrow$ prime, which gives uniqueness of factorization by the standard argument: if $p_1 \cdots p_k = q_1 \cdots q_m$, then $p_1 \mid q_1 \cdots q_m$, so by primality $p_1 \mid q_j$ for some $j$, i.e., $p_1$ is associate to $q_j$. Cancel and induct.

![Unique factorization in a UFD as a finite chain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_5_ufd_chain.png)

**Why this matters.** UFD structure makes "elementary number theory" generalize to other rings: gcd, lcm, prime factorization all work. Failure of UFD (as in $\mathbb{Z}[\sqrt{-5}]$) signals deeper arithmetic complexity, addressed by Dedekind's ideal theory.

Practical consequence: in a UFD $R$, the rational function field $\text{Frac}(R)$ has clean partial fraction decomposition. Every element of $\text{Frac}(R)$ can be written as a sum of terms of the form $u/p^k$ for primes $p$. This is the algebraic root of all the partial-fraction tricks of calculus, generalized to arbitrary UFDs.

### Failure: $\mathbb{Z}[\sqrt{-5}]$

$6 = 2 \cdot 3 = (1 + \sqrt{-5})(1 - \sqrt{-5})$.

**Claim.** All four factors are irreducible.

*For $2$:* If $2 = \alpha\beta$, then $N(\alpha)N(\beta) = N(2) = 4$. If neither is a unit, $N(\alpha) = N(\beta) = 2$. But $a^2 + 5b^2 = 2$ has no integer solutions. So $2$ is irreducible. Similarly for the others.

The four factors are pairwise non-associate (different norms or norms not related by a unit). So $6$ has two essentially distinct irreducible factorizations. UFD fails.

**Why?** In $\mathbb{Z}[\sqrt{-5}]$, irreducible $\not\Rightarrow$ prime. $2$ is irreducible but $2 \mid (1+\sqrt{-5})(1-\sqrt{-5}) = 6$ while $2 \nmid (1 \pm \sqrt{-5})$ in $\mathbb{Z}[\sqrt{-5}]$. The gap between "irreducible" and "prime" is exactly where UFD breaks.

**Class group.** Measures how far a number ring is from being a PID. Class number 1 means PID; for $\mathbb{Z}[\sqrt{-5}]$ class number is $2$.

**Restoring uniqueness via ideals.** Dedekind's insight: although elements may have multiple irreducible factorizations, *ideals* factor uniquely as products of prime ideals in any "Dedekind domain" (a class containing $\mathbb{Z}[\sqrt{-5}]$ and most rings of algebraic integers). For $\mathbb{Z}[\sqrt{-5}]$:

$$(2) = \mathfrak{p}^2, \quad (3) = \mathfrak{q}_1 \mathfrak{q}_2, \quad (1 + \sqrt{-5}) = \mathfrak{p}\mathfrak{q}_1, \quad (1 - \sqrt{-5}) = \mathfrak{p}\mathfrak{q}_2,$$

where $\mathfrak{p} = (2, 1 + \sqrt{-5}), \mathfrak{q}_1 = (3, 1 + \sqrt{-5}), \mathfrak{q}_2 = (3, 1 - \sqrt{-5})$. So $(6) = \mathfrak{p}^2 \mathfrak{q}_1 \mathfrak{q}_2$, the same ideal factorization regardless of which element-level factorization you start from. The "ambiguity" lives in how you group primes into elements, not in the prime structure itself.

## From $\mathbb{Z}$ to $\mathbb{Z}[x]$ to $\mathbb{Q}[x]$: Gauss's Lemma

**Definition.** Content of $f \in \mathbb{Z}[x]$: $c(f) = \gcd$ of coefficients. $f$ is *primitive* if $c(f) = 1$.

For polynomials over a UFD $R$, content is defined as the gcd of coefficients (well-defined up to units). The content extends to a multiplicative function: $c(fg) = c(f) c(g)$ (this is the content version of Gauss's Lemma). Every nonzero $f \in R[x]$ factors uniquely (up to units) as $f = c(f) \cdot f^*$ with $f^*$ primitive.

**Gauss's Lemma.** Product of primitive polynomials is primitive.

*Proof.* If $fg$ has all coefficients divisible by prime $p$, reduce mod $p$: $\bar f \bar g = 0$ in $\mathbb{F}_p[x]$. Integral domain $\Rightarrow$ $\bar f = 0$ or $\bar g = 0$, contradicting primitivity. $\square$

**Corollary.** $f \in \mathbb{Z}[x]$ primitive, $f = gh$ in $\mathbb{Q}[x]$ $\Rightarrow$ exists $g^*, h^* \in \mathbb{Z}[x]$ with $f = g^* h^*$, same degrees.

**Theorem.** $\mathbb{Z}[x]$ is a UFD. More generally, $R$ UFD $\Rightarrow$ $R[x]$ UFD. By induction $R[x_1, \ldots, x_n]$ is UFD.

*Proof sketch for $\mathbb{Z}[x]$.* Use $\mathbb{Z}$ UFD and $\mathbb{Q}[x]$ UFD. For nonzero $f \in \mathbb{Z}[x]$, write $f = c(f) \cdot f^*$ where $f^*$ is primitive. Factor $c(f)$ via $\mathbb{Z}$-UFD, factor $f^*$ via $\mathbb{Q}[x]$-UFD. By Gauss's Corollary, the factors of $f^*$ can be taken in $\mathbb{Z}[x]$ and primitive. Uniqueness: combine $\mathbb{Z}$-uniqueness, $\mathbb{Q}[x]$-uniqueness, and Gauss's Lemma.

The same argument works for any UFD $R$ in place of $\mathbb{Z}$, using $\text{Frac}(R)$ in place of $\mathbb{Q}$.

**Why this matters.** Gauss's Lemma is the bridge between integer-coefficient and rational-coefficient factorization. Without it, the question "is this $\mathbb{Z}[x]$-polynomial irreducible?" would be much harder to answer using $\mathbb{Q}[x]$ tools.

A second consequence: it tells us that "factors clear denominators." If $f \in \mathbb{Z}[x]$ is monic and factors over $\mathbb{Q}$ into monic factors, those factors are automatically in $\mathbb{Z}[x]$. This is why the rational root test produces *integer* roots for monic integer polynomials, not just rational ones.

### Hierarchy Recap

| Level | Example | Why this level |
|-------|---------|----------------|
| Field | $\mathbb{Q}, \mathbb{F}_p$ | Every nonzero element invertible |
| Euclidean, not field | $\mathbb{Z}, F[x]$ | Division algorithm exists |
| PID, not Euclidean | $\mathbb{Z}[(1+\sqrt{-19})/2]$ | Every ideal principal, no Euclidean function |
| UFD, not PID | $\mathbb{Z}[x]$ | Unique factorization holds, $(2, x)$ not principal |
| Integral domain, not UFD | $\mathbb{Z}[\sqrt{-5}]$ | $6 = 2 \cdot 3 = (1+\sqrt{-5})(1-\sqrt{-5})$ |

![The grid structure of monomials inside Z[x]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_6_poly_grid.png)

## Factorization in Practice

**Worked Example 4.** Factor $f = 2x^4 + 3x^3 - 4x^2 - 3x + 2$ over $\mathbb{Q}$.

Rational root test: candidates $\pm 1, \pm 2, \pm 1/2$. $f(1) = 0$, $f(-2) = 0$. Divide out $(x-1)(x+2) = x^2 + x - 2$:

$$f = (x^2 + x - 2)(2x^2 + x - 1) = (x-1)(x+2)(2x-1)(x+1).$$

Sanity check: expand $(x-1)(x+2)(2x-1)(x+1) = (x^2+x-2)(2x^2+x-1)$. Multiply: $2x^4 + x^3 - x^2 + 2x^3 + x^2 - x - 4x^2 - 2x + 2 = 2x^4 + 3x^3 - 4x^2 - 3x + 2$. $\checkmark$

**Worked Example 5.** $f = x^4 - 10x^2 + 1$ irreducible over $\mathbb{Q}$?

Rational root test: $\pm 1$, neither works.

Quadratic factors: try $(x^2 + ax + b)(x^2 + cx + d)$. Comparing: $a + c = 0$ so $c = -a$. $bd = 1$. $a(d - b) = 0$ so $a = 0$ or $d = b$.

Case $a = 0$: $b + d = -10, bd = 1$. Roots of $t^2 + 10t + 1$: $-5 \pm 2\sqrt 6$, irrational.

Case $d = b$: $b^2 = 1$, $b = \pm 1$. $a^2 = 12$ or $a^2 = 8$, both irrational.

So $f$ is irreducible. (This is the minimal polynomial of $\sqrt 2 + \sqrt 3$.)

**Worked Example 5b.** Verify that $\sqrt 2 + \sqrt 3$ is a root of $x^4 - 10x^2 + 1$. Let $\alpha = \sqrt 2 + \sqrt 3$. $\alpha^2 = 2 + 2\sqrt 6 + 3 = 5 + 2\sqrt 6$. $\alpha^2 - 5 = 2\sqrt 6$. $(\alpha^2 - 5)^2 = 24$. So $\alpha^4 - 10\alpha^2 + 25 = 24$, i.e., $\alpha^4 - 10\alpha^2 + 1 = 0$. $\checkmark$

Since $\mathbb{Q}(\sqrt 2, \sqrt 3)$ has degree $4$ over $\mathbb{Q}$ (the four basis elements $1, \sqrt 2, \sqrt 3, \sqrt 6$ are linearly independent), the minimal polynomial of $\alpha$ has degree dividing $4$. Since $\alpha \notin \mathbb{Q}$ and $\alpha^2 \notin \mathbb{Q}$, the minimal polynomial has degree $> 2$. Combined with $f(\alpha) = 0$ and $\deg f = 4$, $f$ is the minimal polynomial.

**Worked Example 6.** $f = x^5 + 6x^4 + 9x^3 + 12x + 3$. Eisenstein with $p = 3$: leading coeff $1$ ($3 \nmid 1$); $6, 9, 0, 12, 3$ all divisible by $3$; constant $3$ not divisible by $9$. Eisenstein applies, so $f$ is irreducible.

**Worked Example: when Eisenstein fails directly.** $f = x^4 + 1$. No prime $p$ makes Eisenstein work directly. But the substitution $x \to x + 1$: $(x+1)^4 + 1 = x^4 + 4x^3 + 6x^2 + 4x + 2$. Eisenstein with $p = 2$: leading $1$, others $4, 6, 4, 2$ all divisible by $2$, constant $2$ not by $4$. So $(x+1)^4 + 1$ is irreducible, hence $x^4 + 1$ is irreducible. (Substitution preserves irreducibility because $x \to x + 1$ is a ring automorphism of $\mathbb{Q}[x]$.)

**Worked Example: $\Phi_5(x) = x^4 + x^3 + x^2 + x + 1$.** Eisenstein with substitution: $\Phi_5(x+1) = ((x+1)^5 - 1)/x$. Direct expansion: $(x+1)^5 - 1 = x^5 + 5x^4 + 10x^3 + 10x^2 + 5x$. Divide by $x$: $x^4 + 5x^3 + 10x^2 + 10x + 5$. Eisenstein with $p = 5$: all the requirements check. So $\Phi_5$ is irreducible. This is the special case $p = 5$ of the general fact about cyclotomic polynomials.

**Worked Example 7.** Factor $x^6 - 1$ over $\mathbb{Q}$. Roots are 6th roots of unity. Factorization:

$$x^6 - 1 = (x^3 - 1)(x^3 + 1) = (x - 1)(x^2 + x + 1)(x + 1)(x^2 - x + 1).$$

Four irreducible factors. Each cyclotomic polynomial $\Phi_d$ for $d \mid 6$ contributes once: $\Phi_1 = x - 1, \Phi_2 = x + 1, \Phi_3 = x^2 + x + 1, \Phi_6 = x^2 - x + 1$.

The general formula: $x^n - 1 = \prod_{d \mid n} \Phi_d(x)$. For $n = 12$: $x^{12} - 1 = \Phi_1 \Phi_2 \Phi_3 \Phi_4 \Phi_6 \Phi_{12}$. Degrees: $1 + 1 + 2 + 2 + 2 + 4 = 12$. $\checkmark$

This factorization is the algebraic shadow of the geometric fact that the $n$-th roots of unity are partitioned by their order $d \mid n$. Each $\Phi_d$ collects the primitive $d$-th roots.

![Roots of x^3 - 2 displayed in the complex plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-polynomial-rings/aa_v2_06_7_root_complex.png)

**Worked Example 8.** Roots of $x^3 - 2$ over $\mathbb{C}$: $\sqrt[3]{2}, \sqrt[3]{2}\omega, \sqrt[3]{2}\omega^2$ where $\omega = e^{2\pi i/3}$. These three points sit at the corners of an equilateral triangle in the complex plane, on the circle of radius $\sqrt[3]{2}$. Over $\mathbb{Q}$, $x^3 - 2$ is irreducible (Eisenstein with $p = 2$). The splitting field is $\mathbb{Q}(\sqrt[3]{2}, \omega)$, a degree-$6$ extension of $\mathbb{Q}$, with Galois group $S_3$.

**Worked Example 9.** Factor $x^4 - 4$ over $\mathbb{Q}$ and over $\mathbb{R}$. Over $\mathbb{Q}$: $x^4 - 4 = (x^2 - 2)(x^2 + 2)$. Both factors are irreducible over $\mathbb{Q}$ (discriminant $\pm 8$). Over $\mathbb{R}$: $x^2 - 2 = (x - \sqrt 2)(x + \sqrt 2)$, but $x^2 + 2$ remains irreducible (no real roots). Over $\mathbb{C}$: complete factorization into linear factors $(x - \sqrt 2)(x + \sqrt 2)(x - i\sqrt 2)(x + i\sqrt 2)$. The factorization gets finer as the field gets larger.

**Worked Example 10.** Factor $x^4 + 4$ over $\mathbb{Z}$. Try Sophie Germain: $x^4 + 4 = x^4 + 4x^2 + 4 - 4x^2 = (x^2 + 2)^2 - (2x)^2 = (x^2 + 2x + 2)(x^2 - 2x + 2)$. Two quadratic factors. Each is irreducible over $\mathbb{Z}$ (discriminants $4 - 8 = -4$, no real roots). So $x^4 + 4$ is *reducible* over $\mathbb{Z}$, even though $x^4 + 1$ is irreducible. The factorization is non-obvious; this is one of the classical traps in irreducibility problems.

**Worked Example 11.** Factor $f(x, y) = x^2 - y^2$ in $\mathbb{Z}[x, y]$. Use difference of squares: $x^2 - y^2 = (x - y)(x + y)$. Both factors are irreducible (degree 1, content 1). So $f$ has two irreducible factors. By contrast, $x^2 + y^2$ is irreducible in $\mathbb{Z}[x, y]$ (because it is irreducible in $\mathbb{R}[x, y]$, hence in $\mathbb{Q}[x, y]$, hence by Gauss's Lemma in $\mathbb{Z}[x, y]$).

**Worked Example 12.** Factor $f(x, y, z) = x^2 + y^2 + z^2 - xy - yz - xz$ over $\mathbb{Q}$. This is the symmetric quadratic form. Try $f(x, x, x) = 3x^2 - 3x^2 = 0$, so $f$ vanishes on the diagonal. This suggests $(x - y)$ is a factor of $f$ when we view $f$ as a polynomial in $x$. Compute: $f$ as polynomial in $x$ is $x^2 - (y + z) x + (y^2 + z^2 - yz)$. Discriminant: $(y + z)^2 - 4(y^2 + z^2 - yz) = y^2 + 2yz + z^2 - 4y^2 - 4z^2 + 4yz = -3y^2 + 6yz - 3z^2 = -3(y - z)^2$. So roots are $x = ((y + z) \pm (y - z)\sqrt{-3})/2$. Not in $\mathbb{Q}$. So $f$ is irreducible over $\mathbb{Q}$. Over $\mathbb{Q}(\sqrt{-3})$, it factors into two linear factors in $x$ (with coefficients in $\mathbb{Q}(\sqrt{-3})[y, z]$).

**Worked Example 13.** $f(x) = x^p - x \in \mathbb{F}_p[x]$ for prime $p$. Over $\mathbb{F}_p$, every element of $\mathbb{F}_p$ is a root (by Fermat's little theorem: $a^p = a$ for $a \in \mathbb{F}_p$). So $f(x) = \prod_{a \in \mathbb{F}_p}(x - a)$, a product of $p$ linear factors. The polynomial $x^p - x$ has all of $\mathbb{F}_p$ as its roots, which is unusual: a degree-$p$ polynomial with $p$ roots in a field of $p$ elements.

**Worked Example 14.** Factor $x^4 + x^3 + x^2 + x + 1$ over $\mathbb{F}_2$. This is $\Phi_5(x)$, the $5$-th cyclotomic polynomial. Over $\mathbb{F}_2$, look at the order of $2$ mod $5$: $2, 4, 3, 1$ (powers of 2 mod 5), so order $4$. By the theory of cyclotomic polynomials over finite fields, $\Phi_5(x)$ is irreducible over $\mathbb{F}_2$ iff $\text{ord}_5(2) = 4 = \deg \Phi_5$. Yes. So $\Phi_5(x)$ is irreducible over $\mathbb{F}_2$. Verify directly: it has no roots in $\mathbb{F}_2$ ($f(0) = 1, f(1) = 5 = 1$); if it factored as two quadratics, the only irreducible quadratic over $\mathbb{F}_2$ is $x^2 + x + 1$, and $(x^2 + x + 1)^2 = x^4 + x^2 + 1$, not equal to $\Phi_5$. So irreducible.

**Worked Example 15.** Factor $x^9 - x$ over $\mathbb{F}_3$. Over $\mathbb{F}_9$, $x^9 - x$ has all $9$ elements of $\mathbb{F}_9$ as roots, so $x^9 - x = \prod_{\alpha \in \mathbb{F}_9}(x - \alpha)$ over $\mathbb{F}_9$. Over $\mathbb{F}_3$, group the roots by their orbit under the Frobenius $\alpha \mapsto \alpha^3$. The fixed points are $\mathbb{F}_3$ (3 elements, contributing 3 linear factors). The remaining $6$ elements split into $3$ orbits of size $2$ under Frobenius (since their minimal polynomial has degree $2$). So the factorization is $3$ linear factors $\times$ $3$ irreducible quadratic factors. Total degree: $3 \cdot 1 + 3 \cdot 2 = 9$. $\checkmark$

## What's Next

We have covered the two pillars of elementary ring theory: general rings/ideals/quotients (Article 5), and polynomial rings/factorization (this article). The hierarchy

$$\text{Fields} \subset \text{Euclidean Domains} \subset \text{PIDs} \subset \text{UFDs} \subset \text{Integral Domains}$$

is fully in place, with strict inclusions and increasing factorization guarantees.

The next article tackles *field theory and extensions*: how do we build larger fields from smaller ones? The answer involves quotients of polynomial rings $F[x]/(f(x))$ and leads to the theory of algebraic extensions, splitting fields, and Galois theory.

A summary worth keeping: polynomial rings turn algebra into a calculation game. The division algorithm is the engine, factorization is the goal, and irreducibility tests are the toolkit. When everything works, we get UFD; when it does not, we have entered the deeper waters of algebraic number theory.

**A preview of where this leads.** $x^2 + 1 \in \mathbb{R}[x]$ is irreducible, so $\mathbb{R}[x]/(x^2 + 1) \cong \mathbb{C}$ is a field. $x^2 + 1 \in \mathbb{Q}[x]$ is also irreducible, so $\mathbb{Q}[x]/(x^2 + 1) \cong \mathbb{Q}(i)$ is a degree-$2$ field extension of $\mathbb{Q}$. In both cases, *which* polynomials are irreducible determines *which* field extensions exist. Galois theory studies the symmetry group of these extensions, and the bridge between polynomial factorization and group theory is the deepest single idea in classical algebra. This is where we are headed.

**Reading recommendations.** Dummit and Foote chapters 9-10 cover polynomial rings and field extensions. Hungerford's *Algebra* is a denser but excellent treatment. For a computational angle, Cox-Little-O'Shea's *Ideals, Varieties, and Algorithms* introduces Gröbner bases and the algorithmic side of multivariate polynomial rings. For irreducibility tests with many worked examples, Lang's *Algebra* chapter 4 is a classic.

**One last reflection.** Polynomial factorization is one of those subjects where every textbook lays out the same theorems but the *practice* requires hands-on familiarity. The way to internalize the irreducibility tests is not to memorize them but to apply them on a hundred examples. After fifty, you start recognizing the structures: cyclotomic shapes, Eisenstein candidates after substitution, polynomials that factor mod $2$ but not mod $3$. The pattern recognition is the real skill, and the theorems are scaffolding for it.

A second reflection: the theory of polynomial rings is the workshop where modern algebra was forged. Galois died at age 20 with a sketch of polynomial Galois theory; Dedekind invented ideals to handle factorization in number rings; Hilbert's Basis Theorem started commutative algebra; Noether's reorganization of all of this around quotient rings and modules is the language of the modern subject. Every name you encounter in classical algebra has, somewhere, a polynomial ring at the heart of their story.

A final concrete tip: when stuck on a polynomial irreducibility problem, try (in order) the rational root test, several reductions mod small primes, Eisenstein at a few primes (with substitution if needed), and direct coefficient comparison. If still stuck, look for special structure: cyclotomic factors, products of binomials, polynomials of the form $f(x^k)$ for some other $f$. Most textbook problems yield to one of these approaches.

---

*This is Part 6 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 5 — Rings and Ideals](/en/abstract-algebra/05-rings-and-ideals/)*

*Next: [Part 7 — Field Extensions](/en/abstract-algebra/07-field-extensions/)*
