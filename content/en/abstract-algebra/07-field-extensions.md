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

Every mathematician, at some point, encounters a polynomial that refuses to be solved within the number system at hand. The ancient Greeks discovered that $\sqrt{2}$ is irrational — that is, $x^2 - 2$ has no solution in $\mathbb{Q}$. The resolution was not to abandon the polynomial, but to enlarge the field. Field extensions formalize this enlargement and give us the structural scaffolding on which Galois theory is built.

I find it useful to think of a field extension as a kind of controlled inflation of a number system. We pump in just enough new elements to solve the equations we care about, and the tower law tells us exactly how much air we used. The resulting picture is much cleaner than I expected when I first met it: every step has a finite degree, the degrees multiply along chains, and the whole thing turns into linear algebra over the base field. This article develops the theory from the ground up: degrees and bases, simple extensions and minimal polynomials, the tower law, splitting fields, and separability. By the end, we will have the full toolkit needed to state and prove the Fundamental Theorem of Galois Theory in the next article.

---

![Finite fields GF(p^n) structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_finite_fields.png)


## Motivation: Solving Polynomials Requires Bigger Fields

Consider the polynomial $f(x) = x^2 + 1$ over $\mathbb{R}$. It has no real roots, since $x^2 \geq 0$ for all $x \in \mathbb{R}$. But if we pass to the larger field $\mathbb{C} = \mathbb{R}(i)$, the polynomial factors as $(x - i)(x + i)$. The strategy is brutally simple: if your equation has no solution, build a field where it does.

This situation is ubiquitous in algebra:

- $x^2 - 2$ has no root in $\mathbb{Q}$, but has roots $\pm\sqrt{2}$ in $\mathbb{Q}(\sqrt{2})$.
- $x^2 - 5$ has no root in $\mathbb{Q}(\sqrt{2})$, but does in $\mathbb{Q}(\sqrt{2}, \sqrt{5})$.
- $x^3 - 2$ has no root in $\mathbb{Q}$, but has a real root $\sqrt[3]{2}$ in $\mathbb{Q}(\sqrt[3]{2})$ and all three roots in $\mathbb{Q}(\sqrt[3]{2}, \omega)$, where $\omega = e^{2\pi i/3}$ is a primitive cube root of unity.
- $x^p - x - 1$ over $\mathbb{F}_p$ never factors completely until you adjoin a root from $\overline{\mathbb{F}_p}$.

The pattern is always the same: given a polynomial over a field $K$ that we cannot factor completely, we build a bigger field $L \supseteq K$ in which the polynomial does factor. The theory of field extensions makes this process precise, answering three fundamental questions: how do we construct these larger fields, how big are they relative to the base field, and when does a minimal such extension exist?

Historically, this line of thinking emerged from centuries of attempts to find root formulas for polynomials. The quadratic formula works in degree 2. Cardano's formula handles degree 3. Ferrari's method extends to degree 4. But degree 5 resisted all attacks. Understanding *why* required a completely new perspective — not on the roots themselves, but on the symmetries of the field extensions they generate. Field extensions are thus not merely a technical convenience; they are the language in which the deepest structural results of algebra are expressed.

![Tower of field extensions: Q ⊂ Q(√2) ⊂ Q(√2,√3)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_1_extension_tower.png)

A small concrete computation to anchor things. Take $K = \mathbb{Q}$ and $L = \mathbb{Q}(\sqrt{2})$. Every element of $L$ has the form $a + b\sqrt{2}$ with $a, b \in \mathbb{Q}$. Multiplication: $(a + b\sqrt{2})(c + d\sqrt{2}) = (ac + 2bd) + (ad + bc)\sqrt{2}$. Inversion: $(a + b\sqrt{2})^{-1} = \frac{a - b\sqrt{2}}{a^2 - 2b^2}$, well-defined as long as $a^2 - 2b^2 \neq 0$, which holds whenever $(a,b) \neq (0,0)$ since $\sqrt{2}$ is irrational. So $L$ really is a field, and as a $\mathbb{Q}$-vector space it has basis $\{1, \sqrt{2}\}$.

**Why this matters.** Once you accept that "solve a polynomial" means "build the right field," a huge swath of classical mathematics becomes uniform. Trisecting an angle, doubling a cube, constructing a regular 17-gon, deciding whether a quintic admits a closed-form solution — all of these reduce to questions about the existence and degree of certain extensions. The Greeks had no way to phrase "$60^\circ$ cannot be trisected" as a clean theorem; we do, and the language is field extensions. The conceptual leap from "find a number" to "construct a field" is the whole point of the subject.

There is also a useful bookkeeping benefit. When you start hopping between $\mathbb{Q}$, $\mathbb{Q}(\sqrt{2})$, $\mathbb{Q}(\sqrt{2}, i)$, and $\overline{\mathbb{Q}}$, you can lose track of which arithmetic identities hold where. Treating each field as a vector space over the previous one gives you dimension-counting as a built-in sanity check: if the degrees don't multiply correctly, you have a bug somewhere in your derivation. I have caught more than one mistake of my own this way.

---

## Field Extensions and Degree

**Definition.** A *field extension* is a pair of fields $K \subseteq L$ (equivalently, an injective field homomorphism $K \hookrightarrow L$). We write $L/K$ and call $K$ the *base field* (or *ground field*) and $L$ the *extension field*. The notation $L/K$ does not mean a quotient — it is simply a conventional way to indicate that $L$ extends $K$.

![Degree formula: [L:K] = [L:F][F:K]](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_degree_formula.png)


Since $L$ is a field containing $K$, it carries the structure of a vector space over $K$: addition in $L$ is the vector addition, and scalar multiplication by elements of $K$ is given by the field multiplication in $L$. This is the linchpin that lets us bring linear algebra to bear on what looks like a question about roots of polynomials. The *degree* of the extension is

$$[L : K] = \dim_K L,$$

the dimension of $L$ as a $K$-vector space. If $[L:K]$ is finite, we say $L/K$ is a *finite extension*; otherwise, it is an *infinite extension*.

**Example 1 ($\mathbb{C}/\mathbb{R}$, degree 2).** Every complex number can be written as $a + bi$ with $a, b \in \mathbb{R}$. The set $\{1, i\}$ is linearly independent over $\mathbb{R}$ (since $a + bi = 0$ with $a,b$ real forces $a = b = 0$) and spans $\mathbb{C}$. So $[\mathbb{C}:\mathbb{R}] = 2$.

**Example 2 ($\mathbb{Q}(\sqrt{2})/\mathbb{Q}$, degree 2).** Define $\mathbb{Q}(\sqrt{2}) = \{a + b\sqrt{2} : a, b \in \mathbb{Q}\}$. This is indeed a field: it is closed under addition and multiplication (using $(\sqrt{2})^2 = 2$), and inverses exist because $1/(a + b\sqrt{2}) = (a - b\sqrt{2})/(a^2 - 2b^2)$, and the denominator is nonzero when $(a,b) \neq (0,0)$ since $\sqrt{2}$ is irrational. A basis over $\mathbb{Q}$ is $\{1, \sqrt{2}\}$.

**Example 3 ($\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$, degree 3).** $\{1, \sqrt[3]{2}, \sqrt[3]{4}\}$ is a $\mathbb{Q}$-basis. Linear independence follows because the minimal polynomial of $\sqrt[3]{2}$ over $\mathbb{Q}$ is $x^3 - 2$, which is irreducible by Eisenstein at $p = 2$; a root of an irreducible cubic cannot satisfy a quadratic.

**Example 4 ($\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$, degree 4).** Basis $\{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$. Independence: suppose $a + b\sqrt{2} + c\sqrt{3} + d\sqrt{6} = 0$, regroup as $(a+b\sqrt{2}) + \sqrt{3}(c+d\sqrt{2}) = 0$. Since $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$ (otherwise $\sqrt{3} = p + q\sqrt{2}$ would force $3 = p^2 + 2q^2 + 2pq\sqrt{2}$, hence $pq = 0$, leading to a contradiction in either branch), both bracketed terms vanish, and Example 2 finishes the job.

**Example 5 ($\mathbb{R}/\mathbb{Q}$, infinite).** $\mathbb{R}$ is uncountable, while any finite-dimensional vector space over $\mathbb{Q}$ is countable. More concretely, the elements $1, \sqrt{2}, \sqrt{3}, \sqrt{5}, \sqrt{7}, \ldots$ (square roots of distinct primes) are $\mathbb{Q}$-linearly independent, which already produces an infinite-dimensional subspace.

**Example 6 ($\mathbb{C}/\mathbb{Q}$, infinite).** Same argument as for $\mathbb{R}/\mathbb{Q}$.

**Why this matters.** The degree is the single most informative invariant of an extension. It tells us how much the field grew, it bounds the number of automorphisms (a fact we will exploit ruthlessly in the Galois correspondence), and via the tower law it composes multiplicatively, turning what could be a tangled lattice computation into arithmetic on integers. When you see "degree" in the rest of this article, read it as "the structural dial we will turn."

Two practical heuristics worth internalizing now:

1. *To lower-bound a degree, exhibit linearly independent elements.* If you can show that $1, \alpha, \alpha^2, \ldots, \alpha^{k-1}$ are $K$-independent, then $[K(\alpha) : K] \geq k$. This is usually the easy half; people get tripped up on the next step.
2. *To upper-bound a degree, exhibit a polynomial.* If $\alpha$ satisfies a degree-$k$ polynomial in $K[x]$, then $[K(\alpha) : K] \leq k$. Combined with the lower bound and irreducibility, this nails the degree exactly.

The same two-sided argument — find a polynomial, then check it is the minimal one — is how almost every degree computation in this article actually proceeds.

---

## Simple Extensions and Minimal Polynomials

**Definition.** An extension $L/K$ is *simple* if $L = K(\alpha)$ for some $\alpha \in L$. The element $\alpha$ is a *primitive element*.

![Minimal polynomial determines the extension](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_minimal_polynomial.png)


**Definition.** Let $\alpha \in L$. We say $\alpha$ is *algebraic over $K$* if $f(\alpha) = 0$ for some nonzero polynomial $f(x) \in K[x]$. Otherwise, $\alpha$ is *transcendental over $K$*.

**Examples.**

- $\sqrt{2}$ is algebraic over $\mathbb{Q}$, satisfying $x^2 - 2 = 0$.
- $i$ is algebraic over $\mathbb{Q}$, satisfying $x^2 + 1 = 0$.
- $\pi$ and $e$ are transcendental over $\mathbb{Q}$ (Lindemann 1882, Hermite 1873). The proofs are not algebraic — they use analytic estimates on power-series remainders — and they are also not easy.
- Every element of $\mathbb{F}_q$ (the field of $q$ elements) is algebraic over $\mathbb{F}_p$, since $\mathbb{F}_q$ is a finite-dimensional $\mathbb{F}_p$-vector space.

![The minimal polynomial determining the degree of an extension](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_2_minimal_poly.png)

**Theorem (Minimal Polynomial).** If $\alpha \in L$ is algebraic over $K$, then there exists a unique monic irreducible polynomial $m_\alpha(x) \in K[x]$ such that $m_\alpha(\alpha) = 0$. Moreover, $m_\alpha(x)$ divides every polynomial $f(x) \in K[x]$ with $f(\alpha) = 0$, and $K(\alpha) \cong K[x]/(m_\alpha)$.

*Proof.* Consider the evaluation homomorphism $\varphi : K[x] \to L$ sending $f(x) \mapsto f(\alpha)$. The kernel is a nonzero ideal of $K[x]$ (since $\alpha$ is algebraic). $K[x]$ is a PID, so the kernel is generated by a single polynomial; the unique monic generator is $m_\alpha$. It is irreducible because $K[x]/\ker\varphi \cong K[\alpha]$ is a subring of $L$, hence an integral domain, and an ideal in a PID has integral-domain quotient iff it is prime iff its generator is irreducible. Maximality of $(m_\alpha)$ then gives that $K[\alpha]$ is itself a field, so $K(\alpha) = K[\alpha]$. $\square$

**Theorem (Structure of Simple Algebraic Extensions).** If $\alpha$ is algebraic over $K$ with minimal polynomial $m_\alpha$ of degree $n$, then $\{1, \alpha, \alpha^2, \ldots, \alpha^{n-1}\}$ is a $K$-basis of $K(\alpha)$, so $[K(\alpha) : K] = n$.

The reason is that polynomials of degree $< n$ are a complete set of coset representatives in $K[x]/(m_\alpha)$; under the isomorphism with $K(\alpha)$ they map to $1, \alpha, \ldots, \alpha^{n-1}$.

**Why this matters.** This single theorem replaces a potentially infinite-dimensional field with a finite-dimensional vector space whose multiplication table you can write on a napkin. Every computation in $\mathbb{Q}(\sqrt[3]{2})$ becomes a computation in $\mathbb{Q}[x]/(x^3 - 2)$ — and the latter is just polynomial arithmetic mod $x^3 - 2$. Inverses come for free via Bezout: if $f(\alpha) \neq 0$, then $\gcd(f, m_\alpha) = 1$, so $sf + tm_\alpha = 1$ for some $s, t \in K[x]$, and evaluating at $\alpha$ gives $f(\alpha)^{-1} = s(\alpha)$, a polynomial in $\alpha$.

There is also a representation-theoretic angle that I find pleasant: multiplication by $\alpha$ is a $K$-linear endomorphism of $K(\alpha)$. In the basis $\{1, \alpha, \ldots, \alpha^{n-1}\}$ it is the *companion matrix* of $m_\alpha$. So $m_\alpha$ is simultaneously the minimal polynomial of $\alpha$ as an algebraic element *and* the minimal polynomial of "multiplication by $\alpha$" as a linear operator. That is not a coincidence — it is the same ideal of polynomials annihilating the same vector. Field theory and linear algebra fit together more tightly than they look on a first pass.

### Worked Example: Arithmetic in $\mathbb{Q}(\sqrt{2})$

Minimal polynomial $m(x) = x^2 - 2$ (irreducible by Eisenstein at $p=2$). Basis $\{1, \sqrt{2}\}$.

Multiplication: $(3 + 5\sqrt{2})(1 - 2\sqrt{2}) = 3 - 6\sqrt{2} + 5\sqrt{2} - 10\cdot 2 = -17 - \sqrt{2}$.

Inversion: $(3 + 5\sqrt{2})^{-1} = \frac{3 - 5\sqrt{2}}{9 - 50} = -\frac{3}{41} + \frac{5}{41}\sqrt{2}$.

### Worked Example: Arithmetic in $\mathbb{Q}(\sqrt[3]{2})$

Let $\alpha = \sqrt[3]{2}$, so $\alpha^3 = 2$. Basis $\{1, \alpha, \alpha^2\}$.

Multiplication: $(1 + \alpha)(2 - \alpha + \alpha^2) = 2 - \alpha + \alpha^2 + 2\alpha - \alpha^2 + \alpha^3 = 2 + \alpha + 2 = 4 + \alpha$.

Inversion: extended Euclidean on $1+x$ and $x^3 - 2$ gives $(1+\alpha)^{-1} = \frac{1}{3}(\alpha^2 - \alpha + 1)$. Verify: $\frac{1}{3}(\alpha^2 - \alpha + 1)(1 + \alpha) = \frac{1}{3}(1 + \alpha^3) = \frac{1}{3}(1 + 2) = 1$. Good.

### The Transcendental Case

If $\alpha$ is transcendental over $K$, the evaluation map $K[x] \to K[\alpha]$ is injective. So $K[\alpha] \cong K[x]$, a polynomial ring (which is *not* a field), and $K(\alpha) \cong K(x)$, the field of rational functions. We have $[K(\alpha) : K] = \infty$. Example: $\mathbb{Q}(\pi) \cong \mathbb{Q}(x)$.

![Algebraic numbers vs transcendental numbers as a Venn-style diagram](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_3_alg_vs_trans.png)

**Algebraic vs. transcendental, in pictures.** The algebraic numbers $\overline{\mathbb{Q}}$ form a countable subfield of $\mathbb{C}$; the transcendentals are everything else (uncountably many). Almost every real number is transcendental, but exhibiting *any specific* transcendental number requires real work. Liouville (1844) gave the first explicit examples by constructing numbers with absurdly good rational approximations; that historical detour is the spiritual ancestor of modern Diophantine approximation. The fact that $\pi$ is transcendental was not proved until 1882 (Lindemann), and that $e + \pi$ is irrational is, somewhat embarrassingly, still open.

A tiny taste of why transcendence is hard: any finite list of polynomials with integer coefficients has only finitely many roots, so the algebraic numbers are a countable union of finite sets. Cardinality alone forces "most" reals to be transcendental. But pointing at a specific real number $r$ and saying "this one is transcendental" requires showing that *no* polynomial in $\mathbb{Z}[x]$ vanishes at $r$, which is a universal statement over an infinite set. Liouville's trick was clever: he wrote down numbers like $\sum_{n \geq 1} 10^{-n!}$ that admit absurdly accurate rational approximations $p/q$ with error $\ll q^{-k}$ for *every* $k$, then showed that algebraic numbers cannot be approximated that well. The same approximation philosophy, refined through the 20th century by Roth and Schmidt, drives a lot of contemporary number theory.

**Closure under arithmetic.** $\overline{\mathbb{Q}}$ is closed under $+, -, \times, \div$: if $\alpha, \beta$ are algebraic, so are $\alpha + \beta$, $\alpha\beta$, and (for nonzero $\beta$) $\alpha/\beta$. The proof is structural — $\mathbb{Q}(\alpha, \beta)$ is a finite extension of $\mathbb{Q}$ by the tower law, hence every element of it is algebraic. The same argument shows that the algebraic *numbers* form a field; this is one of those statements where having the dimension-counting tool turns a potentially messy direct verification into a one-liner.

---

## The Tower Law and Its Consequences

The tower law is the workhorse of dimension counting. Given three nested fields $K \subseteq M \subseteq L$, we can climb $K \to M \to L$ in two steps and the dimensions multiply.

![Tower of field extensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_extension_tower.png)


**Theorem (Tower Law).** If $K \subseteq M \subseteq L$ are fields and $L/M$, $M/K$ are both finite, then $L/K$ is finite and
$$[L : K] = [L : M] \cdot [M : K].$$

*Proof.* Let $\{e_1, \ldots, e_m\}$ be a $K$-basis of $M$ and $\{f_1, \ldots, f_n\}$ an $M$-basis of $L$. We show $\{e_i f_j\}_{i,j}$ is a $K$-basis of $L$.

*Spanning.* Any $\ell \in L$ is $\sum_j \mu_j f_j$ with $\mu_j \in M$. Each $\mu_j = \sum_i a_{ij} e_i$ with $a_{ij} \in K$. Substituting, $\ell = \sum_{i,j} a_{ij} e_i f_j$.

*Linear independence.* Suppose $\sum_{i,j} a_{ij} e_i f_j = 0$ with $a_{ij} \in K$. Group by $j$: $\sum_j (\sum_i a_{ij} e_i) f_j = 0$. By independence of $\{f_j\}$ over $M$, each inner sum is zero. By independence of $\{e_i\}$ over $K$, each $a_{ij} = 0$. $\square$

**Corollary (Degree Divisibility).** If $K \subseteq M \subseteq L$ with $[L:K]$ finite, then $[M:K]$ divides $[L:K]$. In particular, if $\alpha$ is algebraic over $K$ and $\alpha \in L$ with $[L:K] = n$, then $\deg m_\alpha$ divides $n$.

This is the source of nearly all "no such extension exists" arguments.

### Application 1: $\sqrt{2} + \sqrt{3}$ Generates $\mathbb{Q}(\sqrt{2},\sqrt{3})$

Climb the tower $\mathbb{Q} \subset \mathbb{Q}(\sqrt{2}) \subset \mathbb{Q}(\sqrt{2}, \sqrt{3})$. Both steps are degree 2 (the second because $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$, as shown earlier), so the total degree is 4.

Now let $\alpha = \sqrt{2} + \sqrt{3}$. Then $\alpha^2 = 5 + 2\sqrt{6}$, so $\sqrt{6} = (\alpha^2 - 5)/2 \in \mathbb{Q}(\alpha)$. From $\alpha\sqrt{6} = 3\sqrt{2} + 2\sqrt{3}$ and $\alpha = \sqrt{2} + \sqrt{3}$, we recover $\sqrt{2} = \alpha\sqrt{6} - 2\alpha = \alpha(\sqrt{6} - 2)$ and $\sqrt{3} = \alpha - \sqrt{2}$. So $\mathbb{Q}(\alpha) = \mathbb{Q}(\sqrt{2}, \sqrt{3})$.

By the tower law, the minimal polynomial of $\alpha$ over $\mathbb{Q}$ has degree 4. Squaring twice: $\alpha^2 - 5 = 2\sqrt{6} \Rightarrow (\alpha^2 - 5)^2 = 24 \Rightarrow m_\alpha(x) = x^4 - 10x^2 + 1$.

This is the *primitive element theorem* in action: every separable finite extension is in fact simple.

![Constructible numbers form a tower of degree-2 extensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_6_constructible.png)

### Application 2: The Impossibility of Doubling the Cube

Doubling the cube means: given a unit cube, construct one of volume 2 using straightedge and compass. The geometry forces us to construct a length $\sqrt[3]{2}$ using only $+, -, \times, \div, \sqrt{\cdot}$ starting from $\mathbb{Q}$. Each square root extends the field by degree 2 (or 1, if the radicand was already a square). So a constructible number lies in a tower
$$\mathbb{Q} = K_0 \subseteq K_1 \subseteq \cdots \subseteq K_n$$
with $[K_{i+1} : K_i] \in \{1, 2\}$. By the tower law, $[K_n : \mathbb{Q}]$ is a power of 2. But $[\mathbb{Q}(\sqrt[3]{2}) : \mathbb{Q}] = 3$, which does not divide any power of 2. So $\sqrt[3]{2}$ is not constructible. The Greeks struggled with this for two thousand years; we dispatch it in a paragraph.

Trisection of a $60^\circ$ angle reduces to constructing $\cos(20^\circ)$, which satisfies $8x^3 - 6x - 1 = 0$, irreducible over $\mathbb{Q}$. Same conclusion: degree 3, not a power of 2, not constructible.

Squaring the circle reduces to constructing $\sqrt{\pi}$. But $\pi$ is transcendental, so $[\mathbb{Q}(\sqrt{\pi}) : \mathbb{Q}] = \infty$, definitely not a power of 2. Impossible.

The constructible regular $n$-gons are exactly those for which $n = 2^k p_1 \cdots p_r$ with the $p_i$ distinct Fermat primes. Gauss showed at age 19 that the regular 17-gon is constructible (it produces a tower of quadratic extensions of total degree $\varphi(17) = 16 = 2^4$); he also conjectured the converse, finally proved by Wantzel in 1837.

**Why this matters.** A purely algebraic divisibility argument settles three classical problems that had been open for two millennia. This is the kind of leverage abstract algebra provides: by encoding a geometric question as a question about field degrees, the answer becomes a one-line check.

It is worth pausing on what the tower law is *really* doing. Take any finite extension $L/K$, pick an element $\alpha \in L$, and consider the chain $K \subseteq K(\alpha) \subseteq L$. The tower law says
$$[L : K] = [L : K(\alpha)] \cdot [K(\alpha) : K],$$
so the degree of the minimal polynomial of $\alpha$ divides $[L : K]$. Every algebraic element in a finite extension carries a divisibility constraint that is dictated entirely by the dimension of the ambient field. That single observation is the essence of the constructibility arguments above and a good chunk of inverse Galois theory.

A small numerical reality check: for $\mathbb{Q}(\sqrt{2}, \sqrt{3})$ over $\mathbb{Q}$ we have degree 4, and the eight elements $\sqrt{2}, \sqrt{3}, \sqrt{6}, \sqrt{2}+\sqrt{3}, \sqrt{2}+\sqrt{6}, \sqrt{2}\sqrt{3}=\sqrt{6}, \ldots$ all have minimal-polynomial degree dividing 4 (so 1, 2, or 4). You cannot have an element of degree 3 sitting in a degree-4 extension, no matter how baroque your construction.

---

## Splitting Fields and Algebraic Closures

We now move from "adjoin one root" to "adjoin all the roots."

![Splitting field: smallest field containing all roots](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_splitting_field.png)


![Algebraic vs transcendental elements](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_algebraic_elements.png)


**Definition.** Let $f(x) \in K[x]$ be a polynomial of degree $n \geq 1$. A *splitting field* of $f$ over $K$ is a field $L \supseteq K$ such that:

1. $f(x)$ factors into linear factors over $L$: $f(x) = c(x - \alpha_1)(x - \alpha_2) \cdots (x - \alpha_n)$ with $\alpha_i \in L$.
2. $L = K(\alpha_1, \ldots, \alpha_n)$ is generated over $K$ by the roots.

The second condition keeps us honest: we want the *smallest* such field, not just any field containing the roots.

![Constructing the splitting field of a polynomial step by step](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_4_splitting_field.png)

**Theorem (Existence and Uniqueness).** For every $f(x) \in K[x]$, a splitting field exists and is unique up to $K$-isomorphism.

*Existence sketch.* Induct on $\deg f$. Linear case is trivial. Otherwise, factor $f$ into irreducibles over $K$. Pick an irreducible factor $g$ of degree $\geq 2$. Form $K_1 = K[x]/(g(x))$; in $K_1$, $g$ has the root $\overline{x}$, so $f$ factors as $(x - \overline{x}) \cdot h(x)$ in $K_1[x]$, and induction applies to $h$ over $K_1$. $\square$

*Uniqueness sketch.* Induct on $[L_1 : K]$. If $f$ already splits in $K$, both splitting fields equal $K$. Otherwise pick an irreducible factor $p$ with roots $\alpha \in L_1$ and $\beta \in L_2$. Sending $\overline{x} \mapsto \beta$ gives an isomorphism $K(\alpha) \to K(\beta)$. Now $L_1$ is a splitting field for $f/(x-\alpha)$ over $K(\alpha)$ and $L_2$ for $f/(x-\beta)$ over $K(\beta)$; induction extends the isomorphism. $\square$

**Examples.**

- $x^2 - 2$ over $\mathbb{Q}$: splitting field $\mathbb{Q}(\sqrt{2})$, degree 2.
- $x^2 + 1$ over $\mathbb{R}$: splitting field $\mathbb{C}$, degree 2.
- $x^3 - 2$ over $\mathbb{Q}$: splitting field $\mathbb{Q}(\sqrt[3]{2}, \omega)$ with $\omega = e^{2\pi i/3}$, degree 6. The tower $\mathbb{Q} \subset \mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{Q}(\sqrt[3]{2}, \omega)$ has steps 3 and 2.
- $x^4 - 2$ over $\mathbb{Q}$: splitting field $\mathbb{Q}(\sqrt[4]{2}, i)$. Tower: $\mathbb{Q} \subset \mathbb{Q}(\sqrt[4]{2}) \subset \mathbb{Q}(\sqrt[4]{2}, i)$ with steps 4 and 2 (since $i \notin \mathbb{Q}(\sqrt[4]{2}) \subset \mathbb{R}$). Degree 8. We will dissect this one in detail next article.

Note that the degree of a splitting field can be much larger than the degree of the polynomial — or equal to it. The ratio is exactly the size of the Galois group, which is no coincidence at all.

**Definition.** A field $\overline{K}$ is an *algebraic closure* of $K$ if every nonconstant polynomial in $\overline{K}[x]$ has a root in $\overline{K}$ (i.e., $\overline{K}$ is algebraically closed) and $\overline{K}/K$ is algebraic.

**Theorem.** Every field has an algebraic closure, unique up to (non-canonical) $K$-isomorphism.

The existence proof leans on Zorn's lemma. For $\mathbb{Q}$, the algebraic closure $\overline{\mathbb{Q}}$ is the field of all algebraic numbers, sitting inside $\mathbb{C}$. It is countable, infinite-dimensional over $\mathbb{Q}$, and contains every root of every rational polynomial. For $\mathbb{F}_p$, the algebraic closure $\overline{\mathbb{F}_p}$ is the union $\bigcup_{n \geq 1} \mathbb{F}_{p^n}$ of all finite fields of characteristic $p$. The Fundamental Theorem of Algebra says $\mathbb{C}$ is the algebraic closure of $\mathbb{R}$.

![Building the finite field GF(p^n) as a quotient of polynomials](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_5_finite_field.png)

**Finite fields, concretely.** $\mathrm{GF}(p^n) = \mathbb{F}_p[x]/(f(x))$ for any irreducible $f$ of degree $n$. Different choices of $f$ give isomorphic fields. For instance, $\mathrm{GF}(8) = \mathbb{F}_2[x]/(x^3 + x + 1)$. Elements: $\{0, 1, \alpha, \alpha+1, \alpha^2, \alpha^2+1, \alpha^2+\alpha, \alpha^2+\alpha+1\}$ where $\alpha^3 = \alpha + 1$. Multiplication: $\alpha \cdot \alpha^2 = \alpha^3 = \alpha + 1$. Inversion: $\alpha^{-1} = \alpha^2 + 1$ since $\alpha(\alpha^2+1) = \alpha^3 + \alpha = (\alpha+1) + \alpha = 1$. The multiplicative group $\mathrm{GF}(8)^\times$ is cyclic of order 7, generated by $\alpha$.

Every finite field has prime-power order. Every two finite fields of the same order are isomorphic. $\mathbb{F}_{p^n}$ contains $\mathbb{F}_{p^m}$ as a subfield iff $m \mid n$. This makes the lattice of subfields of $\overline{\mathbb{F}_p}$ literally the divisibility lattice of $\mathbb{N}$ — a startling cleanness that does not happen over $\mathbb{Q}$.

The Frobenius endomorphism $\mathrm{Frob}_p : x \mapsto x^p$ is a field automorphism of any field of characteristic $p$, and on $\mathbb{F}_{p^n}$ it generates the entire automorphism group, which is cyclic of order $n$. So $\mathrm{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p) \cong \mathbb{Z}/n\mathbb{Z}$. This is the cleanest example of a Galois group I know — abelian, cyclic, and the same group for every choice of $n$. Compare to $\mathrm{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$, the absolute Galois group of $\mathbb{Q}$, which is profinite, non-abelian, and currently understood only in pieces (the Langlands program, class field theory, Grothendieck's anabelian dreams).

**Why this matters.** Splitting fields are the right universe in which to study a polynomial: every root is present, no factor is hiding. The Galois group, which we meet next article, is the group of $K$-automorphisms of the splitting field, and it controls the entire factorization story. Algebraic closure is the limit of all of this — once you are inside $\overline{K}$, every algebraic question over $K$ has an answer somewhere finite-dimensional.

A subtle but useful observation: the splitting field of $f$ depends only on $f$ up to multiplication by nonzero constants, so we may assume $f$ is monic without losing anything. Also, the splitting field of $f \cdot g$ is generated by the splitting fields of $f$ and $g$, which gives a clean way to build "compositum" extensions out of simpler pieces. You see this trick used constantly in computational Galois theory: factor your big polynomial into irreducibles, compute splitting fields for each factor, glue.

---

## Separability and Perfect Fields

In characteristic 0 every irreducible polynomial has distinct roots, and most readers can blissfully ignore separability. In positive characteristic, repeated roots can appear in irreducible polynomials, breaking the Galois correspondence. Let me make this precise so we know exactly what we are paying for.

**Definition.** A polynomial $f(x) \in K[x]$ is *separable* if it has no repeated roots in any extension field (equivalently, in its splitting field, equivalently, in $\overline{K}$). An algebraic element $\alpha$ is separable if $m_\alpha$ is separable. An algebraic extension $L/K$ is separable if every element of $L$ is.

**Lemma (Derivative criterion).** $f(x)$ is separable iff $\gcd(f, f') = 1$, where $f'$ is the formal derivative.

*Proof.* If $\alpha$ is a repeated root, $f = (x-\alpha)^2 g$, so $f' = 2(x-\alpha)g + (x-\alpha)^2 g'$ and $(x - \alpha) \mid \gcd(f, f')$. Conversely, if $\alpha$ is a simple root $f = (x - \alpha) h$ with $h(\alpha) \neq 0$, then $f'(\alpha) = h(\alpha) \neq 0$. $\square$

**Corollary.** An irreducible $p(x)$ is inseparable iff $p'(x) = 0$.

*Proof.* $\gcd(p, p') \neq 1$ and $\deg p' < \deg p$ force $p' = 0$. $\square$

In characteristic 0, $p' = 0$ forces $p$ constant (contradiction). In characteristic $p$, $p'(x) = 0$ means $p(x) = g(x^p)$ for some $g$. The smallest counterexample lives over $\mathbb{F}_p(t)$: the polynomial $f(x) = x^p - t \in \mathbb{F}_p(t)[x]$ is irreducible (Eisenstein at the prime $t \in \mathbb{F}_p[t]$) but $f'(x) = px^{p-1} = 0$, so it has a single root $\sqrt[p]{t}$ of multiplicity $p$.

**Definition.** A field $K$ is *perfect* if every irreducible polynomial in $K[x]$ is separable. Equivalently:

- $\mathrm{char}(K) = 0$, or
- $\mathrm{char}(K) = p > 0$ and the Frobenius map $x \mapsto x^p$ is surjective (every element is a $p$-th power).

**Examples.**

- All fields of characteristic 0 are perfect.
- All finite fields are perfect (Frobenius is bijective by pigeonhole on a finite set).
- Algebraically closed fields are perfect.
- $\mathbb{F}_p(t)$ is *not* perfect: $t$ has no $p$-th root in $\mathbb{F}_p(t)$.

**Counterexample.** The field $\mathbb{F}_p(t)$ of rational functions over $\mathbb{F}_p$ is *not* perfect: $t$ has no $p$-th root in $\mathbb{F}_p(t)$. The polynomial $f(x) = x^p - t$ is irreducible (Eisenstein at the prime $t \in \mathbb{F}_p[t]$) but $f'(x) = px^{p-1} = 0$, so it has a single root $\sqrt[p]{t}$ of multiplicity $p$. The extension $\mathbb{F}_p(t)(\sqrt[p]{t})/\mathbb{F}_p(t)$ has degree $p$, but its automorphism group over the base is trivial — the only candidate $\sqrt[p]{t} \mapsto \sqrt[p]{t}\zeta$ requires a $p$-th root of unity, and the only $p$-th root of unity in characteristic $p$ is 1. So $|\mathrm{Aut}_K(L)| = 1 \neq p = [L:K]$. This is the prototypical example of how the Galois correspondence breaks for inseparable extensions: the group is too small to see the field.

For the rest of this series we work in characteristic 0 or with finite fields, so separability comes for free.

### Normal Extensions

**Definition.** A finite extension $L/K$ is *normal* if it is the splitting field of some polynomial in $K[x]$. Equivalently, every irreducible polynomial in $K[x]$ that has *one* root in $L$ has *all* its roots in $L$.

**Example.** $\mathbb{Q}(\sqrt{2})/\mathbb{Q}$ is normal: it is the splitting field of $x^2 - 2$. However, $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ is *not* normal: $x^3 - 2$ has the root $\sqrt[3]{2}$ in $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$, but its other two roots $\sqrt[3]{2}\omega$ and $\sqrt[3]{2}\omega^2$ are non-real, hence outside $\mathbb{Q}(\sqrt[3]{2})$. Adjoining one root of an irreducible polynomial does not always give a normal extension; you may need to adjoin all the roots.

**Example.** The splitting field of $x^3 - 2$ over $\mathbb{Q}$ is $\mathbb{Q}(\sqrt[3]{2}, \omega)$, which is normal (being a splitting field), of degree 6.

![Catalog of classical field extensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/07-field-extensions/aa_v2_07_7_extension_examples.png)

**Definition.** A finite extension $L/K$ is *Galois* if it is both normal and separable.

The combination of "all roots present" (normal) and "all roots distinct" (separable) gives us exactly the right setting in which the automorphism group has the maximum allowed size, namely $|\mathrm{Gal}(L/K)| = [L : K]$. This is what makes the Galois correspondence go through, and it is the point at which Part 8 will pick up.

**Why this matters.** Separability and normality are the two technical conditions that make Galois theory clean. Strip them away and you get pathologies: non-bijective correspondences, automorphism groups that are too small, intermediate fields with no group-theoretic shadow. Insisting on Galois extensions is not pedantry; it is the price of admission to the central theorem of the subject. Over $\mathbb{Q}$ and over finite fields — i.e., for almost everyone reading this article — that price is zero, since separability is automatic and we just need the splitting-field condition.

A useful mental picture: the Galois correspondence is a contravariant equivalence between subfields of $L$ containing $K$ and subgroups of $\mathrm{Gal}(L/K)$. Inseparability bloats the field side without bloating the group side; non-normality does the opposite. Either failure breaks the bijection, and you can tell exactly which side broke just by counting.

One more remark to close out this section. There is a useful "transitivity" fact for separability: if $K \subseteq M \subseteq L$ with $M/K$ and $L/M$ both separable, then $L/K$ is separable. (Proof: any element $\alpha \in L$ has a separable minimal polynomial over $M$; the coefficients of that polynomial are themselves separable over $K$; combining, $\alpha$ generates a separable extension of $K$.) The analogous statement for "normal" is *false* — normality does not transit through towers. This is the first sign that normality is the more delicate of the two conditions, and it is why much of next article is spent identifying which subgroups of the Galois group correspond to normal subfields.

---

## What's Next

We have assembled all the ingredients: field extensions and their degrees, the tower law for computing degrees in chains, minimal polynomials that describe simple extensions, splitting fields that give us "complete" factorizations, normality and separability to prevent degenerate behavior. In the next article, we combine these tools into Galois theory proper: the group of automorphisms of a field extension and its remarkable correspondence with the lattice of intermediate fields. This correspondence will ultimately explain why the general quintic cannot be solved by radicals, settling a question that puzzled mathematicians for three centuries.

![Animation: adjoining an element to a field](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/07_adjoin_element.gif)


A short forward-looking checklist, so you know what you are walking into. Given a finite Galois extension $L/K$:

- The order of $\mathrm{Gal}(L/K)$ equals $[L:K]$. (This already pins down the group up to a finite list of possibilities.)
- Subgroups of $\mathrm{Gal}(L/K)$ correspond bijectively to intermediate fields $K \subseteq M \subseteq L$.
- Normal subgroups correspond to *normal* (= Galois over $K$) intermediate extensions.
- "Solvable by radicals" translates into "the Galois group is solvable as a group."
- The general quintic has Galois group $S_5$, which is not solvable. End of story.

If any of those bullets feels mysterious, that is the right state of mind to enter Part 8 with. We will earn each of them, and along the way the entire scaffolding of this article — degrees, towers, splitting fields, minimal polynomials — will pay off all at once.


## Deeper Dive: Computations on Field Extensions

The cleanest way to digest extension theory is to compute degrees, minimal polynomials, and bases for several small towers.

**Computation A: $\mathbb{Q}(\sqrt{2}, \sqrt{3})$ over $\mathbb{Q}$.** Both $\sqrt{2}$ and $\sqrt{3}$ are roots of irreducible degree-$2$ polynomials over $\mathbb{Q}$. The tower $\mathbb{Q} \subset \mathbb{Q}(\sqrt{2}) \subset \mathbb{Q}(\sqrt{2}, \sqrt{3})$ has each step of degree $\leq 2$, so the total degree is at most $4$. Is it exactly $4$? Equivalently, is $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$? Suppose $\sqrt{3} = a + b\sqrt{2}$ with $a, b \in \mathbb{Q}$. Squaring: $3 = a^2 + 2b^2 + 2ab\sqrt{2}$. Since $\sqrt{2} \notin \mathbb{Q}$, we need $ab = 0$. Case $a = 0$: $3 = 2b^2$, so $b^2 = 3/2$, no rational $b$. Case $b = 0$: $3 = a^2$, so $a = \pm\sqrt{3}$, not rational. Contradiction. So $\sqrt{3} \notin \mathbb{Q}(\sqrt{2})$ and $[\mathbb{Q}(\sqrt{2}, \sqrt{3}) : \mathbb{Q}] = 4$. A $\mathbb{Q}$-basis: $\{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$.

A surprise: this field also equals $\mathbb{Q}(\sqrt{2} + \sqrt{3})$. The minimal polynomial of $\alpha = \sqrt{2} + \sqrt{3}$ over $\mathbb{Q}$ is $\alpha^4 - 10\alpha^2 + 1$ (compute $\alpha^2 = 5 + 2\sqrt{6}$, then $(\alpha^2 - 5)^2 = 24$, giving $\alpha^4 - 10\alpha^2 + 1 = 0$). This polynomial is irreducible of degree $4$, confirming the degree from the other direction. The "primitive element theorem" says every separable extension of finite degree is generated by a single element, and here we see it concretely: a single $\alpha$ generates the same field as the two elements $\sqrt{2}, \sqrt{3}$.

**Computation B: degree of a cyclotomic extension.** Take $\zeta = e^{2\pi i / 7}$, a primitive $7$th root of unity. Its minimal polynomial over $\mathbb{Q}$ is the cyclotomic polynomial $\Phi_7(x) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1$, irreducible by Eisenstein after the substitution $x = y + 1$. So $[\mathbb{Q}(\zeta) : \mathbb{Q}] = 6 = \varphi(7)$. More generally, $[\mathbb{Q}(\zeta_n) : \mathbb{Q}] = \varphi(n)$ for any $n$ — a foundational fact of cyclotomic theory.

**Computation C: a non-trivial tower.** Consider $\mathbb{Q}(\sqrt[3]{2}, \omega)$ where $\omega = e^{2\pi i/3}$. Tower: $\mathbb{Q} \subset \mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{Q}(\sqrt[3]{2}, \omega)$. The first step has degree $3$ (minimal polynomial $x^3 - 2$, Eisenstein at $p = 2$). The second step adds $\omega$, with minimal polynomial $x^2 + x + 1$ — irreducible over $\mathbb{Q}(\sqrt[3]{2})$ because $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$ but $\omega \notin \mathbb{R}$, so $\omega$ has degree $> 1$ over $\mathbb{Q}(\sqrt[3]{2})$, hence exactly $2$. Total degree $3 \cdot 2 = 6$. A $\mathbb{Q}$-basis: $\{1, \sqrt[3]{2}, \sqrt[3]{4}, \omega, \omega\sqrt[3]{2}, \omega\sqrt[3]{4}\}$.

This field is the splitting field of $x^3 - 2$ over $\mathbb{Q}$, since the three roots are $\sqrt[3]{2}, \omega\sqrt[3]{2}, \omega^2\sqrt[3]{2}$. Every irreducible polynomial $f \in \mathbb{Q}[x]$ has a *splitting field* — the smallest extension over which $f$ factors into linear factors — and the splitting field is unique up to isomorphism.

**Computation D: an algebraic element via Cayley–Hamilton.** Take $A = \begin{pmatrix} 0 & 1 \\ 2 & 0 \end{pmatrix}$, viewed as an element of $M_2(\mathbb{Q})$. Its characteristic polynomial is $x^2 - 2$. By Cayley–Hamilton, $A^2 - 2I = 0$, i.e., $A^2 = 2I$. So $A$ is a "matrix square root of $2$." The subring $\mathbb{Q}[A] = \{a I + b A : a, b \in \mathbb{Q}\}$ is isomorphic to $\mathbb{Q}(\sqrt{2})$ as a ring — a faithful matrix realization of the field extension. This is the secret ingredient that makes "adjoining $\sqrt{2}$" feel concrete: it is the same as choosing a matrix that squares to $2 I$.

**Computation E: degree-counting argument for transcendence of "most" reals.** The set of algebraic numbers over $\mathbb{Q}$ is countable: each algebraic number is a root of some polynomial in $\mathbb{Q}[x]$, the set of polynomials is countable (each polynomial is a finite sequence of rationals), and each polynomial has finitely many roots. Countable union of finite sets is countable. But $\mathbb{R}$ is uncountable. Hence "most" real numbers are transcendental. This argument, due to Cantor, is the first transcendence proof — and although it does not exhibit a single transcendental number, it shows that they are dense and uncountable.

To exhibit one, the easiest target is Liouville's number $\sum_{n=1}^\infty 10^{-n!} = 0.110001000000000000000001\ldots$ Liouville showed that algebraic numbers cannot be approximated too well by rationals, and this number has rationals approximating it to better than any polynomial bound. Hence transcendental. For $\pi$ and $e$, the proofs (Hermite 1873, Lindemann 1882) require analytic input — symmetric polynomials in roots are key, and Galois theory is in the background.

## The Geometry Hidden in Tower Multiplication

The formula $[L : F] = [L : K][K : F]$ is just multiplicativity of dimensions: a basis for $L$ over $F$ is obtained by multiplying basis elements of $K/F$ with basis elements of $L/K$, giving $[K : F] \cdot [L : K]$ products. Concrete: a $\mathbb{Q}$-basis for $\mathbb{Q}(\sqrt{2}, \sqrt{3})$ is $\{1, \sqrt{2}\} \times \{1, \sqrt{3}\} = \{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$ — exactly $2 \cdot 2 = 4$ elements.

The geometric content: $L$ is a vector space over $F$ of dimension $[L : F]$, and the tower expresses this dimension as a product. When you also have a Galois action, the degree equals the order of the Galois group (in the Galois case), and the multiplicativity of degree mirrors the Lagrange / index multiplicativity for groups: $|\mathrm{Gal}(L/F)| = [\mathrm{Gal}(L/F) : \mathrm{Gal}(L/K)] \cdot |\mathrm{Gal}(L/K)|$.

This is the first hint of the Galois correspondence. Subgroups of the Galois group correspond to intermediate fields, with the larger subgroup pairing with the smaller field, and the indices match exactly. Part 8 will make this precise.

## Common Pitfalls for Beginners

The first pitfall: confusing "algebraic extension" with "finite extension." Every finite extension is algebraic (since every element has a minimal polynomial of degree $\leq [L : F]$), but not every algebraic extension is finite. The algebraic closure $\overline{\mathbb{Q}}$ is algebraic over $\mathbb{Q}$ but has infinite degree.

The second pitfall: assuming that adjoining one root of an irreducible polynomial gives all the roots. It does not, in general. $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$ contains only the real cube root of $2$, not the two complex ones. Adjoining all three roots requires the larger field $\mathbb{Q}(\sqrt[3]{2}, \omega)$, of degree $6$. The extensions where adjoining one root automatically gives all roots are the *normal* extensions — and the term "normal" reappears, exactly because the analogous condition (closure under conjugation, in this case Galois conjugation) is what is being demanded.

The third pitfall: thinking of $\mathbb{Q}(\alpha)$ as "$\mathbb{Q}$ plus the symbol $\alpha$." It is the smallest field containing both, which is concretely $\{p(\alpha) / q(\alpha) : p, q \in \mathbb{Q}[x], q(\alpha) \neq 0\}$ — but if $\alpha$ is algebraic, the denominators clean up and you get $\mathbb{Q}[x]/(\text{minpoly of } \alpha)$. The polynomial-quotient picture is the right computational handle.

## Where This Shows Up

*Number theory.* The ring of integers $\mathcal{O}_K$ of a number field $K = \mathbb{Q}(\alpha)$ is the central object of algebraic number theory. The factorization behaviour of rational primes in $\mathcal{O}_K$ is governed by the splitting of the minimal polynomial of $\alpha$ modulo $p$ — the Dedekind–Kummer theorem. This is how "factor $5$ in $\mathbb{Z}[i]$" reduces to "factor $x^2 + 1$ mod $5$": both give $5 = (2+i)(2-i)$ since $x^2 + 1 \equiv (x-2)(x+2) \pmod 5$.

*Coding theory and BCH codes.* Designing error-correcting codes with prescribed distance requires constructing finite-field extensions of $\mathbb{F}_q$ of specified degree and using elements of these extensions as evaluation points. Reed-Solomon over $\mathbb{F}_{256}$ used in QR codes is the standard application.

*Construction problems in classical geometry.* Three classical Greek problems — squaring the circle, doubling the cube, trisecting a general angle — are unsolvable with compass and straightedge precisely because they would require constructible numbers whose minimal polynomial over $\mathbb{Q}$ has degree $> 2^k$. Doubling the cube needs $\sqrt[3]{2}$ (degree $3$, not a power of $2$). Trisecting $60°$ needs a root of $4x^3 - 3x = 1/2$ (irreducible, degree $3$). Squaring the circle needs $\sqrt{\pi}$, hence $\pi$ algebraic — but $\pi$ is transcendental. All three fall to extension-theoretic arguments.

## What I Want You to Carry Forward

Four questions for Part 8, on Galois theory:

1. *When does an extension $L/F$ have a Galois group of size exactly $[L : F]$?* When the extension is *normal* and *separable*. We will spend significant effort understanding both conditions.
2. *Why is the Galois group of $x^5 - 1$ over $\mathbb{Q}$ abelian, but the Galois group of $x^5 - 2$ over $\mathbb{Q}$ non-abelian?* The first is a cyclotomic extension; the second is a radical extension involving both a cube root and roots of unity.
3. *What is the connection between solvability of a polynomial by radicals and solvability of its Galois group?* The Galois correspondence converts the field-theoretic question into a group-theoretic one — and the latter is decidable.
4. *Why is the general quintic not solvable, while the general quartic is?* $A_5$ is simple non-abelian (and $S_5$ has $A_5$ as its only non-trivial normal subgroup), so $S_5$ is *not solvable*. $S_4$ is, with composition series $S_4 \supset A_4 \supset V_4 \supset \mathbb{Z}/2 \supset \{e\}$.

If the tower computations feel routine, you are ready. If not, work out $\mathbb{Q}(\sqrt{2}, i)$ in detail: degree $4$, minimal polynomial of a primitive element, Galois group (it is the Klein four-group). That single example contains in microcosm everything Galois theory will do.

---
