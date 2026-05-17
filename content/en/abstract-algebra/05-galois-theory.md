---
title: "Galois Theory — Symmetry of Roots"
date: 2021-02-07 09:00:00
tags:
  - abstract-algebra
  - galois-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 5
series_total: 6
translationKey: "abstract-algebra-5"
description: "The Galois group connects field extensions to group theory. The fundamental theorem explains solvability by radicals."
---

## The big picture

You have a polynomial. You want its roots. For quadratics, the quadratic formula works. Cubics and quartics have formulas too (Cardano's formula is ugly but it exists). For quintics? Abel and Ruffini proved no general formula exists — you cannot express the roots of a generic degree-5 polynomial using only $+, -, \times, \div$ and $n$-th roots.

Galois theory explains *why*. The idea: attach a group to a polynomial (its Galois group). The polynomial is solvable by radicals iff that group is a solvable group (in the technical group-theoretic sense from [Part 2](/en/abstract-algebra/02-homomorphisms-and-quotients/)). The symmetric group $S_5$ is not solvable — its only proper normal subgroup is $A_5$, which is simple. A generic quintic has Galois group $S_5$. So: no radical formula.

This is, in my view, the most beautiful theorem in algebra. A question about solving equations gets a complete answer from *group theory*. Two seemingly unrelated subjects — polynomial roots and permutation groups — turn out to be the same problem viewed from different angles. Evariste Galois wrote this down at age 20, the night before he died in a duel. Mathematics has never fully recovered from the romance of it.

## The Galois group

Let $K/F$ be a field extension. An **$F$-automorphism** of $K$ is a field isomorphism $\sigma: K \to K$ that fixes $F$ pointwise: $\sigma(a) = a$ for all $a \in F$.

The **Galois group** $\text{Gal}(K/F)$ is the set of all $F$-automorphisms of $K$, forming a group under composition:

$$
\text{Gal}(K/F) = \{\sigma: K \xrightarrow{\sim} K \mid \sigma|_F = \text{id}_F\}
$$

Why automorphisms? Because they are the symmetries of $K$ that respect the "ground rules" imposed by $F$. An automorphism permutes the roots of every polynomial in $F[x]$ — it cannot send $\sqrt{2}$ to $\pi$, because $\sqrt{2}$ satisfies $x^2 - 2 \in \mathbb{Q}[x]$ and $\pi$ does not.

An extension is **Galois** if it is both normal (a splitting field of some polynomial over $F$) and separable (which is automatic in characteristic 0, as discussed in [Part 4](/en/abstract-algebra/04-fields-and-extensions/)). For Galois extensions, the fundamental equality holds:

$$
|\text{Gal}(K/F)| = [K:F]
$$

This equality fails for non-Galois extensions. For instance, $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ has degree 3 but only one $\mathbb{Q}$-automorphism (the identity) — the other two cube roots of 2 are complex, not in the field. The extension is not normal (it is not a splitting field of $x^3 - 2$, since two roots are missing).

![Galois correspondence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/05-galois-theory/fig_concept.png)

### Example 1: $\text{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$

Any $\mathbb{Q}$-automorphism $\sigma$ of $\mathbb{Q}(\sqrt{2})$ is determined by $\sigma(\sqrt{2})$. Since $\sigma$ preserves the relation $(\sqrt{2})^2 = 2$, we need $\sigma(\sqrt{2})^2 = 2$, so $\sigma(\sqrt{2}) = \pm\sqrt{2}$.

The two automorphisms: the identity $\text{id}$ and conjugation $\tau: a + b\sqrt{2} \mapsto a - b\sqrt{2}$. The Galois group is $\mathbb{Z}/2\mathbb{Z}$.

### Example 2: $\text{Gal}(\mathbb{Q}(\sqrt[3]{2}, \omega)/\mathbb{Q}) \cong S_3$

This is the splitting field of $x^3 - 2$ with degree 6 over $\mathbb{Q}$. The Galois group has order 6. An automorphism is determined by where it sends $\sqrt[3]{2}$ (three choices: $\sqrt[3]{2}, \omega\sqrt[3]{2}, \omega^2\sqrt[3]{2}$) and $\omega$ (two choices: $\omega, \omega^2$). This gives $3 \times 2 = 6$ automorphisms.

The group acts faithfully on the three roots $\{\sqrt[3]{2}, \omega\sqrt[3]{2}, \omega^2\sqrt[3]{2}\}$ by permuting them. Since it has order 6 and embeds in $S_3$ (which also has order 6), the Galois group *is* $S_3$. A non-abelian group arising from a cubic — the non-commutativity reflects the fundamental asymmetry between the roots (one is real, two are complex).

### Example 3: Cyclotomic extensions

Consider $\mathbb{Q}(\zeta_n)/\mathbb{Q}$ where $\zeta_n = e^{2\pi i/n}$. The minimal polynomial is the $n$th cyclotomic polynomial $\Phi_n(x)$ of degree $\varphi(n)$. The Galois group is $(\mathbb{Z}/n\mathbb{Z})^*$, the multiplicative group of units mod $n$. Each automorphism sends $\zeta_n \mapsto \zeta_n^k$ for $\gcd(k, n) = 1$.

For $n = 7$: $\text{Gal}(\mathbb{Q}(\zeta_7)/\mathbb{Q}) \cong (\mathbb{Z}/7\mathbb{Z})^* \cong \mathbb{Z}/6\mathbb{Z}$, a cyclic group of order 6. This cyclicity is what makes $x^7 - 1$ solvable by radicals (and indeed, Gauss showed how to express $\zeta_7$ using nested square and cube roots).

For $n = 8$: $\text{Gal}(\mathbb{Q}(\zeta_8)/\mathbb{Q}) \cong (\mathbb{Z}/8\mathbb{Z})^* \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ — the Klein four-group, not cyclic.

## The Fundamental Theorem of Galois Theory

**Theorem.** Let $K/F$ be a finite Galois extension with Galois group $G = \text{Gal}(K/F)$. There is an inclusion-reversing bijection:

$$
\{\text{intermediate fields } F \subseteq L \subseteq K\} \longleftrightarrow \{\text{subgroups } H \leq G\}
$$

given by:
- $L \mapsto H = \text{Gal}(K/L)$ (automorphisms fixing $L$)
- $H \mapsto L = K^H$ (elements of $K$ fixed by all of $H$)

Moreover:
- $[K:L] = |H|$ and $[L:F] = [G:H]$
- $L/F$ is Galois iff $H \trianglelefteq G$, and then $\text{Gal}(L/F) \cong G/H$

*Proof sketch.* The two maps are inverse: (1) $K^{\text{Gal}(K/L)} = L$ uses the fact that a Galois extension has enough automorphisms to "detect" which field elements are in which intermediate field (this is where we need both normality and separability). (2) $\text{Gal}(K/K^H) = H$ follows from counting: $|H| \leq |\text{Gal}(K/K^H)| = [K:K^H] \leq |H|$ by Artin's lemma (a fixed field of a group of $n$ automorphisms has degree at most $n$). The normality criterion: $L/F$ is Galois iff every $\sigma \in G$ maps $L$ to itself, i.e., $\sigma H \sigma^{-1} = H$ for all $\sigma$, i.e., $H \trianglelefteq G$. $\square$

This theorem is remarkable. Questions about field extensions — potentially infinite, messy, hard to compute with — become questions about finite groups, which are combinatorial and tractable. The correspondence *reverses* inclusion: bigger subgroups correspond to smaller intermediate fields.

### Example 4: The full correspondence for $S_3$

For $K = \mathbb{Q}(\sqrt[3]{2}, \omega)$ with $G = S_3$, the subgroup lattice of $S_3$ gives:

| Subgroup $H$ | $|H|$ | Fixed field $K^H$ | $[K^H:\mathbb{Q}]$ |
|---|---|---|---|
| $\{e\}$ | 1 | $K$ | 6 |
| $\langle(1\;2)\rangle$ | 2 | $\mathbb{Q}(\omega\sqrt[3]{2})$ | 3 |
| $\langle(1\;3)\rangle$ | 2 | $\mathbb{Q}(\omega^2\sqrt[3]{2})$ | 3 |
| $\langle(2\;3)\rangle$ | 2 | $\mathbb{Q}(\sqrt[3]{2})$ | 3 |
| $A_3 = \langle(1\;2\;3)\rangle$ | 3 | $\mathbb{Q}(\omega)$ | 2 |
| $S_3$ | 6 | $\mathbb{Q}$ | 1 |

Only $A_3$ is normal in $S_3$, so only $\mathbb{Q}(\omega)/\mathbb{Q}$ is Galois among the intermediate extensions. The three cubic subfields are conjugate under $G$ — they are isomorphic but not identical inside $K$.

### Example 5: Intermediate fields of $\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$

The Galois group is $V_4 = \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, generated by $\sigma: \sqrt{2} \mapsto -\sqrt{2}, \sqrt{3} \mapsto \sqrt{3}$ and $\tau: \sqrt{2} \mapsto \sqrt{2}, \sqrt{3} \mapsto -\sqrt{3}$.

$V_4$ has three subgroups of order 2: $\langle\sigma\rangle$, $\langle\tau\rangle$, $\langle\sigma\tau\rangle$. The corresponding fixed fields:
- $K^{\langle\sigma\rangle} = \mathbb{Q}(\sqrt{3})$
- $K^{\langle\tau\rangle} = \mathbb{Q}(\sqrt{2})$
- $K^{\langle\sigma\tau\rangle} = \mathbb{Q}(\sqrt{6})$

Since $V_4$ is abelian, every subgroup is normal, so every intermediate extension is Galois over $\mathbb{Q}$.

## Solvability by radicals

A polynomial $f \in F[x]$ is **solvable by radicals** if its roots can be expressed using field operations and $n$-th roots, starting from $F$. Formally: there exists a tower $F = F_0 \subset F_1 \subset \cdots \subset F_m$ where each $F_{i+1} = F_i(\alpha_i^{1/n_i})$, and the splitting field of $f$ is contained in $F_m$.

A group $G$ is **solvable** if it has a subnormal series with abelian quotients:

$$
\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_n = G, \quad G_{i+1}/G_i \text{ abelian}
$$

Equivalently, the derived series $G \supset G' \supset G'' \supset \cdots$ terminates at $\{e\}$ (where $G' = [G, G]$ is the commutator subgroup).

**Theorem (Galois).** A polynomial $f \in F[x]$ (char $F = 0$) is solvable by radicals iff $\text{Gal}(\text{Split}_F(f)/F)$ is a solvable group.

*Proof sketch.* The key steps:

1. **Radical extensions have abelian Galois groups.** If $K = F(\alpha^{1/n})$ and $F$ contains all $n$th roots of unity, then $K/F$ is Galois with $\text{Gal}(K/F) \hookrightarrow \mathbb{Z}/n\mathbb{Z}$ (cyclic, hence abelian). The automorphisms are determined by $\sigma(\alpha^{1/n}) = \zeta^k \alpha^{1/n}$ for $\zeta$ a primitive $n$th root of unity.

2. **Stacking radical extensions gives solvable groups.** A tower of radical extensions corresponds (after adjoining roots of unity) to a tower of Galois extensions with cyclic quotients — exactly a solvable group by definition.

3. **The converse.** If $\text{Gal}(f)$ is solvable, one can construct the radical tower by "peeling off" abelian quotients one at a time, each corresponding to a radical adjunction. $\square$

**Why quintics fail.** $S_5$ is not solvable. Its composition series is $\{e\} \trianglelefteq A_5 \trianglelefteq S_5$, with quotients $A_5$ (simple, non-abelian, order 60) and $\mathbb{Z}/2\mathbb{Z}$. Since $A_5$ is non-abelian simple, the series cannot be refined to have all abelian quotients.

*Proof that $A_5$ is simple.* The conjugacy classes of $A_5$ have sizes 1, 12, 12, 15, 20. A normal subgroup must be a union of conjugacy classes containing the identity (size 1). The only subset of $\{1, 12, 12, 15, 20\}$ containing 1 whose sum divides 60 is $\{1\}$ and $\{1, 12, 12, 15, 20\}$ (the whole group). So $A_5$ has no nontrivial normal subgroups. $\square$

**Non-example (solvable cases).** $S_4$ IS solvable: $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$ with abelian quotients $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, $\mathbb{Z}/3\mathbb{Z}$, $\mathbb{Z}/2\mathbb{Z}$. This is why quartic polynomials always have radical solutions (Cardano-Ferrari formula).

## Specific solvable and unsolvable polynomials

**Solvable:** $x^4 - 2$ over $\mathbb{Q}$. The splitting field is $\mathbb{Q}(\sqrt[4]{2}, i)$ with degree 8. The Galois group is the dihedral group $D_4$ (order 8), which is solvable: $\{e\} \trianglelefteq \langle r^2 \rangle \trianglelefteq \langle r \rangle \trianglelefteq D_4$, with cyclic quotients.

**Unsolvable:** $x^5 - 4x + 2$ over $\mathbb{Q}$. Irreducible by Eisenstein at $p = 2$. Numerical analysis shows exactly three real roots (check signs at $-2, -1, 0, 1, 2$). A theorem: an irreducible quintic over $\mathbb{Q}$ with exactly two complex roots has Galois group $S_5$ (the complex conjugation is a transposition in $S_5$, and any transposition plus a 5-cycle generates all of $S_5$). Therefore $f$ is not solvable by radicals.

**Another unsolvable:** $x^5 - x - 1$ over $\mathbb{Q}$. Irreducible (check mod 2: $x^5 + x + 1 = (x^2+x+1)(x^3+x^2+1)$ — wait, that factors mod 2, so we need another approach. Actually $x^5 - x - 1$ modulo 3 is $x^5 + 2x + 2$, which is irreducible over $\mathbb{F}_3$). This polynomial has Galois group $S_5$ and is not solvable by radicals.

## Computing Galois groups in practice

Determining $\text{Gal}(f)$ uses several techniques:

**1. The discriminant.** For $f \in F[x]$ of degree $n$ with roots $\alpha_1, \ldots, \alpha_n$, the discriminant $\Delta = \prod_{i<j}(\alpha_i - \alpha_j)^2$ satisfies: $\text{Gal}(f) \subseteq A_n$ iff $\sqrt{\Delta} \in F$.

**2. Transitivity.** If $f$ is irreducible, then $\text{Gal}(f)$ acts transitively on the roots. For a separable irreducible polynomial, $|\text{Gal}(f)|$ is divisible by $\deg f$.

**3. Reduction modulo $p$.** If $f \in \mathbb{Z}[x]$ is monic and $\bar f \in \mathbb{F}_p[x]$ factors into irreducibles of degrees $d_1, \ldots, d_k$ (with $\bar f$ separable), then $\text{Gal}(f)$ contains a permutation of cycle type $(d_1, \ldots, d_k)$.

**Worked example for cubics** $x^3 + px + q$ over $\mathbb{Q}$: $\Delta = -4p^3 - 27q^2$. If $\Delta > 0$ is a perfect square, $\text{Gal}(f) = A_3 \cong \mathbb{Z}/3\mathbb{Z}$. If $\Delta$ is not a perfect square, $\text{Gal}(f) = S_3$.

For $x^3 - 3x + 1$: $\Delta = -4(-3)^3 - 27(1)^2 = 108 - 27 = 81 = 9^2$. So $\text{Gal}(x^3 - 3x + 1) = \mathbb{Z}/3\mathbb{Z}$, and the roots can be expressed using only cube roots (in fact, they are $2\cos(2\pi k/9)$ for $k = 1, 2, 4$).

## Infinite Galois theory (a glimpse)

For infinite algebraic extensions, the Galois group acquires a topology (the **Krull topology**, making it a profinite group), and the fundamental theorem holds with the modification that only *closed* subgroups correspond to intermediate fields.

The most important example: $\text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$, the **absolute Galois group** of $\mathbb{Q}$. This is a profinite group of extraordinary complexity — understanding its structure is essentially equivalent to understanding all of algebraic number theory. The Langlands program, one of the deepest research directions in modern mathematics, is in large part about representations of this group.

## What's next

Galois theory is one of the most beautiful syntheses in mathematics — connecting polynomials, fields, and groups through the single idea of symmetry. In the final article, we will see how abstract algebra powers real applications: RSA encryption, error-correcting codes, and the gauge symmetries of fundamental physics.

---

*This is Part 5 of [Abstract Algebra](/en/series/abstract-algebra/) (6 parts).
Previous: [Part 4 — Fields and Field Extensions](/en/abstract-algebra/04-fields-and-extensions/) · Next: [Part 6 — Applications](/en/abstract-algebra/06-applications/)*
