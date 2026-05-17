---
title: "Abstract Algebra (9): Modules — Generalizing Vector Spaces"
date: 2021-09-17 09:00:00
tags:
  - abstract-algebra
  - modules
  - linear-algebra
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "Modules over rings generalize vector spaces over fields — the structure theorem for finitely generated modules over PIDs unifies the theory of abelian groups and canonical forms."
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "abstract-algebra-9"
---

In every linear algebra course, you learn to work over a field: real numbers, complex numbers, or perhaps a finite field. The resulting theory is remarkably clean — every subspace has a complement, every finitely generated vector space has a basis, and all bases have the same cardinality. But what happens when we replace the field with a ring?

The answer is **module theory**, and it is simultaneously a natural generalization and a dramatically richer world. Some of the clean theorems survive; many do not. The payoff for studying this richer world is a single structure theorem that unifies two apparently unrelated classification results: the classification of finitely generated abelian groups and the theory of canonical forms for linear operators.

---

## Why Modules? Vector Spaces over Non-Fields

Recall the definition of a vector space $V$ over a field $F$: it is an abelian group $(V, +)$ equipped with a scalar multiplication $F \times V \to V$ satisfying the usual axioms (distributivity, associativity with field multiplication, and $1 \cdot v = v$). The key property that makes everything work smoothly is that every nonzero element of $F$ has a multiplicative inverse.

But many natural algebraic structures involve "scalar multiplication" by elements of a ring that is not a field.

![Structure theorem for finitely generated modules over PIDs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_fig9_module_structure.png)


**Example 1 (Abelian groups as $\mathbb{Z}$-modules).** Every abelian group $A$ is naturally a module over $\mathbb{Z}$. The scalar multiplication is defined by:
$$n \cdot a = \underbrace{a + a + \cdots + a}_{n \text{ times}}$$
for $n > 0$, with $0 \cdot a = 0$ and $(-n) \cdot a = -(n \cdot a)$. The ring $\mathbb{Z}$ is not a field, so we should not expect field-like behavior. Indeed, the abelian group $\mathbb{Z}/6\mathbb{Z}$ has no "basis" in any reasonable sense — the element $\bar{2}$ satisfies $3 \cdot \bar{2} = \bar{0}$, so nonzero scalars can annihilate nonzero elements.

**Example 2 ($R[x]$-modules and linear operators).** Let $V$ be a vector space over a field $F$ and let $T: V \to V$ be a linear operator. We can make $V$ into a module over the polynomial ring $F[x]$ by defining:
$$f(x) \cdot v = f(T)(v)$$
That is, $x$ acts as $T$, $x^2$ acts as $T^2$, and a general polynomial $a_0 + a_1 x + \cdots + a_n x^n$ acts as $a_0 I + a_1 T + \cdots + a_n T^n$. This single construction will allow us to derive the rational and Jordan canonical forms from the structure theorem for modules.

**Example (Group rings).** Let $G$ be a finite group and $F$ a field. The **group ring** $F[G]$ consists of all formal sums $\sum_{g \in G} a_g g$ with $a_g \in F$, with multiplication extended linearly from the group multiplication. A left $F[G]$-module is the same thing as a representation of $G$ over $F$ — we will explore this connection in detail in the next article on representation theory. For now, the takeaway is that modules over group rings unify representation theory with module theory.

These examples illustrate a general principle: **module theory is the right framework whenever you have an abelian group with a ring acting on it**. The loss of invertibility in the scalars makes the theory harder but also more expressive. In a vector space, no nonzero scalar can annihilate a nonzero vector — this is precisely what makes bases exist and dimensions well-behaved. In a module, the possibility of nonzero annihilation creates a richer landscape: torsion elements, non-free modules, and the need for a more careful structural theory.

---

## Definitions and First Examples

**Definition.** Let $R$ be a ring (with identity $1_R$). A **left $R$-module** is an abelian group $(M, +)$ together with a map $R \times M \to M$, written $(r, m) \mapsto r \cdot m$, satisfying for all $r, s \in R$ and $m, n \in M$:

1. $r \cdot (m + n) = r \cdot m + r \cdot n$
2. $(r + s) \cdot m = r \cdot m + s \cdot m$
3. $(rs) \cdot m = r \cdot (s \cdot m)$
4. $1_R \cdot m = m$

Right modules are defined analogously with scalars acting on the right. When $R$ is commutative, the distinction vanishes.

**Why left vs. right matters.** For a noncommutative ring $R$, left and right modules are genuinely different concepts. If $M$ is a left $R$-module, the axiom $(rs) \cdot m = r \cdot (s \cdot m)$ means that the "inner" scalar $s$ acts first. For a right module, one writes $m \cdot (rs) = (m \cdot r) \cdot s$, so the "outer" scalar $s$ acts first. When $R$ is commutative, the two notions coincide, and we simply speak of $R$-modules. In this article we will mostly work with commutative rings, so the distinction is not critical, but it becomes essential in representation theory and noncommutative algebra.

**Example 3 (Ideals as modules).** If $R$ is a ring, then any left ideal $I \subseteq R$ is a left $R$-module under the ring multiplication. In particular, $R$ itself is a left $R$-module.

**Example 4 (Direct sums).** If $M_1, \ldots, M_n$ are $R$-modules, their direct sum $M_1 \oplus \cdots \oplus M_n$ is an $R$-module under componentwise operations: $r \cdot (m_1, \ldots, m_n) = (r \cdot m_1, \ldots, r \cdot m_n)$.

**Example 5 (Matrix modules).** The set $R^n$ of column vectors with entries in $R$ is a left $R$-module. More generally, the set $M_{m \times n}(R)$ of $m \times n$ matrices over $R$ is both a left $M_{m \times m}(R)$-module and a right $M_{n \times n}(R)$-module.

The translation between module-theoretic language and specific algebraic contexts is worth internalizing:

| Module context | $\mathbb{Z}$-modules | $F$-modules ($F$ a field) | $F[x]$-modules |
|---|---|---|---|
| Module | Abelian group | Vector space | Vector space + operator |
| Submodule | Subgroup | Subspace | $T$-invariant subspace |
| Module homomorphism | Group homomorphism | Linear map | Intertwining operator |

**Torsion.** An element $m \in M$ is a **torsion element** if there exists a nonzero $r \in R$ with $r \cdot m = 0$. The set of all torsion elements $\operatorname{Tor}(M) = \{m \in M : r \cdot m = 0 \text{ for some } r \neq 0\}$ is a submodule when $R$ is an integral domain (if $r_1 m_1 = 0$ and $r_2 m_2 = 0$, then $r_1 r_2 (m_1 + m_2) = r_2(r_1 m_1) + r_1(r_2 m_2) = 0$, and $r_1 r_2 \neq 0$ since $R$ is a domain). A module is **torsion-free** if $\operatorname{Tor}(M) = 0$, and a **torsion module** if $\operatorname{Tor}(M) = M$. In a vector space, the only torsion element is $0$ — so every vector space is torsion-free. But the $\mathbb{Z}$-module $\mathbb{Z}/n\mathbb{Z}$ is entirely torsion: every element $\bar{k}$ satisfies $n \cdot \bar{k} = \bar{0}$.

---

## Submodules, Quotients, and Homomorphisms

The basic constructions from group theory and linear algebra carry over to modules with the expected modifications.

**Definition.** A **submodule** $N$ of an $R$-module $M$ is a subgroup $N \leq M$ (under addition) that is closed under scalar multiplication: $r \cdot n \in N$ for all $r \in R$, $n \in N$.

**Definition.** If $N$ is a submodule of $M$, the **quotient module** $M/N$ is the quotient group $M/N$ with scalar multiplication $r \cdot (m + N) = (r \cdot m) + N$. This is well-defined precisely because $N$ is a submodule.

**Definition.** A **module homomorphism** (or $R$-linear map) $\varphi: M \to N$ is a map satisfying $\varphi(m_1 + m_2) = \varphi(m_1) + \varphi(m_2)$ and $\varphi(r \cdot m) = r \cdot \varphi(m)$ for all $r \in R$ and $m, m_1, m_2 \in M$.

**The isomorphism theorems** hold for modules, with identical statements and nearly identical proofs to the group-theoretic versions:

**First Isomorphism Theorem.** If $\varphi: M \to N$ is an $R$-module homomorphism, then $M / \ker \varphi \cong \operatorname{im} \varphi$.

*Proof sketch.* The map $m + \ker \varphi \mapsto \varphi(m)$ is well-defined (if $m - m' \in \ker \varphi$, then $\varphi(m) = \varphi(m')$), injective by construction, surjective onto $\operatorname{im} \varphi$, and $R$-linear because $\varphi$ is. $\square$

**Example 6.** Consider $\mathbb{Z}$ as a $\mathbb{Z}$-module and the homomorphism $\varphi: \mathbb{Z} \to \mathbb{Z}/n\mathbb{Z}$ sending $k \mapsto \bar{k}$. Then $\ker \varphi = n\mathbb{Z}$, and the first isomorphism theorem gives $\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}/n\mathbb{Z}$ — a tautology here, but the machinery becomes powerful in more complex settings.

**Annihilators.** For an element $m \in M$, the **annihilator** $\operatorname{Ann}(m) = \{r \in R : r \cdot m = 0\}$ is a left ideal of $R$. For a submodule or the whole module, $\operatorname{Ann}(M) = \{r \in R : r \cdot m = 0 \text{ for all } m \in M\}$ is a two-sided ideal. The annihilator captures the "torsion" behavior that distinguishes modules from vector spaces.

**Second and Third Isomorphism Theorems.** These also hold for modules:

**Second Isomorphism Theorem.** If $N_1, N_2$ are submodules of $M$, then $(N_1 + N_2)/N_2 \cong N_1/(N_1 \cap N_2)$.

**Third Isomorphism Theorem.** If $N_1 \subseteq N_2 \subseteq M$ are submodules, then $(M/N_1)/(N_2/N_1) \cong M/N_2$.

The proofs are essentially identical to the group-theoretic versions, with the additional (trivial) check that the isomorphisms preserve scalar multiplication.

**Example.** In the $\mathbb{Z}$-module $\mathbb{Z}$, the submodules $6\mathbb{Z}$ and $4\mathbb{Z}$ satisfy $6\mathbb{Z} + 4\mathbb{Z} = 2\mathbb{Z}$ (since $\gcd(4,6) = 2$) and $6\mathbb{Z} \cap 4\mathbb{Z} = 12\mathbb{Z}$ (since $\operatorname{lcm}(4,6) = 12$). The second isomorphism theorem gives $2\mathbb{Z}/4\mathbb{Z} \cong 6\mathbb{Z}/12\mathbb{Z}$, which is $\mathbb{Z}/2\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z}$ — consistent.

**Direct sum decompositions.** An $R$-module $M$ is the **internal direct sum** of submodules $N_1, \ldots, N_k$ if $M = N_1 + \cdots + N_k$ and $N_i \cap (N_1 + \cdots + N_{i-1} + N_{i+1} + \cdots + N_k) = 0$ for each $i$. This is equivalent to saying that every $m \in M$ can be written uniquely as $m = n_1 + \cdots + n_k$ with $n_i \in N_i$. Internal direct sums are isomorphic to external direct sums: $M \cong N_1 \oplus \cdots \oplus N_k$.

---

## Free Modules and Bases

In a vector space, every element can be written uniquely as a linear combination of basis vectors. Can we do the same for modules?

**Definition.** An $R$-module $M$ is **free** with basis $\{e_i\}_{i \in I}$ if every element $m \in M$ can be written uniquely as $m = \sum_{i \in I} r_i e_i$ where $r_i \in R$ and only finitely many $r_i$ are nonzero. Equivalently, $M \cong \bigoplus_{i \in I} R$ (a direct sum of copies of $R$ as a module over itself).

**Key difference from vector spaces:** Not every module is free, and not every module even has a basis.

**Example 7 (A non-free module).** The $\mathbb{Z}$-module $\mathbb{Z}/2\mathbb{Z}$ is not free. If it were, there would be an element $e$ such that $\mathbb{Z}/2\mathbb{Z} = \{ne : n \in \mathbb{Z}\}$ with $ne = 0$ implying $n = 0$. But $2 \cdot \bar{1} = \bar{0}$ while $2 \neq 0$ in $\mathbb{Z}$, a contradiction.

**Example 8 (Free modules that exist).** The module $\mathbb{Z}^n$ is a free $\mathbb{Z}$-module of rank $n$, with the standard basis vectors $e_1, \ldots, e_n$.

**Rank.** For a commutative ring $R$, the rank of a free $R$-module is well-defined: if $R^m \cong R^n$ as $R$-modules, then $m = n$. (This is proved by reducing modulo a maximal ideal and using the field case.) However, for noncommutative rings, this invariant basis property can fail — there exist rings where $R \cong R^2$ as left modules.

**Theorem (Every module is a quotient of a free module).** For any $R$-module $M$, there exists a free module $F$ and a surjective homomorphism $\pi: F \to M$.

*Proof sketch.* Take $F = \bigoplus_{m \in M} R$ (one copy of $R$ for each element of $M$), and define $\pi$ to send the generator corresponding to $m$ to $m$ itself. This map is surjective and $R$-linear. $\square$

This means every module can be presented as $F/K$ where $F$ is free and $K = \ker \pi$ is a submodule of $F$. When $R$ is a PID and $F$ is finitely generated, we can say much more about the structure of $K$ and hence of $M$.

**Projective and injective modules.** While we will not develop these concepts in full, they are worth mentioning. A module $P$ is **projective** if every surjection $M \to P$ splits (i.e., $P$ is always a direct summand). Free modules are projective, but the converse fails in general. A module $Q$ is **injective** if every injection $Q \to M$ splits. Over a PID, a module is projective if and only if it is free (this is a theorem), and a module is injective if and only if it is divisible (meaning for every $r \neq 0$ and $q \in Q$, there exists $q'$ with $rq' = q$). These concepts become central in homological algebra.

**Exact sequences.** A sequence of module homomorphisms $\cdots \to M_{i-1} \xrightarrow{f_{i-1}} M_i \xrightarrow{f_i} M_{i+1} \to \cdots$ is **exact** at $M_i$ if $\operatorname{im} f_{i-1} = \ker f_i$. A **short exact sequence** $0 \to A \xrightarrow{f} B \xrightarrow{g} C \to 0$ means $f$ is injective, $g$ is surjective, and $\operatorname{im} f = \ker g$. In other words, $A$ is (isomorphic to) a submodule of $B$, and $C \cong B/A$. The short exact sequence **splits** if $B \cong A \oplus C$. Over a field, every short exact sequence of vector spaces splits — but over a general ring, this fails, and the failure is measured by the Ext functor in homological algebra.

---

## The Structure Theorem for Finitely Generated Modules over PIDs

This is the crown jewel of module theory at the undergraduate level. It provides a complete classification of finitely generated modules over principal ideal domains.

**Theorem (Structure Theorem — Invariant Factor Form).** Let $R$ be a PID and $M$ a finitely generated $R$-module. Then there exist unique elements $d_1, d_2, \ldots, d_k \in R$ (nonzero, non-units) with $d_1 \mid d_2 \mid \cdots \mid d_k$, and a unique non-negative integer $r$, such that:
$$M \cong R^r \oplus R/(d_1) \oplus R/(d_2) \oplus \cdots \oplus R/(d_k)$$

The integer $r$ is the **free rank** of $M$, and $d_1, \ldots, d_k$ are the **invariant factors**.

**Theorem (Structure Theorem — Elementary Divisor Form).** Alternatively, there exist (not necessarily unique) irreducible elements $p_1, \ldots, p_s \in R$ and positive integers $a_1, \ldots, a_s$ such that:
$$M \cong R^r \oplus R/(p_1^{a_1}) \oplus R/(p_2^{a_2}) \oplus \cdots \oplus R/(p_s^{a_s})$$

The prime powers $p_i^{a_i}$ are the **elementary divisors**, determined up to associates.

*Proof sketch (key ideas).* The proof proceeds in several steps:

1. **Submodules of free modules over PIDs are free.** If $F$ is a free $R$-module of rank $n$ over a PID $R$, then every submodule of $F$ is free of rank $\leq n$. (This is proved by induction on $n$, using the fact that $R$ is a PID to ensure that the "leading coefficients" form an ideal, hence a principal ideal.)

2. **Smith normal form.** Write $M = F/K$ where $F \cong R^n$. The submodule $K$ is free of rank $m \leq n$. Choose bases for $F$ and $K$ and form the $n \times m$ **relations matrix** $A$. By row and column operations (which correspond to basis changes in $F$ and $K$), reduce $A$ to Smith normal form:
$$A \sim \begin{pmatrix} d_1 & & & \\ & d_2 & & \\ & & \ddots & \\ & & & d_m \\ & & & \\ & & & \end{pmatrix}$$
where $d_1 \mid d_2 \mid \cdots \mid d_m$ and all off-diagonal entries are zero.

3. **Read off the decomposition.** In the new bases, $M \cong R/(d_1) \oplus \cdots \oplus R/(d_m) \oplus R^{n-m}$. The factors $R/(d_i)$ where $d_i$ is a unit give trivial summands (which we discard), and the remaining $d_i$ are the invariant factors.

4. **Elementary divisors from invariant factors.** By the Chinese Remainder Theorem for PIDs, if $d = p_1^{a_1} \cdots p_t^{a_t}$ with the $p_i$ pairwise non-associate irreducibles, then $R/(d) \cong R/(p_1^{a_1}) \oplus \cdots \oplus R/(p_t^{a_t})$.

**Uniqueness** of invariant factors follows from the fact that the $i$-th invariant factor equals $\Delta_i(A)/\Delta_{i-1}(A)$ where $\Delta_i(A)$ is the GCD of all $i \times i$ minors of $A$ (with $\Delta_0 = 1$), and these GCDs are invariant under row/column operations.

**Worked Example.** Let $R = \mathbb{Z}$ and consider the $\mathbb{Z}$-module $M$ presented by the matrix:
$$A = \begin{pmatrix} 2 & 4 \\ 6 & 12 \end{pmatrix}$$

We reduce to Smith normal form. Subtract 3 times row 1 from row 2:
$$\begin{pmatrix} 2 & 4 \\ 0 & 0 \end{pmatrix}$$
Subtract 2 times column 1 from column 2:
$$\begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}$$

So the Smith normal form is $\operatorname{diag}(2, 0)$, giving $M \cong \mathbb{Z}/(2) \oplus \mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}$. The module has free rank 1 and one invariant factor $d_1 = 2$.

**Worked Example 2.** Consider the $\mathbb{Z}$-module presented by the $3 \times 2$ matrix:
$$A = \begin{pmatrix} 2 & 0 \\ 0 & 6 \\ 4 & 6 \end{pmatrix}$$

We compute the Smith normal form step by step. First, $\Delta_1 = \gcd$ of all entries $= \gcd(2, 0, 0, 6, 4, 6) = 2$. For $\Delta_2$, we need the GCD of all $2 \times 2$ minors. The minors are: $\det\begin{pmatrix}2 & 0\\0 & 6\end{pmatrix} = 12$, $\det\begin{pmatrix}2 & 0\\4 & 6\end{pmatrix} = 12$, $\det\begin{pmatrix}0 & 6\\4 & 6\end{pmatrix} = -24$. So $\Delta_2 = \gcd(12, 12, 24) = 12$. The invariant factors are $d_1 = \Delta_1/\Delta_0 = 2/1 = 2$ and $d_2 = \Delta_2/\Delta_1 = 12/2 = 6$. Since $M$ has 3 generators and 2 relations, the free rank is $r = 3 - 2 = 1$.

Therefore $M \cong \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/6\mathbb{Z} \oplus \mathbb{Z}$. In elementary divisor form: $\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z} \oplus \mathbb{Z}$ (since $6 = 2 \cdot 3$ and $\gcd(2,3) = 1$).

**The torsion submodule and the free part.** For any finitely generated module $M$ over a PID $R$, the structure theorem implies $M \cong M_{\mathrm{tor}} \oplus R^r$, where $M_{\mathrm{tor}} = R/(d_1) \oplus \cdots \oplus R/(d_k)$ is the torsion submodule and $R^r$ is the free part. The torsion submodule and the free rank are both invariants of $M$. Note that this clean decomposition into torsion plus free is specific to PIDs; over more general rings, the torsion submodule may not have a complement.

---

## Applications: Abelian Groups and Canonical Forms

The structure theorem, applied to specific choices of PID, immediately yields two major classification results.

### Classification of Finitely Generated Abelian Groups

Take $R = \mathbb{Z}$. Every finitely generated abelian group $G$ is a $\mathbb{Z}$-module, and the structure theorem gives:

$$G \cong \mathbb{Z}^r \oplus \mathbb{Z}/d_1\mathbb{Z} \oplus \cdots \oplus \mathbb{Z}/d_k\mathbb{Z}$$

where $d_1 \mid d_2 \mid \cdots \mid d_k$ (invariant factor form), or equivalently:

$$G \cong \mathbb{Z}^r \oplus \mathbb{Z}/p_1^{a_1}\mathbb{Z} \oplus \cdots \oplus \mathbb{Z}/p_s^{a_s}\mathbb{Z}$$

(elementary divisor form).

**Worked Example.** Classify all abelian groups of order 36. Since $36 = 2^2 \cdot 3^2$, the elementary divisors must be powers of 2 and 3 whose product is 36 (and the free rank is 0). The possibilities are:

| Elementary divisors | Group |
|---|---|
| $2^2, 3^2$ | $\mathbb{Z}/4\mathbb{Z} \oplus \mathbb{Z}/9\mathbb{Z} \cong \mathbb{Z}/36\mathbb{Z}$ |
| $2, 2, 3^2$ | $\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/9\mathbb{Z}$ |
| $2^2, 3, 3$ | $\mathbb{Z}/4\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z}$ |
| $2, 2, 3, 3$ | $\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z}$ |

So there are exactly 4 abelian groups of order 36 (up to isomorphism). Converting to invariant factor form: $\mathbb{Z}/36\mathbb{Z}$; $\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/18\mathbb{Z}$; $\mathbb{Z}/3\mathbb{Z} \oplus \mathbb{Z}/12\mathbb{Z}$; $\mathbb{Z}/6\mathbb{Z} \oplus \mathbb{Z}/6\mathbb{Z}$.

### Rational and Jordan Canonical Forms

Take $R = F[x]$ where $F$ is a field, and let $V$ be a finite-dimensional $F$-vector space with a linear operator $T: V \to V$. As we noted, $V$ becomes an $F[x]$-module.

The structure theorem gives:
$$V \cong F[x]/(d_1(x)) \oplus \cdots \oplus F[x]/(d_k(x))$$
where $d_1(x) \mid \cdots \mid d_k(x)$ are the invariant factors. (There is no free part because $V$ is finite-dimensional but $F[x]$ is infinite-dimensional.)

The largest invariant factor $d_k(x)$ is the **minimal polynomial** of $T$, and the product $d_1(x) \cdots d_k(x)$ is the **characteristic polynomial**. Each summand $F[x]/(d_i(x))$ corresponds to a **companion matrix** block, and stacking these blocks gives the **rational canonical form** of $T$.

If we further decompose using $F[x]/(p(x)^a)$ factors (elementary divisor form), and if $F$ is algebraically closed so that all irreducibles are linear, we get:
$$V \cong \bigoplus_{i} F[x]/((x - \lambda_i)^{a_i})$$

Each summand $F[x]/((x-\lambda)^a)$ corresponds to a **Jordan block** $J_a(\lambda)$, and stacking these gives the **Jordan canonical form**.

**Worked Example.** Let $T: \mathbb{R}^4 \to \mathbb{R}^4$ have characteristic polynomial $(x-1)^2(x-2)^2$ and minimal polynomial $(x-1)^2(x-2)$. The invariant factors must divide the minimal polynomial and their product must be the characteristic polynomial. The only possibility is invariant factors $(x-2), (x-1)^2(x-2)$. (Check: product is $(x-1)^2(x-2)^2$ and the second divides... wait, we need the first to divide the second. We have $(x-2) \mid (x-1)^2(x-2)$, which is true.)

The rational canonical form has two companion matrix blocks corresponding to $d_1(x) = x-2$ and $d_2(x) = (x-1)^2(x-2) = x^3 - 4x^2 + 5x - 2$.

The elementary divisors are $(x-1)^2, (x-2), (x-2)$, giving Jordan form $\operatorname{diag}(J_2(1), J_1(2), J_1(2))$.

**Why this matters.** The beauty of the module-theoretic approach to canonical forms is its conceptual clarity. Instead of ad hoc manipulations of matrices, we apply a single structural result (the structure theorem) to a specific module ($V$ as an $F[x]$-module). The invariant factors and elementary divisors arise naturally, and the canonical forms are just their matrix representations.

Moreover, this approach reveals a deep analogy: **the classification of finitely generated abelian groups and the classification of linear operators up to similarity are the same theorem**, applied to different rings ($\mathbb{Z}$ vs. $F[x]$). This unification is one of the great achievements of abstract algebra.

**Worked Example 2 (Abelian groups of order 12).** We have $12 = 2^2 \cdot 3$. The partitions of the prime powers are: for $2^2$, either $\{4\}$ or $\{2, 2\}$; for $3$, only $\{3\}$. So the abelian groups of order 12 are:
- $\mathbb{Z}/4\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}/12\mathbb{Z}$ (invariant factor: 12)
- $\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/6\mathbb{Z}$ (invariant factors: 2, 6)

Exactly 2 groups, up to isomorphism. Note how the invariant factor form requires $d_1 \mid d_2$: we need 2 | 6, which is satisfied.

---

## What's Next

Modules give us a unified framework that reveals deep connections between abelian group theory and linear algebra. The structure theorem over PIDs is one of the most powerful results in algebra — it takes one theorem to prove what previously required separate arguments for each application.

Let us briefly reflect on the key ideas:

- **Modules generalize vector spaces** by allowing scalars from a ring instead of a field. The loss of invertibility creates torsion elements and non-free modules.
- **Submodules, quotients, and homomorphisms** work exactly as expected, with the full suite of isomorphism theorems carrying over from group theory.
- **Free modules** are the modules with bases, but unlike vector spaces, not every module is free. However, every module is a quotient of a free module.
- **The structure theorem for finitely generated modules over PIDs** provides a complete decomposition into cyclic summands, in either invariant factor or elementary divisor form.
- **Applications** include the classification of finitely generated abelian groups ($R = \mathbb{Z}$) and the rational and Jordan canonical forms for linear operators ($R = F[x]$).

The tools we developed here — especially the interplay between module presentations (relations matrices) and the Smith normal form — are computational workhorses that appear throughout algebra and its applications.

In the next article, we turn to **representation theory**, which asks: given an abstract group $G$, how can we study it by making it act on vector spaces? This is module theory in disguise — a representation of $G$ over a field $F$ is the same thing as a module over the group ring $F[G]$ — and the resulting theory is both beautiful and enormously useful, from pure mathematics to quantum mechanics.

---

*This is Part 9 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 8 — Galois Theory](/en/abstract-algebra/08-galois-theory/)*

*Next: [Part 10 — Representation Theory](/en/abstract-algebra/10-representation-theory/)*
