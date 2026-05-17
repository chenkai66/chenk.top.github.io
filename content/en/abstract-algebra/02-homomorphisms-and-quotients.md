---
title: "Homomorphisms, Normal Subgroups, and Quotient Groups"
date: 2021-01-17 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 6
translationKey: "abstract-algebra-2"
description: "Homomorphisms are structure-preserving maps. Normal subgroups let you build quotient groups. The First Isomorphism Theorem ties it all together."
---

## Maps that respect structure

A function between groups that preserves the operation is called a **homomorphism**. This is the right notion of "map" in group theory — it tells you when two groups are structurally related, even if their elements look completely different.

## The definition

A **group homomorphism** $\varphi: G \to H$ satisfies:

$$
\varphi(ab) = \varphi(a)\varphi(b) \quad \forall\, a, b \in G
$$

Consequences: $\varphi(e_G) = e_H$ and $\varphi(a^{-1}) = \varphi(a)^{-1}$.

The **kernel** and **image** are:

$$
\ker\varphi = \{g \in G : \varphi(g) = e_H\}, \qquad \text{im}\,\varphi = \{\varphi(g) : g \in G\}
$$

A homomorphism is injective iff $\ker\varphi = \{e_G\}$. A bijective homomorphism is an **isomorphism** — the groups are "the same" up to relabeling.

### Example 1: The sign homomorphism

Define $\text{sgn}: S_n \to \{+1, -1\}$ by mapping even permutations to $+1$ and odd permutations to $-1$, where $\{+1, -1\}$ is a group under multiplication. This is a homomorphism with $\ker(\text{sgn}) = A_n$ (the alternating group).

### Example 2: Determinant as homomorphism

The map $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$ is a homomorphism from the general linear group to the multiplicative group of nonzero reals. Its kernel is $SL_n(\mathbb{R})$, the matrices with determinant 1.

## Normal subgroups

A subgroup $N \leq G$ is **normal** (written $N \trianglelefteq G$) if it is invariant under conjugation:

$$
N \trianglelefteq G \iff gNg^{-1} = N \quad \forall\, g \in G
$$

Equivalently: the left coset $gN$ equals the right coset $Ng$ for every $g$.

Why does this matter? Because normality is exactly the condition that makes the set of cosets into a group.

**Key fact.** The kernel of any homomorphism is a normal subgroup. Conversely, every normal subgroup is the kernel of some homomorphism (namely, the projection onto the quotient).

### Example 3: $A_n \trianglelefteq S_n$

Since $A_n = \ker(\text{sgn})$, it is automatically normal in $S_n$. The index is $[S_n : A_n] = 2$. In fact, any subgroup of index 2 is normal: there are only two cosets, and the subgroup itself is one of them, so left and right cosets must coincide.

## Quotient groups

Given $N \trianglelefteq G$, the **quotient group** $G/N$ has elements that are cosets $\{gN : g \in G\}$ with the operation:

$$
(g_1 N)(g_2 N) = (g_1 g_2)N
$$

This is well-defined precisely because $N$ is normal. The order is $|G/N| = [G:N] = |G|/|N|$.

The **canonical projection** $\pi: G \to G/N$ defined by $\pi(g) = gN$ is a surjective homomorphism with kernel $N$.

### Example 4: $\mathbb{Z}/n\mathbb{Z}$ as quotient

Take $G = \mathbb{Z}$ (under addition) and $N = n\mathbb{Z} = \{0, \pm n, \pm 2n, \ldots\}$. Since $\mathbb{Z}$ is abelian, every subgroup is normal. The quotient $\mathbb{Z}/n\mathbb{Z}$ is the group of remainders mod $n$ — the same object we met in the previous article, now constructed as a quotient.

## The First Isomorphism Theorem

This is the central theorem connecting homomorphisms, kernels, and quotients.

**Theorem (First Isomorphism Theorem).** If $\varphi: G \to H$ is a homomorphism, then:

$$
G / \ker\varphi \;\cong\; \text{im}\,\varphi
$$

*Proof sketch.* Define $\bar\varphi: G/\ker\varphi \to \text{im}\,\varphi$ by $\bar\varphi(g\ker\varphi) = \varphi(g)$. Check well-definedness: if $g_1\ker\varphi = g_2\ker\varphi$, then $g_1^{-1}g_2 \in \ker\varphi$, so $\varphi(g_1) = \varphi(g_2)$. It's clearly surjective onto $\text{im}\,\varphi$. Injectivity: $\bar\varphi(g\ker\varphi) = e_H$ implies $g \in \ker\varphi$, so the coset is trivial. $\square$

**Application.** From $\det: GL_n(\mathbb{R}) \to \mathbb{R}^*$: since $\det$ is surjective and $\ker\det = SL_n(\mathbb{R})$, we get $GL_n(\mathbb{R})/SL_n(\mathbb{R}) \cong \mathbb{R}^*$.

## The other isomorphism theorems

**Second Isomorphism Theorem.** If $H \leq G$ and $N \trianglelefteq G$, then $HN/N \cong H/(H \cap N)$.

**Third Isomorphism Theorem.** If $N \subseteq M$ are both normal in $G$, then $(G/N)/(M/N) \cong G/M$.

These are essentially bookkeeping — they say quotients behave as you'd expect when you stack them.

## Cayley's theorem

**Theorem (Cayley).** Every group $G$ is isomorphic to a subgroup of some symmetric group.

*Proof sketch.* For each $g \in G$, define $\lambda_g: G \to G$ by $\lambda_g(x) = gx$. This is a bijection (a permutation of $G$). The map $g \mapsto \lambda_g$ is an injective homomorphism $G \hookrightarrow S_G$. $\square$

So every group, no matter how abstract, can be realized as a permutation group. For finite $G$ with $|G| = n$, this embeds $G$ into $S_n$. The embedding is rarely efficient (a cyclic group of order $n$ doesn't need $n!$ permutations to describe it), but it proves that group theory is, in a precise sense, the study of symmetry.

## What's next

Groups capture one operation. But integers have two — addition and multiplication — and so do polynomials. When you have two operations that interact via distributivity, you get a **ring**. Next: rings, ideals (the ring-theoretic analog of normal subgroups), and why polynomial division works.
