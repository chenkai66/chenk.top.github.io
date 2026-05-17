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

A **field** is a commutative ring with unity in which every nonzero element has a multiplicative inverse. Equivalently, a field has exactly two ideals: $(0)$ and the whole ring.

$$
F \text{ is a field} \iff \forall\, a \neq 0,\; \exists\, a^{-1} : aa^{-1} = 1
$$

The standard examples: $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$, and the finite fields $\mathbb{F}_p = \mathbb{Z}/p\mathbb{Z}$ for prime $p$.

## Field extensions

A **field extension** $F \subseteq K$ (written $K/F$) is a pair of fields where $F$ is a subfield of $K$. We view $K$ as a vector space over $F$. The dimension $[K:F] = \dim_F K$ is the **degree** of the extension.

$$
[K:F] = \dim_F K
$$

**Tower law.** If $F \subseteq L \subseteq K$, then $[K:F] = [K:L] \cdot [L:F]$.

### Example 1: $\mathbb{Q}(\sqrt{2})/\mathbb{Q}$

The field $\mathbb{Q}(\sqrt{2}) = \{a + b\sqrt{2} : a, b \in \mathbb{Q}\}$ is a degree-2 extension of $\mathbb{Q}$. A basis as a $\mathbb{Q}$-vector space is $\{1, \sqrt{2}\}$.

To verify it's a field: given $a + b\sqrt{2} \neq 0$, its inverse is $\frac{a - b\sqrt{2}}{a^2 - 2b^2}$. The denominator $a^2 - 2b^2 \neq 0$ because $\sqrt{2}$ is irrational.

## Algebraic and transcendental elements

An element $\alpha \in K$ is **algebraic** over $F$ if it satisfies some nonzero polynomial with coefficients in $F$. Otherwise, $\alpha$ is **transcendental** over $F$.

For algebraic $\alpha$, the **minimal polynomial** $m_\alpha(x) \in F[x]$ is the unique monic irreducible polynomial with $m_\alpha(\alpha) = 0$.

**Theorem.** If $\alpha$ is algebraic over $F$ with minimal polynomial of degree $n$, then:

$$
F(\alpha) \cong F[x]/(m_\alpha(x)), \quad [F(\alpha):F] = n
$$

A basis for $F(\alpha)$ over $F$ is $\{1, \alpha, \alpha^2, \ldots, \alpha^{n-1}\}$.

*Proof sketch.* The evaluation map $\text{ev}_\alpha: F[x] \to K$ sending $f(x) \mapsto f(\alpha)$ is a ring homomorphism with kernel $(m_\alpha)$. Since $m_\alpha$ is irreducible, $(m_\alpha)$ is maximal in $F[x]$, so $F[x]/(m_\alpha) \cong \text{im}(\text{ev}_\alpha) = F(\alpha)$ is a field. $\square$

### Example 2: $\mathbb{Q}(\sqrt[3]{2})$

The minimal polynomial of $\sqrt[3]{2}$ over $\mathbb{Q}$ is $x^3 - 2$ (irreducible by Eisenstein's criterion at $p=2$). So $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$ and a basis is $\{1, \sqrt[3]{2}, \sqrt[3]{4}\}$.

Multiplication in this field: $(a + b\sqrt[3]{2} + c\sqrt[3]{4})(d + e\sqrt[3]{2} + f\sqrt[3]{4})$ expands and reduces using $(\sqrt[3]{2})^3 = 2$.

## Splitting fields

A polynomial $f \in F[x]$ **splits** over $K$ if it factors into linear factors in $K[x]$. The **splitting field** of $f$ over $F$ is the smallest extension of $F$ over which $f$ splits completely.

$$
\text{Split}_F(f) = F(\alpha_1, \ldots, \alpha_n) \text{ where } f(x) = c(x-\alpha_1)\cdots(x-\alpha_n)
$$

**Theorem.** The splitting field exists and is unique up to isomorphism.

### Example 3: Splitting field of $x^3 - 2$ over $\mathbb{Q}$

The roots of $x^3 - 2$ are $\sqrt[3]{2}$, $\omega\sqrt[3]{2}$, and $\omega^2\sqrt[3]{2}$, where $\omega = e^{2\pi i/3}$ is a primitive cube root of unity. The splitting field is $\mathbb{Q}(\sqrt[3]{2}, \omega)$.

We compute the degree: $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3$. The element $\omega$ satisfies $x^2 + x + 1 = 0$ over $\mathbb{Q}$, so $[\mathbb{Q}(\omega):\mathbb{Q}] = 2$. Since $\gcd(2, 3) = 1$, by the tower law, $[\mathbb{Q}(\sqrt[3]{2}, \omega):\mathbb{Q}] = 6$.

## Finite fields

**Theorem.** For every prime power $q = p^n$, there exists a unique (up to isomorphism) field with $q$ elements, denoted $\mathbb{F}_q$ or $GF(q)$.

Construction: $\mathbb{F}_{p^n}$ is the splitting field of $x^{p^n} - x$ over $\mathbb{F}_p$. Equivalently, take any irreducible polynomial $f$ of degree $n$ in $\mathbb{F}_p[x]$, and form $\mathbb{F}_p[x]/(f)$.

### Example 4: $\mathbb{F}_4$

Take $\mathbb{F}_2[x]/(x^2 + x + 1)$. Let $\alpha$ be the image of $x$. Then $\mathbb{F}_4 = \{0, 1, \alpha, \alpha+1\}$ with $\alpha^2 = \alpha + 1$ (since we're in characteristic 2, $\alpha^2 + \alpha + 1 = 0$ means $\alpha^2 = -\alpha - 1 = \alpha + 1$).

Multiplication table: $\alpha \cdot \alpha = \alpha + 1$, $\alpha \cdot (\alpha + 1) = \alpha^2 + \alpha = (\alpha+1) + \alpha = 1$. So $\alpha^{-1} = \alpha + 1$.

The multiplicative group $\mathbb{F}_4^* = \{1, \alpha, \alpha+1\}$ is cyclic of order 3.

## Algebraic closure

A field $F$ is **algebraically closed** if every non-constant polynomial in $F[x]$ has a root in $F$. The **algebraic closure** $\overline{F}$ is the smallest algebraically closed field containing $F$.

$\mathbb{C}$ is the algebraic closure of $\mathbb{R}$ (by the Fundamental Theorem of Algebra). The algebraic closure $\overline{\mathbb{Q}}$ — the algebraic numbers — is a countable field containing all roots of polynomials with rational coefficients.

## Separability

A polynomial is **separable** if it has no repeated roots (in its splitting field). An algebraic extension $K/F$ is separable if the minimal polynomial of every element is separable.

In characteristic 0 (like $\mathbb{Q}$, $\mathbb{R}$), every irreducible polynomial is separable — this is automatic. In characteristic $p$, things can go wrong: $x^p - a$ might be irreducible but have the repeated root $\alpha$ (where $\alpha^p = a$) in an extension.

Separability matters because it's a hypothesis of the Fundamental Theorem of Galois Theory.

## What's next

We have all the ingredients: field extensions, splitting fields, and the degree formula. Next time: the Galois group of an extension, the fundamental theorem connecting subgroups to intermediate fields, and why there's no formula for quintic roots.
