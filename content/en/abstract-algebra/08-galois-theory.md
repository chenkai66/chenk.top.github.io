---
title: "Abstract Algebra (8): Galois Theory — The Bridge Between Fields and Groups"
date: 2021-09-15 09:00:00
tags:
  - abstract-algebra
  - galois-theory
  - field-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "The Fundamental Theorem of Galois Theory establishes a perfect correspondence between intermediate fields and subgroups — and settles the ancient question of solvability by radicals."
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "abstract-algebra-8"
---

In 1832, a twenty-year-old mathematician named Evariste Galois, on the eve of a duel he would not survive, wrote down the final versions of his mathematical ideas in a letter to a friend. Those ideas — connecting the symmetries of polynomial roots to the structure of groups — would take more than a decade to be understood and published, but they would reshape algebra forever. Galois theory, as we now call it, establishes a precise dictionary between intermediate fields of a field extension and subgroups of a group of symmetries. It explains, in one elegant framework, why the quadratic formula exists, why there is no analogous formula for degree five, and what "solvability" really means.

In the previous article, we built up the machinery of field extensions: degrees, minimal polynomials, the tower law, splitting fields, and separability. We now put all of it to work.

---

## The Galois Group: Automorphisms Fixing the Base Field

**Definition.** Let $L/K$ be a field extension. A *$K$-automorphism* of $L$ is a field isomorphism $\sigma : L \to L$ such that $\sigma(a) = a$ for all $a \in K$. The set of all $K$-automorphisms of $L$ forms a group under composition, called the *Galois group* of $L$ over $K$:

$$\operatorname{Gal}(L/K) = \{\sigma \in \operatorname{Aut}(L) : \sigma|_K = \operatorname{id}_K\}.$$

![Galois correspondence between subgroups and intermediate fields](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_fig8_galois_correspondence.png)


A $K$-automorphism is determined by where it sends the generators of $L$ over $K$. Since $\sigma$ must preserve all polynomial relations with coefficients in $K$, it can only permute the roots of irreducible polynomials over $K$.

**Key observation.** If $\alpha \in L$ is a root of an irreducible polynomial $p(x) \in K[x]$, and $\sigma \in \operatorname{Gal}(L/K)$, then $\sigma(\alpha)$ is also a root of $p$. This is because:

$$p(\sigma(\alpha)) = \sigma(p(\alpha)) = \sigma(0) = 0,$$

where we used the fact that $\sigma$ fixes $K$ (and hence all coefficients of $p$) and preserves addition and multiplication.

**Example 1 ($\operatorname{Gal}(\mathbb{C}/\mathbb{R})$).** Any $\mathbb{R}$-automorphism $\sigma$ of $\mathbb{C}$ must send $i$ to a root of $x^2 + 1$, so $\sigma(i) = i$ or $\sigma(i) = -i$. The first gives the identity, the second gives complex conjugation. Therefore $\operatorname{Gal}(\mathbb{C}/\mathbb{R}) = \{1, \overline{\cdot}\} \cong \mathbb{Z}/2\mathbb{Z}$.

**Example 2 ($\operatorname{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$).** An automorphism $\sigma$ must send $\sqrt{2}$ to a root of $x^2 - 2$, so $\sigma(\sqrt{2}) = \pm\sqrt{2}$. This gives two automorphisms: the identity and $\sigma: a + b\sqrt{2} \mapsto a - b\sqrt{2}$. So $\operatorname{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q}) \cong \mathbb{Z}/2\mathbb{Z}$.

**Example 3 ($\operatorname{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})$).** A $\mathbb{Q}$-automorphism $\sigma$ must send $\sqrt[3]{2}$ to a root of $x^3 - 2$. The roots are $\sqrt[3]{2}$, $\sqrt[3]{2}\omega$, and $\sqrt[3]{2}\omega^2$ (where $\omega = e^{2\pi i/3}$). But $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$, so $\sigma(\sqrt[3]{2})$ must be real. The only real root is $\sqrt[3]{2}$ itself, so $\sigma = \operatorname{id}$. Therefore $\operatorname{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}) = \{1\}$ — the trivial group.

This last example reveals a crucial point: the Galois group can be "too small." We have $|\operatorname{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})| = 1 < 3 = [\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}]$. The extension $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ is not normal (not a splitting field), so it is not a Galois extension. When the extension *is* Galois, we get a perfect match.

**Theorem.** If $L/K$ is a finite Galois extension (i.e., normal and separable), then

$$|\operatorname{Gal}(L/K)| = [L:K].$$

The proof uses the primitive element theorem and properties of separable extensions. We will see this equality at work throughout the examples below.

---

## Fixed Fields and the Galois Correspondence

**Definition.** Let $L$ be a field and $H$ a subgroup of $\operatorname{Aut}(L)$. The *fixed field* of $H$ is

$$L^H = \{a \in L : \sigma(a) = a \text{ for all } \sigma \in H\}.$$

It is straightforward to verify that $L^H$ is indeed a subfield of $L$.

The Galois correspondence is the pair of maps:

$$\{\text{intermediate fields } K \subseteq M \subseteq L\} \quad \longleftrightarrow \quad \{\text{subgroups } H \leq \operatorname{Gal}(L/K)\}$$

given by $M \mapsto \operatorname{Gal}(L/M)$ in one direction, and $H \mapsto L^H$ in the other.

### Worked Example: $\operatorname{Gal}(\mathbb{Q}(\sqrt{2},\sqrt{3})/\mathbb{Q})$

This is the simplest non-trivial example that illustrates the full correspondence. From the previous article, we know $[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}] = 4$.

The field $\mathbb{Q}(\sqrt{2},\sqrt{3})$ is the splitting field of $(x^2-2)(x^2-3)$ over $\mathbb{Q}$, hence normal. Both $x^2 - 2$ and $x^2 - 3$ are separable (we are in characteristic 0). So this is a Galois extension, and $|\operatorname{Gal}(\mathbb{Q}(\sqrt{2},\sqrt{3})/\mathbb{Q})| = 4$.

A $\mathbb{Q}$-automorphism $\sigma$ is determined by $\sigma(\sqrt{2})$ and $\sigma(\sqrt{3})$, each of which is $\pm$ the original. This gives four automorphisms:

| Automorphism | $\sigma(\sqrt{2})$ | $\sigma(\sqrt{3})$ |
|:---:|:---:|:---:|
| $e$ (identity) | $\sqrt{2}$ | $\sqrt{3}$ |
| $\sigma_1$ | $-\sqrt{2}$ | $\sqrt{3}$ |
| $\sigma_2$ | $\sqrt{2}$ | $-\sqrt{3}$ |
| $\sigma_3$ | $-\sqrt{2}$ | $-\sqrt{3}$ |

Since each $\sigma_i$ has order 2, and $\sigma_3 = \sigma_1 \circ \sigma_2$, the group is $G = \{e, \sigma_1, \sigma_2, \sigma_3\} \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ (the Klein four-group $V_4$).

The subgroups of $V_4$ are:

| Subgroup $H$ | $|H|$ | Fixed field $L^H$ | $[L^H : \mathbb{Q}]$ |
|:---:|:---:|:---:|:---:|
| $\{e\}$ | 1 | $\mathbb{Q}(\sqrt{2},\sqrt{3})$ | 4 |
| $\{e, \sigma_1\}$ | 2 | $\mathbb{Q}(\sqrt{3})$ | 2 |
| $\{e, \sigma_2\}$ | 2 | $\mathbb{Q}(\sqrt{2})$ | 2 |
| $\{e, \sigma_3\}$ | 2 | $\mathbb{Q}(\sqrt{6})$ | 2 |
| $G$ | 4 | $\mathbb{Q}$ | 1 |

Let us verify one of these. Consider $H = \{e, \sigma_3\}$. A general element of $\mathbb{Q}(\sqrt{2},\sqrt{3})$ is $a + b\sqrt{2} + c\sqrt{3} + d\sqrt{6}$ with $a,b,c,d \in \mathbb{Q}$. Applying $\sigma_3$: $a - b\sqrt{2} - c\sqrt{3} + d\sqrt{6}$ (since $\sigma_3(\sqrt{6}) = \sigma_3(\sqrt{2}\cdot\sqrt{3}) = (-\sqrt{2})(-\sqrt{3}) = \sqrt{6}$). For the element to be fixed, we need $b = 0$ and $c = 0$. So $L^H = \mathbb{Q}(\sqrt{6})$.

Observe the order-reversing nature: larger subgroups correspond to smaller fixed fields, and vice versa. Also note that $[L : L^H] = |H|$ and $[L^H : \mathbb{Q}] = |G|/|H| = [G:H]$ in each case. These are not coincidences — they are consequences of the Fundamental Theorem.

---

## The Fundamental Theorem of Galois Theory

**Theorem (Fundamental Theorem of Galois Theory).** Let $L/K$ be a finite Galois extension with Galois group $G = \operatorname{Gal}(L/K)$. Then:

**(a) Bijection.** There is an inclusion-reversing bijection
$$\Phi: \{\text{intermediate fields } K \subseteq M \subseteq L\} \to \{\text{subgroups } H \leq G\}$$
given by $\Phi(M) = \operatorname{Gal}(L/M)$, with inverse $\Psi(H) = L^H$.

**(b) Degree = Index.** For any intermediate field $M$:
$$[L:M] = |\operatorname{Gal}(L/M)| \quad \text{and} \quad [M:K] = [G:\operatorname{Gal}(L/M)].$$

**(c) Normality criterion.** An intermediate field $M$ gives a normal extension $M/K$ if and only if $\operatorname{Gal}(L/M)$ is a normal subgroup of $G$. When this holds, the restriction map $G \to \operatorname{Gal}(M/K)$, $\sigma \mapsto \sigma|_M$, is a surjective homomorphism with kernel $\operatorname{Gal}(L/M)$, giving

$$\operatorname{Gal}(M/K) \cong G / \operatorname{Gal}(L/M).$$

*Proof outline.*

*Part (a), well-definedness of $\Phi$.* If $K \subseteq M \subseteq L$, then $L/M$ is Galois (it is a splitting field of the same polynomial restricted to $M$, and separability is inherited). So $\operatorname{Gal}(L/M) \leq G$ is a well-defined subgroup.

*Part (a), well-definedness of $\Psi$.* If $H \leq G$, then $L^H$ is a subfield of $L$ containing $K$ (since $G$ fixes $K$, so does $H$).

*Part (a), $\Psi \circ \Phi = \operatorname{id}$.* We need $L^{\operatorname{Gal}(L/M)} = M$. The inclusion $M \subseteq L^{\operatorname{Gal}(L/M)}$ is clear (elements of $M$ are fixed by automorphisms fixing $M$). For the reverse, one shows that $[L : L^{\operatorname{Gal}(L/M)}] \geq |\operatorname{Gal}(L/M)| = [L:M]$, so $L^{\operatorname{Gal}(L/M)} \subseteq M$ by comparing dimensions.

*Part (a), $\Phi \circ \Psi = \operatorname{id}$.* We need $\operatorname{Gal}(L/L^H) = H$. The inclusion $H \subseteq \operatorname{Gal}(L/L^H)$ is clear. For equality, we use a lemma of Artin: if $H$ is a finite group of automorphisms of $L$, then $[L:L^H] \leq |H|$. Combined with $|\operatorname{Gal}(L/L^H)| = [L:L^H]$ (since $L/L^H$ is Galois), we get $|\operatorname{Gal}(L/L^H)| \leq |H|$, hence equality.

**Artin's Lemma.** *If $H$ is a finite group of automorphisms of a field $L$, then $[L:L^H] \leq |H|$.*

*Proof.* Let $|H| = n$ and suppose for contradiction that $[L:L^H] > n$. Then there exist $n+1$ elements $\alpha_1, \ldots, \alpha_{n+1} \in L$ that are linearly independent over $L^H$. Write $H = \{\sigma_1, \ldots, \sigma_n\}$ and consider the homogeneous system

$$\sum_{j=1}^{n+1} \sigma_i(\alpha_j) x_j = 0, \qquad i = 1, \ldots, n.$$

This is $n$ equations in $n+1$ unknowns over $L$, so it has a nontrivial solution $(c_1, \ldots, c_{n+1}) \in L^{n+1}$, not all $c_j = 0$. Among all such nontrivial solutions, pick one with the fewest nonzero entries. After reindexing, say $c_1, \ldots, c_r \neq 0$ and $c_{r+1} = \cdots = c_{n+1} = 0$, and normalize so $c_r = 1$. We claim all $c_j \in L^H$, which gives a dependence relation over $L^H$ — contradiction.

If some $c_j \notin L^H$, pick $\tau \in H$ with $\tau(c_j) \neq c_j$. Applying $\tau$ to the system and subtracting from the original yields a new nontrivial solution with fewer nonzero entries (since the last entry $c_r = 1$ is fixed by $\tau$, it cancels). This contradicts minimality. $\blacksquare$

*Part (b)* follows from $|\operatorname{Gal}(L/M)| = [L:M]$ (by the Galois extension property) and $[M:K] = [L:K]/[L:M] = |G|/|\operatorname{Gal}(L/M)| = [G:\operatorname{Gal}(L/M)]$.

*Part (c)* is proved as follows. If $M/K$ is normal, then for any $\sigma \in G$ and any $\mathbb{Q}$-automorphism $\tau \in \operatorname{Gal}(L/M)$, the composition $\sigma\tau\sigma^{-1}$ fixes $M$ (because $\sigma$ permutes the roots of any irreducible polynomial over $K$, and $M$ being normal means it contains all conjugates of its elements). The converse uses the fact that if $\operatorname{Gal}(L/M) \trianglelefteq G$, then $G$ acts on $M$ via $\sigma|_M$, and the kernel of this action is $\operatorname{Gal}(L/M)$. $\blacksquare$

---

## Computing Galois Groups: Concrete Examples

### Example 1: Splitting Field of $x^4 - 2$ over $\mathbb{Q}$

From the previous article, the splitting field is $L = \mathbb{Q}(\sqrt[4]{2}, i)$ with $[L:\mathbb{Q}] = 8$.

The roots of $x^4 - 2$ are $\alpha = \sqrt[4]{2}$, $i\alpha$, $-\alpha$, $-i\alpha$. Any $\sigma \in G = \operatorname{Gal}(L/\mathbb{Q})$ is determined by $\sigma(\alpha)$ and $\sigma(i)$:

- $\sigma(\alpha)$ must be a root of $x^4 - 2$: one of $\alpha, i\alpha, -\alpha, -i\alpha$ (4 choices).
- $\sigma(i)$ must be a root of $x^2 + 1$: $i$ or $-i$ (2 choices).

Since $|G| = 8 = 4 \times 2$, all $4 \times 2 = 8$ combinations are realized. Define:

- $\rho$: $\alpha \mapsto i\alpha$, $i \mapsto i$ (a rotation of the four roots).
- $\tau$: $\alpha \mapsto \alpha$, $i \mapsto -i$ (complex conjugation restricted to $L$).

Then $\rho$ has order 4 ($\rho^2(\alpha) = i^2\alpha = -\alpha$, $\rho^3(\alpha) = -i\alpha$, $\rho^4 = e$) and $\tau$ has order 2. Furthermore, $\tau\rho\tau^{-1}(\alpha) = \tau(i\alpha) = (-i)\alpha = -i\alpha = \rho^3(\alpha)$, so $\tau\rho\tau^{-1} = \rho^3 = \rho^{-1}$. This gives the dihedral group relation $\tau\rho = \rho^{-1}\tau$, and therefore:

$$G = \operatorname{Gal}(\mathbb{Q}(\sqrt[4]{2},i)/\mathbb{Q}) \cong D_4,$$

the dihedral group of order 8 (symmetries of a square).

The subgroup lattice of $D_4$ has 10 subgroups (including the trivial group and $D_4$ itself), corresponding to 10 intermediate fields between $\mathbb{Q}$ and $L$. For instance:

- $\langle \rho \rangle \cong \mathbb{Z}/4\mathbb{Z}$ corresponds to $L^{\langle\rho\rangle} = \mathbb{Q}(i)$ (the fixed field of the four rotations is exactly the rationals with $i$ adjoined).
- $\langle \tau \rangle \cong \mathbb{Z}/2\mathbb{Z}$ corresponds to $L^{\langle\tau\rangle} = \mathbb{Q}(\sqrt[4]{2})$ (fixed under conjugation = real elements of $L$, generated by $\sqrt[4]{2}$).
- $\langle \rho^2 \rangle \cong \mathbb{Z}/2\mathbb{Z}$ corresponds to $L^{\langle\rho^2\rangle} = \mathbb{Q}(i, \sqrt{2})$ (since $\rho^2(\alpha) = -\alpha$ and $\rho^2(i) = i$, the fixed elements are those of the form $a + b\sqrt{2} + ci + di\sqrt{2}$).

The normal subgroups of $D_4$ — namely $\{e\}$, $\langle\rho^2\rangle$, $\langle\rho\rangle$, $\{e, \rho^2, \tau, \rho^2\tau\}$, $\{e, \rho^2, \rho\tau, \rho^3\tau\}$, and $D_4$ — correspond to the intermediate fields that give normal (Galois) extensions of $\mathbb{Q}$.

### Example 2: Cyclotomic Fields

Fix a prime $p$ and let $\zeta = e^{2\pi i/p}$, a primitive $p$-th root of unity. The $p$-th *cyclotomic field* is $\mathbb{Q}(\zeta)$. The minimal polynomial of $\zeta$ over $\mathbb{Q}$ is the $p$-th cyclotomic polynomial:

$$\Phi_p(x) = 1 + x + x^2 + \cdots + x^{p-1} = \frac{x^p - 1}{x - 1},$$

which is irreducible over $\mathbb{Q}$ (by Eisenstein's criterion applied to $\Phi_p(x+1)$, after the substitution $x \mapsto x+1$). So $[\mathbb{Q}(\zeta):\mathbb{Q}] = p-1$.

The field $\mathbb{Q}(\zeta)$ is the splitting field of $x^p - 1$ (all roots are $1, \zeta, \zeta^2, \ldots, \zeta^{p-1}$, and they all lie in $\mathbb{Q}(\zeta)$). Any $\sigma \in \operatorname{Gal}(\mathbb{Q}(\zeta)/\mathbb{Q})$ sends $\zeta$ to another primitive $p$-th root of unity, i.e., $\sigma(\zeta) = \zeta^k$ for some $k \in \{1, 2, \ldots, p-1\}$. The map $\sigma \mapsto k \pmod{p}$ is an isomorphism:

$$\operatorname{Gal}(\mathbb{Q}(\zeta)/\mathbb{Q}) \cong (\mathbb{Z}/p\mathbb{Z})^\times \cong \mathbb{Z}/(p-1)\mathbb{Z}.$$

So the Galois group of a $p$-th cyclotomic field is cyclic of order $p-1$. For example, $\operatorname{Gal}(\mathbb{Q}(\zeta_7)/\mathbb{Q}) \cong \mathbb{Z}/6\mathbb{Z}$, and the subgroups of $\mathbb{Z}/6\mathbb{Z}$ (namely $\{0\}$, $\{0,3\}$, $\{0,2,4\}$, and $\mathbb{Z}/6\mathbb{Z}$) correspond to the intermediate fields $\mathbb{Q}(\zeta_7) \supset M_3 \supset M_2 \supset \mathbb{Q}$, where $[M_3:\mathbb{Q}] = 3$ and $[M_2:\mathbb{Q}] = 2$.

---

## Solvability by Radicals and Solvable Groups

The original impetus for Galois theory was the question: for which polynomials can we express the roots using arithmetic operations ($+, -, \times, \div$) and $n$-th roots? This is "solvability by radicals."

**Definition.** A field extension $L/K$ is a *radical extension* if there is a tower

$$K = K_0 \subset K_1 \subset \cdots \subset K_r = L$$

where each $K_{i+1} = K_i(\alpha_i)$ with $\alpha_i^{n_i} \in K_i$ for some positive integer $n_i$. In other words, each step adjoins an $n$-th root of an existing element.

**Definition.** A polynomial $f(x) \in K[x]$ is *solvable by radicals* if its splitting field is contained in some radical extension of $K$.

**Definition.** A group $G$ is *solvable* if it has a subnormal series

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_r = G$$

where each quotient $G_{i+1}/G_i$ is abelian.

**Theorem (Galois's criterion).** Let $f(x) \in K[x]$ (with $\operatorname{char}(K) = 0$) and let $L$ be its splitting field over $K$. Then $f$ is solvable by radicals if and only if $\operatorname{Gal}(L/K)$ is a solvable group.

The proof is substantial and uses:
1. Adjoining an $n$-th root of unity gives a cyclic (hence abelian) Galois group.
2. Adjoining an $n$-th root of an element (with $n$-th roots of unity already present) again gives a cyclic Galois group (Kummer theory).
3. A radical tower gives rise to a sequence of cyclic extensions, and the Galois group of the composite has a subnormal series with abelian (in fact cyclic) quotients.
4. Conversely, a solvable Galois group allows us to build such a tower.

**Example: Solvability of $x^3 - 2$.** The splitting field of $x^3 - 2$ over $\mathbb{Q}$ is $L = \mathbb{Q}(\sqrt[3]{2}, \omega)$ where $\omega = e^{2\pi i/3}$. We showed in the previous article that $[L:\mathbb{Q}] = 6$. The Galois group $G = \operatorname{Gal}(L/\mathbb{Q})$ permutes the three roots $\sqrt[3]{2}, \omega\sqrt[3]{2}, \omega^2\sqrt[3]{2}$ and has order 6. Since $|G| = 6$ and $G$ embeds into $S_3$ (which also has order 6), we get $G \cong S_3$.

The solvability series for $G \cong S_3$ is $\{e\} \trianglelefteq A_3 \trianglelefteq S_3$, with $A_3 \cong \mathbb{Z}/3\mathbb{Z}$ and $S_3/A_3 \cong \mathbb{Z}/2\mathbb{Z}$. Since $G$ is solvable, $x^3 - 2$ is solvable by radicals — and indeed its roots are $\sqrt[3]{2}$, $\omega\sqrt[3]{2}$, $\omega^2\sqrt[3]{2}$, all expressible using cube roots and the radical $\omega = (-1+\sqrt{-3})/2$.

By the Galois correspondence, the unique subgroup $A_3 = \{e, (123), (132)\}$ of index 2 corresponds to the intermediate field $\mathbb{Q}(\omega)$, which is the unique quadratic subfield. The quotient $S_3/A_3 \cong \mathbb{Z}/2\mathbb{Z}$ corresponds to the first step (adjoining $\sqrt{-3}$ to get $\omega$), and the factor $A_3 \cong \mathbb{Z}/3\mathbb{Z}$ corresponds to the second step (adjoining $\sqrt[3]{2}$). This is precisely the structure of Cardano's formula: first extract a square root (the discriminant), then extract cube roots.

### Why Low-Degree Polynomials Are Solvable

For $n \leq 4$, the symmetric group $S_n$ is solvable. The solvability series:

- $S_1 = \{e\}$: trivially solvable.
- $S_2 = \mathbb{Z}/2\mathbb{Z}$: abelian, hence solvable.
- $S_3$: $\{e\} \trianglelefteq A_3 \trianglelefteq S_3$, with $A_3 \cong \mathbb{Z}/3\mathbb{Z}$ and $S_3/A_3 \cong \mathbb{Z}/2\mathbb{Z}$, both abelian.
- $S_4$: $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$, where $V_4$ is the Klein four-group, $A_4/V_4 \cong \mathbb{Z}/3\mathbb{Z}$, $S_4/A_4 \cong \mathbb{Z}/2\mathbb{Z}$.

Since the Galois group of a degree-$n$ polynomial is a subgroup of $S_n$ (it permutes the $n$ roots), and subgroups of solvable groups are solvable, every polynomial of degree $\leq 4$ is solvable by radicals. This is consistent with the existence of the quadratic, cubic, and quartic formulas.

---

## The Insolvability of the General Quintic

**Theorem (Abel-Ruffini).** The general polynomial of degree 5 (and higher) is not solvable by radicals.

By Galois's criterion, it suffices to exhibit a degree-5 polynomial whose Galois group is $S_5$, and then show that $S_5$ is not solvable.

### Step 1: $S_5$ Is Not Solvable

**Proposition.** The alternating group $A_5$ is simple (has no proper normal subgroups).

*Proof.* $|A_5| = 60$. The conjugacy classes of $A_5$ have sizes $1, 12, 12, 15, 20$ (corresponding to the identity, two classes of 5-cycles, products of two disjoint transpositions, and 3-cycles). A normal subgroup is a union of conjugacy classes and must contain the identity (contributing 1). Its order must divide 60. No subset of $\{1, 12, 12, 15, 20\}$ containing 1 sums to a proper divisor of 60 other than 1:

- $1 + 12 = 13$: does not divide 60.
- $1 + 15 = 16$: does not divide 60.
- $1 + 20 = 21$: does not divide 60.
- $1 + 12 + 12 = 25$: does not divide 60.
- $1 + 12 + 15 = 28$: does not divide 60.
- $1 + 12 + 20 = 33$: does not divide 60.
- $1 + 15 + 20 = 36$: does not divide 60.
- $1 + 12 + 12 + 15 = 40$: does not divide 60.
- $1 + 12 + 12 + 20 = 45$: does not divide 60.
- $1 + 12 + 15 + 20 = 48$: does not divide 60.

None works, so the only normal subgroups of $A_5$ are $\{e\}$ and $A_5$. $\blacksquare$

**Corollary.** $S_5$ is not solvable.

*Proof.* If $S_5$ were solvable, then $A_5$ (a subgroup of $S_5$) would also be solvable. But a solvable group has a nontrivial abelian quotient (the first step in its solvability series produces $G_r / G_{r-1}$ abelian and nontrivial). For $A_5$ to have a nontrivial abelian quotient $A_5/N$, we need a proper normal subgroup $N \trianglelefteq A_5$. Since $A_5$ is simple, the only option is $N = \{e\}$, giving $A_5/\{e\} = A_5$ itself — but $A_5$ is non-abelian ($|A_5| = 60 > 1$ and it contains non-commuting elements). Contradiction. $\blacksquare$

### Step 2: A Specific Polynomial with Galois Group $S_5$

Consider $f(x) = x^5 - 4x + 2 \in \mathbb{Q}[x]$.

**Irreducibility.** By Eisenstein's criterion at $p = 2$: $2 \mid (-4)$ and $2 \mid 2$, but $4 \nmid 2$, and $2 \nmid 1$ (leading coefficient). So $f$ is irreducible over $\mathbb{Q}$.

**Galois group is $S_5$.** We use a standard criterion: if $f$ is an irreducible quintic over $\mathbb{Q}$ with exactly three real roots and two complex conjugate roots, then $\operatorname{Gal}(f/\mathbb{Q}) \cong S_5$.

To count real roots, examine $f'(x) = 5x^4 - 4$. Setting $f'(x) = 0$: $x^4 = 4/5$, so $x = \pm(4/5)^{1/4}$. Call these $\pm c$ where $c = (4/5)^{1/4} \approx 0.9457$.

- $f(c) = c^5 - 4c + 2$. We have $c^5 = c \cdot c^4 = c \cdot (4/5) = 4c/5$, so $f(c) = 4c/5 - 4c + 2 = -16c/5 + 2 \approx -16(0.9457)/5 + 2 \approx -3.026 + 2 = -1.026 < 0$.
- $f(-c) = -c^5 + 4c + 2 = -4c/5 + 4c + 2 = 16c/5 + 2 \approx 3.026 + 2 = 5.026 > 0$.

Since $f$ has degree 5 (odd), $f(x) \to +\infty$ as $x \to +\infty$ and $f(x) \to -\infty$ as $x \to -\infty$. With the local maximum at $-c$ (value $> 0$) and local minimum at $c$ (value $< 0$), the real graph crosses the $x$-axis exactly three times. So $f$ has exactly 3 real roots and 2 complex conjugate roots.

**Why this implies $\operatorname{Gal}(f/\mathbb{Q}) = S_5$.** The Galois group $G$, viewed as a subgroup of $S_5$ (acting on the five roots), satisfies:
1. $|G|$ is divisible by 5 (since $f$ is irreducible of degree 5, the Galois group acts transitively on the roots and must contain a 5-cycle).
2. $G$ contains a transposition (complex conjugation swaps the two non-real roots and fixes the three real roots — this is a transposition in $S_5$).

A subgroup of $S_5$ that contains a 5-cycle and a transposition must be all of $S_5$ (a standard group theory exercise: any transposition and any $p$-cycle generate $S_p$ for $p$ prime).

Therefore $\operatorname{Gal}(f/\mathbb{Q}) \cong S_5$, which is not solvable. By Galois's criterion, $f(x) = x^5 - 4x + 2$ is not solvable by radicals. $\blacksquare$

### Historical Perspective

This result — the insolvability of the general quintic — was first proved by Abel in 1824 (without the full Galois framework) and then given its definitive group-theoretic form by Galois. It closed a problem that had been open since the Renaissance: while Tartaglia, Cardano, and Ferrari found formulas for degrees 2, 3, and 4 in the 16th century, all attempts at degree 5 failed. The reason is now clear: it is not a failure of ingenuity but a structural impossibility — $S_5$ is not solvable, and no clever algebraic manipulation can circumvent this.

It is worth emphasizing what the theorem does *not* say. It does not say that degree-5 polynomials have no roots — they always do, in $\mathbb{C}$, by the Fundamental Theorem of Algebra. It does not say that specific quintics are unsolvable: $x^5 - 2$, for instance, has Galois group isomorphic to a solvable group of order 20 (the Frobenius group $F_{20} \cong \mathbb{Z}/5\mathbb{Z} \rtimes \mathbb{Z}/4\mathbb{Z}$), and its roots can be expressed using radicals. What the theorem says is that there is no *uniform formula* using radicals that works for *all* quintics — because some quintics (like $x^5 - 4x + 2$) have $S_5$ as their Galois group, and $S_5$ is not solvable.

The passage from "we cannot find a formula" to "we can *prove* no formula exists" represents one of the great conceptual leaps in mathematics. It is also a triumph of abstraction: the answer to a concrete question about polynomial roots required developing an entirely new subject — group theory — and its interaction with field theory. This synthesis, Galois theory, remains one of the deepest and most elegant achievements in all of algebra.

### A Summary of the Solvability Landscape

| Degree | General Galois group | Solvable? | Formula exists? |
|:---:|:---:|:---:|:---:|
| 1 | $S_1 = \{e\}$ | Yes | Yes (trivial) |
| 2 | $S_2 \cong \mathbb{Z}/2\mathbb{Z}$ | Yes | Yes (quadratic formula) |
| 3 | $S_3$ | Yes | Yes (Cardano's formula) |
| 4 | $S_4$ | Yes | Yes (Ferrari's method) |
| $\geq 5$ | $S_n$ | No | No (Abel-Ruffini) |

---

## What's Next

Galois theory does not end here. In the remaining articles of this series, we will explore further applications and generalizations:

- **Finite fields:** Every finite field has order $p^n$ for some prime $p$, and the Galois group $\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p)$ is cyclic of order $n$, generated by the Frobenius automorphism $x \mapsto x^p$. This connects Galois theory to number theory and coding theory.

- **The inverse Galois problem:** Given a finite group $G$, does there exist a Galois extension $L/\mathbb{Q}$ with $\operatorname{Gal}(L/\mathbb{Q}) \cong G$? This remains one of the major open problems in algebra. It is known for all solvable groups, all symmetric and alternating groups, and many simple groups, but the general case is unresolved.

- **Infinite Galois theory:** For infinite algebraic extensions (like $\overline{\mathbb{Q}}/\mathbb{Q}$), the Galois group carries a natural topology (the Krull topology), and the Fundamental Theorem generalizes to a correspondence between *closed* subgroups and intermediate fields. The absolute Galois group $\operatorname{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$ is one of the most studied and mysterious objects in all of mathematics.

From a polynomial equation that refuses to yield its roots, through the abstract machinery of field extensions and group theory, to a definitive impossibility result and an ongoing research frontier — this is the arc of Galois theory, one of the most beautiful chapters in mathematics.

---

*This is Part 8 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 7 — Field Extensions](/en/abstract-algebra/07-field-extensions/)*

*Next: [Part 9 — Modules](/en/abstract-algebra/09-modules/)*
