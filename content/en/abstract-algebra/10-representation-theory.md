---
title: "Abstract Algebra (10): Representation Theory — Groups Acting on Vector Spaces"
date: 2021-09-19 09:00:00
tags:
  - abstract-algebra
  - representation-theory
  - group-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "Representing abstract groups as matrices makes them concrete and computable — Maschke's theorem, Schur's lemma, and character theory give us powerful classification tools."
disableNunjucks: true
series_order: 10
series_total: 12
translationKey: "abstract-algebra-10"
---

An abstract group is a set with a binary operation satisfying certain axioms. This is elegant but sometimes hard to work with — how do you compute with elements of a group defined by generators and relations, or extract numerical invariants from a multiplication table? The solution, going back to Frobenius and Burnside over a century ago, is to represent group elements as matrices. Matrices are concrete: you can multiply them, take traces, compute determinants, decompose them into eigenspaces. Representation theory is the systematic study of this idea, and it has become one of the most powerful tools in modern algebra, number theory, and mathematical physics.

The connection to our previous article on modules is direct: as we will see, a representation of $G$ over a field $F$ is the same thing as a module over the group ring $F[G]$. The module-theoretic machinery we developed — submodules, quotients, direct sums — translates directly into the language of representations.

---

## Groups Meet Linear Algebra

The fundamental idea is simple: instead of studying a group $G$ in isolation, we study its **actions on vector spaces**. If $G$ acts on a vector space $V$ by linear transformations, we get a homomorphism $\rho: G \to GL(V)$ — each group element becomes an invertible matrix (after choosing a basis). The group's algebraic structure is reflected in the linear algebra of these matrices.

Why is this useful?

1. **Concrete computation.** Matrix multiplication is algorithmic. Given an explicit representation, you can compute products, powers, and conjugates efficiently.

2. **Decomposition.** Vector spaces can be split into direct sums of subspaces. Finding invariant subspaces of a representation corresponds to breaking the group's action into simpler pieces.

3. **Numerical invariants.** The trace of a matrix is a single number that is invariant under conjugation. The function $\chi(g) = \operatorname{tr}(\rho(g))$ — the **character** of the representation — carries a remarkable amount of information about $\rho$.

4. **Classification.** For finite groups over $\mathbb{C}$, representations are completely classified by their characters, and the character table is a finite square matrix that encodes the group's structure.

**Historical note.** Representation theory began with Frobenius's work in the 1890s, motivated by problems in number theory and the study of group determinants. Burnside and Schur developed the theory further in the early 1900s. The subject exploded in the mid-20th century with the work of Brauer (modular representations), Harish-Chandra (infinite-dimensional representations of Lie groups), and Langlands (the Langlands program, which connects representations to number theory). Today, representation theory is a central pillar of mathematics, with connections to algebraic geometry, combinatorics, mathematical physics, and even theoretical computer science.

---


![Character table of S_3](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_fig10_character_table.png)

## Definitions: Representations, Equivalence, Reducibility

**Definition.** A **(linear) representation** of a group $G$ on a vector space $V$ over a field $F$ is a group homomorphism:
$$\rho: G \to GL(V)$$
where $GL(V)$ is the group of invertible linear transformations of $V$. The dimension $\dim_F V$ is the **degree** of the representation.

If $V = F^n$, then $\rho(g)$ is an $n \times n$ invertible matrix for each $g \in G$, and $\rho(g_1 g_2) = \rho(g_1)\rho(g_2)$.

**Definition.** Two representations $\rho: G \to GL(V)$ and $\sigma: G \to GL(W)$ are **equivalent** (or isomorphic) if there exists an invertible linear map $T: V \to W$ such that $T \rho(g) = \sigma(g) T$ for all $g \in G$. In matrix terms: $\sigma(g) = T \rho(g) T^{-1}$ — the representations differ only by a change of basis.

**Definition.** A subspace $W \subseteq V$ is **$G$-invariant** (or $\rho$-invariant) if $\rho(g)(W) \subseteq W$ for all $g \in G$. A representation is **irreducible** (or simple) if $V \neq 0$ and the only $G$-invariant subspaces are $0$ and $V$ itself. A representation is **reducible** if it has a proper nonzero invariant subspace, and **completely reducible** (or semisimple) if it is a direct sum of irreducible representations.

**Example 1 (Trivial representation).** For any group $G$, the map $\rho(g) = I$ (the identity matrix) for all $g$ defines a 1-dimensional representation. This is always irreducible.

**Example 2 (Regular representation).** Let $V = \mathbb{C}[G]$ be the vector space with basis $\{e_g : g \in G\}$. Define $\rho(g)(e_h) = e_{gh}$. This is the **left regular representation**, of degree $|G|$. It is generally not irreducible — in fact, it contains every irreducible representation as a summand.

**Example 3 (Permutation representation of $S_3$).** The symmetric group $S_3$ acts on $\mathbb{C}^3$ by permuting coordinates: $\rho(\sigma)(e_i) = e_{\sigma(i)}$. This 3-dimensional representation is reducible. The subspace $W_1 = \operatorname{span}\{e_1 + e_2 + e_3\}$ is invariant (every permutation fixes the sum), giving a 1-dimensional trivial sub-representation. Its complement $W_2 = \{(a_1, a_2, a_3) : a_1 + a_2 + a_3 = 0\}$ is a 2-dimensional invariant subspace that turns out to be irreducible. So $\mathbb{C}^3 \cong W_1 \oplus W_2$ as representations.

**The module perspective.** As we noted in the previous article, a representation of $G$ over $F$ is the same thing as a module over the group ring $F[G]$. Specifically, if $\rho: G \to GL(V)$ is a representation, then $V$ becomes an $F[G]$-module via $(\sum a_g g) \cdot v = \sum a_g \rho(g)(v)$. The concepts translate: sub-representations correspond to submodules, irreducibility corresponds to simplicity, and complete reducibility corresponds to semisimplicity. This module-theoretic perspective often simplifies proofs and reveals structural patterns.

**Direct sums and tensor products.** Given representations $\rho: G \to GL(V)$ and $\sigma: G \to GL(W)$, we can form:
- The **direct sum** $\rho \oplus \sigma: G \to GL(V \oplus W)$ by $(\rho \oplus \sigma)(g)(v, w) = (\rho(g)v, \sigma(g)w)$.
- The **tensor product** $\rho \otimes \sigma: G \to GL(V \otimes W)$ by $(\rho \otimes \sigma)(g)(v \otimes w) = \rho(g)v \otimes \sigma(g)w$.
- The **dual (contragredient)** $\rho^*: G \to GL(V^*)$ by $\rho^*(g)(\varphi) = \varphi \circ \rho(g)^{-1}$ for $\varphi \in V^*$.

These operations let us build new representations from old ones. The tensor product is particularly important: decomposing tensor products of irreducible representations into irreducible summands (the "Clebsch-Gordan problem") is a central problem in representation theory and mathematical physics.

---

## Complete Reducibility and Maschke's Theorem

The most fundamental result in the representation theory of finite groups says that every representation breaks into irreducible pieces — there are no "indecomposable but reducible" representations lurking in the background.

**Theorem (Maschke).** Let $G$ be a finite group and $F$ a field whose characteristic does not divide $|G|$. Then every finite-dimensional representation of $G$ over $F$ is completely reducible.

Equivalently: if $W \subseteq V$ is a $G$-invariant subspace, then there exists a $G$-invariant complement $W'$ such that $V = W \oplus W'$.

*Proof.* Let $\rho: G \to GL(V)$ be a representation and $W \subseteq V$ a $G$-invariant subspace. We need to find a $G$-invariant complement.

**Step 1:** Choose any complement $U$ of $W$ in $V$ (as a vector space, ignoring the group action). Let $\pi: V \to W$ be the projection onto $W$ along $U$ — that is, $\pi$ is linear, $\pi|_W = \operatorname{id}_W$, and $\ker \pi = U$.

**Step 2:** Average $\pi$ over the group action. Define:
$$\tilde{\pi}(v) = \frac{1}{|G|} \sum_{g \in G} \rho(g) \pi(\rho(g)^{-1} v)$$

Note that dividing by $|G|$ is valid because $\operatorname{char}(F) \nmid |G|$.

**Step 3:** Verify that $\tilde{\pi}$ is a $G$-equivariant projection onto $W$.
- **$\tilde{\pi}$ maps into $W$:** For each $g$, $\pi(\rho(g)^{-1}v) \in W$ (since $\pi$ maps to $W$), and $\rho(g)$ preserves $W$ (since $W$ is invariant). So each summand lies in $W$.
- **$\tilde{\pi}|_W = \operatorname{id}_W$:** If $w \in W$, then $\rho(g)^{-1}w \in W$, so $\pi(\rho(g)^{-1}w) = \rho(g)^{-1}w$, and $\rho(g)\rho(g)^{-1}w = w$. Averaging gives $\tilde{\pi}(w) = w$.
- **$G$-equivariance:** For $h \in G$:
$$\tilde{\pi}(\rho(h)v) = \frac{1}{|G|} \sum_{g \in G} \rho(g)\pi(\rho(g)^{-1}\rho(h)v) = \frac{1}{|G|} \sum_{g \in G} \rho(g)\pi(\rho((h^{-1}g)^{-1})v)$$
Substituting $g' = h^{-1}g$ (which runs over all of $G$ as $g$ does):
$$= \frac{1}{|G|} \sum_{g' \in G} \rho(hg')\pi(\rho(g'^{-1})v) = \rho(h)\tilde{\pi}(v)$$

**Step 4:** Set $W' = \ker \tilde{\pi}$. Then $W'$ is $G$-invariant (since $\tilde{\pi}$ is equivariant), and $V = W \oplus W'$ (since $\tilde{\pi}$ is a projection onto $W$). $\square$

**Remark.** The hypothesis $\operatorname{char}(F) \nmid |G|$ is essential. In **modular representation theory** (where the characteristic divides $|G|$), Maschke's theorem fails, and the theory becomes much more complicated.

**Example (Maschke fails in characteristic $p$).** Consider $G = \mathbb{Z}/2\mathbb{Z} = \{e, g\}$ over $F = \mathbb{F}_2$. The regular representation on $\mathbb{F}_2^2$ maps $g$ to the matrix $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. The subspace $W = \operatorname{span}\{(1,1)\}$ is $G$-invariant (since $g(1,1) = (1,1)$), but its only complement in $\mathbb{F}_2^2$ is $\operatorname{span}\{(1,0)\}$ or $\operatorname{span}\{(0,1)\}$, neither of which is $G$-invariant (since $g(1,0) = (0,1) \notin \operatorname{span}\{(1,0)\}$). So this representation is reducible but not completely reducible. Maschke's theorem does not apply because $\operatorname{char}(\mathbb{F}_2) = 2$ divides $|G| = 2$.

**The group algebra decomposition.** Maschke's theorem has a striking reformulation in terms of the group algebra. When $\operatorname{char}(F) \nmid |G|$ and $F$ is algebraically closed, the group algebra $F[G]$ decomposes as a direct sum of matrix algebras:
$$F[G] \cong M_{d_1}(F) \times M_{d_2}(F) \times \cdots \times M_{d_k}(F)$$
where $d_1, \ldots, d_k$ are the dimensions of the irreducible representations. This is the Artin-Wedderburn theorem applied to the semisimple algebra $F[G]$. It immediately implies $|G| = d_1^2 + d_2^2 + \cdots + d_k^2$ — a dimension count that gives strong constraints on the possible degrees of irreducible representations.

---

## Schur's Lemma and Its Consequences

**Lemma (Schur).** Let $\rho: G \to GL(V)$ and $\sigma: G \to GL(W)$ be irreducible representations over a field $F$, and let $\varphi: V \to W$ be a $G$-equivariant linear map (i.e., $\varphi \rho(g) = \sigma(g) \varphi$ for all $g$). Then:

1. $\varphi$ is either zero or an isomorphism.
2. If $F$ is algebraically closed and $V = W$, $\rho = \sigma$, then $\varphi = \lambda I$ for some $\lambda \in F$.

*Proof of (1).* The kernel $\ker \varphi$ is a $G$-invariant subspace of $V$ (if $v \in \ker\varphi$, then $\varphi(\rho(g)v) = \sigma(g)\varphi(v) = 0$). Since $V$ is irreducible, $\ker\varphi = 0$ or $\ker\varphi = V$. Similarly, $\operatorname{im}\varphi$ is a $G$-invariant subspace of $W$, so $\operatorname{im}\varphi = 0$ or $\operatorname{im}\varphi = W$. If $\ker\varphi = 0$ and $\operatorname{im}\varphi = W$, then $\varphi$ is an isomorphism. Otherwise $\varphi = 0$.

*Proof of (2).* Over an algebraically closed field, $\varphi$ has an eigenvalue $\lambda$. Then $\varphi - \lambda I$ is also $G$-equivariant and has nontrivial kernel. By part (1), $\varphi - \lambda I = 0$. $\square$

**Consequences.** Schur's lemma has far-reaching implications:

- **Endomorphisms of irreducible representations over $\mathbb{C}$ are scalars.** This means $\operatorname{End}_G(V) \cong \mathbb{C}$ for any irreducible complex representation.

- **Non-isomorphic irreducibles are "orthogonal."** Any intertwining map between non-isomorphic irreducible representations is zero.

- **Abelian groups have only 1-dimensional irreducible representations over $\mathbb{C}$.** If $G$ is abelian, each $\rho(g)$ commutes with every $\rho(h)$, so each $\rho(g)$ is an intertwiner, hence a scalar by Schur's lemma.

**Example (Applying Schur's lemma).** Consider the irreducible 2-dimensional representation of $S_3$ (the standard representation on $W_2$ from Example 3). Any $S_3$-equivariant endomorphism $\varphi: W_2 \to W_2$ must be a scalar multiple of the identity by Schur's lemma (since we work over $\mathbb{C}$, which is algebraically closed). This means, for instance, that there is no $S_3$-equivariant way to project $W_2$ onto a 1-dimensional subspace — if there were, the kernel would be an invariant subspace contradicting irreducibility.

**The center of the group algebra.** Schur's lemma also implies that the center of $\mathbb{C}[G]$ (the class functions, or elements that commute with everything) has dimension equal to the number of irreducible representations, which equals the number of conjugacy classes. The central idempotents — one for each irreducible representation — provide a canonical decomposition of $\mathbb{C}[G]$ into simple components.

---

## Characters: Traces as Invariants

The character of a representation is the function that extracts the most important single number from each matrix.

**Definition.** The **character** of a representation $\rho: G \to GL(V)$ is the function $\chi_\rho: G \to F$ defined by:
$$\chi_\rho(g) = \operatorname{tr}(\rho(g))$$

Key properties (immediate from properties of trace):
- $\chi_\rho(e) = \dim V$ (the trace of the identity matrix)
- $\chi_\rho(hgh^{-1}) = \chi_\rho(g)$ (trace is conjugation-invariant), so $\chi$ is a **class function** — constant on conjugacy classes
- $\chi_{\rho \oplus \sigma} = \chi_\rho + \chi_\sigma$
- $\chi_{\rho \otimes \sigma} = \chi_\rho \cdot \chi_\sigma$

**The Inner Product on Class Functions.** For a finite group $G$ and representations over $\mathbb{C}$, define:
$$\langle \chi, \psi \rangle = \frac{1}{|G|} \sum_{g \in G} \chi(g) \overline{\psi(g)}$$

**Theorem (Orthogonality Relations).** Let $\chi_1, \ldots, \chi_k$ be the characters of the distinct irreducible representations of $G$ over $\mathbb{C}$. Then:

$$\langle \chi_i, \chi_j \rangle = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

That is, the irreducible characters form an **orthonormal set** in the inner product space of class functions. Since the number of irreducible characters equals the number of conjugacy classes (which is the dimension of the space of class functions), they actually form an **orthonormal basis**.

*Proof sketch.* By Schur's lemma, if $\rho_i$ and $\rho_j$ are non-isomorphic irreducible representations, then any $G$-equivariant map between them is zero. The averaging trick $\tilde{A} = \frac{1}{|G|}\sum_g \rho_j(g)^{-1} A \rho_i(g)$ applied to an arbitrary linear map $A: V_i \to V_j$ produces a $G$-equivariant map, which must be zero by Schur. Taking $A$ to be the matrix units $E_{pq}$ and summing the trace conditions yields the orthogonality. The $i = j$ case uses Schur's lemma part (2): the average produces a scalar, and computing the trace of both sides gives $1/\dim V_i$ times the trace condition. The full computation involves careful bookkeeping with matrix entries. $\square$

**Corollary.** A representation $\rho$ is determined (up to equivalence) by its character. If $\rho$ decomposes as $\rho \cong m_1 \rho_1 \oplus \cdots \oplus m_k \rho_k$ where $\rho_i$ are irreducible and $m_i$ are multiplicities, then $m_i = \langle \chi_\rho, \chi_i \rangle$.

**Theorem.** The number of irreducible representations of a finite group $G$ (over $\mathbb{C}$) equals the number of conjugacy classes of $G$.

*Proof sketch.* The irreducible characters form an orthonormal set in the space of class functions, which has dimension equal to the number of conjugacy classes. It suffices to show they span this space. If a class function $f$ is orthogonal to every $\chi_i$, then the operator $\sum_g f(g)\rho_i(g) = 0$ for every irreducible $\rho_i$. Since every representation is a sum of irreducibles (Maschke), this operator is zero on every representation, including the regular representation. But the regular representation is faithful, so $f = 0$. $\square$

**Column orthogonality.** In addition to the row orthogonality relations above, there are **column orthogonality relations**:
$$\sum_{i=1}^k \chi_i(g) \overline{\chi_i(h)} = \begin{cases} |C_G(g)| & \text{if } g \text{ and } h \text{ are conjugate} \\ 0 & \text{otherwise} \end{cases}$$
where $|C_G(g)|$ is the order of the centralizer of $g$, which equals $|G|$ divided by the size of the conjugacy class of $g$. These relations provide additional computational checks on character tables and are used in the classification of finite simple groups.

---

## Character Tables of Small Groups

Let us compute character tables for three groups to see the theory in action.

### The cyclic group $\mathbb{Z}/4\mathbb{Z}$

This is abelian, so all irreducible representations are 1-dimensional (by Schur's lemma). The group has 4 elements, hence 4 conjugacy classes (each element is its own class), hence 4 irreducible representations.

A 1-dimensional representation is just a homomorphism $\chi: \mathbb{Z}/4\mathbb{Z} \to \mathbb{C}^*$. Since $\chi(\bar{1})^4 = \chi(\bar{0}) = 1$, we need $\chi(\bar{1})$ to be a 4th root of unity: $1, i, -1, -i$.

| | $\bar{0}$ | $\bar{1}$ | $\bar{2}$ | $\bar{3}$ |
|---|---|---|---|---|
| $\chi_1$ | 1 | 1 | 1 | 1 |
| $\chi_2$ | 1 | $i$ | $-1$ | $-i$ |
| $\chi_3$ | 1 | $-1$ | 1 | $-1$ |
| $\chi_4$ | 1 | $-i$ | $-1$ | $i$ |

One can verify orthogonality: $\langle \chi_j, \chi_k \rangle = \frac{1}{4}\sum_{g} \chi_j(g)\overline{\chi_k(g)} = \delta_{jk}$.

For example, $\langle \chi_2, \chi_4 \rangle = \frac{1}{4}(1 \cdot 1 + i \cdot i + (-1)(-1) + (-i)(-i)) = \frac{1}{4}(1 + i^2 + 1 + (-i)^2) = \frac{1}{4}(1 - 1 + 1 - 1) = 0$. And $\langle \chi_2, \chi_2 \rangle = \frac{1}{4}(1 \cdot 1 + i \cdot (-i) + (-1)(-1) + (-i) \cdot i) = \frac{1}{4}(1 + 1 + 1 + 1) = 1$. Confirmed orthonormal.

### The symmetric group $S_3$

$S_3$ has 6 elements and 3 conjugacy classes: $\{e\}$, $\{(12), (13), (23)\}$, $\{(123), (132)\}$. So there are 3 irreducible representations. Their degrees $d_1, d_2, d_3$ satisfy $d_1^2 + d_2^2 + d_3^2 = 6$, which forces $d_1 = d_2 = 1$, $d_3 = 2$.

The two 1-dimensional representations are:
- **Trivial:** $\chi_1(g) = 1$ for all $g$.
- **Sign:** $\chi_2(g) = \operatorname{sgn}(g)$ (i.e., $+1$ for even permutations, $-1$ for odd).

The 2-dimensional representation is the "standard representation" — the complement of the trivial sub-representation inside the permutation representation on $\mathbb{C}^3$, as in Example 3 above. Its character values are computed by taking traces: $\chi_3(e) = 2$, $\chi_3((12)) = 0$ (the permutation matrix for a transposition has trace 0 on the 2-d subspace), $\chi_3((123)) = -1$.

| | $e$ | $(12)$ | $(123)$ |
|---|---|---|---|
| Size | 1 | 3 | 2 |
| $\chi_1$ | 1 | 1 | 1 |
| $\chi_2$ | 1 | $-1$ | 1 |
| $\chi_3$ | 2 | 0 | $-1$ |

**Verification:** $\langle \chi_3, \chi_3 \rangle = \frac{1}{6}(1 \cdot 4 + 3 \cdot 0 + 2 \cdot 1) = \frac{6}{6} = 1$. Confirmed irreducible.

Let us also verify $\langle \chi_3, \chi_1 \rangle = \frac{1}{6}(1 \cdot 2 \cdot 1 + 3 \cdot 0 \cdot 1 + 2 \cdot (-1) \cdot 1) = \frac{1}{6}(2 + 0 - 2) = 0$. And $\langle \chi_3, \chi_2 \rangle = \frac{1}{6}(2 \cdot 1 + 0 \cdot (-1) \cdot 3 + (-1) \cdot 1 \cdot 2) = \frac{1}{6}(2 + 0 - 2) = 0$. The irreducible characters are indeed pairwise orthogonal.

### The dihedral group $D_4$

$D_4$ (symmetries of the square) has 8 elements and 5 conjugacy classes:
- $\{e\}$, $\{r^2\}$, $\{r, r^3\}$, $\{s, r^2 s\}$, $\{rs, r^3 s\}$

where $r$ is rotation by $90°$ and $s$ is a reflection. There are 5 irreducible representations. The degrees satisfy $\sum d_i^2 = 8$, which forces four 1-dimensional and one 2-dimensional representation: $1^2 + 1^2 + 1^2 + 1^2 + 2^2 = 8$.

The four 1-dimensional representations correspond to the four homomorphisms $D_4 \to \mathbb{C}^*$. Since $D_4/[D_4, D_4] \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$ (the abelianization), there are exactly four.

The 2-dimensional representation comes from the natural action of $D_4$ on $\mathbb{R}^2$ (rotation and reflection matrices):
$$\rho(r) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}, \quad \rho(s) = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

The character table:

| | $e$ | $r^2$ | $r$ | $s$ | $rs$ |
|---|---|---|---|---|---|
| Size | 1 | 1 | 2 | 2 | 2 |
| $\chi_1$ | 1 | 1 | 1 | 1 | 1 |
| $\chi_2$ | 1 | 1 | 1 | $-1$ | $-1$ |
| $\chi_3$ | 1 | 1 | $-1$ | 1 | $-1$ |
| $\chi_4$ | 1 | 1 | $-1$ | $-1$ | 1 |
| $\chi_5$ | 2 | $-2$ | 0 | 0 | 0 |

**Verification:** Column orthogonality (a second set of relations): $\sum_i \chi_i(g)\overline{\chi_i(h)} = \frac{|G|}{|C(g)|}\delta_{C(g), C(h)}$ where $C(g)$ is the conjugacy class of $g$. For the $e$ column: $1 + 1 + 1 + 1 + 4 = 8 = |D_4|/|C(e)| = 8/1$. Verified.

**Using the character table.** Character tables are powerful computational tools. Suppose we are given a representation $\rho$ of $D_4$ with character $\chi(e) = 5$, $\chi(r^2) = 1$, $\chi(r) = 1$, $\chi(s) = -1$, $\chi(rs) = -1$. We can decompose $\rho$ into irreducibles by computing inner products:
$$m_1 = \langle \chi, \chi_1 \rangle = \frac{1}{8}(5 + 1 + 2 - 2 - 2) = \frac{4}{8} = \frac{1}{2}$$

Wait — this is not an integer, which means I made an error. Let me recalculate. $\langle \chi, \chi_1 \rangle = \frac{1}{8}(1 \cdot 5 \cdot 1 + 1 \cdot 1 \cdot 1 + 2 \cdot 1 \cdot 1 + 2 \cdot (-1) \cdot 1 + 2 \cdot (-1) \cdot 1) = \frac{1}{8}(5 + 1 + 2 - 2 - 2) = \frac{4}{8}$. This is not an integer, which is impossible for a character. The issue is that the given $\chi$ values do not actually form a valid character. This illustrates an important point: **the inner product serves as a consistency check** — if $\langle \chi, \chi_i \rangle$ is not a non-negative integer for every $i$, then $\chi$ is not the character of any representation.

Let us instead take a valid example: $\chi(e) = 4$, $\chi(r^2) = 0$, $\chi(r) = 0$, $\chi(s) = 0$, $\chi(rs) = 0$. Then:
$$m_i = \frac{1}{8}\left(1 \cdot 4 \cdot \overline{\chi_i(e)} + 1 \cdot 0 + 2 \cdot 0 + 2 \cdot 0 + 2 \cdot 0\right) = \frac{4 \cdot \chi_i(e)}{8} = \frac{\chi_i(e)}{2}$$

So $m_1 = m_2 = m_3 = m_4 = 1/2$ — again not integers! This means the all-zeros-except-identity character is not a valid character either. Instead consider $\rho = \chi_1 \oplus \chi_2 \oplus \chi_5$, which gives $\chi(e) = 4$, $\chi(r^2) = 0$, $\chi(r) = 0$, $\chi(s) = 0$, $\chi(rs) = 0$. Let me verify: $\chi_1(e) + \chi_2(e) + \chi_5(e) = 1 + 1 + 2 = 4$. $\chi_1(r^2) + \chi_2(r^2) + \chi_5(r^2) = 1 + 1 + (-2) = 0$. $\chi_1(r) + \chi_2(r) + \chi_5(r) = 1 + 1 + 0 = 2$, not 0. So the character with all zeros except $\chi(e) = 4$ does not occur here either.

The correct way to use the character table is to start with a known representation and decompose it. For instance, the regular representation of $D_4$ has character $\chi_{\text{reg}}(e) = 8$ and $\chi_{\text{reg}}(g) = 0$ for all $g \neq e$. Then $m_i = \frac{1}{8} \cdot 8 \cdot d_i = d_i$, confirming that the regular representation contains each irreducible $\rho_i$ with multiplicity equal to its dimension: $\mathbb{C}[D_4] \cong \rho_1 \oplus \rho_2 \oplus \rho_3 \oplus \rho_4 \oplus 2\rho_5$ (with $d_5 = 2$).

**Burnside's theorem.** As a striking application of character theory, we mention (without proof) Burnside's $p^a q^b$ theorem: every group of order $p^a q^b$ (for primes $p, q$) is solvable. The proof — which was originally purely group-theoretic in ambition but resisted all group-theoretic approaches for decades — uses character theory in an essential way. This illustrates how representation theory can prove results about abstract groups that seem to have nothing to do with linear algebra.

---

## What's Next

Representation theory turns abstract group theory into concrete linear algebra, providing a powerful toolkit for understanding group structure through characters and matrix decompositions. The character table is a remarkably compact encoding of a group's representation-theoretic information.

Let us summarize the key results:

- **Maschke's theorem** guarantees that over $\mathbb{C}$ (or any field of characteristic not dividing $|G|$), every representation of a finite group is a direct sum of irreducible representations. This is the representation-theoretic analogue of the spectral theorem in linear algebra.
- **Schur's lemma** provides the fundamental structural constraint: the only intertwining operators between non-isomorphic irreducibles are zero, and endomorphisms of irreducibles over $\mathbb{C}$ are scalars.
- **Character orthogonality** turns the decomposition problem into an inner product computation. The characters of irreducible representations form an orthonormal basis for the space of class functions.
- **The number of irreducible representations equals the number of conjugacy classes**, giving a beautiful connection between the linear-algebraic and combinatorial aspects of group theory.

The theory extends far beyond finite groups: compact groups have Peter-Weyl theory (a continuous analogue of Maschke's theorem), Lie groups have the theory of highest weights, and infinite-dimensional representations appear in number theory (automorphic forms) and quantum field theory.

In the next article, we step back to look at the bigger picture: **category theory** provides a universal language for describing the common structures we have encountered throughout this series — groups, rings, modules, representations, and the maps between them.

---

*This is Part 10 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 9 — Modules](/en/abstract-algebra/09-modules/)*

*Next: [Part 11 — Category Theory](/en/abstract-algebra/11-category-theory/)*
