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

The reason representation theory works at all is a small miracle: every finite group has *enough* finite-dimensional representations to "see" all of its structure. The decomposition of representations into irreducibles is unique. The number of irreducibles equals the number of conjugacy classes. The character of a representation (the trace of each matrix) is a class function, and the irreducible characters form an orthonormal basis for class functions. These four facts — uniqueness of decomposition, irreducible-conjugacy-class match, orthogonality, completeness — turn representation theory into a *computational* discipline, not just a structural one.

This article is the entry point. Most of what I cover specializes to *finite* groups over $\mathbb{C}$, where the theory is the cleanest. But the same ideas extend, with appropriate modifications, to compact Lie groups (Peter-Weyl), to algebraic groups (rational representations), and to infinite-dimensional contexts (automorphic forms, quantum field theory). The core moves stay the same.

---

## What a Representation Is

Let $G$ be a group and $V$ a finite-dimensional vector space over a field $k$ (we will mostly take $k = \mathbb{C}$). A **representation** of $G$ on $V$ is a group homomorphism

$$\rho : G \to \mathrm{GL}(V).$$

In words: every group element $g$ gets assigned an invertible linear map $\rho(g) : V \to V$, in a way compatible with the group operation: $\rho(gh) = \rho(g) \rho(h)$ and $\rho(e) = I$.

If we pick a basis of $V$, $\mathrm{GL}(V) \cong \mathrm{GL}_n(k)$ where $n = \dim V$, and $\rho(g)$ becomes a matrix. The integer $n$ is the **dimension** (or **degree**) of the representation. Different choices of basis give *equivalent* representations (related by conjugation by the change-of-basis matrix), so the right level of abstraction is "linear maps on $V$" rather than "matrices."

Two representations $\rho_1, \rho_2$ on $V_1, V_2$ are **equivalent** (or isomorphic) if there is an invertible linear map $T : V_1 \to V_2$ with $T \rho_1(g) = \rho_2(g) T$ for all $g$. Equivalently, $\rho_2(g) = T \rho_1(g) T^{-1}$ — the same matrices up to a global change of basis.

![Representation rho: G -> GL(V)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_1_rep_def.png)

A few standard examples to fix the notation.

**Trivial representation.** $V = k$, $\rho(g) = 1$ for all $g$. Boring but always there.

**Sign representation of $S_n$.** $V = k$, $\rho(\sigma) = \mathrm{sgn}(\sigma) \in \{\pm 1\}$. The two one-dimensional representations of $S_n$ for $n \ge 2$ are exactly the trivial and the sign.

**Permutation representation.** $G$ acts on a set $X$; let $V = k^X$ have basis $\{e_x : x \in X\}$ and let $\rho(g)$ permute the basis vectors: $\rho(g) e_x = e_{g \cdot x}$. The dimension is $|X|$.

**Regular representation.** $X = G$ acting on itself by left multiplication. The regular representation has dimension $|G|$.

**Standard representation of $S_n$.** Inside the permutation representation on $\mathbb{C}^n$, the subspace $V_0 = \{(v_1, \ldots, v_n) : \sum v_i = 0\}$ is invariant. The restriction is the standard representation, of dimension $n - 1$.

These five constructions together generate most of what you encounter for finite groups. The structural theory below tells us how to decompose them.

---

## Why Use Vector Spaces?

The choice of $V$ as a vector space — rather than, say, an abstract group or a topological space — is forced by what we want to extract. Vector spaces have:

- **Decomposition into subspaces.** A representation can split as $V = V_1 \oplus V_2$ with $G$ acting on each piece. This is the analogue of factoring a number into primes.
- **Linear-algebraic invariants.** Trace, determinant, eigenvalues, characteristic polynomial. These are coordinate-independent and give numerical invariants of $G$.
- **Inner products and unitarity.** Over $\mathbb{C}$, every finite-group representation can be made unitary (preserve a Hermitian inner product). Then $\rho(g^{-1}) = \rho(g)^*$, i.e., the inverse is the conjugate transpose.

The combination of these features is what makes representation theory powerful. Group theory alone gives you elements and subgroups but no quantitative tools. Vector space theory gives you traces and decompositions but no group structure. Putting them together, you can do both at once. The "representation" word is a literal description of what we are doing: we represent each abstract group element by a concrete linear map, and then we work with the linear maps using the powerful machinery of linear algebra.

The "unitary" point deserves emphasis. Over $\mathbb{C}$, given any representation $\rho$ of a finite group on $V$, the average

$$\langle u, v \rangle' = \frac{1}{|G|} \sum_{g \in G} \langle \rho(g) u, \rho(g) v \rangle$$

defines a $G$-invariant Hermitian inner product. So $\rho$ is unitary with respect to $\langle \cdot, \cdot \rangle'$. This is *Weyl's averaging trick*, and it is the foundation of Maschke's theorem. The averaging only works because $G$ is finite (we can sum over it); for infinite groups you need integration over a compact group, which requires Haar measure — but the principle is the same.

---

## Maschke's Theorem: Complete Reducibility

A subspace $W \subseteq V$ is **invariant** if $\rho(g) W \subseteq W$ for all $g \in G$. A representation is **irreducible** if its only invariant subspaces are $\{0\}$ and $V$. The terminology is a deliberate echo of "irreducible polynomial": these are the building blocks that cannot be broken down further.

The first fundamental theorem:

**Maschke's theorem.** Let $G$ be a finite group, $\mathrm{char}(k) \nmid |G|$. Every representation of $G$ over $k$ decomposes as a direct sum of irreducibles.

The proof: given $V$ with an invariant subspace $W$, find a complementary invariant subspace $W'$ such that $V = W \oplus W'$. Pick any linear projection $\pi : V \to W$. Average it over the group:

$$\bar \pi = \frac{1}{|G|} \sum_{g \in G} \rho(g) \circ \pi \circ \rho(g^{-1}).$$

Then $\bar \pi$ is also a projection onto $W$, and it commutes with the $G$-action. Its kernel is an invariant complement. Iterate.

The condition $\mathrm{char}(k) \nmid |G|$ is what makes the averaging well-defined (you need $|G| \neq 0$ in $k$). Over $\mathbb{C}$ (characteristic $0$), this condition is automatic for any finite group. Over $\mathbb{F}_p$, it fails when $p \mid |G|$, and the resulting "modular representation theory" is genuinely harder.

Maschke's theorem is the foundation. It says: to understand all representations of $G$, it's enough to understand the *irreducible* ones, because everything decomposes.

A small technical comment about uniqueness. The decomposition $V = \bigoplus V_i^{m_i}$ is unique only up to the choice of complement at each step. The *isotypic components* — the sum of all copies of a given irreducible — are uniquely determined as subspaces of $V$. The decomposition *within* each isotypic component (which copy is "first") is not canonical, but for most purposes that does not matter.

---

## Irreducible Representations of $S_3$

Let me make this concrete by listing the irreducible representations of $S_3$, the smallest non-abelian group.

$|S_3| = 6$. Conjugacy classes: $\{e\}$, transpositions $\{(12), (13), (23)\}$, three-cycles $\{(123), (132)\}$. Three classes total. By a theorem we will prove below, the number of irreducible representations equals the number of conjugacy classes — so $S_3$ has exactly $3$ irreducible representations.

**Trivial representation.** Dimension $1$, $\rho(\sigma) = 1$ for all $\sigma$. Call this $\mathbf{1}$.

**Sign representation.** Dimension $1$, $\rho(\sigma) = \mathrm{sgn}(\sigma)$. Call this $\mathrm{sgn}$.

**Standard representation.** Dimension $2$. Inside $\mathbb{C}^3$ with the natural permutation action of $S_3$, the subspace $V_0 = \{(v_1, v_2, v_3) : v_1 + v_2 + v_3 = 0\}$ is $S_3$-invariant. As a representation, the matrices in some basis of $V_0$ are:

$$\rho((12)) = \begin{pmatrix} -1 & 1 \\ 0 & 1 \end{pmatrix}, \quad \rho((123)) = \begin{pmatrix} 0 & -1 \\ 1 & -1 \end{pmatrix}.$$

Call this $V$. We will check below that it is irreducible.

The dimensions $1, 1, 2$ satisfy a constraint: $\sum d_i^2 = 1^2 + 1^2 + 2^2 = 6 = |G|$. This is the **sum-of-squares formula**, which holds for any finite group. It is one of the most powerful tools for finding all irreducibles.

![Three irreducible representations of S_3](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_2_s3_irreps.png)

---

## Schur's Lemma

The single most useful structural theorem in representation theory:

**Schur's lemma.** Let $\rho_1, \rho_2$ be irreducible representations of $G$ on $V_1, V_2$ over an algebraically closed field $k$. Let $T : V_1 \to V_2$ be a $G$-equivariant linear map (i.e., $T \rho_1(g) = \rho_2(g) T$ for all $g$). Then:

1. If $\rho_1 \not\cong \rho_2$, then $T = 0$.
2. If $\rho_1 = \rho_2 = \rho$, then $T$ is a scalar multiple of the identity.

The proof of (1): $\ker T$ and $\mathrm{im}\, T$ are invariant subspaces. Since the representations are irreducible, each is $\{0\}$ or all of $V_i$. The only consistent option (when $V_1 \not\cong V_2$) is $T = 0$.

The proof of (2): pick any eigenvalue $\lambda$ of $T$ (which exists because $k$ is algebraically closed). The eigenspace $\ker(T - \lambda I)$ is invariant and nonzero, hence all of $V$. So $T = \lambda I$.

Schur's lemma is short but enormously consequential. A few immediate corollaries.

**Every irreducible representation of an abelian group is one-dimensional.** *Proof:* if $\rho$ is irreducible and $G$ abelian, then for each $g$, $\rho(g)$ commutes with all $\rho(h)$. By Schur, $\rho(g)$ is a scalar. So every $g$ acts by a scalar, and any one-dimensional subspace is invariant. By irreducibility, $\dim V = 1$.

So $\mathbb{Z}/n\mathbb{Z}$ has $n$ one-dimensional irreducible representations, each given by $1 \mapsto \zeta$ where $\zeta$ is an $n$-th root of unity. The total number $n$ matches the number of conjugacy classes (each element is its own class in an abelian group).

**The endomorphism ring of an irreducible representation is just $k$.** *Proof:* $\mathrm{End}_G(V) = k$ by Schur. For non-irreducible representations $V = m_1 V_1 \oplus \cdots \oplus m_r V_r$ (with $V_i$ distinct irreducibles), $\mathrm{End}_G(V) \cong \prod_i M_{m_i}(k)$ — a product of matrix algebras.

The second corollary leads to **Wedderburn's theorem**: the group algebra $k[G]$ for $G$ finite and $k = \mathbb{C}$ is isomorphic to a direct product of matrix algebras, $\prod_i M_{d_i}(\mathbb{C})$, where the $d_i$ are the dimensions of the irreducibles. The sum-of-squares formula falls out: $|G| = \dim k[G] = \sum_i d_i^2$.

![Schur's lemma](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_6_schur.png)

---

## Characters

The **character** of a representation $\rho : G \to \mathrm{GL}(V)$ is the function

$$\chi_\rho : G \to k, \qquad \chi_\rho(g) = \mathrm{tr}(\rho(g)).$$

The character is a class function: $\chi(hgh^{-1}) = \chi(g)$ since trace is invariant under conjugation. So $\chi$ depends only on the conjugacy class of $g$.

Characters are an enormous simplification. Instead of remembering the entire matrix-valued function $\rho$, you remember a single complex-valued function $\chi$ on $G$. Astonishingly, this is enough: $\rho_1 \cong \rho_2$ iff $\chi_{\rho_1} = \chi_{\rho_2}$, over $\mathbb{C}$. So characters classify representations up to isomorphism.

**Character of the trivial representation:** $\chi(g) = 1$ for all $g$.

**Character of the sign representation of $S_n$:** $\chi(\sigma) = \mathrm{sgn}(\sigma)$.

**Character of the standard representation of $S_3$:** $\chi(e) = 2$, $\chi(\text{transposition}) = 0$, $\chi(\text{3-cycle}) = -1$.

The last one deserves a check. The standard representation of $S_3$ has dimension $2$. Trace at the identity: $2$. For a transposition $\sigma$, the matrix $\rho(\sigma)$ has trace $0$ (compute it explicitly using the matrices above, or note that any reflection in two dimensions has trace $0$). For a 3-cycle, $\rho$ acts as a rotation by $120°$, with trace $2 \cos(120°) = -1$.

A representation $\rho$ has $\chi_\rho(e) = \dim V$ — the character at the identity is the dimension. This is one of the most useful sanity checks, and it gets used constantly in computations: if you write down a character table and the first column doesn't read off the dimensions of the irreducibles, you have an error.

![Character table of S_3](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_3_character_table.png)

---

## The Character Table of $S_3$

Putting the data together, the character table of $S_3$ is:

|              | $e$ | $(12)$ | $(123)$ |
|--------------|-----|--------|---------|
| $\mathbf{1}$ | 1   | 1      | 1       |
| $\mathrm{sgn}$| 1   | -1     | 1       |
| $V$          | 2   | 0      | -1      |

A few things to notice. The columns are indexed by conjugacy class representatives. The first column ($g = e$) is the dimensions of the irreducibles. The first row is all $1$s (trivial representation). The remaining rows have a structural pattern: each row, when paired with itself via the inner product on class functions (defined below), has total weight $1$; different rows have inner product $0$.

These are the **orthogonality relations**, the central computational tool of character theory.

---

## Orthogonality Relations

Define an inner product on class functions on $G$:

$$\langle f, g \rangle = \frac{1}{|G|} \sum_{x \in G} f(x) \overline{g(x)} = \sum_{C} \frac{|C|}{|G|} f(c_C) \overline{g(c_C)},$$

where the second sum is over conjugacy classes $C$ with representative $c_C$.

**First orthogonality relation.** The irreducible characters $\chi_1, \ldots, \chi_r$ satisfy

$$\langle \chi_i, \chi_j \rangle = \delta_{ij}.$$

I.e., distinct irreducibles have inner product $0$, and any irreducible has self-inner-product $1$.

**Second orthogonality relation.** For two conjugacy classes $C, C'$:

$$\sum_{i} \chi_i(c_C) \overline{\chi_i(c_{C'})} = \begin{cases} |G|/|C| & \text{if } C = C' \\ 0 & \text{otherwise.} \end{cases}$$

The first relation says rows of the character table are orthonormal (with the right weighting). The second says columns are orthonormal (with a different weighting). The two together give very strong constraints.

**Verification for $S_3$.** The first relation, comparing rows of the character table:

$\langle \chi_{\mathbf{1}}, \chi_{\mathbf{1}} \rangle = (1/6)(1 \cdot 1 \cdot 1 + 3 \cdot 1 \cdot 1 + 2 \cdot 1 \cdot 1) = 6/6 = 1$. ✓

$\langle \chi_{\mathbf{1}}, \chi_{\mathrm{sgn}} \rangle = (1/6)(1 \cdot 1 \cdot 1 + 3 \cdot 1 \cdot (-1) + 2 \cdot 1 \cdot 1) = 0/6 = 0$. ✓

$\langle \chi_V, \chi_V \rangle = (1/6)(1 \cdot 4 + 3 \cdot 0 + 2 \cdot 1) = 6/6 = 1$. ✓

The orthogonality relations are not just elegant; they are *computationally decisive*. Given a candidate character $\chi$, you can compute $\langle \chi, \chi \rangle$. If it's $1$, $\chi$ is irreducible. If it's a positive integer $n$, $\chi$ is a sum of irreducibles whose multiplicities-squared sum to $n$. Computing $\langle \chi, \chi_i \rangle$ tells you the multiplicity of $\chi_i$ in $\chi$.

![Character orthogonality relations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_4_orthogonality.png)

---

## Decomposing the Regular Representation

The **regular representation** of $G$ is the action of $G$ on itself by left multiplication, viewed as a representation of dimension $|G|$ on the vector space $\mathbb{C}[G]$ with basis $\{e_g\}_{g \in G}$. The character is

$$\chi_{\mathrm{reg}}(g) = \begin{cases} |G| & g = e \\ 0 & g \neq e. \end{cases}$$

(*Reason:* $\rho(g)$ permutes the basis $e_h$, and the trace counts fixed points: $h$ is fixed iff $gh = h$ iff $g = e$.)

The multiplicity of an irreducible $V_i$ in the regular representation is

$$\langle \chi_{\mathrm{reg}}, \chi_i \rangle = \frac{1}{|G|} \sum_g \chi_{\mathrm{reg}}(g) \overline{\chi_i(g)} = \frac{1}{|G|} \cdot |G| \cdot \chi_i(e) = \dim V_i.$$

So the regular representation decomposes as

$$\mathbb{C}[G] \cong \bigoplus_i V_i^{\dim V_i}.$$

Each irreducible appears with multiplicity equal to its dimension. The dimensions match: $|G| = \sum_i (\dim V_i)^2$ — the sum-of-squares formula.

For $S_3$: $\mathbb{C}[S_3] \cong \mathbf{1} \oplus \mathrm{sgn} \oplus V \oplus V \cong \mathbf{1} \oplus \mathrm{sgn} \oplus V^2$, with total dimension $1 + 1 + 2 + 2 = 6 = |S_3|$.

The regular representation contains a copy of *every* irreducible, with high multiplicity. This is sometimes phrased as "the regular representation is the universal representation": any irreducible can be found inside it.

![Decomposition of the regular representation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_5_regular_decomp.png)

---

## Number of Irreducibles Equals Number of Conjugacy Classes

This is the deepest structural fact in finite-group representation theory:

**Theorem.** For a finite group $G$, the number of (equivalence classes of) irreducible complex representations equals the number of conjugacy classes.

The proof uses the orthogonality relations. The character table is a square matrix (after we know the count): rows indexed by irreducibles, columns by classes. The first orthogonality relation says rows are linearly independent. The second says columns are linearly independent. Both conditions can hold only if the matrix is square — i.e., $\#\text{irreducibles} = \#\text{classes}$.

The fact has both a combinatorial flavor (counting) and a structural one (the irreducibles are *indexed by* conjugacy classes, in some natural way for many specific groups). For symmetric groups, the irreducibles are parametrized by partitions of $n$ — which are also the conjugacy classes (cycle types). The bijection is not arbitrary; it is mediated by the *Specht modules*, a beautiful combinatorial construction.

A cleaner conceptual statement: the *space of class functions* on $G$ has dimension equal to the number of conjugacy classes. The irreducible characters span this space (by the first orthogonality relation, they are linearly independent; by a separate argument using the regular representation, they span). So the number of irreducible characters equals the dimension of the class-function space, which equals the number of conjugacy classes.

This is one of those theorems where the proof is a bookkeeping argument but the *content* is profound: the categorical level (irreducible representations) and the combinatorial level (conjugacy classes) are bijective for any finite group, with the bijection mediated by characters. There is nothing analogous for infinite groups in general.

---

## Two Concrete Computations

**Example 1: irreducibles of $\mathbb{Z}/n\mathbb{Z}$.** The group is abelian, so all irreducibles are 1-dimensional. The number of conjugacy classes is $n$ (each element its own class). So there are $n$ irreducibles. They are given by

$$\chi_k : 1 \mapsto e^{2\pi i k / n}, \qquad k = 0, 1, \ldots, n-1.$$

These are exactly the $n$ characters of $\mathbb{Z}/n\mathbb{Z}$, and they form the basis for *discrete Fourier analysis* on $\mathbb{Z}/n\mathbb{Z}$. The fact that the regular representation decomposes as a direct sum of these characters is the inversion formula for the discrete Fourier transform. The connection is direct: a function $f : \mathbb{Z}/n\mathbb{Z} \to \mathbb{C}$ is an element of the regular representation, and writing it in the basis of irreducible characters is exactly computing its Fourier transform.

**Example 2: irreducibles of $D_4$ (dihedral group of order 8).** $D_4$ has $5$ conjugacy classes: $\{e\}$, $\{r^2\}$, $\{r, r^3\}$, $\{s, r^2 s\}$, $\{rs, r^3 s\}$. So there are $5$ irreducibles. By sum-of-squares, $\sum d_i^2 = 8$ with five positive integers. The only solution is $1 + 1 + 1 + 1 + 4 = 8$, i.e., four 1-dimensional and one 2-dimensional.

The four 1-dimensional irreducibles come from the abelianization $D_4^{\mathrm{ab}} = D_4 / [D_4, D_4]$, which has order $4$. The 2-dimensional irreducible is the natural matrix representation of $D_4$ as the symmetries of a square (rotations and reflections in the plane). Its character is $\chi(e) = 2$, $\chi(r^2) = -2$, $\chi(r) = 0$, $\chi(s) = 0$, $\chi(rs) = 0$ — easy to compute since the rotation $r$ has trace $2 \cos(90°) = 0$ and reflections have trace $0$.

Sum-of-squares + dimension counting from the abelianization typically pins down all irreducibles for small groups in a few minutes of paper computation. The same approach extends to $D_n$ in general: the abelianization is $\mathbb{Z}/2 \times \mathbb{Z}/2$ for $n$ even and $\mathbb{Z}/2$ for $n$ odd, giving $4$ or $2$ one-dimensional irreducibles, with the remaining irreducibles being the $\lfloor (n-1)/2 \rfloor$ rotation+reflection $2$-dimensional representations.

---

## Tensor Products and Induced Representations

Two more constructions worth knowing.

**Tensor product.** Given representations $\rho_1$ on $V_1$ and $\rho_2$ on $V_2$, the **tensor product** representation acts on $V_1 \otimes V_2$ by $(\rho_1 \otimes \rho_2)(g) = \rho_1(g) \otimes \rho_2(g)$. The character is the pointwise product: $\chi_{V_1 \otimes V_2}(g) = \chi_{V_1}(g) \chi_{V_2}(g)$.

Tensor products are how you *combine* representations. They show up in physics whenever you have two systems and want to talk about the joint state — e.g., two electrons each with spin-$1/2$ live in $\mathbb{C}^2 \otimes \mathbb{C}^2$, which decomposes into a $3$-dimensional triplet and a $1$-dimensional singlet (this is the classic Clebsch-Gordan decomposition).

The decomposition of a tensor product into irreducibles is generally nontrivial. For finite abelian groups, $\chi \otimes \psi = \chi \cdot \psi$ (pointwise product of one-dimensional characters), so tensor products are easy. For non-abelian groups, you have to compute the inner product of the product character with each irreducible character to find the multiplicities.

**Induction and restriction.** Given a subgroup $H \leq G$ and a representation $\rho$ of $H$, the **induced representation** $\mathrm{Ind}_H^G(\rho)$ is a representation of $G$. Concretely, it's the representation on $\mathbb{C}[G] \otimes_{\mathbb{C}[H]} V$, with $G$ acting on the left factor. Going the other way, given a representation of $G$, you can **restrict** it to $H$ and get a representation of $H$.

Frobenius reciprocity ties them together:

$$\langle \mathrm{Ind}_H^G \rho, \sigma \rangle_G = \langle \rho, \mathrm{Res}_H^G \sigma \rangle_H.$$

This is one of the most useful tools for computing decompositions: induce from a small subgroup, decompose into irreducibles of $G$, and use reciprocity to relate the multiplicities to known data on $H$. The trick is most powerful when $H$ is small enough that its representation theory is already known — say, $H$ a Sylow subgroup or a normal subgroup with abelian quotient.

---

## Connection to Quantum Mechanics: SU(2) and Spin

A glimpse of how representation theory generalizes beyond finite groups, and why physicists care.

The Lie group $\mathrm{SU}(2)$ — $2 \times 2$ unitary matrices with determinant $1$ — has a continuous family of irreducible representations $V_n$ of dimension $n+1$ for each integer $n \geq 0$ (indexed by "spin $n/2$"). Concretely, $V_n$ is the space of homogeneous polynomials of degree $n$ in two variables, with the natural $\mathrm{SU}(2)$ action.

This is the mathematical structure underlying **quantum spin**. An electron has spin $1/2$, meaning it lives in $V_1$, the standard 2-dimensional representation. A photon has spin $1$, meaning $V_2$, dimension $3$. The product (for combined systems) decomposes via the **Clebsch-Gordan formula** $V_m \otimes V_n = V_{m+n} \oplus V_{m+n-2} \oplus \cdots \oplus V_{|m-n|}$.

The whole apparatus of "addition of angular momentum" in quantum mechanics is just representation theory of $\mathrm{SU}(2)$. The quantum numbers (total spin, $z$-component of spin) correspond to invariants of the representation, and the selection rules for atomic transitions correspond to which irreducibles can pair up via tensor product.

The same machinery, with $\mathrm{SU}(3)$ in place of $\mathrm{SU}(2)$, classifies quark flavors in the standard model. We will see this in article 12.

![SU(2) representations and quantum spin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_7_su2_spin.png)

---

## Why Representation Theory Works

Stepping back, what is the fundamental reason that the theory has so much structure? A few answers, each capturing part of it:

- **Maschke's theorem** says representations decompose. This reduces the problem to irreducibles.
- **Schur's lemma** provides the fundamental structural constraint: the only intertwining operators between non-isomorphic irreducibles are zero, and endomorphisms of irreducibles over $\mathbb{C}$ are scalars.
- **Character orthogonality** turns the decomposition problem into an inner product computation. The characters of irreducible representations form an orthonormal basis for the space of class functions.
- **The number of irreducible representations equals the number of conjugacy classes**, giving a beautiful connection between the linear-algebraic and combinatorial aspects of group theory.

Each of these statements is "small" in the sense of being a one-line claim, but cumulatively they turn representation theory into a fully computable subject. Pick a finite group, write down its conjugacy classes, count them, you know the number of irreducibles. Compute character values on a few cases, use orthogonality to extend, derive the entire character table. From the character table, derive matrix representations explicitly. Decompose any representation by computing inner products with characters. The whole process is mechanical once the framework is set up — and that mechanical-ness is the source of the theory's practical power. There is no "creative leap" needed in any individual step; the leap was made once, by Frobenius and Schur, and we are just pushing buttons.

The theory extends far beyond finite groups: compact groups have Peter-Weyl theory (a continuous analogue of Maschke's theorem), Lie groups have the theory of highest weights, and infinite-dimensional representations appear in number theory (automorphic forms) and quantum field theory.

---

## A Few Practical Remarks

If you are going to compute with characters, four practical tips.

**Tip 1.** The trivial representation always appears. So $\langle \chi, \chi_{\mathrm{triv}} \rangle$ tells you how many copies of the trivial rep are in your representation. This is the dimension of the *invariants* $V^G = \{v \in V : \rho(g) v = v \text{ for all } g\}$. This single observation is the source of countless concrete calculations: anywhere you see "average over the group" or "fixed points," there is a hidden character computation.

**Tip 2.** For permutation representations, $\chi(g) = \#\{\text{fixed points of } g\}$. So computing the character is just counting fixed points of group elements acting on the underlying set. This connects directly to Burnside's lemma from earlier in this series. As a corollary: for the natural action of $S_n$ on $\{1, \ldots, n\}$, the permutation character splits as $\chi_{\mathrm{perm}} = \chi_{\mathbf{1}} + \chi_V$, where $V$ is the standard representation. The "minus $1$" in $\chi_V(g) = \#\mathrm{Fix}(g) - 1$ is exactly subtracting off the trivial summand.

**Tip 3.** The 1-dimensional representations of $G$ are exactly the characters of $G^{\mathrm{ab}} = G / [G, G]$, the abelianization. So if you know the abelianization, you know the 1-dimensional irreducibles.

**Tip 4.** Tensor products and exterior/symmetric squares are easy to compute on the character level: $\chi_{V \otimes V}(g) = \chi_V(g)^2$, and $\chi_V(g)^2 = \chi_{\mathrm{Sym}^2 V}(g) + \chi_{\Lambda^2 V}(g)$ with explicit formulas $\chi_{\mathrm{Sym}^2 V}(g) = \frac{1}{2}(\chi_V(g)^2 + \chi_V(g^2))$ and $\chi_{\Lambda^2 V}(g) = \frac{1}{2}(\chi_V(g)^2 - \chi_V(g^2))$.

These tips are the kind of small fact that turns "I know the theory" into "I can do the computation." Every representation-theory exam problem I have seen reduces to one of these four moves applied two or three times.

---

## Failures to Be Aware Of

Three places where the clean theory above breaks down.

**Modular representation theory.** When $\mathrm{char}(k) \mid |G|$, Maschke fails. The representation theory is still rich but messier — there are *projective* representations, non-semisimple algebras, Brauer characters as substitutes for ordinary characters. This is a serious subject in its own right and is a big component of finite group theory at the research level.

**Infinite groups.** Without finiteness, the averaging trick fails, and Maschke's theorem doesn't apply. For *compact* topological groups (like $\mathrm{SU}(2), \mathrm{SO}(n), U(n)$), Haar measure plays the role of "$\frac{1}{|G|}\sum$" and most of the theory survives — Peter-Weyl theorem, character theory, complete reducibility. For *non-compact* groups (like $\mathrm{SL}_2(\mathbb{R})$), even the right notion of "irreducible" requires care, and infinite-dimensional representations are essential.

**Real vs complex coefficients.** The cleanest theory is over $\mathbb{C}$ (algebraically closed, characteristic $0$). Over $\mathbb{R}$, irreducibles are not always one-dimensional even for abelian groups — $\mathbb{Z}/3\mathbb{Z}$ has only the trivial as a real irreducible of dimension $1$, with the other irreducible being a real $2$-dimensional rotation. The "complexification" of a real irrep can be a sum of two complex irreps, etc. This is the *Frobenius-Schur indicator* story.

These failure modes are not pathologies; they are different theories with their own structure. Knowing that the clean version exists for finite groups in characteristic zero is what frames everything else.

---

## A Worked Example: Building the Character Table of $S_4$

To consolidate, let me build the character table of $S_4$ from scratch.

$|S_4| = 24$. Conjugacy classes are indexed by cycle types, i.e., partitions of $4$: $1+1+1+1$ (identity, $1$ element), $2+1+1$ (transpositions, $6$ elements), $2+2$ (products of two disjoint transpositions, $3$ elements), $3+1$ ($3$-cycles, $8$ elements), $4$ ($4$-cycles, $6$ elements). Five classes, total $1 + 6 + 3 + 8 + 6 = 24$. ✓

So $S_4$ has $5$ irreducible representations. By the sum-of-squares formula, $\sum d_i^2 = 24$, with $5$ positive integers. The unique solution (up to ordering) is $1 + 1 + 4 + 9 + 9 = 24$, i.e., dimensions $1, 1, 2, 3, 3$.

We can immediately identify some of these:

- $\mathbf{1}$, trivial, dimension $1$.
- $\mathrm{sgn}$, sign representation, dimension $1$.
- Standard representation $V$, dimension $3$. (The $S_4$-action on $\mathbb{C}^4$ permuting coordinates restricts to the $3$-dimensional subspace where coordinates sum to $0$.)
- $V \otimes \mathrm{sgn}$, dimension $3$. (Twisting the standard by the sign gives another $3$-dimensional irreducible.)
- The remaining irreducible has dimension $2$.

The dimension-$2$ irreducible is harder to spot. It comes from the surjection $S_4 \twoheadrightarrow S_4 / V_4 \cong S_3$, where $V_4 = \{e, (12)(34), (13)(24), (14)(23)\}$ is the Klein four-group (a normal subgroup of $S_4$). The standard 2-dimensional irreducible of $S_3$ pulls back to a 2-dimensional irreducible of $S_4$.

Computing characters by hand:

|              | $e$  | $(12)$ | $(12)(34)$ | $(123)$ | $(1234)$ |
|--------------|------|--------|------------|---------|----------|
| size of class | 1   | 6      | 3          | 8       | 6        |
| $\mathbf{1}$ | 1    | 1      | 1          | 1       | 1        |
| $\mathrm{sgn}$| 1   | -1     | 1          | 1       | -1       |
| $W$ (dim 2)  | 2    | 0      | 2          | -1      | 0        |
| $V$ (dim 3)  | 3    | 1      | -1         | 0       | -1       |
| $V \otimes \mathrm{sgn}$ | 3 | -1 | -1   | 0       | 1        |

To check $V$'s character: the standard representation has $\chi_V(g) = (\text{number of fixed points of } g) - 1$. Identity has $4$ fixed points: $\chi_V(e) = 3$. A transposition has $2$ fixed points: $\chi_V((12)) = 1$. A double transposition has $0$ fixed points: $\chi_V((12)(34)) = -1$. A 3-cycle has $1$ fixed point: $\chi_V((123)) = 0$. A 4-cycle has $0$ fixed points: $\chi_V((1234)) = -1$. ✓

To check $W$'s character: $W$ pulls back from $S_3$ via $S_4 \to S_3$ with kernel $V_4$. The characters depend only on the image in $S_3$. The map sends $V_4$ to identity, transpositions to transpositions, 3-cycles to 3-cycles, and 4-cycles to transpositions. So $\chi_W(e) = 2$, $\chi_W((12)) = 0$, $\chi_W((12)(34)) = 2$, $\chi_W((123)) = -1$, $\chi_W((1234)) = 0$. ✓

Verify orthogonality (one example): $\langle \chi_V, \chi_V \rangle = (1/24)(1 \cdot 9 + 6 \cdot 1 + 3 \cdot 1 + 8 \cdot 0 + 6 \cdot 1) = 24/24 = 1$. ✓ So $V$ is irreducible. The whole table is internally consistent.

This is the kind of computation that, with practice, takes about fifteen minutes by hand. Computer algebra systems (GAP, Magma) do it in milliseconds for groups of order up to a few thousand, and use sophisticated algorithms (Dixon-Schneider, in particular) for groups of order up to about $10^7$.

---

## A Final Comment on Why It's Beautiful

I want to end with one observation that I find genuinely satisfying. The character table of a finite group is a very specific finite object — for $S_4$, it's a $5 \times 5$ matrix of integers (well, complex numbers in general, but for $S_4$ they happen to all be integers). This matrix encodes essentially everything about the group's "linear" structure: every representation, every decomposition, every invariant.

But the matrix is *forced* by very few inputs: just the conjugacy classes and their sizes. From those, plus the orthogonality relations and the sum-of-squares formula, you can reconstruct the entire character table by hand. The amount of information you put in (a list of conjugacy class sizes) is much smaller than what you get out (a full description of all linear actions of the group).

This is a kind of mathematical leverage. Group theory + linear algebra + a bit of cleverness about averaging produces character theory, and character theory has an internal coherence (orthogonality, sum-of-squares, conjugacy-irreducible duality) that makes it self-correcting. Make an arithmetic mistake and the orthogonality relations will catch it. Misidentify the dimension and the sum-of-squares formula will show it. The theory is robust to small errors because it has too much structure to allow them.

Whether or not you ever use representation theory in your own work, the experience of seeing this much structure emerge from this little input is, in itself, one of the highlights of an algebra course. It is the closest abstract algebra gets to the kind of computational satisfaction you get from explicit calculation in concrete number theory or geometry.

---

## What's Next

In the next article, we step back to look at the bigger picture: **category theory** provides a universal language for describing the common structures we have encountered throughout this series — groups, rings, modules, representations, and the maps between them. The pattern of "decompose into pieces, find irreducibles, count by conjugacy classes" we just developed for representations is one instance of a general phenomenon, and category theory is the language for stating that generality precisely.

---

*This is Part 10 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 9 — Modules](/en/abstract-algebra/09-modules/)*

*Next: [Part 11 — Category Theory](/en/abstract-algebra/11-category-theory/)*
