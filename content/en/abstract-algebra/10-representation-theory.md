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

This article specializes to *finite* groups over $\mathbb{C}$, where the theory is cleanest. But the same ideas extend to compact Lie groups (Peter-Weyl), algebraic groups (rational representations), and infinite-dimensional contexts (automorphic forms, quantum field theory). The core moves stay the same.

The historical context is worth a sentence. Frobenius invented characters in 1896 to study the factorization of the "group determinant" (a polynomial associated to the group multiplication table). Burnside and Schur developed the matrix-theoretic viewpoint in the 1900s. The subject reached maturity with the proofs of the Burnside $p^a q^b$ theorem (1904, proved using character theory — the first major application) and the eventual proof of the Feit-Thompson theorem (1963, that all groups of odd order are solvable, using character theory as a key ingredient). Character theory is not just a classification tool; it has been essential to the *proof* of major structural theorems about finite groups.

---

## Representations, Complete Reducibility, and the Averaging Trick

Let $G$ be a group and $V$ a finite-dimensional vector space over $\mathbb{C}$. A **representation** of $G$ on $V$ is a group homomorphism $\rho : G \to \mathrm{GL}(V)$. In words: every group element $g$ gets assigned an invertible linear map $\rho(g) : V \to V$, compatible with the group operation: $\rho(gh) = \rho(g)\rho(h)$ and $\rho(e) = I$. Once you pick a basis of $V$, each $\rho(g)$ becomes an invertible matrix, and the group operation becomes matrix multiplication. The integer $n = \dim V$ is the **degree** of the representation. Different choices of basis give *equivalent* representations (related by conjugation $T\rho(g)T^{-1}$), so the natural level of abstraction is "linear maps on $V$" rather than specific matrices.

![Representation rho: G -> GL(V)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_1_rep_def.png)

Standard examples to fix notation. The **trivial representation**: $V = \mathbb{C}$, $\rho(g) = 1$ for all $g$. Boring but always present, and always relevant (it measures invariants). The **sign representation** of $S_n$: $V = \mathbb{C}$, $\rho(\sigma) = \mathrm{sgn}(\sigma) \in \{\pm 1\}$. The **permutation representation**: given a $G$-action on a finite set $X$, let $V = \mathbb{C}^X$ have basis $\{e_x : x \in X\}$ and let $\rho(g)e_x = e_{g \cdot x}$; the dimension is $|X|$ and the character $\chi(g) = \#\mathrm{Fix}(g)$. The **regular representation**: $X = G$ acting on itself by left multiplication, dimension $|G|$, the "biggest" natural representation. The **standard representation** of $S_n$: inside the permutation representation on $\mathbb{C}^n$, the subspace $V_0 = \{(v_1, \ldots, v_n) : \sum v_i = 0\}$ is invariant and has dimension $n-1$; this is the standard representation.

A **subrepresentation** is a subspace $W \subseteq V$ that is $G$-invariant: $\rho(g)W \subseteq W$ for all $g$. A representation is **irreducible** if it has no proper nontrivial subrepresentations — the only invariant subspaces are $\{0\}$ and $V$ itself. Irreducibles are the atoms of the theory; everything else is built from them.

**Maschke's theorem** is the foundational structural result: if $G$ is finite and $\mathrm{char}(k) \nmid |G|$ (automatic over $\mathbb{C}$), then every representation is **completely reducible** — it decomposes as a direct sum of irreducibles: $V \cong V_1^{\oplus m_1} \oplus V_2^{\oplus m_2} \oplus \cdots \oplus V_k^{\oplus m_k}$, where the $V_i$ are distinct irreducibles and $m_i$ are multiplicities. The decomposition is unique up to reordering.

**Why Maschke is true** — the key idea is the averaging trick. Suppose $W \subseteq V$ is $G$-invariant; we want a $G$-invariant complement. Take *any* linear complement $W'$ (which exists as a vector space but is probably not $G$-invariant). Let $\pi : V \to W$ be the projection onto $W$ along $W'$. Now define the averaged projection:

$$\bar{\pi}(v) = \frac{1}{|G|} \sum_{g \in G} \rho(g) \, \pi \, (\rho(g)^{-1} v).$$

Claim: $\bar{\pi}$ is (1) a linear map $V \to W$, (2) a projection ($\bar{\pi}|_W = \mathrm{id}_W$), and (3) $G$-equivariant ($\bar{\pi} \circ \rho(h) = \rho(h) \circ \bar{\pi}$). Properties (1) and (2) are straightforward. Property (3) follows from the fact that averaging over the group commutes with the group action (the sum over $G$ is re-indexed by left multiplication). Then $\ker \bar{\pi}$ is the desired $G$-invariant complement to $W$.

The averaging trick — "start with something non-equivariant, average over $G$ to make it equivariant" — is the engine behind the entire theory. It appears again in the proof of the orthogonality relations, in the construction of unitary structures, and in Weyl's unitary trick for compact Lie groups. The finite sum $\frac{1}{|G|}\sum_g$ is replaced by the Haar integral $\int_G$ in the compact case; the idea is identical.

An equivalent formulation: every representation of a finite group over $\mathbb{C}$ can be made **unitary**. Given any Hermitian inner product $\langle \cdot, \cdot \rangle$ on $V$, define $\langle u, v \rangle_G = \frac{1}{|G|} \sum_g \langle \rho(g)u, \rho(g)v \rangle$. This is a $G$-invariant inner product, and orthogonal complements of invariant subspaces are invariant. Unitarity implies complete reducibility; they are the same theorem in different clothing.

The concrete content of Maschke's theorem is best appreciated by contrast with what happens when it fails. Consider $G = \mathbb{Z}/p\mathbb{Z}$ acting on $V = \mathbb{F}_p^2$ (a 2-dimensional vector space over the field with $p$ elements) by $\rho(1) = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. The subspace $W = \mathrm{span}\{e_1\}$ is invariant (it is the eigenspace for eigenvalue $1$), but there is no invariant complement — the only other eigenspace is $W$ again ($\rho(1) - I$ is nilpotent of rank $1$). This is a 2-dimensional representation that is reducible but *not* completely reducible. It is an "indecomposable" module that cannot be split. This failure is exactly what Maschke's theorem rules out: in characteristic $p$ dividing $|G|$, the averaging trick divides by $0$, and such non-split extensions exist. The entire edifice of character theory rests on this not happening over $\mathbb{C}$.

---

## Schur's Lemma and Character Theory

Once you have irreducibles as atoms, you need to understand the maps between them. **Schur's lemma** provides the definitive answer and is the structural backbone of the entire character theory.

**Schur's Lemma.** Let $V, W$ be irreducible representations of $G$ over $\mathbb{C}$, and let $T : V \to W$ be $G$-equivariant ($T\rho_V(g) = \rho_W(g)T$ for all $g$). Then: (a) either $T = 0$ or $T$ is an isomorphism; (b) if $V = W$, then $T = \lambda I$ for some $\lambda \in \mathbb{C}$.

The proof of (a): $\ker T$ is a $G$-invariant subspace of $V$ (irreducible), hence $\ker T \in \{\{0\}, V\}$. Similarly $\mathrm{Im}\, T$ is $G$-invariant in $W$ (irreducible), hence $\mathrm{Im}\, T \in \{\{0\}, W\}$. Combining: $T = 0$ or $T$ is bijective. For (b): over $\mathbb{C}$, $T$ has an eigenvalue $\lambda$. Then $T - \lambda I$ is equivariant with nontrivial kernel, hence zero by part (a). So $T = \lambda I$.

**Why Schur matters.** It says that $\mathrm{Hom}_G(V_i, V_j)$ — the space of $G$-equivariant maps between irreducibles — is either $0$ (if $V_i \not\cong V_j$) or $\mathbb{C}$ (if $V_i \cong V_j$). This rigidity makes decomposition unique and makes characters work.

**Immediate consequence for abelian groups.** If $G$ is abelian, every $\rho(g)$ commutes with every $\rho(h)$, so each $\rho(g)$ is an equivariant endomorphism of any irreducible. By Schur, $\rho(g) = \lambda_g I$. But then every $1$-dimensional subspace is invariant, so irreducibility forces $\dim V = 1$. All irreducibles of a finite abelian group are one-dimensional. For $\mathbb{Z}/n\mathbb{Z}$, they are $\chi_k(1) = e^{2\pi i k/n}$ for $k = 0, \ldots, n-1$ — the characters of $\mathbb{Z}/n\mathbb{Z}$, which are the basis for discrete Fourier analysis. The DFT is literally the change-of-basis matrix from the standard basis to the irreducible-character basis of the regular representation of $\mathbb{Z}/n\mathbb{Z}$.

For a general finite abelian group $G \cong \mathbb{Z}/n_1 \times \cdots \times \mathbb{Z}/n_r$, the irreducibles are the products $\chi_{k_1} \otimes \cdots \otimes \chi_{k_r}$ — one for each element of $G$ itself. The group of characters $\hat{G} = \mathrm{Hom}(G, \mathbb{C}^\times)$ is isomorphic to $G$ (non-canonically), and the theory of characters of abelian groups is the theory of the Fourier transform on finite abelian groups. This is the starting point for harmonic analysis on locally compact abelian groups (Pontryagin duality) and ultimately for the representation theory of adele groups in number theory.

The **character** of a representation $(\rho, V)$ is $\chi_V : G \to \mathbb{C}$, $\chi_V(g) = \mathrm{tr}(\rho(g))$. Its key properties: $\chi_V(e) = \dim V$; $\chi_V(hgh^{-1}) = \chi_V(g)$ (trace is conjugation-invariant, so characters are class functions); $\chi_V(g^{-1}) = \overline{\chi_V(g)}$ (from unitarity); $\chi_{V \oplus W} = \chi_V + \chi_W$; $\chi_{V \otimes W} = \chi_V \cdot \chi_W$ (pointwise product).

Two representations are isomorphic if and only if they have the same character. This is a non-obvious theorem (the character only records traces, not full matrices), and it is what makes characters a complete invariant for the classification problem. The proof uses complete reducibility: two semisimple modules with the same composition factors (counted with multiplicity) are isomorphic, and the multiplicities are determined by the character inner products. Over fields where Maschke fails, the character no longer determines the representation — in modular representation theory, two non-isomorphic indecomposable modules can have the same Brauer character.

Define an inner product on class functions: $\langle \chi, \psi \rangle = \frac{1}{|G|} \sum_{g \in G} \chi(g)\overline{\psi(g)}$.

**First orthogonality relation.** If $\chi_i, \chi_j$ are irreducible characters, then $\langle \chi_i, \chi_j \rangle = \delta_{ij}$.

The proof uses Schur's lemma at the matrix-coefficient level: $\frac{1}{|G|}\sum_g \rho_i(g)_{ab} \overline{\rho_j(g)_{cd}} = \frac{1}{d_i}\delta_{ij}\delta_{ac}\delta_{bd}$ (where $d_i = \dim V_i$). Setting $a = b$, $c = d$, and summing over all diagonal entries gives the character orthogonality.

**Decomposition formula.** For any representation $V$, the multiplicity of the irreducible $V_i$ in $V$ is: $m_i = \langle \chi_V, \chi_i \rangle = \frac{1}{|G|}\sum_g \chi_V(g)\overline{\chi_i(g)}$. No searching for invariant subspaces, no guessing — just an inner product.

**Irreducibility test.** $V$ is irreducible iff $\langle \chi_V, \chi_V \rangle = 1$. If this inner product equals $m$, then $V$ has $m$ irreducible summands (counted with multiplicity).

---

## Counting Irreducibles and Building Character Tables

The **regular representation** $\mathbb{C}[G]$ (left multiplication of $G$ on itself) decomposes as $\bigoplus_i V_i^{\oplus d_i}$ — each irreducible appears with multiplicity equal to its own dimension. Taking dimensions: $|G| = \sum_i d_i^2$. This is the **sum-of-squares formula**, and it constrains possible dimensions severely.

**Number of irreducibles = number of conjugacy classes.** The irreducible characters are orthonormal in the space of class functions. That space has dimension $k$ (the number of conjugacy classes). An orthonormal set in $\mathbb{C}^k$ has at most $k$ elements; a separate argument (the regular representation decomposes into exactly $k$ distinct irreducibles) shows there are at least $k$. So there are exactly $k$ irreducibles, and their characters form an orthonormal basis.

This bijection — irreducibles $\leftrightarrow$ conjugacy classes — is one of the deepest structural facts in the theory. For symmetric groups, both sides are indexed by partitions of $n$ (conjugacy classes are cycle types, irreducibles are Specht modules parametrized by Young diagrams). For the general finite group, the bijection exists abstractly but need not be canonical — there is no uniform way to pair a specific irreducible with a specific conjugacy class that works for all groups. The search for "natural" parametrizations of irreducibles (by Lusztig characters, by $L$-packets, by nilpotent orbits) is one of the driving forces of modern representation theory.

**Building the character table of $S_4$.** $|S_4| = 24$. Five conjugacy classes (by cycle type): $(1^4)$ (size $1$), $(2,1^2)$ (size $6$), $(2^2)$ (size $3$), $(3,1)$ (size $8$), $(4)$ (size $6$). Five irreducibles. Sum-of-squares: $24 = 1 + 1 + 4 + 9 + 9$, giving dimensions $1, 1, 2, 3, 3$.

Identification: $\mathbf{1}$ (trivial); $\mathrm{sgn}$ (sign); $V$ (standard, dimension $3$, character = fixed points $- 1$); $V \otimes \mathrm{sgn}$ (dimension $3$, twist standard by sign); $W$ (dimension $2$, pulled back from $S_4/V_4 \cong S_3$ via the quotient map).

Computing $\chi_V$: $\chi_V(e) = 3$, $\chi_V((12)) = 2 - 1 = 1$, $\chi_V((12)(34)) = 0 - 1 = -1$, $\chi_V((123)) = 1 - 1 = 0$, $\chi_V((1234)) = 0 - 1 = -1$.

Computing $\chi_W$: the quotient $S_4 \to S_3$ has kernel $V_4 = \{e, (12)(34), (13)(24), (14)(23)\}$. Under this map: $V_4 \mapsto e$, transpositions $\mapsto$ transpositions, $3$-cycles $\mapsto 3$-cycles, $4$-cycles $\mapsto$ transpositions. The standard character of $S_3$ on these images: $\chi_W(e) = 2$, $\chi_W((12)) = 0$, $\chi_W((12)(34)) = 2$, $\chi_W((123)) = -1$, $\chi_W((1234)) = 0$.

| | $e$ | $(12)$ | $(12)(34)$ | $(123)$ | $(1234)$ |
|---|---|---|---|---|---|
| size | 1 | 6 | 3 | 8 | 6 |
| $\mathbf{1}$ | 1 | 1 | 1 | 1 | 1 |
| sgn | 1 | -1 | 1 | 1 | -1 |
| $W$ | 2 | 0 | 2 | -1 | 0 |
| $V$ | 3 | 1 | -1 | 0 | -1 |
| $V \otimes \mathrm{sgn}$ | 3 | -1 | -1 | 0 | 1 |

Verification: $\langle \chi_V, \chi_V \rangle = \frac{1}{24}(9 + 6 + 3 + 0 + 6) = 24/24 = 1$. Irreducible. $\langle \chi_V, \chi_W \rangle = \frac{1}{24}(6 + 0 - 6 + 0 + 0) = 0$. Orthogonal. Column sums-of-squares: $\frac{1}{24}(1 + 1 + 4 + 9 + 9) \cdot (\text{column size}) = 1$ for each column when properly computed. The table is self-consistent.

The systematic procedure — count conjugacy classes, solve sum-of-squares for dimensions, identify 1-dimensional reps from the abelianization $G^{\mathrm{ab}}$, use fixed-point formula for permutation characters, fill remaining entries from orthogonality — works for any group where you can enumerate conjugacy classes.

A useful structural observation: the character table is always a square matrix (rows = irreducibles, columns = conjugacy classes), and it satisfies *column orthogonality* as well as row orthogonality. The second orthogonality relation states: $\sum_i \chi_i(g)\overline{\chi_i(h)} = \frac{|G|}{|C_G(g)|} \delta_{[g],[h]}$, where $[g], [h]$ denote conjugacy classes. This is sometimes more convenient computationally — if you know most of a column, the second orthogonality relation can pin down the missing entry.

For computer algebra, the Dixon-Schneider algorithm computes character tables of groups up to order $\sim 10^6$ efficiently. The basic idea: compute the class multiplication coefficients $a_{ijk}$ (how many ways to write a representative of class $k$ as a product of elements from classes $i$ and $j$), then simultaneously diagonalize the resulting "class matrices." The irreducible characters appear as common eigenvectors. For groups beyond this range (e.g., sporadic simple groups), specialized techniques and massive computation are needed — the character table of the Monster group ($|M| \approx 8 \times 10^{53}$, $194$ conjugacy classes) was completed by Fischer, Livingstone, and Thackray in 1978 and fills several pages.

---

## Tensor Products, Induction, and Frobenius Reciprocity

Two constructions generate new representations from old, and they are related by one of the most useful duality results in the subject.

**Tensor products.** Given representations $(\rho_1, V_1)$ and $(\rho_2, V_2)$ of $G$, the tensor product $V_1 \otimes V_2$ carries the action $g \cdot (v_1 \otimes v_2) = (\rho_1(g)v_1) \otimes (\rho_2(g)v_2)$. Its character is the pointwise product: $\chi_{V_1 \otimes V_2}(g) = \chi_{V_1}(g)\chi_{V_2}(g)$. To decompose a tensor product into irreducibles, compute inner products: $m_i = \langle \chi_{V_1} \cdot \chi_{V_2}, \chi_i \rangle$.

For symmetric and exterior powers: $\chi_{\mathrm{Sym}^2 V}(g) = \frac{1}{2}(\chi_V(g)^2 + \chi_V(g^2))$ and $\chi_{\Lambda^2 V}(g) = \frac{1}{2}(\chi_V(g)^2 - \chi_V(g^2))$. These decompose $V \otimes V = \mathrm{Sym}^2 V \oplus \Lambda^2 V$ and are computable directly from the character of $V$.

**Induction and restriction.** Given $H \leq G$ and a representation $(\sigma, W)$ of $H$, the **induced representation** $\mathrm{Ind}_H^G(W)$ is a representation of $G$ of dimension $[G:H] \cdot \dim W$. Concretely: pick coset representatives $g_1, \ldots, g_m$ for $G/H$; then $\mathrm{Ind}_H^G(W) = \bigoplus_{i=1}^m g_i \otimes W$ as a vector space, with $G$ permuting the coset summands and acting through $\sigma$ within each. The character formula:

$$\chi_{\mathrm{Ind}_H^G W}(g) = \frac{1}{|H|} \sum_{\substack{x \in G \\ x^{-1}gx \in H}} \chi_W(x^{-1}gx).$$

Going the other direction, **restriction** $\mathrm{Res}_H^G V$ just views a $G$-representation as an $H$-representation by forgetting the action of elements outside $H$.

**Frobenius reciprocity** is the fundamental adjunction:

$$\langle \mathrm{Ind}_H^G W, V \rangle_G = \langle W, \mathrm{Res}_H^G V \rangle_H.$$

This says: the multiplicity of an irreducible $V$ of $G$ in an induced representation equals the multiplicity of the restriction of $V$ to $H$ containing $W$. It converts a hard computation (decomposing induced reps of the big group) into an easier one (decomposing restrictions to the small group). The trick is most powerful when $H$ has known representation theory — say, $H$ is cyclic, or a maximal torus, or an abelian normal subgroup.

**Worked example.** Let $G = S_3$, $H = \langle (123) \rangle \cong \mathbb{Z}/3$ (index $2$). Induce the character $\chi_1$ of $H$ defined by $\chi_1((123)) = \omega = e^{2\pi i/3}$ (a non-trivial 1-dimensional rep of $H$). Then $\mathrm{Ind}_H^G(\chi_1)$ has dimension $2 \cdot 1 = 2$.

Using Frobenius reciprocity to decompose: $\langle \mathrm{Ind}_H^G \chi_1, \mathbf{1} \rangle_G = \langle \chi_1, \mathrm{Res}_H \mathbf{1} \rangle_H = \langle \chi_1, \mathbf{1}_H \rangle_H = \frac{1}{3}(1 + \omega + \omega^2) = 0$. $\langle \mathrm{Ind}_H^G \chi_1, \mathrm{sgn} \rangle_G = \langle \chi_1, \mathrm{Res}_H \mathrm{sgn} \rangle_H = \langle \chi_1, \mathbf{1}_H \rangle_H = 0$ (sign restricted to the index-2 subgroup of $3$-cycles is trivial). $\langle \mathrm{Ind}_H^G \chi_1, \chi_V \rangle_G = \langle \chi_1, \mathrm{Res}_H V \rangle_H = \frac{1}{3}(\omega \cdot 2 + \omega^2 \cdot (-1) + 1 \cdot (-1))$ ... actually let me compute $\mathrm{Res}_H V$ directly: $\chi_V(e) = 2$, $\chi_V((123)) = -1$, $\chi_V((132)) = -1$. So $\langle \chi_1, \mathrm{Res}_H V \rangle = \frac{1}{3}(1 \cdot 2 + \omega \cdot (-1) + \omega^2 \cdot (-1)) = \frac{1}{3}(2 - \omega - \omega^2) = \frac{1}{3}(2 + 1) = 1$. So $\mathrm{Ind}_H^G(\chi_1) \cong V$ — the induced representation is the standard irreducible of $S_3$.

This illustrates the general principle: inducing from a small subgroup and decomposing via Frobenius reciprocity is often the most efficient way to construct irreducibles. For symmetric groups, the entire representation theory (Specht modules) can be developed by carefully choosing which representations of which subgroups to induce.

The **Mackey formula** (or Mackey's restriction formula) generalizes Frobenius reciprocity to handle the restriction of an induced representation to a *different* subgroup. If $H, K \leq G$, then

$$\mathrm{Res}_K^G \, \mathrm{Ind}_H^G(W) \;\cong\; \bigoplus_{s \in K \backslash G / H} \mathrm{Ind}_{K \cap sHs^{-1}}^K \, \mathrm{Res}_{K \cap sHs^{-1}}^{sHs^{-1}}({}^s W),$$

where the sum runs over double coset representatives and ${}^sW$ denotes $W$ twisted by conjugation. This looks complicated, but in practice it reduces the decomposition of induced representations to a purely combinatorial problem about double cosets. The Mackey formula is essential for computing branching rules (how irreducibles of $G$ decompose upon restriction to $K$) and for the Harish-Chandra philosophy in the representation theory of reductive groups.

**The group algebra perspective.** There is an equivalent (and sometimes more convenient) formulation of all of this in terms of the group algebra $\mathbb{C}[G] = \{\sum_{g \in G} a_g g : a_g \in \mathbb{C}\}$, a $|G|$-dimensional $\mathbb{C}$-algebra with basis $G$ and multiplication extended linearly from the group operation. Representations of $G$ are exactly left $\mathbb{C}[G]$-modules. Maschke's theorem says $\mathbb{C}[G]$ is semisimple. The Artin-Wedderburn theorem then gives $\mathbb{C}[G] \cong \prod_{i=1}^k M_{d_i}(\mathbb{C})$ — a product of matrix algebras, one per irreducible, with the $i$-th factor having size $d_i \times d_i$. This isomorphism simultaneously explains: why there are $k$ irreducibles (one per matrix factor); why $\sum d_i^2 = |G|$ (the dimensions add up); and why the center of $\mathbb{C}[G]$ has dimension $k$ (one scalar matrix per factor) — which equals the number of conjugacy classes (since the center is spanned by class sums $\sum_{g \in C} g$ for conjugacy classes $C$).

---

## Beyond Finite Groups: Compact Lie Groups and Physics

The structural framework — irreducibles, characters, decomposition, orthogonality — extends to compact Lie groups, where the finite sum $\frac{1}{|G|}\sum_g$ is replaced by the Haar integral $\int_G$. The **Peter-Weyl theorem** is the continuous analogue of Maschke: every unitary representation of a compact group decomposes into finite-dimensional irreducibles, and the matrix coefficients of irreducibles form an orthonormal basis for $L^2(G)$.

The Lie group $\mathrm{SU}(2)$ — $2 \times 2$ unitary matrices with determinant $1$ — has a discrete family of irreducible representations $V_n$ of dimension $n+1$ for each integer $n \geq 0$, indexed by "spin $n/2$." Concretely, $V_n$ is the space of homogeneous polynomials of degree $n$ in two variables $z_1, z_2$, with $\mathrm{SU}(2)$ acting by linear substitution. The character on a rotation by angle $\theta$ (an element conjugate to $\mathrm{diag}(e^{i\theta/2}, e^{-i\theta/2})$) is $\chi_n(\theta) = \frac{\sin((n+1)\theta/2)}{\sin(\theta/2)}$.

![SU(2) representations and quantum spin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/10-representation-theory/aa_v2_10_7_su2_spin.png)

This is the mathematical structure underlying **quantum spin**. An electron (spin $1/2$) lives in $V_1$, dimension $2$: the two basis states are "spin up" and "spin down." A photon (spin $1$) lives in $V_2$, dimension $3$: the three polarization states. Combining systems means tensoring representations, which decomposes via the **Clebsch-Gordan formula**:

$$V_m \otimes V_n \;\cong\; V_{m+n} \oplus V_{m+n-2} \oplus \cdots \oplus V_{|m-n|}.$$

The "addition of angular momentum" rules in quantum mechanics — two spin-$1/2$ particles combine into spin $0$ or spin $1$ — is exactly $V_1 \otimes V_1 \cong V_2 \oplus V_0$, i.e., $\mathbb{C}^2 \otimes \mathbb{C}^2 \cong \mathbb{C}^3 \oplus \mathbb{C}^1$. Selection rules for atomic transitions correspond to which irreducibles appear in specific tensor products with the adjoint representation (dipole transitions correspond to the $V_2$ component).

The same machinery with $\mathrm{SU}(3)$ in place of $\mathrm{SU}(2)$ classifies quark flavors in the standard model of particle physics. The "eightfold way" (Gell-Mann's classification of hadrons) is the adjoint representation of $\mathrm{SU}(3)$, dimension $8$. The prediction and subsequent discovery of the $\Omega^-$ baryon was a triumph of representation theory applied to physics — the particle was predicted to exist because it was needed to complete an irreducible representation.

More broadly, the representation theory of the Poincare group (the symmetry group of special relativity) classifies all possible types of elementary particles: each irreducible unitary representation corresponds to a particle type, labeled by mass and spin. Wigner's classification (1939) showed that the representations are parametrized by these two numbers: $m \geq 0$ (mass, a continuous parameter) and $s \in \{0, 1/2, 1, 3/2, \ldots\}$ (spin, discrete). Massless particles ($m = 0$) have a different structure (helicity replaces spin), which is why photons have only two polarization states rather than three. The fact that "what kinds of particles can exist" is a representation-theoretic question — answerable by classifying irreducibles of a symmetry group — is one of the deepest connections between pure algebra and the physical world.

For *non-compact* groups (like $\mathrm{SL}_2(\mathbb{R})$ or the Poincare group), finite-dimensional representations are inadequate — infinite-dimensional unitary representations become essential. The Langlands program, one of the deepest ongoing research efforts in mathematics, studies these infinite-dimensional representations and their connection to number theory via $L$-functions. The finite-group character theory we developed is the toy model.

One bridge between the finite and Lie-group worlds deserves emphasis: **finite groups of Lie type**. Groups like $\mathrm{GL}_n(\mathbb{F}_q)$, $\mathrm{SL}_n(\mathbb{F}_q)$, and the finite simple groups of types $A_n, B_n, \ldots$ are finite groups whose representation theory shares features with both the finite and Lie-group settings. Deligne and Lusztig (1976) constructed the irreducible representations of these groups using $\ell$-adic cohomology of algebraic varieties — a breathtaking synthesis of algebraic geometry, representation theory, and finite group theory. The character theory of $\mathrm{GL}_2(\mathbb{F}_q)$ is a concrete entry point: it has $q - 1$ one-dimensional representations (characters of the determinant), $(q-1)(q-2)/2$ "principal series" representations of dimension $q + 1$ (induced from the Borel subgroup), and $(q^2 - q)/2$ "cuspidal" representations of dimension $q - 1$ (constructed via the Weil representation or Deligne-Lusztig theory). The total number of irreducibles equals the number of conjugacy classes ($q^2 - 1$), as it must.

---

## Failure Modes and Practical Computation

Three places where the clean theory breaks down, followed by computational tips.

**Modular representation theory** ($\mathrm{char}(k) \mid |G|$). When the characteristic of the field divides the group order, Maschke's theorem fails — the averaging trick requires dividing by $|G|$, which is $0$ in characteristic $p$. Representations need not decompose into irreducibles; there are indecomposable-but-reducible modules (like the $\mathbb{F}_p[\mathbb{Z}/p]$-module from the earlier Maschke failure example). The group algebra $k[G]$ is no longer semisimple — it has a non-zero Jacobson radical. The replacement theory uses **Brauer characters** (defined only on $p$-regular elements, i.e., those of order coprime to $p$), projective indecomposable modules (PIMs), and block decomposition (a partition of modules into chunks controlled by central idempotents of $kG$). The key structural theorem: the number of simple $kG$-modules equals the number of $p$-regular conjugacy classes — a direct substitute for the characteristic-zero "irreducibles = conjugacy classes" result. Modular representation theory is essential to the classification of finite simple groups.

**Infinite non-compact groups.** Without compactness, the Haar integral may not produce a finite $G$-invariant inner product, and representations may be irreducibly infinite-dimensional. The right category is "admissible representations" or "unitary representations on Hilbert spaces," and even the classification of irreducibles is a deep problem (the "unitary dual" problem). For $\mathrm{SL}_2(\mathbb{R})$, the irreducible unitary representations include: the principal series (parametrized by a continuous parameter $s \in i\mathbb{R}$, all infinite-dimensional); the discrete series (parametrized by integers $n \geq 2$, corresponding to holomorphic forms of weight $n$); the complementary series (parametrized by $s \in (0,1)$, the most mysterious); and the trivial representation. The richness of this list — compared to the finite discrete set for compact groups — is what makes the representation theory of non-compact groups a vast and active research area.

**Real versus complex representations.** Over $\mathbb{R}$, Schur's lemma gives $\mathrm{End}_G(V) \in \{\mathbb{R}, \mathbb{C}, \mathbb{H}\}$ (the three real division algebras, by Frobenius's classification). The **Frobenius-Schur indicator** $\nu(\chi) = \frac{1}{|G|}\sum_g \chi(g^2)$ tells you which case you are in: $\nu = 1$ means the representation is real (defined over $\mathbb{R}$, the complexification of a real irreducible); $\nu = -1$ means quaternionic (symplectic — it takes two copies over $\mathbb{R}$ to realize); $\nu = 0$ means genuinely complex (the representation and its complex conjugate are inequivalent). For $\mathbb{Z}/3$: the non-trivial irreducibles $\chi_1, \chi_2$ over $\mathbb{C}$ are complex conjugates of each other (both have $\nu = 0$), and over $\mathbb{R}$ they combine into a single $2$-dimensional real irreducible (a rotation by $2\pi/3$). The quaternion group $Q_8$ has a $2$-dimensional complex irreducible with $\nu = -1$: its endomorphism algebra is $\mathbb{H}$, and over $\mathbb{R}$ it splits into a $4$-dimensional irreducible.

The Frobenius-Schur indicator connects to physics: representations with $\nu = 1$ admit symmetric invariant bilinear forms (and correspond to bosonic symmetries in quantum mechanics); those with $\nu = -1$ admit antisymmetric (symplectic) forms (fermionic symmetries). The trichotomy $\mathbb{R}/\mathbb{C}/\mathbb{H}$ reappears in random matrix theory as the three Dyson ensembles (GOE/GUE/GSE), classified by the same real/complex/quaternionic division.

**Practical tips.** (1) For permutation representations, $\chi(g) = \#\mathrm{Fix}(g)$ — just count fixed points. This connects to Burnside's lemma: $\#\text{orbits} = \langle \chi_{\mathrm{perm}}, \mathbf{1} \rangle = \frac{1}{|G|}\sum_g \#\mathrm{Fix}(g)$. (2) One-dimensional representations are characters of $G^{\mathrm{ab}} = G/[G,G]$. If you know the abelianization, you have the 1-dimensional irreducibles immediately. (3) To verify irreducibility, compute $\langle \chi, \chi \rangle$; if $= 1$, the representation is irreducible. If $= m$, it decomposes into $m$ irreducible constituents (not necessarily distinct). (4) Twisting by a $1$-dimensional character preserves irreducibility: $V \otimes \mathrm{sgn}$ is irreducible whenever $V$ is, since $|\chi_{\mathrm{sgn}}(g)| = 1$ everywhere implies $\langle \chi_V \cdot \chi_{\mathrm{sgn}}, \chi_V \cdot \chi_{\mathrm{sgn}} \rangle = \langle \chi_V, \chi_V \rangle = 1$. This is how the $V \otimes \mathrm{sgn}$ row in the $S_4$ table arises for free. (5) The *kernel* of a character $\chi$ is $\ker \chi = \{g : \chi(g) = \chi(e)\}$ — this is always a normal subgroup of $G$. The intersection of all irreducible character kernels is $\{e\}$ (since the regular representation is faithful), which means characters collectively detect all group elements. No element of $G$ is invisible to character theory.

---

## What's next

In the next article, we step back to look at the bigger picture: **category theory** provides a universal language for describing the common structures we have encountered throughout this series — groups, rings, modules, representations, and the maps between them.

---

*This is Part 10 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 9 — Modules](/en/abstract-algebra/09-modules/)*

*Next: [Part 11 — Category Theory](/en/abstract-algebra/11-category-theory/)*
