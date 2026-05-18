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

The answer is *modules*: the natural generalization of vector spaces, where scalars come from a ring rather than a field. The theory is richer, the pathologies more interesting, and — perhaps most importantly — modules turn out to encompass an enormous range of mathematical objects: abelian groups (modules over $\mathbb{Z}$), vector spaces with a linear endomorphism (modules over $K[x]$), ideals (modules over a ring), and group representations (modules over a group ring). What initially feels like a technical generalization is actually a unifying framework that organizes much of algebra.

I want to flag the mental shift up front. Modules force you to give up two pleasant facts about vector spaces — the existence of bases and the splitting of short exact sequences — and the entire structure theory below is dedicated to figuring out *exactly* how badly those things fail and what survives in their place. The good news is that for the most useful base rings (PIDs, especially $\mathbb{Z}$ and $K[x]$), the failures are completely understood: the structure theorem catalogues all finitely generated modules up to isomorphism. That is why this article focuses on PIDs.

---

## Why Modules? Vector Spaces over Non-Fields

Suppose we want to study an abelian group $A$ as an algebraic object. We can add elements, take inverses, and — implicitly — multiply by integers: $3 \cdot a = a + a + a$, $(-2) \cdot a = -(a + a)$, and so on. This makes $A$ into a structure where $\mathbb{Z}$ acts as scalars. But $\mathbb{Z}$ is not a field; it has nonzero elements without inverses, like 2 or 3.

![Modules vs vector spaces: scalars from a ring](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_module_vs_vector.png)


The catch is that $\mathbb{Z}$-multiplication is no longer reversible: doubling an element does not generally give a doubling-inverse. This single failure of "scalar division" is what turns the simple theory of vector spaces into the messier theory of modules. Reversibility was the secret ingredient that made bases work, and we lose it the moment we leave the field setting.

So a $\mathbb{Z}$-module is "an abelian group with the natural $\mathbb{Z}$-action," and conversely, every $\mathbb{Z}$-module is an abelian group, with the action by integers being forced. This already shows that modules are not a marginal generalization: every abelian group you have ever seen — $\mathbb{Z}$, $\mathbb{Z}/n\mathbb{Z}$, $\mathbb{Q}/\mathbb{Z}$, the additive group of $\mathbb{R}$ — is a $\mathbb{Z}$-module.

A second motivating example. Take a vector space $V$ over a field $K$ together with a linear endomorphism $T : V \to V$. We can act on $V$ by polynomials in $T$: $(2T^3 - 5T + 1)(v) := 2T^3(v) - 5T(v) + v$. This makes $V$ into a $K[x]$-module, where $x$ acts as $T$. The structure theory of $K[x]$-modules turns out to give exactly the theory of canonical forms — Jordan normal form, rational canonical form — that you saw in linear algebra. This is one of those moments where module theory does not just generalize linear algebra; it explains what linear algebra was secretly doing.

This second example deserves a moment of reflection. In an undergraduate linear algebra class, Jordan form is presented as a sequence of clever observations about generalized eigenvectors and chains. The proofs are technical and the steps look ad hoc. The module-theoretic perspective makes the whole thing transparent: a vector space with a linear operator is a $K[x]$-module, and the structure theorem says every finitely generated $K[x]$-module decomposes as a direct sum of cyclic ones. Each cyclic summand $K[x]/((x - \lambda)^e)$ is a Jordan block. End of story. Six pages of clever bookkeeping in the linear algebra book reduce to a single corollary of the structure theorem.

A third example: every ideal $I$ of a ring $R$ is an $R$-submodule of $R$ itself. Module theory thus subsumes the theory of ideals. And every quotient $R/I$ is an $R$-module too. The list goes on.

**Why this matters.** Modules are the universal language for "objects on which a ring acts." If you can describe your structure as "an abelian group together with a ring's worth of linear operators," you are looking at a module, and you can use module theory. This single shift in viewpoint subsumes linear algebra, abelian group theory, ideal theory, representation theory, and homological algebra.

A small piece of dictionary that is useful to keep handy:

| Module-theoretic concept | Specialization to $R = K$ (field) | Specialization to $R = \mathbb{Z}$ |
|---|---|---|
| $R$-module | vector space | abelian group |
| Submodule | subspace | subgroup |
| Quotient module | quotient space | quotient group |
| Module homomorphism | linear map | group homomorphism |
| Free module of rank $n$ | $K^n$ | $\mathbb{Z}^n$ |
| Cyclic module | one-dimensional | cyclic group |
| Torsion submodule | always 0 | torsion subgroup |
| Direct sum | direct sum | direct sum |

The thing to notice is that the column for $\mathbb{Z}$ has a non-trivial "torsion submodule" entry. Over a field, every module is torsion-free; over $\mathbb{Z}$, torsion is the entire reason we need a structure theorem at all. Torsion is the obstruction to being a vector space.

---

## Definitions and First Examples

**Definition.** Let $R$ be a ring (with 1). A *(left) $R$-module* is an abelian group $(M, +)$ together with a scalar multiplication $R \times M \to M$, $(r, m) \mapsto rm$, satisfying:

1. $(r + s)m = rm + sm$ (distributes over scalar addition)
2. $r(m + n) = rm + rn$ (distributes over module addition)
3. $(rs)m = r(sm)$ (associativity)
4. $1 \cdot m = m$ (identity acts trivially)

If $R$ is non-commutative, we distinguish *left* modules (defined as above) from *right* modules (with multiplication $M \times R \to M$). For commutative $R$, the distinction is unnecessary. Most rings in this article are commutative; we work with left modules unless stated otherwise.

When $R$ is a field, an $R$-module is precisely a vector space. Modules generalize vector spaces by allowing the scalars to come from a ring instead of a field.

![Examples of modules: Z-modules, F[x]-modules, vector spaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_1_module_examples.png)

**Examples.**

1. *$\mathbb{Z}$-modules = abelian groups.* The $\mathbb{Z}$-action on an abelian group $A$ is forced: $n \cdot a$ for $n > 0$ is $a + \cdots + a$ ($n$ times); for $n < 0$ it is $-((-n)a)$; and $0 \cdot a = 0$. Both module axioms (distributivity, associativity) follow from the abelian group structure.

2. *Vector spaces over a field $K$ are $K$-modules.* All the vector space axioms are special cases of the module axioms.

3. *$R$ as an $R$-module over itself.* The ring $R$ acts on itself by left multiplication. Submodules of $R$ are exactly the *left ideals* of $R$.

4. *$R^n$ as a free module.* The direct sum $R^n = R \oplus \cdots \oplus R$ ($n$ copies) is an $R$-module under componentwise scalar multiplication. We will see that $R^n$ is the prototype of a "free module."

5. *$K[x]$-modules = vector spaces with a linear operator.* Given a $K$-vector space $V$ and a linear map $T : V \to V$, we make $V$ into a $K[x]$-module by defining $f(x) \cdot v := f(T)(v)$ for $f \in K[x]$. Conversely, any $K[x]$-module is a $K$-vector space (restrict scalars to $K \subset K[x]$), and the action of $x$ defines a linear operator. So *$K[x]$-modules are the same as pairs $(V, T)$*.

6. *Group rings and representations.* For a group $G$ and a field $K$, the group ring $K[G]$ has elements $\sum_{g \in G} a_g g$ with multiplication extending that of $G$. A $K[G]$-module is the same as a $K$-vector space $V$ with a linear $G$-action — that is, a *representation* of $G$. We will study these in Part 10.

7. *Quotient modules.* If $N \subseteq M$ is a submodule, the quotient $M/N$ inherits an $R$-module structure: $r \cdot (m + N) := rm + N$.

**Definition.** A *submodule* of $M$ is a subgroup $N \subseteq M$ closed under scalar multiplication: $r \in R, n \in N \Rightarrow rn \in N$. A map $\varphi : M \to N$ is an *$R$-module homomorphism* if it is additive and $R$-linear: $\varphi(rm) = r\varphi(m)$.

The kernel and image of an $R$-module homomorphism are themselves submodules (of the source and target respectively). The first isomorphism theorem then gives $M / \ker\varphi \cong \mathrm{im}\,\varphi$ as $R$-modules. This and the next two iso theorems are the bread-and-butter computational tools.

**Why this matters.** The flexibility of "ring of scalars" is what makes module theory ubiquitous. The same theorem about modules can specialize to a theorem about abelian groups, vector spaces, ideals, or representations, just by choosing $R$ appropriately.

---

## Submodules, Quotients, and Homomorphisms

The basic structural theorems for modules parallel those for groups and rings. Most of the proofs you saw in Parts 1-3 carry over verbatim, with "linear" replacing "homomorphism."

![Quotient modules and submodules](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_quotient_module.png)


**Isomorphism Theorems.**

1. *(First)* If $\varphi : M \to N$ is an $R$-module homomorphism, then $M/\ker\varphi \cong \mathrm{im}\,\varphi$.
2. *(Second)* If $A, B \subseteq M$ are submodules, then $(A + B)/B \cong A/(A \cap B)$.
3. *(Third)* If $A \subseteq B \subseteq M$ are submodules, then $(M/A)/(B/A) \cong M/B$.

These are proved exactly as for groups, with the additional verification that the isomorphism respects scalar multiplication — automatic, since the homomorphisms in question are $R$-linear by construction.

**Direct sums.** Given modules $M_1, \ldots, M_n$, their direct sum $M_1 \oplus \cdots \oplus M_n$ is the Cartesian product with componentwise addition and scalar multiplication. A finite direct sum of modules is itself a module.

**Generators.** A subset $S \subseteq M$ *generates* $M$ if every element of $M$ is a finite $R$-linear combination of elements of $S$. $M$ is *finitely generated* if it has a finite generating set; *cyclic* if it is generated by a single element.

**Examples.**

- $\mathbb{Z}$ is a cyclic $\mathbb{Z}$-module, generated by 1.
- $\mathbb{Z}/n\mathbb{Z}$ is a cyclic $\mathbb{Z}$-module for any $n \geq 1$.
- $\mathbb{Z}^2 = \mathbb{Z} \oplus \mathbb{Z}$ is finitely generated (by $(1,0), (0,1)$) but not cyclic.
- $\mathbb{Q}$ is *not* finitely generated as a $\mathbb{Z}$-module: any finite collection of fractions has a common denominator $d$, but $\mathbb{Z}\langle a_1/d, \ldots, a_k/d \rangle \subseteq \frac{1}{d}\mathbb{Z}$ does not contain $1/(2d)$.

The fact that $\mathbb{Q}$ is not finitely generated over $\mathbb{Z}$ is striking: an abelian group can be "huge" without being all that complicated, and module theory needs to track this distinction carefully.

**Why this matters.** The "finitely generated" hypothesis is what we will need for the structure theorem. Without it, classification is essentially hopeless; the category of all $\mathbb{Z}$-modules contains all abelian groups, including beasts like $\mathbb{R}$ and $\mathbb{Q}/\mathbb{Z}$ that resist classification.

Two more pieces of vocabulary that show up constantly:

- A module $M$ is *cyclic* if it is generated by a single element, equivalently $M \cong R/I$ for some ideal $I$. The annihilator of $M$ is the ideal $I$.
- A module $M$ is *simple* (or *irreducible*) if it has no submodules other than 0 and $M$. Over a field, simple is the same as one-dimensional. Over $\mathbb{Z}$, simple is the same as $\mathbb{Z}/p\mathbb{Z}$ for $p$ prime.
- A module $M$ is *Noetherian* if every increasing chain of submodules stabilizes; *Artinian* if every decreasing chain does. Over a Noetherian ring, finitely generated modules are Noetherian; this is the abstract reason "finitely generated" is the right finiteness condition.

Tucking these definitions away pays off later: in homological algebra you talk constantly about "simple modules," "indecomposable modules," and the like, and they are all just slight variations on the basic concepts above.

---

## Free Modules and Bases

In linear algebra, every vector space has a basis. For modules, this fails, and the failure is informative.

![Free modules: basis elements generate everything](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_free_module.png)


**Definition.** An $R$-module $M$ is *free* on a set $S$ if every element of $M$ is uniquely a finite $R$-linear combination of elements of $S$. Equivalently, $M \cong R^{|S|}$ via $S \to R^{|S|}$, $s_i \mapsto e_i$.

![A free module R^n with its standard basis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_2_free_module.png)

**Examples of free modules.**

- Every vector space (free over its field).
- $\mathbb{Z}^n$ as a $\mathbb{Z}$-module.
- $R[x]$ as an $R$-module, with basis $\{1, x, x^2, \ldots\}$.

**Examples of non-free modules.**

- $\mathbb{Z}/n\mathbb{Z}$ as a $\mathbb{Z}$-module for $n \geq 2$. There is no basis: any nonzero element $\overline{a}$ satisfies $n\overline{a} = 0$, breaking the uniqueness of representations.
- $\mathbb{Q}$ as a $\mathbb{Z}$-module. Any two elements $a/b, c/d \in \mathbb{Q}$ satisfy $(bc)(a/b) - (ad)(c/d) \cdot 0 = 0$, but actually any two rationals satisfy a $\mathbb{Z}$-linear relation: $(bc) \cdot (a/b) = (ad) \cdot (c/d)$ if $a/b, c/d$ are both nonzero — wait, that is wrong unless $a/b = c/d$. Let me redo. Actually, take $1, 1/2 \in \mathbb{Q}$: $1 \cdot 1 + (-2) \cdot (1/2) = 0$, a nontrivial $\mathbb{Z}$-relation.

The point is that any singleton $\{r\}$ in $\mathbb{Q}$ generates only $\mathbb{Z}r$, so $\mathbb{Q}$ is not generated by any finite set, and even an infinite generating set has nontrivial $\mathbb{Z}$-relations among any two of its elements.

**Theorem.** Over a commutative ring $R$, any two bases of a free module have the same cardinality. This *invariant* is the *rank* of the module.

For non-commutative rings this can fail (the IBN — invariant basis number — property is not automatic), but every commutative ring (and every left/right Noetherian ring) satisfies it. The proof for fields is dimension-counting; for general commutative rings, you reduce mod a maximal ideal $\mathfrak{m}$ to get a vector space over $R/\mathfrak{m}$ where dimension counting works, then lift back.

A pleasant consequence: rank is a well-defined invariant of free modules, just like dimension is for vector spaces. So you can talk about "the rank of a free $\mathbb{Z}$-module" without ambiguity, and the rank tells you how big the module is up to isomorphism.

**Theorem (Submodules of free modules over PIDs).** If $R$ is a PID and $F$ is a free $R$-module of rank $n$, then every submodule of $F$ is free of rank $\leq n$.

This is genuinely special to PIDs. Over $\mathbb{Z}[x]$, for example, the ideal $(2, x)$ is a submodule of the free module $\mathbb{Z}[x]$ but is not itself free — and in fact $(2, x)$ requires two generators with a nontrivial relation, which is the definition of "not free."

The proof uses induction on $n$ together with the key step that any submodule of $R$ (a PID) is a principal ideal $(d)$, hence free of rank 0 or 1. So the rank-1 case is the engine. For higher ranks, you peel off one coordinate at a time and induct.

A concrete consequence: every subgroup of $\mathbb{Z}^n$ is free abelian of rank at most $n$. This means lattice subgroups of $\mathbb{Z}^n$ are themselves lattices — a fact familiar from elementary number theory but here derived from a more general principle.

**Why this matters.** Free modules play the role of "vector-space-like" modules. They are the building blocks; non-free modules will be expressed as quotients of free modules. The structure theorem for modules over a PID can be paraphrased as: "every finitely generated module is a quotient of a free module by a particularly clean kind of submodule."

There is a useful universal property characterization. A free $R$-module on a set $S$ is the unique (up to isomorphism) module $F$ together with a map $S \to F$ such that for any $R$-module $M$ and any function $S \to M$, the function extends uniquely to an $R$-linear map $F \to M$. In other words, "free module on $S$" is "the thing where you can turn any function out of $S$ into a module homomorphism, with no obstructions." This is exactly the same universal property that defines free groups, free monoids, free vector spaces, free abelian groups, and free anything-else-in-a-nice-category. The pattern is so common it gets its own name in category theory: free objects.

A pleasant consequence: every $R$-module $M$ is a quotient of a free $R$-module. To see this, take $S = M$ as a set, form the free module $F = R^{(M)}$, and use the universal property to extend the identity map $M \to M$ to a surjection $F \to M$. This is the *free presentation*, and it is the starting point of homological algebra.

---

## The Structure Theorem for Finitely Generated Modules over PIDs

We now state the central classification result of module theory.

![Structure theorem for finitely generated modules over PID](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_structure_theorem.png)


**Theorem (Structure of finitely generated modules over a PID).** Let $R$ be a PID and $M$ a finitely generated $R$-module. Then there exist a non-negative integer $r$ and elements $d_1, d_2, \ldots, d_k \in R$ (non-zero, non-unit) with $d_1 \mid d_2 \mid \cdots \mid d_k$ such that
$$M \cong R^r \oplus R/(d_1) \oplus R/(d_2) \oplus \cdots \oplus R/(d_k).$$
The integer $r$ (the *free rank*) and the ideals $(d_1), \ldots, (d_k)$ (the *invariant factors*) are uniquely determined by $M$.

![Structure theorem: finitely generated module over a PID decomposes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_4_structure_thm.png)

Equivalently, by the Chinese Remainder Theorem, we can decompose each $R/(d_i)$ further into prime-power pieces:
$$M \cong R^r \oplus R/(p_1^{e_1}) \oplus \cdots \oplus R/(p_m^{e_m})$$
where the $p_j$ are (not necessarily distinct) primes in $R$. These prime-power pieces are the *elementary divisors* of $M$.

**Definition.** The *torsion submodule* of $M$ is
$$M_{\mathrm{tor}} = \{m \in M : rm = 0 \text{ for some nonzero } r \in R\}.$$
Over a PID, $M_{\mathrm{tor}} = R/(d_1) \oplus \cdots \oplus R/(d_k)$ in the decomposition above, and $M/M_{\mathrm{tor}} \cong R^r$ is *torsion-free*.

For a general ring $R$, $M_{\mathrm{tor}}$ might not even be a submodule (the closure under addition can fail when there are zero-divisors in $R$). But over an integral domain, $M_{\mathrm{tor}}$ is always a submodule, and over a PID, the structure theorem makes it a finite direct sum of cyclic modules. Over a Dedekind domain (a step beyond PID), the same finite direct sum structure holds for the torsion part, with the torsion-free part being projective rather than free.

![Torsion in Z/nZ as a Z-module](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_3_torsion.png)

So the structure theorem says: every finitely generated module over a PID splits cleanly into a free part plus a torsion part, and the torsion part is a direct sum of cyclic modules with "nested" annihilators.

**Proof sketch.** Pick a finite generating set $\{m_1, \ldots, m_n\}$ for $M$, giving a surjection $\pi : R^n \twoheadrightarrow M$. The kernel $K = \ker\pi$ is a submodule of the free module $R^n$, hence free of some rank $\leq n$ (by the previous theorem). So $M \cong R^n / K$ where $K$ is itself free.

The submodule $K \subseteq R^n$ is described by a matrix $A$ whose columns are a basis of $K$ written in the standard basis of $R^n$. Different choices of bases for $R^n$ and $K$ change $A$ by left- and right-multiplication by invertible matrices. Smith normal form says we can choose bases so that $A$ is diagonal:
$$A = \mathrm{diag}(d_1, d_2, \ldots, d_k, 0, \ldots, 0)$$
with $d_1 \mid d_2 \mid \cdots \mid d_k$. Then $M \cong R^n / K \cong R/(d_1) \oplus \cdots \oplus R/(d_k) \oplus R^{n-k}$, and $r = n - k$. $\square$

![Smith normal form computation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_smith_normal.png)


![Smith normal form algorithm reducing a matrix over a PID](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_5_smith_normal.png)

The proof reveals that Smith normal form — a constructive algorithm for reducing a matrix over a PID to diagonal form using row and column operations — is the computational heart of the structure theorem. Given an explicit presentation of $M$, you can run Smith normal form on a presentation matrix and read off the invariant factors.

**Why this matters.** This is one of the most powerful classification results in algebra. From a single theorem we will derive the classification of finitely generated abelian groups, the existence of Jordan and rational canonical forms for linear operators, and a host of structure results in algebraic number theory. The unifying mechanism is that "different choices of $R$ give different theorems."

A small running example to ground the proof. Take $R = \mathbb{Z}$ and $M = \mathbb{Z}^2 / \langle (4, 6), (6, 9) \rangle$. We need to compute Smith normal form of the matrix
$$A = \begin{pmatrix} 4 & 6 \\ 6 & 9 \end{pmatrix}.$$
Row-reduce: $R_2 \to R_2 - R_1$ gives $\begin{pmatrix} 4 & 6 \\ 2 & 3 \end{pmatrix}$. Swap rows: $\begin{pmatrix} 2 & 3 \\ 4 & 6 \end{pmatrix}$. $R_2 \to R_2 - 2R_1$: $\begin{pmatrix} 2 & 3 \\ 0 & 0 \end{pmatrix}$. $C_2 \to C_2 - C_1$: $\begin{pmatrix} 2 & 1 \\ 0 & 0 \end{pmatrix}$. Swap columns and negate to get $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$. $C_2 \to C_2 - 2 C_1$: $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$. Smith form: $\mathrm{diag}(1, 0)$. So $M \cong \mathbb{Z}/1 \oplus \mathbb{Z} \cong \mathbb{Z}$. Sanity check: $(4, 6) = 2(2, 3)$ and $(6, 9) = 3(2, 3)$, so the submodule $\langle (4, 6), (6, 9) \rangle = \mathbb{Z} \cdot (2, 3)$ has rank 1, and the quotient is $\mathbb{Z}^2 / \mathbb{Z}(2, 3) \cong \mathbb{Z}$. Matches.

---

## Applications: Abelian Groups and Canonical Forms

### Application 1: Finitely Generated Abelian Groups

Apply the structure theorem with $R = \mathbb{Z}$. Every finitely generated abelian group is isomorphic to
$$A \cong \mathbb{Z}^r \oplus \mathbb{Z}/d_1\mathbb{Z} \oplus \cdots \oplus \mathbb{Z}/d_k\mathbb{Z}$$
with $d_1 \mid \cdots \mid d_k$. The rank $r$ and the invariant factors $d_i$ uniquely determine $A$ up to isomorphism.

The pre-structure-theorem version of this classification was a hard-won result of 19th-century mathematics, proved piecewise for groups of small order and then patched together. The structure theorem proves it in one stroke. It also lets you handle questions like "does $A$ have an element of order 6?" (yes iff some invariant factor is divisible by 6) without enumerating the elements.

**Example.** Classify abelian groups of order 360.

$360 = 2^3 \cdot 3^2 \cdot 5$. The torsion-free part $\mathbb{Z}^r$ contributes nothing to the order, so $r = 0$. We need to write 360 as a product of invariant factors $d_1 \mid d_2 \mid \cdots \mid d_k$.

Equivalently (by CRT), we count partitions of the multiset $\{2,2,2,3,3,5\}$ — that is, pick a partition of the exponent of each prime separately:

- 2-part: partitions of 3, namely $\{3\}, \{2,1\}, \{1,1,1\}$ — 3 options.
- 3-part: partitions of 2, namely $\{2\}, \{1,1\}$ — 2 options.
- 5-part: partition of 1, namely $\{1\}$ — 1 option.

Total: $3 \times 2 \times 1 = 6$ abelian groups of order 360. Listing them via elementary divisors: $\mathbb{Z}/8 \oplus \mathbb{Z}/9 \oplus \mathbb{Z}/5$, $\mathbb{Z}/8 \oplus (\mathbb{Z}/3)^2 \oplus \mathbb{Z}/5$, $\mathbb{Z}/4 \oplus \mathbb{Z}/2 \oplus \mathbb{Z}/9 \oplus \mathbb{Z}/5$, $\mathbb{Z}/4 \oplus \mathbb{Z}/2 \oplus (\mathbb{Z}/3)^2 \oplus \mathbb{Z}/5$, $(\mathbb{Z}/2)^3 \oplus \mathbb{Z}/9 \oplus \mathbb{Z}/5$, $(\mathbb{Z}/2)^3 \oplus (\mathbb{Z}/3)^2 \oplus \mathbb{Z}/5$.

The structure theorem turns "classify abelian groups of order $n$" into a partition-counting exercise. That is the kind of leverage we keep getting.

The same partition-counting also gives the asymptotic growth rate: the number of abelian groups of order $n$ is multiplicative in $n$, equal to $\prod_p P(e_p)$ where $n = \prod p^{e_p}$ and $P$ is the integer partition function. For $n$ a prime, that count is 1 (only $\mathbb{Z}/p$); for $n = p^2$, it is 2 ($\mathbb{Z}/p^2$ or $(\mathbb{Z}/p)^2$); the count grows quickly with the exponents but stays small for square-free $n$.

### Application 2: Canonical Forms for Linear Operators

Take $R = K[x]$ and apply the structure theorem to the $K[x]$-module $V$ defined by a linear operator $T$ on a finite-dimensional $K$-vector space.

Since $V$ is finite-dimensional over $K$, it is a torsion $K[x]$-module (every vector satisfies the characteristic polynomial of $T$). So the structure theorem gives
$$V \cong K[x]/(p_1) \oplus \cdots \oplus K[x]/(p_k)$$
with $p_1 \mid \cdots \mid p_k$ in $K[x]$. The polynomials $p_i$ are the *invariant factors* of $T$. The largest, $p_k$, is the *minimal polynomial* of $T$; the product $p_1 \cdots p_k$ is the *characteristic polynomial*.

Choosing bases of each cyclic summand $K[x]/(p_i)$ that look like $\{1, x, x^2, \ldots, x^{\deg p_i - 1}\}$, the matrix of $T$ becomes block-diagonal with each block the *companion matrix* of $p_i$. This is the *rational canonical form*.

Refining further: $K[x]/(p_i) \cong \bigoplus_j K[x]/(q_{ij}^{e_{ij}})$ by CRT, where the $q_{ij}$ are irreducible factors of $p_i$. This is the *primary decomposition*.

When $K$ is algebraically closed (e.g., $K = \mathbb{C}$), every irreducible $q_{ij}$ has the form $x - \lambda_{ij}$. The submodule $K[x]/(x - \lambda)^e$ corresponds to a *Jordan block* of size $e$ with eigenvalue $\lambda$. This recovers the *Jordan normal form*.

![Jordan normal form arising from F[x]-module structure](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_6_jordan.png)

So Jordan normal form is *not* a deep linear-algebra fact; it is the structure theorem for $K[x]$-modules specialized to algebraically closed $K$. This is, to my taste, the single most satisfying recovery of a "classical" theorem from a structural one.

A small concrete example. Let $T : \mathbb{C}^4 \to \mathbb{C}^4$ have characteristic polynomial $(x-2)^3 (x-5)$ and minimal polynomial $(x-2)^2 (x-5)$. The $\mathbb{C}[x]$-module structure on $\mathbb{C}^4$ has invariant factors that must satisfy: their product is $(x-2)^3(x-5)$ (the char poly), the largest is $(x-2)^2(x-5)$ (the min poly), and they form a divisibility chain. The only choice is $p_1 = (x-2)$, $p_2 = (x-2)^2(x-5)$. Decomposing into prime powers via CRT: $\mathbb{C}[x]/(x-2) \oplus \mathbb{C}[x]/(x-2)^2 \oplus \mathbb{C}[x]/(x-5)$. Jordan form: a single $1 \times 1$ block for eigenvalue 2, a single $2 \times 2$ Jordan block for eigenvalue 2, and a single $1 \times 1$ block for eigenvalue 5. Total dimension $1 + 2 + 1 = 4$, consistent.

**Why this matters.** The same theorem in module theory yields fundamental results in two seemingly unrelated areas: classification of abelian groups (number theory, group theory) and canonical forms for matrices (linear algebra, differential equations, dynamical systems). The unification is not a coincidence — it reflects the deeper truth that abelian groups and "vector-spaces-with-an-endomorphism" are both modules over a PID.

Two more applications worth knowing about, which I won't develop in detail:

- *Algebraic number theory.* The ring of integers $\mathcal{O}_K$ in a number field is a Dedekind domain, not always a PID, but every fractional ideal $I \subset K$ becomes a finitely generated $\mathcal{O}_K$-module, and the structure of these modules controls the class group. The structure theorem fails over Dedekind domains in the sense that "free + torsion" is not enough, but the failure is exactly measured by the class group, which classifies projective $\mathcal{O}_K$-modules of rank 1 up to isomorphism.
- *Linear systems and signal processing.* A linear time-invariant system has state space $V$ and a state-update operator $T : V \to V$. The $K[x]$-module structure on $V$ encodes everything about the system's response. The minimal polynomial of $T$ tells you how long an input takes to "forget"; the Jordan form tells you which modes are coupled. Engineering courses on control theory often re-derive this material from scratch; the structure theorem makes it a one-liner.

---

## Exact Sequences

We close with a notational tool that becomes essential in homological algebra.

![Exact sequences of modules](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_exact_sequence.png)


**Definition.** A sequence of $R$-modules and $R$-module homomorphisms
$$\cdots \to M_{i-1} \xrightarrow{f_{i-1}} M_i \xrightarrow{f_i} M_{i+1} \to \cdots$$
is *exact at $M_i$* if $\mathrm{im}\,f_{i-1} = \ker f_i$. The sequence is *exact* if it is exact at every $M_i$.

A *short exact sequence* is one of the form
$$0 \to A \xrightarrow{f} B \xrightarrow{g} C \to 0,$$
which is equivalent to: $f$ is injective, $g$ is surjective, and $\mathrm{im}\,f = \ker g$. By the first isomorphism theorem, $C \cong B/f(A)$. So a short exact sequence packages the data "$B$ has a submodule isomorphic to $A$ with quotient isomorphic to $C$."

![A short exact sequence and its splitting condition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/09-modules/aa_v2_09_7_exact_seq.png)

**Definition.** A short exact sequence *splits* if there is a homomorphism $s : C \to B$ with $g \circ s = \mathrm{id}_C$. Equivalently, $B \cong A \oplus C$.

**Splitting Lemma.** A short exact sequence $0 \to A \to B \to C \to 0$ splits iff there is a retraction $B \to A$ extending the identity on $A$, iff there is a section $C \to B$ as above.

For *vector spaces*, every short exact sequence splits (just lift any basis of $C$). For modules, splitting is a real condition. The simplest non-splitting example: $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \to \mathbb{Z}/2\mathbb{Z} \to 0$. If this split, $\mathbb{Z} \cong \mathbb{Z} \oplus \mathbb{Z}/2$, but the right side has 2-torsion (the element $(0, 1)$) and $\mathbb{Z}$ does not.

The failure of splitting is exactly what the $\mathrm{Ext}$ functor measures in homological algebra: $\mathrm{Ext}^1_R(C, A)$ classifies short exact sequences from $A$ to $C$ up to equivalence, and the trivial element corresponds to the split sequence. So the entire subject of homological algebra can be motivated as "track exactly how short exact sequences fail to split."

A more dramatic non-splitting example. Consider the $\mathbb{Z}$-module $M = \mathbb{Z}/4$ with submodule $A = 2\mathbb{Z}/4 \cong \mathbb{Z}/2$ and quotient $C = M/A \cong \mathbb{Z}/2$. The exact sequence $0 \to \mathbb{Z}/2 \to \mathbb{Z}/4 \to \mathbb{Z}/2 \to 0$ does not split, because $\mathbb{Z}/4$ has an element of order 4 while $\mathbb{Z}/2 \oplus \mathbb{Z}/2$ does not. This is the prototypical example of a *non-trivial group extension*, and it lives in $\mathrm{Ext}^1_\mathbb{Z}(\mathbb{Z}/2, \mathbb{Z}/2) \cong \mathbb{Z}/2$.

**Why this matters.** Exact sequences become the standard language for stating module-theoretic results in modern algebra. They generalize the isomorphism theorems and provide a clean framework for tracking how submodules and quotients fit together. Once you start writing proofs in terms of exact sequences, you stop writing element-by-element verifications and start writing diagram chases — which is faster, less error-prone, and (eventually) more illuminating.

Three exact-sequence facts worth memorizing:

- *Snake lemma.* Given a commutative diagram with exact rows, you get a connecting morphism that splices the kernel sequence and the cokernel sequence into a single long exact sequence. This is the workhorse of homological algebra.
- *Five lemma.* In a commutative diagram of two rows of five terms each, if the four outer vertical maps are isomorphisms, so is the middle one. This often lets you transport a result from a known case to a more general one.
- *Hom is left exact.* The functor $\mathrm{Hom}_R(-, M)$ converts short exact sequences $0 \to A \to B \to C \to 0$ into left-exact sequences $0 \to \mathrm{Hom}(C, M) \to \mathrm{Hom}(B, M) \to \mathrm{Hom}(A, M)$, but generally not into short exact sequences. The failure to extend on the right is what $\mathrm{Ext}^1$ measures.

You do not need any of this for the structure theorem; the structure theorem stands on its own. But the moment you go beyond PIDs — to commutative Noetherian rings, to non-commutative algebras, to representations — exact sequences are the language you use, and they are more useful than direct-sum decompositions in that broader context. The structure theorem might be thought of as the high point of "module theory by direct sums"; everything past it is module theory by exact sequences.

---

## What's Next

Modules unify a vast range of algebraic structures and provide the language for modern algebra. The structure theorem over PIDs is just the first taste; over more general rings, classification is harder, but the framework remains the right one.

![Animation: torsion elements in a module](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/figures/09_torsion.gif)


In the next article, we focus on a special case of module theory that has a life of its own: *representation theory*. A representation of a group $G$ over a field $K$ is a $K[G]$-module — but the group structure imposes extra rigidity, leading to clean decomposition theorems (Maschke's theorem) and powerful invariants (characters). We will see how representations bring linear algebra to bear on group theory, and why characters are one of the most useful gadgets in algebra.

Some forward references for context. Group representations split (Maschke, Part 10) — meaning every representation is a direct sum of irreducibles — provided the characteristic of $K$ does not divide $|G|$. So in characteristic 0 (or "good" characteristic), representations of finite groups behave like vector spaces with no torsion, and classification is easy. In "bad" characteristic, representations have non-split exact sequences and start looking like the messier modules we have seen in this article. The two regimes are called *ordinary* and *modular* representation theory respectively, and they have very different flavors. We focus on the ordinary case.

In Part 11 we go up another level of abstraction to category theory, where modules become an example rather than the subject. By Part 12 we will see modules show up in cryptography and physics, providing the algebraic substrate for both abstract and applied results. The PID structure theorem is, in a real sense, the technical engine that drives all of this.

---
