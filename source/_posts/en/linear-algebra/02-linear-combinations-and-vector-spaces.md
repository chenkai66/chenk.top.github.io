---
title: "Linear Combinations and Vector Spaces"
date: 2024-04-02 09:00:00
tags:
  - Linear Algebra
  - Span
  - Basis
  - Dimension
  - Linear Independence
description: "If vectors are building blocks, linear combinations are the blueprint. This chapter develops the five concepts that the rest of linear algebra is built on: span, linear independence, basis, dimension, and subspaces."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 2
  total: 18
lang: en
mathjax: true
---

## Why This Chapter Matters

Open a box of crayons that contains only **red, green, and blue**. How many colors can you draw? The honest answer is **infinitely many** — every shade you have ever seen on a screen is just a different mix of those three. Three "ingredients" produce an entire universe.

That recipe — *take a few vectors, scale them, add them up* — is called a **linear combination**. The whole of linear algebra is built on this one move. Once you understand it deeply, you also understand:

- **span** — every place a set of vectors can reach,
- **linear independence** — when none of the ingredients are wasted,
- **basis** — the *smallest* complete set of ingredients,
- **dimension** — how many independent ingredients a space requires,
- **subspaces** — smaller worlds living inside bigger ones.

These five words are the working vocabulary of linear algebra. Every later chapter — matrices, determinants, eigenvalues, SVD — is a sentence written using them.

### Prerequisites

- Chapter 1: vectors, addition, scalar multiplication, and the geometric picture of $\mathbb{R}^2$ and $\mathbb{R}^3$.

---

## 1. What Is a Linear Combination?

### The recipe

Given vectors $\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_k$ and real numbers $c_1, c_2, \ldots, c_k$, their **linear combination** is

$$
c_1 \vec{v}_1 + c_2 \vec{v}_2 + \cdots + c_k \vec{v}_k.
$$

Two operations, nothing more: **scale** each vector, then **add**. The word *linear* means **no squares, no products of components, no nonlinear functions** — just the two basic operations of a vector space.

### Three everyday pictures

**Mixing cocktails.** Two base spirits sit on the shelf:
- Spirit $\vec{a}=(0.40,\,10)$: 40 % alcohol, 10 g/L sugar
- Spirit $\vec{b}=(0.20,\,30)$: 20 % alcohol, 30 g/L sugar

You want a drink with profile $\vec{t}=(0.30,\,20)$. Solving $x\vec{a}+y\vec{b}=\vec{t}$ gives $x=y=0.5$. The target is the linear combination $0.5\vec{a}+0.5\vec{b}$.

**Walking directions.** "300 m east, then 400 m north." Your displacement is the linear combination $300\,\vec{e}_\text{east}+400\,\vec{e}_\text{north}$.

**Pixels on your screen.** Every pixel is

$$
\text{color}=r\!\begin{pmatrix}255\\0\\0\end{pmatrix}+g\!\begin{pmatrix}0\\255\\0\end{pmatrix}+b\!\begin{pmatrix}0\\0\\255\end{pmatrix}.
$$

Three primary colors, infinitely many results.

### Why the word "linear"?

Take a single nonzero $\vec{v}\in\mathbb{R}^2$. As $c$ sweeps over $\mathbb{R}$, the multiples $c\vec{v}$ trace out a **line** through the origin. That straight line — the geometric shadow of scalar multiplication — is where the word *linear* comes from.

Add a second non-parallel vector $\vec{w}$ and the picture explodes from a line into the *whole plane*: every point in $\mathbb{R}^2$ can be written as $a\vec{v}+b\vec{w}$ for exactly one pair $(a,b)$.

![Linear combination of two vectors and the lattice they generate](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig1_linear_combination.png)

The left panel shows one specific combination $1.5\vec{v}+1.2\vec{w}$ built by the parallelogram rule. The right panel shows what happens when $a$ and $b$ are allowed to roam: the dots tile the entire plane.

---

## 2. Span — Everywhere the Vectors Can Reach

### Definition

The **span** of $\vec{v}_1,\ldots,\vec{v}_k$ is the set of *all* their linear combinations:

$$
\operatorname{span}(\vec{v}_1,\ldots,\vec{v}_k)=\{c_1\vec{v}_1+\cdots+c_k\vec{v}_k\mid c_i\in\mathbb{R}\}.
$$

Imagine each vector as a dial on a remote control. Turn the dials however you like; the set of all positions you can reach is the span.

### A catalogue of shapes

| Vectors | Span |
|---------|------|
| One nonzero vector in $\mathbb{R}^2$ or $\mathbb{R}^3$ | A line through the origin |
| Two parallel vectors | Still just that line — the second one adds no direction |
| Two non-parallel vectors in $\mathbb{R}^2$ | All of $\mathbb{R}^2$ |
| Two non-parallel vectors in $\mathbb{R}^3$ | A plane through the origin |
| Three coplanar vectors in $\mathbb{R}^3$ | Still just that plane |
| Three non-coplanar vectors in $\mathbb{R}^3$ | All of $\mathbb{R}^3$ |

Three structural facts hold no matter what:

- The span **always passes through the origin** — set every $c_i=0$.
- The span is **closed**: combine any two of its points and you stay inside.
- Adding a vector that is already reachable **never enlarges** the span.

![Span: line, repeated line, plane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig2_span_visualization.png)

Left: one vector spans a line. Middle: a parallel partner contributes no new direction, the span is the same line. Right: two truly different directions sweep out the whole plane.

### Practical question — can I mix it?

A lab has three solutions:
- $\vec{A}=(5\%,\,10\%)$ acid/salt
- $\vec{B}=(10\%,\,5\%)$
- $\vec{C}=(2\%,\,2\%)$

Can you produce the target $(15\%,\,12\%)$? Since $\vec{A}$ and $\vec{B}$ are not parallel, $\operatorname{span}(\vec{A},\vec{B})=\mathbb{R}^2$ already. The target is therefore reachable — and adding $\vec{C}$ does not help us do anything new, it just makes the recipe non-unique.

---

## 3. Linear Independence — No Wasted Vectors

### The core idea

Sometimes adding a vector buys you nothing because it was *already* in the span. **Linear independence** says: every vector pulls its own weight.

### Definition

Vectors $\vec{v}_1,\ldots,\vec{v}_k$ are **linearly independent** if the *only* way to write the zero vector as a combination of them is to use all-zero coefficients:

$$
c_1\vec{v}_1+\cdots+c_k\vec{v}_k=\vec{0}\;\;\Longrightarrow\;\; c_1=\cdots=c_k=0.
$$

If some non-trivial combination produces $\vec{0}$, the set is **linearly dependent** — at least one vector is a combination of the others, so it is redundant.

### Geometric translation

- In $\mathbb{R}^2$: two vectors are independent $\iff$ they are not parallel.
- In $\mathbb{R}^3$: three vectors are independent $\iff$ they are not coplanar.
- In $\mathbb{R}^n$: more than $n$ vectors **must** be dependent — there are only $n$ truly different directions to go.

![Independent vs dependent vectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig3_linear_independence.png)

On the right, $\vec{v}_3=0.8\vec{v}_1+0.6\vec{v}_2$. The dashed parallelogram exhibits the dependence — $\vec{v}_3$ adds no new direction, so the three vectors are dependent.

### Three ways to test it

1. **Definition.** Solve $c_1\vec{v}_1+\cdots+c_k\vec{v}_k=\vec{0}$. If the only solution is the trivial one, you have independence.
2. **Determinant** (when you have $n$ vectors in $\mathbb{R}^n$). Stack them as columns of a square matrix and compute $\det$. Nonzero $\Rightarrow$ independent.
3. **Rank.** Stack them as columns and compute the rank. Equal to the number of vectors $\Rightarrow$ independent.

```python
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])      # = 2*v2 - v1, so dependent

A = np.column_stack([v1, v2, v3])
print(np.linalg.matrix_rank(A))   # 2, not 3 -> dependent
```

### Why independence is non-negotiable

If $\{\vec{v}_1,\ldots,\vec{v}_k\}$ is independent, then *every* vector in their span has **exactly one** representation as a combination of them. That uniqueness is what makes coordinates well-defined: the pair $(3,5)$ wouldn't mean anything if there were two different ways to build the same point.

---

## 4. Basis — The Smallest Complete Toolbox

### Definition

A **basis** of a vector space $V$ is a set of vectors that is

1. **linearly independent** (no redundancy), and
2. **spans $V$** (covers every vector in the space).

Remove anything and you lose coverage. Add anything and you gain redundancy. A basis is the *minimal* spanning set, simultaneously the *maximal* independent set.

### The standard basis of $\mathbb{R}^n$

$$
\vec{e}_1=\!\begin{pmatrix}1\\0\\\vdots\\0\end{pmatrix},\;
\vec{e}_2=\!\begin{pmatrix}0\\1\\\vdots\\0\end{pmatrix},\;\ldots,\;
\vec{e}_n=\!\begin{pmatrix}0\\\vdots\\0\\1\end{pmatrix}.
$$

When you write $\vec{v}=(3,5)$, what you really mean is $\vec{v}=3\vec{e}_1+5\vec{e}_2$. The standard basis is so familiar that we forget it is a *choice*.

![Standard basis vs a rotated basis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig4_basis_examples.png)

The same point $\vec{u}=(3,2)$ has coordinates $(3,2)$ in the standard basis on the left, and different coordinates in the rotated basis on the right. The arrow itself never moved; the *grid* under it changed.

### Bases are not unique

Each of these is a perfectly valid basis of $\mathbb{R}^2$:

$$
\left\{\!\begin{pmatrix}1\\0\end{pmatrix},\!\begin{pmatrix}0\\1\end{pmatrix}\!\right\},\quad
\left\{\!\begin{pmatrix}1\\1\end{pmatrix},\!\begin{pmatrix}1\\-1\end{pmatrix}\!\right\},\quad
\left\{\!\begin{pmatrix}2\\0\end{pmatrix},\!\begin{pmatrix}0\\3\end{pmatrix}\!\right\}.
$$

Different bases give different coordinates for the same vector — but the vector (the geometric arrow) is the same in all of them.

### Coordinates depend on the basis

The vector $(3,5)$ in the standard basis becomes $(4,-1)$ in the basis $\{(1,1),(1,-1)\}$, because

$$
4\!\begin{pmatrix}1\\1\end{pmatrix}+(-1)\!\begin{pmatrix}1\\-1\end{pmatrix}=\begin{pmatrix}3\\5\end{pmatrix}.
$$

A "vector" is the geometric object. A "coordinate tuple" is what the vector looks like *after* you commit to a basis. This distinction is one of the most freeing ideas in linear algebra — and it's exactly what change of basis is about.

![The same point, two coordinate grids](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig5_change_of_basis.png)

---

## 5. Dimension — Counting the Degrees of Freedom

### Definition

The **dimension** of $V$, written $\dim(V)$, is the number of vectors in any basis of $V$. A theorem (which we'll take on faith for now) guarantees that *every* basis of $V$ has the same size, so the definition makes sense.

### Three equivalent intuitions

Dimension counts:

- The number of **independent parameters** needed to pinpoint a vector,
- The number of **independent directions** of motion,
- The **maximum** number of linearly independent vectors that fit in the space.

| Space | Dimension | Why |
|-------|-----------|-----|
| $\{\vec{0}\}$ | 0 | Nowhere to go |
| A line through origin | 1 | Forward / backward |
| A plane through origin | 2 | Forward/back + left/right |
| $\mathbb{R}^3$ | 3 | Add up/down |
| $\mathbb{R}^n$ | $n$ | $n$ independent directions |

### The dimension theorem

In an $n$-dimensional space:
- More than $n$ vectors are **always** dependent.
- Exactly $n$ independent vectors form a basis.
- Fewer than $n$ vectors **cannot** span the whole space.

This is why dimension feels like the *capacity* of a space — it is the upper bound on how many independent things can coexist inside it.

---

## 6. Subspaces — Spaces Inside Spaces

### Definition

A **subspace** $W$ of $V$ is a non-empty subset that is itself a vector space. Concretely, $W\subseteq V$ is a subspace iff:

1. $\vec{0}\in W$,
2. $\vec{u},\vec{v}\in W \implies \vec{u}+\vec{v}\in W$ (closed under addition),
3. $\vec{v}\in W,\,c\in\mathbb{R} \implies c\vec{v}\in W$ (closed under scaling).

Conditions 2 and 3 say: *you cannot escape the subspace by adding or scaling.* Condition 1 is automatic if 2 and 3 hold and $W$ is non-empty (set $c=0$), but stating it explicitly avoids subtle edge cases.

### The complete list in $\mathbb{R}^3$

There are only four kinds of subspaces of $\mathbb{R}^3$:

- $\{\vec{0}\}$ — dimension 0,
- any **line through the origin** — dimension 1,
- any **plane through the origin** — dimension 2,
- $\mathbb{R}^3$ itself — dimension 3.

A line or plane that does **not** pass through the origin is **not** a subspace — it fails condition 1, and it is also not closed under addition.

![Subspaces in 3D: line through origin, plane through origin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig6_subspaces_3d.png)

### Span always produces a subspace

For any set of vectors $S=\{\vec{v}_1,\ldots,\vec{v}_k\}$, $\operatorname{span}(S)$ is automatically a subspace. This gives the simplest possible recipe: **pick some vectors, take their span, you have a subspace.** Almost every subspace you ever meet is constructed this way.

### Linear vs affine — why the origin matters

A line that misses the origin is an **affine set**, not a linear subspace. The picture below makes the distinction crisp: on the left, $\vec{u}+\vec{w}$ stays on the line; on the right, $\vec{p}_1+\vec{p}_2$ jumps off the line entirely.

![Subspaces must contain the origin: linear vs affine](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/02-linear-combinations-and-vector-spaces/fig7_affine_vs_linear.png)

This is why affine geometry (translations) is *not* the same as linear algebra (rotations and scalings). Linear maps fix the origin; affine maps do not. Whenever you read "subspace," read "passes through the origin and is closed under +/×."

### Dimension formula for sums

For two subspaces $U,W\subseteq V$:

$$
\dim(U+W)=\dim(U)+\dim(W)-\dim(U\cap W).
$$

It is the inclusion–exclusion principle, ported to vector spaces. We will see this formula again in Chapter 5 when we count solutions of linear systems.

---

## 7. Case Study — RGB as a Vector Space

The RGB color model is the cleanest real-world illustration of everything in this chapter:

- Each color is a 3D vector $\vec{c}=(r,g,b)$.
- The basis is $\{\vec{r},\vec{g},\vec{b}\}$, the three primaries.
- $\dim(\text{RGB})=3$ — three independent channels, three dials.

```python
import numpy as np

red, green, blue = np.eye(3) * 255

yellow = red + green          # [255, 255,   0]
purple = red + blue           # [255,   0, 255]
white  = red + green + blue   # [255, 255, 255]
```

**Grayscale** is a 1D subspace: $\vec{c}=k(1,1,1)$ for $k\in[0,255]$ — one dial, one degree of freedom, lying along the diagonal of the RGB cube.

**Color blindness** (some types) projects RGB onto a 2D subspace: the missing dimension is the one a person cannot distinguish.

**Color-space conversion** (RGB → HSV, LAB, …) is a **change of basis**: same colors, new coordinates.

---

## 8. Common Misconceptions

> **"$\vec{v}_1=(1,2)$ and $\vec{v}_2=(2,4)$ span $\mathbb{R}^2$."**
> No. $\vec{v}_2=2\vec{v}_1$, so they span only the line $y=2x$.

> **"Three vectors always span more than two."**
> Only if the third vector is *outside* the span of the first two.

> **"Independent vectors must be perpendicular."**
> No. $(1,0)$ and $(1,1)$ are independent but not orthogonal. Orthogonality is a *stronger* condition than independence.

> **"A space has a unique basis."**
> Every space has *infinitely many* bases. What is unique is the **dimension** — the *number* of vectors in any basis.

> **"Any subset is a subspace."**
> Subspaces must contain $\vec{0}$ and be closed under $+$ and scalar multiplication. Most subsets fail.

---

## 9. Code Lab

### Is a set linearly independent?

```python
import numpy as np

def is_independent(vectors):
    """Return True iff the given vectors are linearly independent."""
    A = np.column_stack(vectors)
    return np.linalg.matrix_rank(A) == len(vectors)

print(is_independent([np.array([1, 2, 3]),
                      np.array([4, 5, 6]),
                      np.array([7, 8, 9])]))   # False
print(is_independent(list(np.eye(3))))         # True
```

### Is a target vector inside a span?

```python
def in_span(target, vectors, tol=1e-8):
    """Check whether `target` lies in span(vectors)."""
    A = np.column_stack(vectors)
    coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
    return np.allclose(A @ coeffs, target, atol=tol)

v1, v2 = np.array([1, 1]), np.array([2, 2])     # parallel
print(in_span(np.array([1, 0]), [v1, v2]))      # False
print(in_span(np.array([3, 3]), [v1, v2]))      # True
```

### Extract a basis from a redundant set

```python
def extract_basis(vectors):
    """Greedy: keep a vector iff it raises the rank."""
    chosen, M = [], np.zeros((len(vectors[0]), 0))
    for v in vectors:
        Mtest = np.column_stack([M, v])
        if np.linalg.matrix_rank(Mtest) > M.shape[1]:
            chosen.append(v)
            M = Mtest
    return chosen

vs = [np.array([1, 0, 0]),
      np.array([0, 1, 0]),
      np.array([1, 1, 0]),    # = v1 + v2  -> redundant
      np.array([0, 0, 1])]
print(len(extract_basis(vs)))   # 3 -> {v1, v2, v4}
```

---

## 10. Chapter Summary

| Concept | Definition | Picture |
|---------|-----------|---------|
| Linear combination | $c_1\vec{v}_1+\cdots+c_k\vec{v}_k$ | Weighted sum of vectors |
| Span | All linear combinations | Every reachable point |
| Linear independence | Zero combination $\Rightarrow$ all coefficients zero | No redundant arrows |
| Basis | Independent **and** spans the space | Smallest complete toolbox |
| Dimension | Size of any basis | Degrees of freedom |
| Subspace | Closed under $+$ and $\cdot$, contains $\vec{0}$ | Space inside a space |

These six ideas thread through everything that follows:

- **Ch 3** — A matrix's column space is the *span* of its columns.
- **Ch 4** — The *determinant* tests linear independence in a single number.
- **Ch 5** — Solution sets of $A\vec{x}=\vec{0}$ are *subspaces* (the null space).
- **Ch 6** — *Eigenvectors* are special bases that diagonalize a matrix.
- **Ch 9** — *SVD* delivers an "optimal" pair of orthonormal bases.

---

## What Comes Next

**Chapter 3 — Matrices as Linear Transformations.** A matrix is not a passive table of numbers; it is an *agent of transformation*. We will see that:

- multiplying $A\vec{x}$ is geometrically the action of $A$ on $\vec{x}$,
- rotation, scaling, shearing, and projection are all matrices,
- matrix multiplication is exactly **composition of transformations**,
- the column space of $A$ is precisely the span we just defined.

---

## Series Navigation

- **Previous:** [Chapter 1 — The Essence of Vectors](/en/chapter-01-the-essence-of-vectors/)
- **Next:** [Chapter 3 — Matrices as Linear Transformations](/en/chapter-03-matrices-as-linear-transformations/)
- **Series:** Essence of Linear Algebra (2 of 18)
