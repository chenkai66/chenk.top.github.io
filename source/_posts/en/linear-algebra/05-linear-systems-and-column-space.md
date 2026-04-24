---
title: "Linear Systems and Column Space"
date: 2024-04-05 09:00:00
tags:
  - Linear Algebra
  - Column Space
  - Null Space
  - Rank
  - Gaussian Elimination
description: "When does Ax = b have a solution? How many? The honest answer is geometric: it depends on whether b lives inside the column space of A, and on how much of the input space A crushes to zero. This chapter weaves Gaussian elimination, column space, null space, rank, and the rank-nullity theorem into one structural picture."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 5
  total: 18
lang: en
mathjax: true
disableNunjucks: true
---

## The Central Question

Almost everything in applied mathematics eventually lands on the same question:

> Given a matrix $A$ and a vector $\vec{b}$, does the equation $A\vec{x} = \vec{b}$ have a solution? If so, how many?

The mechanical answer is "row-reduce and look." The *structural* answer is far more interesting -- and it is the goal of this chapter. Three geometric objects tell you everything:

- **Column space** $C(A)$ -- the set of vectors $A$ can reach. It decides **whether** a solution exists.
- **Null space** $N(A)$ -- the set of vectors $A$ crushes to zero. It decides **how many** solutions exist.
- **Rank** $r$ -- the dimension of the column space. It quantifies how much information $A$ preserves.

Once these three are clear, every linear-systems result -- existence, uniqueness, least squares, the four fundamental subspaces -- becomes the same story told from different angles.

### What You Will Learn

- Two complementary perspectives on $A\vec{x}=\vec{b}$: rows (intersecting hyperplanes) vs. columns (linear combinations)
- Gaussian elimination as the operational tool, and as the LU decomposition in disguise
- Column space, null space, and rank, with their geometric meaning
- The rank-nullity theorem and the four fundamental subspaces
- How to read off the structure of any solution set at a glance

### Prerequisites

- Chapter 2: span, linear independence, basis
- Chapter 3: matrices as linear transformations
- Chapter 4: determinants and invertibility

---

## Two Ways to See $A\vec{x} = \vec{b}$

### Row Perspective: Intersecting Hyperplanes

Consider the system

$$
\begin{cases} x + 2y = 5 \\ 3x - y = 1 \end{cases}
$$

Each equation describes a **line** in the plane. A solution is a point that lies on **both** lines simultaneously -- their intersection $(1, 2)$. In three variables, each equation describes a plane and the solution set is the intersection of those planes (a point, a line, a plane, or nothing).

This is the picture most students meet first. It is geometric and concrete, but it hides what really matters: **the structure of the matrix itself**.

### Column Perspective: Combining Vectors

The same system can be written

$$
x \begin{pmatrix} 1 \\ 3 \end{pmatrix} + y \begin{pmatrix} 2 \\ -1 \end{pmatrix} = \begin{pmatrix} 5 \\ 1 \end{pmatrix}
$$

Now the question becomes: **can we mix the columns of $A$ to produce $\vec{b}$?** Solving the system is choosing the right amounts of each column.

This single shift in viewpoint is the most important idea in the whole chapter. From it, the column space, rank, and existence of solutions fall out for free.

![Ax = b geometrically: solvability lives in the column space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig1_ax_equals_b.png)

The figure above shows both sides of the story. On the **left**, the columns $\vec{a}_1, \vec{a}_2$ span the whole plane, so any target $\vec{b}$ can be assembled from them: pick the right scalars and the parallelogram closes onto $\vec{b}$. On the **right**, the two columns happen to be parallel, so the column space collapses to a single line. A target sitting off that line is unreachable; the best we can do is project it onto the line -- the geometry behind least squares.

**Painter analogy.** You stand in front of an empty canvas with three tubes of paint (the columns of $A$). The column space is the set of every color you can produce by mixing. If two tubes are the same shade, you have not gained any new color; your reachable palette is smaller than it looks. That smaller palette is exactly the column space of a rank-deficient matrix.

---

## Gaussian Elimination: The Operational Tool

### The Three Legal Moves

Elimination simplifies a system without changing its solution set, using only three **elementary row operations**:

1. Swap two rows.
2. Multiply a row by a non-zero constant.
3. Add a multiple of one row to another.

Why are these legal? Because each one is **invertible**: any sequence of operations can be undone, so the set of solutions before and after is identical.

### A Worked Example

Solve

$$
\begin{cases}
x + 2y + z = 2 \\
3x + 8y + z = 12 \\
4y + z = 2
\end{cases}
$$

Write the augmented matrix and eliminate downward, one pivot at a time.

![Gaussian elimination: turning a system into a triangular ladder](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig4_gaussian_elimination.png)

Each highlighted entry is a **pivot** -- the first non-zero in its row. Once the matrix is triangular, **back-substitute**:

- Row 3: $5z = -10 \implies z = -2$.
- Row 2: $2y - 2z = 6 \implies y = 7/2$.
- Row 1: $x + 2y + z = 2 \implies x = -11/2$.

Three pivots in three columns means three independent constraints on three unknowns -- a unique solution.

### Pivots and Free Variables

After elimination, columns split into two kinds:

- **Pivot columns** -- columns that contain a pivot. The corresponding variables are *determined* by the others.
- **Free columns** -- columns without a pivot. The corresponding variables can be chosen *freely*.

This split decides everything:

| Situation | Solution set |
|-----------|--------------|
| Every column is a pivot column | Unique solution |
| Some columns are free | Infinitely many solutions (one per choice of free variables) |
| A row reads $0 = c \neq 0$ | No solution |

### LU Decomposition: Elimination, Stored

Each "subtract a multiple of one row from another" is itself a matrix multiplication on the left by a simple lower-triangular matrix. Multiply them all together and you get a single matrix $L$ such that

$$
A = L \cdot U
$$

where $U$ is the upper-triangular matrix you ended up with, and $L$ is lower-triangular with the elimination multipliers stored in its entries. **LU decomposition is just Gaussian elimination, packaged for re-use:** once you have $L$ and $U$, you can solve $A\vec{x}=\vec{b}$ for any new $\vec{b}$ in $O(n^2)$ instead of $O(n^3)$.

![LU decomposition as two simple shears in sequence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig7_lu_decomposition.png)

Geometrically the picture is delightful. $A$ may look complicated, but elimination splits it into two of the simplest transformations there are: an upper-triangular shear-and-scale ($U$) followed by a lower-triangular shear ($L$). Triangular matrices are easy because their action is **causal** -- each output coordinate depends only on earlier inputs -- which is exactly why back-substitution works.

```python
import numpy as np

A = np.array([[1, 2, 1],
              [3, 8, 1],
              [0, 4, 1]], dtype=float)
b = np.array([2, 12, 2], dtype=float)

x = np.linalg.solve(A, b)
print(f"Solution: {x}")
print(f"Verify Ax = {A @ x}")
```

---

## Column Space: Where the Matrix Can Reach

### Definition

The **column space** of $A$ (written $C(A)$ or $\text{Col}(A)$) is the set of all vectors $A$ can produce:

$$
C(A) = \{ A\vec{x} \mid \vec{x} \in \mathbb{R}^n \} = \text{span}\{ \text{columns of } A \}
$$

Two equivalent ways to read this: it is **everything you can output**, and it is **the span of the columns**.

### The Existence Theorem

> $A\vec{x} = \vec{b}$ has a solution **if and only if** $\vec{b} \in C(A)$.

This is the cleanest statement in the chapter. "Does my equation have a solution?" becomes "is my target in the column space?" -- a purely geometric question.

### What Column Spaces Look Like

![Column space = span of the columns of A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig2_column_space.png)

For a $3 \times 3$ matrix the column space lives inside $\mathbb{R}^3$, and there are only three possibilities:

| Rank | Column space | Meaning |
|------|--------------|---------|
| 1 | A line through the origin | All columns are scalar multiples of one direction |
| 2 | A plane through the origin | Two independent directions; the third column is redundant |
| 3 | All of $\mathbb{R}^3$ | Three independent directions; $A$ is invertible |

The pattern generalises: for an $m \times n$ matrix, the column space is some $r$-dimensional subspace of $\mathbb{R}^m$, where $r$ is the rank.

**Mixer analogy.** Imagine an audio mixer with three faders (the columns) and one master output. The set of all mixes you can produce is the column space. If two channels carry the same instrument, sliding their faders changes nothing genuinely new -- that redundancy is what "rank deficiency" sounds like.

---

## Null Space: What Gets Crushed

### Definition

The **null space** of $A$ (written $N(A)$ or $\ker A$) is the set of inputs that get sent to zero:

$$
N(A) = \{ \vec{x} \mid A\vec{x} = \vec{0} \}
$$

The null space always contains the zero vector (since $A\vec{0} = \vec{0}$ for any matrix). The interesting question is whether it contains anything *else*.

![Null space N(A) = directions A annihilates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig3_null_space.png)

The figure shows the geometric punchline. **Left**: the matrix $A=\begin{pmatrix}1&2\\2&4\end{pmatrix}$ has linearly dependent rows. Its null space is the entire line $\text{span}\{(-2,1)\}$ -- every vector along that direction is mapped to the origin. The image (column space) is a different line, the direction $(1,2)$. **Right**: the projection $\mathbb{R}^3 \to \mathbb{R}^2$ that drops the $z$-coordinate has the entire $z$-axis as its null space; everything vertical is annihilated.

### Why the Null Space Controls Uniqueness

If $\vec{x}_p$ is *any* particular solution to $A\vec{x}=\vec{b}$, then for any $\vec{n} \in N(A)$:

$$
A(\vec{x}_p + \vec{n}) = A\vec{x}_p + A\vec{n} = \vec{b} + \vec{0} = \vec{b}
$$

So $\vec{x}_p + \vec{n}$ is also a solution. The **complete solution set** is always

$$
\{ \vec{x}_p + \vec{n} \mid \vec{n} \in N(A) \}
$$

The geometric picture is simple: take the null space (a subspace through the origin) and shift it by one particular solution. The result is an **affine subspace** parallel to the null space -- exactly the solution set.

- If $N(A) = \{\vec{0}\}$: the solution is **unique** (when it exists).
- If $N(A)$ contains non-zero vectors: there are **infinitely many** solutions, parametrised by the null space.

**Steamroller analogy.** A steamroller compresses a 3D object into a 2D pancake. All vertical motion is lost -- the vertical direction is in the null space. Two objects whose only difference is vertical produce the same flattened image: the null space is exactly the ambiguity in inverting the flattening.

---

## Rank: Effective Dimension

The **rank** of $A$ is

$$
\text{rank}(A) = \dim C(A) = \text{number of pivots after elimination}
$$

It is also the maximum number of linearly independent columns, and (a small miracle) the maximum number of linearly independent **rows**. Row rank equals column rank for any matrix -- it is one of those theorems that looks almost trivial once proved, but says something deep about the symmetry between rows and columns.

### What Rank Tells You

Rank is the count of **effective dimensions** -- how many independent directions the transformation actually preserves.

| $3\times 3$ matrix with rank | Geometric effect |
|------------------------------|------------------|
| 3 (full rank) | Maps $\mathbb{R}^3$ onto $\mathbb{R}^3$; invertible |
| 2 | Squashes 3D space onto a plane |
| 1 | Squashes 3D space onto a line |
| 0 | The zero matrix; everything goes to the origin |

**Information analogy.** Rank is the number of independent information channels. A color photo carries rank-3 information per pixel (R, G, B). Convert it to greyscale and the rank drops to 1; you have lost two whole channels. In machine learning, **low-rank approximation** is the same idea applied to data matrices: keep only the dominant channels and discard the rest.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Rank: {np.linalg.matrix_rank(A)}")  # 2  (row 3 = 2*row 2 - row 1)
```

---

## The Rank-Nullity Theorem

For any $m \times n$ matrix $A$,

$$
\boxed{\;\text{rank}(A) + \dim N(A) = n\;}
$$

In words: **the dimensions you keep plus the dimensions you crush equal the number of input dimensions you started with.** Nothing is created and nothing is lost.

![Rank-Nullity Theorem: every input dimension is either preserved or crushed](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig5_rank_nullity.png)

The bar chart on the left is the theorem in pictures: for every matrix, the blue (rank) and amber (nullity) bars sum to $n$. The pie on the right shows the same thing as a partition of the input space $\mathbb{R}^n$ into a "preserved" part (the row space) and a "crushed" part (the null space).

### Worked Example

Suppose $A$ is $3 \times 5$ with rank $r=2$. Then

$$
\dim N(A) = n - r = 5 - 2 = 3
$$

Three free variables, a 3-dimensional null space, and a 2-dimensional column space living inside $\mathbb{R}^3$ -- the full structure decoded from a single number.

---

## Four Cases for $A\vec{x}=\vec{b}$

For an $m \times n$ matrix of rank $r$, only four scenarios are possible.

![Three faces of Ax = b: unique, infinitely many, none](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/05-linear-systems-and-column-space/fig6_solution_scenarios.png)

### Case 1: $r = m = n$ -- Square and Full Rank

$A$ is invertible. For every $\vec{b}$ there is exactly one solution $\vec{x} = A^{-1}\vec{b}$. The column space is all of $\mathbb{R}^m$ and the null space is just $\{\vec{0}\}$.

### Case 2: $r = n < m$ -- Tall and Full Column Rank (Overdetermined)

More equations than unknowns. The column space is a proper subspace of $\mathbb{R}^m$, so most $\vec{b}$ are unreachable. When a solution does exist it is unique, but in practice we use **least squares** to find the closest reachable $\vec{b}$ -- that is the orange dot in the rightmost panel above.

### Case 3: $r = m < n$ -- Wide and Full Row Rank (Underdetermined)

More unknowns than equations. The column space fills $\mathbb{R}^m$, so every $\vec{b}$ has a solution -- but the null space has dimension $n-m>0$, so there are infinitely many. The middle panel shows the typical picture: the solution set is a line (or plane, or higher) of equally valid answers.

### Case 4: $r < m$ and $r < n$ -- Rank Deficient

The most delicate case. Some $\vec{b}$ have no solution; others have infinitely many. Both pathologies appear at once.

---

## The Four Fundamental Subspaces

For an $m \times n$ matrix of rank $r$, four subspaces tell the whole story:

| Subspace | Symbol | Lives in | Dimension |
|----------|--------|----------|-----------|
| Column space | $C(A)$ | $\mathbb{R}^m$ | $r$ |
| Null space | $N(A)$ | $\mathbb{R}^n$ | $n - r$ |
| Row space | $C(A^T)$ | $\mathbb{R}^n$ | $r$ |
| Left null space | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ |

These four come in two **orthogonal pairs**:

- In $\mathbb{R}^n$: the row space and the null space are orthogonal complements. Every input vector decomposes uniquely into a "useful" part (row space) and a "wasted" part (null space).
- In $\mathbb{R}^m$: the column space and the left null space are orthogonal complements. Every output direction either lies in the column space or is unreachable.

The matrix $A$ acts as a clean bijection from the row space to the column space (both $r$-dimensional), and crushes the null space to zero. Strang calls this the "big picture of linear algebra," and once you internalise it you stop thinking of matrices as numerical tables and start seeing them as geometric machinery.

---

## Applications

### Least Squares: When There Is No Exact Solution

When $A\vec{x}=\vec{b}$ has no exact solution (overdetermined), we minimise the residual $\|A\vec{x}-\vec{b}\|^2$. The minimiser satisfies the **normal equations**:

$$
A^T A \hat{x} = A^T \vec{b}
$$

Geometrically, $A\hat{x}$ is the orthogonal projection of $\vec{b}$ onto the column space -- the closest reachable point.

```python
import numpy as np

# Fit y = ax + b to (1,2), (2,3), (3,5), (4,4)
A = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
b = np.array([2, 3, 5, 4])

x, *_ = np.linalg.lstsq(A, b, rcond=None)
print(f"Best fit: y = {x[0]:.2f}x + {x[1]:.2f}")
```

### Computer Graphics: Projection

Projecting 3D points onto a 2D screen is a matrix product:

$$
P = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$

Its null space is the entire $z$-axis: depth is destroyed, which is why recovering 3D from a single 2D image is genuinely ambiguous (and why you need stereo, motion, or learned priors).

### Circuit Analysis

Kirchhoff's current law, written in matrix form, says $A\vec{i} = \vec{0}$ where $A$ is the network's incidence matrix. The null space of $A$ is the space of valid loop currents, and **its dimension counts the number of independent loops** in the circuit -- a topological fact extracted from pure linear algebra.

---

## Deep Intuition: Three Questions Before Computing

When you see a linear system, do not start eliminating immediately. First ask:

1. **What is the column space?** It tells you which $\vec{b}$ are solvable.
2. **What is the null space?** It tells you whether the answer is unique, and if not, what shape the solution set has.
3. **What is the rank?** It quantifies how much information $A$ preserves.

These three questions are answered by elimination, but elimination is only the bookkeeping. The geometry is what matters.

```python
import numpy as np

def analyze_system(A, b):
    """Print the solution structure of Ax = b."""
    m, n = A.shape
    r = np.linalg.matrix_rank(A)
    r_aug = np.linalg.matrix_rank(np.column_stack([A, b]))

    print(f"Matrix: {m}x{n}, rank={r}, nullity={n-r}")

    if r_aug > r:
        print("  -> No solution (b is outside the column space)")
    elif r == n:
        print("  -> Unique solution")
        print(f"     x = {np.linalg.lstsq(A, b, rcond=None)[0]}")
    else:
        print(f"  -> Infinitely many solutions ({n-r} free variables)")
        print(f"     one particular x = {np.linalg.lstsq(A, b, rcond=None)[0]}")

# Unique solution
analyze_system(np.array([[1, 2], [3, -1]], dtype=float),
               np.array([5, 1], dtype=float))

# Infinitely many solutions
analyze_system(np.array([[1, 2, 3], [2, 4, 6]], dtype=float),
               np.array([1, 2], dtype=float))
```

---

## Chapter Summary

| Concept | What it tells you |
|---------|-------------------|
| Column space $C(A)$ | Which $\vec{b}$ are solvable |
| Null space $N(A)$ | Whether the solution is unique; the shape of the solution set |
| Rank | How many independent directions $A$ preserves |
| Rank-nullity | $\text{rank} + \text{nullity} = n$ -- a conservation law for dimensions |
| Four subspaces | The complete structural picture of any matrix |

The essential thinking of linear algebra is to **understand equations through spaces and dimensions**, not through mechanical computation. Elimination remains the workhorse algorithm, but its real job is to expose the geometry that was already there.

---

## What Comes Next

**Chapter 6: Eigenvalues and Eigenvectors.** Most vectors change direction under a transformation. A few special ones do not -- they only get scaled. These eigenvectors are the natural axes of $A$, the directions in which the matrix becomes a simple stretch. Find them and you understand the long-term behaviour of any linear system.

---

## Series Navigation

- **Previous:** [Chapter 4 -- The Secrets of Determinants](/en/chapter-04-the-secrets-of-determinants/)
- **Next:** [Chapter 6 -- Eigenvalues and Eigenvectors](/en/chapter-06-eigenvalues-and-eigenvectors/)
- **Series:** Essence of Linear Algebra (5 of 18)
