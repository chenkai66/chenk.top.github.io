---
title: "Orthogonality and Projections -- When Vectors Mind Their Own Business"
date: 2024-04-07 09:00:00
tags:
  - Linear Algebra
  - Orthogonality
  - Projections
  - Gram-Schmidt
  - QR Decomposition
  - Least Squares
description: "Orthogonality is what makes GPS work, noise-canceling headphones cancel, and JPEG compress. This chapter builds geometric intuition for orthogonal vectors, projections, Gram-Schmidt, QR decomposition, and least squares -- the backbone of modern scientific computing."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 7
  total: 18
lang: en
mathjax: true
disableNunjucks: true
---

## Why Orthogonality Matters

Two vectors are **orthogonal** when they "do not interfere" with one another. That single idea -- one direction tells you nothing about the other -- powers GPS positioning, noise-canceling headphones, JPEG compression, recommendation systems, and most of numerical linear algebra.

Orthogonality is the single biggest computational shortcut in linear algebra. With a generic basis, finding coordinates is solving a linear system. With an **orthogonal** basis, finding coordinates is one dot product per axis. Hard problem, easy problem, same problem -- just a better basis.

This chapter walks from the everyday intuition of "perpendicular" to the heavy hitters of scientific computing: orthogonal projections, Gram-Schmidt, QR decomposition, and least squares.

### What You Will Learn

- Why a zero dot product means perpendicular -- geometrically and algebraically
- Orthogonal and orthonormal bases, and why they make coordinate computations trivial
- Vector projection: the mathematics of shadows
- Subspace projection and the projection matrix$P=A(A^TA)^{-1}A^T$- Gram-Schmidt orthogonalization: manufacturing orthogonal bases by hand
- QR decomposition: the matrix wrapper around Gram-Schmidt
- Least squares: the best approximate solution when no exact solution exists

### Prerequisites

- Dot product and norm (Chapter 1)
- Linear independence and bases (Chapter 2)
- Matrix-vector products and column space (Chapters 3 and 5)

---

## Orthogonality, Starting from Intuition

### Orthogonality in the Wild

Before any formula, feel what "orthogonal" means.

**City streets.** Manhattan's grid puts north-south streets perpendicular to east-west streets. Walk three blocks east and your north-south position does not change at all. Two directions, zero interference.

**Remote control.** Volume buttons and channel buttons. Pressing volume never switches the channel. Volume and channel are "orthogonal" controls.

**Ingredients.** Salt controls saltiness, sugar controls sweetness. Adding salt does not make the dish sweeter (within reason). Two knobs, two outcomes, no cross-talk.

The recurring pattern: **orthogonal directions carry independent information**.

### The Mathematical Definition

Two vectors$\vec{u}$and$\vec{v}$are **orthogonal** when:$$\vec{u}\cdot\vec{v}=0$$In components:$$u_1v_1+u_2v_2+\cdots+u_nv_n=0$$**Why does zero dot product mean perpendicular?** The geometric form of the dot product is:$$\vec{u}\cdot\vec{v}=\|\vec{u}\|\|\vec{v}\|\cos\theta$$When$\theta=90^\circ$,$\cos\theta=0$, so the dot product vanishes.

![Orthogonal vectors and a generic pair](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig1_orthogonal_vectors.png)

A few special cases worth remembering:

- The zero vector is orthogonal to **every** vector ($\vec{0}\cdot\vec{v}=0$always).
- The standard basis vectors are pairwise orthogonal:$\vec{e}_i\cdot\vec{e}_j=0$for$i\neq j$.
- A nonzero vector is **never** orthogonal to itself ($\vec{v}\cdot\vec{v}=\|\vec{v}\|^2>0$).

### The Deeper Meaning: Information Independence

Orthogonality is really a statement about **information**. When two vectors are orthogonal, knowing the component along one tells you nothing about the component along the other. They are independent measurements of independent things.

A small linguistic example. To describe a person:

- "Height" and "weight" are correlated, hence not orthogonal.
- "Height" and "eye color" are roughly orthogonal -- knowing one tells you nothing useful about the other.

In data analysis, we hunt for orthogonal features because they encode non-redundant information. That is the whole spirit of Principal Component Analysis (PCA), which we will see at the end of the chapter.

---

## Orthogonal Sets and Orthogonal Bases

### Definition

A set$\{\vec{v}_1,\ldots,\vec{v}_k\}$is an **orthogonal set** if every pair is orthogonal:$$\vec{v}_i\cdot\vec{v}_j=0\quad\text{for all }i\neq j$$The standard basis$\{\vec{e}_1,\vec{e}_2,\vec{e}_3\}$in$\mathbb{R}^3$is the canonical example. Three coordinate axes, mutually perpendicular.

### Orthogonal Sets Are Automatically Independent

**Theorem.** Any orthogonal set of nonzero vectors is linearly independent.

**Why intuitively.** Three mutually perpendicular sticks point in completely separate directions. You cannot stack two of them to imitate the third -- there is no overlap to combine.

**Proof in one line.** Suppose$c_1\vec{v}_1+\cdots+c_k\vec{v}_k=\vec{0}$. Dot both sides with$\vec{v}_i$. Every cross term$\vec{v}_j\cdot\vec{v}_i$($j\neq i$) vanishes, leaving$c_i\|\vec{v}_i\|^2=0$. Since$\vec{v}_i\neq\vec{0}$, we get$c_i=0$for every$i$. QED.

Orthogonality buys linear independence for free.

### Orthonormal Bases

If, in addition, every vector has unit length, the set is **orthonormal**. If it spans the whole space, it is an **orthonormal basis**.

For an orthonormal basis$\{\vec{q}_1,\ldots,\vec{q}_n\}$:$$\vec{q}_i\cdot\vec{q}_j=\delta_{ij}=\begin{cases}1&i=j\\0&i\neq j\end{cases}$$### Why Orthogonal Bases Are So Powerful

Take a vector$\vec{v}$and a basis$\{\vec{u}_1,\ldots,\vec{u}_n\}$. We want the coordinates$c_1,\ldots,c_n$such that$\vec{v}=c_1\vec{u}_1+\cdots+c_n\vec{u}_n$.

- **Generic basis.** Solve$U\vec{c}=\vec{v}$. That is Gaussian elimination,$O(n^3)$work.
- **Orthogonal basis.** Each coordinate is just a dot product:$$c_i=\frac{\vec{v}\cdot\vec{u}_i}{\|\vec{u}_i\|^2}$$- **Orthonormal basis.** Even simpler:$$c_i=\vec{v}\cdot\vec{q}_i$$That is$O(n)$per coordinate, with no equation-solving and no risk of catastrophic cancellation. Think of weighing a pile of luggage. With a tangled basis, you have to weigh everything together and back-solve. With an orthogonal basis, you weigh each piece on its own.

---

## Vector Projection: The Mathematics of Shadows

![Projection of b onto a, with the perpendicular error](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig2_vector_projection.png)

### One-Dimensional Projection

Imagine sunlight coming straight down. A tilted stick casts a shadow on the ground. That shadow is the **projection** of the stick onto the ground direction.

The **orthogonal projection** of$\vec{b}$onto$\vec{a}$is:$$\mathrm{proj}_{\vec{a}}\vec{b}=\frac{\vec{a}\cdot\vec{b}}{\vec{a}\cdot\vec{a}}\,\vec{a}$$Reading this slowly:

-$\vec{a}\cdot\vec{b}$measures how much of$\vec{b}$points along$\vec{a}$.
-$\vec{a}\cdot\vec{a}=\|\vec{a}\|^2$normalizes by$\vec{a}$'s squared length.
- The ratio is a scalar; multiplying by$\vec{a}$produces the shadow vector.

The **scalar projection** -- the signed length of the shadow -- is:$$\mathrm{comp}_{\vec{a}}\vec{b}=\frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|}$$It can be negative when$\vec{b}$points roughly opposite to$\vec{a}$.

### Projection as Closest Point

Here is the deep geometric fact:$\mathrm{proj}_{\vec{a}}\vec{b}$is **the point on the line through$\vec{a}$that is closest to$\vec{b}$**.

Why? Call the projection$\hat{\vec{b}}$and the error$\vec{e}=\vec{b}-\hat{\vec{b}}$. By construction,$\vec{e}\perp\vec{a}$. For any other point$t\vec{a}$on the line, the Pythagorean theorem gives:$$\|\vec{b}-t\vec{a}\|^2=\|\vec{e}\|^2+\|t\vec{a}-\hat{\vec{b}}\|^2\geq\|\vec{e}\|^2$$with equality only when$t\vec{a}=\hat{\vec{b}}$. Projection minimizes distance, automatically.

### Orthogonal Decomposition

Every vector$\vec{b}$splits **uniquely** into a part parallel to$\vec{a}$and a part perpendicular to$\vec{a}$:$$\vec{b}=\underbrace{\mathrm{proj}_{\vec{a}}\vec{b}}_{\text{parallel}}+\underbrace{(\vec{b}-\mathrm{proj}_{\vec{a}}\vec{b})}_{\text{perpendicular}}$$The two pieces are mutually orthogonal. The classic physics use case: decomposing gravity into components along and normal to an inclined plane.

---

## Subspace Projection: From Lines to Planes

### The Setup

What if you want to project not onto a line but onto a **plane**, or any higher-dimensional subspace$W$? The projection$\hat{\vec{b}}$of$\vec{b}$onto$W$is defined by:$$\vec{b}-\hat{\vec{b}}\,\perp\,W$$The error vector is orthogonal to **every** vector in$W$, not just one.

![Projection of b onto a plane W (subspace)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig3_subspace_projection.png)

### The Projection Matrix

When$W=\mathrm{Col}(A)$for an$m\times n$matrix$A$with linearly independent columns, the projection has a clean closed form:$$\hat{\vec{b}}=A(A^TA)^{-1}A^T\vec{b}$$The **projection matrix** is:$$P=A(A^TA)^{-1}A^T$$Three properties to memorize:

1. **Idempotent:**$P^2=P$. Once you have a shadow, projecting it again does nothing.
2. **Symmetric:**$P^T=P$. Orthogonal projections are symmetric; oblique ones are not.
3. **Rank:**$\mathrm{rank}(P)=n=\dim W$.

These three together actually **characterize** orthogonal projection matrices.

### The Normal Equations

The coordinate vector$\hat{\vec{x}}$satisfying$\hat{\vec{b}}=A\hat{\vec{x}}$is the solution of the **normal equations**:$$A^TA\hat{\vec{x}}=A^T\vec{b}$$This drops out of the orthogonality condition$\vec{b}-A\hat{\vec{x}}\perp\mathrm{Col}(A)$, which says$A^T(\vec{b}-A\hat{\vec{x}})=\vec{0}$.

### Orthogonal Complements

The **orthogonal complement** of a subspace$W\subseteq\mathbb{R}^n$is the set of all vectors orthogonal to everything in$W$:$$W^{\perp}=\{\vec{v}\in\mathbb{R}^n:\vec{v}\cdot\vec{w}=0\text{ for all }\vec{w}\in W\}$$![Orthogonal complement and unique decomposition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig5_orthogonal_complement.png)

Every vector splits uniquely as$\vec{v}=\vec{v}_W+\vec{v}_{W^{\perp}}$with$\vec{v}_W\in W$and$\vec{v}_{W^{\perp}}\in W^{\perp}$. We write$\mathbb{R}^n=W\oplus W^{\perp}$.

The four fundamental subspaces of any matrix$A$obey:

-$\mathrm{Col}(A)^{\perp}=\mathrm{Null}(A^T)$-$\mathrm{Null}(A)^{\perp}=\mathrm{Row}(A)$These are the "orthogonality structure" Gilbert Strang likes to draw at the bottom of every lecture.

---

## Gram-Schmidt: Manufacturing Orthogonal Bases

### The Problem

You have linearly independent vectors$\vec{a}_1,\ldots,\vec{a}_n$, but they are **not** orthogonal. Can you adjust them into an orthogonal set spanning the same space?

Yes. **Gram-Schmidt** does exactly that, one direction at a time.

![Gram-Schmidt: skewed inputs become an orthogonal basis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig4_gram_schmidt.png)

### The Algorithm

Build orthogonal vectors$\vec{u}_1,\ldots,\vec{u}_n$as follows.

**Step 1.** Take the first vector unchanged:$\vec{u}_1=\vec{a}_1$.

**Step 2.** Subtract the part of$\vec{a}_2$that lies along$\vec{u}_1$:$$\vec{u}_2=\vec{a}_2-\frac{\vec{u}_1\cdot\vec{a}_2}{\vec{u}_1\cdot\vec{u}_1}\,\vec{u}_1$$**Step 3.** Subtract the parts of$\vec{a}_3$along both existing axes:$$\vec{u}_3=\vec{a}_3-\frac{\vec{u}_1\cdot\vec{a}_3}{\vec{u}_1\cdot\vec{u}_1}\,\vec{u}_1-\frac{\vec{u}_2\cdot\vec{a}_3}{\vec{u}_2\cdot\vec{u}_2}\,\vec{u}_2$$**General pattern:**$$\vec{u}_k=\vec{a}_k-\sum_{j=1}^{k-1}\frac{\vec{u}_j\cdot\vec{a}_k}{\vec{u}_j\cdot\vec{u}_j}\,\vec{u}_j$$Normalize each$\vec{u}_k$to length 1 if you want an orthonormal basis:$\vec{q}_k=\vec{u}_k/\|\vec{u}_k\|$.

### The Intuition

Building an orthogonal coordinate system one axis at a time:

- **First axis.** Pick any direction.
- **Second axis.** Take a vector that points "roughly elsewhere" and strip out everything that lay along the first axis. What remains is necessarily perpendicular to it.
- **Third axis.** Take the next vector and strip out its components along **both** existing axes. What remains is perpendicular to both.

Each step removes "contamination" from earlier directions, keeping only the genuinely new information.

### Numerical Stability: Modified Gram-Schmidt

The classical algorithm above has a quiet flaw. Floating-point arithmetic introduces small errors at each subtraction. With many vectors, those errors compound, and the later$\vec{u}_k$drift away from being truly orthogonal to the earlier ones.

**Modified Gram-Schmidt** rearranges the same arithmetic so the running vector is updated immediately after each projection, which is much more stable in practice:

```python
import numpy as np

def modified_gram_schmidt(A):
    """Modified Gram-Schmidt -- numerically stable orthogonalization."""
    m, n = A.shape
    Q = A.copy().astype(float)
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ Q[:, k]
            Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]
    return Q, R
```

For production work, prefer **Householder reflections** (used inside `numpy.linalg.qr`) -- they are even more stable.

---

## QR Decomposition: The Matrix Face of Gram-Schmidt

![QR decomposition: A = QR, columns of A in the orthonormal basis Q](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig7_qr_decomposition.png)

### Definition

Any$m\times n$matrix$A$with linearly independent columns factors as:$$A=QR$$where$Q$is$m\times n$with orthonormal columns ($Q^TQ=I$) and$R$is$n\times n$upper triangular with positive diagonal entries.

### Where$Q$and$R$Come From

The columns of$Q$are exactly the orthonormal vectors that Gram-Schmidt produces from$A$'s columns. The entries of$R$record the projection coefficients:$$r_{ij}=\vec{q}_i\cdot\vec{a}_j$$**Why is$R$upper triangular?** Because$\vec{a}_j$can be written using only$\vec{q}_1,\ldots,\vec{q}_j$-- it never needs orthogonal vectors that come later. Anything below the diagonal is a coefficient on a$\vec{q}_k$with$k>j$, which is necessarily zero.

In picture form:$\vec{a}_j$lives in the span of the first$j$orthonormal axes, and$R$is the change-of-basis matrix from the$\vec{q}$'s back to the$\vec{a}$'s.

### Why QR Matters: Better Least Squares

The normal equations$A^TA\hat{\vec{x}}=A^T\vec{b}$can blow up numerically because forming$A^TA$**squares** the condition number:$\kappa(A^TA)=\kappa(A)^2$. For nearly collinear data, this is fatal.

QR sidesteps the problem. Substitute$A=QR$into the normal equations:$$R^TQ^TQR\hat{\vec{x}}=R^TQ^T\vec{b}$$Since$Q^TQ=I$and$R^T$is invertible, this collapses to:$$R\hat{\vec{x}}=Q^T\vec{b}$$An upper triangular system, solved in$O(n^2)$by back substitution -- both fast and numerically stable.

```python
def qr_decomposition(A):
    """QR decomposition via modified Gram-Schmidt."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
Q, R = qr_decomposition(A)
print("Q =\n", Q)
print("R =\n", R)
print("Verify A = QR:\n", Q @ R)
```

---

## Least Squares: When Equations Have No Solution

![Least squares as projection onto Col(A)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/07-orthogonality-and-projections/fig6_least_squares.png)

### The Problem

Real data is noisy. Fit a line through five measured points? That is five equations in two unknowns -- an **overdetermined** system. Unless the points are exactly collinear, no$\vec{x}$satisfies$A\vec{x}=\vec{b}$.

We do not give up. We change the question.

### The Least Squares Idea

Instead of solving$A\vec{x}=\vec{b}$exactly, find the$\hat{\vec{x}}$that minimizes the **sum of squared errors**:$$\min_{\vec{x}}\|A\vec{x}-\vec{b}\|^2$$**Geometric interpretation.**$A\vec{x}$lives in the column space$\mathrm{Col}(A)$as$\vec{x}$varies. Minimizing$\|A\vec{x}-\vec{b}\|$means finding the point in$\mathrm{Col}(A)$closest to$\vec{b}$-- exactly an orthogonal projection. The answer$\hat{\vec{x}}$satisfies$A\hat{\vec{x}}=\hat{\vec{b}}$, where$\hat{\vec{b}}$is the projection of$\vec{b}$onto$\mathrm{Col}(A)$.

In the figure above, the left panel shows the familiar "best-fit line through scattered points" picture; the right panel shows the same situation as a projection of$\vec{b}$onto the column-space plane. They are the same problem in two different views.

### The Normal Equations (Again)

The orthogonality condition is$\vec{b}-A\hat{\vec{x}}\perp\mathrm{Col}(A)$, so$A^T(\vec{b}-A\hat{\vec{x}})=\vec{0}$:$$A^TA\hat{\vec{x}}=A^T\vec{b}$$You can also derive this with calculus: expand$\|A\vec{x}-\vec{b}\|^2$, take the gradient with respect to$\vec{x}$, set it to zero, and the same equations fall out.

### Linear Regression Example

Fit$(1,2.1),(2,3.9),(3,6.2),(4,7.8),(5,10.1)$with$y=\beta_0+\beta_1 x$:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

A = np.column_stack([np.ones(len(x)), x])  # design matrix

# QR-based solve (preferred)
Q, R = np.linalg.qr(A)
coeffs = np.linalg.solve(R, Q.T @ y)
print(f"Best fit: y = {coeffs[0]:.4f} + {coeffs[1]:.4f}x")
```

### Weighted Least Squares

When some measurements are more trustworthy than others, give them larger weights$w_i$:$$\min_{\vec{x}}\sum_i w_i(\vec{a}_i^T\vec{x}-b_i)^2$$The normal equations become$A^TWA\hat{\vec{x}}=A^TW\vec{b}$with$W=\mathrm{diag}(w_1,\ldots,w_m)$.

---

## Orthogonal Matrices: Preserving Geometry

### Definition

A square matrix$Q$is **orthogonal** if:$$Q^TQ=I$$Equivalently$Q^{-1}=Q^T$. The transpose is the inverse -- which is why orthogonal matrices are so cheap to invert.

### Orthogonal Matrices Preserve Everything

They are "rigid-body" transformations. They preserve:

- **Length:**$\|Q\vec{x}\|=\|\vec{x}\|$- **Inner product:**$(Q\vec{x})\cdot(Q\vec{y})=\vec{x}\cdot\vec{y}$- **Angle:** since the inner product is preserved

### Rotation vs. Reflection

The determinant of an orthogonal matrix is$\pm 1$:

-$\det Q=+1$: a **rotation** (preserves handedness).
-$\det Q=-1$: a **reflection** (reverses handedness).

The 2D rotation by angle$\theta$is the canonical example:$$R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$$**Householder reflection** is another important orthogonal matrix; it is the building block of the most stable QR algorithms in production libraries.

### Why Numerics Loves Orthogonal Matrices

The condition number of an orthogonal matrix is exactly 1:$$\kappa(Q)=\frac{\sigma_{\max}}{\sigma_{\min}}=1$$Computations involving orthogonal matrices never amplify errors. This is the deep reason why so many high-quality numerical algorithms (QR, SVD, Householder, Givens) are built around them.

---

## Applications

### Fourier Analysis

The discrete Fourier transform (DFT) is, mathematically, a change of basis into an **orthogonal basis** of complex exponentials. Each frequency component can be processed independently because the basis vectors do not interfere -- exactly what makes filtering work.

### Noise-Canceling Headphones

Microphones capture external noise, a processor decomposes it into orthogonal frequency components via an FFT, generates a phase-inverted signal, and plays it through the speakers. Because the components are orthogonal, the noise cancels precisely without touching the music.

### Image Compression (JPEG)

JPEG uses the discrete cosine transform (DCT), the real-valued cousin of the DFT. Each$8\times 8$image block is expressed in an orthogonal basis of cosines. High-frequency coefficients (usually small for natural images) are quantized aggressively or dropped, reducing file size while preserving most of the perceived image. Orthogonality is what guarantees that you can keep some coefficients and discard others without smearing the rest.

### CDMA in Mobile Networks

Each user gets a "code" -- and the codes assigned to different users are pairwise orthogonal. Many users share the same frequency simultaneously. To recover user A's signal from the mixed broadcast, the receiver dot-products the signal with A's code; user B's contribution vanishes because$\vec{c}_A\cdot\vec{c}_B=0$.

### PCA: Finding the Most Important Directions

Principal Component Analysis looks for the orthogonal directions of **maximum variance** in a dataset. Concretely, given a centered data matrix$X$, eigendecompose the covariance$\Sigma=\frac{1}{n-1}X^TX=Q\Lambda Q^T$. The columns of$Q$are orthogonal principal axes; the eigenvalues in$\Lambda$measure how much variance each axis carries. Keeping the top$k$columns gives the optimal linear dimensionality reduction.

The orthogonality requirement is what makes the components independent -- each one captures genuinely new structure rather than restating what an earlier one already said.

---

## Python Toolbox

### Gram-Schmidt

```python
import numpy as np

def gram_schmidt(A):
    """Classical Gram-Schmidt orthogonalization.

    A : array, columns are the vectors to orthogonalize.
    Returns Q with orthonormal columns spanning the same space.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float).T
Q = gram_schmidt(A)
print("Q^T Q (should be I):\n", np.round(Q.T @ Q, 10))
```

### Least Squares via QR

```python
def least_squares_qr(A, b):
    """Solve least squares via QR decomposition (numerically stable)."""
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
A = np.column_stack([np.ones(len(x)), x])
print(least_squares_qr(A, y))
```

### Visualizing a Projection

```python
import matplotlib.pyplot as plt

a = np.array([3.0, 1.0])
b = np.array([1.0, 3.0])
proj = (a @ b) / (a @ a) * a
err = b - proj

fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(0, 0, *a, angles='xy', scale_units='xy', scale=1, color='blue', label='a')
ax.quiver(0, 0, *b, angles='xy', scale_units='xy', scale=1, color='green', label='b')
ax.quiver(0, 0, *proj, angles='xy', scale_units='xy', scale=1, color='red', label='proj')
ax.quiver(*proj, *err, angles='xy', scale_units='xy', scale=1, color='orange', label='error')
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5); ax.set_aspect('equal'); ax.grid(True); ax.legend()
plt.show()
```

---

## Exercises

### Warm-Up

1. Verify whether$\vec{u}=(1,2,-1)$and$\vec{v}=(2,-1,0)$are orthogonal. If so, normalize them.
2. Compute the projection of$\vec{b}=(3,4)$onto$\vec{a}=(1,0)$. Verify that the error vector is orthogonal to$\vec{a}$.
3. Apply Gram-Schmidt to$\vec{a}_1=(1,1,0)$,$\vec{a}_2=(1,0,1)$and check that the result is orthogonal.

### Going Deeper

4. Prove that an orthogonal set of nonzero vectors is linearly independent.
5. Prove$P^2=P$for$P=A(A^TA)^{-1}A^T$. Explain geometrically why "the projection of a projection is itself."
6. Compute the QR decomposition of$A=\begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}$by hand and verify$A=QR$.
7. Use least squares to fit$(1,1),(2,3),(3,2),(4,4)$with$y=\beta_0+\beta_1 x$.
8. Prove that the product of two orthogonal matrices is again orthogonal.

### Coding Challenges

9. Implement Modified Gram-Schmidt and compare its orthogonality error against classical Gram-Schmidt on nearly dependent vectors (e.g. columns of the Hilbert matrix).
10. Simulate a 3-user CDMA system with orthogonal codes of length 4. Show that you can separate each user's signal from the mixture by dot-producting with their code.

---

## Chapter Summary

| Concept | Key Formula | One-Sentence Intuition |
|---|---|---|
| Orthogonality |$\vec{u}\cdot\vec{v}=0$| Vectors do not interfere |
| Projection (1D) |$\mathrm{proj}_{\vec{a}}\vec{b}=\dfrac{\vec{a}\cdot\vec{b}}{\vec{a}\cdot\vec{a}}\vec{a}$| The shadow on a line |
| Projection (subspace) |$\hat{\vec{b}}=A(A^TA)^{-1}A^T\vec{b}$| Closest point in the subspace |
| Normal equations |$A^TA\hat{\vec{x}}=A^T\vec{b}$| Heart of least squares |
| Gram-Schmidt | Subtract projections iteratively | Strip out earlier-axis contamination |
| QR decomposition |$A=QR$| Stable orthogonalization in matrix form |
| Orthogonal matrix |$Q^TQ=I$| Preserves length and angle, condition number 1 |

---

## Series Navigation

**Previous:** [Chapter 6 -- Eigenvalues and Eigenvectors](/en/chapter-06-eigenvalues-and-eigenvectors/)

**Next:** [Chapter 8 -- Symmetric Matrices and Quadratic Forms](/en/chapter-08-symmetric-matrices-and-quadratic-forms/)

*This is Chapter 7 of the 18-part "Essence of Linear Algebra" series.*

## References

- Strang, G. (2019). *Introduction to Linear Algebra*, Chapters 4 and 10.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*, Lectures 7-11.
- 3Blue1Brown. *Essence of Linear Algebra*, Chapters 9 and 11.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, Chapter 5.
