---
title: "Symmetric Matrices and Quadratic Forms -- The Best Matrices in Town"
date: 2025-02-19 09:00:00
tags:
  - Linear Algebra
  - Symmetric Matrices
  - Quadratic Forms
  - Positive Definite
  - Spectral Theorem
  - Cholesky Decomposition
description: "Symmetric matrices are the 'nicest' matrices in linear algebra: real eigenvalues, orthogonal eigenvectors, and perfect diagonalization. This chapter builds intuition for quadratic forms, positive definiteness, and why symmetric matrices dominate physics, optimization, and data science."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 8
  total: 18
lang: en
mathjax: true
disableNunjucks: true
series_order: 8
---

## Why Symmetric Matrices Are the "Best"

Of all the matrices you will ever meet, **symmetric matrices** are the most well-behaved. They have:

- only **real** eigenvalues,
- a complete set of **orthogonal** eigenvectors,
- and a **perfect diagonalization** $A = Q\Lambda Q^T$ that costs nothing to invert.

This is not a curiosity. Almost every important matrix you actually compute with in physics, optimization, statistics, or machine learning is symmetric:

- A **covariance matrix** $\Sigma = \tfrac{1}{n}X^TX$ records how features vary together. It is symmetric by construction.
- A **Hessian matrix** $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ records second derivatives. By Clairaut's theorem, mixed partials commute, so $H$ is symmetric.
- A **stiffness matrix** $K$ encodes how connected springs push on each other. Newton's third law forces $K = K^T$.
- A **kernel** or **Gram matrix** $G_{ij} = \langle x_i, x_j \rangle$ measures pairwise similarity. Inner products are symmetric, so $G$ is too.

This chapter explains why symmetry buys you so much, and how the geometry of **quadratic forms** lets you read off the behaviour of a symmetric matrix at a glance.

### What You Will Learn

- Why every real symmetric matrix has real eigenvalues and orthogonal eigenvectors.
- The **Spectral Theorem** -- diagonalizing any symmetric matrix.
- **Quadratic forms** as bowls, saddles, and hilltops.
- **Positive definite** matrices: how to recognize them and why they matter.
- The **Cholesky decomposition** as the "square root" of a positive definite matrix.
- The **Rayleigh quotient** and its link to PCA.
- A first look at the **SVD** as the natural extension of the spectral theorem.

### Prerequisites

- Eigenvalues and eigenvectors (Chapter 6)
- Orthogonality and projections (Chapter 7)
- Matrix transpose and basic properties (Chapter 3)

---

## The Shape of a Symmetric Matrix

A real matrix $A$ is **symmetric** if $A = A^T$, i.e. $a_{ij} = a_{ji}$ for every pair $(i, j)$. The diagonal entries are free; everything else comes in mirrored pairs across the main diagonal.

![Symmetric matrix structure: free diagonal, mirrored off-diagonals](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig1_symmetric_structure.png)

**Geometric picture.** A symmetric transformation only **stretches and compresses** along certain orthogonal directions. There is no twisting, no shearing, no rotation hidden inside. Think of pulling a piece of clay straight along two perpendicular axes -- you change its proportions but never spin it.

**Spring analogy.** Two masses are connected by springs. If pushing mass 1 produces a force on mass 2, then pushing mass 2 produces an equal force back on mass 1. That mutual reciprocity is the physical content of $A = A^T$.

### Where Symmetric Matrices Show Up

| Object | Formula | Why symmetric |
|---|---|---|
| Covariance | $\Sigma = \tfrac{1}{n} X^T X$ | $(X^T X)^T = X^T X$ |
| Hessian | $H_{ij} = \partial^2 f/\partial x_i \partial x_j$ | mixed partials commute |
| Gram matrix | $G_{ij} = \langle x_i, x_j\rangle$ | inner product is symmetric |
| Stiffness | $K\vec{x} = \vec{F}$ | Newton's third law |
| Adjacency (undirected graph) | $a_{ij} = a_{ji}$ | edges have no direction |

---

## Three Superpowers of Symmetry

### Superpower 1: Real Eigenvalues

> **Theorem.** Every eigenvalue of a real symmetric matrix is real.

A general matrix can spin space, and rotation eigenvalues are complex (a $90^\circ$ rotation has eigenvalues $\pm i$). Symmetric matrices never spin -- they only stretch -- so their eigenvalues stay on the real line.

**Proof sketch.** Let $A\vec{v} = \lambda \vec{v}$ with $\vec{v}$ possibly complex. Take $\vec{v}^* A \vec{v}$. Since $A$ is real and symmetric, $A^* = A^T = A$, so

$$
\overline{\vec{v}^* A \vec{v}} = \vec{v}^T A^T \overline{\vec{v}} = \vec{v}^T A \overline{\vec{v}} = \vec{v}^* A \vec{v}.
$$

The number $\vec{v}^* A \vec{v} = \lambda \vec{v}^* \vec{v}$ equals its own conjugate, and $\vec{v}^* \vec{v} > 0$, so $\lambda = \overline{\lambda}$. Hence $\lambda$ is real.

### Superpower 2: Orthogonal Eigenvectors

> **Theorem.** Eigenvectors of a symmetric matrix from **distinct** eigenvalues are orthogonal.

**Proof.** Let $A\vec{v}_1 = \lambda_1 \vec{v}_1$ and $A\vec{v}_2 = \lambda_2 \vec{v}_2$. Compute $\vec{v}_1^T A \vec{v}_2$ two ways:

$$
\vec{v}_1^T (A \vec{v}_2) = \lambda_2 \, \vec{v}_1^T \vec{v}_2,
\qquad
(A \vec{v}_1)^T \vec{v}_2 = \lambda_1 \, \vec{v}_1^T \vec{v}_2.
$$

Both expressions equal $\vec{v}_1^T A \vec{v}_2$ because $A^T = A$. Subtracting,

$$
(\lambda_1 - \lambda_2)\, \vec{v}_1^T \vec{v}_2 = 0.
$$

Since $\lambda_1 \ne \lambda_2$, we must have $\vec{v}_1 \perp \vec{v}_2$.

When eigenvalues repeat, the eigenspace has dimension equal to the multiplicity, and we can hand-pick an orthonormal basis inside it. Either way, **a full orthonormal basis of eigenvectors always exists**.

### Superpower 3: The Spectral Theorem

Combine the previous two and you get the centerpiece of the chapter.

> **Spectral Theorem.** Every real symmetric matrix $A$ admits a factorization
> $A = Q \Lambda Q^T,$
> where $Q$ is orthogonal ($Q^T Q = I$) with the orthonormal eigenvectors as columns, and $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ holds the eigenvalues.

![Spectral theorem: orthogonal eigenvectors of a symmetric matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig2_spectral_theorem.png)

**Three ways to read this:**
1. In the eigenvector basis, $A$ is just a list of stretch factors.
2. $A$ acts as a pure rescaling along $n$ mutually orthogonal axes.
3. Computing $A^k$, $A^{-1}$, $e^A$, $A^{1/2}$, $\ldots$ all collapse to the same operations on the diagonal $\Lambda$.

**Outer-product form.** The same theorem can be written as a sum of rank-1 pieces:

$$
A = \lambda_1 \vec{q}_1 \vec{q}_1^T + \lambda_2 \vec{q}_2 \vec{q}_2^T + \cdots + \lambda_n \vec{q}_n \vec{q}_n^T.
$$

Each $\vec{q}_i \vec{q}_i^T$ is the rank-1 projector onto the $i$-th eigenvector, and $\lambda_i$ is its weight. The word "spectral" is a deliberate analogy with optics: white light is decomposed into pure colours (frequencies); a symmetric matrix is decomposed into pure directions weighted by intensities.

---

## Quadratic Forms: The Geometry of Energy

### Definition

A **quadratic form** in $n$ variables is a homogeneous polynomial of degree two:

$$
Q(\vec{x}) \;=\; \vec{x}^T A \vec{x} \;=\; \sum_{i, j} a_{ij}\, x_i x_j.
$$

We can always assume $A$ is symmetric, because $\vec{x}^T B \vec{x} = \vec{x}^T \tfrac{B + B^T}{2} \vec{x}$.

**Example.** $Q(x_1, x_2) = 3 x_1^2 + 4 x_1 x_2 + x_2^2$ comes from

$$
A = \begin{pmatrix} 3 & 2 \\ 2 & 1 \end{pmatrix},
$$

where the cross-term coefficient $4$ is split symmetrically into two $2$s.

**Physics.** Elastic potential energy of a coupled-spring system is $E = \tfrac{1}{2} \vec{x}^T K \vec{x}$. The matrix $K$ is the stiffness; the energy is its quadratic form.

### Reading the Eigenvalues

Drop $\vec{x}$ into the eigenbasis: $\vec{x} = Q\vec{y}$ gives

$$
Q(\vec{x}) \;=\; \vec{y}^T \Lambda \vec{y} \;=\; \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2.
$$

All cross terms vanish. The shape of the level set $Q = \mathrm{const}$ is now read directly from the **signs of the eigenvalues** -- this is the **principal axis theorem**.

| Eigenvalue signs | Name | 2D shape | Mental picture |
|---|---|---|---|
| All $> 0$ | Positive definite | Ellipse / bowl | Ball at the bottom of a bowl |
| All $< 0$ | Negative definite | Inverted ellipse / hill | Ball perched on a hilltop |
| Mixed signs | Indefinite | Hyperbola / saddle | Saddle point, no min and no max |
| All $\ge 0$, some $= 0$ | Positive semidefinite | Trough | Flat along an axis |

![Quadratic-form signatures: bowl, saddle, hill](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig3_quadratic_signature.png)

A more complete tour of the four cases, with each panel labeled by its eigenvalues:

![The four kinds of symmetric matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig4_definiteness_zoo.png)

### Worked Standardization

Take $Q = 3x_1^2 + 2 x_1 x_2 + 3 x_2^2$, so $A = \bigl(\begin{smallmatrix} 3 & 1 \\ 1 & 3 \end{smallmatrix}\bigr)$. Solving $\det(A - \lambda I) = 0$ gives $\lambda_1 = 4$ and $\lambda_2 = 2$. The standard form is

$$
Q = 4 y_1^2 + 2 y_2^2,
$$

an ellipse whose principal axes are tilted at $45^\circ$ relative to the original $x$-axes.

---

## Positive Definite Matrices

### Definition

A symmetric matrix $A$ is **positive definite** (PD) if for every nonzero $\vec{x}$,

$$
\vec{x}^T A \vec{x} > 0.
$$

Geometrically, the energy increases in every direction away from the origin -- the surface is a true bowl, and the origin is its unique minimum.

Related cousins:
- **Positive semidefinite (PSD):** $\vec{x}^T A \vec{x} \ge 0$ (flat in some directions).
- **Negative definite:** $\vec{x}^T A \vec{x} < 0$ (a hilltop).
- **Indefinite:** takes both positive and negative values (a saddle).

### Four Equivalent Tests

For a real symmetric matrix $A$, the following are equivalent.

1. **Eigenvalue test.** All $\lambda_i > 0$.
2. **Sylvester's criterion.** Every leading principal minor is positive:
   $$a_{11} > 0, \quad \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22}\end{vmatrix} > 0, \quad \ldots, \quad \det(A) > 0.$$
3. **Cholesky exists.** $A = L L^T$ for some lower-triangular $L$ with positive diagonal.
4. **Pivot test.** Gaussian elimination on $A$ produces $n$ positive pivots.

In numerical practice, attempting a Cholesky factorization is the **most reliable** test: it succeeds if and only if $A$ is PD, and it gives you a useful factorization for free.

### Useful Properties

- **Invertible.** $\det(A) = \prod \lambda_i > 0$.
- **Positive diagonal.** Take $\vec{x} = \vec{e}_i$ to see $a_{ii} = \vec{e}_i^T A \vec{e}_i > 0$.
- **Inverse is PD.** If $A = Q\Lambda Q^T$ then $A^{-1} = Q\Lambda^{-1} Q^T$ with positive eigenvalues $1/\lambda_i$.
- **Sum is PD.** If $A, B$ are PD then $A + B$ is PD.
- **Square root.** A unique PD matrix $A^{1/2}$ satisfies $A^{1/2} A^{1/2} = A$.
- **$X^T X$ is PSD,** and PD if and only if $X$ has full column rank.

---

## Cholesky Decomposition: Matrix Square Root

### Definition

For a positive definite $A$, the **Cholesky decomposition** is

$$
A = L L^T,
$$

where $L$ is lower triangular with strictly positive diagonal. It exists and is unique. Think of it as $\sqrt{4} = 2$ generalized to matrices: a PD matrix splits as $L \times L^T$.

### Computing It in 2D

For $A = \bigl(\begin{smallmatrix} a & b \\ b & c \end{smallmatrix}\bigr)$,

$$
L = \begin{pmatrix} \sqrt{a} & 0 \\ b/\sqrt{a} & \sqrt{c - b^2/a} \end{pmatrix}.
$$

**Worked example.** $A = \bigl(\begin{smallmatrix} 4 & 2 \\ 2 & 3 \end{smallmatrix}\bigr)$:

$$
l_{11} = 2, \quad l_{21} = 1, \quad l_{22} = \sqrt{3 - 1} = \sqrt{2},
\qquad
L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{pmatrix}.
$$

Verify $LL^T = A$.

### Why Cholesky Matters

- **Solving $A\vec{x} = \vec{b}$.** Forward-solve $L\vec{y} = \vec{b}$, then back-solve $L^T \vec{x} = \vec{y}$. Roughly **twice as fast** as LU on the same problem and numerically very stable.
- **Sampling correlated Gaussians.** To draw $\vec{z} \sim \mathcal{N}(\vec{0}, \Sigma)$, sample $\vec{u} \sim \mathcal{N}(\vec{0}, I)$ and set $\vec{z} = L\vec{u}$ where $\Sigma = LL^T$. Then $\mathrm{Cov}(\vec{z}) = L L^T = \Sigma$.
- **Definiteness check.** A failed Cholesky (negative under a square root) certifies that $A$ is not PD.

---

## Principal Axes: Geometry of the Level Set

The level set $\vec{x}^T A \vec{x} = 1$ for a PD matrix is an **ellipsoid**. Its principal axes are exactly the eigenvectors $\vec{q}_i$, and the corresponding semi-axis lengths are $1/\sqrt{\lambda_i}$ (a large eigenvalue means a strong "spring" and therefore a short axis).

![Principal axes of an ellipse equal the eigenvectors of A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig5_principal_axes.png)

In the eigenbasis the ellipse aligns with the coordinate axes and the cross term disappears -- the same surface, written in its own natural coordinates.

The same principle in 3D is what makes **moment of inertia** computations tractable: pick the axes of a rigid body's inertia tensor and the rotational equations of motion decouple.

---

## The Rayleigh Quotient

### Definition

For a symmetric matrix $A$, the **Rayleigh quotient** is

$$
R(\vec{x}) \;=\; \frac{\vec{x}^T A \vec{x}}{\vec{x}^T \vec{x}}.
$$

Numerator measures how much $A$ "stretches" in the direction $\vec{x}$; denominator normalizes for length.

### Min--Max Property

> $\lambda_{\min} \;\le\; R(\vec{x}) \;\le\; \lambda_{\max} \quad \text{for every } \vec{x} \ne \vec{0}.$
> The maximum is $\lambda_{\max}$, achieved at the corresponding eigenvector; the minimum is $\lambda_{\min}$, achieved at its eigenvector.

**Proof in one line.** Write $\vec{x} = \sum c_i \vec{q}_i$ in the orthonormal eigenbasis. Then $R(\vec{x}) = \sum \lambda_i c_i^2 / \sum c_i^2$, a weighted average of the eigenvalues with non-negative weights.

![Rayleigh quotient: extrema sit on eigenvectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig6_rayleigh_quotient.png)

This is exactly the **PCA** statement: the direction of maximum variance is the top eigenvector of the covariance matrix, and that maximum variance is the top eigenvalue. Power iteration -- the simplest eigenvalue algorithm -- is a direct consequence of this fact.

---

## Whitening and Decorrelation

If a covariance matrix $\Sigma$ is PD, the spectral theorem gives $\Sigma = Q\Lambda Q^T$, hence

$$
\Sigma^{-1/2} \;=\; Q \Lambda^{-1/2} Q^T.
$$

The **whitening transform** $\vec{z} = \Sigma^{-1/2} \vec{x}$ produces uncorrelated, unit-variance features:

$$
\mathrm{Cov}(\vec{z}) \;=\; \Sigma^{-1/2} \, \Sigma \, \Sigma^{-1/2} \;=\; I.
$$

Many ML algorithms (linear regression, naive Bayes with Gaussian features, ICA) implicitly assume whitened inputs. Whitening is a one-line preprocessing step that often dramatically improves numerical conditioning and convergence speed.

---

## Application Tour

### Covariance and PCA

The sample covariance $\Sigma = \tfrac{1}{n} X^T X$ is symmetric and PSD. Its spectral decomposition is exactly **Principal Component Analysis**: the eigenvectors are the principal directions; the eigenvalues are the variances along them. PCA is nothing more than "diagonalize the covariance matrix and keep the top $k$ pieces".

### Hessian and Optimization

Near a critical point of a smooth function $f$, the second-order Taylor expansion is

$$
f(\vec{x}_0 + \vec{h}) \;\approx\; f(\vec{x}_0) + \tfrac{1}{2} \vec{h}^T H \vec{h}.
$$

The Hessian $H$ is symmetric, and its eigenvalues classify the critical point:

- $H$ PD $\Longrightarrow$ local minimum.
- $H$ ND $\Longrightarrow$ local maximum.
- $H$ indefinite $\Longrightarrow$ saddle point.

This is the multivariable second derivative test, in three lines.

### Ridge Regression

The Ridge estimator is

$$
\hat{\vec{w}} \;=\; (X^T X + \lambda I)^{-1} X^T \vec{y}.
$$

Adding $\lambda I$ shifts every eigenvalue of the symmetric PSD matrix $X^T X$ up by $\lambda$, making the matrix strictly PD and well-conditioned. Geometrically, ridge regression replaces a flat valley in the loss surface with a strict bowl, restoring a unique global minimum.

### Vibrating Systems

A coupled mass--spring system has kinetic energy $T = \tfrac{1}{2} \dot{\vec{x}}^T M \dot{\vec{x}}$ and potential energy $V = \tfrac{1}{2} \vec{x}^T K \vec{x}$ with $M, K$ symmetric and PD. The natural frequencies come from the **generalized eigenvalue problem** $K \vec{v} = \omega^2 M \vec{v}$. The eigenvectors are the **normal modes** -- the patterns in which the system oscillates without changing shape. Every guitar harmonic, every bridge resonance, every molecular vibration is an eigenvalue of a stiffness matrix.

### Portfolio Optimization

Markowitz's mean--variance portfolio minimizes risk $\vec{w}^T \Sigma \vec{w}$ subject to a target return. PD-ness of $\Sigma$ guarantees a unique solution and a well-posed quadratic program. Estimating $\Sigma$ in a way that preserves PD-ness (shrinkage, factor models) is itself a research subfield.

---

## SVD: A First Look

What if $A$ is not symmetric, or not even square? The spectral theorem fails, but its closest cousin is the **Singular Value Decomposition**:

$$
A \;=\; U \Sigma V^T,
$$

where $U, V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries (the **singular values**). The decomposition exists for **any** matrix.

![SVD preview: any matrix maps a unit circle to an ellipse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/fig7_svd_preview.png)

Two clean facts tie SVD back to this chapter:
- The singular values of $A$ are the square roots of the eigenvalues of the symmetric PSD matrix $A^T A$.
- The right singular vectors $\vec{v}_i$ are the eigenvectors of $A^T A$; the left singular vectors $\vec{u}_i$ are the eigenvectors of $A A^T$.

So **SVD is the spectral theorem applied to the symmetric companions $A^T A$ and $A A^T$**. Chapter 9 is devoted to it.

---

## Python Walkthrough

### Spectral Decomposition

```python
import numpy as np

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

# eigh is for symmetric/Hermitian matrices: real eigenvalues, orthonormal Q
eigenvalues, Q = np.linalg.eigh(A)
Lambda = np.diag(eigenvalues)

print("Eigenvalues:", eigenvalues)
print("Q^T Q (should be I):\n", Q.T @ Q)
print("Reconstruction Q Lambda Q^T:\n", Q @ Lambda @ Q.T)
```

### Cholesky Decomposition

```python
A = np.array([[4.0, 2.0],
              [2.0, 3.0]])

L = np.linalg.cholesky(A)
print("L:\n", L)
print("L L^T:\n", L @ L.T)

# Use Cholesky to solve A x = b in two triangular solves
b = np.array([6.0, 5.0])
y = np.linalg.solve_triangular(L, b, lower=True)        # not in numpy.linalg
x = np.linalg.solve_triangular(L.T, y, lower=False)     # use scipy in practice
```

### Quadratic Form and Its Principal Axes

```python
import matplotlib.pyplot as plt

A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

g = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(g, g)
Z = A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2

eigvals, eigvecs = np.linalg.eigh(A)

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(X, Y, Z, levels=15, cmap='viridis')
for i in range(2):
    v = eigvecs[:, i] * 2
    ax.arrow(0, 0, v[0], v[1], head_width=0.1, color='red', linewidth=2)
    ax.text(v[0] * 1.1, v[1] * 1.1, f'$\\lambda$={eigvals[i]:.2f}')
ax.set_aspect('equal')
ax.set_title('Quadratic form contours and eigenvectors')
plt.show()
```

---

## Exercises

### Warm-Up

1. Determine whether $A = \bigl(\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}\bigr)$ is positive definite using both the eigenvalue test and Sylvester's criterion.
2. Write $Q(x_1,x_2) = 5x_1^2 + 4x_1x_2 + 2x_2^2$ as $\vec{x}^T A \vec{x}$ and decide whether $A$ is positive definite.
3. Compute the Cholesky factor of $A = \bigl(\begin{smallmatrix} 9 & 6 \\ 6 & 5 \end{smallmatrix}\bigr)$.

### Going Deeper

4. Prove that if $A$ is PD then $A^{-1}$ is PD.
5. Prove that $X^T X$ is always PSD, and PD if and only if $X$ has full column rank.
6. For $Q = 2x_1^2 + 4x_1x_2 + 5x_2^2$: find the eigenvalues, write the standard form, and sketch the level curve $Q = 1$.
7. Show that $\mathrm{tr}(A) = \sum \lambda_i$ and $\det(A) = \prod \lambda_i$ for any symmetric $A$.
8. Show that a PSD matrix is PD iff it is invertible.

### Coding Challenges

9. Implement Cholesky from scratch (no `numpy.linalg.cholesky`). Test on a random PD matrix.
10. Verify the spectral theorem numerically: generate a random symmetric matrix, decompose it, and check $\| A - Q \Lambda Q^T \|$.
11. Implement PCA on a 2D correlated Gaussian. Plot the data, the principal directions, and the projection onto the first PC.
12. Implement whitening: generate correlated Gaussian samples, transform with $\Sigma^{-1/2}$, and verify that the whitened covariance is close to the identity.

---

## Chapter Summary

| Concept | Key fact | Why it matters |
|---|---|---|
| Symmetric matrix | $A = A^T$ | Stretches without twisting |
| Real eigenvalues | Always | No complex surprises |
| Orthogonal eigenvectors | Distinct $\lambda$ implies orthogonal | Clean decomposition |
| Spectral theorem | $A = Q \Lambda Q^T$ | Foundation of PCA, normal modes |
| Quadratic form | $\vec{x}^T A \vec{x}$ | Bowls, saddles, hilltops |
| Positive definite | $\vec{x}^T A \vec{x} > 0$ | Stable energy, unique minimum |
| Rayleigh quotient | $\lambda_{\min} \le R \le \lambda_{\max}$ | Variational characterization, PCA |
| Cholesky | $A = L L^T$ | Fast, stable "square root" |
| SVD | $A = U \Sigma V^T$ | Spectral theorem for any matrix |

---

## Series Navigation

**Previous:** [Chapter 7 -- Orthogonality and Projections](/en/chapter-07-orthogonality-and-projections/)

**Next:** [Chapter 9 -- Singular Value Decomposition](/en/chapter-09-singular-value-decomposition/)

*This is Chapter 8 of the 18-part "Essence of Linear Algebra" series.*

## References

- Strang, G. (2019). *Introduction to Linear Algebra*, Chapter 6.
- Horn, R. A. & Johnson, C. R. (2012). *Matrix Analysis*, 2nd ed.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
- 3Blue1Brown. *Essence of Linear Algebra* series.
