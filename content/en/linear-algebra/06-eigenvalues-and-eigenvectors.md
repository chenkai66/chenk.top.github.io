---
title: "Essence of Linear Algebra (6): Eigenvalues and Eigenvectors"
date: 2025-02-05 09:00:00
tags:
  - Linear Algebra
  - Eigenvalues
  - Eigenvectors
  - Diagonalization
  - PageRank
description: "Some special vectors survive a matrix transformation with their direction intact -- they only get scaled. These eigenvectors and their eigenvalues reveal the deepest structure of linear transformations, powering everything from Google search to PCA."
categories: Linear Algebra
series: linear-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 6
translationKey: "linear-algebra-6"
---
![Essence of Linear Algebra (6): Eigenvalues and Eigenvectors — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/illustration_1.png)


---

## The Big Question

Apply a matrix to a vector and almost anything can happen. Most vectors get rotated *and* stretched, landing in a brand new direction. But scattered among them are a few special vectors that refuse to leave their span. They come out of the transformation pointing exactly the way they went in — only longer, shorter, or flipped.

These survivors are **eigenvectors**. The factor by which they get scaled is the **eigenvalue**.

Hunting for these "invariant directions" is one of the most powerful moves in all of mathematics. Eigenvalues drive Google's PageRank, predict population growth, govern the stability of bridges and aircraft, and turn matrix problems that would take centuries on a supercomputer into a handful of arithmetic.

### What you will learn

- The definition of eigenvectors and eigenvalues, and a clean geometric picture of both
- How to find them with the **characteristic equation**
- **Diagonalization**: rewriting a matrix as scaling along its own natural axes
- Why **complex eigenvalues** show up exactly when a matrix rotates
- Power iteration, PCA, PageRank, Leslie populations, Fibonacci — one idea, five flavours
- The **spectral theorem** for symmetric matrices

### Prerequisites

[Chapter 3](/en/linear-algebra/03-matrices-as-linear-transformations/) (matrices as transformations), [Chapter 4](/en/linear-algebra/04-the-secrets-of-determinants/) (determinants), and [Chapter 5](/en/linear-algebra/05-linear-systems-and-column-space/) (null space and column space).

---

## What an Eigenvector Really Is

### The formal statement

For an $n \times n$ matrix $A$, if there exists a **non-zero** vector $\vec{v}$ and a scalar $\lambda$ such that
$$
A\vec{v} = \lambda\vec{v},$$then $\vec{v}$ is an **eigenvector** of $A$ and $\lambda$ is the corresponding **eigenvalue**.

The word "eigen" is German for "own" or "intrinsic." Eigenvectors are the matrix's own private set of directions.

### Why non-zero matters

The zero vector trivially satisfies $A\vec{0} = \lambda\vec{0}$ for *any* $\lambda$ and *any* $A$. If we let it count, every scalar would be an eigenvalue and the concept would carry no information. So we insist that eigenvectors are non-zero — eigenvalues, however, are allowed to be zero (they signal that $A$ collapses that direction onto the origin).

### A picture is worth a thousand symbols

The definition becomes obvious once you see it. Pick two vectors in the plane and apply the same matrix $A$ to both. One is on an eigen-line; the other is not.

![A vector that stays on its span survives as an eigenvector; an ordinary vector gets knocked off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig1_eigen_definition.png)

The green vector $\vec{v}$ comes out three times longer but pointing the same way — its line through the origin is preserved. The amber vector $\vec{u}$ is rotated *off* its original line. Eigenvectors are precisely the vectors for which the dashed line in the left panel and the dashed line in the right panel coincide.

### Two intuition pumps

**Kneading dough.** When you knead, almost every point inside the dough moves to a totally new location, mixed and rotated. Yet along the rolling pin's axis, points only get compressed — their direction is preserved. Those are the eigen-directions of that particular squashing.

**A spinning ride.** On a merry-go-round, every direction in your body keeps changing — except the rotation axis, which always points up. That axis is an eigenvector with eigenvalue 1.

---

## What the Eigenvalue Tells You

The eigenvalue $\lambda$ is purely a scaling factor along the eigen-line. Three cases capture the whole story:

![A single eigenvector stretched, kept, or flipped depending on the eigenvalue](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig3_eigenvalue_scaling.png)

The complete dictionary:

| Eigenvalue | What happens to the eigenvector |
|-----------|----------------------------------|
| $\lambda > 1$ | Stretched (longer, same direction) |
| $\lambda = 1$ | Unchanged (a true fixed direction) |
| $0 < \lambda < 1$ | Compressed (shorter, same direction) |
| $\lambda = 0$ | Crushed onto the origin — $A$ is singular along this line |
| $-1 < \lambda < 0$ | Flipped and shrunk |
| $\lambda < -1$ | Flipped and stretched |
| $\lambda \in \mathbb{C}\setminus\mathbb{R}$ | Rotation — no real direction is fixed |

### Worked example: the mirror

Reflection across the y-axis is given by $A = \bigl(\begin{smallmatrix}-1&0\\0&1\end{smallmatrix}\bigr)$.

- $\lambda_1 = -1$, eigenvector $(1,0)$: horizontal vectors get flipped left-right.
- $\lambda_2 = 1$, eigenvector $(0,1)$: vertical vectors do not move at all.

That matches your everyday experience of a vertical mirror: left and right swap, up and down stay put.

### Worked example: the shear

The shear $A = \bigl(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\bigr)$ tilts every vertical line into a slanted line. Solving the characteristic equation gives $\lambda = 1$ as a *repeated* eigenvalue, with the only independent eigenvector $(1,0)$.

In other words, a shear has exactly one direction that survives — the horizontal axis — and no second one. We will see that this kind of matrix, called **defective**, cannot be diagonalised.

---

## Finding Eigenvalues: the Characteristic Equation

Rearranging $A\vec{v} = \lambda\vec{v}$ gives$$(A - \lambda I)\vec{v} = \vec{0}.$$For a *non-zero* $\vec{v}$ to satisfy this, the matrix $A - \lambda I$ must be **singular**, meaning$$\det(A - \lambda I) = 0.$$This is the **characteristic equation**. It is a polynomial of degree $n$ in $\lambda$, so by the fundamental theorem of algebra it has exactly $n$ roots in $\mathbb{C}$ (counted with multiplicity). Those roots are the eigenvalues.

### Full worked example

Find the eigenvalues and eigenvectors of $A = \bigl(\begin{smallmatrix}4&2\\1&3\end{smallmatrix}\bigr)$.

**Step 1 — characteristic polynomial.**$$\det\begin{pmatrix}4-\lambda&2\\1&3-\lambda\end{pmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0.
$$
**Step 2 — solve.** $(\lambda - 5)(\lambda - 2) = 0$, so $\lambda_1 = 5$ and $\lambda_2 = 2$.

**Step 3 — eigenvectors.**

For $\lambda_1 = 5$:$$(A - 5I)\vec{v} = \begin{pmatrix}-1&2\\1&-2\end{pmatrix}\vec{v} = \vec{0}\;\Longrightarrow\; v_1 = 2v_2,\;\; \vec{v}_1 = (2, 1).
$$
For $\lambda_2 = 2$:$$(A - 2I)\vec{v} = \begin{pmatrix}2&2\\1&1\end{pmatrix}\vec{v} = \vec{0}\;\Longrightarrow\; v_1 = -v_2,\;\; \vec{v}_2 = (-1, 1).
$$
**Verification.** $A(2,1)^T = (10,5)^T = 5\,(2,1)^T$. ✓

### Two free sanity checks

For *any* $n \times n$ matrix,$$\operatorname{tr}(A) = \lambda_1 + \cdots + \lambda_n,\qquad \det(A) = \lambda_1\cdots\lambda_n.$$For our example: $\operatorname{tr}(A) = 7 = 5+2$ and $\det(A) = 10 = 5\cdot 2$. Always check this — it catches arithmetic mistakes in seconds.

### Visualising eigenvectors of a $2\times 2$ matrix

Plotting the original grid alongside the deformed grid makes the eigenvectors leap out: they are the lines along which the grid hinges.

![Eigenvectors are the lines that the grid pivots around](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig2_eigenvectors_2x2.png)

The grid in the right panel is sheared and stretched, but the green and purple lines through the eigenvectors stay exactly where they were. Every other line tilts.

---

## Diagonalization: Making a Matrix Trivial

![Essence of Linear Algebra (6): Eigenvalues and Eigenvectors — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/illustration_2.png)

### The core idea

Suppose $A$ has $n$ linearly independent eigenvectors $\vec{v}_1, \ldots, \vec{v}_n$ with eigenvalues $\lambda_1, \ldots, \lambda_n$. Pack the eigenvectors into the columns of a matrix $P$ and put the eigenvalues on the diagonal of $D$. Then$$A = P D P^{-1}.$$The transformation $A$ has been *factored* into three pieces: change to the eigenbasis, scale each eigen-axis independently, change back.

### Picture: three steps that compose to $A$

![Diagonalization as change of basis, then pure scaling, then change back](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig4_diagonalization.png)

In the middle panel the transformation is just $D$ — you stretch the x-axis by 3 and leave the y-axis alone. In the eigenbasis, every diagonalisable matrix is just a list of independent stretches.

### Why this is wonderful

**Cheap matrix powers.** Because $A^k = P D^k P^{-1}$ and $D^k = \operatorname{diag}(\lambda_1^k, \ldots, \lambda_n^k)$, raising $A$ to the 100th power costs almost nothing once the eigendecomposition is known — you raise a few scalars, not matrices.

**Long-term behaviour.** As $k \to \infty$, the dominant eigenvalue (the largest in absolute value) controls everything. If $|\lambda_{\max}| > 1$ the iterated system explodes; if $|\lambda_{\max}| < 1$ it dies; if $|\lambda_{\max}| = 1$ it stays bounded.

**Function calculus on matrices.** Define $f(A) := P f(D) P^{-1}$ for any reasonable function — this is how matrix exponentials in differential equations and continuous-time Markov chains work.

### When can you diagonalise?

- $n$ **distinct** eigenvalues guarantee diagonalisability (eigenvectors of distinct eigenvalues are automatically independent — a fact we will prove later).
- **Real symmetric matrices** (and more generally, normal matrices) are *always* diagonalisable, even with repeated eigenvalues, and even with an *orthonormal* basis of eigenvectors.
- A matrix is **defective** if it has fewer independent eigenvectors than its size. The shear $\bigl(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\bigr)$ is the canonical example: $\lambda = 1$ has algebraic multiplicity 2 but geometric multiplicity 1.

---

## Complex Eigenvalues: the Mathematics of Rotation

### Rotations have no real eigenvectors

The 90-degree rotation matrix is$$A = \begin{pmatrix}0&-1\\1&0\end{pmatrix},$$with characteristic equation $\lambda^2 + 1 = 0$, giving $\lambda = \pm i$. There is no real eigenvector, and that is geometrically obvious: a 90-degree rotation moves *every* direction in the real plane.

![Every real vector gets rotated off its span; iterating traces a circle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig5_complex_eigenvalues.png)

The left panel shows that no matter which direction you try, $R$ kicks it off its line. The right panel iterates $R$ on a single starting vector — the trajectory traces a circle, with constant length, exactly because $|\lambda| = |e^{i\theta}| = 1$.

### General rotation

For rotation by angle $\theta$,$$R_\theta = \begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix},\qquad \lambda = \cos\theta \pm i\sin\theta = e^{\pm i\theta}.$$That is Euler's formula falling out of a determinant. The modulus $|\lambda| = 1$ encodes the fact that rotation preserves length.

### Complex eigenvalues come in conjugate pairs

Real matrices have real characteristic polynomials, so complex roots arrive as conjugates: if $\lambda = a + bi$ is an eigenvalue, then so is $\bar\lambda = a - bi$.

### What complex eigenvalues mean for a system

Write $\lambda = re^{i\theta}$. Each iteration of $A$ scales by $r$ and rotates by $\theta$. So the long-term behaviour of $A^k\vec{x}_0$ is

- $r < 1$: spiral inward (damped oscillation),
- $r = 1$: orbit on a circle/ellipse (pure oscillation),
- $r > 1$: spiral outward (growing oscillation).

This is exactly why an undamped pendulum swings forever, why a damped one settles, and why a slightly mistuned feedback loop in control theory blows up.

---

## Power Iteration: Finding the Dominant Eigenvector

Most large-scale eigenvalue problems do not need *all* eigenvalues — you just want the biggest one. The cheapest algorithm is a one-liner:$$\vec{v}_{k+1} = \frac{A\vec{v}_k}{\|A\vec{v}_k\|}.$$Start with anything (almost any starting vector works), apply $A$, normalise, repeat. The iterates converge to the dominant eigenvector.

![Power iteration: every starting direction snaps onto the dominant eigen-line](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig6_power_iteration.png)

Three different starts — three different colours — all wind up on the same green line. Why does this work? Expand $\vec{v}_0$ in the eigenbasis as $\sum_i c_i \vec{v}_i$. Then$$A^k\vec{v}_0 = \sum_i c_i \lambda_i^k \vec{v}_i.$$After $k$ iterations the term with the largest $|\lambda_i|$ dominates the rest by the factor $(|\lambda_1|/|\lambda_2|)^k$. Normalising kills the absolute scale, and what remains is the dominant eigenvector. Convergence is geometric in the spectral *gap* $|\lambda_2|/|\lambda_1|$ — bigger gap, faster convergence.

```python
import numpy as np

def power_iteration(A, num_iters=100):
    v = np.random.rand(A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(num_iters):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
    eigenvalue = v @ A @ v          # Rayleigh quotient
    return eigenvalue, v

A = np.array([[4, 2], [1, 3]])
val, vec = power_iteration(A)
print(f"Dominant eigenvalue: {val:.4f}")   # ~5.0
```

For *all* eigenvalues, the production-quality algorithm is **QR iteration**, which NumPy, MATLAB, and LAPACK use under the hood:

```python
def qr_algorithm(A, num_iters=200):
    Ak = A.astype(float).copy()
    for _ in range(num_iters):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)

print(qr_algorithm(np.array([[4, 2], [1, 3]])))   # [5, 2]
```

---

## Application 1: Population Growth (Leslie Matrix)

Split a species into age classes; let $\vec{p}_t$ count the population in each class at time $t$. The **Leslie matrix** $L$ has fertilities along the top row and survival probabilities on the sub-diagonal, and the model is just $\vec{p}_{t+1} = L\vec{p}_t$.

For three age classes (juvenile, adult, elderly):$$L = \begin{pmatrix}0&2&0.5\\0.6&0&0\\0&0.8&0\end{pmatrix}.
$$
```python
import numpy as np
L = np.array([[0, 2, 0.5], [0.6, 0, 0], [0, 0.8, 0]])
eigenvalues, eigenvectors = np.linalg.eig(L)
print("Dominant eigenvalue:", max(abs(eigenvalues)))
```

The **dominant eigenvalue** decides everything: $|\lambda_1| > 1$ means the population grows; $|\lambda_1| < 1$ means it goes extinct; $|\lambda_1| = 1$ means it stabilises. The corresponding eigenvector is the **stable age distribution** — whatever ratios you start with, you eventually converge to this one.

---

## Application 2: Google PageRank

Larry Page and Sergey Brin asked: how do you measure a page's importance? Their answer: **a page is important if important pages link to it.** That definition is recursive, and recursion of this shape is solved by an eigenvector.

Build the link matrix $H$ where $H_{ij} = 1/L_j$ when page $j$ links to page $i$ (and $L_j$ is the number of outlinks from $j$). The PageRank vector $\vec{r}$ satisfies$$\vec{r} = H\vec{r},$$which is exactly $H\vec{r} = 1\cdot\vec{r}$ — the eigenvector for eigenvalue 1.

To guarantee convergence and handle dangling pages, Google adds a damping factor $d = 0.85$:$$G = dH + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^T.$$Power iteration on $G$ then converges to PageRank. Operationally PageRank *is* power iteration on a sparse matrix with billions of rows.

```python
import numpy as np
H = np.array([[0,   0, 1, 1/3],
              [1/2, 0, 0, 1/3],
              [1/2, 1, 0, 1/3],
              [0,   0, 0, 0  ]])
d, n = 0.85, 4
G = d * H + (1 - d) / n * np.ones((n, n))
r = np.ones(n) / n
for _ in range(100):
    r = G @ r
print("PageRank:", r)
```

---

## Application 3: Fibonacci and the Golden Ratio

Write the Fibonacci recurrence as a matrix iteration:$$\begin{pmatrix}F_{n+1}\\F_n\end{pmatrix} = \begin{pmatrix}1&1\\1&0\end{pmatrix}^n \begin{pmatrix}1\\0\end{pmatrix}.$$The eigenvalues of $\bigl(\begin{smallmatrix}1&1\\1&0\end{smallmatrix}\bigr)$ are$$\phi = \tfrac{1+\sqrt 5}{2}\approx 1.618 \quad\text{(the golden ratio)},\qquad \hat\phi = \tfrac{1-\sqrt 5}{2}\approx -0.618.$$Diagonalising and reading off the first component gives **Binet's formula**:$$F_n = \frac{\phi^n - \hat\phi^n}{\sqrt 5}.$$Since $|\hat\phi| < 1$, the second term decays and $F_{n+1}/F_n \to \phi$. The golden ratio is not a coincidence — it is the dominant eigenvalue of the Fibonacci matrix.

---

## Application 4: PCA Preview

Take a cloud of data points in $\mathbb{R}^n$, centre it, and compute the covariance matrix $C = \tfrac{1}{n-1}XX^T$. The eigenvectors of $C$ are the **principal components** — the orthogonal directions of greatest variance. The eigenvalues are the variances along those directions.

![Principal components are eigenvectors of the covariance matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/06-eigenvalues-and-eigenvectors/fig7_pca_preview.png)

PC$_1$ points along the longest axis of the cloud; PC$_2$ is perpendicular and shorter. Project onto the first few PCs and you get the lowest-dimensional representation that preserves the most information. That single idea powers face recognition, gene-expression clustering, recommendation systems, and — as we will see in [Chapter 9](/en/linear-algebra/09-singular-value-decomposition/) — the singular value decomposition.

---

## Symmetric Matrices: the Spectral Theorem

Real symmetric matrices ($A = A^T$) are the friendliest matrices in linear algebra:

1. **All eigenvalues are real** (no complex roots).
2. Eigenvectors of *distinct* eigenvalues are **orthogonal**.
3. There is always an **orthonormal** basis of eigenvectors. Equivalently, $A = Q\Lambda Q^T$ with $Q$ orthogonal ($Q^{-1} = Q^T$).

This is the **spectral theorem**, and it is the reason that PCA, image compression, and most of optimisation work as cleanly as they do.

### Spectral decomposition

Every real symmetric matrix unpacks into a sum of rank-one pieces:$$A = \lambda_1\,\vec{q}_1\vec{q}_1^T + \lambda_2\,\vec{q}_2\vec{q}_2^T + \cdots + \lambda_n\,\vec{q}_n\vec{q}_n^T.$$Truncating the sum after the $k$ largest $|\lambda_i|$ gives the best rank-$k$ approximation to $A$ in the Frobenius norm — the engine of low-rank denoising and compression.

### Positive definiteness

For a symmetric $A$, the *signs* of the eigenvalues classify its quadratic form:

| Condition | Quadratic form $\vec{x}^T A\vec{x}$ |
|-----------|--------------------------------------|
| All $\lambda_i > 0$ | Positive definite — bowl with a unique minimum |
| All $\lambda_i \ge 0$ | Positive semi-definite |
| All $\lambda_i < 0$ | Negative definite — dome with a unique maximum |
| Mixed signs | Indefinite — saddle |

In optimisation, the Hessian of an objective at a critical point being positive definite *is* the second-order condition for a strict local minimum.

---

## Properties Worth Memorising

| Object | Eigenvalues | Eigenvectors |
|--------|-------------|--------------|
| $A$ | $\lambda_i$ | $\vec{v}_i$ |
| $A^k$ | $\lambda_i^k$ | $\vec{v}_i$ (same) |
| $A^{-1}$ (if invertible) | $1/\lambda_i$ | $\vec{v}_i$ (same) |
| $A + cI$ | $\lambda_i + c$ | $\vec{v}_i$ (same) |
| $cA$ | $c\lambda_i$ | $\vec{v}_i$ (same) |
| $A^T$ (real) | $\lambda_i$ | generally different |

And the global identities:$$\operatorname{tr}(A) = \sum_i \lambda_i, \qquad \det(A) = \prod_i \lambda_i, \qquad A \text{ invertible} \iff \text{all } \lambda_i \neq 0.
$$
### Similar matrices

If $B = P^{-1}AP$ then $A$ and $B$ are **similar**, and they have the same characteristic polynomial — hence the same eigenvalues with the same multiplicities. Diagonalisation is just the lucky case where the similarity makes $B$ diagonal.

### Algebraic vs. geometric multiplicity

- **Algebraic multiplicity** of $\lambda$: how many times it appears as a root of the characteristic polynomial.
- **Geometric multiplicity** of $\lambda$: $\dim \ker(A - \lambda I)$, the number of independent eigenvectors for that $\lambda$.

The geometric multiplicity is always $\le$ the algebraic multiplicity. A matrix is diagonalisable iff equality holds for every eigenvalue.

---

## Summary

Eigenvalues and eigenvectors uncover the skeleton of a linear transformation:

| Concept | What it tells you |
|---------|-------------------|
| Eigenvector | A direction whose span survives the transformation |
| Eigenvalue | The scaling factor along that direction |
| Diagonalisation | Rewriting the transformation as independent stretches in the eigenbasis |
| Complex eigenvalues | The matrix rotates — no real direction is fixed |
| Dominant eigenvalue | Controls long-term behaviour of $A^k\vec{x}_0$ |
| Spectral theorem | Symmetric $\Rightarrow$ real eigenvalues + orthonormal eigenvectors |

> **Core intuition.** Find the directions whose direction does not change, and a complicated transformation reduces to a list of simple stretches.

From Google search to population biology, from quantum mechanics to deep learning — eigenvalues are everywhere. Master this idea and you hold one of the master keys to applied mathematics.

---

## Numerical Stability: Why We Never Use the Characteristic Polynomial

The textbook way to find eigenvalues is: form $\det(A - \lambda I) = 0$, expand into a polynomial in $\lambda$, find its roots. *Nobody* does this in code, and there is a beautiful reason.

The roots of a polynomial are an *exquisitely* ill-conditioned function of its coefficients. Wilkinson's classic example: the polynomial $\prod_{k=1}^{20}(x - k)$ has roots at $1, 2, \ldots, 20$. Perturb the coefficient of $x^{19}$ by $2^{-23}$ — a single bit at the end of a single coefficient — and the roots near $x=15$ explode into complex pairs scattered across radius $> 2$. A microscopic input error becomes a macroscopic output error. The condition number of root-finding for this polynomial is roughly $10^{14}$.

Now imagine the round-trip:
$$A \;\longrightarrow\; \text{characteristic poly coefficients} \;\longrightarrow\; \text{roots}.$$
The first arrow involves cancellations during the determinant expansion — already an error amplifier. The second arrow is the Wilkinson disaster. Even for a $20\times 20$ matrix with perfectly reasonable entries, the eigenvalues you get back can be off in the leading digit.

Real eigenvalue solvers — LAPACK's `dgeev` for general matrices, `dsyev` for symmetric ones, behind `np.linalg.eig` and `np.linalg.eigh` — never form the polynomial. They use the **QR algorithm**: iteratively factor $A_k = Q_k R_k$, then set $A_{k+1} = R_k Q_k$. The matrix $A_{k+1}$ has the same eigenvalues as $A_k$ (it's a similarity transform), and under mild assumptions $A_k$ converges to upper-triangular, with eigenvalues on the diagonal. With shifts and Hessenberg pre-reduction, the cost is $\Theta(n^3)$ and the algorithm is *backward stable*: the computed eigenvalues are the exact eigenvalues of $A + \delta A$ where $\|\delta A\| \le c\,\varepsilon\,\|A\|$.

For the symmetric case, `dsyev` uses tridiagonal reduction plus divide-and-conquer or MRRR; it's faster ($\sim \tfrac{4}{3}n^3$) and even more stable. The lesson: when you see "find the eigenvalues" in your code, write `np.linalg.eigh` if symmetric, `np.linalg.eig` otherwise, and never look back. The characteristic polynomial is a definition, not an algorithm.

## Connection to Deep Learning: Jacobian Eigenvalues and Vanishing/Exploding Gradients

The most consequential appearance of this chapter in modern ML is in the analysis of recurrent and very deep networks. Consider a simple recurrent network where the hidden state evolves as $h_{t+1} = W h_t + (\text{nonlinearity})$. After $T$ steps, the gradient of the loss with respect to $h_0$ involves the product
$$\frac{\partial h_T}{\partial h_0} \;\approx\; \prod_{t=0}^{T-1} J_t,$$
where $J_t$ is the Jacobian at step $t$. For long sequences ($T = 100, 1000$), the *long-term* behaviour of this product is governed by the eigenvalues of the average Jacobian — exactly the dominant-eigenvalue analysis you just read.

Three consequences:

1. **If the dominant $|\lambda| > 1$**, gradients blow up exponentially: $\|J^T\| \sim |\lambda|^T$. After 100 steps with $|\lambda| = 1.1$, the gradient is multiplied by $\sim 13{,}780$. This is **gradient explosion**, and the standard fix is *gradient clipping*: rescale the gradient whenever its norm exceeds a threshold.
2. **If the dominant $|\lambda| < 1$**, gradients decay exponentially. After 100 steps with $|\lambda| = 0.9$, the gradient shrinks by $\sim 26{,}000$. This is **gradient vanishing**, and it is why vanilla RNNs cannot learn long-range dependencies. LSTM and GRU sidestep it by introducing a near-identity path (the cell state) where the effective Jacobian along the dominant direction has eigenvalue close to $1$.
3. **Orthogonal initialisation** for recurrent weight matrices sets all singular values (and hence the spectral radius bound) to $1$ at the start of training, putting you on the edge of stability rather than deep into either basin.

For feed-forward networks the same analysis applies layer by layer: the eigenvalues of $W^TW$ at each layer determine how gradients are amplified or attenuated as they propagate backward. ResNets work in part because the skip connection $x + F(x)$ adds the identity to the layer's Jacobian, biasing eigenvalues away from zero. *Spectral normalisation* in GANs constrains $\sigma_{\max}(W) \le 1$ for stability. Every one of these tricks is a direct application of the eigenvalue intuition you just built.

So when this chapter says "the dominant eigenvalue controls long-term behaviour of $A^k\vec{x}_0$", read it as: *it controls whether your 100-layer network trains at all.*

---

## What's next

- **[Chapter 7](/en/linear-algebra/07-orthogonality-and-projections/):** Orthogonality and projections — the geometry that makes PCA and least squares possible.
- **[Chapter 8](/en/linear-algebra/08-symmetric-matrices-and-quadratic-forms/):** Symmetric matrices and quadratic forms — a deep dive into the spectral theorem.
- **[Chapter 9](/en/linear-algebra/09-singular-value-decomposition/):** Singular value decomposition — *every* matrix is rotation × scaling × rotation, even non-square ones.
- **Chapters 10-18:** Norms, sparse methods, matrix calculus, and applications across ML, vision, and beyond.
