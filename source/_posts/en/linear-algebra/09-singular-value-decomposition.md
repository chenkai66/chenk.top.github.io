---
title: "Singular Value Decomposition -- The Crown Jewel of Linear Algebra"
date: 2024-04-09 09:00:00
tags:
  - Linear Algebra
  - SVD
  - Singular Value Decomposition
  - PCA
  - Image Compression
  - Dimensionality Reduction
description: "SVD decomposes any matrix -- not just square or symmetric ones. From image compression to Netflix recommendations, from face recognition to gene analysis, SVD is the most powerful and most universal decomposition in linear algebra."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 9
  total: 18
lang: en
mathjax: true
---

## Why SVD Earns the Crown

The spectral theorem of [Chapter 8](/en/chapter-08-symmetric-matrices-and-quadratic-forms/) gave us $A = Q\Lambda Q^T$ -- a beautifully clean factorisation, but **only for symmetric matrices**. Most matrices that show up in practice are not symmetric, and many are not even square:

- a photograph stored as a $1920 \times 1080$ pixel matrix,
- a Netflix-style user--movie rating matrix (millions of rows, thousands of columns),
- a document--term matrix in NLP (documents by vocabulary),
- a gene-expression matrix in bioinformatics.

**Singular Value Decomposition (SVD)** handles every one of them. For *any* $m \times n$ matrix $A$,
$$A = U\,\Sigma\,V^{\!\top}.$$
This is the most powerful, most universally applicable decomposition in all of linear algebra.

### A photography analogy

Picture a photo as a matrix of pixel intensities. SVD says three things at once:

1. **Any photo decomposes into a sum of "basic layers."**
2. **The layers are ranked by importance** -- the first captures gross structure, the second secondary detail, the third still finer detail.
3. **Keeping just the first few layers recovers most of the image.**

Think of a band recording: lead vocal, guitar, bass, drums. Drop a background harmony and the song still works; drop the lead vocal and it falls apart. SVD makes that intuition precise: the singular values measure exactly *how much* each layer contributes.

### What you will learn

- The definition of SVD and its **three-step geometric meaning** (rotate, stretch, rotate).
- How to compute singular values and singular vectors via $A^{\!\top}\!A$ and $AA^{\!\top}$.
- The four fundamental subspaces, read directly off $U$ and $V$.
- **Low-rank approximation** and the Eckart--Young theorem -- the optimality result behind compression.
- The **pseudoinverse**: a universal "best inverse" for matrices that have no honest inverse.
- **PCA as SVD in disguise**.
- Applications: image compression, recommender systems, latent semantic analysis, denoising, eigenfaces.

### Prerequisites

- Eigenvalues and eigenvectors (Chapter 6)
- Orthogonal matrices and projections (Chapter 7)
- Symmetric matrices and the spectral theorem (Chapter 8)

---

## The Definition

### The fundamental theorem

**SVD theorem.** Every $m \times n$ real matrix $A$ admits a factorisation
$$A = U\,\Sigma\,V^{\!\top}$$
where

- $U \in \mathbb{R}^{m\times m}$ is orthogonal (its columns are the **left singular vectors** $u_1, \ldots, u_m$);
- $V \in \mathbb{R}^{n\times n}$ is orthogonal (its columns are the **right singular vectors** $v_1, \ldots, v_n$);
- $\Sigma \in \mathbb{R}^{m\times n}$ has the **singular values** $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ on its main diagonal and zeros elsewhere.

The "economy" form, used in practice for tall matrices, keeps only the $r = \operatorname{rank}(A)$ nonzero singular values:
$$A = U_r\,\Sigma_r\,V_r^{\!\top},\qquad U_r \in \mathbb{R}^{m\times r},\ \Sigma_r \in \mathbb{R}^{r\times r},\ V_r \in \mathbb{R}^{n\times r}.$$

Three facts make the singular values special:

- They are **non-negative real numbers** -- always. (Eigenvalues can be negative or complex.)
- They are **arranged in descending order** by convention.
- **SVD exists for every matrix.** That is exactly the property eigendecomposition lacks, and it is what makes SVD universal.

### Geometric meaning: a transformation in three steps

The factorisation $A = U\Sigma V^{\!\top}$ has a clean visual story. Reading right to left, applying $A$ to a vector means:

1. **Rotate** by $V^{\!\top}$: align the input with the "natural input directions" of $A$.
2. **Stretch** by $\Sigma$: scale the $i$-th coordinate by $\sigma_i$.
3. **Rotate** by $U$: place the result into its final orientation in the output space.

The dough analogy: you knead with a rolling pin. First you turn the dough to a convenient angle ($V^{\!\top}$); then the pin flattens and stretches it ($\Sigma$); finally you rotate the flattened dough where you need it ($U$).

![SVD as three steps: rotate by V^T, stretch by Sigma, rotate by U](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig1_svd_geometry.png)

The figure tracks a unit circle through each stage. The orthogonal basis stays orthogonal under the two rotations; only the middle step (the diagonal $\Sigma$) actually changes shape, turning the circle into an ellipse with semi-axes $\sigma_1$ and $\sigma_2$.

### Unit circle to ellipse

There is a complementary picture that compresses the same content into one input/output pair: the unit circle on the left, the image ellipse $A(\text{circle})$ on the right.

![Right singular vectors v_i map to sigma_i u_i: the principal axes of the ellipse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig2_circle_to_ellipse.png)

Reading the picture:

- The **right singular vectors** $v_1, v_2$ are the orthonormal directions on the input side that get mapped *onto* the principal axes of the output ellipse.
- The **left singular vectors** $u_1, u_2$ are the orthonormal directions of those axes.
- The **singular values** $\sigma_1, \sigma_2$ are the half-lengths of the axes.

In one line:
$$A\,v_i \;=\; \sigma_i\, u_i.$$
This is the equation that makes everything else work.

### Outer-product form

There is a wonderful equivalent way to write SVD as a sum of rank-1 building blocks:
$$A = \sigma_1 u_1 v_1^{\!\top} + \sigma_2 u_2 v_2^{\!\top} + \cdots + \sigma_r u_r v_r^{\!\top}.$$
Each $u_i v_i^{\!\top}$ is a rank-1 matrix, and the singular values are the weights. This perspective is the key to low-rank approximation: keep the largest weights, drop the rest.

---

## Computing SVD

### The bridge to $A^{\!\top}\!A$ and $AA^{\!\top}$

Eigenvectors of two symmetric matrices give us $V$ and $U$. Multiply out $A = U\Sigma V^{\!\top}$ both ways:
$$A^{\!\top}\!A = V\,\Sigma^{\!\top}\!\Sigma\,V^{\!\top}, \qquad AA^{\!\top} = U\,\Sigma\Sigma^{\!\top}\,U^{\!\top}.$$

Both $A^{\!\top}\!A$ and $AA^{\!\top}$ are **symmetric and positive semidefinite**, so the spectral theorem applies. Reading these as spectral decompositions:

- columns of $V$ = orthonormal eigenvectors of $A^{\!\top}\!A$ (the **right singular vectors**),
- columns of $U$ = orthonormal eigenvectors of $AA^{\!\top}$ (the **left singular vectors**),
- $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are the (nonnegative) eigenvalues -- shared by both products.

**Why $A^{\!\top}\!A$?** Think of it as $A$ "acting twice": go forward by $A$, come back by $A^{\!\top}$. The round trip amplifies a direction by exactly $\sigma^2$, which is why eigenvalues of $A^{\!\top}\!A$ are squared singular values.

### Step-by-step computation

Given $m \times n$ matrix $A$ with $m \ge n$:

1. Form $A^{\!\top}\!A$. Compute its (sorted) eigenvalues $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$ and orthonormal eigenvectors $v_1, \ldots, v_n$. These are the columns of $V$.
2. Set $\sigma_i = \sqrt{\lambda_i}$.
3. For each $\sigma_i > 0$, define $u_i = A v_i / \sigma_i$. These are the first $r$ columns of $U$.
4. If $r < m$, extend $\{u_1, \ldots, u_r\}$ to an orthonormal basis of $\mathbb{R}^m$ (e.g. via Gram--Schmidt) to fill out $U$.

In numerical practice nobody computes SVD this way: forming $A^{\!\top}\!A$ squares the condition number. Production code uses bidiagonalisation followed by the QR algorithm or divide-and-conquer methods. The above derivation is conceptual.

### Worked example

For $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$:

$$A^{\!\top}\!A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}, \qquad \det(A^{\!\top}\!A - \lambda I) = \lambda^2 - 3\lambda + 1.$$

So $\lambda = \frac{3 \pm \sqrt{5}}{2}$, giving
$$\sigma_1 = \sqrt{\tfrac{3+\sqrt 5}{2}} \approx 1.618, \qquad \sigma_2 = \sqrt{\tfrac{3-\sqrt 5}{2}} \approx 0.618.$$

Find the eigenvectors of $A^{\!\top}\!A$ to assemble $V$, then $u_i = A v_i / \sigma_i$ gives $U$. (The product $\sigma_1 \sigma_2 = 1 = |\det A|$ is a useful sanity check.)

### Eigenvalues vs singular values: side by side

For symmetric $A$, eigenvalues and singular values agree (up to signs). For general $A$ they do not, and the contrast is illuminating.

![Eigenvectors are not orthogonal; singular vectors are. Eigenvalues describe invariant directions; singular values describe stretching](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig3_eig_vs_svd.png)

| | Eigenvectors | Singular vectors |
|---|---|---|
| Defined by | $A x = \lambda x$ | $A v = \sigma u$ with $u \perp$ each other |
| Orthogonality | not in general | always (both $V$ and $U$ are orthonormal) |
| Values | $\lambda_i \in \mathbb{C}$ | $\sigma_i \in \mathbb{R}_{\ge 0}$ |
| Geometric story | invariant directions, scaled by $\lambda$ | input axes mapped to output axes, scaled by $\sigma$ |
| Exists for | square matrices, often diagonalisable | every matrix |

The figure makes the difference visceral: the same non-symmetric $A$ has eigenvectors at an oblique angle, but the SVD picks a perpendicular pair on each side.

---

## SVD and the Four Fundamental Subspaces

SVD gives the cleanest possible picture of a matrix's structure. For $A \in \mathbb{R}^{m \times n}$ with rank $r$:

| Subspace | Dimension | Orthonormal basis from SVD | Lives in |
|---|---|---|---|
| Row space $\mathcal{C}(A^{\!\top})$ | $r$ | $v_1, \ldots, v_r$ | $\mathbb{R}^n$ |
| Null space $\mathcal{N}(A)$ | $n - r$ | $v_{r+1}, \ldots, v_n$ | $\mathbb{R}^n$ |
| Column space $\mathcal{C}(A)$ | $r$ | $u_1, \ldots, u_r$ | $\mathbb{R}^m$ |
| Left null space $\mathcal{N}(A^{\!\top})$ | $m - r$ | $u_{r+1}, \ldots, u_m$ | $\mathbb{R}^m$ |

The orthonormal basis $\{v_1, \ldots, v_r\}$ of the row space is mapped onto the orthonormal basis $\{u_1, \ldots, u_r\}$ of the column space, with each direction stretched by its $\sigma_i$. Everything in the null space is sent to zero. That is the whole action of $A$ in one sentence.

---

## Low-Rank Approximation: the Theorem Behind Compression

### The Eckart--Young theorem

Truncate the outer-product expansion at $k$ terms:
$$A_k = \sigma_1 u_1 v_1^{\!\top} + \cdots + \sigma_k u_k v_k^{\!\top}.$$

**Theorem (Eckart--Young, 1936).** Among all matrices $B$ of rank at most $k$,
$$\|A - A_k\|_F \;=\; \min_{\operatorname{rank}(B) \le k} \|A - B\|_F \;=\; \sqrt{\sigma_{k+1}^{\,2} + \cdots + \sigma_r^{\,2}}.$$
The same statement holds in the operator (2-)norm with $\|A - A_k\|_2 = \sigma_{k+1}$.

So $A_k$ isn't just *a* low-rank approximation -- it is **provably optimal**. No clever rank-$k$ matrix can do better.

**MP3 analogy.** MP3 compression discards high-frequency components the human ear barely registers. SVD truncation does the same thing for matrices: discard the components carrying the least "energy," keep the loud ones.

### Layer by layer

It helps to *see* the layers stack up. Each $\sigma_i u_i v_i^{\!\top}$ is a single rank-1 image; partial sums approach the original.

![Rank-1 layers stack to rebuild an image; early layers carry most of the energy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig4_low_rank_blocks.png)

Top row: the first three rank-1 layers themselves (positive in red, negative in blue) plus the original. Bottom row: the cumulative sums after 1, 2, 3 layers, with the singular value bar chart on the right. Even three components capture a recognisable amount of the picture.

### Energy

Define the matrix's "energy" as its squared Frobenius norm:
$$\|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2.$$
The fraction of energy retained by the rank-$k$ approximation is
$$\text{energy retained} = \frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_r^2}.$$
For most natural data the spectrum decays quickly: a $1000 \times 1000$ photograph might capture 95% of its energy in the first 50 singular values.

### Image compression in numbers

Storing a rank-$k$ approximation costs $k$ singular values plus the first $k$ columns of $U$ and $V$:
$$\text{numbers stored} = k\,(m + n + 1).$$
For a $500 \times 500$ image with $k = 50$: the original needs $250{,}000$ numbers, the rank-50 approximation needs $50{,}050$ -- a 5x reduction with usually imperceptible loss.

![Original vs k=5, 20, 50; spectrum and cumulative-energy curves](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig5_image_compression.png)

The bottom-left log-scale plot is the most important diagnostic in all of applied SVD: it tells you how aggressively you can truncate before quality collapses.

```python
import numpy as np

def compress(img, k):
    """Rank-k SVD approximation of a 2D array."""
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Diagnostics: how much energy do you keep at each k?
U, s, Vt = np.linalg.svd(img, full_matrices=False)
for k in [5, 20, 50, 100]:
    energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
    print(f"k={k:>3}: {energy:.1f}% energy retained")
```

---

## The Pseudoinverse

### When the inverse does not exist

For $A x = b$, we usually want $x = A^{-1} b$. But $A^{-1}$ exists only when $A$ is square and full-rank. The **Moore--Penrose pseudoinverse** $A^{+}$ supplies a universal "best alternative."

### Definition via SVD

If $A = U \Sigma V^{\!\top}$, then
$$A^{+} = V \Sigma^{+} U^{\!\top}, \qquad
\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0,\\ 0 & \sigma_i = 0,\end{cases}$$
and $\Sigma^{+}$ is transposed to have shape $n \times m$. When $A$ is invertible, $A^{+} = A^{-1}$.

### What $A^{+}$ does

For any $b$, the vector $\hat x = A^{+} b$ is

1. a **least-squares solution**: it minimises $\|A x - b\|_2$;
2. among all least-squares solutions, the one with **minimum norm** $\|x\|_2$.

The two cases:

- **Overdetermined** ($m > n$): typically no exact solution. $A^{+} b$ delivers the least-squares fit.
- **Underdetermined** ($m < n$): infinitely many solutions. $A^{+} b$ picks the shortest.

![Geometry of the pseudoinverse: A^+ b projects b onto col(A); the residual is orthogonal to col(A)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig6_pseudoinverse.png)

The left panel shows the canonical least-squares line fit; the right panel shows the geometry that *makes* it least-squares: the projection of $b$ onto the column space of $A$, with the residual orthogonal to it. That orthogonality is the **normal equation**, and SVD gives it for free.

```python
import numpy as np

# Least-squares line fit y = a x + b via the pseudoinverse
x = np.linspace(-2, 4, 25)
y = 1.2 * x + 0.4 + np.random.default_rng(0).normal(0, 1, 25)

A = np.column_stack([x, np.ones_like(x)])
coef = np.linalg.pinv(A) @ y          # uses SVD internally
print(f"slope={coef[0]:.3f}  intercept={coef[1]:.3f}")
```

---

## PCA via SVD

### The connection

Principal Component Analysis is SVD wearing a statistics hat.

Centre the data matrix $X \in \mathbb{R}^{n\times p}$ (each column has mean zero). Compute its SVD:
$$X_c = U \Sigma V^{\!\top}.$$
Then:

- **Principal directions** = columns of $V$ (right singular vectors). These are the orthogonal axes of maximum variance.
- **Principal scores** = $X_c V = U \Sigma$. These are the data re-expressed in the new basis.
- **Variance along PC$_i$** = $\sigma_i^2 / (n - 1)$.
- **Dimensionality reduction to $k$ components**: $X_k = U_k \Sigma_k$.

### Why PCA works

The first principal direction maximises $\operatorname{Var}(X_c w)$ over unit vectors $w$. A short calculation shows this is $w^{\!\top}\!(X_c^{\!\top} X_c)\,w / (n-1)$, which is maximised at the top eigenvector of $X_c^{\!\top} X_c$ -- exactly the top right singular vector $v_1$.

![Centred data with principal axes; histogram of PC1 scores](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/09-singular-value-decomposition/fig7_pca_via_svd.png)

The dashed ellipse is the 1-$\sigma$ Gaussian fit. The two arrows are the principal axes drawn proportional to the standard deviations $\sigma_i / \sqrt{n-1}$. Projecting onto PC1 collapses 2D to 1D while preserving the lion's share of variance.

```python
import numpy as np

def pca(X, k):
    """PCA via SVD. Returns (scores, components, explained_variance_ratio)."""
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                    # k x p
    scores = Xc @ components.T             # n x k
    var = (s ** 2) / (X.shape[0] - 1)
    return scores, components, (var / var.sum())[:k]
```

---

## Recommender Systems

### The setup

Netflix, Amazon, and Spotify all face the same question: **how do we predict ratings for items a user has not yet seen?** The user--item rating matrix $R \in \mathbb{R}^{m\times n}$ is enormous and mostly missing.

### Matrix factorisation

The modelling assumption is that ratings are driven by a small number of **latent factors** -- for movies, perhaps "action level," "romance," "humour," "art-house depth." Then
$$R \approx U_k \Sigma_k V_k^{\!\top}.$$

- A row of $U_k \Sigma_k$ is one user's taste vector.
- A row of $V_k$ is one item's characteristic vector.
- The predicted rating is just their dot product.

This idea -- learn $U_k$ and $V_k$ to fit the *observed* entries, then read off predictions for the rest -- powered the winning entries of the Netflix Prize (2006--2009).

---

## SVD vs. Eigendecomposition

| Property | Eigendecomposition | SVD |
|---|---|---|
| Applicable to | square (often symmetric) | any matrix |
| Form | $A = P\Lambda P^{-1}$ | $A = U\Sigma V^{\!\top}$ |
| Values | eigenvalues; can be negative or complex | singular values; non-negative real |
| Vectors | eigenvectors; not always orthogonal | singular vectors; always orthogonal |
| Geometric story | invariant directions + scaling | rotate + stretch + rotate |
| Always exists? | no | yes |

### Why SVD earns "crown jewel" status

- **Universal** -- works on any matrix, square or not, full rank or not.
- **Stable** -- numerically robust; the gold standard for rank, conditioning, and least squares.
- **Optimal** -- gives provably best low-rank approximations (Eckart--Young).
- **Insightful** -- exposes rank, the four subspaces, and the operator norm in one shot.
- **Practical** -- image compression, NLP, recommender systems, denoising, control, statistics.

---

## Other Applications

### Latent Semantic Analysis (LSA)

Build a document--term matrix (rows = documents, columns = vocabulary, entries = TF-IDF scores). Apply SVD and keep the top $k$ components. The right singular vectors play the role of "latent topics," and document similarity becomes a cosine similarity in this much smaller space.

### Signal denoising

Low-rank signal + full-rank noise: SVD the observation, keep the large singular values, discard the small ones, reconstruct. This is the workhorse trick behind everything from astronomical image cleaning to seismic-data processing.

### Eigenfaces

Run PCA on a database of aligned face images. The principal components ("eigenfaces") form a basis of typical facial appearance. Any new face is a linear combination of eigenfaces, and recognition reduces to comparing coefficient vectors.

---

## Python Implementation

### Manual SVD via eigendecomposition

```python
import numpy as np

def svd_via_eigen(A):
    """Conceptual SVD via eigendecomposition of A^T A.
    Use np.linalg.svd in production -- this is for teaching only."""
    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)            # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvals, V = eigvals[idx], V[:, idx]

    sigma = np.sqrt(np.maximum(eigvals, 0.0))
    r = int((sigma > 1e-10).sum())
    U = (A @ V[:, :r]) / sigma[:r]              # broadcasts column-wise
    return U, sigma[:r], V[:, :r].T

A = np.array([[3.0, 2.0], [2.0, 3.0]])
U, s, Vt = svd_via_eigen(A)
print("singular values:", s)
print("reconstruction:\n", U @ np.diag(s) @ Vt)
```

### Image compression demo

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_compression(img, ks):
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("original"); axes[0].axis("off")
    for ax, k in zip(axes[1:], ks):
        rec = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
        energy = (s[:k] ** 2).sum() / (s ** 2).sum() * 100
        ax.imshow(rec, cmap="gray")
        ax.set_title(f"k={k}  ({energy:.0f}%)"); ax.axis("off")
    plt.tight_layout(); plt.show()

# plot_compression(your_grayscale_image, [5, 20, 50, 100])
```

---

## Exercises

### Warm-up

1. Explain why singular values are always non-negative, while eigenvalues can be negative or complex.
2. Compute the SVD of $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ by hand.
3. If $A$ is $3 \times 5$, what are the shapes of $U$, $\Sigma$, and $V^{\!\top}$ in the full SVD? In the economy SVD?

### Going deeper

4. Prove that $\operatorname{rank}(A)$ equals the number of nonzero singular values.
5. Prove $\|A\|_F^2 = \sigma_1^2 + \cdots + \sigma_r^2$.
6. If $Q$ is orthogonal, what are its singular values? Why?
7. Show that $U_r U_r^{\!\top}$ is the projection matrix onto the column space, and $V_r V_r^{\!\top}$ is the projection onto the row space.
8. Prove the operator-norm version of Eckart--Young: $\|A - A_k\|_2 = \sigma_{k+1}$.

### Coding challenges

9. **Compression curve.** Load a grayscale image, compute its SVD, and plot rank-$k$ reconstructions for $k \in \{5, 20, 50, 100\}$ together with the singular-value decay on a log scale.
10. **PCA on Iris.** Apply PCA to the Iris dataset. Plot the first two components, colour-coded by species, and report the explained-variance ratio.
11. **Toy recommender.** Build a $5 \times 10$ rating matrix with some entries missing, fill missing values with the row mean, run a rank-3 SVD, and inspect the predicted ratings.

---

## Chapter Summary

| Concept | Key formula | Intuition |
|---|---|---|
| SVD | $A = U\Sigma V^{\!\top}$ | rotate + stretch + rotate |
| Singular values | $\sigma_i \ge 0$ | stretching factors of the ellipse |
| Outer-product form | $A = \sum_i \sigma_i u_i v_i^{\!\top}$ | weighted sum of rank-1 layers |
| Low-rank approximation | $A_k = U_k \Sigma_k V_k^{\!\top}$ | optimal rank-$k$ matrix (Eckart--Young) |
| Pseudoinverse | $A^{+} = V \Sigma^{+} U^{\!\top}$ | minimum-norm least-squares solution |
| PCA | SVD of centred $X$ | maximum-variance directions = right singular vectors |

---

## Series Navigation

**Previous:** [Chapter 8 -- Symmetric Matrices and Quadratic Forms](/en/chapter-08-symmetric-matrices-and-quadratic-forms/)

**Next:** [Chapter 10 -- Matrix Norms and Condition Numbers](/en/chapter-10-matrix-norms-and-condition-numbers/)

*This is Chapter 9 of the 18-part "Essence of Linear Algebra" series.*

## References

- Strang, G. (2019). *Introduction to Linear Algebra*, Chapter 7.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. Johns Hopkins.
- Eckart, C. & Young, G. (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*, 1(3).
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Koren, Y., Bell, R. & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." *Computer*, 42(8).
- 3Blue1Brown. *Essence of Linear Algebra* series.
