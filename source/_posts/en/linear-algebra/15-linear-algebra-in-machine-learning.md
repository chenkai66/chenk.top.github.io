---
title: "Essence of Linear Algebra (15): Linear Algebra in Machine Learning"
date: 2025-03-12 09:00:00
tags:
  - Linear Algebra
  - PCA
  - SVM
  - machine learning
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 15
  total: 18
lang: en
mathjax: true
description: "Machine learning speaks linear algebra as its native language. From PCA to SVMs, from matrix factorization in recommender systems to gradient descent optimization -- see how vectors, matrices, and decompositions power every core ML algorithm."
disableNunjucks: true
series_order: 15
---

Ask any senior ML engineer "what math do you actually use day to day?" and the answer is almost always **linear algebra**. Calculus shows up in derivations; probability shows up in modeling; but the runtime of a real ML system is dominated by matrix-vector multiplies, decompositions, and projections. PyTorch's `Linear`, scikit-learn's `PCA`, Spark MLlib's `ALS`, and a Transformer's attention head are all the same primitive in different costumes.

This chapter walks through the algorithms that production ML systems actually run -- PCA, LDA, SVM with kernels, matrix factorization for recommenders, regularized linear regression, neural network layers, attention -- and shows the linear algebra that makes each of them tick. We focus on intuition first, geometry second, formulas third.

> **What you will learn:**
> - How raw data (images, text, behavior) becomes vectors, and why feature spaces are geometric
> - PCA via eigendecomposition and SVD, plus kernel PCA for nonlinear manifolds
> - LDA for supervised dimensionality reduction and why PCA can fail at it
> - SVMs, the kernel trick, and the Mercer condition that licenses it
> - Matrix factorization for recommender systems (ALS, NMF) and the cold-start problem
> - Matrix form of linear regression, ridge, and LASSO, with the SVD view
> - Why neural network layers, batches, and attention are all matrix multiplications
>
> **Prerequisites:** Eigendecomposition (Chapter 6), SVD (Chapter 9), matrix norms (Chapter 10), matrix calculus basics (Chapter 11).

---

## 1. Vector Representations: How Data Enters the Pipeline

### 1.1 Everything becomes a vector

Before any model can learn, real-world objects must be embedded in $\mathbb{R}^p$:

- A $28 \times 28$ MNIST digit becomes a 784-dimensional vector by flattening pixels.
- A product review becomes a sparse 30,000-dimensional TF-IDF vector or a dense 768-dimensional BERT embedding.
- A user's last 30 days of clicks becomes a feature vector with counts, recency decays, and one-hot category indicators.
- A protein sequence becomes a 1280-dimensional ESM embedding.

The reason is simple: **vectors are the atom of every linear algebra operation.** Once data is vectorized, you can take inner products (similarity), compute distances (k-NN), apply matrices (linear transforms), and decompose covariances (PCA). Without vectorization, you have to write a custom algorithm per data type; with it, the same SVD routine handles photos and protein sequences.

### 1.2 Geometry of feature space

Stack $n$ samples (each $p$-dimensional) row-wise into the **design matrix**:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_n^\top \end{bmatrix} \in \mathbb{R}^{n \times p}
$$

Each row is a point in $\mathbb{R}^p$; each column is a feature observed across all samples. Almost every ML task can be phrased geometrically in this space:

| Task | Geometric statement |
|---|---|
| Classification | Find a hyperplane (or curved surface) separating classes |
| Clustering | Find dense regions of points |
| Dimensionality reduction | Project onto a lower-dimensional subspace |
| Regression | Find a hyperplane closest to all points |
| Anomaly detection | Find points far from the data manifold |

This geometric framing is more than aesthetic: it tells you which algorithm to reach for. "Find the direction of maximum spread" is PCA. "Find the direction that splits two clouds" is LDA. "Find the surface that classes lie on opposite sides of" is an SVM.

### 1.3 Word embeddings: when geometry encodes meaning

Word2Vec, GloVe, and the embedding layers of modern LLMs map tokens to dense vectors so that semantically related words sit close together. The remarkable empirical finding is that simple **vector arithmetic** captures relations:

$$
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})
$$

The "gender" relation is approximately a constant direction; "royalty" is another. Parallelogram analogies work because relations are encoded as **directions** in the embedding space, not as features in any single coordinate.

![Word embedding analogy: parallel directions encode relations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig7_word_embeddings.png)

```python
import numpy as np
from gensim.models import KeyedVectors

# Pre-trained Google News vectors (300-d).
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)

# Analogy: king is to man as ? is to woman.
print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))
# [('queen', 0.7118)]

# Cosine similarity is the natural metric in embedding space.
print(model.similarity('cat', 'dog'))   # 0.76
print(model.similarity('cat', 'galaxy'))# 0.04
```

Caveat: real embeddings are 300- to 4096-dimensional and the analogy property is statistical, not exact. The 2D picture above is a stylized illustration.

---

## 2. Principal Component Analysis (PCA)

### 2.1 The intuition

Throw a handful of paper clips on a table, then look down. They are clearly elongated along some direction. PCA finds that direction automatically: **the axis of greatest variance**. Why care about variance? Because variance encodes information. A feature direction with near-zero variance is essentially constant -- it cannot help any downstream task.

![PCA: principal axes of a 2D data cloud](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig1_pca_projection.png)

The figure shows a correlated 2D cloud, the two principal axes drawn at lengths proportional to $\sqrt{\text{variance}}$, and the 1D projection onto PC1. The orange axis (PC1) explains roughly 95% of the spread; throwing away PC2 loses very little structure.

### 2.2 Derivation

Center the data so $\mathbf{X}^\top\mathbf{1}=\mathbf{0}$. We want a unit vector $\mathbf{u}$ that maximizes the variance of projected coordinates:

$$
\text{Var}(\mathbf{u}) = \frac{1}{n}\sum_{i=1}^n (\mathbf{u}^\top \mathbf{x}_i)^2 = \mathbf{u}^\top \mathbf{C} \mathbf{u}, \qquad \mathbf{C} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}.
$$

Maximizing $\mathbf{u}^\top \mathbf{C}\mathbf{u}$ subject to $\|\mathbf{u}\|=1$ is solved by Lagrange multipliers, which yields $\mathbf{C}\mathbf{u}=\lambda\mathbf{u}$. **The optimal $\mathbf{u}$ is the top eigenvector of the covariance matrix.** The variance along that direction equals the corresponding eigenvalue $\lambda$. Subsequent components are the next eigenvectors, which are mutually orthogonal because $\mathbf{C}$ is symmetric.

### 2.3 PCA via SVD: what production code actually runs

Forming $\mathbf{X}^\top\mathbf{X}$ is wasteful when $p$ is large (think 4096-dim BERT embeddings) and numerically harmful when features are nearly collinear (the condition number squares). Real implementations use the SVD of the centered data matrix:

$$
\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top.
$$

Columns of $\mathbf{V}$ are the principal directions; the projected coordinates are $\mathbf{U}\boldsymbol{\Sigma}$; per-component variances are $\sigma_i^2/n$.

```python
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(42)
X = np.random.randn(200, 2) @ np.array([[2, 0], [0, 0.5]])
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X = X @ R

# scikit-learn (SVD under the hood for n_components < min(n, p)).
pca = PCA(n_components=2).fit(X)
print(pca.components_)              # principal directions, rows are PCs
print(pca.explained_variance_ratio_)# fraction of total variance per PC

# Equivalent manual computation.
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print(Vt)                # same up to sign
print(S**2 / len(X))     # same variances
```

### 2.4 Where PCA actually shows up in production

- **Visualization**: project 768-d sentence embeddings to 2-D for an exploratory plot.
- **Compression**: keep top 50 PCs of MNIST (originally 784-d) before training a small classifier. The accuracy hit is negligible and training is 10x faster.
- **Denoising**: in finance, the leading PCs of a returns matrix capture market and sector factors; the tail is mostly noise (this is the spiked-covariance picture from Chapter 14).
- **Whitening for downstream models**: `PCA(whiten=True)` decorrelates and rescales features so iterative optimizers converge in fewer steps.
- **Eigenfaces**: an early face-recognition system that stored each face as its 100-coefficient PCA representation.

A tip from experience: always center, and almost always standardize before PCA when features are on different scales (e.g., age in years, income in dollars). Otherwise the largest-variance direction is just "the column with the largest unit."

### 2.5 Kernel PCA: when the manifold is curved

Linear PCA finds a *flat* subspace. If your data lives on a Swiss-roll-like manifold, no linear projection unfolds it. Kernel PCA conceptually maps each point through a nonlinear feature map $\phi(\mathbf{x})$ and runs PCA in feature space. Crucially, by the kernel trick (next section) you never compute $\phi$ explicitly -- only inner products $k(\mathbf{x}_i, \mathbf{x}_j)$.

```python
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

X_pca  = PCA(n_components=2).fit_transform(X)
X_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10).fit_transform(X)
# X_kpca neatly separates the two moons; X_pca does not.
```

Kernel PCA has largely been displaced by autoencoders and t-SNE / UMAP for visualization, but it remains a clean illustration of the kernel idea on an unsupervised problem.

---

## 3. Linear Discriminant Analysis (LDA)

### 3.1 PCA's blind spot, fixed by labels

PCA is unsupervised: it ignores class labels and chases variance. That can be exactly wrong for classification. Imagine two long, parallel cigar-shaped clouds for classes A and B. PCA picks the direction along the cigars (max variance), which is the direction that **mixes** A and B. The right direction is perpendicular to the cigars -- the one that splits them.

LDA is supervised. It looks for projections that simultaneously:

1. Push class **centers** as far apart as possible (large between-class scatter).
2. Keep each class **tight** around its own center (small within-class scatter).

![PCA vs LDA: variance is not the same as class separability](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig6_lda_separation.png)

The figure makes the failure mode concrete: PCA's arrow runs along the elongation of the clouds, projecting both classes on top of each other; LDA's arrow runs across the gap, and the bottom-of-panel histograms show clean class separation only on the right.

### 3.2 The math

With $C$ classes (class $c$ has $n_c$ samples, mean $\boldsymbol{\mu}_c$, overall mean $\boldsymbol{\mu}$):

$$
\mathbf{S}_W = \sum_{c=1}^{C}\sum_{\mathbf{x}\in c}(\mathbf{x}-\boldsymbol{\mu}_c)(\mathbf{x}-\boldsymbol{\mu}_c)^\top, \qquad \mathbf{S}_B = \sum_{c=1}^{C} n_c(\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^\top.
$$

LDA maximizes the **generalized Rayleigh quotient**:

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}, \qquad \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}.
$$

For binary classification this collapses to the **Fisher discriminant**:

$$
\mathbf{w}^* \;\propto\; \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2).
$$

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
# Iris (3 classes) reduces cleanly to 2 LDA dimensions.
```

### 3.3 PCA vs LDA at a glance

| Property | PCA | LDA |
|---|---|---|
| Supervision | Unsupervised | Supervised |
| Objective | Maximize variance | Maximize between/within-class ratio |
| Max output dimensions | $\min(n, p)$ | $C - 1$ |
| Typical use | Exploration, compression, denoising | Pre-classification feature shaping |
| Assumptions | Linear structure | Gaussian per class, shared covariance |

The hard cap $C-1$ is often surprising: for binary problems LDA outputs just **one** dimension, the Fisher direction. That is enough for a linear classifier but useless for visualization beyond a 1-D histogram.

---

## 4. Support Vector Machines and the Kernel Trick

### 4.1 Why "maximum margin"?

Many hyperplanes separate two linearly separable classes. SVMs pick the unique one whose margin -- the distance to the nearest training point on either side -- is maximal. The intuition: a wide margin leaves the most room for a new test point to fall on the correct side, so it generalizes better. This intuition is formalized by VC-dimension and PAC-Bayes bounds.

![SVM: maximum-margin hyperplane and support vectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig3_svm_margin.png)

The decision boundary is the central solid line; the dashed amber lines are the margin; the green-circled points sitting **on** the margin are the support vectors. Only those points determine the solution -- delete any non-support-vector and the boundary doesn't move.

### 4.2 The dual is where kernels enter

**Primal:** $\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1.$

**Dual** (after Lagrange multipliers $\alpha_i \ge 0$):

$$
\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \,\mathbf{x}_i^\top \mathbf{x}_j, \qquad \sum_i \alpha_i y_i = 0.
$$

The dual involves data only through the **inner products** $\mathbf{x}_i^\top\mathbf{x}_j$. Replace those inner products with any function $k(\mathbf{x}_i, \mathbf{x}_j)$ and you have implicitly replaced the input space with a different (possibly higher-dimensional) feature space. That is the kernel trick.

### 4.3 Lifting to higher dimensions

If the classes form concentric rings in 2D, no straight line separates them. But map every point through $\phi(x_1, x_2) = (x_1, x_2, x_1^2 + x_2^2)$ and the inner ring sits low in the new $z$-coordinate while the outer ring sits high -- a flat plane separates them.

![Kernel trick: lifting non-separable rings into linearly separable 3D](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig4_kernel_trick.png)

In general the lifted space can be much higher-dimensional, but with a kernel function we never represent it explicitly. We only ever evaluate the inner-product function.

**Common kernels:**

- **Linear:** $k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top\mathbf{y}$
- **Polynomial of degree $d$:** $k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^\top\mathbf{y} + c)^d$
- **RBF (Gaussian):** $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x} - \mathbf{y}\|^2)$ -- corresponds to an *infinite-dimensional* feature space, yet evaluates in $O(p)$.
- **Sigmoid:** $k(\mathbf{x}, \mathbf{y}) = \tanh(\kappa \mathbf{x}^\top\mathbf{y} + c)$ -- not always positive definite; used historically for "neural" SVMs.

### 4.4 The Mercer condition

A function $k$ is a valid kernel iff for any finite sample, the Gram matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is **symmetric positive semidefinite**. This is **Mercer's condition**, and it is exactly the condition that guarantees the existence of some feature map $\phi$ with $k(\mathbf{x}, \mathbf{y}) = \langle\phi(\mathbf{x}), \phi(\mathbf{y})\rangle$.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
print(SVC(kernel='linear').fit(X, y).score(X, y))  # ~0.55
print(SVC(kernel='rbf', gamma=2).fit(X, y).score(X, y))  # ~1.00
```

In modern practice SVMs have been displaced by neural networks for large-scale classification, but RBF-SVMs remain the default for small (a few thousand sample) tabular problems where they routinely match or beat gradient-boosted trees.

---

## 5. Matrix Factorization for Recommender Systems

### 5.1 The Netflix problem

A user-movie rating matrix $\mathbf{R} \in \mathbb{R}^{m\times n}$ is enormous (tens of millions of users, hundreds of thousands of titles) and almost entirely missing -- a typical user rates 0.1% of the catalogue. The goal is to **predict the unobserved cells** so we can recommend the highest-predicted unseen items to each user. The Netflix Prize (2006-2009) established matrix factorization as the workhorse approach, and variants of it still power Spotify, YouTube, and every major e-commerce recommender.

### 5.2 The low-rank assumption

Suppose user taste and movie content can each be summarized by $k$ latent factors -- say, "amount of action," "amount of romance," "indie-vs-blockbuster," and so on. Then the rating user $u$ would give movie $j$ is well approximated by the inner product of their factor vectors:

$$
\hat{r}_{uj} = \mathbf{p}_u^\top \mathbf{q}_j, \qquad \mathbf{R} \approx \mathbf{P}\mathbf{Q}^\top, \quad \mathbf{P} \in \mathbb{R}^{m\times k}, \mathbf{Q} \in \mathbb{R}^{n\times k}.
$$

This is exactly a rank-$k$ approximation of $\mathbf{R}$. With $m=10^7, n=10^5, k=64$, the factor matrices store $\sim 6\times 10^8$ numbers instead of $10^{12}$ -- and they generalize to unseen entries.

![Collaborative filtering as low-rank matrix factorization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig5_matrix_factorization.png)

The figure shows the sparse observed matrix (with `?` for unrated cells), the two factor matrices, and the dense recovered prediction.

### 5.3 Alternating Least Squares (ALS)

We can't run plain SVD because most entries are missing. Instead, optimize over **observed** entries only, with regularization:

$$
\min_{\mathbf{P}, \mathbf{Q}} \sum_{(u,j)\in\Omega}\bigl(r_{uj} - \mathbf{p}_u^\top\mathbf{q}_j\bigr)^2 + \lambda\bigl(\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2\bigr).
$$

This is non-convex jointly in $(\mathbf{P}, \mathbf{Q})$, but **convex when one factor is fixed**. ALS alternates: with $\mathbf{Q}$ fixed, each row of $\mathbf{P}$ is the closed-form solution of an independent ridge regression; then swap. Each pass touches every observation once, and the per-row solves parallelize trivially -- which is why ALS dominates Spark MLlib.

```python
import numpy as np

def als(R, mask, k=10, n_iter=20, lam=0.1):
    m, n = R.shape
    P = np.random.randn(m, k) * 0.1
    Q = np.random.randn(n, k) * 0.1
    I = lam * np.eye(k)
    for _ in range(n_iter):
        for u in range(m):
            j = np.where(mask[u])[0]
            if j.size:
                Qj = Q[j]
                P[u] = np.linalg.solve(Qj.T @ Qj + I, Qj.T @ R[u, j])
        for j in range(n):
            u = np.where(mask[:, j])[0]
            if u.size:
                Pu = P[u]
                Q[j] = np.linalg.solve(Pu.T @ Pu + I, Pu.T @ R[u, j])
    return P, Q
```

### 5.4 Cold-start, biases, and beyond

Pure matrix factorization fails on **cold-start**: a brand-new user with no ratings has no factor vector. Production systems patch this by:

- Adding **side features** (demographics, content metadata) into a hybrid model: $\hat{r}_{uj} = \mathbf{p}_u^\top\mathbf{q}_j + \mathbf{a}^\top \mathbf{f}_u + \mathbf{b}^\top \mathbf{g}_j$.
- Including **user and item biases**: some users rate generously, some movies are universally beloved. Writing $\hat{r}_{uj} = \mu + b_u + b_j + \mathbf{p}_u^\top\mathbf{q}_j$ measurably improves accuracy.
- Switching to **neural CF** (two-tower models) where users and items are embedded by neural networks trained jointly. The inner-product head is still linear algebra.

### 5.5 Non-negative Matrix Factorization (NMF)

When you want each factor to be interpretable as an additive part rather than a signed combination, constrain $\mathbf{P}, \mathbf{Q} \ge 0$. NMF tends to produce **sparse, parts-based** representations: given a corpus of news articles, NMF factors look like topics ("sports," "politics," "tech") with non-negative word loadings. SVD on the same matrix would give you signed factors that mix topics together.

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=2000, stop_words='english')
X   = vec.fit_transform(corpus)               # docs x words, non-negative
nmf = NMF(n_components=10, init='nndsvd').fit(X)
W = nmf.transform(X)        # docs x topics
H = nmf.components_         # topics x words
top = np.argsort(H, axis=1)[:, -10:]
```

---

## 6. Linear Regression in Matrix Form

### 6.1 The model

Multivariate linear regression $y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \epsilon$ stacked over $n$ observations:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \qquad \mathbf{X} \in \mathbb{R}^{n\times(p+1)}.
$$

Here $\mathbf{X}$ is the **design matrix** with a leading column of ones (the intercept).

### 6.2 Least squares as orthogonal projection

We minimize the residual sum of squares $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$. Geometrically, $\mathbf{X}\boldsymbol{\beta}$ ranges over the **column space** of $\mathbf{X}$; we want the point in that subspace closest to $\mathbf{y}$. Closest = perpendicular foot = orthogonal projection.

![Least squares as orthogonal projection of y onto col(X)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/15-linear-algebra-in-machine-learning/fig2_regression_projection.png)

The condition that the residual $\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$ be orthogonal to col$(\mathbf{X})$ gives the **normal equations** $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$, i.e.

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

In code, never form the inverse; use `np.linalg.lstsq(X, y)` (which calls a stable QR/SVD-based solver) or build $\mathbf{X} = \mathbf{Q}\mathbf{R}$ explicitly and back-substitute $\mathbf{R}\hat{\boldsymbol{\beta}} = \mathbf{Q}^\top\mathbf{y}$.

### 6.3 Ridge regression: shrinkage as conditioning

If $\mathbf{X}^\top\mathbf{X}$ is ill-conditioned (collinear features, $p > n$, etc.) the OLS solution is unstable. Ridge adds an $\ell_2$ penalty:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

Adding $\lambda \mathbf{I}$ shifts every eigenvalue of $\mathbf{X}^\top\mathbf{X}$ up by $\lambda$, guaranteeing invertibility and shrinking the condition number from $\sigma_{\max}^2/\sigma_{\min}^2$ to roughly $(\sigma_{\max}^2 + \lambda)/(\sigma_{\min}^2 + \lambda)$. Through the SVD lens, ridge multiplies each singular component of the OLS solution by $\sigma_i^2/(\sigma_i^2 + \lambda)$ -- shrinking small-$\sigma$ (noisy) directions hard while leaving big-$\sigma$ (signal) directions almost unchanged.

### 6.4 LASSO: sparsity through $\ell_1$

LASSO replaces the $\ell_2$ penalty with $\ell_1$:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1.
$$

The $\ell_1$ ball has corners along the coordinate axes; the OLS contour generically touches the ball at one of those corners, where some coordinates are exactly zero. The solver **selects features automatically** -- a huge usability win for sparse, high-dimensional problems (genetics, NLP, high-frequency trading signals).

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
beta_true = np.zeros(p); beta_true[:3] = [3, -2, 1.5]   # only 3 active
y = X @ beta_true + 0.5 * np.random.randn(n)

ols   = LinearRegression(fit_intercept=False).fit(X, y)
ridge = Ridge(alpha=1.0,  fit_intercept=False).fit(X, y)
lasso = Lasso(alpha=0.10, fit_intercept=False).fit(X, y)

print("OLS   nonzero:", (np.abs(ols.coef_)   > 1e-2).sum())   # 20
print("Ridge nonzero:", (np.abs(ridge.coef_) > 1e-2).sum())   # 20
print("LASSO nonzero:", (np.abs(lasso.coef_) > 1e-2).sum())   # ~3
```

---

## 7. Linear Layers in Neural Networks

### 7.1 A fully connected layer is one matmul

Every "Linear" or "Dense" layer is an affine map composed with a pointwise nonlinearity:

$$
\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}).
$$

Without $\sigma$, stacking layers collapses: $\mathbf{W}_3\mathbf{W}_2\mathbf{W}_1$ is just another matrix. Activations break linearity and let the network represent arbitrary continuous functions (the universal approximation theorem).

### 7.2 Batches turn matvecs into matmuls

GPUs are not magic; they are very fast at one thing: large dense matrix multiplies. The reason deep learning training is GPU-friendly is the batching trick. Stacking $B$ inputs into $\mathbf{X} \in \mathbb{R}^{B\times d_{\text{in}}}$:

$$
\mathbf{H} = \sigma\bigl(\mathbf{X}\mathbf{W}^\top + \mathbf{1}\mathbf{b}^\top\bigr).
$$

A single SGEMM call now does what $B$ separate matvecs would, with much higher FLOP utilization.

### 7.3 Backpropagation is matrix calculus

For the layer $\mathbf{h} = \mathbf{W}\mathbf{x}$ the chain rule gives

$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{h}}\,\mathbf{x}^\top, \qquad \frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{h}}.
$$

Backprop through a deep network is just a sequence of these transposes; PyTorch's autograd is essentially a recorder for the matmuls performed in the forward pass.

```python
import numpy as np

class LinearLayer:
    """A matrix-multiply layer with manual forward and backward."""
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad_out, lr=1e-2):
        grad_W = grad_out.T @ self.x
        grad_b = grad_out.sum(axis=0)
        grad_x = grad_out @ self.W
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_x
```

### 7.4 Attention is also linear algebra

The self-attention block at the heart of every Transformer is

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V},
$$

with $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ -- three linear projections of the same input. The product $\mathbf{Q}\mathbf{K}^\top$ is an $n\times n$ similarity matrix between every pair of sequence positions; softmax converts it into a stochastic matrix; multiplying by $\mathbf{V}$ takes a weighted average. The $O(n^2)$ memory cost of that similarity matrix is exactly what the FlashAttention and linear-attention literature is fighting.

---

## 8. Linear Algebra Foundations of Optimization

### 8.1 Gradient and Hessian

The Hessian $\mathbf{H}$ contains all second partials of $f$. A second-order Taylor expansion near a stationary point reads

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f^\top(\mathbf{x} - \mathbf{x}_0) + \tfrac{1}{2}(\mathbf{x} - \mathbf{x}_0)^\top \mathbf{H}(\mathbf{x} - \mathbf{x}_0).
$$

A symmetric positive-definite Hessian means a true minimum; eigenvalues of $\mathbf{H}$ are the curvatures along their corresponding eigenvector directions.

### 8.2 Conditioning and the zigzag

Gradient descent's convergence rate depends on the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ of $\mathbf{H}$. When $\kappa$ is large, contour lines of $f$ are highly elongated ellipses; gradient vectors point across the ellipses rather than down them, producing the famous zigzag. The convergence rate of vanilla gradient descent on a quadratic is $\bigl((\kappa-1)/(\kappa+1)\bigr)^2$ per step; for $\kappa = 1000$ that means $\approx 0.996$, i.e., almost no progress.

### 8.3 What Adam actually does

Newton's method preconditions by $\mathbf{H}^{-1}$, transforming the geometry so the Hessian becomes the identity (one step solves a quadratic). Computing $\mathbf{H}^{-1}$ for a billion-parameter network is hopeless, so Adam approximates it by a **diagonal** preconditioner -- a per-parameter running estimate of the squared gradient, $\hat{v}_t$. The update $\theta \leftarrow \theta - \eta \,\hat{m}_t / \sqrt{\hat{v}_t}$ rescales each coordinate so that high-curvature directions take small steps and low-curvature directions take big ones, taming the zigzag at $O(p)$ extra cost.

---

## 9. Exercises

### Conceptual Understanding

**Exercise 1.** Explain why PCA's principal components are mutually orthogonal. Hint: spectral theorem for symmetric matrices.

**Exercise 2.** LDA can output at most $C-1$ dimensions. Why? Hint: rank of $\mathbf{S}_B$.

**Exercise 3.** Find an explicit feature map $\phi$ such that $k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^\top\mathbf{y})^2$ equals $\langle\phi(\mathbf{x}), \phi(\mathbf{y})\rangle$ for $\mathbf{x}, \mathbf{y} \in \mathbb{R}^2$.

**Exercise 4.** Use a 2D picture of the unit $\ell_1$ vs $\ell_2$ balls to explain why LASSO produces sparse solutions but ridge does not.

### Computational Problems

**Exercise 5.** Take $\mathbf{X} = \begin{bmatrix}1 & 2\\ -1 & 0\\ 0 & -2\end{bmatrix}$ (rows already centered).
(a) Compute $\mathbf{C} = \tfrac{1}{3}\mathbf{X}^\top\mathbf{X}$.
(b) Find its eigenvalues and eigenvectors.
(c) Project the rows onto the first principal component.

**Exercise 6.** With class means $\boldsymbol{\mu}_1 = [1, 2]^\top, \boldsymbol{\mu}_2 = [3, 4]^\top$ and $\mathbf{S}_W = \mathbf{I}$, compute the LDA direction and the projected centers.

**Exercise 7.** Show that ridge regression on $(\mathbf{X}, \mathbf{y})$ is equivalent to OLS on the augmented data $\tilde{\mathbf{X}} = \begin{bmatrix}\mathbf{X}\\\sqrt{\lambda}\mathbf{I}\end{bmatrix}, \; \tilde{\mathbf{y}} = \begin{bmatrix}\mathbf{y}\\\mathbf{0}\end{bmatrix}$.

### Programming Problems

**Exercise 8.** Implement PCA from scratch in NumPy (centering + SVD), apply it to MNIST, and plot the first two PCs colored by digit.

```python
def my_pca(X, n_components):
    # 1. center 2. svd 3. take top components 4. project
    pass
```

**Exercise 9.** Build a synthetic $100 \times 50$ rank-5 + noise matrix, mask 80% of entries, recover with ALS, and report RMSE on the held-out cells.

**Exercise 10.** Write a two-layer MLP for MNIST in PyTorch using only `torch.matmul` (no `nn.Linear`); train to >97% test accuracy.

### Proof Problems

**Exercise 11.** Prove that a Gram matrix is positive semidefinite iff there exists a feature map $\phi$ with $K_{ij} = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$.

**Exercise 12.** Prove that for centered $\mathbf{X}$, the principal directions equal the right singular vectors of $\mathbf{X}$.

**Exercise 13.** State and prove the Gauss-Markov theorem: under $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$ with $\mathbb{E}[\boldsymbol{\epsilon}] = 0$ and $\text{Cov}(\boldsymbol{\epsilon}) = \sigma^2\mathbf{I}$, the OLS estimator is BLUE.

### Application Problems

**Exercise 14.** Design a recommender for 10,000 users, 1,000 movies, with 50 ratings/user. Pick $k$, justify the choice, propose a cold-start strategy, and define an evaluation protocol.

**Exercise 15.** For 50,000 CIFAR-10 images (3072-d): would you PCA before training a CNN? Before training a logistic regression baseline? Why are CNN feature maps fundamentally different from PCA components?

---

## 10. Chapter Summary

- **Vectorize first.** Every ML pipeline starts by mapping data into $\mathbb{R}^p$. Once there, geometry tells you which algorithm to reach for.
- **PCA = top eigenvectors of the covariance.** Compute via SVD for stability. Kernel PCA generalizes to nonlinear manifolds.
- **LDA = supervised analog.** Maximizes the between-to-within scatter ratio; capped at $C-1$ dimensions.
- **SVMs use only inner products.** That makes the kernel trick possible, and Mercer's condition tells you when a function is a valid kernel.
- **Recommenders are low-rank.** ALS and NMF factor the rating matrix into user and item embeddings; biases and side-features patch cold-start.
- **Linear regression is a projection.** Ridge ($\ell_2$) conditions; LASSO ($\ell_1$) selects features; both are still least-squares with different priors.
- **Neural nets are matmuls + nonlinearities.** Forward and backward pass are matrix multiplications; attention is the same primitive applied to similarity scores.
- **Optimization geometry is the Hessian.** Conditioning controls convergence; Adam is a cheap diagonal preconditioner.

Treat the next ML algorithm you meet as a question: *what subspace, projection, or factorization is hiding in here?* Almost always there is one, and seeing it turns the algorithm from a black box into a one-page derivation.

---

## References

- Strang, G. *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press, 2019.
- Murphy, K. P. *Probabilistic Machine Learning: An Introduction.* MIT Press, 2022.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning.* MIT Press, 2016.
- Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006.
- Koren, Y., Bell, R., & Volinsky, C. *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer 42(8), 2009.
- Mikolov, T. et al. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS, 2013.
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS, 2017.

---

## Series Navigation

- **Previous:** [Chapter 14: Random Matrix Theory](/en/chapter-14-random-matrix-theory/)
- **Next:** [Chapter 16: Linear Algebra in Deep Learning](/en/chapter-16-linear-algebra-in-deep-learning/)
- **Full Series:** Essence of Linear Algebra (1--18)
