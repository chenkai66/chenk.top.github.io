---
title: "ML Math Derivations (18): Clustering Algorithms"
date: 2024-03-18 09:00:00
tags:
  - Machine Learning
  - Clustering
  - K-means
  - DBSCAN
  - Spectral Clustering
  - Hierarchical Clustering
  - GMM
  - Mathematical Derivations
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 18
  total: 20
lang: en
mathjax: true
description: "How do you find groups in unlabeled data? This article derives K-means (Lloyd + K-means++), hierarchical, DBSCAN, spectral, and GMM clustering from their mathematical foundations, with seven figures that show why each algorithm encodes a different prior."
disableNunjucks: true
---
## What This Article Covers

A million customer records arrive with no labels. Can you discover meaningful groups automatically? That is **clustering**, the most fundamental unsupervised learning task. Unlike classification, clustering forces you to first answer a slippery question: *what does "similar" even mean?* Every clustering algorithm is, at heart, a different answer to that question -- a different geometric, probabilistic, or graph-theoretic prior on what a "group" is.

**What you will learn:**

1. **K-means** as coordinate descent on a discrete-continuous objective, why Lloyd's algorithm always converges, and how K-means++ tames the initialization problem.
2. **Hierarchical clustering** as a greedy merge process, and how the choice of linkage controls cluster shape.
3. **DBSCAN** as density-based connectivity, and why it can find non-convex clusters and label noise.
4. **Spectral clustering** as a continuous relaxation of NCut, with the graph Laplacian doing the heavy lifting.
5. **Gaussian Mixture Models** as the probabilistic generalization of K-means -- what you gain (soft assignments, ellipsoidal covariance) and what you pay for it.
6. How to **evaluate** clustering quality without labels (silhouette, elbow) and how to choose $K$.

**Prerequisites:** linear algebra (eigendecomposition), basic probability, and the EM algorithm from [Part 13](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/).

---

## 1. Formalizing the Clustering Problem

### 1.1 What Makes a Good Clustering?

**Input:** dataset $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ with $\mathbf{x}_i \in \mathbb{R}^d$.

**Output:** assignments $\{c_1, \dots, c_N\}$ with $c_i \in \{1, \dots, K\}$.

**Two competing principles:**

- **High cohesion:** points in the same cluster should be similar.
- **Low coupling:** points in different clusters should be dissimilar.

Every objective we will write down is just one way to balance these two. The disagreements between K-means, DBSCAN, and spectral clustering all trace back to *how* they measure "similar".

### 1.2 Distance and Similarity Measures

| Measure | Formula | Best for |
|---------|---------|----------|
| Euclidean | $d(\mathbf{x}, \mathbf{y}) = \\|\mathbf{x} - \mathbf{y}\\|_2$ | dense, low-dimensional data |
| Manhattan | $d(\mathbf{x}, \mathbf{y}) = \sum_j \\|x_j - y_j\\|$ | sparse features, anomaly-resistant |
| Cosine | $\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^T\mathbf{y}}{\\|\mathbf{x}\\|\\|\mathbf{y}\\|}$ | text, high-dimensional sparse vectors |
| Mahalanobis | $d(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{y})}$ | correlated features |

**Practical note:** because every distance is sensitive to feature scale, *standardize before clustering* unless your features are already on comparable scales.

### 1.3 Evaluating Clusters Without Labels

The **silhouette coefficient** for point $i$ is

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \in [-1, 1]$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean distance to the *nearest other* cluster. A value near $1$ means $i$ is comfortably inside its cluster; near $0$ it lies on a boundary; negative means it was probably misassigned. Averaging $s(i)$ over the dataset gives a label-free quality score.

---

## 2. K-means: Centroid-Driven Clustering

### 2.1 The Objective

K-means minimizes the **Within-Cluster Sum of Squares (WCSS):**

$$J(\{c_i\}, \{\boldsymbol{\mu}_k\}) = \sum_{k=1}^{K} \sum_{i: c_i = k} \\|\mathbf{x}_i - \boldsymbol{\mu}_k\\|^2 \tag{1}$$

This is a *joint* optimization over discrete assignments $\{c_i\}$ and continuous centroids $\{\boldsymbol{\mu}_k\}$ -- and it is NP-hard in general. Lloyd's algorithm tackles it with **coordinate descent**: alternately optimize one set of variables while fixing the other.

### 2.2 Lloyd's Algorithm

![Lloyd's algorithm: four iterations on a 3-blob dataset showing centroid trajectories and the monotonically decreasing WCSS J](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig1_kmeans_steps.png)

**Step 1 -- Assignment.** Fix centroids, assign each point to the nearest:

$$c_i = \arg\min_{k} \\|\mathbf{x}_i - \boldsymbol{\mu}_k\\|^2$$

**Step 2 -- Update.** Fix assignments, recompute centroids by setting $\partial J / \partial \boldsymbol{\mu}_k = 0$:

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i \tag{2}$$

**Why it converges.** Each step *can only decrease* $J$. Step 1 is optimal because every point ends up at its closest centroid. Step 2 is optimal because the within-cluster mean is the unique minimizer of squared error (the second derivative $2|C_k|\mathbf{I}$ is positive definite). Since $J \geq 0$ and decreases monotonically, the iteration must terminate at a local minimum -- usually within tens of iterations on real data, as the four-panel figure above shows: centroids drift from a poor initialization, decision regions reshape, and $J$ collapses to a steady value.

**Caveat: local optima.** Lloyd's algorithm finds *a* local minimum, not the global one. Bad initialization can trap it on a plateau where two centroids share one true cluster while a third tries to cover two. The standard remedy is to restart Lloyd's algorithm from many random initializations and keep the lowest-$J$ run; a smarter remedy is K-means++.

```python
import numpy as np

def kmeans(X, K, max_iter=100):
    """Vanilla Lloyd's algorithm."""
    N = X.shape[0]
    centroids = X[np.random.choice(N, K, replace=False)]
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array(
            [X[labels == k].mean(axis=0) for k in range(K)]
        )
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

### 2.3 K-means++ Initialization

The idea is simple: spread initial centroids far apart by picking each one *with probability proportional to its squared distance from the existing seeds*.

1. Pick $\boldsymbol{\mu}_1$ uniformly at random from $\mathbf{X}$.
2. For $k = 2, \dots, K$: compute $D(\mathbf{x}_i)^2 = \min_{j < k} \\|\mathbf{x}_i - \boldsymbol{\mu}_j\\|^2$, then sample $\boldsymbol{\mu}_k = \mathbf{x}_i$ with probability $D(\mathbf{x}_i)^2 / \sum_l D(\mathbf{x}_l)^2$.

**Theoretical guarantee** (Arthur & Vassilvitskii, 2007): the expected K-means++ objective is at most $8(\ln K + 2) \cdot J_{\text{opt}}$ -- a logarithmic-in-$K$ approximation, *without even running Lloyd's iterations afterward*.

### 2.4 Choosing $K$: Silhouette and Elbow

K-means makes you specify $K$. The two most common ways to pick it are visualized below.

![Silhouette score versus K: peak at the true number of clusters, with under- and over-fit regions shaded](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig4_silhouette_curve.png)

The silhouette curve typically *peaks* at the right $K$ and decays on both sides. The elbow plot tells the same story from a different angle:

![Elbow plot: WCSS versus K, with the elbow point highlighted as diminishing returns kick in](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig5_elbow.png)

WCSS strictly decreases as $K$ grows (more centroids can always fit better), so we look for the **kink** where extra clusters stop helping much. The point of maximum perpendicular distance from the line connecting the two endpoints is a robust elbow heuristic.

### 2.5 Limitations

| Limitation | Cause | Remedy |
|------------|-------|--------|
| Need to specify $K$ | no built-in model selection | silhouette / elbow / BIC |
| Spherical clusters only | uses Euclidean distance, equal radii | GMM, spectral, kernel K-means |
| Sensitive to outliers | mean is not robust | K-medoids |
| Scale-sensitive | distance is dominated by largest features | standardize first |

---

## 3. Hierarchical Clustering

### 3.1 Agglomerative Approach

Build a **dendrogram** from the bottom up:

1. Start with $N$ singleton clusters.
2. Merge the two closest clusters.
3. Repeat until one cluster (or until you have $K$).

The result is a binary tree where the height of each merge encodes the distance at which it happened. To get a flat partition, draw a horizontal line through the tree -- every branch it cuts is one cluster.

![Ward dendrogram with horizontal cut producing three clusters, plus the resulting scatter plot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig3_dendrogram.png)

### 3.2 Linkage Criteria

The merge rule depends on **linkage**, which is just a choice of how to define distance between two *sets* of points:

| Linkage | $d(C_i, C_j)$ | Character |
|---------|---------------|-----------|
| Single | $\min_{a \in C_i, b \in C_j} d(a, b)$ | finds elongated/chained clusters; fragile to noise bridges |
| Complete | $\max_{a \in C_i, b \in C_j} d(a, b)$ | compact, roughly spherical clusters; noise-robust |
| Average | $\frac{1}{|C_i||C_j|}\sum_{a,b} d(a,b)$ | balanced compromise |
| Ward | minimize variance increase $\Delta\!J$ after merge | most consistent with the K-means objective; produces balanced sizes |

**Ward is the default choice** for most numerical data because its objective coincides with WCSS reduction.

### 3.3 When to Use It

- You don't know $K$ and want to *see* the data's natural granularity.
- You want a **hierarchy** (e.g. taxonomy of products, gene families).
- $N$ is small-to-medium (naive complexity is $O(N^3)$; $O(N^2 \log N)$ with priority queues).

---

## 4. DBSCAN: Density-Based Clustering

### 4.1 Core Concepts

DBSCAN ("Density-Based Spatial Clustering of Applications with Noise") replaces the "every point belongs to some cluster" assumption with a density rule. It uses two parameters: neighborhood radius $\epsilon$ and minimum points $\text{MinPts}$.

- **$\epsilon$-neighborhood:** $N_\epsilon(\mathbf{x}) = \{\mathbf{x}' \in \mathbf{X} : \\|\mathbf{x} - \mathbf{x}'\\| \leq \epsilon\}$.
- **Core point:** $|N_\epsilon(\mathbf{x})| \geq \text{MinPts}$ -- it sits in a dense neighborhood.
- **Border point:** not core, but inside some core point's neighborhood.
- **Noise point:** neither -- DBSCAN explicitly labels it as an outlier.

### 4.2 Density Reachability

DBSCAN grows clusters by chaining core points:

- **Directly density-reachable:** $\mathbf{x}'$ lies in $\mathbf{x}$'s neighborhood, and $\mathbf{x}$ is a core point.
- **Density-reachable:** there is a chain of directly density-reachable links from $\mathbf{x}$ to $\mathbf{x}'$.
- **Density-connected:** both points are density-reachable from a common third point.

**A cluster is a maximal set of density-connected points.** This is why DBSCAN handles arbitrary cluster shapes -- no assumption of convexity, no assumption of equal radii, no fixed $K$.

![DBSCAN on noisy moons: core points filled, border points ringed, noise points marked with x; one core point's epsilon-neighborhood is highlighted](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig2_dbscan_density.png)

The figure makes the bookkeeping concrete. The amber circle around the highlighted core point contains at least $\text{MinPts}=5$ neighbors, qualifying it as a core point. Any other point that falls inside such a ball joins the same cluster, and from there the density-reachability relation propagates outward through the moon shape. Stray points in low-density regions never accumulate enough neighbors and get labeled noise.

### 4.3 Choosing $\epsilon$: the K-Distance Plot

Plot, for every point, the distance to its $k$-th nearest neighbor (with $k = \text{MinPts}$), then sort those values descending. The curve has a knee where it transitions from "dense" to "sparse" distances -- pick $\epsilon$ at that knee.

### 4.4 Strengths and Weaknesses

**Strengths.** No need to specify $K$; finds arbitrary shapes; identifies noise; robust to outliers.

**Weaknesses.** Two parameters that interact ($\epsilon$, $\text{MinPts}$); struggles when clusters have *very* different densities (use **HDBSCAN** for this); suffers in high dimensions where Euclidean distances concentrate.

---

## 5. Spectral Clustering: A Graph Theory Approach

### 5.1 From Data to Graphs

Treat each data point as a graph node and put a weighted edge between every pair. The most common similarity is the Gaussian kernel:

$$W_{ij} = \exp\left(-\frac{\\|\mathbf{x}_i - \mathbf{x}_j\\|^2}{2\sigma^2}\right)$$

For scalability, sparsify with $k$-nearest-neighbor or $\epsilon$-ball graphs.

### 5.2 The Graph Laplacian

Define the **degree matrix** $D_{ii} = \sum_j W_{ij}$ and the **unnormalized Laplacian**

$$\mathbf{L} = \mathbf{D} - \mathbf{W}.$$

The Laplacian's defining property is

$$\mathbf{f}^T \mathbf{L} \mathbf{f} = \tfrac{1}{2}\sum_{i,j} W_{ij}(f_i - f_j)^2 \geq 0.$$

That is, $\mathbf{L}$ measures how much a function $\mathbf{f}$ varies across edges. Smooth functions on the graph -- those that change slowly between strongly connected nodes -- have *small* Laplacian quadratic form, and they are exactly the eigenvectors of $\mathbf{L}$ with the smallest eigenvalues.

The **symmetric normalized Laplacian** rescales by node degree to prevent high-degree nodes from dominating:

$$\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}.$$

### 5.3 Normalized Cut Objective

Spectral clustering minimizes a graph cut:

$$\text{NCut}(A, B) = \frac{\text{cut}(A, B)}{\text{vol}(A)} + \frac{\text{cut}(A, B)}{\text{vol}(B)} \tag{3}$$

where $\text{cut}(A,B) = \sum_{i \in A, j \in B} W_{ij}$ and $\text{vol}(A) = \sum_{i \in A} D_{ii}$. Dividing by volume prevents the trivial "split off one point" solution. Combinatorially this is NP-hard, but a **continuous relaxation** -- letting cluster indicators take real values -- has an exact closed-form solution: it is the eigenvalue problem for $\mathbf{L}_{\text{sym}}$.

### 5.4 The Algorithm

1. Build the similarity matrix $\mathbf{W}$.
2. Compute the Laplacian $\mathbf{L}_{\text{sym}}$.
3. Compute the $K$ eigenvectors corresponding to the *smallest* eigenvalues.
4. Stack them as columns of $\mathbf{U} \in \mathbb{R}^{N \times K}$.
5. Normalize each row of $\mathbf{U}$ to unit length.
6. Run **K-means on the rows** to recover discrete cluster labels.

**Why this works.** Step 3 maps the data into a low-dimensional embedding where graph-connected clusters become *linearly separable point clouds*. K-means, which fails on the original non-convex shape, succeeds in this embedding.

**Complexity.** $O(N^2)$ to build $\mathbf{W}$, $O(N^3)$ for the eigendecomposition. For large $N$, use sparse $k$-NN graphs and Lanczos / Nyström approximation.

---

## 6. Gaussian Mixture Models: K-means with Probabilities

### 6.1 The Generative Model

GMM assumes each point is generated by first sampling a component $z \sim \text{Categorical}(\pi_1, \dots, \pi_K)$, then sampling $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$. The marginal density is

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).$$

Fit by EM (see [Part 13](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/)):

- **E-step:** posterior responsibility $\gamma_{ik} = \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) / \sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$.
- **M-step:** weighted updates $\boldsymbol{\mu}_k = \sum_i \gamma_{ik} \mathbf{x}_i / \sum_i \gamma_{ik}$, similarly for $\boldsymbol{\Sigma}_k$ and $\pi_k$.

### 6.2 Why GMM Generalizes K-means

K-means is the **limit** of GMM with $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ and $\sigma \to 0$: as the variance shrinks, the soft posterior $\gamma_{ik}$ collapses to a one-hot vector, and the M-step becomes the centroid mean. So K-means is GMM with two strong assumptions hard-baked: spherical equal-radius clusters, hard assignments.

GMM relaxes both -- it allows **ellipsoidal covariance** (rotated, stretched clusters) and **soft membership** (a point can be 70% cluster A, 30% cluster B). The price is more parameters and slower convergence.

![GMM versus K-means on anisotropic blobs: K-means imposes straight Voronoi boundaries, GMM recovers tilted Gaussian ellipses](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig6_gmm_vs_kmeans.png)

The figure makes the difference vivid. On stretched, tilted clusters, K-means' Voronoi cells slice across them at the wrong angles, splitting one true cluster between two centroids. GMM's ellipses align with the data's covariance and recover the true partition.

---

## 7. Putting It All Together: When to Use What

Different priors, different victories. The grid below runs K-means, DBSCAN, and spectral clustering on three classic dataset shapes:

![Three algorithms (K-means, DBSCAN, Spectral) on three dataset shapes (Blobs, Moons, Circles)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/18-Clustering-Algorithms/fig7_algo_comparison.png)

Reading the grid:

- **Blobs (top row).** All three succeed -- this is the easy case. K-means is fastest.
- **Moons (middle).** K-means slices each moon in half (its prior is convex, the data isn't). DBSCAN and spectral both follow the curves correctly.
- **Circles (bottom).** K-means again fails for the same reason. DBSCAN traces the inner and outer rings via density chains; spectral exploits the graph structure.

**Rule of thumb:**

| If your data... | Try first |
|-----------------|-----------|
| has roughly spherical, well-separated clusters | K-means |
| has elongated or non-convex clusters with noise | DBSCAN |
| has non-convex clusters in a smooth manifold | Spectral |
| needs soft assignments or covariance modelling | GMM |
| needs a hierarchy / unknown $K$ | Agglomerative (Ward) |

---

## 8. Exercises

**Exercise 1: K-means convergence.** Prove that the update step minimizes WCSS for fixed assignments.

> **Solution.** Take $\partial / \partial \boldsymbol{\mu}_k$ of $\sum_{i \in C_k}\\|\mathbf{x}_i - \boldsymbol{\mu}_k\\|^2$ and set to zero: $\boldsymbol{\mu}_k = \tfrac{1}{|C_k|}\sum_{i \in C_k}\mathbf{x}_i$. The Hessian is $2|C_k|\mathbf{I} \succ 0$, so this is a global minimum.

**Exercise 2: DBSCAN.** With $\text{MinPts} = 3$, $\epsilon = 0.5$, point $\mathbf{x}$ has 2 neighbors within radius 0.5. Is $\mathbf{x}$ a core point?

> **Solution.** The $\epsilon$-neighborhood includes $\mathbf{x}$ itself plus 2 neighbors = 3 points. Since $3 \geq 3$, yes, $\mathbf{x}$ is a core point. (If $\text{MinPts}$ were $4$ it would be a border or noise point.)

**Exercise 3: GMM vs K-means.** When should you reach for GMM instead of K-means?

> **Solution.** When clusters are elliptical (different variances along different axes), when you need soft assignments (e.g. for downstream Bayesian reasoning), or when you want a generative model you can sample from. Stick with K-means when clusters are spherical, $N$ is huge, or you need fast iteration.

**Exercise 4: Linkage choice.** Why does single-linkage often fail on noisy data?

> **Solution.** Single-linkage merges based on the minimum pairwise distance, so a *single* noise point bridging two clusters is enough to chain them together (the "chaining effect"). Complete- and Ward-linkage use aggregate distances and are robust to such bridges.

**Exercise 5: Silhouette.** $a(i) = 2$, $b(i) = 5$. Compute $s(i)$.

> **Solution.** $s = (5 - 2)/\max(2, 5) = 3/5 = 0.6$. A respectable score: point $i$ is roughly twice as far from the next cluster as from its own.

---

## References

[1] Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Trans. Info. Theory*, 28(2), 129-137.

[2] Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027-1035.

[3] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 226-231.

[4] Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. *NIPS*, 849-856.

[5] Von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416.

[6] Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. *IEEE Trans. PAMI*, 22(8), 888-905.

[7] Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301), 236-244.

[8] Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates (HDBSCAN). *PAKDD*, 160-172.

---

*This is Part 18 of the [ML Mathematical Derivations](/tags/Mathematical-Derivations/) series. Next: [Part 19 -- Neural Networks and Backpropagation](/en/Machine-Learning-Mathematical-Derivations-19-Neural-Networks-and-Backpropagation/). Previous: [Part 17 -- Dimensionality Reduction and PCA](/en/Machine-Learning-Mathematical-Derivations-17-Dimensionality-Reduction-and-PCA/).*
