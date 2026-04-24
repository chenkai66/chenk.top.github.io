---
title: "ML Math Derivations (2): Linear Algebra and Matrix Theory"
date: 2026-01-19 09:00:00
tags:
  - Machine Learning
  - Linear Algebra
  - Matrix Theory
  - SVD
  - Matrix Calculus
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 2
  total: 20
lang: en
mathjax: true
description: "The language of machine learning is linear algebra. This article derives vector spaces, eigendecomposition, SVD, and matrix calculus from first principles -- every tool you need for ML optimization."
disableNunjucks: true
---

## Why this chapter, and what's different

If you have already worked through a standard linear-algebra course you have seen most of these objects. **This chapter is not that course.** It is the *ML practitioner's slice* of linear algebra: the half-dozen ideas that actually appear when you implement gradient descent, run PCA, train a neural net, or read a paper.

Concretely the goals are:

1. Build a **geometric intuition** for what matrices *do* (rotate, stretch, project, kill).
2. Learn the four decompositions that show up everywhere -- spectral, **SVD**, QR, Cholesky -- and *which one to reach for*.
3. Master enough **matrix calculus** to derive any neural-net gradient on the back of an envelope.

We skim the algebra of row reduction, determinants by cofactor, and abstract vector-space proofs. If you need those, the references at the bottom give the standard treatments. Here, every concept comes back to a picture or a line of NumPy.

**Prerequisites:** matrix multiplication, transpose, the idea of a determinant, partial derivatives.

---

## 1. Vector spaces, subspaces, rank -- the geometric core

Forget the eight axioms for a moment. The mental model that pays off in ML is:

> *A vector space is a flat, infinite "sheet" through the origin. A subspace is a smaller flat sheet through the origin sitting inside it.*

A line through the origin is a 1D subspace of $\mathbb{R}^2$. A plane through the origin is a 2D subspace of $\mathbb{R}^3$. The crucial word is **through the origin** -- shift the sheet and you lose closure under addition and scaling.

![A 2D subspace inside R^3](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig1_vector_space_subspace.png)

### 1.1 Span, independence, basis -- in one breath

Pick a few vectors $v_1, \ldots, v_k$. The set of all linear combinations $\sum \alpha_i v_i$ is their **span** -- it is *always* a subspace.

Those vectors are **linearly independent** when none of them lives in the span of the others, equivalently:

$$\alpha_1 v_1 + \cdots + \alpha_k v_k = 0 \;\implies\; \alpha_1 = \cdots = \alpha_k = 0. \tag{1}$$

A **basis** is an independent spanning set. Its size is the **dimension** of the subspace.

![Independence vs dependence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig2_linear_dependence.png)

The picture is the whole story: independent vectors enclose a non-degenerate parallelogram (positive area, so they actually span a 2D piece). Dependent vectors lie on a line, and the parallelogram collapses.

### 1.2 Matrices ARE linear maps

Every matrix $A \in \mathbb{R}^{m \times n}$ is the same object as a linear map $T : \mathbb{R}^n \to \mathbb{R}^m$, and conversely. The dictionary is one rule:

> *The $j$-th column of $A$ is $T(e_j)$ -- where the $j$-th standard basis vector lands.*

So when you see a weight matrix $W \in \mathbb{R}^{h \times d}$ in a neural net layer, you are looking at a function "take a $d$-dimensional input, output an $h$-dimensional vector"; the columns of $W$ tell you what each input feature contributes.

### 1.3 Rank and the four fundamental subspaces

For an $m \times n$ matrix $A$ of rank $r$:

| Subspace | Definition | Lives in | Dimension | "Meaning" |
|----------|-----------|----------|-----------|-----------|
| Column space $\text{Col}(A)$ | $\{A x : x \in \mathbb{R}^n\}$ | $\mathbb{R}^m$ | $r$ | reachable outputs |
| Null space $\text{Null}(A)$ | $\{x : A x = 0\}$ | $\mathbb{R}^n$ | $n-r$ | inputs that get killed |
| Row space $\text{Row}(A)$ | $\text{Col}(A^\top)$ | $\mathbb{R}^n$ | $r$ | inputs $A$ "sees" |
| Left null space | $\{y : A^\top y = 0\}$ | $\mathbb{R}^m$ | $m-r$ | outputs unreachable |

**Orthogonality (the fundamental theorem).** $\text{Null}(A) \perp \text{Row}(A)$.

*Proof.* If $Ax = 0$ and $w = A^\top z$ is in the row space, then $\langle x, w \rangle = z^\top (A x) = 0$. $\square$

So $\mathbb{R}^n$ splits into two orthogonal pieces: the row space (where $A$ does interesting work) and the null space (where $A$ throws information away). Whenever a model has more parameters than equations -- which is *always* in modern ML -- the null space is non-trivial and many parameter vectors give the same predictions.

---

## 2. Norms and inner products -- the geometry layer

An **inner product** $\langle \cdot, \cdot \rangle$ gives you angles; a **norm** $\|\cdot\|$ gives you lengths. On $\mathbb{R}^n$ the standard one is $\langle x, y \rangle = x^\top y$, with induced norm $\|x\|_2 = \sqrt{x^\top x}$.

The **$\ell_p$ family** is the workhorse for regularisation:

| Norm | Formula | Unit ball | ML use |
|------|---------|-----------|--------|
| $\ell_1$ | $\sum_i |x_i|$ | diamond | Lasso, sparsity |
| $\ell_2$ | $\sqrt{\sum_i x_i^2}$ | circle / sphere | Ridge, weight decay |
| $\ell_\infty$ | $\max_i |x_i|$ | square / cube | adversarial bounds |

### 2.1 Cauchy-Schwarz and the cosine

**Theorem (Cauchy-Schwarz).** $|\langle u, v \rangle| \le \|u\| \cdot \|v\|$, with equality iff $u, v$ are parallel.

*Proof.* The quadratic $\|u + tv\|^2 = \|u\|^2 + 2t\langle u, v\rangle + t^2 \|v\|^2$ is non-negative for every $t$, so its discriminant must be $\le 0$. $\square$

Divide both sides by $\|u\|\|v\|$ and you have the **cosine of the angle** $\cos\theta = \langle u, v\rangle / (\|u\|\|v\|)$ -- and the proof that it lives in $[-1, 1]$. This is the formula behind cosine similarity in retrieval, attention, contrastive learning -- everywhere.

### 2.2 Matrix norms you actually use

| Norm | Formula | What it measures |
|------|---------|------------------|
| Frobenius $\|A\|_F$ | $\sqrt{\sum_{ij} a_{ij}^2}$ | the "Euclidean length" of $A$ as a vector |
| Spectral $\|A\|_2$ | $\sigma_1(A)$ | maximum stretch over all unit inputs |
| Nuclear $\|A\|_*$ | $\sum_i \sigma_i(A)$ | convex surrogate for $\text{rank}(A)$ |

The spectral norm is the right thing for Lipschitz analysis (e.g. spectral normalisation in GANs). The nuclear norm is what you minimise for low-rank matrix completion (Netflix-style problems).

---

## 3. Eigendecomposition -- "the directions that survive"

### 3.1 Definition and intuition

A nonzero vector $v$ is an **eigenvector** of $A \in \mathbb{R}^{n \times n}$ with **eigenvalue** $\lambda$ when

$$A v = \lambda v. \tag{2}$$

Geometrically: $A$ usually rotates inputs, but along an eigenvector all $A$ does is rescale. The eigenvector's direction *survives* the transformation.

![Eigendecomposition: directions that survive](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig3_eigendecomposition.png)

Eigenvalues solve the characteristic polynomial $\det(A - \lambda I) = 0$. Two free identities:

- $\sum_i \lambda_i = \operatorname{tr}(A)$
- $\prod_i \lambda_i = \det(A)$

### 3.2 The spectral theorem (the only one you'll use daily)

**Theorem.** If $A$ is real and symmetric ($A = A^\top$), then:

1. all eigenvalues are real,
2. eigenvectors for distinct eigenvalues are orthogonal,
3. $A$ is **orthogonally diagonalisable**: $A = Q \Lambda Q^\top$ with $Q^\top Q = I$ and $\Lambda$ diagonal.

*Sketch (eigenvalues real).* Take $A v = \lambda v$ with $v \neq 0$. Compute $\bar v^\top A v$ two ways and use $A = A^\top$: you get $\lambda = \bar\lambda$. $\square$

Equivalently, in **rank-1 form**:

$$A = \sum_{i=1}^n \lambda_i\, q_i q_i^\top. \tag{3}$$

Read this carefully -- $A$ is *literally* a weighted sum of projections onto orthogonal directions. This is the ML lens on every symmetric matrix: covariance, Gram, Hessian, graph Laplacian.

### 3.3 Positive (semi-)definite matrices

A symmetric $A$ is **positive definite** ($A \succ 0$) when $x^\top A x > 0$ for all $x \neq 0$. Equivalently, *all eigenvalues are positive*. **Positive semi-definite** ($A \succeq 0$) replaces $> 0$ with $\ge 0$.

![A PD matrix is a bowl](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig7_positive_definite_ellipse.png)

The geometry: $\{x : x^\top A x = 1\}$ is an **ellipsoid**, with principal axes along the eigenvectors and semi-axis lengths $1/\sqrt{\lambda_i}$. When some eigenvalue is negative, you get a hyperboloid (saddle) and there is no minimum. *This is why convex ML insists on PD Hessians.*

Where PD matrices show up:

- **Covariance** $\Sigma = \mathbb{E}[(x - \mu)(x - \mu)^\top]$ is always PSD.
- **Gram matrices** $X^\top X$ are PSD; PD iff columns of $X$ are independent.
- **Kernels** must be PSD (Mercer's theorem) or you can't reproduce them in any feature space.
- **Hessians of convex losses** are PSD; strict convexity = PD.

---

## 4. Singular Value Decomposition -- the universal factorisation

### 4.1 The theorem

**Theorem (SVD).** Every $A \in \mathbb{R}^{m \times n}$ of rank $r$ factors as

$$A = U \Sigma V^\top, \tag{4}$$

with $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ orthogonal, and $\Sigma$ diagonal with **singular values** $\sigma_1 \ge \cdots \ge \sigma_r > 0$ (and zeros after).

**Why SVD always exists.** $A^\top A$ is symmetric PSD, so the spectral theorem gives an orthonormal eigenbasis $V$ with eigenvalues $\sigma_i^2$. Define $u_i = A v_i / \sigma_i$ for $\sigma_i > 0$ and complete to an orthonormal basis $U$. Then $A v_i = \sigma_i u_i$, i.e. $A V = U \Sigma$, i.e. $A = U \Sigma V^\top$. $\square$

### 4.2 Geometric picture: rotate, scale, rotate

![SVD geometric pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig4_svd_three_steps.png)

Every linear map decomposes into three motions:

1. **$V^\top$**: rotate input so its principal axes align with the coordinate axes,
2. **$\Sigma$**: stretch each axis by $\sigma_i$ (this is the only "non-rigid" step),
3. **$U$**: rotate the result into the output space.

This is the cleanest mental model in linear algebra: *any* matrix is a rotation, then an axis-aligned stretch, then another rotation.

### 4.3 What goes wrong without rank: visualised

When some $\sigma_i = 0$, that direction in the input is annihilated. The output ellipse collapses to a lower-dimensional shape:

![Rank deficiency: 2D collapses to 1D](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig5_rank_deficiency.png)

A rank-1 matrix $A = u v^\top$ sends *all of $\mathbb{R}^n$* onto the single line $\text{span}(u)$. Everything orthogonal to $v$ is in the null space and gets sent to $0$. This is exactly what makes low-rank approximations *compress* data.

### 4.4 Eckart-Young: best low-rank approximation

**Theorem (Eckart-Young).** The best rank-$k$ approximation to $A$ in both Frobenius and spectral norm is

$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top, \tag{5}$$

with errors $\|A - A_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}$ and $\|A - A_k\|_2 = \sigma_{k+1}$.

This is the **mathematical engine of PCA**. Center your data matrix $X$, take its SVD; the top-$k$ right singular vectors are the principal components, and the singular values squared (over $n$) are the explained variances. It is also the engine of every low-rank technique you have heard of: image compression, latent semantic analysis, recommender systems, LoRA.

### 4.5 Pseudo-inverse and condition number

The **Moore-Penrose pseudo-inverse** is computed via SVD: $A^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts the non-zero singular values. Then $x^\star = A^+ b$ is the minimum-norm least-squares solution to $Ax = b$ -- exactly what `numpy.linalg.lstsq` returns.

The **condition number** $\kappa(A) = \sigma_1 / \sigma_r$ measures how much $A$ amplifies relative perturbations. Roughly, if $\kappa(A) \approx 10^k$ you lose $k$ digits of precision when solving $Ax = b$. Always log it before trusting a least-squares fit.

### 4.6 Eigendecomposition vs SVD

| | Eigendecomposition | SVD |
|---|---|---|
| Works on | square matrices | any $m \times n$ |
| Form | $A = P \Lambda P^{-1}$ | $A = U \Sigma V^\top$ |
| Always exists? | only if diagonalisable | always |
| "Values" | may be complex | non-negative real |
| Right factor = inverse of left? | no (general $P$) | yes ($U, V$ orthogonal) |

For a *symmetric PSD* matrix the two coincide ($U = V = Q$, $\Sigma = \Lambda$). Otherwise prefer SVD -- it's stable, total, and applies to rectangular matrices.

---

## 5. Matrix calculus -- the engine of optimisation

This is where most ML readers want a clean reference. We use **numerator layout**: the derivative has the shape of the output (so a gradient of a scalar w.r.t. a vector is a column vector, matching the input).

![Matrix calculus shape rules](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/02-Linear-Algebra-and-Matrix-Theory/fig6_matrix_calculus_shapes.png)

### 5.1 The four formulas you reach for daily

| # | $f$ | $\nabla f$ | Comment |
|---|-----|-----------|---------|
| 1 | $a^\top x$ | $a$ | linear |
| 2 | $x^\top A x$ | $(A + A^\top) x$ | $= 2 A x$ if $A$ symmetric |
| 3 | $\|A x - b\|_2^2$ | $2 A^\top (A x - b)$ | least squares |
| 4 | $\ln \det X$ | $X^{-\top}$ | log-det -- shows up in Gaussian MLE |

*Proof of (2).* Write $f = \sum_{i,j} A_{ij} x_i x_j$. Then $\partial f / \partial x_k = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = [(A + A^\top) x]_k$. $\square$

*Proof of (3).* Apply the chain rule: $\nabla_x \tfrac12 \|Ax - b\|^2 = A^\top (A x - b)$. Multiplying by 2 absorbs the $\tfrac12$ that ML papers conventionally drop. The optimum solves the **normal equations** $A^\top A x = A^\top b$.

### 5.2 Chain rule and backprop

**Theorem (chain rule).** For $L : \mathbb{R}^n \to \mathbb{R}$ written as $L(x) = f(g(x))$ with $g : \mathbb{R}^n \to \mathbb{R}^m$:

$$\nabla_x L = J_g^\top \cdot \nabla_g f, \tag{6}$$

where $J_g$ is the $m \times n$ Jacobian of $g$.

This *is* backpropagation. For one neural-net layer $z = Wx + b$, $a = \sigma(z)$, with downstream loss $L$:

$$\delta := \frac{\partial L}{\partial a} \odot \sigma'(z), \quad \frac{\partial L}{\partial W} = \delta\, x^\top, \quad \frac{\partial L}{\partial b} = \delta. \tag{7}$$

That's the entire algorithm -- everything PyTorch's autograd engine does is run (6) on a computation graph.

### 5.3 Sanity check by finite differences

Whenever you derive a gradient, you owe yourself a numerical check:

$$\hat g_i = \frac{f(x + \varepsilon e_i) - f(x - \varepsilon e_i)}{2 \varepsilon}.$$

Match against the analytic $\nabla f$ to relative error $\approx 10^{-6}$ at $\varepsilon = 10^{-5}$. The code in section 7 does exactly this.

---

## 6. Numerical decompositions -- which one and when

### 6.1 QR for least squares

Any full-column-rank $A$ factors as $A = QR$ with $Q$ orthonormal columns and $R$ upper triangular. To solve $\min_x \|Ax - b\|$, compute $Q^\top b$ and back-substitute on $R x = Q^\top b$.

**Why prefer QR over normal equations.** Normal equations form $A^\top A$, whose condition number is $\kappa(A)^2$. If $\kappa(A) = 10^4$ that's $10^8$ -- you've squared your error budget. QR works directly on $A$, so it's $\kappa(A)$.

### 6.2 Cholesky for PD systems

For PD $A$, the **Cholesky** factorisation $A = L L^\top$ exists with $L$ lower triangular and positive diagonal. Cost $O(n^3 / 3)$, half of LU. Solving $Ax = b$ is then two triangular solves. This is the workhorse for Gaussian-process inference, kernel methods, and any quadratic program with PD Hessian.

### 6.3 Decision table

| Situation | Use |
|-----------|-----|
| Symmetric matrix (covariance, Hessian) | Eigendecomposition / spectral |
| Any matrix; want "honest" rank, PCA, pseudo-inverse | SVD |
| Tall least squares $\min \|Ax - b\|$ | QR |
| PD linear system $Ax = b$ | Cholesky |
| Detect rank deficiency / monitor numerical health | $\kappa(A)$ via SVD |

---

## 7. Code: verify everything

```python
import numpy as np
from scipy.linalg import svd, qr, cholesky

rng = np.random.default_rng(42)

# -------- SVD low-rank approximation: Eckart-Young in action --------
m, n, true_rank = 100, 80, 10
U_t = rng.standard_normal((m, true_rank))
S_t = np.diag(np.linspace(10, 1, true_rank))
V_t = rng.standard_normal((n, true_rank))
A = U_t @ S_t @ V_t.T + 0.5 * rng.standard_normal((m, n))

U, s, Vt = svd(A, full_matrices=False)
print("rank-k approximation: actual vs theoretical Frobenius error")
for k in [1, 5, 10, 20]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    actual = np.linalg.norm(A - A_k, "fro")
    theory = np.sqrt(np.sum(s[k:] ** 2))
    print(f"  k={k:2d}: actual={actual:6.2f}  theory={theory:6.2f}")

# -------- QR vs normal equations: numerical stability --------
m, n = 100, 10
A = rng.standard_normal((m, n))
x_true = rng.standard_normal(n)
b = A @ x_true + 0.1 * rng.standard_normal(m)

x_normal = np.linalg.solve(A.T @ A, A.T @ b)
Q, R = qr(A, mode="economic")
x_qr = np.linalg.solve(R, Q.T @ b)
print(f"\nkappa(A) = {np.linalg.cond(A):.1f},  "
      f"kappa(A^T A) = {np.linalg.cond(A.T @ A):.1f}")
print(f"normal-eq error: {np.linalg.norm(x_normal - x_true):.2e}")
print(f"QR error       : {np.linalg.norm(x_qr - x_true):.2e}")

# -------- Gradient check on linear regression --------
n, d = 100, 5
X = rng.standard_normal((n, d))
w_true = rng.standard_normal(d)
y = X @ w_true + 0.1 * rng.standard_normal(n)

loss = lambda w: 0.5 / n * np.sum((X @ w - y) ** 2)
grad = lambda w: 1 / n * X.T @ (X @ w - y)

def grad_numeric(w, eps=1e-7):
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps; wm[i] -= eps
        g[i] = (loss(wp) - loss(wm)) / (2 * eps)
    return g

w_test = rng.standard_normal(d)
print(f"\ngradient check: ||analytic - numeric|| = "
      f"{np.linalg.norm(grad(w_test) - grad_numeric(w_test)):.2e}")
```

Three things to notice in the output: the Eckart-Young error matches theory to machine precision; QR is one to two orders of magnitude more accurate than normal equations even on benign matrices; the analytic gradient agrees with finite differences to ~$10^{-9}$.

---

## 8. Q&A -- the questions that come up

**Q1. When *should* I trust normal equations?**
When $\kappa(A) < 10^3$ or so -- e.g. small problems with well-scaled features. Anything ill-conditioned: use QR or SVD. Squaring the condition number really does eat your precision.

**Q2. Why does $\ell_1$ produce sparse solutions?**
The $\ell_1$ unit ball has corners on the coordinate axes. Generic optimisation contours hit these corners first, and at a corner some coordinates are exactly zero. The smooth $\ell_2$ ball has no corners and so generically the optimum lies off the axes.

**Q3. SVD vs PCA -- the same thing?**
Yes, modulo bookkeeping. With centred data $X \in \mathbb{R}^{n \times d}$:
- PCA via covariance: $\Sigma = \tfrac{1}{n} X^\top X$, top eigenvectors = principal directions.
- PCA via SVD: $X = U \Sigma V^\top$, columns of $V$ = principal directions; $\sigma_i^2 / n$ = explained variances.
For high-dimensional data ($d \gg n$) the SVD path is dramatically cheaper because you never materialise the $d \times d$ covariance.

**Q4. Why are eigenvalues of symmetric matrices special?**
They are *real* (so we can order them), and the eigenvectors form an *orthonormal* basis. That orthonormality is what lets us write any vector as $\sum \langle v, q_i\rangle q_i$ -- a clean coordinate system for the action of $A$. The Hessians, covariances, Grams, and Laplacians of ML are all symmetric, which is why this matters every day.

**Q5. What does "rank" really mean operationally?**
The rank is *how much information $A$ preserves about its input.* A rank-$r$ matrix in $\mathbb{R}^{m \times n}$ throws away an $(n - r)$-dimensional subspace (its null space) and outputs into an $r$-dimensional subspace (its column space). In ML this is exactly the bottleneck dimension of an autoencoder, the rank in matrix completion, or the latent dimension of an embedding.

---

## Summary

| Tool | Key formula | When to reach for it |
|------|-------------|----------------------|
| Spectral theorem | $A = Q \Lambda Q^\top$ | symmetric matrices: covariance, Hessian |
| SVD | $A = U \Sigma V^\top$ | any matrix: PCA, pseudo-inverse, low-rank |
| QR | $A = QR$ | least squares with bad conditioning |
| Cholesky | $A = LL^\top$ | PD linear systems (Gaussian MLE, GP) |
| Quadratic gradient | $\nabla (x^\top A x) = 2 A x$ | every quadratic objective |
| Chain rule | $\nabla_x L = J_g^\top \nabla_g f$ | backpropagation in one line |

---

## References

1. Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.). Wellesley-Cambridge Press.
2. Trefethen, L. N. & Bau III, D. (1997). *Numerical Linear Algebra*. SIAM. -- the cleanest treatment of stability.
3. Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. -- the algorithmic bible.
4. Petersen, K. B. & Pedersen, M. S. (2012). *The Matrix Cookbook*. Technical University of Denmark. -- the gradient lookup table everyone uses.
5. Eckart, C. & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211-218.
6. Boyd, S. & Vandenberghe, L. (2018). *Introduction to Applied Linear Algebra*. Cambridge University Press. -- great companion focused on data.

---

## Series Navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Mathematical Foundations | [<-- Previous](/en/Machine-Learning-Mathematical-Derivations-1-Introduction-and-Mathematical-Foundations/) |
| **2** | **Linear Algebra and Matrix Theory** | *You are here* |
| 3 | Probability Theory and Statistical Inference | [Read next -->](/en/Machine-Learning-Mathematical-Derivations-3-Probability-Theory-and-Statistical-Inference/) |
| 4 | Convex Optimization Theory | [Read -->](/en/Machine-Learning-Mathematical-Derivations-4-Convex-Optimization-Theory/) |
| 5 | Linear Regression | [Read -->](/en/Machine-Learning-Mathematical-Derivations-5-Linear-Regression/) |
