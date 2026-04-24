---
title: "Low-Rank Matrix Approximation and the Pseudoinverse: From SVD to Regularization"
date: 2024-08-21 09:00:00
tags:
  - ML
  - Linear Algebra
  - Optimization
categories: Algorithm
lang: en
mathjax: true
description: "From the least-squares view to the Moore-Penrose pseudoinverse, the four Penrose conditions, computation via SVD, truncated SVD, Tikhonov regularization, and modern applications from PCA to LoRA."
disableNunjucks: true
---
Real data matrices are almost never both square and full rank: correlated features, too few samples, and noise-induced ill-conditioning all make "matrix inverse" either undefined or numerically useless. The **pseudoinverse** (Moore-Penrose inverse) preserves the *spirit* of an inverse while dropping the impossible-to-meet requirements: it redefines the "solution" of a linear system as the **least-squares solution**, breaking ties by picking the one with **minimum norm**. This post derives the pseudoinverse from that least-squares viewpoint, gives the four Penrose conditions, builds it from the SVD, and connects this single object to **the Eckart-Young low-rank approximation theorem**, **PCA**, **recommender-system matrix factorization**, and **LoRA fine-tuning**.

## What You Will Learn

- **The optimization view**: why "pseudoinverse = least squares" is the right unifying definition
- **The SVD recipe**: how$A = U\Sigma V^{\!\top}$ builds$A^{+}$
- **Eckart-Young**: why truncated SVD is *the* unique optimal low-rank approximation
- **Numerical stability**: how small singular values blow up and why Tikhonov regularization fixes it
- **Modern applications**: a single thread connecting PCA, recommender MF, and LoRA fine-tuning

## Prerequisites

- Linear algebra (matrix multiplication, inner products, orthogonal matrices)
- The concept of least-squares regression
- Python + NumPy

---

# Why the pseudoinverse?

The classical inverse$A^{-1}$ only exists for **square, full-rank** matrices. Three common failure modes:

- **Overdetermined** ($m > n$):$Ax = b$ generally has no exact solution, so we settle for least squares.
- **Underdetermined** ($m < n$): solutions are not unique; we need a principled tie-breaker.
- **Ill-conditioned**: numerically near-singular -- $A^{-1}$ exists in theory but is useless in practice.

The pseudoinverse$A^{+}$ **handles all three uniformly**: it always exists, it is always unique, and it reduces to$A^{-1}$ whenever$A^{-1}$ exists.

# The optimization view: pseudoinverse = least squares

## Definition

For$A \in \mathbb{R}^{m \times n}$, the pseudoinverse$A^{+}$ is defined by

$$A^{+} \;=\; \arg\min_{X \in \mathbb{R}^{n \times m}} \; \|AX - I_m\|_F^2.$$

When the minimizer is not unique (because$A$ does not have full row rank), we add a second tie-breaker: among all minimizers, take the one with the smallest Frobenius norm$\|X\|_F$.

> **Intuition.** When$A$ is square and invertible,$AX = I$ has the exact solution$A^{-1}$. Otherwise no$X$ makes$AX$ exactly$I$, so the pseudoinverse settles for *as close to identity as possible*.

## Frobenius norm in one line

$$\|M\|_F^2 \;=\; \sum_{i,j} M_{ij}^2 \;=\; \mathrm{tr}(M^{\!\top} M).$$

It treats a matrix as one long vector. The trace form is what makes matrix calculus tractable -- you avoid expanding everything element by element.

## Solving the optimization (full column rank case)

If$A$ has full column rank, then$A^{\!\top}A$ is invertible. Setting the gradient to zero,

$$\frac{\partial}{\partial X}\|AX - I\|_F^2 = 2 A^{\!\top}(AX - I) = 0
\;\Longrightarrow\;
X = (A^{\!\top}A)^{-1} A^{\!\top}.$$

This is the **left pseudoinverse**$A^{+} = (A^{\!\top}A)^{-1}A^{\!\top}$, satisfying$A^{+}A = I_n$. The full-row-rank case gives the **right pseudoinverse**$A^{+} = A^{\!\top}(AA^{\!\top})^{-1}$ analogously.

## Penrose's four conditions (the unifying definition)

For *any* matrix$A$, the pseudoinverse$A^{+}$ is the **unique** matrix satisfying

$$\begin{aligned}
&\text{(i)}\;\; A A^{+} A = A
&&\text{(ii)}\;\; A^{+} A A^{+} = A^{+} \\
&\text{(iii)}\;\; (A A^{+})^{\!\top} = A A^{+}
&&\text{(iv)}\;\; (A^{+} A)^{\!\top} = A^{+} A.
\end{aligned}$$

(i)-(ii) say$A$ and$A^{+}$ are mutual pseudoinverses; (iii)-(iv) say both$AA^{+}$ and$A^{+}A$ are **symmetric projection matrices**. That last fact is precisely what gives the pseudoinverse its geometric interpretation.

# Computing the pseudoinverse via SVD

The formula$(A^{\!\top}A)^{-1}A^{\!\top}$ blows up when$A$ has linearly dependent columns and is numerically fragile in general. The SVD route is the **textbook-stable** algorithm.

## Recipe

For any$A \in \mathbb{R}^{m \times n}$, the SVD is

$$A \;=\; U \Sigma V^{\!\top}, \qquad
\Sigma = \mathrm{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r, 0, \ldots, 0),$$

where$U \in \mathbb{R}^{m \times m}$ and$V \in \mathbb{R}^{n \times n}$ are orthogonal,$r = \mathrm{rank}(A)$, and$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$. Then

$$\boxed{\;A^{+} \;=\; V \Sigma^{+} U^{\!\top}, \qquad
\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i & \sigma_i > 0 \\ 0 & \sigma_i = 0\end{cases}\;}$$

A direct check verifies all four Penrose conditions, which is why SVD is the **canonical** way to compute the pseudoinverse.

## Geometry: projection onto the column space

![Pseudoinverse via SVD: least squares = orthogonal projection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/low-rank-approximation-pseudoinverse/fig2_pseudoinverse_svd.png)

Left: an overdetermined regression$y \approx ax + b$, with the least-squares line produced by$\hat\beta = A^{+}b$. Right: geometrically,$AA^{+}b$ is the **orthogonal projection** of$b$ onto the column space$\mathrm{col}(A)$. The residual$b - A\hat\beta$ is perpendicular to$\mathrm{col}(A)$ -- which is exactly the geometric content of the **normal equations**$A^{\!\top}(A\hat\beta - b) = 0$.

## Code

```python
import numpy as np

A = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

# Method 1: textbook SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag(np.where(s > 1e-12, 1.0 / s, 0.0))
A_pinv = Vt.T @ S_pinv @ U.T

# Method 2: built-in (recommended -- internally uses SVD with thresholding)
A_pinv_np = np.linalg.pinv(A)

print(np.allclose(A_pinv, A_pinv_np))   # True
print(A @ A_pinv @ A)                   # should reproduce A (Penrose condition i)
```

`np.linalg.pinv` already applies a **singular-value threshold** (`rcond` defaults to about$\mathrm{eps} \cdot \max(m,n)$), zeroing out the reciprocals of singular values that are numerically indistinguishable from zero. The next section explains why that thresholding is non-negotiable.

# Low-rank approximation: the Eckart-Young theorem

The pseudoinverse is just one consequence of the SVD. The SVD's bigger payoff is **low-rank approximation**: among all rank-$k$ matrices, the truncated SVD is provably optimal.

## The theorem (Eckart-Young, 1936)

Let$A = U\Sigma V^{\!\top}$, and define the **truncated SVD**

$$A_k \;=\; \sum_{i=1}^{k} \sigma_i\, u_i v_i^{\!\top} \;=\; U_{:,1:k} \,\Sigma_{1:k,1:k}\, V_{:,1:k}^{\!\top}.$$

Then for *every* matrix$B$ with$\mathrm{rank}(B) \le k$,

$$\|A - A_k\|_F \;\le\; \|A - B\|_F, \qquad
\|A - A_k\|_F \;=\; \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}.$$

The same statement holds in spectral norm:$\|A - A_k\|_2 = \sigma_{k+1}$. So the truncated SVD is the **unique (up to ties) optimal rank-$k$ approximation**, and it is the theoretical foundation behind nearly every dimensionality-reduction or matrix-compression method in ML.

![Eckart-Young: truncated SVD is provably the best rank-k approximation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/low-rank-approximation-pseudoinverse/fig3_eckart_young.png)

Left: rapidly decaying singular values; the amber tail$\sigma_{k+1}, \sigma_{k+2}, \ldots$ is exactly what gets discarded, and its$\ell_2$ norm equals the approximation error. Right: truncated SVD versus 8 trials of "random rank-$k$ projections" -- any random low-rank compression is **dominated by Eckart-Young by a wide margin**.

## Why "throwing the tail away" works: image compression demo

![Low-rank approximation as image compression](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/low-rank-approximation-pseudoinverse/fig1_lowrank_compression.png)

Take a$96 \times 96$ image (9 216 numbers) and reconstruct from$k = 2, 8, 32$ singular values:

| rank$k$| stored numbers | as % of full | relative Frobenius error |
| --- | --- | --- | --- |
| 2 | 386 | 4% | 16.3% |
| 8 | 1 544 | 17% | 4.0% |
| 32 | 6 176 | 67% | 1.7% |

The bottom row shows the **singular-value spectrum** and **cumulative energy curve**: a small handful of singular values carries almost all of the matrix's "signal." That single observation is why SVD compresses images, denoises signals, and extracts features. **The faster the spectrum decays, the more "low-rank-friendly" the matrix is.**

## Application: PCA = SVD of centered data

PCA computes the SVD of the centered data matrix$X_c = U\Sigma V^{\!\top}$. The columns of$V$ are the principal-component directions; the variances along them are$\sigma_i^2 / (n-1)$. Projecting onto the top$k$ components is exactly the rank-$k$ truncated SVD of$X_c$ -- so PCA isn't a separate algorithm; it's **Eckart-Young, specialized to centered data**.

## Application: matrix factorization for recommender systems

![Recommender systems: a sparse rating matrix is approximated by a low-rank factorization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/low-rank-approximation-pseudoinverse/fig4_recommender_mf.png)

The Netflix-Prize idea: the user-item rating matrix$R$ is **sparse but low rank**, because a small number of latent factors (genre preference, style, price sensitivity, etc.) explain most ratings. Write$R \approx UV^{\!\top}$ with$U \in \mathbb{R}^{n_u \times r}$,$V \in \mathbb{R}^{n_i \times r}$, and a small$r$ (typically 10-200). The figure shows that with only 34% of entries observed, a rank-3 alternating-least-squares fit gets **held-out RMSE down to 0.24** on a 1-5 rating scale.

> A subtlety: SVD itself cannot handle missing entries directly, so production systems solve a **masked, non-convex variant**: `min_{U,V} sum_{(i,j) observed} (R_{ij} - u_i^T v_j)^2`. ALS or SGD does the work, with$\ell_2$ regularization for free generalization (see the next section).

# Numerical stability and regularization

## How small singular values explode the answer

Inside$A^{+} = V \Sigma^{+} U^{\!\top}$ sits$\Sigma^{+}_{ii} = 1/\sigma_i$. If some$\sigma_i$ is near zero, then$1/\sigma_i$ is huge, and **any tiny noise component in the$u_i$ direction is amplified into a numerical disaster**. The standard quantitative measure is the **condition number**

$$\kappa(A) \;=\; \frac{\sigma_{\max}}{\sigma_{\min}}.$$

When$\kappa(A)$ is large (say$10^{10}$), the relative error in solving$Ax = b$ scales like$\kappa(A)$ times the input error -- you lose roughly 10 digits of precision.

## Fix 1: truncated SVD (hard threshold)

The simplest cure: **drop singular values that are numerically too small**.

$$\Sigma^{+}_{ii} = \begin{cases} 1/\sigma_i, & \sigma_i > \tau \\ 0, & \sigma_i \le \tau. \end{cases}$$

Standard choice:$\tau = \mathrm{rcond} \cdot \sigma_1$. This is what `np.linalg.pinv`'s `rcond` parameter does. Truncated SVD treats "signal in small-singular-value directions" as **noise** -- only safe when you actually believe nothing real lives there.

## Fix 2: Tikhonov regularization (soft threshold = ridge regression)

Modify the objective:

$$\min_{x} \;\|Ax - b\|_2^2 \;+\; \lambda \|x\|_2^2,$$

so the normal equations become

$$x_\lambda \;=\; (A^{\!\top}A + \lambda I)^{-1} A^{\!\top} b \;=\; V \,\mathrm{diag}\!\left(\tfrac{\sigma_i}{\sigma_i^2 + \lambda}\right)\, U^{\!\top}\, b.$$

Compared with the hard$1/\sigma_i$, Tikhonov uses the **soft filter**$\sigma_i / (\sigma_i^2 + \lambda)$:

- large singular values ($\sigma_i^2 \gg \lambda$):$\sigma_i / (\sigma_i^2 + \lambda) \approx 1/\sigma_i$, essentially unchanged;
- small singular values ($\sigma_i^2 \ll \lambda$):$\sigma_i / (\sigma_i^2 + \lambda) \approx \sigma_i / \lambda \to 0$, smoothly suppressed.

This is **ridge regression**. Its essence is: **add a low-pass filter, in the singular-value spectrum, to the pseudoinverse.**

```python
def ridge_pinv(A, lam):
    """Tikhonov-regularized pseudoinverse, equivalent to (A^T A + lam I)^-1 A^T."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_reg = s / (s * s + lam)
    return Vt.T @ np.diag(s_reg) @ U.T
```

## Which one to use?

| Situation | Choice |
| --- | --- |
| Small singular values are **pure numerical noise** | Truncated SVD (more aggressive) |
| Small singular values may carry **real but weak signal** | Tikhonov (smoother) |
| High-dim sparse features, want **feature selection** |$\ell_1$ (Lasso, not covered here) |
| Unsure, want hyperparameter tuning | Tikhonov + cross-validation |

# A modern surprise: from Eckart-Young to LoRA

The most striking modern application of low-rank approximation is **parameter-efficient fine-tuning of large language models**.

![LoRA: low-rank approximation makes fine-tuning ~100x cheaper](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/low-rank-approximation-pseudoinverse/fig5_lora_connection.png)

## LoRA's core hypothesis

A pretrained weight$W \in \mathbb{R}^{d \times k}$ (e.g.,$d = k = 4096$, giving$1.7 \times 10^7$ parameters per layer). Full fine-tuning updates all of$W$. **LoRA hypothesizes** that the *task-specific* update$\Delta W$ is itself **low rank**:

$$W' \;=\; W + \Delta W, \qquad \Delta W \;=\; B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k).$$

Parameter count drops from$dk$ to$r(d + k)$. The middle panel above: at$r = 8$, only 0.07 M parameters are trained -- **256x fewer** than full fine-tuning of one layer.

## Why does it work?

The right panel shows the empirical evidence: across many tasks, the singular-value spectrum of$\Delta W$ **decays sharply** -- the effective rank is in the single or low double digits. That's Eckart-Young telling us **a low-rank approximation of$\Delta W$ is essentially lossless**.

> This insight isn't unique to LoRA. Aghajanyan et al. (2020) measured "intrinsic dimensionality" and found that updating only a few hundred dimensions reaches ~90% of full fine-tuning quality. LoRA distilled the observation into a clean, composable, mergeable engineering recipe -- and became the canonical method.

## The unifying picture

| Concept | Form | Common skeleton |
| --- | --- | --- |
| Truncated SVD | $A_k = \sum_{i=1}^k \sigma_i u_i v_i^{\!\top}$| Approximate a high-rank matrix by a low-rank one |
| PCA | $X_c \approx U_k \Sigma_k V_k^{\!\top}$| Same, applied to centered data |
| Recommender MF | $R \approx U V^{\!\top}$| Same, with a missing-entry mask |
| LoRA | $\Delta W = B A$| Same, applied to **parameter updates** |

All four are children of **Eckart-Young**: keep signal in the dominant directions; let go of the rest.

# Practical checklist

- Before solving$Ax = b$, **always look at the condition number** (`np.linalg.cond(A)`). If$\kappa > 10^{12}$, do not trust the raw solution.
- For robust least squares, always go through SVD (`np.linalg.lstsq` or `np.linalg.pinv`); never hand-code$(A^{\!\top}A)^{-1}A^{\!\top}$.
- For PCA, **center the data** then take the SVD; do not eigendecompose the covariance matrix directly (numerically worse and more expensive).
- For matrix factorization (recommenders), pick the rank$r$ via held-out RMSE and add an$\ell_2$ regularizer.
- For LoRA, start at$r = 8$ with$\alpha = 2r$. Lower only if you've measured task complexity.

# Conclusion

The pseudoinverse extends "matrix inverse" to arbitrary matrices through one unifying optimization: **least-squares solution + minimum-norm tie-breaker**. The **SVD simultaneously gives the stable algorithm for$A^{+}$ and the optimal low-rank approximation (Eckart-Young).** Real-world ill-conditioning almost always traces back to small singular values, and there are exactly two cures: truncate (hard threshold) or Tikhonov (soft filter). The same low-rank thread runs from least squares through ridge regression, PCA, recommender-system matrix factorization, all the way to modern LoRA fine-tuning. They look like different methods; underneath, they are all variations on the same Eckart-Young theme: **keep the dominant directions, let the noisy ones go**.

# References

- [Moore-Penrose pseudoinverse (Wikipedia)](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
- [Singular Value Decomposition (Wikipedia)](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Tikhonov regularization (Wikipedia)](https://en.wikipedia.org/wiki/Tikhonov_regularization)
- Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika, 1(3), 211-218.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. arXiv:2012.13255.
