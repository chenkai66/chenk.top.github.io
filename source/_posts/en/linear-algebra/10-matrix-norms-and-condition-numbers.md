---
title: "Matrix Norms and Condition Numbers -- Is Your Linear System Healthy?"
date: 2025-02-20 09:00:00
tags:
  - Linear Algebra
  - Matrix Norms
  - Condition Numbers
  - Numerical Stability
  - Spectral Radius
description: "The condition number is a 'health report' for a linear system -- it tells you whether tiny input errors will explode into catastrophic output errors. This chapter covers vector norms, matrix norms, the spectral norm, condition numbers, and preconditioning."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 10
  total: 18
lang: en
mathjax: true
disableNunjucks: true
---

## The Question That Haunts Engineers

The equations are right. The algorithm is right. So why is the computed answer completely wrong?

The culprit is usually a single number called the **condition number**. It measures how *sensitive* a linear system is — whether a tiny wobble in the input gets amplified into a catastrophic error in the output. To talk about condition numbers we first need a way to measure the "size" of vectors and matrices. That is what norms do.

### What you will learn

- Vector norms ($L^1$, $L^2$, $L^\infty$) and the geometry of their unit balls.
- Matrix norms: Frobenius, induced, and the all-important spectral norm.
- The condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ and what it really means.
- Why ill-conditioned matrices (e.g. the Hilbert matrix) destroy double-precision arithmetic.
- How condition numbers bound the amplification of input error.
- Spectral radius and convergence of iterative methods.
- Preconditioning: how to tame an ill-conditioned system.

### Prerequisites

- Singular values and SVD (Chapter 9).
- Matrix multiplication and inverses (Chapter 3).
- Eigenvalues (Chapter 6).

---

## Vector Norms: Measuring Size

### What is a norm?

A **norm** on a vector space is a function $\|\cdot\|$ obeying three axioms:

1. **Non-negativity.** $\|\vec{x}\| \geq 0$, with equality iff $\vec{x} = \vec{0}$.
2. **Absolute homogeneity.** $\|c\vec{x}\| = |c|\,\|\vec{x}\|$.
3. **Triangle inequality.** $\|\vec{x} + \vec{y}\| \leq \|\vec{x}\| + \|\vec{y}\|$.

These match our everyday sense of "size". Nothing has negative size; doubling something doubles its size; a detour is never shorter than the direct path.

### The three norms you must know

For $\vec{x}\in\mathbb{R}^n$:

**$L^1$ norm — Manhattan distance.** $\|\vec{x}\|_1 = |x_1| + |x_2| + \cdots + |x_n|.$ A taxi in midtown can only drive along the grid; reaching $(3,4)$ from the origin costs $3+4=7$ blocks.

**$L^2$ norm — Euclidean distance.** $\|\vec{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}.$ The straight line "as the crow flies": a crow reaches $(3,4)$ in $\sqrt{9+16}=5$.

**$L^\infty$ norm — Chebyshev distance.** $\|\vec{x}\|_\infty = \max_i|x_i|.$ A chess king moves one square in any of eight directions; reaching $(3,4)$ costs $\max(3,4)=4$ moves because the king can travel diagonally and orthogonally simultaneously.

### The geometry of unit balls

Each norm carves out its own unit ball $\{\vec{x} : \|\vec{x}\| \leq 1\}$:

- **$L^1$:** diamond in 2D, octahedron in 3D — sharp corners on the axes.
- **$L^2$:** circle in 2D, sphere in 3D — perfectly rotation-invariant.
- **$L^\infty$:** square in 2D, cube in 3D — flat faces aligned with the axes.

As $p$ grows from $1$ to $\infty$, the unit ball morphs from diamond to circle to square. Those sharp corners on the $L^1$ ball are exactly why LASSO regression produces sparse solutions: the level sets of the loss tend to touch the $L^1$ constraint at a corner — a corner that lies on a coordinate axis, meaning some coordinates are forced to zero.

![Vector norms: the shape of the unit ball depends on p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig1_norm_unit_balls.png)

### Norm equivalence

**Theorem.** In a finite-dimensional space, all norms are *equivalent*: there exist constants $c, C > 0$ such that
$$c\|\vec{x}\|_a \leq \|\vec{x}\|_b \leq C\|\vec{x}\|_a.$$

Different norms are different rulers measuring the same vector. If a sequence converges in one norm, it converges in all of them. The choice is purely a matter of convenience or physical meaning — until we get to the *speed* of convergence, where the choice of norm starts to matter again.

---

## Matrix Norms: How Strong is the Transformation?

### Frobenius norm — treat the matrix as one long vector

For an $m\times n$ matrix $A$,
$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}.$$
Just flatten every entry into a single long vector and compute its $L^2$ norm.

**Image analogy.** Picture $A$ as a grayscale image, each entry a pixel intensity. The Frobenius norm is the image's *total energy*. It is the standard yardstick for "how different are these two matrices?" in everything from image-compression error to weight-update norms in neural networks.

**Connection to singular values.**
$$\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}.$$
So Frobenius is a *root-sum-of-squares* over **all** singular values — it cares about every direction $A$ stretches.

### Induced norms — the maximum amplification

The **induced norm** (or operator norm) measures the largest factor by which $A$ can stretch a unit vector:
$$\|A\| = \max_{\|\vec{x}\| = 1} \|A\vec{x}\|.$$

Think of $A$ as a magnifying lens: the induced norm asks *what is the biggest zoom this lens can apply?* If $\|A\|_2 = 3$, there exists some input direction whose length triples after applying $A$.

The three induced norms you will see most often:

| Norm | Formula | How to compute |
|---|---|---|
| $\|A\|_1$ (column sum) | $\max_j \sum_i |a_{ij}|$ | Sum the absolute values in each column, take the maximum |
| $\|A\|_2$ (spectral) | $\sigma_{\max}(A)$ | Largest singular value |
| $\|A\|_\infty$ (row sum) | $\max_i \sum_j |a_{ij}|$ | Sum the absolute values in each row, take the maximum |

**Memory trick:** $1$ → column sum, $\infty$ → row sum.

### The spectral norm — the star player

The spectral norm $\|A\|_2 = \sigma_1$ is the most useful matrix norm:

- It equals the **largest singular value**.
- Geometrically, $A$ maps the unit sphere to an ellipsoid; $\|A\|_2$ is the length of the longest semi-axis.
- It captures the **maximum amplification factor** of $A$.

![Operator norm: longest semi-axis of the image ellipse](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig2_operator_norm.png)

The two pictures above tell the whole story. On the left, the unit circle and two extremal directions $v_{\max}$ and $v_{\min}$. On the right, $A$ has rotated and stretched the circle into an ellipse; the orange arrow lands on the longest semi-axis with length $\sigma_{\max} = \|A\|_2$, while the green arrow shrinks to length $\sigma_{\min}$.

### Frobenius vs spectral: two answers to "how big?"

These two norms answer different questions about the same matrix. The spectral norm is the *peak stretch* — useful when you fear the worst direction. The Frobenius norm is the *aggregate stretch* — useful when you care about average behaviour.

![Frobenius vs spectral norm](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig3_frob_vs_spectral.png)

For the matrix shown, $\|A\|_2 = \sigma_1 \approx 2.78$ but $\|A\|_F \approx 2.93$. They are close because $\sigma_2$ is small relative to $\sigma_1$; for a matrix with many comparable singular values they would diverge.

### Submultiplicativity

For induced norms,
$$\|AB\| \leq \|A\|\,\|B\|.$$
Two transformations applied in sequence cannot amplify a vector by more than the product of their individual amplifications. This single inequality is what lets us prove convergence of iterative algorithms — and what controls the worst-case error of repeated multiplication in deep networks.

---

## The Condition Number: a Health Report

### Definition

For an invertible matrix $A$,
$$\kappa(A) = \|A\|\,\|A^{-1}\|.$$
The **spectral condition number** (almost always the one you want):
$$\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}.$$
The ratio of the largest to smallest singular value.

**Allergy analogy.**
- $\kappa \approx 1$: a healthy immune system. A draft does nothing.
- $\kappa \approx 10^{10}$: a severe allergy. A speck of pollen triggers anaphylaxis.

![Condition number: well-conditioned vs ill-conditioned](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig4_condition_number.png)

The well-conditioned matrix maps the input circle to a near-circle (the dashed grey baseline overlaps the green ellipse almost everywhere). The ill-conditioned matrix flattens the same circle into a needle: there are directions in input space along which the matrix barely registers, so to invert we have to amplify those tiny outputs back up — and any noise rides along with the signal.

### Key properties

- **Lower bound:** $\kappa(A) \geq 1$. Equality holds only for scalar multiples of orthogonal matrices.
- **Orthogonal invariance:** $\kappa(QA) = \kappa(A)$ when $Q$ is orthogonal — rotations and reflections cost nothing.
- **Scale invariance:** $\kappa(\alpha A) = \kappa(A)$ — the *shape* of the ellipse, not its size, decides health.
- **Inverse symmetry:** $\kappa(A^{-1}) = \kappa(A)$ — solving and inverting are equally hard.

### Geometric meaning in one line

The condition number is the **eccentricity** of the output ellipse:
$$\kappa = \frac{\text{longest semi-axis}}{\text{shortest semi-axis}}.$$
$\kappa = 1$ means the ellipse is a circle; $\kappa = \infty$ means it has collapsed to a line segment (the matrix is singular).

**Quick example.** $A = I$ has $\kappa = 1$. $B = \begin{pmatrix} 1 & 0 \\ 0 & 10^{-5} \end{pmatrix}$ has $\kappa = 10^5$ — it crushes one direction by a factor of $10^5$.

---

## Ill-Conditioned Matrices: The Nightmare

### The Hilbert matrix

The poster child for ill-conditioning is the **Hilbert matrix**
$$H_{ij} = \frac{1}{i+j-1}.$$
Its condition number grows terrifyingly fast.

| Size $n$ | $\kappa(H_n)$ | Reliable digits in double precision |
|---|---|---|
| 5  | $\sim 10^{5}$  | about 10 |
| 10 | $\sim 10^{13}$ | about 3  |
| 12 | $\sim 10^{16}$ | essentially 0 |
| 15 | $\sim 10^{18}$ | useless |

```python
import numpy as np
from scipy.linalg import hilbert

n = 12
H = hilbert(n)
x_true = np.ones(n)
b = H @ x_true
x_hat = np.linalg.solve(H, b)

print(f"kappa(H_{n})    = {np.linalg.cond(H):.2e}")
print(f"relative error = {np.linalg.norm(x_true - x_hat) / np.linalg.norm(x_true):.2e}")
```

By size 12, the relative error is of order 1: the answer can be 100% wrong even though the algorithm is "correct".

### Visualising the explosion

![Singular value spectrum and Hilbert κ(n) growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig6_singular_values.png)

The left panel shows two 20×20 matrices: a benign one with a flat singular spectrum, and a pathological one whose singular values plunge over six orders of magnitude. The right panel plots $\kappa_2(H_n)$: on a log axis it is essentially a straight line, doubling about every two rows. The dashed red line is the *double-precision wall* at $10^{16}$ — past it, no algorithm executed in IEEE 754 double precision can save you.

### Where ill-conditioning comes from

- **Polynomial fitting.** High-degree fits produce Vandermonde matrices whose condition number grows exponentially in the degree.
- **Fine meshes.** Refining a finite-element mesh by a factor of $h$ multiplies $\kappa$ by roughly $h^{-2}$.
- **Normal equations.** $\kappa(A^TA) = \kappa(A)^2$ — taking a least-squares problem and squaring the condition number is sometimes called *the original sin of numerical linear algebra*.
- **Near singularity.** Two rows or columns that are *almost* linearly dependent.

---

## Error Analysis: How Condition Numbers Amplify Errors

### Right-hand side perturbation

Suppose we solve $A\vec{x} = \vec{b}$ but the right-hand side is contaminated by a small $\delta\vec{b}$. How much does the solution change?

The classical bound:
$$\frac{\|\delta\vec{x}\|}{\|\vec{x}\|} \;\leq\; \kappa(A)\,\frac{\|\delta\vec{b}\|}{\|\vec{b}\|}.$$

The condition number is the **maximum amplification factor** for relative error.

Concretely:
- $\kappa = 10$: a 1% input error gives at most a 10% output error.
- $\kappa = 10^{10}$: a 1% input error can produce a $10^{8}$% output error — your answer is noise.
- $\kappa = 10^{16}$: even *machine-precision* rounding (≈ $10^{-16}$) is enough to destroy the answer.

We can see this happen experimentally. The figure below applies the same 2% wobble to $b$ in many different directions and plots the resulting cloud of solutions $x + \delta x$.

![Sensitivity to perturbation: tiny δb, huge δx](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig5_perturbation.png)

For $\kappa \approx 2$, the cloud is a tiny ring centred on the true $x$ — amplification factor ≈ 1. For $\kappa \approx 200$, the *same* relative wobble in $b$ sends the solution sliding along a long diagonal needle — the worst-case amplification is roughly $\kappa$ itself.

### Matrix perturbation

If the matrix $A$ itself is perturbed by $\delta A$,
$$\frac{\|\delta\vec{x}\|}{\|\vec{x}\|} \;\leq\; \kappa(A)\left(\frac{\|\delta A\|}{\|A\|} + \frac{\|\delta\vec{b}\|}{\|\vec{b}\|}\right).$$
Same condition number, same penalty: errors in the matrix entries get amplified by $\kappa(A)$ too.

### Rule of thumb: lost digits

If $\kappa(A) \approx 10^k$, solving $A\vec{x} = \vec{b}$ in double precision loses roughly $k$ significant digits.

| $\kappa(A)$ | Digits lost | Reliable digits |
|---|---|---|
| $10^{4}$  | 4  | ≈ 12 |
| $10^{8}$  | 8  | ≈ 8  |
| $10^{12}$ | 12 | ≈ 4  |
| $10^{16}$ | 16 | 0 (useless) |

---

## Spectral Radius and Iterative Convergence

### Definition

The **spectral radius** is the largest eigenvalue magnitude:
$$\rho(A) = \max_i |\lambda_i|.$$
It is bounded above by every matrix norm: $\rho(A) \leq \|A\|$ for any $\|\cdot\|$.

### The convergence criterion

The fixed-point iteration $\vec{x}_{k+1} = B\vec{x}_k + \vec{c}$ converges from any starting point **if and only if** $\rho(B) < 1$.

**Shower analogy.** Think of trying to reach the perfect water temperature. If the system is stable ($\rho < 1$) every adjustment moves you closer; if unstable ($\rho \geq 1$) the temperature oscillates between ice water and scalding.

**Convergence speed.** Each iteration shrinks the error by roughly $\rho(B)$, so to reach error $\epsilon$ takes about $\log\epsilon / \log\rho(B)$ iterations. Smaller $\rho$ means dramatically faster convergence.

### The Neumann series

If $\rho(A) < 1$,
$$(I - A)^{-1} = I + A + A^2 + A^3 + \cdots,$$
the matrix analogue of $\frac{1}{1-x} = 1 + x + x^2 + \cdots$. Useful in perturbation analysis and for cheap inverse approximations when $A$ is small.

---

## Numerical Stability: Choosing the Right Algorithm

### Why normal equations are dangerous

Least squares $\min_{\vec{x}}\|A\vec{x} - \vec{b}\|$ has the textbook solution via the **normal equations** $A^TA\hat{\vec{x}} = A^T\vec{b}$. The catch is fatal:
$$\kappa(A^TA) = \kappa(A)^2.$$
If $A$ has $\kappa = 10^{6}$, the normal equations have $\kappa = 10^{12}$ — you have manufactured your own catastrophe.

### QR: the stable alternative

Form $A = QR$ and solve $R\hat{\vec{x}} = Q^T\vec{b}$. Because $Q$ is orthogonal, the condition number of the system stays at $\kappa(A)$ — not squared. This is the default in essentially every modern least-squares routine.

### SVD: most stable, most expensive

Solve via the pseudoinverse $\hat{\vec{x}} = V\Sigma^+U^T\vec{b}$. SVD gracefully handles rank deficiency, returns the minimum-norm solution, and gives you the singular values for free. The cost is roughly 2–3× a QR solve.

### Seeing the difference

The plot below solves the same $50\times 50$ system $Ax = b$ with all three methods as we crank the condition number from $10^2$ up to $10^{14}$:

![Numerical stability: error vs condition number for three solvers](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/10-matrix-norms-and-condition-numbers/fig7_numerical_stability.png)

Three things to notice. (1) QR and SVD track each other; they ride the $\varepsilon_{\text{mach}}\cdot\kappa$ line all the way up. (2) The normal-equations curve is twice as steep on a log–log plot — that is the $\kappa^2$ tax. (3) Around $\kappa \approx 10^{8}$ the normal equations cross the "useless" line while QR/SVD still have eight reliable digits. *Pick your algorithm before the data hits.*

---

## Preconditioning: Taming Ill-Conditioned Systems

### The idea

Solve $A\vec{x} = \vec{b}$ but $\kappa(A)$ is huge. Find an easy-to-invert $M$ and solve instead
$$M^{-1}A\vec{x} = M^{-1}\vec{b}.$$
If $M \approx A$, then $M^{-1}A \approx I$ and $\kappa(M^{-1}A) \approx 1$.

**Analogy.** Trying to weigh an elephant on a kitchen scale is hopeless — but if you know the elephant's weight to within a kilogram, you only need the scale to measure the deviation. Preconditioning takes a problem that lives at the wrong scale and shifts it to one your tools can handle.

### Common preconditioners

| Preconditioner | $M$ | Pros | Cons |
|---|---|---|---|
| Jacobi | $\text{diag}(A)$ | Trivial to apply, perfectly parallel | Limited reduction |
| Gauss–Seidel | Lower-triangular part of $A$ | Better than Jacobi | Hard to parallelise |
| Incomplete LU | Sparse approximate $LU$ | Strong, general-purpose | Fill-in must be controlled |
| Incomplete Cholesky | Sparse approximate $LL^T$ | Half the storage | Symmetric positive definite only |

The trade-off is universal: a stronger preconditioner cuts the iteration count but costs more per iteration. Optimal preconditioners often exploit problem structure (geometric multigrid for PDEs, domain-decomposition for parallel solvers, …).

---

## Applications

### Finite element analysis

The stiffness matrix $K$ in structural mechanics has $\kappa(K) \propto h^{-2}$ where $h$ is the mesh size. Refining the mesh to capture geometric detail makes the matrix harder to solve. Highly non-uniform meshes or large material-property contrasts (think reinforced concrete, where steel and concrete differ by orders of magnitude in stiffness) make things worse — preconditioning is mandatory.

### Image deblurring

In $\vec{y} = B\vec{x} + \text{noise}$, the blur operator $B$ is typically extremely ill-conditioned: the inverse problem amplifies high-frequency noise. **Tikhonov regularisation** replaces the inverse problem with
$$\min_{\vec{x}} \|B\vec{x} - \vec{y}\|^2 + \lambda\|\vec{x}\|^2.$$
The regularisation parameter $\lambda$ trades fidelity for stability and effectively replaces $\sigma_{\min}$ with $\sqrt{\sigma_{\min}^2 + \lambda}$ inside the condition number — even a small $\lambda$ can rescue the problem.

### Deep learning: vanishing and exploding gradients

The spectral norm of weight matrices controls signal propagation:
- $\|W\|_2 > 1$ in many layers → forward signals (and backward gradients) blow up exponentially.
- $\|W\|_2 < 1$ in many layers → signals decay exponentially; gradients vanish.

The fixes — Batch Normalization, careful initialisation (Xavier, He), residual connections, gradient clipping — can all be read as ways of keeping the effective spectral norm of the network's Jacobian close to 1.

### Regularisation in machine learning

Ridge regression adds $\lambda I$ to $X^TX$:
$$\kappa(X^TX + \lambda I) = \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda}.$$
Even a modest $\lambda$ can cut the condition number by orders of magnitude. The same idea (Levenberg–Marquardt, trust regions, weight decay) reappears all over optimisation.

---

## Python Examples

### Computing norms and condition numbers

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]], dtype=float)

print(f"Frobenius norm : {np.linalg.norm(A, 'fro'):.4f}")
print(f"Spectral norm  : {np.linalg.norm(A, 2):.4f}")
print(f"1-norm         : {np.linalg.norm(A, 1):.4f}")
print(f"inf-norm       : {np.linalg.norm(A, np.inf):.4f}")
print(f"Condition #    : {np.linalg.cond(A):.4f}")
```

### Hilbert matrix experiment

```python
from scipy.linalg import hilbert
import matplotlib.pyplot as plt

sizes = range(2, 16)
conds = [np.linalg.cond(hilbert(n)) for n in sizes]

plt.semilogy(list(sizes), conds, "o-")
plt.xlabel("Matrix size n")
plt.ylabel("Condition number")
plt.axhline(1e16, ls="--")  # double precision wall
plt.title("Hilbert matrix: condition number explodes with n")
plt.show()
```

### Comparing least-squares methods

```python
def compare_methods(cond_target=1e8, n=50, seed=0):
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(cond_target), n)
    A = Q1 @ np.diag(s) @ Q2

    x_true = rng.standard_normal(n)
    b = A @ x_true

    x_normal = np.linalg.solve(A.T @ A, A.T @ b)        # the dangerous one
    Q, R = np.linalg.qr(A); x_qr = np.linalg.solve(R, Q.T @ b)
    x_svd = np.linalg.lstsq(A, b, rcond=None)[0]

    print(f"kappa(A)        : {np.linalg.cond(A):.2e}")
    print(f"normal eq error : {np.linalg.norm(x_normal - x_true):.2e}")
    print(f"QR error        : {np.linalg.norm(x_qr - x_true):.2e}")
    print(f"SVD error       : {np.linalg.norm(x_svd - x_true):.2e}")

compare_methods()
```

---

## Exercises

### Warm-up

1. Compute $\|\vec{x}\|_1$, $\|\vec{x}\|_2$, $\|\vec{x}\|_\infty$ for $\vec{x} = (3, -4, 0, 1)$.
2. Compute the Frobenius norm and spectral norm of $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$.
3. For a diagonal matrix $D = \text{diag}(d_1, \ldots, d_n)$, prove $\kappa_2(D) = \max|d_i| / \min|d_i|$.

### Going deeper

4. Prove $\rho(A) \leq \|A\|$ for any matrix norm.
5. Show that if $\|A\| < 1$ (induced norm), then $I - A$ is invertible.
6. Prove that for an orthogonal matrix $Q$, $\kappa_2(Q) = 1$.
7. Using `np.linalg.cond`, estimate $\kappa(H_n)$ for $n = 2, \ldots, 15$ and fit a line on a log scale. What is the empirical growth rate?

### Coding challenges

8. Implement $\|A\|_1$, $\|A\|_2$, $\|A\|_\infty$ from scratch (no `np.linalg.norm`).
9. Reproduce the stability plot in this chapter: error vs condition number for normal equations, QR, and SVD on a least-squares problem.
10. Implement Jacobi iteration with and without diagonal preconditioning. Plot iterations-to-convergence vs the spectral radius of the iteration matrix.

---

## Practical Guidelines

| $\kappa(A)$ | Risk | Recommendation |
|---|---|---|
| $< 10^{4}$ | Low | Standard methods are fine |
| $10^{4}$–$10^{8}$ | Medium | Verify the answer; prefer QR or SVD over normal equations |
| $10^{8}$–$10^{12}$ | High | Add regularisation or use a preconditioner |
| $> 10^{12}$ | Extreme | Step back and reformulate the problem |

---

## Chapter Summary

| Concept | Key formula | Intuition |
|---|---|---|
| Vector norm | $\|\vec{x}\|_p = (\sum |x_i|^p)^{1/p}$ | "How big is this vector?" |
| Frobenius norm | $\|A\|_F = \sqrt{\sum a_{ij}^2}$ | Total energy of the matrix |
| Spectral norm | $\|A\|_2 = \sigma_{\max}$ | Maximum amplification |
| Condition number | $\kappa = \sigma_{\max}/\sigma_{\min}$ | Bound on error amplification |
| Spectral radius | $\rho(A) = \max|\lambda_i|$ | Controls iterative convergence |
| Normal equations | $\kappa(A^TA) = \kappa(A)^2$ | Why they are dangerous |

---

## Series Navigation

**Previous:** [Chapter 9 -- Singular Value Decomposition](/en/chapter-09-singular-value-decomposition/)

**Next:** [Chapter 11 -- Matrix Calculus and Optimization](/en/chapter-11-matrix-calculus-and-optimization/)

*This is Chapter 10 of the 18-part "Essence of Linear Algebra" series.*

## References

- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
- Demmel, J. W. (1997). *Applied Numerical Linear Algebra*.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
