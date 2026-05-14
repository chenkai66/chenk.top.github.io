---
title: 'Optimization (8): Lagrangian Duality and KKT Conditions'
date: 2022-09-24 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 8
lang: en
mathjax: true
description: "How constraints become prices: the Lagrangian, weak duality, Slater's condition for strong duality, the KKT system as necessary and sufficient optimality, and why the SVM dual is much smaller than the SVM primal. Includes complete proofs and the saddle-point characterization."
disableNunjucks: true
translationKey: "optim-08"
---

The most consequential idea in constrained optimization is that **constraints have prices**. The Lagrangian transforms a constrained problem into an unconstrained one by attaching a non-negative multiplier to each inequality and a free multiplier to each equality. The resulting unconstrained problem may be easier (the SVM dual), or it may give a verifiable lower bound (the LP duality used to certify integer programs).

This article develops:

- **Weak duality:** the dual is always a lower bound on the primal — no assumptions needed.
- **Strong duality:** under Slater's condition (or convexity + linear constraints), the gap is zero.
- **KKT conditions:** primal stationarity + dual feasibility + complementary slackness, the practical optimality system.
- **Saddle-point characterization:** the Lagrangian's saddle point coincides with the optimal primal--dual pair.

Each result is proved or carefully cited. We close with the SVM example, where the dual cuts the problem dimension from $d$ (number of features) to $n$ (number of training points) — the original kernel-method magic.

## What You Will Learn

1. Constructing the Lagrangian and the dual function.
2. Proving weak duality (one-line argument).
3. Slater's condition and a clean proof of strong duality for convex problems.
4. The KKT system, when it is necessary, when it is sufficient.
5. Saddle-point view of the Lagrangian and its connection to game theory.
6. Worked example: the SVM dual.

## Prerequisites

[Articles 01](../01-convex-analysis-foundations/)--02 (convex sets, convex functions, subgradients, smoothness, strong convexity).

---

## The setup

Consider the **primal problem**
$$
\begin{aligned}
\text{minimize}\quad & f_0(x) \\
\text{subject to}\quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p,
\end{aligned} \tag{P}
$$
with optimal value $p^\star$. Define the **Lagrangian**
$$
L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x),
$$
with **dual variables** $\lambda \in \mathbb{R}^m_+$ (non-negative for inequalities) and $\nu \in \mathbb{R}^p$ (free for equalities).

Define the **dual function**
$$
g(\lambda, \nu) = \inf_x L(x, \lambda, \nu).
$$
The dual function is **always concave** in $(\lambda, \nu)$, regardless of whether $f_0, f_i, h_j$ are convex — it is a pointwise infimum of affine functions of $(\lambda, \nu)$.

The **dual problem** is
$$
\text{maximize}\quad g(\lambda, \nu) \quad \text{subject to } \lambda \geq 0, \tag{D}
$$
with optimal value $d^\star$.

---

## Weak duality

> **Theorem (Weak duality).** $d^\star \leq p^\star$.

**Proof.** For any feasible $x$ for (P) and any $(\lambda, \nu)$ with $\lambda \geq 0$,
$$
L(x, \lambda, \nu) = f_0(x) + \underbrace{\sum_i \lambda_i f_i(x)}_{\leq 0} + \underbrace{\sum_j \nu_j h_j(x)}_{= 0} \leq f_0(x).
$$
Taking the infimum over $x$ on the left,
$$
g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) \leq L(x, \lambda, \nu) \leq f_0(x).
$$
Taking the infimum over feasible $x$ on the right gives $g(\lambda, \nu) \leq p^\star$. Maximizing the left-hand side over $\lambda \geq 0, \nu$ gives $d^\star \leq p^\star$. $\blacksquare$

This theorem requires **no convexity** — it holds for any constrained optimization problem, however ugly. The dual gives a certificate of optimality: if you find primal-feasible $x$ and dual-feasible $(\lambda, \nu)$ with $f_0(x) = g(\lambda, \nu)$, then $x$ is optimal. This is the basis of branch-and-bound for integer programs.

The gap $p^\star - d^\star \geq 0$ is the **duality gap**. When it is zero, **strong duality** holds.

---


![Weak duality and the duality gap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig1.png)
*Figure 1 — Weak duality. The dual function $g(\lambda)$ never exceeds the primal optimum $p^\star$. Its supremum $d^\star$ is the best lower bound; the shaded region is the duality gap, which collapses to zero under strong duality.*

## Strong duality

### Slater's condition

> **Slater's condition.** There exists $x \in \mathrm{relint}(\mathrm{dom}(f_0))$ with $f_i(x) < 0$ for all non-affine $i$ and $h_j(x) = 0$ for all $j$. (Equivalently: a strictly feasible point exists; affine inequalities can be feasible without strictness.)

> **Theorem (Strong duality, convex case).** If (P) is convex (i.e., $f_0, f_i$ are convex and $h_j$ are affine) and Slater's condition holds, then $d^\star = p^\star$ and the dual optimum is attained.

**Proof sketch.** The proof uses a separating hyperplane argument applied to the **value function**
$$
V(u, v) = \inf\{f_0(x) : f_i(x) \leq u_i, h_j(x) = v_j\}.
$$
Convexity of (P) makes $V$ convex on its domain. Slater's condition guarantees $0 \in \mathrm{relint}(\mathrm{dom}(V))$. The dual function is the conjugate of $V$ restricted to non-negative multipliers:
$$
g(\lambda, \nu) = -V^*(-\lambda, -\nu) \quad \text{for } \lambda \geq 0.
$$
Conjugacy and the fact that $V$ is convex and lower-semicontinuous near $0$ give $V(0) = -V^{**}(0)$, which unwinds to $p^\star = d^\star$.

The full proof is in Boyd & Vandenberghe §5.3.2; the key step is to apply the supporting hyperplane theorem at the boundary point $(0, V(0))$ of the convex set $\{(u, t) : t \geq V(u)\}$. Slater's condition ensures the supporting hyperplane is non-vertical, which gives a finite multiplier. $\blacksquare$


![Value function and supporting hyperplane](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig4.png)
*Figure 4 — Strong duality via the value function. $V(u)$ is the optimal value of the constraint-perturbed problem; convexity plus Slater's condition yields a non-vertical supporting hyperplane at $u=0$ whose slope is exactly $-\lambda^\star$.*

### When Slater fails

If Slater's condition fails, strong duality may still hold (e.g., for LP it always holds without Slater) or it may fail with a positive duality gap. Standard pathological examples:

- $\min x^2/y$ over $y > 0$ — convex, has $p^\star = 0$, dual has $d^\star = -\infty$ (no Slater, no relint feasibility).
- A nonconvex problem can have any gap; SDP relaxations of QCQPs are a famous example where the gap is exactly the integrality gap.

For LPs and convex QPs, strong duality holds whenever (P) and (D) are both feasible. For SDPs, Slater's condition (a strictly positive-definite feasible point) is the typical hypothesis.

---

## The KKT conditions

If $x^\star$ is primal-optimal and $(\lambda^\star, \nu^\star)$ is dual-optimal under strong duality, the **Karush--Kuhn--Tucker** conditions hold:

| Condition                                                                | Name                              |
| ------------------------------------------------------------------------ | --------------------------------- |
| $\nabla f_0(x^\star) + \sum_i \lambda_i^\star \nabla f_i(x^\star) + \sum_j \nu_j^\star \nabla h_j(x^\star) = 0$ | Stationarity (primal)             |
| $f_i(x^\star) \leq 0, \ h_j(x^\star) = 0$                                | Primal feasibility                |
| $\lambda_i^\star \geq 0$                                                 | Dual feasibility                  |
| $\lambda_i^\star f_i(x^\star) = 0$ for all $i$                           | Complementary slackness           |

**Why these hold under strong duality.** Strong duality gives $f_0(x^\star) = L(x^\star, \lambda^\star, \nu^\star) \leq L(x, \lambda^\star, \nu^\star)$ for all $x$. So $x^\star$ minimizes $L(\cdot, \lambda^\star, \nu^\star)$ over $\mathbb{R}^n$, giving stationarity. Primal/dual feasibility hold by definition. The only non-trivial step is complementary slackness: from $L(x^\star, \lambda^\star, \nu^\star) = f_0(x^\star)$ we get $\sum_i \lambda_i^\star f_i(x^\star) = 0$, and since each term is $\leq 0$ they must all be zero individually.


![KKT geometric picture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig3.png)
*Figure 3 — KKT stationarity in two dimensions. At the optimum $x^\star$ the negative objective gradient $-\nabla f_0$ lies in the cone spanned by the active constraint gradients with non-negative weights $\lambda_i^\star$.*

### KKT as sufficient optimality (convex problems)

> **Theorem.** Suppose (P) is convex and $(x^\star, \lambda^\star, \nu^\star)$ satisfies the KKT conditions. Then $x^\star$ is primal-optimal and $(\lambda^\star, \nu^\star)$ is dual-optimal.

**Proof.** Stationarity says $\nabla_x L(x^\star, \lambda^\star, \nu^\star) = 0$. Since $L(\cdot, \lambda^\star, \nu^\star)$ is convex (sum of convex $f_0, \lambda_i^\star f_i$ with $\lambda_i^\star \geq 0$, and affine $\nu_j^\star h_j$), the stationarity condition implies $x^\star$ globally minimizes $L(\cdot, \lambda^\star, \nu^\star)$. Hence
$$
g(\lambda^\star, \nu^\star) = L(x^\star, \lambda^\star, \nu^\star) = f_0(x^\star) + \underbrace{\sum_i \lambda_i^\star f_i(x^\star)}_{= 0 \text{ by CS}} + \underbrace{\sum_j \nu_j^\star h_j(x^\star)}_{= 0} = f_0(x^\star).
$$
Thus weak duality is tight: $f_0(x^\star) = g(\lambda^\star, \nu^\star) \leq p^\star \leq f_0(x^\star)$, so $x^\star$ is optimal. $\blacksquare$

This is the result that makes KKT the practical workhorse: for convex problems, KKT gives a finite system of equations + inequalities whose solution is the optimum.

### When KKT fails

KKT is necessary at an optimum **only if a constraint qualification holds**. Slater's condition is one such qualification; LICQ (linear independence of active constraint gradients) is another. Without one, an optimum may have no Lagrange multipliers, and gradient-based methods that rely on the KKT system can stall.

For non-convex problems, KKT is necessary (with constraint qualification) but not sufficient — KKT points include local optima, saddle points of the Lagrangian, and even some non-stationary points.

---

## Saddle-point characterization

The Lagrangian gives a **min-max** game between primal player ($x$, minimizing) and dual player ($\lambda, \nu$, maximizing).

> **Theorem (Saddle-point principle).** Strong duality holds and $(x^\star, \lambda^\star, \nu^\star)$ is a primal-dual optimum iff $(x^\star, \lambda^\star, \nu^\star)$ is a saddle point of $L$:
> $$L(x^\star, \lambda, \nu) \leq L(x^\star, \lambda^\star, \nu^\star) \leq L(x, \lambda^\star, \nu^\star) \quad \forall x, \forall \lambda \geq 0, \nu.$$
The right inequality is primal optimality of $x^\star$ given the multipliers; the left is dual optimality.


![Saddle surface of the Lagrangian](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig2.png)
*Figure 2 — The Lagrangian as a saddle surface. Minimising over $x$ for each $\lambda$ traces the dual function (green); maximising over feasible $x$ for each $\lambda$ traces the primal value (orange). They meet at the saddle point under strong duality.*

Saddle-point characterization is the foundation of:

- **Augmented Lagrangian** methods: alternate primal and dual updates with a quadratic penalty for stability.
- **ADMM** (article 06): split the primal variable, then do primal-dual ascent on a special Lagrangian.
- **GAN training**: literally a saddle-point game (though non-convex, so the theory does not apply directly).
- **Online primal-dual algorithms**: regret bounds on both players guarantee approximate optimality.

---

## Worked example: the SVM dual

The hard-margin support vector machine on linearly separable data $\{(x_i, y_i)\}_{i=1}^n$ with $y_i \in \{-1, +1\}$:
$$
\begin{aligned}
\min_{w, b} \quad & \tfrac{1}{2} \|w\|_2^2 \\
\text{s.t.} \quad & y_i (w^\top x_i + b) \geq 1, \quad i = 1, \ldots, n.
\end{aligned}
$$
Lagrangian:
$$
L(w, b, \alpha) = \tfrac{1}{2} \|w\|_2^2 - \sum_i \alpha_i [y_i (w^\top x_i + b) - 1].
$$
Setting $\nabla_w L = 0$: $w^\star = \sum_i \alpha_i y_i x_i$. Setting $\partial_b L = 0$: $\sum_i \alpha_i y_i = 0$. Substituting back:
$$
g(\alpha) = -\tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i,
$$
giving the dual
$$
\begin{aligned}
\max_\alpha \quad & \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0.
\end{aligned}
$$
![SVM dual and support vectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig5.png)
*Figure 5 — SVM duality. The maximum-margin separator (black) is determined entirely by support vectors (purple rings); complementary slackness forces $\alpha_i^\star = 0$ for all interior points, and $w^\star = \sum_i \alpha_i^\star y_i x_i$ is a sparse weighted sum.*

Why this is a big deal:

1. **Variables**: dual has $n$ (number of training points), primal has $d + 1$ (number of features + bias). For $d \gg n$, the dual is much smaller.
2. **Kernel trick**: the dual depends on data only through $x_i^\top x_j$. Replace with $K(x_i, x_j)$ and you get nonlinear classifiers without ever touching the lifted feature space.
3. **Sparsity**: by complementary slackness, $\alpha_i > 0$ only when $y_i(w^\top x_i + b) = 1$ — the **support vectors**. Most $\alpha_i$ are zero.

The same structure powers kernel ridge regression, kernel PCA, and Gaussian processes: the dual sees data through inner products, which can be replaced by arbitrary positive-definite kernels.

---

## Where duality fails: noisy and large-scale ML

For $n = 10^9$ training examples (think large-scale ad CTR), the dual SVM has $10^9$ variables — worse than the primal. The dual's elegance breaks down at scale, which is one reason deep learning skipped duality entirely and went back to SGD on the primal.

For convex problems with strong duality, modern practice is:

- **Small to medium $n$** ($\leq 10^4$): solve the dual explicitly, often via QP solvers (libsvm).
- **Large $n$**: **stochastic dual coordinate ascent** (Shalev-Shwartz & Zhang, 2013) — pick one $\alpha_i$ at a time and optimize, retaining duality's guarantees with $O(n)$ memory.
- **Convex-concave saddle problems**: primal-dual methods (Chambolle--Pock, 2011) for image denoising / sparse recovery.

---

## Summary

| Concept                | What it gives you                                                            |
| ---------------------- | ---------------------------------------------------------------------------- |
| Weak duality           | A lower bound on the primal. Always holds.                                    |
| Slater's condition     | A sufficient condition for strong duality on convex problems.                 |
| Strong duality         | Zero gap between primal and dual; multipliers exist.                          |
| KKT system             | Necessary (with CQ) and sufficient (for convex) optimality conditions.        |
| Saddle-point view      | Min-max characterization; bridges to ADMM, augmented Lagrangian, GANs.        |
| Dual problem           | Often smaller, sparser, or kernelizable; the basis of SVMs.                   |

Article 09 takes the constrained problem and solves it via **interior-point methods**: replace the inequality constraints with a barrier and apply Newton's method. We will see that the central-path complexity is $O(\sqrt{n} \log(1/\epsilon))$, which makes interior-point methods the gold standard for medium-scale convex programming.

## References

- Boyd & Vandenberghe, *Convex Optimization*, Ch. 5 — the canonical treatment, every example worked.
- Bertsekas, *Convex Optimization Theory*, Ch. 5 — proofs from a more abstract duality theory.
- Rockafellar, *Conjugate Duality and Optimization*, 1974 — the deep theory of duality via conjugates.
- Shalev-Shwartz & Zhang, *Stochastic Dual Coordinate Ascent for Regularized Loss Minimization*, JMLR 14, 2013 — large-scale duality.
