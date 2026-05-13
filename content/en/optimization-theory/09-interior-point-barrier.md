---
title: 'Optimization (9): Interior-Point Methods and Self-Concordant Barriers'
date: 2022-09-26 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 9
lang: en
mathjax: true
description: "How interior-point methods became the default solver for convex programming: replace inequalities with a logarithmic barrier, parametrize the central path, and apply Newton's method. Includes the self-concordance machinery and the proof of the celebrated O(sqrt(n) log(1/eps)) iteration complexity."
disableNunjucks: true
translationKey: "optim-09"
---

In 1984 Karmarkar showed that LPs could be solved in polynomial time *practically* — not just theoretically (the ellipsoid method had achieved this on paper). His **interior-point method** stayed inside the feasible polytope and converged in $O(n L)$ iterations, far better than the simplex method's exponential worst case. Within a decade, Nesterov & Nemirovski generalized this to **all convex programming** via the **self-concordant barrier** framework. The result — $O(\sqrt{n} \log(1/\epsilon))$ Newton iterations for an $n$-dimensional problem — remains the gold standard for medium-scale convex optimization.

This article unpacks the machinery in three layers:

1. **The barrier method**: replace inequalities with a logarithmic penalty and trace the central path as the penalty weight increases.
2. **Self-concordance**: the analytic property of the barrier that makes Newton's method behave well — convergence regions of size $O(1)$ instead of $O(\mu / L)$.
3. **Primal-dual interior-point**: the modern variant that solves primal and dual simultaneously, used in essentially every commercial LP/QP/SDP solver.

We give the central-path complexity proof in full.

## What You Will Learn

1. The logarithmic barrier and the central path.
2. Self-concordance: definition, three key implications, the $\sqrt{\nu}$ parameter.
3. Damped Newton on a self-concordant function — quadratic convergence in a region of constant size.
4. The barrier method's iteration complexity: $O(\sqrt{\nu} \log(\nu / \epsilon))$ outer Newton steps.
5. Primal-dual interior-point methods and why they dominate in practice.

## Prerequisites

Articles 02 (smoothness), 07 (Newton's method), 08 (Lagrangian, KKT). Some comfort with directional derivatives and reading "Hessian metric" arguments.

---

## 1. The barrier method

Consider the convex problem

$$
\min_x f_0(x) \quad \text{s.t. } f_i(x) \leq 0, \ i = 1, \ldots, m, \quad Ax = b.
$$

Replace the inequalities with a **logarithmic barrier**

$$
\phi(x) = -\sum_{i=1}^m \log(-f_i(x)),
$$

which is finite on the strict interior $\{x : f_i(x) < 0\}$ and blows up at the boundary.
![Logarithmic barrier on a 1D interval](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/09-interior-point-barrier/fig1.png)
*Figure 1. The logarithmic barrier $-\log x - \log(4-x)$ on the open interval $(0,4)$. It is smooth and strictly convex inside the feasible set and diverges to $+\infty$ as $x$ approaches either boundary. The minimizer of the barrier alone is the **analytic center** of the feasible set.*
 For each $t > 0$, solve

$$
\min_x \quad t f_0(x) + \phi(x), \quad Ax = b. \tag{$P_t$}
$$

This is an equality-constrained convex problem; its solution $x^\star(t)$ traces out the **central path**.

### 1.1 Central path properties

For each $t > 0$:

- $x^\star(t)$ is the unique minimizer of ($P_t$) (under mild conditions).
- The KKT conditions for ($P_t$) give $\lambda_i(t) := 1/(t \cdot (-f_i(x^\star(t)))) \geq 0$ and they satisfy a **perturbed complementary slackness**:

  $$
  \lambda_i(t) (-f_i(x^\star(t))) = 1/t.
  $$

  This is the classical KKT system except complementarity is shifted by $1/t$ instead of being exactly zero.

- The duality gap on the central path is exactly $m/t$:

  $$
  f_0(x^\star(t)) - p^\star \leq m / t.
  $$

  Why? Define $\nu(t) = (\nu_1(t), \ldots, \nu_p(t))$ from the equality constraints. Then $(\lambda(t), \nu(t))$ is dual feasible and the Lagrangian evaluated at this dual point is exactly $f_0(x^\star(t)) - m/t$, by direct calculation.

So **as $t \to \infty$, $x^\star(t) \to x^\star$**, and the duality gap shrinks like $1/t$.
![Central path on a 2D polytope](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/09-interior-point-barrier/fig2.png)
*Figure 2. Central path $x^\star(t)$ on a 2D polytope. As the barrier weight $t$ grows, the minimizer moves smoothly from the **analytic center** (where the barrier dominates) toward the **optimum** $x^\star$ at a vertex (where the linear objective dominates). Light gray contours show level sets of the linear objective $c^\top x$.*
 This gives the most basic interior-point algorithm: choose $t = m/\epsilon$, solve ($P_t$), and you have $\epsilon$-suboptimality.

### 1.2 The naive algorithm and its problem

For very large $t$, ($P_t$) becomes badly conditioned — the term $t f_0$ dominates near the boundary and Newton's method has trouble. The fix is to solve a sequence of problems with increasing $t$, **warm-started** from the previous solution:

```sql
Algorithm: Barrier method
Input: strictly feasible x_0, initial t_0 > 0, target tolerance ε
       update factor μ > 1 (typical: μ = 10 or 100)

for k = 0, 1, 2, ...:
    Solve (P_{t_k}) starting from x_k by damped Newton, get x_{k+1}.
    if m / t_k < ε: stop
    t_{k+1} = μ * t_k
return x_{k+1}
```

Each outer iteration multiplies $t$ by $\mu$. The number of outer iterations is $\log(m / (t_0 \epsilon)) / \log \mu$, typically 20--50. The crucial question is **how many Newton steps each inner solve takes** — and this is where self-concordance comes in.

---

## 2. Self-concordance

A convex function $\phi : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is **self-concordant** if for every $x \in \mathrm{dom}(\phi)$ and every direction $u$,

$$
\Big| \frac{d^3}{dt^3} \phi(x + tu) \Big|_{t=0} \Big| \leq 2 \big(u^\top \nabla^2 \phi(x) u \big)^{3/2}.
$$

That is, the third directional derivative is controlled by the Hessian metric.

The most important examples:

- $\phi(x) = -\log x$ on $(0, \infty)$: $\phi'(x) = -1/x$, $\phi''(x) = 1/x^2$, $\phi'''(x) = -2/x^3$. Check: $|\phi'''| / (\phi'')^{3/2} = 2 / x^3 / x^{-3} = 2$. Tight.
- $\phi(X) = -\log \det X$ on $\mathbf{S}^n_{++}$: self-concordant.
- The barrier $-\sum_i \log(-f_i(x))$ for affine $f_i$: self-concordant. (More generally for $f_i$ that are themselves "nice".)

### 2.1 Why self-concordance matters

Self-concordance has three magical consequences:

**1. The Newton decrement is meaningful.** Define

$$
\lambda_\phi(x) := \sqrt{\nabla \phi(x)^\top [\nabla^2 \phi(x)]^{-1} \nabla \phi(x)}.
$$

This is the natural scale-invariant measure of distance-to-optimum for $\phi$. For self-concordant $\phi$,

$$
\phi(x) - \phi^\star \leq \lambda_\phi(x)^2 \quad \text{whenever} \quad \lambda_\phi(x) \leq 0.68.
$$

**2. Damped Newton makes progress with $O(1)$-sized constants.** Consider damped Newton: $x_+ = x - \frac{1}{1 + \lambda} \, [\nabla^2 \phi(x)]^{-1} \nabla \phi(x)$ with $\lambda = \lambda_\phi(x)$. For self-concordant $\phi$,

$$
\phi(x_+) \leq \phi(x) - \omega(\lambda),
$$

where $\omega(\lambda) = \lambda - \log(1 + \lambda)$. As long as $\lambda \geq \frac{1}{4}$, we have $\omega(\lambda) \geq 0.02$, so each Newton step decreases $\phi$ by at least a constant.

**3. Quadratic convergence in a constant region.** If $\lambda_\phi(x) \leq \frac{1}{4}$, then a *full* Newton step gives

$$
\lambda_\phi(x_+) \leq 2 \lambda_\phi(x)^2.
$$

This is quadratic convergence — and the convergence radius $\frac{1}{4}$ is **independent of conditioning**.

These properties are what makes Newton's method robust on a self-concordant function: "damped Newton phase" with constant decrease until $\lambda \leq 1/4$, then "quadratic phase" finishing in $O(\log \log(1/\epsilon))$ steps.
![Two phases of Newton on a self-concordant function](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/09-interior-point-barrier/fig4.png)
*Figure 4. **Left:** the per-step decrease $\omega(\lambda) = \lambda - \log(1+\lambda)$ stays above an absolute constant ($\geq 0.02$) whenever $\lambda \geq 1/4$, so the **damped phase** burns down $\phi$ at a constant rate. **Right:** once the Newton decrement enters the **quadratic region** $\lambda \leq 1/4$, full Newton steps satisfy $\lambda_+ \leq 2\lambda^2$ and convergence is dramatic. Crucially the boundary $1/4$ is independent of conditioning.*


### 2.2 Putting it together for the barrier method

For each outer iteration of the barrier method, we Newton-solve $\min_x t f_0(x) + \phi(x)$. If $f_0$ is self-concordant (or more generally if $t f_0 + \phi$ is) and we warm-start from $x^\star(t/\mu)$ (the previous solution), the warm start lies in the quadratic-convergence region of the new objective. So **each outer iteration costs $O(1)$ Newton steps**, independent of $t$ and the problem size.

Total Newton step count = (number of outer iterations) $\times$ $O(1)$ = $O(\log(m / \epsilon))$.

But there is a subtlety: we need a sharper bound to capture the effect of the **barrier parameter** $\nu$.

---

## 3. The barrier parameter and the $\sqrt{\nu}$ rate

A self-concordant function $\phi$ has **barrier parameter** $\nu \geq 1$ if for all $x \in \mathrm{dom}(\phi)$,

$$
\nabla \phi(x)^\top [\nabla^2 \phi(x)]^{-1} \nabla \phi(x) \leq \nu.
$$

Equivalently: $\lambda_\phi(x)^2 \leq \nu$ everywhere.

For $\phi = -\sum_{i=1}^m \log(-f_i(x))$ with affine $f_i$, $\nu = m$. For $\phi = -\log \det X$ on $n \times n$ PSD matrices, $\nu = n$.

The barrier parameter controls how much the central path moves with $t$:

$$
\|x^\star(t) - x^\star(t')\|_{\nabla^2 \phi} \leq O(\sqrt{\nu} \log(t'/t)).
$$

If we update $t$ by a factor $\mu = 1 + 1/\sqrt{\nu}$ (much smaller than $\mu = 10$), the warm start stays inside the quadratic-convergence region for the new central path point. The total number of outer iterations becomes

$$
\frac{\log(m / (t_0 \epsilon))}{\log(1 + 1/\sqrt{\nu})} = O(\sqrt{\nu} \log(\nu / \epsilon)),
$$

and each inner solve takes $O(1)$ Newton steps. This is the **short-step interior-point algorithm** with celebrated complexity.

> **Theorem (Nesterov--Nemirovski, 1994).** Solving a convex program with self-concordant barrier of parameter $\nu$ to accuracy $\epsilon$ requires $O(\sqrt{\nu} \log(\nu/\epsilon))$ Newton iterations.

For LP with $m$ inequality constraints, $\nu = m$ and the complexity is $O(\sqrt{m} \log(m/\epsilon))$ — Karmarkar's bound. For SDP on $n \times n$ matrices, $\nu = n$, giving $O(\sqrt{n} \log(n/\epsilon))$.

### 3.1 Long-step vs short-step

The textbook short-step method ($\mu = 1 + 1/\sqrt{\nu}$, full Newton steps) has the cleanest theory but is slow in practice — many small outer iterations. **Long-step** algorithms ($\mu = 10$ or $100$, damped Newton) violate the strict theory but converge much faster empirically. The theoretical complexity becomes $O(\nu \log(\nu/\epsilon))$ in the worst case, but the practical performance often matches short-step.

Modern solvers use **predictor-corrector** schemes (Mehrotra, 1992): predict the central path direction with one Newton step, then correct with a second-order term. This is the basis of every commercial LP/QP/SDP solver.
![Barrier weight schedule and duality gap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/09-interior-point-barrier/fig3.png)
*Figure 3. Two outer-iteration schedules for the barrier method on an LP with $m=200$ constraints. Long-step ($\mu=10$) inflates $t_k$ aggressively and reaches tolerance in $\sim 8$ outer iterations; short-step ($\mu = 1 + 1/\sqrt{\nu}$) takes $\sim 90$ tiny steps but each inner solve is provably $O(1)$ Newton steps. Both produce a duality gap $m/t_k$ that decays geometrically.*


---

## 4. Primal-dual interior-point methods

The barrier method updates the primal $x$ explicitly and recovers the dual $(\lambda, \nu)$ as a byproduct. **Primal-dual** methods update both simultaneously.

### 4.1 The primal-dual system

For the inequality-constrained LP $\min c^\top x$ s.t. $Ax = b, x \geq 0$, the central-path conditions are:

$$
\begin{aligned}
A x &= b \quad \text{(primal feasibility)} \\
A^\top \nu + \lambda &= c \quad \text{(dual feasibility)} \\
x_i \lambda_i &= 1/t \quad \text{(perturbed CS)} \\
x, \lambda &\geq 0
\end{aligned}
$$

This is a system in $(x, \nu, \lambda)$ with $1/t$ as a perturbation. Newton's method on this system, with $1/t \to 0$, converges to the primal-dual optimal pair.

### 4.2 Why primal-dual is preferred

- **No need to find a strictly feasible starting point.** Primal-dual methods can start from infeasible iterates and converge to feasibility along the way.
- **Better numerical conditioning.** The Newton system has block structure that can be exploited (and is more stable than the pure primal Newton system as $t \to \infty$).
- **Self-correcting**: errors in $x$ get corrected through $\lambda$ and vice versa.

The vast majority of solvers (Mosek, Gurobi for QP, SDPT3 for SDP, OSQP for QP) implement primal-dual interior-point methods with predictor-corrector and adaptive step sizes.
![Primal-dual residuals over iterations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/09-interior-point-barrier/fig5.png)
*Figure 5. Convergence of a Mehrotra predictor-corrector primal-dual interior-point method. The primal residual $\|Ax-b\|$, dual residual $\|A^\top \nu + \lambda - c\|$ and duality gap $x^\top \lambda$ all drop together at a geometric rate. The **predictor** step computes the affine direction along the central path; the **corrector** step adds a centering correction so the next iterate stays close to the central path.*


---

## 5. Where interior-point shines and where it doesn't

**Where interior-point dominates:**

- LPs and QPs up to $n \approx 10^5$ variables.
- SDPs, second-order cone programs (SOCPs).
- Convex problems with structured constraints (chordal sparsity, etc.).
- High-precision solutions: 12+ digits of accuracy in tens of iterations.

**Where first-order wins:**

- Very large $n$ ($> 10^7$) with sparse structure: each Newton step is $O(n^3)$ in the worst case, so first-order methods scale better.
- Stochastic / streaming data: interior-point requires the full problem in memory.
- Coarse-precision problems where $\epsilon = 10^{-3}$ suffices: SGD and Adam dominate.

The classical workflow in convex modeling: write the problem in CVXPY, the modeling layer translates to standard SDP / SOCP form, an interior-point solver returns the optimum to 10-digit accuracy in seconds. This pipeline made convex optimization a routine engineering tool.

---

## 6. Worked example: LP via the barrier

Consider minimizing $c^\top x$ s.t. $Ax \leq b$ with $A \in \mathbb{R}^{m \times n}$. The barrier objective for parameter $t$:

$$
B_t(x) = t c^\top x - \sum_{i=1}^m \log(b_i - a_i^\top x).
$$

The gradient and Hessian:

$$
\nabla B_t(x) = t c + A^\top \frac{1}{b - Ax}, \quad \nabla^2 B_t(x) = A^\top \mathrm{diag}\big( (b - Ax)^{-2} \big) A.
$$

(Here division is componentwise.)

The Newton direction is the solution of the linear system $\nabla^2 B_t(x) \, \Delta x = -\nabla B_t(x)$. For sparse $A$, this is solved via sparse Cholesky factorization in $O(n^3)$ worst case but typically much less.

A small worked instance with $n = 100$, $m = 200$ to high precision typically takes 30 outer iterations and $\sim 100$ total Newton steps — orders of magnitude faster than first-order methods would need.

---

## 7. Summary

| Concept                        | Role                                                                       |
| ------------------------------ | -------------------------------------------------------------------------- |
| Logarithmic barrier $\phi$     | Smooth penalty replacing inequality constraints.                            |
| Central path $x^\star(t)$      | Smooth curve from analytic center to the optimum as $t \to \infty$.         |
| Self-concordance               | Analytic property making Newton converge in $O(1)$-radius regions.          |
| Barrier parameter $\nu$        | Controls the step size between central-path points; gives the $\sqrt{\nu}$ rate. |
| Primal-dual IPM                | Modern variant; the basis of all commercial convex solvers.                 |

This completes the deterministic, exact-information optimization story. Articles 10 and 11 turn to **stochastic** methods (gradients with noise) and **non-convex** landscapes (no convergence-to-global guarantees) — the regimes where deep learning lives.

## References

- Nesterov & Nemirovski, *Interior-Point Polynomial Algorithms in Convex Programming*, SIAM, 1994 — the foundational monograph.
- Boyd & Vandenberghe, *Convex Optimization*, Ch. 11 — the cleanest exposition for engineers.
- Renegar, *A Mathematical View of Interior-Point Methods*, SIAM, 2001 — self-concordance from a geometric angle.
- Wright, *Primal-Dual Interior-Point Methods*, SIAM, 1997 — the practical algorithms.
- Mehrotra, *On the Implementation of a Primal-Dual Interior Point Method*, SIOPT 2, 1992 — the predictor-corrector scheme used in every modern solver.
