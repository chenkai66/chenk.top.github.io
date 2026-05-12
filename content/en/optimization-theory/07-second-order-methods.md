---
title: 'Optimization (7): Second-Order Methods'
date: 2022-09-22 09:00:00
tags:
  - ML
  - Optimization
  - Numerical Methods
categories: Algorithm
series: optimization-theory
series_order: 7
lang: en
mathjax: true
description: "Second-order methods break the sqrt(kappa) barrier by using curvature. We prove Newton's quadratic local convergence, derive BFGS from a secant condition + low-rank update, walk through L-BFGS's two-loop recursion that makes it the workhorse for medium-scale ML, and analyze trust-region subproblems with the dogleg solution."
disableNunjucks: true
translationKey: "optim-07"
---

First-order methods top out at $O(\sqrt{\kappa})$ iterations to reach $\epsilon$-accuracy (article 05). Second-order methods break this barrier by using curvature: Newton's method has **quadratic** local convergence — the number of correct digits doubles every iteration — and quasi-Newton methods retain most of this speed without computing the Hessian explicitly.

The cost is in the per-iteration work: Newton solves an $n \times n$ linear system per step ($O(n^3)$), BFGS maintains an $n \times n$ matrix ($O(n^2)$ per step + $O(n^2)$ memory), and L-BFGS uses only $O(mn)$ memory for a chosen history $m$ (typically 5--20).

This article gives the convergence proofs, derives the BFGS update from the secant condition, walks through the L-BFGS two-loop recursion line by line, and explains trust-region methods (which use the same Hessian information but with a different globalization strategy).

## What you will learn

1. Newton's method: derivation from second-order Taylor approximation, proof of local quadratic convergence under standard assumptions.
2. Globalization via line search (Wolfe conditions) and via trust regions.
3. The secant condition and how it constrains a quasi-Newton update.
4. BFGS as the unique rank-2 update satisfying the secant condition + symmetry + positive-definiteness, plus a quick derivation.
5. L-BFGS two-loop recursion: why it computes $H_k g_k$ in $O(mn)$ time without ever forming $H_k$.
6. Trust-region methods: subproblem, Cauchy point, dogleg, when to use them.

## Prerequisites

Articles 01--02 (convex analysis, smoothness, strong convexity). Linear algebra fluency: matrix inverse, rank-1 updates, Sherman--Morrison formula.

---

## 1. Newton's method

### 1.1 Derivation

For a twice-differentiable $f$, the second-order Taylor expansion around $x_k$ is
$$
f(x_k + d) \approx f(x_k) + \nabla f(x_k)^\top d + \tfrac{1}{2} d^\top \nabla^2 f(x_k) d.
$$
Minimizing the right-hand side over $d$ (assuming $\nabla^2 f(x_k) \succ 0$) gives
$$
d_k^N = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k).
$$
This is the **Newton direction**. The pure Newton iteration is $x_{k+1} = x_k + d_k^N$.

Geometrically: Newton's method approximates $f$ by the local quadratic and jumps to the minimum of that quadratic. If $f$ is itself a quadratic, Newton converges in one step.

### 1.2 Quadratic local convergence

> **Theorem.** Suppose $f$ is twice continuously differentiable, $\nabla^2 f$ is $L$-Lipschitz (i.e., $\|\nabla^2 f(x) - \nabla^2 f(y)\| \leq L \|x - y\|$), and $\nabla^2 f(x^\star) \succeq \mu I$ at a stationary point $x^\star$. Then for $x_0$ close enough to $x^\star$, Newton's method converges with
> $$
> \|x_{k+1} - x^\star\|_2 \leq \frac{L}{2 \mu} \|x_k - x^\star\|_2^2.
> $$

**Proof.** By definition of the Newton step,
$$
x_{k+1} - x^\star = x_k - x^\star - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k).
$$
Since $\nabla f(x^\star) = 0$,
$$
\nabla f(x_k) = \nabla f(x_k) - \nabla f(x^\star) = \int_0^1 \nabla^2 f(x^\star + t(x_k - x^\star)) (x_k - x^\star) \, dt.
$$
Substituting:
$$
x_{k+1} - x^\star = [\nabla^2 f(x_k)]^{-1} \int_0^1 [\nabla^2 f(x_k) - \nabla^2 f(x^\star + t(x_k - x^\star))] (x_k - x^\star) \, dt.
$$
The integrand is bounded in norm by $L (1 - t) \|x_k - x^\star\|_2$. Integrating gives $\|\cdot\| \leq \frac{L}{2} \|x_k - x^\star\|_2^2$. Combined with $\|[\nabla^2 f(x_k)]^{-1}\| \leq 1/\mu$ (for $x_k$ close enough to $x^\star$), we get the claim. $\blacksquare$

The "doubling of digits" is concrete: if $\|x_k - x^\star\| = 10^{-3}$, then $\|x_{k+1} - x^\star\| \leq C \cdot 10^{-6}$, then $C^2 \cdot 10^{-12}$, etc. To go from $10^{-3}$ to $10^{-12}$ takes 2 iterations, not the $\log(10^9) / \log(\sqrt{\kappa})$ a first-order method would need.

### 1.3 The catch: globalization

The convergence theorem only guarantees fast convergence **in a neighborhood of $x^\star$**. Far from $x^\star$, the Newton direction may not even be a descent direction (when $\nabla^2 f \not\succ 0$), and the step size may be too large.

The standard fix is **damped Newton**: $x_{k+1} = x_k + \alpha_k d_k^N$ where $\alpha_k$ is chosen by a line search to satisfy the **Wolfe conditions**:
- **Sufficient decrease (Armijo):** $f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f(x_k)^\top d_k$, typical $c_1 = 10^{-4}$.
- **Curvature:** $\nabla f(x_k + \alpha d_k)^\top d_k \geq c_2 \nabla f(x_k)^\top d_k$, typical $c_2 = 0.9$ for Newton-like methods.

Once $x_k$ is close to $x^\star$, the unit step $\alpha_k = 1$ satisfies both conditions and the algorithm transitions automatically into the quadratic-convergence regime.

### 1.4 When the Hessian is indefinite

If $\nabla^2 f(x_k) \not\succeq 0$, the Newton direction may point uphill. Common fixes:

- **Modified Cholesky:** factor $\nabla^2 f + E$ where $E$ is the smallest diagonal perturbation that makes the matrix positive definite. The result is a descent direction biased toward Newton.
- **Trust region** (section 4): bound the step length and choose the direction inside the trust region; this handles indefinite Hessians directly.
- **Cubic regularization** (Nesterov & Polyak, 2006): minimize the model $\nabla f^\top d + \tfrac{1}{2} d^\top \nabla^2 f \, d + \tfrac{M}{6} \|d\|^3$ instead. This has global convergence to second-order critical points.

---

## 2. Quasi-Newton methods: the secant equation

Newton needs $\nabla^2 f$, which costs $O(n^2)$ to store and $O(n^3)$ to invert per step. **Quasi-Newton** methods build an approximation $B_k \approx \nabla^2 f(x_k)$ from gradient differences alone, mimicking the curvature implicit in $\nabla f(x_k) - \nabla f(x_{k-1})$.

### 2.1 The secant condition

For a quadratic $f(x) = \frac{1}{2} x^\top A x - b^\top x$,
$$
\nabla f(x_{k+1}) - \nabla f(x_k) = A (x_{k+1} - x_k).
$$
Defining $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$, we have $A s_k = y_k$. For non-quadratic $f$ this becomes the **secant condition**:
$$
B_{k+1} s_k = y_k. \tag{Sec}
$$
Any quasi-Newton update should preserve this: the new approximation $B_{k+1}$ should reproduce the curvature observed in the most recent step.

### 2.2 BFGS: the canonical quasi-Newton update

The BFGS (Broyden--Fletcher--Goldfarb--Shanno) update is the unique rank-2 update of $B_k$ that:

1. Satisfies (Sec).
2. Is symmetric: $B_{k+1} = B_{k+1}^\top$.
3. Stays positive definite if $B_k$ is and $y_k^\top s_k > 0$ (which holds whenever the step satisfies the curvature Wolfe condition).
4. Minimizes a weighted Frobenius norm $\|B_{k+1} - B_k\|$ subject to (Sec) and symmetry.

The closed form is:
$$
B_{k+1} = B_k - \frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k} + \frac{y_k y_k^\top}{y_k^\top s_k}.
$$

For algorithmic purposes we usually want the inverse $H_k = B_k^{-1}$ directly (so we can compute $d_k = -H_k \nabla f(x_k)$ without solving a linear system). Applying the Sherman--Morrison--Woodbury identity twice gives
$$
H_{k+1} = (I - \rho_k s_k y_k^\top) H_k (I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top, \quad \rho_k = \frac{1}{y_k^\top s_k}. \tag{BFGS}
$$
This is the form actually implemented.

### 2.3 Why BFGS works in one paragraph

If $B_k$ approximates $\nabla^2 f$ well in the directions seen so far ($s_0, \ldots, s_{k-1}$), the BFGS update preserves that information while adding the new direction $s_k$. Over $n$ linearly-independent steps on a quadratic, BFGS reconstructs the exact Hessian and from then on behaves like Newton — the **finite termination property**.

For non-quadratic $f$, BFGS achieves **superlinear convergence**: $\|x_{k+1} - x^\star\| / \|x_k - x^\star\| \to 0$. The proof (Dennis & Moré, 1974) is delicate; the intuition is that as $x_k \to x^\star$, the secant condition forces $B_k$ to approximate $\nabla^2 f(x^\star)$ on the directions actually used by the algorithm.

---

## 3. L-BFGS: limited memory

For $n = 10^6$, BFGS would need $10^{12}$ floats (8 TB) just to store $H_k$. **L-BFGS** ("limited memory" BFGS) keeps only the last $m$ pairs $(s_i, y_i)$ — typically $m = 5$ to $20$ — and reconstructs the action $H_k g$ on demand using the **two-loop recursion**.

### 3.1 The two-loop recursion

Given:
- Current gradient $g = \nabla f(x_k)$
- History $\{(s_i, y_i)\}_{i = k-m}^{k-1}$ with $\rho_i = 1 / (y_i^\top s_i)$
- An initial Hessian-inverse approximation $H_k^0$ (commonly $H_k^0 = (s_{k-1}^\top y_{k-1} / y_{k-1}^\top y_{k-1}) I$ — a scalar multiple of identity)

The recursion computes $H_k g$ in $O(mn)$:

```
q ← g
for i = k-1, k-2, ..., k-m:           # first loop, "going backward"
    α_i ← ρ_i s_i^T q
    q ← q - α_i y_i

r ← H_k^0 q                           # apply initial scaling

for i = k-m, k-m+1, ..., k-1:         # second loop, "going forward"
    β ← ρ_i y_i^T r
    r ← r + (α_i - β) s_i

return r                              # r = H_k g
```

Each loop touches each pair once; the total work is $4mn$ inner products plus a vector scaling — totally bypassing the $O(n^2)$ matrix update.

### 3.2 Where the two-loop comes from

Apply (BFGS) recursively to expand $H_k$ in terms of $H_k^0$ and the pairs $(s_i, y_i)$. The first loop unwinds the right-most factor $(I - \rho_i y_i s_i^\top)$ for each $i$ from $k-1$ down to $k-m$, applied to $g$. Multiplying by $H_k^0$ gives the middle. The second loop applies the left factors $(I - \rho_i s_i y_i^\top)$ in the reverse order. The $\alpha_i$ values are reused because they appear symmetrically in both factors. (Nocedal & Wright, *Numerical Optimization*, Algorithm 7.4 has the full derivation.)

### 3.3 Practical L-BFGS

- **Initial $H_0$.** $H_0 = \gamma_k I$ with $\gamma_k = (s_{k-1}^\top y_{k-1}) / (y_{k-1}^\top y_{k-1})$ is the standard choice; it provides scale-invariance.
- **Memory size $m$.** $m = 5$ is a sensible default; $m = 20$ for problems where the gain matters. Larger $m$ does not help past a point because old curvature pairs become irrelevant.
- **Skipping pairs.** If $y_k^\top s_k \leq \epsilon \|s_k\| \|y_k\|$ (say $\epsilon = 10^{-8}$), the curvature condition fails and including the pair would damage positive definiteness; skip it.
- **Line search.** L-BFGS needs a proper Wolfe-condition line search to maintain $y_k^\top s_k > 0$. A pure Armijo backtracking can break BFGS.

L-BFGS is the default solver for many ML problems: PyTorch's `torch.optim.LBFGS`, scikit-learn's `LogisticRegression` for moderate-scale problems, scipy's `minimize` family. It dominates first-order methods for problems that fit in memory and are not too noisy.

---

## 4. Trust-region methods

Line search asks "in which direction?" then "how far?". Trust region asks both at once: **within a region of trust around $x_k$, what is the best step?**

### 4.1 The subproblem

At iterate $x_k$ with gradient $g_k$ and Hessian (or approximation) $B_k$, define the model
$$
m_k(d) = f(x_k) + g_k^\top d + \tfrac{1}{2} d^\top B_k d.
$$
The trust-region subproblem is
$$
d_k^\star = \arg\min_{\|d\|_2 \leq \Delta_k} m_k(d),
$$
where $\Delta_k > 0$ is the trust radius. After computing $d_k^\star$, evaluate the **agreement ratio**
$$
\rho_k = \frac{f(x_k) - f(x_k + d_k^\star)}{m_k(0) - m_k(d_k^\star)}.
$$
If $\rho_k$ is close to 1 the model is good; expand the trust region. If $\rho_k$ is poor or negative, contract the region and reject the step. Standard schedule: shrink $\Delta_k$ by 4 if $\rho_k < 0.25$, expand by 2 if $\rho_k > 0.75$ and the step lies on the boundary.

### 4.2 Solving the subproblem

The exact solution requires the **Moré--Sorensen** algorithm: find $\lambda \geq 0$ such that $(B_k + \lambda I) d = -g_k$ and $\lambda (\|d\|_2 - \Delta_k) = 0$. This is exact but expensive.

Two cheap approximations:

**Cauchy point.** Step along $-g_k$ as far as the model allows or the trust region permits:
$$
d_k^C = -\tau \frac{\Delta_k}{\|g_k\|_2} g_k, \quad \tau = \begin{cases} 1 & g_k^\top B_k g_k \leq 0 \\ \min\big(\frac{\|g_k\|_2^3}{\Delta_k g_k^\top B_k g_k}, 1\big) & \text{otherwise.} \end{cases}
$$
The Cauchy point gives at most a Cauchy decrease — comparable to gradient descent — but is always defined.

**Dogleg.** When $B_k \succ 0$, compute the unconstrained Newton step $d_k^N = -B_k^{-1} g_k$ and the steepest-descent step $d_k^{SD} = -\frac{g_k^\top g_k}{g_k^\top B_k g_k} g_k$. The dogleg path is
- If $\|d_k^N\|_2 \leq \Delta_k$: take $d_k = d_k^N$.
- Else if $\|d_k^{SD}\|_2 \geq \Delta_k$: take $d_k = -\Delta_k \frac{g_k}{\|g_k\|_2}$ (Cauchy direction, capped).
- Else: take the linear combination that lies on the trust region boundary, giving a quasi-Newton step with reduced length.

The dogleg path is a "broken line" from $0$ to $d_k^{SD}$ to $d_k^N$. The model decrease along this path is monotone, so the best feasible point is where the path meets the trust region boundary (or $d_k^N$ if interior).

### 4.3 Convergence

Trust-region methods with Cauchy decrease are **globally convergent** to a stationary point — i.e., $\|\nabla f(x_k)\| \to 0$ — under mild assumptions on $B_k$ (uniformly bounded). When $B_k = \nabla^2 f(x_k)$ and the iterates are close to a strict minimum, trust regions inherit Newton's quadratic convergence.

Trust regions are the method of choice for problems with non-convex Hessians (the model handles indefiniteness gracefully) and for very high-precision solutions where each step's quality matters more than per-step cost.

---

## 5. Choosing among the second-order methods

| Method        | Per-step cost  | Memory    | Convergence near $x^\star$ | When to use                                       |
| ------------- | -------------- | --------- | -------------------------- | ------------------------------------------------- |
| Newton (full) | $O(n^3)$       | $O(n^2)$  | Quadratic                  | Small $n$, exact Hessian available, indefinite handled |
| BFGS          | $O(n^2)$       | $O(n^2)$  | Superlinear                | Medium $n$ ($10^3$--$10^4$), good gradients       |
| L-BFGS        | $O(mn)$        | $O(mn)$   | Superlinear                | Large $n$ ($10^4$--$10^7$), the ML default        |
| Trust region  | $O(n^3)$ exact | $O(n^2)$  | Quadratic                  | Indefinite Hessians, high-precision needs          |
| Gauss--Newton | $O(n m^2)$     | $O(n^2)$  | Quadratic in residual      | Nonlinear least squares ($f = \frac{1}{2} \|r(x)\|^2$) |

For machine learning at the scale of millions of parameters, second-order methods are usually a poor fit because (a) gradients are noisy stochastic estimates, (b) memory is constrained, and (c) we don't need high precision — the validation error stops decreasing well before the optimization gap becomes small. SGD-type methods dominate.

For classical scientific computing — physics simulations, parameter estimation, signal recovery on convex domains — L-BFGS and trust-region Newton are the workhorses. They achieve solutions that gradient descent cannot reach in any reasonable wall-clock time.

---

## 6. Summary

Second-order methods break the $\sqrt{\kappa}$ barrier by using curvature, at a per-iteration cost. The hierarchy:

- **Newton** — fast, expensive, fragile; needs globalization.
- **BFGS** — Newton's curvature, half the cost, full memory.
- **L-BFGS** — BFGS in $O(mn)$ memory; the modern default.
- **Trust region** — robustness to indefinite Hessians and bounded steps.

The next article (08) tackles constrained optimization using Lagrangian duality — the same Newton machinery applies, just to the augmented system.

## Further reading

- Nocedal & Wright, *Numerical Optimization* (2nd ed.), Ch. 3 (line search), 4 (trust region), 6 (BFGS), 7 (L-BFGS) — the canonical reference.
- Boyd & Vandenberghe, *Convex Optimization*, §9.5 (Newton with damping) and §9.6 (self-concordance).
- Nesterov & Polyak, *Cubic regularization of Newton method and its global performance*, MathProg 108, 2006 — modern non-convex Newton.
- Liu & Nocedal, *On the limited memory BFGS method for large scale optimization*, MathProg 45, 1989 — the L-BFGS paper.
