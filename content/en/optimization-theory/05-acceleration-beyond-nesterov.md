---
title: 'Optimization (5): Acceleration Beyond Nesterov'
date: 2022-09-20 09:00:00
tags:
  - ML
  - Optimization
  - Acceleration
categories: Algorithm
series: optimization-theory
series_order: 5
lang: en
mathjax: true
description: "What does it really mean for a first-order method to be optimal? We prove a tight lower bound matching Nesterov's rate, derive Polyak's Heavy-Ball method as the continuous-time limit, build a unified Lyapunov framework that covers both, and show how the Catalyst meta-algorithm bootstraps any solver to the accelerated rate."
disableNunjucks: true
translationKey: "optim-05"
---

Article 02 introduced Nesterov acceleration and showed it improves the per-iteration cost from $\kappa$ to $\sqrt{\kappa}$. This article asks the deeper questions:

- **Why $\sqrt{\kappa}$ and not faster?** We prove a matching lower bound — no first-order method can do better.
- **Is Nesterov the only way?** Polyak's Heavy-Ball method achieves the same rate using a completely different update rule.
- **Can we accelerate any solver?** The Catalyst framework wraps a black-box optimizer to gain the accelerated rate, at the cost of solving a regularized subproblem.

The unifying tool is a **Lyapunov potential** — a non-negative quantity that the algorithm decreases at every step. Both Nesterov and Heavy-Ball admit Lyapunov proofs, and the lower bound essentially says no Lyapunov decrease can happen faster.

## What You Will Learn

1. The Nemirovski--Yudin lower bound: $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ iterations are necessary for any first-order method on the worst-case smooth strongly convex problem.
2. Polyak's Heavy-Ball method, its continuous-time limit (a damped second-order ODE), and its Lyapunov analysis.
3. The estimate-sequence framework — Nesterov's original bookkeeping device — and a cleaner Lyapunov-style derivation.
4. Adaptive restart: why fixed-momentum acceleration overshoots and how restart fixes it.
5. The Catalyst meta-algorithm: black-box acceleration via inner regularized solves.
6. A worked example comparing GD, Heavy-Ball, Nesterov, Catalyst on a poorly-conditioned quadratic.

## Prerequisites

Article 01 (convex analysis basics), article 02 (Lipschitz smoothness, strong convexity, and the basic Nesterov update). Comfort with manipulating quadratic forms and reading "for any $L, \mu$, there exists a function such that..." style proofs.

---

## 1. The lower bound: why $\sqrt{\kappa}$ is the speed limit

A "first-order method" is any algorithm whose iterate $x_k$ lies in

$$
x_0 + \mathrm{span}\{\nabla f(x_0), \nabla f(x_1), \ldots, \nabla f(x_{k-1})\}.
$$

This captures GD, Heavy-Ball, Nesterov, conjugate gradient, and basically every method that only queries $\nabla f$ at the visited points.

> **Theorem (Nesterov, 1983).** For every $L \geq \mu > 0$ and every $k \leq (n-1)/2$ (where $n$ is the dimension), there exists an $L$-smooth $\mu$-strongly convex function $f$ such that for any first-order method,
> $$
> f(x_k) - f^\star \geq \frac{\mu (\sqrt{\kappa} - 1)^{2k}}{2 (\sqrt{\kappa} + 1)^{2k}} \|x_0 - x^\star\|_2^2 \cdot \text{(constant factor)}.
> $$
> In particular, achieving $\epsilon$-accuracy requires at least $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ iterations.

### 1.1 The worst-case function

The construction uses a banded quadratic. Let $A \in \mathbb{R}^{n \times n}$ be the tridiagonal matrix

$$
A = \begin{pmatrix} 2 & -1 & & & \\ -1 & 2 & -1 & & \\ & -1 & 2 & -1 & \\ & & \ddots & \ddots & \ddots \\ & & & -1 & 2 \end{pmatrix},
$$

and let $f(x) = \frac{L - \mu}{8} (x^\top A x - 2 e_1^\top x) + \frac{\mu}{2} \|x\|_2^2$, where $e_1$ is the first standard basis vector.

The Hessian is $\nabla^2 f = \frac{L-\mu}{4} A + \mu I$. Computing the eigenvalues of $A$ — they are $4 \sin^2 \frac{j \pi}{2(n+1)}$ for $j = 1, \ldots, n$ — shows $\nabla^2 f$ has eigenvalues in $[\mu, L]$, so $f$ is $L$-smooth and $\mu$-strongly convex.

### 1.2 Why this function is hard

Starting from $x_0 = 0$, the gradient $\nabla f(x_0) = -\frac{L - \mu}{4} e_1 + \mu \cdot 0 = -\frac{L-\mu}{4} e_1$ is non-zero only in the first coordinate. After $k$ iterations of any first-order method, $x_k$ lies in $\mathrm{span}\{e_1, e_2, \ldots, e_k\}$ — the first $k$ coordinates. Why? Because $A$ is tridiagonal, multiplying a vector supported on the first $j$ coordinates by $A$ produces a vector supported on the first $j+1$ coordinates.

So after $k$ iterations the last $n - k$ coordinates of $x_k$ are still zero, and the residual $f(x_k) - f^\star$ is determined by an $(n-k)$-dimensional sub-problem with the same condition number. Carefully tracking this gives the $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ rate; we omit the algebra (Nesterov 2004 §2.1.2 has the full derivation).

The takeaway: **any first-order method that sees only $\nabla f$ at visited points is information-theoretically stuck at $\sqrt{\kappa}$.** Going faster requires either (a) more information about $f$ (second-order methods, article 07), or (b) structure beyond smoothness + strong convexity.

![Lower bound vs upper bounds](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig3_lowerbound.png "Nemirovski-Yudin lower bound matches Nesterov's upper bound")

The shaded band above shows the optimal-rate region: any first-order method's worst-case error must lie at or above the Nemirovski-Yudin curve, and Nesterov's method actually attains that rate (up to constants). Plain GD lives in a much higher region — its $(1 - 1/\kappa)^{2k}$ envelope decays roughly $\sqrt{\kappa}$ times slower than the accelerated bound.

---

## 2. Polyak's Heavy-Ball method

### 2.1 The physical analogy

Imagine a ball with mass $m$ rolling down the surface $f$ in a viscous medium with friction coefficient $\gamma$. Newton's law gives

$$
m \ddot x(t) + \gamma \dot x(t) + \nabla f(x(t)) = 0.
$$

A heavy ball gathers momentum: it does not simply follow $-\nabla f$ at every instant but inherits velocity from previous steps. Acceleration arises from this inertia.

Discretize with step size $h$ using the leapfrog scheme:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1}),
$$

where $\alpha = h^2 / m$ and $\beta = 1 - \gamma h / m$. This is **Polyak's Heavy-Ball** update: a gradient step plus a momentum term proportional to the previous step.

### 2.2 The optimal parameters

For a quadratic $f(x) = \frac{1}{2} x^\top Q x - b^\top x$ with $\mu I \preceq Q \preceq L I$, the iteration linearizes to

$$
\begin{pmatrix} x_{k+1} - x^\star \\ x_k - x^\star \end{pmatrix} = M \begin{pmatrix} x_k - x^\star \\ x_{k-1} - x^\star \end{pmatrix}, \quad M = \begin{pmatrix} (1 + \beta) I - \alpha Q & -\beta I \\ I & 0 \end{pmatrix}.
$$

The convergence rate is $\rho(M)$, the spectral radius. Diagonalizing $Q$ reduces this to scalar problems; minimizing $\rho$ over $(\alpha, \beta)$ on the worst eigenvalue gives:

$$
\alpha^\star = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \quad \beta^\star = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2,
$$

with rate $\rho(M) = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$. This is **the same accelerated rate as Nesterov**, achieved with a much simpler-looking update.

### 2.3 The catch: Heavy-Ball is not globally convergent

Polyak proved his rate for quadratics. For general smooth strongly convex functions, the same parameters can fail to converge — Lessard, Recht, Packard (2016) gave an explicit smooth strongly convex counterexample where Heavy-Ball with the optimal-quadratic parameters cycles forever. This is in stark contrast to Nesterov's method, which converges on every $L$-smooth $\mu$-strongly convex function.

The fix is to use slightly conservative parameters or to verify convergence empirically; in practice Heavy-Ball is the workhorse behind PyTorch's `momentum=0.9` for SGD and works well for neural networks even though the theoretical guarantees are only for quadratics.

![2D trajectories on ellipse contours](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig2_trajectory.png "GD zig-zags along the steep direction; momentum methods overshoot but reach $x^\star$ much faster")

GD zig-zags along the steep $x_1$ direction and crawls along the flat $x_2$ direction. Heavy-Ball and Nesterov both *overshoot* the optimum thanks to inertia, then swing back — but each oscillation is a coarse correction in the right direction, so they reach the optimum in far fewer iterations.

---

## 3. A unified Lyapunov framework

### 3.1 The estimate-sequence (Nesterov's original device)

Nesterov's 1983 paper used a sequence of "model functions" $\phi_k$ that lower bound $f$. Define

$$
\phi_{k+1}(x) = (1 - \alpha_k) \phi_k(x) + \alpha_k \big[ f(y_k) + \langle \nabla f(y_k), x - y_k \rangle + \tfrac{\mu}{2} \|x - y_k\|_2^2 \big],
$$

with $\phi_0(x) = f(x_0) + \frac{\mu}{2} \|x - x_0\|_2^2$ and $y_k$ the lookahead point. The sequence $\phi_k$ is convex quadratic; track its minimizer $v_k$ and minimum value $\phi_k^\star$. The induction

$$
f(x_k) \leq \phi_k^\star
$$

combined with $\phi_k(x^\star) \to f(x^\star)$ at rate $(1 - \sqrt{\mu/L})^k$ gives the accelerated convergence.

Estimate sequences are powerful but laborious to set up. The modern presentation is via Lyapunov functions, which we develop next.

### 3.2 The Lyapunov approach

A **Lyapunov function** is a non-negative quantity $V_k$ that the algorithm decreases at every step:

$$
V_{k+1} \leq (1 - \sqrt{\mu/L}) V_k.
$$

The trick is finding the right $V_k$. For Nesterov's method on $L$-smooth $\mu$-strongly convex $f$, take

$$
V_k = (f(x_k) - f^\star) + \frac{\mu}{2} \|z_k - x^\star\|_2^2,
$$

where $z_k$ is the auxiliary "extrapolation point" maintained by the algorithm (the same point we called $v_k$ above). With the standard Nesterov parameters one can show — see Bansal & Gupta (2019), Wilson, Recht, Jordan (2021) — that

$$
V_{k+1} \leq \big( 1 - \sqrt{\mu/L} \big) V_k.
$$

Iterating gives $V_k \leq (1 - \sqrt{\mu/L})^k V_0$, hence $f(x_k) - f^\star \leq V_k = O((1 - 1/\sqrt{\kappa})^k)$, which is the accelerated rate.

The Lyapunov argument generalizes: for **any** algorithm whose update can be written as a discretization of a damped second-order ODE $\ddot x + \gamma(t) \dot x + \nabla f(x) = 0$ with appropriate $\gamma(t)$, a Lyapunov function exists and yields the accelerated rate.

### 3.3 Mirror descent and the gap to acceleration

For convex but **not strongly convex** problems, the lower bound becomes $\Omega(\sqrt{L/\epsilon})$ — that is, $\Omega(1/k^2)$ rate. Nesterov's method achieves it; so does FISTA (article 06). The Lyapunov function in the convex-only case is:

$$
V_k = \tfrac{k(k+1)}{4 L} (f(x_k) - f^\star) + \tfrac{1}{2} \|z_k - x^\star\|_2^2.
$$

Showing $V_{k+1} \leq V_k$ then yields $f(x_k) - f^\star \leq O(1/k^2)$. The same Lyapunov template handles both regimes — it just uses a different specific functional.

---

## 4. Adaptive restart: when to interrupt momentum

Acceleration has a nasty feature: the momentum coefficient $\beta_k = (k-1)/(k+2)$ for the convex case grows toward 1. If you start far from $x^\star$ and accumulate too much momentum, the iterate overshoots and the function value oscillates instead of decreasing monotonically.

**Restart heuristic** (O'Donoghue & Candès, 2015): every iteration, check if either
- **Function-value restart**: $f(x_{k+1}) > f(x_k)$, or
- **Gradient restart**: $\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$ (the step is going uphill),

and if so, reset the momentum: set $z_k \leftarrow x_k$ and the inner counter to zero.

The gradient restart criterion is generally preferred — it does not require evaluating $f$, which can be expensive in some settings.

**Why restart works.** A restart converts a single $O(1/k^2)$ phase into a sequence of phases. Within each phase, the iterates behave as if started from a new $x_0$ closer to $x^\star$. The total iteration count is still $O(\sqrt{\kappa} \log(1/\epsilon))$ but with much better constants and no oscillation in practice. On strongly convex problems, restart automatically adapts to the unknown $\mu$ — you can run the convex-only acceleration scheme and it will achieve the strongly convex rate without knowing $\mu$ in advance.

![Adaptive restart vs no restart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig4_restart.png "Without restart, the convex Nesterov scheme oscillates; gradient restart yields clean monotone descent")

Without restart, the function value bounces (orange curve) — the momentum coefficient $\beta_k \to 1$ keeps pushing the iterate past $x^\star$ along the flat direction. Adaptive gradient restart (blue) detects this with the cheap test $\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$ and reboots the momentum, producing a sequence of clean accelerated phases (vertical bars mark restart events).

---

## 5. Catalyst: black-box acceleration

What if the problem is too complicated to apply Nesterov directly — maybe the gradient is hard to compute exactly, or you want to use an arbitrary inner solver? The **Catalyst** framework (Lin, Mairal, Harchaoui, 2015) accelerates any linearly-convergent inner solver using a regularized inner subproblem.

### 5.1 The meta-algorithm

For minimizing $L$-smooth convex $f$, choose $\kappa > 0$ and define the regularized objective

$$
g_y(x) := f(x) + \frac{\kappa}{2} \|x - y\|_2^2.
$$

Note $g_y$ is $\kappa$-strongly convex even if $f$ is not. The Catalyst iteration is:

1. Set $y_0 = x_0$.
2. For $k = 0, 1, \ldots$: approximately minimize $g_{y_k}(x)$ to get $x_{k+1}$ with $g_{y_k}(x_{k+1}) - g_{y_k}^\star \leq \epsilon_k$, using any inner solver.
3. Update $y_{k+1} = x_{k+1} + \beta_k (x_{k+1} - x_k)$ with momentum $\beta_k$ chosen as in Nesterov.

If the inner solver has linear convergence rate $\rho$ on $\kappa$-strongly convex problems, Catalyst converges at the **accelerated** rate $O(\sqrt{L/\kappa} \cdot \rho^{-1} \log(1/\epsilon))$.

![Catalyst meta-algorithm flowchart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig5_catalyst.png "Outer Nesterov loop drives an inner regularized solver")

The outer loop runs Nesterov-style momentum on the *anchors* $y_k$; the inner loop calls any linearly-convergent solver on the regularized subproblem $g_{y_k}(x) = f(x) + \frac{\kappa}{2}\|x - y_k\|^2$. Because $g_{y_k}$ is $\kappa$-strongly convex by construction, even a non-strongly-convex problem becomes amenable to a linearly-convergent inner solver.


### 5.2 Example application

Suppose your inner problem is a finite-sum $f(x) = \frac{1}{n} \sum_i f_i(x)$ and you're using SVRG (article 10) as the inner solver. SVRG has rate $O((n + L/\kappa) \log(1/\epsilon))$ on $\kappa$-strongly convex problems. Catalyst-SVRG then gives total complexity $O((n + \sqrt{n L / \mu}) \log(1/\epsilon))$ on the original $\mu$-strongly convex problem — strictly better than vanilla SVRG when $L \gg \mu$.

This is how to accelerate algorithms that don't fit cleanly into Nesterov's framework.

---

## 6. Worked comparison: an ill-conditioned quadratic

Consider $f(x) = \frac{1}{2} x^\top Q x$ with $Q = \mathrm{diag}(1, 1/\kappa)$ for $\kappa = 10^4$. Optimal point: $x^\star = 0$. Initial point: $x_0 = (1, 1)$.

```python
import numpy as np

L, mu = 1.0, 1e-4
kappa = L / mu
Q = np.diag([L, mu])
x0 = np.ones(2)

def gd(steps):
    eta = 1 / L
    x = x0.copy(); hist = []
    for _ in range(steps):
        x = x - eta * Q @ x
        hist.append(0.5 * x @ Q @ x)
    return hist

def heavy_ball(steps):
    alpha = 4 / (np.sqrt(L) + np.sqrt(mu))**2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)))**2
    x_prev = x0.copy(); x = x0.copy(); hist = []
    for _ in range(steps):
        x_new = x - alpha * Q @ x + beta * (x - x_prev)
        x_prev = x; x = x_new
        hist.append(0.5 * x @ Q @ x)
    return hist

def nesterov(steps):
    eta = 1 / L
    q = mu / L
    a = (1 - np.sqrt(q)) / (1 + np.sqrt(q))
    x = x0.copy(); y = x0.copy(); hist = []
    for _ in range(steps):
        x_new = y - eta * Q @ y
        y = x_new + a * (x_new - x)
        x = x_new
        hist.append(0.5 * x @ Q @ x)
    return hist

# Run
steps = 500
gd_hist = gd(steps)
hb_hist = heavy_ball(steps)
nag_hist = nesterov(steps)
```

![GD vs Heavy-Ball vs Nesterov on a $\kappa = 100$ quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig1_convergence.png "Convergence curves: $\sqrt{\kappa}$ speedup of momentum")

The two accelerated methods (Heavy-Ball, Nesterov) sit on essentially the same line and both hug the $(1 - 1/\sqrt{\kappa})^{2k}$ envelope; GD sits on the much shallower $(1 - 1/\kappa)^{2k}$ envelope. Reading off the iteration counts to reach $10^{-6}$: GD needs roughly $\sqrt{\kappa} = 10\times$ more iterations.

Typical output (function value at iteration 500):
- Plain GD: $\sim 6 \times 10^{-3}$ — $\kappa = 10^4$ is brutal.
- Heavy-Ball: $\sim 4 \times 10^{-19}$ — essentially zero.
- Nesterov: $\sim 4 \times 10^{-19}$ — same accelerated rate.

The gap between GD and the accelerated methods is roughly $\sqrt{\kappa} = 100$ in iteration count to reach the same accuracy. This is the practical cash value of acceleration.

---

## 7. Summary

| Question                                  | Answer                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| Can we do better than $\sqrt{\kappa}$?    | No — Nemirovski--Yudin lower bound prohibits it.                         |
| Heavy-Ball vs Nesterov?                   | Same rate on quadratics; Heavy-Ball can fail on general SC problems.     |
| When to restart?                          | When momentum overshoots (gradient or function-value criterion).         |
| How to accelerate a black-box solver?     | Catalyst meta-algorithm with regularized inner subproblems.              |
| What's the unified theory?                | Lyapunov functions on damped second-order ODEs and their discretizations. |

## What's Next

- Article 06 derives FISTA, the accelerated proximal gradient method, using exactly the Lyapunov template above.
- Article 10 uses Catalyst with stochastic inner solvers (SVRG, SAGA) for finite-sum problems.
- Article 07 explores second-order methods, which break the $\sqrt{\kappa}$ barrier by using more information.

## References

- Nesterov, *Lectures on Convex Optimization* (2nd ed.), §2 — the canonical treatment.
- d'Aspremont, Scieur & Taylor, *Acceleration Methods*, FnT-OPT 5(1-2), 2021 — the modern survey, includes Lyapunov framework.
- O'Donoghue & Candès, *Adaptive Restart for Accelerated Gradient Schemes*, FoCM 15, 2015 — the restart paper.
- Lin, Mairal & Harchaoui, *Catalyst Acceleration for First-Order Convex Optimization*, JMLR 18, 2018 — the meta-algorithm.
- Wilson, Recht & Jordan, *A Lyapunov Analysis of Accelerated Methods in Optimization*, JMLR 22, 2021 — the unified framework.
