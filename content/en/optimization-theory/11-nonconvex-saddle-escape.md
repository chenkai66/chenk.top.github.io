---
title: 'Optimization (11): Non-Convex Optimization and Saddle Escape'
date: 2022-09-29 09:00:00
tags:
  - ML
  - Optimization
  - Deep Learning Theory
categories: Algorithm
series: optimization-theory
series_order: 11
lang: en
mathjax: true
description: "Why does SGD work for training neural networks despite the non-convex landscape? We prove perturbed GD escapes strict saddles in polynomial time, derive convergence under the Polyak-Lojasiewicz condition, and survey what is provably known about deep learning loss surfaces — over-parameterization, NTK, and the implicit bias toward flat minima."
disableNunjucks: true
translationKey: "optim-11"
---

For non-convex $f$, gradient descent has no global guarantee. The best we can say is that $\nabla f(x_t) \to 0$ — we converge to a stationary point, which could be a local min, a saddle, or even a local max. This article asks: when can we say more?

Three positive results:

1. **Saddle escape**: under a "strict saddle" assumption, perturbed GD converges to local minima in polynomial time. Saddle points are unstable; Brownian noise (or just numerical perturbation) escapes them.
2. **PL condition**: a relaxation of strong convexity that holds in over-parameterized neural networks. Under PL, vanilla GD gets the linear rate $O(\log(1/\epsilon))$ even without convexity.
3. **Loss landscape facts**: for sufficiently wide neural networks, all local minima are global, and SGD's noise gives implicit bias toward flat minima with better generalization.

Each is rigorous in its setting. The article also discusses what is **not** known — there is no general theorem saying "SGD finds the global optimum of a deep network."


---

## What You Will Learn

1. Stationary points and the saddle-point classification ($\nabla^2 f$ eigenvalue signs).
2. The strict saddle property and the Ge--Huang--Jin--Yuan (2015) perturbed-GD escape proof.
3. The Polyak--Łojasiewicz (PL) condition and its convergence implications.
4. Over-parameterization, the NTK, and why all local minima can be global.
5. Implicit bias of SGD toward flat minima.

## Prerequisites

[Articles 02](../02-smoothness-strong-convexity-nesterov/) (smoothness), 03 (GD basics), 10 (stochastic methods).

---

## The non-convex landscape

For non-convex $f$, the first-order optimality condition $\nabla f(x) = 0$ has multiple solution types, classified by the Hessian:

| $\nabla^2 f(x^\star)$                          | Type                              |
| ---------------------------------------------- | --------------------------------- |
| $\succ 0$                                      | Strict local minimum              |
| $\preceq 0$                                    | Local maximum                     |
| Indefinite (some $+$, some $-$ eigenvalues)    | Saddle point                      |
| Singular ($0$ eigenvalue)                      | Degenerate; need higher-order info |

GD with small step size converges to a stationary point. **It does not distinguish among the four cases**: GD can stop at a saddle, and on a flat saddle (degenerate) it might never escape.

The classical worry: in dimension $d$, the number of saddles can grow exponentially. For random Gaussian polynomials, almost all critical points are saddles, not minima (Auffinger, Ben Arous, Černý 2013). So a priori we expect to get stuck.

The good news: **most saddles are escapable**.

![Stationary-point taxonomy by Hessian signature](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig2.png)

*Figure: the four classes of stationary points sorted by Hessian eigenvalue signature. Strict saddles (third panel) have at least one strictly negative eigenvalue and are escapable; degenerate critical points (fourth panel) have a zero eigenvalue and require higher-order information.*


---

## Saddle escape via perturbed GD

### Strict saddle property

A function $f$ has the **strict saddle property** if at every stationary point $x^\star$,
$$
\nabla^2 f(x^\star) \succ 0 \quad \text{or} \quad \lambda_{\min}(\nabla^2 f(x^\star)) < 0.
$$
That is, every stationary point is either a strict local min or has a strictly negative eigenvalue (a "strict" saddle, no flat directions).

Many machine-learning problems have the strict saddle property: orthogonal tensor decomposition, generalized phase retrieval, low-rank matrix completion, dictionary learning. For these problems, escaping saddles automatically reaches a local minimum, which is often global.

### Perturbed GD

```text
Algorithm: PGD (perturbed gradient descent)
Input: x_0, step η, perturbation radius r, threshold ε
for t = 0, 1, 2, ...:
    if ||∇f(x_t)|| ≤ ε and "no recent perturbation":
        x_t ← x_t + ξ_t  with ξ_t uniform on ball of radius r
    x_{t+1} = x_t - η ∇f(x_t)
```

The idea: at a stationary point with negative eigenvalue, a random perturbation $\xi_t$ has positive component along the negative eigenvector with overwhelming probability. The GD steps after the perturbation amplify this component (the matrix exponential in the negative-eigenvalue direction), pushing the iterate away from the saddle.

![3D saddle landscape with vanilla and perturbed GD trajectories](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig1.png)

*Figure: the canonical strict saddle $f(x,y) = x^2 - y^2$. Vanilla GD descends the $x$-direction and stalls at the saddle (grey). A small random perturbation gives a non-zero $y$-component, which the negative curvature amplifies exponentially — the iterate escapes (orange).*


> **Theorem (Jin et al. 2017).** For an $L$-smooth, $\rho$-Hessian-Lipschitz function with the strict saddle property, perturbed GD finds an $\epsilon$-second-order stationary point (i.e., $\|\nabla f\| \leq \epsilon$ and $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\rho \epsilon}$) in
> $$O\left( \frac{L \, (f(x_0) - f^\star)}{\epsilon^2} \log^4 d \right) \text{ iterations.}$$
This is the same dependence on $\epsilon$ as plain GD's first-order convergence — the polylog factor is the only price for the second-order guarantee.

The proof is intricate but the intuition is clean: count "stuck epochs" (where the function value barely decreases) and show that each one ends with high probability after $O(\log^4 d)$ iterations of perturbed GD. Each stuck epoch happens at a near-stationary point, and the post-perturbation trajectory escapes if $\lambda_{\min}(\nabla^2 f) < -\sqrt{\rho \epsilon}$.

### SGD's implicit perturbation

In the stochastic setting, the noise from $\nabla f_{i_t}$ already perturbs the iterate. The above analysis carries over: SGD escapes strict saddles in polynomial time without explicit perturbation. This is part of why deep learning practitioners do not worry about saddle points — the noise gets you out for free.

---

## The Polyak--Łojasiewicz (PL) condition

A function $f$ satisfies the **PL inequality** with parameter $\mu > 0$ if
$$
\tfrac{1}{2} \|\nabla f(x)\|_2^2 \geq \mu (f(x) - f^\star) \quad \forall x.
$$
PL is **weaker than strong convexity** — every $\mu$-strongly convex function is PL with the same $\mu$, but PL functions can be non-convex. Examples:

- The squared loss $\|Ax - b\|_2^2$ when $A$ has full row rank — PL but not strongly convex if $A$ is "wide".
- Over-parameterized neural networks at suitable initializations (Liu, Zhu & Belkin 2022) — PL on the loss landscape near the initialization in some regimes.
- Logistic regression with separable data — PL at infinity but not finite minimum.

![PL condition vs strong convexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig3.png)

*Figure: left — strong convexity forces a unique minimum, while PL allows multiple disjoint global minima (the quartic $\tfrac12(x^2-1.2)^2$ has two). Right — both functions satisfy $\|\nabla f\|^2 \geq 2\mu(f - f^\star)$ (log-log scale, dashed line is the PL boundary with $\mu = 0.5$).*


> **Theorem.** If $f$ is $L$-smooth and satisfies the PL inequality with $\mu$, GD with step $\eta = 1/L$ converges linearly:
> $$f(x_T) - f^\star \leq (1 - \mu / L)^T (f(x_0) - f^\star).$$
**Proof.** By smoothness,
$$
f(x_{t+1}) \leq f(x_t) + \nabla f(x_t)^\top (x_{t+1} - x_t) + \tfrac{L}{2} \|x_{t+1} - x_t\|_2^2 = f(x_t) - \tfrac{1}{2 L} \|\nabla f(x_t)\|_2^2.
$$
By PL, $\|\nabla f(x_t)\|_2^2 \geq 2 \mu (f(x_t) - f^\star)$. Substituting:
$$
f(x_{t+1}) - f^\star \leq (1 - \mu / L) (f(x_t) - f^\star). \quad \blacksquare
$$
This is shockingly clean — the proof is two lines. PL is the **right** condition for fast convergence in non-convex settings; it bypasses the global-minimum-uniqueness story that strong convexity needs.

### PL in deep learning

For sufficiently wide neural networks (width polynomial in number of training points), the Neural Tangent Kernel (NTK) regime gives a **provable PL inequality** along the GD trajectory (Du, Lee, Li, Wang, Zhai 2019; Liu et al. 2022). This is one of the few theoretical explanations of why GD finds zero training loss on neural networks despite non-convexity.

The catch: the PL constant $\mu$ scales unfavorably with network width and depth, so the rate, while linear, may be slow. NTK theory describes the lazy regime where the network behaves like a kernel method; real deep learning operates in a richer feature-learning regime where these guarantees are weaker.

---

## Loss landscape of neural networks

What do we actually know about the loss landscape of a deep network?

### Over-parameterization eliminates spurious local minima

For a deep neural network with $\geq n$ parameters where $n$ is the number of training examples, the global minimum value of the empirical loss is $0$ (assuming the architecture has enough capacity to interpolate). Under mild assumptions on the activation, **all local minima of the empirical loss are global** — the loss landscape has the **PL property** along the GD trajectory in suitable regimes.

This is the modern "**over-parameterization helps optimization**" theme: with more parameters than data points, training loss zero is achievable and gradient methods reach it. The 2017--2020 literature on neural-network loss landscapes provided the theoretical grounding for what practitioners had observed since AlexNet.

### The NTK regime

The **Neural Tangent Kernel** (Jacot, Gabriel, Hongler 2018) shows that a wide enough network behaves linearly in its parameters around initialization:
$$
f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0).
$$
GD on the squared loss in this regime converges at the eigenvalues of the NTK — the gradient flow becomes a linear ODE. The smallest eigenvalue is bounded away from zero with high probability (over the random init), giving the linear rate.

NTK is descriptive of "lazy training" — wide networks where weights barely change. Real-world networks operate beyond this regime, with feature learning and qualitative weight changes.

![NTK regime: wide networks stay near init, GD converges linearly](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig4.png)

*Figure: left — in the NTK / lazy regime a wide network's parameters $\theta$ stay inside a tiny neighborhood of the initialization $\theta_0$ (blue), while a narrow network's parameters move substantially (orange, feature learning). Right — under PL with NTK constant $\mu$, GD enjoys the linear rate $(1-\mu/L)^t$, in contrast to the sublinear plateau of generic non-convex landscapes.*


### Implicit bias of SGD

SGD does not just find any global min — it finds **flat** ones. Empirically, SGD-found minima generalize better than the same loss values found by full-batch GD. The conjectured mechanism: the SGD noise has variance proportional to the loss's local curvature, so SGD spends more time in flat regions where curvature is small.

A precise statement (Mandt et al. 2017): SGD's stationary distribution at constant step $\eta$ is approximately Gaussian centered at $x^\star$ with covariance $\eta C$ where $C$ depends on the inverse Hessian and the gradient covariance. Flatter minima (small Hessian) get visited more often.

![Flat vs sharp minima and the generalization gap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig5.png)

*Figure: left — a 1-D loss with one flat (wide) basin and one sharp (narrow) basin at almost identical training-loss values. Right — when the test distribution shifts the loss slightly, the flat minimum's value barely changes (gap $\approx 0.05$) while the sharp minimum's value blows up (gap $\approx 0.93$). This is the geometric intuition behind SGD's preference for flat minima and their better generalization.*


This is part of a larger story — **implicit bias** — where the choice of optimization algorithm shapes the function class effectively learned, even when the loss has many global minima. Linear models with logistic loss converge to the max-margin solution under GD; the same loss with Adam finds different solutions.

---

## Cubic regularization: a non-convex Newton

The non-convex analogue of damped Newton (article 07) is **cubic regularization**:
$$
x_{t+1} = \arg\min_x \, f(x_t) + \nabla f(x_t)^\top (x - x_t) + \tfrac{1}{2} (x - x_t)^\top \nabla^2 f(x_t) (x - x_t) + \tfrac{M}{6} \|x - x_t\|^3.
$$
> **Theorem (Nesterov & Polyak 2006).** Cubic regularization with $M$ Lipschitz-constant of $\nabla^2 f$ converges to a second-order stationary point (i.e., $\|\nabla f\| \leq \epsilon$ and $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\epsilon}$) in $O(1/\epsilon^{1.5})$ iterations.

Compare:

- Plain GD: first-order stationary in $O(1/\epsilon^2)$.
- Perturbed GD: second-order stationary in $\tilde O(1/\epsilon^2)$.
- Cubic regularization: second-order stationary in $O(1/\epsilon^{1.5})$ — **strictly faster**.

The catch is that cubic regularization needs the Hessian (or a cubic subproblem solver). It's been used in some ML applications but is dominated by SGD-family methods at scale.

---

## What is provably hard

To balance the optimism, here's what cannot be done:

- **Finding the global min of a general non-convex function in polynomial time** is NP-hard (decisional version of "is $f^\star \leq c$?" is NP-hard for general non-convex polynomials).
- **Finding $\epsilon$-local min of a smooth bounded $f$** requires $\Omega(1/\epsilon^{1.5})$ Hessian queries (Carmon, Duchi, Hinder, Sidford 2017) — matching the cubic rate.
- **Finding global min of training loss for deep networks** is provably hard in the worst case (Blum & Rivest 1992 for ReLU networks).

The successes of saddle escape, PL, and NTK all rely on **specific structure** of the problem that rules out worst-case behavior. The "ML practice works because of structure, not because optimization is generally easy" theme is now well-established.

---

## Summary

| Concept                       | What it gives you                                                       |
| ----------------------------- | ------------------------------------------------------------------------ |
| Strict saddle property        | Saddle points are unstable; perturbed GD escapes in polynomial time.    |
| PL condition                  | Linear convergence of GD without convexity; holds for wide NNs.         |
| NTK regime                    | Wide networks behave linearly; GD converges with linear rate.           |
| Implicit bias of SGD          | SGD prefers flat minima; explains generalization gap with full-batch GD. |
| Cubic regularization          | Optimal first-order rate to second-order stationary points.             |

This concludes the four-part series on continuous optimization theory. Article 12 closes the series with **discrete and global optimization** — branch-and-bound, integer programming, heuristics — for problems where smoothness is unavailable.

## References

- Jin, Ge, Netrapalli, Kakade, Jordan, *How to Escape Saddle Points Efficiently*, ICML 2017 — the perturbed GD paper.
- Liu, Zhu & Belkin, *Loss landscapes and optimization in over-parameterized non-linear systems and neural networks*, ACHA 59, 2022 — the modern PL+NN result.
- Du, Lee, Li, Wang & Zhai, *Gradient Descent Finds Global Minima of Deep Neural Networks*, ICML 2019 — NTK for deep networks.
- Karimi, Nutini & Schmidt, *Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Łojasiewicz Condition*, ECML 2016 — the PL convergence theory.
- Carmon, Duchi, Hinder & Sidford, *Lower Bounds for Finding Stationary Points*, MathProg 184, 2020 — the matching lower bound for non-convex first-order methods.
- Auffinger, Ben Arous & Černý, *Random Matrices and Complexity of Spin Glasses*, CPAM 66, 2013 — counting saddles in random landscapes.
