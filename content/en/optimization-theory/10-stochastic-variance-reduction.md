---
title: 'Optimization (10): Stochastic Optimization and Variance Reduction'
date: 2022-09-27 09:00:00
tags:
  - ML
  - Optimization
  - Stochastic Methods
categories: Algorithm
series: optimization-theory
series_order: 10
lang: en
mathjax: true
description: "Why does SGD work? We prove the O(1/sqrt(T)) convex rate and the O(1/(mu T)) strongly convex rate from the gradient noise budget. Then variance reduction: SVRG, SAGA, Katyusha — methods that get to the linear rate of full GD using stochastic samples, with complete analysis of why they work."
disableNunjucks: true
translationKey: "optim-10"
---

For the finite-sum problem

$$
\min_x f(x) := \frac{1}{n} \sum_{i=1}^n f_i(x),
$$

deterministic gradient descent costs $O(n)$ per step but converges in $O(\kappa \log(1/\epsilon))$ steps. **Stochastic gradient descent** (SGD) costs $O(1)$ per step but converges in $O(1/\epsilon^2)$ for convex problems and $O(\kappa^2 \log(1/\epsilon))$ for strongly convex ones. Which is faster depends on $n$, $\kappa$, and $\epsilon$.

A modern class of methods — **variance-reduced** SGD — achieves the deterministic rate $O((n + \kappa) \log(1/\epsilon))$ using stochastic samples. They are the missing piece that makes stochastic methods strictly better than full GD on finite-sum problems.

This article:

1. Proves the basic SGD rates from a noise budget.
2. Explains the role of mini-batching and learning rate decay.
3. Derives SVRG and proves its linear convergence on strongly convex objectives.
4. Mentions SAGA and Katyusha and the lower bound that motivated their development.

## What you will learn

1. The two SGD regimes: convex ($O(1/\sqrt{T})$ rate) and strongly convex ($O(1/T)$ rate).
2. The variance bound for SGD, why $\eta = 1/L$ is too aggressive, and where the $\eta_t = 1/(\mu t)$ schedule comes from.
3. The SVRG algorithm and its linear convergence.
4. SAGA, Katyusha, and the lower-bound result $\Omega((n + \sqrt{n \kappa}) \log(1/\epsilon))$.
5. Practical considerations: mini-batches, momentum, and where SGD vs variance-reduced methods matter.

## Prerequisites

Articles 02 (smoothness, strong convexity), 03 (gradient descent and SGD).

---

## 1. The SGD framework

At each iteration $t$, SGD samples an index $i_t$ uniformly from $\{1, \ldots, n\}$ and updates

$$
x_{t+1} = x_t - \eta_t \nabla f_{i_t}(x_t).
$$

Two facts about $\nabla f_{i_t}(x_t)$:

- **Unbiased**: $\mathbb{E}[\nabla f_{i_t}(x_t) \mid x_t] = \nabla f(x_t)$.
- **Bounded variance**: typically we assume

  $$
  \mathbb{E}\big[ \|\nabla f_{i_t}(x_t) - \nabla f(x_t)\|_2^2 \mid x_t \big] \leq \sigma^2,
  $$

  where $\sigma^2$ is the **gradient variance** budget.

These two assumptions (unbiased + bounded variance) are the SGD axioms. The strength of the resulting bounds depends on what additional structure $f$ has.

![SGD vs Full GD trajectories on an ill-conditioned 2D quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig1.png)
*Full GD follows a smooth, deterministic descent path; SGD takes a noisy zigzag in expectation around the same direction. The variance of each SGD step is what the noise budget $\sigma^2$ controls.*

---

## 2. Convex rate: $O(1/\sqrt{T})$

> **Theorem.** Suppose $f$ is convex and the variance bound holds. With constant step $\eta = R / (\sigma \sqrt{T})$ and starting from $x_0$ with $\|x_0 - x^\star\|_2 \leq R$, after $T$ iterations:
> $$
> \mathbb{E}[f(\bar x_T) - f^\star] \leq \frac{R \sigma}{\sqrt{T}},
> $$
> where $\bar x_T = \frac{1}{T} \sum_{t=0}^{T-1} x_t$ is the running average.

**Proof.** Take expectation conditional on $x_t$:

$$
\mathbb{E}\|x_{t+1} - x^\star\|_2^2 = \|x_t - x^\star\|_2^2 - 2 \eta \langle \nabla f(x_t), x_t - x^\star \rangle + \eta^2 \mathbb{E}\|\nabla f_{i_t}(x_t)\|_2^2.
$$

By convexity, $\langle \nabla f(x_t), x_t - x^\star \rangle \geq f(x_t) - f^\star$. By the variance bound,

$$
\mathbb{E}\|\nabla f_{i_t}(x_t)\|_2^2 = \|\nabla f(x_t)\|_2^2 + \mathbb{E}\|\nabla f_{i_t}(x_t) - \nabla f(x_t)\|_2^2 \leq \|\nabla f(x_t)\|_2^2 + \sigma^2.
$$

For now, assume $\|\nabla f\|_2 \leq G$ (Lipschitz $f$); we get $\mathbb{E}\|\nabla f_{i_t}\|_2^2 \leq G^2 + \sigma^2$. Then

$$
\mathbb{E}\|x_{t+1} - x^\star\|_2^2 \leq \|x_t - x^\star\|_2^2 - 2 \eta (f(x_t) - f^\star) + \eta^2 (G^2 + \sigma^2).
$$

Rearranging and summing $t = 0, \ldots, T-1$ (using telescoping):

$$
\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f^\star] \leq \frac{R^2}{2 \eta} + \frac{T \eta (G^2 + \sigma^2)}{2}.
$$

Dividing by $T$ and applying Jensen to $\bar x_T$:

$$
\mathbb{E}[f(\bar x_T) - f^\star] \leq \frac{R^2}{2 \eta T} + \frac{\eta (G^2 + \sigma^2)}{2}.
$$

Optimizing $\eta = R / \sqrt{T (G^2 + \sigma^2)}$ gives $\mathbb{E}[f(\bar x_T) - f^\star] \leq R \sqrt{(G^2 + \sigma^2) / T}$. Replacing $G^2 + \sigma^2$ with $\sigma^2$ (or absorbing $G$ into the noise) gives the form above. $\blacksquare$

The $O(1/\sqrt{T})$ rate is **the** classical SGD rate. Notice it depends on $\sigma^2$, not on the smoothness or condition number of $f$. SGD with proper step sizes is robust to noise but slow.

---

## 3. Strongly convex rate: $O(1/T)$

> **Theorem.** Suppose $f$ is $\mu$-strongly convex and the variance bound holds with $\sigma^2$. With step $\eta_t = 2 / (\mu (t + 1))$, after $T$ iterations,
> $$
> \mathbb{E}[\|x_T - x^\star\|_2^2] \leq \frac{4 \sigma^2}{\mu^2 T}.
> $$

**Proof sketch.** Define $a_t = \mathbb{E}[\|x_t - x^\star\|_2^2]$. The strong-convexity inequality $\langle \nabla f(x_t), x_t - x^\star \rangle \geq \mu \|x_t - x^\star\|_2^2$, combined with variance, gives the recursion

$$
a_{t+1} \leq (1 - 2 \eta_t \mu) a_t + \eta_t^2 \sigma^2 + \eta_t^2 L^2 a_t,
$$

where the last term comes from bounding $\|\nabla f(x_t)\|_2 \leq L \|x_t - x^\star\|_2$ for $L$-smooth strongly convex $f$. With $\eta_t = 2/(\mu(t+1))$ small enough that the $L^2$ term is negligible, induction gives $a_t = O(1/(\mu^2 t))$.

The optimal step decays as $1/t$ — this is **Robbins--Monro**'s classical 1951 schedule and is the basis of all modern adaptive step size schemes for SGD.

### 3.1 Why constant step size doesn't work for SGD on strongly convex $f$

If $\eta_t = \eta$ constant, the recursion has a fixed point at $a^\star = \eta \sigma^2 / (2 \mu)$. The iterates do not converge to $x^\star$; they converge to a noise ball of radius $O(\sqrt{\eta \sigma^2 / \mu})$. To shrink the ball to $\epsilon$ requires $\eta = O(\epsilon \mu / \sigma^2)$, then $T = O(\sigma^2 / (\epsilon \mu^2))$ — the same $1/\epsilon$ dependence as decreasing-step SGD, but you need to manually re-pick $\eta$ for each target accuracy.

In deep learning we don't actually want to converge to $x^\star$ — generalization error stops decreasing well before optimization gap, so a constant step size is fine. This is one of the structural differences between classical theory and DL practice.

---

## 4. Mini-batching: variance scales with batch size

If the batch size is $B$ and we average $B$ stochastic gradients per step,

$$
g_t = \frac{1}{B} \sum_{j=1}^B \nabla f_{i_{t,j}}(x_t),
$$

the variance becomes $\sigma^2 / B$ (assuming independent samples). So mini-batching reduces the noise budget linearly.

The convex rate becomes $O(\sigma / \sqrt{TB})$, which is $\sqrt{B}$ better than SGD with $B = 1$. But each step costs $B$ times more in gradient evaluations. Total gradient evaluations to reach $\epsilon$:

$$
\text{grads} = TB = O(\sigma^2 B / \epsilon^2).
$$

This grows linearly with $B$ — so mini-batching does not save on total compute! It only helps because bigger batches parallelize better on GPUs.

The **linear scaling rule** (Goyal et al., 2017) — batch size $\times k$, learning rate $\times k$ — comes from this analysis: the noise term $\eta^2 \sigma^2 / B$ stays constant if $\eta \propto B$, so larger batches let us take larger steps. This works only up to a "critical batch size" beyond which the noise is no longer the bottleneck (McCandlish et al., 2018).

![Mini-batch variance and the critical batch size](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig4.png)
*Left: gradient variance falls as $\sigma^2/B$ on a log-log plot. Right: the linear scaling rule lets effective step size grow with $B$ — but only up to a critical batch $B^\star$, beyond which speedup saturates because the gradient signal, not noise, becomes the bottleneck.*

---

## 5. Variance reduction: SVRG

SGD's $\sigma^2$ noise budget is unavoidable as long as we use a single $\nabla f_{i_t}$ as the gradient estimate. **Variance reduction** uses additional control variates — extra computation that reduces the variance to zero in the limit.

### 5.1 The SVRG algorithm

(Stochastic Variance-Reduced Gradient, Johnson & Zhang, 2013)

```text
SVRG (with epoch length m, learning rate η):
Initialize w̃_0
for s = 0, 1, 2, ...:                        # outer epochs
    g̃_s = ∇f(w̃_s) = (1/n) Σ ∇f_i(w̃_s)        # full gradient at snapshot
    x_0 = w̃_s
    for t = 0, ..., m-1:                     # inner steps
        Sample i_t ∈ {1, ..., n} uniformly
        g_t = ∇f_{i_t}(x_t) - ∇f_{i_t}(w̃_s) + g̃_s
        x_{t+1} = x_t - η g_t
    w̃_{s+1} = x_m  (or random x_t in the epoch)
```

The key is the **gradient estimator**

$$
g_t = \nabla f_{i_t}(x_t) - \nabla f_{i_t}(\tilde w_s) + \tilde g_s.
$$

Properties:

- **Unbiased**: $\mathbb{E}[g_t \mid x_t] = \nabla f(x_t) - \nabla f(\tilde w_s) + \nabla f(\tilde w_s) = \nabla f(x_t)$.
- **Variance vanishes near $x^\star$**: as $\tilde w_s, x_t \to x^\star$, $\nabla f_{i_t}(x_t) - \nabla f_{i_t}(\tilde w_s) \to 0$, so the noise vanishes.

This is what gives the linear convergence rate.

![SGD vs SVRG gradient samples around a fixed point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig3.png)
*Each light arrow is one stochastic gradient sample; the bold blue arrow is the true $\nabla f(x)$. SGD samples (orange) scatter widely around the mean; SVRG samples (green) cluster tightly because the control variate $-\nabla f_{i_t}(\tilde w_s) + \tilde g_s$ cancels most of the variance.*

### 5.2 SVRG convergence

> **Theorem (Johnson--Zhang 2013).** Suppose each $f_i$ is $L$-smooth and $f$ is $\mu$-strongly convex. With $\eta = \frac{1}{10 L}$ and $m$ chosen large enough (specifically $m \geq 100 L / \mu$), SVRG converges geometrically:
> $$
> \mathbb{E}[f(\tilde w_{s+1}) - f^\star] \leq 0.5 \cdot \mathbb{E}[f(\tilde w_s) - f^\star].
> $$

**Proof sketch.** The variance of $g_t$ satisfies

$$
\mathbb{E}\|g_t - \nabla f(x_t)\|_2^2 \leq L (f(x_t) - f^\star) + L (f(\tilde w_s) - f^\star).
$$

This is the **co-coercivity** lemma. Plugging it into the SGD analysis (as in section 2 but with this bound on $\sigma^2$) and tracking carefully through one SVRG epoch gives a contraction in $f(\tilde w_s) - f^\star$.

### 5.3 Total cost

Each SVRG epoch costs $n + m$ gradient evaluations. Number of epochs to reach $\epsilon$: $\log(1/\epsilon)$. Total: $O((n + L/\mu) \log(1/\epsilon)) = O((n + \kappa) \log(1/\epsilon))$.

Compare to:

- **Full GD**: $O(n \kappa \log(1/\epsilon))$ — the $n$ multiplies $\kappa$.
- **SGD**: $O(\kappa^2 / \epsilon)$ — can be much worse for small $\epsilon$.
- **SVRG**: $O((n + \kappa) \log(1/\epsilon))$ — the $n$ adds.

For $n \approx \kappa$ (typical regularized ML), SVRG is $\sim \kappa \times$ faster than full GD and $\sim \kappa^2 / (\kappa \log(1/\epsilon))$ faster than SGD for small $\epsilon$.

![Convergence rates: SGD, Full GD, SVRG, Katyusha](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig2.png)
*Suboptimality vs total gradient evaluations on a log-log axis. SGD's $1/\sqrt{T}$ rate is the slowest curve; Full GD is geometric but the per-step cost is $n$. SVRG and Katyusha are linear in the number of epochs, eventually beating both.*

---

## 6. SAGA, Katyusha, and the lower bound

**SAGA** (Defazio, Bach & Lacoste-Julien, 2014) is similar to SVRG but maintains a table of the most recent $\nabla f_i$ for each $i$, updating one entry per step. It avoids the snapshot cost but requires $O(nd)$ extra memory. Same $O((n + \kappa) \log(1/\epsilon))$ rate.

**Katyusha** (Allen-Zhu, 2017) combines variance reduction with Nesterov's acceleration, achieving the rate $O((n + \sqrt{n \kappa}) \log(1/\epsilon))$ — better than SVRG when $\kappa \gg n$.

> **Theorem (lower bound, Woodworth & Srebro 2016).** Any randomized first-order finite-sum algorithm requires $\Omega((n + \sqrt{n \kappa}) \log(1/\epsilon))$ gradient evaluations.

So Katyusha is **optimal** for the strongly convex finite-sum setting.

![Total gradient evaluations needed to reach high precision](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig5.png)
*With $n=10^4$ and $\kappa=10^3$, Full GD needs $\sim 10^{11}$ gradients to reach $\epsilon=10^{-4}$; SGD's $O(\kappa^2/\epsilon)$ scaling requires $\sim 10^{10}$. SVRG drops it to $\sim 10^{5}$, and Katyusha shaves another factor of $\sqrt{n/\kappa}$.*

---

## 7. Practical takeaways

| Problem regime                              | Method of choice                        |
| ------------------------------------------- | --------------------------------------- |
| Large $n$, low precision (DL)               | SGD + momentum + cosine schedule        |
| Medium $n$, high precision (classical ML)   | SVRG, SAGA, or quasi-Newton             |
| Strongly convex, ill-conditioned, finite-sum | Katyusha or accelerated SVRG           |
| Convex but not strongly convex              | Polyak averaging on SGD; FISTA-SGD      |
| Online (streaming) data                     | SGD only; variance reduction needs $n$ finite |

In deep learning, vanilla SGD + momentum + a learning-rate schedule still beats variance-reduced methods. The reasons are not fully understood — guesses include: (a) stochasticity helps generalization, (b) the noise scale shrinks naturally as the loss decreases, (c) the implicit bias of SGD steers toward flat minima.

---

## 7. Summary

Stochastic optimization trades per-step cost for noise. The classical SGD rates ($O(1/\sqrt{T})$ convex, $O(1/T)$ strongly convex) come directly from a noise-budget argument. Variance reduction extends SGD's per-step efficiency to the deterministic-rate regime, with Katyusha hitting the matching lower bound.

Article 11 is the next frontier: **non-convex** problems where global convergence is impossible but local guarantees (escape from saddle points, convergence to flat minima) still exist.

## Further reading

- Bottou, Curtis & Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review 60, 2018 — comprehensive survey of SGD and its variants.
- Johnson & Zhang, *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*, NeurIPS 2013 — the SVRG paper.
- Defazio, Bach & Lacoste-Julien, *SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives*, NeurIPS 2014 — the SAGA paper.
- Allen-Zhu, *Katyusha: The First Direct Acceleration of Stochastic Gradient Methods*, JMLR 18, 2017 — accelerated variance reduction.
- Woodworth & Srebro, *Tight Complexity Bounds for Optimizing Composite Objectives*, NeurIPS 2016 — the matching lower bound.
