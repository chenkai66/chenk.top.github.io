---
title: "ML Math Derivations (4): Convex Optimization Theory"
date: 2024-03-04 09:00:00
tags:
  - Machine Learning
  - Convex Optimization
  - Gradient Descent
  - KKT Conditions
  - ADMM
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 4
  total: 20
lang: en
mathjax: true
description: "Nearly every ML algorithm is an optimization problem. This article derives convex sets, convex functions, gradient descent, Newton's method, KKT conditions, and ADMM -- the optimization toolkit for machine learning."
disableNunjucks: true
---

## What This Article Covers

In 1947, George Dantzig proposed the simplex method for linear programming, and a working theory of optimization was born. Eight decades later, optimization has become the engine of machine learning: every model you train, from a one-line linear regression to a 70B-parameter language model, is the answer to *some* optimization problem.

Among all such problems, **convex optimization holds a privileged place**. The defining property is so strong it almost feels like cheating: every local minimum is automatically a global minimum, and a handful of well-understood algorithms come with airtight convergence guarantees. The whole reason we treat "convex" as a green flag and "non-convex" as a yellow one comes down to this single fact.

This article builds the convex toolkit from the ground up. We start with the geometry (convex sets, convex functions, Jensen's inequality), develop the calculus (subgradients for the non-smooth case), and then derive the algorithms that use that geometry: gradient descent, Nesterov acceleration, Adam, Newton's method, BFGS, the KKT conditions for constrained problems, and ADMM for problems with separable structure.

**What you will learn**

1. **Convex sets and convex functions** — the geometric definitions, first- and second-order characterizations, Jensen's inequality
2. **Subgradients** — extending calculus to non-smooth objectives like the L1 norm
3. **First-order methods** — gradient descent, momentum, Nesterov, Adam, with the convergence rates that explain why each one exists
4. **Second-order methods** — Newton's method and BFGS, and the cost-vs-convergence tradeoff
5. **Constrained optimization** — the Lagrangian, weak and strong duality, the KKT conditions
6. **ADMM** — splitting a hard joint problem into two easy alternating subproblems

**Prerequisites:** multivariable calculus (gradients, Hessians, Taylor expansion) and linear algebra (eigenvalues, positive semidefinite matrices). The earlier parts of this series cover both.

---

## 1. Convex Sets and Convex Functions

### 1.1 Convex Sets

**Definition 1 (Convex Set).** A set $C \subseteq \mathbb{R}^n$ is **convex** if for any two points $x, y \in C$ and any $\lambda \in [0, 1]$,

$$
\lambda x + (1 - \lambda) y \in C. \tag{1}
$$

In words: the line segment between any two points of $C$ stays inside $C$. No dents, no holes, no missing chunks. The picture below makes the definition concrete: a chord across a convex set never escapes; a chord across a non-convex set can poke right through a gap.

![Convex vs non-convex sets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig1_convex_sets.png)

A short list of convex sets that you will meet again and again:

- **Hyperplanes** $\{x : a^T x = b\}$ and **halfspaces** $\{x : a^T x \leq b\}$ — the building blocks of every linear-constraint problem.
- **Norm balls** $\{x : \|x\| \leq r\}$ for any norm. The L2 ball is the standard one; the L1 ball gives the famous diamond shape behind sparsity.
- **Ellipsoids** $\{x : (x - x_0)^T A (x - x_0) \leq 1\}$ for $A \succ 0$.
- **Polyhedra** — finite intersections of halfspaces, the natural feasible regions of linear programming.

**Theorem 1 (Intersection).** Any intersection $\bigcap_i C_i$ of convex sets is convex.

*Proof.* If $x, y$ lie in every $C_i$, then $\lambda x + (1-\lambda) y$ lies in every $C_i$ by convexity, hence in their intersection. $\square$

This single fact is the workhorse of practical convex modeling: complicated feasible regions are almost always built by intersecting many simple convex constraints.

### 1.2 Convex Functions

**Definition 2 (Convex Function).** $f : C \to \mathbb{R}$ is **convex** if its domain $C$ is convex and for all $x, y \in C$ and $\lambda \in [0, 1]$,

$$
f(\lambda x + (1-\lambda)y) \;\leq\; \lambda f(x) + (1-\lambda) f(y). \tag{2}
$$

Geometrically, the chord between any two points on the graph of $f$ lies on or above the graph itself. The bowl-shaped surface on the left below is convex; the egg-carton on the right is not. The difference is consequential: the convex bowl has exactly one minimum that any descent method will find, while the egg-carton has many local minima where descent methods get permanently stuck.

![Convex vs non-convex functions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig2_convex_functions.png)

Two strengthenings of convexity matter often enough to name:

- **Strictly convex:** the inequality (2) is strict whenever $x \neq y$ and $\lambda \in (0, 1)$. This rules out flat plateaus and guarantees a *unique* minimizer (when one exists).
- **$\mu$-strongly convex** ($\mu > 0$): $f(x) - \tfrac{\mu}{2}\|x\|^2$ is convex. Equivalently, $f$ has at least quadratic curvature in every direction. Strong convexity is what powers the *linear* convergence rates we will derive shortly.

### 1.3 First-Order and Second-Order Characterizations

Definition 2 is geometric and clean, but it is hard to *check*. The next two theorems give equivalent conditions in terms of derivatives, which is what we actually use in practice.

**Theorem 2 (First-Order Condition).** If $f$ is differentiable, then $f$ is convex if and only if for all $x, y$,

$$
f(y) \;\geq\; f(x) + \nabla f(x)^T (y - x). \tag{3}
$$

In other words, every tangent hyperplane is a *global* lower bound on $f$. This is the deep structural reason gradient descent works on convex problems: the linear approximation $f(x) + \nabla f(x)^T (y-x)$ never overestimates $f$, so following its negative gradient really does decrease $f$ — at least for a small enough step.

**Theorem 3 (Second-Order Condition).** If $f$ is twice differentiable, then $f$ is convex if and only if its Hessian is positive semidefinite everywhere:

$$
\nabla^2 f(x) \;\succeq\; 0 \qquad \text{for all } x. \tag{4}
$$

If $\nabla^2 f \succ 0$ everywhere, $f$ is strictly convex. If $\nabla^2 f \succeq \mu I$ everywhere, $f$ is $\mu$-strongly convex. The Hessian test is by far the most common way to certify convexity in practice — for instance, the loss in linear regression is $\nabla^2 = X^T X$, which is automatically PSD.

### 1.4 Jensen's Inequality

The convexity inequality (2) is a statement about a two-point average. Pushing it to general probability distributions gives what is arguably the single most-used inequality in machine learning.

**Theorem 4 (Jensen's Inequality).** If $f$ is convex and $X$ is an integrable random variable,

$$
f(\mathbb{E}[X]) \;\leq\; \mathbb{E}[f(X)]. \tag{5}
$$

If $f$ is strictly convex and $X$ is non-degenerate, the inequality is strict.

*Proof sketch (discrete case).* Iterate Definition 2 by induction: if $X$ takes values $x_1, \ldots, x_n$ with probabilities $p_1, \ldots, p_n$, then $f(\sum p_i x_i) \leq \sum p_i f(x_i)$ by repeated application of the chord inequality. The continuous case follows by approximation. $\square$

Jensen powers an enormous amount of downstream theory. Two appearances you will see again in this series:

- **Non-negativity of KL divergence.** Apply Jensen to $-\log$ (which is convex): $D_{KL}(p \| q) = \mathbb{E}_p[-\log(q/p)] \geq -\log \mathbb{E}_p[q/p] = 0$.
- **Evidence Lower Bound (ELBO)** in variational inference: the log-marginal $\log p(x)$ is bounded below by $\mathbb{E}_q[\log p(x, z) - \log q(z)]$ via Jensen on $\log$.

A short catalog of convex functions worth memorizing: affine functions $a^T x + b$ (both convex and concave), the exponential $e^{ax}$, all norms $\|x\|$, the negative entropy $x \log x$ on $x > 0$, the squared loss $\tfrac{1}{2}\|x\|^2$, and the log-sum-exp $\log \sum_i e^{x_i}$ that powers softmax.

---

## 2. Subgradients and Non-Smooth Optimization

Many of the most useful objectives in machine learning are *not* differentiable everywhere. The L1 norm $\|x\|_1$ used in Lasso has a kink at every coordinate axis; the hinge loss $\max(0, 1 - y \cdot s)$ used in SVMs has a kink at $s = 1/y$. To do calculus on these objects we generalize the gradient.

### 2.1 Subgradients

**Definition 3 (Subgradient).** A vector $g \in \mathbb{R}^n$ is a **subgradient** of $f$ at $x$ if for all $y$,

$$
f(y) \;\geq\; f(x) + g^T (y - x). \tag{6}
$$

The set of all such $g$ is the **subdifferential** $\partial f(x)$. When $f$ is differentiable at $x$, $\partial f(x) = \{\nabla f(x)\}$ — a single element. At non-differentiable points the subdifferential is typically a non-trivial set, encoding all the "valid" linear lower bounds that touch $f$ at $x$.

**Example (absolute value).** For $f(x) = |x|$,

$$
\partial f(x) = \begin{cases} \{-1\} & x < 0, \\ [-1, 1] & x = 0, \\ \{+1\} & x > 0. \end{cases} \tag{7}
$$

**Optimality.** $x^\star$ is a global minimizer of a convex $f$ if and only if $0 \in \partial f(x^\star)$. This is the non-smooth analogue of $\nabla f(x^\star) = 0$.

### 2.2 Subgradient Method

Replacing the gradient with any subgradient yields the **subgradient method**:

$$
x^{(k+1)} = x^{(k)} - \alpha_k g^{(k)}, \quad g^{(k)} \in \partial f(x^{(k)}). \tag{8}
$$

A subtle but important warning: a subgradient is *not* in general a descent direction — the function value may go up on a single step. To ensure convergence the step size must shrink, e.g. $\alpha_k = 1/\sqrt{k}$. The price for working with non-smooth functions is a slower rate, $O(1/\sqrt{k})$ instead of the $O(1/k)$ rate of gradient descent on smooth problems.

For composite problems of the form $f(x) + h(x)$ where $f$ is smooth and $h$ is non-smooth (Lasso fits this exactly), the **proximal gradient method** recovers the smooth $O(1/k)$ rate by handling $h$ exactly through its proximal operator.

---

## 3. First-Order Optimization Algorithms

### 3.1 Gradient Descent

Gradient descent is the most fundamental algorithm in continuous optimization: at each step, move in the direction of steepest local decrease.

$$
x^{(k+1)} \;=\; x^{(k)} - \alpha \, \nabla f(x^{(k)}). \tag{9}
$$

Why does this work? Because of the first-order condition (3): on a convex function, the negative gradient really is a descent direction, and the linear approximation underestimates $f$, so we cannot "overshoot the bottom" by following gradients carefully. The next figure shows the consequence in two dimensions — a clean monotone path to the minimum on the convex side, and a path that gets pinned to a local minimum on the non-convex side.

![Gradient descent on convex vs non-convex landscapes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig3_gd_convex_vs_nonconvex.png)

**Theorem 5 (Convergence — convex + smooth).** If $f$ is convex with $L$-Lipschitz gradient ($\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$) and we take constant step size $\alpha = 1/L$,

$$
f(x^{(k)}) - f(x^\star) \;\leq\; \frac{L\, \|x^{(0)} - x^\star\|^2}{2k} \;=\; O(1/k). \tag{10}
$$

*Proof sketch.* The Lipschitz-gradient assumption gives the descent inequality $f(y) \leq f(x) + \nabla f(x)^T(y - x) + \tfrac{L}{2}\|y - x\|^2$. Plug in $y = x - \tfrac{1}{L}\nabla f(x)$ and combine with the convexity bound (3); telescoping over $k$ iterations yields (10). $\square$

**Theorem 6 (Convergence — strongly convex).** If in addition $f$ is $\mu$-strongly convex,

$$
f(x^{(k)}) - f(x^\star) \;\leq\; \left(1 - \frac{\mu}{L}\right)^k \big[f(x^{(0)}) - f(x^\star)\big]. \tag{11}
$$

This is **linear convergence**: the error shrinks by a fixed factor each step. The factor depends on the **condition number** $\kappa = L / \mu$. A well-conditioned problem ($\kappa$ near 1) converges in a handful of steps; a badly-conditioned one ($\kappa \sim 10^6$) needs millions. Improving conditioning — through preconditioning, normalization, or regularization — is one of the most effective practical tricks in optimization.

### 3.2 The Learning Rate is a Choice You Cannot Avoid

Gradient descent has one knob: the step size $\alpha$. For the same convex quadratic, three choices produce three completely different stories.

![Learning-rate sweep](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig7_learning_rate.png)

The arithmetic for the 1D quadratic $f(x) = \tfrac{L}{2} x^2$ makes the picture exact. The update is $x^{(k+1)} = (1 - \alpha L) x^{(k)}$, so:

- **$0 < \alpha < 1/L$**: contracting but slowly — many tiny steps.
- **$\alpha = 1/L$**: optimal one-step contraction; on a 1D quadratic this jumps straight to $0$.
- **$1/L < \alpha < 2/L$**: still convergent but oscillating across the minimum.
- **$\alpha > 2/L$**: $|1 - \alpha L| > 1$, so iterates *amplify* and diverge.

The same dichotomy holds in many dimensions, with $L$ replaced by the largest Hessian eigenvalue. This is the single most important practical fact about gradient descent: **the right learning rate is set by curvature, not by guessing**.

### 3.3 Momentum and Nesterov Acceleration

On long, narrow valleys gradient descent zig-zags across the walls instead of running down the floor. Momentum methods fix this by averaging past gradients, damping the oscillations and accumulating speed in consistent directions.

**Heavy ball (Polyak, 1964):**

$$
v^{(k+1)} = \beta v^{(k)} - \alpha \nabla f(x^{(k)}), \qquad x^{(k+1)} = x^{(k)} + v^{(k+1)}. \tag{12}
$$

**Nesterov accelerated gradient (NAG):** evaluate the gradient at a *lookahead* point that anticipates where momentum will carry you,

$$
x^{(k+1)} = y^{(k)} - \alpha \nabla f(y^{(k)}), \qquad y^{(k+1)} = x^{(k+1)} + \beta_k\big(x^{(k+1)} - x^{(k)}\big). \tag{13}
$$

**Theorem 7 (Nesterov acceleration).** For $L$-smooth convex $f$,

$$
f(x^{(k)}) - f(x^\star) \;=\; O(1/k^2). \tag{14}
$$

The bound (14) is **provably optimal**: no first-order method (one that only queries $\nabla f$) can do asymptotically better in the worst case. This is one of the genuine surprises of convex optimization — a tiny algorithmic twist (look ahead, then step) gets you from $O(1/k)$ to $O(1/k^2)$ for free.

### 3.4 Adaptive Learning Rates and Adam

Plain gradient descent uses one global step size for every coordinate. In practice, different coordinates have wildly different effective curvatures — for instance, in a text model, common-word embeddings receive frequent gradient updates while rare-word embeddings receive almost none. **Adaptive methods** give each coordinate its own learning rate, scaled by the recent magnitude of its gradient.

**Adam** combines two ideas: a momentum-style first moment, and an RMSProp-style adaptive second moment.

$$
\begin{aligned}
m^{(k+1)} &= \beta_1 m^{(k)} + (1-\beta_1)\, g^{(k)} \tag{15} \\
v^{(k+1)} &= \beta_2 v^{(k)} + (1-\beta_2)\, (g^{(k)})^{\odot 2} \tag{16} \\
x^{(k+1)} &= x^{(k)} \;-\; \frac{\alpha\, \hat m^{(k+1)}}{\sqrt{\hat v^{(k+1)}} + \varepsilon} \tag{17}
\end{aligned}
$$

with bias-corrected estimates $\hat m^{(k+1)} = m^{(k+1)} / (1 - \beta_1^{k+1})$ and $\hat v^{(k+1)} = v^{(k+1)} / (1 - \beta_2^{k+1})$. The standard hyperparameters $(\beta_1, \beta_2, \alpha, \varepsilon) = (0.9, 0.999, 10^{-3}, 10^{-8})$ work astonishingly well across deep learning workloads, which is part of why Adam became the default optimizer for transformers.

### 3.5 Stochastic Gradient Descent

In large-scale ML, computing the full gradient $\nabla f(x) = \tfrac{1}{N}\sum_{i=1}^N \nabla f_i(x)$ over millions of examples is prohibitive. **Stochastic gradient descent (SGD)** replaces the full sum by a random sample (a mini-batch), trading a noisy step for a much cheaper one.

The picture below shows the qualitative difference: the deterministic GD path is smooth and direct; the SGD path is a noisy walk that stays close to the GD trajectory on average but jiggles around it constantly.

![SGD vs GD on the same landscape](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig6_sgd_path.png)

For convex objectives, SGD converges at $O(1/\sqrt{k})$ with constant step size and $O(1/k)$ with diminishing step size — slower per-step than GD, but each step is $N$ times cheaper, so wall-clock convergence is dramatically better. There is also a soft *benefit* in non-convex settings: noise helps SGD escape shallow saddle points and narrow local minima, which is part of why deep networks trained with SGD generalize better than the same networks trained to a sharper minimum with full-batch methods.

---

## 4. Second-Order Methods

### 4.1 Newton's Method

If a first-order method models $f$ locally as a tangent plane, Newton's method models it as a tangent paraboloid and jumps straight to that paraboloid's minimum. The second-order Taylor expansion at $x^{(k)}$ is

$$
f(x) \;\approx\; f(x^{(k)}) + \nabla f^T (x - x^{(k)}) + \tfrac{1}{2} (x - x^{(k)})^T \nabla^2 f \,(x - x^{(k)}).
$$

Setting the gradient of this quadratic to zero gives the **Newton step**:

$$
\Delta x = -\big[\nabla^2 f(x^{(k)})\big]^{-1} \nabla f(x^{(k)}). \tag{18}
$$

**Theorem 8 (Quadratic convergence).** Near a strongly convex minimum with Lipschitz Hessian,

$$
\|x^{(k+1)} - x^\star\| \;\leq\; C\, \|x^{(k)} - x^\star\|^2. \tag{19}
$$

Quadratic convergence is qualitatively different from anything we have seen so far: roughly, *the number of correct digits doubles each iteration*. Once you are close, Newton finishes in five or six steps to machine precision.

The catch is in three places. (i) Per-iteration cost is $O(n^3)$ to factor the Hessian — prohibitive for $n > 10^4$. (ii) Storage is $O(n^2)$ for the Hessian itself. (iii) If $\nabla^2 f$ is not PD (which happens away from convex regions), the Newton step may not even decrease $f$, so it has to be regularized into something like a trust-region step.

### 4.2 Quasi-Newton: BFGS and L-BFGS

Quasi-Newton methods aim for "almost" Newton-like behavior at first-order cost. Instead of computing the Hessian, they *approximate* its inverse using only the gradients we have already seen.

The BFGS update for the inverse-Hessian approximation $H^{(k)}$ is

$$
H^{(k+1)} = (I - \rho_k s_k y_k^T)\, H^{(k)} \,(I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T, \tag{20}
$$

where $s_k = x^{(k+1)} - x^{(k)}$, $y_k = \nabla f^{(k+1)} - \nabla f^{(k)}$, and $\rho_k = 1/(y_k^T s_k)$. The update is engineered so that $H^{(k+1)}$ satisfies the *secant equation* $H^{(k+1)} y_k = s_k$ (the inverse-Hessian analogue of how the true inverse Hessian relates gradient differences to step differences) and remains symmetric positive definite.

For large $n$, even storing $H$ is impractical, so **L-BFGS** keeps only the last $m \approx 5$–$20$ pairs $(s_i, y_i)$ and applies $H g$ implicitly through a recursive two-loop formula. This is the standard workhorse for medium-to-large smooth convex problems where you want better-than-GD convergence without the Hessian cost.

| Method      | Cost per iter | Convergence  | Best for                   |
|-------------|---------------|--------------|----------------------------|
| GD          | $O(n)$        | $O(1/k)$     | Large-scale smooth         |
| Nesterov    | $O(n)$        | $O(1/k^2)$   | Smooth convex              |
| Newton      | $O(n^3)$      | Quadratic    | $n < 10^4$, high accuracy  |
| BFGS        | $O(n^2)$      | Superlinear  | Medium scale               |
| L-BFGS      | $O(n)$        | Superlinear  | Large smooth convex        |
| SGD/Adam    | $O(n)$ /step  | $O(1/\sqrt{k})$ | Deep learning, huge $N$ |

---

## 5. Constrained Optimization and Duality

### 5.1 Problem Setup

The general constrained convex problem has the form

$$
\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0 \;\; (i = 1, \ldots, m), \quad h_j(x) = 0 \;\; (j = 1, \ldots, p). \tag{21}
$$

We assume $f$ and $g_i$ are convex, and $h_j$ are affine.

### 5.2 The Lagrangian and the Dual Problem

The Lagrangian rolls the constraints into the objective with multipliers $\lambda_i \geq 0$ (for inequalities) and $\nu_j \in \mathbb{R}$ (for equalities):

$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x). \tag{22}
$$

The **dual function** $d(\lambda, \nu) = \min_x \mathcal{L}(x, \lambda, \nu)$ is concave in $(\lambda, \nu)$ regardless of whether the original problem is convex (it is the pointwise minimum of affine functions of $(\lambda, \nu)$).

**Theorem 9 (Weak duality).** For any feasible $\lambda \geq 0$ and any $\nu$,

$$
d(\lambda, \nu) \;\leq\; f(x^\star). \tag{23}
$$

*Proof.* For any feasible $x$, $\sum_i \lambda_i g_i(x) \leq 0$ (because $\lambda_i \geq 0$ and $g_i(x) \leq 0$) and $\sum_j \nu_j h_j(x) = 0$, so $\mathcal{L}(x, \lambda, \nu) \leq f(x)$. Taking the infimum over $x$ gives $d(\lambda, \nu) \leq f(x)$, and since this holds for every feasible $x$, $d(\lambda, \nu) \leq f(x^\star)$. $\square$

The dual problem $\max_{\lambda \geq 0, \nu}\, d(\lambda, \nu)$ has optimal value $d^\star \leq f^\star$. The gap $f^\star - d^\star \geq 0$ is the **duality gap**. When $f^\star = d^\star$ we say **strong duality** holds.

**Slater's condition.** For convex problems, strong duality holds whenever there exists a strictly feasible point — some $\tilde x$ with $g_i(\tilde x) < 0$ for all inequality constraints. (A few mild technical assumptions cover the equality constraints.) Strong duality is what makes the dual problem useful: solving it gives the same optimal value as the primal, often with a simpler structure.

The geometric picture is in the figure below. The image set $\{(g(x), f(x)) : x \in \mathbb{R}^n\}$ tells us everything: $f^\star$ is the smallest $f$-value attained at $g \leq 0$, and $d^\star$ is the highest $f$-axis intercept of any non-vertical hyperplane that supports the image set from below. When the image is convex these two values coincide; when it is not, a "notch" can open above the supporting hyperplane and create a duality gap.

![Primal vs dual: geometric view](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig5_duality_geometry.png)

### 5.3 KKT Conditions

When strong duality holds, the optimal primal and dual variables jointly satisfy a small set of equations and inequalities — the **Karush-Kuhn-Tucker (KKT) conditions**.

**Theorem 10 (KKT).** Suppose strong duality holds and all functions are differentiable. Then $(x^\star, \lambda^\star, \nu^\star)$ is primal-dual optimal if and only if

1. **Primal feasibility:** $g_i(x^\star) \leq 0$ and $h_j(x^\star) = 0$.
2. **Dual feasibility:** $\lambda_i^\star \geq 0$.
3. **Complementary slackness:** $\lambda_i^\star\, g_i(x^\star) = 0$ for all $i$.
4. **Stationarity:** $\nabla f(x^\star) + \sum_i \lambda_i^\star \nabla g_i(x^\star) + \sum_j \nu_j^\star \nabla h_j(x^\star) = 0$.

For convex problems satisfying Slater's condition, KKT is both necessary *and* sufficient — solving these equations is equivalent to solving the original optimization problem.

The geometry of the stationarity condition is striking. At the optimum, $-\nabla f(x^\star)$ (the direction in which the objective wants to move) must lie in the cone of active-constraint normals. If it didn't, you could find a feasible descent direction and the point wouldn't be optimal. The figure makes this concrete on a 2D example: at the constrained optimum on a half-plane, the negative objective gradient and the constraint gradient point in exactly opposite directions, balanced by the multiplier.

![KKT conditions: gradients align at the optimum](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/04-Convex-Optimization-Theory/fig4_kkt_conditions.png)

**Complementary slackness** is the most intuitively powerful KKT clause: for each inequality, *either the constraint is active ($g_i(x^\star) = 0$) or its multiplier is zero*. Inactive constraints contribute nothing to the optimum — they could be dropped without changing the answer. This single observation gives the support-vector machine (SVM) its name: in the SVM dual, only training points exactly on the margin have nonzero multipliers, and those are the support vectors. Everything else is dead weight.

---

## 6. ADMM: Splitting Hard Problems Into Easy Ones

Some problems have a structure where a *joint* optimization over $(x, z)$ is hard, but optimizing $x$ alone (with $z$ fixed) and $z$ alone (with $x$ fixed) are both easy. **ADMM** — the Alternating Direction Method of Multipliers — exploits exactly this structure:

$$
\min_{x, z}\, f(x) + g(z) \quad \text{s.t.} \quad A x + B z = c. \tag{24}
$$

The augmented Lagrangian adds a quadratic penalty to the standard Lagrangian, $\mathcal{L}_\rho(x, z, u) = f(x) + g(z) + u^T(Ax + Bz - c) + \tfrac{\rho}{2}\|Ax + Bz - c\|^2$. ADMM minimizes alternately over $x$ and $z$, then takes a gradient *ascent* step on the dual variable $u$:

$$
\begin{aligned}
x^{(k+1)} &= \arg\min_x\, \Big[f(x) + \tfrac{\rho}{2}\|Ax + Bz^{(k)} - c + u^{(k)}\|^2\Big], \tag{25} \\
z^{(k+1)} &= \arg\min_z\, \Big[g(z) + \tfrac{\rho}{2}\|Ax^{(k+1)} + Bz - c + u^{(k)}\|^2\Big], \tag{26} \\
u^{(k+1)} &= u^{(k)} + Ax^{(k+1)} + Bz^{(k+1)} - c. \tag{27}
\end{aligned}
$$

ADMM is not the fastest algorithm in any single regime, but it is remarkably *robust* and naturally distributable, which is why it dominates large-scale structured problems.

**Application: Lasso.** The Lasso problem $\min \tfrac{1}{2}\|y - X\beta\|^2 + \lambda\|\beta\|_1$ becomes an ADMM problem by introducing a copy: $\min \tfrac{1}{2}\|y - X\beta\|^2 + \lambda\|z\|_1$ s.t. $\beta = z$. Then

- the $\beta$-update is a **ridge regression** with closed form,
- the $z$-update is the **soft-thresholding** operator, applied coordinate-wise:

$$
S_\kappa(a) = \mathrm{sign}(a) \cdot \max(|a| - \kappa, 0). \tag{28}
$$

Both subproblems are essentially free, and the iteration produces the exact Lasso solution. The same recipe works for nuclear-norm minimization, total-variation denoising, consensus optimization across many machines, and more.

---

## 7. Code: Optimization Algorithms

A minimal but working implementation of GD, Newton, BFGS, and ADMM-Lasso:

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve

np.random.seed(42)

# ---- A convex quadratic: f(x) = 0.5 x^T Q x - b^T x ----
n = 10
A = np.random.randn(n, n)
Q = A.T @ A + np.eye(n)        # PD
b = np.random.randn(n)
x_true = np.linalg.solve(Q, b)
f      = lambda x: 0.5 * x @ Q @ x - b @ x
grad_f = lambda x: Q @ x - b

# ---- Gradient Descent ----
x = np.zeros(n)
alpha = 1.0 / np.linalg.eigvalsh(Q).max()    # alpha = 1/L
for _ in range(200):
    x = x - alpha * grad_f(x)
print(f"GD:     ||x - x*|| = {np.linalg.norm(x - x_true):.2e}")

# ---- Newton (Hessian = Q is constant: one step is exact) ----
x = np.zeros(n)
for _ in range(5):
    x = x - np.linalg.solve(Q, grad_f(x))
print(f"Newton: ||x - x*|| = {np.linalg.norm(x - x_true):.2e}")

# ---- BFGS (with Armijo backtracking line search) ----
x, H, g = np.zeros(n), np.eye(n), grad_f(np.zeros(n))
for _ in range(200):
    d = -H @ g
    t = 1.0
    while f(x + t * d) > f(x) + 0.1 * t * (g @ d):
        t *= 0.5
    x_new = x + t * d
    g_new = grad_f(x_new)
    s, y = x_new - x, g_new - g
    rho = 1.0 / (y @ s) if y @ s > 0 else 0.0
    if rho > 0:
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
          + rho * np.outer(s, s)
    x, g = x_new, g_new
print(f"BFGS:   ||x - x*|| = {np.linalg.norm(x - x_true):.2e}")

# ---- ADMM for Lasso ----
ns, nf = 100, 50
X = np.random.randn(ns, nf)
beta_true = np.zeros(nf); beta_true[:10] = np.random.randn(10)
y = X @ beta_true + 0.1 * np.random.randn(ns)
lam, rho = 0.1, 1.0
L = cho_factor(X.T @ X + rho * np.eye(nf))
beta = z = u = np.zeros(nf)
for _ in range(500):
    beta = cho_solve(L, X.T @ y + rho * (z - u))            # ridge-style step
    z    = np.sign(beta + u) * np.maximum(np.abs(beta + u) - lam / rho, 0.0)  # soft-threshold
    u    = u + beta - z                                     # dual ascent
print(f"ADMM:   nonzeros = {int(np.sum(np.abs(z) > 1e-4))}/10, "
      f"err = {np.linalg.norm(z - beta_true):.4f}")
```

---

## 8. Q&A

**Q1: Why study convex optimization if deep learning is non-convex?**

For three reasons that compound. *First*, many of the workhorses of ML — linear and ridge regression, logistic regression, SVMs, most classical statistical estimators — really are convex. *Second*, near a local minimum, *any* sufficiently smooth loss looks quadratic, so the convex convergence rates apply locally. *Third*, modern optimizers like Adam, SGD with momentum, and AdaGrad were all designed and analyzed in the convex setting, then transplanted to deep learning; understanding their convex roots tells you when they are likely to misbehave on a non-convex loss.

**Q2: What does the condition number $\kappa$ actually mean?**

$\kappa = L / \mu$ measures how stretched the level sets are. $\kappa = 1$ means perfectly spherical contours, and gradient descent finishes in one step. $\kappa = 100$ means the bowl is 100× longer in one direction than another, and gradient descent zig-zags. $\kappa \sim 10^6$ means it bounces nearly perpendicular to the descent direction and crawls. Ridge regression's secret weapon is conditioning: adding $\lambda I$ to $X^T X$ raises the smallest eigenvalue from near zero to at least $\lambda$, replacing $\kappa = \lambda_{\max}/\lambda_{\min}$ with $\kappa = (\lambda_{\max}+\lambda)/(\lambda_{\min}+\lambda)$, which can be many orders of magnitude smaller.

**Q3: How do I choose an algorithm for a new problem?**

Follow the structure of the problem.

| Situation                        | Algorithm                             |
|----------------------------------|---------------------------------------|
| Small smooth ($n < 10^3$)        | BFGS or Newton                        |
| Medium smooth ($n < 10^5$)       | L-BFGS                                |
| Large-scale or deep learning     | SGD, Adam                             |
| Non-smooth + structured (e.g. L1)| Proximal gradient, ADMM               |
| Distributed / consensus          | ADMM                                  |
| Constrained convex               | Interior-point (CVXPY)                |

When in doubt, start with L-BFGS for smooth problems, ADMM for non-smooth structured problems, and Adam for anything involving a neural network.

---

## Exercises

**Exercise 1.** Show that $f(x_1, x_2) = x_1^2 - x_1 x_2 + x_2^2$ is strictly convex.

*Solution.* The Hessian is $\begin{pmatrix} 2 & -1 \\ -1 & 2\end{pmatrix}$ with eigenvalues $1$ and $3$. Both are positive, so the Hessian is positive definite and $f$ is strictly convex.

**Exercise 2.** Solve $\min x_1^2 + x_2^2$ subject to $x_1 + x_2 \geq 1$ via the KKT conditions.

*Solution.* Write the constraint as $g(x) = 1 - x_1 - x_2 \leq 0$. Stationarity: $2x_i - \lambda = 0$, so $x_1 = x_2 = \lambda/2$. If the constraint were inactive then $\lambda = 0$ and $x = 0$, which violates the constraint; so the constraint is active, $x_1 + x_2 = 1$, giving $x_1 = x_2 = 1/2$ and $\lambda = 1$.

**Exercise 3.** Derive GD's convergence rate for $f(x) = \tfrac{1}{2} x^T A x - b^T x$ with $A \succ 0$.

*Solution.* The error $e^{(k)} = x^{(k)} - x^\star$ satisfies $e^{(k+1)} = (I - \alpha A)\, e^{(k)}$. Convergence requires $\|I - \alpha A\| < 1$ in spectral norm, i.e. $|1 - \alpha \lambda_i| < 1$ for all eigenvalues $\lambda_i$ of $A$, giving $0 < \alpha < 2/\lambda_{\max}$. The optimal step is $\alpha^\star = 2/(\lambda_{\min} + \lambda_{\max})$, with rate $(\kappa - 1)/(\kappa + 1)$.

**Exercise 4.** Use Jensen's inequality to prove the AM-GM inequality $\tfrac{1}{n}\sum x_i \geq (\prod x_i)^{1/n}$ for $x_i > 0$.

*Solution.* The function $-\log$ is convex on $(0, \infty)$. By Jensen, $-\log\!\big(\tfrac{1}{n}\sum x_i\big) \leq \tfrac{1}{n}\sum (-\log x_i) = -\log(\prod x_i)^{1/n}$. Negating and exponentiating gives the AM-GM inequality.

---

## Summary

| Concept                    | Key Result                                                 | Why it matters                              |
|----------------------------|------------------------------------------------------------|---------------------------------------------|
| Convexity                  | Local min = global min                                     | Optimization becomes tractable              |
| First-order condition      | $f(y) \geq f(x) + \nabla f(x)^T(y - x)$                    | Gradient descent has a global lower bound   |
| GD convergence             | $O(1/k)$ smooth, $(1 - \mu/L)^k$ strongly convex            | Speed is set by the condition number        |
| Nesterov                   | $O(1/k^2)$ — provably optimal first-order                  | Acceleration for free                       |
| Newton                     | Quadratic convergence near $x^\star$                       | Few steps if you can afford the Hessian     |
| KKT                        | Necessary and sufficient (convex + Slater)                 | Characterizes constrained optima            |
| ADMM                       | Splits a hard problem into two easy alternating subproblems | Distributed and structured optimization     |

Convex optimization is the core that the rest of this series will build on: linear regression, logistic regression, SVMs, and many EM-style algorithms all reduce to convex problems for which the algorithms above are the right answer.

---

## References

1. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
2. Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
3. Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization.* Springer.
4. Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
5. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*.
6. Bubeck, S. (2015). Convex Optimization: Algorithms and Complexity. *Foundations and Trends in ML*.
7. Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization Methods for Large-Scale Machine Learning. *SIAM Review*.

---

## Series Navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Mathematical Foundations | [<-- Read](/en/Machine-Learning-Mathematical-Derivations-1-Introduction-and-Mathematical-Foundations/) |
| 2 | Linear Algebra and Matrix Theory | [<-- Read](/en/Machine-Learning-Mathematical-Derivations-2-Linear-Algebra-and-Matrix-Theory/) |
| 3 | Probability Theory and Statistical Inference | [<-- Previous](/en/Machine-Learning-Mathematical-Derivations-3-Probability-Theory-and-Statistical-Inference/) |
| **4** | **Convex Optimization Theory** | *You are here* |
| 5 | Linear Regression | [Read next -->](/en/Machine-Learning-Mathematical-Derivations-5-Linear-Regression/) |
