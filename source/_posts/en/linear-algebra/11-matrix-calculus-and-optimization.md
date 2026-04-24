---
title: "Matrix Calculus and Optimization -- The Engine Behind Machine Learning"
date: 2025-02-24 09:00:00
tags:
  - Linear Algebra
  - Matrix Calculus
  - Gradients
  - Optimization
  - Backpropagation
  - Convex Optimization
description: "Adjusting the shower temperature is a tiny version of training a neural network: you change a parameter based on an error signal. Matrix calculus is the language that scales this idea to millions of parameters, and optimization is the engine that does the adjusting."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 11
  total: 18
lang: en
mathjax: true
disableNunjucks: true
---

## From Shower Knobs to Neural Networks

Every morning you train a tiny neural network. The water comes out too cold, so you nudge the knob -- a *parameter* -- in some direction. A second later you observe a new temperature -- the *error signal* -- and nudge again. After three or four iterations you have converged.

Modern deep learning is the same loop, scaled up by seven orders of magnitude. The "knob" is a matrix$W$with hundreds of millions of entries. The "error" is a scalar loss$L$. And the question is the same: **for each parameter, in which direction should I push, and by how much?** The answer lives in a single object: the gradient$\partial L / \partial W$.

This chapter builds that object from the ground up, and then puts it to work.

### What You Will Learn

- **Gradient** -- the derivative of a scalar with respect to a vector, and why it points uphill
- **Directional derivative** -- the slope along any direction, and why steepest descent is "downhill"
- **Jacobian** -- the derivative of a vector function, the chain rule's main building block
- **Hessian** -- the matrix of second derivatives, which classifies critical points
- **Matrix derivatives** -- the trace and determinant identities you actually need
- **Chain rule and backpropagation** -- one rule, applied recursively over a computation graph
- **Convex optimization** -- the property that turns "search" into "follow the slope"
- **Optimizers** -- gradient descent, Newton, SGD, momentum, Adam

### Prerequisites

- Single-variable calculus (derivatives, the chain rule)
- Vectors and dot products (Chapter 1)
- Matrix multiplication (Chapter 3)
- Symmetric matrices and positive definiteness (Chapter 8)

---

## The Gradient: Slope in Many Dimensions

### From One Variable to Many

You run a bubble tea shop. Profit$f$depends on price$x_1$and ad spend$x_2$. You want to know: *if I nudge the price a little, or the ad budget a little, how does profit change?*

The two partial derivatives answer half the question each:

-$\partial f/\partial x_1$-- holding ad spend fixed, profit per dollar of price change.
-$\partial f/\partial x_2$-- holding price fixed, profit per dollar of ad spend.

Pack them into a vector and you get the **gradient**:

$$
\nabla f = \begin{pmatrix} \partial f/\partial x_1 \\ \partial f/\partial x_2 \end{pmatrix}
$$

For a general function$f: \mathbb{R}^n \to \mathbb{R}$:

$$
\nabla f(\vec{x}) = \begin{pmatrix} \partial f/\partial x_1 \\ \vdots \\ \partial f/\partial x_n \end{pmatrix} \in \mathbb{R}^n
$$

### Three Geometric Meanings That Justify the Whole Story

The gradient is more than a stack of partial derivatives; three geometric facts explain why this particular packaging is the right one.

**1. Direction.** The gradient points in the direction in which$f$increases fastest. Of all unit directions you could walk from$\vec{x}$, the one parallel to$\nabla f$gives you the steepest ascent. Walk against it for the steepest descent -- that single sentence is the entire justification for gradient descent.

**2. Magnitude.**$\|\nabla f\|$is the rate of increase along that steepest direction. A large gradient means the surface is steep here; a small step makes a big change in$f$. A near-zero gradient means you're on a plateau (or at a critical point).

**3. Orthogonality.** The gradient is perpendicular to the level set$\{\vec{y}: f(\vec{y}) = f(\vec{x})\}$. On a topographic map, contour lines mark constant altitude; the gradient points straight uphill, perpendicular to those contours.

![Gradient as the direction of steepest ascent, shown on a 3D bowl and a contour map](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig1_gradient_steepest_ascent.png)

### Three Examples Worth Memorizing

These three formulas appear constantly. Memorize them.

**Linear function.**$f(\vec{x}) = \vec{a}^T\vec{x}$. Then$\nabla f = \vec{a}$.

The graph is a hyperplane and$\vec{a}$is its normal direction -- the unique direction of fastest increase.

**Squared norm.**$f(\vec{x}) = \vec{x}^T\vec{x} = \|\vec{x}\|^2$. Then$\nabla f = 2\vec{x}$.

A bowl centered at the origin. The gradient points radially outward, away from the minimum.

**General quadratic.**$f(\vec{x}) = \vec{x}^TA\vec{x}$with$A$symmetric. Then$\nabla f = 2A\vec{x}$.

Every quadratic loss in this book reduces to this formula. If$A$is not symmetric, replace$A$with$\tfrac{1}{2}(A+A^T)$-- the gradient only sees the symmetric part anyway.

---

## Directional Derivative: Slope Along Any Path

You don't have to walk straight uphill. The **directional derivative** tells you the slope along any unit direction$\vec{d}$:

$$
D_{\vec{d}}f = \nabla f \cdot \vec{d} = \|\nabla f\| \cos\theta
$$

where$\theta$is the angle between$\nabla f$and$\vec{d}$. Three values of$\theta$matter:

-$\theta = 0°$(walk *along* the gradient): maximum increase$\|\nabla f\|$.
-$\theta = 90°$(walk *along the contour*): zero change in$f$.
-$\theta = 180°$(walk *against* the gradient): maximum decrease$-\|\nabla f\|$.

That last line is the theoretical foundation of **gradient descent**: to decrease$f$as fast as possible, walk in the direction$-\nabla f$.

---

## The Jacobian: Vector In, Vector Out

When the output is also a vector, one gradient is no longer enough -- you need a matrix of partial derivatives, one per (input, output) pair.

### A Cooking Analogy

Three seasonings$(x_1, x_2, x_3)$control three taste metrics$(f_1, f_2, f_3)$: saltiness, sweetness, spiciness. There are$3 \times 3 = 9$"how does input$j$affect output$i$?" relationships. Pack them into a$3 \times 3$matrix and you have the Jacobian.

### Definition

For$\vec{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** is the$m \times n$matrix

$$
J = \frac{\partial \vec{f}}{\partial \vec{x}} = \begin{pmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{pmatrix}
$$

Row$i$is the gradient of$f_i$. Entry$(i, j)$is "how much does output$i$change when input$j$is nudged?"

### Geometric Meaning: Best Linear Approximation

Near$\vec{x}_0$, the Jacobian is the best linear approximation to$\vec{f}$:

$$
\vec{f}(\vec{x}_0 + \Delta\vec{x}) \approx \vec{f}(\vec{x}_0) + J\,\Delta\vec{x}
$$

Geometrically,$J$tells you how the function deforms space locally: a small square in input space becomes a small parallelogram in output space, and$J$is exactly the matrix that performs that deformation.

### Classic Example: Polar Coordinates

The change of variables$(r, \theta) \to (x, y) = (r\cos\theta, r\sin\theta)$has Jacobian

$$
J = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}
$$

Its determinant is$\det(J) = r$-- the Jacobian factor that turns$dx\,dy$into$r\,dr\,d\theta$in polar integrals. Calculus students meet that$r$as a mysterious extra factor; here it is, derived directly.

---

## The Hessian: Curvature, and Critical Point Classification

The gradient tells you the slope; the Hessian tells you how the slope changes -- the **curvature**.

### Definition

For$f: \mathbb{R}^n \to \mathbb{R}$, the Hessian is the$n \times n$matrix of second partial derivatives:

$$
H = \begin{pmatrix} \partial^2 f/\partial x_1^2 & \cdots & \partial^2 f/\partial x_1 \partial x_n \\ \vdots & \ddots & \vdots \\ \partial^2 f/\partial x_n \partial x_1 & \cdots & \partial^2 f/\partial x_n^2 \end{pmatrix}
$$

If second partials are continuous,$H$is **symmetric** (Schwarz/Clairaut). That symmetry is what makes the rest of the chapter work -- it lets us talk about eigenvalues of$H$.

### Second-Order Taylor Expansion

The Hessian appears in the second-order Taylor expansion:

$$
f(\vec{x}_0 + \Delta\vec{x}) \approx f(\vec{x}_0) + \nabla f^T \Delta\vec{x} + \frac{1}{2}\Delta\vec{x}^T H \Delta\vec{x}
$$

Three terms: current value, linear correction (gradient), quadratic correction (curvature).

### Classifying Critical Points

At a point where$\nabla f = \vec{0}$(a *critical point*), the linear term vanishes, and the local behavior of$f$is dictated entirely by the Hessian:

| Hessian property | Type | Geometric picture |
|---|---|---|
| Positive definite ($\lambda_i > 0$) | Local minimum | Bottom of a bowl |
| Negative definite ($\lambda_i < 0$) | Local maximum | Top of a hill |
| Indefinite (mixed signs) | Saddle point | Horse saddle |
| Semidefinite (some$\lambda_i = 0$) | Inconclusive | Flat in some directions |

**Worked examples.**$f(x,y) = x^2 + y^2$has$H = \mathrm{diag}(2,2)$, positive definite, so$(0,0)$is a minimum.$f(x,y) = x^2 - y^2$has$H = \mathrm{diag}(2,-2)$, indefinite, so$(0,0)$is a saddle.

![Critical points classified by the eigenvalue signs of the Hessian](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig2_critical_points.png)

> **Why saddle points matter for deep learning.** In high-dimensional non-convex losses, almost every critical point you encounter is a saddle, not a true minimum. Modern optimizers must be able to escape them, which is why naive Newton-style methods struggle and why momentum-based methods dominate.

---

## Matrix Derivatives: When Parameters Are Matrices

In neural networks, parameters are usually matrices (weight matrices). We need "the derivative of a scalar loss with respect to a weight matrix".

### Definition

For$f: \mathbb{R}^{m \times n} \to \mathbb{R}$:

$$
\frac{\partial f}{\partial X} = \begin{pmatrix} \partial f/\partial x_{11} & \cdots & \partial f/\partial x_{1n}\\ \vdots & \ddots & \vdots\\ \partial f/\partial x_{m1} & \cdots & \partial f/\partial x_{mn} \end{pmatrix}
$$

The result has the **same shape** as$X$. This shape rule is the single most useful sanity check in the whole chapter -- if your derivation produces an answer with the wrong shape, you've made an algebra mistake.

### Identities You Will Actually Use

| Function | Derivative w.r.t.$X$| Notes |
|---|---|---|
|$\text{tr}(AX)$|$A^T$| Linear in$X$|
|$\text{tr}(X^TAX)$|$(A + A^T)X$| Quadratic;$2AX$if$A$is symmetric |
|$\text{tr}(AXB)$|$A^TB^T$| The "sandwich" |
|$\det(X)$|$\det(X)\, X^{-T}$| Determinant |
|$\ln\det(X)$|$X^{-T}$| Log-determinant -- ubiquitous in MLE |
|$X^{-1}$(differential) |$\partial(X^{-1}) = -X^{-1}(\partial X)X^{-1}$| Inverse |

The trace appears constantly because *any* scalar function of a matrix can be written as a trace, which then plays well with the cyclic identity$\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$. The full reference is Petersen & Pedersen's *Matrix Cookbook*.

![Vector and matrix derivative shape rules at a glance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig6_shape_rules_cheatsheet.png)

---

## The Chain Rule and Backpropagation

### The Chain Rule, in Matrix Form

If$\vec{u} = \vec{h}(\vec{x})$and$y = g(\vec{u})$with$g$scalar:

$$
\frac{\partial (g \circ \vec{h})}{\partial \vec{x}} = J_h^T \nabla g
$$

If both stages are vector-valued, Jacobians simply *multiply* along the chain:

$$
J_{\text{total}} = J_k\,J_{k-1}\,\cdots\,J_1
$$

**River pollution analogy.** An upstream factory's discharge changes by$\delta_1$; mid-stream concentration responds by$\delta_2 = J_1\delta_1$; downstream ecology responds by$\delta_3 = J_2\delta_2$. The total response is$\delta_3 = J_2 J_1 \delta_1$-- one multiplication per stage.

### Backpropagation: the Chain Rule Done Efficiently

Any composite expression -- including a 1000-layer neural network -- breaks into elementary operations on a directed acyclic graph called a **computation graph**. Backpropagation is the chain rule walked *backwards* over this graph.

**Why backwards?** Suppose you have$n$inputs and$1$output (the usual setup: millions of parameters, one scalar loss).

- *Forward-mode* (a.k.a. dual numbers) computes the Jacobian-vector product$J\vec{v}$. To get all gradients you need$n$such passes.
- *Reverse-mode* (backpropagation) computes the vector-Jacobian product$\vec{v}^TJ$. **A single pass** yields gradients with respect to all$n$inputs simultaneously.

For deep learning, "$n$" is "every parameter in the model" -- so reverse-mode is roughly a million times cheaper than forward-mode. That ratio is the only reason deep learning is computationally tractable at all.

![Forward pass computes values; backward pass propagates gradients along the same graph](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig7_backprop_chain_rule.png)

### Backpropagation Through a Fully Connected Layer

This is the workhorse derivation; if you internalize one calculation in the chapter, make it this one.

**Forward.**

$$
\vec{z} = W\vec{x} + \vec{b}, \qquad \vec{a} = \sigma(\vec{z})\quad(\text{element-wise})
$$

**Backward.** Suppose later layers have already produced$\delta_a := \partial L / \partial \vec{a}$. Then:

1. **Through the activation:**$\delta_z = \delta_a \odot \sigma'(\vec{z})$($\odot$is element-wise product).
2. **Weight gradient:**$\dfrac{\partial L}{\partial W} = \delta_z \vec{x}^T$. Note the shape: outer product of the upstream signal with the input. Same shape as$W$.
3. **Bias gradient:**$\dfrac{\partial L}{\partial \vec{b}} = \delta_z$.
4. **Pass-through to the previous layer:**$\dfrac{\partial L}{\partial \vec{x}} = W^T\delta_z$.

Every modern framework (PyTorch, JAX, TensorFlow) executes exactly this recipe, layer by layer.

### Activation Functions and Their Derivatives

**ReLU.**$\sigma(z) = \max(0, z)$, so$\sigma'(z) = 1$for$z > 0$and$0$otherwise. Cheap, no saturation in the positive region. Downside: zero gradient on the negative side -- the dreaded "dying ReLU".

**Sigmoid.**$\sigma(z) = 1/(1+e^{-z})$, with the elegant identity$\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Output in$(0,1)$, interpretable as probability. Downside: saturates at both ends, so gradients vanish in deep networks.

**Softmax + cross-entropy.** Used at the output of a classifier. The combined gradient simplifies beautifully:

$$
\frac{\partial L}{\partial \vec{z}} = \hat{\vec{y}} - \vec{y}
$$

"Predicted probability minus true label." That cleanliness is exactly why softmax + cross-entropy is the default for classification -- the gradient is local, cheap, and well-scaled.

---

## Convex Optimization: The Property That Makes Things Easy

### What "Convex" Means

A function$f$is **convex** if the chord between any two points on its graph lies above the graph:

$$
f(\alpha\vec{x} + (1-\alpha)\vec{y}) \leq \alpha f(\vec{x}) + (1-\alpha)f(\vec{y}), \quad \alpha \in [0,1]
$$

The single fact that makes convexity precious:

> **Theorem.** Every local minimum of a convex function is a global minimum.

For convex problems, "the optimizer converged to a local minimum" is the *whole* story -- there is no worry about getting stuck in a bad basin.

### Three Equivalent Characterizations (for$C^2$functions)

1.$f$is convex.
2. The graph lies above each tangent:$f(\vec{y}) \geq f(\vec{x}) + \nabla f(\vec{x})^T(\vec{y} - \vec{x})$.
3. The Hessian is positive semidefinite everywhere:$H(\vec{x}) \succeq 0$.

For *strict* convexity, replace$\succeq$with$\succ$.

### A Convex-Function Sampler

| Function | Why it's convex |
|---|---|
|$\vec{a}^T\vec{x} + b$| Affine -- both convex and concave |
|$\|\vec{x}\|_p$($p \geq 1$) | Triangle inequality |
|$\vec{x}^TA\vec{x}$with$A \succeq 0$| Hessian is$2A \succeq 0$|
|$e^x$,$x \log x$| Second derivative$> 0$|
|$-\log x$($x > 0$) | Underlies log-likelihood losses |

![Convex bowl vs non-convex landscape: only one of these gives a globally optimal answer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig5_convex_vs_nonconvex.png)

### KKT Conditions

For the constrained problem$\min f(\vec{x})$subject to$g_i(\vec{x}) \leq 0$and$h_j(\vec{x}) = 0$, the KKT conditions are necessary at any optimum and sufficient when the problem is convex:

1. **Primal feasibility:**$g_i(\vec{x}^*) \leq 0$,$h_j(\vec{x}^*) = 0$.
2. **Dual feasibility:**$\mu_i \geq 0$.
3. **Complementary slackness:**$\mu_i\, g_i(\vec{x}^*) = 0$(an inequality constraint is either tight or its multiplier is zero).
4. **Stationarity:**$\nabla f + \sum_i \mu_i \nabla g_i + \sum_j \nu_j \nabla h_j = \vec{0}$.

KKT is the workhorse of constrained optimization -- and the Lagrangian framework that produces it is also how SVMs, max-entropy models, and PCA are derived.

---

## Optimization Algorithms

### Gradient Descent

$$
\vec{x}_{k+1} = \vec{x}_k - \alpha \nabla f(\vec{x}_k)
$$

Walk downhill. The step size$\alpha$is the **learning rate**: too small and you crawl, too large and you diverge.

For a convex quadratic with condition number$\kappa$, the convergence rate is roughly$\bigl(\tfrac{\kappa - 1}{\kappa + 1}\bigr)^k$-- so an ill-conditioned problem ($\kappa \gg 1$) can be brutally slow, and the trajectory zig-zags across the long axis of the bowl.

![Gradient descent trajectory: well-conditioned (left) vs ill-conditioned zig-zag (right)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig3_gradient_descent_path.png)

### Newton's Method

$$
\vec{x}_{k+1} = \vec{x}_k - H^{-1}\nabla f(\vec{x}_k)
$$

Approximate$f$by its second-order Taylor expansion and jump straight to the minimum of that quadratic. The result is **quadratic convergence**: the number of correct digits roughly doubles per step.

The catch: forming and inverting$H$costs$O(n^3)$, prohibitive when$n$is in the millions. And on non-convex losses, Newton's method may steer toward a saddle or even a maximum -- it goes wherever$\nabla f$is zero.

![Newton's method versus gradient descent on a quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/11-matrix-calculus-and-optimization/fig4_newton_vs_gd.png)

### Stochastic Gradient Descent (SGD)

When the loss is a sum over$N$samples,$L = \tfrac{1}{N}\sum_i \ell_i$, an exact gradient costs$O(N)$. Replace it with a single sample's (or mini-batch's) gradient:

$$
\vec{x}_{k+1} = \vec{x}_k - \alpha \nabla \ell_{i_k}
$$

Massively cheaper per step. The injected noise also helps escape narrow saddles and shallow basins -- a feature, not a bug, for non-convex deep learning.

### Momentum

Plain SGD jitters. Momentum smooths it by accumulating an exponential moving average of past gradients:

$$
\vec{v}_{k+1} = \beta\vec{v}_k + \nabla f(\vec{x}_k), \qquad \vec{x}_{k+1} = \vec{x}_k - \alpha\vec{v}_{k+1}
$$

The standard mental picture: a heavy ball rolling downhill. Inertia smooths out the path and lets it coast through small bumps.

### Adam

Combines momentum (first moment) with per-parameter adaptive step sizes (second moment):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla f)^2
$$

$$
\vec{x}_{t+1} = \vec{x}_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
$$

Parameters with consistently large gradients get a smaller effective learning rate; quiet parameters get a larger one. With$\beta_1 = 0.9$,$\beta_2 = 0.999$,$\alpha = 10^{-3}$it is the default optimizer for most modern deep learning workflows.

---

## Three Worked Applications

### Linear Regression (Closed Form)

Objective:$L(\vec{w}) = \|X\vec{w} - \vec{y}\|^2$.

Gradient:$\nabla L = 2X^T(X\vec{w} - \vec{y})$.

Setting it to zero gives the **normal equation**:

$$
\vec{w}^* = (X^TX)^{-1}X^T\vec{y}
$$

This closed form exists exactly because the loss is a convex quadratic in$\vec{w}$.

### Ridge Regression

Add an$\ell_2$regularizer:$L(\vec{w}) = \|X\vec{w} - \vec{y}\|^2 + \lambda\|\vec{w}\|^2$.

$$
\vec{w}^* = (X^TX + \lambda I)^{-1}X^T\vec{y}
$$

The$\lambda I$term shifts every eigenvalue of$X^TX$by$\lambda$, which guarantees invertibility and tames the condition number -- a direct application of Chapter 10's intuition.

### PCA as Optimization

PCA can be stated as the constrained problem

$$
\max_{\|\vec{w}\|=1} \vec{w}^T\Sigma\vec{w}
$$

The Lagrangian is$\vec{w}^T\Sigma\vec{w} - \lambda(\vec{w}^T\vec{w} - 1)$, and setting its gradient to zero gives$\Sigma\vec{w} = \lambda\vec{w}$. PCA is an eigenvalue problem -- *because* of the KKT stationarity condition. The optimizer is the leading eigenvector of$\Sigma$.

---

## Formula Quick Reference

### Vector Derivatives

|$f(\vec{x})$|$\nabla f$|
|---|---|
|$\vec{a}^T\vec{x}$|$\vec{a}$|
|$\vec{x}^T\vec{x}$|$2\vec{x}$|
|$\vec{x}^TA\vec{x}$($A$symmetric) |$2A\vec{x}$|
|$\|\vec{x}\|_2$|$\vec{x}/\|\vec{x}\|_2$|

### Matrix Derivatives

|$f(X)$|$\partial f/\partial X$|
|---|---|
|$\text{tr}(AX)$|$A^T$|
|$\text{tr}(X^TAX)$|$(A+A^T)X$|
|$\det(X)$|$\det(X)\cdot X^{-T}$|
|$\ln\det(X)$|$X^{-T}$|

---

## Python Examples

### Gradient Checking

The single best debugging tool for any hand-derived gradient: compare to a centered finite difference.

```python
import numpy as np

def gradient_check(f, grad_f, x, epsilon=1e-5):
    """Compare an analytical gradient against a numerical one."""
    analytical = grad_f(x)
    numerical = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += epsilon
        x_minus = x.copy(); x_minus[i] -= epsilon
        numerical[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    rel_error = np.linalg.norm(analytical - numerical) / (
        np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-10)
    return rel_error

# Test: f(x) = x^T A x
A = np.array([[2, 1], [1, 3]], dtype=float)
f = lambda x: x @ A @ x
grad_f = lambda x: 2 * A @ x

x = np.array([1.0, 2.0])
print(f"Relative error: {gradient_check(f, grad_f, x):.2e}")  # ~1e-10
```

A relative error below$10^{-7}$is good; below$10^{-9}$is excellent.

### Comparing GD, Momentum, and Adam

```python
import numpy as np
import matplotlib.pyplot as plt

def optimize_comparison():
    """Compare GD, Momentum, and Adam on a 2D anisotropic quadratic."""
    A = np.array([[10, 0], [0, 1]])  # condition number = 10
    f = lambda x: 0.5 * x @ A @ x
    grad = lambda x: A @ x

    methods = {
        'GD': {'lr': 0.1},
        'Momentum': {'lr': 0.1, 'beta': 0.9},
        'Adam': {'lr': 0.3, 'beta1': 0.9, 'beta2': 0.999}
    }

    x0 = np.array([5.0, 5.0])
    paths = {}

    for name, p in methods.items():
        x, v, m, vv = x0.copy(), np.zeros(2), np.zeros(2), np.zeros(2)
        path = [x.copy()]
        for t in range(1, 50):
            g = grad(x)
            if name == 'GD':
                x = x - p['lr'] * g
            elif name == 'Momentum':
                v = p['beta'] * v + g
                x = x - p['lr'] * v
            elif name == 'Adam':
                m = p['beta1'] * m + (1-p['beta1']) * g
                vv = p['beta2'] * vv + (1-p['beta2']) * g**2
                m_hat = m / (1 - p['beta1']**t)
                v_hat = vv / (1 - p['beta2']**t)
                x = x - p['lr'] * m_hat / (np.sqrt(v_hat) + 1e-8)
            path.append(x.copy())
        paths[name] = np.array(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    t = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(t, t)
    Z = 0.5 * (A[0,0]*X**2 + A[1,1]*Y**2)
    ax.contour(X, Y, Z, levels=20, alpha=0.5)
    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], 'o-', markersize=3, label=name)
    ax.legend(); ax.set_aspect('equal')
    ax.set_title('Optimization paths on a 2D quadratic')
    plt.show()

optimize_comparison()
```

---

## Exercises

### Warm-Up

1. Compute the gradient of$f(\vec{x}) = 3x_1^2 + 2x_1x_2 + x_2^2$.
2. Find all critical points of$f(x,y) = x^3 - 3xy + y^3$and classify each one using the Hessian.
3. Prove:$\nabla(\vec{x}^TA\vec{x}) = (A + A^T)\vec{x}$, working from the partial-derivative definition.

### Going Deeper

4. Derive the backpropagation formulas for a two-layer network$f = \sigma_2(W_2\sigma_1(W_1\vec{x} + \vec{b}_1) + \vec{b}_2)$. Pay attention to shapes at every step.
5. Prove that Newton's method converges in *one* step on any quadratic function.
6. Prove: any local minimum of a convex function is a global minimum (use the chord-above-graph definition).

### Coding Challenges

7. Implement the gradient checker above and verify it on three different functions, including one matrix-valued.
8. Implement SGD, Momentum, and Adam from scratch and compare their convergence on the Rosenbrock function. Plot the paths.
9. Implement a two-layer neural network on a toy 2D classification dataset using only numpy, with hand-coded backpropagation (no autograd).

---

## Chapter Summary

| Concept | Key fact | Role in ML |
|---|---|---|
| Gradient | Direction of steepest ascent | Drives gradient descent |
| Jacobian | Best linear approximation | Building block of the chain rule |
| Hessian | Curvature matrix | Classifies critical points |
| Chain rule | Jacobians multiply | Powers backpropagation |
| Convexity | Local min = global min | Guarantee for many losses |
| Adam | Adaptive learning rates | Default optimizer for deep learning |

The single most important takeaway: **for a scalar loss with a million parameters, one backward pass over the computation graph yields all million gradients.** Everything else in this chapter exists to make that one fact precise and trustworthy.

---

## Series Navigation

**Previous:** [Chapter 10 -- Matrix Norms and Condition Numbers](/en/chapter-10-matrix-norms-and-condition-numbers/)

**Next:** [Chapter 12 -- Sparse Matrices and Compressed Sensing](/en/chapter-12-sparse-matrices-and-compressed-sensing/)

*This is Chapter 11 of the 18-part "Essence of Linear Algebra" series.*

## References

- Petersen, K. B. & Pedersen, M. S. *The Matrix Cookbook*. (The reference for matrix derivative identities.)
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Chapter 6 -- backpropagation in detail.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization* -- the standard text on convexity, Lagrangians, and KKT.
- Nocedal, J. & Wright, S. (2006). *Numerical Optimization* -- the canonical reference for Newton, quasi-Newton, and trust-region methods.
- Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *ICLR*.
