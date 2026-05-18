---
title: "ML Math Derivations (19): Neural Networks and Backpropagation"
date: 2026-02-07 09:00:00
tags:
  - Machine Learning
  - Neural Networks
  - Deep Learning
  - Backpropagation
  - Vanishing Gradients
  - Weight Initialization
  - Mathematical Derivations
categories: Machine Learning
series: ml-math-derivations
lang: en
mathjax: true
description: "How does a neural network learn? This article derives forward propagation, the chain rule mechanics of backpropagation, vanishing/exploding gradients, and initialization strategies (Xavier, He)."
disableNunjucks: true
series_order: 19
series_total: 20
translationKey: "ml-math-derivations-19"
---
![ML Math Derivations (19): Neural Networks and Backpropagation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/illustration_1.png)


---

> **Hook.** In 1969 Minsky and Papert proved that a single perceptron could not learn XOR, and connectionist research went into a fifteen-year freeze. The thaw came when Rumelhart, Hinton and Williams realised that *stacking* perceptrons makes the problem disappear — and that the same chain rule everyone learns in calculus, applied carefully, computes every gradient in a multilayer network for the cost of a single extra forward pass. That algorithm is backpropagation. Every gradient in every Transformer, every diffusion model, every GPT trained today still runs on it.

## What You Will Learn

A single perceptron cannot solve XOR. Stack enough of them with nonlinear activations and you obtain a *universal function approximator*. The remaining question is how such a network learns from data. The answer — **backpropagation**, an efficient application of the chain rule that recycles intermediate results during a single backward sweep — is the engine behind every deep learning library written in the last forty years. Understanding it mathematically reveals two further truths: why deep networks suffer from vanishing or exploding gradients, and why the choice of weight initialization is much less arbitrary than it first appears.

**What you will learn:**

1. The perceptron: model, learning rule, and convergence theorem.
2. Forward propagation in matrix form for multilayer networks.
3. Backpropagation: deriving the chain rule layer by layer and storing the right quantities.
4. Why vanishing and exploding gradients arise (and how to read them from a single product of Jacobians).
5. Xavier and He initialization: variance-preservation derivations and when each one applies.

**Prerequisites:** calculus (chain rule, partial derivatives), linear algebra (matrix multiplication), and basic probability.

---

## The Perceptron: Where It All Starts

### Model

Given an input $\mathbf{x}\in\mathbb{R}^d$, weights $\mathbf{w}\in\mathbb{R}^d$, and bias $b\in\mathbb{R}$, the perceptron computes
$$z = \mathbf{w}^{T}\mathbf{x} + b, \qquad \hat{y} = \operatorname{sign}(z) = \begin{cases} +1 & z \geq 0,\\ -1 & z < 0.\end{cases}$$
Geometrically, the equation $\mathbf{w}^{T}\mathbf{x} + b = 0$ defines a hyperplane that splits the input space into two half-spaces, and $\hat y$ records which side the point falls on.

### Learning Algorithm

For a misclassified point $(\mathbf{x}_i, y_i)$ — meaning $y_i(\mathbf{w}^{T}\mathbf{x}_i + b) \leq 0$ — Rosenblatt's update is
$$\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i, \qquad b \leftarrow b + \eta\, y_i.$$
Equivalently, this is stochastic subgradient descent on the *perceptron loss* $\sum_{i \in M} -y_i(\mathbf{w}^{T}\mathbf{x}_i + b)$, where $M$ is the set of currently misclassified points.

### Convergence Theorem

**Theorem (Novikoff, 1962).** If the data is linearly separable — i.e. there exist $\mathbf{w}^{*}$ and $\gamma > 0$ such that $y_i\,\mathbf{w}^{*\,T}\mathbf{x}_i \geq \gamma$ for all $i$ — then the perceptron converges in at most
$$\frac{\|\mathbf{w}^{*}\|^{2}\, R^{2}}{\gamma^{2}} \qquad \text{updates,}$$
where $R = \max_i \|\mathbf{x}_i\|$. The proof bounds $\mathbf{w}_k^{T}\mathbf{w}^{*}$ from below (it grows at least linearly in $k$) and $\|\mathbf{w}_k\|$ from above (it grows at most like $\sqrt{k}$); combining them gives a finite cap on $k$.

### The XOR Problem

The four points $(0,0)\!\to\!0$, $(0,1)\!\to\!1$, $(1,0)\!\to\!1$, $(1,1)\!\to\!0$ are *not* linearly separable: no single hyperplane separates the diagonal pairs. Minsky and Papert's 1969 observation of this fact stalled connectionist research for more than a decade, until multilayer networks made the problem trivially solvable — by drawing two hyperplanes, then combining them.

---

## Multilayer Networks and Forward Propagation

### Architecture

![Multilayer perceptron architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig1_mlp_architecture.png)

A feedforward network is a chain of affine maps interleaved with element-wise nonlinearities:

- **Input layer:** $\mathbf{h}^{(0)} = \mathbf{x}\in\mathbb{R}^{d_0}$.
- **Hidden layers:** $\mathbf{h}^{(l)}\in\mathbb{R}^{d_l}$ for $l = 1, \ldots, L-1$.
- **Output layer:** $\mathbf{h}^{(L)} = \hat{\mathbf{y}}\in\mathbb{R}^{d_L}$.

### Forward Pass (Matrix Form)

![Forward propagation flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig2_forward_propagation.png)

At layer $l$,
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\,\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \tag{1}$$

$$\mathbf{h}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right), \tag{2}$$
with $\mathbf{W}^{(l)}\in\mathbb{R}^{d_l\times d_{l-1}}$ and $\mathbf{b}^{(l)}\in\mathbb{R}^{d_l}$. During the forward pass we **cache** $\mathbf{h}^{(l-1)}$ and $\mathbf{z}^{(l)}$ at every layer; backpropagation will need them.

### Activation Functions

![Activation functions and their derivatives](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig4_activations.png)

| Function | Formula | Derivative | Notes |
|----------|---------|------------|-------|
| Sigmoid | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $\sigma(z)\bigl(1-\sigma(z)\bigr)$ | Output in $(0,1)$; saturates and so vanishes the gradient. |
| Tanh    | $\tanh(z)$ | $1 - \tanh^{2}(z)$ | Zero-centred — better than sigmoid, but still saturates. |
| ReLU    | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | No vanishing for $z>0$; can produce *dead neurons*. |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha\!\approx\!0.01$ | $1$ or $\alpha$ | Fixes dead ReLU. |
| GELU    | $z\,\Phi(z)$ | smooth, bell-shaped near $0$ | Default in Transformers. |
| Swish   | $z\,\sigma(z)$ | $\sigma(z) + z\,\sigma(z)(1-\sigma(z))$ | Self-gated, smooth. |

The right panel of the figure makes one fact obvious: sigmoid's derivative never exceeds $0.25$, while ReLU's is exactly $1$ in the active region. That single difference governs much of what follows.

### Universal Approximation Theorem

![Universal approximation: a 1-hidden-layer ReLU MLP fits diverse targets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig5_universal_approx.png)

**Theorem (Cybenko 1989; Hornik 1991).** For any continuous $f$ on a compact subset of $\mathbb{R}^d$ and any $\varepsilon > 0$, there exists a single-hidden-layer network
$$g(\mathbf{x}) = \sum_{j=1}^{M} v_j\, \sigma\!\left(\mathbf{w}_j^{T}\mathbf{x} + b_j\right)$$
with $\|f - g\|_\infty < \varepsilon$. The figure above shows the theorem in action: a tiny 64-unit ReLU MLP comfortably fits a smooth wave, an absolute value, and even a discontinuous step function.

**Caveat.** The theorem is an *existence* result: it does not bound how large $M$ must be, nor does it guarantee that gradient descent will find a good $g$. In practice, **depth is exponentially more efficient than width** — a depth-$L$ network can express functions that would require a width $\Omega(2^{L})$ in a shallow one (Telgarsky, 2016).

---

## Backpropagation: The Chain Rule at Scale

![ML Math Derivations (19): Neural Networks and Backpropagation — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/illustration_2.png)

### Loss Functions

For regression we typically use mean-squared error,
$$\mathcal{L} = \tfrac{1}{2}\,\bigl\|\hat{\mathbf{y}} - \mathbf{y}\bigr\|^{2},$$
while for classification we use cross-entropy on top of softmax,
$$\mathcal{L} = -\sum_{c} y_c \log \hat{y}_c, \qquad \hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{z}^{(L)}).$$
### The Key Idea

![Backpropagation gradient flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig3_backprop_chain.png)

We need $\partial\mathcal{L}/\partial\mathbf{W}^{(l)}$ and $\partial\mathcal{L}/\partial\mathbf{b}^{(l)}$ for every layer. A naive approach would re-traverse the network for each parameter, costing $\mathcal{O}(P^2)$ for $P$ parameters. Backpropagation exploits the fact that *every gradient shares the same suffix path through the network*: by computing one **error signal** per layer in a single right-to-left sweep, all parameter gradients fall out by simple outer products. The cost drops to $\mathcal{O}(P)$.

### Deriving the Error Signal

Define the error signal at layer $l$:
$$\boldsymbol{\delta}^{(l)} \;=\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}. \tag{3}$$
**Output layer ($l = L$, softmax + cross-entropy).** A short calculation (multiplying by $\partial\hat{\mathbf{y}}/\partial \mathbf{z}^{(L)}$ and using $\sum_c y_c = 1$) collapses to
$$\boldsymbol{\delta}^{(L)} \;=\; \hat{\mathbf{y}} - \mathbf{y}. \tag{4}$$
This algebraic miracle is one of the main reasons softmax + cross-entropy is the default classification head: the gradient is *prediction minus target*, full stop.

**Hidden layers (backward recursion).** For $l < L$ apply the chain rule through $\mathbf{z}^{(l+1)} = \mathbf{W}^{(l+1)}\sigma(\mathbf{z}^{(l)}) + \mathbf{b}^{(l+1)}$:
$$\boldsymbol{\delta}^{(l)} \;=\; \bigl(\mathbf{W}^{(l+1)\,T}\boldsymbol{\delta}^{(l+1)}\bigr) \,\odot\, \sigma'\!\left(\mathbf{z}^{(l)}\right). \tag{5}$$
In words: take the error from the layer above, push it back through the transposed weight matrix, then *gate* it element-wise by the local activation derivative.

### Parameter Gradients

Once the $\boldsymbol{\delta}^{(l)}$'s are in hand the parameter gradients are immediate:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \;=\; \boldsymbol{\delta}^{(l)}\, \mathbf{h}^{(l-1)\,T}, \tag{6}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} \;=\; \boldsymbol{\delta}^{(l)}. \tag{7}$$
Each weight gradient is the **outer product** of the error signal at the layer's output with the activations that arrived at its input — a quantity we already cached during the forward pass.

### Worked Numerical Example: backprop on a 2-layer MLP, by hand

Take the smallest network that exhibits every step: input $x \in \mathbb{R}$, one hidden unit with sigmoid, one linear output, MSE loss against target $y$. Concrete numbers:

- $w_1 = 0.5$, $b_1 = 0$, $w_2 = -1.0$, $b_2 = 0.1$.
- Single training point $(x, y) = (1.0, 0.0)$.

**Forward pass.**

$z_1 = w_1 x + b_1 = 0.5 \cdot 1.0 + 0 = 0.5$.

$h_1 = \sigma(z_1) = 1 / (1 + e^{-0.5}) \approx 0.6225$.

$z_2 = w_2 h_1 + b_2 = -1.0 \cdot 0.6225 + 0.1 = -0.5225$.

$\hat y = z_2 = -0.5225$ (linear output).

$\mathcal{L} = \tfrac{1}{2}(\hat y - y)^2 = \tfrac{1}{2}(-0.5225)^2 \approx 0.1365$.

**Backward pass.**

Output error: $\delta_2 = \partial \mathcal{L}/\partial z_2 = \hat y - y = -0.5225$.

Output-layer gradients (use $h_1$ from cache):

$\partial \mathcal{L}/\partial w_2 = \delta_2 \cdot h_1 = -0.5225 \cdot 0.6225 \approx -0.3253$.

$\partial \mathcal{L}/\partial b_2 = \delta_2 = -0.5225$.

Push the error back through $w_2$ and gate by the sigmoid derivative. Sigmoid derivative: $\sigma'(z_1) = h_1 (1 - h_1) = 0.6225 \cdot 0.3775 \approx 0.2350$.

Hidden error: $\delta_1 = (w_2 \cdot \delta_2) \cdot \sigma'(z_1) = (-1.0 \cdot -0.5225) \cdot 0.2350 = 0.5225 \cdot 0.2350 \approx 0.1228$.

Hidden-layer gradients (use $x$ from cache):

$\partial \mathcal{L}/\partial w_1 = \delta_1 \cdot x = 0.1228 \cdot 1.0 \approx 0.1228$.

$\partial \mathcal{L}/\partial b_1 = \delta_1 \approx 0.1228$.

**SGD step with $\eta = 0.5$.**

$w_2 \leftarrow -1.0 - 0.5 \cdot (-0.3253) = -0.8374$.

$b_2 \leftarrow 0.1 - 0.5 \cdot (-0.5225) = 0.3613$.

$w_1 \leftarrow 0.5 - 0.5 \cdot 0.1228 = 0.4386$.

$b_1 \leftarrow 0 - 0.5 \cdot 0.1228 = -0.0614$.

Recomputing the forward pass with the updated weights gives $\hat y \approx -0.236$, $\mathcal{L} \approx 0.0279$ — the loss dropped by a factor of $\approx 4.9$ in one step. Two structural facts are visible. The hidden-layer gradient is *attenuated* by $\sigma'(z_1) \approx 0.235$ on the way back: with a deeper network this factor compounds and produces vanishing gradients, exactly the problem analysed in the next section. And the gradients at every layer are products of three things that were already computed during the forward pass (an upstream error, a local Jacobian, a cached activation) — no Jacobian matrix is ever materialised, and the per-parameter cost is $\mathcal{O}(1)$.


```python
def backprop(X, y, weights, biases, activations):
    """One forward + one backward pass for a feedforward network."""
    L = len(weights)

    # Forward pass — store activations and pre-activations.
    h = [X]
    z = []
    for l in range(L):
        z_l = weights[l] @ h[-1] + biases[l]
        z.append(z_l)
        h.append(activations[l](z_l))

    # Backward pass — assumes softmax + cross-entropy at the output.
    delta = h[-1] - y
    grad_W = [None] * L
    grad_b = [None] * L
    for l in range(L - 1, -1, -1):
        grad_W[l] = delta @ h[l].T
        grad_b[l] = delta.sum(axis=1, keepdims=True)
        if l > 0:
            delta = (weights[l].T @ delta) * activations[l].deriv(z[l - 1])
    return grad_W, grad_b
```

The whole algorithm is essentially three lines: one outer product, one sum, one transposed matrix–vector product gated by an activation derivative. Modern autograd engines automate the bookkeeping but execute exactly this recursion under the hood.

---

## Vanishing and Exploding Gradients

### The Vanishing Gradient Problem

Iterating equation (5) from layer $L$ down to layer $1$ shows that the gradient at layer $1$ is a *product* of $L-1$ Jacobian-like factors:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} \;\propto\; \prod_{l=2}^{L} \Bigl[\mathbf{W}^{(l)\,T}\, \operatorname{diag}\bigl(\sigma'(\mathbf{z}^{(l-1)})\bigr)\Bigr].$$
For sigmoid, $\sigma'(z)\leq 0.25$. If each weight matrix has spectral norm near $1$, the magnitude of the gradient at layer $1$ is roughly $0.25^{L-1}$. At depth $L=20$ that is about $3.6\times 10^{-12}$ — gradient updates are numerically zero, and the early layers stop learning.

### The Exploding Gradient Problem

Reverse the inequality: if the spectral norms of the weight matrices exceed $1$ and the activation derivatives are not strongly contracting, the same product *grows* exponentially. Numerical overflow follows in a few hundred steps. RNNs are notorious for this because the same weight matrix is multiplied $T$ times in unrolled time.

### Reading the Curves

![Vanishing vs. exploding gradients across depth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig6_vanishing_exploding.png)

The figure reproduces the effect numerically: a unit gradient signal is back-propagated through random networks of increasing depth. Sigmoid (Xavier-initialized) decays by orders of magnitude per layer; ReLU with He initialization stays close to the healthy regime; ReLU with no rescaling explodes by similar factors. The y-axis is logarithmic — every grid line is a factor of ten.

### Solutions

| Problem | Solution | How it helps |
|---------|----------|--------------|
| Vanishing | ReLU activation | $\sigma'(z) = 1$ for $z>0$, no exponential decay. |
| Vanishing | Residual connections | $\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + F(\mathbf{h}^{(l-1)})$ — the gradient has an *identity* path that bypasses the nonlinearities. |
| Vanishing | Batch normalization | Stabilises activation distributions layer by layer. |
| Exploding | Gradient clipping | Rescale $\mathbf{g}\!\leftarrow\!(c/\mid\mathbf{g}\mid)\,\mathbf{g}$ when $\mid\mathbf{g}\mid > c$. |
| Both | Proper initialization | Keep the variance of activations and gradients stable across layers (next section). |

---

## Weight Initialization Strategies

### Why Initialization Matters

- **All zeros.** Every neuron in a layer computes the same function and receives the same gradient; the network cannot break symmetry.
- **Too large.** Pre-activations land in the saturating tails of the activation; gradients vanish.
- **Too small.** Activations collapse toward zero; the signal disappears as it propagates forward.

The goal is therefore to *preserve the variance of activations and gradients across layers*. The right scale is determined by the layer's fan-in and the activation function.

### Xavier (Glorot) Initialization

Consider a single neuron in layer $l$ with fan-in $n_{\text{in}}$ and fan-out $n_{\text{out}}$:
$$z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji}\, h_i.$$
If $w_{ji}$ and $h_i$ are independent with zero mean,
$$\operatorname{Var}(z_j) = n_{\text{in}}\cdot \operatorname{Var}(w)\cdot \operatorname{Var}(h).$$
Demanding $\operatorname{Var}(z) = \operatorname{Var}(h)$ on the *forward* pass gives $\operatorname{Var}(w) = 1/n_{\text{in}}$; the same analysis on the *backward* pass yields $\operatorname{Var}(w) = 1/n_{\text{out}}$. Glorot and Bengio (2010) compromise:
$$\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad w \sim \mathcal{U}\!\left(-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right). \tag{8}$$
Best for **sigmoid and tanh** activations.

### He Initialization

For ReLU, half of the units are zeroed out in expectation, halving the variance contribution of $h$. He et al. (2015) therefore scale up by a factor of two:
$$\operatorname{Var}(w) = \frac{2}{n_{\text{in}}}, \qquad w \sim \mathcal{N}\!\left(0,\, \tfrac{2}{n_{\text{in}}}\right). \tag{9}$$
Best for **ReLU and its variants**.

### Summary Table

| Activation | Initialization | $\operatorname{Var}(w)$ |
|------------|---------------|--------------------------|
| Sigmoid / Tanh | Xavier | $\dfrac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU | He | $\dfrac{2}{n_{\text{in}}}$ |
| Leaky ReLU ($\alpha$) | He (modified) | $\dfrac{2}{(1 + \alpha^{2})\, n_{\text{in}}}$ |

The middle column of the gradient figure above is a direct experimental confirmation: the green curve (ReLU + He) is the only one that stays near order $1$ regardless of depth.

---

## The Loss Landscape

![Non-convex loss landscape with a SGD trajectory](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig7_loss_landscape.png)

Even a tiny network has a loss surface that is sharply non-convex: two deep basins, a saddle, and a small ridge. The figure traces SGD from a poor initialization. Two qualitative facts deserve emphasis:

1. **Most local minima are good enough.** As networks become wider, the minima of the empirical loss tend to lie in flat basins of comparable depth (Choromanska et al., 2015). The intuition that we must find the *global* minimum is a holdover from convex optimization that does not transfer.
2. **Saddle points dominate critical points in high dimensions.** In $\mathbb{R}^d$, the probability that all $d$ Hessian eigenvalues have the same sign at a random critical point is exponentially small. SGD's noise actively helps escape these saddles (Dauphin et al., 2014).

These two observations explain why simple first-order methods continue to dominate deep learning despite the lack of convexity guarantees.

---

## Backprop in PyTorch: what `loss.backward()` actually does

The autograd machinery in PyTorch is not magic — it is a literal implementation of the chain rule we derived above, with two engineering layers on top.

```python
import torch, torch.nn as nn

x = torch.randn(32, 784, requires_grad=False)
y = torch.randint(0, 10, (32,))

net = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
loss_fn = nn.CrossEntropyLoss()

logits = net(x)            # forward: builds the computation graph
loss = loss_fn(logits, y)
loss.backward()            # backward: chain rule, populates .grad on every Parameter

for p in net.parameters():
    print(p.shape, p.grad.norm().item())
```

Three things are happening under `loss.backward()`. First, every tensor with `requires_grad=True` accumulates references to the operation that produced it during the forward pass — this is the *tape*. Second, `backward` walks the tape in reverse topological order; each node knows how to multiply its incoming cotangent by its local Jacobian (without materialising the Jacobian when a more efficient vector-Jacobian product exists). Third, gradients accumulate into `.grad` rather than overwrite — this is why training loops always start with `optimizer.zero_grad()`.

Two PyTorch behaviours trip people up. (1) `loss.backward()` frees the graph by default; calling it twice raises `RuntimeError: Trying to backward through the graph a second time`. Pass `retain_graph=True` if you need to. (2) In-place operations on tensors that participate in autograd will silently corrupt the gradient if the original value was needed for the backward pass. The error message — `one of the variables needed for gradient computation has been modified by an inplace operation` — is the second-most-common training bug after shape mismatches.

## Common misinterpretations

Five things students reliably get wrong about backprop.

**"Backprop is gradient descent."** It is not. Backprop computes $\nabla_{\boldsymbol\theta} L$. SGD, Adam, L-BFGS are *consumers* of that gradient. You can backprop and then do nothing with the gradient; you can also do gradient descent on a function whose gradient you compute by finite differences instead of backprop. Conflating the two leads to confusion when you switch optimisers but keep the same training loop.

**"Vanishing gradients are caused by deep networks."** Half-true. The depth amplifies a per-layer property: if every layer's local Jacobian has spectral norm $< 1$, the product across $L$ layers shrinks as $\rho^L$. ResNets fix this not by changing depth but by adding identity skip connections so the local Jacobian has eigenvalue 1 by construction. Depth is not the disease; the spectral profile of the Jacobian is.

**"ReLU avoids the saturation problem entirely."** Only on the positive side. A ReLU unit whose pre-activation is consistently negative receives zero gradient forever — *dead ReLU*. On a typical training run, 5-15% of ReLU units die in the first few epochs and never recover. Leaky ReLU and GELU exist mostly to mitigate this.

**"More parameters always overfit."** Empirically false in the modern regime. Networks with $10\times$ more parameters than samples routinely generalise well, the *double descent* phenomenon — test error has a peak around the interpolation threshold, then drops again as width grows. The classical bias-variance picture ([Part 20](/en/ml-math-derivations/20-regularization-and-model-selection/)) is incomplete for over-parameterised models.

**"Initialisation only affects training speed."** Wrong. With sufficiently bad init, a deep ReLU net's pre-activations either die out completely or saturate, and *no* amount of training recovers it. He initialisation (variance $2/n_{\text{in}}$) is not an optimisation; it is what makes the network trainable at all.

---

## Exercises

**Exercise 1 — Chain rule.** For $y = \sigma(wx + b)$ with $\sigma(z) = 1/(1+e^{-z})$, find $\partial y/\partial w$.

> **Solution.** $\partial y/\partial w = \sigma'(z)\,x = \sigma(z)\bigl(1 - \sigma(z)\bigr)\,x$.

**Exercise 2 — Vanishing gradient.** Why does sigmoid cause vanishing gradients but ReLU does not?

> **Solution.** $\sigma'(z)\leq 0.25$ uniformly, so over $L$ layers the gradient is multiplied by at most $0.25^{L}$, which decays exponentially. ReLU has $\sigma'(z) = 1$ in the active region, so no exponential decay occurs.

**Exercise 3 — Batch normalization.** How does BatchNorm help training?

> **Solution.** It re-centres and re-scales each layer's pre-activations to roughly zero mean and unit variance. This stabilises gradient magnitudes, allowing larger learning rates, and the mini-batch noise acts as a mild regularizer.

**Exercise 4 — Dropout at test time.** Training uses dropout with $p = 0.5$. What changes at test time?

> **Solution.** Keep all neurons active and multiply weights by $(1 - p) = 0.5$ to preserve the expected output. Most modern libraries use *inverted dropout*: divide activations by $(1-p)$ during training so test-time inference needs no adjustment.

**Exercise 5 — Xavier.** Why does Xavier use $\operatorname{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$?

> **Solution.** Variance preservation on the forward pass requires $n_{\text{in}}\operatorname{Var}(w) = 1$; preservation on the backward pass requires $n_{\text{out}}\operatorname{Var}(w) = 1$. The harmonic-style compromise $\operatorname{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ approximately satisfies both.

---

## What's next

Neural networks give expressivity, but expressivity does not automatically give generalization. When model capacity dwarfs the dataset, training loss can be driven to zero while test loss explodes — that is overfitting. The final chapter of the series targets exactly this: **regularization and model selection**.

Regularization is the safety valve between capacity and data — L2 pulls parameters toward the origin, L1 induces sparsity, Dropout breaks neurons during training, early stopping halts before overfitting kicks in. Model selection tunes those valves to the right strength — cross-validation, AIC/BIC, information criteria. I also touch the two counterintuitive phenomena of the deep-learning era: double descent and the implicit regularization of overparametrized networks. This is the closing chapter not because it is the hardest, but because every "tune the hyperparameters" line in every preceding algorithm uses the language built here.

## References

[1] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408.

[2] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.

[3] Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.

[4] Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.

[5] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 249–256.

[6] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*, 1026–1034.

[7] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456.

[8] Dauphin, Y., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*.

[9] Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G., & LeCun, Y. (2015). The loss surfaces of multilayer networks. *AISTATS*.

[10] Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.

[11] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 6.

---

*This is Part 19 of the [ML Mathematical Derivations](/en/tags/mathematical-derivations/) series. Next: [Part 20 — Regularization and Model Selection](/en/ml-math-derivations/20-regularization-and-model-selection). Previous: [Part 18 — Clustering Algorithms](/en/ml-math-derivations/18-clustering-algorithms).*
