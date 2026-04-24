---
title: "ML Math Derivations (19): Neural Networks and Backpropagation"
date: 2024-03-19 09:00:00
tags:
  - Machine Learning
  - Neural Networks
  - Deep Learning
  - Backpropagation
  - Vanishing Gradients
  - Weight Initialization
  - Mathematical Derivations
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 19
  total: 20
lang: en
mathjax: true
description: "How does a neural network learn? This article derives forward propagation, the chain rule mechanics of backpropagation, vanishing/exploding gradients, and initialization strategies (Xavier, He)."
disableNunjucks: true
---

## What This Article Covers

A single perceptron cannot solve XOR. Stack enough of them with nonlinear activations and you obtain a *universal function approximator*. The remaining question is how such a network learns from data. The answer — **backpropagation**, an efficient application of the chain rule that recycles intermediate results during a single backward sweep — is the engine behind every deep learning library written in the last forty years. Understanding it mathematically reveals two further truths: why deep networks suffer from vanishing or exploding gradients, and why the choice of weight initialization is much less arbitrary than it first appears.

**What you will learn:**

1. The perceptron: model, learning rule, and convergence theorem.
2. Forward propagation in matrix form for multilayer networks.
3. Backpropagation: deriving the chain rule layer by layer and storing the right quantities.
4. Why vanishing and exploding gradients arise (and how to read them from a single product of Jacobians).
5. Xavier and He initialization: variance-preservation derivations and when each one applies.

**Prerequisites:** calculus (chain rule, partial derivatives), linear algebra (matrix multiplication), and basic probability.

---

## 1. The Perceptron: Where It All Starts

### 1.1 Model

Given an input $\mathbf{x}\in\mathbb{R}^d$, weights $\mathbf{w}\in\mathbb{R}^d$, and bias $b\in\mathbb{R}$, the perceptron computes

$$
z = \mathbf{w}^{T}\mathbf{x} + b, \qquad \hat{y} = \operatorname{sign}(z) = \begin{cases} +1 & z \geq 0,\\ -1 & z < 0.\end{cases}
$$

Geometrically, the equation $\mathbf{w}^{T}\mathbf{x} + b = 0$ defines a hyperplane that splits the input space into two half-spaces, and $\hat y$ records which side the point falls on.

### 1.2 Learning Algorithm

For a misclassified point $(\mathbf{x}_i, y_i)$ — meaning $y_i(\mathbf{w}^{T}\mathbf{x}_i + b) \leq 0$ — Rosenblatt's update is

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i, \qquad b \leftarrow b + \eta\, y_i.
$$

Equivalently, this is stochastic subgradient descent on the *perceptron loss* $\sum_{i \in M} -y_i(\mathbf{w}^{T}\mathbf{x}_i + b)$, where $M$ is the set of currently misclassified points.

### 1.3 Convergence Theorem

**Theorem (Novikoff, 1962).** If the data is linearly separable — i.e. there exist $\mathbf{w}^{*}$ and $\gamma > 0$ such that $y_i\,\mathbf{w}^{*\,T}\mathbf{x}_i \geq \gamma$ for all $i$ — then the perceptron converges in at most

$$
\frac{\|\mathbf{w}^{*}\|^{2}\, R^{2}}{\gamma^{2}} \qquad \text{updates,}
$$

where $R = \max_i \|\mathbf{x}_i\|$. The proof bounds $\mathbf{w}_k^{T}\mathbf{w}^{*}$ from below (it grows at least linearly in $k$) and $\|\mathbf{w}_k\|$ from above (it grows at most like $\sqrt{k}$); combining them gives a finite cap on $k$.

### 1.4 The XOR Problem

The four points $(0,0)\!\to\!0$, $(0,1)\!\to\!1$, $(1,0)\!\to\!1$, $(1,1)\!\to\!0$ are *not* linearly separable: no single hyperplane separates the diagonal pairs. Minsky and Papert's 1969 observation of this fact stalled connectionist research for more than a decade, until multilayer networks made the problem trivially solvable — by drawing two hyperplanes, then combining them.

---

## 2. Multilayer Networks and Forward Propagation

### 2.1 Architecture

![Multilayer perceptron architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig1_mlp_architecture.png)

A feedforward network is a chain of affine maps interleaved with element-wise nonlinearities:

- **Input layer:** $\mathbf{h}^{(0)} = \mathbf{x}\in\mathbb{R}^{d_0}$.
- **Hidden layers:** $\mathbf{h}^{(l)}\in\mathbb{R}^{d_l}$ for $l = 1, \ldots, L-1$.
- **Output layer:** $\mathbf{h}^{(L)} = \hat{\mathbf{y}}\in\mathbb{R}^{d_L}$.

### 2.2 Forward Pass (Matrix Form)

![Forward propagation flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig2_forward_propagation.png)

At layer $l$,

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\,\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \tag{1}
$$

$$
\mathbf{h}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right), \tag{2}
$$

with $\mathbf{W}^{(l)}\in\mathbb{R}^{d_l\times d_{l-1}}$ and $\mathbf{b}^{(l)}\in\mathbb{R}^{d_l}$. During the forward pass we **cache** $\mathbf{h}^{(l-1)}$ and $\mathbf{z}^{(l)}$ at every layer; backpropagation will need them.

### 2.3 Activation Functions

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

### 2.4 Universal Approximation Theorem

![Universal approximation: a 1-hidden-layer ReLU MLP fits diverse targets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig5_universal_approx.png)

**Theorem (Cybenko 1989; Hornik 1991).** For any continuous $f$ on a compact subset of $\mathbb{R}^d$ and any $\varepsilon > 0$, there exists a single-hidden-layer network

$$
g(\mathbf{x}) = \sum_{j=1}^{M} v_j\, \sigma\!\left(\mathbf{w}_j^{T}\mathbf{x} + b_j\right)
$$

with $\|f - g\|_\infty < \varepsilon$. The figure above shows the theorem in action: a tiny 64-unit ReLU MLP comfortably fits a smooth wave, an absolute value, and even a discontinuous step function.

**Caveat.** The theorem is an *existence* result: it does not bound how large $M$ must be, nor does it guarantee that gradient descent will find a good $g$. In practice, **depth is exponentially more efficient than width** — a depth-$L$ network can express functions that would require a width $\Omega(2^{L})$ in a shallow one (Telgarsky, 2016).

---

## 3. Backpropagation: The Chain Rule at Scale

### 3.1 Loss Functions

For regression we typically use mean-squared error,

$$
\mathcal{L} = \tfrac{1}{2}\,\bigl\|\hat{\mathbf{y}} - \mathbf{y}\bigr\|^{2},
$$

while for classification we use cross-entropy on top of softmax,

$$
\mathcal{L} = -\sum_{c} y_c \log \hat{y}_c, \qquad \hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{z}^{(L)}).
$$

### 3.2 The Key Idea

![Backpropagation gradient flow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig3_backprop_chain.png)

We need $\partial\mathcal{L}/\partial\mathbf{W}^{(l)}$ and $\partial\mathcal{L}/\partial\mathbf{b}^{(l)}$ for every layer. A naive approach would re-traverse the network for each parameter, costing $\mathcal{O}(P^2)$ for $P$ parameters. Backpropagation exploits the fact that *every gradient shares the same suffix path through the network*: by computing one **error signal** per layer in a single right-to-left sweep, all parameter gradients fall out by simple outer products. The cost drops to $\mathcal{O}(P)$.

### 3.3 Deriving the Error Signal

Define the error signal at layer $l$:

$$
\boldsymbol{\delta}^{(l)} \;=\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}. \tag{3}
$$

**Output layer ($l = L$, softmax + cross-entropy).** A short calculation (multiplying by $\partial\hat{\mathbf{y}}/\partial \mathbf{z}^{(L)}$ and using $\sum_c y_c = 1$) collapses to

$$
\boldsymbol{\delta}^{(L)} \;=\; \hat{\mathbf{y}} - \mathbf{y}. \tag{4}
$$

This algebraic miracle is one of the main reasons softmax + cross-entropy is the default classification head: the gradient is *prediction minus target*, full stop.

**Hidden layers (backward recursion).** For $l < L$ apply the chain rule through $\mathbf{z}^{(l+1)} = \mathbf{W}^{(l+1)}\sigma(\mathbf{z}^{(l)}) + \mathbf{b}^{(l+1)}$:

$$
\boldsymbol{\delta}^{(l)} \;=\; \bigl(\mathbf{W}^{(l+1)\,T}\boldsymbol{\delta}^{(l+1)}\bigr) \,\odot\, \sigma'\!\left(\mathbf{z}^{(l)}\right). \tag{5}
$$

In words: take the error from the layer above, push it back through the transposed weight matrix, then *gate* it element-wise by the local activation derivative.

### 3.4 Parameter Gradients

Once the $\boldsymbol{\delta}^{(l)}$'s are in hand the parameter gradients are immediate:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \;=\; \boldsymbol{\delta}^{(l)}\, \mathbf{h}^{(l-1)\,T}, \tag{6}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} \;=\; \boldsymbol{\delta}^{(l)}. \tag{7}
$$

Each weight gradient is the **outer product** of the error signal at the layer's output with the activations that arrived at its input — a quantity we already cached during the forward pass.

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

## 4. Vanishing and Exploding Gradients

### 4.1 The Vanishing Gradient Problem

Iterating equation (5) from layer $L$ down to layer $1$ shows that the gradient at layer $1$ is a *product* of $L-1$ Jacobian-like factors:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} \;\propto\; \prod_{l=2}^{L} \Bigl[\mathbf{W}^{(l)\,T}\, \operatorname{diag}\bigl(\sigma'(\mathbf{z}^{(l-1)})\bigr)\Bigr].
$$

For sigmoid, $\sigma'(z)\leq 0.25$. If each weight matrix has spectral norm near $1$, the magnitude of the gradient at layer $1$ is roughly $0.25^{L-1}$. At depth $L=20$ that is about $3.6\times 10^{-12}$ — gradient updates are numerically zero, and the early layers stop learning.

### 4.2 The Exploding Gradient Problem

Reverse the inequality: if the spectral norms of the weight matrices exceed $1$ and the activation derivatives are not strongly contracting, the same product *grows* exponentially. Numerical overflow follows in a few hundred steps. RNNs are notorious for this because the same weight matrix is multiplied $T$ times in unrolled time.

### 4.3 Reading the Curves

![Vanishing vs. exploding gradients across depth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig6_vanishing_exploding.png)

The figure reproduces the effect numerically: a unit gradient signal is back-propagated through random networks of increasing depth. Sigmoid (Xavier-initialized) decays by orders of magnitude per layer; ReLU with He initialization stays close to the healthy regime; ReLU with no rescaling explodes by similar factors. The y-axis is logarithmic — every grid line is a factor of ten.

### 4.4 Solutions

| Problem | Solution | How it helps |
|---------|----------|--------------|
| Vanishing | ReLU activation | $\sigma'(z) = 1$ for $z>0$, no exponential decay. |
| Vanishing | Residual connections | $\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + F(\mathbf{h}^{(l-1)})$ — the gradient has an *identity* path that bypasses the nonlinearities. |
| Vanishing | Batch normalization | Stabilises activation distributions layer by layer. |
| Exploding | Gradient clipping | Rescale $\mathbf{g}\!\leftarrow\!(c/\|\mathbf{g}\|)\,\mathbf{g}$ when $\|\mathbf{g}\| > c$. |
| Both | Proper initialization | Keep the variance of activations and gradients stable across layers (next section). |

---

## 5. Weight Initialization Strategies

### 5.1 Why Initialization Matters

- **All zeros.** Every neuron in a layer computes the same function and receives the same gradient; the network cannot break symmetry.
- **Too large.** Pre-activations land in the saturating tails of the activation; gradients vanish.
- **Too small.** Activations collapse toward zero; the signal disappears as it propagates forward.

The goal is therefore to *preserve the variance of activations and gradients across layers*. The right scale is determined by the layer's fan-in and the activation function.

### 5.2 Xavier (Glorot) Initialization

Consider a single neuron in layer $l$ with fan-in $n_{\text{in}}$ and fan-out $n_{\text{out}}$:

$$
z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji}\, h_i.
$$

If $w_{ji}$ and $h_i$ are independent with zero mean,

$$
\operatorname{Var}(z_j) = n_{\text{in}}\cdot \operatorname{Var}(w)\cdot \operatorname{Var}(h).
$$

Demanding $\operatorname{Var}(z) = \operatorname{Var}(h)$ on the *forward* pass gives $\operatorname{Var}(w) = 1/n_{\text{in}}$; the same analysis on the *backward* pass yields $\operatorname{Var}(w) = 1/n_{\text{out}}$. Glorot and Bengio (2010) compromise:

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad w \sim \mathcal{U}\!\left(-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right). \tag{8}
$$

Best for **sigmoid and tanh** activations.

### 5.3 He Initialization

For ReLU, half of the units are zeroed out in expectation, halving the variance contribution of $h$. He et al. (2015) therefore scale up by a factor of two:

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}}}, \qquad w \sim \mathcal{N}\!\left(0,\, \tfrac{2}{n_{\text{in}}}\right). \tag{9}
$$

Best for **ReLU and its variants**.

### 5.4 Summary Table

| Activation | Initialization | $\operatorname{Var}(w)$ |
|------------|---------------|--------------------------|
| Sigmoid / Tanh | Xavier | $\dfrac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU | He | $\dfrac{2}{n_{\text{in}}}$ |
| Leaky ReLU ($\alpha$) | He (modified) | $\dfrac{2}{(1 + \alpha^{2})\, n_{\text{in}}}$ |

The middle column of the gradient figure above is a direct experimental confirmation: the green curve (ReLU + He) is the only one that stays near order $1$ regardless of depth.

---

## 6. The Loss Landscape

![Non-convex loss landscape with a SGD trajectory](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/19-Neural-Networks-and-Backpropagation/fig7_loss_landscape.png)

Even a tiny network has a loss surface that is sharply non-convex: two deep basins, a saddle, and a small ridge. The figure traces SGD from a poor initialization. Two qualitative facts deserve emphasis:

1. **Most local minima are good enough.** As networks become wider, the minima of the empirical loss tend to lie in flat basins of comparable depth (Choromanska et al., 2015). The intuition that we must find the *global* minimum is a holdover from convex optimization that does not transfer.
2. **Saddle points dominate critical points in high dimensions.** In $\mathbb{R}^d$, the probability that all $d$ Hessian eigenvalues have the same sign at a random critical point is exponentially small. SGD's noise actively helps escape these saddles (Dauphin et al., 2014).

These two observations explain why simple first-order methods continue to dominate deep learning despite the lack of convexity guarantees.

---

## 7. Exercises

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

*This is Part 19 of the [ML Mathematical Derivations](/tags/Mathematical-Derivations/) series. Next: [Part 20 — Regularization and Model Selection](/en/Machine-Learning-Mathematical-Derivations-20-Regularization-and-Model-Selection/). Previous: [Part 18 — Clustering Algorithms](/en/Machine-Learning-Mathematical-Derivations-18-Clustering-Algorithms/).*
