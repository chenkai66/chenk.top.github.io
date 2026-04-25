---
title: "Essence of Linear Algebra (16): Linear Algebra in Deep Learning"
date: 2025-04-16 09:00:00
tags:
  - Linear Algebra
  - neural networks
  - deep learning
  - transformers
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 16
  total: 18
lang: en
mathjax: true
description: "Deep learning is large-scale matrix computation. From backpropagation as the chain rule in matrix form, to im2col turning convolutions into GEMM, to attention as soft retrieval via dot products -- see every core DL operation through the lens of linear algebra."
disableNunjucks: true
series_order: 16
---

Strip away the marketing and a deep network is one thing: a long pipeline of matrix multiplications glued together by elementwise nonlinearities. Forward pass, backward pass, convolution, attention, normalization, fine-tuning -- every "trick" is a small twist on the same algebraic theme. Once you see the matrices, the field stops looking like a bag of recipes and starts looking like a single language.

This chapter rebuilds the modern stack from that single language. We follow one signal -- a vector $\mathbf{x}$ -- as it flows through linear layers, gets convolved, gets attended to, gets normalized, and gets adapted by a low-rank update. At each step we name the matrix that does the work and the property of that matrix (rank, conditioning, transpose) that makes the trick succeed.

> **What you will learn**
> - How a neural network IS a chain of matrix multiplications -- and why batching is mandatory on a GPU
> - Backpropagation as the matrix chain rule, with $W^{\top}$ as the universal adjoint
> - Convolution rewritten as a single GEMM via the im2col trick
> - Scaled dot-product attention, decomposed into four matrix steps you can read off the page
> - Why initialization, normalization and residual connections are all spectral-radius arguments in disguise
> - LoRA: parameter-efficient fine-tuning as a low-rank update $\Delta W = BA$
>
> **Prerequisites:** Matrix calculus (Chapter 11), SVD (Chapter 9), and the previous chapter on classical ML (Chapter 15).

---

## 1. The Network Is the Matmul Chain

### 1.1 One neuron, one inner product

A neuron does the simplest thing imaginable: take a weighted sum, add a bias, squash it.

$$h \;=\; \sigma(\mathbf{w}^{\top}\mathbf{x} + b)$$

That is **one inner product plus one nonlinearity**. Stop here and the rest of deep learning is just stacking and broadcasting this primitive.

### 1.2 Stack neurons -- get a matrix

Pack $m$ neurons' weight vectors as the rows of a matrix $\mathbf{W} \in \mathbb{R}^{m \times d}$:

$$\mathbf{h} \;=\; \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

Geometrically, $\mathbf{W}$ is a linear map from a $d$-dimensional input space into an $m$-dimensional feature space; $\sigma$ then bends that space so it is no longer flat. Without $\sigma$ a stack of layers would collapse to a single matrix product -- the nonlinearity is what breaks the closure of matrix multiplication and makes universal approximation possible.

### 1.3 Batch is not optional

GPUs are GEMM machines. A single sample wastes nearly all of their FLOPs. Stack $B$ samples as rows of $\mathbf{X} \in \mathbb{R}^{B \times d}$ and a layer becomes one giant matmul:

$$\mathbf{H} \;=\; \sigma(\mathbf{X}\mathbf{W}^{\top} + \mathbf{1}\mathbf{b}^{\top})$$

Bigger $B$ means bigger matrices means higher arithmetic intensity means happier silicon.

![Neural network as a chain of matrix multiplications](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig1_network_as_matmul.png)

```python
import torch
import torch.nn as nn

linear = nn.Linear(in_features=784, out_features=256)
x_batch = torch.randn(32, 784)   # 32 samples
h_batch = linear(x_batch)        # (32, 256)

print(linear.weight.shape)       # torch.Size([256, 784])
print(linear.bias.shape)         # torch.Size([256])
```

### 1.4 Reading a trained weight matrix

After training, $\mathbf{W}$ is not random noise -- each row is a learned **template** the neuron fires on. Heatmapping the matrix and reshaping each row back to the input geometry gives you a direct picture of what the network has learned.

![Reading a weight matrix: rows are the neurons' filters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig2_weight_heatmap.png)

For an MLP on images you see oriented edges and blobs -- the same primitives Hubel and Wiesel found in V1. For a language model you find feature directions corresponding to syntactic roles. The matrix is interpretable; you just have to look at it.

---

## 2. Backpropagation Is the Matrix Chain Rule

Backprop has a reputation for being mysterious. It isn't. It is the chain rule, written with matrices, with one rule of thumb: **whichever matrix multiplied on the forward pass, its transpose multiplies on the backward pass.**

### 2.1 One layer, four steps

Consider $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, $\mathbf{h} = \sigma(\mathbf{z})$, with some scalar loss $L$ downstream.

1. **Receive** the upstream gradient $\partial L/\partial \mathbf{h}$ from the next layer.
2. **Push through the activation** (Hadamard product with the elementwise derivative):

$$\frac{\partial L}{\partial \mathbf{z}} \;=\; \frac{\partial L}{\partial \mathbf{h}} \odot \sigma'(\mathbf{z})$$

3. **Compute parameter gradients** (outer products):

$$\frac{\partial L}{\partial \mathbf{W}} \;=\; \frac{\partial L}{\partial \mathbf{z}}\,\mathbf{x}^{\top}, \qquad \frac{\partial L}{\partial \mathbf{b}} \;=\; \frac{\partial L}{\partial \mathbf{z}}$$

4. **Pass to the previous layer** (transposed map):

$$\frac{\partial L}{\partial \mathbf{x}} \;=\; \mathbf{W}^{\top}\,\frac{\partial L}{\partial \mathbf{z}}$$

Why $\mathbf{W}^{\top}$? Because $\mathbf{W}$ pushes $\mathbf{x}$ forward; its transpose -- the **adjoint** -- pulls gradients back. This is exactly the duality theorem for linear maps, dressed up in calculus notation.

### 2.2 Batched form

For a batch $\mathbf{X} \in \mathbb{R}^{B \times d}$ with post-activation gradient $\boldsymbol{\Delta}$:

$$\frac{\partial L}{\partial \mathbf{W}} \;=\; \boldsymbol{\Delta}^{\top}\mathbf{X}, \qquad \frac{\partial L}{\partial \mathbf{b}} \;=\; \boldsymbol{\Delta}^{\top}\mathbf{1}, \qquad \frac{\partial L}{\partial \mathbf{X}} \;=\; \boldsymbol{\Delta}\,\mathbf{W}$$

Notice the parameter gradient is a **sum of outer products** -- contributions from every sample, all packaged in a single matmul.

```python
import torch
import torch.nn.functional as F

class ManualLinear:
    def __init__(self, in_features, out_features):
        scale = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.randn(out_features, in_features) * scale
        self.b = torch.zeros(out_features)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W.T + self.b
        return F.relu(self.z)

    def backward(self, grad_h):
        grad_z = grad_h * (self.z > 0).float()   # ReLU derivative
        self.grad_W = grad_z.T @ self.x
        self.grad_b = grad_z.sum(dim=0)
        return grad_z @ self.W                    # to previous layer
```

### 2.3 The Jacobian view

For any $\mathbf{y} = f(\mathbf{x})$ the Jacobian $\mathbf{J}_{ij} = \partial y_i / \partial x_j$ is the local linear approximation. The chain rule then reads as a product of Jacobians:

$$\nabla_{\!\mathbf{x}} L \;=\; \mathbf{J}^{\top}\,\nabla_{\!\mathbf{y}} L$$

For a linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$ the Jacobian is literally $\mathbf{W}$. For a deep network it is $\mathbf{J}_L \mathbf{J}_{L-1} \cdots \mathbf{J}_1$, and the gradient norm is bounded by the product of the operator norms. That single observation explains every "vanishing/exploding gradient" pathology you have ever read about.

---

## 3. Convolution Is Just GEMM in Disguise

### 3.1 One dimension: a Toeplitz matrix

A 1D convolution with kernel $\mathbf{w} = [w_0, w_1, w_2]$ acting on a length-5 input is the same as multiplying by a banded **Toeplitz matrix**:

$$\mathbf{T} \;=\; \begin{bmatrix} w_2 & w_1 & w_0 & 0 & 0 \\ 0 & w_2 & w_1 & w_0 & 0 \\ 0 & 0 & w_2 & w_1 & w_0 \end{bmatrix}, \qquad \mathbf{y} = \mathbf{T}\mathbf{x}$$

So convolution was always a matrix product. The only reason we don't write it that way is that $\mathbf{T}$ is enormous and almost entirely zero.

### 3.2 Two dimensions: im2col

In 2D, frameworks pull the same trick using **im2col**. The idea is to **unfold** every receptive field into a column, stack them, and convert the whole convolution into a single dense matmul that BLAS adores.

![im2col: turning convolution into one matrix multiplication](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig3_im2col.png)

The recipe:

1. **Unfold input.** For each output position, take the corresponding input patch ($C_{\rm in} \times K \times K$ values), flatten it, drop it into one column of $\mathbf{X}_{\rm col}$.
2. **Flatten kernel** into one row $\mathbf{w}_{\rm row}$.
3. **Multiply**: $\mathbf{Y}_{\rm flat} = \mathbf{w}_{\rm row}\,\mathbf{X}_{\rm col}$.
4. **Reshape** the result back to the spatial output.

Yes, im2col duplicates input values (each pixel appears in $K^2$ patches). You pay in memory and you win in throughput, because cuBLAS's GEMM is hand-tuned to feed every tensor core in the SM. On modern GPUs this trade is almost always worth it.

```python
import torch
import torch.nn.functional as F

def conv2d_via_im2col(x, weight, stride=1, padding=0):
    """2D convolution implemented as a single GEMM."""
    B = x.shape[0]
    C_out, C_in, kh, kw = weight.shape
    if padding > 0:
        x = F.pad(x, [padding] * 4)
    _, _, h_pad, w_pad = x.shape
    out_h = (h_pad - kh) // stride + 1
    out_w = (w_pad - kw) // stride + 1

    col = torch.zeros(B, C_in * kh * kw, out_h * out_w)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            col[:, :, i*out_w + j] = patch.reshape(B, -1)

    weight_col = weight.reshape(C_out, -1)
    out = weight_col @ col
    return out.reshape(B, C_out, out_h, out_w)

x = torch.randn(2, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print((conv2d_via_im2col(x, w, padding=1) - F.conv2d(x, w, padding=1))
      .abs().max().item())   # ~1e-6
```

### 3.3 Depthwise separable convolution = low-rank factorization

A standard $K\times K$ convolution costs $C_{\rm out}\,C_{\rm in}\,K^2$ parameters. **Depthwise-separable** convolution factors that tensor into two cheaper pieces:

- **Depthwise:** each input channel convolved independently -- $C_{\rm in}\,K^2$ params.
- **Pointwise ($1\times 1$):** mixes channels -- $C_{\rm out}\,C_{\rm in}$ params.

This is a **low-rank decomposition of the convolution weight tensor**. MobileNet, EfficientNet, ConvNeXt and every modern mobile vision model leans on it.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k,
                                   padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

print(sum(p.numel() for p in nn.Conv2d(64, 128, 3, padding=1).parameters()))
print(sum(p.numel() for p in DepthwiseSeparableConv(64, 128).parameters()))
# 73,856   vs   8,896  -> ~8x fewer params
```

---

## 4. Attention Is a Soft Lookup -- Done with Three Matmuls

### 4.1 The library metaphor

You walk into a library with a question. Each book has a **key** (its keywords) and a **value** (its content). You compare your **query** against every key, normalise the scores into weights, and return a weighted blend of the values.

That is the entire mechanism. The clever part is that all of it is matmuls.

### 4.2 Scaled dot-product attention, four steps

$$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \;=\; \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

Read the four panels left to right:

![Scaled dot-product attention as four matrix steps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig4_attention.png)

1. $\mathbf{Q}\mathbf{K}^{\top}$ -- an $n\times n$ matrix of all pairwise query-key dot products.
2. Divide by $\sqrt{d_k}$. If each entry of $\mathbf{Q},\mathbf{K}$ is i.i.d. $\mathcal{N}(0,1)$, the dot product has variance $d_k$. Without the scale, large $d_k$ pushes softmax into saturation and the gradient dies. The $\sqrt{d_k}$ keeps the variance at 1, where softmax is well-conditioned.
3. Row-wise softmax turns scores into a probability distribution -- the **attention weights**.
4. Multiply by $\mathbf{V}$. Each output row is a convex combination of all value vectors, weighted by relevance.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (B, h, n, d_k)"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

### 4.3 Multi-head: many subspaces in parallel

One attention head learns one notion of "relevance". Real language wants many -- syntactic, semantic, positional. Multi-head attention runs $h$ heads in parallel, each in a $d_k = d_{\rm model}/h$ subspace, then mixes them.

$$\mathrm{MultiHead}(\mathbf{X}) = \mathrm{Concat}(\text{head}_1, \ldots, \text{head}_h)\,\mathbf{W}^O$$

$$\text{head}_i = \mathrm{Attention}(\mathbf{X}\mathbf{W}_i^Q,\,\mathbf{X}\mathbf{W}_i^K,\,\mathbf{X}\mathbf{W}_i^V)$$

The projection matrices $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ carve up the $d_{\rm model}$-dimensional embedding into $h$ disjoint subspaces; $\mathbf{W}^O$ glues the head outputs back together.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out), w
```

The cost of vanilla attention is $O(n^2 d_k)$ in time and $O(n^2)$ in memory -- the $n\times n$ score matrix is the bottleneck for long sequences. FlashAttention re-tiles the computation so that matrix never lives in HBM; mathematically it is identical, just kinder to the memory hierarchy.

---

## 5. Putting It Together: A Transformer Block

A Transformer encoder layer is just four ingredients in a fixed pattern.

- **Multi-head self-attention** (Section 4).
- **Position-wise FFN.** A two-layer MLP, applied independently at each token position, that expands and re-projects:

$$\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

- **Residual connections.** Every sublayer outputs $\mathbf{x} + \mathrm{sublayer}(\mathbf{x})$. The Jacobian becomes $\mathbf{I} + \mathbf{J}$, with eigenvalues clustered near 1 -- gradients always have an unobstructed shortcut backwards.
- **Layer normalization** (Section 6).

The decoder adds **cross-attention** (queries from the decoder, keys/values from the encoder) and a **causal mask** that zeros out scores from the future:

```python
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

Because there is no recurrence, the model needs **positional encoding** to know where each token sits. Sinusoidal encodings,

$$\mathrm{PE}_{(\mathrm{pos}, 2i)} = \sin(\mathrm{pos}/10000^{2i/d}), \qquad \mathrm{PE}_{(\mathrm{pos}, 2i+1)} = \cos(\mathrm{pos}/10000^{2i/d})$$

have the elegant property that $\mathrm{PE}_{\mathrm{pos}+k}$ is a **linear function** of $\mathrm{PE}_{\mathrm{pos}}$ -- relative position is encoded as a fixed rotation in feature space.

---

## 6. Normalization: Standardize Along Different Axes

Activations drift during training. Normalization layers pull them back to a known distribution at every forward pass.

### 6.1 BatchNorm vs LayerNorm

Both apply $\hat{x} = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$ followed by a learnable affine $\gamma\hat{x} + \beta$. The difference is **which axis the mean and variance are computed over**:

- **BatchNorm:** for each feature, average over the batch dimension. Ties samples together; estimate quality depends on batch size; needs running averages at inference.
- **LayerNorm:** for each sample, average over the feature dimension. Sample-independent, batch-size-independent, trivially parallel -- which is why every Transformer uses it.

Reading the matrix mental model: BatchNorm standardises **columns** of the activation matrix, LayerNorm standardises **rows**.

### 6.2 RMSNorm: drop the mean

LLaMA-style models simplify further. Skip the mean subtraction and just rescale by the root-mean-square:

$$\hat{x} = \frac{x}{\mathrm{RMS}(x)} \cdot \gamma, \qquad \mathrm{RMS}(x) = \sqrt{\tfrac{1}{d}\sum_i x_i^2}$$

Roughly half the FLOPs of LayerNorm, comparable downstream quality. Modern LLMs adopt it almost universally.

---

## 7. Initialization, Conditioning, and Gradient Flow

Every issue with training depth ultimately reduces to one quantity: **the singular values of the layerwise Jacobian, multiplied together**.

### 7.1 The variance-preservation principle

Want signals to neither explode nor vanish through $L$ layers? Then every layer should approximately preserve variance. For a linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$ with i.i.d. zero-mean inputs and weights, $\mathrm{Var}(y_i) = n_{\rm in}\,\mathrm{Var}(W_{ij})\,\mathrm{Var}(x)$. Setting this equal to $\mathrm{Var}(x)$ gives the initialization rules.

- **Xavier (Glorot)**, designed for $\tanh$/sigmoid:

$$w_{ij} \sim \mathcal{U}\!\left[-\sqrt{\tfrac{6}{n_{\rm in} + n_{\rm out}}},\;\sqrt{\tfrac{6}{n_{\rm in} + n_{\rm out}}}\right]$$

- **He (Kaiming)**, designed for ReLU (which kills half the units, halving the variance, so we double the scale):

$$w_{ij} \sim \mathcal{N}\!\left(0,\,\tfrac{2}{n_{\rm in}}\right)$$

### 7.2 What the spectrum says

The product $\mathbf{W}_L \mathbf{W}_{L-1} \cdots \mathbf{W}_1$ has top singular value roughly $\prod_\ell \sigma_{\max}(\mathbf{W}_\ell)$. If each factor has $\sigma_{\max} > 1$ the product blows up; if each has $\sigma_{\max} < 1$ it crashes to zero. He / Xavier are calibrated so each factor sits around 1.

![Why initialization matters: singular value distributions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig6_init_eigenvalues.png)

The right panel shows the catastrophe directly: naive $\mathcal{N}(0,1)$ init explodes by orders of magnitude per layer, while He init keeps the top singular value of the product around $O(1)$ no matter the depth.

### 7.3 The same story, in the gradient

Run a forward and backward pass through a 6-layer linear network with three different initializations:

![Gradient flow through a deep network](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig5_backprop_flow.png)

Naive init explodes. Tiny init vanishes. Only He init keeps both the activations and the gradients on a flat trajectory across layers. **Modern tricks for training depth** -- residual connections, careful init, normalization layers, gradient clipping -- are all variations on a single theme: keep the per-layer Jacobian's spectral radius near 1.

---

## 8. LoRA: Fine-Tuning as a Low-Rank Update

Frontier LLMs have hundreds of billions of parameters. Full fine-tuning is wasteful in compute, memory, and storage (one full copy of the weights per task). **LoRA** observes that the *update* you actually need is often very low-rank, even though the base weights are not.

### 8.1 The formula

Freeze the pretrained $\mathbf{W}_0$. Learn an additive update factored into two skinny matrices:

$$\mathbf{W}' \;=\; \mathbf{W}_0 + \Delta\mathbf{W}, \qquad \Delta\mathbf{W} = \mathbf{B}\mathbf{A}$$

with $\mathbf{A} \in \mathbb{R}^{r \times d_{\rm in}}$, $\mathbf{B} \in \mathbb{R}^{d_{\rm out} \times r}$, and $r \ll \min(d_{\rm in}, d_{\rm out})$.

| | parameters |
| --- | --- |
| Full fine-tuning of one layer | $d_{\rm in}\,d_{\rm out}$ |
| LoRA (rank $r$) | $r\,(d_{\rm in} + d_{\rm out})$ |

For $d_{\rm in} = d_{\rm out} = 4096$ and $r = 8$: 16.8M parameters become 65K -- **a 256x reduction**.

![LoRA: low-rank decomposition of the weight update](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/16-linear-algebra-in-deep-learning/fig7_lora_decomposition.png)

The top row visualises the factorisation: a fat $\Delta\mathbf{W}$ on the left equals a tall $\mathbf{B}$ times a wide $\mathbf{A}$ on the right. The bottom-left chart shows the parameter savings; the bottom-right shows reconstruction error collapsing to zero once $r$ reaches the true intrinsic rank of the update.

### 8.2 Why low-rank works

Empirically, the change in weights induced by fine-tuning lives in a low-dimensional subspace -- the "intrinsic rank" hypothesis of Aghajanyan et al. and Hu et al. By fixing $\mathrm{rank}(\Delta\mathbf{W}) \le r$ a priori, LoRA both saves parameters and acts as **structural regularization**: you can only move along $r$ directions, which prevents the catastrophic forgetting that full fine-tuning often suffers from.

```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))   # zero init -> identity
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def merge(self):
        """Fold B A into the base weight - zero overhead at inference."""
        self.base.weight.data += (self.B @ self.A) * self.scaling
```

### 8.3 The LoRA family

- **QLoRA.** Store $\mathbf{W}_0$ in 4-bit NF4, train LoRA in BF16. Fine-tune a 65B model on a single 48GB GPU.
- **DoRA.** Decompose each weight column into magnitude $m$ and direction $\hat{v}$; LoRA-adapt the direction, learn the magnitude separately.
- **AdaLoRA.** Allocate the rank budget unevenly across layers using importance scores from sensitivity analysis.

All of them are linear-algebra moves: LoRA itself is rank constraint, DoRA is polar decomposition, AdaLoRA is rank allocation. Same algebra, different knobs.

---

## 9. The Big Picture

Lay the chapter end to end:

| Operation | Linear-algebra primitive | Why it works |
| --- | --- | --- |
| Fully connected layer | matrix-vector product | linear map + nonlinearity |
| Backpropagation | adjoint ($W^{\top}$) | chain rule = product of Jacobians |
| Convolution (im2col) | block-Toeplitz $\to$ GEMM | trade memory for cuBLAS throughput |
| Depthwise-separable conv | low-rank tensor factorization | drop FLOPs and params |
| Attention | $\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$ | content-addressable lookup |
| Multi-head | direct sum of $h$ subspaces | parallel relevance patterns |
| Residual connection | $\mathbf{I} + \mathbf{J}$ | spectrum centred at 1 |
| BatchNorm / LayerNorm / RMSNorm | row vs column standardization | control activation scale |
| Xavier / He init | variance preservation | $\sigma_{\max}$ near 1 per layer |
| LoRA | $\Delta W = BA$, $\mathrm{rank} \le r$ | exploit low intrinsic dimension |

The same handful of ideas -- linear maps, transposes, ranks, singular values -- explain everything modern deep learning does. The architectures that come and go are reshufflings of the same primitives.

---

## Exercises

### Warm-up

1. Show that if $\mathbf{W}$ is orthogonal ($\mathbf{W}^{\top}\mathbf{W} = \mathbf{I}$), the linear layer $\mathbf{h} = \mathbf{W}\mathbf{x}$ preserves the $\ell_2$ norm both forward and backward.
2. Input $\mathbb{R}^{100}$ feeds an MLP $100 \to 256 \to 128 \to 10$. Write each weight matrix's shape and the total parameter count (don't forget biases).
3. Why does multi-head attention use $d_k = d_{\rm model}/h$ instead of full $d_{\rm model}$ per head? Compare parameter counts and expressive power.
4. For a convolutional feature map of shape $(B, C, H, W)$, list the dimensions BatchNorm and LayerNorm normalise over.

### Deeper

5. Suppose every entry of $\mathbf{Q}$ and $\mathbf{K}$ is i.i.d. $\mathcal{N}(0,1)$. Compute the mean and variance of an entry of $\mathbf{Q}\mathbf{K}^{\top}$, and use the result to justify the $\sqrt{d_k}$ scaling.
6. For an im2col implementation with input $(1, 3, 8, 8)$, kernel $(16, 3, 3, 3)$, stride 1, padding 1: compute the shape of $\mathbf{X}_{\rm col}$ and the memory blow-up factor over the original input.
7. Prove that $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ with $\mathbf{B}\in\mathbb{R}^{m\times r}$, $\mathbf{A}\in\mathbb{R}^{r\times n}$ has $\mathrm{rank}(\Delta\mathbf{W}) \le r$.
8. Analyse the gradient flow through a ResNet block $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$. Show that there is always a "shortcut" path along which the gradient flows without passing through $F$.

### Code

9. Implement a complete Transformer encoder (multi-layer) and use it for a toy sequence-classification task.
10. Apply LoRA to a small pretrained model. Report parameter counts, training memory, and downstream accuracy versus full fine-tuning.
11. Generate a random batch and visualise the activation distribution before and after BatchNorm, LayerNorm, and RMSNorm.
12. For BERT-tiny, profile FLOPs by component (attention, FFN, normalization). Plot total compute as a function of sequence length.

### Open-ended

13. Why do Transformers universally prefer LayerNorm over BatchNorm? Argue from training stability, variable sequence length, and parallelism.
14. LoRA's premise is that fine-tuning updates are low-rank. Sketch a setting where this assumption fails, and design an experiment to detect it.
15. Migrating 2D CNNs to 3D data (video, MRI): how do FLOPs and parameter counts scale, and what is the matrix-algebra reason?

---

## Chapter Summary

- Every neural network layer is a matrix multiplication wrapped in a nonlinearity. Batches turn this into one big GEMM, the operation GPUs were born to run.
- Backpropagation is the matrix chain rule. The forward map's transpose is the backward map -- this is the adjoint duality, full stop.
- Convolutions become GEMMs via im2col. Depthwise-separable convolutions are a low-rank factorization of the convolution tensor.
- Attention is a soft lookup: $\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$. Multi-head attention parallelises it across subspaces.
- Initialization, normalization, and residual connections all serve one purpose: keep the per-layer Jacobian's spectral radius near 1.
- LoRA exploits the empirically low intrinsic rank of fine-tuning updates: $\Delta W = BA$ with $r \ll \min(d_{\rm in}, d_{\rm out})$.

Master these primitives and the next architecture won't look new -- it will look like a remix.

---

## References

- Vaswani, A. et al. **"Attention Is All You Need."** NeurIPS 2017.
- Hu, E. et al. **"LoRA: Low-Rank Adaptation of Large Language Models."** ICLR 2022.
- Aghajanyan, A. et al. **"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning."** ACL 2021.
- Ba, J., Kiros, J., Hinton, G. **"Layer Normalization."** arXiv 2016.
- Zhang, B., Sennrich, R. **"Root Mean Square Layer Normalization."** NeurIPS 2019.
- Ioffe, S., Szegedy, C. **"Batch Normalization."** ICML 2015.
- He, K. et al. **"Deep Residual Learning for Image Recognition."** CVPR 2016.
- He, K. et al. **"Delving Deep into Rectifiers."** ICCV 2015. (He init)
- Glorot, X., Bengio, Y. **"Understanding the Difficulty of Training Deep Feedforward Neural Networks."** AISTATS 2010. (Xavier init)
- Dao, T. et al. **"FlashAttention."** NeurIPS 2022.
- Howard, A. et al. **"MobileNets."** arXiv 2017. (depthwise-separable convolution)

---

## Series Navigation

- **Previous:** [Chapter 15: Linear Algebra in Machine Learning](/en/chapter-15-linear-algebra-in-machine-learning/)
- **Next:** [Chapter 17: Linear Algebra in Computer Vision](/en/chapter-17-linear-algebra-in-computer-vision/)
- **Full Series:** Essence of Linear Algebra (1--18)
