---
title: "Essence of Linear Algebra (13): Tensors and Multilinear Algebra"
date: 2024-04-25 09:00:00
tags:
  - Linear Algebra
  - tensor decomposition
  - deep learning
  - recommender systems
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 13
  total: 18
lang: en
mathjax: true
description: "From scalars to high-dimensional data cubes -- tensors generalize vectors and matrices to arbitrary dimensions. Learn CP and Tucker decomposition, see how tensors compress neural networks and power recommender systems, and build intuition with NumPy/TensorLy code."
disableNunjucks: true
---

If you've used PyTorch or TensorFlow, you've met the word "tensor" hundreds of times. PyTorch calls every array `torch.Tensor`; TensorFlow puts it in the product name. But what *is* a tensor, and why did frameworks borrow this physics-flavored word for what looks like a multi-dimensional array?

The short answer of this chapter:

> A tensor is the natural generalization of a scalar, vector, and matrix to **arbitrary** dimensions. Everything you know about matrices either lifts cleanly to tensors, or breaks in instructive ways.

We'll start from familiar objects, build up the language (fibers, slices, unfolding), then learn the two workhorse decompositions -- **CP** and **Tucker** -- and see how they compress neural networks and power context-aware recommenders.

> **What you will learn:**
> - Tensor order, shape, fibers, slices, and unfolding
> - Core operations: contraction, outer product, n-mode product
> - CP decomposition, Tucker decomposition, and HOSVD
> - Applications in neural-network compression and recommender systems
>
> **Prerequisites:** Eigendecomposition (Chapter 6), SVD (Chapter 9), matrix norms (Chapter 10)

---

## From Scalars to Tensors: A Natural Generalization

### A number, a row, a table, a cube

Look at how the objects pile up:

- **Scalar** (order 0): a single number. Today's temperature, your bank balance. Magnitude only.
- **Vector** (order 1): a list of numbers. A 2D location $$(x, y)$$, an RGB color $$(r, g, b)$$. Magnitude *and* direction.
- **Matrix** (order 2): a table of numbers. An Excel sheet. A grayscale image where each cell is a pixel intensity.
- **3rd-order tensor**: a "data cube." A color image is exactly this: height $$\times$$ width $$\times$$ 3 color channels.

Each step adds one more axis along which numbers can be indexed. There is nothing magical about stopping at three.

![From scalars to tensors: each step adds one more axis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig1_tensor_hierarchy.png)

The general definition just continues the pattern:

> A **tensor** is a generalization of vectors and matrices to arbitrary dimensions. An $$N$$th-order tensor is an $$N$$-dimensional array, denoted $$\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$$, where $$I_1, I_2, \ldots, I_N$$ are the sizes of each axis.

### Tensors are everywhere

You handle tensors every day, often without naming them:

| Data | Shape | Order | What each axis means |
|---|---|---|---|
| Sentiment score of one sentence | $$(\,)$$ | 0 | a single number |
| One city's weekly temperature | $$(7,)$$ | 1 | 7 days of temperatures |
| Grayscale photo | $$(H, W)$$ | 2 | rows $$\times$$ columns of pixel intensities |
| Color photo | $$(H, W, 3)$$ | 3 | every pixel carries an RGB triple |
| A short video clip | $$(T, H, W, 3)$$ | 4 | $$T$$ frames stacked together |
| A training mini-batch of images | $$(N, 3, H, W)$$ | 4 | $$N$$ images packed for the GPU |
| User--movie--time ratings | $$(U, M, T)$$ | 3 | user $$u$$'s rating of movie $$m$$ at time $$t$$ |

**A concrete example.** Suppose you're building a video recommender. You have 1000 users, 10000 movies, and 52 weeks of preference history. That data lives most naturally as a $$1000 \times 10000 \times 52$$ third-order tensor. Flattening time away into a user--movie matrix throws away exactly the signal you came for.

### Order vs. shape vs. rank

Three words that beginners confuse. Keep them apart:

- **Order** (or "mode"): how many axes there are. A vector is order 1; a $$480 \times 640 \times 3$$ image is order 3.
- **Shape**: the size along each axis. The same image has shape $$(480, 640, 3)$$.
- **Rank**: a measure of complexity, not of dimensions. Defined later in this chapter -- and as you'll see, tensor rank behaves quite differently from matrix rank.

---

## The Internal Structure: Fibers and Slices

To work with a tensor, you need ways to "see into" it. Two essential views are **fibers** and **slices**.

### Fibers: toothpicks through a tensor

Picture a Rubik's cube (a 3rd-order tensor). Push a toothpick straight through it along one direction. The string of small cubes you pierce is a **fiber** -- a 1-D slice through the tensor.

**Definition.** Fix all indices except one; what you get is a vector -- that's a fiber.

For a third-order tensor $$\mathcal{X} \in \mathbb{R}^{I \times J \times K}$$:

- **Mode-1 fiber** $$\mathbf{x}_{:jk}$$: fix $$j$$ and $$k$$, vary the first index -- a vector of length $$I$$
- **Mode-2 fiber** $$\mathbf{x}_{i:k}$$: fix $$i$$ and $$k$$ -- a vector of length $$J$$
- **Mode-3 fiber** $$\mathbf{x}_{ij:}$$: fix $$i$$ and $$j$$ -- a vector of length $$K$$

For an RGB image $$\mathcal{X} \in \mathbb{R}^{H \times W \times 3}$$, the mode-3 fiber $$\mathbf{x}_{ij:}$$ at position $$(i, j)$$ is exactly that pixel's color vector $$(R, G, B)$$.

### Slices: thin sheets through a tensor

Same Rubik's cube, but now slice it with a knife. Each cut gives you a matrix.

**Definition.** Fix one index; what you get is a matrix -- that's a slice.

For $$\mathcal{X} \in \mathbb{R}^{I \times J \times K}$$:

- **Horizontal slice** $$\mathbf{X}_{i::}$$: fix $$i$$, get a $$J \times K$$ matrix
- **Lateral slice** $$\mathbf{X}_{:j:}$$: fix $$j$$, get an $$I \times K$$ matrix
- **Frontal slice** $$\mathbf{X}_{::k}$$: fix $$k$$, get an $$I \times J$$ matrix

For a video $$\mathcal{V} \in \mathbb{R}^{H \times W \times T}$$, the frontal slice $$\mathbf{V}_{::t}$$ is just frame $$t$$. The whole video is a stack of frontal slices.

![A 3D tensor visualized as a stack of frontal slices (matrices)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig2_tensor_as_stack.png)

This stack-of-matrices picture is the right mental model: anywhere you can already work with a matrix, you can lift the operation up by sliding it across the third axis.

### Unfolding (matricization)

Sometimes you want to flatten a tensor into a matrix so existing matrix tools apply. This operation is called **unfolding** or **matricization**.

**Mode-$$n$$ unfolding.** Reshape the tensor so that the mode-$$n$$ fibers become the *columns* of the resulting matrix. For $$\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$$ this gives:

$$\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times (I_1 \cdots I_{n-1} I_{n+1} \cdots I_N)}$$

For $$\mathcal{X} \in \mathbb{R}^{3 \times 4 \times 2}$$ the three unfoldings have shape $$3 \times 8$$, $$4 \times 6$$, and $$2 \times 12$$ respectively.

![Mode-1 unfolding: each mode-1 fiber becomes one column of the resulting matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig6_mode_n_unfolding.png)

Why bother? Because most tensor algorithms reduce some sub-step to "do an SVD on the mode-$$n$$ unfolding," and we already have powerful, stable matrix tools.

---

## Basic Tensor Operations

### Addition and scalar multiplication

These work exactly like vectors and matrices: same-shape tensors add elementwise, and scalar multiplication scales every entry.

$$(\mathcal{X} + \mathcal{Y})_{i_1 \cdots i_N} = x_{i_1 \cdots i_N} + y_{i_1 \cdots i_N}, \qquad (\alpha \mathcal{X})_{i_1 \cdots i_N} = \alpha\, x_{i_1 \cdots i_N}$$

The skip connection in ResNet, $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$, is just tensor addition.

### Tensor contraction (Einstein summation)

Contraction is the single most important tensor operation, and it's just the generalization of matrix multiplication you'd guess at.

**The idea.** Pick one axis from each of two tensors that have the *same length*. Multiply paired entries, sum along that shared axis, and the axis disappears.

For matrix multiplication this is the familiar

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

The shared index $$j$$ is contracted away.

**General contraction.** If $$\mathcal{A} \in \mathbb{R}^{I_1 \times I_2 \times J}$$ and $$\mathcal{B} \in \mathbb{R}^{J \times K_1 \times K_2}$$, contracting along the shared mode $$J$$ gives a 4th-order tensor:

$$\mathcal{C}_{i_1 i_2 k_1 k_2} = \sum_j \mathcal{A}_{i_1 i_2 j}\, \mathcal{B}_{j k_1 k_2}$$

![Tensor contraction: pick a shared index, multiply, sum it away](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig3_contraction_einsum.png)

This is exactly the rule behind `np.einsum` and `torch.einsum`: write the input axes, write the output axes, and any index that appears on the input but not the output is summed away.

### Outer products and rank-1 tensors

The outer product runs in the *opposite* direction: it builds higher-order tensors out of lower-order ones.

**Two vectors** $$\mathbf{a} \circ \mathbf{b} = \mathbf{a} \mathbf{b}^T$$, a matrix with $$(i,j)$$-entry $$a_i b_j$$.

**Three vectors** $$(\mathbf{a} \circ \mathbf{b} \circ \mathbf{c})_{ijk} = a_i b_j c_k$$, a third-order tensor.

A tensor that can be written as a single outer product of vectors is called **rank-1**. It is the simplest possible tensor: its full content is determined by a few one-dimensional pieces.

**Key fact.** *Every* tensor can be written as a sum of rank-1 tensors. That single fact is the theoretical foundation of tensor decomposition -- both CP and Tucker grow out of it.

### n-mode product

The n-mode product applies a matrix to one axis of a tensor.

**Definition.** For $$\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$$ and $$\mathbf{U} \in \mathbb{R}^{J \times I_n}$$:

$$(\mathcal{X} \times_n \mathbf{U})_{i_1 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n} x_{i_1 \cdots i_N}\, u_{j i_n}$$

The result has the same shape as $$\mathcal{X}$$ except mode $$n$$ has size $$J$$ instead of $$I_n$$.

**Matrix view.** In terms of mode-$$n$$ unfolding it's just a matrix multiply: $$\mathbf{Y}_{(n)} = \mathbf{U}\, \mathbf{X}_{(n)}$$.

**Picture.** "Apply the linear transformation $$\mathbf{U}$$ along the $$n$$th axis only." For an RGB image $$\mathcal{I} \in \mathbb{R}^{H \times W \times 3}$$, multiplying by a $$3 \times 3$$ color-correction matrix $$\mathbf{M}$$ as $$\mathcal{I} \times_3 \mathbf{M}$$ does white-balance / tone mapping by linearly mixing each pixel's color triple.

**Useful properties:**

- Different modes commute: $$\mathcal{X} \times_m \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n \mathbf{B} \times_m \mathbf{A}$$ when $$m \neq n$$.
- Same mode composes as matrix product: $$\mathcal{X} \times_n \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n (\mathbf{B}\mathbf{A})$$.

### Kronecker and Khatri-Rao products

These two appear constantly in tensor decompositions; learn to recognise them.

**Kronecker product** $$\mathbf{A} \otimes \mathbf{B}$$. For $$\mathbf{A} \in \mathbb{R}^{I \times J}$$ and $$\mathbf{B} \in \mathbb{R}^{K \times L}$$, the result lives in $$\mathbb{R}^{IK \times JL}$$:

$$\mathbf{A} \otimes \mathbf{B} = \begin{bmatrix} a_{11}\mathbf{B} & \cdots & a_{1J}\mathbf{B} \\ \vdots & \ddots & \vdots \\ a_{I1}\mathbf{B} & \cdots & a_{IJ}\mathbf{B} \end{bmatrix}$$

Think of it as "replace each entry of $$\mathbf{A}$$ by that entry times the whole matrix $$\mathbf{B}$$."

**Khatri-Rao product** $$\mathbf{A} \odot \mathbf{B}$$ (column-wise Kronecker). For $$\mathbf{A} \in \mathbb{R}^{I \times R}$$ and $$\mathbf{B} \in \mathbb{R}^{K \times R}$$ (same number of columns):

$$\mathbf{A} \odot \mathbf{B} = [\mathbf{a}_1 \otimes \mathbf{b}_1, \; \mathbf{a}_2 \otimes \mathbf{b}_2, \; \ldots, \; \mathbf{a}_R \otimes \mathbf{b}_R]$$

The Khatri-Rao product is the algebraic glue holding the CP-ALS update formulas together.

---

## Tensor Norms

### Frobenius norm

Same recipe as for matrices: square every entry, add, square-root.

$$\|\mathcal{X}\|_F = \sqrt{\sum_{i_1, \ldots, i_N} x_{i_1 \cdots i_N}^2}$$

**Useful invariant.** The Frobenius norm is preserved by unfolding: $$\|\mathcal{X}\|_F = \|\mathbf{X}_{(n)}\|_F$$ for any mode $$n$$. So you can compute it on whichever shape is convenient.

### Inner product

For two tensors of the same shape:

$$\langle \mathcal{X}, \mathcal{Y} \rangle = \sum_{i_1, \ldots, i_N} x_{i_1 \cdots i_N}\, y_{i_1 \cdots i_N}, \qquad \|\mathcal{X}\|_F = \sqrt{\langle \mathcal{X}, \mathcal{X} \rangle}$$

---

## Tensors as Multilinear Maps

A more abstract view that pays off: tensors *are* multilinear maps.

**Linear maps** are the familiar $$f(\mathbf{x}) = \mathbf{A}\mathbf{x}$$, satisfying $$f(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha f(\mathbf{x}) + \beta f(\mathbf{y})$$.

**Bilinear maps** take two vector inputs and are linear in each one separately:

$$f(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{A} \mathbf{y}$$

is bilinear because $$f(\alpha \mathbf{x}_1 + \beta \mathbf{x}_2, \mathbf{y}) = \alpha f(\mathbf{x}_1, \mathbf{y}) + \beta f(\mathbf{x}_2, \mathbf{y})$$ and similarly in $$\mathbf{y}$$.

**Multilinear maps** continue the pattern: linear in each of $$N$$ vector inputs.

**Tensor product spaces.** Given vector spaces $$V$$ and $$W$$, their tensor product $$V \otimes W$$ is the unique vector space with the property that every bilinear map $$f : V \times W \to Z$$ factors through it as a linear map $$\tilde{f} : V \otimes W \to Z$$ with $$f(\mathbf{v}, \mathbf{w}) = \tilde{f}(\mathbf{v} \otimes \mathbf{w})$$.

The payoff: tensors let you turn a multilinear problem into a linear one, which means you get to bring all of linear algebra to bear.

---

## CP Decomposition: Tensor as a Sum of Simple Pieces

![CP decomposition: any tensor as a sum of rank-1 outer products](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig4_cp_decomposition.png)

### What is CP decomposition?

**CP decomposition** (CANDECOMP / PARAFAC) writes a tensor as a weighted sum of rank-1 tensors:

$$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \, \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r$$

Collecting the vectors into **factor matrices** $$\mathbf{A} = [\mathbf{a}_1, \ldots, \mathbf{a}_R]$$, $$\mathbf{B} = [\mathbf{b}_1, \ldots, \mathbf{b}_R]$$, $$\mathbf{C} = [\mathbf{c}_1, \ldots, \mathbf{c}_R]$$, the standard shorthand is $$\mathcal{X} \approx [\![\boldsymbol{\lambda}; \mathbf{A}, \mathbf{B}, \mathbf{C}]\!]$$.

### The intuition: superimposing simple patterns

Take the user--movie--time rating tensor. CP decomposition says

$$\text{Rating}(u, m, t) \approx \sum_{r=1}^{R} \lambda_r \cdot \text{user}_r(u) \cdot \text{movie}_r(m) \cdot \text{time}_r(t)$$

and each component $$r$$ is one "simple pattern" you can almost name out loud:

- Component 1: *young users, action movies, weekend evenings*.
- Component 2: *middle-aged users, art films, weekday late nights*.
- ...

Sum a handful of these and you reconstruct most of what users actually do. The decomposition is essentially a soft clustering on three axes at once.

### Tensor rank: where it stops behaving like the matrix kind

The **rank** of a tensor is the smallest number of rank-1 components needed to represent it exactly:

$$\operatorname{rank}(\mathcal{X}) = \min\bigl\{ R : \mathcal{X} = \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r \bigr\}$$

Three things that look like the matrix story but aren't:

- **Computing tensor rank is NP-hard.** SVD gives matrix rank for free; tensors have no analogue.
- **The best low-rank approximation may not exist.** For a matrix, SVD always achieves it. For tensors, the optimum can be approached but never reached -- a phenomenon called *border rank*.
- **Tensor rank can exceed every individual dimension.** Matrix rank is bounded by $$\min(m, n)$$; tensor rank has no such ceiling.

### Why CP is special: essential uniqueness

A real advantage of CP over matrix factorizations: under mild conditions, the decomposition is **essentially unique**. The factors are determined up to a permutation of components and a rescaling of columns -- nothing else.

**Kruskal's condition.** If $$k_{\mathbf{A}} + k_{\mathbf{B}} + k_{\mathbf{C}} \geq 2R + 2$$, where $$k_{\mathbf{A}}$$ is the *Kruskal rank* of $$\mathbf{A}$$ (the largest $$k$$ such that any $$k$$ columns are linearly independent), then CP is essentially unique.

Compare to SVD, where $$\mathbf{U}$$ and $$\mathbf{V}$$ can be rotated by any orthogonal matrix without changing the product. CP doesn't have that ambiguity, which is why CP factors are usable as *interpretable* user / item / time profiles in recommender systems.

### The ALS algorithm

The standard recipe for fitting CP is **Alternating Least Squares (ALS)**: hold all factor matrices fixed except one, and the remaining problem is plain old least squares.

1. Randomly initialize $$\mathbf{A}, \mathbf{B}, \mathbf{C}$$.
2. Update $$\mathbf{A}$$ holding $$\mathbf{B}, \mathbf{C}$$ fixed:
   $$\mathbf{A} \leftarrow \mathbf{X}_{(1)}\, (\mathbf{C} \odot \mathbf{B})\, \bigl[(\mathbf{C}^T \mathbf{C}) * (\mathbf{B}^T \mathbf{B})\bigr]^{\dagger}$$
3. Update $$\mathbf{B}$$ similarly, then $$\mathbf{C}$$.
4. Repeat until the reconstruction error stops shrinking.

Here $$*$$ is the Hadamard (elementwise) product and $$\dagger$$ the Moore--Penrose pseudo-inverse.

ALS doesn't guarantee a global optimum, but in practice it works well -- run it from several random starts and keep the best fit.

```python
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# Simulated user-movie-time rating tensor:
# 100 users, 50 movies, 12 months, ratings in [0, 5]
np.random.seed(42)
n_users, n_movies, n_months = 100, 50, 12
X = np.random.rand(n_users, n_movies, n_months) * 5

# CP decomposition with 5 latent components
weights, factors = parafac(tl.tensor(X), rank=5, n_iter_max=100)
user_factors, movie_factors, time_factors = factors

print(user_factors.shape)   # (100, 5)
print(movie_factors.shape)  # (50, 5)
print(time_factors.shape)   # (12, 5)
```

Each row of `user_factors` is a 5-D embedding of one user; rows of `movie_factors` and `time_factors` play the same role for movies and months. CP simultaneously embeds all three axes into a shared 5-D latent space.

---

## Tucker Decomposition: A More Flexible Form

![Tucker decomposition: a small core tensor plus one factor matrix per mode](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig5_tucker_decomposition.png)

### The definition

**Tucker decomposition** is the more general form:

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}$$

In components:

$$x_{ijk} \approx \sum_{p=1}^{P} \sum_{q=1}^{Q} \sum_{r=1}^{R} g_{pqr}\, a_{ip}\, b_{jq}\, c_{kr}$$

Here $$\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$$ is the **core tensor** -- a small tensor that holds the *interaction structure* -- and $$\mathbf{A}, \mathbf{B}, \mathbf{C}$$ are the factor matrices that map back out to the original sizes.

### Tucker vs. CP

- **CP is a special case of Tucker.** If $$\mathcal{G}$$ is "superdiagonal" (only $$g_{rrr}$$ nonzero), Tucker collapses to CP.
- **Tucker is strictly more flexible.** CP forces all three factor matrices to share the same number of columns $$R$$. Tucker lets each mode have its *own* truncation rank $$(P, Q, R)$$, which is a much better fit when the three axes have very different intrinsic dimensions (a $$1000 \times 1000 \times 12$$ tensor probably wants $$P, Q$$ in the dozens but $$R = 4$$).
- **CP has uniqueness, Tucker doesn't.** Tucker has the same rotational ambiguity as SVD: replace $$\mathbf{A}$$ by $$\mathbf{A}\mathbf{Q}$$ and absorb $$\mathbf{Q}^{-1}$$ into the core, and you get the same approximation. So Tucker compresses well but its factors are not directly interpretable the way CP's are.

### HOSVD: Higher-Order SVD

**HOSVD** is the canonical orthogonal Tucker decomposition. The recipe is delightfully simple:

1. For each mode $$n$$, compute the SVD of the mode-$$n$$ unfolding $$\mathbf{X}_{(n)}$$.
2. Take the top $$R_n$$ left singular vectors as the factor matrix $$\mathbf{U}^{(n)}$$.
3. Form the core $$\mathcal{G} = \mathcal{X} \times_1 \mathbf{U}^{(1)T} \times_2 \mathbf{U}^{(2)T} \times_3 \mathbf{U}^{(3)T}$$.

The analogy with matrix SVD is exact:

| Matrix SVD | HOSVD |
|---|---|
| $$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$ | $$\mathcal{X} = \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}$$ |
| $$\boldsymbol{\Sigma}$$ diagonal | $$\mathcal{G}$$ "all-orthogonal" but not diagonal |
| $$\mathbf{U}, \mathbf{V}$$ orthogonal | every $$\mathbf{U}^{(n)}$$ orthogonal |

HOSVD is workhorse machinery for compression, denoising, and as a warm start for nonlinear Tucker fitting (e.g. HOOI -- Higher-Order Orthogonal Iteration).

```python
from tensorly.decomposition import tucker

# Tucker on the same rating tensor
# Compress 100 users -> 10 dims, 50 movies -> 8, 12 months -> 4
core, factors = tucker(tl.tensor(X), rank=[10, 8, 4])

print(core.shape)                       # (10, 8, 4)
print([f.shape for f in factors])       # [(100, 10), (50, 8), (12, 4)]

original_size   = 100 * 50 * 12
compressed_size = 10 * 8 * 4 + 100 * 10 + 50 * 8 + 12 * 4
print(f"Compression: {original_size / compressed_size:.2f}x")
```

### Multilinear rank

Tucker comes with its own notion of rank, the **multilinear rank**:

$$\operatorname{rank}_{\text{ml}}(\mathcal{X}) = \bigl(\operatorname{rank}(\mathbf{X}_{(1)}),\, \operatorname{rank}(\mathbf{X}_{(2)}),\, \operatorname{rank}(\mathbf{X}_{(3)})\bigr)$$

Each entry is the rank of one mode-$$n$$ unfolding, i.e. how many independent directions the data uses along that axis. Unlike CP rank, multilinear rank is *easy* to compute -- one matrix SVD per mode.

---

## Tensors in Deep Learning

There's a reason the framework is called TensorFlow. Almost every operation in a neural network is a tensor operation in disguise.

### Convolution from a tensor point of view

A standard 2D convolutional layer involves four tensors:

- **Input** $$\mathcal{X} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$$ -- batch, input channels, height, width.
- **Kernel** $$\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W}$$.
- **Output** $$\mathcal{Y} \in \mathbb{R}^{B \times C_{\text{out}} \times H' \times W'}$$.

The convolution operation itself is one big contraction over $$C_{\text{in}}, K_H, K_W$$ together with a sliding window over spatial dimensions -- a structured 4-way contraction.

### Compressing networks with CP

A convolutional layer is basically a 4-way tensor of weights, and "rank" matters here because the *parameter count is the rank's bottleneck*. CP-decompose $$\mathcal{W}$$ as a sum of $$R$$ rank-1 tensors and a single fat convolution becomes a chain of four small ones:

1. $$1 \times 1$$ conv mapping $$C_{\text{in}} \to R$$ channels;
2. $$K_H \times 1$$ depthwise conv on each of the $$R$$ channels;
3. $$1 \times K_W$$ depthwise conv;
4. $$1 \times 1$$ conv mapping $$R \to C_{\text{out}}$$.

Compare parameter counts:

| | Original | After CP-rank-$$R$$ |
|---|---|---|
| Parameters | $$C_{\text{out}} \cdot C_{\text{in}} \cdot K_H \cdot K_W$$ | $$R \cdot (C_{\text{out}} + C_{\text{in}} + K_H + K_W)$$ |

When $$R$$ is small the saving is dramatic. A $$512 \times 512 \times 3 \times 3$$ VGG-16 layer has $$\approx$$ 2.36 M parameters; rank-64 CP brings that to $$\approx$$ 66 K -- about 35$$\times$$ compression, with modest accuracy loss after fine-tuning.

### Compressing networks with Tucker

Tucker is the more popular choice in practice because it lets you compress input and output channels separately, which matches how CNNs are actually structured. The recipe is:

1. $$1 \times 1$$ conv reducing $$C_{\text{in}} \to P$$;
2. small $$K_H \times K_W$$ conv working in the compressed $$P \times Q$$ space;
3. $$1 \times 1$$ conv expanding $$Q \to C_{\text{out}}$$.

This "bottleneck" pattern looks suspiciously like the building block of MobileNets and SqueezeNet -- which is no accident; both can be read as hand-designed Tucker decompositions of a standard conv.

---

## Tensor Decomposition for Recommender Systems

### From matrix factorization to tensor factorization

Classical collaborative filtering uses a **user-item matrix** $$\mathbf{R} \in \mathbb{R}^{U \times M}$$ where $$r_{um}$$ is user $$u$$'s rating of item $$m$$. Matrix factorization -- SVD, NMF, ALS -- finds embeddings

$$\mathbf{R} \approx \mathbf{P} \mathbf{Q}^T,\quad \mathbf{P} \in \mathbb{R}^{U \times K},\; \mathbf{Q} \in \mathbb{R}^{M \times K}$$

so that $$\hat{r}_{um} = \mathbf{p}_u^T \mathbf{q}_m$$.

This loses something important: people's preferences depend on context. You don't want the same movie on a Tuesday morning commute as on a Saturday night.

**The tensor lift.** Add a context axis (time, location, device, mood, ...) to get a third-order tensor $$\mathcal{R} \in \mathbb{R}^{U \times M \times T}$$ and CP-decompose it:

$$r_{umt} \approx \sum_{r=1}^{R} \lambda_r\, p_{ur}\, q_{mr}\, t_{tr}$$

The new factor $$\mathbf{t}_r$$ tells you how component $$r$$ is modulated over time. Thanks to CP's essential uniqueness, you can actually *interpret* what each component captures.

### Sparse data: only fit what you observed

Real rating data is brutally sparse -- often well below 0.1% of entries are observed. So you don't fit the full tensor, you fit only the observed positions:

$$\min_{\mathbf{A}, \mathbf{B}, \mathbf{C}} \sum_{(i,j,k) \in \Omega} \left( x_{ijk} - \sum_{r=1}^{R} a_{ir} b_{jr} c_{kr} \right)^2 + \lambda \bigl(\|\mathbf{A}\|_F^2 + \|\mathbf{B}\|_F^2 + \|\mathbf{C}\|_F^2\bigr)$$

where $$\Omega$$ is the set of observed indices and $$\lambda$$ is a regularization parameter that prevents the embeddings from blowing up.

This is exactly how production context-aware recommenders work -- only with neural extensions on top.

---

## An Application You Use Every Day: Images

Every JPEG you open is a 3rd-order tensor. The picture below shows it explicitly: an $$H \times W \times 3$$ array decomposed into three monochrome "channel" matrices stacked along the depth axis.

![A color image is a 3rd-order tensor: H x W x 3, with one matrix per color channel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/13-tensors-and-multilinear-algebra/fig7_image_as_tensor.png)

Once you see images this way, it's clear why so much of computer vision is just tensor algebra. Color conversion is an n-mode product against a $$3 \times 3$$ matrix. Image compression is low-rank approximation. Convolutional features are tensor contractions. Even the "channels" of a feature map deep inside a CNN are no different in kind from R, G, B -- they're just learned channels instead of physically given ones.

---

## Other Tensor Decompositions

### Tensor Train (TT)

For very high-order tensors -- think quantum many-body wavefunctions or the natural representation of an order-50 tensor with each axis of size 2 -- both CP and Tucker complexity blow up. **Tensor Train** factors the tensor as a chain of small "carriages":

$$\mathcal{X}(i_1, i_2, \ldots, i_N) = \mathbf{G}_1(i_1)\, \mathbf{G}_2(i_2)\, \cdots\, \mathbf{G}_N(i_N)$$

where each $$\mathbf{G}_k(i_k)$$ is a matrix of size $$r_{k-1} \times r_k$$. Wins:

- Parameter count grows as $$O(N\, d\, r^2)$$ -- linear in the order $$N$$, not exponential.
- Stable construction algorithm (TT-SVD) exists.
- This is the working representation in modern many-body quantum simulators (DMRG / MPS).

### Non-negative tensor factorization (NTF)

When entries are physically non-negative -- pixel intensities, word counts, chemical concentrations -- you usually want non-negative factors as well:

$$\mathcal{X} \approx \sum_{r=1}^{R} \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r,\quad \text{with } \mathbf{a}_r, \mathbf{b}_r, \mathbf{c}_r \geq 0$$

The non-negativity constraint forces components to look like *parts* (additive features that build up into the whole) instead of a mix of positive and negative cancellation. This is why NMF / NTF gives interpretable topics and parts-based decompositions while plain SVD often does not.

---

## Exercises

### Conceptual

**Exercise 1.** What is the order of each of the following data structures?

- A mono MP3 song (44.1 kHz)
- A stereo song
- A 5-minute 1080p RGB video at 30 fps
- The ImageNet dataset (1 million $$224 \times 224$$ RGB images)
- A Transformer attention tensor $$\in \mathbb{R}^{B \times H \times L \times L}$$

**Exercise 2.** For $$\mathcal{X} \in \mathbb{R}^{3 \times 4 \times 2}$$:

- How many mode-1 fibers are there, and how long is each?
- How many frontal slices are there, and what is the shape of each?
- What is the shape of the mode-2 unfolding $$\mathbf{X}_{(2)}$$?

### Computation

**Exercise 3.** Let $$\mathbf{a} = [1, 2]^T$$, $$\mathbf{b} = [3, 4, 5]^T$$, $$\mathbf{c} = [6, 7]^T$$.

- Compute the outer product $$\mathbf{a} \circ \mathbf{b}$$.
- Form the third-order tensor $$\mathcal{T} = \mathbf{a} \circ \mathbf{b} \circ \mathbf{c}$$ and write down $$t_{121}$$.
- Compute $$\|\mathcal{T}\|_F$$.

**Exercise 4.** Let $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$, $$\mathbf{B} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$.

- Compute the Kronecker product $$\mathbf{A} \otimes \mathbf{B}$$.
- Verify the identity $$(\mathbf{A} \otimes \mathbf{B})\, \operatorname{vec}(\mathbf{X}) = \operatorname{vec}(\mathbf{B} \mathbf{X} \mathbf{A}^T)$$ for a chosen $$\mathbf{X}$$.

### Decomposition

**Exercise 5.** Consider the rank-2 tensor $$\mathcal{X} = \mathbf{a}_1 \circ \mathbf{b}_1 \circ \mathbf{c}_1 + \mathbf{a}_2 \circ \mathbf{b}_2 \circ \mathbf{c}_2$$ with

$$\mathbf{a}_1 = [1, 0]^T,\; \mathbf{b}_1 = [1, 1, 0]^T,\; \mathbf{c}_1 = [1, 0]^T$$

$$\mathbf{a}_2 = [0, 1]^T,\; \mathbf{b}_2 = [0, 1, 1]^T,\; \mathbf{c}_2 = [0, 1]^T$$

(a) Compute several entries of $$\mathcal{X}$$.
(b) Write the three factor matrices.
(c) Compute the mode-1 unfolding $$\mathbf{X}_{(1)}$$.

**Exercise 6.** For the $$3 \times 3 \times 3$$ identity tensor ($$\delta_{ijk} = 1$$ iff $$i = j = k$$):

- Compute its Frobenius norm.
- What is its CP rank? (Hint: can it be written with fewer than 3 rank-1 terms?)
- What is its multilinear rank?

### Applications

**Exercise 7** (Recommender System). A platform has 3 users, 4 movies, and 2 time periods. Observed ratings:

| User | Movie | Time | Rating |
|---|---|---|---|
| 1 | 1 | 1 | 5 |
| 1 | 2 | 1 | 4 |
| 2 | 1 | 2 | 3 |
| 2 | 3 | 1 | 5 |
| 3 | 2 | 2 | 2 |
| 3 | 4 | 1 | 4 |

- Construct the rating tensor $$\mathcal{R} \in \mathbb{R}^{3 \times 4 \times 2}$$ (treat unobserved as 0).
- For rank-2 CP, how many parameters need to be estimated?
- Discuss the trade-off between rank-1 and rank-2 decomposition.

**Exercise 8** (Image compression). A $$1024 \times 768$$ RGB image is a $$1024 \times 768 \times 3$$ tensor.

- How many bytes for raw storage (1 byte / value)?
- How many bytes after Tucker decomposition with rank $$(100, 75, 3)$$?
- What is the compression ratio?
- What about CP decomposition with rank 100?

### Programming

**Exercise 9.** Implement in Python:

```python
# (a) Mode-n unfolding for a third-order tensor
def unfold(X, mode):
    """X: ndarray (I, J, K); mode in {0, 1, 2}; returns the unfolded matrix."""
    pass

# (b) n-mode product
def mode_n_product(X, A, mode):
    """X: (I, J, K); A: (P, I_mode); returns the product tensor."""
    pass

# (c) CP-ALS
def simple_cp_als(X, rank, n_iter=100):
    """Returns (A, B, C) factor matrices."""
    pass
```

**Exercise 10.** Using the `tensorly` library:

- Build a rank-3 random $$20 \times 15 \times 10$$ tensor (generate factors first, then construct via outer products).
- Run CP decomposition with ranks 2, 3, 4, 5 and compare reconstruction errors.
- Plot reconstruction error vs. rank.
- Apply HOSVD and compare against CP results.

### Proofs

**Exercise 11.** Prove that for any mode $$n$$, $$\|\mathcal{X}\|_F = \|\mathbf{X}_{(n)}\|_F$$.

**Exercise 12.** Prove the n-mode product properties:

(a) $$\mathcal{X} \times_m \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n \mathbf{B} \times_m \mathbf{A}$$ when $$m \neq n$$.
(b) $$\mathcal{X} \times_n \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n (\mathbf{B}\mathbf{A})$$.

**Exercise 13.** For the rank-1 tensor $$\mathcal{X} = \mathbf{a} \circ \mathbf{b} \circ \mathbf{c}$$, prove:

(a) $$\|\mathcal{X}\|_F = \|\mathbf{a}\|\, \|\mathbf{b}\|\, \|\mathbf{c}\|$$.
(b) $$\mathbf{X}_{(1)} = \mathbf{a}\, (\mathbf{c} \otimes \mathbf{b})^T$$.

---

## Chapter Summary

**Concepts.** Tensors generalize scalars, vectors, and matrices to arbitrary order. Fibers, slices, and unfolding are the basic tools for "looking inside" a tensor. The core operations -- addition, scalar multiplication, contraction, outer product, n-mode product -- are all natural lifts of matrix operations.

**Decompositions.** CP writes a tensor as a sum of rank-1 outer products, and is *essentially unique* under mild conditions, which makes its factors interpretable. Tucker is more flexible (different rank per mode) but suffers from rotational ambiguity; HOSVD provides the canonical orthogonal version.

**Beware of "rank".** Tensor rank is NP-hard to compute, the best low-rank approximation may not exist, and tensor rank can exceed every individual axis size -- none of which are true for matrices.

**Applications.** Compressing convolutional layers (CP / Tucker), context-aware recommendation (user--item--time CP), and the natural representation of images / videos / EEG / quantum states. The unifying theme: *decompose complex high-dimensional structure into combinations of simple components.*

---

## References

- Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. *SIAM Review*, 51(3), 455--500.
- Sidiropoulos, N. D., et al. (2017). Tensor decomposition for signal processing and machine learning. *IEEE Transactions on Signal Processing*, 65(13), 3551--3582.
- Cichocki, A., et al. (2015). Tensor decompositions for signal processing applications. *IEEE Signal Processing Magazine*, 32(2), 145--163.
- [TensorLy Documentation](https://tensorly.org/) -- Python tensor learning library.
- Strang, G. (2019). *Linear Algebra and Learning from Data*, Chapter 7.

---

## Series Navigation

- **Previous:** [Chapter 12: Sparse Matrices and Compressed Sensing](/en/chapter-12-sparse-matrices-and-compressed-sensing/)
- **Next:** [Chapter 14: Random Matrix Theory](/en/chapter-14-random-matrix-theory/)
- **Full Series:** Essence of Linear Algebra (1--18)
