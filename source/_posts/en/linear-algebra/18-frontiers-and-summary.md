---
title: "Essence of Linear Algebra (18): Frontiers and Summary"
date: 2025-04-30 09:00:00
tags:
  - Linear Algebra
  - quantum computing
  - graph neural networks
  - large language models
  - tensor networks
  - topological data analysis
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 18
  total: 18
lang: en
mathjax: true
description: "Series finale: quantum gates as unitary matrices, graph convolution as Laplacian filtering, attention as soft retrieval, LoRA as low-rank adaptation, tensor networks, the matrix exponential, free probability, and a topological data analysis preview -- plus the 18-chapter dependency graph and the geometry / numerics / computation triangle that runs through everything."
disableNunjucks: true
series_order: 18
---

We have walked the long road of linear algebra together. We started with arrows in the plane and ended at the gates of quantum computers, the inner workings of large language models, and the topology of data clouds. The remarkable thing -- the thing this series has tried to make visible -- is that the same handful of ideas keeps coming back. A vector is a state. A matrix is a transformation. A decomposition is the structure hiding inside the transformation. A norm tells you when you can trust your computation. Once you internalise that loop, every "frontier" looks less like a foreign country and more like another dialect of a language you already speak.

This last chapter does two things. First, it walks you through the frontiers -- quantum information, graph neural networks, large models, tensor networks, randomised numerical linear algebra, the matrix exponential as a bridge to Lie theory, free probability, and topological data analysis -- and points to the linear-algebraic skeleton inside each. Second, it steps back and gives you the whole eighteen-chapter map, the recurring themes, the most important theorems, and a path forward.

> **What you will leave with**
> - The unitary picture of quantum computation: qubits as unit vectors, gates as unitaries, entanglement from CNOT.
> - Why a graph Laplacian is the Fourier basis of a network, and how GCN is its first-order Chebyshev filter.
> - Transformer mathematics distilled: attention as soft retrieval, RoPE as complex rotation, LoRA as low-rank adaptation.
> - Sparse and linear attention, quantisation, pruning -- the same matrix story, told under a memory budget.
> - Tensor networks, randomised SVD, NeRF, PINNs, Neural ODEs as continuations of earlier chapters.
> - A complete eighteen-chapter map, the recurring "geometry / numerics / computation" triangle, and reading lists.
>
> **Prerequisites:** comfort with the whole series, especially eigendecomposition (ch. 6), SVD (ch. 9), tensors (ch. 13), random matrices (ch. 14), and the deep learning chapter (ch. 16).

---

## The 18-chapter dependency graph

![Linear algebra: an 18-chapter dependency graph](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig1_concept_map.png)

Before we look forward, look back. The figure above is the actual dependency graph of the series: foundations in blue (vectors, vector spaces, linear maps), structural results in purple (determinants, linear systems, eigenvalues, orthogonality), the two great decompositions in green (the spectral theorem and the SVD), the computational layer in amber (norms and conditioning, matrix calculus, sparsity, tensors, random matrices), the application chapters in red (machine learning, deep learning, computer vision), and this finale chapter in dark.

Notice two things. First, the graph is not a chain -- it is a thin layered network in which several earlier chapters fan into each later chapter. SVD (ch. 9) alone feeds chapters 10, 13, 14, 15, 16, 17, and 18. That is not an accident; SVD is the single most useful theorem in applied linear algebra. Second, the frontier chapter does not introduce new mathematics out of nowhere -- it just reuses the ideas you already know on bigger objects.

---

## Quantum computing: linear algebra at the smallest scale

![Quantum computing in linear algebra: state, gate, entanglement](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig2_quantum_computing.png)

### Qubits as unit vectors

A classical bit is either $|0\rangle$ or $|1\rangle$. A qubit is a unit vector in $\mathbb{C}^2$,

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1,
$$

with computational basis $|0\rangle = \begin{bmatrix}1\\0\end{bmatrix}$ and $|1\rangle = \begin{bmatrix}0\\1\end{bmatrix}$. The Bloch sphere on the left of the figure is the geometric picture: the north pole is $|0\rangle$, the south pole is $|1\rangle$, and every other point on the sphere is a legitimate quantum state. Tensoring $n$ qubits gives a unit vector in $\mathbb{C}^{2^n}$, and *that* is the vector space in which quantum algorithms operate.

### Gates as unitary matrices

A gate is a linear map that preserves the unit-norm condition. That is exactly the definition of a **unitary matrix**: $\mathbf{U}^{\dagger}\mathbf{U} = \mathbf{I}$. Unitarity preserves inner products, which preserves probability -- the linear-algebraic origin of physical reversibility.

The Hadamard gate creates equal superposition,

$$
\mathbf{H} = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}, \qquad \mathbf{H}|0\rangle = \tfrac{1}{\sqrt{2}}(|0\rangle + |1\rangle),
$$

which the middle panel of the figure shows as a bar chart: a probability-one state at $|0\rangle$ becomes amplitude $1/\sqrt{2}$ on each basis state. The Pauli matrices

$$
\mathbf{X} = \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix},\quad \mathbf{Y} = \begin{bmatrix}0 & -i\\ i & 0\end{bmatrix},\quad \mathbf{Z} = \begin{bmatrix}1 & 0\\ 0 & -1\end{bmatrix}
$$

are the three basic rotations -- and an arbitrary single-qubit gate is the matrix exponential $e^{-i\theta(\mathbf{n}\cdot\boldsymbol{\sigma})/2}$, which we will revisit in the Lie-algebra section.

The two-qubit CNOT gate

$$
\text{CNOT} = \begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{bmatrix}
$$

is the engine of entanglement. Apply Hadamard to the first qubit, then CNOT, and an unentangled $|00\rangle$ becomes the **Bell state**

$$
|\Phi^+\rangle = \tfrac{1}{\sqrt{2}}(|00\rangle + |11\rangle),
$$

shown in the right panel of the figure as the amplitude vector after each gate. No tensor product of single-qubit states equals $|\Phi^+\rangle$ -- entanglement is a property of multi-qubit vector spaces that has no classical analogue.

### Two emblematic algorithms

**Grover's search.** Find a marked item among $N$ in $O(\sqrt{N})$ queries instead of the classical $O(N)$. The whole algorithm is two reflections: the oracle flips the sign of the marked basis state, then the diffusion operator $2|\psi\rangle\langle\psi| - \mathbf{I}$ reflects across the uniform superposition. Two reflections compose to a rotation, and after $O(\sqrt{N})$ rotations the amplitude has rotated to the marked state. This is the orthogonal-matrix story from chapter 7, in $\mathbb{C}^N$.

**Shor's algorithm.** Factor an integer in polynomial time using the **Quantum Fourier Transform**. The QFT is the same DFT matrix you have met before, applied to the amplitude vector with $O(n^2)$ gates instead of $O(n 2^n)$ scalar multiplications -- the exponential speedup is what threatens RSA.

---

## Graph neural networks: linear algebra on networks

### Three matrices for one graph

A graph $G = (V, E)$ supports three matrices that you will see everywhere: the adjacency matrix $\mathbf{A}$ ($A_{ij}=1$ if $i \sim j$), the diagonal degree matrix $\mathbf{D}$, and the **graph Laplacian** $\mathbf{L} = \mathbf{D} - \mathbf{A}$. The Laplacian is positive semi-definite, has $\mathbf{1}$ as a zero eigenvector, the multiplicity of zero counts connected components, and

$$
\mathbf{x}^{T}\mathbf{L}\mathbf{x} = \sum_{(i,j)\in E}(x_i - x_j)^2
$$

is a *smoothness measure* of the signal $\mathbf{x}$ on the graph. The normalised Laplacian $\tilde{\mathbf{L}} = \mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}$ has eigenvalues in $[0,2]$.

### A Fourier transform on a graph

Eigendecomposing $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{T}$ gives a basis $\mathbf{U}$ in which "frequency" makes sense: small eigenvalues correspond to slowly varying eigenvectors (close-by nodes have close-by values), large eigenvalues to oscillatory ones. The **graph Fourier transform** is $\hat{\mathbf{x}} = \mathbf{U}^{T}\mathbf{x}$, and a spectral filter is just elementwise multiplication in this basis. Spectral clustering -- embedding nodes using the bottom non-trivial eigenvectors and then running k-means -- is the same idea: cluster nodes that look the same to the low-frequency basis.

### From spectral filters to GCN

Spectral convolution costs $O(n^3)$ because of the eigendecomposition. ChebNet replaces the filter by a degree-$K$ Chebyshev polynomial in $\mathbf{L}$, which only needs $K$-hop neighbours and costs $O(K|E|)$. Take $K = 1$ and a careful renormalisation, and you obtain the GCN layer

$$
\mathbf{H}' = \sigma\!\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\,\mathbf{H}\,\mathbf{W}\right),
$$

with $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ adding a self-loop. Reading the layer from right to left it is "linear transform $\mathbf{W}$, aggregate normalised neighbour features, apply nonlinearity" -- a one-line message-passing scheme that powers everything from molecular property prediction (atoms as nodes, bonds as edges) to recommender systems (user-item bipartite graphs) to the structural side of AlphaFold.

---

## Large language models: attention is matrix multiplication wearing a hat

### Self-attention as soft retrieval

Self-attention -- the inner loop of every modern Transformer -- is

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_k}}\right)\mathbf{V}.
$$

The $n \times n$ matrix $\mathbf{Q}\mathbf{K}^{T}$ holds every pairwise similarity between tokens. The softmax turns each row into a probability distribution over keys, and multiplying by $\mathbf{V}$ takes the corresponding weighted sum of values. Geometrically: query is "what I am looking for," key is "what I have," value is "what I provide." Attention is differentiable database lookup. Multi-head attention runs the same operation in several learned subspaces in parallel so that one head can pick up syntax while another picks up co-reference.

### Position information as rotation

Pure self-attention is permutation-equivariant, which would be a disaster for language. The fix is to inject position information. The classic sinusoidal encoding $PE_{(\text{pos},2i)} = \sin(\text{pos}/10000^{2i/d})$ has the property that $PE_{(\text{pos}+k)}$ is a linear function of $PE_{(\text{pos})}$ -- relative position is encoded as a rotation. The modern **Rotary Position Embedding (RoPE)** pushes this all the way: it rotates each pair of coordinates by an angle proportional to position, so that the *inner product* between query and key only depends on the relative offset. RoPE is, quite literally, complex multiplication.

### LoRA: low-rank adaptation

The biggest practical idea from the LLM era is that fine-tuning is intrinsically low-rank. Instead of updating $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out}\times d_\text{in}}$, LoRA freezes $\mathbf{W}_0$ and learns

$$
\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}, \qquad \mathbf{B} \in \mathbb{R}^{d_\text{out}\times r}, \quad \mathbf{A} \in \mathbb{R}^{r\times d_\text{in}}, \quad r \ll d.
$$

For $d = 4096$ and $r = 8$ this is a 256x parameter reduction, and at inference time you can fold $\mathbf{B}\mathbf{A}$ back into $\mathbf{W}_0$ for free. **QLoRA** combines this with 4-bit quantisation of $\mathbf{W}_0$ and lets you fine-tune 65B models on a single consumer GPU.

### KV cache and the cost of memory

In autoregressive generation the keys and values for past tokens never change, so caching them turns each new-token step into "compute Q/K/V for the new token, then attend." The cache occupies $O(2 \cdot L \cdot n \cdot d)$ for $L$ layers and is often the binding constraint at long context. This is space-for-time at industrial scale, and understanding *which* axis of which tensor is large is the difference between a model that runs and a model that doesn't.

---

## Sparse and efficient computation

### Sparse, linear, and approximate attention

Standard attention is $O(n^2)$ in sequence length, which is prohibitive for documents and long videos. Sparse attention sets most entries of $\mathbf{Q}\mathbf{K}^{T}$ to $-\infty$ -- locally-windowed (Longformer), strided (Sparse Transformer), or local plus a few global tokens (BigBird). Linear attention takes a different route: replace softmax by a kernel feature map $\phi$ so that

$$
\text{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) \approx \phi(\mathbf{Q})\bigl(\phi(\mathbf{K})^{T}\mathbf{V}\bigr),
$$

which is $O(nd^2)$ instead of $O(n^2 d)$ because the parenthesised product is a small $d \times d$ matrix.

### Quantisation

Symmetric INT$b$ quantisation maps $w$ to $\text{round}(w/s)$ for a per-tensor or per-channel scale $s$. Going from FP16 to INT4 is a 4x memory saving and -- on hardware that supports it -- a substantial speedup. The principled version, **GPTQ**, treats quantisation as a layer-wise weighted approximation problem with the empirical Hessian as the weight, and solves it with Cholesky-based updates. Quantisation is, again, a low-precision version of a matrix-approximation problem.

### Pruning

Remove the small-magnitude weights. Unstructured pruning gives 90%+ sparsity but is hard to accelerate; structured pruning (rows, columns, heads) is friendlier to hardware. NVIDIA Ampere ships a 2:4 sparse tensor-core path that runs structured-sparse matrix multiplication at full speed. Compressed storage (CSR, CSC) is the same chapter-12 vocabulary, dressed for 2024.

---

## Tensor networks: factorising exponentially big tensors

![Tensor networks: graphical calculus for high-dimensional tensors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig3_tensor_networks.png)

A tensor with $N$ indices each of size $d$ has $d^N$ entries. You cannot store it. **Tensor networks** are the right factorisation language for these objects, and -- as the figure shows -- they have a beautiful diagrammatic calculus where each node is a small tensor, each edge is a contracted bond, and each open leg is a remaining physical index.

The simplest tensor network is the **Matrix Product State** (also called Tensor Train),

$$
\mathcal{X}(i_1,\ldots,i_N) = \mathbf{G}_1(i_1)\,\mathbf{G}_2(i_2) \cdots \mathbf{G}_N(i_N),
$$

with each $\mathbf{G}_k(i_k)$ a small matrix. Storage drops from $d^N$ to $N d r^{2}$, where $r$ is the bond dimension. MPS is the structure underneath the **DMRG** algorithm in quantum many-body physics, and -- as randomised tensor train sketches -- in machine learning compression. **PEPS** generalises to two-dimensional lattices; **MERA** stacks isometries and disentanglers in a hierarchy that captures critical (scale-free) systems. The same picture also explains why deep networks can represent functions exponentially efficiently: each layer is one renormalisation step.

---

## The matrix exponential and a peek at Lie algebras

![Matrix exponential of so(3) generators traces great circles on $S^2$](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig4_lie_rotations.png)

The matrix exponential $e^{\mathbf{A}} = \sum_{k\ge 0}\mathbf{A}^k/k!$ is the bridge from linear algebra to *continuous* symmetry. For a skew-symmetric $\mathbf{K} \in \mathfrak{so}(3)$, the curve $t \mapsto e^{t\mathbf{K}}$ is a one-parameter group of rotations of $\mathbb{R}^3$. The figure shows orbits generated by the three so(3) basis elements $L_x, L_y, L_z$ acting on a common starting point: each orbit is a great circle on the unit sphere, and the tangent arrows at the starting point are the *generators* themselves.

Three things to take away. First, eigenvalue decomposition is what makes the exponential computable: if $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ then $e^{\mathbf{A}} = \mathbf{V}\,\text{diag}(e^{\lambda_i})\,\mathbf{V}^{-1}$. Second, the matrix exponential is exactly how single-qubit gates are parameterised, $e^{-i\theta(\mathbf{n}\cdot\boldsymbol{\sigma})/2}$, so the quantum and classical rotation stories are the same story. Third, **Neural ODEs** parameterise a hidden state by $d\mathbf{h}/dt = f(\mathbf{h},t,\theta)$, with the Euler discretisation reproducing residual connections; integrating that ODE is, locally, the matrix exponential of the Jacobian.

---

## From random matrices to free probability

![From random matrices (ch. 14) to free probability](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig5_random_to_free.png)

Chapter 14 told you that high-dimensional randomness is not chaos -- it is regularity in disguise. Two universal laws appear again on the left of the figure above: the eigenvalues of a sample covariance matrix at aspect ratio $\gamma = p/n$ pile up under the **Marchenko-Pastur** density, with sharp spectral edges at $(1\pm\sqrt{\gamma})^2$.

The right-hand panel takes the next step. If $W_1$ and $W_2$ are two large independent Wigner matrices, they are *asymptotically free* -- a non-commutative analogue of independence -- and the spectrum of $W_1 + W_2$ is again a semicircle, with variance equal to the sum of the variances. This is the **free central limit theorem**, and it is the start of a whole calculus (the $R$-transform) that lets you predict spectra of sums and products of random matrices analytically. Free probability is now a working tool inside neural-network theory, deep ensembles, and high-dimensional statistics.

---

## Topological data analysis: shape that survives noise

![Topological data analysis: shape persists across scales](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig6_persistence.png)

Sometimes the structure in your data is not a cluster or a low-dimensional subspace -- it is a **hole**. The figure above shows a noisy point cloud sampled from an annulus and its **persistence diagram**, the central object of TDA. Sweep a radius $r$ from 0 to large; at each $r$ build the Vietoris-Rips complex (connect any two points within $r$). Track when topological features are born and when they die. Plot every feature as a point $(\text{birth}, \text{death})$.

Short-lived points hug the diagonal and represent noise. Long-lived points -- the conspicuous diamond above the diagonal in the figure -- represent real features. Here a single $H_1$ point is far above the diagonal: it is the loop, and it survived precisely because the underlying shape really is annular. The whole pipeline rests on linear algebra: persistence is computed by reducing a *boundary matrix* over the field $\mathbb{F}_2$, which is Gaussian elimination with a clever pivot rule. TDA shows up in materials science, biology, and -- increasingly -- in inspecting the loss landscape of neural networks.

---

## Other frontier directions in one paragraph each

**Randomised numerical linear algebra.** Randomised SVD computes a rank-$k$ factorisation in $O(mnk)$ instead of $O(mn\min(m,n))$ by sketching the column space with a random Gaussian matrix and then doing a small dense SVD. The theoretical justification is the **Johnson-Lindenstrauss lemma**: random projection preserves pairwise distances. Sketching now underlies modern preconditioners, large-scale least squares, and log-determinant estimation.

**Implicit neural representations.** NeRF (Neural Radiance Fields) represents a 3D scene as an MLP from $(\mathbf{x}, \mathbf{d})$ to (density, colour). The Fourier-feature positional encoding is the same trick that lets Transformers see position; without it, MLPs systematically underfit high-frequency detail.

**Neural PDE solvers.** Physics-Informed Neural Networks parameterise the solution of a PDE by a neural network and add the PDE residual to the loss. Automatic differentiation -- an instance of the chain rule, hence of matrix multiplication -- makes arbitrary-order derivatives free. **Neural ODEs** view the network itself as a continuous dynamical system.

**Equivariant networks and geometric deep learning.** Build the symmetry group into the architecture by replacing matrix multiplication with group convolution. SO(3)-equivariant networks are now standard in molecular modelling.

---

## The eighteen chapters in one table

| Ch | Topic | Core insight |
|---|---|---|
| 1 | Vectors | Magnitude and direction; also elements of any vector space (functions, matrices, signals). |
| 2 | Vector spaces | Eight axioms specify exactly where linear combinations make sense. |
| 3 | Linear maps | Matrices and linear maps correspond one-to-one once you fix bases. |
| 4 | Determinants | Signed volume scaling; zero iff singular. |
| 5 | Linear systems | Solution structure is determined by the four fundamental subspaces. |
| 6 | Eigenvalues | Eigenvectors are the invariant directions of a transformation. |
| 7 | Orthogonality | Inner products give length and angle; orthogonal bases are numerically best. |
| 8 | Symmetric matrices | Real symmetric is orthogonally diagonalisable with real eigenvalues. |
| 9 | SVD | Every matrix factors as rotation - non-negative scaling - rotation. |
| 10 | Norms and conditioning | The condition number is the amplification factor of input error. |
| 11 | Matrix calculus | Gradient = direction of steepest ascent; chain rule = matrix product. |
| 12 | Sparsity | The L1 norm induces sparsity; compressed sensing breaks Nyquist. |
| 13 | Tensors | Multi-index arrays; decomposition exposes hidden structure. |
| 14 | Random matrices | High-dimensional randomness has surprising regularity (Wigner, MP). |
| 15 | Machine learning | PCA maximises variance; the kernel trick is an implicit feature map. |
| 16 | Deep learning | A neural network is layered matrix multiplication plus nonlinearity. |
| 17 | Computer vision | A camera is a projective matrix; reconstruction is an inverse problem. |
| 18 | Frontiers | Quantum gates are unitary; graph convolution is Laplacian filtering. |

### The most important theorems

- **Rank-nullity theorem.** $\dim\text{null}(\mathbf{A}) + \text{rank}(\mathbf{A}) = n$.
- **Spectral theorem.** Real symmetric matrices are orthogonally diagonalisable with real eigenvalues.
- **SVD existence.** Every $m \times n$ matrix has a singular value decomposition.
- **Eckart-Young theorem.** Truncated SVD is the optimal low-rank approximation in any unitarily invariant norm.
- **Johnson-Lindenstrauss lemma.** Random projection embeds high-dimensional points into low dimension with controlled distortion.
- **Cayley-Hamilton theorem.** Every matrix satisfies its own characteristic polynomial.
- **Courant-Fischer min-max.** Eigenvalues of a symmetric matrix are extrema of Rayleigh quotients on subspaces.

---

## The recurring triangle: geometry, numerics, computation

![The recurring triangle of the series: geometry, numerics, computation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/18-frontiers-and-summary/fig7_three_pillars.png)

If there is one diagram that summarises the whole series, it is this triangle. **Geometry** is the source of intuition -- vectors as arrows, matrices as transformations of space, eigenvectors as invariant directions, the Bloch sphere. **Numerics** is what tells you when you can trust the computation -- norms, conditioning, stable algorithms, floating-point reality. **Computation** is what makes any of it useful at scale -- sparse kernels, randomised methods, GPU tensor cores, low-precision arithmetic.

The three pillars are not separate disciplines. They lean on each other. Spectral theory bridges geometry and numerics: the spectrum of a matrix tells you both its geometric character and its sensitivity to perturbation. Sketching bridges geometry and computation: if you keep distances approximately, you keep the geometry while shrinking the cost. Low-precision arithmetic bridges numerics and computation: you give up bits and gain throughput, and you have to use conditioning to know how many bits you can afford to lose.

The SVD sits at the centre because all three pillars agree on it. Geometrically it is the right way to look at any linear map. Numerically it is the most stable factorisation we know. Computationally it admits a randomised version that scales to enormous data. If you remember nothing else from this series, remember: *when in doubt, take the SVD*.

---

## Learning advice and resources

### How to keep going

**Visualise.** GeoGebra, Manim (the library 3Blue1Brown uses), or just NumPy and matplotlib. Watching what a matrix does to a grid will teach you more than re-reading any formula.

**Compute small examples by hand.** A surprising amount of linear algebra is invisible until you have ground out a $3 \times 3$ Gram-Schmidt or a $4 \times 4$ SVD on paper. Do it once.

**Always ask why.** Why is the determinant defined this way? Why is the SVD always real? Why does the GCN layer have a self-loop? Refusing to settle for "this is the formula" is the difference between using linear algebra and understanding it.

**Connect every theorem to an application.** Eigendecomposition: PageRank. SVD: latent semantic analysis, recommender systems, NeRF camera poses. Sparsity: compressed sensing MRI. Random matrices: covariance cleaning in finance. The connections are not afterthoughts -- they are how the field moves.

### Reading list

**Classic textbooks.**

- Gilbert Strang, *Introduction to Linear Algebra* -- intuition first, the canonical first book.
- Sheldon Axler, *Linear Algebra Done Right* -- elegant, determinant-free, abstract perspective.
- Trefethen and Bau, *Numerical Linear Algebra* -- the right way to learn the numerical side.
- Golub and Van Loan, *Matrix Computations* -- the engineer's reference.
- Strang, *Linear Algebra and Learning from Data* -- the bridge to machine learning.

**Frontier reading.**

- Nielsen and Chuang, *Quantum Computation and Quantum Information* -- the standard quantum text.
- Bronstein et al., *Geometric Deep Learning* -- equivariance, GNNs, Transformers under one umbrella.
- Halko, Martinsson, Tropp, "Finding Structure with Randomness" -- the randomised NLA manifesto.
- Edelsbrunner and Harer, *Computational Topology* -- the right introduction to TDA.

**Online courses.** MIT 18.06 by Strang on YouTube; 3Blue1Brown's *Essence of Linear Algebra*; Stanford CS229 for the ML angle.

**Software.** NumPy / SciPy and PyTorch / JAX for everyday work; Julia for numerical research; Manim for animations.

---

## Exercises

### Quantum computing

1. Verify that the Hadamard matrix is unitary by computing $\mathbf{H}^{\dagger}\mathbf{H}$.
2. Compute $\mathbf{H}|0\rangle$ and $\mathbf{H}|1\rangle$ and locate both states on the Bloch sphere.
3. Show that the Pauli matrices anticommute, e.g. $\mathbf{X}\mathbf{Y} = -\mathbf{Y}\mathbf{X}$, and that $\mathbf{X}^2 = \mathbf{Y}^2 = \mathbf{Z}^2 = \mathbf{I}$.
4. Prove that the Bell state $\tfrac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ cannot be written as a tensor product $|\phi\rangle \otimes |\psi\rangle$.
5. Design a quantum circuit that maps $|00\rangle$ to $\tfrac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$.

### Graph neural networks

6. For the four-node cycle $1{-}2,\ 1{-}3,\ 2{-}4,\ 3{-}4$, write $\mathbf{A}$, $\mathbf{D}$, and $\mathbf{L}$.
7. Compute the eigenvalues of that Laplacian; confirm the smallest is 0 and explain what its multiplicity tells you.
8. Prove $\mathbf{x}^{T}\mathbf{L}\mathbf{x} = \sum_{(i,j)\in E}(x_i - x_j)^2$ and interpret it physically.
9. Show that the normalised Laplacian has eigenvalues in $[0, 2]$.
10. What goes wrong with $\mathbf{H}' = \sigma(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{H}\mathbf{W})$ if you forget the self-loop?

### Large models and efficient computing

11. In scaled dot-product attention, what fails for large $d_k$ if you remove the $\sqrt{d_k}$ normalisation?
12. With $d_\text{in} = d_\text{out} = 4096$ and rank $r = 16$, give the parameter count of the original layer, of the LoRA adapter, and the reduction ratio.
13. For sequence length $n$, $L$ layers, per-layer K/V dimension $d$, derive the size of the KV cache.
14. Design INT8 symmetric quantisation for a tensor with values in $[-2.5, 2.5]$. State both the quantise and dequantise formulas.
15. Sliding-window attention with window $w$ has what asymptotic complexity? Compare to standard $O(n^2)$.

### Frontier topics

16. Show that $e^{\mathbf{A}}e^{\mathbf{B}} \ne e^{\mathbf{A} + \mathbf{B}}$ in general, and that equality does hold when $\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$.
17. For an MPS with $N = 50$, physical dim $d = 4$, bond dim $r = 32$, compute the parameter count and compare to the dense tensor with $d^N$ entries.
18. Implement randomised SVD: draw a Gaussian $\boldsymbol{\Omega}$, form $\mathbf{Y} = \mathbf{A}\boldsymbol{\Omega}$, take a thin QR, and SVD the small projected matrix.
19. Sample 100 points from a circle and 100 from a disk. Compute the persistence diagram of each (any TDA library is fine). Explain the difference in $H_1$.
20. Combine GNN and Transformer for molecular property prediction: describe the architecture, the inductive biases of each piece, and where you would put the cross-attention.

### Programming

21. Implement Hadamard and CNOT gates in NumPy and simulate $|00\rangle \xrightarrow{\mathbf{H}\otimes\mathbf{I}} \xrightarrow{\text{CNOT}}$.
22. Implement a one-layer GCN in PyTorch and run node classification on Karate Club.
23. Implement LoRA on a small linear layer; verify that with $r = \min(d_\text{in}, d_\text{out})$ it is equivalent to a full update.
24. Implement INT8 symmetric quantise/dequantise; measure the per-channel error on a pretrained linear layer.
25. Compare wall-clock time of standard vs. sliding-window attention as $n$ grows; plot the ratio.

---

## A short closing

Linear algebra is both ancient and young. Ancient because its core ideas have been settled for two centuries. Young because every generation of technology finds a new use for them: equation solving in the nineteenth century, quantum mechanics and operations research in the twentieth, machine learning and large-scale inference in the twenty-first.

The remarkable continuity is the point. Quantum gates are unitary matrices. Graph convolution is Laplacian filtering. Attention is softmax of $\mathbf{Q}\mathbf{K}^{T}$ times $\mathbf{V}$. LoRA is a low-rank update. Tensor networks factorise exponentials. NeRF and PINNs lean on the matrix exponential. Free probability extends the central limit theorem to non-commuting matrices. None of this is foreign. It is *the same vocabulary* applied at a new scale or in a new setting.

If this series has done its job, when you next see a paper that intimidates you with "spectral" or "tensor" or "attention" in the title, you will recognise it as a friend. Open it. Look for the matrices. Look for the decomposition. Look for the conditioning. The mathematics will yield.

Thank you for walking the eighteen chapters with me. The end of a series is not the end of the journey -- it is the start of yours.

---

## References

- Nielsen, M. A., and Chuang, I. L. *Quantum Computation and Quantum Information*. Cambridge University Press.
- Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
- Bronstein, M., Bruna, J., Cohen, T., and Velickovic, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* 2021.
- Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
- Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
- Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
- Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
- Halko, N., Martinsson, P.-G., and Tropp, J. A. "Finding Structure with Randomness." *SIAM Review*, 2011.
- Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020.
- Chen, R. T. Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
- Orus, R. "A Practical Introduction to Tensor Networks." *Annals of Physics*, 2014.
- Edelsbrunner, H., and Harer, J. *Computational Topology: An Introduction.* AMS, 2010.
- Voiculescu, D., Dykema, K. J., and Nica, A. *Free Random Variables.* AMS, 1992.
- Strang, G. *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press, 2019.

---

## Series Navigation

- **Previous:** [Chapter 17: Linear Algebra in Computer Vision](/en/chapter-17-linear-algebra-in-computer-vision/)
- **Full Series:** Essence of Linear Algebra (1--18)
