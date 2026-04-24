---
title: "PDE and Machine Learning (8): Reaction-Diffusion Systems and Graph Neural Networks"
date: 2024-12-12 09:00:00
tags:
  - PDE
  - Machine Learning
  - Reaction-Diffusion
  - Graph Neural Networks
  - GNN
  - Turing Instability
  - Over-smoothing
categories: PDE and Machine Learning
series:
  name: "PDE and Machine Learning"
  part: 8
  total: 8
lang: en
mathjax: true
description: "Deep GNNs collapse because they are diffusion equations on graphs. Turing's reaction-diffusion theory tells us how to fix it -- and closes the eight-chapter PDE+ML series."
disableNunjucks: true
---

## What This Article Covers

Stack 32 layers of GCN on a citation graph and accuracy collapses from 81 % to 20 %. Every node converges to the same vector. This is **over-smoothing**, the GNN equivalent of heat death — and the diagnosis comes straight from PDE theory. **A GCN layer is one explicit-Euler step of the heat equation on a graph**, and the heat equation has exactly one fixed point: the constant. The cure was published in 1952. Alan Turing showed that adding a *reaction* term to a diffusion equation can make a uniform state spontaneously break apart into stripes, spots, or labyrinths. The same trick — a learned reaction term — keeps deep GNNs alive.

This is also Part 8 of the *PDE + Machine Learning* series. We have spent seven chapters arguing that essentially every modern neural architecture is, secretly, the discretisation of a PDE. Reaction-diffusion + GNNs closes the loop: it is the most explicitly PDE-shaped architecture of all, and it lets us re-examine *every* preceding chapter through one final lens.

**What you will learn**

1. Reaction-diffusion equations on continuous space — Gray-Scott, FitzHugh-Nagumo, the morphologies they produce.
2. Turing instability — the linear-stability argument that explains how diffusion can *create* structure.
3. Graph Laplacians — the discrete analogue of $\nabla^2$, and why its spectrum dictates GNN behaviour.
4. GCN $=$ discretised graph diffusion — and the spectral proof of over-smoothing.
5. Reaction-diffusion GNNs (GRAND, GRAND++, RDGNN) — adding a reaction term that keeps node features distinct.
6. A retrospective on the entire PDE + ML series.

**Prerequisites:** linear algebra (eigen-decomposition), basic PDE concepts (the diffusion equation), and familiarity with message-passing GNNs.

---

![Four Turing morphologies produced by the Gray-Scott model — spots, stripes, labyrinth, holes — together with a sketch of where each lives in the $(F,k)$ regime map.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig1_turing_patterns.png)
*Four Turing morphologies produced by the Gray-Scott model — spots, stripes, labyrinth, holes — together with a sketch of where each lives in the $(F,k)$ regime map.*

## 1. Reaction-Diffusion in Continuous Space

### 1.1 The General Form

A reaction-diffusion (RD) equation couples spatial diffusion with local nonlinear reactions:
$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D}\,\nabla^2\mathbf{u} + \mathbf{R}(\mathbf{u}). \tag{1}
$$
- The diffusion term $\mathbf{D}\nabla^2\mathbf{u}$ is **linear** and **smoothing** — it always reduces gradients.
- The reaction term $\mathbf{R}(\mathbf{u})$ is **local** (no spatial derivatives) and **nonlinear** — it can either reinforce or oppose the smoothing.

Two perspectives are useful. *Physically*, $\mathbf{u}$ is a vector of concentrations; diffusion is Fick's law, reaction is the local rate equation. *Mathematically*, (1) is a semilinear parabolic PDE — the heat equation we met in Chapter 7, plus a pointwise nonlinear forcing.

The remarkable thing — and Turing's insight — is that the *competition* between these two terms can produce stable, non-trivial spatial patterns from a uniform initial state. Call this **diffusion-driven instability**.

### 1.2 Gray-Scott

Gray-Scott is the canonical two-component model:
$$
\partial_t u = D_u \nabla^2 u - u v^2 + F(1-u),\qquad
\partial_t v = D_v \nabla^2 v + u v^2 - (F+k)\,v.
$$
- $u$ is a substrate fed in at rate $F$; $v$ is an autocatalyst that consumes $u$ via $u + 2v \to 3v$ and decays at rate $k$.
- With $D_u > D_v$ (substrate diffuses faster than the catalyst), small patches of $v$ stabilise into the morphologies of Fig. 1.

The same equation, with different $(F, k)$, gives **spots**, **stripes**, **labyrinths**, **holes**, **moving spots**, even **self-replicating spots** — Pearson (1993) catalogued a dozen distinct regimes.

### 1.3 FitzHugh-Nagumo

Originally a simplified neuron model:
$$
\partial_t v = D \nabla^2 v + v - \tfrac{v^3}{3} - w + I,\qquad
\partial_t w = \varepsilon\,(v + \beta - \gamma w),\quad \varepsilon \ll 1.
$$
- $v$ is the fast membrane potential; $w$ is a slow recovery variable.
- The cubic nonlinearity makes $v$ excitable: a super-threshold push sets off a stereotyped pulse that the slow $w$ then resets.

In 2D you get spiral waves and target patterns — exactly the patterns observed in cardiac tissue during arrhythmia and in the developing chick retina (see Fig. 6 below).

---

## 2. Turing Instability: Patterns from Uniformity

### 2.1 The Question

Pick a uniform steady state $\bar{\mathbf{u}}$ with $\mathbf{R}(\bar{\mathbf{u}}) = \mathbf{0}$ that is **stable in the well-mixed (no diffusion) system**. Can adding diffusion ever *destabilise* it?

The naive answer is no — diffusion only smooths, surely it can only *help* stability. Turing (1952) proved that intuition wrong.

### 2.2 The Argument

Linearise (1) around $\bar{\mathbf{u}}$ with perturbation $\delta\mathbf{u}(\mathbf{x}, t) = \mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}\,e^{\sigma t}$:
$$
\sigma\,\mathbf{q} \;=\; \underbrace{\bigl(\mathbf{J} - |\mathbf{k}|^2\,\mathbf{D}\bigr)}_{\mathbf{A}(|\mathbf{k}|^2)}\,\mathbf{q},\qquad
\mathbf{J} = \nabla_{\mathbf{u}}\mathbf{R}(\bar{\mathbf{u}}). \tag{2}
$$
The mode $\mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}$ grows when $\mathbf{A}(|\mathbf{k}|^2)$ has an eigenvalue with positive real part. The full **Turing condition** is then a list of four inequalities (Fig. 2, right):

1. $\mathrm{tr}\,\mathbf{J} < 0$ and $\det\,\mathbf{J} > 0$ — **the well-mixed system is stable**.
2. The Jacobian has activator-inhibitor structure: $f_u > 0$, $g_v < 0$, with $f_v\,g_u < 0$.
3. **Diffusion asymmetry**: $D_v \gg D_u$ — the inhibitor diffuses much faster than the activator.
4. There exists some $|\mathbf{k}|^2$ with $\det\,\mathbf{A}(|\mathbf{k}|^2) < 0$, i.e. an unstable wavenumber.

Conditions 1-3 are algebraic facts about the kinetics. Condition 4 follows once the first three hold and is the actual mechanism: a *band* of wavenumbers becomes unstable, and the most unstable mode $|\mathbf{k}_*|$ sets the **characteristic length scale of the resulting pattern**, $\ell \sim 2\pi/|\mathbf{k}_*|$.

![Left: dispersion relation $\sigma(|\mathbf{k}|^2)$ for an activator-inhibitor system. With equal diffusion (blue) the system is stable everywhere; making the inhibitor diffuse faster (red) opens a band of unstable wavenumbers around $|\mathbf{k}_*|^2 \approx 3.4$. Right: the four Turing conditions in a glance.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig2_turing_instability.png)
*Left: dispersion relation $\sigma(|\mathbf{k}|^2)$ for an activator-inhibitor system. With equal diffusion (blue) the system is stable everywhere; making the inhibitor diffuse faster (red) opens a band of unstable wavenumbers around $|\mathbf{k}_*|^2 \approx 3.4$. Right: the four Turing conditions at a glance.*

### 2.3 The Intuition Behind Diffusion-Driven Instability

Why does asymmetric diffusion destabilise an otherwise-stable steady state? Imagine a tiny local bump of activator. *Locally*, the activator self-amplifies (positive feedback). It also produces inhibitor — but the inhibitor diffuses away quickly, so its concentration *near* the bump stays low, while *far away* the inhibitor builds up and suppresses other potential bumps. This is **short-range activation, long-range inhibition** — the universal recipe behind animal coat patterns, vegetation stripes, sand ripples, and (we'll see in §5) the architecture of deep GNNs.

---

## 3. From Grids to Graphs

### 3.1 Why Graphs?

Finite differences (FDM) and finite elements (FEM) discretise PDEs on regular grids or carefully designed meshes. They are extremely powerful when the domain is simple. But for **molecular structures, social networks, citation graphs, road networks, brain connectomes**, there is no natural notion of a regular grid — the connectivity is the geometry.

A graph $G = (V, E)$ generalises both: it is just a set of nodes plus a relation between them. A GNN solves a "PDE" on this graph. The job of this section is to write down what that PDE looks like.

![From regular grids to irregular graphs. Both discretise $\nabla^2$, but the grid stencil is replaced by a node's neighbourhood. The continuous PDE $\partial_t u = D\nabla^2 u + R(u)$ admits both flavours of discretisation; one Euler step of the graph version is exactly a GCN layer.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig3_grid_to_graph.png)
*From regular grids to irregular graphs. Both discretise $\nabla^2$, but the grid stencil is replaced by a node's neighbourhood. The continuous PDE $\partial_t u = D\nabla^2 u + R(u)$ admits both flavours of discretisation; one Euler step of the graph version is exactly a GCN layer.*

### 3.2 The Graph Laplacian

For a weighted, undirected graph with adjacency matrix $\mathbf{A}$ and degree matrix $\mathbf{D} = \mathrm{diag}(d_i)$:

| Variant | Formula | Spectrum lives in |
|---------|---------|-------------------|
| Combinatorial | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | $[0, 2 d_{\max}]$ |
| Symmetric normalised | $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ | $[0, 2]$ |
| Random-walk | $\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$ | $[0, 2]$ |

All three share a defining property:
$$
\mathbf{x}^{\!\top}\!\mathbf{L}\mathbf{x} \;=\; \tfrac{1}{2}\sum_{(i,j) \in E} w_{ij}\,(x_i - x_j)^2 \;\geq\; 0. \tag{3}
$$
The Laplacian is the discrete analogue of $-\nabla^2$ in the sense that it integrates the squared gradient. It is symmetric positive semi-definite, with spectral decomposition $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\!\top}$, eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$.

The smallest eigenvalue is always zero, with eigenvector proportional to $\mathbf{1}$ (the constant). The second smallest, $\lambda_2$ — the *algebraic connectivity* — measures how well-connected the graph is.

### 3.3 The Graph Heat Equation

Write down the obvious continuous-time dynamics
$$
\frac{d\mathbf{X}}{dt} = -\mathbf{L}\mathbf{X}. \tag{4}
$$
This is the graph heat equation. Its closed-form solution is $\mathbf{X}(t) = e^{-\mathbf{L}t}\mathbf{X}(0)$, and in spectral coordinates the dynamics decouple completely:
$$
\hat x_k(t) = e^{-\lambda_k t}\,\hat x_k(0),\qquad \hat x_k = \mathbf{u}_k^{\!\top}\mathbf{X}(0).
$$
Every mode decays exponentially at its own rate $\lambda_k$ — except $\lambda_1 = 0$, which is preserved forever. As $t \to \infty$ only the constant survives.

![Graph heat equation in action. A random initial signal on a 50-node small-world graph is annihilated by diffusion: by $t = 6$ every node carries the same value. The right panel shows why — the $k$-th mode decays as $e^{-\lambda_k t}$, and only $\lambda_1 = 0$ survives.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig4_graph_laplacian.png)
*Graph heat equation in action. A random initial signal on a 50-node small-world graph is annihilated by diffusion: by $t = 6$ every node carries the same value. The right panel shows why — the $k$-th mode decays as $e^{-\lambda_k t}$, and only $\lambda_1 = 0$ survives.*

This is **over-smoothing in its purest form**, and we have not even mentioned neural networks yet.

---

## 4. GCN Is the Heat Equation

### 4.1 The Identification

The standard GCN layer (Kipf & Welling, 2017) is
$$
\mathbf{H}^{(\ell+1)} = \sigma\bigl(\tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}\,\mathbf{W}^{(\ell)}\bigr),
\qquad \tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}(\mathbf{A} + \mathbf{I})\tilde{\mathbf{D}}^{-1/2}.
$$
Strip away the nonlinearity and the linear projection ($\sigma = \mathrm{id}$, $\mathbf{W} = \mathbf{I}$). What remains is
$$
\mathbf{H}^{(\ell+1)} = \tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}
\;=\; \bigl(\mathbf{I} - \tilde{\mathbf{L}}_{\text{sym}}\bigr)\mathbf{H}^{(\ell)}.
\tag{5}
$$
This is exactly the **explicit Euler step** of the graph heat equation $\dot{\mathbf{H}} = -\tilde{\mathbf{L}}_{\text{sym}}\mathbf{H}$ with step size $h = 1$. The "self-loops" trick $\mathbf{A} + \mathbf{I}$ is just the standard FDM stabilisation that pushes the spectrum of $\tilde{\mathbf{L}}_{\text{sym}}$ into $[0, 2)$ so the explicit scheme remains stable.

### 4.2 The Spectral Proof of Over-Smoothing

After $L$ layers (still ignoring nonlinearities and weight matrices),
$$
\mathbf{H}^{(L)} = \tilde{\mathbf{A}}^L\,\mathbf{H}^{(0)}.
$$
The eigenvalues of $\tilde{\mathbf{A}}$ lie in $(-1, 1]$, with the eigenvalue $1$ corresponding to the constant eigenvector. Take a power and everything except the leading eigenspace dies:
$$
\tilde{\mathbf{A}}^L \xrightarrow[L \to \infty]{} \pi_{\text{const}}.
$$
Every node feature collapses onto the same vector. **This is not a quirk of GCN — it is a theorem about iterating any low-pass filter.** Adding ReLU and learning the weight matrices delays the collapse but does not prevent it: Oono & Suzuki (2020) proved that for an arbitrary weight matrix sequence with bounded singular values, GCN features still converge to a low-dimensional subspace.

### 4.3 Continuous-Depth GNNs

If GCN is one Euler step, why not solve the ODE properly? **GRAND** (Chamberlain et al., 2021) is the continuous-time GNN:
$$
\frac{d\mathbf{X}}{dt} = -\mathcal{L}_\theta(\mathbf{X})\,\mathbf{X},\qquad \mathbf{X}(T) = \text{output.}
$$
$\mathcal{L}_\theta$ is a learned attention-weighted Laplacian, and the integration is done with an off-the-shelf ODE solver (Dormand-Prince, etc.). This does *not* fix over-smoothing — solving a heat equation more accurately is still solving a heat equation. **GRAND++** (Thorpe et al., 2022) adds a *source* term, and **RDGNN** (Eliasof et al., 2024 and predecessors) adds a full *reaction* term. We now build the latter.

---

## 5. RDGNN: Reaction-Diffusion Graph Neural Networks

### 5.1 The Architecture

The continuous-time RD-GNN is the natural graph version of (1):
$$
\frac{d\mathbf{H}}{dt} = -\epsilon_d\,\mathbf{L}\,\mathbf{H} \;+\; \epsilon_r\,R_\theta(\mathbf{H}, \mathbf{H}^{(0)}).
\tag{6}
$$
One Lie-Trotter (operator-splitting) step gives the discrete update:
$$
\boxed{\;\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} \;-\; \epsilon_d\,\mathbf{L}\,\mathbf{H}^{(\ell)} \;+\; \epsilon_r\,R_\theta\bigl(\mathbf{H}^{(\ell)},\,\mathbf{H}^{(0)}\bigr).\;} \tag{7}
$$
Three blocks (Fig. 5):

- **Diffusion** $-\epsilon_d \mathbf{L}\mathbf{H}^{(\ell)}$: standard graph smoothing. Step size constraint: $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$ for explicit-Euler stability.
- **Reaction** $\epsilon_r R_\theta(\mathbf{H}^{(\ell)}, \mathbf{H}^{(0)})$: a learned, *purely local* transform — typically a small MLP applied node-wise. Conditioning on $\mathbf{H}^{(0)}$ acts like the input-skip in a ResNet and prevents drift.
- **Skip** $\mathbf{H}^{(\ell)}$ in the update: keeps the dynamics close to the identity, which is what makes large $L$ stable in practice.

A common reaction-term design is the FitzHugh-style activator-decay pair
$$
R_\theta(\mathbf{H}, \mathbf{H}^{(0)}) = \mathrm{MLP}_\theta\bigl([\mathbf{H} \,\Vert\, \mathbf{H}^{(0)}]\bigr) \;-\; \alpha\,\mathbf{H}.
$$

![A reaction-diffusion GNN layer. The diffusion branch performs the usual graph-Laplacian smoothing; the reaction branch is a learned, node-wise nonlinear update; an input skip from $\mathbf{H}^{(0)}$ provides the standard "anchor" against drift. Repeating the block $L$ times yields a deep GNN that, unlike GCN, does not collapse.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig5_rdgnn_architecture.png)
*A reaction-diffusion GNN layer. The diffusion branch performs the usual graph-Laplacian smoothing; the reaction branch is a learned, node-wise nonlinear update; an input skip from $\mathbf{H}^{(0)}$ provides the standard "anchor" against drift. Repeating the block $L$ times yields a deep GNN that, unlike GCN, does not collapse.*

### 5.2 Why It Works

There are two ways to see why a reaction term cures over-smoothing.

**Spectral view.** Pure diffusion *attenuates* every non-constant mode at rate $e^{-\lambda_k t}$. The reaction term is *not* a function of $\mathbf{L}$ and therefore can have arbitrary spectral content; in particular it can pump energy back into the high-frequency modes that diffusion is killing. The net effect is a non-trivial steady-state distribution of energy across $k$.

**Turing view.** If $R_\theta$ has activator-inhibitor structure (which a sufficiently expressive MLP can learn), and the diffusion strength $\epsilon_d$ is chosen so that the eigenvalues of $\mathbf{J} - \epsilon_d\,\lambda_k$ are unstable for some $k$, the network exhibits **node-level Turing patterns** — different nodes converge to different feature values, organised by the graph spectrum. This is the GNN equivalent of stripes on a fish.

### 5.3 PyTorch Implementation

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class RDGNN(nn.Module):
    """Reaction-diffusion GNN.

    H^(l+1) = H^(l) - eps_d * L H^(l) + eps_r * R(H^(l), H^(0))
    The diffusion step is implemented via GCNConv, which contains the
    normalised Laplacian; the reaction step is a per-node MLP conditioned
    on the input embedding.
    """

    def __init__(self, in_dim, hidden, out_dim, n_layers,
                 eps_d=0.1, eps_r=0.1, alpha=0.1):
        super().__init__()
        self.eps_d, self.eps_r, self.alpha = eps_d, eps_r, alpha
        self.encoder = nn.Linear(in_dim, hidden)
        self.diff = GCNConv(hidden, hidden, normalize=True, add_self_loops=True)
        self.react = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden),
                          nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h0 = self.encoder(x)
        h = h0
        for r in self.react:
            # Diffusion step: -eps_d * L h  ~~  eps_d * (GCN(h) - h)
            h_diff = self.diff(h, edge_index) - h
            h_react = r(torch.cat([h, h0], dim=-1)) - self.alpha * h
            h = h + self.eps_d * h_diff + self.eps_r * h_react
        return self.decoder(h)
```

Notice how lean the architecture is: one shared GCN for diffusion, $L$ small MLPs for reaction, two linear projections at the ends. The accuracy curves in Fig. 6c show that this is enough to maintain Cora performance up to 64 layers — eight times deeper than the regime in which GCN collapses.

---

## 6. Where Reaction-Diffusion Already Wins

The same equation, three application stories.

![From biology to GNNs. (a) The Gray-Scott model produces realistic fur-and-skin patterns — exactly the mechanism Turing proposed for biological morphogenesis. (b) FitzHugh-Nagumo dynamics produce spiral waves, observed during cardiac arrhythmia and in early visual-cortex development. (c) The same RD principle, applied to graphs, produces deep GNNs that do not collapse — RDGNN keeps Cora accuracy near 80 % at 64 layers, while GCN, GAT, and even pure-diffusion GRAND fall below 25 %.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig6_applications.png)
*From biology to GNNs. (a) The Gray-Scott model produces realistic fur-and-skin patterns — exactly the mechanism Turing proposed for biological morphogenesis. (b) FitzHugh-Nagumo dynamics produce spiral waves, observed during cardiac arrhythmia and in early visual-cortex development. (c) The same RD principle, applied to graphs, produces deep GNNs that do not collapse — RDGNN keeps Cora accuracy near 80 % at 64 layers, while GCN, GAT, and even pure-diffusion GRAND fall below 25 %.*

**Morphogenesis.** Murray's *Mathematical Biology* uses Turing's mechanism to fit the spot-and-stripe transitions on big-cat coats: large cats (jaguar) get spots, intermediate cats (leopard) get rosettes, the tail (where the local geometry constrains $|\mathbf{k}|$) gets stripes — all from a single set of reaction parameters and the geometry of the embryo. The same maths predicts vegetation stripes in semi-arid landscapes (water as inhibitor, biomass as activator) and the spiral arms of *Belousov-Zhabotinsky* chemistry experiments.

**Neural development.** During retinal mosaic formation, neighbouring photoreceptors *inhibit* each other from differentiating into the same subtype while *activating* the same subtype across longer ranges via diffusing morphogens. This is mathematically a Turing system, and the resulting cone arrangements have measurable wavelength $\ell \sim 2\pi/|\mathbf{k}_*|$. Spiral waves in cortical electrical activity — pathological during epilepsy, important during developmental wiring — are FitzHugh-Nagumo solutions on a 2D excitable medium.

**Deep GNNs.** On standard benchmarks, the depth-vs-accuracy story is dramatic (Fig. 6c, replicated from Eliasof et al. (2024) on Cora). GCN and GAT collapse beyond 8 layers as the spectral proof predicts; pure-diffusion GRAND merely *delays* the collapse by being more accurate; only RDGNN — explicit reaction term — maintains accuracy at $L = 64$. The same effect is even larger on **heterophilic** graphs (where neighbours tend to have *different* labels), because pure smoothing is actively harmful there: the reaction term can learn to *amplify* differences between connected nodes.

---

## 7. Series Finale: Eight Chapters, One Idea

We are at the end of the *PDE + Machine Learning* series. Step back.

![The eight-chapter journey. Chapters 1-2 use neural networks to *solve* PDEs. Chapters 3-4 reinterpret training as a variational principle. Chapters 5-6 build networks that respect the geometric structure of the underlying flow. Chapters 7-8 apply the same machinery to generative modelling and graph learning.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/fig7_series_journey.png)
*The eight-chapter journey. Chapters 1-2 use neural networks to *solve* PDEs. Chapters 3-4 reinterpret training as a variational principle. Chapters 5-6 build networks that respect the geometric structure of the underlying flow. Chapters 7-8 apply the same machinery to generative modelling and graph learning.*

The eight chapters fall into four pairs.

| Pair | Chapters | The PDE in the picture |
|------|----------|------------------------|
| Solving PDEs with NNs | [1](/en/PDE-and-Machine-Learning-1-Physics-Informed-Neural-Networks/) PINN, [2](/en/PDE-and-Machine-Learning-2-Neural-Operator-Theory/) Neural Operators | The target equation itself becomes the loss. |
| Variational view | [3](/en/PDE-and-Machine-Learning-3-Variational-Principles/) Deep Ritz, [4](/en/PDE-and-Machine-Learning-4-Variational-Inference/) VI / Fokker-Planck | Loss $=$ free energy; gradient flow $=$ continuity equation. |
| Structure-preserving flows | [5](/en/PDE-and-Machine-Learning-5-Symplectic-Geometry/) Symplectic nets, [6](/en/PDE-and-Machine-Learning-6-Continuous-Normalizing-Flows/) Neural ODE / CNF | Network respects the symplectic / volume / divergence structure of the flow. |
| Generative + graph PDEs | [7](/en/PDE-and-Machine-Learning-7-Diffusion-Models/) Diffusion models, **8** RD + GNN | Forward / reverse Fokker-Planck; reaction-diffusion on graphs. |

Underneath every chapter sits one slogan:

> **A neural architecture is a discretised PDE. Choose the architecture by choosing the right PDE.**

Concretely:

- **Want extrapolation off the training grid?** Choose an operator-learning PDE (Chapter 2).
- **Want a network that reproduces conserved quantities?** Choose a symplectic integrator (Chapter 5).
- **Want a tractable likelihood?** Choose a continuity equation and learn its drift (Chapter 6).
- **Want to sample a complicated distribution?** Choose a Fokker-Planck and learn its score (Chapter 7).
- **Want a deep GNN that does not collapse?** Choose a reaction-diffusion equation, not just diffusion (this chapter).

The PDE perspective is not the only useful lens on deep learning, but it is uncommonly *generative*: every time we have asked "what would the corresponding numerical analysis say?" we have learned something concrete — a stability bound, a step-size constraint, a structural fix. This is what physics-style thinking buys you in machine learning, and it is why the conversation between the two fields is far from over.

---

## 8. Exercises

**Exercise 1.** Show that for a connected graph the only solution of $\mathbf{L}\mathbf{x} = \mathbf{0}$ is the constant. Conclude that the heat equation on a connected graph drives every initial condition to its mean.

> *Solution.* From (3), $\mathbf{x}^\top\!\mathbf{L}\mathbf{x} = \tfrac{1}{2}\sum w_{ij}(x_i - x_j)^2 = 0$ forces $x_i = x_j$ across every edge; on a connected graph this means $\mathbf{x}$ is constant. The kernel of $\mathbf{L}$ is therefore one-dimensional. Every other eigenvalue is strictly positive, so $e^{-\mathbf{L}t}$ kills all non-constant components and preserves the mean. $\blacksquare$

**Exercise 2.** Derive the explicit-Euler stability bound $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$ for the diffusion step.

> *Solution.* In spectral coordinates the Euler update is $\hat{x}_k^{(\ell+1)} = (1 - \epsilon_d \lambda_k)\,\hat{x}_k^{(\ell)}$. For non-growth we need $|1 - \epsilon_d \lambda_k| \leq 1$ for every $k$, which gives $0 \leq \epsilon_d \lambda_k \leq 2$, i.e. $\epsilon_d \leq 2/\lambda_{\max}$. Asymptotic monotone decay (no oscillation) requires the stricter $\epsilon_d < 1/\lambda_{\max}$. $\blacksquare$

**Exercise 3.** Why does RDGNN help especially on heterophilic graphs?

> *Solution.* On heterophilic graphs neighbours tend to carry *opposite* labels. The pure diffusion step averages neighbouring features and therefore actively destroys the discriminative signal — the more layers, the worse. The reaction term operates *node-wise* and is conditioned on $\mathbf{H}^{(0)}$, so it can produce node-specific updates that *increase* the difference between neighbours, restoring class separation. $\blacksquare$

**Exercise 4.** Show that the discrete RDGNN update (7) is the first-order operator-splitting (Lie-Trotter) discretisation of the continuous RD-GNN (6).

> *Solution.* Operator splitting writes $\dot{\mathbf{H}} = (\mathcal{D} + \mathcal{R})\,\mathbf{H}$ with $\mathcal{D}\mathbf{H} = -\mathbf{L}\mathbf{H}$ and $\mathcal{R}\mathbf{H} = R_\theta(\mathbf{H})/\epsilon_r$, then alternates one Euler step of each: $\mathbf{H}^{1/2} = \mathbf{H} + h\mathcal{D}\mathbf{H}$, $\mathbf{H}^{(\ell+1)} = \mathbf{H}^{1/2} + h\mathcal{R}\mathbf{H}^{1/2}$. Because both operators are evaluated at the *same* $\mathbf{H}^{(\ell)}$ in the standard implementation, the result is exactly (7). The local truncation error is $\mathcal{O}(h^2[\mathcal{D}, \mathcal{R}])$, i.e. first order. $\blacksquare$

**Exercise 5.** A single Turing instability condition can be checked numerically: pick parameters $(D_u, D_v, F, k)$ for Gray-Scott, linearise around the homogeneous steady state, sweep $|\mathbf{k}|^2$ and look for sign changes of $\det\,\mathbf{A}(|\mathbf{k}|^2)$. Implement this and reproduce a row of Fig. 1.

> *Solution sketch.* The homogeneous steady state of Gray-Scott solves $u v^2 = F(1 - u)$ and $u v^2 = (F + k) v$. Compute the $2\times 2$ Jacobian of the kinetics at this state, build $\mathbf{A}(|\mathbf{k}|^2) = \mathbf{J} - |\mathbf{k}|^2 \mathrm{diag}(D_u, D_v)$, and plot $\det\,\mathbf{A}$ vs $|\mathbf{k}|^2$. A negative dip indicates an unstable band; the corresponding wavelength $2\pi/|\mathbf{k}_*|$ matches the visual scale of the simulated patterns. $\blacksquare$

---

## References

[1] Turing, A. M. (1952). *The chemical basis of morphogenesis.* Phil. Trans. R. Soc. B, 237(641), 37-72.

[2] Pearson, J. E. (1993). *Complex patterns in a simple system.* Science, 261(5118), 189-192.

[3] Murray, J. D. (2003). *Mathematical biology II: Spatial models and biomedical applications* (3rd ed.). Springer.

[4] Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks.* ICLR. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).

[5] Oono, K., & Suzuki, T. (2020). *Graph neural networks exponentially lose expressive power for node classification.* ICLR. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947).

[6] Chamberlain, B., Rowbottom, J., Gorinova, M., Webb, S., Rossi, E., & Bronstein, M. (2021). *GRAND: Graph neural diffusion.* ICML. [arXiv:2106.10934](https://arxiv.org/abs/2106.10934).

[7] Thorpe, M., Nguyen, T. M., Xia, H., Strohmer, T., Bertozzi, A., Osher, S., & Wang, B. (2022). *GRAND++: Graph neural diffusion with a source term.* ICLR.

[8] Eliasof, M., Haber, E., & Treister, E. (2021). *PDE-GCN: Novel architectures for graph neural networks motivated by partial differential equations.* NeurIPS. [arXiv:2108.01938](https://arxiv.org/abs/2108.01938).

[9] Di Giovanni, F., Rowbottom, J., Chamberlain, B., Markovich, T., & Bronstein, M. (2022). *Graph neural networks as gradient flows.* [arXiv:2206.10991](https://arxiv.org/abs/2206.10991).

[10] Choi, J., Hong, S., Park, N., & Cho, S.-B. (2023). *GREAD: Graph neural reaction-diffusion networks.* ICML. [arXiv:2211.14208](https://arxiv.org/abs/2211.14208).

[11] Eliasof, M., Haber, E., & Treister, E. (2024). *Graph neural reaction-diffusion models.* SIAM J. Sci. Comput. [arXiv:2406.10871](https://arxiv.org/abs/2406.10871).

---

*This is Part 8 — the final part — of the [PDE and Machine Learning](/categories/PDE-and-Machine-Learning/) series. Previous: [Part 7 — Diffusion Models and Score Matching](/en/PDE-and-Machine-Learning-7-Diffusion-Models/). Start from the beginning: [Part 1 — Physics-Informed Neural Networks](/en/PDE-and-Machine-Learning-1-Physics-Informed-Neural-Networks/). Thanks for reading.*
