---
title: "Graph Neural Networks for Learning Equivariant Representations of Neural Networks"
date: 2024-12-20 09:00:00
tags:
  - Meta-Learning
  - Representation Learning
  - GNN
categories: Paper
lang: en
mathjax: true
description: "Represent a neural network as a directed graph (neurons as nodes, weights as edges) and use a GNN to produce permutation-equivariant embeddings. The right symmetry unlocks generalisation prediction, network classification, retrieval, and model merging across architectures and widths."
disableNunjucks: true
---

You can shuffle the hidden neurons of a trained MLP and get the *exact* same function back -- but the flat parameter vector now looks completely different. This single fact ruins most attempts at "learning over neural networks": naive representations treat two functionally identical models as two unrelated points in parameter space, and the downstream learner wastes capacity rediscovering a symmetry it should have for free. This paper -- *Graph Neural Networks for Learning Equivariant Representations of Neural Networks* (Kofinas et al., ICML 2024) -- proposes the clean fix: turn the network itself into a graph, then use a GNN whose architecture *natively* respects the relevant permutation symmetry.

## What you will learn

- Why hidden-neuron permutations are the right symmetry to design against
- How an MLP, CNN, or Transformer maps onto a single typed graph -- the *neural graph*
- What "equivariant" means here, formally and operationally
- How message passing on a neural graph is constructed, and what `PNA + FiLM` adds
- Four downstream tasks where the equivariance pays off: predicting generalisation, classifying networks, retrieving similar models, merging weights
- Practical concerns: probe features, normalisation, positional embeddings, scaling

## Prerequisites

- Basic GNN literacy (message passing, node features, pooling)
- Standard MLP / CNN / Transformer structure
- Comfort with the words *invariance* and *equivariance*

---

# Why equivariance matters for "learning over networks"

A growing class of tasks treats an entire trained neural network as a single data point:

- **Predicting generalisation** from weights, without re-running validation
- **Classifying networks** by task, dataset, or training recipe (SGD vs Adam, ResNet vs VGG, ...)
- **Retrieving similar networks** from a model zoo by functional similarity
- **Meta-learning** across populations of trained models
- **Model merging**: combining the weights of independently trained models that solve the same task

All five share the same nuisance: an MLP has a *huge* discrete symmetry group acting on its parameters that leaves the function unchanged. For a single hidden layer with permutation matrix $P$,

$$
f(x;\,W_1, b_1, W_2, b_2) \;=\; f(x;\,P W_1, P b_1, W_2 P^\top, b_2),
$$

and the same applies, independently, to every hidden layer. With per-layer widths $n_1, \ldots, n_L$, the symmetry group is the direct product

$$
\mathcal{S} \;=\; S_{n_1} \times S_{n_2} \times \cdots \times S_{n_L},
$$

so the number of equivalent parameter vectors representing the same function explodes combinatorially. A learner that ignores $\mathcal{S}$ either has to (i) memorise all its orbits, which is hopeless, or (ii) hope that its training distribution happens to cover them, which is naive.

![Permutation equivariance: hidden-unit symmetry of an MLP](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gnn-equivariant-representations/fig1_permutation_equivariance.png)

Figure 1 makes the symmetry concrete. Permute the three hidden units in any order, permute the rows of $W_1$ and the columns of $W_2$ correspondingly, and the function $f(x)$ is byte-identical -- yet `vec(W_1, b_1, W_2, b_2)` is a wildly different point in $\mathbb{R}^d$.

## Why the obvious baselines fail

Three baselines occur to almost everyone, and each fails for a different reason.

**Flatten the weights into a vector.** Concatenate every parameter into a single $\theta \in \mathbb{R}^d$ and feed it to an MLP. This is *not* permutation-equivariant: shuffling hidden units changes $\theta$ entirely. It also has no notion of architecture: the dimension $d$ depends on widths and depths, so two networks of different sizes cannot even be compared in the same space.

**Aggregate weight statistics.** Compute means, variances, histograms, or moments of each weight tensor and feed those. This *is* invariant to hidden permutations, but it throws away all relational information -- which weights connect which neurons -- and collapses functionally distinct networks with similar weight distributions to the same point.

**Treat the weight matrix as an image.** Apply a CNN over the matrix grid. CNNs are translation-equivariant on a 2D grid, but the hidden-unit symmetry is permutation, not translation -- the rows and columns of $W$ are *not* arranged on a regular lattice. The architecture also assumes a fixed shape, so it cannot transfer across widths.

The pattern is the same: the wrong symmetry, the wrong invariant, or the wrong topology. We need a representation that *is* the symmetry.

# Neural graphs: turning weights into a typed graph

The fix is to build a directed graph whose structure mirrors the computation graph of the network itself.

## Construction for an MLP

For an MLP with widths $n_0, n_1, \ldots, n_L$, weights $W_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, and biases $b_\ell \in \mathbb{R}^{n_\ell}$:

- **Nodes**: one per neuron, $\sum_\ell n_\ell$ in total.
- **Node features** $V$: the bias of that neuron, optionally concatenated with positional / type embeddings (input vs hidden vs output, layer index, activation type).
- **Edges**: one per weight, directed from the source neuron to the target neuron.
- **Edge features** $E$: the scalar weight value, optionally concatenated with edge-type embeddings (forward vs residual, conv vs linear, ...).

![MLP weights vs the same MLP as a neural graph](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gnn-equivariant-representations/fig2_neural_graph.png)

Figure 2 contrasts the two representations. On the left, the parameters live as separate tensors $(W_1, b_1, W_2, b_2)$, and the only natural way to feed them to a downstream learner is `vec(...)` -- which loses topology. On the right, the *same* parameters are arranged on a graph. The bias of $h_1$ becomes a node feature; the weight from $x_1$ to $h_1$ becomes an edge feature. The graph as a structured object is *invariant* to relabelling the hidden nodes, because graph identity is up to isomorphism: the labels `h1, h2, h3` are arbitrary.

## Why this is the right object

The neural graph carries exactly the right symmetry. The symmetry group of the graph -- relabelling all $N = \sum_\ell n_\ell$ nodes -- is $S_N$, much larger than the per-layer product $\mathcal{S}$ that the MLP actually has. But that is a feature, not a bug:

- $\mathcal{S}$ is a *subgroup* of $S_N$ (you can only legally permute *within* a layer, not across).
- Any model that is $S_N$-equivariant is automatically $\mathcal{S}$-equivariant.
- A single $S_N$-equivariant model can therefore handle *many* different architectures (different per-layer widths) without retraining or redesign.

This is the practical headline of the paper: prior work (e.g. DeepSet-style permutation networks) was designed for one fixed $\mathcal{S}$, and a model trained for one width could not consume another. The neural-graph + GNN approach handles the entire family at once.

## Extending to CNNs, Transformers, and the rest

The same recipe extends with small adjustments:

- **Convolutions.** A conv layer with kernel shape $c_\text{out} \times c_\text{in} \times k \times k$ becomes a bipartite block of edges between $c_\text{in}$ source channels and $c_\text{out}$ target channels. The spatial $k \times k$ kernel is flattened into a vector and used as multi-dimensional edge features. To handle different kernel sizes in the same model, all kernels are zero-padded to the largest one.
- **Flatten + linear head.** A linear layer on a flattened conv output is treated as a $1 \times 1$ conv, which makes it look identical in graph form. Adaptive pooling absorbs varying spatial resolutions so the graph topology is independent of input image size.
- **Normalisation layers.** A `LayerNorm` or `BatchNorm` with scale $\gamma$ and bias $\beta$ becomes a *diagonal* edge block (one edge per channel, edge feature $= \gamma_i$) plus per-output-node biases $\beta_i$. This preserves the diagonal structure exactly.
- **Residual connections.** A skip $y = x + f(x)$ adds an edge with feature $1$ from each source node to the matching destination node, which is mathematically the identity matrix made explicit.
- **Attention.** Multi-head self-attention has parameters $W_Q^h, W_K^h, W_V^h, W_O$. Each becomes a typed edge block over input / per-head / output node groups. The attention computation itself is parameter-free, so it is *not* explicitly modelled in the graph -- the GNN approximates its effect.
- **Activations.** Non-linearity type per layer (ReLU, GELU, SiLU, ...) becomes a learned embedding added to the corresponding node features.

The point is that one uniform graph language captures every standard layer type, so the *same* GNN can ingest MLPs, CNNs, ResNets, and Transformers with no architecture-specific code in the downstream learner.

# Equivariance: the formal property and its operational meaning

It is worth pinning down exactly what is being preserved.

For a function $f$ on graphs and a permutation $\pi$ of node labels:

- $f$ is **invariant** if $f(\pi \cdot G) = f(G)$. Use this for *graph-level* outputs (one prediction per network).
- $f$ is **equivariant** if $f(\pi \cdot G) = \pi \cdot f(G)$. Use this for *node-level* outputs (one prediction per neuron).

Standard message-passing GNNs are equivariant by construction: the update at node $v$,

$$
h_v^{(\ell+1)} \;=\; \mathrm{UPDATE}\!\left(\,h_v^{(\ell)},\;\bigoplus_{u \in \mathcal{N}(v)} \mathrm{MSG}(h_u^{(\ell)}, e_{uv})\,\right),
$$

is identical for every node and uses a permutation-invariant aggregator $\bigoplus$ (sum / mean / max / attention). Relabelling the nodes therefore commutes with the GNN: permuted input graph $\Rightarrow$ permuted node embeddings, same edge structure $\Rightarrow$ same scalar functions of those embeddings.

![Invariance vs equivariance: same symmetry, different output](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gnn-equivariant-representations/fig4_equivariant_vs_invariant.png)

Figure 4 makes the distinction operational. On the left, a graph-level pooling (sum, mean, attention) crushes the node-embedding matrix into a single vector $z_G$ that is *the same* whether you permuted the nodes or not -- this is what you want for "predict generalisation" or "classify task". On the right, the per-node embedding matrix $Z(G)$ permutes *along with* the input -- this is what you need for "align neurons across two networks", which underlies model merging and architecture editing.

The slogan: **equivariance is the stronger property; invariance is what you usually report at the end**. The cleanest pipeline therefore keeps message passing equivariant throughout, then applies a single invariant pool only when a graph-level prediction is required.

# Architecture: GNN + Transformer variants for neural graphs

The paper considers two backbones, both adapted to the unusual fact that *edge* features carry the bulk of the information (weights are the parameters, after all).

## NG-GNN: PNA with edge updates and FiLM modulation

The base is **PNA** (Principal Neighborhood Aggregation), chosen because it supports edge features and combines several aggregators in parallel (mean, max, std, scaled by node degree). The standard PNA does not *update* edges; the paper adds a per-layer edge MLP

$$
e_{uv}^{(\ell+1)} \;=\; \phi^{(\ell)}_E\!\left(\,e_{uv}^{(\ell)},\, h_u^{(\ell)},\, h_v^{(\ell)}\,\right),
$$

so that edge features evolve through depth alongside node features. To strengthen the multiplicative interaction between the weight (edge) and the neuron states (nodes), the message uses **FiLM** modulation:

$$
\mathrm{MSG}(h_u, e_{uv}) \;=\; (\gamma(e_{uv}) \odot h_u) + \beta(e_{uv}),
$$

where $\gamma, \beta$ are small MLPs. This lets the *weight* gate the message coming from the *source neuron*, which mirrors what a real network actually does at inference.

## NG-T: a relational Transformer

The Transformer variant treats the neural graph as a fully connected graph and uses **relational attention**: edge features enter the attention computation as a bias on the value matrix,

$$
V_{uv} \;=\; (\gamma(e_{uv}) \odot V_u) + \beta(e_{uv}),
$$

so the attention from $v$ to $u$ is conditioned on the *weight* connecting them. This is the same FiLM trick, ported to attention. Empirically NG-T tends to be stronger on dense graphs (small networks) and NG-GNN scales better to large, sparse ones.

## End-to-end pipeline

![End-to-end GNN pipeline for processing neural-net parameters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gnn-equivariant-representations/fig3_gnn_pipeline.png)

Figure 3 puts the five stages in order: take the trained network, build its neural graph, run $L$ layers of equivariant message passing with edge updates, pool to a graph embedding, and feed a small MLP head. The equivariance is built in at stage 3 -- everything before is data, everything after either preserves it (more equivariant ops) or collapses it on purpose (a single pool).

# Engineering details that matter

These look minor on paper but the ablations show they each matter.

**Probe features.** For each network, run a fixed set of $k$ probe inputs forward and record every intermediate activation; concatenate the per-neuron activation vector to that neuron's node feature. This injects functional information that is *also* permutation-equivariant (probes interact only with neurons, not with their labels), and it is trivially preserved under symmetries of the network that preserve its function. In practice the probes are *learned* -- you can backprop into the probe inputs -- and they give a substantial boost on tasks where weight statistics alone underspecify the function.

**Normalisation that respects the symmetry.** Most parameter-space methods normalise by the per-neuron mean and std computed across the training set. That is a *symmetry-breaking* operation: there is no such thing as "neuron 7 across networks" if neurons are permutable. The fix is to normalise per *layer* instead -- one mean / std per layer for weights, one for biases -- so the statistics are themselves $\mathcal{S}$-invariant.

**Positional embeddings without breaking permutation.** Each node gets a learned positional embedding tied to its *layer index*, not its in-layer index. All nodes in the same hidden layer share the same positional vector, so within-layer permutation symmetry is preserved. Input and output nodes get *unique* positional embeddings, because permuting them *would* change the function (they are the externally visible interface).

**Reverse edges.** Adding edges in the reverse direction (with their own learned type embedding) doubles the message-passing bandwidth and lets gradient-flow-like information propagate backward through the graph in a single layer rather than $L$ layers. Cheap, consistent gain.

# Downstream tasks: what equivariance buys you

## 1. Predicting generalisation

The setup: take a model zoo of trained networks with known test accuracies, train the GNN to regress test accuracy from weights alone.

![Predicting generalisation from weights alone](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gnn-equivariant-representations/fig5_generalization_prediction.png)

Figure 5 shows the qualitative gap between an equivariant predictor (tight scatter around $y = x$) and a flat-vector MLP baseline (a noisy cloud that struggles to discriminate). Equivariance prevents the model from "wasting" its sample efficiency on parameter permutations and lets it focus on the parts of the weights that actually correlate with generalisation -- spectra, alignment of layers, sharpness proxies.

## 2. Classifying networks by behaviour

Same pipeline, classification head: predict which dataset / task / optimiser produced the network. The interesting result is that the embedding learned for one classification task transfers to others: the GNN learns a general "feature space of networks", not just a task-specific decision boundary.

## 3. Retrieving similar networks

Embed every model in a zoo with the GNN, then use cosine similarity for retrieval. Functionally similar networks (e.g. two CIFAR-10 classifiers trained from different seeds) end up close, *despite* having parameter-space distances that are essentially random. This is exactly what equivariance is supposed to give you: the embedding metric is induced by the *function* the network computes, not by its arbitrary parameterisation.

## 4. Model merging via neuron alignment

Use the *equivariant* (per-node) embeddings, not the pooled vector. Match neurons between two networks by Hungarian / optimal transport on their node-embedding distance, then merge weights along the resulting alignment. The traditional approach (activation matching with probe inputs) becomes a *special case*: probe activations are one of the node features the GNN consumes.

# How it stacks up vs the baselines

| Method | Equivariant? | Cross-architecture? | Captures topology? |
| --- | --- | --- | --- |
| Flatten weights + MLP | No | No (dim depends on widths) | No |
| Weight statistics | Yes (invariant only) | Yes | No (loses relations) |
| CNN over weight matrix | No (translation $\ne$ permutation) | Partial | Partial |
| DeepSet-style per-layer permutation nets | Yes (one fixed $\mathcal{S}$) | No (one architecture only) | Partial |
| **Neural graph + GNN (this paper)** | **Yes ($S_N$-equivariant)** | **Yes** | **Yes** |

The empirical pattern in the paper is consistent with this table: the GNN approach matches or beats specialised per-architecture methods on each architecture, and is the only one that handles *all* of them at once.

# Limitations and open questions

- **Scale.** A neural graph for a billion-parameter model has a billion edges. Sparse GNN libraries help, but the present formulation is comfortable in the millions of parameters, not billions. Layerwise / blockwise neural graphs are the obvious next step.
- **Architecture coverage.** The paper covers MLPs, CNNs, ResNets, and Transformers; arbitrary computation graphs (mixtures of experts, dynamic routing, recursive structures) are open.
- **Probe design.** Probes are learned, but what *type* of probes -- adversarial, random, in-distribution, OOD -- is best for which downstream task is mostly empirical so far.
- **Behaviour under non-symmetric initialisations.** The story assumes parameters that respect $\mathcal{S}$-orbit structure. Specific weight-tying schemes or structured sparsity may break that assumption and need modelling.

# Takeaways

1. **The right symmetry to design against is per-layer hidden permutation**, not "all of $\theta$". The neural graph encodes that symmetry exactly.
2. **GNNs are permutation-equivariant for free**; combining them with a graph that mirrors network topology gives you the right inductive bias automatically -- no special parameter-sharing scheme needed.
3. **One model, many architectures.** Because $\mathcal{S}$ is a subgroup of the full $S_N$ that the GNN is equivariant to, the same trained GNN consumes any compatible architecture.
4. **Pool *only at the end*.** Keep node-level equivariance through every message-passing layer; collapse with a single invariant pool when (and only when) you need a graph-level scalar.
5. **The big wins are tasks that meta-process networks**: predicting generalisation, retrieving similar models, merging weights. All of them previously required either ignoring symmetry or hand-coding it.

# Further reading

- Original paper: [Graph Neural Networks for Learning Equivariant Representations of Neural Networks](https://arxiv.org/abs/2403.12143) (Kofinas et al., ICML 2024).
- Background on PNA: [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) (Corso et al., NeurIPS 2020).
- FiLM modulation: [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) (Perez et al., AAAI 2018).
- Permutation-invariant networks for parameter spaces: [Equivariant Architectures for Learning in Deep Weight Spaces](https://arxiv.org/abs/2301.12780) (Navon et al., ICML 2023) and [Permutation Equivariant Neural Functionals](https://arxiv.org/abs/2302.14040) (Zhou et al., NeurIPS 2023).
- Generalisation prediction from weights: [Predicting Neural Network Accuracy from Weights](https://arxiv.org/abs/2002.11448) (Unterthiner et al., 2020).
- Model merging via permutation: [Git Re-Basin](https://arxiv.org/abs/2209.04836) (Ainsworth et al., ICLR 2023).
