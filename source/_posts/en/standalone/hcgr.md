---
title: "HCGR: Hyperbolic Contrastive Graph Representation Learning for Session-based Recommendation"
date: 2024-12-18 09:00:00
tags:
  - Contrastive Learning
  - Recommender Systems
  - GNN
  - Hyperbolic Geometry
categories: Paper
lang: en
mathjax: true
description: "HCGR embeds session graphs in the Lorentz model of hyperbolic space and trains them with InfoNCE-style contrastive learning. This review unpacks why hierarchical session intent fits hyperbolic geometry, how Lorentz attention works in tangent space, and what the ablations actually prove."
disableNunjucks: true
---

A user opens a sneaker app, taps "running shoes", drills into a brand, then a price band, then a single SKU. That trajectory is a *tree*: each click narrows the candidate set roughly multiplicatively. In Euclidean space you need many dimensions to keep all the leaves of that tree apart, because Euclidean volume only grows polynomially with radius. In hyperbolic space volume grows *exponentially* with radius, so the tree fits naturally — a few dimensions are enough to keep the whole long tail untangled.

[**HCGR**](https://arxiv.org/abs/2107.05366) (Guo et al., 2021) takes this seriously. It embeds session-graph nodes on the **Lorentz hyperboloid**, runs an attention-weighted GNN aggregator in the tangent space, and adds a contrastive auxiliary loss that pulls together two augmented views of the same session while pushing other sessions away. The result is a session recommender that beats strong Euclidean GNN baselines like SR-GNN and GCE-GNN, with the largest gains exactly where hyperbolic geometry should help: long-tail items and deep-hierarchy datasets like Last.FM.

## What you will learn

- Why session intent has the structure that hyperbolic geometry was built for
- The Lorentz model: distance, tangent space, exponential and logarithmic maps in a form you can implement
- How HCGR does GNN aggregation and attention without ever leaving the manifold
- The two-view contrastive objective and how it interacts with hyperbolic distance
- How to read HCGR-style ablations critically (Euclidean vs hyperbolic at *matched* capacity, with vs without contrastive)
- When this machinery is worth its complexity, and when plain SR-GNN is enough

## Prerequisites

- Session-based recommendation basics (ideally SR-GNN or GCE-GNN)
- Standard GNN message passing
- Light differential-geometry intuition; we keep formulas operational rather than rigorous

---

## 1. Why session intent fits hyperbolic geometry

Three properties of session-recommendation data keep showing up:

1. **Power-law popularity.** A handful of head items absorb most clicks; the long tail is enormous and sparsely interacted.
2. **Taxonomic narrowing.** A session usually walks down a tree: category → subcategory → attribute → SKU.
3. **Exponential branching.** As you move from a coarse concept outward, the number of fine-grained candidates explodes.

These are the defining properties of *trees*, and trees do not embed cleanly in Euclidean space. To see why, take a balanced binary tree of depth $L$. It has $2^L$ leaves but only $L$ "ring" levels. In a 2D Euclidean plane, the number of points that fit at radius $r$ with separation $\delta$ is $O(r/\delta)$ — linear. In hyperbolic space the boundary of the disk at radius $r$ has length $\sinh(r) \sim e^r$, so it fits $O(e^r/\delta)$ separated points — exponential. The tree fits; nothing has to be squeezed.

{% asset_img fig1_poincare_vs_euclidean.png "Embedding the same depth-4 binary tree in the Euclidean plane (left) and the Poincaré disk (right). On the disk, every leaf has room because the boundary expands exponentially." %}

The picture above is the elevator pitch for hyperbolic embeddings. On the left, the Euclidean tree runs out of room at depth 4 — leaves crowd together and the embedding has to either grow the radius or invent extra dimensions. On the right, the Poincaré disk pushes leaves toward the boundary where exponential length absorbs them with even spacing.

## 2. Session graphs, briefly

HCGR is a *graph* model. Given a session $s = [v_1, v_2, \dots, v_n]$ it constructs a directed session graph $G_s = (V_s, E_s)$ where:

- nodes are unique items in the session,
- edges are observed transitions $v_i \to v_{i+1}$, optionally weighted by frequency.

This is the SR-GNN-family setup, and HCGR keeps everything that already works about it: local transition structure is captured by message passing on $G_s$. The only thing HCGR changes is the *geometry of the embeddings* and the *training signal*.

## 3. The Lorentz model in operational form

There are several equivalent models of hyperbolic geometry — the Poincaré ball, the Klein model, and the Lorentz (a.k.a. hyperboloid) model. HCGR uses **Lorentz** because its formulas are more numerically stable than the Poincaré ball's and gradients behave better near the boundary.

### 3.1 The hyperboloid

Define the Lorentzian inner product on $\mathbb{R}^{d+1}$:

$$
\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} \;=\; -x_0 y_0 + \sum_{i=1}^{d} x_i y_i.
$$

The hyperboloid (curvature $c = -1$ for simplicity) is the upper sheet:

$$
\mathbb{H}^d \;=\; \bigl\{\, \mathbf{x} \in \mathbb{R}^{d+1} \;:\; \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -1,\; x_0 > 0 \,\bigr\}.
$$

Concretely, an embedding is a $(d+1)$-vector that lives on this curved surface. The "extra" dimension is the price you pay to write hyperbolic operations as clean linear algebra in an ambient Euclidean space.

### 3.2 Distance

The Lorentz distance is

$$
d_{\mathcal{L}}(\mathbf{x}, \mathbf{y}) \;=\; \mathrm{arcosh}\!\bigl( -\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} \bigr).
$$

The key qualitative fact: $d_{\mathcal{L}}$ grows roughly like $\mathrm{arcosh}$ of an inner product that itself can grow exponentially with how far points are pushed up the hyperboloid. So distances expand fast, which is exactly what we want when we are trying to keep an exponentially branching tree apart.

### 3.3 Tangent space, exp and log

You cannot do gradient descent directly on a curved manifold without leaving it. The standard trick is to operate in the **tangent space** at a base point (usually the origin $\mathbf{o} = (1, 0, \dots, 0)$), which is locally Euclidean, then map back.

- **Logarithmic map** $\log_{\mathbf{o}}: \mathbb{H}^d \to T_{\mathbf{o}}\mathbb{H}^d$ — projects a manifold point into the tangent plane.
- **Exponential map** $\exp_{\mathbf{o}}: T_{\mathbf{o}}\mathbb{H}^d \to \mathbb{H}^d$ — its inverse, lifts a tangent vector back onto the manifold.

You don't need the closed forms by heart. The pattern that matters: anything you would do in a Euclidean GNN — sum, attention, MLP — happens in the tangent space, and you `exp_map` the result back. This is the only way "addition" makes sense on the hyperboloid.

## 4. HCGR end-to-end

Here is the full pipeline before we dive into pieces:

{% asset_img fig2_hcgr_architecture.png "HCGR pipeline: session sequence → session graph → Lorentz embeddings → tangent-space attention aggregation → session readout → next-item scoring (CE) plus a contrastive auxiliary on two augmented views." %}

Three things are happening in parallel:

1. **Geometric backbone.** Items live on $\mathbb{H}^d$. All updates are pulled into the tangent space, recombined, then pushed back.
2. **Recommendation head.** A session readout produces $\mathbf{s}$, and the next-item score is the negative Lorentz distance to each candidate item embedding. Cross-entropy supervises the click target.
3. **Contrastive head.** Two augmented views of $G_s$ are encoded into $\mathbf{s}^a$ and $\mathbf{s}^b$. An InfoNCE loss pulls them together and pushes other sessions in the batch away.

### 4.1 Hyperbolic GNN aggregation

In a vanilla GAT-style aggregator you would do $\mathbf{h}_i = \sigma\!\bigl(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W \mathbf{h}_j\bigr)$. You cannot sum points on the hyperboloid directly: their sum doesn't lie on the manifold. HCGR does the obvious workaround:

$$
\mathbf{h}_i^{(l+1)} \;=\; \exp_{\mathbf{o}}\!\Biggl( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} \, \log_{\mathbf{o}}\!\bigl(\mathbf{h}_j^{(l)}\bigr) \Biggr).
$$

Read it from inside out: bring neighbours into the tangent space, attention-weighted-sum them there, exp them back. Attention weights $\alpha_{ij}$ are computed from the tangent vectors with the standard GAT trick (a learned linear layer on concatenated features, softmax over neighbours).

A separate parallel-transport step is needed if you want to move tangent vectors between different base points — HCGR uses it to keep multi-layer aggregation consistent — but conceptually nothing changes.

### 4.2 Non-linearity that respects curvature

Plain ReLU on a hyperboloid coordinate vector is meaningless. HCGR threads the activation through the tangent map of one layer's curvature and the exp map of the next layer's:

$$
\sigma_{\mathbb{H}}^{l \to l+1}(\mathbf{x}) \;=\; \exp_{\mathbf{o}}^{c_{l+1}}\!\Bigl(\, \sigma\!\bigl(\, \log_{\mathbf{o}}^{c_l}(\mathbf{x})\,\bigr)\, \Bigr).
$$

This is the standard "tangent-space activation" pattern from hyperbolic neural networks. It lets you stack layers with different curvatures while keeping every intermediate state on a valid manifold.

## 5. The contrastive auxiliary

Session graphs are noisy. A real session has accidental clicks, repeated items, exploration, and back-tracks. Pure cross-entropy on the next item amplifies that noise: anything that helps predict the click target is reinforced, even if the *representation* of the session is brittle. Contrastive learning fixes this by enforcing a structural property of the encoder: two perturbations of the *same* session should land in the same place; two *different* sessions should not.

{% asset_img fig3_contrastive_views.png "Two-view contrastive scheme. Edge dropout and node dropout each produce one view; both views go through the HCGR encoder; InfoNCE pulls the positive pair together and pushes negatives apart, all measured by Lorentz distance in the disk." %}

### 5.1 Augmentations

HCGR uses graph-level augmentations on the session graph $G_s$:

- **Edge dropout** — drop a random fraction of transitions.
- **Node dropout** — remove a small set of items and inherit their incoming/outgoing edges where appropriate.

Two independent augmentations of $G_s$ produce views $G_s^a$ and $G_s^b$. Both are encoded by the same HCGR network into session vectors $\mathbf{s}^a, \mathbf{s}^b \in \mathbb{H}^d$.

### 5.2 InfoNCE in hyperbolic space

The contrastive loss is the standard InfoNCE form, but with similarity defined through hyperbolic distance:

$$
\mathcal{L}_{\mathrm{cl}} \;=\; -\, \log \frac{\exp\!\bigl( \mathrm{sim}(\mathbf{s}^a, \mathbf{s}^b) / \tau \bigr)}{\sum_{k} \exp\!\bigl( \mathrm{sim}(\mathbf{s}^a, \mathbf{s}^b_k) / \tau \bigr)},
$$

where $\mathrm{sim}(\mathbf{u}, \mathbf{v}) = -\, d_{\mathcal{L}}(\mathbf{u}, \mathbf{v})$ (or, for stability, an inner product in the tangent space at $\mathbf{o}$) and $\tau$ is the InfoNCE temperature. The denominator runs over all sessions in the mini-batch, treating other sessions as negatives.

### 5.3 Total objective

The recommendation cross-entropy and contrastive auxiliary are simply added:

$$
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{rec}} \;+\; \lambda \, \mathcal{L}_{\mathrm{cl}}.
$$

A typical $\lambda$ is small (0.05–0.2). The contrastive loss is *regularising* the encoder, not replacing the supervision; making it too large hurts ranking quality.

## 6. Distance and dimension: the capacity argument

Whether HCGR is "really" using hyperbolic geometry or just stacking parameters comes down to one question: does the same hierarchy fit in fewer dimensions?

{% asset_img fig4_distance_growth.png "Left: pairwise distance grows linearly in the Euclidean plane (blue) but exponentially with radius in the hyperbolic plane (purple). Right: capacity for hierarchical structure grows much faster with embedding dimension in hyperbolic space." %}

The left panel is the geometry: $\sinh(r)$ versus $r$. Past $r \approx 1.5$ the curves diverge sharply, so two items at the periphery of the manifold are pushed apart almost for free. The right panel is the practical consequence in a recommender: at $d = 16$ Euclidean embeddings still struggle to keep a deep taxonomy untangled, while a 16-dim hyperbolic embedding has room to spare. This is why HCGR usually wins more on long-tail-heavy datasets (Last.FM) than on shorter, fatter sessions (Yoochoose).

## 7. What the numbers actually show

{% asset_img fig5_performance.png "Indicative Recall@20 and MRR@20 across three standard session datasets. The hyperbolic-only ablation already beats Euclidean baselines; full HCGR (hyperbolic + contrastive) widens the gap." %}

Two patterns are worth flagging from the paper's own ablations:

- **Geometry alone helps.** Replacing the Euclidean GNN with the Lorentz aggregator (no contrastive loss) already moves Recall@20 by a meaningful margin. This is the cleanest evidence that the gain is *geometric*, not just "more parameters in a fashionable shape".
- **Contrastive on top is consistent.** The contrastive auxiliary delivers further improvement across all three datasets. The lift is largest where session graphs are noisiest, which is also where you would expect view-based augmentation to help most.

## 8. Reading HCGR-style results critically

If you're reviewing HCGR or any other hyperbolic recommender, here is the checklist I run through:

1. **Matched capacity.** Compare Euclidean and hyperbolic at the *same* embedding dimension and the *same* parameter count. Hyperbolic often wins at lower $d$ but ties at high $d$ — the headline matters less than the curve.
2. **Where the gain lives.** Ask for a head-vs-tail breakdown. The whole motivation of hyperbolic geometry is the long tail; if all of the gain is on head items, something else is doing the work.
3. **Ablation of each piece.** Hyperbolic alone, contrastive alone, both. Without this, you cannot tell whether it's the geometry or the contrastive trick.
4. **Reproducibility.** Lorentz optimisation can be touchy. Look for fixed seeds, multiple runs, and at least one mention of stability tricks (clipping, careful initialisation near the origin, projection back to the manifold).
5. **Curvature sensitivity.** Curvature $c$ is a hyperparameter. If results require fine curvature tuning per dataset, deployment will be painful.

## 9. Practical takeaways

If you're considering this for a real session recommender:

- **Start Euclidean.** A well-tuned SR-GNN or GCE-GNN is a strong baseline. Don't pay for hyperbolic complexity until you have a clean Euclidean number.
- **Bring in hyperbolic when your data is genuinely hierarchical.** Catalogue with deep taxonomy, heavy long tail, multi-level intent — these are the wins.
- **Add contrastive whenever you can define meaningful augmentations.** Edge dropout is almost free and tends to help on its own, even with Euclidean encoders.
- **Budget for instability.** Manifold projection, gradient clipping, and a warm-up schedule on $\lambda$ are all worth implementing from day one. Hyperbolic optimisation diverges quietly.
- **Tune three knobs deliberately.** Curvature $c$, contrastive temperature $\tau$, and contrastive weight $\lambda$. The other dials matter much less.

## 10. Limitations and open questions

The paper is honest about the costs, and so should we be:

- **Computational overhead.** Every layer pays for `log_map` and `exp_map`. On large catalogues with deep encoders the hit is real.
- **Optimisation fragility.** Naive SGD can fall off the manifold; Riemannian optimisers help but add a second-order configuration burden.
- **Tooling gap.** PyTorch and TensorFlow are Euclidean by default. Geoopt and similar libraries close the gap, but you are off the well-trodden path.
- **Interpretability.** "Closer in Lorentz distance" is harder to explain to a product team than "cosine similarity above threshold".

For most teams, the right reading of HCGR is: *the geometry is the contribution, the contrastive part is a portable trick.* Steal the contrastive auxiliary for your existing Euclidean recommender first; pay the hyperbolic tax only when the data shape clearly demands it.

## References

- Guo, Tang, Yu, Xu, Yu, Yang, Lu. *HCGR: Hyperbolic Contrastive Graph Representation Learning for Session-based Recommendation*. 2021. [arXiv:2107.05366](https://arxiv.org/abs/2107.05366)
- Wu et al. *Session-based Recommendation with Graph Neural Networks (SR-GNN)*. AAAI 2019.
- Wang et al. *Global Context Enhanced Graph Neural Networks for Session-based Recommendation (GCE-GNN)*. SIGIR 2020.
- Nickel and Kiela. *Poincaré Embeddings for Learning Hierarchical Representations*. NeurIPS 2017.
- Ganea, Bécigneul, Hofmann. *Hyperbolic Neural Networks*. NeurIPS 2018.
- Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*. ICML 2020.
