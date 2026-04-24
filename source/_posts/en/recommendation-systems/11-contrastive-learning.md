---
title: "Recommendation Systems (11): Contrastive Learning and Self-Supervised Learning"
date: 2024-05-12 09:00:00
tags:
  - Recommendation Systems
  - Contrastive Learning
  - Self-Supervised
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 11
  total: 16
lang: en
mathjax: true
description: "A practitioner's guide to contrastive learning for recommendations: InfoNCE and the role of temperature, SimCLR vs MoCo negatives, SGL graph augmentations, CL4SRec sequence augmentations, XSimGCL's noise-only trick, with intuition, math, and clean PyTorch."
disableNunjucks: true
---

Classical recommenders learn from one signal: did a user click, watch, or buy? That signal is precious, but it is also brutally sparse. Most users touch fewer than 1% of the catalogue, most items are touched by fewer than 0.1% of users, and a brand-new item or user has nothing at all. Optimising a model directly against such sparse labels almost guarantees overfitting on the head and silence on the tail.

Contrastive learning offers a different bargain. Instead of asking "what label should this example have?", it asks "which two examples should look alike, and which two should look different?". That question is cheap to answer — you can derive it from the data itself by perturbing the same user/item/sequence in two ways and declaring the two perturbations a positive pair. Every other example in the batch is a negative. The model learns geometry: similar things end up close, dissimilar things end up far. Once the geometry is good, the supervised recommendation head only needs a small nudge.

This article walks through the core machinery (InfoNCE, temperature, augmentations) and the four families that matter in practice for recommendations: **SimCLR-style** in-batch contrast, **MoCo-style** queue-based contrast, **SGL** graph augmentations, and **CL4SRec** sequence augmentations. We finish with **XSimGCL**, the surprising result that you can throw the augmentations away entirely and just inject noise.

## What you will learn

- **Why** contrastive learning attacks the sparsity, cold-start, and popularity-bias problems from a different angle than "more data"
- **InfoNCE in detail**: where the loss comes from, why temperature $\tau$ matters more than you'd think, and how it shapes gradients
- **SimCLR vs MoCo**: in-batch negatives vs a momentum-encoder queue, and when each wins
- **SGL** (Wu et al., SIGIR 2021): node dropout, edge dropout, random-walk subgraphs as graph views
- **CL4SRec** (Xie et al., ICDE 2022): crop, mask, reorder for sequential recommenders
- **SimGCL / XSimGCL** (Yu et al., 2022/2023): why a tiny embedding-noise trick beats elaborate graph augmentation
- Working **PyTorch** for each piece

## Prerequisites

- PyTorch fundamentals (modules, autograd, loss functions)
- Graph neural networks, especially LightGCN ([Part 7](/en/recommendation-systems-7-graph-neural-networks/))
- Embedding spaces and similarity ([Part 5](/en/recommendation-systems-5-embedding-techniques/))

---

## Why contrastive learning for recommendations?

### The data sparsity problem, restated

Sparsity is not just "we have few labels". It is structural:

1. **Cold start.** New users and new items have *zero* interactions, so any model that depends on them as features (matrix factorisation, two-tower, GNN) cannot place them anywhere meaningful in the embedding space.
2. **Overfitting on the head.** With a long-tailed click distribution, the loss is dominated by a handful of popular items. A model that memorises them looks fine on training metrics and useless on the tail.
3. **Popularity bias.** Even when you serve diverse candidates, the scoring head will rank popular items higher because that is what minimised the training loss. The system collapses toward sameness.

### The contrastive bargain

Contrastive learning trades one expensive signal (labels) for a much cheaper one (consistency under perturbation). Suppose you take a user's behaviour graph, drop 20% of the edges, encode it, then drop a different 20% of the edges and encode again. Two views, same user. The bargain is simple: **the two embeddings should be nearly identical**, while embeddings for *different* users in the batch should be different.

That single objective gives you three things at once:

- **A free training signal**, available in unlimited quantity (any user can be perturbed any number of times).
- **A representation prior**: the model learns features that survive perturbation, which by construction are the *robust* ones — exactly what you want for cold-start and tail items.
- **A regulariser** against collapse onto popular items: the loss explicitly forces different users apart.

![Anchor user pulled toward two augmented views (positives) and pushed away from other users in the batch (negatives)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig1_contrastive_pairs.png)

The figure above captures the entire intuition. The blue anchor is one user. The two green points are augmented views of *that same user*, and we pull them in. The amber points are other users from the same training batch, and we push them out. Doing this for every user, every batch, shapes a geometry where semantic similarity equals embedding similarity.

---

## InfoNCE: the loss that does the work

Almost every contrastive recommender uses some variant of the **InfoNCE** loss (van den Oord et al., 2018). Given an anchor $x$, a positive $x^+$, and a set of negatives $\{x_i^-\}$, with encoder $f$ and similarity $\mathrm{sim}(\cdot,\cdot) = z\cdot z'$ on $\ell_2$-normalised embeddings:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\!\big(\mathrm{sim}(f(x), f(x^+)) / \tau\big)}{\exp\!\big(\mathrm{sim}(f(x), f(x^+)) / \tau\big) + \sum_{i} \exp\!\big(\mathrm{sim}(f(x), f(x_i^-)) / \tau\big)}
$$

This is exactly the cross-entropy of an $(N+1)$-way classifier whose correct class is "the positive". The numerator says "make the positive likely"; the denominator forces the model to *rank* the positive above every negative. That ranking pressure is what prevents the trivial solution where all embeddings collapse to the same vector.

### Temperature: the most under-appreciated knob

The temperature $\tau$ controls how sharply the softmax distinguishes positives from negatives. It is not a cosmetic hyperparameter; it changes what the loss optimises.

![Left: InfoNCE loss vs positive similarity at different temperatures. Right: gradient with respect to positive similarity at the same temperatures](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig2_contrastive_loss.png)

Two things to read off the right-hand panel:

- **Low $\tau$ (0.05)** makes the gradient very sharp around the decision boundary. The model focuses essentially all of its capacity on the *hardest* negatives — the ones nearly as similar as the positive. This is great for fine-grained discrimination but unstable if those hard negatives are actually mislabelled (e.g. a missing-not-at-random click).
- **High $\tau$ (1.0)** spreads gradient mass across all negatives, even easy ones. The optimisation is smoother but the resulting embeddings cluster less tightly.
- **The sweet spot for recommendations is roughly $\tau \in [0.1, 0.2]$** — sharp enough for useful contrast, soft enough not to chase noise.

A useful mental model: $1/\tau$ is a "magnification". Halving $\tau$ doubles every similarity gap, which doubles the gradient pressure to separate near-positives from near-negatives.

### Why we need negatives at all

If you trained only the numerator — pull positives together — every embedding would collapse to a constant vector and the loss would happily drop to zero. Negatives are not a side dish; they are the load-bearing constraint. This is why **batch size matters so much in SimCLR-style setups**: a batch of 256 gives you 510 negatives per anchor, a batch of 4096 gives you 8190. More negatives means a denser, more informative denominator.

---

## SimCLR vs MoCo: where do the negatives come from?

Two paradigms dominate self-supervised vision and have been carried over wholesale into recommendations.

![SimCLR uses every other view in the batch as a negative (single shared encoder); MoCo maintains a momentum-updated key encoder and a FIFO queue of cached negatives](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig3_simclr_vs_moco.png)

**SimCLR** (Chen et al., ICML 2020) keeps things simple: one encoder, one projection head, two augmentations per example, and **every other view in the same minibatch is a negative**. The drawback is the obvious one — your number of negatives is bounded by the GPU memory budget for the batch.

**MoCo** (He et al., CVPR 2020) decouples the two: a query encoder updated by SGD, a *momentum* key encoder updated as an exponential moving average of the query encoder $\xi \leftarrow m\xi + (1-m)\theta$, and a FIFO queue holding the last $K$ encoded keys (typically $K = 65{,}536$). Negatives come from the queue, so $K$ is independent of batch size.

In recommendation systems, the SimCLR pattern wins more often: minibatches in CTR/CVR training are large anyway, the encoder is usually a GNN whose forward cost is dominated by graph traversal rather than embedding lookup, and a queue of millions of *user* keys ages quickly because the embedding table itself is being updated. MoCo-style queues do reappear in long-sequence retrieval models, where the encoder is heavy and you genuinely want to amortise its cost across many anchors.

### A reference SimCLR loss in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Symmetric SimCLR-style InfoNCE on two views.

    z1, z2 are L2-normalised embeddings of shape (B, D). The positive
    for z1[i] is z2[i] (and vice-versa); every other entry in the
    concatenated 2B batch is a negative.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                  # (2B, D)
    sim = z @ z.T / tau                             # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))               # exclude self-similarity
    # Positive of row i in [0..B-1] is at column i+B; row i in [B..2B-1] at column i-B.
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

Three things worth highlighting in this 10-line implementation:

1. The embeddings are assumed pre-normalised. The cosine similarity then is just the dot product, and the temperature $\tau$ has the geometric meaning above.
2. We mask the diagonal with $-\infty$, *not* zero. With `cross_entropy` working in log-softmax space, $-\infty$ disappears from the partition function; zero would silently shift gradients.
3. The loss is symmetric: each of the $2B$ rows contributes one cross-entropy term. Half the rows treat the second view as the target, half treat the first view.

---

## SGL: contrastive learning on the user-item graph

SGL (Wu et al., SIGIR 2021, *Self-supervised Graph Learning for Recommendation*) was the paper that brought contrastive learning into the recommendation mainstream. The idea is to bolt an InfoNCE head onto a LightGCN backbone and treat **two perturbed copies of the user-item graph** as positive pairs of every node.

### Three ways to perturb a graph

![Original user-item bipartite graph and three SGL augmentations: edge dropout, node dropout, random-walk subgraph](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig4_sgl_augmentations.png)

- **Edge dropout (ED)**: each edge survives independently with probability $1-p$. Cheap, structure-preserving, the most-used variant.
- **Node dropout (ND)**: each node (and all of its incident edges) is dropped with probability $p$. Stronger augmentation; can destabilise small-degree nodes.
- **Random-walk subgraph (RW)**: sample a subgraph by a length-$L$ random walk from each anchor. Different walks give different views.

In the original SGL ablations, edge dropout consistently matches or beats the other two while being trivial to implement. Use ED unless you have a specific reason not to.

### The full SGL training step

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_dropout(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, keep]


class SGL(nn.Module):
    """LightGCN backbone + node-level InfoNCE on two edge-dropout views."""

    def __init__(self, n_users, n_items, dim=64, n_layers=3,
                 drop_p=0.1, tau=0.2, lam=0.1):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers, self.drop_p, self.tau, self.lam = n_layers, drop_p, tau, lam
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def _propagate(self, x, edge_index):
        """Symmetric-normalised LightGCN propagation, layer-averaged."""
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).float()
        norm = deg.pow(-0.5)
        layers = [x]
        for _ in range(self.n_layers):
            msg = x[col] * (norm[row] * norm[col]).unsqueeze(1)
            agg = torch.zeros_like(x).index_add_(0, row, msg)
            x = agg
            layers.append(x)
        return torch.stack(layers, dim=0).mean(dim=0)

    def encode(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        return self._propagate(x, edge_index)

    def cl_loss(self, z1, z2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim = z1 @ z2.T / self.tau
        targets = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, targets)

    def bpr_loss(self, z, users, pos, neg):
        u, p, n = z[users], z[self.n_users + pos], z[self.n_users + neg]
        return -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()

    def forward(self, edge_index, users, pos_items, neg_items):
        # Two augmented views for the contrastive signal
        z1 = self.encode(edge_dropout(edge_index, self.drop_p))
        z2 = self.encode(edge_dropout(edge_index, self.drop_p))
        # One clean pass for the recommendation signal
        z = self.encode(edge_index)
        return self.bpr_loss(z, users, pos_items, neg_items) \
             + self.lam * self.cl_loss(z1, z2)
```

A few non-obvious implementation details:

- **Three forward passes, not two.** The contrastive views are dropped *separately* from the clean graph used for BPR. Sharing the dropped graph with the recommendation loss biases the supervised signal toward the lucky surviving edges.
- **The contrastive loss is computed at the node level**, on the concatenation of user and item embeddings. This is what gives both sides of the bipartite graph a self-supervised signal.
- **Loss weight $\lambda$ matters.** Too small ($<10^{-2}$) and the contrastive signal disappears; too large ($>1$) and BPR loses its grip. The SGL paper sweeps $\lambda \in \{0.005, 0.05, 0.1, 0.5, 1.0\}$ and reports $0.1$ as a robust default — start there.

---

## CL4SRec: contrastive learning for sequential recommenders

For sequence-based recommenders (SASRec, BERT4Rec, GRU4Rec...), the analogous question is "how do I create two views of *the same behaviour sequence*?". CL4SRec (Xie et al., ICDE 2022) proposed three augmentations that have become the defaults.

![Three augmentations on a behaviour sequence: crop a contiguous span, mask a random fraction with [M], reorder a contiguous chunk](./11-contrastive-learning/fig5_cl4srec_augmentations.png)

- **Crop**: keep a contiguous subsequence of length $\eta L$. Preserves local order; teaches invariance to "starting later" or "stopping earlier".
- **Mask**: replace a random $\gamma$ fraction of items with a special `[M]` token. The same idea as masked language modelling — the encoder must infer hidden items from context.
- **Reorder**: shuffle a contiguous chunk of length $\beta L$. Teaches the model that *exact* positions inside a session matter less than the bag of items themselves.

CL4SRec randomly samples *one* of the three augmentations per view, giving nine possible (view A, view B) pairs. This stochastic mixture is itself a regulariser: the encoder cannot overfit to any single augmentation policy.

```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqAugment:
    """The three CL4SRec augmentations. Sequences are LongTensors of item IDs;
    item id 0 is reserved for padding/mask."""

    def __init__(self, crop_eta=0.6, mask_gamma=0.3, reorder_beta=0.6, mask_id=0):
        self.crop_eta = crop_eta
        self.mask_gamma = mask_gamma
        self.reorder_beta = reorder_beta
        self.mask_id = mask_id

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        op = random.choice(["crop", "mask", "reorder"])
        L = seq.size(0)
        if op == "crop":
            k = max(1, int(L * self.crop_eta))
            s = random.randint(0, L - k)
            return seq[s:s + k]
        if op == "mask":
            out = seq.clone()
            n = max(1, int(L * self.mask_gamma))
            idx = torch.randperm(L)[:n]
            out[idx] = self.mask_id
            return out
        # reorder
        out = seq.clone()
        k = max(2, int(L * self.reorder_beta))
        s = random.randint(0, L - k)
        chunk = out[s:s + k][torch.randperm(k)]
        out[s:s + k] = chunk
        return out
```

The encoder is whatever sequence model you already have (Transformer, GRU, ...). Pool the final hidden state, project, normalise, drop into the same `info_nce` we wrote earlier — and add it as an auxiliary loss to your usual next-item prediction.

---

## XSimGCL: when the augmentations themselves don't matter

A surprising empirical result from Yu et al. (2022/2023) deserves its own section. They asked: how much of SGL's gain actually comes from the *graph augmentations*, and how much from the contrastive loss itself? Their answer: almost all of it comes from the loss. Replacing graph dropout with **a tiny amount of uniform noise added to the propagated embeddings** matches or beats SGL, while removing all the bookkeeping around graph perturbation.

The trick (SimGCL / XSimGCL, the latter being the streamlined variant) is roughly:

1. Run LightGCN propagation as usual.
2. At each layer, add a small noise $\Delta$ to the embeddings, where $\Delta$ is a unit-norm random direction scaled by a small $\epsilon$ (e.g. 0.1) and pointing in the *same hemisphere* as the embedding (so it perturbs but does not flip).
3. Run the propagation a *second* time with a different noise sample to get the second view. No graph dropout.

```python
def add_noise(x: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Same-hemisphere uniform noise on the unit sphere (SimGCL)."""
    noise = torch.rand_like(x)
    noise = F.normalize(noise, dim=-1) * torch.sign(x) * epsilon
    return x + noise
```

The deeper lesson, articulated in the SimGCL analysis, is that the contrastive loss is doing two things simultaneously:

- **Alignment**: pulling positive pairs together (the numerator).
- **Uniformity**: pushing all embeddings to spread evenly over the unit hypersphere (the denominator).

Uniformity is the regulariser that breaks popularity bias, and it is **the dominant effect**. Once you have uniformity, the exact mechanism that produces the second view (graph dropout, embedding noise, anything reasonable) hardly matters.

---

## Does this actually help? Reported gains.

![Recall@20 and NDCG@20 on four standard benchmarks: LightGCN baseline vs +SGL vs +XSimGCL](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig6_performance_gain.png)

The numbers above are illustrative of magnitudes reported in the SGL (Wu et al., 2021) and SimGCL/XSimGCL (Yu et al., 2022/2023) papers on Yelp2018, Amazon-Book, Alibaba-iFashion, and Gowalla. The pattern is consistent: **a contrastive auxiliary loss adds 5–20% to Recall@20 and NDCG@20** over the same backbone with no contrastive signal, and the gains are concentrated on tail items and cold users — exactly where you cared.

What that gain looks like in the embedding space:

![Item embeddings before vs after contrastive training, projected with t-SNE: from entangled blob to well-separated interest clusters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/11-contrastive-learning/fig7_embedding_space.png)

Before contrastive training, the item embeddings cluster weakly — the dominant axis of variation is popularity, and most semantic structure is buried. After contrastive training, the items separate into tight, well-spaced groups corresponding to interest categories. The downstream scoring head then has a much easier job: a near-linear boundary suffices.

---

## Frequently asked questions

### Why not just use more data instead of contrastive learning?

You can't get "more data" for cold users — they are cold by definition. You can't get "more data" for tail items — they are tail because almost no one interacts with them. Contrastive learning manufactures a training signal that does not require new interactions, only new perturbations of the interactions you already have. It is doing something *categorically different* from collecting more clicks.

### How do I choose between graph augmentations and embedding noise?

Default to embedding noise (XSimGCL): it is faster, simpler, and the recent literature consistently finds it competitive with or superior to graph augmentation. Reach for graph augmentations (SGL) when you want to *also* regularise the GNN's structural inductive bias, or when you have a good prior on which edges/nodes are most informative.

### How do I set the temperature?

Start at $\tau = 0.2$. If your hardest negatives are reliably real negatives (e.g. you have explicit dwell-time signals), drop $\tau$ to 0.05–0.1 to sharpen. If your negatives are noisy (e.g. random in-batch users in a high-dimensional catalogue), keep $\tau \geq 0.2$ to avoid overfitting to false negatives. Tune on a small grid; the loss landscape is smooth in $\tau$.

### Do I need a projection head?

For SimCLR / SGL on top of a GNN, yes — discard it after pretraining. The projection head lets the encoder's intermediate representation stay general while the head specialises for the contrastive metric. XSimGCL is the notable exception: it contrasts the propagated GNN embeddings directly without a projection head, and works fine.

### How do I combine contrastive and recommendation losses?

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda \cdot \mathcal{L}_{\text{CL}}$$

Start with $\lambda = 0.1$. Sweep $\{0.01, 0.05, 0.1, 0.5, 1.0\}$ on validation if results matter. Sensitivity to $\lambda$ is much lower than sensitivity to $\tau$ — don't burn your hyperparameter budget here.

### Does this work with implicit feedback?

Yes, and arguably *better* than with explicit feedback. The whole framework treats every interaction as a positive label and lets the InfoNCE structure handle negatives implicitly. SGL, SimGCL, XSimGCL, and CL4SRec are all designed for implicit feedback (clicks, plays, purchases).

### How small can my dataset be?

Contrastive learning helps most precisely when supervised signal is scarce. SGL's reported gains on Yelp2018 (sparsity ~99.87%) are larger than on denser datasets. Below ~10K interactions the variance from random augmentation dominates and you may need stronger priors (cross-domain transfer, content features); above ~100K interactions you should expect clean improvements.

### How do I evaluate contrastive recommenders?

Standard top-K metrics (Recall@K, NDCG@K, HR@K) for parity with baselines, then *additionally*:

- **Cold-start slice**: bucket users/items by interaction count and report metrics per bucket.
- **Tail coverage**: fraction of recommended items in the long tail (often defined as the bottom 80% by popularity).
- **Embedding diagnostics**: visualise with t-SNE/UMAP, or compute uniformity ($\log \mathbb{E}\,e^{-2\|z_i - z_j\|^2}$) and alignment ($\mathbb{E}\,\|z - z^+\|^2$) per Wang & Isola (2020). High uniformity + low alignment = healthy.

---

## Conclusion

Contrastive learning solves a problem that more data cannot: it gives a model a way to learn *geometry* from sparse interactions, by demanding consistency under perturbation. The recipe in 2024 is well-understood:

1. **Pick an augmentation**, or none at all (XSimGCL noise).
2. **Apply InfoNCE** with $\tau \approx 0.2$.
3. **Add it as an auxiliary loss** with weight $\lambda \approx 0.1$ on top of your existing supervised objective.

For graph-based recommenders, start with XSimGCL — it is the simplest thing that works. If you want a more interpretable baseline, SGL with edge dropout is still a strong choice. For sequence recommenders, CL4SRec's three-way augmentation menu is the standard. In every case, expect the largest wins on the cold and the tail.

---

## Series Navigation

This article is **Part 11** of the 16-part Recommendation Systems series.

| Previous | | Next |
|:---------|:-:|-----:|
| [Part 10: Deep Interest Networks](/en/recommendation-systems-10-deep-interest-networks/) | [All Parts](/tags/Recommendation-Systems/) | [Part 12: LLM-Based Recommendation](/en/recommendation-systems-12-llm-recommendation/) |
