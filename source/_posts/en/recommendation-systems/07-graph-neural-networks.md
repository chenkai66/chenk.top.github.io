---
title: "Recommendation Systems (7): Graph Neural Networks and Social Recommendation"
date: 2025-11-08 09:00:00
tags:
  - Recommendation Systems
  - GNN
  - Social Recommendation
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 7
  total: 16
lang: en
mathjax: true
description: "A deep, intuition-first walkthrough of graph neural networks for recommendation: GCN, GAT, GraphSAGE, PinSage, LightGCN, NGCF, social signals, scalable sampling, and cold start. Diagrams plus working PyTorch."
permalink: "en/recommendation-systems-7-graph-neural-networks/"
disableNunjucks: true
---
When Netflix decides what to recommend next, it does not look at your watch history in isolation. Behind the scenes there is a web of relationships: movies that share actors, users with overlapping taste, ratings that ripple through the catalogue. The "graph" view is not a metaphor — every interaction matrix *is* a graph, and treating it as one unlocks ideas that flat user/item embeddings cannot express.

**Graph neural networks** (GNNs) are the tool that lets us reason over that graph. Instead of learning each user and each item in isolation, a GNN says: *your representation is shaped by the company you keep.* That single shift powers Pinterest's billion-node PinSage, the strikingly simple LightGCN that beats heavier baselines on collaborative filtering, and the social-recommendation systems that fuse "what you watched" with "what your friends watched."

This article is an intuition-first walk through the landscape. We start from the bipartite user-item graph, build up the message-passing framework, then dissect GCN, GAT, GraphSAGE, PinSage, LightGCN, NGCF, social GNNs, and the sampling tricks that keep all of this tractable at scale. Every model comes with working PyTorch.

---

## What You Will Learn

- Why recommendation data is naturally graph-shaped, and what that buys you
- The core GNN building blocks — GCN, GAT, GraphSAGE — and when to pick which
- How industrial models (PinSage, LightGCN, NGCF) put GNNs into production
- How social signals add 5–15% lift, and the failure mode to watch for
- Mini-batch neighbour sampling that scales GNNs from MovieLens to Pinterest
- A clean recipe for cold-start users via inductive aggregation

## Prerequisites

- Comfort with Python and PyTorch (tensors, `nn.Module`, training loops)
- Basic embeddings and collaborative filtering ([Part 2](/en/recommendation-systems-2-collaborative-filtering/))
- Comfort with matrix notation; no spectral graph theory required — we build everything by hand first

---

## Why GNNs for Recommendation?

### Recommendation Data Is Already a Graph

![User-item bipartite graph showing connections between users and movies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig1_bipartite_graph.png)

Take any recommendation system and look closely at the interaction matrix: rows are users, columns are items, and each non-zero entry is a click, watch, purchase, or rating. Redraw that matrix as a **bipartite graph**:

- **Nodes** are users and items.
- **Edges** connect a user to every item they interacted with.
- **Collaborative signal** is hidden in the topology: users who share many edges (interacted with similar items) are implicitly connected through those shared items.

In the figure above, Alice and Bob never met, but the highlighted edges reveal they share two movies. A GNN can propagate this signal across hops: *"Bob liked Blade Runner, Bob is graph-similar to Alice through their shared movies, so Alice probably likes Blade Runner too."* Matrix factorisation never asks that question — it only sees individual cells.

### What Traditional Approaches Miss

| Method | What it captures | What it misses |
|---|---|---|
| **Matrix factorisation** | Latent factors for users/items, dot product score | Each embedding learned in isolation; no notion of "neighbours of neighbours" |
| **Neural CF / autoencoders** | Nonlinear interactions | Still per-pair; no propagation along the graph |
| **GNN** | Node embedding shaped by its $L$-hop neighbourhood | (Eventually: scaling, over-smoothing — we will fix both) |

The slogan: **traditional methods learn embeddings in isolation; GNNs learn embeddings in context.**

### What GNNs Bring to the Table

- **Explicit graph modelling** — operate directly on topology.
- **Neighbourhood aggregation** — collaborative signal propagates as a side-effect of the architecture.
- **Multi-hop reasoning** — stack two layers and you capture "users who liked items similar to items you liked."
- **Inductive variants** (GraphSAGE) — embed brand-new users/items without retraining.
- **Heterogeneous graphs** — fold in social ties, knowledge graphs, attributes.

---

## Graph Neural Network Fundamentals

### Graph Basics (Quick Refresher)

A graph $G=(V,E)$ has vertices $V=\{v_1,\dots,v_n\}$ and edges $E\subseteq V\times V$. The **adjacency matrix** $A\in\{0,1\}^{n\times n}$ has $A_{ij}=1$ exactly when there is an edge between $v_i$ and $v_j$.

Three graph flavours show up everywhere in recommendation:

| Type | Description | Example |
|---|---|---|
| **Bipartite** | Two node types; edges only cross types | User ↔ item |
| **Weighted** | Edge weights encode strength | Rating, click count |
| **Attributed** | Nodes carry feature vectors | Demographics, item categories |

### Message Passing in One Picture

![Three-step message passing: each neighbour computes a message, the target node aggregates them, then updates its state](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig2_message_passing.png)

Every GNN, however dressed up, is a variation of the same three-step recipe:

1. **Message.** Each neighbour computes what to send.
2. **Aggregate.** The target combines incoming messages (sum, mean, attention-weighted, …).
3. **Update.** The target blends the aggregate with its own previous state.

Formally, at layer $l$, node $v$ updates as:

$$\mathbf{m}_v^{(l)} = \mathrm{AGGREGATE}^{(l)}\!\bigl(\{\mathbf{h}_u^{(l-1)} : u\in\mathcal{N}(v)\}\bigr)$$

$$\mathbf{h}_v^{(l)} = \mathrm{UPDATE}^{(l)}\!\bigl(\mathbf{h}_v^{(l-1)},\; \mathbf{m}_v^{(l)}\bigr)$$

In plain English: *"Look at what your neighbours know, summarise it, then blend it with what you already know."* Stack $L$ layers and each node sees its $L$-hop neighbourhood.

**Telephone analogy.** Imagine a game of telephone where every person whispers to all their friends simultaneously. After one round you know what your friends think; after two, what your friends' friends think. GNNs do exactly that with learned, differentiable functions.

---

## Graph Convolutional Networks (GCN)

### The Key Idea

GCN (Kipf & Welling, 2017) defines a convolution-like operation on graphs. Just as a CNN aggregates pixels from a local neighbourhood, a GCN aggregates feature vectors from a node's graph neighbourhood.

### The GCN Layer

A single GCN layer computes:

$$\mathbf{H}^{(l+1)} = \sigma\!\Bigl(\tilde{D}^{-1/2}\, \tilde{A}\, \tilde{D}^{-1/2}\, \mathbf{H}^{(l)}\, \mathbf{W}^{(l)}\Bigr)$$

That looks dense. Here is what each piece does:

| Symbol | Meaning |
|---|---|
| $\tilde{A} = A + I$ | Adjacency matrix with self-loops added (each node also listens to itself) |
| $\tilde{D}$ | Degree matrix of $\tilde{A}$ |
| $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ | Symmetric normalisation — prevents high-degree nodes from dominating |
| $\mathbf{H}^{(l)}$ | Node feature matrix at layer $l$ (each row = one node's embedding) |
| $\mathbf{W}^{(l)}$ | Learnable weight matrix |
| $\sigma$ | Nonlinearity, typically ReLU |

For a single node $v$ this unfolds to:

$$\mathbf{h}_v^{(l+1)} = \sigma\!\left(\sum_{u\in\mathcal{N}(v)\cup\{v\}} \frac{1}{\sqrt{d_v\,d_u}}\; \mathbf{h}_u^{(l)}\, \mathbf{W}^{(l)}\right)$$

1. **Gather** the embeddings of all neighbours (plus yourself).
2. **Normalise** each contribution by $1/\sqrt{d_v\cdot d_u}$ — popular nodes are scaled down so they do not shout over everyone else.
3. **Transform** with a shared weight matrix $\mathbf{W}$.
4. **Activate** with ReLU.

### Implementation: GCN Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


class GCNLayer(nn.Module):
    """Single Graph Convolutional layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge list [2, num_edges]
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Step 1: add self-loops so each node hears itself
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: symmetric normalisation coefficients
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: linear transform
        x = self.linear(x)

        # Step 4: scatter-add normalised neighbour features into targets
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand_as(x), norm.unsqueeze(1) * x[row])
        return out
```

### Multi-Layer GCN

Stacking layers extends the receptive field. Two layers = two-hop neighbourhood, three layers = three hops. But beware **over-smoothing**: too many layers and every node's embedding converges to the same value. Two to three layers is the sweet spot for recommendation.

```python
class GCN(nn.Module):
    """Multi-layer GCN for learning node embeddings."""

    def __init__(self, num_nodes, in_channels, hidden_channels,
                 out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Optional: learnable embeddings when nodes have no input features
        self.embedding = None
        if in_channels == 0:
            self.embedding = nn.Embedding(num_nodes, hidden_channels)
            in_channels = hidden_channels

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNLayer(in_channels, out_channels))
        else:
            self.convs.append(GCNLayer(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            self.convs.append(GCNLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        if self.embedding is not None:
            x = self.embedding.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
```

---

## Graph Attention Networks (GAT)

### Why Attention?

GCN treats every neighbour equally (up to degree normalisation). But not all neighbours are equally informative — your best friend's movie taste matters more than a random acquaintance's. GAT (Veličković et al., 2018) fixes this by learning **adaptive attention weights** per edge.

### How GAT Works

For each edge from $j$ to $i$, GAT computes an attention logit:

$$e_{ij} = \mathrm{LeakyReLU}\!\bigl(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\bigr)$$

where $\mathbf{W}$ projects node features into a shared space, $\|$ is concatenation, and $\mathbf{a}$ is a learnable attention vector. Logits are normalised across neighbours with softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}(i)} \exp(e_{ik})}$$

The updated embedding is a weighted sum:

$$\mathbf{h}_i^{(l+1)} = \sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}} \alpha_{ij}\, \mathbf{W}^{(l)}\, \mathbf{h}_j^{(l)}\right)$$

In plain English: *"For each neighbour, learn how relevant they are to me, then take a weighted average of their features."*

### Multi-Head Attention

As in Transformers, GAT uses multiple heads to stabilise learning. Each head computes its own attention and aggregation; results are concatenated (intermediate layers) or averaged (final layer):

$$\mathbf{h}_i^{(l+1)} = \big\|_{k=1}^{K}\; \sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}} \alpha_{ij}^{(k)}\, \mathbf{W}^{(k)}\, \mathbf{h}_j^{(l)}\right)$$

### Implementation: GAT Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GATLayer(MessagePassing):
    """Graph Attention Network layer with multi-head attention."""

    def __init__(self, in_channels, out_channels, heads=1,
                 dropout=0.0, concat=True):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Project into multi-head space: [N, heads, out_channels]
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index, x=x)
        return out.view(-1, self.heads * self.out_channels) if self.concat else out.mean(dim=1)

    def message(self, x_i, x_j, index):
        # x_i: target features, x_j: source features
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index)  # normalise per target node
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
```

---

## GraphSAGE: Scalable Inductive Learning

### The Problem with GCN

GCN is **transductive**: it needs the entire graph at training time and has no embedding for nodes that appear later. If a new user signs up tomorrow, GCN cannot score them without retraining.

GraphSAGE (Hamilton et al., 2017) fixes this with **inductive learning**. Instead of memorising one fixed embedding per node, it learns *how to aggregate* neighbour information. When a new node appears, GraphSAGE runs the same aggregation on its neighbours to produce an embedding on the fly.

### Sample and Aggregate

GraphSAGE has two key ideas:

1. **Neighbour sampling.** Instead of using *all* neighbours (expensive), sample a fixed number $k$ at each hop.
2. **Learned aggregation.** Apply a trainable function (mean, LSTM, max-pool) to the sampled neighbours.

For node $v$ at layer $l$:

$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \mathrm{AGGREGATE}^{(l)}\!\bigl(\{\mathbf{h}_u^{(l-1)} : u\in\mathcal{N}_{\text{sampled}}(v)\}\bigr)$$

$$\mathbf{h}_v^{(l)} = \sigma\!\bigl(\mathbf{W}^{(l)}\cdot[\mathbf{h}_v^{(l-1)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(l)}]\bigr)$$

In plain English: *"Sample some neighbours, summarise their features, concatenate with your own, push through a linear layer."*

| Aggregator | How it works |
|---|---|
| **Mean** | Average neighbour embeddings: $\tfrac{1}{|\mathcal{N}|}\sum_u \mathbf{h}_u$ |
| **Max-pool** | Element-wise max after a shared MLP: $\max(\sigma(\mathbf{W}_{\text{pool}}\mathbf{h}_u + \mathbf{b}))$ |
| **LSTM** | Run an LSTM over neighbours (order-sensitive) |

### How the Three Differ at a Glance

![GCN vs GAT vs GraphSAGE: equal weighting, learned attention, and sampled aggregation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig3_gcn_gat_sage.png)

GCN gives every neighbour the same (degree-normalised) say. GAT learns who matters and turns up their weight. GraphSAGE picks a manageable subset of neighbours so the whole thing stays affordable on huge graphs.

### Implementation: GraphSAGE Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SAGEConv(MessagePassing):
    """GraphSAGE convolution with mean aggregation."""

    def __init__(self, in_channels, out_channels, normalize=False):
        super().__init__(aggr='mean')
        self.normalize = normalize
        self.lin_self = nn.Linear(in_channels, out_channels)   # own features
        self.lin_neigh = nn.Linear(in_channels, out_channels)  # neighbour features

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        neigh_agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + self.lin_neigh(neigh_agg)
        return F.normalize(out, p=2, dim=1) if self.normalize else out

    def message(self, x_j):
        return x_j
```

The ability to embed brand-new nodes is the single biggest reason GraphSAGE shows up in production. Pinterest's PinSage (next section) is a direct descendant.

---

## PinSage: Billion-Scale Recommendation at Pinterest

PinSage (Ying et al., 2018) is Pinterest's production GNN, running on **3 billion nodes and 18 billion edges**. Three innovations make this possible:

### 1. Random Walk Sampling

Instead of sampling neighbours uniformly, PinSage runs short random walks from each node and keeps the top-$k$ most frequently visited neighbours. This focuses on the most *important* neighbours, not just the closest ones.

### 2. Importance-Weighted Aggregation

Neighbour features are weighted by random-walk visit counts:

$$\mathbf{h}_v^{(l)} = \sigma\!\Bigl(\mathbf{W}^{(l)}\cdot \bigl[\mathbf{h}_v^{(l-1)} \,\big\|\, \mathrm{AGG}\bigl(\{\alpha_{uv}\, \mathbf{h}_u^{(l-1)} : u\in\mathcal{N}_{\text{top-}k}(v)\}\bigr)\bigr]\Bigr)$$

### 3. Hard Negative Mining

During training, PinSage samples *hard negatives* — items that score high but are not actual positives. This forces the model to make finer distinctions instead of just separating obviously irrelevant items from positives.

### Implementation: PinSage Convolution

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict


class PinSageConv(nn.Module):
    """PinSage convolution with importance-weighted aggregation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.combine = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, importance_weights=None):
        row, col = edge_index
        neighbor_feats = x[col]
        if importance_weights is not None:
            neighbor_feats = neighbor_feats * importance_weights.unsqueeze(1)

        aggregated = torch.zeros(x.size(0), x.size(1), device=x.device)
        aggregated.scatter_add_(
            0, row.unsqueeze(1).expand_as(neighbor_feats), neighbor_feats
        )
        aggregated = self.aggregator(aggregated)
        return self.combine(torch.cat([x, aggregated], dim=1))


def compute_random_walk_importance(adj_list, num_nodes, num_walks=10,
                                   walk_length=5, top_k=10):
    """Compute neighbour importance via short random walks."""
    visit_counts = Counter()
    for start in range(num_nodes):
        if start not in adj_list:
            continue
        for _ in range(num_walks):
            current = start
            for _ in range(walk_length):
                neighbors = adj_list.get(current, [])
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                visit_counts[(start, current)] += 1

    per_source = defaultdict(list)
    for (src, tgt), cnt in visit_counts.items():
        per_source[src].append((tgt, cnt))

    scores = {}
    for src, neighbors in per_source.items():
        neighbors.sort(key=lambda t: t[1], reverse=True)
        top = neighbors[:top_k]
        total = sum(c for _, c in top)
        for tgt, cnt in top:
            scores[(src, tgt)] = cnt / total if total > 0 else 0
    return scores
```

---

## LightGCN: Less Is More

### The Surprising Insight

LightGCN (He et al., 2020) asks a provocative question: *"What if we strip everything out of GCN — no learned weight matrices, no nonlinear activations, no self-loops — and just keep the neighbourhood aggregation?"*

The answer, on collaborative filtering, is **it works better.** The graph structure alone carries enough signal. The feature transforms and activations of a standard GCN actually *hurt* by adding unnecessary complexity and overfitting risk.

![Standard GCN keeps self-loops, weights, and ReLU; LightGCN keeps only aggregation and layer combination](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig4_lightgcn_simplify.png)

### Architecture

LightGCN uses the simplest possible aggregation:

$$\mathbf{e}_u^{(l+1)} = \sum_{i\in\mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)|\cdot|\mathcal{N}(i)|}}\; \mathbf{e}_i^{(l)}$$

$$\mathbf{e}_i^{(l+1)} = \sum_{u\in\mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(u)|\cdot|\mathcal{N}(i)|}}\; \mathbf{e}_u^{(l)}$$

In plain English: *"Average your neighbours' embeddings (normalised by degrees). No activation, no weight matrix. That is one layer."*

The final embedding combines all layers:

$$\mathbf{e}_u = \sum_{l=0}^{L} \alpha_l\, \mathbf{e}_u^{(l)}, \qquad \mathbf{e}_i = \sum_{l=0}^{L} \alpha_l\, \mathbf{e}_i^{(l)}$$

with $\alpha_l = \tfrac{1}{L+1}$ (equal weighting). Layer 0 is the raw embedding, layer 1 is one-hop, layer 2 is two-hop, and so on. Combining them gives a multi-scale representation without committing to a single depth.

### Why It Works

- **Smoothing effect.** Aggregation pulls similar nodes' embeddings closer — exactly what collaborative filtering wants.
- **Layer combination.** Mixing layers balances direct signals (layer 0) with higher-order collaborative signals (deeper layers) and dampens over-smoothing.
- **Simplicity as regularisation.** Fewer parameters means less overfitting on sparse interaction data.

### Implementation: LightGCN

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNLayer(MessagePassing):
    """One layer of LightGCN: normalised neighbour aggregation, nothing else."""

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.unsqueeze(1) * x_j


class LightGCN(nn.Module):
    """Complete LightGCN model for collaborative filtering."""

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embeddings = [x]  # layer 0 = raw embeddings
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embeddings.append(x)

        # Equal-weight combination across all layers
        final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return final[:self.num_users], final[self.num_users:]

    def predict(self, user_emb, item_emb, user_ids, item_ids):
        return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=1)
```

---

## NGCF: Neural Graph Collaborative Filtering

### How It Differs from LightGCN

NGCF (Wang et al., 2019) takes the opposite philosophy: explicit feature transformations and user-item interaction terms during message passing should give richer embeddings. Where LightGCN strips, NGCF adds.

### Message Construction

At each layer, messages include a **feature interaction** term:

$$\mathbf{m}_{u\leftarrow i} = \frac{1}{\sqrt{|\mathcal{N}(u)|\cdot|\mathcal{N}(i)|}}\;\bigl(\mathbf{W}_1\, \mathbf{e}_i^{(l)} + \mathbf{W}_2\, (\mathbf{e}_i^{(l)} \odot \mathbf{e}_u^{(l)})\bigr)$$

The $\odot$ (element-wise product) captures how user and item features interact. If a user dimension is high for "action" and the item dimension is high for the same, the product amplifies the signal.

After aggregation:

$$\mathbf{e}_u^{(l+1)} = \mathrm{LeakyReLU}\!\bigl(\mathbf{m}_{u\leftarrow u} + \sum_{i\in\mathcal{N}(u)} \mathbf{m}_{u\leftarrow i}\bigr)$$

### Implementation: NGCF

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class NGCFLayer(MessagePassing):
    """NGCF layer with feature transformation and interaction."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_channels, out_channels, bias=False)
        self.W2 = nn.Linear(in_channels, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        self_emb = self.W_self(x)
        neighbor_emb = self.propagate(edge_index, x=x, norm=norm)
        out = F.leaky_relu(self_emb + neighbor_emb, negative_slope=0.2)
        return F.dropout(out, p=self.dropout, training=self.training)

    def message(self, x_i, x_j, norm):
        # W1 * neighbour + W2 * (user . item)  -- feature interaction
        msg = self.W1(x_j) + self.W2(x_i * x_j)
        return norm.unsqueeze(1) * msg


class NGCF(nn.Module):
    """Neural Graph Collaborative Filtering."""

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([
            NGCFLayer(embedding_dim, embedding_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
        return x[:self.num_users], x[self.num_users:]
```

### LightGCN vs. NGCF: Which Should You Use?

| | LightGCN | NGCF |
|---|---|---|
| **Parameters** | Only embeddings | Embeddings + weight matrices |
| **Performance** | Often better on sparse data | Better when features are rich |
| **Overfitting risk** | Low | Higher (more parameters) |
| **Training speed** | Fast | Slower |
| **Recommended for** | Most CF tasks | Tasks with dense features |

**Default choice: LightGCN.** Switch to NGCF only with strong item/user features and a large enough dataset to justify the extra parameters.

---

## Why Stack Layers? Multi-Hop Aggregation

![Multi-hop neighbourhood: target user collects from items, then from similar users, then from candidate items](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig5_multihop.png)

The reason recommendation cares about depth at all becomes obvious once you draw the hops:

- **Layer 1** — target user collects features from items they touched.
- **Layer 2** — signal reaches similar users (the classic collaborative-filtering hop).
- **Layer 3** — signal reaches the items those similar users liked: candidate recommendations.

Two layers already capture the "users who liked similar items" pattern that matrix factorisation cannot express. Three layers occasionally help on sparse graphs but flirt with over-smoothing — defer to LightGCN-style layer combination if you want depth without collapse.

---

## Social Recommendation

### The Idea

Your friends influence what you buy, watch, and listen to. Social recommendation adds **social edges** (friendships, follows, trust links) to the user-item graph. The bet: socially connected users tend to share preferences.

### Graph Structure

A social recommendation graph has two edge types:

- **Interaction edges** $(u_i, i_j)$: user $u_i$ interacted with item $i_j$.
- **Social edges** $(u_i, u_j)$: users $u_i$ and $u_j$ are friends.

### Two Mechanisms

**Homophily.** Friends have similar tastes *because* they are friends (shared background, culture).

**Social influence.** Friends *become* more similar over time because they influence each other's choices.

For the model, both reduce to the same prescription: propagate preferences along social edges.

### Implementation: Social GCN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class SocialGCNLayer(MessagePassing):
    """GCN layer that handles both user-item and social edges."""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index_ui, edge_index_social, num_users):
        x_ui = self.propagate(edge_index_ui, x=x)
        x_social = self.propagate(edge_index_social, x=x[:num_users])
        x_combined = x_ui.clone()
        x_combined[:num_users] = x_combined[:num_users] + x_social
        return self.linear(x_combined)


class SocialGCN(nn.Module):
    """Social recommendation using GCN."""

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.convs = nn.ModuleList([
            SocialGCNLayer(embedding_dim, embedding_dim)
            for _ in range(num_layers)
        ])

    def forward(self, edge_index_ui, edge_index_social):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index_ui, edge_index_social, self.num_users)
            x = F.relu(x)
        return x[:self.num_users], x[self.num_users:]
```

Empirical studies report **5–15% lift** in NDCG when social signals are informative — meaningful in production. The failure mode: noisy social ties (random Facebook friends with no shared taste) hurt instead of help. Use attention to down-weight irrelevant social edges, or filter the social graph by interaction overlap before training.

---

## Graph Sampling for Scalability

### The Problem

Full-batch GNN training means loading the *entire* graph into memory and computing embeddings for *all* nodes every forward pass. Fine for MovieLens (100K interactions). Impossible for Pinterest (18 billion edges).

![Full graph vs. sampled mini-batch: pick a target node, sample a few hop-1 neighbours, then a few hop-2 neighbours, and only those participate in the forward pass](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig6_minibatch_sampling.png)

### Sampling Strategies

| Strategy | How it works | Example |
|---|---|---|
| **Neighbour sampling** | Sample a fixed number of neighbours per node, per layer | GraphSAGE: 10 at hop 1, 5 at hop 2 |
| **Node sampling** | Sample a subset of nodes, use their neighbourhoods | FastGCN |
| **Subgraph sampling** | Sample connected subgraphs | Cluster-GCN, GraphSAINT |
| **Random walk sampling** | Use random walks to find important neighbours | PinSage |

Neighbour sampling is the workhorse: bound the receptive field per node so memory stays predictable, then re-sample every epoch for variance reduction.

### Implementation: Neighbour Sampler

```python
import numpy as np
import torch


class NeighborSampler:
    """GraphSAGE-style neighbour sampler for mini-batch training."""

    def __init__(self, edge_index, num_nodes, num_neighbors=(10, 5)):
        """
        Args:
            num_neighbors: Tuple of (hop1_samples, hop2_samples, ...)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors

        self.adj = {i: [] for i in range(num_nodes)}
        row, col = edge_index.cpu().numpy()
        for r, c in zip(row, col):
            self.adj[r].append(c)

    def sample(self, target_nodes):
        """Sample multi-hop neighbourhoods for target nodes."""
        current_nodes = set(target_nodes.tolist())
        all_layers = []

        for num_nbrs in self.num_neighbors:
            edges, next_nodes = [], set()
            for node in current_nodes:
                neighbors = self.adj[node]
                if not neighbors:
                    continue
                if len(neighbors) > num_nbrs:
                    sampled = np.random.choice(
                        neighbors, num_nbrs, replace=False
                    ).tolist()
                else:
                    sampled = neighbors
                for nbr in sampled:
                    edges.append([nbr, node])
                    next_nodes.add(nbr)

            if edges:
                all_layers.append(torch.tensor(edges).t().contiguous())
            else:
                all_layers.append(torch.empty((2, 0), dtype=torch.long))
            current_nodes = next_nodes
        return all_layers
```

---

## Training Techniques

### BPR Loss (Bayesian Personalised Ranking)

The default loss for implicit-feedback recommendation:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \ln\, \sigma(\hat{r}_{ui} - \hat{r}_{uj}) + \lambda \|\Theta\|^2$$

In plain English: *"For each user $u$, make the score of a positive item $i$ (one they interacted with) higher than a negative item $j$ (one they did not). The sigmoid and log turn this into a smooth, differentiable objective."*

- $(u,i,j)$ = (user, positive item, negative item) triplet
- $\hat{r}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i$ = dot-product score
- $\lambda \|\Theta\|^2$ = L2 regularisation

### Negative Sampling Strategies

| Strategy | Description | When to use |
|---|---|---|
| **Uniform** | Pick random non-interacted items | Baseline |
| **Popularity-based** | Sample proportional to item popularity | Helps with long-tail |
| **Hard negatives** | High-scoring negatives | Late-stage refinement |

### Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim


class BPRLoss(nn.Module):
    """BPR loss with L2 regularisation."""

    def __init__(self, reg_lambda=1e-4):
        super().__init__()
        self.reg_lambda = reg_lambda

    def forward(self, pos_scores, neg_scores, *embeddings):
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg = self.reg_lambda * sum(emb.norm(2).pow(2) for emb in embeddings)
        return bpr + reg


def train_lightgcn(model, edge_index, train_pairs, num_items,
                   num_epochs=100, lr=0.001, batch_size=1024):
    """Training loop for LightGCN. train_pairs: tensor of [user, pos_item]."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BPRLoss()

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(train_pairs))
        total_loss, num_batches = 0.0, 0

        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[perm[start:start + batch_size]]
            user_ids, pos_items = batch[:, 0], batch[:, 1]
            neg_items = torch.randint(0, num_items, (len(batch),))

            optimizer.zero_grad()
            user_emb, item_emb = model(edge_index)

            pos_scores = (user_emb[user_ids] * item_emb[pos_items]).sum(1)
            neg_scores = (user_emb[user_ids] * item_emb[neg_items]).sum(1)

            loss = criterion(pos_scores, neg_scores,
                             user_emb[user_ids],
                             item_emb[pos_items],
                             item_emb[neg_items])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_batches:.4f}")
```

### Evaluation Metrics

```python
import numpy as np
import torch


def evaluate_topk(model, edge_index, test_user_items, train_user_items, k=10):
    """Compute Recall@K and NDCG@K."""
    model.eval()
    recalls, ndcgs = [], []

    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        for user_id, test_items in test_user_items.items():
            scores = (user_emb[user_id] * item_emb).sum(dim=1)

            # Mask out training items
            train_items = train_user_items.get(user_id, [])
            scores[train_items] = -float('inf')

            _, top_k = torch.topk(scores, k)
            top_k = set(top_k.cpu().numpy())

            hits = len(top_k & set(test_items))
            recalls.append(hits / len(test_items) if test_items else 0)

            # NDCG: rewards hits at higher positions
            dcg = sum(1.0 / np.log2(idx + 2)
                      for idx, item in enumerate(top_k) if item in test_items)
            idcg = sum(1.0 / np.log2(idx + 2)
                       for idx in range(min(k, len(test_items))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(recalls), np.mean(ndcgs)
```

---

## Cold Start: Where Graph Models Shine

![Matrix factorisation has no embedding for a brand-new user; GraphSAGE-style aggregation builds one from a single interaction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/07-graph-neural-networks/fig7_cold_start.png)

A new user appears with one interaction. Matrix factorisation cannot help — it has no row for them, and adding one means retraining. An inductive GNN like GraphSAGE simply runs the same aggregation function on whatever neighbours the new user has. Even a single interaction yields a meaningful embedding because the aggregation pulls in everything we already know about that item's other admirers.

A practical recipe for handling cold-start users:

1. **Inductive aggregation.** Use GraphSAGE/PinSage-style models so any node with at least one neighbour gets an embedding for free.
2. **Feature initialisation.** Seed cold-start nodes with side-feature embeddings (demographics, device, registration channel) before any aggregation.
3. **Mean-of-similar fallback.** Until a user has enough interactions, blend their embedding with the mean of users who share the same side features.
4. **Meta-learning** (MAML-style). Train the model to adapt rapidly to new users from a few interactions if you can afford it.

---

## Complete Example: LightGCN on MovieLens

```python
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. Synthetic MovieLens-like data ---
num_users, num_items = 1000, 2000
num_interactions = 10000

users = np.random.randint(0, num_users, num_interactions)
items = np.random.randint(0, num_items, num_interactions)
train_u, test_u, train_i, test_i = train_test_split(
    users, items, test_size=0.2, random_state=42
)

# Bipartite edge index (undirected)
edge_src = np.concatenate([train_u, train_i + num_users])
edge_dst = np.concatenate([train_i + num_users, train_u])
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

# --- 2. Train LightGCN ---
model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = BPRLoss()

train_pairs = torch.tensor(np.stack([train_u, train_i], axis=1), dtype=torch.long)

for epoch in range(50):
    model.train()
    perm = torch.randperm(len(train_pairs))
    for start in range(0, len(train_pairs), 512):
        batch = train_pairs[perm[start:start + 512]]
        user_ids, pos_items = batch[:, 0], batch[:, 1]
        neg_items = torch.randint(0, num_items, (len(batch),))

        optimizer.zero_grad()
        user_emb, item_emb = model(edge_index)
        pos_scores = (user_emb[user_ids] * item_emb[pos_items]).sum(1)
        neg_scores = (user_emb[user_ids] * item_emb[neg_items]).sum(1)
        loss = criterion(pos_scores, neg_scores,
                         user_emb[user_ids],
                         item_emb[pos_items],
                         item_emb[neg_items])
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete!")
```

---

## Frequently Asked Questions

### Q1: Why do GNNs outperform matrix factorisation for recommendation?

Matrix factorisation learns each embedding independently — it never asks "what do my neighbours look like?" GNNs explicitly propagate collaborative signals through the graph, so similar users (connected through shared items) end up with similar embeddings as a side-effect of the architecture. Multi-layer GNNs capture higher-order patterns ("users who liked items similar to items you liked") that matrix factorisation cannot express in a single dot product.

### Q2: GCN vs. GAT vs. GraphSAGE — when do I use which?

| | GCN | GAT | GraphSAGE |
|---|---|---|---|
| **Strengths** | Simple, efficient | Adaptive neighbour weighting | Scalable, handles new nodes |
| **Weaknesses** | Needs full graph, fixed weights | Expensive for large neighbourhoods | Sampling introduces variance |
| **Best for** | Small/medium static graphs | Heterogeneous edges | Large, evolving graphs |

**Rule of thumb:** start with LightGCN (simplified GCN) for collaborative filtering. Reach for GraphSAGE when you need inductive learning, and GAT when different neighbours carry very different importance.

### Q3: How many GNN layers should I use?

Two to three layers is standard for recommendation:

- **1 layer** — direct neighbours only; misses collaborative patterns.
- **2 layers** — friends-of-friends; sweet spot for most tasks.
- **3 layers** — three-hop patterns; slight gain, watch for over-smoothing.
- **4+ layers** — over-smoothing dominates; embeddings become indistinguishable.

Use layer combination (LightGCN-style) to get the benefit of multiple scales without committing to a single deep stack.

### Q4: How do I handle cold-start users in a GNN?

1. **GraphSAGE-style induction.** Aggregation works for any node with at least one neighbour.
2. **Feature initialisation.** Initialise the new user's embedding from side features (demographics, device) or the mean of similar users.
3. **Meta-learning.** Use MAML-style approaches to adapt rapidly from a few interactions.
4. **Hybrid models.** Combine GNN embeddings with content-based features so cold nodes still have useful representations.

See the cold-start figure above for the inductive case.

### Q5: What about computational cost?

| Model | Forward pass | Memory |
|---|---|---|
| GCN / LightGCN | $O(L\cdot|E|\cdot d)$ | $O(|V|\cdot d + |E|)$ |
| GAT | $O(L\cdot|E|\cdot d\cdot H)$ | $O(|V|\cdot d\cdot H + |E|)$ |
| GraphSAGE (sampled) | $O(L\cdot|V|\cdot k\cdot d)$ | $O(|V|\cdot d + k\cdot|V|)$ |

with $L$ = layers, $|E|$ = edges, $d$ = embedding dim, $H$ = attention heads, $k$ = sampled neighbours. For graphs with billions of edges, sampling (GraphSAGE, PinSage) or subgraph partitioning (Cluster-GCN) is mandatory.

### Q6: Does social recommendation always help?

Only when social ties genuinely reflect shared preferences. Empirical studies show 5–15% NDCG lift on informative social graphs. Noisy social connections (random follower lists) can hurt. Use attention to down-weight irrelevant edges, or pre-filter the social graph by interaction overlap.

### Q7: Can GNNs handle dynamic graphs?

Yes, with modifications:

- **Time-aware aggregation** — weight neighbours by recency.
- **Temporal encoding** — add time embeddings to edges.
- **Incremental updates** — fine-tune on new edges instead of retraining.
- **Temporal GNNs** — TGN and friends handle streaming graphs natively.

```python
def temporal_aggregate(neighbor_embs, timestamps, current_time, decay_rate=0.1):
    """Weight neighbours by how recent they are."""
    time_diffs = current_time - timestamps
    weights = torch.exp(-decay_rate * time_diffs)
    weights = weights / weights.sum()
    return (neighbor_embs * weights.unsqueeze(1)).sum(dim=0)
```

### Q8: How do I prevent over-smoothing?

1. **Limit depth** to 2–3 layers.
2. **Layer combination** (LightGCN-style) — equal-weight all layers so the final representation still contains the original embedding.
3. **Residual connections** — $\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \mathrm{GNN}(\mathbf{h}^{(l)})$.
4. **Edge dropout** — randomly drop edges during training to reduce structural over-reliance.

---

## Key Takeaways

- **Recommendation data is inherently graph-structured.** GNNs exploit this structure by propagating collaborative signals through neighbourhood aggregation.
- **LightGCN proves simplicity wins.** Stripping GCN down to bare aggregation often outperforms heavier architectures on collaborative filtering.
- **GraphSAGE enables inductive learning.** Critical for production systems where new users and items appear constantly — and the cleanest path to handling cold start.
- **PinSage demonstrates billion-scale feasibility.** Random-walk sampling and importance weighting make GNNs practical at Pinterest scale.
- **Social signals provide 5–15% lift** — but only when social connections genuinely reflect shared preferences.
- **Two to three layers is the sweet spot.** Deeper risks over-smoothing; layer combination buys depth without collapse.

---

## Series Navigation

This article is **Part 7** of the 16-part Recommendation Systems series.

| Part | Topic |
|---:|:---|
| 1 | [Fundamentals and Core Concepts](/en/recommendation-systems-1-fundamentals/) |
| 2 | [Collaborative Filtering](/en/recommendation-systems-2-collaborative-filtering/) |
| 3 | [Deep Learning Basics](/en/recommendation-systems-3-deep-learning-basics/) |
| 4 | [CTR Prediction](/en/recommendation-systems-4-ctr-prediction/) |
| 5 | [Embedding Techniques](/en/recommendation-systems-5-embedding-techniques/) |
| 6 | [Sequential Recommendation](/en/recommendation-systems-6-sequential-recommendation/) |
| **7** | **Graph Neural Networks and Social Recommendation (you are here)** |
| 8 | [Knowledge Graph-Enhanced Recommendation](/en/recommendation-systems-8-knowledge-graph/) |
| 9 | [Multi-Task and Multi-Objective Learning](/en/recommendation-systems-9-multi-task/) |
| 10 | [Deep Interest Networks](/en/recommendation-systems-10-deep-interest-networks/) |
| 11 | [Contrastive Learning](/en/recommendation-systems-11-contrastive-learning/) |
| 12 | [LLM-Enhanced Recommendation](/en/recommendation-systems-12-llm-recommendation/) |
| 13 | [Fairness and Explainability](/en/recommendation-systems-13-fairness-explainability/) |
| 14 | [Cross-Domain and Cold Start](/en/recommendation-systems-14-cross-domain-cold-start/) |
| 15 | [Real-Time and Online Learning](/en/recommendation-systems-15-real-time-online/) |
| 16 | [Industrial Practice](/en/recommendation-systems-16-industrial-practice/) |
