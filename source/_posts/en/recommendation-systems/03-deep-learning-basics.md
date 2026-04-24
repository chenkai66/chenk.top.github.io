---
title: "Recommendation Systems (3): Deep Learning Foundations"
date: 2024-05-04 09:00:00
tags:
  - Recommendation Systems
  - Deep Learning
  - Neural Networks
  - Embeddings
categories:
  - Recommendation Systems
series: recommendation-systems
lang: en
mathjax: true
permalink: en/recommendation-systems-3-deep-learning-basics/
description: "From MLPs to embeddings to NeuMF, YouTube DNN, and Wide & Deep -- a progressive walkthrough of the deep learning building blocks that power every modern recommender, with verified architectures and runnable PyTorch code."
---

In June 2016, Google published a one-page paper that quietly redrew the map of recommendation systems. The paper described **Wide & Deep Learning**, the model then powering app recommendations inside Google Play -- a billion-user product. Within a year, every major tech company had a deep model in production. By 2019, the industry standard had shifted: matrix factorization was a baseline, not a system.

What changed? Multi-layer neural networks brought four capabilities classical methods could not deliver:

- **Learned representations.** Embedding layers replace one-hot vectors with dense, semantic vectors -- learned end-to-end from clicks.
- **Nonlinear interactions.** A two-layer MLP with ReLU can fit XOR; a dot product cannot.
- **Multimodal fusion.** Text, images, and behavior sequences flow through the same gradient.
- **End-to-end optimization.** No more hand-tuned feature crosses; the loss decides.

This article walks the path that took the field from `dot(p_u, q_i)` to NeuMF, YouTube DNN, and Wide & Deep -- with the architectures verified against the original papers, and runnable PyTorch code at every step.

## What you will build a feel for

- **The MLP intuition** -- why stacking linear layers with ReLU is a universal interaction engine.
- **Embeddings** -- not just `nn.Embedding`, but *why* gradients pull similar IDs together.
- **NeuMF** (He et al., WWW 2017) -- two paths, one objective.
- **YouTube DNN** (Covington et al., RecSys 2016) -- the two-stage pipeline used by every large-scale recommender today.
- **Wide & Deep** (Cheng et al., DLRS 2016) -- the textbook fusion of memorization and generalization.

## Prerequisites

- Comfort with PyTorch (`nn.Module`, autograd, `DataLoader`).
- [Part 2 of this series](/en/recommendation-systems-2-collaborative-filtering/) (matrix factorization, implicit vs. explicit feedback).
- Basic linear algebra: dot products, matrix-vector multiplies.

---

## 1. Why deep learning, and why now

### The ceiling classical methods hit

![Bar chart and trend line comparing AUC of MF, FM, Wide & Deep, DeepFM, and DIN on a CTR benchmark, with deep models showing 4-13% improvement over MF baseline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig1_dl_vs_traditional.png)

The numbers above are typical of public CTR benchmarks (Criteo, Avazu, MovieLens-1M with implicit feedback). They tell a consistent story: **every additional source of nonlinearity buys a few AUC points, and a few AUC points are worth a lot of GMV.**

To see *why*, look at what each classical method can express.

**Matrix factorization** predicts a rating with a dot product:

$$\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$

In plain terms: each user and each item is a short vector; the prediction is their alignment. Beautiful, but linear -- it cannot capture that you love sci-fi *and* action *together* while disliking either alone.

**Factorization machines** add pairwise feature interactions:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j$$

This is a strict superset of MF -- but it stops at second order. A "young user × Friday night × thriller" three-way effect requires manual cross-feature engineering.

**Collaborative filtering** sidesteps modeling entirely and just looks for similar users or items. It works well until the matrix gets sparse, which it always does in production.

The shared limitation: **all three are at most second-order, and all three need a human to design the right features.**

### What deep models add

A neural network with one hidden layer and a nonlinearity is, in theory, a universal function approximator. In practice, that means:

- An MLP on top of an embedding can fit *any* finite-order interaction the data warrants.
- The same backbone consumes images (CNN), text (Transformer), and sequences (RNN) -- all jointly trained.
- Cold-start gets a hook: if the new item has *content* (a title, an image), pre-trained encoders give it a sensible initial vector.

The price: more compute, less interpretability, more hyperparameters. Section 7 covers the engineering discipline that makes this trade pay off.

---

## 2. The MLP intuition: from dot product to nonlinear interaction

Before diving into named architectures, it helps to internalize what an MLP buys you over a dot product.

A dot product $\mathbf{p}^\top \mathbf{q} = \sum_k p_k q_k$ adds up coordinate-wise products. It is symmetric, linear, and incapable of expressing "feature A matters *only when* feature B is also present."

Concatenate $[\mathbf{p}; \mathbf{q}]$ and pass through `Linear → ReLU → Linear`:

$$f(\mathbf{p}, \mathbf{q}) = \mathbf{w}^\top \, \text{ReLU}\!\big(\mathbf{W} [\mathbf{p}; \mathbf{q}] + \mathbf{b}\big)$$

Now the ReLU gates each hidden unit on or off depending on which combination of input dimensions is active. With enough hidden units, this is exactly the universal-approximation result. **The interaction is no longer a fixed formula -- it is learned.**

This single substitution -- replace dot product with MLP -- is the seed from which NeuMF, YouTube DNN, and Wide & Deep all grow.

---

## 3. Embeddings: the bridge from sparse IDs to learned semantics

### One-hot is the enemy

Picture a catalog with 10 million users and 1 million items. One-hot encoding gives every user a 10-million-dimensional vector with a single 1. Three things go wrong at once:

1. **Storage and compute** explode -- a single user input becomes a 40 MB float vector.
2. **Information density** collapses -- 99.99999% of the vector is zero.
3. **All distances are equal** -- $\|\mathbf{e}_i - \mathbf{e}_j\|_2 = \sqrt{2}$ for every $i \ne j$. User 42 is no closer to user 43 than to user 9,999,999.

An embedding layer fixes all three. It maps each ID to a dense vector of, say, 64 dimensions. After training, **users with similar tastes land near each other** in that 64-D space.

> **Analogy.** Think of one-hot as a phone book where every name is on its own island, all islands the same distance apart. Embedding is the geographer who rearranges the islands so that close-by islands have related people on them. Suddenly "find similar users" becomes a nearest-neighbor lookup.

### What an embedding actually is

Mathematically, the embedding layer is a learned matrix $\mathbf{P} \in \mathbb{R}^{m \times d}$ where row $i$ is user $i$'s vector. The "lookup" operation $\mathbf{p}_i = \mathbf{P}[i, :]$ is mathematically equivalent to $\mathbf{P}^\top \mathbf{e}_i$, but implemented as a row index for speed.

In code, that is one line:

```python
import torch
import torch.nn as nn

user_embedding = nn.Embedding(num_embeddings=10_000_000, embedding_dim=64)
user_vec = user_embedding(torch.LongTensor([42]))   # shape: [1, 64]
```

### How gradients teach embeddings to mean something

Embeddings start random. They become semantic only because gradients push them.

Take a click prediction loss. When user $u$ clicks item $i$, the loss tells the optimizer: "make $\mathbf{p}_u^\top \mathbf{q}_i$ larger." Backprop nudges $\mathbf{p}_u$ slightly toward $\mathbf{q}_i$ and vice versa. Over millions of clicks, this single rule produces stunning structure: items watched by overlapping audiences end up close, items co-purchased end up close, songs from the same artist end up close.

You did not tell the model what a "genre" is. The geometry of the gradient discovered it.

### Visualizing what was learned

![t-SNE projection of learned item embeddings showing tight clusters for sci-fi, action, documentary, romcom, and horror films, with hybrid items bridging related clusters and a long-tail halo of low-interaction items at the periphery](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig5_embedding_space.png)

Project a trained item embedding matrix to 2D with t-SNE and you typically see exactly this picture: tight clusters of same-category items, bridges of hybrid items between related categories (sci-fi action sits between sci-fi and action), and a halo of long-tail items at the periphery. This visualization is your single best diagnostic for whether the model is learning anything at all.

### Choosing the dimension

| Catalog size | Recommended $d$ | Notes |
|---|---|---|
| < 100K | 8 -- 32 | Larger $d$ overfits. |
| 100K -- 1M | 32 -- 64 | The sweet spot for most domains. |
| 1M -- 100M | 64 -- 128 | Diminishing returns above 128. |
| Web-scale | 128 -- 256 | Only if you have billions of interactions. |

A useful heuristic from the YouTube paper and confirmed in many follow-ups: **start at $d = 32$, double until validation AUC stops moving by more than ~0.5%, then stop.** Memory and serving latency are linear in $d$; quality is concave.

### A clean, reusable embedding layer

```python
import torch
import torch.nn as nn

class IdEmbedding(nn.Module):
    """Embedding layer with Xavier init and optional padding index."""

    def __init__(self, vocab_size: int, dim: int, padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.emb.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[padding_idx].zero_()

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        return self.emb(ids)
```

For multi-field categorical inputs (user ID, item ID, category, city, ...), give each field its own table and stack:

```python
class MultiFieldEmbedding(nn.Module):
    def __init__(self, field_dims: list[int], dim: int):
        super().__init__()
        self.tables = nn.ModuleList([IdEmbedding(v, dim) for v in field_dims])

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: [batch, num_fields] -> [batch, num_fields, dim]
        return torch.stack([t(x[:, i]) for i, t in enumerate(self.tables)], dim=1)
```

This `[batch, num_fields, dim]` tensor is the input shape every model below expects.

---

## 4. Neural Collaborative Filtering (NCF and NeuMF)

### The promise: replace the dot product with something smarter

He, Liao, Zhang, Nie, Hu, and Chua introduced **Neural Collaborative Filtering** at WWW 2017. Their pitch was disarmingly simple: matrix factorization is a special case of a network. If we generalize it, we can do better.

The NCF paper proposed three siblings:

- **GMF** (Generalized Matrix Factorization) -- a learnable weighted version of the dot product.
- **MLP** -- pure deep concatenation, no inductive bias toward inner products.
- **NeuMF** -- the fusion of GMF and MLP, with separate embeddings for each path.

NeuMF is the one that matters in practice. Here is its architecture, faithful to the paper.

### NeuMF architecture

![NeuMF architecture diagram showing two paths fused at the top: GMF path with element-wise product of separate user/item embeddings on the left, MLP path with concatenated embeddings flowing through three ReLU layers on the right, and a final sigmoid head consuming the concatenation of both path outputs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig2_ncf_architecture.png)

Read the diagram bottom-up:

1. **Two embedding tables per side.** GMF and MLP do *not* share embeddings -- the paper found this matters. Each path learns the representation that suits its objective.
2. **GMF path.** Element-wise product $\mathbf{p}_u^\text{GMF} \odot \mathbf{q}_i^\text{GMF}$. With a learned weight on top, this is a generalization of the standard dot product.
3. **MLP path.** Concatenate $[\mathbf{p}_u^\text{MLP}; \mathbf{q}_i^\text{MLP}]$, then 2--3 dense + ReLU layers (typical: $128 \to 64 \to 32$).
4. **Fusion.** Concatenate the two path outputs and project to a scalar through sigmoid:

$$\hat{y}_{ui} = \sigma\!\left(\mathbf{h}^\top \begin{bmatrix} \mathbf{p}_u^\text{GMF} \odot \mathbf{q}_i^\text{GMF} \\ \mathbf{z}_L^\text{MLP} \end{bmatrix}\right)$$

For implicit feedback (clicks, plays, purchases), the loss is binary cross-entropy:

$$\mathcal{L} = -\sum_{(u, i) \in \mathcal{D}^+ \cup \mathcal{D}^-} \big[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \big]$$

The negative set $\mathcal{D}^-$ is built by sampling -- typically 4 negatives per positive.

### NeuMF, end to end

```python
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    """NeuMF: fuses GMF and MLP, separate embeddings per path (He et al., 2017)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        mlp_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        # Separate embeddings for the two paths -- key NeuMF design choice.
        self.user_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_gmf = nn.Embedding(num_items, embedding_dim)
        self.user_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP tower over concatenated user/item vectors.
        layers, in_dim = [], embedding_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Final head sees [GMF vector ; MLP vector].
        self.head = nn.Linear(embedding_dim + mlp_layers[-1], 1)

        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, users: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)           # [B, d]
        mlp_in = torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)                                  # [B, mlp_layers[-1]]
        logits = self.head(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)


# Smoke test
model = NeuMF(num_users=10_000, num_items=5_000, embedding_dim=32, dropout=0.2)
users = torch.randint(0, 10_000, (8,))
items = torch.randint(0, 5_000, (8,))
labels = torch.randint(0, 2, (8,)).float()
preds = model(users, items)
loss = nn.BCELoss()(preds, labels)
print(f"preds={preds.detach().numpy().round(3)}  loss={loss.item():.4f}")
```

### Practical NeuMF training tips

Three things from the paper that people forget:

- **Pre-train each path.** Train GMF alone, then MLP alone, then initialize NeuMF from the two checkpoints. The paper reports a ~2% AUC boost from this trick alone.
- **Negative sampling ratio matters.** 4 negatives per positive is the canonical default; 1:1 underfits, 1:10 wastes compute.
- **No L2 on biases.** Weight decay on bias terms degrades performance; mask them in your optimizer's parameter groups.

---

## 5. YouTube DNN: the two-stage pipeline that runs the internet

Covington, Adams, and Sargin (RecSys 2016) is the most-cited industrial recommender paper of the deep-learning era. Its two-stage decomposition -- **candidate generation** then **ranking** -- is the template for almost every large-scale recommender shipped since: TikTok, Spotify, Pinterest, Instagram, Taobao.

### Why two stages?

You cannot score billions of videos for every request. You also cannot use rich features for billions of items in real time. The fix:

1. **Candidate generation** narrows the corpus from millions of items to a few hundred plausible ones, using a fast model and cheap features.
2. **Ranking** then applies a much heavier model with rich features to those few hundred and picks the top-K.

### Architecture

![YouTube DNN two-stage pipeline shown side by side: candidate generation tower on the left consuming watch history, search tokens, geo, age, and gender, producing a 256-dim user vector that hits an ANN index over video embeddings; ranking tower on the right consuming richer features through a four-layer MLP and predicting expected watch time via weighted logistic regression](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig3_youtube_dnn.png)

**Candidate generation** is framed as extreme multi-class classification: "given this user state, predict which video they will watch next, out of millions." The user tower averages embeddings of recently watched videos, concatenates demographic features, runs three ReLU layers ($1024 \to 512 \to 256$), and outputs a 256-D **user vector**. At training time the loss is sampled softmax over the full video corpus. At serving time, the learned video embeddings live in an ANN index (HNSW or ScaNN), and the user vector becomes a nearest-neighbor query -- single-digit milliseconds for billions of items.

**Ranking** is a heavier feed-forward network ($1024 \to 512 \to 256 \to 128$) over much richer features: the impression video's embedding, embeddings of previously watched videos in the same channel, *time since last watch*, *position in feed*, language match, and so on. Critically, the head is a **weighted logistic regression** trained to predict expected watch time -- not click. The paper showed this aligns better with long-term satisfaction than CTR alone.

### What to copy from the YouTube paper

Three design choices have aged extremely well:

- **Average-pool the user's recent behavior.** Cheap, parallel, and a strong baseline. Sequence models (Part 6) only beat it once you have enough data.
- **Treat candidate generation as classification, not regression.** Sampled softmax + ANN serving is the dominant pattern industry-wide.
- **Choose your label carefully.** "Watch time," "long click" (>30s dwell), or "completed view" generalize better than raw click. Your loss is the product spec.

A skeleton in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class YouTubeCandidateTower(nn.Module):
    """Candidate generation tower: produces a user vector for ANN lookup."""

    def __init__(
        self,
        num_videos: int,
        num_searches: int,
        num_geos: int,
        embedding_dim: int = 64,
        user_dim: int = 256,
    ):
        super().__init__()
        self.video_emb = nn.Embedding(num_videos, embedding_dim, padding_idx=0)
        self.search_emb = nn.Embedding(num_searches, embedding_dim, padding_idx=0)
        self.geo_emb = nn.Embedding(num_geos, embedding_dim)

        in_dim = 3 * embedding_dim + 2  # video pool + search pool + geo + age + gender
        self.tower = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, user_dim), nn.ReLU(),
        )

    def forward(self, watched, searched, geo, age, gender):
        # watched, searched: [B, L] padded; mean-pool over non-padding tokens.
        v = self._masked_mean(self.video_emb(watched), watched != 0)
        s = self._masked_mean(self.search_emb(searched), searched != 0)
        g = self.geo_emb(geo)
        x = torch.cat([v, s, g, age.unsqueeze(-1), gender.unsqueeze(-1)], dim=-1)
        return F.normalize(self.tower(x), dim=-1)  # cosine-friendly user vector

    @staticmethod
    def _masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (emb * m).sum(1) / m.sum(1).clamp(min=1.0)
```

At training time, pair this user vector with a softmax over a sampled set of candidate video embeddings; at serving time, dot-product against an ANN index. That single tower is the workhorse of modern industrial recall.

---

## 6. Wide & Deep: memorization meets generalization

### The insight

Cheng et al. (DLRS 2016) noticed that a deep model alone, while better at generalizing, sometimes over-recommends -- it suggests reasonable-but-wrong items because the embeddings smooth too much. Conversely, a linear model with cross features memorizes specific co-occurrences perfectly but cannot extrapolate.

> **Analogy.** Memorization is the friend who says "you liked Inception, you'll like Tenet" -- specific, accurate, but never adventurous. Generalization is the friend who says "you like cerebral thrillers, try Primer" -- broader, sometimes wrong, but capable of surprise. A great recommender is both friends in one.

Their fix: train them **jointly**, summing the two scores before the sigmoid.

### Architecture

![Wide and Deep architecture diagram: wide linear branch on the left consuming sparse and manually crossed features through a single linear layer to produce a wide score, deep branch on the right embedding categorical fields and running them through three ReLU layers to produce a deep score, both summed and passed through a sigmoid to predict click probability, with FTRL+L1 noted for the wide side and AdaGrad/Adam for the deep side](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig4_wide_and_deep.png)

**Wide.** A linear layer over the original sparse features and **manually constructed cross features** $\phi(\mathbf{x})$, e.g., `installed_app=Pandora AND impression_app=YouTube`. Outputs $\hat{y}^w = \mathbf{w}^\top [\mathbf{x}, \phi(\mathbf{x})] + b$.

**Deep.** Embed every categorical field, concatenate, run through 3 ReLU layers ($256 \to 128 \to 64$). Outputs $\hat{y}^d$.

**Joint head.** $\hat{y} = \sigma(\hat{y}^w + \hat{y}^d)$. Both sides receive gradients from the same loss; the optimizer decides how much each contributes.

The paper used a deliberate split: **FTRL with L1 for the wide side** (sparse, interpretable, picks features), **AdaGrad for the deep side** (dense, smooth). This bi-optimizer setup is essential -- using one optimizer for both hurts.

### Implementation

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    """Wide & Deep (Cheng et al., 2016) for binary CTR prediction."""

    def __init__(
        self,
        wide_dim: int,                  # dimension of the sparse + cross feature vector
        field_dims: list[int],          # vocab sizes of categorical fields for the deep side
        embedding_dim: int = 32,
        deep_layers: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        # Wide side: a single big sparse linear layer.
        self.wide = nn.Linear(wide_dim, 1)

        # Deep side: per-field embeddings + MLP.
        self.embeddings = nn.ModuleList([nn.Embedding(v, embedding_dim) for v in field_dims])
        in_dim = len(field_dims) * embedding_dim
        layers = []
        for out_dim in deep_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x_wide: torch.Tensor, x_deep: torch.LongTensor) -> torch.Tensor:
        wide_score = self.wide(x_wide).squeeze(-1)
        deep_in = torch.cat([emb(x_deep[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1)
        deep_score = self.deep(deep_in).squeeze(-1)
        return torch.sigmoid(wide_score + deep_score)


# Two-optimizer training -- the production-grade detail many tutorials miss.
model = WideAndDeep(wide_dim=10_000, field_dims=[50_000, 10_000, 100, 20])
opt_wide = torch.optim.Adagrad(model.wide.parameters(), lr=0.05)
opt_deep = torch.optim.Adam(
    [p for n, p in model.named_parameters() if not n.startswith("wide.")],
    lr=1e-3, weight_decay=1e-5,
)
```

In practice you can use a single Adam optimizer and still train successfully -- but the joint Wide+Deep loss is what gives the model its name. **It is not an ensemble of two separately trained models.** The wide and deep parameters see each other's gradients, and that interaction is the point.

### Direct descendants

Wide & Deep spawned a family that automated away the manual cross features:

| Model | Replaces "Wide" with | Year |
|---|---|---|
| **DeepFM** | A factorization machine layer that learns 2nd-order crosses automatically | 2017 |
| **DCN** | A "Cross Network" that learns arbitrary-order crosses with $O(d)$ parameters per order | 2017 |
| **xDeepFM** | A Compressed Interaction Network that explicitly models high-order crosses | 2018 |
| **AutoInt** | Self-attention over feature embeddings | 2019 |

[Part 4 of this series](/en/recommendation-systems-4-ctr-prediction/) covers these in depth.

---

## 7. Training discipline that decides whether any of this works

A correct architecture is necessary but nowhere near sufficient. The following details routinely move offline AUC by more than the choice of model.

![Two-panel training dynamics chart for MF, NeuMF, and Wide & Deep over 50 epochs: left panel shows BCE training loss with deeper models reaching lower minima, right panel shows validation AUC with the gap between models holding steady and an early-stopping marker at the AUC plateau](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/03-deep-learning-basics/fig6_training_curves.png)

### Negative sampling

For implicit feedback, every user has thousands of "negatives" (items they did not see). Three sampling strategies, ordered by sophistication:

- **Uniform random.** The default, surprisingly hard to beat for retrieval.
- **Popularity-weighted.** Sample popular items more often -- if a user ignored a hit, that is a strong negative signal.
- **In-batch / hard negatives.** Use other positives in the same batch as negatives. Or, periodically retrieve top-scoring negatives from the current model. Improves discrimination but needs careful temperature tuning.

```python
import numpy as np

def sample_negatives(user_history: set[int], catalog_size: int, k: int = 4) -> list[int]:
    """Uniform random negatives, rejecting items in the user's history."""
    out = []
    while len(out) < k:
        cand = int(np.random.randint(catalog_size))
        if cand not in user_history:
            out.append(cand)
    return out
```

### Optimizer, schedule, regularization

Sensible defaults that work for >90% of recommender models:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
```

- **Dropout 0.2 -- 0.3** in MLP towers. Higher hurts ranking quality.
- **Gradient clipping** (`torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`) if you see loss spikes.
- **Early stopping** on validation AUC, patience 5--10 epochs.

### A complete training loop

```python
def train(model, train_loader, val_loader, *, epochs=50, patience=8, device="cuda"):
    model = model.to(device)
    bce = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)

    best_val, bad = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            opt.zero_grad()
            preds = model(users, items)
            loss = bce(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                val_loss += bce(model(users, items), labels).item() * len(labels)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}; best val_loss={best_val:.4f}")
                break
        print(f"epoch {epoch:>3}  train={train_loss:.4f}  val={val_loss:.4f}")
```

### Metrics that match the task

| Use case | Primary metric | What it really measures |
|---|---|---|
| CTR prediction | **AUC** | Does the model rank positives above negatives? |
| CTR prediction | **LogLoss** | Are predicted probabilities calibrated? |
| Top-K retrieval | **Recall@K**, **HitRate@K** | Did the right item appear in the top K? |
| Top-K ranking | **NDCG@K** | How highly were the right items ranked? |
| Rating prediction | **RMSE** | Average squared error in the predicted score |

If you measure CTR with RMSE or rating prediction with AUC, you will optimize the wrong knob. Pick once, document, and never silently switch.

---

## 8. Frequently asked, honestly answered

**Q: How big should my embedding dimension be?**
Start at 32 for catalogs under 1M items, 64 above. Double until validation AUC moves by less than ~0.5%. Above $d = 256$ you almost always overfit unless you have billions of interactions.

**Q: Should I always use NeuMF over MF?**
No. Below ~1M interactions, MF with proper regularization frequently beats NeuMF -- the deep model overfits. NeuMF starts to dominate clearly above ~10M interactions and rich side features.

**Q: Wide & Deep -- can I skip the wide side and just use deep?**
You can, and many do (DeepFM, DCN). The thing the wide side gives you that pure deep does not is *exact* memorization of high-cardinality co-occurrences -- "users who installed app X also installed app Y." If your business depends on those specific patterns being captured precisely, keep the wide side. Otherwise, automated cross networks (DeepFM, DCN) are usually the better trade.

**Q: Where does YouTube DNN fit if I have only a few million users?**
The two-stage pattern is overkill below ~100K items. Run a single ranker over the whole catalog. Adopt two-stage when scoring everything in your catalog at request time stops fitting your latency budget.

**Q: How do I handle a brand-new item with no interactions?**
Three options, in order of effectiveness:
1. **Content-based init** -- compute the embedding from the item's text/image with a pre-trained encoder (BERT, CLIP).
2. **Category-mean init** -- average the embeddings of items in the same category.
3. **Bandit exploration** -- expose the new item to a small fraction of traffic to gather initial signal.

**Q: How do I prevent overfitting on a deep recommender?**
Layered defenses: weight decay $10^{-5}$ on embeddings, dropout 0.2--0.3 in MLPs, early stopping on validation AUC, smaller $d$ if all else fails. Add complexity only when validation moves with it.

**Q: How do I speed training up?**
The 80/20 list: GPU first (10 -- 100x), then bigger batches (better GPU util), then mixed precision (`torch.cuda.amp` ~ 2x), then `num_workers > 0` in your `DataLoader`. Pre-compute all features offline; never join in the training loop.

---

## 9. Where this leaves us

Deep learning did not invent recommendations. It changed three constraints:

- **Representations are learned**, not designed. Embeddings replace one-hot, and the gradient discovers structure you could not have hand-coded.
- **Interactions are arbitrary**, not second-order. An MLP fits whatever the data warrants.
- **The pipeline is end-to-end**, not stages glued together. Loss flows from prediction to raw IDs.

NeuMF showed that a learned interaction beats a fixed inner product. YouTube DNN showed how to do this at billion-item scale by splitting recall and ranking. Wide & Deep showed that memorization and generalization are best learned together, not separately.

[Part 4 of this series](/en/recommendation-systems-4-ctr-prediction/) takes the next step: **CTR-prediction models that automate the cross-feature engineering Wide & Deep still relies on**, including DeepFM, DCN, xDeepFM, AutoInt, and FiBiNet.

---

## Series navigation

| Part | Topic | Link |
|---|---|---|
| 1 | Introduction to Recommendation Systems | [Read Part 1](/en/recommendation-systems-1-introduction/) |
| 2 | Collaborative Filtering | [Read Part 2](/en/recommendation-systems-2-collaborative-filtering/) |
| **3** | **Deep Learning Foundations** | **You are here** |
| 4 | CTR Prediction Models | [Read Part 4](/en/recommendation-systems-4-ctr-prediction/) |
| 5 | Embedding Techniques | [Read Part 5](/en/recommendation-systems-5-embedding-techniques/) |
| 6 | Sequential Recommendation | [Read Part 6](/en/recommendation-systems-6-sequence-models/) |
| ... | ... | ... |
| 16 | Industrial Practice and MLOps | [Read Part 16](/en/recommendation-systems-16-production/) |
