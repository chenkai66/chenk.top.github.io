---
title: "Recommendation Systems (5): Embedding and Representation Learning"
date: 2024-05-06 09:00:00
tags:
  - Recommendation Systems
  - Embedding
  - Representation Learning
categories: Recommendation Systems
lang: en
mathjax: true
series: recommendation-systems
series_title: "Part 5 of 16: Embedding and Representation Learning"
permalink: "en/recommendation-systems-5-embedding-techniques/"
description: "How modern recommenders learn dense vector representations for users and items: Word2Vec / Item2Vec, Node2Vec, two-tower DSSM and YouTube DNN, negative sampling, FAISS/HNSW serving, and how to evaluate embedding quality. Concept, math, code and production guidance in one place."
disableNunjucks: true
---

When Netflix suggests *Inception* to someone who just finished *The Dark Knight*, the magic is not a hand-crafted "if-watched-Nolan-then" rule. It is geometry. Both films sit close together in a 128-dimensional **embedding space** that the model has learned from billions of viewing events. Geometry replaces enumeration: instead of comparing a movie to fifteen thousand others through brittle similarity rules, the system asks a single question — **how far apart are these two vectors?**

This article unpacks how those vectors get learned and served at production scale. We will move from the underlying intuition through five families of techniques (sequence, graph, two-tower, attention-pooled, contrastive), the engineering of negative sampling, and the millisecond-level realities of approximate nearest-neighbour search. Every section is paired with code that compiles and runs.

## What you will learn

- **What an embedding actually is** — and why "low-dimensional" is the entire point
- **Sequence-based learning** with Word2Vec and Item2Vec, including the Skip-gram derivation
- **Graph-based learning** with Node2Vec's biased random walks
- **Two-tower architectures** (DSSM, YouTube DNN) that dominate large-scale retrieval
- **Negative sampling strategies** — uniform, popularity-aware, hard, in-batch
- **ANN serving** with FAISS, HNSW and Annoy, and how to read the latency / recall trade-off
- **Evaluation** through both intrinsic (coherence, clustering) and extrinsic (HR@K, NDCG) metrics

## Prerequisites

- Linear algebra basics (vectors, dot products, matrix multiplication)
- Familiarity with neural networks and PyTorch
- Recommendation system fundamentals — see [Part 1](/en/recommendation-systems-1-fundamentals/)
- Helpful but not required: deep learning for recommendations — see [Part 3](/en/recommendation-systems-3-deep-learning-basics/)

---

## 1. Foundations: what an embedding is, and why it matters

### 1.1 The compression view

An **embedding** is a learned function $f : \mathcal{I} \to \mathbb{R}^{d}$ that maps a discrete object — a user, a movie, a SKU, a node in a graph — to a dense vector of $d$ real numbers, where $d$ is much smaller than the catalogue size.

> **Analogy.** Picture a Netflix-style catalogue with a million titles. A naïve representation would need a million-dimensional one-hot vector per movie. Embeddings replace that with, say, 128 numbers per movie — a *vector of latent qualities* that nobody labelled but the model discovered: how cerebral, how violent, how visually dark, how 1990s.

Three properties make embeddings the lingua franca of modern recommenders:

| Property | What it gives you |
|---|---|
| **Density** | Every dimension carries information; no wasted bits compared to one-hot vectors. |
| **Geometry** | "Similar" becomes a measurable quantity — a dot product or a cosine. |
| **Composability** | Embeddings can be averaged, concatenated, attended over, and indexed for retrieval. |

### 1.2 Why this beats sparse representations

Real interaction matrices are absurdly sparse. A platform with $10^{8}$ users and $10^{7}$ items has $10^{15}$ possible cells but typically observes fewer than $10^{11}$ — under 0.01% density. Storing or factoring that matrix directly is infeasible. Compressing each row and column into a $d$-dimensional vector turns the problem into something modern hardware loves: dense matrix multiplications.

![Item embedding space, t-SNE projection, four clusters by category with a query item and its k-NN region highlighted](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig1_embedding_space.png)

The picture above is the central intuition of this entire article. Items belonging to the same category form clusters; the query item's neighbours in vector space are exactly the items we want to recommend. **Recommendation reduces to nearest-neighbour search in an embedding space.**

### 1.3 The learning objective in one sentence

Almost every embedding method, from matrix factorisation to BERT4Rec, optimises the same idea:

> **Items that appear in similar contexts should end up with similar vectors; items that do not should be pushed apart.**

This is the [distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics) borrowed from linguistics — "you shall know a word by the company it keeps." In recommender land, "context" can be:

- a co-occurrence in a user session (Item2Vec)
- adjacency in a graph (Node2Vec, GraphSAGE)
- a click on the same query (DSSM)
- a positive label in a CTR log (DeepFM, DLRM)

The choice of context defines the method.

---

## 2. Sequence-based embeddings: Word2Vec and Item2Vec

### 2.1 From words to items

Word2Vec (Mikolov et al., 2013) had two flavours:

- **Skip-gram** — given a centre word, predict its surrounding context.
- **CBOW (continuous bag-of-words)** — given the surrounding context, predict the centre word.

> **Analogy.** Skip-gram is like handing someone a single jigsaw piece and asking what surrounds it. CBOW is the reverse — show the surrounding pieces and ask what the missing centre piece is.

Item2Vec (Barkan & Koenigstein, 2016) is the trivial-looking but powerful adaptation: **treat a user's interaction sequence as a sentence and each item as a word.** All the Word2Vec machinery transfers verbatim.

### 2.2 The Skip-gram objective, derived

Given a sequence $S = [i_1, i_2, \dots, i_T]$ and a window size $c$, Skip-gram maximises the log-probability of seeing each context item given its centre:

$$
\mathcal{L} \;=\; \sum_{t=1}^{T} \;\sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p\!\left(i_{t+j}\,\middle|\,i_t\right).
$$

The naïve probability is a softmax over the whole catalogue:

$$
p\!\left(i_{t+j}\,\middle|\,i_t\right) \;=\; \frac{\exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_{t+j}}\right)}{\displaystyle\sum_{k=1}^{|\mathcal{I}|} \exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{k}\right)}.
$$

Here $\mathbf{e}_i$ is the **input** (centre) embedding and $\mathbf{e}'_i$ is the **output** (context) embedding. Two sets of vectors per item — a small surprise the first time you see it.

Computing that denominator over millions of items is unworkable. **Negative sampling** replaces it with a binary classification problem: for each true (centre, context) pair, sample $K$ random "noise" items and ask the model to discriminate them. The objective becomes

$$
\mathcal{L} \;=\; \sum_{(i_t,\,i_c)} \!\left[\, \log\sigma(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_c})
\;+\; \sum_{k=1}^{K} \mathbb{E}_{i_k \sim P_n}\!\big[\log\sigma(-\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_k})\big]\right],
$$

where $\sigma$ is the sigmoid. The noise distribution $P_n$ is the unigram frequency raised to the $3/4$ power — a famous Mikolov heuristic that nudges rarer items into the negative pool more than pure frequency would.

![Item2Vec Skip-gram architecture: a sliding context window selects a centre item, an embedding lookup produces a vector, and the model contrasts the positive context items against K negatives sampled from the 3/4-power unigram distribution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig2_item2vec_skipgram.png)

### 2.3 Implementation

A self-contained Item2Vec with Skip-gram + negative sampling. Read it once paying attention to the **two embedding tables**, the **score clamping**, and the **sigmoid trick**.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random


class Item2Vec(nn.Module):
    """Skip-gram with Negative Sampling.

    Two embedding tables:
      - input_embeddings: used at inference; this is the "true" item vector
      - output_embeddings: a context-side helper, discarded after training
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_negatives: int = 5):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_negatives = num_negatives

        # Xavier init keeps the initial dot products small enough that
        # sigmoid does not saturate on iteration 1.
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, target, context, negatives):
        """
        target    : (B,)          centre item ids
        context   : (B,)          true neighbour ids
        negatives : (B, K)        K random items per row
        """
        t = self.input_embeddings(target)      # (B, d)
        c = self.output_embeddings(context)    # (B, d)
        n = self.output_embeddings(negatives)  # (B, K, d)

        # Positive term: pull centre and true context together.
        pos = torch.clamp((t * c).sum(-1), -10, 10)
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-10)

        # Negative term: push centre away from K random items.
        neg = torch.clamp(torch.bmm(n, t.unsqueeze(-1)).squeeze(-1), -10, 10)
        neg_loss = -torch.log(torch.sigmoid(-neg) + 1e-10).sum(-1)

        return (pos_loss + neg_loss).mean()

    @torch.no_grad()
    def vector(self, item_id: int) -> np.ndarray:
        return self.input_embeddings.weight[item_id].cpu().numpy()


# --- data plumbing -------------------------------------------------------- #

def build_vocab(sequences):
    counter = Counter(item for seq in sequences for item in seq)
    vocab = {item: idx for idx, item in enumerate(counter)}
    return vocab, counter


def make_pairs(sequences, vocab, window=5):
    pairs = []
    for seq in sequences:
        ids = [vocab[i] for i in seq if i in vocab]
        for t, centre in enumerate(ids):
            lo, hi = max(0, t - window), min(len(ids), t + window + 1)
            for j in range(lo, hi):
                if j != t:
                    pairs.append((centre, ids[j]))
    return pairs


def make_neg_sampler(counter, vocab):
    items = list(counter)
    probs = np.array([counter[i] ** 0.75 for i in items], dtype=np.float64)
    probs /= probs.sum()
    idx_lookup = np.array([vocab[i] for i in items])

    def sample(n_rows: int, k: int) -> np.ndarray:
        chosen = np.random.choice(len(items), size=(n_rows, k), p=probs)
        return idx_lookup[chosen]

    return sample


# --- training loop -------------------------------------------------------- #

if __name__ == "__main__":
    sequences = [
        [1, 2, 3, 4, 5], [2, 3, 4, 6, 7],
        [1, 3, 5, 8, 9], [4, 5, 6, 10, 11],
    ]
    vocab, counter = build_vocab(sequences)
    pairs = make_pairs(sequences, vocab, window=2)
    neg = make_neg_sampler(counter, vocab)

    model = Item2Vec(len(vocab), embedding_dim=64, num_negatives=5)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    BATCH = 32
    for epoch in range(10):
        random.shuffle(pairs)
        running = 0.0
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            t = torch.tensor([p[0] for p in batch], dtype=torch.long)
            c = torch.tensor([p[1] for p in batch], dtype=torch.long)
            ng = torch.tensor(neg(len(batch), 5), dtype=torch.long)

            opt.zero_grad()
            loss = model(t, c, ng)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"epoch {epoch + 1:>2}  loss {running / max(1, len(pairs) // BATCH):.4f}")
```

### 2.4 Design decisions worth defending

| Decision | Choice | Why it matters |
|---|---|---|
| Two embedding tables | Separate centre / context | Gives the model more room to encode "what I am" vs. "what I appear next to". Standard in Word2Vec. |
| Negative count $K$ | 5 – 20 | Small $K$ trains fast; large $K$ better discriminates long-tail items. Diminishing returns past 20. |
| Window size $c$ | 2 – 5 | Bigger windows pick up more co-occurrence noise. Sessions are short; a small window is usually enough. |
| Negative distribution | $P_n \propto f^{0.75}$ | Pure-frequency over-samples blockbusters; uniform over-samples obscure long tails. The 3/4 exponent is the empirical sweet spot Mikolov found. |
| `clamp(-10, 10)` on dot products | Numerical guard | A single overflowed sigmoid can NaN the whole epoch. Cheap insurance. |

### 2.5 Field-tested gotchas

- **Cold-start items.** An item with zero training interactions has no embedding. Three remedies: (1) train a side network from item content (text, image) to predict its embedding; (2) initialise from the average of categorically similar items; (3) accept randomness and let early online traffic pull it into place.
- **Variable session length.** Truncate to the most recent $N$ items (typically 50 – 100) and pad shorter sessions. For very long sessions, segment into windows.
- **Negative collisions.** With a million-item catalogue, drawing a "false negative" that is actually positive happens with probability $< 0.1\%$ — ignore it. With a 1k-item catalogue, explicitly exclude positives.
- **Repeated items.** Power users replay the same song fifty times. De-duplicate consecutive repeats before building pairs, otherwise the model just learns "this song is similar to itself."

### 2.6 CBOW: the symmetric variant

CBOW averages the surrounding context embeddings into one vector and uses *that* to predict the centre. It trains faster and tends to be slightly worse on long-tail items — Skip-gram sees each context item independently while CBOW averages everything into a single shot.

```python
class Item2VecCBOW(nn.Module):
    """Predict the centre item from an averaged window of context items."""

    def __init__(self, vocab_size, embedding_dim, num_negatives=5):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embedding_dim)
        self.out_emb = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, context, target, negatives):
        ctx = self.in_emb(context).mean(dim=1)        # (B, d)
        tgt = self.out_emb(target)                    # (B, d)
        neg = self.out_emb(negatives)                 # (B, K, d)

        pos = torch.clamp((ctx * tgt).sum(-1), -10, 10)
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-10)

        neg = torch.clamp(torch.bmm(neg, ctx.unsqueeze(-1)).squeeze(-1), -10, 10)
        neg_loss = -torch.log(torch.sigmoid(-neg) + 1e-10).sum(-1)
        return (pos_loss + neg_loss).mean()
```

> **Pick Skip-gram for recommenders by default.** Item popularity follows a long tail; Skip-gram allocates gradient to rare items more generously than CBOW.

---

## 3. Graph-based embeddings: Node2Vec

### 3.1 When sequences are not enough

Item2Vec only sees what is *adjacent in time*. But the relationships in a marketplace look more like a graph: items are co-purchased, co-viewed, share a category, share a brand. The same item can be relevant in two completely different "neighbourhoods" — a hiking tent is near "camping" *and* near "cycling gear". Sequence models flatten that structure; graph models preserve it.

### 3.2 The biased random walk

Node2Vec (Grover & Leskovec, 2016) takes a graph and produces "sentences" by walking from node to node. The trick is **how** it walks: a tunable bias makes the walker prefer either staying close to home (BFS-like, captures community) or wandering far (DFS-like, captures structural roles).

> **Analogy.** Imagine exploring a city. BFS-style is "every shop on this street, then the next street" — you map out one neighbourhood thoroughly. DFS-style is "follow the main road for ten kilometres" — you discover how neighbourhoods connect to each other. Node2Vec lets you dial between the two.

Two parameters do the work. After arriving at node $v$ from $t$, the unnormalised probability of moving to a neighbour $x$ is

$$
\alpha_{p, q}(t, x) \;=\;
\begin{cases}
1/p & \text{if } d_{t,x} = 0 \;\;\text{(go back to } t\text{)} \\
1   & \text{if } d_{t,x} = 1 \;\;\text{(}x\text{ is also a neighbour of } t\text{)} \\
1/q & \text{if } d_{t,x} = 2 \;\;\text{(}x\text{ is one step further away)}
\end{cases}
$$

| Setting | Walk behaviour | Captures |
|---|---|---|
| small $p$, large $q$ | BFS-like | Community / cluster structure |
| large $p$, small $q$ | DFS-like | Structural equivalence, hub-spoke patterns |
| $p = q = 1$ | Plain random walk (DeepWalk) | Mix of both |

![Biased random walk on a user-item bipartite graph: arrows trace the path U2 -> I3 -> U4 -> I8 -> I7 -> U3 -> I4, illustrating how a single walk visits both user and item nodes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig4_random_walk.png)

Once you have a corpus of walks, you feed them into Skip-gram exactly as in Item2Vec — the algorithm does not care whether the "sentence" came from user history or graph traversal.

### 3.3 Implementation

```python
import numpy as np
import networkx as nx
import random


class Node2Vec:
    """Biased random walks + Skip-gram on a NetworkX graph."""

    def __init__(self, graph, dimensions=64, walk_length=80,
                 num_walks=10, p=1.0, q=1.0, window_size=10):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size

    def _walk(self, start):
        walk = [start]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            nbrs = list(self.graph.neighbors(cur))
            if not nbrs:
                break

            if len(walk) == 1:
                walk.append(random.choice(nbrs))
                continue

            prev = walk[-2]
            weights = []
            for x in nbrs:
                w = self.graph[cur][x].get("weight", 1.0)
                if x == prev:                         # distance 0
                    weights.append(w / self.p)
                elif self.graph.has_edge(prev, x):    # distance 1
                    weights.append(w)
                else:                                 # distance 2
                    weights.append(w / self.q)
            weights = np.array(weights)
            weights /= weights.sum()
            walk.append(np.random.choice(nbrs, p=weights))
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for n in nodes:
                walks.append(self._walk(n))
        return walks

    def fit(self):
        from gensim.models import Word2Vec
        walks = [[str(n) for n in w] for w in self.generate_walks()]
        model = Word2Vec(walks, vector_size=self.dimensions,
                         window=self.window_size, sg=1,
                         negative=5, min_count=0, workers=4)
        return {n: model.wv[str(n)] for n in self.graph.nodes()}
```

### 3.4 Building the graph in the first place

Most production deployments do not start with a clean graph; they construct one from interaction logs. A common recipe uses **Jaccard similarity over user sets** to weight edges:

```python
from collections import defaultdict

def co_occurrence_graph(interactions, min_jaccard=0.1, max_users=None):
    """Item-item graph from (user, item) interactions.

    Edge weight = Jaccard similarity over the users who touched both items.
    """
    item_users = defaultdict(set)
    for u, i in interactions:
        item_users[i].add(u)

    G = nx.Graph()
    items = list(item_users)
    for a in range(len(items)):
        ua = item_users[items[a]]
        if max_users and len(ua) > max_users:
            continue
        G.add_node(items[a])
        for b in range(a + 1, len(items)):
            ub = item_users[items[b]]
            inter = len(ua & ub)
            if inter == 0:
                continue
            j = inter / len(ua | ub)
            if j >= min_jaccard:
                G.add_edge(items[a], items[b], weight=j)
    return G
```

> **Practitioner tip.** The pairwise loop above is $O(N^2)$ in items. For a real catalogue, build an inverted index by user, iterate per user, and only generate candidate pairs that share at least one user. That brings it down to $O(\sum_u |I_u|^2)$, which is dramatically smaller in practice.

---

## 4. Two-tower models: separate user and item encoders

### 4.1 Why two towers, not one

Concatenate user features with item features, push them through a single network, get a score. Why not? **Because at serving time you would have to run the network for every (user, candidate) pair.** With ten million candidates that is ten million forward passes per request.

Two-tower architectures factor the model into a **user tower** $f_u(\mathbf{x}_u)$ and an **item tower** $f_i(\mathbf{x}_i)$, then predict with a similarity function — usually cosine — on the two outputs:

$$
\mathbf{e}_u = f_u(\mathbf{x}_u; \theta_u), \qquad
\mathbf{e}_i = f_i(\mathbf{x}_i; \theta_i), \qquad
s(u, i) = \cos(\mathbf{e}_u, \mathbf{e}_i).
$$

> **Analogy.** Think of a dating app. One tower writes everyone's profile; the other tower writes everyone's "what I look for" preferences. At match time you just compare profiles — you do not re-run the matching algorithm from scratch for every potential pair.

The architectural payoff is enormous:

1. **Pre-compute all item vectors offline.** Run the item tower once per item and store the result.
2. **At serving time, run only the user tower.** One forward pass per request.
3. **Use ANN search** to find the top-K nearest item vectors in milliseconds.

This is the canonical recipe for the *retrieval* (a.k.a. *recall*) stage of a modern recommendation pipeline.

![Two-tower DSSM: user features flow up the left tower, item features flow up the right tower, both end in L2-normalised d-dimensional vectors, and a cosine similarity at the top produces the score](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig3_two_tower.png)

### 4.2 DSSM in code

DSSM (Huang et al., Microsoft, 2013) is the canonical two-tower model. Originally designed for web search, it ships in essentially every large recommender today.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_tower(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DSSM(nn.Module):
    """Symmetric two-tower with cosine similarity head."""

    def __init__(self, user_dim, item_dim, embedding_dim=128, hidden=(256, 128)):
        super().__init__()
        self.user_tower = make_tower(user_dim, list(hidden), embedding_dim)
        self.item_tower = make_tower(item_dim, list(hidden), embedding_dim)

    def encode_user(self, x):
        return F.normalize(self.user_tower(x), p=2, dim=-1)

    def encode_item(self, x):
        return F.normalize(self.item_tower(x), p=2, dim=-1)

    def forward(self, user_x, item_x):
        u = self.encode_user(user_x)
        i = self.encode_item(item_x)
        return (u * i).sum(-1)        # cosine because both are L2-normalised


class SampledSoftmaxLoss(nn.Module):
    """Cross-entropy where the positive must beat the in-batch negatives."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, u, pos, neg):
        # u   : (B, d)
        # pos : (B, d)
        # neg : (B, K, d)
        pos_score = (u * pos).sum(-1, keepdim=True) / self.t          # (B, 1)
        neg_score = torch.bmm(u.unsqueeze(1),
                              neg.transpose(1, 2)).squeeze(1) / self.t  # (B, K)
        logits = torch.cat([pos_score, neg_score], dim=1)             # (B, 1+K)
        target = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
        return F.cross_entropy(logits, target)
```

A few non-obvious choices baked into this 30-line implementation:

- **L2-normalise the outputs.** Cosine similarity equals an inner product on unit-norm vectors, so this lets you index with FAISS's inner-product flavour.
- **Temperature.** Dividing logits by a small $\tau \in [0.05, 0.2]$ makes the softmax sharper. Without it, cosine similarities live in $[-1, 1]$ — a very flat distribution that learns slowly.
- **`BatchNorm` between layers.** Keeps activations on a sane scale through the depth of the tower; especially important when input features mix wildly different magnitudes.
- **Cross-entropy with the positive at index 0.** Cleanly extends to in-batch negatives by using *every other item in the batch* as a negative — see Section 5.

---

## 5. YouTube DNN: pooling a user's history

### 5.1 The setup

YouTube DNN (Covington et al., 2016) is a celebrated two-tower variant tailored to video recommendation, where each user is described by a *sequence* of recently watched videos rather than a flat feature vector. The user tower's job is to **pool that sequence** into one vector.

Original YouTube DNN does the simplest possible thing: **mean-pool the embeddings of the watched videos**, then concatenate with demographics, then push through a 2-layer MLP. Average pooling is unbeatable on latency, and at YouTube's scale every saved millisecond adds up to data-centre money.

```python
class YouTubeDNN(nn.Module):
    """Mean-pool the watch history; concatenate side features; project to d."""

    def __init__(self, num_videos, num_categories, d=64,
                 user_hidden=(256, 128), item_hidden=(128, 64)):
        super().__init__()
        self.video = nn.Embedding(num_videos, d)
        self.category = nn.Embedding(num_categories, 16)

        self.user_mlp = make_tower(d + 16, list(user_hidden), d)
        self.item_mlp = make_tower(d + 16, list(item_hidden), d)

    def user_vector(self, history_ids, history_cats):
        h = self.video(history_ids).mean(dim=1)
        c = self.category(history_cats).mean(dim=1)
        return F.normalize(self.user_mlp(torch.cat([h, c], dim=-1)), dim=-1)

    def item_vector(self, item_id, item_cat):
        e = self.video(item_id)
        c = self.category(item_cat)
        return F.normalize(self.item_mlp(torch.cat([e, c], dim=-1)), dim=-1)

    def forward(self, hist_ids, hist_cats, item_id, item_cat):
        u = self.user_vector(hist_ids, hist_cats)
        i = self.item_vector(item_id, item_cat)
        return (u * i).sum(-1)
```

### 5.2 When pooling is too crude: attention

Average pooling treats every watched video equally. In reality, a user who just watched ten cooking videos and one ten-second car ad probably wants more cooking, not more cars. Self-attention lets the model *learn* the weighting:

```python
class YouTubeDNNWithAttention(nn.Module):
    def __init__(self, num_videos, num_categories, d=64, n_heads=4):
        super().__init__()
        self.video = nn.Embedding(num_videos, d)
        self.category = nn.Embedding(num_categories, 16)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.user_mlp = make_tower(d + 16, [256], d)
        self.item_mlp = make_tower(d + 16, [128], d)

    def user_vector(self, history_ids, history_cats):
        h = self.video(history_ids)                        # (B, T, d)
        h, _ = self.attn(h, h, h)                          # self-attention
        h = h.mean(dim=1)                                  # pool the attended seq
        c = self.category(history_cats).mean(dim=1)
        return F.normalize(self.user_mlp(torch.cat([h, c], dim=-1)), dim=-1)

    def item_vector(self, item_id, item_cat):
        e = self.video(item_id)
        c = self.category(item_cat)
        return F.normalize(self.item_mlp(torch.cat([e, c], dim=-1)), dim=-1)
```

> **The classic trade-off.** Mean pooling is $O(T)$ in history length and trivially fast. Self-attention is $O(T^2)$ in compute and $O(T)$ in memory but typically improves recall by a few percent. For real-time retrieval over a long history, target attention against the candidate item ([DIN, Zhou et al., 2018](https://arxiv.org/abs/1706.06978)) is often a better engineering compromise.

---

## 6. Negative sampling strategies

Positives are expensive — they come from real user behaviour. Negatives are cheap — there are millions of them. The art is **picking the right cheap negatives**, because random ones are too easy and trivially-hard ones destabilise training.

### 6.1 The four strategies

| Strategy | Mechanism | When to use |
|---|---|---|
| **Uniform** | Pick negatives uniformly at random from the catalogue. | Simple baseline; works on small catalogues. |
| **Popularity-aware** | Sample with $P_n \propto f^{0.75}$. | Default for large catalogues; counters popularity bias. |
| **Hard-negative mining** | Pick items that *look* relevant to the user but were not interacted with. | Late-stage fine-tuning; sharpens the decision boundary. |
| **In-batch negatives** | Use other positives in the same minibatch as your negatives. | Two-tower training at scale; effectively free. |

### 6.2 Why in-batch negatives are the workhorse

In a batch of $B$ (user, positive item) pairs, every other positive's item embedding can serve as a negative for every user. That gives you $B - 1$ negatives per row at no extra cost. Variant: **add a popularity correction term** to undo the bias of in-batch sampling — popular items appear in batches more often and get penalised too hard. The standard correction (Yi et al., 2019, "Sampling-Bias-Corrected Neural Modeling") subtracts $\log p(i)$ from the logit before softmax.

### 6.3 Implementation snippets

```python
import numpy as np

def uniform_neg(catalog_size: int, k: int, exclude: set | None = None) -> list[int]:
    out = set()
    while len(out) < k:
        i = int(np.random.randint(catalog_size))
        if exclude is None or i not in exclude:
            out.add(i)
    return list(out)


def popularity_neg(probs: np.ndarray, k: int) -> np.ndarray:
    """probs is a length-N array, already normalised by f^0.75."""
    return np.random.choice(len(probs), size=k, replace=False, p=probs)


def in_batch_neg(item_emb: torch.Tensor) -> torch.Tensor:
    """item_emb: (B, d). Returns (B, B-1, d) using the other rows of the batch."""
    B = item_emb.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=item_emb.device)
    idx = (~eye).nonzero(as_tuple=False)[:, 1].view(B, B - 1)
    return item_emb[idx]


def hard_neg(user_emb, item_emb, positives_mask, top_k=100, k=10):
    """Hard negatives: items most similar to the user that are NOT positives."""
    sims = user_emb @ item_emb.T
    sims[positives_mask] = -1e9
    top = sims.topk(top_k, dim=-1).indices
    pick = torch.randint(0, top_k, (user_emb.size(0), k), device=user_emb.device)
    return top.gather(1, pick)
```

> **Curriculum.** Start training with uniform / popularity-aware negatives for stability. After a few epochs, mix in 10 – 30% hard negatives to sharpen the boundary. Pure hard-negative training from scratch tends to collapse — the model has not learned coarse separation yet, so "hard" is just "noise."

---

## 7. Approximate nearest-neighbour search

### 7.1 The serving problem

You have a one-million-item index and a hundred-millisecond budget. A brute-force scan is $O(Nd)$ per query — about $1.3 \times 10^{8}$ multiply-adds for $d=128$. Doable on a beefy CPU, undoable on a fleet of them at QPS in the tens of thousands. **Approximate** nearest-neighbour search trades 1 – 5% recall for a 10 – 100× speedup.

![ANN search: left panel shows IVF index probing the centroid nearest the query and returning the k-NN inside that cluster; right panel compares query latency vs Recall@10 for Flat, IVF, HNSW, Annoy, IVFPQ on a 1M-item, d=128 benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig5_ann_search.png)

### 7.2 The three index families

| Family | Idea | Strength |
|---|---|---|
| **Inverted file (IVF)** | k-means clusters the vectors; at query time only the nearest few centroids are scanned. | Tunable speed / recall; great for medium-to-large indexes. |
| **Hierarchical Navigable Small World (HNSW)** | A multi-layer proximity graph with greedy navigation. | Best raw query speed at high recall; modest memory cost. |
| **Product Quantisation (PQ)** | Compresses vectors into a few bytes by quantising sub-vectors independently. Usually layered onto IVF (IVFPQ). | Massive memory savings; mandatory at billion-scale. |

Annoy (Spotify) uses random projection trees — simpler API, excellent for quick prototypes but typically slower than HNSW at the same recall.

### 7.3 FAISS in production

```python
import faiss
import numpy as np


class ANNIndex:
    """Thin wrapper around FAISS with three common index types."""

    def __init__(self, dim: int, kind: str = "IVF",
                 nlist: int = 100, hnsw_m: int = 32):
        if kind == "Flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "IVF":
            quantiser = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif kind == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m,
                                             faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(kind)
        self._trained = False

    def build(self, vectors: np.ndarray) -> None:
        x = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(x)             # cosine via inner product
        if hasattr(self.index, "train") and not self._trained:
            self.index.train(x)
            self._trained = True
        self.index.add(x)

    def query(self, vector: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(vector.reshape(1, -1), dtype="float32")
        faiss.normalize_L2(q)
        sims, ids = self.index.search(q, k)
        return sims[0], ids[0]

    def set_nprobe(self, n: int) -> None:
        """For IVF: how many clusters to scan per query. Higher = slower & more recall."""
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = n
```

### 7.4 Benchmarks at a glance

| Index | Build time | Query time | Memory | Recall@10 | Best for |
|---|---|---|---|---|---|
| Flat | seconds | $O(N)$ — slow | $4Nd$ B | 100% | Ground truth, small catalogues |
| IVF | minutes | fast | $4Nd$ B | 95 – 99% | General large-scale, tunable |
| HNSW | tens of minutes | very fast | $4Nd + $ graph | 95 – 99% | Tight latency budgets |
| IVFPQ | minutes | fast | $\sim N$ B (8× compression) | 90 – 95% | Billion-scale, RAM-bound |
| Annoy | minutes | fast | medium | 90 – 95% | Static indexes, simple deployment |

> **Operational rules of thumb.** Use IVF with `nprobe = sqrt(nlist)` as a starting point. Set `nlist ≈ sqrt(N)`. For an HNSW graph, `M = 32` and `efSearch = 100` are sane defaults. Always benchmark on **your** vectors — recall numbers from generic benchmarks are surprisingly volatile when the cluster geometry changes.

---

## 8. Evaluating embedding quality

Embeddings are not directly visible to users, so you need proxies. Use both **intrinsic** metrics (purely about the vectors) and **extrinsic** metrics (about the recommendations they produce).

![Embedding similarity heatmap for 12 movies grouped into four categories; the block-diagonal pattern shows that within-category cosine similarity is high while cross-category similarity is near zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/05-embedding-techniques/fig6_similarity_heatmap.png)

### 8.1 Intrinsic — does the geometry look right?

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def neighbour_overlap(emb: dict, gold: dict, k: int = 10) -> float:
    """Average Jaccard between the embedding's top-k and a ground-truth top-k."""
    scores = []
    items = list(emb)
    matrix = np.stack([emb[i] for i in items])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9

    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)

    for idx, item in enumerate(items):
        if item not in gold:
            continue
        top_emb = set(items[j] for j in np.argsort(-sims[idx])[:k])
        top_gold = set(sorted(gold[item], key=gold[item].get, reverse=True)[:k])
        scores.append(len(top_emb & top_gold) / k)
    return float(np.mean(scores)) if scores else 0.0


def cluster_silhouette(emb: dict, n_clusters: int = 10) -> float:
    matrix = np.stack(list(emb.values()))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(matrix)
    return silhouette_score(matrix, labels)
```

### 8.2 Extrinsic — does the system recommend better?

```python
def hit_rate_ndcg(user_vec: dict, item_vec: dict, test, k: int = 10):
    items = list(item_vec)
    matrix = np.stack([item_vec[i] for i in items])
    item_index = {i: idx for idx, i in enumerate(items)}

    hits = 0
    ndcg = []
    for user_id, true_item in test:
        if user_id not in user_vec or true_item not in item_index:
            continue
        sims = matrix @ user_vec[user_id]
        top = np.argsort(-sims)[:k]
        if item_index[true_item] in top:
            hits += 1
            rank = int(np.where(top == item_index[true_item])[0][0]) + 1
            ndcg.append(1.0 / np.log2(rank + 1))
        else:
            ndcg.append(0.0)
    return {"hit_rate@k": hits / len(test), "ndcg@k": float(np.mean(ndcg))}
```

> **A real evaluation suite tracks more than accuracy.** Coverage (what fraction of the catalogue ever gets recommended), diversity (intra-list dissimilarity), serendipity (recommendations far from the user's history that still convert), and freshness all matter. A model that wins HR@10 but only ever recommends the top 1% of items is a popularity baseline in disguise.

---

## 9. Frequently asked questions

### Q1. Item2Vec vs. matrix factorisation — when do I pick which?

Matrix factorisation (MF) learns from the whole user-item matrix and captures *global* preference patterns. Item2Vec learns from sequences and captures *temporal co-occurrence*. If your data is naturally sequential (playlists, sessions, watch logs) Item2Vec is usually stronger; if you have explicit ratings or stable long-term preferences, MF is competitive and far simpler to train.

### Q2. How do I pick the embedding dimension?

| Catalogue size | Typical $d$ |
|---|---|
| < 10K items | 32 – 64 |
| 10K – 1M items | 64 – 128 |
| > 1M items | 128 – 512 |

Doubling $d$ doubles index size and query latency without doubling quality — diminishing returns set in fast. Validate on HR@K and NDCG, not on training loss.

### Q3. Skip-gram or CBOW for items?

Skip-gram for almost everything in recommendations. It allocates more gradient to long-tail items, which is exactly the regime where recommenders need help.

### Q4. When is Node2Vec strictly better than Item2Vec?

When your relationships are not naturally sequential. Co-purchase graphs, social networks, knowledge graphs, item-attribute graphs — anything where "neighbour" has a meaning beyond "appeared next to in time."

### Q5. Why not just train one big model end-to-end?

A single-tower model that takes both user and item features cannot factor into independent encoders, so retrieval becomes $O(N)$ per query. Two-tower trades a bit of expressiveness for the ability to pre-compute and index — at scale, that trade is non-negotiable.

### Q6. How do I handle cold-start?

- **New users:** rely on demographic / context features in the user tower (no ID needed).
- **New items:** use a content-based item tower (text, image, category embeddings).
- **Hybrid:** sum a learned ID embedding with a content embedding; the ID part learns once data accumulates, the content part carries the new item.

### Q7. What is the right negative sampling recipe?

Start with uniform or popularity-aware sampling for stability. After a few epochs, blend in 10 – 30% hard negatives. Use in-batch negatives whenever you can — they are essentially free.

### Q8. FAISS vs. HNSW vs. Annoy?

- **FAISS** is the production standard for very large indexes (> 10M items), supports GPUs, and has the richest set of index types.
- **HNSW** (via `hnswlib` or FAISS's `IndexHNSWFlat`) gives the best raw query speed at high recall and is the right default for tight latency budgets.
- **Annoy** is great for prototyping or static indexes you can rebuild offline.

### Q9. How do I sanity-check embeddings without labels?

Inspect nearest neighbours visually for a handful of items you know well. Plot a t-SNE / UMAP and check whether known categories cluster. Compute a silhouette score against any auxiliary categorical label.

### Q10. Dot product or cosine similarity?

Cosine similarity is direction-only; magnitudes cancel. Dot product also rewards larger magnitudes — which can be useful if you *want* popular items to score higher (their embeddings tend to grow longer during training). For most retrieval setups, **L2-normalise everything and use dot product = cosine.** That also lets you index with FAISS's inner-product mode.

---

## 10. Summary

- **Embeddings are dense, learned vectors that turn discrete identity into geometry.** Recommendation becomes nearest-neighbour search.
- **Choose the method by the shape of your data.** Sequences → Item2Vec. Graphs → Node2Vec. Rich features and large catalogues → two-tower (DSSM, YouTube DNN).
- **Negative sampling is half of the model.** Mix uniform / popularity-aware / hard / in-batch and watch curriculum effects.
- **Two-tower architectures are the production default for retrieval** because they make item vectors pre-computable and indexable.
- **ANN search makes serving feasible.** Pick FAISS-IVF, HNSW or Annoy by your latency / recall / memory budget; benchmark on *your* data.
- **Evaluate with intrinsic and extrinsic metrics.** Track coverage and diversity alongside HR@K and NDCG, or you will silently drift toward popularity.

---

## Series navigation

- **Previous:** [Part 4 — CTR Prediction and Click-Through Rate Modeling](/en/recommendation-systems-4-ctr-prediction/)
- **Next:** [Part 6 — Sequential Recommendation and Session-based Modeling](/en/recommendation-systems-6-sequential-recommendation/)
- [Back to Series Overview (Part 1)](/en/recommendation-systems-1-fundamentals/)
