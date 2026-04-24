---
title: "Recommendation Systems (6): Sequential Recommendation and Session-based Modeling"
date: 2025-11-04 09:00:00
tags:
  - Recommendation Systems
  - Sequential Recommendation
  - Session Modeling
categories: Recommendation Systems
lang: en
mathjax: true
series: recommendation-systems
series_title: "Part 6 of 16: Sequential Recommendation and Session-based Modeling"
permalink: "en/recommendation-systems-6-sequential-recommendation/"
description: "How recommenders use the order of user actions to predict the next one. Markov chains, GRU4Rec, Caser, SASRec, BERT4Rec, BST, and SR-GNN, with implementations and intuition."
disableNunjucks: true
series_order: 6
---

When you scroll TikTok, every recommendation feels eerily on-point — not because the system reads your mind, but because it reads the **order** of what you just watched. A cooking video followed by a travel vlog tells a different story than the same two clips in reverse. That ordering is exactly the signal that sequential recommenders are built to exploit.

Compare two friends recommending shows. The first knows your favourite genres but never asks what you watched last week. The second says, *"You just finished three sci-fi thrillers in a row — try this one."* Traditional collaborative filtering is friend one. Sequential recommendation is friend two.

![A user's interaction sequence, where each step depends on the ones before it, ending in a next-item prediction](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig1_sequence_timeline.png)

## What you will learn

- **Why order matters**, and how sequential models depart from set-based collaborative filtering
- **Markov chains** as the simplest sequential baseline — interpretable, sparse, and surprisingly resilient
- **GRU4Rec**, the first deep-learning model to take session-based recommendation seriously
- **Caser**, which treats the sequence as an "image" and runs CNN filters across it
- **SASRec** and **BERT4Rec**, the Transformer-era unidirectional and bidirectional models
- **BST**, the Behavior Sequence Transformer that pulls in side features
- **SR-GNN**, which represents a session as a directed graph
- **Evaluation metrics** (HR@K, NDCG@K, MRR) and **production tradeoffs**

## Prerequisites

- Comfort with neural networks (RNNs, CNNs, Transformers)
- Basic PyTorch
- Recommendation fundamentals from [Part 1](/en/recommendation-systems-1-fundamentals/)
- Embedding techniques from [Part 5](/en/recommendation-systems-5-embedding-techniques/) help

---

## What is sequential recommendation?

### Definition

A **sequential recommender** models user preferences using the **temporal order** of interactions. Where traditional collaborative filtering treats a user's history as a bag of items, a sequential model treats it as a stream — and that stream carries information.

Formally, given a user $u$ with interaction sequence $S_u = [i_1, i_2, \dots, i_t]$, we want to estimate

$$
P(i_{t+1} \mid S_u) = P(i_{t+1} \mid i_1, i_2, \dots, i_t).
$$

In plain English: *given everything the user has done so far, in order, what comes next?* The probability depends not just on which items appeared but on **how they were arranged in time**.

### Why this matters

There are four reasons sequential modelling pays off in production:

- **Drifting taste.** A user who binged action films last month might be on a documentary kick this week. Static models miss the shift; sequential ones absorb it.
- **Local context.** After watching a movie trailer, the next thing a user wants is the movie itself, not a random recommendation. Sequential context captures that.
- **Session intent.** On a shopping site, a user browsing laptops will probably look at laptop bags next — a pattern that lives in the transition, not in the marginal popularity of bags.
- **Cold-start dampening.** Even users with three interactions carry signal in their *order*; set-based models throw that signal away.

### Sequential vs. classical recommendation

| Aspect | Classical CF | Sequential |
|---|---|---|
| Input | Unordered set of interactions | Ordered sequence |
| Temporal modelling | Ignored | First-class |
| Prediction target | $P(i \mid u)$ | $P(i_{t+1} \mid S_u)$ |
| Strength | Long-term taste | Next-action prediction |
| Typical pitch | "Users like you also liked..." | "Based on your last few actions..." |

### Three flavours

- **User-based sequential.** Treats the user's entire history as one long sequence. Good for capturing slow taste evolution (months of music listening).
- **Session-based.** Treats each short session as an independent sequence. Ideal when users are anonymous or intent is short-lived (a single shopping session).
- **Hybrid.** Uses long-term embeddings for "who you are" and the current session for "what you want right now." Production systems usually land here.

---

## Markov chain models

### First-order chains

The simplest sequential model is a **first-order Markov chain**, which assumes the next item depends only on the current one:

$$
P(i_{t+1} \mid i_1, \dots, i_t) = P(i_{t+1} \mid i_t).
$$

> **Analogy.** This is like predicting the next word in a sentence using only the current word. If the current word is "ice," you might guess "cream" — but you have no idea whether the conversation is about dessert or hockey.

We learn a transition matrix $M \in \mathbb{R}^{|I| \times |I|}$, where $M_{ij} = P(j \mid i)$ is estimated from counts:

$$
M_{ij} = \frac{\text{count}(i \to j)}{\text{count}(i)}.
$$

In words: how often does $j$ follow $i$, divided by how often $i$ appears (excluding final positions).

### Higher-order chains

A $k$-th order chain conditions on the last $k$ items:

$$
P(i_{t+1} \mid i_1, \dots, i_t) = P(i_{t+1} \mid i_{t-k+1}, \dots, i_t).
$$

Higher orders capture more context but burn through your data: with 10,000 items a second-order model has $10^8$ possible transitions, most of which are never observed. This is the **curse of dimensionality** in disguise.

### Implementation

```python
import numpy as np
from collections import defaultdict
from typing import List

class MarkovChainRecommender:
    """First-order Markov chain. Simple, interpretable, a strong baseline."""

    def __init__(self, order: int = 1):
        self.order = order
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.items = set()

    def fit(self, sequences: List[List[int]]):
        for seq in sequences:
            self.items.update(seq)
            for i in range(len(seq) - self.order):
                context = tuple(seq[i:i + self.order])
                next_item = seq[i + self.order]
                self.transition_counts[context][next_item] += 1
                self.context_counts[context] += 1

    def predict_next(self, sequence: List[int], top_k: int = 10):
        if len(sequence) < self.order:
            return []
        context = tuple(sequence[-self.order:])
        if context not in self.transition_counts:
            return []
        total = self.context_counts[context]
        probs = [(item, count / total)
                 for item, count in self.transition_counts[context].items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]


# --- Example ---
sequences = [
    [1, 2, 3, 4],   # laptop -> mouse -> keyboard -> monitor
    [1, 2, 5],      # laptop -> mouse -> mousepad
    [2, 3, 4],      # mouse  -> keyboard -> monitor
    [1, 6, 7],      # laptop -> charger -> bag
]
model = MarkovChainRecommender(order=1)
model.fit(sequences)
for item, prob in model.predict_next([1, 2], top_k=3):
    print(f"  Item {item}: {prob:.3f}")
# After [laptop, mouse], "keyboard" wins because mouse->keyboard fired twice.
```

### Where Markov chains break

**Sparsity.** For large catalogues most transitions never appear. Smoothing helps but does not fix the underlying problem.

**Tunnel vision.** Even a high-order chain only sees a fixed window. It cannot represent a statement like *"the user has been into photography for the past month."*

**No generalization.** If items A and B are functionally similar, the chain still treats them as completely independent — there is no shared representation.

These limitations are exactly the gap that neural sequence models fill.

---

## GRU4Rec: RNN-based sequential recommendation

### Architecture overview

GRU4Rec (Hidasi et al., 2015) was the first deep model to take session-based recommendation seriously. It uses **Gated Recurrent Units** to consume items one at a time and maintain a running summary of the session.

> **Key idea.** Instead of treating sessions as bags of items, GRU4Rec walks through them sequentially, updating a hidden state that compresses everything seen so far.

![GRU4Rec architecture: items embedded, then passed through a GRU whose hidden state feeds a softmax over items](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig2_gru4rec_architecture.png)

### How a GRU updates its state

A GRU cell maintains hidden state $h_t$. Given the input $x_t$ (embedding of item $i_t$) and the previous hidden state $h_{t-1}$:

**Reset gate** (what to forget):

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

**Update gate** (how much to refresh):

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

**Candidate activation** (proposed new memory):

$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$

**Hidden state** (blend old and new):

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

> **Plain English.** The GRU reads items one at a time and keeps a "memory" vector. The reset gate decides how much of the old memory to wipe; the update gate decides how much new information to mix in. The final hidden state is a learned summary of the whole session.

The full pipeline is: **Embedding → GRU → Linear projection → Softmax** over the item vocabulary, trained with a ranking loss (BPR or TOP1) for implicit feedback.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class GRU4Rec(nn.Module):
    """Session-based recommendation with a GRU."""

    def __init__(self, num_items: int, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.25):
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output = nn.Linear(hidden_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequences, hidden=None):
        x = self.dropout(self.item_embedding(sequences))
        gru_out, hidden = self.gru(x, hidden)
        return self.output(self.dropout(gru_out)), hidden

    def predict_next(self, sequence, top_k=10):
        self.eval()
        with torch.no_grad():
            seq = torch.LongTensor([sequence]).to(next(self.parameters()).device)
            logits, _ = self(seq)
            scores, indices = torch.topk(logits[0, -1, :], k=min(top_k, self.num_items))
            return list(zip(indices.tolist(), scores.tolist()))


class SessionDataset(Dataset):
    """Pad/truncate to max_len; input = all but last, target = all but first."""

    def __init__(self, sessions, max_len=20):
        self.sessions = sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        s = self.sessions[idx][-self.max_len:]
        s = [0] * (self.max_len - len(s)) + s
        return torch.LongTensor(s[:-1]), torch.LongTensor(s[1:])


def train_gru4rec(model, loader, num_epochs=10, lr=1e-3, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{num_epochs}  loss={total / len(loader):.4f}")
```

### Strengths and limitations

| Strengths | Limitations |
|---|---|
| Captures sequential dependencies naturally | Sequential by construction — no parallelism over time steps |
| Handles variable-length sequences gracefully | Long-range dependencies fade with distance |
| Compact, well-understood architecture | Fixed hidden state caps memory capacity |

---

## Caser: convolutional sequence embedding

### Motivation

Caser (Tang and Wang, 2018) flips the problem on its head. Instead of walking through the sequence step by step, it **lays the embedded sequence out as an image** and runs convolutions over it. Different filter sizes catch patterns of different lengths in parallel.

> **Analogy.** If GRU4Rec reads a sentence word by word, Caser looks at it all at once with a set of differently-sized magnifying glasses — one for pairs, one for triplets, one for longer sub-phrases.

### Architecture

Caser stacks two filter families on the $t \times d$ embedding matrix $\mathbf{E}$ (sequence length $t$, embedding dim $d$):

- **Horizontal filters** slide along the sequence to capture **union-level n-gram patterns**. Heights of 2, 3, 4 give bigram, trigram, and 4-gram detectors.
- **Vertical filters** sweep the embedding dimension to capture **point-level patterns** — latent features of individual items aggregated across time.

$$
\mathbf{c}_h = \text{ReLU}(\text{Conv}_h(\mathbf{E})) \quad\text{(horizontal, height } h\text{)}
$$

$$
\mathbf{c}_v = \text{ReLU}(\text{Conv}_v(\mathbf{E})) \quad\text{(vertical, full height)}
$$

The two outputs are pooled, concatenated, and pushed through a fully-connected head.

### Implementation

```python
class Caser(nn.Module):
    """CNN-based sequential recommendation."""

    def __init__(self, num_items: int, embedding_dim: int = 50,
                 max_len: int = 50, num_horizon: int = 16,
                 num_vertical: int = 8,
                 horizon_sizes: list = [2, 3, 4]):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.horizon_convs = nn.ModuleList([
            nn.Conv2d(1, num_horizon, (h, embedding_dim)) for h in horizon_sizes
        ])
        self.vertical_conv = nn.Conv2d(1, num_vertical, (max_len, 1))
        total = num_horizon * len(horizon_sizes) + num_vertical
        self.fc1 = nn.Linear(total, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, sequences):
        emb = self.item_embedding(sequences).unsqueeze(1)        # (B, 1, L, D)
        h_outs = []
        for conv in self.horizon_convs:
            o = F.relu(conv(emb))
            h_outs.append(F.max_pool2d(o, (o.size(2), 1)).squeeze(-1).squeeze(-1))
        h = torch.cat(h_outs, dim=1)
        v = F.relu(self.vertical_conv(emb)).squeeze(2).squeeze(2)
        x = self.dropout(F.relu(self.fc1(torch.cat([h, v], dim=1))))
        return self.fc2(x)
```

### Why Caser matters

- **Parallel by design.** CNNs process the whole sequence in one forward pass — much faster than RNNs on a GPU.
- **Multi-scale patterns.** Different filter heights pick up bigrams, trigrams, and longer phrases simultaneously.
- **Local pattern detector.** Excels at short-range sequential patterns where RNNs are overkill and Transformers are wasteful.

---

## SASRec: self-attention for sequential recommendation

### Why Transformers

SASRec (Kang and McAuley, 2018) brings the **Transformer encoder** to sequential recommendation. Self-attention solves two problems that bedevil RNNs at once: it lets every position connect *directly* to every earlier one (no vanishing gradients), and it processes all positions in parallel.

> **Why it works.** RNNs read one item at a time, which makes them slow and makes it hard to link items that are far apart in time. Self-attention lets each item *look* at every earlier item in a single matrix multiplication, capturing both nearby and distant relationships at once.

![SASRec self-attention weights with causal mask. Hatched cells are blocked because the model cannot peek at future items.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig3_attention_heatmap.png)

The heatmap above shows a typical attention pattern. Each row is one position acting as the **query**; the columns are the items it attends to. The diagonal is dark (every position pays attention to itself), recent items get more weight than older ones, and the upper triangle is hatched — those cells are masked out so the model cannot cheat by looking ahead.

### Building blocks

**1. Self-attention.** Each position attends to all previous positions:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
$$

The $\sqrt{d_k}$ scaling keeps the dot products from saturating the softmax as the dimension grows.

**2. Causal mask.** Position $t$ may only attend to positions $1, \dots, t$. Without this mask the model would trivially solve the prediction task by reading the answer.

**3. Positional encoding.** Self-attention is permutation-equivariant — by itself it does not know the order of inputs. We inject order through positional encodings, classically sinusoidal:

$$
PE_{(p, 2i)} = \sin(p / 10000^{2i/d}), \qquad
PE_{(p, 2i+1)} = \cos(p / 10000^{2i/d}).
$$

The next figure shows what this addition looks like in practice.

![Item embeddings (left) plus positional encodings (centre) yield the input the Transformer actually consumes (right). The position term is what turns a bag of items into a sequence.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig5_position_plus_item.png)

**4. Residuals and LayerNorm.** Standard Transformer plumbing — they make deeper stacks trainable.

### Implementation

```python
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items, d_model=128, num_heads=2,
                 num_layers=2, d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, num_items + 1)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, sequences):
        seq_len = sequences.size(1)
        x = self.item_embedding(sequences) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        causal = torch.triu(torch.ones(seq_len, seq_len, device=sequences.device),
                            diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=causal)
        return self.output(x)
```

### Why SASRec became the default

- **Long-range dependencies** without vanishing gradients — any two positions are one attention hop apart.
- **Parallel training** over all positions at once.
- **Interpretability for free** — attention weights tell you which past actions drove each prediction.
- **Strong scaling** with sequence length and model size.

---

## BERT4Rec: bidirectional encoder for sequential recommendation

### Motivation

BERT4Rec (Sun et al., 2019) takes the same Transformer backbone but flips the training objective. Instead of next-item prediction with a causal mask, it borrows BERT's **cloze task**: randomly mask items in the sequence and ask the model to fill them in using **both left and right** context.

> **Wait — how can we look at future items?** During training we hide a few positions and let the bidirectional encoder reconstruct them. At inference we append a `[MASK]` token to the end of the sequence and predict what should go there. Same backbone, different pretext task, richer representations.

![BERT4Rec reconstructs masked positions using a bidirectional encoder. Two items are hidden; the encoder uses every other position as context to recover them.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig4_bert4rec_masking.png)

### Differences from SASRec at a glance

| Feature | SASRec | BERT4Rec |
|---|---|---|
| Attention direction | Causal (left → right) | Bidirectional |
| Training task | Predict the next item | Predict masked items |
| Masking | Causal mask during training | Random masking during training |
| Inference | Use the last position's output | Append `[MASK]`, read its output |

### Implementation

```python
class BERT4Rec(nn.Module):
    """Bidirectional Transformer with cloze-style training."""

    def __init__(self, num_items, d_model=128, num_heads=2,
                 num_layers=2, d_ff=256, max_len=50,
                 dropout=0.1, mask_prob=0.15):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1                                  # [MASK] id
        self.item_embedding = nn.Embedding(num_items + 2, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, num_items + 1)

    def _mask_sequence(self, sequences):
        import random
        masked = sequences.clone()
        positions = torch.zeros_like(sequences, dtype=torch.bool)
        for i in range(sequences.size(0)):
            for j in range(sequences.size(1)):
                if sequences[i, j] != 0 and random.random() < self.mask_prob:
                    positions[i, j] = True
                    r = random.random()
                    if r < 0.8:
                        masked[i, j] = self.mask_token                  # 80%: [MASK]
                    elif r < 0.9:
                        masked[i, j] = random.randint(1, self.num_items)# 10%: random
                    # 10%: keep original
        return masked, positions

    def forward(self, sequences, training=True):
        if training:
            sequences, positions = self._mask_sequence(sequences)
        else:
            positions = torch.zeros_like(sequences, dtype=torch.bool)
        x = self.item_embedding(sequences) * math.sqrt(self.item_embedding.embedding_dim)
        x = self.dropout(self.pos_encoding(x))
        for block in self.blocks:                                       # NO causal mask
            x = block(x)
        return self.output(x), positions

    def predict_next(self, sequence, top_k=10):
        self.eval()
        with torch.no_grad():
            if len(sequence) >= self.max_len:
                sequence = sequence[-(self.max_len - 1):]
            sequence = sequence + [self.mask_token]
            sequence = [0] * (self.max_len - len(sequence)) + sequence
            seq = torch.LongTensor([sequence])
            logits, _ = self(seq, training=False)
            mask_pos = sequence.index(self.mask_token)
            scores, indices = torch.topk(logits[0, mask_pos, :],
                                         k=min(top_k, self.num_items))
            return list(zip(indices.tolist(), scores.tolist()))
```

### Tradeoffs

**Wins.** Bidirectional context produces richer representations, and masked training makes the model robust to missing or noisy items.

**Costs.** Inference needs the `[MASK]`-append trick, which is less natural than autoregressive prediction. Training is a touch more complex, and large catalogs often see diminishing returns over a well-tuned SASRec.

---

## BST: Behavior Sequence Transformer

### What makes BST different

BST (Chen et al., 2019, Alibaba) extends the Transformer to incorporate **rich side features** beyond item IDs. In a real e-commerce system you have item categories, brands, prices, shop IDs, user demographics — BST embeds all of them and feeds the concatenation through a Transformer.

> **Insight.** Real user behaviour is not a sequence of IDs, it is a sequence of *events*. Each event bundles an item with its category, brand, price bucket, time gap, and so on. BST treats the bundle as the unit of input.

### Implementation

```python
class FeatureEmbedding(nn.Module):
    """Embed multiple categorical fields and concatenate."""

    def __init__(self, feature_dims, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim + 1, embedding_dim, padding_idx=0)
            for name, dim in feature_dims.items()
        })

    def forward(self, features):
        return torch.cat([self.embeddings[n](t) for n, t in features.items()], dim=-1)


class BST(nn.Module):
    """Behavior Sequence Transformer for feature-rich e-commerce."""

    def __init__(self, item_vocab_size, feature_dims,
                 embedding_dim=64, num_heads=2, num_layers=2,
                 d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.item_embedding = nn.Embedding(item_vocab_size + 1, embedding_dim, padding_idx=0)
        self.feature_embedding = FeatureEmbedding(feature_dims, embedding_dim)
        total = embedding_dim * (1 + len(feature_dims))
        self.pos_encoding = PositionalEncoding(total, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(total, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(total, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

    def forward(self, item_ids, features):
        x = torch.cat([self.item_embedding(item_ids),
                       self.feature_embedding(features)], dim=-1)
        x = self.dropout(self.pos_encoding(x))
        seq_len = item_ids.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=item_ids.device),
                            diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=causal)
        return self.mlp(x).squeeze(-1)
```

---

## Session-based recommendation

A **session** is a short burst of interactions — one shopping visit, one playlist, one news-reading session. Session-based recommendation is sequential recommendation with extra constraints:

- **No persistent identity.** The user may be anonymous.
- **Short sequences.** Typically 5 to 20 items.
- **Independent.** Each session is treated on its own; cross-session history is unavailable or ignored.
- **Real-time.** Predictions must keep up with the user's clicks.

| Domain | Example session |
|---|---|
| E-commerce | Laptops → laptop bags → laptop stands |
| News | Politics → sports → weather |
| Music | Jazz → late-night jazz → classical |
| Video | Three cooking tutorials in a row |

---

## SR-GNN: graph neural networks for session recommendation

### Motivation

SR-GNN (Wu et al., 2019) takes a step sideways: instead of treating a session as a flat sequence, it models it as a **directed graph** where nodes are items and edges are transitions.

> **Why a graph?** Consider a session $[A, B, C, B, D]$ where item $B$ appears twice, creating a loop. A graph captures this naturally; a flat sequence model has to fight to represent it.

### Graph construction

For session $S = [i_1, i_2, \dots, i_t]$:

- **Nodes** are unique items in the session.
- **Edges** are directed from each item to the next item in the original order.
- **Edge weights** record the frequency of each transition (handling repeated visits).

### How SR-GNN works

SR-GNN runs a **Gated Graph Neural Network** to propagate information between neighbouring items. The update equations look like a GRU but operate on graph neighbours:

**Message passing.** Each node aggregates from its neighbours:

$$
\mathbf{m}_v^{(l)} = \sum_{u \in \mathcal{N}(v)} \mathbf{A}_{uv}\,\mathbf{h}_u^{(l-1)}
$$

**Gated update.** The node updates its representation with GRU-style gates:

$$
\mathbf{z}_v = \sigma(\mathbf{W}_z \mathbf{m}_v + \mathbf{U}_z \mathbf{h}_v),\quad
\mathbf{r}_v = \sigma(\mathbf{W}_r \mathbf{m}_v + \mathbf{U}_r \mathbf{h}_v)
$$

$$
\tilde{\mathbf{h}}_v = \tanh(\mathbf{W}_h \mathbf{m}_v + \mathbf{U}_h (\mathbf{r}_v \odot \mathbf{h}_v))
$$

$$
\mathbf{h}_v = (1 - \mathbf{z}_v) \odot \mathbf{h}_v + \mathbf{z}_v \odot \tilde{\mathbf{h}}_v
$$

> **Plain English.** Each item "talks" to the items that appeared right before or after it in the session. After a few rounds of message passing, every item's representation has absorbed information about its local neighbourhood in the session graph. The session is then summarised by attention-pooling the node embeddings.

### Implementation (simplified)

```python
class SRGNN(nn.Module):
    """Session-as-graph recommendation. Simplified GGNN with attention pooling."""

    def __init__(self, num_items, embedding_dim=100, num_gnn_layers=1, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.num_gnn_layers = num_gnn_layers
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.W_z = nn.Linear(embedding_dim, embedding_dim)
        self.U_z = nn.Linear(embedding_dim, embedding_dim)
        self.W_r = nn.Linear(embedding_dim, embedding_dim)
        self.U_r = nn.Linear(embedding_dim, embedding_dim)
        self.W_h = nn.Linear(embedding_dim, embedding_dim)
        self.U_h = nn.Linear(embedding_dim, embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.output = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)

    def _gnn_step(self, node_embs, adj):
        m = torch.matmul(adj, node_embs)
        z = torch.sigmoid(self.W_z(m) + self.U_z(node_embs))
        r = torch.sigmoid(self.W_r(m) + self.U_r(node_embs))
        h = torch.tanh(self.W_h(m) + self.U_h(r * node_embs))
        return (1 - z) * node_embs + z * h

    def forward_session(self, session):
        unique = list(dict.fromkeys(session))                           # preserve order
        idx = {item: i for i, item in enumerate(unique)}
        n = len(unique)
        adj = torch.zeros(n, n)
        for a, b in zip(session, session[1:]):
            adj[idx[b], idx[a]] += 1
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1)

        h = self.item_embedding(torch.LongTensor(unique))
        for _ in range(self.num_gnn_layers):
            h = self.dropout(self._gnn_step(h, adj))

        seq_emb = h[[idx[i] for i in session]]
        attn = F.softmax(self.attention(seq_emb), dim=0)
        session_emb = (attn * seq_emb).sum(dim=0)
        return self.output(session_emb)
```

### Why SR-GNN stands out

- **Graph structure** captures complex transition patterns flat sequences miss.
- **Repeated items** are first-class — they collapse into the same node and accumulate edge weights.
- **Local message passing** picks up tight neighbourhood signals like "this item is the centre of a small interest cluster within the session."

---

## How long should the sequence be?

Different model families plateau at very different sequence lengths, and they pay very different prices for it. The figure below shows the typical pattern on a benchmark like MovieLens or Amazon Beauty.

![Left: HR@10 vs maximum sequence length for four model families. Transformers keep gaining as the window grows. Right: relative training cost — RNNs are sequential and scale linearly, Transformers are GPU-parallel and stay nearly flat.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/06-sequential-recommendation/fig6_performance_vs_length.png)

Three things stand out:

- **Markov chains plateau early.** Their fixed-window assumption simply cannot exploit longer context.
- **GRU4Rec peaks around 50–75 items**, then degrades as the hidden state struggles to compress more history into a fixed-size vector.
- **SASRec and BERT4Rec keep climbing**. With direct attention between any two positions there is no compression bottleneck — they convert longer context into higher quality.

The cost picture is the mirror image. RNNs cannot parallelize across time, so cost grows linearly with sequence length. Transformers exploit GPU parallelism and pay almost nothing extra at moderate lengths. Both effects together explain why Transformers became the default in production sequential recommenders.

---

## Evaluation metrics

### Three you need to know

**Hit Rate (HR@K).** What fraction of test cases have the correct next item in the top-$K$ predictions?

$$
\text{HR@K} = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}\!\left[\text{rank}(i_t^*) \leq K\right]
$$

**NDCG@K.** Like HR@K, but rewards higher ranks logarithmically:

$$
\text{NDCG@K} = \frac{1}{|T|} \sum_{t \in T} \frac{\mathbb{1}\!\left[\text{rank}(i_t^*) \leq K\right]}{\log_2(\text{rank}(i_t^*) + 1)}
$$

**MRR.** The mean reciprocal rank of the correct item:

$$
\text{MRR} = \frac{1}{|T|} \sum_{t \in T} \frac{1}{\text{rank}(i_t^*)}
$$

> **Which to use?** HR@K is the most interpretable ("did we get it right at all?"). NDCG@K is the right metric for ranking quality ("how high did we put it?"). MRR is great when there is exactly one relevant item per query.

### Implementation

```python
import numpy as np
from typing import List

def hit_rate_at_k(preds: List[List[int]], truth: List[int], k: int = 10) -> float:
    return sum(t in p[:k] for p, t in zip(preds, truth)) / len(truth)

def ndcg_at_k(preds: List[List[int]], truth: List[int], k: int = 10) -> float:
    scores = []
    for p, t in zip(preds, truth):
        if t in p[:k]:
            scores.append(1.0 / np.log2(p[:k].index(t) + 2))
        else:
            scores.append(0.0)
    return float(np.mean(scores))

def mrr(preds: List[List[int]], truth: List[int]) -> float:
    scores = [1.0 / (p.index(t) + 1) if t in p else 0.0 for p, t in zip(preds, truth)]
    return float(np.mean(scores))
```

---

## Practical considerations

### Data preprocessing

**Padding and truncation.** Most pipelines pad short sequences with zeros (prepended) and truncate long ones to keep the most recent items. Dynamic batching by length minimizes wasted compute. For very long histories, a sliding window of overlapping fixed-length subsequences works well.

**Negative sampling.** Implicit-feedback data only has positive interactions, so you sample negatives. Random sampling is the baseline; popularity-weighted and hard-negative mining produce better gradients but cost more (see [Part 5](/en/recommendation-systems-5-embedding-techniques/)).

**Augmentation.** Three cheap tricks:

- **Sequence cropping** — turn each session into many overlapping prefixes.
- **Item masking** — BERT4Rec-style random masking, even for non-BERT4Rec models.
- **Light shuffling** — perturb non-adjacent items to encourage robustness.

### Training

- **Loss.** Cross-entropy, BPR, and sampled softmax are the standard menu. Sampled softmax is essential for vocabularies in the millions.
- **Regularization.** Dropout (0.1–0.5), L2 weight decay, and early stopping on validation HR@K cover most cases.
- **Optimization.** AdamW with a warmup-then-decay schedule is the default for Transformer-based models; gradient clipping at norm 1.0 is good insurance.

### Scaling to millions of items

Six levers, used in roughly this order:

1. **Negative sampling.** Don't compute over the full vocabulary at training time.
2. **Approximate nearest neighbour** (FAISS, HNSW) for the final retrieval step.
3. **Two-stage retrieval.** Embedding-based candidate generation, then a heavier ranker on the top few hundred.
4. **Distillation.** Train a smaller, faster student that mimics a heavyweight teacher.
5. **Quantization.** FP16 or INT8 inference for tight latency budgets.
6. **Caching.** Item embeddings, popular sequences, and even predictions for stable users.

---

## Model comparison and selection

| Model | Architecture | Parallel | Long-range | Side features | Best for |
|---|---|---|---|---|---|
| Markov chain | Statistical | n/a | No | No | Baselines, very cold start |
| GRU4Rec | RNN | No | Medium | No | Streaming, simple sessions |
| Caser | CNN | Yes | Short | No | Short sessions, local patterns |
| SASRec | Transformer | Yes | Yes | No | General-purpose default |
| BERT4Rec | Transformer | Yes | Yes | No | When bidirectional context truly helps |
| BST | Transformer | Yes | Yes | Yes | Feature-rich e-commerce |
| SR-GNN | GNN | Partly | Medium | No | Sessions with repeated items |

**Picking a starting point:**

- Default to **SASRec**. It is the strongest single bet across most datasets.
- Use **GRU4Rec** when you need online updates from a streaming session.
- Use **Caser** when sessions are short and locality dominates.
- Use **BST** when item-side features are rich and meaningful.
- Use **SR-GNN** when sessions contain repeated items and complex transitions.
- Use **BERT4Rec** when you can afford pre-training on a large corpus.

---

## Questions and answers

### Q1. When should I pick session-based over user-based?

**Session-based** wins for anonymous users, short and self-contained intent (e-commerce browsing, news), and real-time low-latency settings. **User-based** wins when you have stable user IDs, long histories, and care about preference evolution (music streaming, video accounts). Most production systems do both — use a long-term embedding for "who you are" and the session for "what you want right now."

### Q2. How do I pick `max_len`?

Look at the empirical distribution of session lengths and target the 90th percentile. Common choices: **20–50 for sessions**, **50–200 for user histories**. Longer is not free — every extra position is compute, and old interactions are often noisier than helpful.

### Q3. Why are Transformers usually better than RNNs here?

Three reasons that matter in practice: parallel training, direct attention between any two positions (no vanishing gradients), and free interpretability via attention weights. RNNs are still useful when memory budgets are tight or when you need true streaming inference with state passed across requests.

### Q4. Does BERT4Rec's bidirectional attention actually help?

Sometimes. The bidirectional context gives you richer item representations and makes the model robust to gaps. But the `[MASK]`-append inference trick is awkward, training is more involved, and on many real catalogs a well-tuned SASRec matches it. Try SASRec first; reach for BERT4Rec when you have a large pretraining corpus or noisy logged sessions.

### Q5. How do I handle cold start?

For **new items** use content features (category, brand, price, text, image) — feature-aware models like BST do this naturally. For **new users or sessions** start with popular and trending items, lean on session-based models that need no history, or wrap the recommender in an exploration layer (epsilon-greedy or contextual bandits).

### Q6. Which evaluation metric should I report?

For most leaderboards, **HR@10 and NDCG@10 together**. HR@10 is interpretable; NDCG@10 captures ranking quality. **MRR** is a good third number when each query has exactly one ground-truth item. In production, always pair offline metrics with online A/B tests on click-through rate, conversion, and revenue — offline wins do not always translate.

### Q7. Can I combine sequential models with other approaches?

Yes — and you usually should. Common hybrids:

- **Sequential + collaborative filtering.** Combine temporal signals with user-item similarity.
- **Sequential + content.** Use item features alongside the sequence (BST, content-aware SASRec).
- **Sequential + knowledge graph.** Inject relational structure between items.
- **Multi-task.** Predict next item *and* category, *and* dwell time, etc.

### Q8. What is on the research frontier?

The 2023–2025 wave is dominated by:

- **LLM-based recommenders** that prompt or fine-tune a language model on serialized user histories.
- **Contrastive learning** to produce better-separated item representations.
- **Multi-modal sequences** mixing IDs with images, text, and audio.
- **Linear and sub-quadratic attention** for very long histories.
- **Causal inference** tools to ask *why* users behave a certain way, not just predict *what* comes next.

---

## Summary

Sequential recommendation explicitly models the temporal order of user interactions to predict what comes next. The field has evolved from the simplicity of Markov chains to the sophistication of Transformers and graph neural networks.

**Key takeaways:**

- **Order is signal.** The arrangement of past interactions carries information that bag-of-items models throw away.
- **Architecture progression.** Markov chains → RNNs (GRU4Rec) → CNNs (Caser) → Transformers (SASRec, BERT4Rec) → GNNs (SR-GNN) — each step solved a real limitation of its predecessor.
- **Session vs. user.** Session-based models thrive on anonymous, short-term intent; user-based models capture taste evolution. Most production systems blend both.
- **Picking an architecture** depends on sequence length, the value of side features, latency, and how repetitive sessions are.
- **Evaluate with HR@K, NDCG@K, MRR**, but rely on online A/B tests for the final word.
- **Scaling tricks** — negative sampling, ANN search, two-stage retrieval, distillation, quantization, caching — are not optional in production.

---

## Series Navigation

- **Previous:** [Part 5 — Embedding and Representation Learning](/en/recommendation-systems-5-embedding-techniques/)
- **Next:** [Part 7 — Graph Neural Networks for Recommendation](/en/recommendation-systems-7-graph-neural-networks/)
- [Back to Series Overview (Part 1)](/en/recommendation-systems-1-fundamentals/)
