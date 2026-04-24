---
title: "Recommendation Systems (10): Deep Interest Networks and Attention Mechanisms"
date: 2024-05-11 09:00:00
tags:
  - Recommendation Systems
  - DIN
  - Attention Mechanism
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 10
  total: 16
lang: en
mathjax: true
description: "From DIN's target attention to DIEN's AUGRU and BST's Transformer — how Alibaba taught CTR models to read a user's history like a chef reads the room."
---

A good chef doesn't cook the same dish for every guest. She watches you walk in, notes the wine you order, glances at how you eyed the chalkboard — and only then decides whether tonight's special should be the steak or the risotto. Your past visits matter, but only the parts that fit *this* mood.

A recommendation model used to be a worse chef. It would take everything the user had ever clicked, average it into a single vector, and serve the same dish to everyone in the room. That vintage leather jacket you viewed last week and the random phone charger you clicked six months ago carried equal weight, regardless of what you were looking at right now.

**Deep Interest Networks (DIN)** taught the model to read the room. The idea is unreasonably simple: when scoring a candidate item, weight each past behavior by how relevant it is to *that* candidate. The same user gets a different representation for every item — exactly as a chef cooks a different dish for every mood.

This article walks through the family of attention-based CTR models that grew out of that insight: DIN (target attention), DIEN (interest evolution with GRU + AUGRU), DSIN (session-aware), and BST (Transformer over behaviors). We'll keep the math honest, the code runnable, and the intuition sharp.

## What you will learn

- Why averaging user history loses critical information, and how attention fixes it
- **DIN** — target attention with a Local Activation Unit
- **DIEN** — modeling interest evolution with GRU + AUGRU + auxiliary loss
- **DSIN** — capturing session-level browsing patterns
- **BST** — Transformer over the behavior sequence + candidate
- Production tricks: Dice activation, mini-batch aware regularization, sequence truncation

## Prerequisites

- PyTorch basics (modules, forward pass, loss computation)
- Embeddings ([Part 5](/en/recommendation-systems-5-embedding-techniques/))
- Familiarity with RNN/GRU concepts (helpful but not required)

---

## 1. From averaging to attention

### The problem with averaging

Consider a user who has clicked five action movies, three rom-coms, two documentaries, and one horror film. When you score a new action movie, those five action clicks should dominate. A simple average treats all eleven equally — the horror outlier pulls the user's representation away from the very thing you're recommending.

Formally, the traditional approach computes a fixed user vector:

$$\mathbf{v}_u = \frac{1}{T} \sum_{j=1}^{T} \mathbf{e}_{b_j}$$

where $\mathbf{e}_{b_j}$ is the embedding of behavior $b_j$. The vector ignores the candidate entirely. Whether you're scoring an action movie or a documentary, the user looks the same.

### The attention fix

![DIN attention weights — same user, different weights for different candidates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig1_attention_weights.png)

Attention computes a relevance score $\alpha_j$ for each historical behavior $b_j$ with respect to the candidate item $i$:

$$\alpha_j = \text{score}(\mathbf{e}_{b_j}, \mathbf{e}_i)$$

The user representation becomes a **weighted** sum:

$$\mathbf{v}_u(i) = \sum_{j=1}^{T} \alpha_j \, \mathbf{e}_{b_j}$$

Now $\mathbf{v}_u(i)$ depends on $i$. Score an action movie and the action clicks light up. Score a rom-com and the rom-com clicks take over. Same history, different reading. The figure above shows exactly this: ten clicks from one user, two candidates, two completely different attention profiles. The model didn't change. The question did.

### Choosing a scoring function

Three common choices, in order of expressiveness:

- **Dot product** — $\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$. Cheap. Limited.
- **Scaled dot product** — divide by $\sqrt{d}$ to keep magnitudes stable. Used in Transformers.
- **Additive (MLP)** — $\mathbf{v}^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k} + \mathbf{b})$. Most expressive. **DIN's choice.**

DIN goes further still — instead of just concatenating $\mathbf{q}$ and $\mathbf{k}$, it feeds the MLP four things: query, key, query−key, and query⊙key. Subtraction captures *difference*, element-wise product captures *interaction*. The MLP learns a non-linear compatibility function over all four.

---

## 2. Deep Interest Network (DIN)

DIN was introduced by Alibaba in 2018 (Zhou et al., KDD'18) and remains the foundational attention-based CTR model. Its workhorse is the **Local Activation Unit** — a small MLP that scores each historical behavior against the candidate.

### How DIN works

Given a user's behavior sequence $[b_1, b_2, \ldots, b_T]$ and a candidate item $i$:

1. **Embed** behaviors, candidate, user features, context.
2. **Score** every behavior against the candidate via the activation unit.
3. **Weighted sum** of behavior embeddings → the "activated" user representation.
4. **Concatenate** with other features and pass through an MLP for CTR prediction.

The activation unit's score is:

$$\text{score}(\mathbf{e}_{b_j}, \mathbf{e}_i) = \text{MLP}\big([\,\mathbf{e}_{b_j};\ \mathbf{e}_i;\ \mathbf{e}_{b_j} - \mathbf{e}_i;\ \mathbf{e}_{b_j} \odot \mathbf{e}_i\,]\big)$$

A subtle but important detail: **DIN does not apply softmax** in the original paper. The authors found that letting weights sum to anything (not just 1) preserves the *intensity* of interest — a user with many strong matches should produce a larger user vector than a user with weak matches. We'll show both forms in the code.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalActivationUnit(nn.Module):
    """DIN's Local Activation Unit.

    Scores each historical behavior against the candidate item using a small
    MLP that consumes four interaction views: behavior, candidate, their
    difference, and their element-wise product.
    """

    def __init__(self, embedding_dim, hidden_dims=(80, 40), use_softmax=False):
        super().__init__()
        # Input = [behavior; candidate; behavior - candidate; behavior * candidate]
        in_dim = embedding_dim * 4
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.PReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.use_softmax = use_softmax

    def forward(self, behaviors, candidate, mask=None):
        """
        behaviors: (B, T, D)   historical behavior embeddings
        candidate: (B, D)      candidate item embedding
        mask:      (B, T)      1 for real positions, 0 for padding
        Returns:
            user_repr: (B, D)
            weights:   (B, T)
        """
        B, T, D = behaviors.shape
        cand = candidate.unsqueeze(1).expand(B, T, D)

        # Four interaction views
        feats = torch.cat([behaviors, cand, behaviors - cand, behaviors * cand], dim=-1)
        scores = self.mlp(feats).squeeze(-1)              # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if self.use_softmax:
            weights = F.softmax(scores, dim=1)
        else:
            # DIN's default: keep raw weights (no normalization)
            weights = scores
            if mask is not None:
                weights = weights * mask

        user_repr = torch.bmm(weights.unsqueeze(1), behaviors).squeeze(1)
        return user_repr, weights


class DIN(nn.Module):
    """Deep Interest Network for CTR prediction."""

    def __init__(self, item_dim=64, user_dim=32, ctx_dim=16,
                 mlp_hidden=(200, 80), dropout=0.5):
        super().__init__()
        self.activation = LocalActivationUnit(item_dim)

        in_dim = item_dim + item_dim + user_dim + ctx_dim
        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_feats, behaviors, candidate, ctx_feats, mask=None):
        user_repr, attn = self.activation(behaviors, candidate, mask)
        x = torch.cat([user_repr, candidate, user_feats, ctx_feats], dim=1)
        logit = self.mlp(x).squeeze(-1)
        return logit, attn


# Quick smoke test
model = DIN(item_dim=64, user_dim=32, ctx_dim=16)
B, T = 32, 20
logits, attn = model(
    user_feats=torch.randn(B, 32),
    behaviors=torch.randn(B, T, 64),
    candidate=torch.randn(B, 64),
    ctx_feats=torch.randn(B, 16),
    mask=torch.ones(B, T),
)
print(logits.shape, attn.shape)   # torch.Size([32]) torch.Size([32, 20])
```

### Training and the Alibaba production tricks

DIN is trained with binary cross-entropy on logits:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \big[ y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \big]$$

Three tricks the paper credits with most of the lift:

- **Dice activation** — a data-adaptive PReLU that shifts its inflection point with the batch distribution (Section 7).
- **Mini-batch aware regularization** — instead of L2-regularizing every embedding (millions of items, mostly never seen this batch), only regularize embeddings that appear in the current batch, weighted by their frequency. Roughly the same regularization signal at a fraction of the cost.
- **Gradient clipping** — long behavior sequences tend to explode gradients early in training.

---

## 3. Deep Interest Evolution Network (DIEN)

DIN treats history as a bag of behaviors. It ignores time. But interests *move* — last month you were researching laptops, this week you're chasing laptop accessories, next week the obsession shifts to ergonomic chairs.

![User interest shifts over weeks — DIEN models this trajectory, DIN does not](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig5_interest_evolution.png)

The figure tells the story DIN cannot. A user's "interests" aren't a single vector — they are a *time series*, with peaks that crest and recede in a predictable order. DIN sees the union of all peaks at once. DIEN (Zhou et al., AAAI'19) adds two layers on top of behavior embeddings to capture the trajectory.

### Architecture in one picture

![DIEN: GRU extracts interest at each time step, AUGRU evolves it toward the candidate](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig2_dien_architecture.png)

**Layer 1 — Interest Extractor (GRU).** A standard GRU over the behavior sequence:

$$\mathbf{h}_t = \text{GRU}(\mathbf{e}_{b_t}, \mathbf{h}_{t-1})$$

Each hidden state $\mathbf{h}_t$ is the user's interest at time $t$.

**Layer 2 — Interest Evolution (AUGRU).** A modified GRU whose update gate is multiplied by an attention weight $a_t$ — the relevance of $\mathbf{h}_t$ to the candidate:

$$\tilde{u}_t = a_t \cdot u_t \qquad \mathbf{h}'_t = (1 - \tilde{u}_t) \odot \mathbf{h}'_{t-1} + \tilde{u}_t \odot \tilde{\mathbf{h}}_t$$

Read it as: when a past interest is highly relevant to the candidate, let it drive the evolution. When it's irrelevant, freeze the state — don't let noise wash out the signal. The arrows in the figure are drawn with thickness proportional to $a_t$; thick arrows pump information forward, thin arrows leave the previous state mostly unchanged.

### The auxiliary loss trick

A bare GRU can learn lazy hidden states that minimize the CTR loss without actually representing interest. DIEN solves this with an **auxiliary loss** that forces $\mathbf{h}_t$ to predict the *next* behavior $b_{t+1}$:

$$\mathcal{L}_{\text{aux}} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \Big[ \log \sigma(\mathbf{h}_t^\top \mathbf{e}_{b_{t+1}}^+) + \log\big(1 - \sigma(\mathbf{h}_t^\top \mathbf{e}_{b_{t+1}}^-)\big)\Big]$$

Plain English: if the hidden state at time $t$ can predict what the user clicks at time $t+1$ (positive sample) and *can't* predict a randomly sampled negative, then it has captured something real.

The total objective is $\mathcal{L} = \mathcal{L}_{\text{ctr}} + \lambda \cdot \mathcal{L}_{\text{aux}}$ with $\lambda$ typically in $[0.1, 1.0]$.

### AUGRU implementation

```python
class AUGRUCell(nn.Module):
    """GRU cell with an attention-weighted update gate."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, a_t):
        """
        x_t:    (B, input_dim)   interest at time t (from GRU layer)
        h_prev: (B, hidden_dim)  previous evolved state
        a_t:    (B, 1)           attention weight wrt candidate
        """
        r = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev))
        n = torch.tanh(self.W_in(x_t) + r * self.W_hn(h_prev))
        z_tilde = a_t * z                       # ← the AUGRU twist
        h_t = (1 - z_tilde) * h_prev + z_tilde * n
        return h_t


class DIEN(nn.Module):
    def __init__(self, item_dim=64, user_dim=32, ctx_dim=16,
                 hidden_dim=64, mlp_hidden=(200, 80), dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(item_dim, hidden_dim, batch_first=True)
        self.attn = LocalActivationUnit(hidden_dim, use_softmax=True)
        self.augru = AUGRUCell(hidden_dim, hidden_dim)

        in_dim = hidden_dim + item_dim + user_dim + ctx_dim
        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_feats, behaviors, candidate, ctx_feats):
        # Layer 1: interest extraction
        interest, _ = self.gru(behaviors)               # (B, T, H)

        # Attention weights against candidate
        _, attn = self.attn(interest, candidate)        # (B, T)

        # Layer 2: interest evolution via AUGRU
        B, T, H = interest.shape
        h = torch.zeros(B, H, device=behaviors.device)
        for t in range(T):
            h = self.augru(interest[:, t, :], h, attn[:, t:t+1])
        final_interest = h                              # (B, H)

        x = torch.cat([final_interest, candidate, user_feats, ctx_feats], dim=1)
        return self.mlp(x).squeeze(-1), interest        # interest used for aux loss
```

In production, the per-timestep Python loop is replaced with a custom CUDA kernel — but conceptually this is what AUGRU does.

---

## 4. Deep Session Interest Network (DSIN)

User behavior tends to come in bursts. You spend fifteen minutes browsing laptops at lunch, come back at night to skim headphones, then look at running shoes the next morning. Each burst is internally coherent; the gaps between them often mark a shift in mood.

![DSIN: behaviors split into sessions by 30-minute gaps; self-attention within, Bi-LSTM across](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig3_dsin_sessions.png)

DSIN (Feng et al., IJCAI'19) makes that structure explicit. The figure traces the full pipeline on nine actions split across three sessions:

1. **Session split** — break the behavior sequence whenever the gap exceeds 30 minutes (the original paper's threshold).
2. **Intra-session self-attention** — within each session, multi-head self-attention captures the local pattern (which items in this burst relate to which).
3. **Inter-session Bi-LSTM** — across sessions, a Bi-LSTM models how interest drifts from one session to the next.
4. **Target attention** — finally, attention over session vectors weights them by relevance to the candidate.

The intuition: a session is the model's "thought unit." Treating thirty clicks as one undifferentiated bag throws away the fact that they came in three coherent chunks. Treating each click as its own time step ignores that clicks within a session are usually close in topic. Sessions are the right granularity in the middle.

```python
class DSINSessionLayer(nn.Module):
    """Self-attention within a session → average pool → session vector."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, session_behaviors, mask=None):
        # session_behaviors: (B, S, D)
        attended, _ = self.attn(session_behaviors, session_behaviors, session_behaviors,
                                key_padding_mask=mask)
        return attended.mean(dim=1)                      # (B, D)


def split_sessions(timestamps, gap_seconds=1800):
    """Split a behavior sequence into session boundaries by time gap."""
    sessions, start = [], 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > gap_seconds:
            sessions.append((start, i))
            start = i
    sessions.append((start, len(timestamps)))
    return sessions
```

### When to reach for which model

| Model | Key innovation | Best fit |
|-------|---------------|----------|
| **DIN** | Target attention on flat behavior list | Short histories, no clear time structure |
| **DIEN** | GRU + AUGRU + auxiliary loss | Long histories where interests evolve smoothly |
| **DSIN** | Intra-session self-attn + inter-session Bi-LSTM | Browsing patterns with clear session boundaries |
| **BST** | Transformer over behaviors + candidate | Long histories, parallelizable serving |

---

## 5. Behavior Sequence Transformer (BST)

By 2019 the Transformer had eaten NLP. Alibaba's Taobao team asked: what if we just put one over the behavior sequence and call it a day?

![BST: Transformer over the behavior sequence + candidate, then MLP for CTR](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig4_bst_architecture.png)

**BST** (Chen et al., DLP-KDD'19) treats the behavior sequence + the candidate item as a single token sequence and runs a Transformer encoder over it. Multi-head self-attention lets every behavior attend to every other behavior *and* to the candidate. Position embeddings encode time order.

The whole architecture is essentially:

$$\mathbf{Z} = \text{TransformerBlock}\big(\,[\mathbf{e}_{b_1} + \mathbf{p}_1,\, \ldots,\, \mathbf{e}_{b_T} + \mathbf{p}_T,\, \mathbf{e}_i + \mathbf{p}_{T+1}]\,\big)$$

Then concat $\mathbf{Z}$ with side features and feed an MLP. The reported lift on Taobao logs over a WDL baseline was ~7.5% AUC at the time. Notice what BST is *not* doing: it doesn't have an explicit "target attention" step. It doesn't need one. Self-attention over `[history, candidate]` already gives the candidate token direct access to every behavior — and, less obviously, gives every behavior direct access to every other behavior, which DIN never modeled.

```python
class BST(nn.Module):
    def __init__(self, item_dim=64, max_len=50, num_heads=8, num_layers=2,
                 user_dim=32, ctx_dim=16, mlp_hidden=(1024, 512, 256), dropout=0.2):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len + 1, item_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=item_dim, nhead=num_heads,
            dim_feedforward=item_dim * 4, dropout=dropout,
            batch_first=True, activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        in_dim = item_dim * (max_len + 1) + user_dim + ctx_dim
        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_feats, behaviors, candidate, ctx_feats, mask=None):
        # Concatenate candidate to the end of the behavior sequence
        seq = torch.cat([behaviors, candidate.unsqueeze(1)], dim=1)   # (B, T+1, D)
        pos = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos_embed(pos).unsqueeze(0)

        z = self.transformer(seq, src_key_padding_mask=mask)          # (B, T+1, D)
        flat = z.reshape(z.size(0), -1)
        x = torch.cat([flat, user_feats, ctx_feats], dim=1)
        return self.mlp(x).squeeze(-1)
```

---

## 6. How much do these tricks actually buy you?

![AUC progression on Amazon Books CTR benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig6_performance_comparison.png)

Numbers from the original DIN/DIEN/DSIN/BST papers on the Amazon Books CTR benchmark, normalized to comparable settings. Two things to notice:

- **The biggest jump is from sum/avg pooling to DIN.** Adding attention is the single most impactful change. The rest is incremental — DIEN adds a few tenths of a percent on top, DSIN a bit more, BST roughly matches or slightly beats DSIN depending on the dataset.
- **AUC gains look small but matter at scale.** A 0.005 AUC lift on Taobao translates to several percent CTR improvement and hundreds of millions in incremental GMV. This is why teams keep iterating on what looks like noise to outsiders.

Beyond accuracy, each model has a different cost profile: DIN serves cheaply because attention is just one MLP per behavior; DIEN's sequential AUGRU is the slowest; BST is fast on GPUs but heavy on memory; DSIN's bookkeeping (sessionizing on the fly) is the operational headache.

A reasonable rule of thumb: start with DIN. It captures 80% of the lift with 20% of the engineering. Reach for DIEN when behavior sequences are long and topical *order* matters (subscription-style products, hobbies that ramp). Reach for DSIN when sessions are obvious and frequent (short-video apps, e-commerce browsing). Reach for BST when you want one mental model that covers everything and your serving stack already loves Transformers.

---

## 7. Production tricks that actually move the needle

### Dice — a data-adaptive activation

![PReLU vs Dice — Dice shifts its inflection point with the batch distribution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/10-deep-interest-networks/fig7_activation_functions.png)

PReLU has a hard switch at $x = 0$ — fine if your activations are centered there, awkward if the batch distribution is shifted. Dice (Data-adaptive Activation) replaces the hard switch with a smooth sigmoid centered on the batch's running mean:

$$\text{Dice}(x) = p(x) \cdot x + (1 - p(x)) \cdot \alpha x, \qquad p(x) = \sigma\!\left(\frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)$$

The transition point now follows the data. The right panel of the figure shows three batches with different means — Dice's inflection rides along, while PReLU's stays nailed to zero. Different layers, different distributions, different effective activations — for free.

```python
class Dice(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.bn = nn.BatchNorm1d(dim, eps=eps, affine=False)

    def forward(self, x):
        x_norm = self.bn(x)
        p = torch.sigmoid(x_norm)
        return p * x + (1 - p) * self.alpha * x
```

### Mini-batch aware regularization

L2-regularizing 100M item embeddings every step is wasteful — the gradient hits ~99.99% zeros. Restrict the regularization to embeddings that appear in this batch, scaled by their batch frequency:

$$\mathcal{L}_{\text{reg}} = \frac{\lambda}{2} \sum_{j \in \mathcal{B}} \frac{n_{j,\mathcal{B}}}{n_j} \|\mathbf{e}_j\|^2$$

where $n_{j,\mathcal{B}}$ is item $j$'s count in batch $\mathcal{B}$ and $n_j$ is its global count. Same effect, orders of magnitude cheaper.

### Variable-length sequences

Real users have wildly different history lengths. Pad to a fixed max, then mask:

```python
def pad_and_mask(sequences, max_len, pad_value=0):
    padded, masks = [], []
    for seq in sequences:
        seq = seq[-max_len:]                                    # keep most recent
        pad_len = max_len - len(seq)
        padded.append([pad_value] * pad_len + list(seq))
        masks.append([0] * pad_len + [1] * len(seq))
    return torch.LongTensor(padded), torch.FloatTensor(masks)
```

In attention, set masked scores to $-10^9$ before softmax — this drives their weights to zero.

### Serving at scale

For millions of QPS:

- **Pre-compute and cache item embeddings offline.** Item table is static-ish; recompute nightly.
- **Truncate to the most recent N behaviors.** N = 50–100 captures most signal at a fraction of the cost.
- **Quantize.** FP16 or INT8 cuts model size 2–4x with negligible AUC loss.
- **Batch inference.** GPUs love batches of 64+ requests.
- **Replace AUGRU's Python loop with a custom CUDA op** if you really need DIEN in production.

---

## 8. FAQ

**Why target attention instead of self-attention in DIN?**
Target attention answers "which past behaviors are relevant to *this* candidate?" Self-attention only looks within the history ("laptop and phone are both electronics") — useful, but it doesn't condition on the candidate, which is the whole point. BST eventually shows you can have both at once with a Transformer.

**Why doesn't DIN use softmax?**
The authors found that softmax destroys *intensity*. A user with many strong matches and a user with one weak match would produce equally-normalized vectors. Without softmax, the magnitude of the user vector itself signals interest strength.

**Does the auxiliary loss really help?**
Yes — significantly on long sequences. Without it, the GRU can collapse to trivial states that minimize CTR loss without representing interest. The DIEN paper reports the aux loss alone is worth ~0.3% AUC on Amazon datasets.

**What about computational cost?**
Attention is $O(T^2 \cdot d)$ in sequence length — fine for $T \le 100$, painful beyond. For long histories, options are: truncate (most common), use sparse/linear attention, or two-stage retrieval (e.g., SIM hard search → DIN).

**How do you handle cold-start users?**
Fall back to user profile features (demographics, location, device) and category-level priors. Content-based item embeddings (from titles, images) help when behavior data is sparse on either side.

**Are attention weights actually interpretable?**
Mostly yes, with caveats. They show *which* past behaviors the model leaned on for a given recommendation, which is great for debugging and trust. But softmax-normalized weights are *relative* — high weight doesn't mean high absolute relevance, just relatively higher than the rest of the sequence.

---

## Conclusion

Deep Interest Networks brought one durable idea to recommendation: **not all past behaviors matter equally, and the model should figure out which ones do, every single time.**

The rest is variations on that theme:

1. **DIN** — weight behaviors by relevance to the candidate.
2. **DIEN** — model how those interests evolve in time.
3. **DSIN** — group them into sessions and respect the structure.
4. **BST** — let the Transformer figure out all of it.

A good chef doesn't cook the same dish for every guest. After DIN, neither does a good recommender.

---

## Series Navigation

This article is **Part 10** of the 16-part Recommendation Systems series.

| Previous | | Next |
|:---------|:-:|-----:|
| [Part 9: Multi-Task Learning](/en/recommendation-systems-9-multi-task-learning/) | [All Parts](/tags/Recommendation-Systems/) | [Part 11: Contrastive Learning](/en/recommendation-systems-11-contrastive-learning/) |
