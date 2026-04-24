---
title: "Recommendation Systems (4): CTR Prediction and Click-Through Rate Modeling"
date: 2024-05-05 09:00:00
tags:
  - Recommendation Systems
  - CTR Prediction
  - Deep Learning
  - Feature Interactions
categories:
  - Recommendation Systems
series: recommendation-systems
lang: en
mathjax: true
permalink: en/recommendation-systems-4-ctr-prediction/
description: "A practical guide to CTR prediction models -- from Logistic Regression and Factorization Machines to DeepFM, xDeepFM, DCN, AutoInt, and FiBiNet -- with intuitive explanations and PyTorch implementations."
---

Every time you scroll through a social-media feed, click a product recommendation, or watch a suggested video, a CTR (click-through rate) model decided what to show you. These models answer one deceptively small question:

> **"What is the probability that this specific user will click on this specific item, right now?"**

Behind that question is one of the most economically valuable problems in machine learning. A 1% lift in CTR translates into millions of dollars at Google, Amazon, or Alibaba scale -- and the same models also drive video feeds, app stores, news apps, and dating apps. CTR prediction sits at the heart of the **ranking** stage: candidate generation gives you a few thousand items, and the CTR model decides which dozen actually reach the user.

This article is a tour through the decade-long evolution of CTR models, from a single-line logistic regression to attention-based architectures. We will not just look at formulas. For each model we will ask three questions:

1. **What problem in the previous model forced this design?**
2. **What is the geometric or probabilistic intuition?**
3. **How would you actually implement and ship it?**

By the end you should be able to read any modern CTR paper, sketch its architecture from memory, and pick the right baseline for your own system.

## What You Will Learn

- The CTR prediction problem and **why** it is uniquely hard (it is not just classification with imbalanced labels)
- **Logistic Regression** as both a baseline and a sanity check -- and exactly where it breaks
- **Factorization Machines (FM)** and **Field-aware FM (FFM)** for automatic pairwise interactions on sparse data
- **DeepFM** -- the industry workhorse that combines FM and a deep network
- **xDeepFM** -- explicit *high-order* interactions through the Compressed Interaction Network
- **DCN** -- bounded-degree feature crosses with linear parameter cost
- **AutoInt** -- self-attention applied to feature interactions
- **FiBiNet** -- learning *which features matter* with SENet plus richer bilinear interactions
- Training reality: class imbalance, calibration, AUC vs Logloss, and how to evaluate offline before A/B tests

## Prerequisites

- Comfortable Python and PyTorch (`nn.Module`, training loops, embeddings)
- Basic deep-learning concepts and the embedding view of categorical features ([Part 3](/en/recommendation-systems-3-deep-learning-basics/))
- Familiarity with binary classification, sigmoid, and cross-entropy

---

## Understanding the CTR Prediction Problem

### What Is CTR Prediction?

CTR prediction is **binary classification with extreme structure**. Given a user, an item, and the surrounding context, we estimate

$$P(y = 1 \mid \mathbf{x}) \quad\text{where } y \in \{0, 1\},\;\; 1 = \text{click}.$$

The feature vector $\mathbf{x}$ is the concatenation of three families:

| Family | Examples |
|---|---|
| User | user id, age bucket, gender, history, country |
| Item | item id, brand, category, price band, freshness |
| Context | hour of day, device, network, query, position |

Empirically, $\text{CTR} = \text{clicks} / \text{impressions}$, and the model output is later used to **rank** candidates, **filter** low-quality ones, and feed a downstream business objective (e.g. eCPM = CTR x bid for ads, or a multi-objective score for feeds).

### Why CTR Prediction Is Hard

Five properties make CTR prediction look like a standard classification task and behave like nothing of the sort:

**1. Extreme class imbalance.** Display ads sit at 0.1-2%, e-commerce at 1-5%, news feeds at 2-10%. A "predict no" model gets 95%+ accuracy and is useless -- AUC and Logloss replace accuracy.

**2. High-dimensional, ultra-sparse features.** After one-hot encoding, the feature space is $10^6$ to $10^9$ dimensions. Each sample lights up only dozens of them. Storing a weight per feature pair is impossible.

**3. The signal lives in interactions.** "Young user" alone is a weak signal; "young user x action movie x evening" is gold. Capturing those crosses *automatically* and *cheaply* is the central modelling problem.

**4. Distribution shift is constant.** New items, viral trends, weekday/weekend cycles. Models retrain daily or hourly; offline AUC alone never tells the full story.

**5. Hard latency budget.** Ranking has to score thousands of candidates in well under 100 ms (often under 10 ms p99). Model size, embedding lookup, and batching matter as much as architecture.

### The CTR Prediction Pipeline

The end-to-end view from a raw click log to a ranked list and back to model retraining looks like this:

![End-to-end CTR pipeline: raw logs, feature engineering, embedding, model, ranking, A/B and feedback loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig6_pipeline.png)

A few things to notice in the pipeline:

- **Feature engineering** still dominates real systems. Embeddings learn what they can, but explicit cross features and statistical features (rolling CTR per user, per item, per slot) routinely win the largest A/B tests.
- **Embeddings are shared infrastructure.** All deep CTR models (FM, DeepFM, xDeepFM, DCN, AutoInt, FiBiNet) read from the same embedding table. The architecture mostly defines *how the embeddings interact*.
- **Online feedback closes the loop.** Yesterday's serving log is today's training data. Model freshness often beats model sophistication.

With that mental map, let us walk the architecture timeline.

---

## Logistic Regression: The Foundation (and the Reason FM Exists)

Despite living next to giant neural networks in production, Logistic Regression (LR) refuses to die. It is the universal baseline, the calibration anchor, and -- in latency-bounded systems -- still the actual scorer for a non-trivial fraction of requests.

### How It Works

LR models the click probability as a single linear scoring function passed through a sigmoid:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}.$$

> **Plain English:** "Take a weighted sum of every feature, add a bias, then squash to $[0, 1]$."

We train it by minimising binary cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \big[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \big].$$

### Why LR Is Both Beloved and Insufficient

The geometry tells the whole story. LR can only learn a hyperplane in feature space. Any pattern that requires "feature A is good *only when* feature B is also active" is invisible to it. The classic illustration is XOR-shaped click behaviour:

![Left: LR cannot separate XOR-shaped click data with a single line. Right: a non-linear interaction term recovers the structure.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig1_lr_limitation.png)

In the left panel, "young + action" and "old + comedy" both click, but "young + comedy" and "old + action" do not. No linear boundary works -- AUC stays near 0.5. The right panel adds a single interaction term ($x_1 \cdot x_2$) and instantly recovers the structure. Every CTR model after LR is, at heart, an answer to the question:

> **"How do we discover and represent useful feature crosses, automatically and at scale?"**

### Implementation

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class LogisticRegression(nn.Module):
    """Logistic Regression for CTR prediction."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_lr(X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    scaler = StandardScaler()
    X_train_s = torch.FloatTensor(scaler.fit_transform(X_train))
    X_val_s = torch.FloatTensor(scaler.transform(X_val))
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    model = LogisticRegression(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_s), y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_s), y_val_t)
            print(f"Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

    return model, scaler
```

### Where LR Falls Short -- Concretely

1. **No feature interactions.** Treats every feature as independent.
2. **Manual feature engineering.** To capture interactions you must hand-craft `user_age x item_category` columns -- impossible past two- or three-way crosses.
3. **Linear decision boundary.** Visible above; no representation power for XOR-style structure.

These three failures motivate every subsequent architecture in this article.

---

## Factorization Machines (FM): Automatic Pairwise Interactions

Steffen Rendle's 2010 Factorization Machines were the first model that made automatic pairwise interactions both practical and statistically efficient on sparse data.

### The Core Insight

A naive "interaction-aware LR" would learn a separate weight $w_{ij}$ for every pair of features. With $d$ features that is $O(d^2)$ parameters -- and most pairs are *never observed together* in the training set, so they cannot be learned anyway.

FM replaces the per-pair weight with the dot product of two learnable vectors:

$$w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f} \, v_{j,f}.$$

> **Analogy.** Imagine 1,000 movies. Storing a weight for every pair needs a million numbers, most never observed. Instead, give each movie a $k$-dimensional "personality vector". Two movies interact strongly iff their vectors point similarly. We now have $1000 \cdot k$ numbers, and we can predict an interaction even for a pair we have *never seen together* -- because each vector was learned from many other co-occurrences.

That last property -- generalisation to unseen pairs -- is the real magic. It is why FM still works on extreme sparsity where decision trees and linear models stall.

### Mathematical Formulation

$$\hat{y}(\mathbf{x}) = \underbrace{w_0}_{\text{bias}} + \underbrace{\sum_{i=1}^{d} w_i x_i}_{\text{linear}} + \underbrace{\sum_{i=1}^{d} \sum_{j=i+1}^{d} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j}_{\text{pairwise interactions}}.$$

The interaction term *looks* $O(d^2)$ but admits a beautiful $O(k \cdot d)$ closed form:

$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \left[ \left(\sum_i \mathbf{v}_i x_i \right)^2 - \sum_i (\mathbf{v}_i x_i)^2 \right].$$

> **Why this works.** Squaring the sum gives all $i \cdot j$ products including $i = j$; subtracting the sum of squares removes the diagonal; halving removes the double-count.

### Implementation

```python
import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """Factorization Machine for CTR prediction (field-style indices)."""

    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.linear = nn.Linear(sum(field_dims), 1)
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

    def forward(self, x):
        """x: [batch, num_fields] of categorical indices (one per field)."""
        # Linear part on a one-hot view
        linear_out = self.linear(self._one_hot(x))

        # Stack per-field embeddings: [batch, num_fields, embed_dim]
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(len(self.field_dims))],
            dim=1,
        )

        # Efficient pairwise interaction (the (sum^2 - sum_of_squares)/2 trick)
        sum_square = torch.sum(embs, dim=1) ** 2
        square_sum = torch.sum(embs ** 2, dim=1)
        interaction = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)

        return torch.sigmoid(linear_out + interaction)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh


model = FactorizationMachine(field_dims=[10, 20, 15], embed_dim=16)
x = torch.LongTensor([[0, 5, 2], [3, 10, 8], [1, 7, 1], [9, 15, 12]])
print(model(x).squeeze())
```

### FM: Strengths and Limitations

**Strengths.** Pairwise interactions for free, $O(kd)$ compute, and statistical generalisation to unseen pairs.

**Limitations.** Only pairwise, and a feature uses *the same* embedding regardless of which other field it is interacting with -- which is sometimes wrong. That single observation gave us FFM.

---

## Field-aware Factorization Machines (FFM)

FFM (2016) extends FM with one targeted change: **each feature gets a separate embedding for each field it is interacting with.**

### The Intuition

In FM, the embedding for "action movie" is the same vector regardless of whether it is interacting with "user age" or "time of day". But intuitively, age-vs-genre and hour-vs-genre are different stories. FFM gives every feature one embedding *per opposite field*.

$$\hat{y}(\mathbf{x}) = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_{i, f_j}, \mathbf{v}_{j, f_i} \rangle x_i x_j.$$

The notation $\mathbf{v}_{i, f_j}$ reads "feature $i$'s embedding *when interacting with field* $f_j$".

### Implementation

```python
class FFM(nn.Module):
    """Field-aware Factorization Machine."""

    def __init__(self, field_dims, num_fields, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = num_fields
        self.linear = nn.Linear(sum(field_dims), 1)
        # Each feature gets one embedding per opposite field
        self.embeddings = nn.ModuleList([
            nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(num_fields)])
            for dim in field_dims
        ])

    def forward(self, x):
        b = x.size(0)
        linear_out = self.linear(self._one_hot(x))

        interaction = torch.zeros(b, 1, device=x.device)
        for i in range(len(self.field_dims)):
            for j in range(i + 1, len(self.field_dims)):
                v_i_fj = self.embeddings[i][j](x[:, i])  # i's emb for field j
                v_j_fi = self.embeddings[j][i](x[:, j])  # j's emb for field i
                interaction += (v_i_fj * v_j_fi).sum(dim=1, keepdim=True)

        return torch.sigmoid(linear_out + interaction)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

### FFM vs FM Trade-offs

| Aspect | FM | FFM |
|---|---|---|
| Parameters | $O(d \cdot k)$ | $O(d \cdot F \cdot k)$, with $F$ fields |
| Expressiveness | Same embedding for all interactions | Field-aware embeddings |
| Domain knowledge | Not required | Need a field schema |
| Typical use | First baseline | Won early Criteo / Avazu Kaggle competitions |

Both stop at pairwise interactions. To go higher we have two options: **stack non-linearities** (deep networks) or **build interactions explicitly** (CIN, Cross). DeepFM does the first; xDeepFM and DCN do the second.

Before continuing, here is a one-look summary of the *interaction primitives* used by the rest of the article:

![Side-by-side comparison of the interaction operators used by FM, FFM, DeepFM, DCN, and AutoInt](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig2_interaction_methods.png)

---

## DeepFM: Combining FM with Deep Learning

DeepFM (Huawei, 2017) is, with little exaggeration, the default starting point for deep CTR models. Its idea is structurally simple: run an FM and a deep network **in parallel**, sharing the embedding table.

### Why This Combination Works

- The **FM branch** captures pairwise (low-order) interactions explicitly.
- The **Deep branch** captures higher-order interactions implicitly through stacked non-linearities.
- **Shared embeddings** halve the parameter count and force both branches to agree on what each feature *means*.

> **Analogy.** Two detectives on the same case. FM is the rule-based investigator who is great with simple clues ("these two features always co-occur with clicks"). The deep MLP is the pattern-matcher who finds long, fuzzy chains of evidence. They argue, then add up their scores.

The architecture diagram makes the parallel structure obvious:

![DeepFM architecture: shared embedding feeding parallel FM and Deep branches, summed before the sigmoid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig3_deepfm_arch.png)

### Mathematical Formulation

$$\hat{y}(\mathbf{x}) = \sigma\big(y_{\text{FM}} + y_{\text{Deep}}\big),$$

where $y_{\text{FM}}$ is the standard FM expression and $y_{\text{Deep}}$ flows through an MLP over the concatenated embeddings:

$$\mathbf{h}_0 = [\mathbf{v}_1; \mathbf{v}_2; \ldots; \mathbf{v}_m], \quad \mathbf{h}_l = \text{ReLU}(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad y_{\text{Deep}} = \mathbf{w}^\top \mathbf{h}_L + b.$$

### Implementation

```python
class DeepFM(nn.Module):
    """DeepFM: parallel FM and Deep network with a shared embedding table."""

    def __init__(self, field_dims, embed_dim=16, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )

        # FM branch
        fm_linear = self.linear(self._one_hot(x))
        sum_sq = torch.sum(embs, dim=1) ** 2
        sq_sum = torch.sum(embs ** 2, dim=1)
        fm_interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
        fm_out = fm_linear + fm_interaction

        # Deep branch (over the SAME embeddings)
        deep_out = self.mlp(embs.view(embs.size(0), -1))

        return torch.sigmoid(fm_out + deep_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

DeepFM is the **go-to baseline**. If you are bootstrapping a new CTR system, start here, then ablate the FM branch, ablate the Deep branch, and only invest in something more exotic if either ablation costs you AUC.

The next two models came out of an honest observation: a deep MLP learns interactions *implicitly*, and you cannot tell which interactions it actually learned. That motivates xDeepFM (CIN) and DCN (cross network), which both make the high-order structure explicit.

---

## xDeepFM: Explicit High-Order Feature Interactions

xDeepFM (eXtreme Deep Factorization Machine, 2018) introduces the **Compressed Interaction Network (CIN)**, which builds higher-order interactions layer by layer in the embedding space.

### How CIN Works

Think of CIN as a pyramid of interactions:

- **Layer 0:** the original embeddings (degree-1 features).
- **Layer 1:** every Layer-0 feature crossed elementwise with every original embedding (degree 2).
- **Layer 2:** every Layer-1 feature crossed with every original embedding (degree 3).
- ...

At each layer, the cross is followed by a learnable convolutional compression:

$$\mathbf{X}^k_{h, *} = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} W^{k,h}_{i,j} \big(\mathbf{X}^{k-1}_{i,*} \circ \mathbf{X}^0_{j,*}\big),$$

where $\circ$ is the Hadamard (elementwise) product and $W$ are learned weights.

> **Plain English.** "Take every feature map from the previous layer, cross it elementwise with every original embedding, then apply a learned 1x1 convolution to compress all those crosses back down to a manageable number of feature maps. Stack."

The full xDeepFM is **Linear + CIN + Deep MLP** -- a three-tower model, summed before the sigmoid.

### Implementation

```python
class CIN(nn.Module):
    """Compressed Interaction Network used by xDeepFM."""

    def __init__(self, num_fields, embed_dim, cin_layer_sizes=(100, 100)):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.cin_layers = nn.ModuleList()
        prev_size = num_fields
        for layer_size in cin_layer_sizes:
            self.cin_layers.append(
                nn.Conv1d(prev_size * num_fields, layer_size, kernel_size=1)
            )
            prev_size = layer_size

    def forward(self, embeddings):
        """embeddings: [batch, num_fields, embed_dim]"""
        b = embeddings.size(0)
        X_0 = embeddings
        X_k = X_0
        outputs = []
        for cin_layer in self.cin_layers:
            H = X_k.size(1)
            inter = X_k.unsqueeze(2) * X_0.unsqueeze(1)        # [b, H, m, D]
            inter = inter.view(b, H * self.num_fields, self.embed_dim)
            X_k = torch.relu(cin_layer(inter))                   # [b, layer, D]
            outputs.append(X_k.sum(dim=2))                       # sum-pool over D
        return torch.cat(outputs, dim=1)


class xDeepFM(nn.Module):
    """xDeepFM = Linear + CIN + Deep MLP (over a shared embedding)."""

    def __init__(self, field_dims, embed_dim=16,
                 cin_layer_sizes=(100, 100), mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        self.cin = CIN(self.num_fields, embed_dim, cin_layer_sizes)
        self.cin_proj = nn.Linear(sum(cin_layer_sizes), 1)

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        linear_out = self.linear(self._one_hot(x))
        cin_out = self.cin_proj(self.cin(embs))
        deep_out = self.mlp(embs.view(embs.size(0), -1))
        return torch.sigmoid(linear_out + cin_out + deep_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

### xDeepFM vs DeepFM

| Aspect | DeepFM | xDeepFM |
|---|---|---|
| Low-order interactions | Explicit (FM) | Explicit (FM + CIN) |
| High-order interactions | Implicit (deep MLP only) | Explicit (CIN) + Implicit (deep) |
| Interpretability | Limited | Better -- you can probe CIN feature maps |
| Inference cost | Lower | Higher (CIN dominates) |
| When to pick it | Default starting point | Complex datasets where DeepFM plateaus |

---

## Deep & Cross Network (DCN): Bounded-Degree Cross Features

DCN (Google, 2017) takes a different route. Instead of stacking elementwise products with learnable convolutions, it adds a tiny module called the **Cross Network** that increases the polynomial degree of the interaction by exactly one per layer, with O($d$) parameters per layer.

### The Cross Layer

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \cdot (\mathbf{w}_l^\top \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l.$$

> **Plain English.** "Take a learned scalar projection of the current state, multiply it by the *original* input vector, add a bias, plus a residual." Each step injects $\mathbf{x}_0$ once more, raising the interaction degree by one.

After $L$ cross layers you have learned a polynomial of degree $L+1$ in the original features -- but with only $L \cdot d$ parameters in the cross stack.

The picture below shows both ideas: how each cross layer adds degree, and how dramatically cheaper this is than naively expanding all polynomial monomials.

![Left: each cross layer injects x0 again, raising the polynomial degree by one. Right: parameter cost vs degree -- DCN scales linearly while explicit polynomials explode](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig4_dcn_cross.png)

The right panel is the punchline. At degree 6 with 100 input features, an explicit polynomial expansion needs $10^{12}$ parameters. The cross network needs 600.

### Implementation

```python
class CrossNetwork(nn.Module):
    """Cross Network: bounded-degree interactions, linear in d per layer."""

    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x):
        x_0 = x
        x_l = x
        for layer in self.layers:
            x_l_w = layer(x_l)              # scalar projection [batch, 1]
            x_l = x_0 * x_l_w + x_l         # broadcast + residual
        return x_l


class DCN(nn.Module):
    """Deep & Cross Network."""

    def __init__(self, field_dims, embed_dim=16,
                 cross_layers=3, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        input_dim = self.num_fields * embed_dim

        self.cross_net = CrossNetwork(input_dim, cross_layers)

        layers = []
        prev = input_dim
        for dim in mlp_dims:
            layers += [
                nn.Linear(prev, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        self.final = nn.Linear(input_dim + 1, 1)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        flat = embs.view(embs.size(0), -1)

        cross_out = self.cross_net(flat)     # [batch, input_dim]
        deep_out = self.mlp(flat)            # [batch, 1]

        combined = torch.cat([cross_out, deep_out], dim=1)
        return torch.sigmoid(self.final(combined))
```

**DCN advantages.**

- Explicit, *bounded* interaction degree -- no surprises in production.
- Cross stack is tiny relative to the deep MLP, so latency stays close to a plain MLP.
- Successfully deployed at Google scale; v2 of the paper introduces DCN-Mix for even higher capacity.

---

## AutoInt: Attention as a Feature-Interaction Engine

AutoInt (2019) brings **multi-head self-attention** -- the engine inside Transformers -- to feature interactions. The key claim: not all interactions matter equally, and attention can learn *which* feature pairs to focus on, with multiple heads learning multiple notions of "related".

### How It Works

Treat each field's embedding as a token. Project to query, key, value:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}.$$

> **Plain English.** "Each feature asks 'whose embedding should I read from?' (Q), advertises what it knows (K), and offers content to be aggregated (V). Softmax over similarity gives the routing weights."

With $H$ heads, the model learns $H$ parallel notions of feature relatedness. Stacking $L$ AutoInt blocks lets information flow more than once, building deeper compositions.

### Implementation

```python
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention used inside an AutoInt block."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: [batch, num_fields, embed_dim]"""
        B, N, D = x.size()
        residual = x
        x = self.norm(x)

        def reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = reshape(self.W_q(x)), reshape(self.W_k(x)), reshape(self.W_v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.dropout(self.W_o(out)) + residual


class AutoInt(nn.Module):
    """AutoInt: stacked multi-head self-attention over feature embeddings."""

    def __init__(self, field_dims, embed_dim=16, num_attn_layers=3,
                 num_heads=4, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        h = embs
        for layer in self.attn_layers:
            h = layer(h)

        linear_out = self.linear(self._one_hot(x))
        mlp_out = self.mlp(h.view(h.size(0), -1))
        return torch.sigmoid(linear_out + mlp_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

**AutoInt advantages.**

- Discovers which interactions matter without manual schema design.
- Attention weights are *inspectable*, which helps debugging and reporting.
- Multi-head structure naturally captures multiple flavours of feature relationship.

---

## FiBiNet: Feature Importance + Bilinear Interactions

FiBiNet (2019) tackles two assumptions other models bake in silently:

1. **All features deserve equal attention.** They do not. Some carry strong signal; some are noise. FiBiNet uses **SENet** to learn a per-field importance gate.
2. **Interactions are well captured by elementwise products.** Sometimes they are not. FiBiNet replaces the Hadamard product with a **bilinear** form that can model asymmetric, richer interactions.

### SENet: Learning Feature Importance

Three steps:

- **Squeeze.** Average each field's embedding along the embedding dimension to a scalar -- one importance score per field.
- **Excitation.** A two-layer MLP (bottleneck) maps the scalars to per-field gates.
- **Reweight.** Multiply each embedding by its gate.

> **Analogy.** A DJ adjusting volume sliders for each track based on what is currently playing.

### Bilinear Interaction

Replace $\mathbf{v}_i \odot \mathbf{v}_j$ with $\mathbf{v}_i^\top \mathbf{W} \mathbf{v}_j$ where $\mathbf{W}$ is learned. Variants share $\mathbf{W}$ across all field pairs (Field-All), per field (Field-Each), or per pair (Field-Interaction).

### Implementation

```python
class SENet(nn.Module):
    """Squeeze-and-Excitation gating over fields."""

    def __init__(self, num_fields, reduction=4):
        super().__init__()
        reduced = max(1, num_fields // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced),
            nn.ReLU(),
            nn.Linear(reduced, num_fields),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [batch, num_fields, embed_dim]"""
        z = x.mean(dim=2)                            # squeeze
        weights = self.excitation(z).unsqueeze(2)    # excite
        return x * weights                           # reweight


class BilinearInteraction(nn.Module):
    """Bilinear interaction with a single shared W."""

    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x):
        n = x.size(1)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                vi_W = torch.matmul(x[:, i:i + 1, :], self.W)
                out.append((vi_W * x[:, j:j + 1, :]).squeeze(1))
        return torch.stack(out, dim=1)


class FiBiNet(nn.Module):
    """Feature Importance and Bilinear-feature-interaction Network."""

    def __init__(self, field_dims, embed_dim=16, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)
        self.senet = SENet(self.num_fields)
        self.bilinear = BilinearInteraction(embed_dim)

        num_pairs = self.num_fields * (self.num_fields - 1) // 2
        mlp_input = self.num_fields * embed_dim * 2 + num_pairs * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        linear_out = self.linear(self._one_hot(x))
        senet_embs = self.senet(embs)
        bilinear_out = self.bilinear(embs)

        mlp_in = torch.cat([
            embs.view(embs.size(0), -1),
            senet_embs.view(senet_embs.size(0), -1),
            bilinear_out.view(bilinear_out.size(0), -1),
        ], dim=1)
        return torch.sigmoid(linear_out + self.mlp(mlp_in))

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

---

## Model Comparison and Selection Guide

Now the question every practitioner asks: **does any of this actually move AUC?**

The figure below summarises typical relative ordering on Criteo-style benchmarks. Numbers are illustrative -- absolute values vary by dataset, embedding size, and training budget -- but the *gap pattern* is consistent across published reports.

![Bar charts comparing AUC and Logloss across LR, FM, FFM, DeepFM, xDeepFM, DCN, AutoInt, FiBiNet on Criteo-style benchmarks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/04-ctr-prediction/fig5_auc_logloss.png)

Two observations matter more than the absolute numbers:

1. **The biggest single jump is LR -> FM.** Adding pairwise interactions, even cheaply, is worth more AUC than any later architectural refinement.
2. **DeepFM and beyond live within ~0.005 AUC of each other.** That sounds tiny. At Google or Meta scale, 0.5 milli-AUC is real money. At a startup with 1M users, it is in the noise -- features and freshness will dominate.

### Computational Complexity

| Model | Parameters | Training Speed | Inference Speed |
|---|---|---|---|
| LR | $O(d)$ | very fast | very fast |
| FM | $O(d \cdot k)$ | fast | fast |
| FFM | $O(d \cdot F \cdot k)$ | medium | medium |
| DeepFM | $O(d \cdot k + \text{MLP})$ | medium | medium |
| xDeepFM | $O(d \cdot k + \text{CIN} + \text{MLP})$ | slow | medium |
| DCN | $O(d \cdot k + L \cdot d + \text{MLP})$ | medium | medium |
| AutoInt | $O(d \cdot k + L \cdot \text{Attn} + \text{MLP})$ | medium | medium |
| FiBiNet | $O(d \cdot k + \text{SE} + \text{Bilinear} + \text{MLP})$ | medium | medium |

### Feature Interaction Capabilities

| Model | Low-Order | High-Order | Explicit | Implicit |
|---|---|---|---|---|
| LR | linear only | no | no | no |
| FM | pairwise | no | yes | no |
| FFM | pairwise (field-aware) | no | yes | no |
| DeepFM | pairwise | yes | yes (FM) | yes (DNN) |
| xDeepFM | pairwise | bounded | yes (CIN) | yes (DNN) |
| DCN | bounded degree | yes | yes (Cross) | yes (DNN) |
| AutoInt | all orders | yes | yes (Attention) | yes (DNN) |
| FiBiNet | bilinear pairs | yes | yes (Bilinear) | yes (DNN) |

### A Decision Flowchart You Can Actually Use

- **First system / proof of concept.** Use **LR** or **FM**. Get the pipeline, evaluation, and serving right before you add layers.
- **First "real" model.** **DeepFM**. Strongest performance-to-effort ratio in the table.
- **DeepFM plateaued and you have GPU budget.** Try **DCN** (cheaper) or **xDeepFM** (richer), not both at once.
- **Heterogeneous fields, want interpretability of interaction weights.** **AutoInt**.
- **Long, noisy feature lists where you suspect feature-importance varies a lot.** **FiBiNet**.
- **Ultra-low latency, edge serving.** **LR** or **FM** for online; deeper model offline for re-ranking or retrieval bootstrapping.

---

## Training Strategies and Best Practices

### Handling Class Imbalance

CTR data is brutally imbalanced. Three reliable tools:

**1. Weighted BCE loss.**

```python
# pos_weight upweights the minority (positive) class in the loss
pos_weight = torch.tensor([num_negatives / num_positives])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**2. Negative downsampling.** Standard at Facebook-scale; just remember that downsampling miscalibrates the predicted probability and you have to re-calibrate before serving.

```python
import random

def sample_negatives(positives, item_pool, user_history_fn, k=4):
    out = []
    for user_id, pos_item in positives:
        candidates = item_pool - set(user_history_fn(user_id))
        for neg in random.sample(list(candidates), min(k, len(candidates))):
            out.append((user_id, neg, 0))
    return out
```

**3. Focal loss.** Down-weights easy examples so the gradient focuses on the few hard ones.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.where(targets == 1, preds, 1 - preds)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

### Regularisation

- **Dropout 0.2-0.5** in MLP layers; never on embedding lookups directly.
- **L2** (weight decay 1e-5 to 1e-6) on dense weights; embeddings often need *less* regularisation than weights.
- **Early stopping** on validation AUC, patience 3-10 epochs.

```python
def train_with_early_stopping(model, train_loader, val_loader,
                              epochs=100, patience=10):
    best_loss, wait, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)
        if val_loss < best_loss:
            best_loss, wait, best_state = val_loss, 0, model.state_dict().copy()
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model
```

### Evaluation Metrics

**AUC-ROC** is the headline metric. It measures the probability that a random positive sample is scored above a random negative one -- by construction, it is invariant to the label imbalance.

```python
from sklearn.metrics import roc_auc_score, log_loss


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).squeeze().cpu().numpy())
            labels.extend(y.cpu().numpy())
    return {
        'AUC': roc_auc_score(labels, preds),
        'LogLoss': log_loss(labels, preds),
    }
```

**Calibration matters too**, often more than AUC for downstream auctions. Predicted CTR of 0.05 should match an empirical 5% click rate inside that probability bucket. Use `sklearn.calibration.calibration_curve` to plot reliability diagrams; fix systematic over/under-prediction with **Platt scaling** or **isotonic regression** before serving.

---

## Frequently Asked Questions

### Q1: Why is CTR prediction binary classification, not regression?

The target is a probability (the chance of a click), and Bernoulli is the right likelihood. Binary classification has well-established metrics (AUC, Logloss), handles imbalance gracefully, and produces interpretable scores between 0 and 1. Regression on click counts is sometimes used for revenue or watch-time estimation, but for click prediction specifically, BCE is the standard.

### Q2: How do I choose the embedding dimension?

Start with 16. For small datasets (< 1M samples), 4-8 is usually enough. For huge datasets (> 100M), try 16-64. Run a quick ablation: if doubling the dimension lifts AUC by less than 0.001, go back to the smaller value. Embedding tables dominate model memory and serving cost.

### Q3: What is the difference between FM and matrix factorisation?

Matrix factorisation decomposes a *single* user-item rating matrix into user and item embeddings. FM is strictly more general: it factorises pairwise interactions among *any* features, so it can absorb side information (age, city, time of day) into the same factorised form. MF is the special case of FM with two fields.

### Q4: When should I use DeepFM vs xDeepFM?

Default to **DeepFM**. Try xDeepFM only after DeepFM clearly plateaus and your dataset is rich enough that the *third*- and *fourth*-order interactions plausibly matter. The CIN component nearly doubles inference cost.

### Q5: How do I handle cold-start items?

Four levers, usually combined: (1) initialise embeddings from content features (text/image encoders); (2) fall back to popularity for the first few impressions; (3) explore via a contextual bandit so new items get *some* impressions; (4) pre-train embeddings on a related task. The rule of thumb: *never* let a model see only the item id of a new item.

### Q6: Feature engineering vs model architecture -- which matters more?

Almost always feature engineering, by 2-3x. Good cross features, sensible bucketisation, proper missing-value handling, and rolling user/item statistics typically yield 10-30% AUC improvement. Switching architectures within the deep CTR family yields 2-10%. Do feature engineering first; reach for fancier architectures last.

### Q7: How do I handle missing features?

Four options: (1) default value (0, mean, mode); (2) add a binary `is_missing` indicator; (3) reserve a special "missing" embedding for categorical features; (4) impute with KNN or a simple model. Choose based on whether *missingness itself is informative* -- if a logged-out user is missing demographics, that fact predicts behaviour and should be a feature.

### Q8: How do I evaluate offline vs online?

**Offline:** time-based train/validation/test split (never random!). Metrics: AUC and Logloss. Fast and cheap, but downstream effects (diversity, freshness, position bias) are invisible. **Online:** A/B test with real users. Metrics: realised CTR, conversions, revenue, retention. Slow and expensive but the only authoritative signal. Always validate offline first; never ship without an online test.

### Q9: How do I deploy CTR models in production?

The big four: (1) serve from TorchServe / Triton / TF-Serving with batched requests; (2) target < 10 ms p99 via INT8 quantisation, embedding sharding, and pre-fetching; (3) monitor predicted-CTR distribution drift -- if the histogram shifts, retrain or roll back; (4) version your models *and your feature pipelines together* -- a feature schema mismatch silently destroys AUC.

### Q10: What are the latest trends (2024-2025)?

Transformer-based interaction stacks at scale, multi-task learning (jointly predicting CTR + conversion + watch time), graph neural networks over user-item graphs, AutoML for embedding dimension and architecture search, debiasing via causal inference and inverse-propensity weighting, and federated learning for privacy. The fundamentals -- feature quality, interaction modelling, calibration, freshness -- remain the dominant levers regardless of the trend.

---

## Summary

CTR prediction is the heart of modern ranking. We walked the architecture timeline from a single linear layer to attention-based interaction discovery:

1. **LR** -- simple, calibrated, but blind to feature interactions.
2. **FM / FFM** -- automatic pairwise interactions on sparse data; FFM adds field awareness at a parameter cost.
3. **DeepFM** -- the industry workhorse: explicit pairwise (FM) + implicit deep, sharing one embedding table.
4. **xDeepFM** -- explicit higher-order interactions through CIN.
5. **DCN** -- bounded-degree polynomial crosses with linear parameter cost.
6. **AutoInt** -- multi-head self-attention for interaction discovery and inspection.
7. **FiBiNet** -- learnable feature importance (SENet) plus bilinear interactions.

**Practical takeaways.**

- **Start simple.** LR -> FM -> DeepFM, in that order. Stop the moment you stop improving AUC.
- **Features first, architecture second.** A new cross-feature usually beats a new model.
- **Handle imbalance deliberately.** Pick one of pos_weight, downsampling+calibration, or focal loss -- and stick with it.
- **Evaluate honestly.** Time-based splits offline, A/B tests online, and watch calibration alongside AUC.
- **Iterate forever.** CTR systems are never done. Distributions shift, items churn, and yesterday's model is today's baseline.

The "best" model is the one that wins *your* A/B test under *your* latency budget on *your* data. Understand the problem first; pick the smallest tool that solves it.

---

## Series Navigation

| Part | Topic | Link |
|---|---|---|
| 1 | Introduction to Recommendation Systems | [Read Part 1](/en/recommendation-systems-1-introduction/) |
| 2 | Collaborative Filtering | [Read Part 2](/en/recommendation-systems-2-collaborative-filtering/) |
| 3 | Deep Learning Basics for RecSys | [Read Part 3](/en/recommendation-systems-3-deep-learning-basics/) |
| **4** | **CTR Prediction Models** | **You are here** |
| 5 | Feature Interaction Models | [Read Part 5](/en/recommendation-systems-5-feature-interactions/) |
| 6 | Sequence-Based Recommendations | [Read Part 6](/en/recommendation-systems-6-sequence-models/) |
| ... | ... | ... |
| 16 | Production Systems and MLOps | [Read Part 16](/en/recommendation-systems-16-production/) |
