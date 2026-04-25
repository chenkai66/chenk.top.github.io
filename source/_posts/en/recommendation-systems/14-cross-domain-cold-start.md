---
title: "Recommendation Systems (14): Cross-Domain Recommendation and Cold-Start Solutions"
date: 2026-01-09 09:00:00
tags:
  - Recommendation Systems
  - Cross-Domain
  - Cold Start
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 14
  total: 16
lang: en
mathjax: true
description: "Cold-start and cross-domain recommendation in depth: the three faces of cold-start, EMCDR/PTUPCDR cross-domain bridges, MeLU/MAML meta-learning, UCB bandits for exploration, and the cold-to-warm production stack."
disableNunjucks: true
series_order: 14
---

> When Netflix launches in a new country, it inherits millions of users with zero history and a catalog with no local ratings. Amazon faces the same problem each time it opens a new product category. Pure collaborative filtering — the workhorse of warm-state recommendation — has nothing to compute on. The discipline that makes recommendations work in this regime is a stack of techniques: bootstrap heuristics for the first request, meta-learning after a handful of interactions, cross-domain transfer when a related domain is rich, and bandits to keep exploring once the model is confident. This post walks through that stack, anchored to the papers it descends from.

## What you will learn

- **The three cold-start regimes** — new user, new item, new system — and why each demands a different lever
- **Cross-domain bridges** — from EMCDR's shared MLP to PTUPCDR's per-user meta network
- **Meta-learning for recsys** — MAML's bilevel optimization and how MeLU specialises it for users
- **Exploration vs exploitation** — UCB1, ε-greedy, Thompson sampling and their regret bounds
- **The cold-to-warm progression** — which method dominates at which interaction count
- **Content-based fallback** — the always-on safety net for new items

## Prerequisites

- Working knowledge of collaborative filtering and matrix factorization (Parts 3-4)
- Comfort with PyTorch and basic gradient descent (Part 7)
- Willingness to read one or two equations carefully

---

## The Three Faces of Cold-Start

![Three cold-start regimes shown as user-item interaction matrices: missing row for new user, missing column for new item, mostly empty grid for new system](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig1_cold_start_scenarios.png)

A recommendation system trained on history fails in three distinct ways when history is missing. Each looks like a different shape of hole in the user-item matrix.

### User cold-start

A user signs up. We may know their device, IP region, and the campaign that brought them in — possibly an age band from a sign-up form. We have **zero** explicit interactions. Their row in the rating matrix is blank. Industry data points are blunt: poor first-session recommendations correlate with roughly 3× higher 30-day churn on consumer apps. The lever here is **inference from sparse signals plus aggressive exploration**.

### Item cold-start

A merchant uploads a new SKU. A studio releases a new film. A creator publishes a new video. The column is blank. Collaborative filtering cannot retrieve the item because nobody has co-interacted with it. Without intervention, the item never gets exposure, never accumulates ratings, and stays invisible — the so-called rich-get-richer problem. The lever is **content features**: title, image, embedding from a pretrained encoder, plus seeding via similar warm items.

### System cold-start

A new platform launches, or an existing platform expands into a new vertical. The matrix is mostly empty. Both rows and columns are sparse simultaneously. The lever is **transfer**: borrow embeddings, mappings, or even ranking models from a related, data-rich domain.

### A formalism that covers all three

Given users $U$, items $I$, and an interaction matrix $R \in \mathbb{R}^{|U| \times |I|}$, define cold-start subsets $U_{\text{cold}} \subset U$ with $|R_{u,\cdot}| < K$ for $u \in U_{\text{cold}}$, and similarly $I_{\text{cold}}$. The objective is to predict $R_{u,i}$ for $(u, i)$ pairs where at least one side is cold. Standard CF learns embeddings $\mathbf{e}_u, \mathbf{e}_i$ from co-occurrence patterns; on cold rows or columns these embeddings either don't exist or are random initializations with no gradient signal. Every method below is essentially an answer to the question: **what do we use instead of $\mathbf{e}_u$ or $\mathbf{e}_i$ when the data isn't there?**

---

## Cross-Domain Recommendation

![Cross-domain pipeline: source-domain interactions feed embeddings, a bridge function maps them into target-domain space, target predictor scores cold-start users](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig2_cross_domain_transfer.png)

The intuition is old: a user who loves Stanley Kubrick's *2001: A Space Odyssey* probably reads Arthur C. Clarke. Their movie behavior is a strong prior on their book preferences even though the items are different. The technical question is how to operationalize "strong prior" — what function maps a user's representation in the movie domain to a representation in the book domain?

### EMCDR — the shared MLP bridge

[Man et al., IJCAI 2017](https://www.ijcai.org/proceedings/2017/0343.pdf) introduced **EMCDR (Embedding and Mapping for Cross-Domain Recommendation)**, the cleanest formulation. It has three stages.

1. **Train domain-specific MF.** Run matrix factorization on source interactions $R^S$ and target interactions $R^T$ separately, yielding user embeddings $U^S, U^T$ and item embeddings $V^S, V^T$.
2. **Train a mapping on overlap users.** For users $i \in U_o = U^S \cap U^T$ that exist in both domains, learn an MLP $f_\phi$ that minimizes $\sum_{i \in U_o} \|f_\phi(\mathbf{u}_i^S) - \mathbf{u}_i^T\|^2$.
3. **Predict for cold target users.** For a user who only exists in the source, set $\hat{\mathbf{u}}_i^T = f_\phi(\mathbf{u}_i^S)$ and feed it into the target predictor.

The bridge is **global** — every user is mapped through the same $f_\phi$.

### PTUPCDR — personalize the bridge itself

EMCDR's weakness is that one mapping cannot capture how different users translate across domains. A horror-movie fan and a documentary fan probably need very different mappings into the book domain. [Zhu et al., WSDM 2022](https://arxiv.org/abs/2110.11154) propose **PTUPCDR (Personalized Transfer of User Preferences for Cross-Domain Recommendation)**: instead of one $\phi$, generate a per-user $\phi_i$ from the user's source-domain behavior using a meta network.

$$
\phi_i = h_\theta\bigl(\{\mathbf{v}_j^S : j \in \mathcal{H}_i^S\}\bigr), \qquad \hat{\mathbf{u}}_i^T = f_{\phi_i}(\mathbf{u}_i^S)
$$

In words: read the user's source-domain history, summarize it into a small set of bridge weights, then apply that personalized bridge to the user's source embedding. PTUPCDR reports MAE reductions of 5–10% over EMCDR on the standard Amazon cross-category benchmarks.

![Side-by-side: EMCDR uses a single shared MLP for all users; PTUPCDR generates per-user bridge parameters via a meta network](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig3_emcdr_ptupcdr.png)

### A minimal cross-domain skeleton

```python
import torch
import torch.nn as nn


class CrossDomainBridge(nn.Module):
    """Shared-bridge cross-domain model in the EMCDR style.

    Stage 1 (not shown): pretrain source/target MF separately.
    Stage 2: this module learns f_phi on overlap users.
    Stage 3: for a cold target user we run u_S through f_phi
             and score against target items.
    """

    def __init__(self, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_emb_source: torch.Tensor) -> torch.Tensor:
        return self.bridge(user_emb_source)


def emcdr_loss(bridge, u_source, u_target_true):
    """Train on overlap users only — supervised regression in embedding space."""
    u_target_pred = bridge(u_source)
    return torch.mean((u_target_pred - u_target_true) ** 2)
```

### Negative transfer — the failure mode to watch for

Cross-domain transfer is not free. If the source and target are weakly related (say, news-reading patterns and grocery purchases), the bridge can actively **hurt** target performance. The signal is straightforward: run an A/B against a target-only baseline and watch the cold-user metric. If transfer doesn't beat the baseline at the same training cost, the domains are too far apart.

---

## Meta-Learning for Cold-Start

The cross-domain story assumes you have a related rich domain. Meta-learning takes a different angle: even within a single domain, can we train the model so that it **adapts quickly** to a new user from just a handful of interactions?

### MAML in one paragraph

[Finn, Abbeel, and Levine (ICML 2017)](https://arxiv.org/abs/1703.03400) proposed **MAML (Model-Agnostic Meta-Learning)**: instead of learning parameters $\theta$ that minimize expected loss across tasks, learn $\theta$ that, after a few gradient steps on any task, performs well on that task. It's a bilevel optimization. The inner loop adapts:

$$
\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(f_\theta, \mathcal{S}_i)
$$

The outer loop optimizes the initialization through the adapted parameters:

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{T_i \sim p(\mathcal{T})} \mathcal{L}_{T_i}(f_{\theta'_i}, \mathcal{Q}_i)
$$

Geometrically, MAML pushes $\theta$ to a region of parameter space where every task's optimum is just a few gradient steps away.

![MAML loss landscape with three task minima and a meta-initialization equidistant from all three; right panel shows inner and outer loop equations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig4_maml_meta_learning.png)

### MeLU — MAML specialised for recommendation

[Lee et al. (KDD 2019)](https://arxiv.org/abs/1908.00413) adapt MAML to recommendation as **MeLU (Meta-Learned User preference estimator)**. The key engineering choice: only the **decision layers** are adapted in the inner loop; the **embedding layers** stay shared and slow-moving. This matches the inductive bias that item / category embeddings should be stable, while the way a user combines them is what changes per user.

Each "task" is one user. The support set $\mathcal{S}_i$ is the user's first 1–5 interactions; the query set $\mathcal{Q}_i$ is held out for the outer-loop loss. After meta-training on hundreds of thousands of users, a brand-new user can be handled by:

1. Take their first $K$ ratings as the support set.
2. Run $K{=}1$ to $5$ gradient steps on the decision layers.
3. Score all candidate items with the adapted model.

```python
class MeLU(nn.Module):
    """MeLU-style recommender: shared embeddings, adaptive decision head.

    Inner-loop adaptation only touches decision_head.parameters().
    Embeddings are meta-learned in the outer loop only.
    """

    def __init__(self, num_items, num_genres, embedding_dim=32):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        self.decision_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, item_ids, genre_ids):
        x = torch.cat(
            [self.item_embedding(item_ids), self.genre_embedding(genre_ids)],
            dim=-1,
        )
        return self.decision_head(x).squeeze(-1)


def melu_inner_adapt(model, support_items, support_genres, support_ratings,
                     inner_lr=0.01, n_steps=3):
    """Take a few gradient steps on the decision head only.

    Returns adapted parameters as a list of tensors with requires_grad=True
    so the outer loop can backprop through them.
    """
    fast_params = [p.clone().detach().requires_grad_(True)
                   for p in model.decision_head.parameters()]

    for _ in range(n_steps):
        # Forward pass with current fast_params
        x = torch.cat([model.item_embedding(support_items),
                       model.genre_embedding(support_genres)], dim=-1)
        for i in range(0, len(fast_params), 2):
            w, b = fast_params[i], fast_params[i + 1]
            x = torch.nn.functional.linear(x, w, b)
            if i < len(fast_params) - 2:
                x = torch.relu(x)
        pred = x.squeeze(-1)

        loss = torch.mean((pred - support_ratings) ** 2)
        grads = torch.autograd.grad(loss, fast_params, create_graph=True)
        fast_params = [p - inner_lr * g for p, g in zip(fast_params, grads)]

    return fast_params
```

### When meta-learning is worth the cost

MAML/MeLU costs 3–10× more training compute than a vanilla model because of the inner-loop unrolling and second-order gradients. Use it when:

- You have many users, each with very few interactions (the long-tail user distribution).
- You can identify a clear "task" boundary — one user, one session, one cohort.
- Bootstrap heuristics aren't enough but you don't have a related domain to transfer from.

If second-order gradients are too expensive, **FOMAML** drops them and recovers most of the benefit.

---

## Bandits — Exploration vs Exploitation

Once a model has *some* confidence about a user, the next question is what to actually serve. Always pick the highest-scoring item and you'll never learn whether the user might love something the model rates lower. Always pick randomly and you'll burn the user's session. The textbook framework is the **multi-armed bandit**.

### UCB1 — confidence bounds

[Auer, Cesa-Bianchi, and Fischer (2002)](http://aima.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer+al:2002.pdf) prove that the UCB1 rule

$$
a_t = \arg\max_a \left[ \hat\mu_a + \sqrt{\frac{2 \ln t}{n_a}} \right]
$$

achieves $O(\log t)$ cumulative regret — i.e., the gap between UCB and an oracle that always picks the best arm grows only logarithmically in the number of rounds. The formula has a clean interpretation: pick the item whose **upper confidence bound** is highest. Items with few pulls $n_a$ get a large exploration bonus and are tried; items with many pulls have tight bounds and are picked only if their estimated mean is genuinely high.

![Left: bar chart of estimated rewards with UCB whiskers showing items with few pulls have wide uncertainty bonuses; right: cumulative regret curves showing UCB and Thompson achieve logarithmic regret while greedy and random grow linearly](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig5_ucb_exploration.png)

### Thompson sampling — Bayesian alternative

Thompson sampling maintains a posterior over each arm's reward and samples from it; pick the arm whose sample is highest. Empirically it usually matches or beats UCB1 and is trivial to implement for Beta-Bernoulli rewards.

```python
import numpy as np


class UCB1Recommender:
    """UCB1 for K candidate items. Rewards in [0, 1]."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.counts = np.zeros(n_items, dtype=np.int64)
        self.means = np.zeros(n_items, dtype=np.float64)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        # Pull each arm once first
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        ucb = self.means + np.sqrt(2 * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, item: int, reward: float) -> None:
        self.counts[item] += 1
        n = self.counts[item]
        self.means[item] += (reward - self.means[item]) / n


class ThompsonSampling:
    """Beta-Bernoulli Thompson sampling for binary rewards (e.g. click / no click)."""

    def __init__(self, n_items: int, alpha=1.0, beta=1.0):
        self.alpha = np.full(n_items, alpha)
        self.beta = np.full(n_items, beta)

    def select(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, item: int, reward: float) -> None:
        self.alpha[item] += reward
        self.beta[item] += 1 - reward
```

In production, contextual bandits like LinUCB and neural variants extend these ideas to use user / item features rather than just per-arm counters. They are the canonical tool for the **few-shot regime** where meta-learning has produced a model but you still need to gather data efficiently.

---

## Content-Based Fallback

For new items, no amount of meta-learning helps if there's no signal at all. Content-based retrieval is the always-on safety net.

![Cold-start item flows through a feature encoder (BERT / CLIP / TF-IDF) into a content embedding, which is matched against warm catalog items by cosine similarity; predicted rating aggregates the top-K nearest warm items' ratings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig7_content_fallback.png)

The recipe is mechanical:

1. **Encode the new item** with a domain-appropriate encoder. Text → BERT / sentence-transformers. Images → CLIP. Tabular → handcrafted features + a small MLP.
2. **Find the K nearest warm items** by cosine similarity on the content embedding.
3. **Predict the rating** as a similarity-weighted average of those neighbors' ratings:

$$
\hat r_{u,i} = \frac{\sum_{j \in N_K(i)} \mathrm{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in N_K(i)} \mathrm{sim}(i, j)}
$$

```python
import numpy as np


class ContentFallback:
    """Predict ratings for a cold item by borrowing from K nearest warm items."""

    def __init__(self, item_features: np.ndarray, K: int = 20):
        # item_features: (n_items, d) precomputed embeddings (e.g. BERT, CLIP)
        norms = np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-8
        self.normed = item_features / norms
        self.K = K

    def predict(self, cold_item_idx: int, warm_ids: np.ndarray,
                warm_ratings: np.ndarray) -> float:
        sims = self.normed[warm_ids] @ self.normed[cold_item_idx]
        top = np.argpartition(-sims, kth=min(self.K, len(sims) - 1))[: self.K]
        s, r = sims[top], warm_ratings[top]
        s = np.maximum(s, 0)  # negative similarity → ignore
        return float((s * r).sum() / (s.sum() + 1e-8))
```

The same pattern works for **user cold-start** if the user provides any content-style signal (a survey on sign-up, a single onboarding click): encode the signal, find similar warm users, borrow their preferences.

---

## The Cold-to-Warm Production Stack

No single method dominates across the whole interaction-count axis. Production systems route requests to different methods based on how much data the user has accumulated.

![Performance vs interaction count with five curves: pure CF starts near zero and ramps slowly, content-based stays flat at a decent floor, meta-learning ramps fast in the few-shot region, cross-domain is strong from the start, hybrid envelope is on top throughout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/14-cross-domain-cold-start/fig6_cold_to_warm.png)

The shape of each curve tells a story:

- **Pure CF** is useless below ~5 interactions and slow to climb until ~20.
- **Content-based** has a flat floor — never great, never terrible, always available.
- **Meta-learning (MeLU)** is the steepest mover in the 1–10 interaction range — exactly what it's designed for.
- **Cross-domain (PTUPCDR)** starts highest because it inherits source-domain knowledge for free, but flattens earlier.
- **Hybrid stack** is the upper envelope: route to whichever method dominates at the current interaction count.

A practical routing rule:

| Interactions | Primary method               | Fallbacks                         |
|--------------|------------------------------|-----------------------------------|
| 0            | Cross-domain or popularity   | Content-based                     |
| 1-3          | Cross-domain + bandit        | Content-based                     |
| 3-20         | Meta-learning (MeLU)         | Cross-domain                      |
| 20+          | Full CF / DIN / sequential   | Meta-learning for new sessions    |

Keep a popularity baseline running in parallel as a circuit breaker — if any of the above methods produces low-confidence predictions, fall back to popular items in the user's inferred segment.

---

## Q&A

### My platform has no related domain to transfer from. Where do I start?

Lead with content-based fallback for items and a popularity prior for users, then meta-train MeLU on whatever interaction data you accumulate. Cross-domain is a 10–20% relative boost when available, not a prerequisite.

### How many interactions before MeLU outperforms a content baseline?

In published benchmarks (MovieLens, Bookcrossing) MeLU starts winning at 2–3 interactions and dominates by 5. Your mileage depends on how informative your support interactions are — diverse genres beat homogeneous ones.

### Is FOMAML really good enough, or should I bite the bullet on second-order MAML?

For recommendation, FOMAML is within 1–2% of MAML on standard metrics and trains 3× faster. Use second-order MAML only when you've measured that the gap matters for your business metric.

### How do I detect negative transfer in cross-domain?

Run a target-only baseline alongside the cross-domain model. If target performance on cold users does not improve, the domains are too distant. Common culprit: aligning users by ID across domains where the same user has wildly different behavior (work account vs personal account).

### Are bandits worth it for top-K recommendation, or just for single-pick problems?

Worth it for the **first slot or two** of a feed — that's where exploration value is highest. After the first few items, switch to ranking. Combinatorial bandits exist but are operationally expensive.

### How do I evaluate a cold-start system offline?

Hold out users entirely (not random interactions). For each held-out user, expose only their first $K$ interactions to the model as "support" and predict on the rest. Report metrics stratified by $K \in \{1, 3, 5, 10\}$ — a single number hides the cold-to-warm transition.

---

## Summary

Cold-start is not a single problem with a single solution; it's a **regime** that demands different machinery at different points on the interaction-count axis.

- The taxonomy (user / item / system) determines which lever — exploration, content, transfer — is actually available.
- **EMCDR** and **PTUPCDR** turn a data-rich source domain into a prior on a data-sparse target. PTUPCDR's per-user bridge is the current practical sweet spot.
- **MAML** and its recsys specialisation **MeLU** train models whose initialisations adapt in a few steps — perfect for the 3-10 interaction window.
- **UCB1** and **Thompson sampling** give the few-shot regime a principled exploration rule with logarithmic regret.
- **Content-based fallback** is the unglamorous always-on safety net.
- The **hybrid stack** routes by interaction count and keeps a popularity circuit breaker.

Build the stack incrementally. Start with content + popularity, layer in meta-learning once you have enough users to meta-train, add cross-domain when a related domain becomes available. Measure cold-user metrics separately from warm metrics; aggregate numbers will hide the regime where you're actually losing money.

---

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400). *ICML*.
- Lee, H., Im, J., Jang, S., Cho, H., & Chung, S. (2019). [MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation](https://arxiv.org/abs/1908.00413). *KDD*.
- Man, T., Shen, H., Jin, X., & Cheng, X. (2017). [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/proceedings/2017/0343.pdf). *IJCAI*.
- Zhu, Y., Tang, Z., Liu, Y., Zhuang, F., Xie, R., Zhang, X., Lin, L., & He, Q. (2022). [Personalized Transfer of User Preferences for Cross-domain Recommendation](https://arxiv.org/abs/2110.11154). *WSDM*.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](http://aima.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer+al:2002.pdf). *Machine Learning*, 47, 235-256.
- Snell, J., Swersky, K., & Zemel, R. (2017). [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175). *NeurIPS*.

---

## Series Navigation

This is **Part 14 of 16** in the Recommendation Systems series.

**Previous**: [Part 13 -- Fairness, Debiasing, and Explainability](/en/recommendation-systems-13-fairness-explainability/)
**Next**: [Part 15 -- Real-Time Recommendation and Online Learning](/en/recommendation-systems-15-real-time-online/)

[View all parts in the Recommendation Systems series](/categories/Recommendation-Systems/)
