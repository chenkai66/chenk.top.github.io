---
title: "Transfer Learning (4): Few-Shot Learning"
date: 2025-04-27 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Few-Shot Learning
  - Meta-Learning
  - MAML
  - Prototypical Networks
  - Metric Learning
categories:
  - Transfer Learning
series:
  name: "Transfer Learning"
  part: 4
  total: 12
lang: en
mathjax: true
description: "Learn new concepts from a handful of examples. Covers the N-way K-shot protocol, metric learning (Siamese, Prototypical, Matching, Relation networks), meta-learning (MAML, Reptile), episodic training, miniImageNet benchmarks, and a complete Prototypical Network implementation."
disableNunjucks: true
---

Show a child one photograph of a pangolin and they will spot pangolins for life. Show a deep learning model one photograph and it will give you a uniformly random guess. Few-shot learning is the field that closes that gap: building classifiers that work with only one to ten labeled examples per class.

The trick is not to memorize individual classes harder. It is to learn *how to learn* from very few examples, then carry that ability over to brand-new classes at test time. This article covers the two families that dominate the field today: **metric learning**, which learns a good distance function, and **meta-learning**, which learns a good initialization.

## What You Will Learn

- The N-way K-shot evaluation protocol and why standard training fails on it
- Metric learning: Siamese, Prototypical, Matching, and Relation networks
- Meta-learning: MAML and its first-order cousins (FOMAML, Reptile)
- Episodic training: matching training-time difficulty to test-time difficulty
- A clean, end-to-end Prototypical Network implementation in PyTorch

**Prerequisites:** Parts 1-2 of this series; comfort with PyTorch and basic optimization.

---

## The Few-Shot Challenge

![5-way 1-shot episode: support set on the left, query set on the right](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig1_nway_kshot.png)

### Problem Setup: N-way K-shot

The community uses a single, shared evaluation protocol so that papers are comparable:

- **N-way:** the model must classify among $N$ classes.
- **K-shot:** for each class, only $K$ labeled examples are available.

A "5-way 1-shot" task is therefore: here is one labeled image from each of five classes you have never seen before; now classify a batch of new query images.

Each evaluation episode consists of:

- a **support set** $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{NK}$ -- the $N \times K$ labeled examples,
- a **query set** $\mathcal{Q} = \{(x_j, y_j)\}_{j=1}^{NQ}$ -- the unlabeled images to classify (with hidden labels used only to measure accuracy).

Reported numbers are averages over hundreds or thousands of episodes drawn from a held-out **novel-class** set, with 95% confidence intervals because the per-episode variance is large.

### Why Standard Training Fails

Three forces conspire against a vanilla classifier:

- **Data scarcity.** With $K = 1$ you literally cannot estimate a within-class variance. With $K = 5$ you can, but barely.
- **Overfitting.** A high-capacity network will memorize the support examples instead of learning a class-discriminative rule.
- **Inter-class similarity.** Novel classes drawn from the same domain (e.g. two breeds of dog) often differ only in subtle features that a randomly initialized classifier has no reason to attend to.

Empirical risk minimization with weight decay is not enough: regularization stops parameters from blowing up, but it does not inject the inductive bias required to generalize from a single image.

### The Core Insight

To learn from few samples you need **prior knowledge.** Few-shot learning gets that prior by training on a large set of *base classes* (with many examples each), then evaluating on disjoint *novel classes* (with few). The two main routes are:

1. **Metric learning** -- train a backbone whose embedding space already separates classes, so a fresh class can be characterized by the location of its few support points. Classify queries by their distance in this space.
2. **Meta-learning** -- train across many simulated few-shot tasks so the network *learns to be adapted* by a few gradient steps. Treat "fast adaptation" itself as the thing to optimize.

Both share the same data split (base vs. novel) but invest the prior knowledge differently: metric learning bakes it into the embedding; meta-learning bakes it into the optimization initialization.

---

## Metric Learning: Classification by Distance

![Prototypical Network embedding space with class clusters, prototypes, and decision regions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig2_prototypical.png)

The metric-learning recipe is one sentence long: learn an embedding $f_\theta$ such that same-class samples cluster together and different-class samples lie far apart, then classify a query by its proximity to the support points.

### Siamese Networks

The earliest member of the family. Two weight-shared encoders $f_\theta$ embed a pair of inputs and the distance is

$$d(x_1, x_2) = \|f_\theta(x_1) - f_\theta(x_2)\|_2.$$

Training uses the **contrastive loss**

$$\mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2,$$

with $y = 1$ for same-class pairs (pull together) and $y = 0$ for different-class pairs (push apart until the distance exceeds margin $m$). At test time, classify a query by the label of its nearest support sample.

### Prototypical Networks

Prototypical networks improve on the pairwise picture by collapsing each support class into a single point.

#### Computing prototypes

For class $c$ with support examples $\{x_1^c, \ldots, x_K^c\}$, the prototype is the mean embedding:

$$\mathbf{c}_c = \frac{1}{K} \sum_{k=1}^{K} f_\theta(x_k^c).$$

Geometrically it is the centroid of the class in embedding space.

#### Classification

Score each class by the negative squared Euclidean distance from the query embedding to the prototype, then take a softmax:

$$P(y = c \mid x_q) = \frac{\exp\bigl(-d(f_\theta(x_q), \mathbf{c}_c)\bigr)}{\sum_{c'} \exp\bigl(-d(f_\theta(x_q), \mathbf{c}_{c'})\bigr)}, \qquad d(u, v) = \|u - v\|_2^2.$$

Train end-to-end with cross-entropy on the query predictions of each episode.

#### Why prototypes are principled

If we model class-conditional embeddings as Gaussians with shared isotropic covariance, $P(x \mid y = c) = \mathcal{N}(\mu_c, \sigma^2 I)$, then the maximum-likelihood class is exactly the nearest centroid. Prototypical networks are therefore the deep-learning incarnation of the Bayes-optimal classifier under that (admittedly strong) assumption -- which is why it tends to work so well in practice.

A second, cleaner, observation: under squared-Euclidean distance the decision boundary between any two classes is a hyperplane in embedding space. So Prototypical networks are equivalent to a *linear* classifier in the learned space, but with the linear weights tied to the prototype geometry.

### Matching Networks

Matching networks replace the hard nearest-prototype rule with a soft attention over the entire support set.

![Matching Network attention: cosine similarity becomes a softmax weight over support samples](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig3_matching.png)

The prediction is a label-weighted sum:

$$P(y \mid x_q, \mathcal{S}) = \sum_{i=1}^{NK} a(x_q, x_i) \cdot y_i, \qquad a(x_q, x_i) = \mathrm{softmax}_i\bigl(\cos(f(x_q), g(x_i))\bigr).$$

Here $y_i$ is a one-hot label vector, so the prediction is a convex combination of one-hots weighted by attention.

The other contribution of the paper is **full context embeddings**: a bidirectional LSTM is run over the support set so each support embedding is aware of every other support sample. The intuition is that what counts as a discriminative feature depends on the other classes you are trying to separate from -- and the LSTM lets the network express that.

### Relation Networks

Relation networks take the next step: instead of choosing a fixed metric (Euclidean, cosine), they *learn* one. A small network $g_\phi$ takes the concatenated embeddings and outputs a scalar similarity:

$$r_{q, c} = g_\phi\bigl(\mathrm{concat}(f_\theta(x_q),\, \mathbf{c}_c)\bigr) \in [0, 1].$$

![Relation Network: shared embedding + learned relation module](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig5_relation.png)

The training target is $r_{q, c} = \mathbb{1}\{y_q = c\}$ with mean-squared-error loss; both modules are trained jointly. Why bother? Fixed metrics implicitly assume the embedding space is isotropic -- every dimension counts equally. A learned metric lets the network downweight dimensions that turn out to be uninformative for the task at hand.

---

## Meta-Learning: Learning to Learn

Where metric learning bakes the prior into the embedding, meta-learning bakes it into the optimization process itself. The model is trained across many tasks so that *adapting* it to a new task takes only a handful of gradient steps.

### MAML: Model-Agnostic Meta-Learning

MAML's idea is simple and surprisingly effective: search for an initialization $\theta$ such that one or two gradient steps on any new task's support set already produce a good model.

![MAML: a single meta-initialization adapts via inner-loop steps to many tasks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig4_maml.png)

#### Algorithm

For each sampled task $\mathcal{T}_i$ (with its own support and query sets):

1. **Inner loop (per-task adaptation).** Take one (or a few) gradient steps on the support loss:
   $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{support}}(\theta).$$
2. **Outer loop (meta-update).** Evaluate the *adapted* parameters on the query set and update the initialization:
   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta_i').$$

The outer-loop gradient differentiates *through* the inner-loop update, which involves second derivatives of the support loss with respect to $\theta$ -- a Hessian-vector product.

#### First-order approximation (FOMAML)

The exact second-order MAML update costs $O(d^2)$ memory in the parameter dimension $d$ and is fiddly to implement. FOMAML drops the second-order term and approximates

$$\nabla_\theta \mathcal{L}(\theta_i') \approx \nabla_{\theta_i'} \mathcal{L}(\theta_i'),$$

which is just the gradient at the adapted point, evaluated as if $\theta_i'$ did not depend on $\theta$. Cost drops to $O(d)$, and reported accuracies barely change.

#### Geometric intuition

MAML pushes $\theta$ toward a region of the loss landscape that is *flat with respect to fast adaptation*: from this point a few steps in any task-specific direction reach a low loss. Think of $\theta$ as a universal launching pad rather than a universally-good model.

### Reptile: Even Simpler

Reptile drops the inner-loop differentiation entirely. Sample a task, run $k$ ordinary SGD steps on it to get $\tilde{\theta}$, then nudge the meta-parameters toward the result:

$$\theta \leftarrow \theta + \epsilon \,(\tilde{\theta} - \theta).$$

That's the whole algorithm. Despite the simplicity it works almost as well as MAML, because moving the meta-parameters toward task-specific solutions across many tasks ends up locating $\theta$ near a shared sweet spot.

| Method  | Gradient order | Per-step cost   | Implementation | miniImageNet (5w-5s)\* |
|---------|----------------|-----------------|----------------|------------------------|
| MAML    | Second-order   | High (Hessian)  | Hard           | ~63%                   |
| FOMAML  | First-order    | Medium          | Easy           | ~62%                   |
| Reptile | First-order    | Low             | Trivial        | ~66%                   |

\*Reported in the original papers; numbers vary across implementations.

---

## Episodic Training

Standard supervised training shows the network the entire base-class dataset and asks it to classify. Episodic training reframes the entire training loop to look like the test loop.

![Episodic training: each step is a fresh N-way K-shot task sampled from the base classes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig7_episodic.png)

### How an episode is built

Each iteration:

1. Sample $N$ classes from the base-class pool.
2. Sample $K$ examples per class for the **support set**.
3. Sample $Q$ additional examples per class for the **query set**.
4. Train the model to classify the queries given only that support set.

```python
for epoch in range(num_epochs):
    for episode in range(episodes_per_epoch):
        classes = sample(base_classes, N)
        support = sample_from_classes(classes, K)
        query   = sample_from_classes(classes, Q)

        prototypes = compute_prototypes(support)
        logits     = -distance(query, prototypes)
        loss       = cross_entropy(logits, query_labels)

        loss.backward()
        optimizer.step()
```

### Why this matters

The model never gets to see the full base-class dataset at once. Every gradient update simulates a few-shot task, so the inductive bias the network develops is precisely the one needed at test time. This is curriculum learning where the curriculum *is* the test-time conditions.

A good sanity check: turn off episodic training and just train a flat $|C_{\text{base}}|$-way classifier, then drop a linear head on the frozen features. With a strong backbone (deep ResNet, large augmentation) this "Baseline++" recipe is competitive with metric- and meta-learning approaches -- a result Chen et al. (ICLR 2019) used to argue that episodic training matters less than people thought, and that backbone capacity matters more.

---

## How Well Does Any of This Work?

![miniImageNet 5-way benchmark accuracies across representative methods](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig6_mini_imagenet.png)

The numbers above are from the original papers (with later work routinely surpassing them by using larger backbones and pre-training tricks). Two things to take away:

- **The 1-shot vs. 5-shot gap is huge.** Going from one example to five typically adds 10-20 percentage points -- a reminder that even a tiny amount of data dominates clever architecture choices.
- **Methods cluster.** Once the backbone is held fixed, Prototypical, Matching, Relation, and MAML-family numbers land within a few points of each other. Pick by engineering taste (simplicity, compute budget, tooling) rather than chasing the last point of accuracy.

---

## Complete Implementation: Prototypical Networks

```python
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool, the standard miniImageNet building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ProtoNetEncoder(nn.Module):
    """4-layer CNN encoder. Maps an 84x84 RGB image to a 1600-d vector."""
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),   # 84 -> 42
            ConvBlock(hidden_dim, hidden_dim),    # 42 -> 21
            ConvBlock(hidden_dim, hidden_dim),    # 21 -> 10
            ConvBlock(hidden_dim, hidden_dim),    # 10 ->  5
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def compute_prototypes(self, support_emb, support_labels, n_way):
        """Average embeddings per class to form prototypes."""
        prototypes = [
            support_emb[support_labels == c].mean(dim=0)
            for c in range(n_way)
        ]
        return torch.stack(prototypes)  # (n_way, embed_dim)

    def forward(self, support_imgs, support_lbls, query_imgs, n_way):
        support_emb = self.encoder(support_imgs)
        query_emb   = self.encoder(query_imgs)
        prototypes  = self.compute_prototypes(support_emb, support_lbls, n_way)
        # Negative squared Euclidean distance == logits
        dists  = torch.cdist(query_emb, prototypes, p=2)
        return -dists


class EpisodeSampler:
    """Yields N-way K-shot episodes from a flat (data, labels) array."""
    def __init__(self, data, labels, n_way, n_support, n_query, n_episodes):
        self.data, self.labels = data, labels
        self.n_way, self.n_support = n_way, n_support
        self.n_query, self.n_episodes = n_query, n_episodes
        self.classes = np.unique(labels)
        self.class_indices = {c: np.where(labels == c)[0] for c in self.classes}

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._sample()

    def _sample(self):
        chosen = np.random.choice(self.classes, self.n_way, replace=False)
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []

        for new_label, c in enumerate(chosen):
            idxs = np.random.choice(
                self.class_indices[c],
                self.n_support + self.n_query,
                replace=False,
            )
            for idx in idxs[:self.n_support]:
                s_imgs.append(self.data[idx]); s_lbls.append(new_label)
            for idx in idxs[self.n_support:]:
                q_imgs.append(self.data[idx]); q_lbls.append(new_label)

        return (
            torch.stack([torch.FloatTensor(x) for x in s_imgs]),
            torch.LongTensor(s_lbls),
            torch.stack([torch.FloatTensor(x) for x in q_imgs]),
            torch.LongTensor(q_lbls),
        )


def train(model, train_data, train_lbls, val_data, val_lbls,
          n_way=5, n_support=5, n_query=15, n_episodes=100,
          num_epochs=50, lr=1e-3, device='cpu'):
    """Episodic training with periodic validation."""
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        sampler = EpisodeSampler(train_data, train_lbls,
                                 n_way, n_support, n_query, n_episodes)
        loss_sum, acc_sum = 0.0, 0.0
        for s_img, s_lbl, q_img, q_lbl in tqdm(sampler, desc=f'Epoch {epoch+1}'):
            s_img, s_lbl = s_img.to(device), s_lbl.to(device)
            q_img, q_lbl = q_img.to(device), q_lbl.to(device)

            logits = model(s_img, s_lbl, q_img, n_way)
            loss = crit(logits, q_lbl)

            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item()
            acc_sum  += (logits.argmax(1) == q_lbl).float().mean().item()

        # ---- validate ----
        model.eval()
        val_sampler = EpisodeSampler(val_data, val_lbls,
                                     n_way, n_support, n_query, n_episodes)
        val_acc = 0.0
        with torch.no_grad():
            for s_img, s_lbl, q_img, q_lbl in val_sampler:
                s_img, s_lbl = s_img.to(device), s_lbl.to(device)
                q_img, q_lbl = q_img.to(device), q_lbl.to(device)
                logits = model(s_img, s_lbl, q_img, n_way)
                val_acc += (logits.argmax(1) == q_lbl).float().mean().item()
        val_acc /= n_episodes
        print(f"  train_loss={loss_sum/n_episodes:.4f}  "
              f"train_acc={acc_sum/n_episodes:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), 'best_protonet.pt')


# ---- demo with random data ----
if __name__ == '__main__':
    num_classes, samples_per_class, img_size = 64, 600, 84
    all_data = np.random.randn(num_classes * samples_per_class,
                               3, img_size, img_size).astype(np.float32)
    all_labels = np.repeat(np.arange(num_classes), samples_per_class)

    train_classes = int(num_classes * 0.8)
    train_mask = all_labels < train_classes
    val_mask   = all_labels >= train_classes

    encoder = ProtoNetEncoder()
    model = PrototypicalNetwork(encoder)
    train(model, all_data[train_mask], all_labels[train_mask],
          all_data[val_mask], all_labels[val_mask],
          n_way=5, n_support=5, n_query=15, num_epochs=10)
```

### Code Walkthrough

| Component            | Role                                                            |
|----------------------|-----------------------------------------------------------------|
| `ProtoNetEncoder`    | 4-block CNN, the standard backbone for miniImageNet experiments |
| `compute_prototypes` | Averages support embeddings per class                           |
| `forward`            | Returns negative Euclidean distances as logits                  |
| `EpisodeSampler`     | Builds an N-way K-shot episode each iteration                   |
| `train`              | Episodic training loop with periodic validation                 |

Two implementation notes worth highlighting:

- **`torch.cdist(..., p=2)` returns the Euclidean (not squared) distance.** Negating it as logits is fine for argmax but technically does not match the Bayes-optimal Gaussian-mean derivation. In practice the difference does not matter; if you want exact correspondence, square it.
- **Always relabel the support classes to $0, \ldots, N-1$** inside the sampler so that the cross-entropy targets have the expected shape.

---

## FAQ

**How is few-shot learning different from ordinary transfer learning?**
It is the limit case. Transfer learning assumes you have at least hundreds of target labels, so a fine-tuned head can do most of the work. Few-shot learning has 1-10. That gap is large enough that you need *training-time machinery* -- episodic sampling, metric or meta objectives -- not just a downstream training trick.

**Why do Prototypical networks use the mean as the prototype?**
Under Gaussian class-conditionals with shared isotropic covariance, the class mean is the Bayes-optimal classifier. The mean is also robust enough to be useful even when that assumption fails -- especially for $K \ge 5$.

**MAML or Prototypical Networks -- which should I use?**
Default to Prototypical Networks: simpler, faster, the prototypes are interpretable, and they tend to match or beat MAML on standard image benchmarks. Reach for MAML when (a) the tasks are diverse and look qualitatively different from one another, (b) the data is non-image and you do not have a great pretrained embedding, or (c) you specifically need adaptation that updates the *entire* network rather than just a final classifier.

**How many base classes do I need?**
More is always better for generalization. Standard benchmarks use 64 base classes (miniImageNet) up to 1200+ (Omniglot). With fewer than ~30 base classes you tend to see severe overfitting to the base set itself, and novel-class accuracy collapses.

**Does any of this work for non-image data?**
Yes. Prototypical Networks work for anything with a meaningful embedding -- text (use a transformer encoder), graphs (use a GNN), audio (use a spectrogram CNN). MAML and Reptile are model-agnostic by design. The episodic protocol does not care about modality.

**Why are confidence intervals always reported?**
The per-episode accuracy variance is large -- a single hard episode can swing 10-20 points. Reporting the mean over a few hundred episodes plus a 95% CI is the only way to make numbers comparable across papers.

---

## Summary

Few-shot learning attacks deep learning's biggest practical bottleneck: data scarcity in the long tail.

- **Metric learning** (Siamese, Prototypical, Matching, Relation Networks) learns an embedding space where distance equals dissimilarity. Simple, fast, interpretable. Prototypical Networks are the workhorse default.
- **Meta-learning** (MAML, FOMAML, Reptile) learns an initialization from which a few gradient steps reach the optimum of any new task. More flexible, costlier, less interpretable.
- **Episodic training** is the unifying training paradigm: each iteration is a fresh few-shot task, so training-time difficulty matches test-time difficulty.

Across families, accuracies converge once the backbone is held fixed -- a reminder that backbone capacity and pretraining quality matter at least as much as the few-shot algorithm on top.

Next: [Part 5 -- Knowledge Distillation](/en/transfer-learning-5-knowledge-distillation/), where we compress a large teacher model into a small student that mimics it.

---

## References

1. Snell et al. (2017). *Prototypical Networks for Few-shot Learning.* NeurIPS. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
2. Finn et al. (2017). *Model-Agnostic Meta-Learning (MAML).* ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)
3. Vinyals et al. (2016). *Matching Networks for One Shot Learning.* NeurIPS. [arXiv:1606.04080](https://arxiv.org/abs/1606.04080)
4. Sung et al. (2018). *Learning to Compare: Relation Network for Few-Shot Learning.* CVPR. [arXiv:1711.06025](https://arxiv.org/abs/1711.06025)
5. Nichol et al. (2018). *On First-Order Meta-Learning Algorithms (Reptile).* [arXiv:1803.02999](https://arxiv.org/abs/1803.02999)
6. Koch et al. (2015). *Siamese Neural Networks for One-shot Image Recognition.* ICML Deep Learning Workshop.
7. Chen et al. (2019). *A Closer Look at Few-shot Classification.* ICLR. [arXiv:1904.04232](https://arxiv.org/abs/1904.04232)
8. Wang et al. (2020). *Generalizing from a Few Examples: A Survey on Few-Shot Learning.* ACM Computing Surveys. [arXiv:1904.05046](https://arxiv.org/abs/1904.05046)

---

## Series Navigation

| Part | Topic |
|------|-------|
| [1](/en/transfer-learning-1-fundamentals-and-core-concepts/) | Fundamentals and Core Concepts |
| [2](/en/transfer-learning-2-pre-training-and-fine-tuning/) | Pre-training and Fine-tuning |
| [3](/en/transfer-learning-3-domain-adaptation/) | Domain Adaptation |
| **4** | **Few-Shot Learning** (you are here) |
| [5](/en/transfer-learning-5-knowledge-distillation/) | Knowledge Distillation |
| [6](/en/transfer-learning-6-multi-task-learning/) | Multi-Task Learning |
