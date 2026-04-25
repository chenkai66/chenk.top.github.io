---
title: "Transfer Learning (1): Fundamentals and Core Concepts"
date: 2025-05-01 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Domain Adaptation
  - Machine Learning
categories:
  - Transfer Learning
series:
  name: "Transfer Learning"
  part: 1
  total: 12
lang: en
mathjax: true
description: "A beginner-friendly guide to transfer learning fundamentals: why it works, formal definitions, taxonomy, negative transfer, and a complete feature-transfer implementation with MMD domain adaptation."
disableNunjucks: true
series_order: 1
---

You spent two weeks training an ImageNet classifier on a rack of GPUs. On Monday morning your team lead asks for a chest-X-ray pneumonia model -- and the entire labelled dataset is **two hundred images**. Do you book another two weeks of GPU time and start from scratch?

Of course not. You take what the ImageNet model already knows about edges, textures and shapes, swap out the last layer, and fine-tune on the X-rays. Two hours later you have a model that beats anything you could have trained from random weights with so little data. That is **transfer learning**, and it is the reason most real-world deep-learning projects ship in days instead of months.

This article is the foundation for the rest of the series. We will cover the seven things you need before any of the more specialised topics make sense:

1. **Why** training from scratch is not always an option;
2. The **formal definitions** of domain, task, source and target;
3. The **taxonomy** -- inductive, transductive, unsupervised;
4. **What transfers** at each layer of a deep network;
5. **Negative transfer**: when borrowed knowledge backfires;
6. The **Ben-David bound** and **MMD** -- how to tell whether transfer will work;
7. A **runnable implementation** of feature transfer with MMD alignment.

**Prerequisites:** basic ML vocabulary (loss, gradient descent, classification) and comfort reading Python.

---

## 1. Why We Need Transfer Learning

### The dilemma of training from scratch

A textbook supervised pipeline assumes three things, and none of them holds in real industrial settings:

- **Massive labelled data.** Modern deep networks need tens of thousands to millions of labelled examples to generalise. Few real teams have that.
- **Plentiful compute.** Training a ResNet-50 from random initialisation costs hundreds of GPU-hours; a Transformer from scratch can run into the tens of thousands.
- **No knowledge reuse.** Even strongly related tasks -- chest X-rays vs. chest CT -- start back at zero.

Real medical projects, by contrast, give you a few hundred cases of a rare disease, annotators who must be board-certified physicians, and a deadline measured in weeks. Transfer learning resolves the mismatch with a simple promise: **take a model trained on a large generic dataset, and adapt it cheaply to your data-scarce task.**

The picture below shows what "data-scarce, distribution-shifted" actually looks like. Same two classes, but the target domain has been rotated and shifted -- a faithful cartoon of the kind of covariate shift you encounter when moving from ImageNet photos to medical scans, or from one hospital's scanner to another's.

![Source vs target domain showing distribution shift](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig1_domain_shift.png)

### The intuition

Humans transfer knowledge constantly:

- A cyclist learns to ride a motorbike in an afternoon, not a year.
- A Python programmer reads Java syntax and immediately recognises classes, loops and exceptions.
- Anyone who has seen a house cat will instantly classify a lion as "some kind of feline".

Deep networks have the same property. The early convolutional layers of a vision model learn near-universal primitives -- oriented edges, colour blobs, simple textures. Those primitives are useful for almost any visual task. Higher layers learn more specialised concepts (fur patterns, eye shapes), but even these are recyclable across related domains. Transfer learning is the engineering discipline of exploiting that overlap.

### The core idea, in one sentence

> Given a **source** with abundant labelled data and a **target** with very little, transfer learning moves knowledge from the source so that the target model performs better than it would in isolation.

The only requirement is some correlation between source and target. They do not have to share a feature space, a label set, or even a modality -- but the more they share, the more there is to transfer.

---

## 2. Formal Definitions

To talk about transfer learning precisely, we need to separate two ideas that beginners often confuse: a **domain** is about *the inputs you see*; a **task** is about *what you have to predict*. Pan and Yang's 2010 survey makes this distinction the cornerstone of the whole field.

### Domain

A domain is a pair $\mathcal{D} = \{\mathcal{X}, P(X)\}$ where $\mathcal{X}$ is a feature space and $P(X)$ a marginal distribution over it.

- **Source:** ImageNet RGB photos, $\mathcal{X} = \mathbb{R}^{224 \times 224 \times 3}$, $P_S(X)$ is the distribution of natural-scene pixel statistics.
- **Target:** chest CT slices, same dimensionality but $P_T(X)$ has utterly different intensity histograms, structural priors and noise characteristics.

### Task

A task is a pair $\mathcal{T} = \{\mathcal{Y}, f(\cdot)\}$: a label space and the predictor we want to learn. For supervised learning the task implicitly fixes the conditional $P(Y \mid X)$.

- **Task 1:** ImageNet 1000-way classification, $|\mathcal{Y}| = 1000$.
- **Task 2:** pneumonia binary classification, $|\mathcal{Y}| = 2$.

### Source vs. target

|              | Source                                              | Target                                              |
| ------------ | --------------------------------------------------- | --------------------------------------------------- |
| Domain       | $\mathcal{D}_S$ with task $\mathcal{T}_S$           | $\mathcal{D}_T$ with task $\mathcal{T}_T$           |
| Labels       | abundant                                            | scarce or absent                                    |
| Distribution | $P_S(X), P_S(Y\|X)$                                 | $P_T(X), P_T(Y\|X)$                                 |

Transfer learning explicitly does **not** require source and target to be identical -- that is its whole reason for existing. They may differ in any combination of feature space ($\mathcal{X}_S \neq \mathcal{X}_T$), marginal ($P_S(X) \neq P_T(X)$), label space ($\mathcal{Y}_S \neq \mathcal{Y}_T$), or conditional ($P_S(Y\mid X) \neq P_T(Y\mid X)$). Each combination gives rise to a different sub-problem.

### The formal statement

> Given a source domain $\mathcal{D}_S$ with task $\mathcal{T}_S$ and a target domain $\mathcal{D}_T$ with task $\mathcal{T}_T$, transfer learning aims to improve the target predictive function $f_T(\cdot)$ using knowledge from $\mathcal{D}_S$ and $\mathcal{T}_S$, where $\mathcal{D}_S \neq \mathcal{D}_T$ or $\mathcal{T}_S \neq \mathcal{T}_T$.

Letting $\epsilon_0$ be the target error you would achieve without transfer and $\epsilon_T$ the error with transfer, success means
$$
\epsilon_T < \epsilon_0
$$
or, equivalently, *the same accuracy with fewer target labels*. This second framing is usually the more honest measure of value: transfer learning rarely lifts a saturated model, but it dramatically cuts the labelling bill at every operating point below saturation.

---

## 3. Taxonomy of Transfer Learning

The field looks chaotic until you organise it by **what is missing on the target side**. That gives the three-way split below.

![Transfer learning taxonomy: inductive, transductive, unsupervised](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig2_taxonomy.png)

### Inductive transfer

- Source and target **tasks differ** ($\mathcal{T}_S \neq \mathcal{T}_T$).
- Target has **some labels** -- usually a small set.
- Methods: pretrain-then-finetune, multi-task learning, self-training.
- Canonical example: an ImageNet-pretrained ResNet head-swapped onto a chest-X-ray classifier with a few hundred labelled scans. This is the workflow behind most medical imaging papers.

### Transductive transfer

- Tasks are the **same** but **domains differ** ($\mathcal{T}_S = \mathcal{T}_T$, $\mathcal{D}_S \neq \mathcal{D}_T$).
- Target has **no labels at all**.
- Methods: domain adaptation (feature alignment, adversarial alignment), instance reweighting.
- Canonical example: a semantic segmentation model trained on GTA5 driving footage, deployed on real Cityscapes data without any new labels. Self-driving research lives here.

### Unsupervised transfer

- **Neither side has labels.** What transfers is structure -- representations, clusters, manifolds.
- Methods: self-supervised pretraining (MoCo, SimCLR, MAE), deep clustering.
- Canonical example: Word2Vec or BERT trained on a generic corpus, then used as a feature extractor for any downstream NLP task. This is also where modern foundation-model pipelines start.

In practice these categories blur: a typical foundation-model workflow does **unsupervised** pretraining, **inductive** transfer to a downstream task, and -- if the deployment environment differs from the labelled training set -- **transductive** domain adaptation on top. The taxonomy is a vocabulary, not a strict partition.

---

## 4. What Transfers at Each Layer

A deep network is not one knowledge unit -- it is a stack of representations of increasing specificity. Yosinski et al. (2014) ran a now-classic experiment: take a CNN trained on one half of ImageNet, freeze the first $k$ layers, retrain the rest on the other half, and measure the accuracy gap as you slide $k$. The result, sketched below, is the single most useful empirical fact about transfer learning.

![Layer-wise transferability of CNN features](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig3_layer_transferability.png)

The story has three parts:

1. **Low-level features (`conv1`-`conv3`) are general.** Edges and textures are nearly universal across visual tasks. Freezing these costs you almost nothing -- in fact transferring them often beats training them from scratch on a small target.
2. **High-level features (`conv5` onwards) are specific.** Filter banks tuned to ImageNet object categories do not align with target classes. Freezing them imposes a real penalty -- the orange "frozen" curve drops sharply.
3. **Fine-tuning recovers specificity for free.** The blue "fine-tuned" curve stays flat: as long as you let high-level layers adapt, you keep the low-level priors *and* match the target distribution. This is why "freeze low, fine-tune high" is the default recipe.

This single picture explains most of the practical advice in the rest of the series: *which* layers to freeze, *which* learning rate to use per block, and *why* parameter-efficient methods like LoRA target only the deeper layers.

---

## 5. Negative Transfer

Transfer learning is not free. When the source and target are too different, the inherited weights become a liability rather than an asset. We call this **negative transfer**: the transferred model performs *worse* than a model trained on the target alone.
$$
\epsilon_T > \epsilon_0
$$

The diagram below makes the regime visible. As source-target divergence grows, transfer accuracy crosses below the from-scratch baseline. Past that crossover, you are paying to import bad inductive biases.

![Negative transfer: positive vs negative regions as a function of divergence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig4_negative_transfer.png)

### Why it happens

1. **Divergent input distributions.** Photographic textures are useless for hand-drawn sketches; high-frequency statistics are completely different.
2. **Conflicting tasks.** The optimum for the source loss may sit far from the target optimum, and gradient descent cannot always escape the source basin.
3. **Source overfitting.** The pretrained model may have memorised dataset-specific noise (e.g. the consistent watermark on a corner of every ImageNet photo) that actively misleads target predictions.

### How to avoid it

- **Measure first.** Run MMD or a domain classifier on raw or shallow features. If divergence is huge, transfer less aggressively or change source datasets.
- **Selective transfer.** Freeze the general low-level layers, retrain the specific high-level layers. The layer-transferability curve above is your guide.
- **Regularised fine-tuning.** Add an $L_2$ penalty pulling parameters toward the pretrained values, or use techniques like *L2-SP*. This bounds how far you can drift.
- **Ensemble as a safety net.** When in doubt, average a transfer model and a from-scratch model. The ensemble strictly dominates the worse of the two on most benchmarks.

The crossover in the figure is not theoretical -- it is the operating curve every transfer-learning practitioner is implicitly walking along. Your job is to know which side of it you are on **before** you ship.

---

## 6. Quantifying Transfer Feasibility

Two tools let you predict, before training, whether transfer is likely to help.

### The Ben-David bound

For any hypothesis $h$ in a class $\mathcal{H}$, the target-domain error decomposes as
$$
\epsilon_T(h) \;\leq\; \epsilon_S(h) \;+\; \tfrac{1}{2}\, d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) \;+\; \lambda^{*}.
$$
The three terms are independent levers:

- $\epsilon_S(h)$ -- source error, reducible by **better training**.
- $d_{\mathcal{H}\Delta\mathcal{H}}$ -- divergence between source and target distributions, reducible by **domain adaptation**.
- $\lambda^{*}$ -- the irreducible joint error of the best classifier on both domains. This is fixed by the problem itself: if no single hypothesis can do well on both, no amount of clever alignment will save you.

The bound's practical implication is brutal: if either the divergence or $\lambda^{*}$ is large, **stop and pick a different source domain** rather than burning compute on a doomed adaptation.

### Maximum Mean Discrepancy

To estimate divergence in practice, MMD compares the means of source and target features after mapping into a Reproducing Kernel Hilbert Space:
$$
\mathrm{MMD}(\mathcal{D}_S, \mathcal{D}_T) \;=\; \big\lVert \mathbb{E}[\phi(X_S)] - \mathbb{E}[\phi(X_T)] \big\rVert_{\mathcal{H}}.
$$
With an RBF kernel, this has a closed-form unbiased estimator that you can compute from a single mini-batch and add directly to your loss:
$$
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{task}} \;+\; \lambda \cdot \mathrm{MMD}^{2}.
$$
Minimising the second term aligns the source and target feature distributions, which is exactly what the Ben-David bound asks you to do.

A useful rule of thumb after pretraining: MMD below 0.1 in the embedding space is a strong positive sign; above 0.5 you are firmly in negative-transfer territory.

---

## 7. Putting It Together: The Standard Recipe

### Pretrained backbone + new head

The ninety-percent solution that solves most production problems is also the simplest:

![Pretrained backbone with a new task-specific head](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig5_backbone_new_head.png)

Take the convolutional or Transformer body of a model trained on a large generic dataset, throw away the original classification head, attach a new head sized to your label space, and fine-tune. You decide layer by layer whether to freeze (cheap, conservative) or unfreeze (expensive, expressive). For most tabular budgets the right answer is "freeze the backbone for the first few epochs, then unfreeze with a tiny learning rate".

### Why this works in the low-data regime

The data-efficiency curve is the single most compelling business case for transfer learning. With ten labels, training from scratch is barely better than guessing; transfer already gives you something useful. With one hundred labels, the gap is enormous. With ten thousand labels, the curves converge -- which is exactly *why* transfer is most valuable to teams that cannot afford to label ten thousand examples.

![Data-efficiency: target accuracy vs. number of target labels](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig6_data_efficiency.png)

Read this chart as a contract: every horizontal slice tells you how many labels you save by transferring; every vertical slice tells you how much accuracy you gain. The two views are equivalent and both are routinely worth orders of magnitude.

### The unsupervised case: domain adaptation

When there are zero target labels, you cannot fine-tune in the usual sense. The dominant strategy -- developed in detail in part 3 of this series -- is to learn a **shared encoder** that pulls source and target features into the same subspace, then train a classifier on the (labelled) source side and apply it to the (now-aligned) target.

![Domain adaptation problem setup with a shared encoder](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/01-fundamentals-and-core-concepts/fig7_domain_adaptation.png)

The training objective is exactly the loss we wrote down above: classification on the source plus an MMD penalty pulling source and target embeddings together. The next section shows that this is fewer than 100 lines of PyTorch.

---

## Transfer Learning vs. Related Concepts

Transfer learning shares borders with several adjacent fields. The vocabulary matters:

| Dimension       | Transfer learning                  | Multi-task learning                 |
| --------------- | ---------------------------------- | ----------------------------------- |
| Goal            | optimise target performance        | optimise all tasks simultaneously   |
| Training        | sequential (source then target)    | parallel (jointly)                  |
| Data assumption | domains can differ                 | tasks assumed related               |
| Typical pattern | pretrain-finetune                  | shared encoder, multiple heads      |

| Dimension      | Transfer learning              | Meta-learning                          |
| -------------- | ------------------------------ | -------------------------------------- |
| Goal           | transfer specific knowledge    | learn *how to learn* new tasks         |
| Training data  | one or few source domains      | many diverse tasks                     |
| Adaptation     | usually requires fine-tuning   | fast few-shot updates                  |

| Dimension           | Transfer learning           | Domain generalisation              |
| ------------------- | --------------------------- | ---------------------------------- |
| Test-time access    | target data available       | target domain unknown              |
| Method              | domain adaptation           | learn domain-invariant features    |

These distinctions affect which papers, benchmarks and tools you should reach for.

---

## Complete Implementation: Feature Transfer with MMD

Below is a self-contained example of the workflow this article has been describing. We simulate a domain shift in 2D, then compare three strategies: train from scratch on the target, transfer naively from the source, and properly align with MMD.

```python
"""
Feature Transfer and Domain Adaptation Demo
Method: feature extraction + MMD alignment + fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

np.random.seed(42)
torch.manual_seed(42)


# --- Data generation: simulate domain shift -------------------------------

def generate_source_domain(n_samples=1000):
    """Source domain: two well-separated Gaussian clusters."""
    X0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([-2, -2])
    X1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    return X, y


def generate_target_domain(n_samples=200):
    """Target domain: rotated 45 degrees and shifted (covariate shift)."""
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    X0 = (np.random.randn(n_samples // 2, 2) * 0.6 + np.array([-1, -1])) @ rotation.T
    X1 = (np.random.randn(n_samples // 2, 2) * 0.6 + np.array([ 1,  1])) @ rotation.T
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    return X, y


X_source,        y_source        = generate_source_domain(1000)
X_target_train,  y_target_train  = generate_target_domain(50)   # few labelled
X_target_test,   y_target_test   = generate_target_domain(200)  # held-out

print(f"source: {X_source.shape}  |  target train: {X_target_train.shape}  "
      f"|  target test: {X_target_test.shape}")


# --- Method 1: train from scratch on the few target labels ----------------

clf_scratch = SVC(kernel="rbf", gamma="auto").fit(X_target_train, y_target_train)
acc_scratch = accuracy_score(y_target_test, clf_scratch.predict(X_target_test))
print(f"[from scratch]    accuracy: {acc_scratch:.4f}")


# --- Method 2: train on source, apply directly to target (no adaptation) --

clf_direct = SVC(kernel="rbf", gamma="auto").fit(X_source, y_source)
acc_direct = accuracy_score(y_target_test, clf_direct.predict(X_target_test))
print(f"[direct transfer] accuracy: {acc_direct:.4f}")


# --- Method 3: feature transfer + MMD alignment + fine-tuning -------------

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Classifier(nn.Module):
    def __init__(self, input_dim=16, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def compute_mmd(x_source, x_target, gamma=1.0):
    """RBF-kernel MMD between two batches (biased estimator, fine for SGD)."""
    xx = torch.sum(x_source ** 2, dim=1, keepdim=True)
    yy = torch.sum(x_target ** 2, dim=1, keepdim=True)
    K_ss = torch.exp(-gamma * (xx + xx.t() - 2 * x_source @ x_source.t()))
    K_tt = torch.exp(-gamma * (yy + yy.t() - 2 * x_target @ x_target.t()))
    K_st = torch.exp(-gamma * (xx + yy.t() - 2 * x_source @ x_target.t()))
    n_s, n_t = x_source.size(0), x_target.size(0)
    return K_ss.sum() / n_s ** 2 + K_tt.sum() / n_t ** 2 - 2 * K_st.sum() / (n_s * n_t)


def train_with_mmd(X_source, y_source, X_target_unlabeled,
                   X_target_labeled, y_target_labeled,
                   epochs=100, lambda_mmd=0.5):
    """Two-stage training: source classification + MMD, then target fine-tune."""
    X_s   = torch.FloatTensor(X_source)
    y_s   = torch.LongTensor(y_source.astype(int))
    X_t_u = torch.FloatTensor(X_target_unlabeled)
    X_t_l = torch.FloatTensor(X_target_labeled)
    y_t_l = torch.LongTensor(y_target_labeled.astype(int))

    feat = FeatureExtractor()
    clf  = Classifier()
    optimizer = optim.Adam(list(feat.parameters()) + list(clf.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X_s, y_s), batch_size=32, shuffle=True)

    # Stage 1: source classification + MMD alignment on unlabeled target
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss_cls = criterion(clf(feat(X_batch)), y_batch)
            loss_mmd = compute_mmd(feat(X_batch), feat(X_t_u))
            (loss_cls + lambda_mmd * loss_mmd).backward()
            optimizer.step()

    # Stage 2: light fine-tuning on the few labelled target samples
    for _ in range(50):
        optimizer.zero_grad()
        criterion(clf(feat(X_t_l)), y_t_l).backward()
        optimizer.step()

    return feat, clf


feat, clf_mmd = train_with_mmd(
    X_source, y_source,
    X_target_unlabeled=X_target_test,
    X_target_labeled=X_target_train,
    y_target_labeled=y_target_train,
)

feat.eval(); clf_mmd.eval()
with torch.no_grad():
    preds = torch.argmax(clf_mmd(feat(torch.FloatTensor(X_target_test))), dim=1).numpy()

acc_transfer = accuracy_score(y_target_test, preds)
print(f"[feature + MMD]   accuracy: {acc_transfer:.4f}")
print(f"\nimprovement over from-scratch: "
      f"{(acc_transfer - acc_scratch) / acc_scratch * 100:.1f}%")
```

### What each piece does

| Component                          | Purpose                                                       |
| ---------------------------------- | ------------------------------------------------------------- |
| `generate_source/target_domain`    | Manufactures a controlled covariate shift (rotation + offset) |
| `compute_mmd`                      | RBF-kernel MMD between two embedding batches                  |
| Stage 1 training                   | Classify source while pulling target embeddings closer        |
| Stage 2 fine-tuning                | Light adaptation using the 50 labelled target samples         |

**Knobs worth understanding.** `lambda_mmd=0.5` controls how aggressively the encoder is pulled toward distributional alignment -- too small and you ignore the target, too large and you destroy classification accuracy on the source. The 100/50 epoch split allocates most of the budget to alignment and a small refinement pass to specialisation.

You can swap the synthetic 2D data for any real pair of datasets (Office-31, DomainNet, VisDA) and the structure of the code does not change. That is the point of the exercise.

---

## FAQ

### Is transfer learning always better than training from scratch?

No. It depends on domain relatedness, target-data quantity, and task similarity. The honest rule of thumb: consider transfer whenever target data is below roughly 10% of what you would need to train competitively from scratch. Above that threshold, gains shrink and the engineering overhead may not be worth it.

### How do I pick a good source domain?

Pick something large, diverse, and in the same modality. For vision the default is ImageNet; for NLP, a strong pretrained Transformer (BERT, GPT, Llama). Sanity-check overlap by t-SNE on shallow features and by computing MMD: under 0.1 is a green light, over 0.5 is a red flag.

### Which layers should I freeze?

Start by freezing the bottom 30-50% of the network and fine-tuning the rest. For NLP it is more common to fine-tune *all* layers but with a learning rate one order of magnitude smaller than pretraining. With very little data (under 100 examples), only train the final 1-2 layers; everything else will overfit.

### How do I detect negative transfer?

Always run a from-scratch baseline. If the transfer model underperforms it, or if validation loss climbs during fine-tuning, or if MMD between source and target embeddings stays above 0.5 after pretraining, you have negative transfer. Remedies in order of cost: transfer fewer layers, switch source dataset, or move to adversarial domain adaptation (covered in part 3).

---

## Summary

We covered the seven pieces every transfer-learning project needs:

- **Motivation** -- data scarcity, expensive compute, the universal value of knowledge reuse.
- **Formal definitions** -- domain vs. task, source vs. target, with the four ways they can differ.
- **Taxonomy** -- inductive, transductive, unsupervised; one vocabulary, three problem shapes.
- **Layer-wise transferability** -- general low-level features, specific high-level features; freeze low, fine-tune high.
- **Negative transfer** -- where it comes from, how to detect it, how to bound the damage.
- **Theory** -- the Ben-David decomposition and MMD as a practical divergence estimator.
- **Recipe and code** -- pretrained backbone plus new head, with a runnable MMD-aligned implementation.

Transfer learning is not magic. It is a disciplined exploitation of the structure that real datasets share. Used well, it is one of the highest-leverage techniques in the modern deep-learning toolkit; used carelessly, it silently makes your model worse. The rest of this series is about using it well.

---

## References

1. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10), 1345-1359.
2. Weiss, K., Khoshgoftaar, T. M., & Wang, D. (2016). A survey of transfer learning. *Journal of Big Data*, 3(1), 1-40.
3. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS*.
4. Rosenstein, M. T. et al. (2005). To transfer or not to transfer. *NeurIPS Workshop on Transfer Learning*.
5. Ben-David, S. et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1), 151-175.
6. Gretton, A. et al. (2012). A kernel two-sample test. *JMLR*, 13, 723-773.

---

## Series Navigation

| Part | Topic |
|------|-------|
| **1** | **Fundamentals and Core Concepts** (you are here) |
| [2](/en/transfer-learning-2-pre-training-and-fine-tuning/) | Pre-training and Fine-tuning |
| [3](/en/transfer-learning-3-domain-adaptation/) | Domain Adaptation |
| [4](/en/transfer-learning-4-few-shot-learning/) | Few-Shot Learning |
| [5](/en/transfer-learning-5-knowledge-distillation/) | Knowledge Distillation |
| [6](/en/transfer-learning-6-multi-task-learning/) | Multi-Task Learning |
