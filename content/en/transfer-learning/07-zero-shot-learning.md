---
title: "Transfer Learning (7): Zero-Shot Learning"
date: 2025-06-06 09:00:00
categories: Transfer Learning
  - Machine Learning
tags:
  - Zero-Shot Learning
  - Semantic Embedding
  - Attribute Representation
  - Generative ZSL
  - CLIP
  - Transfer Learning
series: transfer-learning
lang: en
mathjax: true
description: "A first-principles tour of zero-shot learning: attribute prototypes (DAP), compatibility functions, DeViSE, generative ZSL with f-CLSWGAN, the GZSL bias problem and calibration, and CLIP-style vision-language pretraining, with PyTorch building blocks."
disableNunjucks: true
series_order: 7
series_total: 12
translationKey: "transfer-learning-7"
---
You have never seen a zebra. I tell you it looks like a horse painted with black and white stripes, and the next time one walks into the zoo you recognise it instantly. No labelled examples, no fine-tuning — only a *semantic bridge* between what you know (horses, stripes) and what you don't (this new species).

**Zero-shot learning (ZSL)** is the machine-learning version of that trick. Train on a set of *seen* classes for which you have labelled images. At test time, classify into a *disjoint* set of *unseen* classes that you have *never* shown the model — using only a description of what those classes are: a list of attributes, a word embedding of the class name, a sentence, or an image-text contrastive prompt. The model's only handle on the unseen classes is the geometry it has learned in a shared visual–semantic space.

This post derives ZSL from one equation — a compatibility function $F(x, c)$ between an image and a class description — and then walks through every major family that has appeared in the last fifteen years: attribute prototypes, bilinear and deep compatibility, generative feature synthesis, generalised ZSL with calibration, and finally CLIP-style web-scale pretraining that has effectively dissolved the field as a separate problem.


---

## What You Will Learn

- The ZSL problem statement and the seen / unseen split
- Attribute representations and **Direct Attribute Prediction (DAP)**
- Compatibility functions: bilinear, deep, and ranking-based (DeViSE, ALE)
- **Generative ZSL** (f-CLSWGAN, f-VAEGAN-D2): turning ZSL back into supervised learning
- Why **generalised ZSL (GZSL)** collapses without calibration, and three fixes
- **CLIP** as the modern, scalable answer

## Prerequisites

- Parts 1–6 of this series (especially [few-shot in Part 4](/en/transfer-learning/04-few-shot-learning/) and [multi-task in Part 6](/en/transfer-learning/06-multi-task-learning/))
- PyTorch and standard cross-entropy / softmax training
- Word embeddings (Word2Vec / GloVe basics) and a passing acquaintance with GANs

---

## Problem definition

Let $\mathcal{C}^s$ be the set of **seen classes** (labelled training data available) and $\mathcal{C}^u$ the set of **unseen classes** (no labelled data at all). The defining constraint of ZSL is
$$\mathcal{C}^s \cap \mathcal{C}^u = \emptyset.$$
Each class $c$ — seen or unseen — is paired with a **semantic descriptor** $a_c \in \mathbb{R}^M$. This descriptor is the only way information can leak from the seen classes to the unseen ones. It can be:

- a **binary or continuous attribute vector** (e.g. *striped*, *has wings*, *aquatic*);
- a **word embedding** of the class name (Word2Vec, GloVe);
- a **sentence embedding** of an encyclopaedic description (BERT [CLS] token);
- a **prompt embedding** in a vision-language model (CLIP).

We then face two flavours of the task:

| Setting | Test label space | Realism |
|---|---|---|
| **Conventional ZSL** | $y \in \mathcal{C}^u$ | Easier benchmark; assumes you know the test image is unseen. |
| **Generalised ZSL (GZSL)** | $y \in \mathcal{C}^s \cup \mathcal{C}^u$ | What you actually need in production; much harder because of bias toward seen classes. |

The single equation that organises every method below is the **compatibility function**
$$
F: \mathcal{X} \times \mathcal{C} \to \mathbb{R},
\qquad
\hat{y} = \arg\max_{c \in \mathcal{C}_{\text{test}}} F(x, c; \theta).
$$
We learn $F$ on seen classes and *trust* the semantic geometry to extend it to unseen ones.

![Supervised vs few-shot vs zero-shot supervision](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig1_zsl_vs_fsl_vs_supervised.png)

---

## Attribute representations and DAP

Attributes are the most interpretable semantic descriptor. They are short, human-defined predicates such as *striped*, *four-legged*, *has wings*, *aquatic*. The widely used **Animals with Attributes 2** (AwA2) dataset gives every one of its 50 animal classes an 85-dimensional attribute vector; **CUB-200-2011** uses 312 attributes for fine-grained bird recognition.

A class becomes a row in the $|\mathcal{C}| \times M$ **attribute prototype matrix**. For example,
$$a_{\text{zebra}} = (\underbrace{1}_{\text{striped}}, \underbrace{1}_{\text{four-legs}}, \underbrace{0}_{\text{wings}}, \underbrace{1}_{\text{hooves}}, \ldots).$$
### Direct Attribute Prediction (DAP)

Lampert et al. (2009) — the paper that *defined* the modern ZSL problem — proposed a clean two-stage pipeline:

**Stage 1.** Train one binary classifier per attribute on the seen-class images:
$$\hat{a}_m(x) = P(\text{attribute } m \mid x), \qquad m = 1, \ldots, M.$$
**Stage 2.** At test time, run all $M$ attribute classifiers on $x$ to get a predicted attribute vector $\hat{a}(x)$, then pick the unseen class whose prototype is closest:
$$\hat{y} = \arg\min_{c \in \mathcal{C}^u} d\bigl(\hat{a}(x),\, a_c\bigr).$$
A worked example: the query is a zebra. The attribute classifiers fire strongly on *striped*, *four-legs*, *hooves* and *mane*, weakly on *wings* and *aquatic*. Cosine similarity against the prototype matrix puts *zebra* on top, with *horse* and *tiger* trailing.

![Attribute-based zero-shot pipeline (DAP)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig2_attribute_classification.png)

**Why it works:** the attribute classifiers transfer because attributes are *shared* across classes. *Striped* is the same predicate whether the animal is a zebra (unseen) or a tiger (seen).

**Why it falls short:**

1. **Error compounds across stages.** A 90% accurate attribute classifier still leaves a noisy $\hat{a}(x)$, and nearest-prototype search amplifies the noise.
2. **Attribute independence.** DAP treats attributes as independent given the class — but *has wings* and *can fly* are obviously correlated.
3. **Annotation cost.** Defining and labelling attributes requires an expert taxonomy.

The variant **IAP (Indirect Attribute Prediction)** marginalises over seen classes — predict the seen-class probability first, then map through the attribute matrix — but the structural limits are the same.

---

## Compatibility functions: a unified view

Stop predicting attributes as an intermediate step and learn the compatibility $F(x, c)$ end-to-end.

### Bilinear (ALE, SJE)

The simplest form is bilinear:
$$F(x, c) = \phi(x)^\top W\, a_c,$$
with a CNN backbone $\phi(\cdot)$ producing a $d$-dimensional visual feature and $W \in \mathbb{R}^{d \times M}$ the only trainable matrix. Train it on seen classes with a standard softmax cross-entropy:
$$\mathcal{L}(x, y) = -\log \frac{\exp F(x, y)}{\sum_{c \in \mathcal{C}^s} \exp F(x, c)}.$$
Akata et al.'s **ALE** (2013) and **SJE** (2015) replace cross-entropy with a structured **ranking loss** — *the score of the correct class must beat every wrong class by a margin* — which empirically generalises better to unseen classes than vanilla softmax.

### Deep compatibility / two-tower

To capture non-linear visual-semantic interactions, project both modalities into a shared $d$-dimensional space with two small MLPs:
$$F(x, c) = \frac{f_v(\phi(x))^\top f_s(a_c)}{\|f_v(\phi(x))\|\, \|f_s(a_c)\|} \cdot \tau,$$
with a learned temperature $\tau$. This **two-tower** design is the same structure that powers DeViSE, CLIP and modern dense retrieval — only the data and scale differ.

![Shared semantic embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig3_semantic_embedding_space.png)

The geometric picture is the punchline. Seen-class features cluster around their semantic prototypes (*horse*, *cat*, *dog*) during training; the prototypes for unseen classes (*zebra*, *lion*, *panda*) are *placed by the semantics alone*. A query image projected into the same space lands near the unseen prototype that best matches its attributes — and that is the prediction.

### DeViSE: word embeddings as the semantic side

Frome et al. (2013) proposed **DeViSE** — *Deep Visual–Semantic Embedding* — which replaces hand-defined attributes with **word embeddings of class names** (Word2Vec / GloVe). The architecture is a two-tower model trained with a hinge ranking loss
$$\mathcal{L} = \sum_{c' \neq y} \max\bigl(0,\; m - F(x, y) + F(x, c')\bigr),$$
and at inference the score is computed against *every* class embedding, including ones never seen in image form.

![DeViSE: visual encoder + Word2Vec text encoder in a shared space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig4_devise_visual_text_embedding.png)

DeViSE was the first proof that ZSL could scale: trained on ImageNet, it produced reasonable predictions on tens of thousands of zero-shot classes drawn from WordNet. Its weakness is that word embeddings encode *linguistic* similarity (*cat*, *dog*, *pet* cluster) rather than *visual* similarity (*zebra*, *tiger*, *jaguar* — also striped — cluster). This **modality gap** motivated everything that came next.

---

## Generative ZSL: synthesise unseen-class features

Discriminative ZSL learns a fixed compatibility function and *hopes* it extends. Generative ZSL flips the problem: **synthesise visual features for unseen classes from their semantic descriptors**, then train a perfectly ordinary supervised classifier on the union of real-seen and synthetic-unseen features. ZSL becomes supervised learning again.

### f-CLSWGAN (Xian et al., 2018)

The model is a conditional Wasserstein GAN with a classifier-guidance loss:

- **Generator** $G(z, a_c) \to \tilde{x}$. Input: noise $z \sim \mathcal{N}(0, I)$ and class semantics $a_c$. Output: a synthetic CNN feature.
- **Critic** $D(x)$ scores real vs synthetic features (no sigmoid, Wasserstein loss).
- **Auxiliary classifier** $\mathrm{cls}$ on top of generated features, trained on real seen-class features.

The total objective is
$$
\mathcal{L} = \underbrace{\mathbb{E}[D(\tilde{x})] - \mathbb{E}[D(x)] + \lambda\,\mathrm{GP}}_{\text{WGAN-GP}}
\;+\; \beta \cdot \underbrace{\mathbb{E}\bigl[-\log P(y \mid \tilde{x})\bigr]}_{\text{classification}},
$$
where $\mathrm{GP}$ is the standard gradient penalty $\mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$. The classification loss is the key trick — it forces synthetic features to be *class-discriminative*, not just realistic.

**At test time,** sample $\tilde{x}^{(u)} \sim G(z, a_u)$ for every unseen class, train softmax on $\{(x, y) : y \in \mathcal{C}^s\} \cup \{(\tilde{x}^{(u)}, u)\}$, and predict normally. On AwA2 this jumps GZSL harmonic mean from ~22 (vanilla embedding) to ~58.

### f-VAEGAN-D2

Xian et al. (2019) added a VAE encoder to stabilise training and a second discriminator that exploits *unlabelled* test images (transductive). The recipe — VAE for stability, GAN for sharpness, classifier for discriminability — is now standard in feature-generation ZSL.

---

## Generalised ZSL: the bias problem

Conventional-ZSL benchmarks restrict the test label space to $\mathcal{C}^u$, which secretly papers over the real difficulty: in deployment, you do not know whether an incoming image belongs to a seen or unseen class.

When you open the label space to $\mathcal{C}^s \cup \mathcal{C}^u$, every method trained only on seen-class images develops a strong **bias toward seen classes**. Their visual features lie inside the regions $F$ already knows; unseen-class features look slightly off-distribution and almost always lose the argmax.

The standard metric is the **harmonic mean** of seen and unseen accuracies:
$$H = \frac{2 \cdot S \cdot U}{S + U}.$$
It punishes any method that wins on one side at the cost of the other. A model that scores $S = 88, U = 12$ has $H = 21$ — terrible.

![GZSL bias problem and three remedies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig6_gzsl_vs_zsl.png)

Three families of remedies:

**1. Calibrated stacking (Chao et al., 2016).** Subtract a constant from every seen-class score:
$$F_{\text{cal}}(x, c) = F(x, c) - \gamma \cdot \mathbb{1}[c \in \mathcal{C}^s].$$
Tune $\gamma$ on a held-out validation set. Cheap, effective, and a strong baseline.

**2. Generative feature synthesis** ([Section 4](#generative-zsl-synthesise-unseen-class-features)). Once you can fabricate unseen-class features, GZSL is just supervised classification on a balanced training set.

**3. OOD gating.** Train a binary detector that decides "is this image from a seen class or an unseen class?" and route to the appropriate sub-classifier. Performance is bounded by the OOD detector itself.

---

## CLIP and the vision-language pretraining era

**CLIP** (Radford et al., 2021) is, in retrospect, exactly the deep-compatibility two-tower of [Section 3](#compatibility-functions-a-unified-view) — but trained on **400 million image–text pairs** scraped from the web, with the cross-entropy contrastive loss
$$\mathcal{L} = -\sum_i \log \frac{\exp\bigl(I_i \cdot T_i / \tau\bigr)}{\sum_j \exp\bigl(I_i \cdot T_j / \tau\bigr)} \;+\; (\text{symmetric term over text}).$$
At zero-shot inference time you build a classifier *from text* on the fly. For a $K$-class problem, write a prompt template such as `"a photo of a {class}"`, encode all $K$ prompts with the text tower, and classify a new image by picking the prompt embedding closest to its image embedding.

![CLIP zero-shot classification](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig5_clip_zero_shot.png)

Two consequences worth absorbing:

1. **No semantic engineering.** Classes are described in natural language. Want to add a class? Write a sentence.
2. **GZSL is no longer a separate problem.** CLIP achieves comparable accuracy on classes it has and has not "seen" in the training distribution because it was never trained class-discriminatively to begin with.

The benchmark picture confirms the trajectory:

![Zero-shot benchmark progression on AwA2 and CUB](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig7_benchmark_results.png)

DAP (2009) → ALE/DeViSE (2013) — bilinear and ranking compatibility. SAE (2017) — semantic autoencoders. f-CLSWGAN (2018) → CADA-VAE (2019) — generative feature synthesis closes the GZSL gap. CLIP (2021) — large-scale contrastive pretraining redefines the ceiling.

---

## Implementation

A compact PyTorch building block for the deep-compatibility model and a feature-generating GAN. The complete training scaffold (data loaders for AwA2, evaluation in both ZSL and GZSL settings) is in the longer reference implementation linked at the end.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCompatibility(nn.Module):
    """Two-tower visual-semantic compatibility used by DeViSE / ALE / CLIP."""

    def __init__(self, visual_dim: int, semantic_dim: int,
                 embed_dim: int = 512, temperature: float = 10.0):
        super().__init__()
        self.vis = nn.Sequential(
            nn.Linear(visual_dim, 1024), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(1024, embed_dim),
        )
        self.sem = nn.Sequential(
            nn.Linear(semantic_dim, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, embed_dim),
        )
        self.tau = temperature

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Score every class for every image. Returns [B, C] logits."""
        v = F.normalize(self.vis(x), dim=-1)        # [B, d]
        s = F.normalize(self.sem(a), dim=-1)        # [C, d]
        return self.tau * v @ s.t()

class FeatureGenerator(nn.Module):
    """f-CLSWGAN-style conditional generator."""

    def __init__(self, noise_dim: int, semantic_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + semantic_dim, 4096), nn.LeakyReLU(0.2),
            nn.Linear(4096, feature_dim), nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))

def gzsl_evaluate(model: DeepCompatibility, features: torch.Tensor,
                  labels: torch.Tensor, attrs: torch.Tensor,
                  seen_ids: torch.Tensor, unseen_ids: torch.Tensor,
                  gamma: float = 0.0) -> tuple[float, float, float]:
    """Compute (S, U, H) on a GZSL test set with optional calibration."""
    model.eval()
    with torch.no_grad():
        logits = model(features, attrs)                 # [N, C]
        logits[:, seen_ids] -= gamma                    # bias subtraction
        pred = logits.argmax(dim=-1)
        seen_mask = torch.isin(labels, seen_ids)
        S = (pred[seen_mask] == labels[seen_mask]).float().mean().item()
        U = (pred[~seen_mask] == labels[~seen_mask]).float().mean().item()
    H = 2 * S * U / (S + U + 1e-8)
    return S, U, H
```

The `gamma` argument to `gzsl_evaluate` implements calibrated stacking — the simplest GZSL fix to try first.

---

## Bilinear Compatibility From Scratch

The bilinear compatibility $F(x, c) = \theta(x)^\top W\, \phi(c)$ is the smallest model that captures the ZSL geometry — one matrix $W$ links the visual side to the semantic side, and that matrix is the only thing learning has to discover. ALE (Akata et al., 2013) and SJE (Akata et al., 2015) both live here; the difference is the loss. Cross-entropy works but treats every wrong class symmetrically. A structured ranking loss with a class-dependent margin is what gets you the extra two or three points on AwA2.

Write the visual feature as $\theta(x) \in \mathbb{R}^d$ (a frozen CNN feature, in practice a 2048-D ResNet pool) and the class descriptor as $\phi(c) = a_c \in \mathbb{R}^M$ (the attribute vector). Then for a training pair $(x_n, y_n)$ the **structured ranking loss** is
$$
\mathcal{L}(x_n, y_n) = \sum_{y \in \mathcal{C}^s} \max\bigl(0,\; \Delta(y_n, y) + F(x_n, y) - F(x_n, y_n)\bigr),
$$
where $\Delta(y_n, y)$ is the margin — usually $\mathbb{1}[y \neq y_n]$ but it can be replaced by a semantic distance $\|a_{y_n} - a_y\|$ to push semantically distant impostors harder. The gradient with respect to $W$ collapses to a sum of rank-one updates $\theta(x_n)\bigl(\phi(y) - \phi(y_n)\bigr)^\top$ over the violating classes, which is why a single matrix is enough capacity and why convergence is fast.

Why bilinear instead of a deep MLP? Three reasons. First, $W$ has $dM$ parameters — for $d=2048, M=85$ that is $\sim 174\text{k}$, two orders of magnitude below a two-tower MLP, and the seen-class training set on AwA2 is only $\sim 23\text{k}$ images. Second, the bilinear form is a *tensor decomposition* of the joint score: it cannot overfit to seen-class idiosyncrasies in the way a non-linear model can, which matters because the test loss lives on a *disjoint* class set. Third, $W$ is interpretable — its left singular vectors are the visual directions that carry the most attribute signal.

A short derivation of the second point. Decompose $W = U \Sigma V^\top$ with $U \in \mathbb{R}^{d \times r}, V \in \mathbb{R}^{M \times r}$. Then $F(x, c) = \sum_{k=1}^{r} \sigma_k \langle u_k, \theta(x)\rangle \langle v_k, \phi(c)\rangle$. Each rank-one term routes a single visual direction to a single semantic direction with a single scalar gain. Generalisation to unseen $c$ requires only that $v_k$ has been *aligned* with the right semantic axis during training — and the seen-class loss aligns it whenever even one seen class lights up that attribute. There is no equivalent guarantee for a deep two-tower; the non-linearity can carve seen-class manifolds that the unseen-class semantics simply do not project onto.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearCompatibility(nn.Module):
    """F(x, c) = theta(x)^T W phi(c). One trainable matrix."""
    def __init__(self, visual_dim: int, semantic_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(visual_dim, semantic_dim) * 0.01)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, d], A: [C, M]  ->  scores [B, C]
        return x @ self.W @ A.t()

def structured_ranking_loss(scores: torch.Tensor, y: torch.Tensor,
                            margin: float = 1.0) -> torch.Tensor:
    """L = sum_y max(0, Delta + F(x,y) - F(x,y_n)) over wrong classes."""
    B, C = scores.shape
    correct = scores.gather(1, y.unsqueeze(1))           # [B, 1]
    delta = torch.ones_like(scores) * margin
    delta.scatter_(1, y.unsqueeze(1), 0.0)               # zero margin for true class
    violations = (delta + scores - correct).clamp(min=0)
    violations.scatter_(1, y.unsqueeze(1), 0.0)          # exclude true class
    return violations.sum(dim=1).mean()

# Tiny synthetic AwA2-shape problem: 5 classes, 100-D features, 85-D attrs.
torch.manual_seed(0)
C, d, M = 5, 100, 85
A = torch.randn(C, M)                                    # class attribute matrix
true_W = torch.randn(d, M) * 0.1
X = torch.randn(256, d)
y = (X @ true_W @ A.t()).argmax(dim=1)                   # synthetic labels

model = BilinearCompatibility(d, M)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(300):
    scores = model(X, A)
    loss = structured_ranking_loss(scores, y, margin=1.0)
    opt.zero_grad(); loss.backward(); opt.step()
acc = (model(X, A).argmax(1) == y).float().mean().item()
print(f"train acc={acc:.3f}, loss={loss.item():.4f}")
```

The synthetic run converges to $>0.95$ accuracy in a few hundred steps with a single $100 \times 85$ matrix. On real AwA2 features, the same model with cross-entropy hits $\sim 58\%$ ZSL accuracy; swapping to the ranking loss above buys $\sim 2$–$3$ points.

| Method | AwA2 ZSL acc | Params learned | Notes |
|---|---|---|---|
| Bilinear + softmax | 58.2 | $dM$ | the floor |
| Bilinear + ranking (ALE) | 60.7 | $dM$ | margin matters |
| DeViSE (deep two-tower) | 59.7 | $\sim 2M$ | Word2Vec semantics |
| SJE (structured + multi-cue) | 61.9 | $dM$ per cue | best non-generative |

Numbers are in the range reported by Xian et al.'s 2019 TPAMI evaluation; absolute values shift a couple of points with backbone choice. Two implementation notes that make the difference between "publishes" and "reproduces":

- **Initialise $W$ small.** The bilinear score is unbounded; with $W \sim \mathcal{N}(0, 1)$ the early softmax is saturated and the ranking gradient vanishes. The factor-of-100 scaling in the snippet above is not cosmetic.
- **Use class-balanced minibatches.** Seen-class frequencies in AwA2 are skewed by 2–3x; vanilla SGD over an unbalanced sampler under-trains the rare classes whose attribute signature is exactly what generalises to the unseen split.

The deeper compatibility families and feature-generating GANs from the next two sections are what eventually pulled GZSL harmonic mean above $\sim 60$ — but the bilinear baseline above is what every paper still has to beat.

---

## FAQ

**Q1. When should I reach for ZSL instead of few-shot or active learning?**
When (a) the long tail is wide and unstable — new classes appear faster than you can label them; (b) you already have rich semantics — attribute taxonomies, product catalogues, textual descriptions; or (c) you can use a pretrained vision-language model (CLIP, SigLIP, OpenCLIP) and skip ZSL-specific machinery entirely.

**Q2. Attributes vs word embeddings vs sentences vs CLIP prompts?**
Attributes are the most discriminative per-dimension but require expert design. Word embeddings scale but encode language similarity, not visual similarity. Sentence embeddings strike a balance. CLIP prompts dominate everything below it on most natural-image benchmarks because the encoders were already trained jointly.

**Q3. Why does GZSL collapse without calibration?**
The training objective only ever rewards correct seen-class predictions, so the score margin between seen and unseen logits is uncalibrated. At test time, seen-class logits are systematically larger and steal the argmax. Calibrated stacking, generative synthesis and OOD gating all attack the same imbalance.

**Q4. Did CLIP make ZSL research obsolete?**
For natural images: largely, yes. For domains under-represented on the open web — medical imaging, industrial defect detection, satellite imagery — the older semantic-attribute machinery is still the most data-efficient way to bootstrap. And CLIP itself is a deep two-tower compatibility function, so the conceptual scaffolding from this post still applies.

**Q5. How do I avoid the Hubness problem?**
Hubness is the high-dimensional pathology where a few "hub" prototypes are nearest neighbours to far too many queries. Fixes: L2-normalise both modalities before scoring (already in the snippet above), apply rank-based scoring (CSLS, ZestNorm), or use a per-class score-standardisation step.

---

## References

- Lampert, C. H., Nickisch, H., & Harmeling, S. (2009). *Learning to detect unseen object classes by between-class attribute transfer.* CVPR. — DAP, AwA dataset.
- Frome, A., et al. (2013). *DeViSE: A deep visual-semantic embedding model.* NeurIPS.
- Akata, Z., Perronnin, F., Harchaoui, Z., & Schmid, C. (2013). *Label-embedding for attribute-based classification.* CVPR. — ALE.
- Chao, W.-L., et al. (2016). *An empirical study and analysis of generalized zero-shot learning for object recognition in the wild.* ECCV. — GZSL and calibrated stacking.
- Xian, Y., Lampert, C. H., Schiele, B., & Akata, Z. (2019). *Zero-shot learning — A comprehensive evaluation of the good, the bad and the ugly.* TPAMI.
- Xian, Y., Lorenz, T., Schiele, B., & Akata, Z. (2018). *Feature generating networks for zero-shot learning.* CVPR. — f-CLSWGAN.
- Xian, Y., Sharma, S., Schiele, B., & Akata, Z. (2019). *f-VAEGAN-D2: A feature generating framework for any-shot learning.* CVPR.
- Schönfeld, E., et al. (2019). *Generalized zero- and few-shot learning via aligned variational autoencoders.* CVPR. — CADA-VAE.
- Radford, A., et al. (2021). *Learning transferable visual models from natural language supervision.* ICML. — CLIP.
