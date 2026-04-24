---
title: "Transfer Learning (7): Zero-Shot Learning"
date: 2025-05-09 09:00:00
categories:
  - Transfer Learning
  - Machine Learning
tags:
  - Zero-Shot Learning
  - Semantic Embedding
  - Attribute Representation
  - Generative ZSL
  - CLIP
  - Transfer Learning
series:
  name: "Transfer Learning"
  order: 7
  total: 12
lang: en
mathjax: true
description: "A first-principles tour of zero-shot learning: attribute prototypes (DAP), compatibility functions, DeViSE, generative ZSL with f-CLSWGAN, the GZSL bias problem and calibration, and CLIP-style vision-language pretraining, with PyTorch building blocks."
disableNunjucks: true
---

You have never seen a zebra. I tell you it looks like a horse painted with black and white stripes, and the next time one walks into the zoo you recognise it instantly. No labelled examples, no fine-tuning — only a *semantic bridge* between what you know (horses, stripes) and what you don't (this new species).

**Zero-shot learning (ZSL)** is the machine-learning version of that trick. Train on a set of *seen* classes for which you have labelled images. At test time, classify into a *disjoint* set of *unseen* classes that you have *never* shown the model — using only a description of what those classes are: a list of attributes, a word embedding of the class name, a sentence, or an image-text contrastive prompt. The model's only handle on the unseen classes is the geometry it has learned in a shared visual–semantic space.

This post derives ZSL from one equation — a compatibility function $F(x, c)$ between an image and a class description — and then walks through every major family that has appeared in the last fifteen years: attribute prototypes, bilinear and deep compatibility, generative feature synthesis, generalised ZSL with calibration, and finally CLIP-style web-scale pretraining that has effectively dissolved the field as a separate problem.

## What you will learn

- The ZSL problem statement and the seen / unseen split
- Attribute representations and **Direct Attribute Prediction (DAP)**
- Compatibility functions: bilinear, deep, and ranking-based (DeViSE, ALE)
- **Generative ZSL** (f-CLSWGAN, f-VAEGAN-D2): turning ZSL back into supervised learning
- Why **generalised ZSL (GZSL)** collapses without calibration, and three fixes
- **CLIP** as the modern, scalable answer

## Prerequisites

- Parts 1–6 of this series (especially few-shot in Part 4 and multi-task in Part 6)
- PyTorch and standard cross-entropy / softmax training
- Word embeddings (Word2Vec / GloVe basics) and a passing acquaintance with GANs

---

## 1. Problem definition

Let $\mathcal{C}^s$ be the set of **seen classes** (labelled training data available) and $\mathcal{C}^u$ the set of **unseen classes** (no labelled data at all). The defining constraint of ZSL is

$$
\mathcal{C}^s \cap \mathcal{C}^u = \emptyset.
$$

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

## 2. Attribute representations and DAP

Attributes are the most interpretable semantic descriptor. They are short, human-defined predicates such as *striped*, *four-legged*, *has wings*, *aquatic*. The widely used **Animals with Attributes 2** (AwA2) dataset gives every one of its 50 animal classes an 85-dimensional attribute vector; **CUB-200-2011** uses 312 attributes for fine-grained bird recognition.

A class becomes a row in the $|\mathcal{C}| \times M$ **attribute prototype matrix**. For example,

$$
a_{\text{zebra}} = (\underbrace{1}_{\text{striped}}, \underbrace{1}_{\text{four-legs}}, \underbrace{0}_{\text{wings}}, \underbrace{1}_{\text{hooves}}, \ldots).
$$

### Direct Attribute Prediction (DAP)

Lampert et al. (2009) — the paper that *defined* the modern ZSL problem — proposed a clean two-stage pipeline:

**Stage 1.** Train one binary classifier per attribute on the seen-class images:

$$
\hat{a}_m(x) = P(\text{attribute } m \mid x), \qquad m = 1, \ldots, M.
$$

**Stage 2.** At test time, run all $M$ attribute classifiers on $x$ to get a predicted attribute vector $\hat{a}(x)$, then pick the unseen class whose prototype is closest:

$$
\hat{y} = \arg\min_{c \in \mathcal{C}^u} d\bigl(\hat{a}(x),\, a_c\bigr).
$$

A worked example: the query is a zebra. The attribute classifiers fire strongly on *striped*, *four-legs*, *hooves* and *mane*, weakly on *wings* and *aquatic*. Cosine similarity against the prototype matrix puts *zebra* on top, with *horse* and *tiger* trailing.

![Attribute-based zero-shot pipeline (DAP)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig2_attribute_classification.png)

**Why it works:** the attribute classifiers transfer because attributes are *shared* across classes. *Striped* is the same predicate whether the animal is a zebra (unseen) or a tiger (seen).

**Why it falls short:**

1. **Error compounds across stages.** A 90% accurate attribute classifier still leaves a noisy $\hat{a}(x)$, and nearest-prototype search amplifies the noise.
2. **Attribute independence.** DAP treats attributes as independent given the class — but *has wings* and *can fly* are obviously correlated.
3. **Annotation cost.** Defining and labelling attributes requires an expert taxonomy.

The variant **IAP (Indirect Attribute Prediction)** marginalises over seen classes — predict the seen-class probability first, then map through the attribute matrix — but the structural limits are the same.

---

## 3. Compatibility functions: a unified view

Stop predicting attributes as an intermediate step and learn the compatibility $F(x, c)$ end-to-end.

### Bilinear (ALE, SJE)

The simplest form is bilinear:

$$
F(x, c) = \phi(x)^\top W\, a_c,
$$

with a CNN backbone $\phi(\cdot)$ producing a $d$-dimensional visual feature and $W \in \mathbb{R}^{d \times M}$ the only trainable matrix. Train it on seen classes with a standard softmax cross-entropy:

$$
\mathcal{L}(x, y) = -\log \frac{\exp F(x, y)}{\sum_{c \in \mathcal{C}^s} \exp F(x, c)}.
$$

Akata et al.'s **ALE** (2013) and **SJE** (2015) replace cross-entropy with a structured **ranking loss** — *the score of the correct class must beat every wrong class by a margin* — which empirically generalises better to unseen classes than vanilla softmax.

### Deep compatibility / two-tower

To capture non-linear visual-semantic interactions, project both modalities into a shared $d$-dimensional space with two small MLPs:

$$
F(x, c) = \frac{f_v(\phi(x))^\top f_s(a_c)}{\|f_v(\phi(x))\|\, \|f_s(a_c)\|} \cdot \tau,
$$

with a learned temperature $\tau$. This **two-tower** design is the same structure that powers DeViSE, CLIP and modern dense retrieval — only the data and scale differ.

![Shared semantic embedding space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig3_semantic_embedding_space.png)

The geometric picture is the punchline. Seen-class features cluster around their semantic prototypes (*horse*, *cat*, *dog*) during training; the prototypes for unseen classes (*zebra*, *lion*, *panda*) are *placed by the semantics alone*. A query image projected into the same space lands near the unseen prototype that best matches its attributes — and that is the prediction.

### DeViSE: word embeddings as the semantic side

Frome et al. (2013) proposed **DeViSE** — *Deep Visual–Semantic Embedding* — which replaces hand-defined attributes with **word embeddings of class names** (Word2Vec / GloVe). The architecture is a two-tower model trained with a hinge ranking loss

$$
\mathcal{L} = \sum_{c' \neq y} \max\bigl(0,\; m - F(x, y) + F(x, c')\bigr),
$$

and at inference the score is computed against *every* class embedding, including ones never seen in image form.

![DeViSE: visual encoder + Word2Vec text encoder in a shared space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig4_devise_visual_text_embedding.png)

DeViSE was the first proof that ZSL could scale: trained on ImageNet, it produced reasonable predictions on tens of thousands of zero-shot classes drawn from WordNet. Its weakness is that word embeddings encode *linguistic* similarity (*cat*, *dog*, *pet* cluster) rather than *visual* similarity (*zebra*, *tiger*, *jaguar* — also striped — cluster). This **modality gap** motivated everything that came next.

---

## 4. Generative ZSL: synthesise unseen-class features

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

## 5. Generalised ZSL: the bias problem

Conventional-ZSL benchmarks restrict the test label space to $\mathcal{C}^u$, which secretly papers over the real difficulty: in deployment, you do not know whether an incoming image belongs to a seen or unseen class.

When you open the label space to $\mathcal{C}^s \cup \mathcal{C}^u$, every method trained only on seen-class images develops a strong **bias toward seen classes**. Their visual features lie inside the regions $F$ already knows; unseen-class features look slightly off-distribution and almost always lose the argmax.

The standard metric is the **harmonic mean** of seen and unseen accuracies:

$$
H = \frac{2 \cdot S \cdot U}{S + U}.
$$

It punishes any method that wins on one side at the cost of the other. A model that scores $S = 88, U = 12$ has $H = 21$ — terrible.

![GZSL bias problem and three remedies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig6_gzsl_vs_zsl.png)

Three families of remedies:

**1. Calibrated stacking (Chao et al., 2016).** Subtract a constant from every seen-class score:

$$
F_{\text{cal}}(x, c) = F(x, c) - \gamma \cdot \mathbb{1}[c \in \mathcal{C}^s].
$$

Tune $\gamma$ on a held-out validation set. Cheap, effective, and a strong baseline.

**2. Generative feature synthesis** (Section 4). Once you can fabricate unseen-class features, GZSL is just supervised classification on a balanced training set.

**3. OOD gating.** Train a binary detector that decides "is this image from a seen class or an unseen class?" and route to the appropriate sub-classifier. Performance is bounded by the OOD detector itself.

---

## 6. CLIP and the vision-language pretraining era

**CLIP** (Radford et al., 2021) is, in retrospect, exactly the deep-compatibility two-tower of Section 3 — but trained on **400 million image–text pairs** scraped from the web, with the cross-entropy contrastive loss

$$
\mathcal{L} = -\sum_i \log \frac{\exp\bigl(I_i \cdot T_i / \tau\bigr)}{\sum_j \exp\bigl(I_i \cdot T_j / \tau\bigr)} \;+\; (\text{symmetric term over text}).
$$

At zero-shot inference time you build a classifier *from text* on the fly. For a $K$-class problem, write a prompt template such as `"a photo of a {class}"`, encode all $K$ prompts with the text tower, and classify a new image by picking the prompt embedding closest to its image embedding.

![CLIP zero-shot classification](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig5_clip_zero_shot.png)

Two consequences worth absorbing:

1. **No semantic engineering.** Classes are described in natural language. Want to add a class? Write a sentence.
2. **GZSL is no longer a separate problem.** CLIP achieves comparable accuracy on classes it has and has not "seen" in the training distribution because it was never trained class-discriminatively to begin with.

The benchmark picture confirms the trajectory:

![Zero-shot benchmark progression on AwA2 and CUB](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-zero-shot-learning/fig7_benchmark_results.png)

DAP (2009) → ALE/DeViSE (2013) — bilinear and ranking compatibility. SAE (2017) — semantic autoencoders. f-CLSWGAN (2018) → CADA-VAE (2019) — generative feature synthesis closes the GZSL gap. CLIP (2021) — large-scale contrastive pretraining redefines the ceiling.

---

## 7. Implementation

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

## 8. Q&A

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

---

## Series Navigation

- Previous: [Part 6 — Multi-Task Learning](/en/transfer-learning-6-multi-task-learning/)
- Next: [Part 8 — Multimodal Transfer](/en/transfer-learning-8-multimodal-transfer/)
- [View all 12 parts in this series](/tags/Transfer-Learning/)
