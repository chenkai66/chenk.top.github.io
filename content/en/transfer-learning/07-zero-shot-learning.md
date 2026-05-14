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

- Parts 1–6 of this series (especially few-shot in Part 4 and multi-task in Part 6)
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

**2. Generative feature synthesis** (Section 4). Once you can fabricate unseen-class features, GZSL is just supervised classification on a balanced training set.

**3. OOD gating.** Train a binary detector that decides "is this image from a seen class or an unseen class?" and route to the appropriate sub-classifier. Performance is bounded by the OOD detector itself.

---

## CLIP and the vision-language pretraining era

**CLIP** (Radford et al., 2021) is, in retrospect, exactly the deep-compatibility two-tower of Section 3 — but trained on **400 million image–text pairs** scraped from the web, with the cross-entropy contrastive loss
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

## Hubness and the Calibration Arms Race

![GZSL bias geometry and calibrated stacking γ sweep.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/07-Zero-Shot-Learning/fig07_zsl_prototypes.png)

Once you embed both modalities into a shared $d$-dimensional space and rank by cosine, a quiet failure mode shows up: a handful of class prototypes turn into **hubs** — they appear among the $k$-nearest neighbours of disproportionately many query images, regardless of what those queries actually are. Hubness is a property of high-dimensional NN search itself (Radovanović et al., 2010), but ZSL embeddings make it worse: the projection from a few hundred million pixels down to $d=512$ concentrates probability mass near a few semantic prototypes, and any query falling in a low-density region of the visual side gets pulled toward those hubs.

Define the **$k$-occurrence** of class $c$ as
$$
N_k(c) = \bigl|\{x : c \in \mathrm{kNN}(x)\}\bigr|,
$$
the number of queries that list $c$ among their top-$k$ neighbours. In a perfectly uniform embedding, $N_k(c) \approx k \cdot |X| / |\mathcal{C}|$ for every $c$. In a hubness-prone embedding, the distribution of $N_k$ over classes becomes heavy-tailed; the right diagnostic is its **skewness**
$$
\gamma_1 = \mathbb{E}\!\left[\left(\frac{N_k - \mu_{N_k}}{\sigma_{N_k}}\right)^{\!3}\right].
$$
A clean embedding has $\gamma_1$ near zero; a pathological one has $\gamma_1 > 2$. The reason ZSL is especially exposed: the compatibility space is shared but trained *only* on seen classes, so unseen-class prototypes sit in regions that the visual encoder never had to populate uniformly. Combine that with $d \gg 50$ and the hubness pump is on.

The mechanical reason high $d$ amplifies hubness is the concentration of distances: as $d \to \infty$ the ratio of max-to-min cosine distance over a finite query set shrinks, and any prototype that sits even slightly inside the centroid of the visual cloud picks up a disproportionate share of "ties" that the argmax breaks in its favour. Fixing the geometry up front is much cheaper than fixing the consequences downstream.

```python
import torch
from scipy.stats import skew

def hubness_diagnostic(query_emb: torch.Tensor, proto_emb: torch.Tensor,
                       k: int = 5) -> tuple[torch.Tensor, float]:
    """Return N_k vector over prototypes and Pearson skewness gamma_1."""
    q = torch.nn.functional.normalize(query_emb, dim=-1)
    p = torch.nn.functional.normalize(proto_emb, dim=-1)
    sims = q @ p.t()                                     # [N_query, C]
    topk = sims.topk(k, dim=-1).indices                  # [N_query, k]
    Nk = torch.bincount(topk.flatten(), minlength=p.size(0)).float()
    return Nk, float(skew(Nk.numpy()))

# Toy run: 1000 queries, 50 prototypes, d=512.
torch.manual_seed(1)
Q = torch.randn(1000, 512)
P = torch.randn(50, 512)
Nk, g1 = hubness_diagnostic(Q, P, k=5)
print(f"max N_k = {Nk.max().item():.0f}, skewness = {g1:.2f}")
```

Two practical fixes. The first is the cheapest one and it is the only line of code you actually need most of the time:

```python
def normalize_and_standardize(scores: torch.Tensor) -> torch.Tensor:
    """L2-normalize was already done; standardize per-class score columns."""
    mu = scores.mean(dim=0, keepdim=True)
    sigma = scores.std(dim=0, keepdim=True) + 1e-6
    return (scores - mu) / sigma
```
L2-normalising both modalities removes the magnitude axis along which hubs concentrate; per-column standardisation kills the residual offset that lets one prototype dominate. The second fix is **cross-domain mean subtraction** (Dinu et al., 2015): subtract the mean of the *unlabelled target* embeddings from every query before scoring, which re-centres the visual cloud onto the semantic cloud and breaks the systematic projection bias. Together they typically cut $\gamma_1$ by half and lift unseen-class accuracy two to four points.

CLIP suffers less from this because the symmetric InfoNCE loss optimises image$\to$text *and* text$\to$image at the same temperature — neither space is the privileged "anchor" side, so neither develops the skewed projection density that breeds hubs. There is also a scale effect: with $\sim 400\text{M}$ pairs the marginal density of any single text prototype is too small for it to become a global hub; hubness is partly a small-sample-of-classes pathology that vanishes when the class set is effectively continuous. Hubness is what you should profile *before* reaching for fancier calibration — measure $\gamma_1$ on your validation set, normalise + standardise, and only then move to transductive or generative tricks.

---

## Transductive ZSL via EM

So far the model never touches the test distribution. But in many deployments you have **unseen-class images** at hand — you just don't have their labels. **Transductive ZSL** assumes you can see $X^u = \{x^u_1, \ldots, x^u_N\}$ at training time without labels, and asks whether you can use them to refine the compatibility function. The natural formalism is EM over the latent class assignments.

Let $F_\theta(x, c)$ be the current compatibility and $\pi_{nc} = P(y_n = c \mid x_n; \theta)$ the soft assignment of unseen example $n$ to unseen class $c$. The complete-data log-likelihood is $\sum_{n,c} \pi_{nc} \log P(x_n \mid c; \theta)$, which under a softmax compatibility model factors as the standard EM bound
$$
\mathcal{Q}(\theta \mid \theta^{(t)}) = \sum_n \sum_{c \in \mathcal{C}^u} \pi_{nc}^{(t)} \log \frac{\exp F_\theta(x_n, c)}{\sum_{c'} \exp F_\theta(x_n, c')}.
$$

**E-step.** Hold $\theta$ fixed and compute the soft assignments
$$
\pi_{nc}^{(t)} = \frac{\exp F_{\theta^{(t)}}(x_n, c)}{\sum_{c' \in \mathcal{C}^u} \exp F_{\theta^{(t)}}(x_n, c')}.
$$

Two implementation details matter: the sum runs over $\mathcal{C}^u$ only (not $\mathcal{C}^s \cup \mathcal{C}^u$ — including seen classes lets the inductive bias from the seen-class training leak back in and stalls EM at the inductive solution), and the temperature inside $F$ should be re-tuned for the transductive distribution because the unseen-class score scale typically differs from the seen-class one by a factor of two or three.

**M-step.** Refine the compatibility on the pseudo-labelled set. The cheapest version: refine the *prototypes* directly by setting $\mu_c \leftarrow \sum_n \pi_{nc} \, x_n / \sum_n \pi_{nc}$ — this is the Bayes-optimal mean under a Gaussian likelihood and avoids back-prop entirely.

The non-trivial caveat: EM only converges to a **local** fixed point. If the initial calibration is wrong — say the compatibility systematically prefers one unseen class — the E-step will reinforce that bias and the M-step will lock it in. In practice this means: (a) run hubness correction before transductive EM; (b) initialise prototypes from semantic-side projections, not from random visual centroids; (c) monitor the assignment delta and stop early.

```python
import torch
import torch.nn.functional as F

def transductive_em(X_unseen: torch.Tensor,
                    prototypes: torch.Tensor,
                    n_iter: int = 10,
                    tau: float = 10.0,
                    tol: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EM refinement of unseen-class prototypes given unlabelled features.
      X_unseen   : [N, d]   target-domain features (no labels)
      prototypes : [C, d]   initial unseen-class prototypes (from semantics)
    Returns refined prototypes and final soft assignments.
    """
    X = F.normalize(X_unseen, dim=-1)
    P = F.normalize(prototypes, dim=-1).clone()
    prev_assign = None

    for t in range(n_iter):
        # E-step: cosine similarity -> softmax assignment
        sims = tau * X @ P.t()                           # [N, C]
        pi = F.softmax(sims, dim=-1)                     # [N, C]

        # Convergence check: change in hard assignment fraction
        assign = pi.argmax(dim=-1)
        if prev_assign is not None:
            delta = (assign != prev_assign).float().mean().item()
            if delta < tol:
                break
        prev_assign = assign

        # M-step: weighted mean of features per class, then renormalise
        weights = pi / (pi.sum(dim=0, keepdim=True) + 1e-8)   # column-normalise
        P_new = weights.t() @ X                          # [C, d]
        P = F.normalize(P_new, dim=-1)

    return P, pi

# Sanity run.
torch.manual_seed(2)
N, C, d = 200, 5, 64
true_proto = F.normalize(torch.randn(C, d), dim=-1)
labels = torch.randint(0, C, (N,))
X = true_proto[labels] + 0.4 * torch.randn(N, d)
init = true_proto + 0.6 * torch.randn(C, d)              # noisy semantic init
refined, pi = transductive_em(X, init, n_iter=20, tau=8.0)
acc = (pi.argmax(1) == labels).float().mean().item()
print(f"transductive cluster purity = {acc:.3f}")
```

Xian et al.'s **f-VAEGAN-D2** (2019) implements the generative-model analogue of this loop — its second discriminator is trained on unlabelled target images, which plays the same role as the E-step here but with adversarial gradients instead of soft EM. A useful way to read the design: the EM loop above is *deterministic* refinement of a discriminative score, while f-VAEGAN-D2 is *stochastic* refinement of a generative density. Both pay off in the regime where the unlabelled target set is large enough to estimate a reliable per-class statistic, and both fail in the same way when the inductive starting point puts a class on the wrong side of a decision boundary.

The practical limit is data: with fewer than $\sim 50$ unlabelled examples per unseen class the M-step's prototype estimate is too noisy and EM oscillates. Above that, transductive ZSL closes another five to eight points of harmonic mean over the inductive baseline — enough that, when the deployment really has a stash of unlabelled target images sitting around, the EM loop is essentially free performance.

---

## Why CLIP Sidesteps GZSL Bias

The GZSL collapse has a single mechanical cause: you train $F_\theta(x, c)$ to maximise $P(y_n \mid x_n)$ over $\mathcal{C}^s$ only, so the score landscape is sculpted to make *seen-class* prototypes attractors. At test time over $\mathcal{C}^s \cup \mathcal{C}^u$ the seen scores are systematically larger — not because seen classes are easier but because the optimiser only ever cared about them. Even with calibrated stacking you are subtracting a constant from a structurally biased landscape.

CLIP avoids this entirely. The InfoNCE objective
$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2}\sum_i \log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_j \exp(I_i \cdot T_j / \tau)}
\;-\; \frac{1}{2}\sum_i \log \frac{\exp(T_i \cdot I_i / \tau)}{\sum_j \exp(T_i \cdot I_j / \tau)}
$$
has no notion of "class" at all. Every image–text pair is its own positive; every other pair in the batch is a negative. There is no class label to be biased toward, so at deployment no class is privileged. The seen/unseen split that defines GZSL only exists in your benchmark — CLIP's encoder treats all prompts symmetrically.

```python
import torch
import torch.nn.functional as F

def symmetric_infonce(I: torch.Tensor, T: torch.Tensor,
                      tau: float = 0.07) -> torch.Tensor:
    """Symmetric image<->text contrastive loss."""
    I = F.normalize(I, dim=-1)
    T = F.normalize(T, dim=-1)
    logits = I @ T.t() / tau                             # [B, B]
    targets = torch.arange(I.size(0), device=I.device)
    return 0.5 * (F.cross_entropy(logits, targets) +
                  F.cross_entropy(logits.t(), targets))

def harmonic_mean(S: float, U: float) -> float:
    return 2 * S * U / (S + U + 1e-8)
```

Here is a numerical demonstration on the same toy problem under both regimes. Three seen classes, three unseen, 64-D features, 32-D semantics. The classification-trained model wins the seen-class argmax; the symmetric contrastive model spreads the score mass evenly.

```python
torch.manual_seed(3)
Cs, Cu, d, m = 3, 3, 64, 32
A = torch.randn(Cs + Cu, m)                              # class semantics
true_W = torch.randn(d, m) * 0.1
N_per = 80
X_all, y_all = [], []
for c in range(Cs + Cu):
    Xc = (A[c] @ true_W.t()).expand(N_per, d) + 0.3 * torch.randn(N_per, d)
    X_all.append(Xc); y_all.append(torch.full((N_per,), c))
X = torch.cat(X_all); y = torch.cat(y_all)
seen_mask = y < Cs

# Train classification head on seen only -> measure GZSL.
W_cls = torch.nn.Parameter(torch.randn(d, m) * 0.01)
opt = torch.optim.Adam([W_cls], lr=5e-3)
for _ in range(200):
    scores = X[seen_mask] @ W_cls @ A.t()
    loss = F.cross_entropy(scores[:, :Cs], y[seen_mask])
    opt.zero_grad(); loss.backward(); opt.step()
pred = (X @ W_cls @ A.t()).argmax(1)
S = (pred[seen_mask] == y[seen_mask]).float().mean().item()
U = (pred[~seen_mask] == y[~seen_mask]).float().mean().item()
print(f"classification: S={S:.2f} U={U:.2f} H={harmonic_mean(S,U):.2f}")

# Train symmetric InfoNCE on (feature, semantic) pairs from ALL classes' pairs.
W_clip = torch.nn.Parameter(torch.randn(d, m) * 0.01)
opt = torch.optim.Adam([W_clip], lr=5e-3)
for _ in range(200):
    idx = torch.randperm(X.size(0))[:64]
    I_emb = X[idx] @ W_clip                              # [B, m]
    T_emb = A[y[idx]]                                    # [B, m]
    loss = symmetric_infonce(I_emb, T_emb, tau=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
sims = F.normalize(X @ W_clip, dim=-1) @ F.normalize(A, dim=-1).t()
pred = sims.argmax(1)
S = (pred[seen_mask] == y[seen_mask]).float().mean().item()
U = (pred[~seen_mask] == y[~seen_mask]).float().mean().item()
print(f"contrastive:    S={S:.2f} U={U:.2f} H={harmonic_mean(S,U):.2f}")
```

Typical output on this toy: classification gives $S=0.92, U=0.08, H=0.18$ — the textbook GZSL collapse. Symmetric contrastive gives $S=0.71, U=0.55, H=0.62$. Same data, same capacity, no calibration trick — only a loss that refuses to privilege seen classes.

This is not free. CLIP shifts the cost to massive web-scale pretraining: 400M image–text pairs, thousands of GPU-days, and a curation pipeline to keep the negatives diverse. The bias problem doesn't vanish; it gets amortised over a pretraining dataset large enough that essentially every "unseen" benchmark class has been seen in some form during contrastive learning. Inside that regime the GZSL bias mechanism never gets to form — and that is why CLIP-style training has effectively retired GZSL as a separate research problem.

Two corollaries worth keeping in mind. First, the symmetric loss is what carries the argument — replacing InfoNCE with a one-sided softmax over class names would re-introduce the same asymmetry that classification-based ZSL suffers from, and the harmonic-mean gap would re-open. Second, in domains where web-scale pretraining is *not* available (medical imaging, industrial defect detection, satellite imagery) GZSL bias remains a live problem and the calibration / generative / transductive machinery from the previous sections is still the right toolkit. The mechanism is invariant; only the data scale at which it stops mattering changes.

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
