---
title: "Transfer Learning (8): Multimodal Transfer"
date: 2025-05-13 09:00:00
categories:
  - Transfer Learning
  - Machine Learning
tags:
  - Multimodal Learning
  - CLIP
  - Contrastive Learning
  - Vision-Language Models
  - Transfer Learning
series:
  name: "Transfer Learning"
  order: 8
  total: 12
lang: en
mathjax: true
description: "Derive contrastive learning (InfoNCE), CLIP's vision-language pretraining, BLIP's Q-Former bridge to LLMs, cross-modal alignment, and multimodal fusion strategies. Includes a from-scratch CLIP implementation in PyTorch."
disableNunjucks: true
---

How can a model classify an image of a Burmese cat correctly without ever having seen a label "Burmese cat"? Traditional supervised learning needs millions of labeled examples per class. CLIP, released by OpenAI in 2021, sidesteps that constraint entirely: it learns to put images and natural-language descriptions into the same vector space, and then "classification" reduces to picking which sentence — out of any candidate sentences you write down — sits closest to the image.

The trick is not architecture. The trick is supervision. CLIP scraped 400 million (image, alt-text) pairs from the web and trained a contrastive objective: for every image, its true caption should be more similar than the captions paired with the other images in the batch. That single constraint, applied at scale, is enough to align two modalities so well that downstream tasks — zero-shot classification, retrieval, captioning prompting — fall out almost for free.

This post derives the math behind that alignment, walks through CLIP and its successor BLIP-2, compares the three families of fusion strategies, and ends with a from-scratch CLIP implementation in PyTorch.

## What you will learn

- The InfoNCE loss as mutual-information maximization, and the role of temperature $\tau$
- CLIP's dual-encoder design and zero-shot classification protocol
- BLIP / BLIP-2: the Q-Former bridge that connects a frozen ViT to a frozen LLM
- Cross-modal retrieval ($R@K$), captioning, VQA, and visual grounding
- Three fusion strategies: early, late, and cross-attention — when to use which
- A 100-line CLIP implementation that you can train

## Prerequisites

- Neural network training in PyTorch
- Cosine similarity, softmax, cross-entropy
- Transfer learning fundamentals (Parts 1–6)

---

## 1. CLIP: a dual-encoder vision-language model

CLIP has only two moving parts: an image encoder $f_v$ (ViT or ResNet) and a text encoder $f_t$ (Transformer). Both project to a shared $d$-dimensional space and L2-normalize, so similarity is just a dot product on the unit hypersphere.

![CLIP dual-encoder architecture: image encoder + text encoder share an L2-normalised embedding space, trained with symmetric InfoNCE.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig1_clip_architecture.png)

Two design choices carry almost all of CLIP's power:

1. **No classification head.** Output is an embedding vector, not a logit over a fixed label set. The model never commits to "1000 ImageNet classes", so it is free to be applied to any concept that can be written as text.
2. **Symmetric contrastive loss.** Image-to-text and text-to-image are trained simultaneously. The encoder doesn't learn an asymmetric skill (e.g. "describe this image"); it learns a *bidirectional* alignment, which is exactly what zero-shot classification and retrieval both need.

### Zero-shot classification protocol

Given an image and $K$ candidate classes:

1. Wrap each class name in a prompt template: `"a photo of a {class}"`.
2. Encode the image once and all $K$ prompts once.
3. Pick the class whose text embedding has the highest cosine similarity to the image embedding.

That's it. No gradient descent on the target task, no labeled examples — just $K$ class names. CLIP achieves 76% top-1 on ImageNet this way, comparable to a fully supervised ResNet-50 trained on ImageNet's 1.28M labels.

| Aspect | Supervised classifier | CLIP zero-shot |
|---|---|---|
| Labels needed | One per training image | None at deployment |
| Output space | Fixed $K$ classes | Any text |
| New class cost | Retrain or fine-tune | Add a new prompt |
| Web-scale data | Manually curated | Naturally exists |

---

## 2. Contrastive learning: the math behind alignment

For a batch of $B$ image-text pairs $\{(\mathbf{v}_i, \mathbf{t}_i)\}$, all $B \times B$ pairwise similarities form a matrix. The diagonal entries are positives (true pairs); the off-diagonal entries are negatives (mismatched pairs).

![Contrastive image-text alignment: diagonal of the batch similarity matrix is pulled up, off-diagonal is pushed down.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig2_contrastive_alignment.png)

The image-to-text InfoNCE loss is a row-wise softmax cross-entropy over this matrix, with the diagonal as the target:

$$
\mathcal{L}_{i \to t} \;=\; -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{v}_i^\top \mathbf{t}_i / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{v}_i^\top \mathbf{t}_j / \tau)}
$$

The text-to-image loss $\mathcal{L}_{t \to i}$ is the column-wise version. CLIP minimizes the symmetric average $\mathcal{L} = \tfrac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$.

### Why it works: a mutual-information lower bound

InfoNCE is not just a heuristic; it is a tractable lower bound on the mutual information $I(V; T)$ between the two modalities (Oord et al., 2018):

$$
I(V; T) \;\geq\; \log B \;-\; \mathcal{L}_{\text{InfoNCE}}
$$

So minimizing the loss directly *maximizes* a lower bound on how much knowing the image tells you about the text (and vice versa). The bound tightens as $B$ grows — which is exactly why CLIP uses an enormous batch size of 32,768.

### Temperature $\tau$: focus vs. spread

The temperature controls the softmax sharpness:

- **Small $\tau$ (≈0.01):** distribution is peaked, gradients concentrate on the hardest negative. Risks overfitting to noise in web data.
- **Large $\tau$ (≈1.0):** distribution is flat, all negatives contribute equally. Slow, weak signal.
- **CLIP's choice:** $\tau = 0.07$, and crucially it is *learned* (parameterized as $\log(1/\tau)$ to keep it positive), so the model can settle on its own sharpness.

### The role of batch size

Each anchor image sees $B - 1$ negatives in its batch. A bigger batch means more negatives, a tighter MI bound, and stronger learning signal — until you run out of GPU memory.

| Batch size | Notes |
|---|---|
| 256 | Typical academic baseline; works but slow to converge |
| 4,096 | MoCo-style queues compensate at this scale |
| 32,768 | CLIP's setting; required massive distributed training |

When you can't increase the batch directly, two tricks help: **MoCo-style memory banks** (keep a queue of recent negatives) and **gradient accumulation** (only feasible for the loss, since the softmax denominator must see all negatives at once).

---

## 3. BLIP and BLIP-2: bridging vision encoders to LLMs

CLIP gives you a great *embedding*, but it cannot generate text. To unlock captioning, VQA, and instruction-following on images, you need a generative head. BLIP-2 (Li et al., 2023) provides a particularly elegant recipe.

![BLIP-2 architecture: a small Q-Former bridges a frozen image encoder and a frozen LLM, trained with three contrastive/matching/generative objectives.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig3_blip_architecture.png)

The core idea: keep the expensive parts (a pretrained ViT and a pretrained LLM) **frozen**, and train only a small **Q-Former** (≈100M params) in between. The Q-Former is a transformer with a small set of *learned query vectors* that cross-attend to image features and produce a fixed-length "soft visual prompt" the LLM can consume.

### Two-stage training

**Stage 1** trains the Q-Former with three complementary losses:

- **ITC** (Image-Text Contrastive): the CLIP-style alignment loss, so the queries learn to capture text-relevant image content.
- **ITM** (Image-Text Matching): a binary classifier on (image, text) pairs, including hard negatives. Forces fine-grained image-text matching that contrastive loss misses.
- **ITG** (Image-grounded Text Generation): teach the Q-Former to extract information sufficient to *generate* the caption (autoregressive language modelling conditioned on visual queries).

**Stage 2** plugs the Q-Former's output into the frozen LLM's input embedding space and fine-tunes only the projection layer with a generative loss. Because the LLM already knows language, learning to "read" visual prompts is cheap.

### Why this matters for transfer learning

BLIP-2 is the template for almost every modern vision-language model (LLaVA, MiniGPT-4, Qwen-VL, GPT-4V): freeze the heavy encoders, train a thin connector. Transfer cost drops by 1–2 orders of magnitude compared to training a multimodal model end-to-end.

---

## 4. Cross-modal retrieval

A shared embedding space buys you retrieval *for free*. Encode the query in one modality, encode a database in the other, return the top-$K$ nearest neighbours by cosine similarity.

![Cross-modal retrieval: image-to-text and text-to-image both reduce to nearest-neighbour search in the shared embedding space.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig4_cross_modal_retrieval.png)

The standard metric is **Recall@K** — the fraction of queries for which the true match appears in the top $K$ results. Modern VLMs report $R@1$, $R@5$, $R@10$ on benchmarks like MS-COCO (5K test images, 25K captions) and Flickr30K.

Three engineering wins follow from the dual-encoder design:

1. **Database embeddings are precomputed.** At query time you only encode the query, then do an ANN search (FAISS / ScaNN). This scales to billions of items.
2. **Modality-symmetric.** The same index serves image-to-text and text-to-image queries.
3. **Composable.** You can mix: encode an image *and* a text refinement ("…but in winter"), average them, and retrieve.

---

## 5. Cross-modal alignment: how tightly should we couple?

The *granularity* of alignment is a design knob. Three regimes are common:

- **Global alignment (CLIP, ALIGN).** One vector per image, one vector per sentence. Fast and scalable; misses fine-grained spatial reasoning.
- **Region alignment (OSCAR).** Detect object regions in the image, align each region to noun phrases in the caption. Object tags act as anchor concepts. Better for VQA where the question targets a specific object.
- **Dense alignment (GLIP, GroundingDINO).** Pixel-level or token-level correspondence. Required for visual grounding ("which woman in the photo?") and open-vocabulary detection.

Joint multimodal embeddings cluster by *semantic concept*, not by modality:

![t-SNE of joint multimodal embeddings: images and captions of the same concept land in the same cluster.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig5_embedding_tsne.png)

That clustering structure is what makes zero-shot transfer work: a new image of "a beach" lands near the *concept* of beach, regardless of which captions the model has seen before.

---

## 6. Downstream tasks

Once you have a good multimodal encoder, a small head — or sometimes no head at all — unlocks a wide task surface.

![Vision-language downstream tasks: VQA, Captioning, Retrieval, Visual Grounding.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig6_vl_tasks.png)

| Task | Setup | Model family |
|---|---|---|
| Zero-shot classification | Cosine similarity vs. prompted class names | CLIP, ALIGN |
| Image-text retrieval | ANN over precomputed embeddings; $R@K$ | CLIP, ALIGN, BLIP |
| Image captioning | Image → autoregressive text decoder | BLIP-2, LLaVA |
| Visual question answering | (Image, question) → answer | BLIP-2, LLaVA, Flamingo |
| Visual grounding | (Image, expression) → bounding box | GLIP, GroundingDINO |
| Open-vocabulary detection | Image + class prompts → boxes | OWL-ViT, GLIP |

---

## 7. Fusion strategies

Whenever you mix two modalities, you have to decide *where* to fuse them. Three patterns dominate.

![Three fusion strategies for multimodal models: early (raw concat), late (independent encoders + combine), cross-attention (deep interaction).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig7_fusion_strategies.png)

**Early fusion.** Concatenate raw features and feed a single encoder: $\mathbf{h} = f([\mathbf{v}; \mathbf{t}])$. Simple but throws away pretrained unimodal encoders. Rare in modern systems.

**Late fusion.** Encode each modality independently, then combine: $\mathbf{h} = g(f_v(\mathbf{v}), f_t(\mathbf{t}))$. This is CLIP. Modular, scalable, retrieval-friendly, but interaction between modalities is shallow — the encoders don't "see" each other.

**Cross-attention (deep) fusion.** Encoders interleave cross-attention layers where queries from one modality attend to keys/values from the other:

$$
\text{CrossAttn}(\mathbf{V}, \mathbf{T}) = \text{softmax}\!\left(\frac{(\mathbf{V}\mathbf{W}_Q)(\mathbf{T}\mathbf{W}_K)^\top}{\sqrt{d}}\right) \mathbf{T}\mathbf{W}_V
$$

Used in ViLBERT, LXMERT, BLIP. Richer interaction, better on tasks needing fine-grained reasoning (VQA, grounding); slower at retrieval because every (image, text) pair must be re-scored.

| Strategy | Pretrained encoders | Interaction depth | Retrieval cost |
|---|---|---|---|
| Early | No | High but unprincipled | Low |
| Late | Yes | Shallow | $O(N + M)$ encodings, then ANN |
| Cross-attention | Yes | Deep | $O(N \cdot M)$ — re-score every pair |

Practical rule: use **late fusion** when retrieval is on the critical path (search, recommendation), **cross-attention** when reasoning quality matters more than latency (VQA, captioning).

---

## 8. Implementation: a minimal CLIP

This is a complete, runnable contrastive learner. The image encoder is intentionally a stub (a small MLP over fake 2048-d features) so the focus stays on the contrastive machinery; swap it for a real ViT in production.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Stand-in for a ViT/ResNet feature extractor + projection head."""

    def __init__(self, embed_dim: int = 512, in_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)  # unit hypersphere


class TextEncoder(nn.Module):
    """Small Transformer text encoder; uses the [CLS] position as the summary."""

    def __init__(self, vocab_size: int = 10_000, embed_dim: int = 512,
                 max_len: int = 77, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(max_len, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=4 * embed_dim, batch_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens) + self.pos[: tokens.size(1)]
        x = self.tf(x)
        return F.normalize(self.proj(x[:, 0]), dim=-1)  # CLS pooling


class CLIP(nn.Module):
    def __init__(self, embed_dim: int = 512, vocab_size: int = 10_000):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # Learnable temperature, parameterised in log-space to stay positive.
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        v = self.image_encoder(images)            # (B, d)
        t = self.text_encoder(tokens)             # (B, d)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_i2t = scale * v @ t.t()            # (B, B)
        return logits_i2t, logits_i2t.t()


def contrastive_loss(logits_i2t: torch.Tensor,
                     logits_t2i: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: diagonal indices are the positive targets."""
    B = logits_i2t.size(0)
    target = torch.arange(B, device=logits_i2t.device)
    return 0.5 * (F.cross_entropy(logits_i2t, target)
                  + F.cross_entropy(logits_t2i, target))


@torch.no_grad()
def zero_shot_classify(model: CLIP, image: torch.Tensor,
                       class_token_lists: list[torch.Tensor]) -> torch.Tensor:
    """Classify `image` into one of K classes, given tokenised class prompts."""
    model.eval()
    v = model.image_encoder(image.unsqueeze(0))                      # (1, d)
    t = torch.stack([model.text_encoder(c.unsqueeze(0)).squeeze(0)
                     for c in class_token_lists])                    # (K, d)
    return (model.logit_scale.exp() * v @ t.t()).softmax(-1).squeeze(0)
```

Three details that newcomers often get wrong:

1. **Always L2-normalize before the dot product.** Otherwise the loss collapses to "make the embedding norm large".
2. **Clamp `logit_scale.exp()`.** Empirically, capping it around 100 stabilises late-stage training when the model has learned a near-perfect alignment.
3. **The cross-entropy targets are `arange(B)`.** The "label" of pair $i$ is just its row/column index — this is what implements "pair $i$ matches itself".

---

## 9. Q&A

**Q1: Where does CLIP's zero-shot ability really come from?** Three things stacked: (i) 400M web pairs cover an enormous concept distribution; (ii) natural language is a far richer label space than discrete classes — you supervise on a *description*, not a category; (iii) contrastive learning aligns the modalities so that any text becomes a usable "prototype" for classification.

**Q2: Why is batch size so important?** Each example sees $B-1$ negatives, and the InfoNCE bound on mutual information tightens as $B$ grows. Below $B \approx 1{,}024$, results degrade noticeably; CLIP uses 32,768. If you can't go big, use a MoCo-style queue.

**Q3: What does CLIP fail at?** Counting ("how many cats?"), spatial relations ("the cat *left of* the lamp"), fine-grained categories (dog breeds), text *inside* images, and abstract concepts. Contrastive learning on noisy web text rewards coarse matching.

**Q4: How should I fine-tune CLIP on my own data?** Three tiers: (a) **freeze + linear probe** for ≤10K labelled examples; (b) **LoRA on attention layers** for 10K–100K; (c) **full fine-tune at low LR** for ≥100K, but freeze the temperature and use the contrastive loss, not cross-entropy on labels.

**Q5: When is cross-attention worth the cost?** When you need *interaction* — VQA, grounding, multi-step reasoning. For pure retrieval at scale, late fusion (CLIP-style) wins because you can precompute and ANN-search.

**Q6: Do I have to scrape 400M pairs to train a useful VL model?** No. The BLIP-2 recipe — freeze a pretrained vision encoder *and* a pretrained LLM, train only a small connector — works with low-millions of pairs and is what most modern open-source VLMs do.

---

## References

- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- Jia, C., et al. (2021). Scaling up visual and vision-language representation learning with noisy text supervision (ALIGN). *ICML*.
- Li, J., et al. (2022). BLIP: Bootstrapping language-image pre-training. *ICML*.
- Li, J., et al. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML*.
- Li, X., et al. (2020). Oscar: Object-semantics aligned pre-training. *ECCV*.
- Lu, J., et al. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic representations. *NeurIPS*.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding (InfoNCE). *arXiv:1807.03748*.

---

## Series Navigation

- Previous: [Part 7 -- Zero-Shot Learning](/en/transfer-learning-7-zero-shot-learning/)
- Next: [Part 9 -- Parameter-Efficient Fine-Tuning](/en/transfer-learning-9-parameter-efficient-fine-tuning/)
- [View all 12 parts in this series](/tags/Transfer-Learning/)
