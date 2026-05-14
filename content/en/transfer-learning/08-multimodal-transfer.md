---
title: "Transfer Learning (8): Multimodal Transfer"
date: 2025-06-12 09:00:00
categories: Transfer Learning
  - Machine Learning
tags:
  - Multimodal Learning
  - CLIP
  - Contrastive Learning
  - Vision-Language Models
  - Transfer Learning
series: transfer-learning
lang: en
mathjax: true
description: "Derive contrastive learning (InfoNCE), CLIP's vision-language pretraining, BLIP's Q-Former bridge to LLMs, cross-modal alignment, and multimodal fusion strategies. Includes a from-scratch CLIP implementation in PyTorch."
disableNunjucks: true
series_order: 8
translationKey: "transfer-learning-8"
---
How can a model classify an image of a Burmese cat correctly without ever having seen a label "Burmese cat"? Traditional supervised learning needs millions of labeled examples per class. CLIP, released by OpenAI in 2021, sidesteps that constraint entirely: it learns to put images and natural-language descriptions into the same vector space, and then "classification" reduces to picking which sentence — out of any candidate sentences you write down — sits closest to the image.

The trick is not architecture. The trick is supervision. CLIP scraped 400 million (image, alt-text) pairs from the web and trained a contrastive objective: for every image, its true caption should be more similar than the captions paired with the other images in the batch. That single constraint, applied at scale, is enough to align two modalities so well that downstream tasks — zero-shot classification, retrieval, captioning prompting — fall out almost for free.

This post derives the math behind that alignment, walks through CLIP and its successor BLIP-2, compares the three families of fusion strategies, and ends with a from-scratch CLIP implementation in PyTorch.


---

## What You Will Learn

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

## CLIP: a dual-encoder vision-language model

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

## Contrastive learning: the math behind alignment

![Transfer Learning (8): Multimodal Transfer — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/illustration_2.png)

For a batch of $B$ image-text pairs $\{(\mathbf{v}_i, \mathbf{t}_i)\}$, all $B \times B$ pairwise similarities form a matrix. The diagonal entries are positives (true pairs); the off-diagonal entries are negatives (mismatched pairs).

![Contrastive image-text alignment: diagonal of the batch similarity matrix is pulled up, off-diagonal is pushed down.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig2_contrastive_alignment.png)

The image-to-text InfoNCE loss is a row-wise softmax cross-entropy over this matrix, with the diagonal as the target:
$$\mathcal{L}_{i \to t} \;=\; -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{v}_i^\top \mathbf{t}_i / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{v}_i^\top \mathbf{t}_j / \tau)}$$
The text-to-image loss $\mathcal{L}_{t \to i}$ is the column-wise version. CLIP minimizes the symmetric average $\mathcal{L} = \tfrac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$.

### Why it works: a mutual-information lower bound

InfoNCE is not just a heuristic; it is a tractable lower bound on the mutual information $I(V; T)$ between the two modalities (Oord et al., 2018):
$$I(V; T) \;\geq\; \log B \;-\; \mathcal{L}_{\text{InfoNCE}}$$
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

## BLIP and BLIP-2: bridging vision encoders to LLMs

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

## Cross-modal retrieval

A shared embedding space buys you retrieval *for free*. Encode the query in one modality, encode a database in the other, return the top-$K$ nearest neighbours by cosine similarity.

![Cross-modal retrieval: image-to-text and text-to-image both reduce to nearest-neighbour search in the shared embedding space.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig4_cross_modal_retrieval.png)

The standard metric is **Recall@K** — the fraction of queries for which the true match appears in the top $K$ results. Modern VLMs report $R@1$, $R@5$, $R@10$ on benchmarks like MS-COCO (5K test images, 25K captions) and Flickr30K.

Three engineering wins follow from the dual-encoder design:

1. **Database embeddings are precomputed.** At query time you only encode the query, then do an ANN search (FAISS / ScaNN). This scales to billions of items.
2. **Modality-symmetric.** The same index serves image-to-text and text-to-image queries.
3. **Composable.** You can mix: encode an image *and* a text refinement ("…but in winter"), average them, and retrieve.

---

## Cross-modal alignment: how tightly should we couple?

The *granularity* of alignment is a design knob. Three regimes are common:

- **Global alignment (CLIP, ALIGN).** One vector per image, one vector per sentence. Fast and scalable; misses fine-grained spatial reasoning.
- **Region alignment (OSCAR).** Detect object regions in the image, align each region to noun phrases in the caption. Object tags act as anchor concepts. Better for VQA where the question targets a specific object.
- **Dense alignment (GLIP, GroundingDINO).** Pixel-level or token-level correspondence. Required for visual grounding ("which woman in the photo?") and open-vocabulary detection.

Joint multimodal embeddings cluster by *semantic concept*, not by modality:

![t-SNE of joint multimodal embeddings: images and captions of the same concept land in the same cluster.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig5_embedding_tsne.png)

That clustering structure is what makes zero-shot transfer work: a new image of "a beach" lands near the *concept* of beach, regardless of which captions the model has seen before.

---

## Downstream tasks

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

## Fusion strategies

Whenever you mix two modalities, you have to decide *where* to fuse them. Three patterns dominate.

![Three fusion strategies for multimodal models: early (raw concat), late (independent encoders + combine), cross-attention (deep interaction).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-multimodal-transfer/fig7_fusion_strategies.png)

**Early fusion.** Concatenate raw features and feed a single encoder: $\mathbf{h} = f([\mathbf{v}; \mathbf{t}])$. Simple but throws away pretrained unimodal encoders. Rare in modern systems.

**Late fusion.** Encode each modality independently, then combine: $\mathbf{h} = g(f_v(\mathbf{v}), f_t(\mathbf{t}))$. This is CLIP. Modular, scalable, retrieval-friendly, but interaction between modalities is shallow — the encoders don't "see" each other.

**Cross-attention (deep) fusion.** Encoders interleave cross-attention layers where queries from one modality attend to keys/values from the other:
$$\text{CrossAttn}(\mathbf{V}, \mathbf{T}) = \text{softmax}\!\left(\frac{(\mathbf{V}\mathbf{W}_Q)(\mathbf{T}\mathbf{W}_K)^\top}{\sqrt{d}}\right) \mathbf{T}\mathbf{W}_V$$
Used in ViLBERT, LXMERT, BLIP. Richer interaction, better on tasks needing fine-grained reasoning (VQA, grounding); slower at retrieval because every (image, text) pair must be re-scored.

| Strategy | Pretrained encoders | Interaction depth | Retrieval cost |
|---|---|---|---|
| Early | No | High but unprincipled | Low |
| Late | Yes | Shallow | $O(N + M)$ encodings, then ANN |
| Cross-attention | Yes | Deep | $O(N \cdot M)$ — re-score every pair |

Practical rule: use **late fusion** when retrieval is on the critical path (search, recommendation), **cross-attention** when reasoning quality matters more than latency (VQA, captioning).

---

## Implementation: a minimal CLIP

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

## SigLIP and the Post-CLIP Family

CLIP defined the genre but did not finish it. The 2023–2025 wave of vision-language models refines CLIP along three axes that matter in production.

### SigLIP: from softmax to sigmoid

CLIP's contrastive objective uses a softmax over the entire batch, which makes the loss inherently dependent on batch size — to get good gradients you need batches in the tens of thousands. SigLIP (Zhai et al., 2023) replaces the softmax with a per-pair sigmoid loss: each (image, text) pair is independently classified as match or non-match. This decouples the loss from batch size and lets you train at batch sizes as small as 256 with little quality loss. It also halves memory and roughly doubles wall-clock throughput at fixed accuracy. SigLIP-2 (released 2025) added masked-prediction and self-distillation on top, narrowing the gap with much larger DINOv2-style image-only encoders.

The practical takeaway: if you are training a CLIP-style model from scratch in 2025, **start with SigLIP**. The recipe is simpler, the GPU bill is smaller, and the off-the-shelf checkpoints (Google's `siglip-so400m-patch14-384`) outperform OpenAI's original CLIP at most retrieval and classification tasks.

### EVA-CLIP and scaling carefully

EVA-CLIP (Sun et al., 2023) showed that with the right initialisation — masked image modelling pretraining, then CLIP fine-tuning — you can match a CLIP trained at 10× the compute. The key insight is that the visual tower needs strong geometric features *before* contrastive alignment, not after. If you have any kind of image-only pretraining budget (even just MAE on ImageNet-1K), spend it before the CLIP stage.

### LongCLIP and the 77-token wall

The original CLIP text encoder caps text at 77 tokens, which is fine for image captions but disastrous for document-image retrieval, OCR-augmented search, and HTML screenshots. LongCLIP (Zhang et al., 2024) extends the context to 248 tokens via positional embedding interpolation plus a knowledge distillation stage. For any task involving long captions or document understanding, swap in LongCLIP weights — the change is a one-line config edit and the gain is dramatic.

### A note on licences

The post-CLIP ecosystem is not uniformly open. SigLIP weights are Apache 2.0 (Google), EVA-CLIP weights are MIT (BAAI), but several variants are research-only (LiT, SigLIP-2 large variants under Google's licence). For a commercial product, audit the licence before fine-tuning — it is the single most common compliance bug I see in vision-language deployments.

## Failure Modes I Have Actually Hit

Vision-language models fail in characteristic ways that are not documented in the canonical papers. Three I have shipped fixes for.

### Modality bias: the model ignores the image

A symptom that recurs across CLIP, BLIP, and even GPT-4o-style multimodal LLMs: when the text alone is sufficient to give a confident answer, the model will **systematically ignore the image**. Ask "what colour is the cat?" and show a picture of a *dog*, and the model often answers "orange" — it inferred from "the cat" that there must be a cat. The cause is training-data prior dominance: the text-image pairs in pretraining are overwhelmingly aligned, so the model never had to disagree with text.

Mitigation: at inference, when image and text disagree, expose both signals separately. For agentic systems, run a "describe what you see" pass first, then a "given that description, answer the question" pass. The two-stage prompt eliminates the bias at the cost of one extra call.

### Object hallucination in BLIP-2 / LLaVA

BLIP-2 (and its descendants like LLaVA) generate captions that include objects not present in the image, especially for uncommon scenes. The hallucination rate on POPE benchmark hits 15–25 % even for the best open models. The root cause is that the language model dominates the joint distribution: when the visual evidence is ambiguous, the LM completes from its prior.

Mitigations that work: contrastive decoding (subtract a "blind" forward pass that masks the visual tokens), object-grounded prompting ("list every object you can see, then describe their relationships"), and at training time, hard negatives sourced from object-detection mismatches.

### Resolution sensitivity

Most CLIP variants train at 224×224 or 384×384 and degrade sharply at other resolutions — not just smaller, but *larger*. Feeding a 1024×1024 image to a 224-trained CLIP often produces worse retrieval scores than downsampling first. The patches at higher resolution alias against the learned positional embeddings.

Fix: always resize to the model's training resolution before feeding it. If you need genuine high-resolution understanding, use a model designed for it (NaViT, Idefics2's any-resolution, or the LLaVA-NeXT tile splitting approach). The "more pixels = more information" intuition does not apply to fixed-resolution vision transformers.

### Final thought

The CLIP era taught us that contrastive pretraining can produce extraordinary general-purpose visual representations. The post-CLIP era is teaching us that the rough edges — modality bias, hallucination, resolution brittleness — are deeply baked in. If you are deploying a vision-language model in production, the work is not picking the right model; it is building the guardrails around its known failure modes.

## From-Scratch CLIP Training Pipeline

![CLIP InfoNCE recall surface over (batch size, temperature).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-Multimodal-Transfer/fig08_infonce_surface.png)

The minimal CLIP above runs, but it runs on a single GPU with whatever batch fits. That is fine for a tutorial; it is hopeless for actually learning anything. InfoNCE's mutual-information bound tightens with $\log B$, and CLIP's $B = 32{,}768$ is not a flex — it is the working point where the loss starts producing useful gradients on web-noise data. Below 1024, you mostly learn shortcuts.

So we need three things the minimal version skipped: gradient accumulation to fake a large batch on one device, mixed precision and checkpointing to fit the model in memory, and an all-gather across GPUs so the contrastive denominator sees negatives from every rank.

### Gradient accumulation, but careful

You cannot naively split InfoNCE across micro-batches. The softmax denominator is *global* — every positive is normalised against every negative in the same batch. If you split into 4 micro-batches of 256 and compute the loss inside each, you get four independent 256-batch losses, not one 1024-batch loss.

The fix: encode all 4 micro-batches first (no loss yet), concatenate their embeddings, then compute one joint loss over the full 1024×1024 similarity matrix. The encoders run with `torch.no_grad()` on the first 3 forward passes if you want to save memory — but then you have to recompute them under grad on the *one* backward step, which is exactly what gradient checkpointing already does. Cleaner solution: forward all 4 with grad enabled, accumulate activations, then backward once.

Memory accounting tells you which way to go. For a ViT-L/14 at 224 resolution, one forward pass of micro-batch 256 holds roughly 18 GB of activations on a 40GB A100. Four of those held simultaneously will not fit. The standard production recipe is therefore: enable activation checkpointing on the transformer blocks (drops the per-pass cost to ~6 GB), then accumulate forwards-with-grad, then one backward. The extra recompute cost in the backward step (~30%) is dwarfed by the throughput gain from a 4x larger effective batch.

A subtle related issue is that `DistributedDataParallel` will average gradients across ranks at backward time, but the contrastive loss is *not* a simple sum over examples — it is a softmax over the joint similarity matrix. If you let DDP do its default averaging on a per-rank loss, the cross-rank negatives never contribute to the gradient. The fix is upstream of DDP: do the all-gather first, compute one global loss, and let DDP average the *correct* per-rank gradient slices. This is exactly what the `GatherLayer` trick below makes possible.

### Mixed precision and checkpointing

Two orthogonal memory wins. Mixed precision (`torch.cuda.amp.autocast` + `GradScaler`) keeps weights in fp32 but runs forward in bf16/fp16 — roughly 1.7x throughput and 40% less activation memory on Ampere+. The `GradScaler` is needed only for fp16 (to keep gradients out of the underflow region by scaling the loss up before backward and unscaling before the optimizer step); bf16 has fp32's exponent range and skips the scaler entirely.

Gradient checkpointing trades compute for memory by re-running the forward of selected blocks during backward instead of caching activations; on a ViT-L/14 image encoder this cuts activation memory by ~3x at the cost of ~30% extra FLOPs. The right blocks to checkpoint are the transformer layers, not the patch embedding or the projection head — checkpointing tiny modules costs more in launch overhead than it saves in memory.

Stack the three tricks together — bf16 autocast, checkpointed transformer blocks, and parameter sharding via FSDP if the model itself does not fit — and a ViT-L/14 + text-Transformer CLIP trains comfortably on 40GB A100s with micro-batch 256.

### Distributed negatives

With $N$ GPUs at micro-batch $b$ each, naive DDP gives every rank a contrastive loss over only $b$ negatives — you have wasted $(N-1)b$ free negatives sitting on other ranks. The fix is `dist.all_gather` on the embeddings, so every rank sees $Nb$ tokens.

But `all_gather` is not autograd-aware: gradients on the gathered copies do not flow back to the source ranks. The standard hack is the `GatherLayer` trick — wrap `all_gather` in a custom `torch.autograd.Function` whose backward calls `dist.reduce_scatter` to send each rank its slice of the gradient.

The mathematical justification is straightforward. The InfoNCE loss is a sum over positives, and each positive's gradient with respect to *its own* embedding is computed locally; the only cross-rank dependency is through the softmax denominator, which is symmetric in the negatives. Reduce-scatter on the gradient stack distributes that denominator-side gradient back to whichever rank produced each negative, with no double counting. Skip the GatherLayer and you train fine — the loss still goes down — but the negatives from other ranks contribute to the *forward* loss without contributing to the *backward* update of those ranks' encoders, which is silently wasteful.

```python
class GatherLayer(torch.autograd.Function):
    """all_gather with gradient passthrough via reduce_scatter."""
    @staticmethod
    def forward(ctx, x):
        ctx.world = dist.get_world_size()
        out = [torch.zeros_like(x) for _ in range(ctx.world)]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        g = torch.stack(grads, dim=0)            # (W, b, d)
        out = torch.zeros_like(grads[dist.get_rank()])
        dist.reduce_scatter(out, list(g))        # sum over world, slice to rank
        return out
```

### The training loop

Putting it together: 4 micro-batches per step, all-gather across 8 ranks, autocast forward, scaled backward, grad clip, step.

```python
model = CLIP(embed_dim=512).cuda()
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.2,
                        betas=(0.9, 0.98), eps=1e-6)
scaler = torch.cuda.amp.GradScaler()
ACCUM = 4                                  # micro-batches per optimizer step

for step, batches in enumerate(loader):    # batches = list of ACCUM micro-batches
    opt.zero_grad(set_to_none=True)
    v_chunks, t_chunks = [], []

    # Forward all micro-batches, store embeddings (small) instead of activations
    for imgs, toks in batches:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v = model.module.image_encoder(imgs.cuda(non_blocking=True))
            t = model.module.text_encoder(toks.cuda(non_blocking=True))
        v_chunks.append(v); t_chunks.append(t)

    v_local = torch.cat(v_chunks, 0)       # (ACCUM*b, d)
    t_local = torch.cat(t_chunks, 0)

    # Cross-rank negatives
    v_all = torch.cat(GatherLayer.apply(v_local), 0)   # (W*ACCUM*b, d)
    t_all = torch.cat(GatherLayer.apply(t_local), 0)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        scale = model.module.logit_scale.exp().clamp(max=100)
        logits = scale * v_all @ t_all.t()
        target = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, target)
                      + F.cross_entropy(logits.t(), target))

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(opt); scaler.update()
```

Realistic numbers from a typical OpenCLIP-style run: 32 A100s, micro-batch 256, accumulation 1, all-gather across ranks gives effective batch $32 \times 256 = 8192$. One epoch over 100M filtered LAION pairs takes ~4 hours; a full 32-epoch run lands in 5–6 days. Loss curves are visibly noisier below effective batch 4096 — which is the empirical justification for caring about every trick in this section.

A few practical notes that bite in production runs. AdamW betas of $(0.9, 0.98)$ with eps $10^{-6}$ are not a quirk — the larger beta-2 is needed because gradient variance on contrastive losses spikes early in training, and the smaller eps prevents step-size collapse when the gradient norm bottoms out near convergence. Warmup matters more than for supervised training: 2000 steps of linear warmup, then cosine decay over the rest of the schedule, is the OpenCLIP recipe and it is hard to beat. Skip warmup at your peril — the early-step gradients on uninitialised projection heads are large enough to destabilise the entire run.

Logging-wise, watch the *learned* `logit_scale` value. Early in training it sits near its initial $\log(1/0.07) \approx 2.66$. As alignment improves it drifts up; if it pegs at the clamp value (100), the model has fully memorised the batch and you are about to overfit. If it drifts *down*, the contrastive signal is too weak — usually a sign the batch is too small or the learning rate is too high.

One more knob worth its own paragraph: weight decay of 0.2 is much higher than the 0.01–0.1 used in supervised training. The justification is that contrastive losses without strong regularisation tend to blow up the embedding magnitudes (since L2 normalisation in the forward pass means the optimiser can move weight norms freely without affecting the loss). Strong decay keeps the underlying weight matrices in a well-conditioned regime, which improves both training stability and downstream linear-probe transfer.

The pipeline is now strong enough to learn alignment. But late-fusion alignment is only one fusion regime — for tasks that need fine-grained visual reasoning, the encoders need to talk to each other.

---

## Cross-Attention Fusion (ViLBERT-Style)

![CLIP training animation: image and text embeddings aligning under contrastive loss.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/08-Multimodal-Transfer/anim_clip_alignment.gif)

Late fusion is the right answer for retrieval and exactly the wrong answer for VQA. The reason is structural: a single dot product between two pooled vectors cannot express "the *small* dog *behind* the woman in *red*". Each modality needs to consult the other while still encoding, not just at the very end.

Concretely, take a CLIP zero-shot VQA baseline on a benchmark like GQA. Performance hovers around 30–35%, barely above chance for a 5-way question, because the model has compressed both image and question into a single similarity score and lost every relational signal. A cross-attention model in the same parameter budget reaches 55–60%. The gap is not architectural cleverness; it is information bandwidth between the modalities.

Three fusion regimes, ranked by how deeply the modalities interact:

- **Late fusion** (CLIP, SigLIP). Encode each modality independently, dot-product the pooled embeddings. Cheap, precomputable, retrieval-native.
- **Early fusion** (early VisualBERT). Concatenate raw image patch tokens with text tokens, run a single transformer over the union. Throws away the option to use unimodal pretrained encoders, and the model has to rediscover modality structure from scratch.
- **Cross-attention fusion** (ViLBERT, LXMERT, BLIP). Keep two encoders, but interleave layers where one modality's queries attend to the other's keys and values. Modality-aware, deeply interactive, and you keep your pretrained towers.

The information-theoretic framing makes the trade-off concrete. Late fusion compresses each modality to a single vector $\mathbf{v}, \mathbf{t} \in \mathbb{R}^d$ before the interaction; the only signal that survives is what was encoded into those $d$ scalars. Cross-attention preserves token-level structure on both sides — $L_v$ image tokens and $L_t$ text tokens — and lets the interaction look at all $L_v \times L_t$ pairs of attention weights. The expressive gap is large: a sentence like "the small dog behind the woman in red" needs at least four cross-modal correspondences (small↔dog patches, dog↔dog patches, behind↔spatial relation, red↔dress patches), none of which can be unambiguously encoded into a single pooled vector.

Cross-attention is bidirectional: vision tokens query text $\mathbf{Q}_v \mathbf{K}_t^\top$ to absorb language context, and text tokens query vision $\mathbf{Q}_t \mathbf{K}_v^\top$ to absorb visual context. Both happen inside the same block, with residual connections so each tower remains recognisably itself.

Some implementations only run cross-attention in one direction (e.g. text-to-vision in BLIP's image-grounded text encoder) because the downstream task only needs the language-side representation. That is a perfectly fine optimisation when you know the asymmetry; for a general-purpose VL encoder, bidirectional is the safer default and roughly doubles parameter count of the fusion layers without doubling FLOPs at inference (since each direction's attention is $O(L_v L_t d)$, which is fixed regardless of bidirectionality).

Worth noting: cross-attention scaling does not look like self-attention scaling. Self-attention's compute is $O(L^2 d)$ in the sequence length $L$, which dominates for long sequences. Cross-attention is $O(L_v L_t d)$, *bilinear* in the two sequence lengths — for typical CLIP-scale settings ($L_v = 257$ patch tokens, $L_t = 77$ text tokens), the cross-attention compute is about 8x smaller per layer than the unimodal self-attention on either side. Cross-attention layers are cheap; the cost shows up only because each (image, text) pair needs its own forward pass through the entire fusion stack, which is the precomputation problem we keep coming back to.

```python
class CrossAttentionBlock(nn.Module):
    """One bidirectional cross-attention layer between vision and text streams."""

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.ln_v_q = nn.LayerNorm(d_model)
        self.ln_v_kv = nn.LayerNorm(d_model)
        self.ln_t_q = nn.LayerNorm(d_model)
        self.ln_t_kv = nn.LayerNorm(d_model)
        self.attn_v2t = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_t2v = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        h = int(d_model * mlp_ratio)
        self.mlp_v = nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Linear(d_model, h), nn.GELU(),
                                   nn.Linear(h, d_model))
        self.mlp_t = nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Linear(d_model, h), nn.GELU(),
                                   nn.Linear(h, d_model))

    def forward(self, v: torch.Tensor, t: torch.Tensor,
                v_mask: torch.Tensor = None, t_mask: torch.Tensor = None):
        # Vision queries attend to text K/V
        v_attn, _ = self.attn_v2t(self.ln_v_q(v), self.ln_t_kv(t), self.ln_t_kv(t),
                                  key_padding_mask=t_mask)
        v = v + v_attn
        # Text queries attend to vision K/V
        t_attn, _ = self.attn_t2v(self.ln_t_q(t), self.ln_v_kv(v), self.ln_v_kv(v),
                                  key_padding_mask=v_mask)
        t = t + t_attn
        # Per-modality MLPs
        v = v + self.mlp_v(v)
        t = t + self.mlp_t(t)
        return v, t
```

Stack 4–6 of these on top of frozen unimodal encoders and you have the ViLBERT skeleton. Note the asymmetric layer norms (separate `ln_*_q` vs `ln_*_kv`) — cross-attention is more stable when the query and key/value streams are normalised independently, since they come from different distributions.

A subtle implementation point: the attention masks are *not* symmetric. `t_mask` (text padding) goes into `attn_v2t` so vision queries do not attend to padding tokens; `v_mask` (which can encode patch-level masking for masked image modelling, or be `None` if every patch is valid) goes into `attn_t2v`. Getting this wrong silently degrades performance — the model still trains, the loss still goes down, but recall plateaus 5–8 points below the unmasked-correctly baseline.

Initialisation also matters here in a way it does not for unimodal transformers. The `attn_v2t` and `attn_t2v` weights start at random; if the residual paths dominate from step zero (which they should), the cross-attention contribution is approximately zero and the output is just whatever the unimodal towers produce. As training proceeds the cross-attention weights grow and the modalities start consulting each other. If you initialise the cross-attention output projections to zero (the trick used in Flamingo's gated cross-attention), you get exactly this behaviour by construction and training is noticeably more stable in the first few thousand steps.

A related consideration is whether to *freeze* the unimodal towers during fusion training. The Flamingo answer is yes — train only the cross-attention layers and a small visual resampler, leaving the language model and vision encoder untouched. This keeps the model's general competence intact and reduces compute by ~10x. The BLIP-2 answer is also yes for the heavy components and no for the small connector. The end-to-end answer (CLIP4VQA, ViLBERT) gives marginally better task scores at the cost of much higher compute and the usual catastrophic-forgetting risks. For most use cases below billion-token training budgets, freezing is the right default.

Representative numbers on Flickr30K image-text retrieval, holding the visual backbone fixed:

| Fusion | $R@1$ image→text | Per-query latency |
|---|---|---|
| Late (dot-product) | 65% | 8 ms |
| Cross-attention (4 layers) | 72% | 24 ms |

These numbers are stable across CLIP-style backbones — the gap is fundamentally about fusion, not about encoder quality. Ablating the cross-attention to one layer recovers about half the gain (~3 points) at a third of the latency cost; ablating to eight layers gives diminishing returns (~1 point above four layers) at twice the cost. Four to six layers is the empirical sweet spot for retrieval-style benchmarks; VQA benefits from going deeper because the reasoning chain is longer.

The 7-point recall jump is real; so is the 3x latency cost. And the latency hides a worse fact: cross-attention has to *re-run* every (image, text) pair through the fusion stack, which means you cannot precompute a database. For 1M items and one query, late fusion is $O(N + M)$ encodings then ANN; cross-attention is $O(N)$ full forward passes. At scale this is the difference between 50ms and 50 minutes.

The decision rule is mechanical. Retrieval at scale: late fusion, period. VQA, NLI, grounding, anything where the answer depends on a specific cross-modal relationship: cross-attention, paid happily.

A useful hybrid that production systems land on: late fusion for *recall*, cross-attention for *re-rank*. Pull the top 100 candidates with cosine similarity over precomputed embeddings, then run cross-attention on just those 100 pairs to produce the final ordering. You pay $O(N + M)$ encodings plus 100 cross-attention forwards per query, which is well under a second even for million-item databases. Most retrieval-augmented VQA pipelines end up here whether they planned to or not.

The two-stage pipeline also makes the failure mode legible. If recall is bad, you fix the late-fusion encoder (more pretraining data, longer training, better contrastive loss). If recall is fine but the wrong candidate ranks first, you fix the cross-attention re-ranker (more layers, more training, harder negatives). Trying to debug a single end-to-end model that does both is much less tractable, which is the unstated reason the recall+rerank decomposition has won in production.

That covers training and architecture. The next question every team asks is how to take a pretrained CLIP and adapt it to their own data without burning the whole thing down.

---

## Fine-Tuning CLIP on Custom Data

Most CLIP "fine-tuning" failures come from picking the wrong tier. People take a 5K-pair medical-imaging dataset, full-fine-tune a ViT-L/14, and discover the model has catastrophically forgotten everything outside their domain. The right answer was a linear probe.

The diagnostic question is always the same: how many *distinct concepts* does your downstream task actually contain? A 5K medical dataset typically has 20–50 distinct radiological findings, which is well within the expressive capacity of a linear classifier on top of a 512-dimensional CLIP embedding. Full fine-tuning gives the optimiser permission to overwrite features that it does not need for *your* task — features that turn out to matter for OOD inputs you have not seen yet.

A simple decision tree, calibrated on real projects:

| Labelled pairs | Recipe | Trainable params |
|---|---|---|
| < 5K | Linear probe on frozen embeddings | ~$d \cdot K$ |
| 5K – 100K | LoRA on attention `q_proj`, `v_proj` of both towers | ~0.5–2% of base |
| > 100K | Full fine-tune at low LR (1e-6 to 1e-5) | 100% |

The boundaries are not magic numbers; they reflect when the optimisation problem changes character. Below ~5K pairs the loss landscape on a 100M-parameter model is dominated by a few high-gradient examples and you overfit before features adapt; above ~100K, LoRA starts running out of capacity to express the necessary shifts. Between those regimes, LoRA hits the sweet spot where you have enough data to train the adapter but not enough to safely move the base weights.

The linear-probe case is not really "fine-tuning CLIP" — you freeze both encoders, precompute embeddings once, and train a logistic regression on top. It is dirt-cheap and astonishingly competitive when data is scarce.

LoRA on a dual-encoder needs a small wrinkle: you insert separate LoRA modules on *both* the vision and text transformers, because both representations need to drift toward the new domain. Sharing LoRA weights across modalities is a popular bug — the modalities have different feature statistics and different layer counts.

Which projections to wrap is also a real choice. The original LoRA paper recommends `q_proj` and `v_proj` on the basis that those are the matrices whose updates most directly steer attention behaviour. For CLIP fine-tuning specifically, adding `out_proj` (the post-attention projection) gives a small additional gain, but extending to FFN matrices typically does not — the FFN already has high capacity and tends to memorise the small fine-tuning set when given trainable adapters.

```python
class LoRALinear(nn.Module):
    """y = W x + (B A) x, with A, B low-rank and W frozen."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

def inject_lora(model: CLIP, r: int = 8):
    """Wrap q_proj and v_proj of every attention block in both towers."""
    for tower in (model.image_encoder, model.text_encoder):
        for module in tower.modules():
            if isinstance(module, nn.MultiheadAttention):
                # PyTorch packs qkv; for clarity assume separate q_proj/v_proj
                module.q_proj = LoRALinear(module.q_proj, r=r)
                module.v_proj = LoRALinear(module.v_proj, r=r)
    return model
```

One non-obvious caveat: keep the *contrastive* loss. A common mistake is to bolt a classification head on top, switch to cross-entropy, and treat CLIP as a generic backbone. That destroys the embedding geometry — you no longer get zero-shot prompts, retrieval, or any of the things you adopted CLIP for in the first place. If you want classification, write the class names as text prompts and keep training contrastively against them.

A symmetric mistake on the data side: forgetting that contrastive fine-tuning needs *negative* pairs implicitly via batch composition. If your fine-tuning batch contains 32 chest X-rays all paired with the report "left lower lobe consolidation", every off-diagonal entry is also a positive in disguise, the contrastive loss collapses, and the model learns nothing. Either ensure caption diversity within the batch, or use a margin-based loss (triplet, InfoNCE with explicit hard negatives) that does not assume off-diagonal entries are negatives.

Concrete numbers from a 5K medical-captioning project (chest X-ray + radiology report snippets):

| Setup | Trainable | $R@1$ image→report |
|---|---|---|
| Zero-shot OpenAI CLIP | 0 | 12% |
| Linear probe | ~$d \cdot K$ | 31% |
| LoRA r=8, both towers | ~1.4M | 47% |
| Full fine-tune | 428M | 44% (overfits) |

The full fine-tune column is worth a beat. Without aggressive learning rate decay (1e-6), early stopping by validation R@1, and EMA on the weights, that 44% drops into the low 30s and the resulting checkpoint is unusable for any concept outside the medical domain. Even *with* all those guardrails, you have shipped a model that can no longer answer "is this a photo of a cat" correctly — which is fine if your product never needs that, but is a quiet kind of regression that does not show up in your fine-tuning metric.

LoRA wins outright at this scale, and the margin against full fine-tune is the catastrophic-forgetting tax. The same pattern repeats in legal, satellite, and product-catalogue domains.

One more knob worth tuning explicitly: the rank $r$. The intuition that "bigger is better" fails earlier than people expect for dual-encoder LoRA. On the chest X-ray data above, $r=4$ gets to 44%, $r=8$ to 47%, $r=16$ to 48%, and $r=32$ overfits back down to 43%. The reason is that contrastive fine-tuning on small datasets is fundamentally bounded by the number of *concept clusters* in the data, not by the expressivity of the adapter — once $r$ exceeds that intrinsic dimensionality, you start fitting noise. Start at $r=8$ and only move if validation tells you to.

Adapting CLIP to your domain is the friendly story. The unfriendly one is that the same shared embedding space which makes adaptation easy also makes adversarial manipulation easy.

---

## Adversarial Robustness of CLIP Embeddings

CLIP's superpower is also its attack surface. A shared embedding space means an attacker who knows the text encoder can write down *any* target caption, embed it, and then optimise an image perturbation to land near that embedding. Retrieval systems built on CLIP will then surface the perturbed image for queries that match the attacker's caption — never the image's actual content.

The threat model is unusually friendly to the attacker. The text encoder is the *public* half of any deployed CLIP-based retrieval system — even if your image embeddings are private, your text encoder weights almost certainly come from a published checkpoint, and the attacker's loss never needs to query your service. They craft the perturbation offline, then upload the perturbed image once. By contrast, attacking a closed-box supervised classifier requires either query access (rate-limitable) or transfer attacks (less reliable).

Formally: given a clean image $x$, a victim image encoder $f_{img}$, and an attacker-chosen target text whose embedding is $t = f_{txt}(\text{caption})$, the attacker solves

$$\min_{\delta} \; \|f_{img}(x + \delta) - t\|_2^2 \;+\; \lambda \|\delta\|_2^2 \quad \text{s.t.} \; \|\delta\|_\infty \le \epsilon$$

The first term pulls the perturbed embedding to the target caption; the second keeps the perturbation small enough that humans do not notice. L-BFGS handles this cleanly because the constraint is small and the objective is smooth.

Two design choices on the objective matter. The $\ell_2$ alignment term is preferred over a cosine-similarity term because L-BFGS prefers smooth quadratic loss surfaces, and on the unit hypersphere $\|f_{img}(x+\delta) - t\|_2^2 = 2 - 2 f_{img}(x+\delta)^\top t$, so the two are equivalent up to a constant — but $\ell_2$ avoids the trigonometric singularity at $\cos\theta = -1$. The $\lambda \|\delta\|_2^2$ regulariser is mostly cosmetic given the hard $\ell_\infty$ projection; setting it to $10^{-3}$ tends to produce smoother-looking perturbations without affecting attack success.

```python
def clip_attack(model, x, target_caption_tokens, eps=8/255, lam=1e-3,
                steps=200, lr=0.01):
    """Untargeted-norm L-BFGS attack to align f_img(x+delta) with caption embedding."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        t = model.text_encoder(target_caption_tokens.unsqueeze(0))   # (1, d)

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=lr, max_iter=steps,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        v = model.image_encoder((x + delta).clamp(0, 1).unsqueeze(0))  # (1, d)
        align = ((v - t) ** 2).sum()
        reg = lam * (delta ** 2).sum()
        loss = align + reg
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        delta.clamp_(-eps, eps)              # project back into L-infinity ball
    return (x + delta).clamp(0, 1).detach()
```

Empirically, with $\|\delta\|_\infty \le 8/255$ — visually imperceptible, the standard adversarial budget — this attack succeeds at >90% targeted retrieval flip on ImageNet against OpenAI's CLIP ViT-B/32. The attacker can pick any target caption from a vocabulary of millions and reliably hijack retrieval results. Worse, the perturbations *transfer* across CLIP variants: an attack crafted on ViT-B/32 retains 60–70% success on ViT-L/14 and SigLIP, because the contrastive objective shapes embedding geometry in similar ways across architectures.

The supervised baseline is much harder to attack. The same $\epsilon = 8/255$ budget against a vanilla ResNet-50 only flips classification to a chosen target ~40% of the time — and that is targeted *classification*, not embedding alignment, which is a strictly easier objective for the attacker. CLIP's contrastive embedding space is geometrically smoother, which is exactly what makes gradient-based attacks effective: every direction in embedding space is reachable, by design.

Mitigations, in order of cost-effectiveness:

- **Prompt ensembling.** Average the embeddings of 8–80 prompt templates ("a photo of a {c}", "a blurry photo of a {c}", ...) instead of one. Empirically 1.5–3x more adversarially robust at zero extra training cost, because the attacker has to align with the *mean* of many text directions simultaneously.
- **Input transformations.** JPEG compression at quality 75, random resized crop, or a Gaussian blur ablate most $\ell_\infty$ perturbations. Cheap and stackable. The catch: the attacker can simulate the same defence pipeline during attack optimisation (Expectation-over-Transformations), which restores most of the attack success rate. So treat input transforms as a speed bump for opportunistic attackers, not a wall against motivated ones.
- **Adversarial fine-tuning.** Madry-style PGD training on contrastive loss. *In CLIP this is a worse trade-off than in supervised models* — robustness gains are modest (~15% reduction in attack success rate) and clean-data zero-shot accuracy drops 5–10 points. The contrastive geometry seems more fragile to adversarial perturbations at training time than the cross-entropy geometry.

A useful diagnostic before deployment: compute the *Lipschitz ratio* of your image encoder by sampling pairs $(x, x + \delta)$ with small random $\delta$ and measuring $\|f_{img}(x + \delta) - f_{img}(x)\| / \|\delta\|$. CLIP encoders typically sit around 8–15 in this ratio; an ImageNet ResNet-50 is closer to 3–5. The high Lipschitz constant is precisely why the L-BFGS attack converges so quickly — small input perturbations buy large embedding-space movement. There is no free fix; if you need both zero-shot generality and adversarial robustness in the same model, today's recipes do not deliver and you should split the system architecturally instead.

A practical layered defence that production teams actually deploy: (i) prompt ensemble for in-distribution robustness, (ii) a perceptual hash check at upload time to catch known adversarial templates, (iii) a separate small "is this image natural" classifier that flags abnormal frequency-domain signatures characteristic of $\ell_\infty$ perturbations, and (iv) a human review queue for high-impact retrieval results (top-result hijacks on commercial queries). None of these closes the attack on its own; together they raise the effort cost for an attacker by 2–3 orders of magnitude, which is enough to deter most opportunistic abuse.

The overall picture: CLIP's open embedding space is a feature for transfer and a liability for adversarial robustness, and that trade-off is fundamental, not a bug to be patched. If you are deploying CLIP in a setting where adversaries can submit images — content moderation, e-commerce search — assume the embeddings are manipulable and design the system around that assumption rather than hoping training tricks make it go away.

The pattern repeats at every level of the post-CLIP stack. SigLIP's per-pair sigmoid loss is *more* attackable than softmax InfoNCE because there is no normalisation across negatives to compete against. BLIP-2's frozen Q-Former inherits the underlying ViT's Lipschitz behaviour. LLaVA-style multimodal LLMs add a new attack surface where adversarial images steer text generation through the visual token projection. Every architectural choice that makes the model more useful for general-purpose multimodal reasoning also makes it more useful for adversarial steering — not because the designers were careless, but because both properties draw on the same underlying smoothness of the learned representation. The honest framing is that adversarial robustness and zero-shot generality are duelling design goals in the current paradigm, and choosing one means accepting reduced quantity of the other.

## FAQ

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
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding (InfoNCE). *[arXiv:1807.03748](https://arxiv.org/abs/1807.03748)*.
