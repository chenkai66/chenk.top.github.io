---
title: "Transfer Learning (9): Parameter-Efficient Fine-Tuning"
date: 2025-05-17 09:00:00
categories:
  - Transfer Learning
  - Machine Learning
tags:
  - PEFT
  - LoRA
  - Adapter
  - Parameter Efficiency
  - Transfer Learning
series:
  name: "Transfer Learning"
  order: 9
  total: 12
lang: en
mathjax: true
description: "Derive LoRA's low-rank adaptation, the Adapter bottleneck, Prefix-Tuning, Prompt-Tuning, BitFit and QLoRA. Includes a from-scratch LoRA implementation with weight merging and a method-selection guide."
disableNunjucks: true
series_order: 9
---

How do you fine-tune a 175B-parameter model on a single GPU? Update only 0.1% of the parameters. Parameter-Efficient Fine-Tuning (PEFT) makes this possible -- and on most benchmarks it matches full fine-tuning. This post derives the math behind LoRA, Adapter, Prefix-Tuning, Prompt-Tuning, BitFit and QLoRA, and gives you a single picture for choosing among them.

## What You Will Learn

- Why the low-rank assumption holds for weight updates
- LoRA: derivation, initialization, scaling, and weight merging
- Adapter: bottleneck architecture and where to insert it
- Prefix-Tuning vs Prompt-Tuning vs P-Tuning v2
- QLoRA: how 4-bit quantisation gets a 65B model on one GPU
- Method comparison and a selection guide grounded in GLUE numbers

## Prerequisites

- Transformer architecture (attention, FFN, residual + LayerNorm)
- Matrix decomposition basics (rank, SVD)
- Transfer learning fundamentals (Parts 1-6)

---

## The Full Fine-Tuning Problem

Full fine-tuning updates every parameter $\boldsymbol{\theta}$:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$$

For GPT-3 (175B params) this means roughly **700 GB of FP32 weights**, plus gradients, plus optimiser states -- and one full copy per task. Even after the model fits, the per-task storage and serving cost is brutal: 100 customers means 100 copies of a 700 GB checkpoint.

PEFT replaces this with an additive decomposition:

$$\boldsymbol{\theta}^* = \boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta}, \qquad |\Delta\boldsymbol{\theta}| \ll |\boldsymbol{\theta}_0|.$$

Freeze $\boldsymbol{\theta}_0$, ship a tiny task-specific $\Delta\boldsymbol{\theta}$. The pretrained weights become a shared backbone, and adaptation becomes a thin delta you can store, version and route per request.

| Method | Trainable params | Storage saving |
|--------|-----------------|------------|
| Full Fine-Tuning | 100% | 0% |
| LoRA | 0.1 - 1% | 99 - 99.9% |
| Adapter | 0.5 - 2% | 98 - 99.5% |
| Prefix-Tuning | ~0.1% | ~99.9% |
| Prompt-Tuning | <0.01% | >99.99% |
| BitFit | ~0.1% | ~99.9% |

---

## LoRA: Low-Rank Adaptation

![LoRA: W = W_0 + (alpha/r) * B * A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig1_lora_decomposition.png)

### The Idea in One Equation

LoRA assumes the *update* $\Delta\mathbf{W}$ has low rank, so it parameterises it as a product of two thin matrices:

$$\mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{r}\, \mathbf{B}\mathbf{A},
\qquad \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}},\;
        \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r},\;
        r \ll \min(d_{\text{in}}, d_{\text{out}}).$$

The frozen $\mathbf{W}_0$ has $d_{\text{in}} d_{\text{out}}$ parameters; the trainable update has only $r(d_{\text{in}} + d_{\text{out}})$. For $d=4096, r=8$ that is **0.39%** of the original.

### Why Low Rank Works

Two empirical results justify the assumption:

1. **Intrinsic dimensionality (Aghajanyan et al., 2020).** Fine-tuning trajectories live in a low-dimensional subspace of weight space. A few hundred parameters can already match full fine-tuning on many tasks.
2. **Spectrum of pretrained matrices.** The singular values of attention projections decay quickly: a handful of directions carry most of the energy. The *update* needed to specialise to a new task is even more concentrated.

In other words, the pretrained model already knows almost everything; adaptation just rotates a small slice.

### Initialization, Scaling and the Forward Pass

A few details make LoRA work in practice:

- **Init.** $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$ with Kaiming-style variance, $\mathbf{B} = \mathbf{0}$. This guarantees $\Delta\mathbf{W} = \mathbf{0}$ at step 0, so training starts from the pretrained behaviour exactly.
- **Forward.** Compute $\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \tfrac{\alpha}{r}\,\mathbf{B}(\mathbf{A}\mathbf{x})$. **Never materialise $\mathbf{B}\mathbf{A}$**: it is a $d_{\text{out}}\times d_{\text{in}}$ dense matrix, exactly what we were avoiding.
- **Scaling.** The factor $\alpha/r$ decouples step size from rank: doubling $r$ does not double the effective learning rate, so $\alpha$ can stay fixed across rank sweeps.
- **Merging.** At inference we can collapse the adapter: $\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + (\alpha/r)\,\mathbf{B}\mathbf{A}$. This adds **zero latency** -- you ship a single weight matrix.
- **Where to apply.** Empirically, attaching LoRA to the **query** and **value** projections beats attention-only or MLP-only, and matches "all linear layers" at much lower cost.

### How Much Smaller Is "Smaller"?

![Trainable parameters per method, log scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig2_param_count.png)

On a 175B base, the difference between Full FT and LoRA $r=8$ is roughly **five orders of magnitude** in trainable parameters -- but, as we will see in the GLUE chart at the end, almost no difference in score.

---

## Adapter: Bottleneck Modules in the Block

![Adapter placement in a Transformer block](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig3_adapter_placement.png)

Houlsby et al. (2019) take a different route: instead of editing existing weights, **insert** small trainable modules into each Transformer block. Each adapter is a residual bottleneck:

$$\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}}\, \sigma(\mathbf{W}_{\text{down}}\, \mathbf{h}),
\qquad \mathbf{W}_{\text{down}} \in \mathbb{R}^{m \times d},\;
        \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times m},\; m \ll d.$$

The "down then up" projection is the same trick as in LoRA, but applied as an additional layer rather than a delta to an existing one. Initialising $\mathbf{W}_{\text{up}}$ near zero again makes the block start as identity.

**Adapter vs LoRA** -- a useful side-by-side:

| | Adapter | LoRA |
|---|---------|------|
| What changes | New module added | Existing weights re-parameterised |
| Inference latency | Yes (extra serial layer) | None (merge into $W_0$) |
| Per-task storage | $\sim 2md$ per layer | $\sim r(d_\text{in}+d_\text{out})$ per layer |
| Best for | Encoder models (BERT) | Generative models (GPT-family) |

Two follow-ups are worth knowing. *Pfeiffer adapters* keep only one adapter per block (after the FFN) and recover most of the quality at half the parameters. *Parallel adapters* (He et al., 2021) compute the bottleneck **in parallel** with the FFN to avoid the serial dependency on a GPU, narrowing the latency gap with LoRA.

---

## Prefix-Tuning

![Prefix-Tuning: learnable virtual tokens at every layer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig4_prefix_tuning.png)

Prefix-Tuning (Li & Liang, 2021) does not touch any model weights at all. Instead it prepends $m$ **learnable "virtual tokens"** to the key/value sequence at every layer:

$$\bigl[\mathbf{P}_1, \ldots, \mathbf{P}_m,\; \mathbf{x}_1, \ldots, \mathbf{x}_n\bigr] \to \text{Transformer}.$$

Only the prefix matrices are trained, $m \times d \times L \times 2$ parameters in total (key + value at $L$ layers). Two practical notes:

- Direct optimisation of the prefix is unstable. The original paper trains the prefix through a small MLP, then drops the MLP at inference.
- The information bottleneck is real: a 20-token prefix can only carry so much, so on hard tasks Prefix-Tuning lags LoRA by 0.5-1 point.

Conceptually, the prefix steers attention from the very first layer, biasing what the model "looks at" for every real token that follows. It is closest in spirit to soft prompts but operates inside the attention mechanism rather than the embedding table.

---

## Prompt-Tuning vs P-Tuning v2

![Prompt-Tuning vs P-Tuning v2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig5_prompt_vs_ptuning.png)

**Prompt-Tuning** (Lester et al., 2021) is the radical simplification: tune $m$ soft prompt vectors at the *input layer only*. Trainable parameters are just $m \times d$ -- often a few thousand.

The catch: it only really works on **very large** models. Lester et al. show the gap to full fine-tuning closes monotonically with scale; below ~1B parameters the loss is severe, but at 10B+ it essentially vanishes. The intuition: a 100-billion-parameter model has enough capacity to "interpret" arbitrary continuous prompts, while a small one needs more localised editing power.

**P-Tuning v2** (Liu et al., 2022) bridges Prompt-Tuning and Prefix-Tuning: trainable prompts are added at *every* layer, not only at the input. The result matches full fine-tuning across model sizes (including small ones) on most tasks, at ~0.1% trainable parameters.

## BitFit

The most reductive method of all: **fine-tune only the bias terms** (Zaken et al., 2021). That is around 0.08% of parameters for BERT, yet it is competitive with full fine-tuning on small-data GLUE tasks. Why does it work? Biases shift the input distribution to each non-linearity, which is enough to re-route activations without changing the linear maps themselves. As a baseline it is essentially free; as a serious method it underperforms LoRA by a small margin.

---

## QLoRA: Quantisation + LoRA

![QLoRA: 4-bit base + paged optimiser + bf16 LoRA adapters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig6_qlora_stack.png)

QLoRA (Dettmers et al., 2023) is the engineering advance that put 65B fine-tuning on a single 48 GB GPU. It composes three ideas:

1. **NF4 (4-bit NormalFloat) quantisation** of the frozen base weights. The codebook is information-theoretically optimal under the assumption that pretrained weights are zero-mean Gaussian, which they approximately are.
2. **Double quantisation**: re-quantise the per-block scale constants themselves, saving a further ~0.4 bits per parameter.
3. **Paged optimisers**: page Adam's state between CPU and GPU through unified memory, avoiding OOM during long sequences.

Crucially, **gradients still flow through the quantised matmul** to the bf16 LoRA adapters above. The frozen base never needs FP16 weights; only the small adapters do. The right-hand chart shows the punchline: full FT of a 65B model needs ~700 GB; QLoRA fits in ~50 GB.

QLoRA matches full FT on the Vicuna benchmark with <2% degradation, while running on hardware orders of magnitude cheaper.

---

## Implementation: LoRA from Scratch

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """W' = W_0 + (alpha / r) * B @ A.

    The base weight is frozen; only A and B receive gradients.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank

        # Frozen pretrained weights (loaded later).
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(
            torch.zeros(out_features), requires_grad=False)

        # Trainable low-rank factors.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self._merged:
            return base
        # IMPORTANT: never materialise B @ A explicitly.
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base + self.scaling * delta

    @torch.no_grad()
    def merge_weights(self) -> None:
        """Fold the LoRA update into the base weight; zero overhead at inference."""
        if self._merged:
            return
        self.weight.add_(self.scaling * (self.lora_B @ self.lora_A))
        self.lora_A.zero_(); self.lora_B.zero_()
        self._merged = True


def apply_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0,
               targets: tuple[str, ...] = ("query", "value")) -> None:
    """Replace selected nn.Linear layers in-place with LoRALayer."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            lora = LoRALayer(child.in_features, child.out_features, rank, alpha)
            lora.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora.bias.data.copy_(child.bias.data)
            setattr(module, name, lora)
        else:
            apply_lora(child, rank, alpha, targets)
```

Three things to notice in the implementation:

1. The forward pass **never builds $\mathbf{B}\mathbf{A}$** -- it left-multiplies by $\mathbf{A}$ first, producing an intermediate of width $r$.
2. `merge_weights` is a one-time, in-place operation. After merging, the layer behaves bit-for-bit like a plain `nn.Linear`, so latency at serving time is unchanged.
3. `apply_lora` walks the module tree, swapping only the layers named `"query"` / `"value"` -- the canonical recommendation from the LoRA paper.

---

## Method Comparison: GLUE in One Picture

![PEFT efficiency vs GLUE accuracy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig7_glue_comparison.png)

The bubble chart sums up the design space. With RoBERTa-base on GLUE:

- **LoRA $r=8$ at 0.24% trainable parameters** essentially ties full fine-tuning (within 0.1 points).
- **Adapter** matches LoRA's quality but at ~4x the trainable parameter count and with non-trivial inference latency.
- **Prefix-Tuning** trades a small accuracy drop for very low parameter count, useful when you must store hundreds of task adapters.
- **Prompt-Tuning** is only competitive when the base model is much larger than RoBERTa-base; on smaller models it leaves 3-4 points on the table.
- **BitFit** is a remarkable baseline given how trivial it is.

## Method Selection Guide

| Scenario | First choice | Reason |
|----------|-------------|--------|
| GPT-style decoder fine-tuning | LoRA (q,v) | No latency, easy to merge, scales to 70B+ |
| BERT-style classification, many tasks | Adapter (Pfeiffer) | Per-task storage tiny, well-trodden tooling |
| Very large base (>10B), many tasks | Prompt-Tuning | Negligible storage, scales gracefully |
| Need zero inference overhead | LoRA + merge | Final model is a single matrix |
| Few-shot / few-thousand examples | Prefix-Tuning or BitFit | Strong inductive bias, less overfitting |
| Single 24-48 GB GPU, 30B+ model | QLoRA | NF4 + paged Adam unlocks the size class |

**Picking LoRA's rank.** Start at $r = 8$. Bump to 16 if validation lags by more than ~0.5 points; drop to 4 if you are memory-bound. Larger base models tolerate (and often prefer) **smaller** $r$ -- the intrinsic dimensionality of adaptation does not grow with model size.

---

## Q&A

### How much do you give up vs full fine-tuning?

For models above ~10B parameters, less than 1 point on most benchmarks (often within noise). Below ~1B, gaps of 2-5 points are common, especially on small datasets.

### What learning rate should LoRA use?

One to two orders of magnitude higher than full fine-tuning. The trainable parameters start near zero, so they can take much larger steps without diverging. A typical schedule is $\text{lr} = 1\text{e-}4$ to $5\text{e-}4$ with cosine decay.

### Can I combine PEFT with quantisation?

Yes -- that is exactly QLoRA. NF4 + double quantisation + bf16 LoRA fine-tunes 65B models on a single 48 GB GPU with <2% performance drop.

### Should I apply LoRA to all linear layers?

Not necessary. The original paper finds query + value already captures most of the gain; adding key and output gives marginal improvement at 2x the parameters. For decoder-only LLMs, also adding the MLP up/down projections sometimes helps on long-context tasks.

### Adapter vs LoRA at serving time?

LoRA wins after merging: the model is a single weight matrix, identical to full fine-tuning at inference. Adapters add 1-3% latency per block due to the extra serial sub-layer; with parallel adapters this drops to under 1%.

---

## References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2020). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL*.
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML*.
- Pfeiffer, J., et al. (2021). AdapterFusion: Non-Destructive Task Composition for Transfer Learning. *EACL*.
- He, J., et al. (2021). Towards a Unified View of Parameter-Efficient Transfer Learning. *ICLR*.
- Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. *ACL*.
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. *EMNLP*.
- Liu, X., et al. (2022). P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-Tuning Universally Across Scales and Tasks. *ACL*.
- Zaken, E. B., Ravfogel, S., & Goldberg, Y. (2021). BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language-Models. *ACL*.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.

---

## Series Navigation

- Previous: [Part 8 -- Multimodal Transfer](/en/transfer-learning-8-multimodal-transfer/)
- Next: [Part 10 -- Continual Learning](/en/transfer-learning-10-continual-learning/)
- [View all 12 parts in this series](/tags/Transfer-Learning/)
