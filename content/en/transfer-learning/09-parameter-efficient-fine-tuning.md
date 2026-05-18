---
title: "Transfer Learning (9): Parameter-Efficient Fine-Tuning"
date: 2025-06-18 09:00:00
categories: Transfer Learning
  - Machine Learning
tags:
  - PEFT
  - LoRA
  - Adapter
  - Parameter Efficiency
  - Transfer Learning
series: transfer-learning
lang: en
mathjax: true
description: "Derive LoRA's low-rank adaptation, the Adapter bottleneck, Prefix-Tuning, Prompt-Tuning, BitFit and QLoRA. Includes a from-scratch LoRA implementation with weight merging and a method-selection guide."
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "transfer-learning-9"
---
How do you fine-tune a 175B-parameter model on a single GPU? Update only 0.1% of the parameters. Parameter-Efficient Fine-Tuning (PEFT) makes this possible — and on most benchmarks it matches full fine-tuning. This post derives the math behind LoRA, Adapter, Prefix-Tuning, Prompt-Tuning, BitFit and QLoRA, and gives you a single picture for choosing among them.

![Transfer Learning (9): Parameter-Efficient Fine-Tuning — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/illustration_1.png)


---

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
For GPT-3 (175B params) this means roughly **700 GB of FP32 weights**, plus gradients, plus optimiser states — and one full copy per task. Even after the model fits, the per-task storage and serving cost is brutal: 100 customers means 100 copies of a 700 GB checkpoint.

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

![Transfer Learning (9): Parameter-Efficient Fine-Tuning — Chapter summary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/illustration_2.png)


![LoRA: W = W_0 + (alpha/r) * B * A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig1_lora_decomposition.png)

### The Idea in One Equation

LoRA assumes the *update* $\Delta\mathbf{W}$ has low rank, so it parameterises it as a product of two thin matrices:
$$
\mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{r}\, \mathbf{B}\mathbf{A},
\qquad \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}},\;
        \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r},\;
        r \ll \min(d_{\text{in}}, d_{\text{out}}).
        $$
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
- **Merging.** At inference we can collapse the adapter: $\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + (\alpha/r)\,\mathbf{B}\mathbf{A}$. This adds **zero latency** — you ship a single weight matrix.
- **Where to apply.** Empirically, attaching LoRA to the **query** and **value** projections beats attention-only or MLP-only, and matches "all linear layers" at much lower cost.

### How Much Smaller Is "Smaller"?

![Trainable parameters per method, log scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig2_param_count.png)

On a 175B base, the difference between Full FT and LoRA $r=8$ is roughly **five orders of magnitude** in trainable parameters — but, as we will see in the GLUE chart at the end, almost no difference in score.

---

## Adapter: Bottleneck Modules in the Block

![Adapter placement in a Transformer block](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/09-parameter-efficient-fine-tuning/fig3_adapter_placement.png)

Houlsby et al. (2019) take a different route: instead of editing existing weights, **insert** small trainable modules into each Transformer block. Each adapter is a residual bottleneck:
$$
\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}}\, \sigma(\mathbf{W}_{\text{down}}\, \mathbf{h}),
\qquad \mathbf{W}_{\text{down}} \in \mathbb{R}^{m \times d},\;
        \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times m},\; m \ll d.
        $$
The "down then up" projection is the same trick as in LoRA, but applied as an additional layer rather than a delta to an existing one. Initialising $\mathbf{W}_{\text{up}}$ near zero again makes the block start as identity.

**Adapter vs LoRA** — a useful side-by-side:

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

**Prompt-Tuning** (Lester et al., 2021) is the radical simplification: tune $m$ soft prompt vectors at the *input layer only*. Trainable parameters are just $m \times d$ — often a few thousand.

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

1. The forward pass **never builds $\mathbf{B}\mathbf{A}$** — it left-multiplies by $\mathbf{A}$ first, producing an intermediate of width $r$.
2. `merge_weights` is a one-time, in-place operation. After merging, the layer behaves bit-for-bit like a plain `nn.Linear`, so latency at serving time is unchanged.
3. `apply_lora` walks the module tree, swapping only the layers named `"query"` / `"value"` — the canonical recommendation from the LoRA paper.

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

**Picking LoRA's rank.** Start at $r = 8$. Bump to 16 if validation lags by more than ~0.5 points; drop to 4 if you are memory-bound. Larger base models tolerate (and often prefer) **smaller** $r$ — the intrinsic dimensionality of adaptation does not grow with model size.

---

## DoRA, OFT, and the Post-LoRA Wave

LoRA dominated 2022–2024 because the recipe was simple and the numbers held up. The 2024–2025 wave of papers refines the same idea in three orthogonal directions, and a few are now production-relevant.

### DoRA: decompose the weight, then LoRA the direction

DoRA (Liu et al., 2024) starts from the observation that any weight matrix can be written as a magnitude-times-direction pair: $W = m \cdot \frac{V}{\|V\|}$. Full fine-tuning updates *both*; LoRA implicitly conflates them. DoRA freezes the magnitude $m$ as a learnable scalar per output channel and applies low-rank updates to $V$ alone:
$$
W' = (m + \Delta m) \cdot \frac{W + \Delta V}{\|W + \Delta V\|}.
$$
On Llama-2-7B reasoning benchmarks DoRA closes about half the gap between LoRA and full fine-tuning at the same parameter count. Cost: a single extra `Linear(out_dim, 1)` per adapted matrix and a normalisation in the forward pass — negligible at inference because you can fuse it back into $W$ once training stops.

### OFT: orthogonal fine-tuning that preserves geometry

Orthogonal Fine-Tuning (Qiu et al., 2023) replaces the additive update with an orthogonal *rotation*: $W' = R \cdot W$ where $R$ is parameterised as a Cayley transform of a small skew-symmetric matrix. Because $R$ is orthogonal, the singular values of $W$ are preserved exactly. This matters when the base model's calibration is fragile — text-to-image fine-tuning is the killer app, where LoRA tends to drift the base model's style budget while OFT keeps it intact. The downside is that OFT is more expensive per step than LoRA at matched rank.

### VeRA, IA³, and the further compression frontier

VeRA (Kopiczko et al., 2024) shares a single random low-rank pair $(A, B)$ across all layers and learns only a per-layer scaling vector. The result is a 10× parameter reduction over LoRA at near-equal accuracy on GLUE — useful when you are storing thousands of customer-specific adapters and disk is the binding constraint. IA³ goes further by learning only three rescaling vectors per Transformer layer (one for keys, one for values, one for the FFN), and is the right pick when you genuinely need adapters smaller than 100 KB.

The honest summary: for general-purpose LLM fine-tuning in 2025, **LoRA is still the right default**. Switch to DoRA when you are giving up too much accuracy at small ranks, OFT for image-generation work, and VeRA only when adapter storage is your bottleneck.

## Serving Multi-LoRA at Scale

The architectural advantage of LoRA — adapters are tiny matrices that can be added to a frozen base — only translates into a serving advantage if you actually exploit it. Three production patterns I have seen work.

### Hot-swappable adapter cache

Keep the base model resident in GPU memory and swap the LoRA matrices in and out per request. With rank 16 on a 7 B model, each adapter is roughly 16 MB; you can hold hundreds in CPU RAM and PCIe-transfer the right one in under 5 ms. This is the right pattern for multi-tenant SaaS where each customer has their own fine-tune. Frameworks: vLLM's `--enable-lora`, S-LoRA, or Punica all do this with subtly different memory layouts.

### Batched heterogeneous requests

The hard problem is when the same batch contains requests for different LoRAs. Naive serving sequentialises them and throws away most of the batch advantage. S-LoRA (Sheng et al., 2023) solves this by paging adapter matrices into a unified buffer and using a custom kernel that gathers the right $A$ and $B$ per request. On A100s the throughput gap between "single LoRA" and "100 LoRAs in one batch" shrinks from 8× to under 1.5×. If you are serving more than a handful of distinct adapters concurrently, this is the pattern to copy.

### When to bake the adapter back in

If a single adapter is hot enough that 95 %+ of your traffic uses it, the right move is to **fold it back into the base weights** (`W' = W + BA`) and serve a single merged model. This costs you the multi-tenant advantage but recovers single-model serving simplicity and a small amount of latency from skipping the adapter forward pass. We made this trade-off on a customer-facing assistant where a single corporate persona accounted for 98 % of QPS — merging cut p99 latency by 12 % at zero accuracy cost.

The decision tree: if adapters are roughly equal in QPS, keep them separate and serve via S-LoRA-style batching; if one dominates, merge it; if you have hundreds of low-QPS adapters, go fully hot-swap with a CPU-side cache.

## Adapter Latency Bottleneck Analysis

The Adapter section above said adapters add latency at inference. That is true but the magnitude is worth quantifying, because the trade-off between serial and parallel placement depends sharply on batch size and is the reason LoRA eventually won the production argument.

### Serial vs Parallel Placement

A serial adapter sits inside the residual path:
$$
h_{\text{out}} = h_{\text{in}} + \text{Adapter}(\text{FFN}(h_{\text{in}})).
$$
The adapter's forward pass cannot start until the FFN finishes. On a GPU this is a hard dependency — two sequential CUDA launches with their own kernel-launch overhead and SM scheduling stalls.

A parallel adapter (He et al., 2021) sidesteps this by computing the adapter from $h_{\text{in}}$ directly, in parallel with the FFN:
$$
h_{\text{out}} = h_{\text{in}} + \text{FFN}(h_{\text{in}}) + \text{Adapter}(h_{\text{in}}).
$$
On hardware with enough free SMs, the two branches overlap and the adapter's wall-clock cost approaches zero. The accuracy is slightly worse on some tasks because the adapter no longer sees the FFN's transformed activations.

### A Microbenchmark

```python
import time
import torch
import torch.nn as nn

d, m = 1024, 64
ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)).cuda()
adapter = nn.Sequential(nn.Linear(d, m), nn.GELU(), nn.Linear(m, d)).cuda()

def serial_forward(x):
    return x + adapter(ffn(x))

def parallel_forward(x):
    return x + ffn(x) + adapter(x)

def bench(fn, x, n=200):
    for _ in range(20): fn(x)               # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3  # ms

for bs in [1, 8, 32, 128]:
    x = torch.randn(bs, 128, d, device="cuda")
    base = bench(lambda z: z + ffn(z), x)
    ser = bench(serial_forward, x)
    par = bench(parallel_forward, x)
    print(f"bs={bs:3d}  base={base:.3f}  serial={ser:.3f} (+{(ser/base-1)*100:.1f}%)"
          f"  parallel={par:.3f} (+{(par/base-1)*100:.1f}%)")
```

On an A100 with FP32 weights I get something close to:

| batch | base (ms) | serial | overhead | parallel | overhead |
|-------|-----------|--------|----------|----------|----------|
| 1     | 0.41      | 0.48   | +18%     | 0.43     | +6%      |
| 8     | 0.43      | 0.49   | +14%     | 0.44     | +4%      |
| 32    | 0.62      | 0.68   | +9%      | 0.63     | +2%      |
| 128   | 1.85      | 1.93   | +4%      | 1.88     | +1.5%    |

Two patterns. First, the relative overhead falls with batch size — kernel-launch costs amortise across more work. Second, parallel beats serial at every batch size, with the gap largest at low batch.

### Practical Takeaway

For serving, parallel adapters are strictly better than serial. For training, serial sometimes wins by 0.2-0.5 points on tasks where the adapter's job is to refine FFN outputs rather than to inject orthogonal signal — NLI is the clearest example. The choice is therefore a small accuracy-latency knob.

Both lose to LoRA at inference. Once merged, LoRA contributes zero new ops to the forward pass; the layer is bit-for-bit a plain linear. This is the structural advantage the next sections build on: parameter efficiency that is also runtime efficiency.

### A Note on Multi-Adapter Serving

The latency story changes again when you serve many adapters concurrently. A serial adapter forces per-request branch divergence — different customers want different bottleneck weights, so you cannot share the kernel launch. A parallel adapter has the same problem but at least overlaps with the FFN, so the divergence cost is partially hidden. LoRA pays nothing here either: the merged path is one matmul regardless of which adapter you logically applied, provided you are willing to keep separate merged copies in memory. If you are not (because there are too many), the S-LoRA-style gather kernel discussed earlier in this article restores most of the throughput. Adapters have no equivalent escape hatch.

---

## NF4 Quantization From First Principles

QLoRA's headline trick is fine-tuning a 65B model on a single 48 GB GPU. The enabling technology is the NF4 codebook for the frozen base weights. It is worth seeing why it beats uniform int4 by enough to matter.

### Why Uniform int4 Wastes Bits

Uniform int4 takes a value range $[-w_{\max}, w_{\max}]$ and divides it into 16 equally spaced levels. If the underlying distribution of weights were uniform, this would be optimal. It is not. Pretrained Transformer weights are well-modelled as zero-mean Gaussian with small variance — most weights cluster near zero, with a thin tail. Equispaced levels put half the codebook into a region containing very few actual weights, and the other half too coarse for the dense centre. The result is large quantisation error around zero, where it hurts most.

### The NormalFloat Construction

The information-theoretically optimal codebook for a known distribution is the one that puts each level at the conditional mean of an equiprobable bin under that distribution. For a unit normal $\mathcal{N}(0, 1)$ with 16 bins, you place the level boundaries at the quantiles
$$
q_i = \Phi^{-1}\!\left(\frac{i}{16}\right), \qquad i = 1, \ldots, 15,
$$
and use the bin midpoints (or conditional means) as the 16 codebook values. Dettmers et al. (2023) tweak this slightly to ensure exactly one level lands at zero, which matters because pretrained weights have a literal zero mode after pruning-style regularisation.

The codebook values come out approximately:
$$
\{-1.00, -0.70, -0.53, -0.39, -0.28, -0.18, -0.09, 0.00,
0.08, 0.16, 0.25, 0.34, 0.44, 0.56, 0.72, 1.00\}.
$$
Notice the asymmetric spacing: levels are dense near zero where the Gaussian mass concentrates, and sparse near the tails.

### A From-Scratch Implementation

```python
import torch
from torch.distributions import Normal

def build_nf4_codebook() -> torch.Tensor:
    """16 levels, equispaced in CDF of a unit normal, anchored at 0 and +-1."""
    normal = Normal(0.0, 1.0)
    # Asymmetric split: 8 negative, 1 zero, 7 positive (Dettmers convention).
    neg = normal.icdf(torch.linspace(0.5 / 8, 0.5, 8))
    pos = normal.icdf(torch.linspace(0.5, 1 - 0.5 / 7, 7))[1:]
    levels = torch.cat([neg, torch.tensor([0.0]), pos])
    levels = levels / levels.abs().max()           # normalise to [-1, 1]
    return levels.sort().values

NF4 = build_nf4_codebook()

def quantize_nf4(W: torch.Tensor, blocksize: int = 64):
    """Per-block absmax scaling, nearest-codebook rounding."""
    W_flat = W.flatten()
    pad = (-W_flat.numel()) % blocksize
    W_flat = torch.cat([W_flat, W_flat.new_zeros(pad)])
    blocks = W_flat.view(-1, blocksize)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normed = blocks / scale                                    # in [-1, 1]
    # Nearest codebook index for each entry.
    dist = (normed.unsqueeze(-1) - NF4.to(W).view(1, 1, -1)).abs()
    idx = dist.argmin(dim=-1).to(torch.uint8)                  # 4-bit indices
    return idx, scale.squeeze(1), W.shape, pad

def dequantize_nf4(idx, scale, shape, pad):
    levels = NF4.to(scale).gather(0, idx.long().flatten()).view_as(idx)
    out = (levels * scale.unsqueeze(1)).flatten()
    if pad: out = out[:-pad]
    return out.view(shape)

# Backward pass uses straight-through: gradient of dequant is identity.
class NF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, scale, shape, pad):
        W = dequantize_nf4(idx, scale, shape, pad)
        ctx.save_for_backward(x, W)
        return x @ W.T
    @staticmethod
    def backward(ctx, gy):
        x, W = ctx.saved_tensors
        return gy @ W, None, None, None, None
```

A quick reconstruction-error check on a random Gaussian matrix the size of a Llama-7B attention projection:

```python
W = torch.randn(4096, 4096)
idx, scale, shape, pad = quantize_nf4(W)
W_hat = dequantize_nf4(idx, scale, shape, pad)
mse_nf4 = ((W - W_hat) ** 2).mean().item()

# Uniform int4 baseline.
s = W.abs().max() / 7
W_uni = (W / s).round().clamp(-8, 7) * s
mse_uniform = ((W - W_uni) ** 2).mean().item()
print(f"NF4 MSE = {mse_nf4:.4f}    uniform int4 MSE = {mse_uniform:.4f}")
```

Typical numbers: NF4 around 0.04, uniform int4 around 0.11 — almost 3x worse. The gap is even larger on real Llama weights, which are more sharply concentrated near zero than a fitted Gaussian.

### Double Quantization

NF4 stores one FP32 absmax per 64-element block. That overhead is $32 / 64 = 0.5$ bits per parameter — non-trivial when the weights themselves are 4 bits. Double quantisation re-quantises the absmax constants to int8 with a second-level FP32 scale per 256-block group. The extra error is negligible (the absmax distribution is itself well-behaved) and the saving is roughly 0.4 bits per parameter, or about 0.4 GB on a 7B model. On a 65B model the saving compounds to several GB — the difference between fitting and not fitting on a 48 GB card.

### Why This Unlocks 65B on One GPU

The arithmetic: 65B parameters at NF4 + double-quant is approximately $65 \cdot 10^9 \cdot 4.5 \,\text{bits} / 8 = 36.6$ GB for the frozen base. Add the bf16 LoRA adapters (~200 MB at $r=64$ across all linears), Adam state on those adapters only (~0.8 GB), and activations for the longest sequence you can manage, and you are under 48 GB. None of this is possible with FP16 base weights, which alone would cost 130 GB.

This is also why QLoRA's accuracy holds up: the gradient signal still flows through a faithful dequantisation of the base weights, so the LoRA adapters are learning corrections to a near-true model, not a corrupted one.

### Where the Straight-Through Estimator Hides

The backward pass through `dequantize_nf4` is technically non-differentiable — `argmin` and integer indexing have zero gradient almost everywhere. The `NF4Linear.backward` above sidesteps this by treating the dequantised weight as if it were the parameter, propagating $\partial \mathcal{L} / \partial W$ directly. This is the straight-through estimator (STE), and it is correct in the QLoRA setting for a subtle reason: the frozen base weights never receive gradient updates, so STE bias does not accumulate. The only gradients that matter are on the LoRA adapters above the quantised matmul, and those flow exactly. If you ever try to backpropagate into the quantised weights themselves — for quantisation-aware training, say — you have to revisit STE design carefully.

---

## FAQ

### How much do you give up vs full fine-tuning?

For models above ~10B parameters, less than 1 point on most benchmarks (often within noise). Below ~1B, gaps of 2-5 points are common, especially on small datasets.

### What learning rate should LoRA use?

One to two orders of magnitude higher than full fine-tuning. The trainable parameters start near zero, so they can take much larger steps without diverging. A typical schedule is $\text{lr} = 1\text{e-}4$ to $5\text{e-}4$ with cosine decay.

### Can I combine PEFT with quantisation?

Yes — that is exactly QLoRA. NF4 + double quantisation + bf16 LoRA fine-tunes 65B models on a single 48 GB GPU with <2% performance drop.

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
