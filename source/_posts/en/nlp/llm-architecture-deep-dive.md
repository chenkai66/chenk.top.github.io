---
title: "NLP (9): Deep Dive into LLM Architecture"
date: 2025-11-10 09:00:00
tags:
  - NLP
  - LLM
  - Transformer
  - RoPE
  - Flash Attention
  - MoE
categories: Natural Language Processing
series: NLP
part: 9
total_parts: 12
lang: en
mathjax: true
description: "Inside modern LLMs: pre-norm + RMSNorm + SwiGLU + RoPE + GQA, KV cache mechanics, FlashAttention's IO-aware schedule, sparse Mixture-of-Experts, and INT8 / INT4 quantization."
disableNunjucks: true
series_order: 9
---

The 2017 Transformer paper drew one block. Every production LLM today still uses that diagram as a silhouette, but almost every internal piece has been replaced. Pre-norm replaced post-norm. RMSNorm replaced LayerNorm. SwiGLU replaced GELU. Rotary embeddings replaced sinusoids. Multi-head attention became grouped-query attention. The dense FFN sometimes became a sparse mixture of experts. And the inference loop is dominated by a data structure that doesn't appear in the original paper at all: the KV cache.

This article walks through those changes in the order they actually matter when you implement, train, or deploy a model. We start with the modern decoder block, then the inference data structure that funds long contexts (the KV cache), then how positions are now encoded (RoPE / ALiBi), then the attention layout that makes the cache cheap (GQA / MQA), then the IO-aware kernel that makes attention fast (FlashAttention), and finally how to grow the model without growing per-token compute (MoE) or shrink it without losing quality (quantization).

## What you will learn

- **Modern block layout** — pre-norm + RMSNorm + SwiGLU + RoPE + GQA, and why each replacement happened.
- **KV cache mechanics** — how it converts the attention prefix from $O(n^2)$ recompute to $O(n)$ amortized cost, and what it costs in memory.
- **Position encoding** — sinusoidal vs RoPE vs ALiBi as three different answers to "where is this token?"
- **Attention variants** — MHA, MQA, GQA: trading head diversity for cache size, with concrete numbers on a 70B-class model.
- **FlashAttention** — exact attention with an IO-aware schedule that keeps tiles in SRAM and never materializes the $n \times n$ score matrix.
- **MoE** — sparse top-$k$ routing that grows total parameters without growing per-token FLOPs.
- **Quantization** — FP16 → INT8 → INT4 with GPTQ and AWQ, and what to expect for accuracy and memory.

## Prerequisites

- The base Transformer block (see [Part 4 — Attention and Transformers](/en/nlp-attention-transformer/)).
- Pre-trained model concepts (see [Part 5 — BERT and Pre-training](/en/nlp-bert-pretrained-models/)).
- Comfortable reading PyTorch and basic linear algebra.

---

## Three families: encoder-only, decoder-only, encoder-decoder

Before we get into the modern block, it helps to remember why almost all general-purpose LLMs ended up decoder-only. The three families differ only in their attention mask:

| Family | Mask | Pre-training objective | Strength | Examples |
|---|---|---|---|---|
| Encoder-only | bidirectional | masked LM (MLM) | semantic understanding | BERT, RoBERTa, DeBERTa |
| Decoder-only | causal (lower-triangular) | next-token prediction (LM) | generation, in-context learning | GPT, LLaMA, Qwen, Mistral |
| Encoder–Decoder | bidirectional encoder + causal decoder + cross-attn | denoising / span corruption | seq-to-seq | T5, BART, FLAN-T5 |

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

enc  = AutoModel.from_pretrained("bert-base-uncased")            # encoder-only
dec  = AutoModelForCausalLM.from_pretrained("gpt2")               # decoder-only
ed   = AutoModelForSeq2SeqLM.from_pretrained("t5-small")          # encoder-decoder
```

The decoder-only family won the scaling race for two reasons. First, every task can be cast as next-token prediction, so a single objective and dataset format scales cleanly. Second, the causal mask makes prefix caching (the KV cache below) very cheap — the encoder–decoder family has to cache cross-attention as well, and the encoder-only family can't generate at all without an external decoder. Today, "LLM" almost always means a decoder-only model with the additions in the next section.

---

## The modern decoder block

![Figure 1 — Modern LLM decoder block](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig1_modern_block.png)

A LLaMA-style block (LLaMA, LLaMA-2, Mistral, Qwen, Yi, DeepSeek, ...) differs from the 2017 block in five places. Each change has a small individual effect, but together they buy a model that trains stably without warmup tuning, generalizes to longer contexts, and runs faster at inference.

**1. Pre-norm instead of post-norm.** The original block applies LayerNorm *after* the residual add (`x + Sublayer(x)` then norm). Modern blocks normalize *before* the sublayer (`x + Sublayer(Norm(x))`). Pre-norm leaves a clean identity path through the residual, which keeps gradients well-scaled in deep stacks and removes the need for the famous Transformer learning-rate warmup.

**2. RMSNorm instead of LayerNorm.** LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm only divides by the root-mean-square — no mean, no bias term:

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \cdot g.
$$

It saves a reduction and a parameter per layer with no measurable quality loss.

**3. SwiGLU instead of GELU.** The standard FFN is `Linear → GELU → Linear`. SwiGLU adds a gating linear:

$$
\mathrm{SwiGLU}(x) = \big(\mathrm{Swish}(W_1 x) \odot (W_3 x)\big) W_2.
$$

The element-wise product gives the FFN a multiplicative interaction, which empirically improves perplexity by 1–2% at iso-parameters. To keep the parameter budget the same as a vanilla FFN, the hidden dimension is shrunk by a factor of $2/3$.

**4. RoPE instead of learned absolute positions.** Position is injected by *rotating* the query and key vectors at attention time, not added to the embedding. We expand on this in the next section.

**5. GQA / MQA instead of MHA.** A subset of query heads share each KV head. We expand on this in the attention-variants section.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.g

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # value
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down-proj
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LlamaBlock(nn.Module):
    """Pre-norm, RMSNorm, SwiGLU, with attention pluggable from below."""
    def __init__(self, d_model, attn, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = attn
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, d_ff)
    def forward(self, x, **kw):
        x = x + self.attn(self.norm1(x), **kw)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## KV cache: the data structure that pays for long contexts

![Figure 2 — KV cache turns O(n²) prefix recompute into O(n)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig2_kv_cache.png)

In autoregressive decoding, each generation step appends one token. Naïvely, computing attention for the new position requires the keys and values of *all previous tokens* — and re-projecting them from scratch every step costs $O(n)$ work per step, $O(n^2)$ in total. The KV cache observes that those projections are deterministic functions of past tokens, so they can be computed once and stored.

Two facts make the cache work:

- The model is **causal**, so a previously written cache entry never needs to change when new tokens arrive.
- The K and V projections are **linear** in the input, so caching the post-projection tensor is mathematically identical to recomputing it.

```python
class KVCache:
    """Per-layer KV cache. Layout: [B, H_kv, T, D_h]."""
    def __init__(self, B, H_kv, D_h, max_T, device, dtype=torch.float16):
        self.k = torch.empty(B, H_kv, max_T, D_h, device=device, dtype=dtype)
        self.v = torch.empty(B, H_kv, max_T, D_h, device=device, dtype=dtype)
        self.t = 0                                     # current length

    def append(self, k_new, v_new):
        T_new = k_new.size(2)
        self.k[:, :, self.t : self.t + T_new] = k_new
        self.v[:, :, self.t : self.t + T_new] = v_new
        self.t += T_new
        return self.k[:, :, : self.t], self.v[:, :, : self.t]

def attention_step(q_new, cache: KVCache, k_new, v_new, scale):
    """One decode step: q_new is [B, H_q, 1, D_h]."""
    K, V = cache.append(k_new, v_new)                  # full prefix
    scores = (q_new @ K.transpose(-2, -1)) * scale     # [B, H, 1, T]
    attn   = scores.softmax(-1)
    return attn @ V                                    # [B, H, 1, D_h]
```

The cost is real: cache memory grows linearly with sequence length and is the dominant memory term during decoding. For a LLaMA-2-70B-shape model with 80 layers, 64 KV heads, head_dim=128, fp16, vanilla MHA needs

$$
2 \cdot 80 \cdot 64 \cdot 128 \cdot 2\text{ B} = 2.6\text{ MB per token},
$$

so a 32K context would cost 84 GB just for the cache — more than the weights themselves. This is the pressure that motivated GQA, MQA, and PagedAttention (vLLM).

---

## Position encoding: sinusoidal, RoPE, ALiBi

![Figure 3 — Sinusoidal vs RoPE vs ALiBi](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig3_position_encoding.png)

Self-attention has no built-in notion of order: permuting the inputs permutes the outputs. The three dominant solutions inject position in three different places.

**Sinusoidal absolute (Vaswani et al., 2017).** A fixed $\sin / \cos$ vector is *added* to the token embedding at the input layer. Information about position has to survive every subsequent linear projection. Works fine in-distribution but extrapolates poorly past the training length.

**RoPE — rotary position embedding (Su et al., 2021).** Instead of adding to the embedding, RoPE *rotates* the Q and K vectors at attention time. Pair the $d$ head dimensions into $d/2$ 2D planes. In plane $i$, define a frequency $\theta_i = 10000^{-2i/d}$, and at position $m$ rotate the $i$-th plane by angle $m\theta_i$. The key identity is that

$$
\langle R_m q,\; R_n k \rangle = \langle q,\; R_{n-m} k \rangle,
$$

so the dot product depends only on the *relative* offset $n - m$. This is why RoPE generalizes to longer contexts than seen in training, and why every modern model (LLaMA, Qwen, Mistral, Yi, DeepSeek, GPT-NeoX) uses it.

```python
def precompute_rope(d_head, max_T, base=10000.0, device="cpu"):
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_T, device=device).float()
    angles = torch.outer(t, freqs)                     # [T, d_head/2]
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    """x: [B, H, T, D_h]. Rotate dim-pairs (i, i+half)."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[: x.size(-2)].to(x.dtype)
    sin = sin[: x.size(-2)].to(x.dtype)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)
```

**ALiBi — attention with linear biases (Press et al., 2021).** Skip position embeddings entirely; instead, add a per-head linear penalty $-m_h \cdot |i - j|$ to the pre-softmax score. Heads with small $m_h$ behave globally; heads with large $m_h$ focus locally. ALiBi extrapolates farther than RoPE in the original paper but injects no relative-phase information, so it tends to lose to RoPE on knowledge-intensive long-context benchmarks. Used by BLOOM and MPT.

For long-context post-training, RoPE has practical knobs (NTK-aware scaling, YaRN, position interpolation) that let a model trained at 4K extend to 32K–128K with mild fine-tuning, which is now the dominant approach.

---

## Attention variants: MHA → GQA → MQA

![Figure 4 — MHA vs MQA vs GQA](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig4_attention_variants.png)

Standard multi-head attention (MHA) keeps a separate K and V projection per query head. The KV cache cost is proportional to the number of KV heads, which dominates inference memory for long contexts (see the 84 GB number above). Two compromises:

- **MQA — multi-query attention (Shazeer, 2019).** All query heads share a *single* KV head. KV cache shrinks by $H$×, but quality drops noticeably on hard tasks because the model loses head diversity in the K/V projection.
- **GQA — grouped-query attention (Ainslie et al., 2023).** The middle ground used by LLaMA-2-70B, Mistral, and most recent models. Group $H_q$ query heads into $G$ groups; each group shares one KV head. With $H_q = 64$ and $G = 8$ (LLaMA-2-70B), the cache shrinks 8× while quality matches MHA within noise.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.h_q  = n_q_heads
        self.h_kv = n_kv_heads
        self.rep  = n_q_heads // n_kv_heads
        self.d_h  = d_model // n_q_heads
        self.wq = nn.Linear(d_model, n_q_heads  * self.d_h, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_h, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_h, bias=False)
        self.wo = nn.Linear(n_q_heads * self.d_h, d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h_q,  self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h_kv, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h_kv, self.d_h).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(k, v)
        # Broadcast each KV head to its query group.
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

**Numbers, LLaMA-2-70B-shape, fp16, per-token cache:**

- MHA, 64 KV heads → 2.56 MB/token  → 80 GB at 32K context.
- GQA-8, 8 KV heads → 0.32 MB/token → 10 GB at 32K context.
- MQA, 1 KV head → 0.04 MB/token → 1.25 GB at 32K context.

GQA-8 captures most of MQA's memory savings without measurable quality loss, which is why every recent open-weight model defaults to it.

---

## FlashAttention: same math, IO-aware schedule

![Figure 5 — FlashAttention memory hierarchy](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig5_flash_attention.png)

A vanilla attention kernel computes $S = QK^\top$, materializes the full $n \times n$ score matrix in HBM (the GPU's main memory), runs a softmax in HBM, then multiplies by $V$. For $n = 8192$ in fp16, $S$ alone is 128 MB *per head per layer* and most of that traffic is wasted: every entry of $S$ is read and written multiple times.

FlashAttention (Dao et al., 2022) is the same math with a different schedule:

1. Split $Q$, $K$, $V$ into row/column tiles that fit in on-chip SRAM (~192 KB per SM on A100/H100).
2. Compute one $S$ tile at a time inside SRAM.
3. Run softmax with the **online softmax trick** — keep a running max $m$ and a running denominator $\ell$ so each tile updates the partial output without ever seeing the whole row.
4. Write only the final $O$ back to HBM. The full $S$ never appears.

The result is exact (numerically identical up to floating-point reordering) attention with $O(n)$ HBM traffic instead of $O(n^2)$, giving 2–4× wall-clock speedup at $n \geq 2048$ and up to 8× memory reduction. FlashAttention-2 added better warp-level work partitioning, reaching ≈70% of an A100's peak FP16 throughput.

```python
# In modern transformers/torch, you don't write the kernel — you opt in:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",   # also: "sdpa", "eager"
    torch_dtype=torch.float16,
    device_map="auto",
)
# Or use PyTorch's built-in scaled_dot_product_attention, which dispatches to
# FlashAttention / mem-efficient / math kernels automatically.
```

The takeaway is that FlashAttention does not change what your model computes — only the order of operations on the GPU. Always enable it for sequence lengths above ~1K.

---

## Mixture of Experts: capacity without compute

![Figure 6 — Sparse MoE: top-k routing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig6_moe.png)

A dense Transformer spends most of its parameters and most of its FLOPs in the FFN. MoE replaces the single FFN with $N$ "expert" FFNs and a tiny router that, *per token*, picks the top-$k$ experts to run:

$$
y = \sum_{i \in \mathrm{TopK}(W_g x)} g_i(x)\, E_i(x),\qquad g(x) = \mathrm{softmax}(W_g x).
$$

Per-token FLOPs grow with $k$ (typically 2), not $N$ — so an 8-expert model has roughly $8\times$ the FFN parameters of a dense model but spends only $2 \times$ the FFN compute. Mixtral 8×7B has 47B total parameters but activates ~13B per token, giving 70B-class quality at 13B-class inference cost.

The honest cost is elsewhere:

- **Memory.** All experts must be resident even though only $k$ run. Mixtral 8×7B needs ~94 GB in fp16 — more than a dense 70B, ironically — though INT4 brings it under 24 GB.
- **Load balancing.** Without help, the router collapses to a few favored experts. Real MoE training adds an auxiliary load-balancing loss (Shazeer 2017, Switch Transformer) plus a small router noise term to encourage exploration.
- **All-to-all communication.** In multi-GPU training each token must travel to its chosen experts and back, so MoE is sensitive to interconnect topology.

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.k = k

    def forward(self, x):
        logits = self.gate(x)                                   # [*, N]
        topv, topi = logits.topk(self.k, dim=-1)
        weights = topv.softmax(-1)                              # renormalize
        return weights, topi                                    # [*, k], [*, k]

class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff) for _ in range(n_experts)]
        )
        self.router = TopKRouter(d_model, n_experts, k)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)
        w, idx = self.router(flat)                              # [BT, k]
        out = torch.zeros_like(flat)
        for slot in range(self.router.k):
            for e in range(len(self.experts)):
                mask = idx[:, slot] == e
                if mask.any():
                    out[mask] += w[mask, slot, None] * self.experts[e](flat[mask])
        return out.view(B, T, D)
```

(Production MoE implementations dispatch tokens to experts in parallel via grouped GEMM and capacity-limited buckets — the loop above is for clarity, not speed.)

---

## Quantization: fewer bits per weight

![Figure 7 — Quantization: FP16 → INT8 → INT4](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/llm-architecture-deep-dive/fig7_quantization.png)

Modern LLM weights are trained in BF16 (2 bytes / parameter), so a 70B model is 140 GB — too large for a single A100 80 GB or H100 80 GB. Quantization replaces each weight with a low-bit integer plus a per-block scale, shrinking the footprint at small accuracy cost.

**Symmetric INT8.** Pick a per-tensor (or per-channel) scale $s = \max|w| / 127$, store $\hat w = \mathrm{round}(w / s) \in [-127, 127]$. At compute time recover $w \approx s \cdot \hat w$. Memory drops 2×, and on an INT8 tensor core throughput roughly doubles too.

```python
def quantize_int8_per_channel(W):
    """W: [out, in]. Returns int8 weight + per-row fp16 scale."""
    scale = W.abs().amax(dim=1, keepdim=True) / 127.0          # [out, 1]
    Wq = (W / scale).round().clamp(-127, 127).to(torch.int8)
    return Wq, scale.to(torch.float16)
```

**INT4 with GPTQ (Frantar et al., 2022).** Quantizing each weight independently to 4 bits would crater accuracy. GPTQ instead quantizes column-by-column and, after each column, *adjusts the remaining un-quantized columns* to compensate for the rounding error, using a Hessian estimated from a small calibration set. The result: 4-bit weights with <1% perplexity loss on 7B+ models.

**INT4 with AWQ (Lin et al., 2023).** AWQ observes that ~1% of weight channels are "salient" — driven by large activation magnitudes — and that protecting *just those* channels (by per-channel scaling, not by keeping them in fp16) recovers most of the quality. AWQ is faster to compute than GPTQ and is the default for many quantized open-weight checkpoints.

| Precision | Bytes / param | LLaMA-2-7B weights | LLaMA-2-70B weights | Typical PPL gap |
|---|---|---|---|---|
| FP16 / BF16 | 2.0 | 13.5 GB | 140 GB | reference |
| INT8 (RTN) | 1.0 | 6.7 GB | 70 GB | <0.5% |
| INT4 (GPTQ / AWQ) | 0.5 | 3.4 GB | 35 GB | <2% |
| INT3 (advanced) | 0.375 | 2.5 GB | 26 GB | 3–6% |

```python
# Loading a pre-quantized AWQ checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ", device_map="auto", torch_dtype="auto",
)
tok = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")
```

Activations are usually left in FP16/BF16 because their distributions are heavy-tailed (LLM.int8(), Dettmers 2022, demonstrated that a few "outlier" features carry most of the activation energy and must be kept in higher precision). Weight-only INT4 is the current sweet spot for inference.

---

## Putting it together: high-throughput inference

The pieces compose. A modern serving stack like **vLLM** combines:

- A LLaMA-style decoder block (pre-norm + RMSNorm + SwiGLU + RoPE + GQA).
- Per-layer KV cache, but laid out as fixed-size **pages** (PagedAttention) so contexts can grow and shrink without fragmentation.
- FlashAttention kernels for both prefill and decode.
- INT4 weight quantization (AWQ / GPTQ) for memory.
- Continuous batching: when a request finishes, its slot is filled by the next prompt mid-batch instead of waiting for the slowest sequence.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
for o in llm.generate(["Explain MoE in two sentences."], params):
    print(o.outputs[0].text)
```

In practice, a single A100 80 GB serves a 7B AWQ model at 3000–5000 tokens/s aggregate throughput, and a single H100 serves a Mixtral 8×7B AWQ at >1000 tokens/s — numbers that were unimaginable with the 2017 block as written.

---

## Frequently asked questions

### Why decoder-only and not encoder–decoder?

A causal decoder-only model trained to predict the next token can be turned into a classifier, a translator, or a chatbot by changing the prompt. Encoder–decoder models need separate cross-attention parameters and a more complex training pipeline, and they don't benefit from cheap prefix caching during generation. Decoder-only also scales more cleanly because every token in the corpus is a training signal.

### RoPE or ALiBi — which one wins in practice?

RoPE for almost everything: it gives the model true relative-phase information and supports clean post-training context extension via NTK / YaRN scaling. ALiBi is appealing if you need extrapolation without any fine-tuning, but at the cost of weaker in-distribution quality. Every leading open-weight LLM today uses RoPE.

### Does FlashAttention change the model output?

No. FlashAttention is exact attention, modulo floating-point reduction order. The numerical differences vs the naive kernel are well below training noise.

### Is MoE always cheaper than dense?

Cheaper in compute, more expensive in memory. If you are GPU-memory-bound (e.g., serving on a single 24 GB card), a quantized dense model usually beats an MoE. If you have multi-GPU memory but are FLOPs-bound at decode time, MoE wins.

### How much accuracy does INT4 cost on a 7B?

With GPTQ or AWQ on a calibration set of ~128 samples, you should see <2% perplexity increase and indistinguishable quality on most downstream tasks. For 70B models the gap is usually <1%. Below 7B, quantization gets noticeably more painful.

### Why is the KV cache the bottleneck for long contexts?

Cache size scales as $2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot \mathrm{bytes}$. For a 70B-class model at 32K tokens with MHA, that's ~80 GB — bigger than the model weights. GQA brings it to ~10 GB, and PagedAttention keeps fragmentation under 5%, which together unlock long-context serving on commodity hardware.

---

## Series navigation

- **Previous**: [Part 8 — Model Fine-tuning and PEFT](/en/nlp-fine-tuning-peft/)
- **Next**: [Part 10 — RAG and Knowledge Enhancement](/en/nlp-rag-knowledge-enhancement/)
- [View all 12 parts in the NLP series](/tags/NLP/)
