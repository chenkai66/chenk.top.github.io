---
title: "LLM Engineering (1): Architectures from Transformer to MoE"
date: 2026-03-27 09:00:00
tags:
  - LLM
  - Transformer
  - MoE
  - Architecture
  - mamba
categories: LLM Engineering
series: llm-engineering
series_order: 1
series_total: 12
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "MHA → GQA → MQA, sparse MoE routing in Mixtral and Qwen3-MoE, sliding-window attention, and the state-space alternatives Mamba and RWKV — what each costs and where each wins."
translationKey: "llm-engineering-1"
---

The 2017 Transformer block is still the silhouette of every production LLM in 2026, but almost every internal piece has been swapped, sparsified, or specialized. This series covers the modern stack end to end — architecture, training, inference, retrieval, evaluation, safety, deployment. Chapter 1 is about the block itself: what attention looks like in a 2026 model, how MoE breaks the param-FLOPs link, and where the non-attention alternatives (Mamba, RWKV) actually beat the Transformer.

I'll assume you know the original Transformer block. If you don't, the [NLP series part 4](/en/nlp/attention-transformer/) covers it. This chapter is what's *different* now.

![LLM Engineering (1): Architectures from Transformer to MoE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/illustration_1.png)

---

## What changed and why

![fig5: architecture timeline 2017-2026](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/fig5_timeline.png)


A modern decoder block — LLaMA-3, Qwen3, Mistral, DeepSeek-V3, Yi — looks like this:

```python
# pseudocode for a single decoder layer
def layer(x, kv_cache):
    h = x + attention(rms_norm(x), kv_cache)   # pre-norm + RMSNorm
    h = h + ffn_or_moe(rms_norm(h))             # SwiGLU FFN, sometimes routed
    return h
```

Five swaps relative to "Attention Is All You Need" [Vaswani et al., 2017]:

1. **Pre-norm** instead of post-norm — gradients flow through a clean residual identity, no warmup needed. The original post-norm Transformer needed careful learning-rate warmup (~10K steps) to avoid the first few gradient updates blowing up the norm-then-residual path. Pre-norm, popularized by GPT-2 and rigorously analyzed by [Xiong et al., 2020], removes that requirement and trains stably from step zero. Every production LLM since 2020 uses pre-norm.
2. **RMSNorm** instead of LayerNorm — drop the mean, keep the RMS divisor. One reduction less per layer. [Zhang & Sennrich, 2019] showed RMSNorm matches LayerNorm quality at ~7-64 % faster wall-clock on transformer FFNs. T5 and the entire LLaMA lineage adopted it; in 2026, only a handful of legacy architectures still use mean-centering.
3. **SwiGLU** instead of GELU — gated FFN, ~2-3 % perplexity win, +50 % FLOPs in the FFN that turn out to be worth it. The "GLU variants" paper by [Shazeer, 2020] systematically swept gated activations; SwiGLU (Swish gating) won on nearly every benchmark. Convention: shrink the FFN inner dim by 2/3 to keep total FFN params constant, since GLU triples the projection count.
4. **RoPE** instead of sinusoidal positions — [Su et al., 2021] proposed rotary embeddings that encode relative position by rotating Q and K vectors in 2D subspaces. RoPE plus context extension tricks (NTK scaling, YaRN) is how every long-context model in 2026 reaches 128K-1M tokens. [Chapter 6](/en/llm-engineering/06-long-context/) covers this in depth.
5. **GQA / MQA** instead of MHA — smaller KV cache, same quality. Critical at long context.

The dense FFN is also increasingly *replaced* by a sparse mixture of experts (MoE), which is the biggest architectural change of the last three years and the focus of half this chapter.

A historical note worth holding in mind: the *names* of these techniques (pre-norm, RMSNorm, SwiGLU, RoPE, GQA) suggest a tidy progression, but every one of them was contested for at least a year after its first paper. Pre-norm vs post-norm debate ran 2018-2020. RoPE vs ALiBi vs NoPE [Press et al., 2022; Kazemnejad et al., 2023] ran 2022-2024 — and ALiBi still wins at extreme length-extrapolation in some studies. The "settled" architecture in 2026 is settled because it ships, not because the math is final.

## The attention math, properly

The scaled dot-product attention from [Vaswani et al., 2017]:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
Three things in this formula deserve more than the standard one-line treatment.

**Why $\sqrt{d_k}$ in the denominator.** $Q$ and $K$ are projections of a unit-variance input through linear layers initialized so each column has unit variance. The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ is then a sum of $d_k$ unit-variance terms, so its variance is $d_k$ and its standard deviation is $\sqrt{d_k}$. Without the divisor, dot products grow as $d_k$ grows, the softmax saturates at one row, and the gradient through the softmax vanishes (the Jacobian of softmax at a one-hot input is zero). Dividing by $\sqrt{d_k}$ keeps the pre-softmax logits at unit variance regardless of head dimension. This is the single most important detail in the entire Transformer paper — the model does not train without it.

**Why softmax and not something else.** Softmax is differentiable, normalized to a probability simplex, and gives a sharp focus when one logit dominates. It's also the bottleneck. Linear attention [Katharopoulos et al., 2020] replaces $\text{softmax}(QK^\top)V$ with $(\phi(Q)\phi(K)^\top)V$ for a feature map $\phi$, then uses associativity to compute $\phi(K)^\top V$ first — this changes the cost from $O(n^2 d)$ to $O(n d^2)$. Linear attention sounds great until you measure. It loses 2-5 perplexity points consistently because the implicit kernel is poorly matched to language. The cleanest study, [Schlag et al., 2021], showed linear attention behaves like a fixed-capacity associative memory and saturates well before scaled-dot-product does.

**Why FlashAttention is not just an optimization.** [Dao et al., 2022] showed that the $O(n^2)$ memory of vanilla attention isn't from the FLOPs — it's from materializing the $n \times n$ score matrix. FlashAttention tiles $Q$, $K$, $V$ into blocks that fit in SRAM, computes per-block softmax with online numerically-stable normalization, and never writes the full score matrix to HBM. The result: 2-4× wall-clock speedup at training and a memory drop from $O(n^2)$ to $O(n)$. FlashAttention-2 [Dao, 2023] reorganized the work to better overlap matmul and reduction; FlashAttention-3 [Shah et al., 2024] added asynchronous warp-specialized scheduling for H100. Every production LLM training framework in 2026 (PyTorch SDPA, JAX TPU attention, vLLM, SGLang) calls a FlashAttention-derived kernel.

The online softmax trick at the heart of FlashAttention is worth understanding. Standard softmax requires two passes over the row: one for $\max$ and one for $\sum e^{x - \max}$. FlashAttention does it in one pass by maintaining a running $(m_t, \ell_t)$ pair — the running max and the running sum-of-exps — and updating them as new blocks arrive. When you see a new max $m_{t+1} > m_t$, you rescale the existing sum: $\ell_{t+1} = e^{m_t - m_{t+1}} \ell_t + e^{x - m_{t+1}}$. This is the same trick used in numerically stable streaming statistics, applied to the softmax inside attention. The output is bit-exact to the un-tiled reference within the limits of fp32 accumulation.

## GQA, MQA, MHA: the real cost of the KV cache

![fig1: MHA → GQA → MQA head sharing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/fig1_attention_heads.png)


Multi-head attention (MHA) projects every token to $h$ separate Q, K, V vectors of dimension $d_{\text{head}}$. Multi-query attention (MQA) projects $h$ Q vectors but only **one** K and one V, shared across heads. Grouped-query attention (GQA) is the middle: $g$ groups, each sharing one K/V across $h/g$ heads.

The KV cache memory is what matters at long context. For LLaMA-3-70B with $h=64$, $d_{\text{head}}=128$, $L=80$ layers, FP16:
$$\text{KV bytes per token} = 2 \cdot L \cdot 2 \cdot h_{\text{kv}} \cdot d_{\text{head}}$$
For a 32K-token context:

| Variant | $h_{\text{kv}}$ | KV / token | KV / 32K context |
|---|---|---|---|
| MHA | 64 | 32 KB | 1.0 GB |
| GQA-8 | 8 | 4 KB | 128 MB |
| MQA | 1 | 0.5 KB | 16 MB |

GQA-8 is the dominant choice in 2026. It keeps almost all of MHA's quality (the GQA paper [Ainslie et al., 2023] reports under 0.5 % degradation on most tasks) and gives an 8× KV cache reduction. MQA was tried in early models (PaLM, Falcon-7B) but loses too much quality at scale, especially for long context where head diversity matters.

A LLaMA-3-style attention block with GQA:

```python
import torch
import torch.nn.functional as F

class GQA(torch.nn.Module):
    def __init__(self, d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128):
        super().__init__()
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, k_cache=None, v_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        # repeat K, V to match Q heads (n_heads / n_kv groups share)
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
```

The `repeat_interleave` is conceptual — actual kernels (FlashAttention-2, FlexAttention) handle the broadcast inside the SRAM tile without materializing the repeated K/V.

**Why GQA-8 specifically.** The choice of group count is empirical, not theoretical. Ainslie et al. swept $g \in \{1, 2, 4, 8, 16, 64\}$ on T5-XXL and found that quality drops sharply between $g=8$ and $g=4$, then again between $g=2$ and $g=1$ (MQA). At $g=8$ the validation perplexity differential to full MHA was 0.05 — within run-to-run noise. At $g=1$ it was 0.4 — large enough to notice on downstream tasks. LLaMA-2-70B picked $g=8$ from this paper directly; LLaMA-3 kept it; Qwen3 kept it. Eight is not magic — it's the experimental knee of the curve.

**Multi-query latent attention (MLA).** DeepSeek-V2 [DeepSeek-AI, 2024] introduced a different KV-saving idea. Instead of duplicating K/V across heads, project tokens into a small *latent* dimension $d_c \ll h \cdot d_{\text{head}}$ (typically $d_c = 512$), cache only that latent vector, and re-project to per-head K and V on the fly. KV cache drops to roughly $d_c$ bytes per token per layer — even smaller than MQA — while attention quality matches MHA. The cost is more compute at attention time, but for memory-bound long-context inference that trade is favorable. DeepSeek-V3 ships MLA in production. As of 2026, MLA is the architectural choice on the open frontier for any new 100B+ model targeting long context.

**KV-cache compression via quantization.** Orthogonal to the architecture choice, you can store the KV cache in INT8 or INT4 instead of FP16. KIVI [Liu et al., 2024] and FP8-KV showed INT4 KV with per-channel asymmetric quantization is effectively lossless on most benchmarks down to a 4× memory cut. Combined with GQA-8 you get 32× total KV reduction vs MHA-FP16. vLLM, SGLang, and TensorRT-LLM all ship INT8/FP8 KV cache as an option in 2026.

## Sliding window attention

Mistral-7B introduced **sliding window attention (SWA)** with window 4096 [Jiang et al., 2023]. Each token only attends to the previous $w$ tokens. The receptive field still grows linearly with depth — a token at position $n$ in layer $L$ sees back $L \cdot w$ tokens — so a 32-layer model with $w=4096$ has an effective receptive field of 131K tokens.

SWA bounds the KV cache at $w$ tokens regardless of context length. The catch: the model still needs to learn to use the layered receptive field, which means SWA models often underperform full-attention models on tasks that need precise long-range retrieval ("needle in a haystack"). Most 2026 long-context models combine SWA with **attention sinks** [Xiao et al., 2024] (Mistral-Large, Qwen3) — keeping the first 4-8 tokens in every layer's attention window, which dramatically stabilizes long-context behavior. [Chapter 6](/en/llm-engineering/06-long-context/) covers this.

The attention sink phenomenon is one of the strangest empirical observations in modern LLMs. Xiao et al. observed that streaming inference (where you slide a fixed window forward and drop tokens that fall off the back) catastrophically diverges after a few thousand tokens — perplexity rises to thousands. The fix turned out to be trivial: never evict the first 4 tokens. With sinks preserved, perplexity stays flat out to millions of tokens. The mechanistic explanation is that softmax always sums to 1, so any "extra" attention mass that doesn't have a meaningful target gets dumped on the first few tokens during training. Evicting those tokens removes the dump site and the attention distribution explodes. Modern models are pretrained with sinks reserved explicitly to avoid this fragility.

## Mixture of experts: more parameters, same FLOPs

![LLM Engineering (1): Architectures from Transformer to MoE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/illustration_2.png)


![fig3: sparse MoE vs dense compute](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/fig3_sparse_vs_dense.png)


![fig2: MoE top-2 routing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/fig2_moe_routing.png)


The fundamental MoE trick: replace the dense FFN with $E$ FFNs (the "experts") plus a small router that picks $k$ of them per token. Total parameters scale with $E$; per-token FLOPs scale with $k$.

The lineage of MoE in deep learning starts much earlier than the modern wave. [Jacobs et al., 1991] proposed "mixture of experts" as an ensemble idea. [Shazeer et al., 2017] scaled it to 137B-param language models with sparse top-$k$ gating — that paper is the direct ancestor of every current sparse MoE design. GShard [Lepikhin et al., 2020] and Switch Transformer [Fedus et al., 2022] productized expert parallelism on TPUs. The 2024-2026 wave (Mixtral, DeepSeek, Qwen3) is the third generation: top-$k$ routing with auxiliary-loss-free balancing, fine-grained experts (more, smaller experts), and shared-expert designs.

Mixtral-8x7B is the textbook example. It has 8 experts per layer, top-2 routing, and 32 layers. Total params: 46.7B (not 56B — attention is shared). Active params per token: 12.9B. So you pay 12.9B-FLOPs of compute for the modeling capacity of a 46.7B model. That's a 3.6× parameter-to-compute ratio.

Qwen3-235B-A22B (released late 2025) pushes this further: 235B total, 22B active, ratio 10.7×. DeepSeek-V3 [DeepSeek-AI, 2024] has 671B total / 37B active, ratio 18× — the highest ratio shipped to date.

A minimal MoE FFN looks like this:

```python
class MoEBlock(torch.nn.Module):
    def __init__(self, d_model, d_ffn, n_experts, top_k):
        super().__init__()
        self.gate = torch.nn.Linear(d_model, n_experts, bias=False)
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model, d_ffn) for _ in range(n_experts)
        ])
        self.top_k = top_k

    def forward(self, x):                        # x: [B, T, d]
        scores = self.gate(x)                    # [B, T, E]
        topk_w, topk_i = scores.topk(self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_i[..., k]                 # [B, T]
            w   = topk_w[..., k:k+1]             # [B, T, 1]
            for e in range(len(self.experts)):
                mask = (idx == e)
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out
```

Two things this naive code hides that matter in production:

**Load balancing.** If the router always picks the same expert, you get a dense model with extra parameters wasted. The fix is an auxiliary loss that penalizes unbalanced routing — Switch Transformer's $\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_e f_e \cdot p_e$ where $f_e$ is the fraction of tokens routed to expert $e$ and $p_e$ is the average router probability. DeepSeek-V3 ditched the aux loss for an **auxiliary-loss-free** balancing scheme [Wang et al., 2024]: a per-expert bias $b_e$ added to the gate logits, updated by gradient descent to equalize $f_e$. Cleaner, no quality penalty.

**Expert parallelism.** With 256 experts (DeepSeek-V3) you can't fit them all on one GPU. Experts are sharded across GPUs and tokens are routed there via all-to-all. The all-to-all latency is the bottleneck — DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP) gets 156 GB/s on NVLink which is close to the hardware ceiling.

The sharp edge of MoE: total VRAM scales with total params, even though only $k$ are active. You can't run DeepSeek-V3 on a single 80 GB H100 — you need ~700 GB of weight memory. Active-param-equivalence is a *compute* claim, not a memory one.

## MoE math: routing, capacity, and balancing in detail

The router is a linear map $g(x) = W_g x \in \mathbb{R}^E$ followed by softmax over a top-$k$ subset. Top-$k$ specifically — not "pick all experts above a threshold," because the FLOPs per token must be deterministic for the kernels to work. Mixtral uses $k=2$, DeepSeek-V3 uses $k=8$ out of 256 experts plus 1 always-on shared expert, Qwen3-MoE uses $k=8$ out of 128.

The Switch Transformer auxiliary loss decomposes into two factors. Let $T$ be the batch of tokens, $E$ the number of experts. Define $f_e = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}[\arg\max_e g(x_t) = e]$ (the fraction of tokens whose top-1 went to expert $e$) and $p_e = \frac{1}{|T|} \sum_{t \in T} \text{softmax}(g(x_t))_e$ (the average router probability for expert $e$). The loss is
$$\ell_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot p_e.$$
This is minimized when both $f_e = 1/E$ and $p_e = 1/E$ for all $e$. The clever part is using both quantities: $f_e$ alone has a non-differentiable arg-max, $p_e$ alone is differentiable but doesn't penalize hard imbalance. Multiplying makes the router learn to be balanced both in soft probability and in the actual hard assignment.

**Expert capacity** is the maximum tokens any one expert can receive in a batch. If capacity is $c \cdot |T| \cdot k / E$ (capacity factor $c$), tokens routed to a full expert are *dropped* — they bypass the FFN with a zero contribution. Switch Transformer used $c=1.25$. Lower $c$ saves memory and all-to-all bandwidth but drops more tokens; higher $c$ avoids drops but wastes capacity. Modern training (DeepSeek-V3, Qwen3) often uses $c=1.0$ during training plus aux-loss-free balancing, then $c=\infty$ at inference (which is fine because batch sizes are small enough that capacity is rarely hit).

The auxiliary-loss-free approach from [Wang et al., 2024], adopted by DeepSeek-V3, replaces the aux loss with a per-expert bias added to the routing logits before top-$k$:
$$g'_e(x) = g_e(x) + b_e,\qquad b_e \leftarrow b_e - \eta \cdot \text{sign}(f_e - 1/E)$$
The bias updates after each batch to push tokens toward under-utilized experts. The softmax weights used for the convex combination of expert outputs still come from the un-biased $g_e(x)$ — only the assignment uses $g'_e$. This decouples "which expert" from "how strongly" and avoids the quality penalty that aux-loss imposes when it fights the model's natural routing preference.

## Mixtral vs Qwen3-MoE vs DeepSeek-V3 architecture comparison

Three sparse-MoE designs, three different parameter-vs-compute trade-offs:

| Property | Mixtral 8x7B | Qwen3-235B-A22B | DeepSeek-V3 |
|---|---|---|---|
| Total params | 46.7B | 235B | 671B |
| Active params | 12.9B | 22B | 37B |
| Sparsity ratio | 3.6× | 10.7× | 18.1× |
| Layers | 32 | 94 | 61 |
| Experts per layer | 8 | 128 | 256 + 1 shared |
| Top-$k$ | 2 | 8 | 8 |
| Expert size (FFN inner) | 14336 | 1536 | 2048 |
| Attention | GQA-8 | GQA-8 | MLA |
| Balancing | aux loss | aux-loss-free | aux-loss-free |
| Tokenizer vocab | 32K | 152K | 129K |

The trend is clear: more, smaller experts and fewer top-$k$ activations relative to total. Mixtral's "8 large experts, pick 2" is the OG sparse design. DeepSeek-V3's "256 small experts, pick 8 + 1 always-on" is the current frontier. The intuition is that fine-grained experts allow more specialization; the always-on shared expert captures common patterns that don't need routing (saving the routed experts' capacity for genuinely specialized work).

The "shared expert" idea from DeepSeek-V2 deserves a paragraph. Without it, frequent boring patterns (English function words, common code idioms) compete with rare specialized patterns for routing slots, and the router has to learn a wasteful mapping for each. With it, the shared expert gobbles the universal patterns, and the routed experts specialize. Empirically this halves the routing entropy (more decisive routing) while improving downstream quality 1-2 % on average.

Mixtral's "8 experts, top-2" was chosen partly for inference efficiency on a single 8-GPU node — one expert per GPU, top-2 means each token activates 2 GPUs in all-to-all. DeepSeek's "256 experts, top-8" requires more sophisticated expert parallelism but spreads load better and gets the higher sparsity ratio. The architectures encode different inference deployment assumptions.

## Hybrid architectures: Jamba, Zamba, Samba

Pure attention is $O(n^2)$. Pure state-space (Mamba) is $O(n)$ but loses on copy-style tasks. The natural answer is to mix.

**Jamba** [Lieber et al., 2024] is the first widely-deployed hybrid. The Jamba block alternates: 7 Mamba layers, 1 attention layer, repeating. Of the 32 layers in Jamba-1.5-Large, 4 are attention and 28 are Mamba. Plus MoE on top — the FFNs are sparse with 16 experts and top-2 routing. The result: 398B total params, 94B active per token, 256K context window, ~5× faster inference at long context than a comparable dense Transformer.

**Zamba** [Glorioso et al., 2024] interleaves a single shared attention block among many Mamba blocks. The shared attention amortizes the parameter cost — instead of N attention layers, you have N references to one attention block. Zamba-7B-v2 uses Mamba-2 layers with one attention layer shared across the network at multiple depths. This pattern saves 30 % parameters at the cost of slightly more compute (the shared block runs N times).

**Samba** [Ren et al., 2024] is the most aggressive hybrid: 1:1 alternation between Mamba and sliding-window attention. The 3.8B Samba model claims to match Phi-3-3.8B on most benchmarks while extrapolating cleanly to 1M-token context — something pure Transformers struggle with even with RoPE extension tricks.

The empirical lesson from all three: **a small fraction (10-50 %) of attention layers is enough to recover what pure Mamba loses on copy/lookup tasks**, while still keeping most of the linear-time advantage. The exact ratio is contested. For copy-heavy tasks (in-context learning of new patterns), more attention helps. For general language modeling, less attention is fine.

## State-space models: Mamba and the linear-time alternative

![fig4: state-space vs attention complexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/01-architectures/fig4_mamba_vs_attention.png)


Attention is $O(n^2)$ in sequence length. Linear-time alternatives keep coming back: linear attention [Katharopoulos et al., 2020], Performer [Choromanski et al., 2021], Linformer, Reformer. None of them stuck — they were all worse than vanilla attention at scale.

Then Mamba happened. Mamba [Gu & Dao, 2023] is a **selective state-space model**: each layer maintains a fixed-size hidden state $h_t \in \mathbb{R}^N$ and updates it with a recurrence:
$$h_t = \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t, \quad y_t = C_t \, h_t.$$
The "selective" part: $A$, $B$, $C$ are *input-dependent*, computed from $x_t$. This was the missing piece — earlier SSMs (S4 [Gu et al., 2022]) had time-invariant dynamics and couldn't do content-based memory. Mamba can.

Mamba-2 [Dao & Gu, 2024] made $\bar{A}_t$ scalar-times-identity, which lets the recurrence be expressed as a structured matmul (the State Space Duality, SSD) and run efficiently on GPUs. At 2.7B params it matches Pythia-2.8B perplexity with 5× faster inference and constant memory per token — no KV cache.

What Mamba is not: a Transformer killer. The Jamba paper [Lieber et al., 2024] and several subsequent hybrids (Zamba, Samba, Falcon-Mamba) found that **pure Mamba underperforms on in-context learning and copy tasks** — specifically, anything that needs to look up an exact token from earlier in the sequence. The fix is to interleave a few attention layers among many Mamba layers. Jamba is roughly 1 attention layer per 7 Mamba layers; Samba is 1:1.

The mechanistic reason Mamba struggles with copy is that its hidden state has fixed dimension $N$ (typically 64-128). To copy a token from 5000 positions back, the model must compress and route the relevant token into the hidden state and carry it forward without overwriting it for 5000 steps. Attention sidesteps this by recomputing similarity against all past tokens at every step. [Jelassi et al., 2024] proved formally that Mamba cannot solve associative-recall problems beyond a sequence length proportional to its hidden state size, while attention can.

In 2026, hybrid architectures are the practical state of the art for long context. Pure Transformers (Qwen3, GPT-4o, Claude-4.5) still dominate the general-purpose LLM market, but for >256K context windows, hybrid Mamba-attention models (Jamba-1.5-Large, Falcon3-Mamba) are competitive at 5-10× lower inference cost.

## RWKV: the third path

RWKV [Peng et al., 2023] is a recurrent network designed to be parallelizable like a Transformer at training time. It uses a time-mixing block (linear attention with exponential decay) and a channel-mixing block (gated FFN). RWKV-7 (2025) introduced "Goose" — a learned dynamic state evolution that closes most of the quality gap with attention.

I include RWKV here for completeness. In practice, every team I've seen ship a non-Transformer LLM in the last 12 months has gone with Mamba-2 hybrids. RWKV has a smaller community and weaker tooling. If you're building production, default to attention; consider Mamba hybrids for niche long-context workloads; treat RWKV as research.

## Worked example: KV cache and FLOPs for a 70B model at 32K context

Let me put concrete numbers on what an attention block does during a single decoding step. LLaMA-3-70B, GQA-8, $h=64$, $d_{\text{head}}=128$, $L=80$, $d_{\text{model}}=8192$, vocab $V=128256$, FFN inner $d_{\text{ffn}}=28672$ (SwiGLU triple-projection equivalent of ~57K standard FFN).

**Per-layer parameter count.**

- Attention QKVO projections: $d_{\text{model}} \cdot (n_{\text{heads}} \cdot d_{\text{head}} + 2 \cdot n_{\text{kv}} \cdot d_{\text{head}} + d_{\text{model}}) = 8192 \cdot (8192 + 2048 + 8192) = 152 \text{M}$
- FFN SwiGLU (gate + up + down): $3 \cdot d_{\text{model}} \cdot d_{\text{ffn}} = 3 \cdot 8192 \cdot 28672 = 705 \text{M}$
- RMSNorm × 2: negligible (~32K)
- Per-layer total: ~857M

Times 80 layers = 68.6B. Plus embeddings ($V \cdot d_{\text{model}} = 1.05 \text{B}$, tied), plus output norm, totals ~70B. ✓

**Per-token KV cache at 32K.** From the formula above: $2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 = 327{,}680$ bytes per token at FP16, or 4 KB. At 32K tokens that's 128 MB. ✓

**FLOPs per decoded token.**

- Attention QKV proj: $2 \cdot d_{\text{model}} \cdot (n_{\text{heads}} + 2 n_{\text{kv}}) \cdot d_{\text{head}} = 2 \cdot 8192 \cdot 80 \cdot 128 \approx 168 \text{M}$
- Attention compute (one new query against 32K cached K/V): $4 \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot 32{,}768 \approx 1.07 \text{G}$
- O proj: $2 \cdot d_{\text{model}}^2 \approx 134 \text{M}$
- FFN: $2 \cdot 3 \cdot d_{\text{model}} \cdot d_{\text{ffn}} \approx 1.41 \text{G}$
- Per-layer: ~2.78 GFLOPs
- 80 layers: ~222 GFLOPs per decoded token

On an H100 with ~2 TFLOPs/W FP16 effective throughput, that's ~110 ms of pure compute per token — but decoding is *memory-bound*, not compute-bound. The actual bottleneck is reading the 70B param weights from HBM at 3.35 TB/s, which gives a hard floor of ~21 ms per token (one decode requires touching every weight once). Production serving fights this with batching, KV quantization, and speculative decoding (chapter 5).

## Production reality: what frontier labs actually ship

Architecture papers describe the public 1 % of what shipping models do. The other 99 % is plumbing. Three things every frontier lab does that you won't find in a model card:

**Custom CUDA kernels for attention + FFN fusion.** vLLM and SGLang both ship hand-tuned kernels for the LLaMA family that fuse RMSNorm + QKV projection + RoPE + attention + output projection into a small number of kernel launches. The naive PyTorch graph would launch ~20 kernels per layer per token; a fused kernel launches 2-3. At 80 layers and 64-batch decoding, the kernel-launch overhead alone can dominate small models. NVIDIA's TensorRT-LLM goes further and JIT-compiles a per-architecture kernel from a graph IR.

**FP8 attention compute.** H100 and B200 have native FP8 (E4M3 / E5M2) tensor cores. FlashAttention-3 [Shah et al., 2024] runs the QK matmul in FP8 with FP32 accumulation, which doubles throughput vs FP16 attention. The accuracy loss is < 0.1 % perplexity on standard benchmarks; production systems (GPT-4o, Claude-4.5) use FP8 for both prefill and decode. Training in FP8 (NVIDIA's Transformer Engine, [Micikevicius et al., 2022]) is becoming standard for new pretraining runs.

**Per-layer LR scaling and weight tying.** LLaMA-3 paper [Dubey et al., 2024] reveals that the attention output projection and the FFN down projection are initialized with smaller variance than other layers, to avoid early-training instability. Most production training scripts also tie the embedding and LM head matrices, halving 1-2 % of total params. These tweaks are individually small but compound into noticeable training stability and quality differences.

The fact that "Mixtral 8x7B" is one model with one architecture is partly a fiction. The deployed Mistral API runs Mixtral with several inference-time tricks (paged attention, speculative decoding via a small draft model, FP8 KV cache) that aren't part of the architecture proper but do affect quality and latency. When you read benchmark numbers, you're seeing a deployment, not just an architecture.

## Common Pitfalls

Five things I've seen go wrong with these architectures.

**1. Hard-coding $h_{\text{kv}} = 1$ assuming MQA.** A custom training script for a 7B model assumed MQA after copying example code from a tutorial. Quality was 3-5 perplexity points worse than baseline. The fix was switching to GQA-8. If you're starting from scratch, GQA-8 is the safe default.

**2. Forgetting to share K/V across heads in the cache.** I've seen FlashAttention wrappers that allocate KV cache as $[B, T, n_{\text{heads}}, d_{\text{head}}]$ even for GQA models, then do `repeat_interleave` at runtime on the cached K/V. This wastes 8× memory on a GQA-8 model. The cache should be $[B, T, n_{\text{kv}}, d_{\text{head}}]$ and the broadcast should happen inside the attention kernel.

**3. RMSNorm $\epsilon$ too small.** Default $\epsilon = 10^{-6}$ in PyTorch. For FP16 training this can underflow. Use $\epsilon = 10^{-5}$ for FP16, $10^{-6}$ for BF16. We chased a divergence-at-step-12000 bug for two days that was just this.

**4. MoE router gradients silently zeroed.** A common mistake when implementing custom MoE: only the chosen experts' outputs flow gradients to the router. If you compute the router gate and then re-softmax over the top-$k$, you must use the *original* logits in the gradient path, not just the top-$k$ subset. Otherwise the router can never learn to *not* pick a bad expert. Both Mixtral's and DeepSeek-V3's reference implementations do this correctly; rolled-from-scratch ones often don't.

**5. SwiGLU FFN inner dim not adjusted.** A standard Transformer FFN has inner dim $4 \cdot d_{\text{model}}$. Naively swapping to SwiGLU triples the projection count, blowing up params 3×. The convention is to shrink inner dim by 2/3 (so $\approx 2.67 \cdot d_{\text{model}}$) to keep total FFN params roughly equal. Several open-source forks of LLaMA broke this rule and ended up with different param counts than expected.

## Research frontier 2024-2026

What's coming after the current "MoE Transformer with GQA" consensus:

**Differential attention** [Ye et al., 2024] subtracts two attention maps with different parameters. Empirically this suppresses attention noise and improves long-context retrieval. Already showing up in some 2025-2026 model releases.

**Native sparsity in attention.** Native Sparse Attention (NSA) and similar work train the model to attend to a sparse subset of keys natively, rather than retrofitting sparsity after the fact. [Yuan et al., 2025] showed native sparse attention can match dense attention at much lower compute and KV cost on long-context benchmarks.

**Linear attention is back.** Several 2024-2025 papers (Gated Linear Attention, RetNet [Sun et al., 2023], TransNormerLLM) have closed the quality gap with full attention by adding gating and decay. Whether they actually displace softmax attention at production scale is still TBD, but the gap is narrower than it's been in 5 years.

**Diffusion language models.** [Lou et al., 2024] (SEDD) and [Sahoo et al., 2024] (MDLM) showed discrete diffusion can match autoregressive perplexity on text. Mercury Coder, released 2025, claims sub-50ms latency for 1000-token generation via diffusion, which is faster than any autoregressive model can achieve at that length. Whether the quality holds at frontier scale is unclear, but it's the most credible non-autoregressive contender in years.

**Test-time compute scaling.** [Snell et al., 2024] showed that adding inference-time compute (chain-of-thought, self-consistency, MCTS) can substitute for pretraining compute at the same quality. The o1 / DeepSeek-R1 / Claude-thinking line of models takes this seriously: a 32B-param thinking model can outperform a 70B-param non-thinking one on hard reasoning tasks by spending 10× more inference compute. This is not an architectural change but it changes the architecture-vs-cost trade.

## What to use when

| Workload | Architecture | Why |
|---|---|---|
| General-purpose chat, code | Dense Transformer (LLaMA-3, Qwen3-Dense) | Best quality per param, mature tooling |
| Cost-sensitive serving | MoE (Mixtral, DeepSeek-V3, Qwen3-MoE) | 3-10× param-to-FLOPs ratio |
| 256K+ context, low latency | Hybrid Mamba-attention (Jamba) | Constant memory per token |
| Edge inference < 4 GB | Quantized small dense (Qwen3-1.7B INT4, Phi-4-mini) | Memory-bound, MoE doesn't help |
| Reasoning-heavy (math, code) | Dense Transformer + thinking RL | Quality scales with inference compute |

The choice is rarely "pick the best architecture." It's "pick the architecture whose constraints match my serving constraints." MoE wins when you have many GPUs and are FLOPs-bound. Dense wins on a single GPU where total VRAM is the cap. Hybrids win when context is the bottleneck.

## What's Next

Modern LLMs are still Transformers, but the block has been re-engineered piece by piece for stability (pre-norm, RMSNorm), quality (SwiGLU, RoPE), inference cost (GQA, sliding window), and parameter efficiency (MoE). Pure non-attention models (Mamba, RWKV) underperform in general but win on long context when hybridized.

Next chapter goes one level down: **tokenization**. Why CJK tokens cost 2-3× more than English, what BPE actually does on a byte stream, and how chat template tokens get baked into a model's behavior. It's the layer everyone skips and then later regrets.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.
- Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *NeurIPS*.
- Lepikhin, D., Lee, H., Xu, Y., et al. (2020). GShard: Scaling giant models with conditional computation and automatic sharding. *[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)*.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive Transformers with linear attention. *ICML*.
- Xiong, R., Yang, Y., He, D., et al. (2020). On layer normalization in the Transformer architecture. *ICML*.
- Shazeer, N. (2020). GLU variants improve Transformer. *[arXiv:2002.05202](https://arxiv.org/abs/2002.05202)*.
- Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. *[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)*.
- Choromanski, K., Likhosherstov, V., Dohan, D., et al. (2021). Rethinking attention with Performers. *ICLR*.
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers are secretly fast weight programmers. *ICML*.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*.
- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*.
- Press, O., Smith, N., & Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation (ALiBi). *ICLR*.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces (S4). *ICLR*.
- Micikevicius, P., Stosic, D., Burgess, N., et al. (2022). FP8 formats for deep learning. *[arXiv:2209.05433](https://arxiv.org/abs/2209.05433)*.
- Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023). GQA: Training generalized multi-query Transformer models from multi-head checkpoints. *EMNLP*.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)*.
- Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)*.
- Jiang, A., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. *[arXiv:2310.06825](https://arxiv.org/abs/2310.06825)*.
- Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer era. *EMNLP Findings*.
- Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive network: A successor to Transformer for large language models (RetNet). *[arXiv:2307.08621](https://arxiv.org/abs/2307.08621)*.
- Kazemnejad, A., Padhi, I., Ramamurthy, K., et al. (2023). The impact of positional encoding on length generalization in Transformers. *NeurIPS*.
- Jiang, A., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of experts. *[arXiv:2401.04088](https://arxiv.org/abs/2401.04088)*.
- Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *ICML*.
- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient streaming language models with attention sinks. *ICLR*.
- DeepSeek-AI. (2024). DeepSeek-V3 technical report. *[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)*.
- DeepSeek-AI. (2024). DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. *[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)*.
- Lieber, O., Lenz, B., Bata, H., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. *[arXiv:2403.19887](https://arxiv.org/abs/2403.19887)*.
- Glorioso, P., Anthony, Q., Tokpanov, Y., et al. (2024). Zamba: A compact 7B SSM hybrid model. *[arXiv:2405.16712](https://arxiv.org/abs/2405.16712)*.
- Ren, L., Liu, Y., Lu, Y., et al. (2024). Samba: Simple hybrid state space models for efficient unlimited context language modeling. *[arXiv:2406.07522](https://arxiv.org/abs/2406.07522)*.
- Wang, L., Gao, H., Zhao, C., et al. (2024). Auxiliary-loss-free load balancing strategy for mixture-of-experts. *[arXiv:2408.15664](https://arxiv.org/abs/2408.15664)*.
- Liu, Z., Yuan, J., Jin, H., et al. (2024). KIVI: A tuning-free asymmetric 2bit quantization for KV cache. *ICML*.
- Shah, J., Bikshandi, G., Zhang, Y., et al. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *NeurIPS*.
- Jelassi, S., Brandfonbrener, D., Kakade, S., & Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. *ICML*.
- Ye, T., Dong, L., Xia, Y., et al. (2024). Differential Transformer. *[arXiv:2410.05258](https://arxiv.org/abs/2410.05258)*.
- Lou, A., Meng, C., & Ermon, S. (2024). Discrete diffusion modeling by estimating the ratios of the data distribution (SEDD). *ICML*.
- Sahoo, S., Arriola, M., Schiff, Y., et al. (2024). Simple and effective masked diffusion language models (MDLM). *NeurIPS*.
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *[arXiv:2408.03314](https://arxiv.org/abs/2408.03314)*.
- Dubey, A., Jauhri, A., Pandey, A., et al. (2024). The Llama 3 herd of models. *[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)*.
- Yuan, J., Gao, H., Dai, D., et al. (2025). Native sparse attention: Hardware-aligned and natively trainable sparse attention. *[arXiv:2502.11089](https://arxiv.org/abs/2502.11089)*.
- Jacobs, R., Jordan, M., Nowlan, S., & Hinton, G. (1991). Adaptive mixtures of local experts. *Neural Computation* 3(1):79-87.
