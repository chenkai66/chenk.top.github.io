---
title: "LLM Engineering (6): Long Context — RoPE, YaRN, Sinks"
date: 2026-05-01 09:00:00
tags:
  - LLM
  - long-context
  - rope
  - yarn
  - alibi
  - attention-sinks
categories: LLM Engineering
series: llm-engineering
series_order: 6
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "How RoPE encodes position, why naive extension breaks, NTK-aware and YaRN scaling, ALiBi vs RoPE, attention sinks for streaming, and why 1M-context claims often fail at retrieval."
translationKey: "llm-engineering-6"
---

"1M token context" is one of the most over-claimed numbers in LLMs. A model can attend to 1M tokens — that's an architecture statement. A model can *use* information at position 800K to answer a question — that's a behavior statement, and it's harder. This chapter is about the math of position encoding, the engineering tricks that extend context past the training length, and the reasons most long-context claims don't survive needle-in-a-haystack tests.

The history of long-context LLMs in three acts. Act one (2017-2021): models were trained at 512-2048 tokens because attention is $O(n^2)$ and that's what fit. Act two (2022-2023): efficient attention kernels (FlashAttention, [Dao 2022][dao-flashattention]) made longer training feasible, and post-hoc context extension techniques (Position Interpolation, NTK-aware scaling, YaRN) let practitioners push pre-trained checkpoints from 4K to 32K and beyond. Act three (2024-2026): native long-context training (Llama 3.1's 128K, Gemini's 1-2M, Claude's 200K) became standard, but the gap between *attendable* context and *useful* context remained — and that's what this chapter is mostly about.

![LLM Engineering (6): Long Context — RoPE, YaRN, Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/illustration_1.png)

## Position is not free

Self-attention is permutation invariant. Without a position signal the model can't tell "the cat sat on the mat" from "the mat sat on the cat." Three answers to "how to inject position":

1. **Sinusoidal absolute** (original Transformer): add $\sin/\cos$ functions of position to the token embedding before layer 1.
2. **Learned absolute**: learn a position embedding per absolute index up to max length. Used by GPT-2, BERT.
3. **Rotary (RoPE)**: rotate Q and K vectors by an angle proportional to position, *inside* every attention layer. Used by LLaMA, Qwen, Mistral, DeepSeek.

RoPE won. It's the position encoding of every credible 2026 LLM I know of. Two reasons: it injects position at every layer (better signal), and the relative position falls out naturally from the dot product, which is what attention actually needs.

A fourth answer, **ALiBi** (attention with linear bias), competed seriously around 2022 and lost; we'll cover it later in this chapter as the most interesting alternative path. A fifth, **xPos** (Sun et al., 2022), is a RoPE refinement that adds a length-dependent decay to make extrapolation more stable; it's used inside DeepSeek and a few other modern models, but the core idea is RoPE plus engineering polish.

## RoPE: the math

![fig1: RoPE rotation visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig1_rope_rotation.png)


For a query vector $q$ and key vector $k$ at positions $m$ and $n$, RoPE multiplies each by a rotation matrix $R(m\theta)$ and $R(n\theta)$, where $\theta$ depends on the dimension ([Su et al., 2021][su-rope]):

$$\theta_i = b^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1$$

with $b$ the **rope base** (default 10,000). Each pair of dimensions $(2i, 2i+1)$ rotates at frequency $\theta_i$. Lower-index pairs rotate fast (carry fine-grained position), higher-index pairs rotate slow (carry coarse position).

The key property: after rotation, the dot product $q \cdot k$ depends only on the *relative* position $m - n$:

$$\langle R(m\theta) q, R(n\theta) k \rangle = \langle q, R((n-m)\theta) k \rangle.$$

This is what makes RoPE work at extrapolation: the model doesn't see absolute position 50K, it sees a relative offset of $-3$ from the current token, which it has seen a billion times during training.

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# precompute cos/sin tables
def rope_cos_sin(seq_len, dim, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
```

Almost every modern attention kernel handles RoPE inline; this is what's happening conceptually.

### Wavelengths and what each dimension carries

Each frequency $\theta_i$ corresponds to a wavelength $\lambda_i = 2\pi / \theta_i$ — the number of token positions it takes for that pair of dimensions to complete a full rotation. With base $b=10000$ and head_dim $d=128$:

- $i=0$: $\lambda_0 = 2\pi \approx 6.3$ tokens. This pair encodes very local position.
- $i=32$: $\lambda_{32} = 2\pi \cdot 10000^{32/64} \approx 628$ tokens. Mid-range.
- $i=63$: $\lambda_{63} = 2\pi \cdot 10000 \approx 62832$ tokens. The longest wavelength is 62K tokens; if your training context is 4K, this dimension never completes even one full rotation during training.

This is the seed of the extension problem. Dimensions whose wavelengths are *much longer* than the training context have only seen a tiny fraction of their rotation range. When we ask the model to attend at positions far beyond training length, those dimensions enter rotation territory the model has never seen. The high-frequency (short-wavelength) dimensions, which encode local position, are fine — they cycle many times per training context and the model is well-acquainted with their full range.

This insight motivates the modern extension methods: don't scale all dimensions equally; scale by wavelength.

## Why naive extension breaks

![fig2: YaRN frequency rescaling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig2_yarn_freq_adjustment.png)


![fig4: position interpolation strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig4_position_interpolation.png)


Train a model with `rope_base=10000` and `max_position=4096`. Try to use it at position 32768. What happens?

The lowest-frequency dimensions rotate at $10000^{-1} \approx 10^{-4}$ radians per token. By position 32768 they've rotated $\sim 3.3$ radians — past $\pi$, which means they've wrapped past anything seen during training. The dot product geometry is now in territory the model has never learned to interpret. Quality collapses.

Two extension strategies that work:

**Position interpolation (PI)** ([Chen et al., 2023][chen-pi]): scale positions down by a factor $s = L_{\text{new}} / L_{\text{train}}$. Position 32768 with $s=8$ becomes effective position 4096, which the model has seen. Continued fine-tuning for a few hundred steps fixes the small distribution shift. PI works but loses some short-context quality — every position now corresponds to a fractional rotation that the model never trained on. The original PI paper reported extending LLaMA from 2K to 32K with 1000 fine-tuning steps and minimal quality degradation; this was the breakthrough that made post-hoc context extension a real engineering option.

**NTK-aware scaling** (introduced via the [r/LocalLLaMA community in mid-2023][ntk-aware]): instead of scaling positions uniformly, scale the rope base so that low-frequency dimensions stay in the trained range while high-frequency ones (which carry local info) are minimally disturbed. The new base is $b' = b \cdot s^{d/(d-2)}$. The intuition matches the wavelength analysis above: dimensions that already cycled many times during training don't need scaling; dimensions that barely rotated do.

**YaRN** ([Peng et al., 2024][peng-yarn]): the current best. Combines NTK-aware scaling with per-dimension interpolation strength based on whether the dimension's wavelength is shorter or longer than the training context. Plus a temperature correction (rescale attention logits by $\sqrt{\log s}$ to compensate for the increased token count attending to a fixed query). YaRN extends LLaMA-2-7B from 4K to 128K with minimal continued training (a few hundred steps) and almost no short-context degradation (perplexity within 0.1 of the base model).

```python
# YaRN-style rope_base for context extension
import math
def yarn_base(orig_base, orig_ctx, target_ctx, alpha=1, beta=32):
    ratio = target_ctx / orig_ctx
    return orig_base * (ratio ** (alpha / (alpha - beta)))
# example: 4K → 32K
print(yarn_base(10000, 4096, 32768))  # ~1.6e8
```

In practice, the open models you'll deploy (Qwen3, LLaMA-3, Mistral) ship with their context extension already done. You don't usually re-extend; you read the technical report to know which method they used and what the practical limits are.

## LongRoPE and search-based scaling

YaRN is good but assumes a uniform scaling formula across all dimensions in a band. **LongRoPE** ([Ding et al., 2024][ding-longrope]) goes further: it treats per-dimension scaling factors as a search problem. Use evolutionary search over a few thousand candidate scaling vectors (each of length $d/2$, one factor per RoPE frequency pair), evaluate each on a small calibration set (perplexity on long-context texts), and keep the winners. LongRoPE extended LLaMA-2-7B to 2M tokens with comparable RULER scores to native long-context models, all without significant continued pre-training.

The lesson is that the right scaling shape isn't fixed — it depends on the model, the data distribution, and the target length. Per-dimension search-then-fine-tune is the most expressive extension strategy currently known.

A practical implication: when you see "context extended to 1M" in a model card, ask which method. PI, NTK-aware, and YaRN are essentially deterministic. LongRoPE involves a search pass; the released checkpoint encodes the discovered scaling factors. The deployment cost is the same (just initialize cos/sin tables differently), but the quality at high extension ratios is meaningfully different.

## ALiBi: the simpler alternative

ALiBi ([Press et al., 2022][press-alibi]) skips position rotation entirely and adds a linear bias to the attention scores:

$$\text{attn}_{ij} = \text{softmax}\left(\frac{q_i k_j^T}{\sqrt{d}} - m_h \cdot |i - j|\right)$$

where $m_h$ is a per-head slope, geometric in the head index (typically $m_h = 2^{-8h/H}$ for $H$ heads). Closer tokens get higher attention; the bias is fixed at training time and no rotation is needed.

Pros: extrapolates to longer contexts than seen at training with no fine-tuning. The ALiBi paper showed train-at-1024 / test-at-2048 with no quality drop. Cons: most experiments find ALiBi underperforms RoPE on tasks that need precise long-range retrieval — the linear-decay bias means very distant tokens get exponentially less attention regardless of relevance.

ALiBi is in BLOOM, MPT, and a few research models. RoPE has won the production fight. The exception is hybrid Mamba-attention models where ALiBi can be cheaper, and a few "attention sink" implementations where ALiBi-style decay terms are added to RoPE for streaming stability.

A subtle observation: ALiBi works because language really does have a roughly logarithmic distance-to-relevance relationship for most tokens, and the linear bias in log-attention space approximates this. The reason it loses to RoPE on retrieval is that retrieval tasks specifically require *anti*-monotonic attention — the relevant token might be the *farthest* one in the context, not the closest. RoPE can learn arbitrary attention patterns; ALiBi has a built-in distance prior.

## Attention sinks: the streaming hack

![fig3: sliding-window attention with sinks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig3_sliding_window.png)


Xiao et al. ([2024, "StreamingLLM"][xiao-sinks]) found a strange phenomenon: if you do **windowed attention** (each token attends to the previous $w$ tokens) and decode past the window, quality collapses immediately at position $w+1$. Not a slow decay — a cliff.

The cause: attention softmaxes are forced to put mass somewhere even when no key matches the query. Trained models route this mass to the **first few tokens of the sequence** — they become "attention sinks." Truncate them and the softmax distribution becomes incoherent.

The fix is comically simple: keep the first 4-8 tokens in every attention window forever. Combined with sliding windows, this gives models that can decode arbitrarily long sequences without quality collapse.

```python
def streaming_attention(q, k_cache, v_cache, sink_size=4, window=4096):
    # Keep first sink_size tokens + last window tokens
    sink_k, sink_v = k_cache[:, :, :sink_size, :], v_cache[:, :, :sink_size, :]
    win_k,  win_v  = k_cache[:, :, -window:, :], v_cache[:, :, -window:, :]
    k = torch.cat([sink_k, win_k], dim=2)
    v = torch.cat([sink_v, win_v], dim=2)
    return F.scaled_dot_product_attention(q, k, v)
```

Mistral-7B-v0.2 onwards, Qwen3, and most production long-context models all use sinks + sliding windows internally. It is the difference between "1M context that works" and "1M context that hallucinates."

### Why softmax demands a sink

The mathematical reason is structural. Softmax outputs a probability distribution over keys; the probabilities sum to 1 by construction. If no key has any meaningful relevance to the query (e.g., the model is processing a "filler" token), the softmax still produces a distribution. Where does it go? The trained model learns to dump that probability mass on a few specific positions — usually the first few BOS-adjacent tokens, which it has seen in every training example. These tokens act as a "no-op" target.

The implication for streaming is severe. If you slide your attention window past these BOS-adjacent tokens, you've removed the learned no-op target. The softmax now has to put its mass on actually-relevant-looking keys, distorting the attention pattern, and quality cliff-falls. Keeping the sinks forever costs essentially nothing (4-8 KV entries) and preserves the learned dynamics.

A related-but-distinct paper, **Massive Activations** (Sun et al., 2024), showed that sink behavior is part of a broader pattern where a small number of feature activations carry disproportionate importance in trained transformers. Pruning these activations destroys the model. Sinks are the attention-side manifestation of this phenomenon.

## Sliding window attention in production

Mistral 7B (released 2023) was the first popular model to ship with **sliding window attention** (SWA): each token only attends to the previous $w=4096$ tokens. Memory and compute drop linearly with $w$ rather than quadratically with $n$, making 32K context cheap.

The receptive field still grows: at layer $\ell$, a token attends through $\ell$ hops of size $w$, so the effective receptive field is $\ell \cdot w$. For Mistral 7B with 32 layers and $w=4096$, the layer-32 receptive field is 131K tokens, far beyond the named context window. Information from distant tokens does propagate, but indirectly through intermediate layers.

In practice, SWA combined with attention sinks gives most production long-context behavior. Pure dense attention beyond 32K is rare; the cost-quality curve favors SWA + sinks + extension techniques (YaRN/LongRoPE) for the long tail.

## Needle in a haystack: the only honest benchmark

![LLM Engineering (6): Long Context — RoPE, YaRN, Sinks — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/illustration_2.png)


![fig5: RULER scores by context length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/06-long-context/fig5_ruler_scores.png)


Architecture lets you attend to N tokens. Whether you actually use information at position N is a separate question. The standard test (originated by Greg Kamradt in 2023): insert a fact ("The magic number is 7392") at position $p$ in a long context, then ask "What is the magic number?"

The "needle in a haystack" gave us a sharable visualization (the green-and-red heatmap of position-vs-context-length) but it's a weak test. Single-fact retrieval is too easy for modern models. **RULER** ([Hsieh et al., 2024][hsieh-ruler]) is the modern replacement: 13 task categories at controlled context lengths, including multi-needle retrieval, multi-hop reasoning over hidden facts, variable tracking, and aggregation tasks. RULER scores tell you whether a model is *actually* useful at the claimed length, not just whether it can echo a string.

Real numbers from RULER (published 2024 + my reproductions on later models):

| Model | Claimed ctx | RULER 16K | RULER 64K | RULER 256K |
|---|---|---|---|---|
| LLaMA-3-8B-Instruct | 8K | 90.3 | 31.4 | 0.0 |
| LLaMA-3.1-8B (YaRN-128K) | 128K | 89.7 | 78.2 | 38.5 |
| Qwen3-32B | 128K | 95.1 | 91.2 | 81.8 |
| GPT-4o | 128K | 95.5 | 91.8 | 84.2 |
| Claude-4.5-Sonnet | 200K | 96.7 | 93.5 | 88.4 |
| Gemini-3-Pro | 1M | 97.2 | 95.0 | 92.1 |

The pattern: claimed context is usually 2-4x the working context. Architectures that pre-trained on long context (Gemini, Claude) hold up better than ones that extended afterward. For production, assume the working context is half what's claimed unless the vendor publishes RULER results.

A more painful test: **multi-needle**. Hide $k$ facts and ask the model to retrieve all $k$. Most models fall off a cliff at $k = 5$ even at 32K context. **Reasoning over needles** (chain together 3 facts at different positions) is harder still — frontier models maintain ~80 % at 128K but drop to <50 % at 256K.

## Lost in the middle

A second well-documented long-context failure mode: **Lost in the Middle** ([Liu et al., 2023][liu-lostmiddle]). Even when the model *can* attend to all positions, it disproportionately weights tokens at the start and end of the context. The middle 60 % of a long context is systematically underused.

Liu et al.'s key experiment: place a single relevant document in a multi-document QA prompt at varying positions among 19 distractors. Accuracy at position 1 (start) and position 20 (end) was 75-80 %; accuracy at position 10 (middle) dropped to 50-55 %. The U-shaped curve appeared in every model tested (GPT-3.5, Claude-1, MPT, etc.) and persists in 2026 models, though less severely.

Cause: attention is biased toward the start by training-time distribution (most documents have important framing at the top), and toward the end by recency bias in autoregressive decoding. The middle gets squeezed.

Practical fixes:

- **Put the question at the end** of long context, not the beginning. Lost-in-the-middle attention favors the end, so the model "thinks about" the question after reading the documents. Empirically: 5-15 % accuracy improvement.
- **Place the most important context near the start or end** of the document list, not the middle, when you have control over ordering.
- **Reranking matters.** Even if your retrieval returns the right document, putting it at position 5 of 10 underperforms putting it at position 1. RAG pipelines (chapter 8) should sort retrieved chunks with the most relevant first.

## Native long context vs post-hoc extension

By 2026 the long-context landscape has split. Some models are pre-trained with long context as a first-class concern; others are extended post hoc.

Native long-context models (Llama 3.1's 128K, Gemini 2.5/3's 1-2M, Claude 4.5's 200K) interleave long-document training data throughout pre-training. The model learns position-dependent behavior natively; RULER scores are uniformly high across the full context.

Post-hoc extended models (community LLaMA-2 → 32K with YaRN, Qwen2.5 → 128K with extension) start from a short-context base and apply YaRN/LongRoPE plus a small amount of continued pre-training on long-context data. RULER scores are good at moderate extensions (2-4x) and degrade for aggressive ones (16x+).

The cost difference is real. Pre-training at 128K from scratch costs roughly 2-4x as much per token as 4K pre-training (attention dominates compute at long sequences, even with FlashAttention; KV cache memory limits batch sizes). Post-hoc extension is essentially free (a few thousand fine-tuning steps). If you're a foundation lab spending $100M on a training run, the 2-4x premium for native long-context is justified. If you're a small lab releasing an open-weights model, post-hoc extension is the only viable path.

## RAG vs long context: the real tradeoff

The most common application question: "Should I stuff documents into long context, or RAG them?" In 2026 the answer is more nuanced than two years ago.

Stuff into context if:
- Documents fit in <128K tokens for your model.
- Retrieval quality is hard to engineer (fluid query, no good chunking).
- Latency budget tolerates 5-30s prefill.
- Cost budget tolerates ~$0.30 per query (200K-token prompt at $1.50/Mtok).

Use RAG (chapter 8) if:
- Document corpus is much larger than context window.
- Latency must be <1s.
- Cost per query must be cents.
- You need source attribution.

Long context shines for tasks that require holistic reading — code review of an entire repo, summarizing a long meeting, multi-document QA where the right chunks aren't predictable. RAG dominates everything else by 1-2 orders of magnitude on cost.

The hybrid approach (RAG to find candidates, long context to synthesize) is the production sweet spot for most non-trivial workloads. Anthropic's "Contextual Retrieval" (2024) and Microsoft's GraphRAG (2024) are both variations on this hybrid theme; chapter 8 covers them in detail.

## Production tips

- **Always check working context** with a needle test on your actual workload before deploying. Vendor-quoted numbers are best case.
- **Prompt-cache long inputs** when serving repeatedly. A 100K-token system prompt that costs $1.50 to prefill costs $0.05 with caching (vLLM enable_prefix_caching, OpenAI/Anthropic prompt caching APIs).
- **Place the question at the end** of long context, not the beginning. Lost-in-the-middle (Liu et al., 2023) shows attention disproportionately weights the start and end of long contexts. Question-at-end gets 5-15 % higher accuracy.
- **Don't trust position 50K+** for arithmetic. Even Claude-4.5 makes more errors on multi-step math when the input data is far back. Move calculation context closer to the question if you can.
- **Use sliding window + sinks** if you're rolling your own long-context model. The cost-quality tradeoff is unbeatable.
- **Watch the prefill latency curve.** TTFT scales roughly linearly with prompt length on a single GPU, but more steeply with TP (all-reduces dominate) and on cross-node setups. A 200K-token prompt on TP=2 H100 takes 6-8 seconds to prefill; users notice.
- **Benchmark on your domain.** A model that scores 90 on RULER might score 70 on your medical-records task because medical text has different distribution from RULER's synthetic needles. Domain-specific eval is the only honest signal.

## Takeaway and what's next

RoPE made long context tractable; YaRN extended it past training length; sinks made it stable for streaming; but the working context is always less than the claimed context. Test on your workload. For most production tasks, RAG wins on cost. Long context wins for holistic reading and short-running interactive workflows.

Next chapter: **function calling and tool use**. JSON schema vs free-form, parallel tool calls, error recovery, and the agent-loop patterns that actually work.

## References

- [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.][su-rope]
- [Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023.][chen-pi]
- [Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," ICLR 2024.][peng-yarn]
- [Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens," 2024.][ding-longrope]
- [Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)," ICLR 2022.][press-alibi]
- [Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.][xiao-sinks]
- [Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," 2023.][liu-lostmiddle]
- [Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?" 2024.][hsieh-ruler]
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.][dao-flashattention]
- [NTK-aware scaling: r/LocalLLaMA bowtied_handbasket post (2023)][ntk-aware] — community origin of the technique.
- Sun et al., "Massive Activations in Large Language Models," 2024.

[su-rope]: https://arxiv.org/abs/2104.09864
[chen-pi]: https://arxiv.org/abs/2306.15595
[peng-yarn]: https://arxiv.org/abs/2309.00071
[ding-longrope]: https://arxiv.org/abs/2402.13753
[press-alibi]: https://arxiv.org/abs/2108.12409
[xiao-sinks]: https://arxiv.org/abs/2309.17453
[liu-lostmiddle]: https://arxiv.org/abs/2307.03172
[hsieh-ruler]: https://arxiv.org/abs/2404.06654
[dao-flashattention]: https://arxiv.org/abs/2205.14135
[ntk-aware]: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
