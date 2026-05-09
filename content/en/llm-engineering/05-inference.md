---
title: "LLM Engineering (5): Inference Optimization"
date: 2026-04-30 09:00:00
tags:
  - llm
  - inference
  - vllm
  - quantization
  - paged-attention
  - speculative-decoding
categories: LLM Engineering
series: llm-engineering
series_order: 5
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "KV cache mechanics, paged attention, continuous batching, speculative decoding, INT8/INT4/AWQ/GPTQ quantization, and the vLLM vs SGLang vs TensorRT-LLM tradeoffs."
translationKey: "llm-engineering-5"
---

Inference is where the money goes. A single 70B-class model serving 1000 concurrent users at 50 tok/s eats the GPU budget that trained the model in about 3 months. Everything in this chapter is in service of two numbers: time-to-first-token (TTFT) and inter-token latency (ITL). And one ratio: GPU-seconds per million output tokens.

Training is a one-time capital expense — you compress the cost over millions of inference calls. Inference is the recurring operating expense, and unlike training it does not amortize. A 0.5x improvement in tokens-per-GPU-second compounds every day for the life of the product. This is why every serious LLM team has at least one full-time engineer on inference, and why the open-source community has shipped four distinct waves of inference engines (FasterTransformer → DeepSpeed-Inference → vLLM → SGLang/TensorRT-LLM/llama.cpp) in five years.

![LLM Engineering (5): Inference Optimization — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/illustration_1.png)

## The two phases that don't share characteristics

![fig1: prefill vs decode compute pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/fig1_prefill_vs_decode.png)


Every LLM inference call is two phases:

1. **Prefill (prompt processing)**: take the input tokens, run them through the model in parallel, fill the KV cache. Compute-bound. A 4K-token prompt on a 70B model is about 280 TFLOP — saturates an H100 in ~70 ms.
2. **Decode (generation)**: produce one token at a time, attend over the cached keys and values. Memory-bound. Each decode step reads the entire KV cache (gigabytes) to produce one token.

The asymmetry is everything. Prefill batches well across users (same kernel, different sequences). Decode batches *poorly* with naive batching because each user is at a different sequence position. The major inference engines are all built around this asymmetry.

A common heuristic: TTFT is dominated by prefill, ITL is dominated by decode memory bandwidth. To reduce TTFT, throw FLOPs at it (more SMs, tensor parallelism). To reduce ITL, throw memory bandwidth at it (HBM3 over GDDR, fewer parameters via quantization).

The arithmetic intensity argument makes this rigorous. Prefill on a 4K prompt of a 70B model: the model weights (140 GB at BF16) get loaded once and operate on 4096 tokens. Arithmetic intensity ≈ 4096 FLOP per parameter byte read. Decode on the same model, single token: weights loaded once, operate on 1 token. Arithmetic intensity ≈ 1 FLOP per byte. An H100 at 989 TFLOPS BF16 and 3.35 TB/s HBM has a ridge point of ~295 FLOP/byte. Prefill at 4096 FLOP/byte sits far above the ridge (compute-bound). Decode at 1 FLOP/byte is two orders of magnitude below (memory-bound). The two phases want different hardware properties, and a serving stack that doesn't separate them leaves performance on the floor.

## KV cache: the data structure that funds long context

![fig2: KV cache size growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/fig2_kv_cache_growth.png)


The KV cache stores the projected K and V vectors for every prior token in every layer. For a 70B model with GQA-8 at 32K context (numbers from chapter 1):

$$\text{KV} = 2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 \cdot 32{,}768 \cdot 2 \text{ bytes} = 8.6 \text{ GB}$$

Per request. With 50 concurrent requests, that's 430 GB of KV cache — far more than your model weights. KV cache is the bottleneck, not weights.

The naive implementation allocates a contiguous tensor per request sized to `max_context`. Two failures:

1. **Internal fragmentation.** A request that uses 1K tokens still reserves 32K of memory.
2. **Cannot grow.** Beyond max_context, you OOM mid-decode.

A third failure shows up under load: **external fragmentation**. Requests arrive and depart at different times. Free memory exists but is scattered across non-contiguous regions, none large enough for a new request. Servers running for hours under variable load lose 20-40 % of their addressable KV memory to external fragmentation alone. This is the same problem operating systems solved with paging in the 1960s, and it has the same solution.

## Paged attention

![fig3: paged attention block table](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/fig3_paged_attention.png)


vLLM's killer feature, from the 2023 paper [Kwon et al.][kwon-vllm], is **paged attention**. KV cache is allocated in fixed-size **blocks** (typically 16 tokens worth) and a per-request **block table** maps logical positions to physical blocks. Like virtual memory in an OS.

```
Request A: needs 47 tokens of KV → 3 blocks (16+16+15)
Request B: needs 200 tokens of KV → 13 blocks
Block table for A: [0x47, 0x12, 0x3a]
Block table for B: [0x05, 0x09, 0x21, ...]
```

Benefits:

- **Near-zero waste**: only the last block per request has internal fragmentation, and even then only up to 15 tokens.
- **Memory sharing**: prefix-shared requests (system prompt, few-shot examples, tool definitions) can share the same physical blocks. This is where prompt caching comes from.
- **Easy preemption**: under memory pressure, swap out a request's blocks to CPU; bring them back when we have headroom.

vLLM's measured serving throughput improvement over a naive Hugging Face Transformers loop on LLaMA-13B was 14-24x on the original paper's benchmarks. The numbers have held up — though the realistic production gap depends heavily on workload mix. For batch-1 single-stream serving, the win is closer to 2x; for high-concurrency mixed-length traffic, 5-10x is typical; the 14-24x figure shows up when comparing to *truly* naive HF code with no batching at all. The point isn't the exact multiplier; it's that paged attention makes KV memory a tractable resource instead of a hard constraint.

A minimal use:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-7B", gpu_memory_utilization=0.9,
          max_model_len=32768, enable_prefix_caching=True)
prompts = ["Explain MoE in one paragraph.",
           "Explain GQA in one paragraph."]
out = llm.generate(prompts, SamplingParams(max_tokens=200, temperature=0.7))
```

`enable_prefix_caching=True` enables automatic block sharing for repeated prompt prefixes. With a system prompt repeated across 1000 requests, this is a 5-50x prefill speedup for those requests.

### The block manager: how paging actually works

The block manager sits at the heart of every paged-attention engine. It owns a pool of physical KV blocks (typically 16 tokens × 2 (KV) × num_layers × num_heads × head_dim bytes per block), maintains free lists, and atomically allocates blocks for incoming requests. When a request needs to grow (next decode step), the manager pops a free block, appends its physical address to the request's block table, and returns. Under copy-on-write semantics, prefix-shared blocks are reference-counted: forking a sequence (e.g., for parallel sampling, beam search, or speculation rollback) increments the refcount; freeing decrements; physical memory is released only when refcount hits zero.

The "evict to CPU" path matters in production. When the GPU runs out of free blocks but new requests are queued, the scheduler picks a victim request (usually FIFO or longest-running), copies its blocks to pinned host memory over PCIe, and frees the GPU blocks. When memory becomes available, the request is brought back. Eviction overhead is real (PCIe at 64 GB/s for Gen5 ×16 means 100 ms to evict and reload an 8 GB request) but it lets the system honor latency SLOs by not refusing requests outright.

## Continuous batching

![fig4: continuous vs static batching](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/fig4_batching_timeline.png)


The other revolution. Static batching waits for the slowest sequence in a batch to finish before starting the next batch. With variable-length outputs (which is always), you waste a lot of GPU time idle.

**Continuous batching** ([Yu et al., Orca, OSDI 2022][yu-orca], popularized by vLLM): every decode step, evict completed sequences and admit waiting ones. The GPU is never running fewer than `max_batch_size` sequences as long as work is queued.

For 1000 prompts where output lengths vary from 50 to 2000 tokens:

- Static batching: total wall time ≈ time for 2000-token output (worst case dominates).
- Continuous batching: total wall time ≈ average across all outputs.

The throughput improvement compounds with paged attention because admitting a new sequence mid-batch is just allocating new KV blocks — no full-tensor reallocation needed.

### Orca's iteration-level scheduling

The Orca paper introduced two mechanisms still ubiquitous in 2026 engines. **Iteration-level scheduling** decouples the batching boundary from the request boundary: every transformer iteration is independently scheduled, so a request that finishes at step 500 doesn't hold up a request that needs 1500 steps. **Selective batching** handles the prefill/decode mismatch by allowing the attention operator to operate on per-sequence shapes while the linear layers (which don't care about position) batch together. Modern engines fuse this further: vLLM and SGLang both run "chunked prefill" where prefill of one request is interleaved with decode of others in the same forward pass, keeping GPU utilization above 90 % even when the decode queue is sparse.

The scheduler decision matters: when do you admit a new request mid-batch? Naively, immediately. But admitting a 32K-token prompt for prefill alongside ongoing decodes spikes per-iteration latency dramatically (prefill of 32K is ~100x the FLOPS of a single decode step). Production schedulers cap per-iteration "budget" and split long prefills into chunks, interleaving them with decode steps. This is the hidden tradeoff knob that determines whether your TTFT p99 is 500 ms or 5 seconds under load.

## Speculative decoding

![LLM Engineering (5): Inference Optimization — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/illustration_2.png)


![fig5: speculative decoding tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/05-inference/fig5_speculative_tree.png)


A clever idea, now standard. Decode is memory-bound: each step reads the entire model weights to produce one token. What if you could verify *N* tokens at once?

Speculative decoding ([Leviathan et al., 2023][leviathan-spec]) uses a **draft model** (small, fast — say a 1B model) to propose $k$ tokens. The target model (the 70B you actually want to serve) verifies them in a single forward pass: it sees the proposed prefix and computes probability for $k+1$ positions in parallel. Tokens that match the target's argmax stay; the first mismatch and everything after are discarded; resample from there.

Math: if the draft is right $\alpha$ fraction of the time, and decode is memory-bound (so a $k+1$-token forward pass costs roughly the same as a 1-token pass), the expected speedup is $(1 - \alpha^{k+1})/(1 - \alpha)$. For $\alpha=0.7$, $k=4$: 2.5x.

In practice, draft models are usually distilled from the target. Three families compete:

- **Draft-target two-model pairs**, the original Leviathan formulation: a small distilled model proposes, the target verifies. Quality is high but maintaining two synced checkpoints is operational pain.
- **Medusa** ([Cai et al., 2024][cai-medusa]): adds 4-5 extra prediction heads to the target model itself. Each head predicts position $t+1, t+2, t+3, ...$ from the same backbone activations. No separate draft model. Speedup: 2-3x with no quality loss; trains in a few GPU-days on top of the base.
- **EAGLE / EAGLE-2** ([Li et al., 2024][li-eagle]): an autoregressive head that predicts on top of the second-to-last layer's hidden state plus the previous token. EAGLE-2 introduces a dynamic tree-based verification that explores multiple draft branches per step. Hits 3-4x on most workloads; current SOTA for autoregressive speculation.

```python
# vLLM speculative decoding
llm = LLM(model="Qwen/Qwen3-32B",
          speculative_model="Qwen/Qwen3-1.7B",
          num_speculative_tokens=5)
```

The catch: speculation only helps if your target's GPU is memory-bound, not FLOPs-bound. Big batches are FLOPs-bound (the weights are loaded once and amortized across all sequences in the batch); spec decoding doesn't help. Single-stream low-latency serving is memory-bound and benefits enormously.

A subtle point on correctness: the verification step does a *rejection sample* against the target distribution. If draft probability for token $t$ is $p_d(t)$ and target probability is $p_t(t)$, accept with probability $\min(1, p_t(t)/p_d(t))$; on rejection, sample from a corrected distribution proportional to $\max(0, p_t(t) - p_d(t))$. This guarantees the output distribution is *exactly* the target's, regardless of draft quality. Spec decoding is lossless: a worse draft just means lower acceptance rate, not worse outputs. This is why teams ship it without quality regression tests — there's no quality to regress.

## Quantization: INT8, INT4, FP8

Inference quantization is bf16 → INT8 / INT4 / FP8 to shrink memory and bandwidth. The model weights are typically quantized; activations sometimes too.

### INT8 weight-only quantization

Easiest. Quantize each weight matrix per-channel: $w_q = \text{round}(w / s)$, where $s$ is a per-output-channel scale. At inference, dequantize on-the-fly into bf16 before the matmul.

Memory: weights drop 2x (16-bit → 8-bit). Quality: typically <0.5 % perplexity loss with no calibration. Free win.

### INT4: GPTQ vs AWQ

INT4 is much harder. Naive round-to-nearest loses 2-5 % perplexity. The two algorithms that work:

**GPTQ** ([Frantar et al., 2022][frantar-gptq]): an extension of OBQ (Optimal Brain Quantization) made tractable for billion-parameter models. Quantize columns of the weight matrix one at a time; after each column, update the remaining un-quantized columns to compensate for the introduced error using a Cholesky-decomposed Hessian inverse. The math is essentially a structured least-squares solve at each column. Computational cost: a few hours for 70B on a single A100, peak memory roughly equal to one layer's activations on a small calibration set (typically 128 samples × 2048 tokens).

**AWQ** ([Lin et al., 2023][lin-awq]): observe that 1 % of weights are "salient" (large activations multiplied through them). Apply a per-channel scaling $s$ that protects salient weights from quantization noise — equivalently, multiply input activations by $s$ and divide weights by $s$ before quantization. This is mathematically a no-op at full precision but reshapes the quantization landscape. Then quantize uniformly. Faster than GPTQ (no Hessian solve, just a search over scaling factors), slightly better quality on most models, much friendlier to MoE.

**SmoothQuant** ([Xiao et al., 2022][xiao-smoothquant]): orthogonal to GPTQ/AWQ — addresses *activation* quantization. The challenge is that activations have outlier channels (a few channels with magnitudes 100x the others) that destroy INT8 activation quantization. SmoothQuant migrates the activation magnitude into weights via a per-channel scaling, making both sides quantizable to INT8. Used as a preprocessing step before INT8 W8A8 quantization.

Both GPTQ and AWQ deliver INT4 weight-only with <1 % perplexity loss on LLaMA-class models. For MoE, the routing complicates calibration — different experts see very different activation distributions, and a uniform scaling factor underprotects rarely-used experts. Per-expert AWQ is the current best practice.

```python
# Loading an AWQ-quantized model with vLLM
llm = LLM(model="Qwen/Qwen3-32B-AWQ", quantization="awq",
          dtype="float16", max_model_len=32768)
```

### FP8 (and the H100/H200 hardware path)

H100 and newer GPUs have FP8 tensor cores running at 2x BF16 throughput. FP8 inference is the modern path: weights stored FP8, activations cast to FP8 in compute, accumulation in FP32. Quality loss is negligible (<0.1 %) because FP8 has more dynamic range than INT8.

FP8 has two formats: E4M3 (4 exponent bits, 3 mantissa) for activations and weights — small range, more precision; and E5M2 (5 exp, 2 mantissa) for gradients and KV cache — larger range, less precision. The hardware fuses dequant scaling into the matmul, so there's no "INT4 dequant overhead" cost; FP8 is the speed-of-light path on H100/H200.

Calibration matters for FP8 even though the perplexity loss is tiny. The standard recipe: collect activation statistics over 128-512 calibration samples, compute per-tensor or per-token scales, store as part of the checkpoint. Per-token activation scales (computed at runtime from the actual batch) give the best quality at small extra latency cost. NVIDIA's TransformerEngine and vLLM's `--quantization fp8` both implement this.

The catch: FP8 needs hardware support. INT4/INT8 work on any GPU. FP8 is H100/H200/B200 only. If you're shipping on A100s, INT8/INT4 are your options.

### KV cache quantization

The other half of the memory bill. Compress KV cache to INT8 (free 2x), or INT4 (1-2 % quality loss with careful per-token calibration). vLLM and SGLang both support FP8 KV cache; INT4 KV is research-grade but FlashInfer has kernels.

KV cache at FP8 is the workhorse in 2026. It cuts memory in half (so 2x more concurrent requests on the same GPU) at <0.1 % quality loss. Per-token scaling is essentially free because the scales are computed inline. INT4 KV needs more care — the per-channel salience structure that AWQ exploits for weights doesn't apply directly to KV (which is data, not parameters). The current best-practice is per-token-per-head scaling with outlier preservation: keep ~1 % of channels at higher precision, quantize the rest aggressively.

## SGLang and RadixAttention

vLLM's prefix caching shares blocks between requests with identical prefixes. SGLang ([Zheng et al., 2024][zheng-sglang]) generalizes this with **RadixAttention**: a radix tree of all active KV blocks, where each path from root to leaf represents a token sequence. New requests look up the longest matching prefix in the tree, share those blocks, and only compute the suffix.

Why this matters more than vLLM's exact-prefix caching: agent workloads branch. A ReAct agent might issue 5 tool-calling sub-queries, each with the same system prompt + tools + conversation history but different short suffixes. Linear prefix caching catches each sub-query against the shared root, but doesn't help the sub-queries share state with each other. RadixAttention does.

Reported gains in the SGLang paper: 5x throughput on agent and structured-output workloads vs vLLM-equivalent prefix caching, ~1.5x on simple chat. The win shrinks for traffic that doesn't share prefixes.

The implementation is roughly: maintain a radix tree keyed on token IDs, where nodes hold KV block references. On a new request, walk the tree as deep as possible matching the input prefix; bump refcount on all touched nodes; allocate new blocks for the unmatched suffix; on completion, decrement refcounts; LRU-evict subtrees when memory pressure rises. The data structure cost is negligible compared to the savings.

## TensorRT-LLM specifics

NVIDIA's TensorRT-LLM is the third major engine. Its differentiators:

- **Compiled kernels**: every model is compiled to TensorRT engines for the specific GPU, batch size profile, and sequence lengths. Compilation takes 10-30 minutes; the resulting engine is 1.1-1.3x faster than vLLM's runtime-compiled equivalents on the workloads it supports.
- **Plugin model**: custom kernels for attention (FMHA, paged FMHA, FlashAttention-3 paths), MoE routing, and quantization are exposed as plugins. The plugin model is more fragile than vLLM's flexibility but allows aggressive fusion.
- **In-flight batching**: the TensorRT-LLM equivalent of continuous batching, with NVIDIA-tuned schedulers.
- **Tight FP8 integration**: TransformerEngine FP8 is first-class. On H100 with FP8, TensorRT-LLM is usually the throughput leader by 5-15 %.
- **Painful conversion**: each new model architecture needs a custom Python builder script. New model support lags vLLM by weeks to months.

Pick TensorRT-LLM if you serve a fixed set of well-known models at high scale on NVIDIA hardware and you have the engineering bandwidth. Otherwise, the operational simplicity of vLLM almost always wins.

## The serving framework choice in 2026

Three major options:

- **vLLM** — the de facto open standard. Best community, fastest to support new models and features. Paged attention, continuous batching, speculative decoding, prefix caching all out of the box. Defaults are good. *Pick this unless you have reasons not to.*
- **SGLang** — newer (2024), better for **structured generation** (constrained JSON, regex), **front-end caching** (RadixAttention for prompt-tree sharing), and high-fanout agents that branch a lot. Lower TTFT for shared-prefix workloads.
- **TensorRT-LLM** — NVIDIA's framework. Best raw throughput on H100 for the models it supports (compiled kernels, fused FlashAttention paths), but conversion is painful and adds new model support slowly. *Pick this if you have NVIDIA support and need every last token/s.*

vLLM 0.6+ and SGLang both support most of: speculative decoding, FP8, AWQ/GPTQ, MoE, multi-LoRA, function-call constrained decoding. The gap is closing. Default to vLLM, swap to SGLang if your workload is structured-output or agent-heavy.

For pure throughput numbers on Qwen3-32B at FP8 on a single H100 (my benchmarks, late 2025):

| Engine | Throughput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|
| vLLM 0.6 | 7400 | 95 | 13 |
| SGLang 0.4 | 7800 | 72 | 14 |
| TensorRT-LLM | 8900 | 88 | 11 |

The differences are noise for most workloads. Pick on developer experience.

## Parallelism modes for serving

For a 70B model in BF16 at full quality, you need >140 GB. Single H100 (80 GB) can't fit it. Options branch into three orthogonal axes — tensor parallelism (TP), pipeline parallelism (PP), and sequence parallelism (SP) — that compose differently from the training case (chapter 4).

### Tensor parallelism (TP)

Split each weight matrix across GPUs. The classic Megatron-LM [Shoeybi et al., 2019][shoeybi-megatron] decomposition: column-parallelize the QKV projection (each GPU gets a subset of attention heads), row-parallelize the attention output projection (with all-reduce afterward), then column-parallel and row-parallel for the two FFN matmuls. Two all-reduces per transformer layer.

For serving, TP within a node (NVLink, ~600-900 GB/s bidirectional) is fine; TP across nodes (InfiniBand, ~50 GB/s per link) introduces unacceptable latency for the all-reduces. The standard cap is TP ≤ 8 (one node).

- TP=2 H100s (same node, NVLink). Roughly 1.7-1.9x throughput of one (not 2x — TP introduces all-reduce communication).
- TP=4 if you need long context (KV cache also splits across TP).
- INT4 quantization, single H100. Lower throughput than TP=2 but cheaper.

`vllm serve Qwen/Qwen3-72B --tensor-parallel-size 2` is the one-liner.

### Pipeline parallelism (PP)

Split the layers across GPUs. GPU 0 holds layers 0-19, GPU 1 holds 20-39, etc. Activations flow through the pipeline. The classic problem is the **pipeline bubble**: the time when GPUs at the end of the pipeline are idle waiting for the start, and vice versa. For serving, PP introduces TTFT latency proportional to the pipeline depth × time-per-layer (GPipe / Megatron-LM analysis); modern engines amortize this with **micro-batching** (split the batch into smaller micro-batches that flow through the pipeline back-to-back).

PP is the cross-node parallelism for inference. TP within node, PP across nodes is the standard 200B+ deployment. Most teams cap at one node for latency-sensitive workloads — the PP bubble is real.

### Sequence parallelism (SP)

For very long sequences, the activations themselves don't fit on one GPU. Sequence parallelism splits along the token dimension and uses ring or all-to-all communication for the attention all-reduce. For inference, SP shows up at 512K+ contexts where the per-layer activation buffer exceeds GPU memory even after TP. Most production serving doesn't need it.

### Math: when does each parallelism mode pay off?

Roughly, TP cost per layer is $2 \cdot \text{params}/\text{TP} / \text{HBM-BW} + 2 \cdot \text{batch} \cdot \text{seq} \cdot \text{hidden} / \text{NVLink-BW}$. The first term shrinks linearly with TP; the second grows. There's a sweet spot, usually TP=2 to TP=4 on H100. Beyond TP=8 the all-reduce dominates and you lose. For a 70B model with TP=2 you get ~1.85x throughput; TP=4 gets ~3.2x; TP=8 gets ~5.5x. Diminishing returns are real.

PP latency cost is roughly $\text{depth} \cdot \text{time-per-layer}$ for a single-batch query. With micro-batching and steady-state queue, throughput approaches $\text{batch} / (\text{time-per-stage})$. PP is throughput-friendly, latency-hostile.

## What I'd actually deploy

A 7B-class model: single L40S or 4090, FP8, vLLM, 16K context, $0.10-$0.15 per million tokens served at 80 % utilization.

A 32B-class model: single H100 with AWQ INT4 *or* TP=2 L40S with FP8. Both work. H100 is faster per-token; L40S is cheaper per-hour. Decide on $/Mtok.

A 70B-class model: TP=2 H100 with FP8. INT4 saves 30 % $ but loses ~1 % quality, often too much for prod.

A 200B+ MoE: distinct optimization problem (chapter 12).

## Observability: what to actually measure

In production, the metrics that matter are not "tokens/sec" — that's a single-batch number. The real metrics are per-percentile latencies under load. The minimum dashboard:

- **TTFT p50 / p95 / p99** (ms): time from request received to first token emitted. p99 is what your users feel; p50 is what they brag about.
- **ITL p50 / p95 / p99** (ms): inter-token latency in steady state. Spikes here mean preemption or memory pressure.
- **Throughput (tok/s)** at various concurrency levels: 1, 8, 32, 128, 512. Plot as a curve; the inflection point is where you're saturating either FLOPs or memory bandwidth.
- **Queue depth** and **queue wait time**: are requests piling up because the engine can't keep up?
- **KV cache utilization (%)**: are you about to start preempting?
- **Prefix cache hit rate (%)**: low hit rate means your prompt-engineering team is generating non-shareable prompts.
- **GPU SM utilization** and **HBM bandwidth utilization**: from `dcgm-exporter` or `nvidia-smi dmon`. If both are <70 %, you have headroom; if HBM is at 95 % and SM is at 40 %, you're memory-bound and quantization is the answer.

Set SLOs explicitly. A typical production target: TTFT p95 < 500 ms, ITL p95 < 50 ms, error rate < 0.1 %. Anything above that is a fire.

## Cost arithmetic

The key number is **dollars per million output tokens** ($/Mtok-out). For a self-hosted vLLM serving Qwen3-32B at FP8 on a single H100 ($2.50/hr on-demand, $1.20/hr 1-year reserved, $0.80/hr 3-year reserved or savings plan):

- Throughput at concurrency 32: ~7400 tok/s
- Reserved hourly cost: $1.20/hr = $0.00033/sec
- $/Mtok-out = $0.00033 / 7400 \cdot 10^6 ≈ $0.045

Compare to API pricing in mid-2026:
- Claude-4.5-Sonnet: $15/Mtok output
- GPT-5: $12/Mtok output
- DeepSeek-V3.2: $1.10/Mtok output
- Qwen3-Max API: $0.80/Mtok output

The 30-300x margin between self-hosted open-weights and frontier API is the entire reason teams self-host. The catch is: at <100M tokens/month, the engineering cost dominates the savings. At >1B tokens/month, the savings dominate and self-hosting is obvious.

A subtle gotcha: input tokens are typically 5-10x cheaper than output tokens on APIs (because they batch better and don't autoregress). Self-hosted, they're still cheaper but less dramatically (3-5x), because both phases run on your hardware. Cost models that assume parity will mislead you in either direction.

## Takeaway and what's next

Inference is two asymmetric phases (prefill, decode), and the modern serving stack — paged attention + continuous batching + speculation + quantization + FP8 hardware — exists to make the worst of both phases tolerable. vLLM is the right default. Quantization (INT8 always, INT4 if budget tight, FP8 if H100+) is essentially free. Spec decoding is a 2-3x win for low-batch low-latency serving and noise for high-throughput. The cost arithmetic in 2026 strongly favors self-hosting open weights at scale.

Next chapter: **long context**. RoPE scaling, YaRN, NTK-aware interpolation, ALiBi, attention sinks, and why most "1M context" claims fall apart on actual retrieval tasks.

## References

- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.][kwon-vllm] The original vLLM paper.
- [Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022.][yu-orca] Continuous batching + iteration-level scheduling.
- [Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024.][zheng-sglang] RadixAttention.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023.][leviathan-spec]
- [Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," ICML 2024.][cai-medusa]
- [Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees," 2024.][li-eagle]
- [Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024.][lin-awq]
- [Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023.][frantar-gptq]
- [Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023.][xiao-smoothquant]
- [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019.][shoeybi-megatron]
- [vLLM project documentation](https://docs.vllm.ai/)
- [SGLang project documentation](https://docs.sglang.ai/)
- [NVIDIA TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM)

[kwon-vllm]: https://arxiv.org/abs/2309.06180
[yu-orca]: https://www.usenix.org/conference/osdi22/presentation/yu
[zheng-sglang]: https://arxiv.org/abs/2312.07104
[leviathan-spec]: https://arxiv.org/abs/2211.17192
[cai-medusa]: https://arxiv.org/abs/2401.10774
[li-eagle]: https://arxiv.org/abs/2406.16858
[lin-awq]: https://arxiv.org/abs/2306.00978
[frantar-gptq]: https://arxiv.org/abs/2210.17323
[xiao-smoothquant]: https://arxiv.org/abs/2211.10438
[shoeybi-megatron]: https://arxiv.org/abs/1909.08053
