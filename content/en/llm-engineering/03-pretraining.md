---
title: "LLM Engineering (3): Pretraining at Scale"
date: 2026-03-29 09:00:00
tags:
  - LLM
  - pretraining
  - fsdp
  - zero
  - data-mixing
  - scaling-laws
categories: LLM Engineering
series: llm-engineering
series_order: 3
series_title: "LLM Engineering"
lang: en
mathjax: true
disableNunjucks: true
description: "Data mixing, deduplication, contamination, μP, FSDP vs ZeRO-3 vs pipeline parallel, the practical 200B-token cliff, and the failure modes that only appear above 1000 GPUs."
translationKey: "llm-engineering-3"
---

Pretraining is where most of an LLM's capability comes from, and it's also where the leaderboard-vs-reality gap is widest. Most published runs are heroic engineering more than they are scientific results. This chapter is about the parts of pretraining that you actually have to get right when you're not OpenAI: the data, the parallelism choice, and the failure modes that only show up when the cluster is large enough to make a single bad NCCL all-reduce kill a 30-day run.

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_1.png)

## The data mixture is more important than the architecture

![fig3: data mixture composition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig3_data_mixture.png)

Every credible scaling study in the last three years agrees: at the same compute, the difference between two LLaMA-style architectures is small (~5 % perplexity), but the difference between two data mixtures is large (>30 %). The Chinchilla paper's compute-optimal scaling laws assumed a fixed data distribution; once that's allowed to vary, data dominates.

A modern pretraining mix looks roughly like this (FineWeb-Edu [Penedo et al., 2024], RedPajama-V2 [Together AI, 2024], Dolma [Soldaini et al., 2024] — all open mixes are within a few points of each other):

| Source | Share | Notes |
|---|---|---|
| Filtered web (CommonCrawl) | 50-65 % | Dominant by volume |
| Code (GitHub, StackExchange) | 8-15 % | Boosts reasoning, not just code |
| Books | 5-10 % | Long-context coherence |
| Wikipedia | 2-5 % | Disproportionately useful per token |
| Academic / arXiv | 3-5 % | Math, citations |
| Math (proofs, textbooks) | 2-4 % | Specialized for reasoning |
| Multilingual web | 5-15 % | Quality varies wildly by language |

The most important number nobody talks about: **deduplication rate**. CommonCrawl 2024 deduplicated at the document level keeps about 25 % of raw bytes. Deduplicated at the line level (more aggressive), about 12 %. The Lee et al. (2022) paper showed that aggressive dedup *improves* perplexity even though it removes 75 % of the data. Repetition is poison for LMs — it teaches them to memorize instead of generalize.

DeepSeek's pretraining writeup (DeepSeek-V3 technical report [DeepSeek-AI, 2024], Dec 2024) is the most honest about ratios I've seen: 14.8T tokens, 87 % code+math+web, 13 % "high-quality books and synthetic data". The synthetic-data fraction is doing more work than people admit.

## DataComp-LM: data quality > quantity, with receipts

The most rigorous public study of data quality is the DataComp for Language Models benchmark [Li et al., 2024]. They fixed the architecture and training compute, then ran 416 controlled experiments varying only the data mixture. Findings worth memorizing:

- **Quality filters dominate.** A model-based quality classifier (FastText trained to predict whether text resembles "high-quality" reference data) trained on the same compute as a baseline beat that baseline by 6.6 percentage points on MMLU and 3.5 on Core (a 22-task benchmark).
- **Aggressive filtering wins.** Keeping only the top 10 % of documents by quality classifier score outperformed keeping the top 50 %, even though it discarded 5× more tokens.
- **Deduplication interacts with filtering.** MinHash-LSH deduplication at 0.7 Jaccard threshold gave +2.1 MMLU; combined with quality filtering it gave +8.7. The two interventions are super-additive — dedup removes near-duplicates of low-quality content that quality filters might miss.
- **The optimal mix is task-specific.** A model trained for code-heavy downstream use prefers a different mix than one for general chat. Universal optima don't exist.

The 7B model trained on the DCLM-Baseline data (3T tokens) outperformed Llama 2 7B (which used 2T tokens of unspecified, less-aggressively-filtered data) by 11.5 MMLU points despite using 33 % less compute per parameter. That's the size of effect data quality can have.

## Synthetic data: the dirty secret

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_2.png)

Until ~2023 the going wisdom was "synthetic data is contamination, never use it." That changed when Phi-1 [Gunasekar et al., 2023] (Microsoft, 2023) showed a 1.3B model could match much larger code models if trained entirely on synthetic textbook-style data generated by GPT-4. The Phi series ran with this; everyone else followed quietly.

In 2026, top open models all use synthetic data:

- **Qwen3 technical report**: synthetic instruction data, synthetic code, "diverse" synthetic Q&A.
- **Llama 3.1** [Dubey et al., 2024]: synthetic data for code, math, multilingual.
- **DeepSeek-V3**: synthetic data for "high-quality books" (legally cleaner than scraping).

The risk is **mode collapse** — synthesizing from a single teacher model concentrates the distribution. The defense is multi-teacher synthesis and aggressive filtering. The other risk is **contamination** of evaluation benchmarks. Phi-3 was caught with MMLU questions in its training set. The standard countermeasure is exact-match decontamination (drop any 13-gram overlap with eval sets) plus held-out novel benchmarks.

A subtler risk identified by [Shumailov et al., 2024] is **model collapse**: training on data sampled from previous-generation models progressively narrows the data distribution and eventually degrades quality. Their experiments showed that recursively training on model outputs — even high-quality ones — converges to a degenerate distribution within 5-10 generations. The mitigation in production is to always mix synthetic data with substantial human-authored data (typically 50-70 % human, 30-50 % synthetic) and to use synthetic data from many independent teachers rather than one strong model.

The economics matter. Generating 1B high-quality synthetic tokens with GPT-4o costs roughly $30K. Generating with a smaller, fine-tuned-for-synthesis model (e.g., Qwen3-32B-Instruct on a self-hosted cluster) drops it to a few hundred dollars. Most production synthetic-data pipelines in 2026 use a mix: a strong frontier model (Claude, GPT-4o) for "seed" diverse generations, then a self-hosted mid-size model for scale-up.

## Scaling laws: Chinchilla, then 200B

The Chinchilla law [Hoffmann et al., 2022] said compute-optimal training uses approximately **20 tokens per parameter**. A 7B model wants 140B tokens; a 70B model wants 1.4T.

The earlier scaling law from [Kaplan et al., 2020] gave a different ratio (~1.7 tokens per param) because they varied compute by changing model size more than data size. Hoffmann et al. fixed Kaplan's methodology by sweeping data and model size jointly. The Chinchilla law became the default planning heuristic for 2022-2023 frontier runs.

Chinchilla is wrong in practice. Or rather, it's right for *training-compute-optimal*, but for **inference-compute-optimal** — where the model gets used billions of times — you should drastically over-train small models. [Sardana et al., 2023] formalized this as "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws." LLaMA-3 8B was trained on 15T tokens (~1900 tokens per parameter), 95× past Chinchilla. The model is much better than a Chinchilla-optimal 8B and serves vastly more cheaply than a Chinchilla-optimal 70B.

The cliff: most open data sources max out around 200B unique tokens after dedup. Beyond that you're either retraining on the same data (epoching) or generating synthetic. Multi-epoch training works up to about 4 epochs on high-quality data, then the model starts overfitting [Muennighoff et al., 2023]. After that, synthetic data is the only way forward — which is why the post-2024 frontier models all generate their own.

The best public estimate of the practical ceiling on natural-language data is from [Villalobos et al., 2024]: roughly 300T tokens of "high-quality" English text exists on the public internet. Frontier labs have processed and filtered this aggressively. Beyond ~50T high-quality tokens (where Llama 3 and DeepSeek-V3 sit), synthetic data and proprietary data sources become the only way to scale data further. This is one of the reasons that "scaling pretraining" as a strategy is starting to plateau in 2026 — we're approaching the data wall.

## Curriculum: anneal from low to high quality

A robust 2024-2025 finding: training is more efficient if you don't shuffle all data uniformly, but instead structure the order of presentation. The basic pattern, used in DeepSeek-V3 and several other recent runs:

1. **Bulk phase** (first ~80 % of tokens): broadly mixed, web-heavy, lower quality bar.
2. **Anneal phase** (last ~20 % of tokens): high-quality mix — books, papers, math, curated code, instruction-tuned data.
3. Learning rate decays through both phases on a cosine schedule.

The intuition: early training establishes broad linguistic competence, late training shapes the model toward the desired distribution. Annealing on high-quality data at low LR is much more sample-efficient at improving downstream benchmarks than mixing high-quality data uniformly throughout.

Llama 3 [Dubey et al., 2024] reports a separate "annealing run" at the end where they continued training on a heavily filtered, high-quality 40B-token mix at 1/10 the peak learning rate. This single phase contributed several MMLU points despite being a tiny fraction of total training tokens.

## Long-context pretraining: the staged sequence length recipe

Training a model with 128K context from scratch is wasteful — most of the early pretraining benefits from short-sequence data, and the FLOPs cost of long-sequence attention is quadratic. Modern long-context models stage the sequence length:

- **Stage 1** (0-90 % of tokens): 4K context. Builds general capability.
- **Stage 2** (90-95 %): 32K context. Introduces long-range patterns.
- **Stage 3** (95-100 %): 128K context, with curated long-document data (books, code repos, long papers).

Llama 3 used a 4-stage curriculum reaching 128K. NVIDIA's Nemotron-340B used a similar staged approach. Qwen3 uses staged pretraining reaching 128K with YaRN extension to 256K at the end.

The key insight is that the model's general language modeling capability is established at short context, and the long-context phase only needs to teach the model how to use the extended attention. This is much cheaper than training on long sequences throughout. Sequence packing — concatenating multiple short documents into a single long sequence with attention masks — gives nearly free compute utilization during the short-sequence stages.

## μP: the parameterization that lets you tune at small scale

![fig4: μP scaling across widths](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig4_mup_scaling.png)

A practical headache of large-scale training: hyperparameters tuned at 1B don't transfer to 70B. Learning rate, especially. The standard parameterization breaks because activation magnitudes scale with width.

**μP (Maximal Update Parameterization)**, from [Yang et al., 2022], fixes this. Under μP, you scale initialization variances and learning rates by the layer width such that activations and gradients have the same magnitude across model widths. Net effect: tune learning rate at 100M params, transfer it to 100B params.

```python
# Standard: per-layer LR doesn't depend on width
# μP: scale LR for hidden Linear layers by 1 / fan_in_ratio
def mup_lr(base_lr, fan_in, base_fan_in=256):
    return base_lr * (base_fan_in / fan_in)
```

Cerebras and Microsoft Research have both shown experiments where μP-tuned 7B → 70B → 700B models track each other's loss curves to within 1 % at the same fraction of training. Without μP you'd burn millions of dollars finding the right LR at each scale.

The full μP recipe scales four things differently at each layer width:
1. Initialization standard deviation: $\sigma \propto 1/\sqrt{\text{fan}_{\text{in}}}$
2. Learning rate for hidden layers: $\eta \propto 1/\text{fan}_{\text{in}}$
3. Output layer multiplier: $\text{logits} = (1/d) \cdot W_{\text{out}} \cdot h$ rather than $W_{\text{out}} \cdot h$
4. Embedding layer LR: held constant (not scaled)

Most production runs in 2026 use μP for the LR transfer at minimum. Some go further with μTransfer for batch size and weight decay. [Wortsman et al., 2024] showed μP also makes training more robust to noise and easier to recover from instability.

## Parallelism: FSDP, ZeRO, pipeline, tensor

![fig1: ZeRO/FSDP memory stages](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig1_zero_stages.png)

Once a model exceeds single-GPU memory, you have four orthogonal axes of parallelism:

1. **Data parallelism (DP)** — replicate model on each GPU, split batch.
2. **Sharded data parallelism (FSDP / ZeRO-3)** [Rajbhandari et al., 2020] — shard parameters across DP ranks, gather them just-in-time before each layer's forward pass.
3. **Tensor parallelism (TP)** [Shoeybi et al., 2019] — split each matmul across GPUs (Megatron style).
4. **Pipeline parallelism (PP)** [Huang et al., 2019] — split layers across GPUs, micro-batch through them.

For a 70B model on 64 H100s the typical recipe is:

```text
TP=8 within node (NVLink)
PP=2 across nodes (200 Gbps Infiniband)
DP=4 with FSDP across the remaining axis
```

This gives you 64 GPUs total (8 × 2 × 4), with the most communication-heavy parallelism (TP) confined to the fast NVLink within each node, and the PP bubbles overlapped with the next micro-batch. ZeRO-3 (the FSDP variant) is the default in 2026 because it gets you to 70B with no code changes:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = TransformerModel(...)
model = FSDP(model,
             sharding_strategy=ShardingStrategy.FULL_SHARD,
             mixed_precision=MixedPrecision(
                 param_dtype=torch.bfloat16,
                 reduce_dtype=torch.float32,
             ),
             auto_wrap_policy=transformer_auto_wrap_policy(
                 transformer_layer_cls={LlamaDecoderLayer},
             ))
```

The `mixed_precision` block is critical. **Always reduce in fp32** — bf16 reductions silently underflow when summing 1000s of tiny gradients across DP ranks. We've debugged loss-divergence runs that turned out to be a single missed `reduce_dtype=fp32`.

For the giant models (DeepSeek-V3 671B, Qwen3-Max), even FSDP isn't enough — you need 4D parallelism with custom code. Megatron-LM, NVIDIA NeMo, ColossalAI, and DeepSpeed are the four production frameworks. Most labs have pulled custom forks of one of them.

## Worked example: parallelism choice for 70B on 64 H100s

![fig2: pipeline parallelism schedule](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig2_pipeline_parallelism.png)

The math of why TP=8, PP=2, DP=4 is the right answer for this configuration. Each H100 has 80 GB HBM. Total HBM: 64 × 80 = 5120 GB.

**Memory budget per GPU.** Per-rank we need:
- Weights: 70B × 2 bytes (BF16) / TP / PP = 70e9 × 2 / 16 = 8.75 GB
- Gradients: same as weights = 8.75 GB
- Adam optimizer state: 4 bytes/param × 2 (m, v) / TP / PP / DP_FSDP = 70e9 × 8 / 64 = 8.75 GB if FSDP shards optimizer
- Activations (with gradient checkpointing): scales with batch × seqlen × model dim, typically 30-50 GB for 70B at seqlen 8K
- KV/scratch buffers: ~5 GB

Total: ~62 GB out of 80, leaves ~18 GB headroom for FlashAttention scratch, NCCL buffers, peak activation moments. Tight but works.

**Why not pure FSDP=64?** Pure FSDP on 64 GPUs requires all-gathering 70B params before each layer's forward pass. At 70B × 2 bytes, that's 140 GB of data per gather, divided across 64 ranks = 2.2 GB/rank/gather, on a per-layer basis. With 80 layers, that's ~175 GB of gather traffic per forward pass per rank. Even at NVLink's 900 GB/s, this is ~200 ms of pure communication per forward pass — competitive with the compute time, killing throughput.

**Why not pure TP=64?** TP communicates after every matmul (all-reduce on row-parallel layers). The all-reduce volume per token is $O(d_{\text{model}} \times \text{TP})$. At TP=64 on inter-node links (200 Gbps Infiniband), this dominates. TP only works well within a single NVLink-connected node (max 8 GPUs in current generations).

**Why TP=8, PP=2, DP=4?** TP=8 inside each node uses fast NVLink for the all-reduce-heavy TP traffic. PP=2 splits the model across two nodes, with PP communication being relatively rare (once per micro-batch). DP=4 with FSDP shards the optimizer state across 4 DP ranks, each holding 1/4 of the weights × 1/(TP×PP) of the layers. The PP pipeline bubble (10-20 % of step time) is acceptable; the alternative parallelism configurations have worse overhead.

This is one configuration. Different model sizes and cluster topologies have different optima. The general principle: minimize cross-node communication, fit TP within the fastest interconnect, use PP for the next layer of granularity, fill remaining parallelism with FSDP/DP.

## Failure modes that only appear above 1000 GPUs

Things that don't matter at 8 GPUs and bring down 8000-GPU runs:

**Silent data corruption.** A single GPU returning bad math (cosmic ray, voltage glitch) creates a NaN that gets all-reduced. Detect by checksumming gradients periodically — if rank 0's gradient norm differs from rank 100's by more than $10^{-3}$, halt and re-shard.

**Loss spikes.** A bad batch (a Python parser dump, an HTML soup, etc.) causes a gradient spike that shifts the optimizer momentum and never recovers. Defense: aggressive grad clipping (1.0 default, sometimes 0.3 for very large models) plus a "skip this batch" trigger when grad norm > 5x running median. The Llama 3 paper mentions "checkpoint rollback" as standard procedure: when a loss spike isn't recoverable within 50 steps, roll back to the last good checkpoint, lower the LR by 30 %, and resume.

**NCCL hangs.** A single node's NIC firmware bug causes that rank's all-reduce to never return. The other 999 ranks block forever. Defense: NCCL_TIMEOUT_S=300 and a watchdog that reschedules.

**Checkpoint corruption.** A 2.7 TB FSDP checkpoint (Qwen3-32B optimizer state) takes 20+ minutes to write. If a node OOMs mid-write, you've lost a step *and* the previous good checkpoint if you overwrote in place. Always write to `step_xxx.tmp/`, fsync, then `mv`.

**Heterogeneous hardware.** Even within "same SKU" H100s, clock speeds vary 2-3 % batch to batch. Slow ranks straggle. Defense: sort GPUs by per-rank throughput at warmup, place stragglers on TP=1 ranks.

**Embedding collapse.** If the embedding matrix's gradient flow gets clipped or NaN-corrupted in the early epochs, embeddings can collapse to near-identical vectors. Symptom: training loss stuck at log(V), perplexity at V. Diagnosis: monitor singular values of the embedding matrix; if the top singular value dominates by 10× over the next, the embeddings have collapsed. Fix: rewind to before the collapse, increase embedding LR by 2×, restart.

**Gradient norm divergence.** Distinct from loss spikes. Gradient norms slowly trend upward over thousands of steps despite stable loss. This typically presages a future collapse. Monitor `||g||_2` per parameter group, not just global. Specific layer types (often the LM head or final transformer layer) are usually the culprit.

The Meta OPT-175B logbook [Zhang et al., 2022] is still the best public document on what an actual large run looks like. They had a node failure on average every 1.7 days during their 33-day run.

## What actually happens during a run

![fig5: training loss curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig5_loss_curve.png)

A modern 70B pretrain run (LLaMA-3 70B, 15T tokens, ~1900 H100s for 60 days) burns roughly:

- **Tokens**: 15T input, batch size ~16M tokens
- **Steps**: 15T / 16M ≈ 940K
- **Wallclock per step**: ~5.5 s (with sequence packing)
- **Total compute**: ~$2.5e25$ FLOPs ≈ 6000 H100-years

LR schedule is cosine decay from a brief warmup to ~10 % of peak. Peak LR is around 1.5e-4 for 70B (lower for bigger models, μP-scaled if you're disciplined). Weight decay 0.1, β2=0.95, AdamW.

Eval is checkpoint-by-checkpoint: every 2000 steps, evaluate on 5-10 small benchmarks (Hellaswag, ARC-easy, OpenBookQA, GSM8K-100, HumanEval-50). Loss is cheap to track; downstream eval is the actual signal.

## Production reality: what frontier labs actually do

The papers describe the architecture; the production system is mostly logistics. Three things every frontier pretraining team does that don't make it into model cards:

**Multiple parallel runs at different scales.** A 70B target run is preceded by 1B and 7B "scout" runs at 1/100 and 1/10 the data, with the same architecture, optimizer, and data mix. The scout runs validate the recipe and let you tune hyperparameters that μP can't transfer (mostly schedule details). When the scout runs look good, the 70B kicks off with high confidence the recipe is sound. Llama 3 paper mentions running 405B alongside 70B to validate scaling behavior continuously.

**Aggressive monitoring.** Loss curves, grad norms, individual parameter stats, attention pattern histograms, expert utilization rates (for MoE), activation magnitudes per layer, eval scores. Monitoring dashboards are not optional infrastructure — they're the only way to catch problems before they accumulate. Anthropic open-sourced parts of their monitoring (clipping ratios, activation stats) in their model cards; OpenAI's described in the GPT-4 paper.

**Continuous data refinement.** The training data is not static. The team continuously runs new quality classifiers, removes newly identified contamination, adds new synthetic data sources. Llama 3 explicitly mentions running the entire training data through a retrained quality classifier mid-run and re-weighting subsequent batches accordingly. The "data is fixed before training" model is academic; production teams iterate.

**On-the-fly tokenization vs pre-tokenized.** Pre-tokenizing 15T tokens takes weeks and 100+ TB of storage. Streaming tokenization during training requires the tokenizer to keep up with the data loader, which means highly optimized Rust tokenizers. Most production setups precompute token IDs but stream them from a distributed file system rather than loading raw text.

## Common Pitfalls

Five things I've personally seen wreck pretraining runs.

**1. Forgetting to set NCCL collectives to bf16/fp32 mismatch.** Default NCCL all-reduce in fp16 mode silently underflows when summing thousands of small gradient values. Always: `NCCL_PROTO=Simple` for stability and `reduce_dtype=fp32` for FSDP. The Llama 3 paper notes they had to add this to recover from instability around step 500K.

**2. Wrong RoPE base for context length.** RoPE uses a base frequency $\theta = 10000$. At 32K context, frequencies aliased and the model couldn't learn long-range patterns. Fix: scale base to $10000 \times (\text{seqlen}/2048)^{d/(d-2)}$ per [bloc97 NTK-aware] or use YaRN [Peng et al., 2024]. We chased a 32K-context degradation for a week before realizing the RoPE base was still set for the 4K pretraining stage.

**3. Loss masking on packed sequences.** Sequence packing concatenates multiple documents into one training sequence. If you don't mask cross-document attention, the model learns spurious cross-document correlations and quality drops 1-3 % MMLU. Always emit a `position_ids` and `attention_mask` that resets at document boundaries.

**4. Activation checkpointing with non-deterministic ops.** Re-running the forward during the checkpoint-recompute phase requires deterministic kernels. Some FlashAttention and GeLU implementations have non-deterministic numerical reductions. Result: gradients computed against different forward values than were saved. Subtle quality regression that only shows up after a few thousand steps. Always set `torch.use_deterministic_algorithms(True)` for checkpointed regions or verify the kernel is deterministic.

**5. Optimizer state not sharded to disk.** For a 70B model, AdamW optimizer state is 70B × 8 bytes = 560 GB. Saving this to a single shared filesystem with thousands of nodes simultaneously can deadlock. Use FSDP's sharded checkpointing (each rank writes its shard) with parallel writes. We had a checkpoint that took 4 hours to save because of inadvertent serialization.

## Research frontier 2024-2026

What's coming after the current Llama-style "filter + dedup + train" recipe:

**Data efficiency at scale.** [Marion et al., 2023] (When Less is More) showed that training on a curated 200B-token subset can match training on a 1T-token unfiltered set. This is essentially "early stopping for data" — find the most informative examples and skip the rest.

**Curriculum-driven mixing.** [Albalak et al., 2024] showed dynamic data mixing — adjusting the proportion of each data source over the course of training based on how the model is improving — beats fixed mixing by 1-2 MMLU. Doremi [Xie et al., 2023] proposed a min-max optimization for finding optimal weights automatically.

**Compute-optimal beyond Chinchilla.** Mixture-of-Experts changes the scaling law. [DeepSeek-AI, 2024] reported their own MoE scaling laws derived from V3 experiments: optimal active-to-total ratio, optimal expert count, optimal expert size all depend on training compute in non-trivial ways. The community is still calibrating MoE-specific scaling laws.

**Continual pretraining.** Most production models are now updated, not retrained from scratch. Llama 3.1 was Llama 3 + continued pretraining on additional data + post-training. Continued pretraining is much cheaper but requires careful learning-rate scheduling to avoid catastrophic forgetting.

## What's Next

Pretraining is 70 % data engineering and 30 % distributed-systems engineering. The architecture choice is the smallest of the three. Get the data mix right, dedup hard, train past Chinchilla for inference-cost reasons, use μP so your hyperparameters transfer, pick FSDP plus TP plus PP based on your hardware topology, and write defensive code for the failure modes that only appear at scale.

Next chapter: **post-training**. SFT, DPO, RLHF, RLAIF — what each actually optimizes, when reward models fail (and they fail constantly), the LoRA-vs-full-FT debate, and the production recipes that turn a base model into something a customer will use.

## References

- Shoeybi, M., Patwary, M., Puri, R., et al. (2019). Megatron-LM: Training multi-billion parameter language models using model parallelism. *[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)*.
- Huang, Y., Cheng, Y., Bapna, A., et al. (2019). GPipe: Efficient training of giant neural networks using pipeline parallelism. *NeurIPS*.
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling laws for neural language models. *[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)*.
- Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. *SC'20*.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training compute-optimal large language models (Chinchilla). *NeurIPS*.
- Lee, K., Ippolito, D., Nystrom, A., et al. (2022). Deduplicating training data makes language models better. *ACL*.
- Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and efficient foundation language models. *[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)*.
- Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)*.
- Yang, G., Hu, E., Babuschkin, I., et al. (2022, 2023). Tuning large neural networks via zero-shot hyperparameter transfer (μP / μTransfer). *NeurIPS*.
- Zhang, S., Roller, S., Goyal, N., et al. (2022). OPT: Open pre-trained Transformer language models. *[arXiv:2205.01068](https://arxiv.org/abs/2205.01068)*. (See also OPT-175B logbook.)
- Sardana, N., Portes, J., Dohmen, S., & Frankle, J. (2023). Beyond Chinchilla-Optimal: Accounting for inference in language model scaling laws. *[arXiv:2401.00448](https://arxiv.org/abs/2401.00448)*.
- Gunasekar, S., Zhang, Y., Aneja, J., et al. (2023). Textbooks are all you need (Phi-1). *[arXiv:2306.11644](https://arxiv.org/abs/2306.11644)*.
- Muennighoff, N., Rush, A., Barak, B., et al. (2023). Scaling data-constrained language models. *NeurIPS*.
- Marion, M., Üstün, A., Pozzobon, L., et al. (2023). When less is more: Investigating data pruning for pretraining LLMs at scale. *[arXiv:2309.04564](https://arxiv.org/abs/2309.04564)*.
- Xie, S., Pham, H., Dong, X., et al. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. *NeurIPS*.
- DeepSeek-AI. (2024). DeepSeek-V3 technical report. *[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)*.
- Dubey, A., Jauhri, A., Pandey, A., et al. (2024). The Llama 3 herd of models. *[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)*.
- Penedo, G., Kydlíček, H., et al. (2024). The FineWeb datasets: Decanting the web for the finest text data at scale. *NeurIPS*.
- Soldaini, L., Kinney, R., Bhagia, A., et al. (2024). Dolma: An open corpus of three trillion tokens for language model pretraining research. *ACL*.
- Li, J., Fang, A., Smyrnis, G., et al. (2024). DataComp-LM: In search of the next generation of training sets for language models. *NeurIPS*.
- Together AI. (2024). RedPajama-V2 technical report.
- Villalobos, P., Sevilla, J., Heim, L., et al. (2024). Position: Will we run out of data? Limits of LLM scaling based on human-generated data. *ICML*.
- Shumailov, I., Shumaylov, Z., Zhao, Y., et al. (2024). The curse of recursion: Training on generated data makes models forget. *Nature*.
- Wortsman, M., Liu, P., Xiao, L., et al. (2024). Small-scale proxies for large-scale Transformer training instabilities. *ICLR*.
- Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2024). YaRN: Efficient context window extension of large language models. *ICLR*.
- Albalak, A., Pan, L., Raffel, C., et al. (2024). Efficient online data mixing for language model pre-training. *NeurIPS*.
