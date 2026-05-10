---
title: "大模型工程（五）：推理加速的十八般武艺"
date: 2026-04-30 09:00:00
tags:
  - LLM
  - inference
  - vllm
  - quantization
  - paged-attention
  - speculative-decoding
categories: 大模型工程
series: llm-engineering
series_order: 5
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "KV cache 力学、paged attention、continuous batching、speculative decoding、INT8/INT4/AWQ/GPTQ 量化，以及 vLLM、SGLang、TensorRT-LLM 的取舍。"
translationKey: "llm-engineering-5"
---
钱其实是花在推理上的。单个 70B 级别的模型，要是服务 1000 个并发用户，每秒生成 50 个 token，大概跑 3 个月就把训练这台模型的 GPU 预算烧光了。本章所有内容都围绕两个指标展开：首 token 延迟（TTFT）和 token 间延迟（ITL）。还有一个比率：每百万输出 token 消耗的 GPU 秒数。

训练是一次性资本支出——你把成本分摊到数百万次推理调用上。推理是持续的运营支出，而且不像训练那样能摊销。tokens-per-GPU-second 提升 0.5 倍，会在产品整个生命周期里每天复利累积。这就是为什么每个正经的 LLM 团队至少有一名全职工程师搞推理，也是为什么开源社区在五年内推出了四波不同的推理引擎（FasterTransformer → DeepSpeed-Inference → vLLM → SGLang/TensorRT-LLM/llama.cpp）。

![LLM Engineering (5): Inference Optimization — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/illustration_1.png)

## 两个特性截然不同的阶段

![fig1: prefill vs decode compute pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig1_prefill_vs_decode.png)

每次 LLM 推理调用都分两个阶段：

1.  **Prefill（提示词处理）**：输入 token 并行跑过模型，填满 KV cache。这是计算密集型（Compute-bound）。70B 模型处理 4K token 的 prompt 大概需要 280 TFLOP——能把一张 H100 饱和运行约 70 ms。
2.  **Decode（生成）**：一次生成一个 token，基于缓存的 key 和 value 做注意力。这是内存密集型（Memory-bound）。每个 decode 步都要读取整个 KV cache（几 GB 大小）才能产出一个 token。

这种不对称性就是一切。Prefill 可以在用户间很好地批处理（同一个 kernel，不同序列）。Decode 用 naive batching 效果很差，因为每个用户所处的序列位置都不一样。主流推理引擎全是围绕这种不对称性设计的。

有个常用的经验法则：TTFT 主要由 prefill 主导，ITL 主要由 decode 的内存带宽主导。想降低 TTFT，就堆 FLOPs（更多 SMs，tensor parallelism）。想降低 ITL，就堆内存带宽（HBM3 胜过 GDDR，或者通过量化减少参数量）。

用算术强度来论证就更严谨了。70B 模型处理 4K prompt 的 prefill：模型权重（BF16 下 140 GB）加载一次，操作 4096 个 token。算术强度 ≈ 每读取一个参数字节对应 4096 FLOP。同样是这个模型，单 token 的 decode：权重加载一次，操作 1 个 token。算术强度 ≈ 每字节 1 FLOP。H100 在 BF16 下算力 989 TFLOPS，HBM 带宽 3.35 TB/s，平衡点大概在 ~295 FLOP/byte。Prefill 的 4096 FLOP/byte 远高于平衡点（计算密集型）。Decode 的 1 FLOP/byte 低于平衡点两个数量级（内存密集型）。这两个阶段需要的硬件特性不同， Serving 栈如果不把它们分开处理，性能就会躺在地板上起不来。

## KV cache：支撑长上下文的数据结构

![fig2: KV cache size growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig2_kv_cache_growth.png)

KV cache 存储了每一层中每个 prior token 的投影 K 和 V 向量。对于 70B 模型，GQA-8，32K 上下文（数据来自第一章）：

$$\text{KV} = 2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 \cdot 32{,}768 \cdot 2 \text{ bytes} = 8.6 \text{ GB}$$

这是单个请求的量。如果有 50 个并发请求，那就是 430 GB 的 KV cache——远超你的模型权重。瓶颈是 KV cache，不是权重。

Naive 实现会为每个请求分配一个大小为 `max_context` 的连续 tensor。这有两个问题：

1.  **内部碎片化。** 一个只用了 1K token 的请求依然预留了 32K 的内存。
2.  **无法增长。** 超过 max_context，你在 decode 中途就会 OOM。

高负载下会出现第三个问题：**外部碎片化**。请求在不同时间到达和离开。空闲内存确实存在，但散落在不连续的区域，没有任何一块足够大给新请求。在可变负载下运行几小时的服务器，仅外部碎片化就会损失 20-40% 的可寻址 KV 内存。这和操作系统上世纪 60 年代用分页解决的问题一样，解决方案也一样。

## Paged attention

![fig3: paged attention block table](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig3_paged_attention.png)

vLLM 的杀手锏，来自 2023 年的论文 [Kwon et al.][kwon-vllm]，就是 **paged attention**。KV cache 按固定大小的 **block** 分配（通常是 16 个 token 的量），每个请求通过 **block table** 映射逻辑位置到物理 block。就像操作系统的虚拟内存。

```
Request A: needs 47 tokens of KV → 3 blocks (16+16+15)
Request B: needs 200 tokens of KV → 13 blocks
Block table for A: [0x47, 0x12, 0x3a]
Block table for B: [0x05, 0x09, 0x21, ...]
```

好处很明显：

-   **几乎零浪费**：每个请求只有最后一个 block 有内部碎片，而且最多也就 15 个 token。
-   **内存共享**：前缀共享的请求（系统 prompt、few-shot 示例、工具定义）可以共享相同的物理 block。这就是 prompt caching 的来源。
-   **易于抢占**：内存压力下，把请求的 block  swap 到 CPU；有空闲再搬回来。

原论文基准测试中，vLLM 在 LLaMA-13B 上的服务吞吐量比 naive 的 Hugging Face Transformers 循环提高了 14-24 倍。这些数据至今依然成立——虽然实际生产环境的差距高度依赖工作负载混合。对于 batch-1 单流服务，提升接近 2 倍；对于高并发混合长度流量，5-10 倍是常态；14-24 倍这个数字是在对比 *真正* naive 的 HF 代码（完全不做 batching）时出现的。重点不在于具体的倍数，而是 paged attention 让 KV 内存变成了可管理的资源，而不是硬约束。

一个最小化用法：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-7B", gpu_memory_utilization=0.9,
          max_model_len=32768, enable_prefix_caching=True)
prompts = ["Explain MoE in one paragraph.",
           "Explain GQA in one paragraph."]
out = llm.generate(prompts, SamplingParams(max_tokens=200, temperature=0.7))
```

`enable_prefix_caching=True` 启用了针对重复 prompt 前缀的自动 block 共享。如果系统 prompt 在 1000 个请求中重复，这对那些请求来说是 5-50 倍的 prefill 加速。

### Block Manager：分页到底怎么运作

Block Manager 位于每个分页注意力引擎的核心。它拥有一池物理 KV block（通常每个 block 大小为 16 tokens × 2 (KV) × num_layers × num_heads × head_dim bytes），维护空闲列表，并为 incoming 请求原子性地分配 block。当请求需要增长（下一个 decode 步）时，manager 弹出一个空闲 block，将其物理地址追加到请求的 block table，然后返回。在 copy-on-write 语义下，前缀共享的 block 是引用计数的：fork 一个序列（例如为了并行采样、beam search 或 speculation rollback）会增加 refcount；释放则减少；只有当 refcount 归零时才释放物理内存。

生产环境中，“驱逐到 CPU"这条路径很重要。当 GPU 空闲 block 用完但新请求在排队时，调度器会选一个受害者请求（通常是 FIFO 或运行时间最长的），通过 PCIe 将其 block 复制到 pinned host 内存，然后释放 GPU block。当内存可用时，请求再被搬回。驱逐开销是实打实的（Gen5 ×16 的 PCIe 带宽 64 GB/s，意味着驱逐和重载一个 8 GB 请求需要 100 ms），但这让系统能够通过不直接拒绝请求来遵守延迟 SLO。

## Continuous batching

![fig4: continuous vs static batching](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig4_batching_timeline.png)

另一场革命。Static batching 会等待 batch 中最慢的序列完成后再开始下一个 batch。面对变长输出（这总是常态），你会浪费大量 GPU 时间空闲等待。

**Continuous batching**（[Yu et al., Orca, OSDI 2022][yu-orca]，由 vLLM 普及）：每个 decode 步，驱逐完成的序列并接纳等待的序列。只要有工作在排队，GPU 运行的序列数永远不会低于 `max_batch_size`。

对于 1000 个 prompt，输出长度从 50 到 2000 token 不等：

-   Static batching：总墙钟时间 ≈ 2000 token 输出的时间（最坏情况主导）。
-   Continuous batching：总墙钟时间 ≈ 所有输出的平均值。

吞吐量提升会与 paged attention 产生复利效应，因为在 batch 中途接纳新序列只是分配新的 KV block——不需要重新分配整个 tensor。

### Orca 的迭代级调度

Orca 论文引入了两种机制，在 2026 年的引擎中依然无处不在。**迭代级调度**将 batching 边界与请求边界解耦：每个 transformer 迭代独立调度，所以一个在第 500 步完成的请求不会阻塞一个需要 1500 步的请求。**选择性 batching** 通过允许注意力算子按每序列形状操作，而线性层（不关心位置）一起 batching，来处理 prefill/decode 不匹配的问题。现代引擎进一步融合了这一点：vLLM 和 SGLang 都运行 "chunked prefill"，其中一次 forward  pass 中一个请求的 prefill 与其他请求的 decode 交错进行，即使 decode 队列稀疏也能保持 GPU 利用率在 90% 以上。

调度器的决策至关重要：你什么时候在 batch 中途接纳新请求？Naive 的做法是立即接纳。但在 ongoing decode 旁边接纳一个 32K token 的 prompt 进行 prefill 会 dramatically  spike 每次迭代的延迟（32K 的 prefill FLOPS 大约是单个 decode 步的 100 倍）。生产调度器会限制每次迭代的“预算”，并将长 prefill 拆分成 chunk，与 decode 步交错执行。这是一个隐藏的权衡旋钮，决定了你的 TTFT p99 在负载下是 500 ms 还是 5 秒。
## Speculative decoding

![LLM Engineering (5): Inference Optimization — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/illustration_2.png)


![fig5: speculative decoding tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig5_speculative_tree.png)


这是个巧点子，现在已经是标配了。Decode 阶段通常是 memory-bound 的：每一步都得读遍整个模型权重才能吐出一个 token。要是能一次性验证 *N* 个 token 会怎样？

Speculative decoding ([Leviathan et al., 2023][leviathan-spec]) 的核心思路是用一个 **draft model**（小模型，速度快——比如 1B 模型）先 proposes $k$ 个 token。目标模型（也就是你真正要服务的 70B 模型）在单次 forward pass 里验证它们：它看到提出的前缀，并行计算 $k+1$ 个位置的概率。跟目标模型 argmax 匹配的 token 保留；第一个不匹配的 token 及其后面的全部丢弃；从这里重新 sample。

数学上：如果 draft 模型有 $\alpha$ 的概率猜对，且 decode 是 memory-bound 的（所以 $k+1$ 个 token 的 forward pass 成本跟 1 个 token 差不多），期望加速比是 $(1 - \alpha^{k+1})/(1 - \alpha)$。对于 $\alpha=0.7$，$k=4$：加速 2.5 倍。

实践中，draft 模型通常是从目标模型蒸馏出来的。主要有三派竞争：

- **Draft-target two-model pairs**，Leviathan 最初的 formulation：小蒸馏模型提出，目标模型验证。质量高，但维护两个同步的 checkpoint 是运维噩梦。
- **Medusa** ([Cai et al., 2024][cai-medusa])：给目标模型本身加 4-5 个额外的预测头。每个头从相同的 backbone activations 预测位置 $t+1, t+2, t+3, ...$。不需要单独的 draft 模型。加速 2-3 倍且无质量损失；在 base 模型上训几个 GPU-days 就行。
- **EAGLE / EAGLE-2** ([Li et al., 2024][li-eagle])：一个 autoregressive head，基于倒数第二层的 hidden state 加上前一个 token 进行预测。EAGLE-2 引入了基于动态树的 verification，每一步探索多个 draft 分支。在大多数 workload 上能达到 3-4 倍加速；目前是 autoregressive speculation 的 SOTA。

```python
# vLLM speculative decoding
llm = LLM(model="Qwen/Qwen3-32B",
          speculative_model="Qwen/Qwen3-1.7B",
          num_speculative_tokens=5)
```

有个坑：只有当目标 GPU 是 memory-bound 而不是 FLOPs-bound 时，speculation 才有用。大 batch 是 FLOPs-bound 的（权重加载一次，摊销到 batch 里所有 sequence 上）；spec decoding 帮不上忙。单 stream 低延迟 serving 是 memory-bound 的，收益巨大。

correctness 上有个细节：验证步骤其实是针对目标分布做 *rejection sample*。如果 token $t$ 的 draft 概率是 $p_d(t)$，目标概率是 $p_t(t)$，接受概率为 $\min(1, p_t(t)/p_d(t))$；拒绝时，从正比于 $\max(0, p_t(t) - p_d(t))$ 的修正分布中 sample。这保证输出分布 *exactly* 是目标模型的分布，不管 draft 质量如何。Spec decoding 是无损的：draft 差只意味着接受率低，不会导致输出变差。这也是为什么团队上线不用做质量回归测试——根本没质量可回归。

## Quantization: INT8, INT4, FP8

推理量化就是把 bf16 转成 INT8 / INT4 / FP8，为了省显存和带宽。通常量化模型权重；有时候也量化 activations。

### INT8 weight-only quantization

最简单。按通道量化每个权重矩阵：$w_q = \text{round}(w / s)$，其中 $s$ 是每个输出通道的 scale。推理时，在 matmul 之前即时反量化成 bf16。

显存：权重减半（16-bit → 8-bit）。质量：通常 perplexity 损失 <0.5 %，无需 calibration。白捡的便宜。

### INT4: GPTQ vs AWQ

INT4 就难多了。Naive 的 round-to-nearest 会丢 2-5 % perplexity。能用的算法 mainly 两个：

**GPTQ** ([Frantar et al., 2022][frantar-gptq])：OBQ (Optimal Brain Quantization) 的扩展，让它在十亿参数模型上可行。逐列量化权重矩阵；每量化一列后，更新剩余未量化列以补偿引入的误差，使用 Cholesky 分解的 Hessian 逆矩阵。数学上本质是每列做一次结构化 least-squares 求解。计算成本：单张 A100 上 70B 模型需要几小时，峰值显存大约等于小 calibration 集上一层的 activations（通常 128 samples × 2048 tokens）。

**AWQ** ([Lin et al., 2023][lin-awq])：观察到 1 % 的权重是 "salient" 的（大 activations 穿过它们）。应用 per-channel scaling $s$ 保护 salient 权重免受量化噪声影响——等价于量化前将输入 activations 乘以 $s$，权重除以 $s$。全精度下这在数学上是 no-op，但重塑了量化 landscape。然后均匀量化。比 GPTQ 快（不用解 Hessian，只需搜索 scaling 因子），大多数模型上质量略好，对 MoE 更友好。

**SmoothQuant** ([Xiao et al., 2022][xiao-smoothquant])：正交于 GPTQ/AWQ —— 解决 *activation* 量化。挑战在于 activations 有 outlier 通道（少数通道量级是其他的 100 倍），这会毁掉 INT8 activation 量化。SmoothQuant 通过 per-channel scaling 把 activation 量级迁移到权重上，让两边都能量化到 INT8。用作 INT8 W8A8 量化前的预处理步骤。

GPTQ 和 AWQ 都能在 LLaMA 类模型上实现 INT4 weight-only 且 perplexity 损失 <1 %。对于 MoE，路由让 calibration 变复杂了——不同 expert 看到的 activation 分布差异很大，统一 scaling 因子会保护不足那些 rarely-used 的 experts。Per-expert AWQ 是目前的最佳实践。

```python
# Loading an AWQ-quantized model with vLLM
llm = LLM(model="Qwen/Qwen3-32B-AWQ", quantization="awq",
          dtype="float16", max_model_len=32768)
```

### FP8 (and the H100/H200 hardware path)

H100 及更新的 GPU 有 FP8 tensor cores，吞吐量是 BF16 的 2 倍。FP8 推理是现代路径：权重存 FP8，计算时 activations 转 FP8，累加用 FP32。质量损失忽略不计（<0.1 %），因为 FP8 比 INT8 动态范围更大。

FP8 有两种格式：E4M3（4 位 exponent，3 位 mantissa）用于 activations 和权重——范围小，精度高；E5M2（5 位 exp，2 位 mantissa）用于 gradients 和 KV cache——范围大，精度低。硬件把反量化 scaling 融合进 matmul，所以没有 "INT4 反量化开销" 成本；FP8 是 H100/H200 上的光速路径。

校准对 FP8 很重要，尽管 perplexity 损失很小。标准 recipe：收集 128-512 个 calibration 样本上的 activation 统计信息，计算 per-tensor 或 per-token scales，作为 checkpoint 的一部分存储。Per-token activation scales（在运行时从实际 batch 计算）能以微小的延迟成本换取最佳质量。NVIDIA 的 TransformerEngine 和 vLLM 的 `--quantization fp8` 都实现了这个。

坑在于：FP8 需要硬件支持。INT4/INT8 在任何 GPU 上都能跑。FP8 仅限 H100/H200/B200。如果你在 A100 上部署，INT8/INT4 是你的选项。

### KV cache quantization

显存账单的另一半。压缩 KV cache 到 INT8（省 2 倍），或 INT4（仔细做 per-token calibration 损失 1-2 % 质量）。vLLM 和 SGLang 都支持 FP8 KV cache；INT4 KV 还是 research-grade，但 FlashInfer 有 kernels。

FP8 KV cache 是 2026 年的主力。它以 <0.1 % 的质量损失将显存减半（所以同一 GPU 上并发请求翻倍）。Per-token scaling 几乎是免费的，因为 scales 是 inline 计算的。INT4 KV 需要更多操心——AWQ 利用权重的 per-channel salience 结构并不直接适用于 KV（这是数据，不是参数）。目前的最佳实践是 per-token-per-head scaling 加 outlier  preservation：保留约 1 % 的通道用更高精度，其余激进量化。

## SGLang and RadixAttention

vLLM 的前缀缓存是在具有相同前缀的请求间共享 blocks。SGLang ([Zheng et al., 2024][zheng-sglang]) 用 **RadixAttention**  generalized 了这个：所有活跃 KV blocks 的 radix 树，其中从根到叶的每条路径代表一个 token 序列。新请求在树中查找最长匹配前缀，共享那些 blocks，只计算 suffix。

为什么这比 vLLM 的 exact-prefix 缓存更重要：agent  workload 会分支。ReAct  agent 可能发出 5 个 tool-calling 子查询，每个都有相同的 system prompt + tools + conversation history，但 suffix 不同。线性前缀缓存能让每个子查询匹配共享的根，但帮不了子查询之间共享状态。RadixAttention 可以。

SGLang 论文里报告的提升：相比 vLLM 等价的前缀缓存，agent 和 structured-output  workload 吞吐量提升 5 倍，简单 chat 提升约 1.5 倍。对于不共享前缀的流量，收益会缩小。

实现大致是：维护一个以 token IDs 为 key 的 radix 树，节点持有 KV block 引用。新请求到来时，尽可能深地遍历树匹配输入前缀；增加所有触及节点的 refcount；为未匹配的 suffix 分配新 blocks；完成后减少 refcounts；显存压力升高时 LRU-evict 子树。数据结构成本相比节省的资源微不足道。

## TensorRT-LLM specifics

NVIDIA 的 TensorRT-LLM 是第三大引擎。它的差异化特点：

- **Compiled kernels**：每个模型都针对特定 GPU、batch size  profile 和 sequence lengths 编译成 TensorRT 引擎。编译耗时 10-30 分钟； resulting 引擎在支持的 workload 上比 vLLM 的 runtime-compiled  equivalents 快 1.1-1.3 倍。
- **Plugin model**：attention 的自定义 kernels（FMHA, paged FMHA, FlashAttention-3 路径）、MoE 路由和量化作为插件暴露。插件模型比 vLLM 的灵活性更脆弱，但允许激进融合。
- **In-flight batching**：相当于 continuous batching 的 TensorRT-LLM 版本，带有 NVIDIA 调优的 schedulers。
- **Tight FP8 integration**：TransformerEngine FP8 是一等公民。在 H100 上用 FP8，TensorRT-LLM 通常是以 5-15 % 的优势领跑吞吐量。
- **Painful conversion**：每个新模型架构都需要自定义 Python builder 脚本。新模型支持比 vLLM 滞后数周到数月。

如果你在 NVIDIA 硬件上高规模服务固定的一组知名模型，且有工程带宽，选 TensorRT-LLM。否则，vLLM 的运维 simplicity 几乎总是胜出。
## 2026 年的服务框架选型

主要就三个选择：

- **vLLM** —— 事实上的开放标准。社区最活跃，支持新模型和新特性最快。Paged attention、continuous batching、speculative decoding、prefix caching 全部开箱即用。默认配置就很香。*除非你有特殊理由，否则选它准没错。*
- **SGLang** —— 新秀（2024 年），在 **结构化生成**（受限 JSON、regex）、**前端缓存**（用于 prompt 树共享的 RadixAttention）以及高分支 agent 场景下表现更好。对于共享 prefix 的负载，TTFT 更低。
- **TensorRT-LLM** —— NVIDIA 亲儿子。在它支持的模型上，H100 的裸吞吐量最高（编译内核、 fused FlashAttention 路径），但模型转换过程挺折磨人，支持新模型的速度也慢。*如果你有 NVIDIA 支持且需要榨干每一个 token/s，选这个。*

vLLM 0.6+ 和 SGLang 都支持大部分特性：speculative decoding、FP8、AWQ/GPTQ、MoE、multi-LoRA、function-call 受限解码。差距正在缩小。默认选 vLLM，如果你的负载是结构化输出或重度 agent 场景，再换 SGLang。

下面是我在 2025 年末测的纯吞吐量数据，单卡 H100 跑 FP8 精度的 Qwen3-32B：

| Engine | Throughput (tok/s) | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|
| vLLM 0.6 | 7400 | 95 | 13 |
| SGLang 0.4 | 7800 | 72 | 14 |
| TensorRT-LLM | 8900 | 88 | 11 |

对大多数业务来说，这点差别纯属噪声。选哪个主要看开发体验。

## 服务端的并行模式

跑一个 BF16 精度的 70B 全量模型，你需要 >140 GB 显存。单卡 H100（80 GB）塞不下。选项分支为三个正交维度——tensor parallelism (TP)、pipeline parallelism (PP) 和 sequence parallelism (SP)——它们的组合方式与训练场景（第 4 章）不同。

### Tensor parallelism (TP)

把权重矩阵切分 across GPUs。经典的 Megatron-LM [Shoeybi et al., 2019][shoeybi-megatron] 分解法：QKV 投影做列并行（每张卡拿一部分 attention heads），attention 输出投影做行并行（之后做 all-reduce），然后两个 FFN matmul 分别做列并行和行并行。每个 transformer 层两次 all-reduce。

对于服务来说，节点内 TP（NVLink，双向 ~600-900 GB/s）没问题；跨节点 TP（InfiniBand，每链路 ~50 GB/s）会给 all-reduce 带来无法接受的延迟。通常上限是 TP ≤ 8（单节点内）。

- TP=2 张 H100（同节点，NVLink）。吞吐量大概是单卡的 1.7-1.9 倍（不是 2 倍——TP 引入了 all-reduce 通信）。
- 如果需要长 context，用 TP=4（KV cache 也会随 TP 切分）。
- INT4 量化，单卡 H100。吞吐量不如 TP=2，但胜在便宜。

`vllm serve Qwen/Qwen3-72B --tensor-parallel-size 2` 这一条命令就能搞定。

### Pipeline parallelism (PP)

把层切分 across GPUs。GPU 0 持第 0-19 层，GPU 1 持第 20-39 层，以此类推。激活值在流水线中流动。经典问题是 **pipeline 气泡**：流水线末端的 GPU 空闲等待起点，反之亦然。对于服务，PP 会引入与流水线深度 × 每层时间成正比的 TTFT 延迟（GPipe / Megatron-LM 分析）；现代引擎通过 **微批次**（micro-batching，把 batch 切分成更小的微批次背靠背流过流水线）来摊销这个开销。

PP 是跨节点并行的首选方案。节点内 TP，跨节点 PP 是 200B+ 部署的标准配置。大多数团队对延迟敏感的业务会卡在单节点内——PP 气泡是真实存在的。

### Sequence parallelism (SP)

序列特别长时，激活值本身塞不进单卡。Sequence parallelism 沿 token 维度切分，attention all-reduce 使用 ring 或 all-to-all 通信。对于推理，SP 出现在 512K+ context 场景，此时即使做了 TP，每层激活缓冲也超出 GPU 内存。大多数生产服务用不到它。

### 数学：什么时候哪种并行模式划算？

粗略来说，TP 每层成本是 $2 \cdot \text{params}/\text{TP} / \text{HBM-BW} + 2 \cdot \text{batch} \cdot \text{seq} \cdot \text{hidden} / \text{NVLink-BW}$。第一项随 TP 线性减小；第二项增大。有个甜蜜点，通常在 H100 上是 TP=2 到 TP=4。超过 TP=8，all-reduce 占主导，你就亏了。对于 70B 模型，TP=2 吞吐量约 ~1.85 倍；TP=4 约 ~3.2 倍；TP=8 约 ~5.5 倍。边际递减效应是真实的。

PP 延迟成本对于单 batch 查询大约是 $\text{depth} \cdot \text{time-per-layer}$。有了微批次和稳态队列，吞吐量接近 $\text{batch} / (\text{time-per-stage})$。PP 对吞吐量友好，对延迟不友好。

## 我自己会怎么部署

7B 级别模型：单卡 L40S 或 4090，FP8，vLLM，16K context，80% 利用率下每百万 token 服务成本 $0.10-$0.15。

32B 级别模型：单卡 H100 跑 AWQ INT4 *或者* 双卡 L40S 跑 FP8。两者都行。H100 单 token 更快；L40S 每小时更便宜。看 $/Mtok 决定。

70B 级别模型：双卡 H100 跑 FP8。INT4 能省 30% 成本但损失 ~1% 质量，生产环境往往接受不了。

200B+ MoE：这是个独立的优化问题（第 12 章）。

## 可观测性：到底该监控什么

生产环境中，重要的指标不是 "tokens/sec"——那是单 batch 数据。真正的指标是负载下的分位数延迟。最小化看板配置：

- **TTFT p50 / p95 / p99** (ms)：从收到请求到发出第一个 token 的时间。p99 是用户感受到的，p50 是你拿来吹牛的。
- **ITL p50 / p95 / p99** (ms)：稳态下的 token 间延迟。这里出现尖峰意味着发生了抢占或内存压力。
- **Throughput (tok/s)** 在不同并发级别：1, 8, 32, 128, 512。画成曲线；拐点就是你 saturating FLOPs 或内存带宽的地方。
- **Queue depth** 和 **queue wait time**：请求是不是因为引擎跟不上而堆积了？
- **KV cache utilization (%)**：是不是快要开始抢占了？
- **Prefix cache hit rate (%)**：命中率低说明你的 prompt 工程团队生成的 prompt 无法共享。
- **GPU SM utilization** 和 **HBM bandwidth utilization**：来自 `dcgm-exporter` 或 `nvidia-smi dmon`。如果两者都 <70%，你还有余量；如果 HBM 到了 95% 而 SM 只有 40%，你就是内存瓶颈，量化是解决方案。

明确设定 SLO。典型的生产目标：TTFT p95 < 500 ms，ITL p95 < 50 ms，错误率 < 0.1%。超过这个值就是事故。

## 成本算账

关键数字是 **每百万输出 token 的成本** ($/Mtok-out)。对于自托管 vLLM 服务单卡 H100 跑 FP8 精度的 Qwen3-32B（按需 $2.50/hr，1 年预留 $1.20/hr，3 年预留或节省计划 $0.80/hr）：

- 并发 32 时的吞吐量：~7400 tok/s
- 预留每小时成本：$1.20/hr = $0.00033/sec
- $/Mtok-out = $0.00033 / 7400 \cdot 10^6 ≈ $0.045

对比 2026 年中的 API 定价：
- Claude-4.5-Sonnet: $15/Mtok output
- GPT-5: $12/Mtok output
- DeepSeek-V3.2: $1.10/Mtok output
- Qwen3-Max API: $0.80/Mtok output

自托管开源权重与前沿 API 之间 30-300 倍的差价，就是团队选择自托管的全部理由。坑在于：月用量 <1 亿 token 时，工程成本压倒节省的费用。月用量 >10 亿 token 时，节省占主导，自托管显而易见。

有个容易被忽视的细节：API 上 input token 通常比 output token 便宜 5-10 倍（因为它们 batching 更好且不需要 autoregress）。自托管时，它们确实更便宜但没那么夸张（3-5 倍），因为两个阶段都跑在你的硬件上。假设两者 parity 的成本模型会在两个方向上误导你。

## 总结与下一章

推理分两个不对称阶段（prefill, decode），现代服务栈——paged attention + continuous batching + speculation + quantization + FP8 硬件——存在的目的就是让这两个阶段最糟糕的部分变得可容忍。vLLM 是正确的默认选择。量化（INT8 永远上，预算紧用 INT4，H100+ 用 FP8）基本等于免费。Spec decoding 对于低 batch 低延迟服务是 2-3 倍的增益，对于高吞吐量则是噪声。2026 年的成本算账强烈支持大规模自托管开源权重。

下一章：**长 context**。RoPE scaling、YaRN、NTK-aware interpolation、ALiBi、attention sinks，以及为什么大多数 "1M context" 宣称在实际检索任务中会崩盘。

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