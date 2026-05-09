---
title: "大模型工程（五）：推理优化"
date: 2026-04-30 09:00:00
tags:
  - llm
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
钱都花在推理上。一个 70B 的模型，服务 1000 个用户，每秒处理 50 个 token，3 个月就烧光训练时的 GPU 预算。

本章内容就盯住两个指标：首 token 时间（TTFT）和 token 延迟（ITL）。再加一个比例：每百万 token 耗多少 GPU 秒。

训练是一锤子买卖，成本分摊到百万次推理里。推理是日常开销，省不了。tokens-per-GPU-second 提升 0.5 倍，产品生命周期内每天都在赚。

每个正经 LLM 团队都有专人搞推理优化。开源社区五年出了四代推理引擎：FasterTransformer → DeepSpeed-Inference → vLLM → SGLang/TensorRT-LLM/llama.cpp。

![大模型工程（五）：推理优化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/illustration_1.jpg)
## 两个特性完全不同的阶段

![fig1: prefill vs decode compute pattern](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig1_prefill_vs_decode.png)

每次 LLM 推理都分两个阶段。

1. **Prefill（处理提示）**：输入 token 并行跑模型，填充 KV 缓存。计算密集型。70B 模型跑 4K token 提示，大约需要 280 TFLOP。一张 H100 显卡 70 毫秒就能打满。
2. **Decode（生成）**：每次生成一个 token，注意力操作依赖缓存的 K 和 V。内存带宽密集型。每步解码读取整个 KV 缓存（数 GB），只生成一个 token。

不对称性是关键。Prefill 能跨用户批量处理，用同一个内核跑不同序列。Decode 在朴素批处理下效果差，因为每个用户处于不同序列位置。主流推理引擎都围绕这种不对称性设计。

经验法则很简单。TTFT 主要由 prefill 决定，ITL 则看 decode 的内存带宽。想降低 TTFT，堆 FLOPs 就行，比如增加 SM 或用张量并行。想降低 ITL，提升内存带宽更有效，比如用 HBM3 替代 GDDR，或者通过量化减少参数。

算术强度让这一点更清楚。70B 模型跑 4K 提示的 prefill，模型权重（BF16 下 140 GB）加载一次，处理 4096 个 token。算术强度约 4096 FLOP/字节。同一模型单 token decode，权重加载一次，只处理 1 个 token。算术强度约 1 FLOP/字节。H100 在 BF16 下 989 TFLOPS，HBM 带宽 3.35 TB/s，ridge point 约 295 FLOP/字节。Prefill 的 4096 FLOP/字节远高于 ridge，属于计算受限。Decode 的 1 FLOP/字节低两个数量级，属于内存受限。这两个阶段需要不同硬件特性。服务栈如果不分开处理，性能会大打折扣。

下一节会具体讲如何优化这两个阶段的性能。
## KV cache：支撑长上下文的数据结构

![fig2: KV cache size growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig2_kv_cache_growth.png)

KV cache 存储每一层每个历史 token 的 K 和 V 向量。以 70B GQA-8 模型为例，上下文长度为 32K：

$$\text{KV} = 2 \cdot 80 \cdot 2 \cdot 8 \cdot 128 \cdot 32{,}768 \cdot 2 \text{ 字节} = 8.6 \text{ GB}$$

每次请求都要这么多内存。50 个并发请求就是 430 GB。这比模型权重大多了。瓶颈在 KV cache，不在权重。

最简单的实现是为每个请求分配一块连续张量，大小为 `max_context`。但这种方法有两个问题：

1. **内部碎片**。即使只用 1K token，也得预留 32K 内存。
2. **无法扩展**。超出 `max_context` 就会 OOM。

负载下还有第三个问题：**外部碎片**。请求来来去去，空闲内存散落在不连续区域，没有一块够大。服务器运行几小时后，仅外部碎片就占掉 20-40% 的可用 KV 内存。这个问题和操作系统在 1960 年代遇到的分页问题一样，解法也类似。
## Paged attention

![fig3: paged attention block table](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig3_paged_attention.png)

vLLM 的杀手级特性是 2023 年论文 [Kwon 等][kwon-vllm] 提出的 **paged attention**。KV 缓存按固定大小分配，每块通常是 16 个 token。每个请求用一个块表将逻辑位置映射到物理块。这和操作系统的虚拟内存类似。

```
请求 A：47 个 token 的 KV → 3 块（16+16+15）
请求 B：200 个 token 的 KV → 13 块
A 的块表：[0x47, 0x12, 0x3a]
B 的块表：[0x05, 0x09, 0x21, ...]
```

好处很明显：

- 几乎零浪费。每个请求只有最后一块可能有碎片，最多 15 个 token。
- 支持内存共享。前缀相同的请求（如系统提示、few-shot 示例）可以共用物理块。这就是 prompt 缓存的来源。
- 抢占方便。内存不足时，把请求的块换到 CPU；空闲时再拉回来。

vLLM 在 LLaMA-13B 上的吞吐量比朴素的 Hugging Face Transformers 循环快了 14 到 24 倍（基于原论文基准）。这个数字至今有效，但实际生产中的差距取决于负载。单流服务提升约 2 倍；高并发混合流量提升 5 到 10 倍；14 到 24 倍是对完全没有批处理的 HF 代码的对比结果。重点不是倍数，而是 paged attention 让 KV 内存从硬约束变成了可控资源。

最简用法如下：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-7B", gpu_memory_utilization=0.9,
          max_model_len=32768, enable_prefix_caching=True)
prompts = ["用一段话讲 MoE。",
           "用一段话讲 GQA。"]
out = llm.generate(prompts, SamplingParams(max_tokens=200, temperature=0.7))
```

`enable_prefix_caching=True` 自动启用重复 prompt 前缀的块共享。如果一个系统提示在 1000 个请求中复用，这些请求的 prefill 速度能提高 5 到 50 倍。

### Block manager：分页如何工作

Block manager 是 paged-attention 引擎的核心。它管理一个物理 KV 块池，每块大小通常是 16 个 token × 2(KV) × 层数 × 头数 × head_dim 字节。它维护空闲列表，并为新请求分配块。当请求需要扩展（下一个解码步骤），manager 弹出一个空闲块，将其地址追加到请求的块表，然后返回。

在 copy-on-write 语义下，前缀共享的块使用引用计数。fork 序列（如并行采样、beam search 或推测回滚）会增加引用计数；释放时减少引用计数；引用计数归零时，才真正释放物理内存。

"换出到 CPU" 路径在生产环境中很重要。GPU 没有空闲块但仍有新请求排队时，调度器会选择一个受害请求（通常是 FIFO 或运行时间最长的）。通过 PCIe 将其块拷贝到 pinned host memory，然后释放 GPU 块。内存可用时，再把请求拉回来。

换出开销确实存在。PCIe Gen5 ×16 速率为 64 GB/s，换出和重新加载一个 8 GB 请求需要 100 毫秒。但它让系统能在不拒绝请求的情况下满足延迟 SLO。
## Continuous batching

![fig4: continuous vs static batching](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig4_batching_timeline.png)

静态批处理有个大问题：它要等最慢的序列跑完，才开始下一批。输出长度不固定时，GPU 会浪费大量时间空转。

**连续批处理**（[Yu 等，Orca，OSDI 2022][yu-orca]，vLLM 推广）解决了这个问题。每次解码时，踢掉已完成的序列，加入等待的序列。只要任务队列不空，GPU 始终跑满 `max_batch_size` 条序列。

假设我有 1000 个提示，输出长度从 50 到 2000 不等：

- 静态批处理：总耗时接近 2000 token 的输出时间。
- 连续批处理：总耗时接近所有输出的平均值。

结合分页注意力机制，效果更明显。中途加入新序列只需分配新的 KV 块，不用重新分配整个张量。

### Orca 的迭代级调度

Orca 论文提出两个机制，到 2026 年依然流行。第一个是**迭代级调度**，把批处理边界和请求边界分开。每个 Transformer 迭代独立调度，一个请求在第 500 步完成，不会拖累需要 1500 步的请求。

第二个是**选择性批处理**，解决预填充和解码不匹配的问题。注意力算子按序列形状操作，线性层则一起批处理。现代引擎更进一步，比如 vLLM 和 SGLang 都支持“分块预填充”。一个请求的预填充可以和其他请求的解码交错进行，即使解码队列稀疏，GPU 利用率也能保持在 90% 以上。

调度决策很关键：什么时候接受新请求？简单点，直接接受。但 32K token 的预填充和解码一起跑，每步延迟会飙升。预填充的 FLOPS 是单步解码的约 100 倍。生产环境的做法是设一个“预算”上限，把长预填充切分成小块，和解码步骤交错执行。这个隐藏旋钮决定了负载下 TTFT p99 是 500 毫秒还是 5 秒。
## Speculative decoding

![大模型工程（五）：推理优化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/illustration_2.jpg)


![fig5: speculative decoding tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/05-inference/fig5_speculative_tree.png)

这个想法很聪明，现在已经是标配了。解码是内存瓶颈：每生成一个 token，都要读取整个模型权重。如果能一次验证 N 个 token 呢？

Speculative decoding（[Leviathan 等，2023][leviathan-spec]）用一个小而快的 **draft 模型**（比如 1B 参数）提议 $k$ 个 token。目标模型（比如 70B 参数的大模型）在一次前向传递中验证这些 token。它会看提议的前缀，并行计算 $k+1$ 个位置的概率。匹配的 token 留下，第一个不匹配的 token 和后续的全丢掉，然后重新采样。

数学上，假设 draft 模型正确率是 $\alpha$，解码是内存瓶颈（$k+1$ 个 token 的前向传递成本和单 token 差不多），那么加速比是 $(1 - \alpha^{k+1})/(1 - \alpha)$。当 $\alpha=0.7$、$k=4$ 时，加速比是 2.5 倍。

实际中，draft 模型通常从目标模型蒸馏而来。目前主流方法有三种：

- **Draft-target 双模型对**：小蒸馏模型提议，目标模型验证。质量高，但维护两个同步检查点很麻烦。
- **Medusa**（[Cai 等，2024][cai-medusa]）：在目标模型上加 4-5 个预测头。每个头从相同的 backbone 激活预测 $t+1, t+2, t+3, ...$。不需要独立 draft 模型。加速比 2-3 倍，质量无损；在基础模型上训练只需几天 GPU 时间。
- **EAGLE / EAGLE-2**（[Li 等，2024][li-eagle]）：基于倒数第二层隐藏状态和前一个 token 的自回归头。EAGLE-2 引入动态树验证，每步探索多个 draft 分支。大多数任务能加速 3-4 倍，是当前 SOTA 方法。

```python
# vLLM 的 speculative decoding
llm = LLM(model="Qwen/Qwen3-32B",
          speculative_model="Qwen/Qwen3-1.7B",
          num_speculative_tokens=5)
```

推测解码只在目标 GPU 是内存瓶颈时有用。如果是 FLOPs 瓶颈，就没啥帮助。大 batch 通常是 FLOPs 瓶颈（权重加载一次后分摊到所有序列），推测解码帮不上忙。单流低延迟服务是内存瓶颈，收益很大。

关于正确性有个细节：验证步骤对目标分布做拒绝采样。如果 draft 模型对 token $t$ 的概率是 $p_d(t)$，目标模型是 $p_t(t)$，接受概率是 $\min(1, p_t(t)/p_d(t))$。拒绝时，从修正分布 $\propto \max(0, p_t(t) - p_d(t))$ 重新采样。这保证输出分布完全等同目标模型，与 draft 质量无关。推测解码是无损的：draft 差只会降低接受率，不会影响输出质量。这也是为什么团队上线时不用测质量——因为质量不会退化。
## 量化：INT8、INT4、FP8

推理时，把 bf16 转成 INT8/INT4/FP8，能省内存和带宽。一般只量化权重，有时也量化激活。

### INT8 仅权重量化

最简单。按通道量化每个权重矩阵：$w_q = \text{round}(w / s)$，其中 $s$ 是每输出通道的缩放因子。推理时，实时反量化回 bf16 再做矩阵乘法。

内存：权重从 16 位降到 8 位，减半。质量：困惑度损失通常小于 0.5%，不用校准。纯赚。

### INT4：GPTQ vs AWQ

INT4 难得多。直接四舍五入会丢 2-5% 的困惑度。目前有两种靠谱算法：

**GPTQ**（[Frantar 等，2022][frantar-gptq]）：OBQ 的改进版，适合大模型。逐列量化权重矩阵，每列后用 Cholesky 分解的 Hessian 逆更新未量化的列，补偿误差。数学上就是每列做一次结构化最小二乘求解。计算成本：单 A100 处理 70B 参数要几小时，峰值内存约等于一层激活（典型值是 128 样本 × 2048 token 的小校准集）。

**AWQ**（[Lin 等，2023][lin-awq]）：发现 1% 的权重是“显著”的（穿过它们的激活值较大）。用每通道缩放因子 $s$ 保护这些权重——相当于量化前将输入激活乘以 $s$，权重除以 $s$。全精度下这是恒等变换，但改变了量化分布。然后均匀量化。比 GPTQ 快（不用解 Hessian，只需搜索缩放因子），多数模型质量略好，对 MoE 更友好。

**SmoothQuant**（[Xiao 等，2022][xiao-smoothquant]）：与 GPTQ 和 AWQ 正交，解决激活量化问题。难点是激活中有异常通道（少数通道幅值是其他通道的 100 倍），会破坏 INT8 激活量化。SmoothQuant 把激活幅值迁移到权重中，通过每通道缩放让两边都能量化到 INT8。作为 INT8 W8A8 量化前的预处理步骤。

对 LLaMA 类模型，GPTQ 和 AWQ 在 INT4 仅权重量化时，困惑度损失都小于 1%。MoE 中路由复杂化了校准——不同专家看到的激活分布差异很大，统一缩放因子保护不了少用的专家。当前最佳实践是每个专家单独用 AWQ。

```python
# vLLM 加载 AWQ 量化模型
llm = LLM(model="Qwen/Qwen3-32B-AWQ", quantization="awq",
          dtype="float16", max_model_len=32768)
```

### FP8（以及 H100/H200 硬件路径）

H100 及更新 GPU 支持 FP8 tensor core，吞吐量是 BF16 的两倍。FP8 推理是现代路径：权重存 FP8，激活在计算时转 FP8，累积用 FP32。质量损失可忽略（<0.1%），因为 FP8 动态范围比 INT8 大。

FP8 有两种格式：E4M3（4 位指数、3 位尾数）用于激活和权重——范围小、精度高；E5M2（5 位指数、2 位尾数）用于梯度和 KV 缓存——范围大、精度低。硬件把反量化缩放融合到矩阵乘法中，没有“INT4 反量化开销”。FP8 是 H100/H200 上的速度上限。

校准对 FP8 很重要，尽管困惑度损失很小。标准方法是在 128-512 个校准样本上收集激活统计，计算每张量或每 token 的缩放因子，存到检查点里。运行时从实际批次计算每 token 的激活缩放因子，能在小延迟开销下获得最佳质量。NVIDIA TransformerEngine 和 vLLM `--quantization fp8` 都实现了这一点。

FP8 需要硬件支持。INT4/INT8 在任何 GPU 上都能跑，FP8 只限 H100/H200/B200。如果用 A100 部署，只能选 INT8/INT4。

### KV 缓存量化

KV 缓存压缩是内存优化的另一半。压到 INT8（节省一半内存），或者 INT4（1-2% 质量损失，需仔细校准）。vLLM 和 SGLang 都支持 FP8 KV 缓存；INT4 KV 缓存还在研究阶段，FlashInfer 提供了相关内核。

预计 2026 年，FP8 会成为主力。它能把内存减半（同 GPU 上并发请求翻倍），质量损失小于 0.1%。每 token 缩放因子几乎免费，因为是在线计算的。INT4 KV 缓存需要更多注意——AWQ 用于权重的每通道显著性结构不直接适用于 KV（KV 是数据，不是参数）。当前最佳实践是每 token 每头缩放加异常值保留：保留约 1% 的通道在高精度，其余部分激进量化。
## SGLang 与 RadixAttention

vLLM 的前缀缓存会在前缀相同的请求间共享块。SGLang（[Zheng 等，2024][zheng-sglang]）更进一步，用 **RadixAttention** 扩展了这个思路。它把所有活跃的 KV 块组织成一棵 radix 树。树中每条从根到叶的路径表示一个 token 序列。新请求来了，先在树里找最长匹配前缀，共享这些块，只计算后缀。

为什么这比 vLLM 的精确前缀缓存更有用？因为 agent 负载会分支。比如，ReAct agent 可能发出 5 个工具调用子查询。每个子查询的系统提示、工具和对话历史都一样，但后缀不同。线性前缀缓存只能让子查询匹配共享根，无法让子查询之间共享状态。RadixAttention 却能做到。

SGLang 论文提到，在 agent 和结构化输出负载上，RadixAttention 的吞吐量是 vLLM 等价前缀缓存的 5 倍。简单聊天任务上也有约 1.5 倍提升。如果流量不共享前缀，优势会缩小。

实现方法很简单。维护一棵以 token ID 为键的 radix 树，节点保存 KV 块引用。新请求来了，尽量深地匹配输入前缀。匹配到的节点引用计数加一。未匹配的后缀分配新块。完成后减少引用计数。内存压力大时，用 LRU 算法驱逐子树。数据结构开销很小，远低于节省的资源。
## TensorRT-LLM 细节

NVIDIA 的 TensorRT-LLM 是第三大引擎。它的特点很鲜明。

- **编译内核**：每个模型会针对特定 GPU、批处理大小和序列长度编译成 TensorRT 引擎。编译耗时 10 到 30 分钟。生成的引擎在支持的工作负载上，比 vLLM 的运行时编译快 1.1 到 1.3 倍。
- **插件模型**：attention（FMHA、paged FMHA、FlashAttention-3 路径）、MoE 路由和量化用自定义内核实现，以插件形式提供。灵活性不如 vLLM，但融合更激进。
- **飞行批处理**：类似连续批处理，调度器经过 NVIDIA 优化。
- **FP8 深度集成**：TransformerEngine FP8 是首选。在 H100 上用 FP8 时，TensorRT-LLM 吞吐量通常领先 5% 到 15%。
- **转换麻烦**：每种新模型架构都需要一个自定义 Python 构建脚本。新模型支持比 vLLM 滞后几周到几个月。

如果我在 NVIDIA 硬件上大规模运行固定的一组成熟模型，并且有足够工程资源，我会选 TensorRT-LLM。否则，vLLM 的操作简便性几乎总是更优。
## 2026 年的服务框架选型

三个主要选项：

- **vLLM**——事实上的开源标准。社区最活跃，支持新模型和功能最快。Paged attention、连续批处理、推测解码、前缀缓存开箱即用。默认配置就够用。*除非有特殊原因，直接选它。*
- **SGLang**——2024 年推出，适合结构化生成（约束 JSON、正则表达式）、前端缓存（RadixAttention 用于 prompt 树共享）和高扇出 agent 场景。共享前缀负载的 TTFT 更低。
- **TensorRT-LLM**——NVIDIA 的框架。H100 上支持的模型吞吐量最高（编译内核、融合 FlashAttention 路径）。但转换麻烦，新增模型支持慢。*如果有 NVIDIA 支持，又想榨干每秒 token，就选它。*

vLLM 0.6+ 和 SGLang 都支持推测解码、FP8、AWQ/GPTQ、MoE、multi-LoRA、函数调用约束解码。差距越来越小。默认用 vLLM，工作负载涉及结构化输出或大量 agent 时，换成 SGLang。

Qwen3-32B 在 FP8 单 H100 上的吞吐数据（我的基准测试，2025 年底）：

| 引擎 | 吞吐（tok/s） | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|
| vLLM 0.6 | 7400 | 95 | 13 |
| SGLang 0.4 | 7800 | 72 | 14 |
| TensorRT-LLM | 8900 | 88 | 11 |

对大多数负载来说，这些差异可以忽略。按开发体验选就行。
## 服务并行模式

70B 模型用 BF16 精度，内存需求超过 140 GB。单张 H100（80 GB）搞不定。解决办法分三路：张量并行（TP）、流水线并行（PP）、序列并行（SP）。组合方式和训练时不一样（第 4 章）。

---

### 张量并行（TP）

把权重矩阵分到多张 GPU 上。经典方法是 Megatron-LM 分解（[Shoeybi 等，2019][shoeybi-megatron]）。QKV 投影按列切分，每张卡负责一部分注意力头。注意力输出投影按行切分，后面加 all-reduce。FFN 的两个矩阵乘法分别按列和按行切分。每个 transformer 层需要两次 all-reduce。

节点内 TP 没问题，NVLink 带宽 600-900 GB/s。跨节点 TP 就不行了，InfiniBand 每条链路只有 50 GB/s，all-reduce 延迟太高。通常 TP 不超过 8，限制在一节点内。

- 两张 H100（同一节点，NVLink 连接），吞吐量是单卡的 1.7 到 1.9 倍。不是 2 倍，因为有通信开销。
- 长上下文可以用 TP=4，KV 缓存也会切分。
- INT4 量化，单卡搞定。吞吐量比 TP=2 低，但成本更便宜。

`vllm serve Qwen/Qwen3-72B --tensor-parallel-size 2`，一行命令就行。

---

### 流水线并行（PP）

把模型层分到多张 GPU 上。比如 GPU 0 负责层 0-19，GPU 1 负责层 20-39。激活值在流水线里流动。经典问题是 **流水线气泡**：前端空闲等后端，或者反过来。推理任务中，PP 会引入 TTFT 延迟，延迟和流水线深度 × 每层时间成正比。现代引擎用 **微批处理**摊销延迟，把批次拆成小块连续流过流水线。

PP 是跨节点并行的主力。节点内用 TP，跨节点用 PP，这是 200B+ 模型的标准部署。延迟敏感的任务，多数团队只用一个节点——流水线气泡确实是个坑。

---

### 序列并行（SP）

超长序列时，激活值可能装不下。SP 沿 token 维度切分，用环形或全对全通信做注意力 all-reduce。推理任务中，上下文长度超过 512K，且每层激活缓冲区在 TP 后仍超出 GPU 内存时，才会用到 SP。大多数生产环境用不上。

---

### 数学：每种并行模式何时划算？

粗略估算，TP 每层成本为 $2 \cdot \text{params}/\text{TP} / \text{HBM-BW} + 2 \cdot \text{batch} \cdot \text{seq} \cdot \text{hidden} / \text{NVLink-BW}$。第一项随 TP 减少，第二项增加。H100 上 TP=2 到 TP=4 是甜点。超过 TP=8，all-reduce 占主导，性能下降。

70B 模型，TP=2 吞吐量约 1.85 倍，TP=4 约 3.2 倍，TP=8 约 5.5 倍。边际递减很明显。

PP 单批次查询延迟约为 $\text{深度} \cdot \text{每层时间}$。微批处理加稳态队列时，吞吐量接近 $\text{batch} / (\text{每阶段时间})$。PP 提升吞吐量，但延迟高。

下一节会聊具体踩过的坑和血泪经验。
## 我会真正部署什么

7B 级模型，单张 L40S 或 4090。用 FP8，跑 vLLM，上下文长度 16K。80% 利用率时，每百万 token 成本 0.10-0.15 美元。

32B 级模型，两种选择。单 H100 配 AWQ INT4，或者双 L40S（TP=2）用 FP8。H100 单 token 处理更快，L40S 每小时更便宜。按 $/Mtok 决定。

70B 级模型，双 H100（TP=2），FP8。INT4 能省 30% 成本，但质量损失约 1%，生产环境通常不划算。

200B+ MoE 是另一类优化问题（第 12 章）。
## 可观测性：实际要测什么

生产环境里，别盯着 "tok/s" 看。这只是单批次的数字，没太大意义。真正重要的是负载下的百分位延迟。

看板至少得包含这些指标：

- **TTFT p50 / p95 / p99**（ms）：从收到请求到吐出第一个 token 的时间。p99 是用户的真实感受，p50 是他们会吹的数字。
- **ITL p50 / p95 / p99**（ms）：稳态下 token 之间的延迟。如果这里出现尖峰，可能是抢占或者内存压力导致的。
- **吞吐量（tok/s）**：测试不同并发数下的表现，比如 1、8、32、128、512。画成曲线后，拐点就是 FLOPs 或内存带宽的瓶颈位置。
- **队列深度** 和 **队列等待时间**：请求堆积了？说明引擎处理不过来。
- **KV 缓存利用率（%）**：快接近上限了吗？接近的话可能要开始抢占了。
- **Prefix 缓存命中率（%）**：命中率低？说明你的 prompt 工程团队在生成不可复用的 prompt。
- **GPU SM 利用率** 和 **HBM 带宽利用率**：用 `dcgm-exporter` 或 `nvidia-smi dmon` 查看。如果两者都低于 70%，还有优化空间；如果 HBM 达到 95%，而 SM 只有 40%，那就是内存瓶颈，量化能解决问题。

SLO 得明确写清楚。我的经验是，生产环境的目标通常是：TTFT p95 < 500 ms、ITL p95 < 50 ms、错误率 < 0.1%。超过这些值，基本就是起火了，赶紧救。
## 成本算术

关键数字是 **每百万输出 token 的成本**（$/Mtok-out）。以自托管 vLLM 为例，用单块 H100 跑 Qwen3-32B FP8。H100 的价格如下：按需 $2.50/小时，1 年预留 $1.20/小时，3 年预留或 savings plan $0.80/小时。

并发 32 时，吞吐量约 7400 tok/s。  
预留成本是 $1.20/小时，换算成秒就是 $0.00033/秒。  
计算一下，$/Mtok-out = $0.00033 / 7400 \cdot 10^6 ≈ $0.045。

再看看 2026 年中期的 API 定价：  
Claude-4.5-Sonnet：$15/Mtok 输出。  
GPT-5：$12/Mtok 输出。  
DeepSeek-V3.2：$1.10/Mtok 输出。  
Qwen3-Max API：$0.80/Mtok 输出。

自托管开源权重和前沿 API 的成本差距在 30 到 300 倍之间。这就是团队选择自托管的核心原因。每月处理少于 1 亿 token 时，工程成本会吃掉大部分节省。但每月超过 10 亿 token 时，节省就非常明显了，自托管几乎是唯一选择。

还有一个坑容易被忽略：API 上输入 token 比输出 token 便宜 5 到 10 倍，因为输入可以批量处理且不需要自回归。自托管时，输入 token 仍然更便宜，但差距缩小到 3 到 5 倍，毕竟两个阶段都跑在你的硬件上。如果假设输入和输出 token 价格相同，成本模型就会误导你。
## 小结与下一篇

推理分两个不对称阶段：prefill 和 decode。现代服务栈用 paged attention、continuous batching、speculation、量化和 FP8 硬件，目标是让这两个阶段的最差表现也能接受。vLLM 是首选，默认没错。量化基本无成本，INT8 通用，预算紧选 INT4，H100+ 用 FP8。Spec decoding 在低 batch、低延迟场景下能提升 2-3 倍性能，但高吞吐量时效果不明显。到 2026 年，大规模自托管开源权重的成本优势会非常明显。

下一篇聊**长上下文**。RoPE 缩放、YaRN、NTK-aware 插值、ALiBi、attention sinks 都会提到。还会分析为什么大多数 "1M 上下文" 的说法在实际检索任务中站不住脚。
## 参考文献

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
