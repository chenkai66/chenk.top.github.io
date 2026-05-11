---
title: "大模型工程（三）：预训练的规模之道"
date: 2026-03-29 09:00:00
tags:
  - LLM
  - pretraining
  - fsdp
  - zero
  - data-mixing
  - scaling-laws
categories: 大模型工程
series: llm-engineering
series_order: 3
series_title: "大模型工程"
lang: zh
mathjax: true
disableNunjucks: true
description: "数据混合、去重、benchmark 污染、μP，FSDP / ZeRO-3 / Pipeline 并行，实战意义上的 200B token 悬崖，以及 1000 卡以上才会出现的失败模式。"
translationKey: "llm-engineering-3"
---
预训练是大模型能力的源头，也是榜单成绩和实际表现差距最大的地方。大多数公开的训练记录与其说是科学成果，不如说是工程奇迹。这一章聊聊当你不是 OpenAI 时，预训练必须搞对的几个部分：数据、并行策略，以及只有集群大到一定程度才会暴露的故障模式——比如一次失败的 NCCL all-reduce 就可能导致整个持续 30 天的训练任务中断。

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/illustration_1.png)

## 数据配比比架构更重要

![fig3: data mixture composition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig3_data_mixture.png)

过去三年里所有靠谱的 scaling study 都达成共识：在算力相同的情况下，两个 LLaMA 式架构之间的差异很小 (~5 % perplexity)，但不同数据配比带来的性能差异极为显著（>30%）。Chinchilla 论文的 compute-optimal 缩放定律假设了数据分布固定；一旦允许这个分布变化，数据就占据了主导地位。

现代预训练配比大致如下（FineWeb-Edu [Penedo et al., 2024], RedPajama-V2 [Together AI, 2024], Dolma [Soldaini et al., 2024] — 所有开放配比彼此之间相差无几）：

| 来源 | 占比 | 备注 |
|---|---|---|
| 过滤后的网页 (CommonCrawl) | 50-65 % | 体量主导 |
| 代码 (GitHub, StackExchange) | 8-15 % | 提升推理能力，不只是代码 |
| 书籍 | 5-10 % | 长上下文连贯性 |
| Wikipedia | 2-5 % | 单位 token 的效用显著更高 |
| 学术 / arXiv | 3-5 % | 数学、引用 |
| 数学 (证明、教科书) | 2-4 % | 专为推理优化 |
| 多语言网页 | 5-15 % | 质量因语言差异巨大 |

有个最重要的数字没人提：**去重率**。CommonCrawl 2024 在文档级别去重后保留约 25 % 的原始字节。在行级别去重（更激进），约 12 %。Lee et al. (2022) 的论文表明，激进去重即便移除了 75% 的数据，仍可降低困惑度（perplexity）。重复对语言模型是毒药——它教会模型记忆而不是泛化。

DeepSeek 的预训练说明（DeepSeek-V3 technical report [DeepSeek-AI, 2024], Dec 2024）是我见过对配比最诚实的：14.8T tokens，87 % code+math+web，13 % "high-quality books and synthetic data"。合成数据的占比比大家承认的要高。

## DataComp-LM：数据质量胜过数量，且有据可查

关于数据质量最严谨的公开研究是 DataComp for Language Models 基准 [Li et al., 2024]。他们固定了架构和训练算力，然后跑了 416 次对照实验，只变数据配比。值得记住的发现：

- **质量过滤器主导。** 一个基于模型的质量分类器（FastText 训练用于预测文本是否类似"高质量"参考数据），在与基线相同的算力下训练，在 MMLU 上超越基线 6.6 个百分点，在 Core（一个 22 任务基准）上超越 3.5。
- **激进过滤胜出。** 只保留质量分类器得分前 10 % 的文档，表现优于保留前 50 %，即使它丢弃了 5× 更多的 tokens。
- **去重与过滤相互作用。** 0.7 Jaccard 阈值的 MinHash-LSH 去重带来 +2.1 MMLU；结合质量过滤带来 +8.7。这两种干预是超加性的——去重移除了质量过滤器可能漏掉的低质内容的近重复项。
- **最优配比取决于任务。** 为代码密集型下游用途训练的模型，偏好的配比与通用聊天模型不同。不存在通用的最优解。

在 DCLM-Baseline 数据（3T tokens）上训练的 7B 模型，尽管每参数算力少了 33 %，但在 MMLU 上比 Llama 2 7B（使用了 2T 未指定、过滤较少的数据）高出 11.5 分。这体现了数据质量所能带来的性能提升幅度。

## 合成数据：不可告人的秘密

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/illustration_2.png)

直到 ~2023 年，普遍的观点还是"合成数据是污染，绝不能用"。当 Phi-1 [Gunasekar et al., 2023] (Microsoft, 2023) 展示了一个 1.3B 模型如果完全在 GPT-4 生成的合成教科书式数据上训练，可以匹配大得多的代码模型时，这一观点开始转变。Phi 系列继续沿用这个路线；其他团队则悄然跟进。

2026 年，顶级开源模型都在用合成数据：

- **Qwen3 technical report**：合成指令数据，合成代码，"diverse"合成问答。
- **Llama 3.1** [Dubey et al., 2024]：用于代码、数学、多语言的合成数据。
- **DeepSeek-V3**：用于"high-quality books"的合成数据（比爬取法律风险更低）。

主要风险是模式坍缩（mode collapse）：仅依赖单一教师模型生成合成数据，会导致输出分布过度集中。防御措施是多教师合成和激进过滤。另一个风险是评估基准的 **contamination**。Phi-3 曾被发现训练集里包含 MMLU 问题。标准对策是精确匹配去污染（丢弃任何与评估集有 13-gram 重叠的数据）加上持留的新颖基准。

[Shumailov et al., 2024] 指出的一个更隐蔽的风险是 **model collapse**：在从前代模型采样的数据上训练会逐渐 narrowing 数据分布，最终降低质量。他们的实验表明，递归地在模型输出上训练——即使是高质量的——也会在 5-10 代内收敛到退化分布。生产环境中的标准缓解措施是始终将合成数据与大量人工撰写的数据混合（通常 50-70 % 人类，30-50 % 合成），并使用来自许多独立教师的合成数据，而不是单一强模型。

经济账也很重要。使用 GPT-4o 生成 10 亿（1B）高质量合成 token 的成本约为 3 万美元。用较小的、针对合成微调的模型（例如自托管集群上的 Qwen3-32B-Instruct）生成，成本降至几百美元。2026 年大多数生产环境的合成数据管道采用混合模式：用强 frontier 模型（Claude, GPT-4o）生成"seed"多样数据，然后用自托管中型模型进行规模放大。

## 缩放定律：从 Chinchilla 到 200B

Chinchilla 定律 [Hoffmann et al., 2022] 指出 compute-optimal 训练大约使用 **每参数 20 tokens**。7B 模型想要 140B tokens；70B 模型想要 1.4T。

早期 [Kaplan et al., 2020] 的缩放定律给出了不同的比例 (~每参数 1.7 tokens)，因为他们通过改变模型大小多于数据大小来变动算力。Hoffmann et al. 通过联合扫描数据和模型大小修正了 Kaplan 的方法论。Chinchilla 定律成为 2022-2023 年 frontier runs 的默认规划启发式。

实践中 Chinchilla 是错的。或者更准确地说，它对 *training-compute-optimal* 是对的，但对于 **inference-compute-optimal** —— 模型被使用数十亿次的场景 —— 你应该 drastically 过度训练小模型。[Sardana et al., 2023] 将其形式化为 "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws." LLaMA-3 8B 在 15T tokens 上训练 (~每参数 1900 tokens)，是 Chinchilla 的 95×。该模型比 Chinchilla-optimal 8B 好得多，且服务成本远低于 Chinchilla-optimal 70B。

悬崖在于：大多数开放数据源去重后最多约 200B unique tokens。超过这个数，你要么在同一数据上重训（epoching），要么生成合成数据。多 epoch 训练在高质量数据上最多有效到约 4 个 epoch，之后模型开始过拟合 [Muennighoff et al., 2023]。在那之后，合成数据是唯一的前进方向——这就是为什么 2024 年后的 frontier 模型都在生成自己的数据。

关于自然语言数据实际 ceiling 的最佳公开估计来自 [Villalobos et al., 2024]：公共互联网上大约存在 300T tokens 的"high-quality"英文文本。Frontier labs 已经对此进行了激进的处理和过滤。超过 ~50T 高质量 tokens（Llama 3 和 DeepSeek-V3 所在的位置），合成数据和私有数据源成为进一步缩放数据的唯一途径。这是"缩放预训练"作为策略在 2026 年开始 plateau 的原因之一——我们正接近数据墙。

## 课程学习：从低质量向高质量退火

2024-2025 的一个稳健发现：如果你不是均匀 shuffle 所有数据，而是结构化呈现顺序，训练效率更高。DeepSeek-V3 和最近几次运行使用的基本模式：

1. **Bulk phase**（前 ~80 % tokens）：广泛混合，web-heavy，质量门槛较低。
2. **Anneal phase**（后 ~20 % tokens）：高质量配比——书籍、论文、数学、 curated 代码、指令微调数据。
3. 学习率在两个阶段都按 cosine schedule 衰减。

直觉是：早期训练建立广泛的语言能力，晚期训练将模型塑造向期望的分布。在低 LR 下对高质量数据退火，比在整个过程中均匀混合高质量数据，在提升下游基准方面 sample-efficient 得多。

Llama 3 [Dubey et al., 2024] 报告了最后单独的一次"annealing run"，他们在 1/10 峰值学习率下继续在 heavily filtered、高质量的 40B-token 配比上训练。尽管这只是总训练 tokens 的很小一部分，但这一阶段贡献了几个 MMLU 点数。

## 长上下文预训练：分阶段序列长度方案

从头训练一个 128K 上下文的模型是浪费的——大多数早期预训练受益于短序列数据，而长序列 attention 的 FLOPs 成本是二次方的。现代长上下文模型分阶段设置序列长度：

- **Stage 1**（0-90 % tokens）：4K 上下文。建立通用能力。
- **Stage 2**（90-95 %）：32K 上下文。引入长程模式。
- **Stage 3**（95-100 %）：128K 上下文，使用 curated 长文档数据（书籍、代码库、长论文）。

Llama 3 使用了达到 128K 的 4 阶段课程。NVIDIA 的 Nemotron-340B 使用了类似的分阶段方法。Qwen3 使用分阶段预训练达到 128K，最后通过 YaRN 扩展至 256K。

关键洞察是，模型的通用语言建模能力是在短上下文建立的，长上下文阶段只需要教会模型如何使用 extended attention。这比全程在长序列上训练便宜得多。Sequence packing——将多个短文档拼接成单个长序列并带上 attention masks——在短序列阶段几乎免费地提供了算力利用率。
## μP: 让你能在小规模上调参的参数化方法

![fig4: μP scaling across widths](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig4_mup_scaling.png)

大模型训练有个头疼的事：1B 模型调好的超参，放到 70B 上根本没法用。尤其是学习率。标准参数化之所以失效，是因为激活值的量级会随着宽度变化。

**μP（Maximal Update Parameterization）**，出自 [Yang et al., 2022]，就是来解决这个问题的。在 μP 下，你按层宽缩放初始化方差和学习率，让不同宽度模型的激活值和梯度保持同一量级。效果很简单：在 1 亿参数模型上调好学习率，直接用到 1000 亿参数模型上。

```python
# Standard: per-layer LR doesn't depend on width
# μP: scale LR for hidden Linear layers by 1 / fan_in_ratio
def mup_lr(base_lr, fan_in, base_fan_in=256):
    return base_lr * (base_fan_in / fan_in)
```

Cerebras 和微软研究院都做过实验，μP 调优后的 7B、70B 到 700B 模型，在相同训练比例下 loss 曲线偏差不到 1%。要是没用 μP，每个尺度都得烧几百万美元去摸索学习率。

完整的 μP 方案会在每层宽度上差异化处理四件事：
1. 初始化标准差：$\sigma \propto 1/\sqrt{\text{fan}_{\text{in}}}$
2. 隐藏层学习率：$\eta \propto 1/\text{fan}_{\text{in}}$
3. 输出层乘数：$\text{logits} = (1/d) \cdot W_{\text{out}} \cdot h$ 而不是 $W_{\text{out}} \cdot h$
4. Embedding 层学习率：保持不变（不缩放）

到了 2026 年，绝大多数生产环境至少会用 μP 来做学习率迁移。有些团队更进一步，用 μTransfer 处理 batch size 和权重衰减。[Wortsman et al., 2024] 还发现 μP 能让训练对噪声更鲁棒，从不稳定状态恢复也更容易。

## 并行策略：FSDP, ZeRO, Pipeline, Tensor

![fig1: ZeRO/FSDP memory stages](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig1_zero_stages.png)

模型一旦超过单卡显存，你就得面对四个正交的并行维度：

1. **数据并行（DP）** — 每张 GPU 复制模型，切分 batch。
2. **分片数据并行（FSDP / ZeRO-3）** [Rajbhandari et al., 2020] — 跨 DP rank 分片参数，每层 forward 前即时 gather。
3. **张量并行（TP）** [Shoeybi et al., 2019] — 跨 GPU 切分每个 matmul（Megatron 风格）。
4. **流水线并行（PP）** [Huang et al., 2019] — 跨 GPU 切分层，micro-batch 穿过它们。

对于 64 张 H100 上的 70B 模型，典型配置是：

```
TP=8 within node (NVLink)
PP=2 across nodes (200 Gbps Infiniband)
DP=4 with FSDP across the remaining axis
```

这样总共凑齐 64 张卡（8 × 2 × 4）。通信最密集的 TP 被限制在节点内高速 NVLink 上，PP 的 bubble 则能和下一个 micro-batch 重叠。2026 年 ZeRO-3（FSDP 的一种变体）是默认选项，因为它能让你不用改代码就跑起 70B 模型：

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

`mixed_precision` 这块至关重要。**务必用 fp32 做 reduce**——当跨 DP rank 累加成千上万个微小梯度时，bf16 会静默下溢。我们排查过 loss 发散的问题，最后发现就是因为漏了一个 `reduce_dtype=fp32`。

到了巨型模型（比如 DeepSeek-V3 671B, Qwen3-Max），光靠 FSDP 就不够了，得上定制代码搞 4D 并行。Megatron-LM、NVIDIA NeMo、ColossalAI 和 DeepSpeed 是四大生产框架。大多数实验室都是基于其中某一个魔改出来的。

## 实战案例：64 张 H100 上跑 70B 的并行选择

![fig2: pipeline parallelism schedule](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig2_pipeline_parallelism.png)

下面算笔账，为什么这个配置下 TP=8, PP=2, DP=4 是最优解。每张 H100 有 80 GB HBM，总共 5120 GB。

**单卡显存预算**。每个 rank 需要：
- 权重：70B × 2 bytes (BF16) / TP / PP = 70e9 × 2 / 16 = 8.75 GB
- 梯度：同权重 = 8.75 GB
- Adam 优化器状态：4 bytes/param × 2 (m, v) / TP / PP / DP_FSDP = 70e9 × 8 / 64 = 8.75 GB (如果 FSDP 分片优化器)
- 激活值（带梯度检查点）：随 batch × seqlen × model dim 缩放，70B 在 seqlen 8K 时通常 30-50 GB
- KV/scratch 缓冲：~5 GB

总共：~62 GB / 80 GB，剩下 ~18 GB 余量给 FlashAttention scratch、NCCL  buffers、峰值激活值。很紧，但能跑。

**为什么不用纯 FSDP=64？** 纯 FSDP 在 64 张卡上意味着每层 forward 前都要 all-gather 70B 参数。70B × 2 bytes 就是 140 GB 数据，分摊到 64 个 rank 是 2.2 GB/rank/gather，按层计算。80 层下来，每个 forward pass 每个 rank 要跑 ~175 GB 的 gather 流量。哪怕 NVLink 有 900 GB/s，这也得耗费 ~200 ms 纯通信时间——跟计算时间差不多了，吞吐量直接报废。

**为什么不用纯 TP=64？** TP 每次 matmul 后都要通信（行并行层上的 all-reduce）。每个 token 的 all-reduce 体积是 $O(d_{\text{model}} \times \text{TP})$。如果在节点间链路（200 Gbps Infiniband）上跑 TP=64，通信会彻底主导耗时。TP 只有在单节点 NVLink 内部（目前最多 8 卡）才能跑得好。

**为什么是 TP=8, PP=2, DP=4？** 节点内 TP=8 利用高速 NVLink 处理密集的 all-reduce 流量。PP=2 把模型切到两个节点，PP 通信相对稀疏（每个 micro-batch 一次）。DP=4 配合 FSDP 把优化器状态切分到 4 个 DP rank 上，每个持有 1/4 权重 × 1/(TP×PP) 层。PP 的 pipeline bubble（占 step 时间的 10-20%）是可以接受的；其他并行配置开销更大。

这只是其中一种配置。不同模型规模和集群拓扑有不同的最优解。通用原则是：最小化跨节点通信，把 TP 塞进最快的互联网络，用 PP 处理下一层粒度，剩下的并行度用 FSDP/DP 填满。

## 千卡以上才会暴露的故障模式

有些事在 8 卡上无关紧要，但在 8000 卡上能直接搞崩训练：

**静默数据损坏。** 单张卡算错（宇宙射线、电压 glitch）会产生 NaN 并被 all-reduce。防御方法是定期校验梯度 checksum——如果 rank 0 的梯度范数和 rank 100 相差超过 $10^{-3}$，立即 halt 并重新分片。

**Loss 尖峰。** 坏批次（比如 Python 解析器 dump、HTML 乱码）会导致梯度尖峰，偏移优化器动量且无法恢复。防御：激进的 grad clipping（默认 1.0，超大模型有时 0.3），加上当 grad norm > 5 倍运行中位数时“跳过该批次”的触发器。Llama 3 论文提到“检查点回滚”是标准流程：如果 50 步内 loss 尖峰无法恢复，回滚到上一个好检查点，学习率降低 30%，然后继续。

**NCCL 挂死。** 单节点网卡固件 bug 会导致该 rank 的 all-reduce 永远不返回。其他 999 个 rank 永久阻塞。防御：设置 `NCCL_TIMEOUT_S=300` 以及一个负责重新调度的 watchdog。

**检查点损坏。** 一个 2.7 TB 的 FSDP 检查点（Qwen3-32B 优化器状态）写入需要 20 多分钟。如果节点在写入中途 OOM，你不仅丢了当前 step，连上一个好检查点也覆盖了。务必先写到 `step_xxx.tmp/`，fsync，然后再 `mv`。

**硬件异构性。** 即使是“同型号”H100，批次间时钟频率也有 2-3% 差异。慢卡会拖后腿。防御：预热时按单卡吞吐量排序，把慢卡放在 TP=1 的 rank 上。

**Embedding 坍塌。** 如果早期 epoch 中 embedding 矩阵的梯度流被 clipping 或 NaN 污染，embeddings 可能坍塌成近乎相同的向量。症状：训练 loss 卡在 log(V)， perplexity 卡在 V。诊断：监控 embedding 矩阵的奇异值；如果最大奇异值比第二大 dominates 10 倍，说明坍塌了。修复：回滚到坍塌前，embedding 学习率翻倍，重启。

**梯度范数发散。** 这和 loss 尖峰不同。梯度范数在数千步内缓慢上升，尽管 loss 稳定。这通常预示着未来的坍塌。不要只监控全局，要监控每个参数组的 `||g||_2`。特定层类型（通常是 LM head 或最终 transformer 层）往往是罪魁祸首。

Meta 的 OPT-175B 日志 [Zhang et al., 2022] 仍然是关于实际大模型训练最好的公开文档。他们在 33 天的运行中，平均每 1.7 天就会发生一次节点故障。

## 训练期间实际发生了什么

![fig5: training loss curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig5_loss_curve.png)

现代 70B 预训练运行（LLaMA-3 70B, 15T tokens, ~1900 张 H100 跑 60 天）大致消耗如下：

- **Tokens**: 15T 输入，batch size ~16M tokens
- **Steps**: 15T / 16M ≈ 940K
- **Wallclock per step**: ~5.5 s (带 sequence packing)
- **Total compute**: ~$2.5e25$ FLOPs ≈ 6000 H100-years

学习率调度是从 brief warmup 余弦衰降到峰值的 ~10%。70B 的峰值学习率大约在 1.5e-4（更大模型更低，如果你守纪律就用 μP 缩放）。权重衰减 0.1，β2=0.95，AdamW。

评估是每个检查点做一次：每 2000 步，在 5-10 个小基准上评估（Hellaswag, ARC-easy, OpenBookQA, GSM8K-100, HumanEval-50）。追踪 loss 很便宜；下游评估才是真正的信号。

## 生产现实：前沿实验室实际在做什么

论文描述架构；生产系统大半是在搞工程调度。前沿预训练团队都在做这三件事，但不会写进 model cards：

**不同规模的多路并行运行。** 70B 目标运行之前，会有 1B 和 7B 的“侦察”运行，数据量分别是 1/100 和 1/10，架构、优化器和数据混合相同。侦察运行验证配方，让你能调优 μP 无法迁移的超参（主要是调度细节）。侦察运行看起来不错，70B 才能高置信度启动。Llama 3 论文提到同时跑 405B 和 70B 以持续验证缩放行为。

**激进监控。** Loss 曲线、梯度范数、单个参数统计、注意力模式直方图、专家利用率（针对 MoE）、每层激活值量级、评估分数。监控仪表盘不是可选基础设施——这是问题累积前捕捉它们的唯一方法。Anthropic 在 model cards 中开源了部分监控（clipping ratios, activation stats）；OpenAI 在 GPT-4 论文中有描述。

**持续数据 refinement。** 训练数据不是静态的。团队持续运行新的质量分类器，移除新发现的污染，添加新的合成数据源。Llama 3 明确提到在运行中途用重新训练的质量分类器过一遍整个训练数据，并据此重新加权后续批次。“训练前固定数据”是学术界的模型；生产团队在迭代。

**实时 tokenization 对比预 tokenization。** 预 tokenization 15T tokens 需要几周时间和 100+ TB 存储。训练期间流式 tokenization 要求 tokenizer 跟上 data loader，这意味着高度优化的 Rust tokenizer。大多数生产设置预计算 token IDs，但从分布式文件系统流式传输，而不是加载原始文本。
## 常见坑点

我自己亲眼见过五个能把预训练跑废的大坑。

**1. 忘了处理 NCCL 集合通信的 bf16/fp32 不匹配问题。** NCCL 默认的 all-reduce 在 fp16 模式下，累加成千上万个小梯度值时会 silently underflow（静默下溢）。稳一点的做法是：设 `NCCL_PROTO=Simple` 保证稳定性，FSDP 里 `reduce_dtype=fp32`。Llama 3 论文里提过，他们就是在 50 万 step 左右遇到不稳定，加上这个才救回来的。

**2. 上下文长度对应的 RoPE base 设错了。** RoPE 默认基频 $\theta = 10000$。到了 32K 上下文，频率发生混叠（aliasing），模型根本学不到长程模式。解决办法：按 [bloc97 NTK-aware] 把 base 缩放到 $10000 \times (\text{seqlen}/2048)^{d/(d-2)}$，或者直接用 YaRN [Peng et al., 2024]。我们有次为了 32K 上下文性能下降查了一周，最后发现 RoPE base 还停留在 4K 预训练阶段的设置。

**3. 打包序列（Sequence packing）上的 Loss masking 没做好。** Sequence packing 是把多篇文档拼成一个训练序列。如果不把跨文档的 attention mask 掉，模型会学到虚假的跨文档关联，MMLU 质量直接掉 1-3 %。务必在文档边界重置 `position_ids` 和 `attention_mask`。

**4. 激活值检查点（Activation checkpointing）混用了非确定性算子。** 检查点重计算阶段要重跑 forward，这就要求 kernel 必须是确定性的。有些 FlashAttention 和 GeLU 实现里的数值归约是非确定性的。结果就是：重算时的梯度跟保存时的 forward 值对不上。这种细微的质量回退往往几千步之后才暴露。在检查点区域务必设 `torch.use_deterministic_algorithms(True)`，或者确认 kernel 是确定性的。

**5. 优化器状态没分片存盘。** 70B 模型的 AdamW 优化器状态是 70B × 8 bytes = 560 GB。几千个节点同时往单个共享文件系统写这个大小的文件，很容易死锁。用 FSDP 的分片检查点（每个 rank 写自己的分片）配合并行写入。我们有过一个检查点因为意外串行化，存了 4 个小时。

## 2024-2026 研究前沿

现在的 Llama 式“过滤 + 去重 + 训练”配方之后，接下来会是这些：

**大规模下的数据效率。** [Marion et al., 2023] (When Less is More) 表明，用精心 curated 的 200B-token 子集训练，效果能匹敌 1T-token 未过滤全集。这本质上是“数据的早停”——挑出信息量最大的样本，剩下的跳过。

**课程驱动的混合策略。** [Albalak et al., 2024] 展示了动态数据混合——根据模型提升情况调整各数据源比例——比固定混合高出 1-2 MMLU。Doremi [Xie et al., 2023] 提出用 min-max 优化自动找最优权重。

**超越 Chinchilla 的算力最优。** 混合专家模型（MoE）改变了缩放律。[DeepSeek-AI, 2024] 基于 V3 实验报告了他们自己的 MoE 缩放律：最优 active-to-total 比例、专家数量、专家大小都跟训练算力有非 trivial 的依赖关系。社区还在校准 MoE 特定的缩放律。

**持续预训练。** 大多数生产模型现在是更新，而不是从头重训。Llama 3.1 就是 Llama 3 + 额外数据持续预训练 + 后训练。持续预训练便宜得多，但需要仔细调度学习率以避免灾难性遗忘。

## 总结与下一步

预训练 70 % 是数据工程，30 % 是分布式系统工程。架构选择反而是三者里最次要的。把数据混合配比搞对，狠狠去重，为了推理成本训练超过 Chinchilla 最优值，用 μP 让超参数可迁移，根据硬件拓扑选 FSDP 加 TP 加 PP，然后为那些只在大规模下才出现的故障模式写防御性代码。

下一章：**后训练**。SFT、DPO、RLHF、RLAIF——它们到底优化了什么，奖励模型什么时候会失效（而且它们经常失效），LoRA 和全量微调之争，以及把基座模型变成客户能用产品的生产级配方。

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