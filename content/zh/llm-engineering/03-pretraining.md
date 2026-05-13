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
预训练是大模型能力的源头，也是榜单成绩与实际表现差距最大的地方。大多数公开的训练记录更像是工程奇迹，而非科学成果。本章将聚焦于当你不是 OpenAI 时，预训练中真正必须做对的关键环节：数据、并行策略，以及那些只有在集群规模足够大时才会暴露的故障模式——比如一次失败的 NCCL all-reduce 就可能让为期 30 天的训练任务功亏一篑。

![LLM 工程（3）：大规模预训练 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_1.png)

## 数据配比比架构更重要

![图3：数据混合组成](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig3_data_mixture.png)
\n过去三年所有可信的缩放研究都达成共识：在相同算力下，两个 LLaMA 式架构之间的性能差异很小（约 5% 困惑度），但不同数据配比带来的差距却极为显著（超过 30%）。Chinchilla 论文提出的计算最优缩放定律假设数据分布固定；一旦允许数据分布变化，数据质量便成为主导因素。
\n现代预训练数据配比大致如下（FineWeb-Edu [Penedo et al., 2024]、RedPajama-V2 [Together AI, 2024]、Dolma [Soldaini et al., 2024]——所有开源配比彼此相差无几）：

| 来源 | 占比 | 备注 |
|---|---|---|
| 过滤后的网页 (CommonCrawl) | 50–65% | 体量主导 |
| 代码 (GitHub, StackExchange) | 8–15% | 提升推理能力，不只是写代码 |
| 书籍 | 5–10% | 增强长上下文连贯性 |
| Wikipedia | 2–5% | 单位 token 的效用远高于平均 |
| 学术 / arXiv | 3–5% | 数学、引用格式 |
| 数学 (证明、教科书) | 2–4% | 专为逻辑推理优化 |
| 多语言网页 | 5–15% | 质量因语言差异极大 |
\n最被忽视却至关重要的指标是：**去重率**。CommonCrawl 2024 在文档级别去重后仅保留约 25% 的原始字节；若在行级别进行更激进的去重，则仅剩约 12%。Lee et al. (2022) 的研究表明，即使激进去重移除了 75% 的数据，模型困惑度反而更低。重复内容对语言模型而言是毒药——它教会模型死记硬背，而非泛化理解。
\nDeepSeek 的预训练说明（DeepSeek-V3 技术报告 [DeepSeek-AI, 2024]，2024 年 12 月）是我见过对配比最坦诚的：14.8T tokens 中，87% 来自代码、数学和网页，13% 是“高质量书籍和合成数据”。其中合成数据的实际贡献远超公开承认的程度。

## DataComp-LM：数据质量 > 数量，且有据可查
\n关于数据质量最严谨的公开研究来自 DataComp for Language Models 基准 [Li et al., 2024]。他们固定了模型架构和训练算力，运行了 416 次对照实验，仅改变数据配比。以下发现值得铭记：

- **质量过滤器起决定性作用**。一个基于 FastText 训练的质量分类器（用于预测文本是否接近“高质量”参考数据），在与基线相同的算力下训练，MMLU 成绩高出 6.6 个百分点，Core（22 任务基准）高出 3.5 分。
- **激进过滤效果更好**。仅保留质量得分前 10% 的文档，表现优于保留前 50%，尽管丢弃了 5 倍以上的 tokens。
- **去重与过滤协同增效**。在 0.7 Jaccard 阈值下使用 MinHash-LSH 去重可提升 MMLU 2.1 分；若结合质量过滤，则提升达 8.7 分。两者具有超加性——去重能清除质量过滤器可能遗漏的低质近重复内容。
- **最优配比因任务而异**。面向代码密集型下游任务的模型，其理想数据配比与通用聊天模型截然不同。不存在放之四海皆准的“通用最优解”。
\n在 DCLM-Baseline 数据（3T tokens）上训练的 7B 模型，尽管每参数算力少了 33%，MMLU 成绩仍比 Llama 2 7B（使用 2T 未明确说明、过滤较弱的数据）高出 11.5 分。这充分体现了数据质量所能带来的巨大影响。

## 合成数据：不可告人的秘密

![LLM 工程（3）：大规模预训练 —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_2.png)
\n直到 2023 年左右，主流观点仍是“合成数据即污染，绝不可用”。这一认知在 Phi-1 [Gunasekar et al., 2023]（Microsoft, 2023）发布后彻底改变——该研究证明，一个 1.3B 模型若完全在 GPT-4 生成的合成教科书式数据上训练，性能可媲美更大的代码专用模型。Phi 系列顺势而为，其他团队则悄然跟进。
\n到 2026 年，顶级开源模型普遍采用合成数据：

- **Qwen3 技术报告**：合成指令数据、合成代码、“多样化”合成问答。
- **Llama 3.1** [Dubey et al., 2024]：用于代码、数学和多语言的合成数据。
- **DeepSeek-V3**：用于“高质量书籍”的合成数据（法律风险低于网络爬取）。
\n主要风险之一是 **模式坍缩（mode collapse）**——仅依赖单一教师模型生成数据会导致输出分布过度集中。防御手段包括多教师合成与激进过滤。另一风险是评估基准的 **污染（contamination）**。Phi-3 曾被发现训练集中包含 MMLU 原题。标准对策是精确匹配去污染（丢弃任何与评估集存在 13-gram 重叠的数据），并辅以预留的新颖基准。

[Shumailov et al., 2024] 指出了一种更隐蔽的风险：**模型坍缩（model collapse）**。在由前代模型生成的数据上持续训练，会逐步收窄数据分布，最终导致性能退化。其实验表明，即使使用高质量模型输出进行递归训练，也会在 5–10 代内收敛至退化分布。生产环境中的缓解措施是：始终将合成数据与大量人工撰写内容混合（通常 50–70% 人类数据，30–50% 合成数据），并确保合成数据来自多个独立教师模型，而非单一强大模型。
\n经济账同样关键。使用 GPT-4o 生成 10 亿（1B）高质量合成 tokens 的成本约为 3 万美元；若改用较小的、专为合成微调的模型（如自托管集群上的 Qwen3-32B-Instruct），成本可降至数百美元。2026 年，大多数生产级合成数据管道采用混合策略：先用前沿模型（如 Claude 或 GPT-4o）生成多样化的“种子”数据，再用自托管的中等规模模型进行规模化扩产。

## 缩放定律：从 Chinchilla 到 200B
\nChinchilla 定律 [Hoffmann et al., 2022] 指出，计算最优训练应使用约 **每参数 20 个 tokens**。这意味着 7B 模型需 140B tokens，70B 模型需 1.4T。
\n早期 [Kaplan et al., 2020] 的缩放定律给出不同比例（约每参数 1.7 tokens），因其主要通过调整模型大小而非数据量来改变算力。Hoffmann 等人通过联合扫描模型与数据规模修正了方法论，使 Chinchilla 定律成为 2022–2023 年前沿训练的默认规划准则。
\n然而，Chinchilla 在实践中并不完全适用。更准确地说，它适用于 *训练算力最优*，但对于 **推理算力最优**（即模型被调用数十亿次的场景），应大幅超量训练小型模型。[Sardana et al., 2023] 在《超越 Chinchilla 最优》中形式化了这一观点。LLaMA-3 8B 在 15T tokens 上训练（约每参数 1900 tokens），是 Chinchilla 推荐值的 95 倍。该模型不仅显著优于 Chinchilla 最优的 8B 模型，服务成本也远低于 Chinchilla 最优的 70B 模型。
\n现实瓶颈在于：大多数开源数据源经过去重后，高质量唯一 tokens 总量不超过 200B。超过此阈值后，要么在同一数据上重复训练（多轮 epoching），要么依赖合成数据。研究表明，在高质量数据上，多轮训练最多有效至约 4 个 epoch，之后模型开始过拟合 [Muennighoff et al., 2023]。此后，合成数据成为唯一可行路径——这也解释了为何 2024 年后的前沿模型纷纷转向自产数据。

[Villalobos et al., 2024] 对自然语言数据上限的最佳估计指出：公共互联网上约有 300T tokens 的“高质量”英文文本。前沿实验室已对此进行了高强度清洗与过滤。当训练数据超过约 50T 高质量 tokens（Llama 3 和 DeepSeek-V3 所处的位置），进一步扩展只能依赖合成数据或私有数据源。这也是“预训练规模扩展”策略在 2026 年趋于停滞的原因之一——我们正逼近数据墙。

## 课程学习：从低质量向高质量退火

2024–2025 年的一项稳健发现是：若非均匀打乱所有数据，而是结构化地安排训练顺序，效率会更高。DeepSeek-V3 及其他近期训练采用的基本模式如下：

1. **主体阶段（Bulk phase）**（前 ~80% tokens）：广泛混合、以网页为主、质量门槛较低。
2. **退火阶段（Anneal phase）**（后 ~20% tokens）：高质量配比——书籍、论文、数学、精选代码、指令微调数据。
3. 学习率在整个过程中按余弦调度衰减。
\n其直觉在于：早期训练建立广泛的语言能力，晚期训练则将模型塑造成目标分布。在低学习率下对高质量数据进行退火，比全程均匀混合高质量数据，在提升下游任务表现上要高效得多。
\nLlama 3 [Dubey et al., 2024] 报告了一次独立的“退火运行”：在峰值学习率的 1/10 下，继续在高度过滤的 40B-token 高质量数据上训练。尽管这部分仅占总训练量的极小比例，却贡献了数个 MMLU 分数。

## 长上下文预训练：分阶段序列长度方案
\n从零开始直接训练 128K 上下文模型是低效的——早期预训练主要受益于短序列数据，而长序列注意力的 FLOPs 成本呈平方增长。现代长上下文模型普遍采用分阶段策略：

- **阶段 1**（0–90% tokens）：4K 上下文，构建通用语言能力。
- **阶段 2**（90–95%）：32K 上下文，引入长程依赖模式。
- **阶段 3**（95–100%）：128K 上下文，配合精选长文档数据（书籍、代码库、长论文）。
\nLlama 3 采用了四阶段课程达到 128K；NVIDIA 的 Nemotron-340B 也使用类似方法；Qwen3 则通过分阶段预训练达到 128K，并在最后借助 YaRN 扩展至 256K。
\n核心洞见在于：模型的基础语言建模能力在短上下文阶段已基本建立，长上下文阶段只需教会模型如何利用扩展后的注意力机制。这比全程使用长序列训练经济得多。序列打包（sequence packing）——将多个短文档拼接成单个长序列并配合注意力掩码——在短序列阶段几乎免费提升了计算利用率。

## μP：让你能在小规模上调参的参数化方法

![图4：μP 在不同宽度下的扩展](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig4_mup_scaling.png)
\n大规模训练的一大痛点是：在 1B 模型上调优的超参数无法直接迁移到 70B 模型，尤其是学习率。标准参数化失效的原因在于激活值的量级随模型宽度变化而变化。

**μP（Maximal Update Parameterization）** [Yang et al., 2022] 正是为解决此问题而生。在 μP 下，通过按层宽缩放初始化方差和学习率，使得不同规模模型的激活值与梯度保持一致量级。其效果立竿见影：在 1 亿参数模型上调好的学习率，可直接用于 1000 亿参数模型。

```python
# Standard: per-layer LR doesn't depend on width
# μP: scale LR for hidden Linear layers by 1 / fan_in_ratio
def mup_lr(base_lr, fan_in, base_fan_in=256):
    return base_lr * (base_fan_in / fan_in)
```
\nCerebras 与微软研究院的实验均表明，经 μP 调优的 7B → 70B → 700B 模型，在相同训练进度下的损失曲线偏差不超过 1%。若不使用 μP，每个规模都需耗费数百万美元重新摸索学习率。
\n完整的 μP 方案在每层宽度上差异化处理四个要素：
1. 初始化标准差：$\sigma \propto 1/\sqrt{\text{fan}_{\text{in}}}$
2. 隐藏层学习率：$\eta \propto 1/\text{fan}_{\text{in}}$
3. 输出层乘数：$\text{logits} = (1/d) \cdot W_{\text{out}} \cdot h$ 而非 $W_{\text{out}} \cdot h$
4. Embedding 层学习率：保持不变（不缩放）
\n到 2026 年，绝大多数生产训练至少会采用 μP 进行学习率迁移。部分团队更进一步，使用 μTransfer 处理 batch size 与权重衰减。[Wortsman et al., 2024] 还发现，μP 能提升训练对噪声的鲁棒性，并简化从不稳定状态中的恢复过程。

## 并行策略：FSDP、ZeRO、Pipeline、Tensor

![图1：ZeRO/FSDP 内存阶段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig1_zero_stages.png)
\n当模型超出单 GPU 显存时，需考虑四个正交的并行维度：

1. **数据并行（DP）** —— 每张 GPU 复制完整模型，切分批次。
2. **分片数据并行（FSDP / ZeRO-3）** [Rajbhandari et al., 2020] —— 将参数跨 DP rank 分片，每层前向计算前即时聚合。
3. **张量并行（TP）** [Shoeybi et al., 2019] —— 将每个矩阵乘法跨 GPU 切分（Megatron 风格）。
4. **流水线并行（PP）** [Huang et al., 2019] —— 将模型层跨 GPU 切分，通过微批次（micro-batch）流水执行。
\n对于在 64 张 H100 上训练 70B 模型，典型配置为：

```text
TP=8 within node (NVLink)
PP=2 across nodes (200 Gbps Infiniband)
DP=4 with FSDP across the remaining axis
```
\n该配置共使用 64 张 GPU（8 × 2 × 4），将通信最密集的 TP 限制在节点内高速 NVLink 上，PP 的气泡（bubble）则与下一个微批次重叠。到 2026 年，ZeRO-3（FSDP 的一种变体）已成为默认选择，因其无需修改代码即可支持 70B 模型训练：

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

`mixed_precision` 配置至关重要。**务必使用 fp32 进行 reduce 操作**——当跨 DP rank 累加成千上万个微小梯度时，bf16 会静默下溢。我们曾排查过损失发散问题，最终发现根源竟是遗漏了 `reduce_dtype=fp32`。
\n对于巨型模型（如 DeepSeek-V3 671B、Qwen3-Max），仅靠 FSDP 已不够，需定制代码实现四维并行。Megatron-LM、NVIDIA NeMo、ColossalAI 和 DeepSpeed 是四大主流生产框架，多数实验室均基于其中之一进行深度定制。

## 实战案例：64 张 H100 上跑 70B 的并行选择

![图2：流水线并行调度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig2_pipeline_parallelism.png)
\n为何在此配置下 TP=8、PP=2、DP=4 是最优解？每张 H100 拥有 80 GB HBM，总计 5120 GB。

**单卡显存预算**。每个 rank 需占用：
- 权重：70B × 2 字节（BF16）/ TP / PP = 70e9 × 2 / 16 = 8.75 GB
- 梯度：同权重 = 8.75 GB
- Adam 优化器状态：4 字节/参数 × 2（m, v）/ TP / PP / DP_FSDP = 70e9 × 8 / 64 = 8.75 GB（若 FSDP 分片优化器）
- 激活值（启用梯度检查点）：随 batch × seqlen × model dim 变化，70B 模型在 seqlen=8K 时通常为 30–50 GB
- KV/临时缓冲区：约 5 GB
\n总计约 62 GB，剩余 18 GB 用于 FlashAttention 临时空间、NCCL 缓冲及峰值激活。虽紧张但可行。

**为何不采用纯 FSDP=64？** 此配置需在每层前向计算前 all-gather 70B 参数。70B × 2 字节 = 140 GB，分摊至 64 个 rank 为 2.2 GB/rank/gather。80 层累计约 175 GB/gather/forward。即便 NVLink 带宽达 900 GB/s，也需约 200 ms 纯通信时间——与计算时间相当，严重拖累吞吐。

**为何不采用纯 TP=64？** TP 在每次矩阵乘法后需通信（行并行层上的 all-reduce）。每 token 的通信量为 $O(d_{\text{model}} \times \text{TP})$。若在节点间链路（200 Gbps Infiniband）上运行 TP=64，通信将完全主导耗时。TP 仅在单节点 NVLink 内部（当前最多 8 卡）才高效。

**为何选择 TP=8、PP=2、DP=4？** 节点内 TP=8 利用高速 NVLink 处理密集通信；PP=2 将模型跨两节点切分，PP 通信频率较低（每微批次一次）；DP=4 配合 FSDP 将优化器状态分片至 4 个 DP rank。PP 气泡（占 step 时间 10–20%）可接受，其他配置开销更大。
\n此仅为一例。不同模型规模与集群拓扑各有最优解，通用原则是：最小化跨节点通信，将 TP 限制在最快互联内，用 PP 处理下一层粒度，剩余并行度由 FSDP/DP 填补。

## 千卡以上才会暴露的故障模式
\n有些问题在 8 卡集群中无关紧要，但在 8000 卡规模下足以摧毁整个训练：

**静默数据损坏**。单 GPU 因宇宙射线或电压波动返回错误计算结果，产生 NaN 并经 all-reduce 扩散。防御方法：定期校验梯度 checksum——若 rank 0 与 rank 100 的梯度范数差异超过 $10^{-3}$，立即暂停并重新分片。

**Loss 尖峰**。坏批次（如 Python 解析器 dump、HTML 乱码）引发梯度尖峰，偏移优化器动量且无法恢复。防御措施：激进梯度裁剪（默认 1.0，超大模型有时设为 0.3），并设置触发器——当梯度范数超过运行中位数 5 倍时跳过该批次。Llama 3 论文提到“检查点回滚”是标准流程：若 50 步内无法恢复，回滚至上一良好检查点，学习率降低 30%，然后继续。

**NCCL 挂死**。单节点网卡固件 bug 导致该 rank 的 all-reduce 永不返回，其余 999 个 rank 永久阻塞。防御：设置 `NCCL_TIMEOUT_S=300` 并部署 watchdog 自动重调度。

**检查点损坏**。2.7 TB 的 FSDP 检查点（如 Qwen3-32B 优化器状态）写入需 20 多分钟。若节点中途 OOM，不仅丢失当前 step，还可能覆盖上一良好检查点。务必先写入 `step_xxx.tmp/`，执行 fsync，再原子移动（`mv`）。

**硬件异构性**。即使是同型号 H100，批次间时钟频率也有 2–3% 差异，慢卡会拖累整体。防御：预热阶段按单卡吞吐排序，将慢卡分配至 TP=1 的 rank。

**Embedding 坍塌**。若早期 epoch 中 embedding 梯度流被裁剪或 NaN 污染，embeddings 可能坍缩为近似相同向量。症状：训练 loss 卡在 log(V)，困惑度稳定在 V。诊断：监控 embedding 矩阵奇异值——若最大奇异值比第二大值高 10 倍以上，即已坍塌。修复：回滚至坍塌前，embedding 学习率翻倍，重启训练。

**梯度范数发散**。不同于 loss 尖峰，此现象表现为梯度范数在数千步内缓慢上升，尽管 loss 稳定。这通常预示未来崩溃。需监控各参数组的 `||g||_2`（而非仅全局），LM head 或最终 transformer 层常为罪魁祸首。
\nMeta 的 OPT-175B 日志 [Zhang et al., 2022] 仍是描述真实大规模训练的最佳公开文档——其 33 天运行中平均每 1.7 天发生一次节点故障。

## 训练期间实际发生了什么

![图5：训练损失曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/fig5_loss_curve.png)
\n现代 70B 预训练运行（如 LLaMA-3 70B，15T tokens，约 1900 张 H100 运行 60 天）大致消耗：

- **Tokens**：15T 输入，batch size ≈ 16M tokens
- **Steps**：15T / 16M ≈ 940K
- **单步耗时**：≈ 5.5 秒（启用 sequence packing）
- **总算力**：≈ $2.5e25$ FLOPs ≈ 6000 H100-年
\n学习率按余弦调度从短暂 warmup 衰减至峰值的约 10%。70B 模型峰值学习率约 1.5e-4（更大模型更低，若遵循 μP 则自动缩放）。权重衰减 0.1，β2=0.95，优化器为 AdamW。
\n评估按检查点进行：每 2000 步，在 5–10 个小基准上测试（Hellaswag、ARC-easy、OpenBookQA、GSM8K-100、HumanEval-50）。追踪 loss 成本低廉，但下游评估才是真实信号。

## 生产现实：前沿实验室实际在做什么
\n论文描述架构，生产系统则主要是工程调度。前沿预训练团队普遍做三件事，却从不写入 model cards：

**多规模并行运行**。70B 目标训练前，会先进行 1B 和 7B 的“侦察”运行（数据量分别为 1/100 和 1/10），使用相同架构、优化器与数据配比。侦察运行用于验证配方，并调优 μP 无法迁移的超参（主要是调度细节）。待侦察运行表现良好，70B 训练才高置信度启动。Llama 3 论文提到同时运行 405B 与 70B 以持续验证缩放行为。

**激进监控**。包括 loss 曲线、梯度范数、单参数统计、注意力模式直方图、专家利用率（MoE）、各层激活量级、评估分数等。监控仪表盘并非可选组件，而是防止问题累积的唯一手段。Anthropic 在 model cards 中开源了部分监控指标（裁剪比率、激活统计）；OpenAI 在 GPT-4 论文中亦有描述。

**持续数据精炼**。训练数据并非静态。团队持续运行新质量分类器，移除新发现的污染，添加新合成数据源。Llama 3 明确提到在训练中途用重训的质量分类器处理全部数据，并据此动态调整后续批次权重。“训练前固定数据”只是学术理想，生产团队始终在迭代。

**流式分词 vs 预分词**。预分词 15T tokens 需数周时间和 100+ TB 存储。流式分词要求 tokenizer 跟上数据加载速度，需高度优化的 Rust 实现。多数生产系统选择预计算 token IDs，但从分布式文件系统流式读取，而非加载原始文本。

## 常见坑点
\n以下五个问题我曾亲眼见证其摧毁预训练运行：

**1. 忽略 NCCL 集合通信的 bf16/fp32 不匹配**。NCCL 默认在 fp16 模式下 all-reduce，累加大量微小梯度时会静默下溢。稳妥做法：设置 `NCCL_PROTO=Simple` 保稳定，FSDP 中指定 `reduce_dtype=fp32`。Llama 3 论文提到，他们在约 50 万 step 时因未设此选项导致不稳定，添加后才恢复。

**2. RoPE 基频未适配上下文长度**。RoPE 默认基频 $\theta = 10000$。在 32K 上下文下，频率混叠导致模型无法学习长程模式。解决方案：按 [bloc97 NTK-aware] 将基频缩放为 $10000 \times (\text{seqlen}/2048)^{d/(d-2)}$，或直接使用 YaRN [Peng et al., 2024]。我们曾为 32K 性能下降排查一周，最终发现 RoPE 基频仍停留在 4K 预训练阶段的设置。

**3. 序列打包未正确掩码跨文档注意力**。序列打包将多篇文档拼接为单序列。若未在文档边界重置 `position_ids` 和 `attention_mask`，模型会学习虚假跨文档关联，MMLU 成绩下降 1–3%。

**4. 激活检查点混用非确定性算子**。检查点重计算需重跑前向传播，要求算子确定性。部分 FlashAttention 与 GeLU 实现有非确定性数值归约，导致重算梯度与保存的前向值不一致。此类细微退化常在数千步后显现。务必在检查点区域设置 `torch.use_deterministic_algorithms(True)`，或验证算子确定性。

**5. 优化器状态未分片存储**。70B 模型的 AdamW 优化器状态达 560 GB（70B × 8 字节）。数千节点同时写入共享文件系统易死锁。应使用 FSDP 分片检查点（各 rank 写自身分片）配合并行写入。我们曾因意外串行化，导致检查点写入耗时 4 小时。

## 2024–2026 研究前沿
\n继当前 Llama 式“过滤 + 去重 + 训练”范式之后，新方向包括：

**大规模数据效率**。[Marion et al., 2023]（《少即是多》）表明，用精心筛选的 200B-token 子集训练，效果可匹敌 1T-token 未过滤全集。这本质是“数据早停”——挑选信息量最大的样本，跳过冗余内容。

**课程驱动的动态混合**。[Albalak et al., 2024] 证明，根据模型进展动态调整各数据源比例，可比固定混合提升 1–2 MMLU。DoReMi [Xie et al., 2023] 提出 min-max 优化自动寻找最优权重。

**超越 Chinchilla 的算力最优**。混合专家（MoE）改变了缩放定律。[DeepSeek-AI, 2024] 基于 V3 实验报告了 MoE 专属缩放律：最优激活/总参数比、专家数量、专家大小均与训练算力存在非平凡依赖关系。社区仍在校准 MoE 特定缩放律。

**持续预训练**。多数生产模型现采用更新而非从头训练。Llama 3.1 即 Llama 3 + 额外数据持续预训练 + 后训练。持续预训练成本更低，但需精细调度学习率以避免灾难性遗忘。

## 总结与下一步
\n预训练 70% 是数据工程，30% 是分布式系统工程，架构选择反而是三者中最次要的。务必做对：数据配比、强力去重、为推理成本超量训练、用 μP 实现超参迁移、根据硬件拓扑选择 FSDP+TP+PP 组合，并为大规模特有故障编写防御代码。
\n下一章：**后训练**。SFT、DPO、RLHF、RLAIF——它们究竟优化了什么？奖励模型何时失效（且频繁失效）？LoRA 与全量微调之争？以及将基座模型转化为客户可用产品的生产级配方。

## 参考文献

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
