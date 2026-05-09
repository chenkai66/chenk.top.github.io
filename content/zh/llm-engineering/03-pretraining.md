---
title: "大模型工程（三）：规模化预训练"
date: 2026-04-28 09:00:00
tags:
  - llm
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
预训练决定了 LLM 的大部分能力，也是 leaderboard 和现实差距最大的地方。公开的训练 run 大多是工程奇迹，科学成分反而少。这篇不聊 OpenAI 那种规模的事，只聊你不是 OpenAI 时必须搞对的几件事：数据、并行选型，还有集群大到一次糟糕的 NCCL all-reduce 就能毁掉 30 天 run 的失败模式。

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_1.png)
## 数据混合比架构更重要

![fig3: 数据混合构成](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig3_data_mixture.png)

过去三年，所有靠谱的 scaling 研究都指向一个结论：算力相同的情况下，两种 LLaMA 风格架构的差异很小，大约 5% perplexity。但数据混合的差异却很大，超过 30%。Chinchilla 论文提出的 compute-optimal scaling law 假设数据分布固定。一旦允许数据分布变化，数据就成了最关键的因素。

现代预训练数据混合大致如下（FineWeb-Edu [Penedo et al., 2024]、RedPajama-V2 [Together AI, 2024]、Dolma [Soldaini et al., 2024] 这些公开数据集的差异都在几个百分点内）：

| 来源 | 占比 | 备注 |
|---|---|---|
| 过滤后的网页（CommonCrawl） | 50-65% | 数据量最大 |
| 代码（GitHub、StackExchange） | 8-15% | 提升推理能力，不止是代码 |
| 书籍 | 5-10% | 提供长上下文连贯性 |
| Wikipedia | 2-5% | 每个 token 的价值更高 |
| 学术 / arXiv | 3-5% | 包含数学公式和引用 |
| 数学（证明、教材） | 2-4% | 专为推理优化 |
| 多语言网页 | 5-15% | 各语言质量差异大 |

很少有人提到一个关键数字：**去重率**。CommonCrawl 2024 在文档级别去重后，只保留了约 25% 的原始字节。如果按行级别去重，只剩 12%。Lee 等人在 2022 年的论文中指出，激进去重虽然移除了 75% 的数据，但反而降低了 perplexity。重复数据对语言模型来说是毒药，它会让模型死记硬背，而不是学会泛化。

DeepSeek 的预训练报告（DeepSeek-V3 技术报告 [DeepSeek-AI, 2024]，2024 年 12 月发布）是我见过最坦诚的。他们明确列出了数据比例：14.8T token 中，87% 是代码、数学和网页数据，13% 是“高质量书籍和合成数据”。合成数据的作用被大大低估了，实际上它的贡献远超认知。

下一节会详细聊聊合成数据的设计思路和踩过的坑。
## DataComp-LM：数据质量大于数量，证据确凿

DataComp for Language Models 基准测试 [Li et al., 2024] 是目前对数据质量最严谨的公开研究。他们固定了模型架构和训练算力，跑了 416 组对照实验，只调整数据组合。结果有几个亮点。

- **质量过滤是关键。** 用 FastText 训练一个质量分类器，判断文本是否像“高质量”参考数据。同样算力下，比基线模型 MMLU 高 6.6 个百分点，Core（22 项任务基准）高 3.5 个百分点。
- **激进过滤更有效。** 只保留质量评分前 10% 的文档，效果比保留前 50% 还好。虽然丢弃了 5 倍的 token，但性能更强。
- **去重和过滤配合更好。** MinHash-LSH 在 0.7 Jaccard 阈值下去重，MMLU 提升 2.1。结合质量过滤后，提升幅度达到 8.7。两者叠加效果显著——去重能清理质量过滤漏掉的低质内容近似副本。
- **最佳数据组合因任务而异。** 训练代码密集型下游任务的模型，偏好不同的数据组合。通用聊天模型则不同。没有普适最优解。

在 DCLM-Baseline 数据集（3T token）上训练的 7B 模型，性能超越了 Llama 2 7B。后者用了 2T token 的未明确说明且过滤不严的数据。前者 MMLU 高出 11.5 分，每参数算力还少了 33%。这就是数据质量的力量。
## 合成数据：不能说的秘密

![LLM Engineering (3): Pretraining at Scale — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/llm-engineering/03-pretraining/illustration_2.png)

2023 年之前，大家都觉得合成数据就是污染，碰都别碰。Phi-1 [Gunasekar et al., 2023]（Microsoft，2023）打破了这个观念。它证明了一个 1.3B 参数的模型，如果完全用 GPT-4 生成的教科书风格数据训练，性能能媲美更大的代码模型。Phi 系列一路坚持，其他公司也悄悄跟上。

到了 2026 年，顶级开源模型全都离不开合成数据：

- **Qwen3 技术报告**：指令、代码、问答，全是合成的。
- **Llama 3.1** [Dubey et al., 2024]：代码、数学、多语言任务，合成数据撑起一片天。
- **DeepSeek-V3**：合成高质量书籍，比爬取数据更合法。

风险主要有两个。一是 **mode collapse**，单靠一个 teacher 模型生成数据，分布会过于集中。解决办法是多 teacher 模型生成，再加严格过滤。二是 **评测集污染**。比如 Phi-3 的训练数据里就被发现混入了 MMLU 题目。标准应对方法是精确匹配去重（去掉与评测集有 13-gram 重叠的数据），同时保留全新的评测基准。

[Shumailov et al., 2024] 提出了一个更隐蔽的风险：**model collapse**。如果一直用前一代模型生成的数据训练，数据分布会逐渐变窄，最终导致模型质量下降。实验表明，即使递归训练用的是高质量模型输出，5 到 10 代之内也会退化。实际生产中，我通常会混合使用合成数据和人类编写的数据（人类数据占 50%-70%，合成数据占 30%-50%）。另外，尽量从多个独立 teacher 模型生成数据，而不是依赖单一强模型。

经济成本也很重要。用 GPT-4o 生成 10 亿个高质量 token 大约要花 $30K。但如果用更小的、专门为合成任务微调的模型（比如 Qwen3-32B-Instruct 在自托管集群上跑），成本能降到几百美元。2026 年，大多数合成数据 pipeline 都采用混合策略：先用强大的前沿模型（如 Claude 或 GPT-4o）生成多样化的“种子”数据，再用自托管的中型模型进行规模化扩展。
## Scaling law：Chinchilla，然后是 200B

Chinchilla law [Hoffmann et al., 2022] 提出，计算最优训练需要 **每参数 20 个 token**。7B 模型要 140B token，70B 模型要 1.4T。

[Kaplan et al., 2020] 更早提出过一个比例，约 **每参数 1.7 个 token**。他们调整算力时更关注模型大小，而不是数据量。Hoffmann 等人改进了方法，同时调整数据量和模型大小。于是，Chinchilla law 成为 2022-2023 年前沿实验的默认参考。

但实际中，Chinchilla 并不完全对。它适合 *训练算力最优* 的场景。但在 **推理算力最优** 的情况下——比如模型会被反复调用几十亿次——应该大幅过度训练小模型。[Sardana et al., 2023] 把这总结为 "Beyond Chinchilla-Optimal"。LLaMA-3 8B 训练用了 15T token，约 **每参数 1900 个 token**，是 Chinchilla 推荐值的 95 倍。这个模型不仅比 Chinchilla 最优的 8B 模型强得多，部署成本也远低于 Chinchilla 最优的 70B 模型。

数据天花板是个大问题。大多数开源数据源去重后，高质量 token 总量在 200B 左右就到顶了。再往后，要么重复训练，要么生成合成数据。高质量数据上，多轮训练最多撑到 4 轮。再多，模型就会过拟合 [Muennighoff et al., 2023]。之后，合成数据成了唯一出路。这也是为什么 2024 年后的前沿模型都开始自己生成数据。

目前对自然语言数据天花板的最佳估计来自 [Villalobos et al., 2024]：公开互联网上大约有 300T 高质量英文文本。前沿实验室已经对这些数据进行了激进处理和过滤。超过 50T 高质量 token 后（Llama 3 和 DeepSeek-V3 大概在这个范围），合成数据和私有数据源成了扩展的唯一选择。这也是 "scaling 预训练" 策略在 2026 年逐渐趋于平缓的原因之一——我们正在逼近数据墙。
## Curriculum：从低到高的退火策略

2024-2025 年有个重要发现。训练时，数据不能简单打乱，顺序设计很关键。DeepSeek-V3 和其他几个近期训练都用了类似模式。

1. **主体阶段**（前 80% token）：数据广泛混合，网页内容为主，质量门槛较低。  
2. **退火阶段**（后 20% token）：高质量数据混合，包括书籍、论文、数学、精选代码和指令调优数据。  
3. 学习率全程按余弦曲线衰减。

逻辑很简单。早期训练建立广泛语言能力，后期用高质量数据调整模型分布。低学习率下退火，比全程混入高质量数据更高效，尤其在提升下游基准测试表现时。

Llama 3 [Dubey et al., 2024] 提到一个“退火阶段”。他们在训练末尾用严格过滤的 40B 高质量 token 数据集，以峰值学习率的 1/10 继续训练。这个阶段占总 token 很小一部分，却让 MMLU 指标提升了好几点。
## 长上下文预训练：分阶段序列长度方案

从零开始训一个 128K 上下文的模型，太浪费了。早期预训练靠短序列数据就够了，长序列注意力的计算成本是平方级增长的。现代长上下文模型都用分阶段调整序列长度的方法。

- **第一阶段**（0-90% 的 token）：4K 上下文，主要用来打基础。
- **第二阶段**（90-95%）：32K 上下文，引入长程依赖模式。
- **第三阶段**（95-100%）：128K 上下文，用精选的长文档数据（比如书籍、代码仓库、长篇论文）。

Llama 3 用了四阶段课程扩展到 128K。NVIDIA 的 Nemotron-340B 也用了类似的分阶段方法。Qwen3 则通过分阶段预训练达到 128K，最后用 YaRN 扩展到 256K。

核心思路很简单。短上下文阶段已经能建立通用语言建模能力，长上下文阶段只需要教模型怎么用扩展的注意力机制。这比全程用长序列训练便宜多了。短序列阶段可以用“序列打包”——把多个短文档拼成一个长序列，再加注意力掩码——几乎不浪费计算资源。
## μP：小规模调参，大规模通用

![fig4: μP 跨宽度缩放](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig4_mup_scaling.png)

大规模训练有个痛点。1B 参数调好的超参数，搬到 70B 模型上就崩了。特别是学习率。标准参数化下，激活值幅值随宽度变化，规模一变就出问题。

**μP（Maximal Update Parameterization）** 解决了这个麻烦。这是 [Yang et al., 2022] 提出的方法。用 μP，初始化方差和学习率按层宽度调整。激活值和梯度幅值在不同宽度下保持一致。最终效果不错：100M 参数调好的学习率，直接能用到 100B 参数模型上。

```python
# 标准做法：单层学习率不依赖宽度
# μP：隐藏 Linear 层学习率按 1 / fan_in_ratio 缩放
def mup_lr(base_lr, fan_in, base_fan_in=256):
    return base_lr * (base_fan_in / fan_in)
```

Cerebras 和 Microsoft Research 做过实验。用 μP 调参的 7B、70B 和 700B 模型，loss 曲线几乎完全对齐。相同训练进度下，误差控制在 1% 以内。没有 μP，每个规模都得花几百万美元找学习率。

完整的 μP 方法调整四样东西：
1. 初始化标准差：$\sigma \propto 1/\sqrt{\text{fan}_{\text{in}}}$
2. 隐藏层学习率：$\eta \propto 1/\text{fan}_{\text{in}}$
3. 输出层乘子：$\text{logits} = (1/d) \cdot W_{\text{out}} \cdot h$，而不是 $W_{\text{out}} \cdot h$
4. 嵌入层学习率：保持不变，不缩放。

到了 2026 年，大多数生产环境至少会用 μP 迁移学习率。有些人更进一步，用 μTransfer 处理 batch size 和 weight decay。[Wortsman et al., 2024] 发现，μP 让训练对噪声更鲁棒，也更容易从不稳定中恢复。
## 并行：FSDP、ZeRO、Pipeline、Tensor

![fig1: ZeRO/FSDP 显存阶段](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig1_zero_stages.png)

模型超出单卡内存后，有四个并行方向：

1. **数据并行（DP）**——每张 GPU 复制一份模型，分批处理数据。
2. **分片数据并行（FSDP / ZeRO-3）** [Rajbhandari et al., 2020]——参数分片到不同 DP 节点，前向计算时按需聚合。
3. **张量并行（TP）** [Shoeybi et al., 2019]——矩阵乘法拆分到多张 GPU，类似 Megatron 的做法。
4. **流水线并行（PP）** [Huang et al., 2019]——层分布到不同 GPU，用 micro-batch 流水执行。

70B 参数的模型，64 张 H100 的典型配置如下：

```
TP=8 节点内（NVLink）
PP=2 跨节点（200 Gbps Infiniband）
DP=4 剩余维度用 FSDP
```

总共 64 张 GPU（8 × 2 × 4）。通信量最大的张量并行限制在 NVLink 内，流水线并行的空闲时间被下一个 micro-batch 覆盖。到 2026 年，ZeRO-3（FSDP 的一种实现）会成为默认选择。它无需改代码就能支持 70B 模型。

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

`mixed_precision` 部分很关键。梯度规约一定要用 fp32。bf16 在跨 DP 节点求和上千个小梯度时会悄无声息地溢出。我曾经调试过 loss 发散的问题，最后发现就是漏写了 `reduce_dtype=fp32`。

对于超大模型（比如 DeepSeek-V3 671B 和 Qwen3-Max），FSDP 也不够用了。必须用四维并行加自定义代码。主流生产框架有四个：Megatron-LM、NVIDIA NeMo、ColossalAI 和 DeepSpeed。大多数实验室都会基于其中一个框架拉分支开发。
## 实战案例：70B 模型在 64 张 H100 上的并行策略选择

![fig2: pipeline 并行调度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig2_pipeline_parallelism.png)

为什么 TP=8、PP=2、DP=4 是最佳选择？每张 H100 显卡有 80 GB HBM，总共 64 × 80 = 5120 GB。

**每张显卡的内存分配。**  
- 权重：70B × 2 字节（BF16）÷ TP ÷ PP = 8.75 GB  
- 梯度：和权重一样，也是 8.75 GB  
- Adam 优化器状态：4 字节/参数 × 2（m、v）÷ TP ÷ PP ÷ DP_FSDP = 8.75 GB（FSDP 切分优化器）  
- 激活值（带梯度 checkpointing）：随 batch × seqlen × model dim 变化，seqlen 8K 下通常 30-50 GB  
- KV/scratch 缓冲区：约 5 GB  

总计约 62 GB，剩下 18 GB 给 FlashAttention、NCCL 缓冲区和峰值激活用。虽然紧张，但能跑起来。

**为什么不直接用纯 FSDP=64？**  
纯 FSDP 每次前向传播前需要 all-gather 70B 参数。70B × 2 字节 = 140 GB 数据，分摊到 64 个 rank，每个 rank 每次 gather 处理 2.2 GB。模型有 80 层，每次前向传播每个 rank 要处理约 175 GB 流量。即使 NVLink 带宽 900 GB/s，通信时间也要 200 毫秒，几乎和计算时间持平，吞吐量直接崩了。

**为什么不直接用纯 TP=64？**  
TP 每次矩阵乘法后都需要通信（行并行层上的 all-reduce）。每 token 的 all-reduce 数据量是 $O(d_{\text{model}} \times \text{TP})$。跨节点运行时（200 Gbps Infiniband），通信开销会成为瓶颈。TP 只能在单个 NVLink 节点内高效运行，当前最多支持 8 张显卡。

**为什么选择 TP=8、PP=2、DP=4？**  
TP=8 在节点内部利用高速 NVLink 处理密集通信。PP=2 将模型切分到两个节点，PP 通信频率低（每个 micro-batch 一次）。DP=4 配合 FSDP 切分优化器状态，每个 rank 存储 1/4 权重和 1/(TP×PP) 层数。PP 的流水线气泡占步长时间 10-20%，可以接受。其他策略开销更大。

这只是其中一个配置。不同模型规模和集群拓扑有不同的最优解。通用原则是：减少跨节点通信，将 TP 限制在最快互联范围内，用 PP 提供下一层粒度，剩下的并行性用 FSDP 或 DP 填补。
## 1000 卡以上才出现的失败模式

8 卡上无所谓，但到了 8000 卡就能搞垮整个训练任务。

**静默数据损坏。** 某张卡因为宇宙射线或者电压波动返回错误结果，产生一个 NaN，随后被 all-reduce 扩散。解决办法是定期校验梯度。如果 rank 0 和 rank 100 的梯度范数差距超过 $10^{-3}$，立即停止并重新分片。

**Loss 突增。** 坏 batch（比如 Python 解析器乱码、HTML 垃圾数据）会导致梯度突增，优化器动量偏移后无法恢复。应对方法是激进的梯度裁剪，默认值 1.0，超大模型有时降到 0.3。再加一个“跳过 batch”触发器，当梯度范数超过跑动中位数 5 倍时触发。Llama 3 论文提到，“checkpoint 回滚”是标准操作。如果 loss 突增 50 步内无法恢复，回滚到上一个正常 checkpoint，学习率降低 30%，继续训练。

**NCCL 挂起。** 某个节点网卡固件 bug 会让 all-reduce 永不返回，其他 999 个 rank 也会阻塞。防御措施是设置 NCCL_TIMEOUT_S=300，并加一个看门狗重新调度任务。

**Checkpoint 损坏。** 2.7 TB 的 FSDP checkpoint（比如 Qwen3-32B 的优化器状态）写入需要 20 多分钟。如果某个节点中途 OOM，不仅丢掉当前 step，还会覆盖上一个正常 checkpoint。解决办法是先写到 `step_xxx.tmp/` 目录，fsync 后用 `mv` 移动到目标位置。

**异构硬件。** 即使是“同型号”H100，不同批次时钟速度也可能差 2-3%。慢的 rank 拖累整体性能。解决办法是在 warmup 阶段按吞吐量排序，把慢的 rank 放到 TP=1 的位置。

**嵌入坍缩。** 嵌入矩阵梯度流在早期 epoch 被裁剪或损坏，嵌入可能坍缩成几乎相同的向量。症状是训练 loss 卡在 log(V)，困惑度停留在 V。诊断方法是监控奇异值，如果最大奇异值比次大的高 10 倍以上，说明嵌入已坍缩。修复方法是回退到坍缩之前，嵌入学习率提高 2 倍，重启训练。

**梯度范数发散。** 这和 loss 突增不同。即使 loss 稳定，梯度范数也可能在几千步内缓慢上升，通常预示未来崩溃。监控时要注意每个参数组的 `||g||_2`，而不仅是全局梯度范数。LM head 或最后一个 transformer 层往往是罪魁祸首。

Meta OPT-175B 的运行日志 [Zhang et al., 2022] 至今仍是描述真实大规模训练的最佳公开文档。他们在 33 天训练中，平均每 1.7 天遇到一次节点故障。
## 一次运行中到底发生了什么

![fig5: 训练 loss 曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/llm-engineering/03-pretraining/fig5_loss_curve.png)

现代 70B 参数预训练任务（LLaMA-3 70B，15T tokens，约 1900 张 H100，耗时 60 天）的资源消耗如下：

- 输入 15T tokens，batch size 约 16M tokens。  
- 总步数：15T / 16M ≈ 940K。  
- 每步耗时：约 5.5 秒（使用 sequence packing 技术）。  
- 总算力：约 $2.5 \times 10^{25}$ FLOPs，相当于 6000 张 H100 运行一年。  

学习率调度用余弦衰减（cosine decay）。先短暂 warmup，再衰减到峰值的 10% 左右。70B 模型的峰值学习率约为 1.5e-4。更大模型的学习率更低，严格按 μP 缩放规则会更严谨。优化器选 AdamW，参数设置为 weight decay 0.1，β2=0.95。

评估按 checkpoint 进行。每 2000 步，在 5 到 10 个小型基准测试集上跑一遍。包括 Hellaswag、ARC-easy、OpenBookQA、GSM8K-100 和 HumanEval-50。跟踪 loss 成本低，但下游任务的评估结果才是真正的信号。
## 生产真相：前沿实验室真正在做的事

论文讲架构，生产拼后勤。每个大模型预训练团队都有几件事不会写进模型卡，但踩过的坑都绕不开。

**多规模并行跑。** 跑 70B 模型前，先用 1/100 和 1/10 数据量跑 1B 和 7B 的“侦察”模型。架构、优化器、数据混合完全一致。侦察模型验证训练配方，调那些 μP 搞不定的超参数，比如学习率调度细节。侦察模型跑稳了，70B 才敢放心启动。Llama 3 论文提到，他们同时跑了 405B 和 70B，持续验证扩展行为。

**监控不能少。** 损失曲线、梯度范数、单参数统计、注意力模式直方图、MoE 专家利用率、每层激活值、评估分数，一个都不能漏。监控仪表盘不是可选，而是提前发现问题的唯一手段。Anthropic 开源了部分监控工具，比如裁剪比和激活统计；OpenAI 在 GPT-4 论文里也详细描述了他们的做法。

**数据要迭代。** 训练数据不是固定不变的。团队会不断跑新的质量分类器，移除新发现的污染数据，加入新的合成数据源。Llama 3 明确提到，他们在训练中途用重新训练的质量分类器过滤整个数据集，并根据结果重新加权后续批次。学术界常说“数据固定”，但生产环境里，数据必须迭代。

**分词怎么搞？** 预分词 15T token 要几周时间，存储超过 100TB。训练时流式分词要求分词器跟上数据加载速度，这需要高度优化的 Rust 分词器。大多数生产环境选择预先计算 token ID，但从分布式文件系统流式读取，而不是直接加载原始文本。

下一节会聊聊这些实践背后的血泪经验。
## 常见踩坑

我亲眼见过 5 个搞砸预训练任务的错误，分享出来给大家避坑。

**1. NCCL collectives 的 bf16/fp32 不匹配问题。** 默认 NCCL all-reduce 在 fp16 模式下求和上千个小梯度值时会悄悄下溢。解决方法很简单：设置 `NCCL_PROTO=Simple` 确保稳定性，同时为 FSDP 设置 `reduce_dtype=fp32`。Llama 3 论文提到，他们在 step 500K 附近遇到不稳定问题，后来靠这个方法才恢复。

**2. RoPE 基础频率与上下文长度不匹配。** RoPE 的基础频率是 $\theta = 10000$。在 32K 上下文长度下，频率混叠，模型学不到长程依赖关系。解决方法有两种：一是按 [bloc97 NTK-aware] 的建议，调整基础频率为 $10000 \times (\text{seqlen}/2048)^{d/(d-2)}$；二是直接用 YaRN [Peng et al., 2024]。我们曾花一周排查 32K 上下文质量下降的问题，最后发现 RoPE 基础频率还停留在 4K 预训练阶段。

**3. 打包序列时未正确处理 loss masking。** Sequence packing 把多个文档拼成一个训练序列。如果不屏蔽跨文档注意力，模型会学到虚假的跨文档关联，导致 MMLU 指标下降 1-3%。正确的做法是生成 `position_ids` 和 `attention_mask`，并在文档边界处重置。

**4. 激活 checkpointing 中使用非确定性操作。** 在 checkpoint-recompute 阶段重新运行前向传播时，必须确保内核是确定性的。一些 FlashAttention 和 GeLU 实现中有非确定性的数值缩减操作。这会导致梯度计算与保存的前向值不一致，引发微妙的质量退化，通常几千步后才会显现。解决方法是在 checkpoint 区域设置 `torch.use_deterministic_algorithms(True)`，或者手动验证内核是否确定性。

**5. Optimizer 状态未分片存储到磁盘。** 对于 70B 参数的模型，AdamW 优化器状态占用 70B × 8 字节 = 560 GB 空间。上千个节点同时写入单个共享文件系统，容易死锁。推荐用 FSDP 的分片 checkpointing 功能（每个 rank 只写自己的分片），并启用并行写入。我们曾因不小心串行化存储，导致一个 checkpoint 花了整整 4 小时才保存完成。
## 研究前沿 2024-2026

Llama 风格的 "过滤 + 去重 + 训练" 方法之后，接下来会有什么新突破？

**大规模数据效率。** [Marion et al., 2023]（When Less is More）证明，用 200B token 的精选子集训练，效果能媲美 1T token 的未过滤数据集。这其实是 "数据的提前停止"。找到最有价值的样本，跳过其他部分。

**Curriculum 驱动的数据混合。** [Albalak et al., 2024] 发现，动态调整数据源比例，根据模型改进实时优化，比固定比例高出 1-2 MMLU。Doremi [Xie et al., 2023] 提出 min-max 优化方法，自动寻找最优权重。

**超越 Chinchilla 的算力最优解。** Mixture-of-Experts 改变了 scaling law。[DeepSeek-AI, 2024] 报告了基于 V3 实验的 MoE scaling law：激活/总参数比、专家数量、专家规模，都复杂依赖于训练算力。社区仍在探索 MoE 特有的规律。

**持续预训练。** 大多数生产模型现在选择更新，而不是从头训练。比如，Llama 3.1 是在 Llama 3 基础上，用额外数据持续预训练，再结合 post-training 得到的。持续预训练成本低得多，但需要精心设计学习率调度，避免灾难性遗忘。
## 小结与下一篇

预训练，七成靠数据工程，三成靠分布式系统。架构选型反而最不重要。先把数据混合调好，彻底去重。训练时间要超过 Chinchilla，这是为了降低推理成本。用 μP 确保超参数能迁移。根据硬件拓扑选 FSDP、TP 和 PP。为大规模场景的故障模式写好防御代码。

下一篇聊 **Post-training**。SFT、DPO、RLHF、RLAIF——我会讲每种方法优化了什么，奖励模型什么时候会崩（它们经常崩）。还有 LoRA 和全量微调的争论，以及如何把基础模型变成客户能用的产品。
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
