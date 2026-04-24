---
title: "Prefix-Tuning：为生成任务优化连续提示"
tags:
  - PEFT
  - 参数高效微调
  - Prefix-Tuning
categories: 论文笔记
lang: zh-CN
mathjax: true
description: "Prefix-Tuning 冻结整个语言模型，只学习一组注入到注意力层的连续向量来引导生成。本文从注意力公式、重参数化、KV cache 机制到 GPT-2 上的实验，把这套方法和 Adapter、Prompt Tuning、LoRA 的边界讲清楚。"
---

把 GPT-2 微调到一个具体任务上，意味着要再多存一份 1.5B 参数的权重。换十几个任务，存储和上线成本就能直接劝退一个团队，更别提"一份基模 + 多任务共享"这种工程上很想要的架构。**Prefix-Tuning**（Li & Liang, 2021）走了一条相反的路：模型权重一个不动，只学一小段连续向量——也就是论文里所说的"前缀"——在每一层注意力里被当作"已经在那里的上下文"喂进去。模型本身没变，换一段前缀就等于换了一种"任务人格"。

## 你将学到什么

- 前缀到底是什么、为什么把它注入到每一层的 K/V 比只在输入端拼软提示要强
- 完整的注意力公式、参数量推导，以及那个用于稳定训练的重参数化 MLP 的作用
- 在自回归解码时，前缀如何与 KV cache 协同工作
- Prefix-Tuning 与 Prompt Tuning、Adapter、LoRA 各自的边界
- 关于前缀长度、多任务部署、常见踩坑的工程经验

## 前置知识

- Transformer 注意力的 Q/K/V 投影，以及多头结构
- PEFT（参数高效微调）想解决什么问题
- 自回归语言模型的基本概念

---

## 1. 动机：在不动权重的前提下让大模型适配新任务

全量微调要更新模型的全部参数。这件事在算力上贵、在存储上更贵——任务一多，你会发现绝大部分硬盘其实都是同一个基模的近似副本。PEFT 的核心目标可以浓缩成三条：

- 减少可训练参数（也就减少了优化器状态的显存占用）
- 把每个任务的"补丁"从 GB 级压到 MB 级
- 让一份冻结的基模能同时服务很多任务

Prefix-Tuning 是最早专门针对**生成任务**（table-to-text、摘要、对话）设计的 PEFT 方法之一，到今天它仍然是讲 PEFT 时一个非常干净的入门基线。

## 2. Prefix-Tuning 里的"前缀"到底是什么

![Prefix-Tuning 架构：可学习的 K/V 前缀注入到每一层冻结的注意力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/prefix-tuning-optimizing-continuous-prompts-for-generation/fig1_architecture.png)

Transformer 是堆叠的"注意力 + MLP"块。每一层的注意力把隐藏状态投影成查询、键、值 $Q, K, V \in \mathbb{R}^{n \times d}$，再算

$$
\text{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V .
$$

如果想给模型添一段"看起来像上下文"的可学习参数，自然有两个位置：

- **输入前缀**。在词嵌入前面拼一段可学习的向量，这是 *Prompt Tuning*（Lester et al., 2021）的做法。只有第一层直接看到它，深层只能间接受影响。
- **逐层 K/V 前缀**。在每一层都把可学习的键/值向量直接拼到 $K$ 和 $V$ 前面。这就是 *Prefix-Tuning*。每一层都拥有自己专属的"旋钮"。

第二种方案表达力更强：它在每一层都加了容量，而不是只在最入口处。在生成任务上，这种"深进去"的注入也确实更管用。

## 3. 注意力前缀的公式

设第 $\ell$ 层的可学习前缀矩阵为 $P^{(\ell)} \in \mathbb{R}^{L_{\text{prefix}} \times 2 d}$，沿最后一维劈成键和值两半：

$$
P^{(\ell)} = \big[\, P^{(\ell)}_K \;\Vert\; P^{(\ell)}_V \,\big],
\quad P^{(\ell)}_K, P^{(\ell)}_V \in \mathbb{R}^{L_{\text{prefix}} \times d}.
$$

在每次前向传播里，把它和冻结模型本来产生的真 K/V 拼起来：

$$
\tilde{K}^{(\ell)} = \big[\, P^{(\ell)}_K \;;\; K^{(\ell)} \,\big],
\qquad
\tilde{V}^{(\ell)} = \big[\, P^{(\ell)}_V \;;\; V^{(\ell)} \,\big],
$$

再把标准注意力替换成

$$
\text{Attn}\!\big(Q^{(\ell)},\, \tilde{K}^{(\ell)},\, \tilde{V}^{(\ell)}\big).
$$

有两点细节值得点出来：

1. 查询 $Q^{(\ell)}$ 没有被扩。前缀只出现在注意力的"被看"那一侧——别的位置可以读它，但它不会自己产出输出 token。
2. 因为每一层的 K 和 V 都被加长了，每一个被生成的 token 在每一层都能"看到"前缀。整段前缀就像一条贯穿整个 Transformer 栈、可微的工作记忆。

## 4. 参数量：为什么说它"轻"

设层数为 $L$，隐藏维度为 $d$，前缀长度为 $L_{\text{prefix}}$，可训练参数量是

$$
| \theta_{\text{prefix}} | = 2 \cdot L \cdot d \cdot L_{\text{prefix}} .
$$

代入 GPT-2 medium（$L = 24$，$d = 1024$）和典型的 $L_{\text{prefix}} = 10$，大约是 **0.5 M** 参数，对比模型本身的 **355 M** 参数，差不多缩小了 700 倍。换成 GPT-2 XL，比例还会更夸张。

每个任务的"权重"从 GB 级掉到了几百 KB。这是工程上的关键卖点：你可以发一份基模加一个装满小前缀文件的目录，每个文件对应一个任务。

## 5. 重参数化为什么有用

如果你直接把 $P$ 当成参数去训，会发现训练相当脆弱——尤其当前缀稍微长一点，loss 容易抖、最终质量也低于这个参数量"应有"的水平。Li & Liang 的处理是给前缀套一个小 MLP 做**重参数化**：

$$
P^{(\ell)}_K, P^{(\ell)}_V = \text{MLP}_\phi\!\big( P'^{(\ell)} \big),
$$

其中 $P'^{(\ell)} \in \mathbb{R}^{L_{\text{prefix}} \times d'}$ 是一个更小维度的潜在前缀，$d' \ll d$。这步为什么有用：

- MLP 把优化曲面磨光滑了。梯度回传到一个低维潜变量上，比直接打在又宽又高的 $L_{\text{prefix}} \times 2d$ 矩阵上要好优化。
- 非线性提供了纯靠前缀矩阵无法表达的额外容量——而且不需要解冻骨干网络。
- 一个共享的小 MLP 让你在层之间获得部分参数共享，同时仍能输出每层不同的前缀。

训练完之后，这个 MLP 可以**直接扔掉**：只保留实例化好的 $P^{(\ell)}_K, P^{(\ell)}_V$ 张量供推理用。也就是说"额外容量"在部署阶段是免费的。

## 6. 一图比清楚：三种适配冻结大模型的方式

![全量微调、Prefix-Tuning 与 Prompt Tuning 的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/prefix-tuning-optimizing-continuous-prompts-for-generation/fig2_method_comparison.png)

这三种方法可以放到同一条谱系上看：

- **全量微调**：什么都更新。容量最大、成本最贵、模型无法共享。
- **Prefix-Tuning**：只更新逐层的 K/V 张量。约 0.1% 的参数量，在生成任务上很强，骨干模型保持冻结。
- **Prompt Tuning**：只更新输入端的软提示。约 0.01% 的参数量，实现最简单；在小模型上通常弱于 Prefix-Tuning，但模型规模大了之后差距会显著缩小（Lester et al., 2021）。

LoRA（Hu et al., 2021）是结构上很不一样的第四种思路：它不动激活，而是在权重矩阵上加一个低秩增量 $W \mapsto W + \alpha BA$，其中 $A, B$ 都是低秩矩阵。LoRA 后来在指令微调场景里几乎成了默认选择，主要因为：（a）推理时它可以**合并进权重**，几乎零额外延迟；（b）"秩"这个超参对大多数用户来说比"前缀长度"更直观。Prefix-Tuning 仍然有不可替代的位置——当你必须保证基模权重一比特都不动、当你要在解码阶段按请求热切换 adapter、或者当任务本身就是"靠上下文驱动"的生成型任务时，它是更自然的选择。

## 7. 前缀长度：是甜点而不是越大越好

![前缀长度与质量、参数成本的关系](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/prefix-tuning-optimizing-continuous-prompts-for-generation/fig3_prefix_length_sweep.png)

经验上，E2E table-to-text 上的 BLEU 从 $L_{\text{prefix}} = 1$ 涨到 $L_{\text{prefix}} \approx 10$ 时增长很快，之后基本拉平，长到 200 以上反而会**轻微回落**——这更像是优化层面的问题而非容量瓶颈，但结论是一样的：单纯把前缀加长不是免费午餐。一个还算实用的起点：

- **分类型或短生成**：$L_{\text{prefix}} \in [5, 10]$
- **长文本生成、少样本场景**：$L_{\text{prefix}} \in [10, 20]$
- **超过 100**：很少能赚回它带来的参数和推理序列长度成本

## 8. 自回归解码时 KV cache 的机制

![解码过程中前缀 K/V 在 KV cache 中的位置](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/prefix-tuning-optimizing-continuous-prompts-for-generation/fig4_kv_cache_prepend.png)

Prefix-Tuning 与自回归解码标配的 KV cache 配合得相当自然。每一层都维护一份缓存，里面是模型已经"看过"的所有 $(K, V)$。加上 Prefix-Tuning 之后：

1. **$t = 0$ 时**，缓存先**用前缀 K/V 初始化**（这些张量在请求加载时一次性从硬盘读出来），然后把 prompt 编码成的 K/V 也追加进去。
2. **每一个解码步 $t$**，新生成 token 的 K/V 像普通解码一样追加到缓存末尾。
3. **每一步的注意力**都跑在完整的扩展缓存上：前缀 + prompt + 已经生成的内容。

所以前缀对每步注意力多带来的开销是一个加性的 $O(L_{\text{prefix}})$，对每个活跃请求多占的内存是固定的 $2 \cdot L \cdot d \cdot L_{\text{prefix}}$ 个浮点数。$L_{\text{prefix}} = 10$ 时基本可以忽略；如果你真用了 200，那就相当于解码时多带了 200 个 token 的上下文，这个成本是值得测一下的。

这套机制顺便让"按请求切换任务"变得很便宜：换一份前缀文件、清掉缓存，同一个基模就立刻表现得像另一种微调结果。

## 9. 应用：GPT-2 上的两个生成任务

![Prefix-Tuning vs 全量微调 vs Adapter 在 E2E 和 XSum 上的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/prefix-tuning-optimizing-continuous-prompts-for-generation/fig5_gpt2_application.png)

原论文跑的两个核心实验是 GPT-2 medium 上的：

- **E2E**（table-to-text）：Prefix-Tuning 用 0.1% 的参数量做到与全量微调相当甚至略优的 BLEU。
- **XSum**（抽象摘要）：随着数据增多，全量微调慢慢追上来；但在**少数据场景**下 Prefix-Tuning 优势明显——5% 训练数据以下时，"你只能拧一个低维上下文旋钮"这个归纳偏置反而压住了全量微调常见的过拟合。

少数据上的这个观察后来被反复印证。LoRA 在指令微调上的少数据表现也是同样的故事：当任务数据不多时，把假设空间收窄到一个低维 adapter 本身就是一种正则，而这个正则的收益通常比全量微调多出来的容量更值钱。

## 10. 工程实践注记

**前缀加在哪里。** 在 decoder-only 模型里，每一层的 self-attention 都加前缀是默认配方。在 encoder-decoder 模型里你得做选择：encoder self-attention、decoder self-attention、decoder cross-attention，或者三个一起。论文在条件生成任务上发现 cross-attention 前缀最关键。

**初始化技巧。** 随机初始化能用，但收敛慢。用一段**真实的提示句**（比如 "summarize the following article:"）的词嵌入去初始化前缀，能在同样的任务上明显加快收敛。一个小动作，但效果是真的。

**多任务部署。** 一个任务存一份前缀张量。30 个 GPT-2 medium 任务的前缀加起来也就是几 MB，对比 30 份全量微调的 ~10 GB 是数量级的差距。线上切任务的逻辑就是"重新加载小前缀文件"，不用重载模型。

**常见踩坑。**

- *训练了很久毫无起色*：通常是 $L_{\text{prefix}}$ 太小、学习率太低，或者在需要重参数化的任务上漏掉了 MLP。
- *loss 看起来稳但生成是乱码*：检查拼接维度是不是错了，以及前缀是否**只**出现在 K/V 一侧。一个出乎意料常见的 bug 是把它也拼到了 $Q$ 上。
- *前缀很长时质量崩盘*：基本上是优化问题。降学习率、加强重参数化，或者干脆缩短前缀。

## 11. 什么时候选 Prefix-Tuning

适合 Prefix-Tuning 的场景：

- 必须保证基模权重逐比特不变（合规或共享需求）。
- 一份基模背后挂多个任务，需要按请求热切换 adapter。
- 任务本身偏生成、且数据预算中等。

更适合用 LoRA 的场景：

- 追求极致的推理延迟（LoRA 可以**合并**进权重）。
- 需要在指令微调那种宽分布数据上的整体表现。
- 偏好用"秩"这个超参来思考问题。

更适合做全量微调的场景：

- 数据真的很多、并且对每一个点的指标都很在意。
- 不在乎每个任务都得多存一份完整模型。

## 一句话收束

Prefix-Tuning 把"适配新任务"重新表述为：**学习一段贯穿全栈、按层注入的小型上下文记忆**，而模型本身保持冻结。结构干净，参数量在亚百分点级别，少数据上的归纳偏置实打实有用。即便后来 LoRA 在 PEFT 实战里几乎一统江湖，Prefix-Tuning 仍然是讲清楚"通过给模型喂可学上下文来适配它"这个想法时最干净的心智模型，也仍然是一些真实部署场景下更自然的选择。

## 参考资料

- Li, X. L., & Liang, P. (2021). [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190). *ACL 2021*. 原始论文。
- Lester, B., Al-Rfou, R., & Constant, N. (2021). [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691). *EMNLP 2021*. Prompt Tuning 的代表作。
- Houlsby, N., et al. (2019). [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751). *ICML 2019*. Adapter 的奠基性论文。
- Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022*. PEFT 实战上的现任默认选项。
- He, J., et al. (2022). [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366). *ICLR 2022*. 把 Adapter / Prefix / LoRA 放进一个统一框架。
- Yang, Z., et al. (2022). [Robust Prefix-Tuning for Text Classification](https://thumtblog.github.io/2022/04/05/robust-prefix-tuning/). 关于前缀对对抗扰动鲁棒性的分析。
