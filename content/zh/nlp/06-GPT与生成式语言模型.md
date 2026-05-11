---
title: "自然语言处理（六）：GPT与生成式语言模型"
date: 2025-10-26 09:00:00
tags:
  - NLP
  - GPT
  - 深度学习
  - 语言模型
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从GPT-1到GPT-4：理解自回归语言建模、解码策略（贪心、束搜索、top-k、top-p）、上下文学习，并用HuggingFace构建聊天机器人。"
disableNunjucks: true
series_order: 6
translationKey: "nlp-6"
polished_by_qwen_max: true
---
当你向 ChatGPT 提问，看到一段流畅的多段落回答逐字逐句地生成时，你其实正在见证一个看似简单却极其强大的循环过程：将“到目前为止生成的所有内容”输入 Transformer 解码器，观察它在词汇表上输出的概率分布，从中选择一个词，追加到已生成的内容末尾，然后重复这一过程。这就是自回归语言模型的核心逻辑。真正令人惊叹的并非这个循环本身，而是当我们将循环背后的神经网络扩展至数千亿参数，并用近乎整个互联网的文本数据进行训练时，它所涌现出的强大能力。

如果说 BERT（第五篇）是“理解”的王者，那么 GPT 就是“生成”的霸主。本文将带你完整回顾 GPT 的发展历程，深入剖析自回归解码的运行机制，让你直观感受到不同解码策略的实际影响，最终还会教你如何在自己的笔记本电脑上运行一个功能完备的聊天机器人。


<!-- wanx-hero -->
![自然语言处理（六）：GPT与生成式语言模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/gpt-generative-models/illustration_1.png)
## 你将学到什么
![自然语言处理（六）：GPT与生成式语言模型 — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/gpt-generative-models/illustration_2.png)

- 仅使用解码器的 Transformer 是如何通过“预测下一个 token”实现通用人工智能的
- 因果（掩码）自注意力机制——这一设计点是 GPT 区别于 BERT 的关键所在
- 四种经典的解码策略：贪心搜索、束搜索、top-k 采样和 top-p 采样，以及温度参数在其中的作用
- 上下文学习（零样本/少样本）为何无需更新梯度就能奏效
- 实验中观察到的缩放定律，以及模型能力涌现的奇特现象
- 如何评估生成文本的质量（BLEU、ROUGE、困惑度），以及这些指标在哪些场景下会失效
- 如何基于 GPT-2 和 HuggingFace 构建一个支持多轮对话的聊天机器人

**前置知识**：第 4 篇（Transformer 架构）、第 5 篇（预训练与 BERT）。
## 1. 仅解码器 Transformer
![NLP (6): GPT 和生成式语言模型 —— 图示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/illustration_2.png)

![Decoder-only 架构与因果掩码](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig1_decoder_only_arch.png)

最初的 Transformer（Vaswani 等，2017）由两部分组成：编码器负责处理输入句子，解码器负责生成输出句子。BERT 只保留了编码器部分，而 GPT 则选择了另一条路——只保留了解码器，并且引入了一个关键改动：在每一层自注意力机制中使用**因果掩码**，确保位置 $i$ 只能关注到 $1, 2, \ldots, i$ 的内容，完全屏蔽掉未来的信息。

为什么这个设计如此重要？因为它让训练和推理过程完全一致。在训练时，虽然模型能看到整个句子，但因果掩码会挡住未来的内容，迫使每个位置只能根据过去的信息预测下一个词——这正是推理时的场景。训练与推理过程完全一致，无需额外切换“生成模式”；训练时用于计算损失的前向传播，即等价于推理时生成下一个 token 的过程。

### 前向传播流程详解

假设输入序列为 $x_1, \ldots, x_t$：

1. **嵌入表示**：将每个 token 转换为嵌入向量并加入位置编码：  
   $h^0_i = E_{\text{tok}}(x_i) + E_{\text{pos}}(i)$
   
2. **堆叠 $L$ 层 Transformer 模块**：  
   每一层的计算如下：  
   $$\tilde h = h + \text{MaskedMHA}(\text{LN}(h)), \quad h \leftarrow \tilde h + \text{FFN}(\text{LN}(\tilde h))$$

3. **投影输出**：通过线性变换得到 logits：  
   $z_i = W_o\,h^L_i$，然后通过 softmax 计算下一个 token 的概率分布：  
   $P(x_{i+1}\mid x_{\le i}) = \text{softmax}(z_i)$。

### 因果掩码的具体实现

带掩码的注意力机制本质上还是缩放点积注意力，只是在 softmax 之前对禁止的位置加上 $-\infty$：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right) V,
\quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

经过 softmax 后，$-\infty$ 会被映射为 $0$，因此未来位置对当前结果没有任何影响。右图直观地展示了这一点：每一行代表一个查询位置，每一列代表一个被注意的位置，只有下三角区域（即过去的部分）是有效的。

### 训练目标

GPT 的训练目标是最大化语料库的对数似然，也就是对每个位置的交叉熵进行累加：

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_1, \ldots, x_{i-1})$$

用这一个简单的损失函数，在 TB 级别的文本数据上进行训练，这就是 GPT 的全部训练逻辑。
## 2. 自回归生成，一步步来

![自回归生成：逐个生成 token](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig2_autoregressive_step.png)

在推理阶段，模型每次只生成一个 token。每一步的生成过程可以分为三个关键环节：

1. **前向计算**：将当前序列输入模型，仅提取最后一个位置的 logits。
2. **概率分布转换**：将 logits 转换为词汇表上的概率分布（可选地引入温度参数进行调节）。
3. **选择与追加**：根据概率分布选择下一个 token（可以是确定性选择，也可以通过采样），将其追加到序列末尾，然后重复上述步骤，直到生成结束符 `<eos>` 或达到最大长度限制。

上图的下半部分展示了在输入提示 `"The cat sat on the"` 后，下一个 token 的概率分布情况。可以看到，`mat` 的概率最高，约为 42%，但其他合理的选项（如 `floor`、`couch`、`sofa` 等）也占据了相当一部分概率。模型的输出是每次都一成不变，还是时不时带来一些惊喜，完全取决于你如何从这个分布中进行采样——这部分内容我们将在第 4 节深入探讨。

> **关于速度的实用建议**。如果直接生成 $T$ 个 token，意味着需要执行 $T$ 次前向计算，而每次计算的复杂度会随着序列长度平方增长——这种开销显然是无法接受的。在实际实现中，几乎所有的系统都会采用 **KV cache** 技术：将过去 token 的 keys 和 values 缓存起来并重复利用，这样每次生成新 token 时只需对缓存的向量进行注意力计算即可。这种方法将生成的整体计算复杂度从 $O(T^3)$ 降低到了 $O(T^2)$，也是现代生成系统能够高效运行的核心原因。
## 3. GPT 家族：5 年，参数增长近万倍

![GPT-1 到 GPT-4 的演进](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig4_gpt_evolution.png)

GPT 系列的发展历程可以看作是一个简单想法不断放大的实验：当我们将规模推向极限时，会发生什么？

### GPT-1（2018 年 6 月）——概念验证的起点

- 12 层解码器，**1.17 亿**参数，训练数据来自 BooksCorpus（约 4.5 GB）。
- 方法论：先用语言建模目标进行预训练，然后针对每个下游任务微调一个小型任务头。
- 关键启示：一个通用的 Transformer 解码器，仅通过原始文本的预训练，就能在多种基准测试中超越手工设计的任务专用架构。

### GPT-2（2019 年 2 月）——零样本能力初露锋芒

- **15 亿**参数，训练数据来自 WebText（约 40 GB，抓取自 Reddit 出站链接）。
- 新思路：完全跳过微调。只需在输入提示中描述任务，模型往往就能完成任务。

```text
Translate to French:
English: The cat sat on the mat.
French:
```

尽管 GPT-2 从未见过平行语料，它依然能生成不错的翻译结果。这是因为模型在网页上接触了大量双语文本，学会了“English: … / French: …”这种模式。

### GPT-3（2020 年 5 月）——规模带来质变

- **1750 亿**参数（比 GPT-2 大约 100 倍），训练数据约 570 GB，训练计算量达到 $3.14\times 10^{23}$ FLOPs。
- 核心突破：**少样本上下文学习**。只需在输入提示中提供几个输入-输出样例，模型就能即时掌握任务，完全无需梯度更新。
- 许多能力（如三位数算术、代码补全、多步推理）在 GPT-2 规模时几乎不存在，但在 GPT-3 规模下却突然显现。这就是我们将在第 7 节深入探讨的**涌现**现象。

### GPT-4（2023 年 3 月）——多模态与指令优化的巅峰

- 架构和参数量未公开（传闻是万亿参数级别的混合专家模型）。
- 支持文本和图像两种输入形式。
- 通过 **RLHF**（基于人类反馈的强化学习）大幅提升了安全性、实用性和指令遵循能力。
- 在律师资格考试中达到前 10% 的水平，AP 考试获得满分 5 分，还能处理早期模型无法胜任的长程推理任务。

| 模型 | 参数量 | 训练数据 | 标志性能力 |
|------|--------|----------|------------|
| GPT-1 | 1.17 亿 | 4.5 GB | 预训练 + 微调有效 |
| GPT-2 | 15 亿 | 40 GB | 从提示实现零样本 |
| GPT-3 | 1750 亿 | 570 GB | 少样本上下文学习 |
| GPT-4 | （未公开） | （未公开） | 多模态、强推理、RLHF |

---
## 4. 解码策略
仅通过调整解码策略，就能让生成内容从单调重复、缺乏变化，转变为出人意料又紧扣主题——全程无需重新训练模型。接下来，我们用第 2 节中提到的**同一个**下一 token 分布，直观展示四种经典解码策略的效果。

![贪心、top-k、top-p 和温度对同一分布的影响](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig5_sampling_strategies.png)

### 4.1 贪心解码

每一步都选择概率最高的 token：$x_t = \arg\max_w P(w \mid x_{<t})$。

```python
def greedy_decode(model, tokenizer, prompt, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**优点**：结果确定性强，运行速度快。  
**缺点**：容易陷入退化循环（比如 `"the the the"`），生成内容单调乏味、缺乏惊喜。通常仅用于需要可复现结果的调试场景。

### 4.2 束搜索（Beam Search）

在每一步维护 $k$ 条最优的部分序列，按累计对数概率排序（通常会加入长度惩罚项，避免偏好短序列）。

```python
def beam_search(model, tokenizer, prompt, beam=5, max_new=100, lp=0.6):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    beams = [(ids, 0.0)]
    for _ in range(max_new):
        cands = []
        for seq, score in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id:
                cands.append((seq, score)); continue
            logp = torch.log_softmax(model(seq).logits[:, -1, :], dim=-1)
            top_lp, top_id = logp.topk(beam, dim=-1)
            for p, idx in zip(top_lp[0], top_id[0]):
                new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                cands.append((new_seq, score + p.item()))
        cands.sort(key=lambda x: x[1] / (x[0].size(1) ** lp), reverse=True)
        beams = cands[:beam]
    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)
```

**优点**：生成内容的似然值比贪心解码更高，是机器翻译和文本摘要任务的核心方法。  
**缺点**：在开放式生成中，容易产生平淡无奇、缺乏个性的内容——这就是所谓的“束搜索诅咒”（似然值最高的句子往往最无聊）。

### 4.3 Top-$k$ 采样

只从概率最高的 $k$ 个 token 中进行采样（采样前重新归一化它们的概率）：

```python
def top_k_sample(model, tokenizer, prompt, k=50, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        top_logits, top_ids = logits.topk(k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        nxt = top_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**问题**：$k$ 是固定的。如果模型非常自信（某个 token 占了 95% 的概率），top-$k$ 还是会考虑 $k$ 个候选，可能导致不合理的结果；如果模型在几百个 token 上都不确定，$k=50$ 又显得过于局限。

### 4.4 Top-$p$（核采样）

选出累计概率 $\ge p$ 的最小 token 集合——称为**核（nucleus）**——然后从中采样。核的大小会根据模型的置信度动态调整。

```python
def top_p_sample(model, tokenizer, prompt, p=0.9, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        sorted_logits, sorted_ids = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep = cum <= p
        keep[..., 0] = True               # 至少保留 top-1
        keep[..., 1:] = keep[..., :-1].clone() | (cum[..., :-1] <= p)
        sorted_logits[~keep] = -float("inf")
        nucleus_probs = torch.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(nucleus_probs, 1)
        nxt = sorted_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

上图的 (c) 子图中，核包含 5 个 token，因为前 5 个就覆盖了 85% 的概率。换一个分布，核可能只有 1 个 token（模型非常自信）或 200 个（分布很平）。这种自适应性让 top-$p$ 成为开放式生成任务的首选策略。

### 4.5 温度

温度 $T$ 在 softmax **之前**对 logits 做缩放：

$$P_T(w) = \text{softmax}(z / T)$$

上图的 (d) 子图展示了同一组 logits 在 $T = 0.5$ 和 $T = 1.5$ 下的变化。直观来说，$T$ 控制分布的“尖锐程度”：

- $T \to 0$：分布塌缩到 argmax 对应的 one-hot（等价于贪心解码）。
- $T = 1$：原始分布（不变）。
- $T \to \infty$：分布趋于均匀。

经验法则：**top-$p = 0.9$ 配合 $T = 0.7$–$0.9$** 是大多数对话和创意写作任务的默认配置。

### 策略速查表

| 策略     | 多样性   | 质量       | 速度   | 最佳场景           |
|----------|----------|------------|--------|--------------------|
| 贪心     | 无       | 低         | 快     | 可复现调试         |
| 束搜索   | 低       | 似然值高   | 慢     | 翻译、摘要         |
| Top-$k$  | 中       | 中-高      | 中     | 通用生成           |
| Top-$p$  | 中-高    | 高         | 中     | 对话、故事、创意写作 |
## 5. 缩放定律：涌现前的可预测性
![缩放定律：损失随规模幂律下降](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig3_scaling_laws.png)

2020 年，Kaplan 等人揭示了一个有趣的现象：测试损失会随着三个关键因素——模型参数量 $N$、数据集大小 $D$、训练计算量 $C$——呈**清晰的幂律**下降，但前提是这三个因素中没有一个成为瓶颈。

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

实验结果显示，这些幂律的指数非常小（$\alpha_N \approx 0.076$，$\alpha_C \approx 0.050$）。在对数-对数坐标系下，这种关系表现为一条直线，因此上图中的两个面板都呈现线性趋势。这一发现带来了两个重要启示：

1. **大模型的表现可以提前预测**。通过 $\le 1$ B 参数的小规模实验，我们就能估算出 1750 亿参数模型的损失值。这正是 GPT-3 投资决策的关键依据——团队在投入数百万美元购买 GPU 之前，就已经大致清楚损失曲线会落在什么范围。
2. **损失存在一个无法突破的下限**（图中虚线），即数据本身的熵。无论模型规模如何扩大，都无法跨越这个极限。

到了 2022 年，**Chinchilla** 论文进一步深化了这一规律：在固定计算预算下，GPT-3 的训练量实际上远远不足。最优的训练策略是让每个参数对应大约**20 个 token**。因此，用 1.4 T tokens 训练的 70 B 模型（Chinchilla），在相同计算资源下性能超过了用 300 B tokens 训练的 175 B 模型（GPT-3）。如今，主流的开源模型（如 LLaMA、Mistral、Qwen）都采用了 Chinchilla 式的缩放方法。
## 6. 涌现能力
![涌现能力：随规模出现的尖锐相变](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig6_emergent_capabilities.png)

根据缩放定律，模型的**整体损失**会随着规模的增加而逐渐下降。然而，具体到某些**单项任务**时，情况却并非如此平滑——这些任务在很长一段时间内表现得像“猜谜”一样随机，直到某个临界点突然展现出极高的准确性。

图中清晰地展示了这一现象的特点：情感分类任务（蓝色曲线）从小规模开始就逐步提升；而少样本上下文学习、三位数算术运算以及思维链推理（紫色、橙色和绿色曲线）则在达到某个关键模型规模之前几乎毫无起色，随后性能迅速飙升。Wei 等人（2022）在 **BIG-Bench** 基准测试中总结了 137 个类似的任务，揭示了这种模式的普遍性。

关于这种涌现现象究竟是“真实存在”，还是仅仅是我们测量方式（例如离散任务中的精确匹配准确率）带来的假象，学术界仍有争议。Schaeffer 等人（2023）的研究表明，如果改用连续指标来评估，部分涌现曲线确实会显得更加平滑。但从工程实践的角度来看，结论是明确的：当模型规模较小时，它确实**无法完成**某些任务；但当参数量达到 $10^{10}$ 到 $10^{11}$ 之间时，它却能**出色地完成**这些任务。因此，在制定技术路线图时，这一点必须纳入考虑范围。
## 7. 上下文学习
![少样本上下文学习](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/06-GPT与生成式语言模型/fig7_in_context_learning.png)

GPT 类大模型最令人惊叹的特性之一是：你只需要在输入提示中给出几个示例，就能让模型学会完成新任务。无需反向传播，也无需优化器或参数更新——模型完全通过“阅读”提示内容来适应新任务。

### 零样本与少样本

```text
# 零样本
将以下英文句子翻译成法语：
The bird flew in the sky.
法语：

# 少样本
英译法。
English: The cat sat on the mat. -> French: Le chat s'est assis sur le tapis.
English: The dog ran in the park. -> French: Le chien a couru dans le parc.
English: The bird flew in the sky. -> French:
```

少样本提示为模型提供了一个可以模仿的**模板**。实践表明，对于大多数任务来说，2 到 5 个示例已经足够；超过 8 个示例后，效果提升会趋于平缓。

### 为什么这种方法有效？

这一现象背后的机制仍然是研究的热点问题，目前主流的解释包括以下几种：

- **大规模模式匹配**。在预训练阶段，模型接触过互联网上的大量输入输出格式，而提示内容会激活其中与任务匹配的模板。
- **隐式梯度下降**。Garg 等（2022）的研究表明，在线性回归等简单场景中，Transformer 在前向传播过程中可以通过上下文示例**模拟一步梯度下降**。模型规模越大，模拟的步数可能越多。
- **贝叶斯隐任务推断**。Xie 等（2022）将 ICL 解释为从示例中推断出一个隐藏的“任务变量”，然后基于这个变量生成结果。
- **规模引发的相变效应**。ICL 只有在模型参数量达到数十亿以上时才有效——这与涌现现象的描述是一致的。

### 实用的 Prompt 工程技巧

```python
def few_shot_prompt(task, examples, query):
    parts = [task, ""]
    for inp, out in examples:
        parts.append(f"Input: {inp}\nOutput: {out}\n")
    parts.append(f"Input: {query}\nOutput:")
    return "\n".join(parts)
```

以下是一些经过验证有效的建议：

1. **明确任务描述**。在提示开头用简洁的语言清楚说明任务的目标。
2. **选择具有代表性的示例**。确保示例能够覆盖测试时可能遇到的各种情况。
3. **保持格式一致性**。分隔符、大小写和标签风格需要统一。
4. **推理任务使用思维链**。要求模型“一步步思考”，或者在少样本示例中展示完整的推理过程。这种方法在数学和逻辑类任务上能显著提高准确率。
## 8. 评估生成文本

世上没有完美的评估指标，但了解该用什么近似方法、以及它的局限性在哪里，却是至关重要的。

### BLEU（翻译任务）

BLEU 的核心思想是通过对比生成文本和参考文本的 n-gram 精确率，并结合一个长度惩罚因子来计算得分。公式如下：

$$\text{BLEU} = \text{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

下面是用 Python 实现 BLEU 的代码示例：

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu(generated: str, reference: str) -> float:
    return sentence_bleu([reference.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method1)
```

在机器翻译任务中，BLEU 得分与人工评价的相关性较高；但在开放式生成任务中，这种相关性会显著下降。

### ROUGE（摘要生成）

ROUGE 主要衡量的是生成文本对参考文本内容的覆盖程度，也就是 n-gram 的**召回率**。简单来说，它关注的是参考文本中有多少内容被成功还原到了生成结果中。

以下是使用 ROUGE 的代码实现：

```python
from rouge_score import rouge_scorer
def rouge(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in s.items()}
```

### 困惑度（内在质量指标）

困惑度用来衡量模型对未见过的文本的“意外程度”。数值越低，说明模型对文本的预测能力越强：

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i\mid w_{<i})\right)$$

下面是一个计算困惑度的代码示例：

```python
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**enc, labels=enc.input_ids).loss
    return torch.exp(loss).item()
```

虽然困惑度是训练过程中非常有用的内在指标，但需要明确的是，**低困惑度并不等于高质量的对话系统**。模型可能只是擅长捕捉常见模式，却仍然会生成不准确或虚构的内容。

### 不同任务适合哪些评估指标？

| 任务类型         | 推荐使用的评估方法                     |
|------------------|---------------------------------------|
| 机器翻译         | BLEU、chrF、COMET（学习式指标）       |
| 文本摘要         | ROUGE、BERTScore                      |
| 开放式对话       | 人工评估 + 多样性指标（distinct-n）   |
| 指令遵循         | LLM-as-judge + 人工评估               |
| 语言建模         | 困惑度                                |

对于任何直接面向用户的场景，**人工评估始终是最可靠的标准**。
## 9. 用 GPT-2 搭建一个聊天机器人

GPT-2 的规模适中，既能在普通 CPU 上流畅运行，又能生成足够有趣的内容。HuggingFace 提供的 `transformers` 库几乎屏蔽了所有繁琐的底层实现，让开发者可以专注于核心逻辑。

### 单轮对话

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class ChatBot:
    def __init__(self, name: str = "gpt2") -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def respond(self, user: str, max_new: int = 100,
                temperature: float = 0.8, top_p: float = 0.9) -> str:
        prompt = f"User: {user}\nAssistant:"
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Assistant:")[-1].strip()

print(ChatBot().respond("What is machine learning?"))
```

### 多轮对话（支持滚动历史记录）

```python
class MultiTurnBot(ChatBot):
    def __init__(self, name: str = "gpt2") -> None:
        super().__init__(name)
        self.history: list[str] = []

    def respond(self, user: str, max_turns: int = 5, **gen_kwargs) -> str:
        # 只保留最近几轮对话，确保总长度不超过 1024 token 的上下文限制
        recent = self.history[-2 * max_turns:]
        prompt = "\n".join(recent + [f"User: {user}", "Assistant:"])
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=gen_kwargs.get("max_new", 100),
                do_sample=True,
                temperature=gen_kwargs.get("temperature", 0.8),
                top_p=gen_kwargs.get("top_p", 0.9),
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        reply = self.tokenizer.decode(out[0], skip_special_tokens=True)
        reply = reply.split("Assistant:")[-1].split("User:")[0].strip()
        self.history += [f"User: {user}", f"Assistant: {reply}"]
        return reply
```

### 一行代码快速实现（使用 pipeline API）

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("User: What is deep learning?\nAssistant:",
          max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)[0]["generated_text"])
```

> 需要注意的是，GPT-2 的回答**不会**像 ChatGPT 那样自然，因为它没有经过指令微调（Instruction Tuning）或基于人类反馈的强化学习（RLHF）。如果希望实现类似 ChatGPT 的效果，可以选择一些开源的指令微调模型，例如 `meta-llama/Llama-3-8B-Instruct` 或 `Qwen/Qwen2.5-7B-Instruct`，或者自己用指令数据集对 GPT-2 进行微调。

---
## 10. GPT 和 BERT 的对比，一张表搞定

| 维度         | BERT                                      | GPT                                      |
|--------------|-------------------------------------------|------------------------------------------|
| 架构设计     | 编码器架构，采用双向自注意力机制          | 解码器架构，使用**因果性**自注意力机制   |
| 预训练目标   | 掩码语言模型（Masked LM）+ 下一句预测      | 因果语言建模                             |
| 上下文范围   | 同时关注左侧和右侧的上下文信息            | 仅关注左侧（历史）信息                   |
| 擅长领域     | 分类任务、命名实体识别（NER）、问答系统、检索 | 文本生成、对话系统、代码生成、创意写作   |
| 适配方式     | 针对不同任务添加任务头进行微调            | 使用 Prompt（零样本/少样本学习）或微调   |
| 规模效应     | 参数量超过约 1B 后效果提升趋于平缓        | 随参数规模增加持续提升，新能力逐步显现   |

一个直观的理解方式：**BERT 更像是搜索引擎，而 GPT 则是内容创作者**。如果需要将输入转化为向量表示（如检索或分类），BERT 系列模型会是更好的选择；如果目标是生成高质量的输出文本（如文章、代码或对话），GPT 系列则更为适合。
## 11. 必须坦诚面对的局限性

- **幻觉问题**。当模型遇到它不了解的内容时，默认行为是生成看似流畅但事实错误的文本。应对方法包括：检索增强（第 10 节提到的技术）、工具调用，以及通过校准训练让模型学会拒绝回答。
- **上下文长度限制**。从 GPT-2 的 1024 个 token 到 GPT-3 的 2048，再到 GPT-4 的 32K 和 128K，如今支持 1M token 的模型正逐渐普及。然而，处理长上下文的成本很高，而且在长上下文中，中间部分的信息容易被忽略或遗忘（即“中间迷失”现象）。
- **计算成本高昂**。对于一个拥有 1750 亿参数的模型，单次前向传播就需要数百 GB 的 GPU 显存。在整个模型生命周期中，推理阶段的开销远远超过训练阶段的成本。
- **训练数据中的偏差**。模型会忠实反映其训练数据中的统计规律，包括可能存在的刻板印象和偏见。
- **可控性有限**。如果需要对风格、格式或事实准确性施加严格的约束，则必须依赖额外的技术手段，例如系统提示（system prompts）、约束解码、基于人类反馈的强化学习（RLHF），或者后处理过滤器。
## 12. 核心要点
- GPT 模型的本质非常简单，它是一个**只有解码器的 Transformer，并使用因果掩码**，目标是预测下一个 token。这种简洁性正是它的精髓所在。
- **规模决定一切**：参数量、训练数据和计算资源的结合，会让模型的损失以幂律形式持续下降，同时在某些临界点还会出现能力的突飞猛进。
- **上下文学习**让一个固定的模型通过设计不同的提示词（prompt）就能完成成千上万种任务，绝大多数场景下完全不需要额外微调。
- **解码策略**是灵活可调的关键：对于对话和创意写作，推荐使用 **top-$p$ 采样配合温度值 0.7–0.9**；而对于翻译或需要稳定输出的任务，则更适合用贪心搜索或束搜索。
- **BERT 擅长理解，GPT 擅长生成**。两者相辅相成，覆盖了现代 NLP 的全貌。接下来的第 7 到 12 篇内容，将深入探讨如何高效地运用 GPT 类模型（包括提示工程、微调、RAG、多模态等技术）。