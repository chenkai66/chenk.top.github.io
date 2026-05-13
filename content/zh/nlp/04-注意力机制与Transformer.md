---
title: "自然语言处理（四）：注意力机制与 Transformer"
date: 2025-10-16 09:00:00
tags:
  - 注意力机制
  - NLP
  - Transformer
categories: 自然语言处理
series: nlp
lang: zh
mathjax: true
description: "从 Seq2Seq 的瓶颈到 Attention Is All You Need，建立缩放点积注意力、多头注意力、位置编码和因果掩码的直觉，并用 PyTorch 从零搭一个完整 Transformer。"
disableNunjucks: true
series_order: 4
translationKey: "nlp-4"
polished_by_qwen_max: true
---
2017 年 6 月，Google Brain 和 Google Research 的八位研究者发表了一篇标题相当引人注目的论文：*Attention Is All You Need*。这篇论文提出的 **Transformer** 架构彻底抛弃了循环结构，不再使用 LSTM 或 GRU，也不再需要从左到右逐步扫描句子；相反，序列中的每个 token 都可以通过缩放点积注意力直接“看到”其他所有 token。

这一设计带来了深远影响：它不仅充分发挥了 GPU 的大规模并行计算能力，还从根本上解决了困扰 RNN 数十年的长距离依赖问题；更重要的是，Transformer 成为了后续一系列大模型的基础架构，包括 BERT、GPT、T5、LLaMA、Claude 等几乎所有现代大语言模型。只要扎实理解这篇论文，后续内容本质上都是在此基础上的延伸与变体。

从“带注意力机制的 RNN”到完整的 Transformer，这条路并不算长，但每一步都至关重要。接下来，我们将一步步仔细拆解。

<!-- wanx-hero -->
![自然语言处理（四）：注意力机制与Transformer — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/illustration_1.png)
## 你将学到什么
![自然语言处理（四）：注意力机制与Transformer — 配图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/illustration_2.png)

- 为什么固定长度的上下文向量会让传统的 Seq2Seq 模型在处理长句子时力不从心，而注意力机制又是如何扭转这一局面的
- Bahdanau 和 Luong 注意力：从经典注意力到自注意力的思想桥梁
- Query、 Key 和 Value 的抽象概念解析，缩放点积注意力的核心原理，以及 $\sqrt{d_k}$ 这一缩放因子的作用
- 多头注意力的设计初衷：为何要同时运行多个“视角”并行处理信息
- 正弦位置编码与可学习位置编码的优劣比较
- 因果掩码的作用、残差连接的意义以及 LayerNorm 的工作原理
- 一个完整的 PyTorch 实现的 Transformer 模型，完全从零构建，可以在你的笔记本上直接运行
- BERT、 GPT 和 T5 如何基于相同的架构模块适配不同的任务需求

**前置知识**：第三篇（RNN 和 Seq2Seq），基本线性代数知识（矩阵乘法、 softmax），以及对 PyTorch 的基本使用能力。
## 1. 驱动注意力机制的瓶颈问题

![NLP (4)：注意力机制与Transformer —— 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/illustration_2.png)

回顾一下第三部分提到的基础编解码模型。编码器 RNN 按顺序逐个读取源句子的 token，最终把所有信息压缩成一个固定长度的向量 $c = h_T^{\text{enc}}$。解码器则完全依赖这个向量来生成目标语言序列。

举个例子，假设我们要翻译句子 *"The cat that chased the mouse that ate the cheese was very tired"* 到法语。编码器需要将“猫”“老鼠”“奶酪”“追逐”“吃”“疲惫”及其语法关系统统塞进一个 512 维的向量中，而解码器只能依靠这 512 个数字来还原整句话，且无法再回头查看源句。

这种方法存在两个致命问题：
- **信息容量有限**。一个固定大小的向量无法无损地存储任意长度的序列信息。实验表明，传统的 Seq2Seq 模型在处理超过 30 个词的句子时，BLEU 分数会显著下降。
- **缺乏聚焦能力**。当解码器生成法语单词“猫”时，应关注源句中的“猫”，而不是“奶酪”。然而，静态上下文向量对所有内容一视同仁，无法区分哪些信息更重要。

打个比方：传统方法如同先全文背诵再闭卷翻译；而注意力机制则类似专业译者——始终将原文置于手边，逐词翻译时动态聚焦相关片段。**注意力机制让神经网络学会像专业译者那样工作。**
## 2. Bahdanau 注意力（2015）：每一步都回头看
![翻译对齐的注意力热力图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig1_attention_heatmap.png)

在论文 *Neural Machine Translation by Jointly Learning to Align and Translate* 中， Bahdanau、 Cho 和 Bengio 提出了首个被广泛采用的注意力机制。这个方法的核心思想非常直观：不再使用单一固定的上下文向量，而是在每个解码步骤动态地计算编码器隐藏状态的加权组合。

假设当前解码步骤为 $t$，上一时刻的解码器状态为 $s_{t-1}$，编码器的状态序列为 $h_1, \ldots, h_n$。整个过程可以分为四个关键步骤：

**第一步：计算相关性分数**  
通过一个小的前馈网络，评估每个编码器状态 $h_j$ 与当前解码状态 $s_{t-1}$ 的相关性：
$$e_{tj} = \mathbf{v}^\top \tanh(W_s s_{t-1} + W_h h_j)$$
**第二步：归一化为概率分布**  
利用 softmax 函数将这些分数转化为一个概率分布，确保所有权重之和为 1：
$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{n} \exp(e_{tk})}$$
**第三步：生成上下文向量**  
根据计算出的概率分布 $\alpha_{tj}$，对编码器的所有隐藏状态进行加权求和，得到上下文向量：
$$c_t = \sum_{j=1}^{n} \alpha_{tj}\, h_j$$
**第四步：更新解码器状态**  
RNN 同时接收上下文向量 $c_t$ 和上一时刻的输出 $y_{t-1}$，更新当前解码状态：
$$s_t = \text{RNN}(s_{t-1}, [c_t; y_{t-1}])$$
上面的热力图展示了一个英译法的小例子中，$\alpha_{tj}$ 的具体表现。每一行的概率值加起来等于 1，而亮色区域正好反映了语言学上的对齐关系（例如 Le ↔ The、 chat ↔ cat、 tapis ↔ mat）。值得注意的是，这些对齐关系并非人工标注，而是模型在最小化翻译损失的过程中自主学习得到的。从这一刻起，注意力机制不再仅仅是一个提升性能的工具，而是成为了一个具有语言学意义且高度可解释的研究对象。
## 3. Luong 注意力：更简单的打分函数

几个月后， Luong、 Pham 和 Manning 提出了一种比 Bahdanau 的小型前馈打分器更为简洁的设计方案。这些方法不仅计算高效，还为后续研究奠定了基础：

| 类型   | 打分函数                                          | 特点说明                                |
|--------|---------------------------------------------------|-----------------------------------------|
| 点积   | $e_{tj} = s_t^\top h_j$                            | 计算速度最快，但要求输入维度一致         |
| 通用   | $e_{tj} = s_t^\top W h_j$                          | 引入可学习的双线性变换，支持维度不匹配   |
| 拼接   | $e_{tj} = \mathbf{v}^\top \tanh(W [s_t; h_j])$    | 与 Bahdanau 的方法基本一致               |

其中，点积形式的概念后来成为了 Transformer 中注意力机制的核心思想，两年后被进一步发扬光大。此外， Luong 还提出了 **局部注意力** 的概念，即仅在对齐点周围大小为 $2D+1$ 的窗口内计算注意力，从而显著降低了处理长序列时的计算开销。
## 4. 关键一跃：彻底抛弃循环结构
Bahdanau 和 Luong 的注意力机制虽然建立在 RNN 的基础上，能够加速收敛并提升效果，但 RNN 的本质依然是串行计算：必须先处理第 $t$ 个 token，才能继续处理第 $t+1$ 个。在拥有数千核心的 GPU 上，这种设计无疑是对硬件资源的巨大浪费。

Vaswani 等人提出了一个简洁而深刻的问题：能否完全摒弃循环结构，仅依靠注意力机制？ 如果每个 token 都可以直接与其他所有 token 建立联系，那么我们就能获得两大显著优势：

1. **完全并行化**。所有位置的计算可以通过一次矩阵乘法同时完成。
2. **路径长度恒定**。任意两个 token 之间的交互只需一步操作，无论它们在序列中相隔多远。梯度消失问题从此成为历史。

实现这一目标的核心机制是**自注意力**：不再是解码器关注编码器的状态，而是让序列中的每个 token 直接关注同一序列中的所有其他 token （包括自身）。

### 一个直观的例子

来看这句话："*The animal didn't cross the street because **it** was too tired.*"（那只动物没有过马路，因为**它**太累了。）

要正确表示 **it**，模型需要知道 **it** 指代的是 **animal** 而不是 **street**。在自注意力机制中，**it** 的新表示是句子中所有其他 token 表示的加权和。一个训练得当的注意力头会给 **animal** 分配高权重，给 **street** 分配低权重，并利用这个加权和来精炼 **it** 的含义。

### Query、 Key、 Value：三位一体

对于每个 token 的嵌入 $x_i$，我们学习三组线性投影：
$$q_i = W_Q\, x_i, \qquad k_i = W_K\, x_i, \qquad v_i = W_V\, x_i$$
这三个角色可以类比一次字典查询：

- **Query** $q_i$：我在找什么？——这是当前 token 提出的问题。
- **Key** $k_i$：我有什么？——用来判断这个 token 是否与问题相关。
- **Value** $v_i$：如果真的关注我，你能得到什么信息？

为位置 $i$ 计算注意力时，我们会用 $q_i$ 和所有 $k_j$ 进行打分，再根据分数决定从每个 $v_j$ 中读取多少信息。

### 缩放点积注意力，分四步走

![缩放点积注意力的四个步骤](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig2_qkv_computation.png)

将所有 query、 key、 value 堆叠成矩阵 $Q, K, V \in \mathbb{R}^{n \times d_k}$，公式如下：
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$
结合上图的四个步骤：

1. **打分**：通过 $Q K^\top$ 计算出一个 $n \times n$ 的矩阵，其中第 $(i, j)$ 个元素是点积 $q_i \cdot k_j$，值越大表示越相关。
2. **缩放**：将每个元素除以 $\sqrt{d_k}$，避免数值过大。
3. **softmax**：沿每一行进行 softmax 操作，将缩放后的分数转化为概率分布，总和为 1。第 $i$ 行告诉我们：“要计算位置 $i$ 的输出，请按这个比例混合各个 value。”
4. **加权求和**：与 $V$ 相乘，得到每个位置的新表示。

### 为什么要除以 $\sqrt{d_k}$？

这是面试中最常被问到的细节之一。假设 $q$ 和 $k$ 的每一维独立、均值为 0、方差为 1，那么：
$$\text{Var}(q \cdot k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$
当 $d_k = 64$ 时，点积的标准差会达到 8。如果直接将这种量级的数值输入 softmax，分布会被推得非常尖锐，几乎变成 one-hot 向量。除了最大值的位置，其他位置的梯度基本为 0，训练过程会陷入停滞。

除以 $\sqrt{d_k}$ 将方差拉回 1，确保 softmax 工作在有意义的区间内。这行看似简单的代码，对训练稳定性至关重要。
## 5. 多头注意力
机制

![多头注意力架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig3_multihead_attention.png)

单个注意力操作只能从一种加权的角度去“观察”序列，但语言本身却包含了许多同时存在的复杂结构：比如主谓一致、指代关系、句法依赖、语义相似性以及位置邻近性等。仅靠一个注意力头显然无法全面捕捉这些多样化的特征。

为了解决这个问题，多头注意力机制通过并行运行 $h$ 个独立的注意力操作来实现分工协作。每个注意力头都有自己的参数矩阵进行投影：
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
接着，将所有头的输出拼接起来，并通过一个最终的线性变换层：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$
在原始论文中，模型的维度 $d_{\text{model}} = 512$，并且使用了 $h = 8$ 个头，因此每个头的维度为 $d_k = d_v = 512 / 8 = 64$。虽然总的计算量与使用一个大头时相当，但这种设计让模型能够专注于不同的任务。后续研究发现，不同的头会自动学习捕捉不同的语言特性，例如句法依赖关系和指代消解 [Author Year]。

### 因果掩码

![因果掩码可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig5_causal_mask.png)

在自回归解码器中，训练时需要确保位置 $i$ 不能看到未来的位置 $j > i$ 的信息，否则模型可能会直接“偷看”正确答案。为此，我们引入了**加性掩码**机制：
$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
其中，掩码矩阵 $M$ 的定义是：当 $j \le i$ 时，$M_{ij} = 0$（允许关注）；当 $j > i$ 时，$M_{ij} = -\infty$（禁止关注）。由于 softmax 函数的性质，$-\infty$ 会被映射为精确的 0——图中右侧的上三角区域被完全清零。

正是这个简单的技巧，使得像 GPT 这样的模型能够在一次前向传播中高效地处理整个序列，同时在推理阶段依然表现出逐词生成的行为。
## 6. 位置编码：让顺序回归

![正弦位置编码](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig4_positional_encoding.png)

自注意力机制有一个特性：它是**置换不变的**。换句话说，如果你打乱输入 token 的顺序，输出也会以同样的方式被打乱，而注意力权重本身不会发生任何变化。比如，“猫吃鱼”和“鱼吃猫”在模型看来会产生完全相同的内部表示——这显然不符合我们的预期。

为了解决这个问题，在输入进入第一层之前，我们需要为每个 token 的嵌入添加一个**位置编码** $\text{PE}(\text{pos}) \in \mathbb{R}^{d_{\text{model}}}$，从而让模型能够感知序列中的顺序信息。

### 正弦编码

最初的 Transformer 使用了一种固定的（不可学习的）正弦函数方案来生成位置编码：
$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right), \qquad PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)$$
上图左侧展示的是编码矩阵的热力图，右侧则绘制了几个具体维度的变化曲线。可以看到，低维部分振荡频率较快（捕捉精细的位置信息），而高维部分振荡频率较慢（捕捉粗略的位置信息）。这种设计为每个位置生成了一个独一无二的“指纹”。此外，由于 $\sin$ 和 $\cos$ 函数满足简单的线性关系，模型理论上可以学会诸如“向前三个位置”这样的相对偏移信息。

### 可学习的位置嵌入

另一种常见的方法是将位置视为一个标准的嵌入表，其形状为 $(\text{max\_len}, d_{\text{model}})$。 BERT 和 GPT-2 都采用了这种方式。这种方法实现起来更简单，实际训练效果也略胜一筹。不过，它有一个明显的缺点：无法外推到训练时未见过的更长序列。

### 为什么选择相加而不是拼接？

通过相加的方式，位置编码和内容嵌入共享 $d_{\text{model}}$ 的完整维度，后续的线性投影（$W_Q, W_K, W_V$）可以根据需要自行学习如何分离或结合这两部分信息。如果采用拼接的方式，则要么会增加整体的维度，要么会挤占内容嵌入的表达能力，导致效率下降。

近年来，许多现代大模型已经不再使用原始的正弦编码方案。取而代之的是 **RoPE**（旋转位置编码）、**ALiBi**（注意力分数线性偏置）以及 **NoPE** 等方法，它们主要针对长度外推问题进行了优化，如今已成为生产环境中广泛采用的标准技术。尽管如此，原始的正弦编码方案仍然因其简洁性和可解释性，成为了一个重要的基线参考。
## 7. 完整的 Transformer 架构
![Transformer 编码器和解码器块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig6_transformer_block.png)

Transformer 的架构可以看作是由多个编码器层堆叠在一起，再连接到同样由多层组成的解码器。无论是编码器还是解码器，它们的设计都共享了三个核心理念：**残差连接包裹子层、 LayerNorm 和 dropout**。

### 编码器层

每个编码器层包含两个主要的子模块：

1. **多头自注意力机制**，用于处理输入的源序列。
2. **逐位置前馈网络**，对序列中的每个位置独立进行计算。
$$\text{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2$$
这个前馈网络会先把维度扩展到 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$（通常是 2048），然后再将结果投影回原来的维度。它是模型中参数最密集的部分，也是 token 级非线性特征提取的核心所在。每个子模块都会通过以下方式封装：
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$
残差连接的作用是让梯度能够直接从任意一层传递到之前的任意一层。这种设计在模型深度达到 6 层、 12 层、 24 层甚至 96 层时显得尤为重要，因为它有效缓解了深层网络中的梯度消失问题。

### 解码器层

每个解码器层则包含三个子模块：

1. **掩码多头自注意力机制**，只关注当前已经生成的目标 token （因果关系）。
2. **交叉注意力机制**：其中 query 来自解码器，而 key 和 value 则来自编码器的最后一层输出。这一步是连接源序列和目标序列的关键桥梁。
3. **前馈网络**，其结构与编码器中的完全一致。

上图展示了编码器和解码器的具体结构，并且明确标注了交叉注意力的连接路径。

### 整体流程

以基础版 Transformer 为例（$N = 6$、$d_{\text{model}} = 512$、$h = 8$、$d_{\text{ff}} = 2048$），整个模型大约有 6500 万个参数。而 GPT-3 的改进思路非常直接：把 $N$、$d_{\text{model}}$ 和 $h$ 这些超参数大幅放大，去掉编码器部分，然后用互联网上的海量数据进行训练。
## 8. 用 PyTorch 从零实现
 Transformer

下面的代码实现尽量简洁，每一部分都直接对应前面提到的公式。运行时用 CPU 就够了，重点是帮助理解，而不是用来训练实际模型。

### 缩放点积注意力机制

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """公式：softmax(QK^T / sqrt(d_k)) V。

    输入形状：
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  可广播到 (batch, heads, seq_q, seq_k)，0 表示屏蔽
    输出：
        output:  (batch, heads, seq_q, d_v)
        weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights
```

### 多头注意力机制

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 每个角色用一个大矩阵，后面再切分成多头
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        b, t, _ = x.size()
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        b, _, t, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q, K, V = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.dropout(self._merge(out))), weights
```

### 正弦位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]
```

### 逐位置前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### 编码器层与解码器层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        sa, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(sa))
        ca, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop2(ca))
        x = self.norm3(x + self.drop3(self.ffn(x)))
        return x
```

### 完整的 Transformer 模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_pad_mask(seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size):
        return ~torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    def encode(self, src, src_mask=None):
        x = self.dropout(self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)
```

### 简单测试

```python
torch.manual_seed(0)
model = Transformer(src_vocab=10000, tgt_vocab=10000, num_layers=2)
src = torch.randint(1, 10000, (2, 20))
tgt = torch.randint(1, 10000, (2, 25))

src_mask = Transformer.make_pad_mask(src)
tgt_mask = Transformer.make_causal_mask(tgt.size(1) - 1)

logits = model(src, tgt[:, :-1], src_mask, tgt_mask)
print(logits.shape)  # torch.Size([2, 24, 10000])
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

当 $N=2$ 时，模型大约有 24M 参数；而标准的 base 版本（$N=6$）大约有 65M 参数。
## 9. 自注意力 vs. RNN vs. CNN
![感受野对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-注意力机制与Transformer/fig7_receptive_field.png)

Transformer 为什么能全面取代 RNN 和 CNN，成为序列建模的首选？答案其实就藏在每一层的三个核心指标中：

| 架构              | 每层计算复杂度        | 串行操作次数 | 最大路径长度   |
|-------------------|----------------------|--------------|----------------|
| **自注意力**      | $O(n^2 \cdot d)$     | $O(1)$       | $O(1)$         |
| **RNN （LSTM/GRU）**| $O(n \cdot d^2)$    | $O(n)$       | $O(n)$         |
| **CNN （核 $k$）**  | $O(k \cdot n \cdot d^2)$ | $O(1)$   | $O(\log_k n)$ |

自注意力机制虽然在计算复杂度上达到了 $O(n^2)$，但它带来了两个显著的优势：路径长度恒定，并且可以完全并行化。对于常见的序列长度（$n < 1000$），现代硬件对矩阵运算的高效支持使得这种权衡非常值得。而对于超长序列（比如几万个 token），像 **FlashAttention**、**Longformer**、**Performer** 以及 **Mamba** 风格的状态空间模型等高效变体则更受青睐。
## 10. 三种 Transformer 的工业应用

最初的 Transformer 模型采用了编码器-解码器架构。但随着技术的发展，两种主流变体分别简化了其中的一部分：

| 类别             | 架构         | 预训练目标            | 适用场景                          | 典型模型           |
|------------------|-------------|----------------------|----------------------------------|--------------------|
| **仅编码器**     | 编码器栈     | 掩码语言建模          | 分类、命名实体识别、检索、问答      | BERT、 RoBERTa、 DeBERTa |
| **仅解码器**     | 解码器栈     | 下一个 token 预测（因果）| 文本生成、对话、代码生成、推理     | GPT、 LLaMA、 Claude   |
| **编码器-解码器**| 双栈架构     | 片段还原 / seq2seq    | 翻译、摘要生成、结构化任务         | T5、 BART、 mT5       |

在大模型的竞赛中，仅解码器模型逐渐占据了主导地位。这背后的原因在于，基于原始网页文本进行下一个 token 预测不仅扩展性极强，还能将几乎所有任务统一为文本生成的形式。

### HuggingFace 快速上手指南

在实际项目中，我们几乎不会从零开始训练模型。以下三个代码片段分别展示了每种架构的典型应用场景。

```python
# 仅编码器：用 BERT 实现情感分类
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
inputs = tok("This movie was fantastic!", return_tensors="pt")
logits = model(**inputs).logits
print("正面" if logits.argmax() == 1 else "负面")
```

```python
# 仅解码器：用 GPT-2 生成文本
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tok("Once upon a time", return_tensors="pt")
out = model.generate(
    inputs.input_ids, max_new_tokens=40, top_p=0.95, temperature=0.8, do_sample=True
)
print(tok.decode(out[0], skip_special_tokens=True))
```

```python
# 编码器-解码器：用 T5 实现翻译任务
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
inputs = tok(
    "translate English to French: The cat is sleeping on the mat.",
    return_tensors="pt",
)
out = model.generate(inputs.input_ids, max_new_tokens=40, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```
## 11. 常见问题
**为什么解码器需要掩码，而编码器不需要？**  
编码器的任务是处理整个源句子，因此它需要双向地捕捉全局上下文信息。而解码器则是逐个生成 token，如果在训练时不屏蔽未来的 ground-truth token，模型就会“作弊”，直接利用这些未来的信息。像 BERT 这样的纯编码器模型之所以能够实现双向性，正是因为它们没有因果掩码的限制。

**显存的二次方开销从哪里来？**  
这主要源于注意力机制中的 $n \times n$ 注意力分数矩阵，在没有任何降维操作之前，这个矩阵就已经占据了大量显存。举个例子，当 $n = 4000$ 且有 $h = 16$ 个头时，仅这一部分在 float16 精度下每层就需要超过 500 MB 的显存。 FlashAttention 的巧妙之处在于，它通过分块流式计算 softmax，避免了完整矩阵的生成。

**为什么 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$？**  
这是一个基于经验的选择。 2017 年时， 4 倍的比例被认为是一个合理的折中方案，并逐渐成为默认配置。不过，近年来的一些工作（如 PaLM 和 LLaMA）开始尝试调整这一比例，或者用 SwiGLU 或 GeGLU 替代 ReLU，以获得一些微小的性能提升。

**Pre-LN 还是 Post-LN？**  
原始论文中使用的是 Post-LN （即 LayerNorm 放在残差连接**之后**）。然而，现代实现几乎都采用了 Pre-LN （即 LayerNorm 放在子层**之前**），因为这种方式在深层网络中训练更加稳定，同时减少了对复杂学习率 warmup 的依赖。

**现代大模型还用正弦位置编码吗？**  
基本上已经不用了。目前主流的大模型大多采用 **RoPE**（如 LLaMA、 GPT-NeoX）或 **ALiBi**（如 BLOOM）。这些方法不仅能自然地扩展到更长的上下文，还能通过旋转或加性偏置的方式无缝集成到多头注意力机制中。
## 总结
- 普通的 Seq2Seq 模型在处理长输入时效果不佳，主要原因是单个上下文向量的信息容量有限。**注意力机制**解决了这个问题，它让解码器能够灵活地访问编码器的所有状态。
- **自注意力机制**摒弃了传统的循环结构，每个位置可以在 $O(1)$ 的时间内直接“看到”其他所有位置的信息。
- **缩放点积注意力**的公式非常简洁：$\text{softmax}(QK^\top / \sqrt{d_k}) V$。其中的 $\sqrt{d_k}$ 是确保训练稳定性的关键，少了它可能会导致梯度消失或爆炸。
- **多头注意力**通过并行运行多个小规模的注意力计算，让不同的“头”专注于捕捉输入序列中不同类型的关系。
- **位置编码**为模型补充了纯注意力机制所丢失的序列顺序信息。
- **Transformer 块**的结构很简单：由多头注意力和前馈网络组成，每部分都嵌套在残差连接和 LayerNorm 中。将这样的块堆叠 $N$ 层，整个架构就完成了。
- **BERT、 GPT、 T5** 都是基于这一模板的变体，分别对应仅编码器、仅解码器以及编码器-解码器的实现。

接下来的两篇文章会深入探讨 BERT 和 GPT 的细节。一旦理解了 Transformer 的基本架构，剩下的重点就是如何设计巧妙的预训练目标以及如何扩展模型规模了。