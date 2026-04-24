---
title: "自然语言处理（四）：注意力机制与Transformer"
date: 2025-08-27 09:00:00
tags:
  - 注意力机制
  - NLP
  - Transformer
categories: 自然语言处理
series:
  name: "自然语言处理"
  part: 4
  total: 12
lang: zh-CN
mathjax: true
description: "从 Seq2Seq 的瓶颈到 Attention Is All You Need，建立缩放点积注意力、多头注意力、位置编码和因果掩码的直觉，并用 PyTorch 从零搭一个完整 Transformer。"
disableNunjucks: true
series_order: 4
---

2017 年 6 月，Google 的八位研究者发了一篇标题相当大胆的论文：*Attention Is All You Need*。论文里提出的 **Transformer** 架构干脆把循环结构整个扔掉了——没有 LSTM，没有 GRU，也不再从左到右一个一个地读句子。取而代之，序列里的每个 token 都可以通过一个数学操作直接看到其他所有 token：缩放点积注意力。

这一个设计决定，解锁了 GPU 上的大规模并行训练，顺手解决了困扰 RNN 几十年的长距离依赖问题，并且成为了 BERT、GPT、T5、LLaMA、Claude 以及今天几乎所有大模型的底座。把这一篇读懂，本系列后面的内容基本上就是同一个主题的不同变奏。

从"带注意力的 RNN"到完整的 Transformer 这条路并不长，但每一步都不能跳。我们慢慢走。

## 你将学到什么

- 为什么固定大小的上下文向量会让朴素 Seq2Seq 在长句子上崩溃，注意力机制是怎么救场的
- Bahdanau 和 Luong 注意力——通往自注意力的桥梁
- Query / Key / Value 抽象、缩放点积注意力，以及那个看似不起眼的 $\sqrt{d_k}$ 究竟在干什么
- 多头注意力的直觉：为什么并行跑多个"视角"
- 正弦位置编码 vs. 可学习位置编码
- 因果掩码、残差连接和 LayerNorm
- 一个能在笔记本上跑起来的、从零写的 PyTorch Transformer
- BERT、GPT、T5 怎么用同一套积木拼出三种不同的模型

**前置知识**：第三篇（RNN 与 Seq2Seq），基本的线性代数（矩阵乘法、softmax），以及一点 PyTorch 经验。

---

## 1. 让人头疼的瓶颈问题

回忆一下第三篇里的朴素编解码器：编码器 RNN 一个 token 一个 token 地把源句子读完，把所有信息压成一个固定大小的向量 $c = h_T^{\text{enc}}$；解码器只拿着这一个向量去生成整个目标序列。

设想要把"那只追逐了吃奶酪的老鼠的猫非常疲惫"翻译成英文。编码器必须把猫、老鼠、奶酪、追逐、吃、疲惫，以及它们之间的语法关系全都塞进 512 个数字里。解码器接下来就要靠这 512 个数字把整句话还原出来，**而且不能再回头看一眼源句**。

这件事崩溃的方式有两种，原因不一样：

- **信息容量有限**。一个固定向量装不下任意长度的序列。实测里，朴素 Seq2Seq 的 BLEU 分数在句子超过 30 个词左右就开始急剧下滑。
- **没法选择性地聚焦**。生成英文的 "cat" 时，模型应该看着源句里的"猫"，而不是"奶酪"。可是静态上下文向量对所有源词一视同仁。

打个比方：让你背下一段话，然后合上书凭记忆把它翻译出来；和让你把原文摊在桌上、每写一个译词前都回头看一眼相关的部分。**注意力做的就是后一件事，只不过是在神经网络里做的。**

---

## 2. Bahdanau 注意力（2015）：每一步都回头看一眼

![翻译对齐的注意力热力图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig1_attention_heatmap.png)

Bahdanau、Cho 和 Bengio 在 *Neural Machine Translation by Jointly Learning to Align and Translate* 里提出了第一个被广泛使用的注意力机制。核心想法只有一句话：**别只用一个固定的上下文向量了，每一个解码步骤都重新算一个所有编码器状态的加权和**。

设解码器在第 $t$ 步的隐藏状态是 $s_{t-1}$，编码器隐藏状态是 $h_1, \ldots, h_n$，过程分四步：

**第 1 步——打分**。一个小型前馈网络给每个编码器状态打一个相关性分数：

$$e_{tj} = \mathbf{v}^\top \tanh(W_s s_{t-1} + W_h h_j)$$

**第 2 步——归一化**。沿 $j$ 做 softmax，把分数变成一个加起来等于 1 的概率分布：

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{n} \exp(e_{tk})}$$

**第 3 步——求上下文向量**。它是编码器状态的凸组合：

$$c_t = \sum_{j=1}^{n} \alpha_{tj}\, h_j$$

**第 4 步——更新解码器**：

$$s_t = \text{RNN}(s_{t-1}, [c_t; y_{t-1}])$$

上面那张图就是这些 $\alpha_{tj}$ 在英译法小例子上的样子。每一行加起来都是 1，亮的格子正好对上语言学上的对齐关系（Le ↔ The、chat ↔ cat、tapis ↔ mat）。关键是：**没人告诉模型这些对齐**，它们是模型为了把翻译损失降下去而自己学出来的。从这一刻起，注意力不再只是一个性能技巧，而成了一个有语言学意义、可解释的东西。

---

## 3. Luong 注意力：把打分函数变简单

几个月后，Luong、Pham 和 Manning 提出了几个更简单的打分方式：

| 名称  | 打分函数                                          | 备注                                |
|------|---------------------------------------------------|-------------------------------------|
| 点积  | $e_{tj} = s_t^\top h_j$                            | 最快，要求两边维度相同               |
| 通用  | $e_{tj} = s_t^\top W h_j$                          | 加一个可学习矩阵，处理维度不一致     |
| 拼接  | $e_{tj} = \mathbf{v}^\top \tanh(W [s_t; h_j])$    | 基本就是 Bahdanau 那一套             |

点积那一行就是两年之后 Transformer 全面接管时所采用的打分方式的祖宗。Luong 还顺手提了 **局部注意力**：只在对齐点附近 $2D+1$ 个位置里算注意力，长序列时省一大笔计算。

---

## 4. 关键一跃：彻底扔掉循环结构

Bahdanau 和 Luong 的注意力还是套在 RNN 上面的。它们能让训练收敛得更快、效果更好，但 RNN 仍然把计算串行化了：第 $t$ 步必须算完才能算第 $t+1$ 步。在一块有几千个核的 GPU 上，这是不可饶恕的浪费。

Vaswani 等人问了一个显而易见的问题：**如果只用注意力呢？** 如果每个 token 都能直接关注其他所有 token，我们能拿到两件大礼：

1. **完全并行**。所有位置可以在一次矩阵乘法里同时算出来。
2. **路径长度恒定**。任意两个 token 之间永远只隔一次操作，无论序列多长。100 个时间步上的梯度消失？再见。

实现这个的机制就是**自注意力**：不再是解码器去看编码器状态，而是同一个序列里的每个 token 都去看序列里的所有 token（包括自己）。

### 一个直觉例子

来看："*The animal didn't cross the street because **it** was too tired.*"（那只动物没有过马路，因为**它**太累了。）

要正确地表示 **it**，模型必须知道 **it** 指的是 animal 而不是 street。在自注意力里，**it** 的新表示就是其他所有 token 表示的加权和；一个训练好的注意力头会给 **animal** 高权重、给 **street** 低权重，然后用这个加权和去精炼 **it** 的含义。

### Query、Key、Value：三位一体

对每个 token 的嵌入 $x_i$，我们学三组线性投影：

$$q_i = W_Q\, x_i, \qquad k_i = W_K\, x_i, \qquad v_i = W_V\, x_i$$

这三个角色刚好对应一次字典查询：

- **Query** $q_i$：我在找什么？——这个位置正在问的问题。
- **Key** $k_i$：我里面有什么？——用来判断"这个 token 跟你要找的相关吗"。
- **Value** $v_i$：要是真的关注我，你能拿到什么信息？

给位置 $i$ 算注意力，就是用 $q_i$ 跟所有的 $k_j$ 打分，再决定从每个 $v_j$ 里读多少出来。

### 缩放点积注意力，分四步走

![缩放点积注意力的四个步骤](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig2_qkv_computation.png)

把所有的 query、key、value 堆成矩阵 $Q, K, V \in \mathbb{R}^{n \times d_k}$：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

对照上图四个面板：

1. **打分**：$Q K^\top$ 给出一个 $n \times n$ 的矩阵，第 $(i, j)$ 个元素就是点积 $q_i \cdot k_j$，越大越相关。
2. **缩放**：每个元素都除以 $\sqrt{d_k}$。
3. **softmax**：沿行做 softmax，把缩放后的分数变成一个加起来等于 1 的概率分布。第 $i$ 行告诉你："要算位置 $i$ 的输出，请按这个比例混合各个 value"。
4. **加权求和**：与 $V$ 相乘，得到每个位置的新表示。

### 为什么要除以 $\sqrt{d_k}$？

这是面试里出现频率最高的细节。假设 $q$ 和 $k$ 的每一维都独立、均值 0、方差 1，那么：

$$\text{Var}(q \cdot k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

$d_k = 64$ 时，点积的标准差就到了 8。把这种量级的数喂给 softmax，分布会被推得非常尖锐，几乎变成 one-hot 向量；除了那一个最大位置，其他位置上的梯度基本是 0，**训练直接卡死**。

除以 $\sqrt{d_k}$ 把方差拉回 1，让 softmax 留在有意义的工作区间。一行代码，效果出奇地大。

---

## 5. 多头注意力

![多头注意力架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig3_multihead_attention.png)

单个注意力操作只能给出"一种"加权视角。可是语言里同时存在很多种结构：主谓一致、指代消解、句法依赖、语义相似、位置邻近……一个头根本忙不过来。

多头注意力的做法是并行跑 $h$ 套注意力，每一套有自己的投影矩阵：

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

把所有头的输出拼起来，再过一个线性层：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

原论文里 $d_{\text{model}} = 512$、$h = 8$，所以每个头的 $d_k = d_v = 512 / 8 = 64$。每层的总计算量和只用一个大头基本一样，但模型可以分工：后续的探针实验发现，不同的头确实学会了关注不同的关系，比如句法依赖和指代。

### 因果掩码

![因果掩码可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig5_causal_mask.png)

在自回归解码器里，训练时不能让位置 $i$ 看到位置 $j > i$ 的内容，否则模型直接把答案抄一遍就赢了。我们用一个**加性掩码**来强制：

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

其中 $M_{ij} = 0$（允许，$j \le i$）或 $M_{ij} = -\infty$（禁止，$j > i$）。$-\infty$ 经过 softmax 会变成精确的 0——上图右侧的上三角被干净地抹平了。

正是这个小技巧，让 GPT 类模型能在一次前向传播里把整个序列的 loss 全算完，但推理时表现得仿佛它真的在一个 token 一个 token 地往外吐字。

---

## 6. 位置编码：把"顺序"找回来

![正弦位置编码](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig4_positional_encoding.png)

自注意力是**置换等变**的。把输入 token 打乱，输出也跟着同步打乱，注意力权重本身完全不变。这就意味着"猫吃鱼"和"鱼吃猫"对模型来说是一样的——这显然不行。

解决办法是在第一层之前给每个 token 嵌入加上一个**位置编码** $\text{PE}(\text{pos}) \in \mathbb{R}^{d_{\text{model}}}$。

### 正弦编码

原版 Transformer 用了一个固定（不学习）的正弦方案：

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right), \qquad PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)$$

上图左边是整个编码矩阵的热力图，右边把几个具体维度画成波形：低维振荡很快（精细位置），高维振荡很慢（粗略位置）。这种组合给每个位置一个独一无二的指纹；又因为 $\sin$ 和 $\cos$ 满足简单的线性恒等式，模型理论上可以学到"往后三个位置"这种相对偏移。

### 可学习的位置嵌入

也可以把位置当成一个标准的嵌入表，形状 $(\text{max\_len}, d_{\text{model}})$。BERT 和 GPT-2 都是这么干的——更简单，实际效果常常还略好一点；唯一缺点是训练时没见过的更长的序列就外推不了了。

### 为什么是相加而不是拼接？

相加保留了 $d_{\text{model}}$ 完整的维度给"内容"和"位置"两部分，后续的 $W_Q, W_K, W_V$ 投影自己会学着把它们再分开。拼接要么把维度撑大，要么挤占了内容的容量，两头不讨好。

现在的主流大模型已经基本不用原版正弦编码了：**RoPE**（旋转位置编码）、**ALiBi**（注意力分数线性偏置）、**NoPE** 都是为了解决长度外推这个老大难问题。原版正弦方案现在更多是作为一个清晰的入门基线。

---

## 7. 完整的 Transformer 架构

![Transformer 编码器和解码器块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig6_transformer_block.png)

一个 Transformer 就是一摞编码器层接一摞解码器层。两边都共享三个结构性想法：**子层外面套残差连接、LayerNorm 和 dropout**。

### 编码器层

$N$ 个编码器层每层有两个子层：

1. **多头自注意力**，作用在源序列上。
2. **逐位置前馈网络**，对每个位置独立地做：

$$\text{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2$$

FFN 把维度先扩到 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$（通常 2048），再投影回来。这里住了模型的大部分参数，也是 token 级非线性处理发生的地方。每个子层都被这样包起来：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

残差让梯度能从任意一层直接流回任意一层之前，这对深度 6、12、24、96 的网络来说至关重要。

### 解码器层

$N$ 个解码器层每层有三个子层：

1. **掩码多头自注意力**，看已经生成出来的目标 token（带因果掩码）。
2. **交叉注意力**：query 来自解码器，key 和 value 来自编码器最后一层的输出。这是连接源和目标的桥梁。
3. **前馈网络**，跟编码器里那个完全一样。

上图就是把两边并排画出来，并显式画了交叉注意力的连线。

### 拼起来

```
源 token -> 嵌入 + 位置编码 -> [编码器层 x N] -> 编码器输出
                                                       |
                                                 K, V  |
                                                       v
目标 token -> 嵌入 + 位置编码 -> [解码器层 x N] -> 线性 -> softmax -> 概率
```

base 版（$N = 6$、$d_{\text{model}} = 512$、$h = 8$、$d_{\text{ff}} = 2048$）大约 6500 万参数。GPT-3 不过是把 $N$、$d_{\text{model}}$、$h$ 都放大、扔掉编码器，然后扔到整个互联网上训。

---

## 8. PyTorch 从零实现

下面这套实现刻意写得很短，每一段都对得上前面的公式。CPU 上就能跑，目的是理解，不是真去训练一个翻译模型。

### 缩放点积注意力

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """对应公式 softmax(QK^T / sqrt(d_k)) V。

    形状：
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  可广播到 (batch, heads, seq_q, seq_k)，0 表示禁止
    返回：
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

### 多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 每个角色用一个大矩阵，再 reshape 成多头
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

### 编码器层 / 解码器层

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

### 完整 Transformer

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

### 跑一下

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

$N=2$ 时大约 24M 参数，$N=6$ 的标准 base 版大约 65M。

---

## 9. 自注意力 vs. RNN vs. CNN

![感受野对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/nlp/04-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8ETransformer/fig7_receptive_field.png)

为什么 Transformer 能这么彻底地把 RNN 和 CNN 在序列建模上的位置取代掉？答案藏在这三个数字里（每层）：

| 架构              | 单层计算量            | 串行步数  | 任意两点最大路径长度 |
|-------------------|----------------------|-----------|---------------------|
| **自注意力**      | $O(n^2 \cdot d)$      | $O(1)$    | $O(1)$              |
| **RNN（LSTM/GRU）** | $O(n \cdot d^2)$    | $O(n)$    | $O(n)$              |
| **CNN（核 $k$）**  | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$      |

自注意力的代价是 $O(n^2)$ 的计算，但回报是路径长度恒定加完全并行。对常见的序列长度（$n < 1000$）和喜欢做大矩阵乘法的现代硬件来说，这笔账非常划算。序列长到几万 token 时，**FlashAttention**、**Longformer**、**Performer** 以及 **Mamba** 类的状态空间模型才会变得有吸引力。

---

## 10. 三种工业 Transformer 变体

原版 Transformer 是编码器-解码器的。两个广泛使用的变体各砍掉了一半：

| 家族                | 架构          | 预训练目标             | 擅长                                | 代表                    |
|---------------------|---------------|------------------------|-------------------------------------|-------------------------|
| **仅编码器**        | 编码器栈       | 掩码语言建模            | 分类、NER、检索、QA                  | BERT、RoBERTa、DeBERTa   |
| **仅解码器**        | 解码器栈       | 下一个 token 预测（因果）| 生成、对话、写代码、推理              | GPT、LLaMA、Claude       |
| **编码器-解码器**    | 两个都有       | 片段还原 / seq2seq      | 翻译、摘要、有结构的任务              | T5、BART、mT5            |

仅解码器路线最终在大模型竞赛里胜出，是因为"下一个 token 预测"在原始网页文本上 scale 得极其漂亮，而且能把几乎所有任务都统一成"文本生成"。

### HuggingFace 三个最小例子

实际项目里基本不会从零训练，下面三段代码各自展示一个家族的味道。

```python
# 仅编码器：BERT 做分类
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
# 仅解码器：GPT-2 做生成
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
# 编码器-解码器：T5 做翻译
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

---

## 11. 常见问题

**为什么只有解码器需要掩码，编码器不用？**
编码器看的是完整源句，本来就该双向地看；解码器一个 token 一个 token 地生成，训练时如果不挡住未来，模型就直接抄答案了。BERT 这种仅编码器模型是双向的，正是因为它压根没有因果掩码。

**$O(n^2)$ 的显存到底花在哪儿？**
就花在那个 $n \times n$ 的注意力分数矩阵上，还没做任何聚合之前。$n = 4000$、$h = 16$ 个头、float16 精度，光这一个缓冲区单层就要 500 MB 以上。FlashAttention 的核心思想就是**永远不显式地把这个矩阵物化出来**，而是分块流式地算 softmax。

**为什么 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$？**
经验值。2017 年觉得 4 倍是个不错的选择，就一路沿用下来。最近的工作（PaLM、LLaMA）有时会改这个比例，或者把 ReLU 换成 SwiGLU、GeGLU 拿一点边际收益。

**Pre-LN 还是 Post-LN？**
原论文是 Post-LN（LayerNorm 在残差**之后**）。现在的实现几乎清一色 Pre-LN（LayerNorm 在子层**之前**），深层网络下训练稳得多，也不太需要那种精心设计的学习率 warmup。

**今天的大模型还在用正弦位置编码吗？**
基本不用了。主流大模型用 **RoPE**（LLaMA、GPT-NeoX）或 **ALiBi**（BLOOM），因为它们能更自然地外推到更长的上下文，并且通过旋转或加性偏置自然地嵌进多头注意力里。

---

## 12. 核心要点

- 朴素 Seq2Seq 在长输入上崩溃，是因为单一上下文向量太小。**注意力**让解码器动态访问每一个编码器状态。
- **自注意力**把循环结构去掉了：任意两个位置之间只隔 $O(1)$ 步操作。
- **缩放点积注意力**就一个公式：$\text{softmax}(QK^\top / \sqrt{d_k}) V$。那个 $\sqrt{d_k}$ 是训练能不能稳住和梯度直接崩掉的分界线。
- **多头注意力**并行跑多个小注意力，让不同的头分工去抓不同关系。
- **位置编码**把纯注意力丢掉的顺序信息找回来。
- **Transformer 块** = 多头注意力 + 前馈网络，每个外面都套残差和 LayerNorm。摞 $N$ 层。就这么多。
- **BERT、GPT、T5** 分别是这套模板的"仅编码器""仅解码器""编码器-解码器"三种特化形式。

接下来两篇会深入讲 BERT 和 GPT——一旦架构看明白了，剩下的主要就是聪明的预训练目标加规模。

---

## 系列导航

| 部分 | 主题 | 链接 |
|------|------|------|
| 1 | NLP入门与文本预处理 | [<-- 阅读](/zh/自然语言处理-一-NLP入门与文本预处理/) |
| 2 | 词向量与语言模型 | [<-- 阅读](/zh/自然语言处理-二-词向量与语言模型/) |
| 3 | RNN与序列建模 | [<-- 上一篇](/zh/自然语言处理-三-RNN与序列建模/) |
| **4** | **注意力机制与Transformer（本文）** | |
| 5 | BERT与预训练模型 | [下一篇 -->](/zh/自然语言处理-五-BERT与预训练模型/) |
| 6 | GPT与生成式语言模型 | [阅读 -->](/zh/自然语言处理-六-GPT与生成式语言模型/) |
