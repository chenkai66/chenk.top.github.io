---
title: "时间序列模型（四）：Attention 机制 -- 直接的长程依赖"
date: 2024-10-16 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Attention
categories: 时间序列
series: time-series
lang: zh-CN
mathjax: true
description: "自注意力、多头注意力和位置编码在时间序列中的应用。逐步推导数学公式，附 PyTorch 实现和注意力可视化。"
disableNunjucks: true
series_order: 4
translationKey: "time-series-4"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/illustration_1.jpg)
## 本章要点

- 循环网络处理长程依赖时为什么力不从心，注意力机制又是如何轻松解决的。
- Query / Key / Value 的工作机制、Scaled dot-product 公式，以及为什么需要除以 $\sqrt{d_k}$。
- 两种经典打分函数：**Bahdanau**（加性）和 **Luong**（乘性）。
- 如何将 **Attention 融入 LSTM 编码器/解码器** 来做时间序列预测。
- **多头注意力**在时序任务中的分工：不同头分别捕捉近期、周期、异常等特征。
- $O(n^2)$ 显存瓶颈，稀疏注意力和线性注意力如何突破这一限制。
- 一个完整的 **股价预测案例**，附带注意力权重叠加图解析模型决策。

**前置知识**：RNN/LSTM/GRU 的基本概念（第 2-3 部分）、线性代数基础、PyTorch 基本操作。

---
## 1. 为什么需要 Attention：循环结构的瓶颈

在长度为 $n$ 的循环网络中，两个相距 $k$ 个时间步的位置之间，信息传递需要经过 **$O(k)$ 步**。每一步都把所有信息压缩到一个隐状态向量里，每一步都有可能让梯度逐渐衰减。

但真实世界的时间序列数据可不会配合这种设计：

- 几分钟前的 ECG 异常，比最近 200 个基线样本更重要。
- 今天的电力负荷，通常和上周三同一时间最相似。
- 股价还在对几周前的 **财报事件** 做出反应。

注意力机制提出了一种全新的思路：每个时间步都直接与其他所有时间步建立一条**可学习的连接**。任意两点之间的路径长度缩短为 $O(1)$，而连接的强度（即 *注意力权重*）本身还能解释。

![24 步窗口下的注意力权重热图：明亮的对角线表示近期偏置，偏离对角线的部分显示 12 步周期性，第 5 列的纵向亮带是对该位置异常的持续记忆。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig1_attention_heatmap.png)
*图 1. 因果注意力图天然编码了三种有用的先验知识：近期偏置、周期性和对异常的持续记忆，完全不需要手工设计特征。*

---
## 2. 从头理解 Scaled Dot-Product Attention

把输入序列堆成一个矩阵 $X \in \mathbb{R}^{n \times d}$，每行代表一个时间步。用三个可学习的线性变换生成三种不同的数据视角：

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V,$$

其中 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$。

- **Query** $Q$ -- 这个时间步在关注什么？
- **Key** $K$ -- 这个时间步提供了哪些信息？
- **Value** $V$ -- 这个时间步实际传递了什么？

Query 和 Key 的匹配程度通过点积计算。公式写成矩阵形式就是：

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

### 为什么需要除以 $\sqrt{d_k}$？

如果 $Q$ 和 $K$ 的元素是独立同分布，且方差为 1，那么每个点积 $q_i^\top k_j$ 的方差会变成 $d_k$。当 $d_k$ 很大时，softmax 的输入值会变得非常大，导致 softmax 饱和，梯度几乎全被压到 0。除以 $\sqrt{d_k}$ 把方差重新拉回 1，确保梯度保持健康范围。

这个细节看似不起眼，但如果忽略它，训练会直接失败，损失函数卡在初始值附近动不了。

### 最简实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (batch, seq_len, d). mask: (batch, seq_len, seq_len) 或可广播形状。"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)            # (B, n, n)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                          # 行随机矩阵
    return weights @ V, weights                                  # (B, n, d_v), (B, n, n)
```

整个机制就是两次矩阵乘法夹一个 softmax。模型的表达能力完全依赖于三个学习矩阵 $W^Q, W^K, W^V$。

---
## 3. Bahdanau 和 Luong：两种经典打分函数

Transformer 出现之前，Bahdanau 等人在 2015 年提出了 **加性注意力**，用于序列到序列的翻译任务。随后，Luong 等人在同年提出了 **乘性（点积）注意力** 的变体。即使在今天，当你把注意力机制嵌入 RNN 时，这两种方法依然很有用。

![Bahdanau（加性）和 Luong（乘性）注意力打分函数对比。加性用一个小 MLP，乘性用点积；Transformer 选择了乘性并加入了 $1/\sqrt{d_k}$ 缩放因子。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig2_bahdanau_vs_luong.png)
*图 2. 两种计算 Query-Key 相容度的方法：Bahdanau 使用 MLP，Luong 使用点积。*

| 特性 | Bahdanau（加性） | Luong（乘性） |
|---|---|---|
| 打分公式 | $v^\top \tanh(W_1 h_i + W_2 s_{t-1})$ | $s_t^\top W h_i$ |
| 单次计算成本 | 一次 MLP 前向传播 | 一次点积运算 |
| 参数 | $v, W_1, W_2$ | $W$（通常为单位矩阵） |
| 适用场景 | Query 和 Key 属于不同空间 | Query 和 Key 共享同一空间 |
| 现代 Transformer 中的应用 | 很少使用 | 标准选择（带 $1/\sqrt{d_k}$ 缩放） |

两者的输出都是 softmax 归一化前的分数，最终都通过 softmax + 加权求和完成计算。Transformer 只是选了**更高效**的一种，并加上了缩放因子 $1/\sqrt{d_k}$。
## 4. Self-Attention 在时间序列中的应用

在 seq2seq 模型里，Query 来自解码器，Key 和 Value 来自编码器，这是两个不同的序列。**Self-Attention** 不再区分这些：同一个序列同时充当 $Q$、$K$ 和 $V$。每个时间步都会关注同一窗口内的所有其他时间步。

这正是时间序列预测需要的。假设我要用一个 12 步的窗口预测下一个值，注意力权重会告诉我模型依赖哪些历史时间步来做预测。

![从最新时间步出发的自注意力，弧线粗细表示权重大小；下方柱状图显示相同权重分布。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig3_self_attention_ts.png)
*图 3. 在 $t=11$ 处的预测 Query 不仅关注 $t=10$，还强烈关注 $t=5$（六步前），因为底层信号的周期约为 6。注意力机制自己发现了季节性，无需任何提示。*

### 因果掩码

做预测时，必须防止第 $i$ 步看到未来的信息。标准做法是使用 **因果掩码**：通过一个下三角矩阵，将上三角部分填充为 $-\infty$，这样 softmax 后这些位置的值就会被置零：

```python
def causal_mask(n, device):
    return torch.tril(torch.ones(n, n, device=device)).bool()  # 对角线及以下为 1

scores = scores.masked_fill(~causal_mask(n, scores.device), float("-inf"))
```

这就是预测 Transformer 和分类 Transformer 的唯一区别。
## 5. 多头注意力：专为时间序列设计的“分工”

单头注意力会把所有模式平均到一张图上。多头注意力则将嵌入维度分成 $h$ 块，同时运行 $h$ 个独立的注意力机制，最后拼接起来通过一个线性层：

$$\text{MultiHead}(X) = [\text{head}_1; \dots; \text{head}_h] \, W^O,
\qquad
\text{head}_j = \text{Attention}(X W^{Q}_j, X W^{K}_j, X W^{V}_j).$$

每个头都有自己独立的 $W^Q_j, W^K_j, W^V_j \in \mathbb{R}^{d \times (d/h)}$，可以专注于捕捉不同的模式。在时间序列任务中，训练后通常能看到四种典型的头分工：

![四个注意力头在同一个 18 步窗口上的权重图：分别对应近期、长程趋势、周期 7、以及对 t=4 异常事件的持续记忆。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig6_multihead_for_time.png)
*图 4. 同一个窗口被四个头看出四种结构。多头注意力本质上是一个学得的"时序卷积核集合"。*

| 头 | 学到的模式 | 时序意义 |
|---|---|---|
| Local | 锐利的对角线 | 短期动量 |
| Long-range | 弥漫的三角形 | 慢漂移、状态转换 |
| Periodic | 偏移的对角条纹 | 日 / 周周期 |
| Anomaly | 单列纵向亮带 | "记住第 $k$ 步的尖峰" |

PyTorch 实现：

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        # 投影后 reshape 成 (B, h, n, d_k)
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, n, self.h, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v                                # (B, h, n, d_k)
        out = out.transpose(1, 2).reshape(B, n, -1)      # (B, n, d_model)
        return self.W_o(out), weights
```

**用几个头？** 当 $d_\text{model} = 64\!-\!128$ 时，先从 4 个头开始。训练完成后可视化每个头的权重图：如果多个头几乎一样，就减少头的数量；如果某个头试图同时编码多种不同模式，就增加头的数量。
## 6. 位置编码：把时间加回去

自注意力机制是 **置换不变** 的。输入打乱，输出也会跟着乱。对于时间序列来说，这等于丢掉了数据中最重要的信息。所以，必须显式地加入位置信息。

### 正弦位置编码

原始 Transformer 使用一组几何间隔频率的正弦和余弦函数：

$$PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),
\qquad
PE_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).$$

为什么选这种形式？

- **有界性**：每个值都在 $[-1, 1]$ 范围内，和 $p$ 无关。
- **线性位移等变性**：$PE_{p+\Delta}$ 是 $PE_p$ 的固定线性变换。模型可以通过一次线性投影学会“往前看 7 步”这样的相对偏移。
- **多尺度特性**：低维分量变化慢，表示长程位置；高维分量变化快，表示细粒度位置。

### 时间感知编码（不规则采样）

如果采样间隔不均匀，比如传感器数据或交易数据，应该用 **实际的时间差**，而不是索引。常见实现如下：

```python
def time_features(timestamps, d_model):
    """timestamps: (B, n)，单位秒；返回 (B, n, d_model)。"""
    deltas = timestamps - timestamps[:, :1]              # 距窗口起点的秒数
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    args = deltas.unsqueeze(-1) * freqs                  # (B, n, d_model/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

这段代码可以统一处理 1Hz 的 IoT 数据、不规则的交易 tick 和缺失样本的情况。
## 7. Attention + LSTM：实用的混合架构

纯 Transformer 在处理长序列时确实很强大，但需要大量数据支撑。如果窗口长度在 **50-500 步**之间，混合架构往往是最佳选择。LSTM 负责高效提取局部时序特征，而注意力机制则动态决定每个预测步骤中哪些编码状态更重要。

![LSTM 编码器 + Attention + LSTM 解码器：编码器生成隐藏状态 h1...h5，注意力对它们打分后生成上下文向量，用于解码器预测。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig5_attention_lstm_hybrid.png)
*图 5. 混合架构保留了 LSTM 对局部时序结构的强大归纳偏置，同时利用注意力机制作为内容寻址的“指针”，灵活回溯历史信息。*

```python
class LSTMAttention(nn.Module):
    def __init__(self, n_features, hidden, horizon):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(n_features + hidden, hidden, batch_first=True)
        # Luong 风格乘性注意力
        self.W_a = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden * 2, 1)
        self.horizon = horizon

    def forward(self, x, last_obs):
        H, (h, c) = self.encoder(x)                     # H: (B, n, hidden)
        outs = []
        y_prev = last_obs                                # (B, 1, n_features)
        for _ in range(self.horizon):
            s = h[-1]                                    # (B, hidden)
            scores = (self.W_a(s).unsqueeze(1) * H).sum(-1)         # (B, n)
            alpha = F.softmax(scores, dim=-1)
            ctx = (alpha.unsqueeze(-1) * H).sum(1)                  # (B, hidden)
            dec_in = torch.cat([y_prev, ctx.unsqueeze(1)], dim=-1)
            o, (h, c) = self.decoder(dec_in, (h, c))
            y = self.head(torch.cat([o.squeeze(1), ctx], dim=-1))
            outs.append(y)
            y_prev = y.unsqueeze(1).expand(-1, 1, x.size(-1))
        return torch.cat(outs, dim=1), alpha
```

实际测试表明，这种架构（包括 DA-RNN、双阶段注意力等变体）在 M-competition 风格的基准测试中表现突出，尤其适合短预测视野和数据量有限的场景。
## 8. $O(n^2)$ 的显存瓶颈与解决方法

注意力矩阵包含 $n^2$ 个元素，每个元素都需要计算和存储。以窗口长度为 4096、数据类型为 float32 为例，单头每层每样本就要占用 64 MB 显存。这个瓶颈是真实存在的。

![计算与显存复杂度：RNN 是 O(n)，全注意力是 O(n^2)；稀疏注意力 O(n log n)，线性注意力 O(n)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig4_complexity_vs_length.png)
*图 6. RNN 胜在显存占用低，注意力胜在并行能力强。计算量的交叉点大致在 $n \approx d$。当 $n \gg d$ 时，必须使用次平方复杂度的变体。*

| 变体 | 时间复杂度 | 空间复杂度 | 核心思想 |
|---|---|---|---|
| 全注意力 | $O(n^2 d)$ | $O(n^2)$ | 暴力计算所有对 |
| 稀疏 / 跨步 | $O(n \log n \cdot d)$ | $O(n \log n)$ | 局部窗口 + 膨胀跳跃（Longformer、BigBird） |
| 线性注意力 | $O(n d^2)$ | $O(n d)$ | 用核特征替换 softmax（Linformer、Performer） |
| Informer ProbSparse | $O(n \log n \cdot d)$ | $O(n \log n)$ | 只保留 top-$\log n$ 个 query（第 8 部分会讲） |

大多数时序问题中，$n$ 在几百量级，$d$ 在几十到几百之间，和 RNN 的交叉点对我们有利。只有当标准实现显存不足时，才需要考虑次平方复杂度的变体。
## 9. 案例研究：股票价格预测

为了让整个流程更具体，我用一个合成的股票序列做例子。这个序列包含三个部分：缓慢的趋势、30天的周期波动，以及第60天的财报事件。目标是预测未来10天的价格。模型选用 LSTM+attention，同时对比无注意力机制的基线模型。

![股价预测：LSTM+Attention（橙色）成功捕捉财报后的新趋势；无注意力基线（灰色虚线）低估了变化。下方柱状图显示预测 query 对过去30天的注意力权重，财报当天用红色标出。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig7_stock_attention_app.png)
*图 7. 注意力权重不是黑箱。从图中能直接看出，模型对财报事件和最近一周的数据给予了更多关注。*

有三点值得注意：

1. **财报日权重特别高** -- 注意力机制没有被告知什么是财报事件，但它自己学会了识别这种事件型记忆。
2. **30天周期被保留下来** -- 橙色曲线跟随周期波动，而基线模型则退化成了接近线性的外推。
3. **可解释性是“免费”的** -- 驱动预测的矩阵同时也解释了预测。LSTM 需要事后工具（比如积分梯度、SHAP）来分析；而注意力机制直接输出 softmax 行作为解释。

需要提醒的是：注意力权重 **与重要性相关，但并不等同于因果关系**。在高风险场景下，应该通过扰动测试验证解释（比如将某个关键步骤置零，观察预测的变化），而不是直接把热图当作真相。
## 10. 时序注意力的实用技巧

1. **标准化输入数据**。注意力分数是通过点积计算的，如果不标准化，数值大的特征会占据主导地位。
2. **加入位置编码**。等间隔采样用正弦编码，非等间隔采样用时间感知编码。
3. **预测任务一定要加因果掩码**，训练和推理阶段都不能省略。
4. **从 4 个头开始，$d_\text{model} \in [64, 128]$**。只有当验证损失需要时才考虑扩大规模。
5. **在注意力模块前加 LayerNorm**，对注意力权重和前馈网络块分别加 dropout。
6. **学习率要比 RNN 低**，设置为 $10^{-4}$ 到 $5 \cdot 10^{-4}$，前几百步用 warm-up。
7. **尽早检查注意力头的分布**。如果多个头的模式趋于一致，减少头的数量或者加入多样性正则化。
8. **注意 $O(n^2)$ 的计算瓶颈**。如果需要处理 $n > 1024$ 的情况，直接换成次平方复杂度的变体或者使用 Informer（第 8 部分）。
## 11. 常见问题

- **忘了除以 $\sqrt{d_k}$** -- 训练几步就卡死了。
- **掩码搞错** -- 数据泄露很隐蔽，训练时指标虚高，部署时直接崩。
- **注意力被填充 token 污染** -- 忘了加 padding mask，特殊 token 的信号会扩散到每个位置。
- **把权重当因果解释** -- 权重只是证据，不是结论。
- **窗口长度太短** -- 如果 10 步就能装下所有有用的历史信息，LSTM 很可能比 Transformer 更快，效果也不差。
## 12. 总结

注意力机制用 **直接、按内容寻址的查找** 替代了 RNN 的 **顺序传递和有损信息通道**。数学上就是两次矩阵乘法加一个 softmax，但带来的影响却非常深远：

- 任意两个时间步之间的路径长度缩短为 $O(1)$。
- 训练完全并行化，所有位置一次性计算完成。
- 注意力矩阵本身提供了天然的可解释性。
- 多头注意力为多尺度时序模式提供了一个清晰的抽象。

代价是显存占用达到 $O(n^2)$，并且需要显式注入位置信息。不过对大多数时序问题来说，这点成本完全值得。接下来第 5、6、8 部分会详细介绍 Transformer、TCN 和 Informer 如何进一步拓展这一思想。

> **记忆口诀** -- *Q 提问，K 回答，V 携带；除以 $\sqrt{d_k}$，softmax 转权重，乘 V 得结果；多个头，多个视角。*
## 参考资料

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
3. Luong, Pham, Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP 2015.
4. Qin et al., *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*, IJCAI 2017.
5. Kitaev, Kaiser, Levskaya, *Reformer: The Efficient Transformer*, ICLR 2020.
6. Beltagy, Peters, Cohan, *Longformer: The Long-Document Transformer*, 2020.
7. Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. -- 第 8 部分会详细讲。

---

> 
>
> 本文是时间序列模型系列的**第 4 篇**，共 8 篇。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - [第 2 篇：LSTM —— 门控机制与长期依赖](/zh/time-series/02-lstm/)
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru/)
> - **第 4 篇：Attention 机制 —— 直接的长程依赖**（当前）
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构/)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn/)
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - [第 8 篇：Informer —— 高效长序列预测](/zh/time-series/08-informer长序列预测/)
