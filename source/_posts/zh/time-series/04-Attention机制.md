---
title: "时间序列模型（四）：Attention 机制 -- 直接的长程依赖"
date: 2024-11-04 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Attention
categories: 时间序列
series:
  name: "时间序列模型"
  part: 4
  total: 8
lang: zh-CN
mathjax: true
description: "自注意力、多头注意力和位置编码在时间序列中的应用。逐步推导数学公式，附 PyTorch 实现和注意力可视化。"
---

> **系列**：时间序列模型 -- 第 4 部分，共 8 部分
> [<-- 上一篇：GRU](/zh/时间序列模型-三-GRU/) | [下一篇：Transformer -->](/zh/时间序列模型-五-Transformer架构/)

## 本章要点

- 循环网络在长程依赖上为什么吃亏，注意力如何一击破解。
- Query / Key / Value 机制、Scaled dot-product 公式，以及为什么必须除以 $\sqrt{d_k}$。
- 两种经典打分函数：**Bahdanau**（加性）和 **Luong**（乘性）。
- 如何把 **Attention 接到 LSTM 编码器/解码器** 上做时间序列预测。
- **多头注意力**在时序场景下的"四种典型分工"：近期、长程、周期、异常。
- $O(n^2)$ 显存墙，以及稀疏 / 线性注意力如何绕过去。
- 一个完整的 **股价预测案例**，并用注意力权重叠加图解释模型决策。

**前置**：RNN/LSTM/GRU 的基本概念（第 2-3 部分）、线性代数、PyTorch 基本操作。

---

## 1. 为什么需要 Attention：循环结构的瓶颈

在长度为 $n$ 的循环网络里，相距 $k$ 个时间步的两个位置之间，信息要走 **$O(k)$ 步**。每一步都要把所有有用的信号塞进同一个隐状态向量里，每一步都有梯度衰减的风险。

而真实的时间序列基本不配合这种几何：

- 几分钟前出现的 ECG 异常，比刚刚 200 个稳态样本更重要；
- 今天的电力负荷，往往最像 **上周同一时刻**；
- 股价对几周前的 **财报事件** 仍然在反应。

注意力提出了一种完全不同的几何：**任意两个时间步之间都有一条直连的、可学习的边**。任意两点的路径长度变成 $O(1)$，而边的强度（即 *注意力权重*）本身就具备可解释性。

![24 步窗口下的注意力权重热图：明亮的对角线是近期偏置，平行偏移的对角线是 12 步周期，第 5 列的纵向亮带是对该位置异常的持续记忆。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig1_attention_heatmap.png)
*图 1. 一张因果注意力图就同时编码了三种有用先验：近期、周期、对异常的持续记忆 -- 完全无需手工特征。*

---

## 2. 从零推导 Scaled Dot-Product Attention

把输入序列堆成矩阵 $X \in \mathbb{R}^{n \times d}$，每一行对应一个时间步。三个可学习的线性映射给出同一份数据的三种"视角"：

$$
Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V,
$$

其中 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$。

- **Query** $Q$ -- "这个时间步在找什么？"
- **Key** $K$ -- "这个时间步广告什么？"
- **Value** $V$ -- "这个时间步真正携带的内容是什么？"

Query 与 Key 的相容度用点积衡量。整体写成矩阵形式：

$$
\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

### 为什么要除以 $\sqrt{d_k}$？

如果 $Q, K$ 的元素是均值 0、方差 1 的独立分布，那么单个点积 $q_i^\top k_j$ 的方差就是 $d_k$。当 $d_k$ 很大时，softmax 的输入幅度也变大，softmax 进入饱和区，绝大多数位置的梯度会被压成接近 0。除以 $\sqrt{d_k}$ 把方差拉回 1，让梯度回到健康量级。

这是一个看似"工程小细节"，但忘了这一步训练就直接失败 -- 损失会卡在初始值附近不动。

### 最小实现

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

整个机制就是 **两次矩阵乘法夹一次 softmax**。模型的全部表达能力都藏在三个学习矩阵 $W^Q, W^K, W^V$ 里。

---

## 3. Bahdanau vs Luong：两种经典打分函数

在 Transformer 出现之前，Bahdanau et al. (2015) 在机器翻译里提出了 **加性注意力**，Luong et al. (2015) 紧接着提出了 **乘性（点积）** 变体。在 RNN+Attention 的混合架构里，这两种打分函数仍然非常常用。

![Bahdanau（加性）vs Luong（乘性）打分函数。加性用一个小 MLP，乘性用一次点积；Transformer 选择了乘性 + 1/sqrt(d_k) 缩放。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig2_bahdanau_vs_luong.png)
*图 2. 两种 Query-Key 相容度打分方式：Bahdanau 用 MLP，Luong 用点积。*

| 性质 | Bahdanau（加性） | Luong（乘性） |
|---|---|---|
| 打分函数 | $v^\top \tanh(W_1 h_i + W_2 s_{t-1})$ | $s_t^\top W h_i$ |
| 单对成本 | 一次 MLP 前向 | 一次点积 |
| 参数 | $v, W_1, W_2$ | $W$（甚至可省） |
| 适用场景 | Q/K 维度不同或语义不同空间 | Q/K 同空间 |
| 现代 Transformer | 几乎不用 | 标准选项（配 $1/\sqrt{d_k}$） |

两者的下游流程完全相同：softmax 归一化 + 加权求和。Transformer 只是选了**便宜**的那一个，并加上了缩放因子。

---

## 4. Self-Attention 在时间序列上的直觉

在 seq2seq 框架里，Query 来自解码器、Key/Value 来自编码器，这是两个不同的序列。**自注意力（Self-Attention）** 抹掉这个区别：同一个序列同时充当 $Q, K, V$ -- 每个时间步都看自己窗口里的所有其他步。

对时间序列来说这正好是我们要的。给定一个 12 步的预测窗口，从"现在"出发的注意力权重直接告诉我们：模型在依赖哪些历史步做预测？

![从最新时间步出发的自注意力，弧的粗细对应权重大小；下方柱状图显示同一行权重。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig3_self_attention_ts.png)
*图 3. $t=11$ 处的预测 Query 不只盯着 $t=10$，它对 $t=5$（六步前）也施加了很强的关注 -- 因为底层信号周期约为 6。注意力在没有任何提示的情况下自己发现了季节性。*

### 因果掩码

预测任务必须禁止第 $i$ 步看到未来。标准做法是 **因果掩码**：一个下三角矩阵，把上三角填 $-\infty$，softmax 之后这些位置就归零：

```python
def causal_mask(n, device):
    return torch.tril(torch.ones(n, n, device=device)).bool()  # 对角线及以下为 1

scores = scores.masked_fill(~causal_mask(n, scores.device), float("-inf"))
```

预测 Transformer 与分类 Transformer 的 **唯一** 区别就是这一行。

---

## 5. 多头注意力：为时序定制的"分工"

单头注意力会把所有模式平均到一张图上。多头注意力把嵌入维度切成 $h$ 块，并行跑 $h$ 个独立注意力，再拼回来过一个线性层：

$$
\text{MultiHead}(X) = [\text{head}_1; \dots; \text{head}_h] \, W^O,
\qquad
\text{head}_j = \text{Attention}(X W^{Q}_j, X W^{K}_j, X W^{V}_j).
$$

每个头都有自己的 $W^Q_j, W^K_j, W^V_j \in \mathbb{R}^{d \times (d/h)}$，可以专门捕捉不同模式。在时间序列里我们经常观察到四种典型分工：

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

**用几个头？** 当 $d_\text{model} = 64\!-\!128$ 时，先取 4 个头作为基线。训练完可视化每个头的权重图：若多个头长得几乎一样，就 *减少*；若某个头在试图同时编码多个不同模式，就 *增加*。

---

## 6. 位置编码：把"时间"放回去

自注意力是 **置换不变** 的 -- 把输入打乱，输出也跟着同样打乱。对时间序列而言，这等于扔掉了数据集里最重要的变量。我们必须显式注入位置信息。

### 正弦位置编码

原版 Transformer 用一组几何分布频率的正余弦：

$$
PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),
\qquad
PE_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$

这个公式不是随便选的：

- **有界**：每个分量都在 $[-1, 1]$ 内，与 $p$ 无关。
- **位移线性等变**：$PE_{p+\Delta}$ 是 $PE_p$ 的固定线性函数，模型用一个线性投影就能学到"往前看 7 步"。
- **多尺度**：低维分量变化慢（编码长程位置），高维分量变化快（编码细粒度位置）。

### 时间感知编码（不规则采样）

当采样不等间隔时（传感器、tick 数据），喂进去的应当是 **真实的时间差**，而不是索引。常见写法：

```python
def time_features(timestamps, d_model):
    """timestamps: (B, n)，单位秒；返回 (B, n, d_model)。"""
    deltas = timestamps - timestamps[:, :1]              # 距窗口起点的秒数
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    args = deltas.unsqueeze(-1) * freqs                  # (B, n, d_model/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

这套代码同时适用于 1Hz 的 IoT 数据、不规则的成交 tick 和带缺失的样本。

---

## 7. Attention + LSTM：实战中最稳的混合架构

纯 Transformer 在长序列上很强，但需要大量数据。窗口在 **50 - 500 步** 范围时，混合架构往往是最强基线：LSTM 廉价地提取局部时序特征，注意力再决定每个预测步要回看哪些编码状态。

![LSTM 编码器 + Attention + LSTM 解码器：编码器输出 h1...h5，注意力对它们打分得到上下文向量 c_t，再喂给解码器生成预测。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig5_attention_lstm_hybrid.png)
*图 5. 混合架构保留了 LSTM 对局部时序结构的强归纳偏置，同时把注意力当作一个内容寻址的"指针"，按需回看历史。*

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

经验上这个架构（以及它的变体 DA-RNN、双阶段注意力等）在 M-competition 风格的基准上常常拿到第一梯队，尤其是在短预测视野、数据量有限的情况下。

---

## 8. $O(n^2)$ 的"显存墙"与突破方案

注意力矩阵有 $n^2$ 个元素，每一个都要算、都要存。窗口长度 4096、float32 时，一张图就是 64 MB -- 还要乘上头数、层数、batch。这堵墙是真实存在的。

![计算与显存复杂度：RNN 是 O(n)，全注意力是 O(n^2)；稀疏注意力 O(n log n)，线性注意力 O(n)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig4_complexity_vs_length.png)
*图 6. RNN 在显存上更省，注意力在并行性上更强。计算量交叉点大致在 $n \approx d$，当 $n \gg d$ 时必须换次平方变体。*

| 变体 | 时间 | 空间 | 思路 |
|---|---|---|---|
| 全注意力 | $O(n^2 d)$ | $O(n^2)$ | 暴力计算所有对 |
| 稀疏 / 跨步 | $O(n \log n \cdot d)$ | $O(n \log n)$ | 局部窗口 + 膨胀跳跃（Longformer、BigBird） |
| 线性注意力 | $O(n d^2)$ | $O(n d)$ | 用核特征替换 softmax（Linformer、Performer） |
| Informer ProbSparse | $O(n \log n \cdot d)$ | $O(n \log n)$ | 只保留 top-$\log n$ 个 query（第 8 部分会讲） |

对绝大多数时序问题，$n$ 在几百量级，$d$ 在几十到几百，与 RNN 的交叉点站在我们这边。**只有当标准实现 OOM 时**，再考虑次平方变体。

---

## 9. 案例：股价预测的注意力解释

把整套流程落到一个具体案例上。合成股价含三种成分 -- 慢趋势、30 日周期、第 60 天的财报事件，预测未来 10 天。对比 LSTM+Attention 和无注意力基线：

![股价预测：LSTM+Attention（橙色）成功跟上财报后的新均值水平；无注意力基线（灰色虚线）低估了上行。下方柱状图是预测 query 对过去 30 天的注意力权重，财报当天用红色标出。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention%E6%9C%BA%E5%88%B6/fig7_stock_attention_app.png)
*图 7. 注意力权重并非黑盒。我们能直接读出：模型在"重押"财报事件 + 最近一周的数据。*

三个观察：

1. **财报当天的权重显著偏大** -- 注意力在没有"事件"先验的前提下，自己学会了识别事件型记忆。
2. **30 日周期被保留** -- 橙色预测沿着周期波动，而基线塌缩成线性外推。
3. **可解释性近乎免费** -- 同一张矩阵既驱动了预测、也解释了预测。LSTM 需要事后工具（积分梯度、SHAP）；注意力直接给出 softmax 行。

提醒一句：注意力权重 **与重要性相关，但不等同于因果重要性**。高风险部署需用扰动实验（把某个 key 步置零，看预测如何变）来验证，而不是把热图当作真理。

---

## 10. 时序注意力的实践 checklist

1. **标准化输入**。注意力是点积；不归一化的话，量级大的特征会主导分数。
2. **加位置编码**。等间隔采样用正弦，不等间隔用时间感知版本。
3. **预测任务必须用因果掩码**，训练和推理都不能少。
4. **从 4 头、$d_\text{model} \in [64, 128]$ 起步**。验证集要求时再加。
5. **Pre-LN**：在注意力之前做 LayerNorm；注意力权重和 FFN 块都加 dropout。
6. **学习率比 RNN 低**：$10^{-4}$ 到 $5 \cdot 10^{-4}$，前几百步加 warm-up。
7. **早期就可视化每个头**。塌缩到几乎相同就减头，或加多样性正则。
8. **小心 $O(n^2)$ 墙**。需要 $n > 1024$ 时直接换次平方变体或 Informer（第 8 部分）。

---

## 11. 常见坑

- **忘了除以 $\sqrt{d_k}$** -- 训练几步就停滞。
- **掩码错位** -- 微妙的数据泄露，训练指标虚高，部署时翻车。
- **注意力被填充 token 污染** -- 忘了 padding mask，特殊 token 的信号会渗入每个位置。
- **把权重当作因果解释** -- 它是证据，不是证明。
- **窗口太短** -- 如果 10 步内就装得下所有有用历史，LSTM 大概率比 Transformer 跑得更快、效果不差。

---

## 12. 总结

注意力把 RNN 那条 **顺序、有损的信息通道** 替换成 **直接、可寻址的查表机制**。数学只是两次矩阵乘法 + 一次 softmax，但带来的影响是结构性的：

- 任意两个时间步之间的路径长度变成 $O(1)$；
- 训练完全并行，每个位置同时计算；
- 注意力矩阵自带可解释性；
- 多头注意力给出多尺度时序模式的天然抽象。

代价是 $O(n^2)$ 显存，以及必须显式注入位置。对绝大多数时序问题而言，这笔账划得很 -- 接下来的第 5、6、8 部分会展示 Transformer、TCN 和 Informer 如何把这套思想推得更远。

> **记忆口诀** -- *Q 提问，K 应答，V 装货；除根号、softmax、得权重；多头并行、各看一面。*

---

## 参考资料

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
3. Luong, Pham, Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP 2015.
4. Qin et al., *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*, IJCAI 2017.
5. Kitaev, Kaiser, Levskaya, *Reformer: The Efficient Transformer*, ICLR 2020.
6. Beltagy, Peters, Cohan, *Longformer: The Long-Document Transformer*, 2020.
7. Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. -- 第 8 部分会详细讲。

---

**系列导航**

| | |
|---|---|
| **上一篇** | [GRU](/zh/时间序列模型-三-GRU/) |
| **当前** | 第四部分：Attention 机制 |
| **下一篇** | [Transformer 架构](/zh/时间序列模型-五-Transformer架构/) |
