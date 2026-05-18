---
title: "时间序列模型（四）：Attention 机制——直接的长程依赖"
date: 2024-10-16 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Attention
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "自注意力、多头注意力和位置编码在时间序列中的应用。逐步推导数学公式，附 PyTorch 实现和注意力可视化。"
disableNunjucks: true
series_order: 4
series_total: 8
translationKey: "time-series-4"
---

RNN 和 LSTM 解决了"时间步太多"的问题，但留下了另一个更隐蔽的限制：信息必须**逐步传递**。要让第 100 步看到第 1 步的内容，得让那个信号沿着隐藏状态一路传 99 次——每一步都有衰减，每一步都得经过非线性挤压。即使 LSTM 的细胞状态再"高速公路"，也终究是单条车道、单向通行。

注意力机制的核心想法非常简单：**为什么不让任意两个时间步直接对话？**与其让第 100 步从前面 99 步那里"层层听说"第 1 步发生了什么，不如直接计算"第 100 步对第 1 步的关注权重"，然后用这个权重加权读取第 1 步的内容。这就把任意两点之间的距离从 99 步缩短到 1 步——梯度不再需要穿越整条序列才能更新远处的权重。

这听上去像在堆资源（每两步都要算关系，复杂度从 O(n) 暴增到 O(n²)），但换来的好处是革命性的：长程依赖不再是难题、训练可以并行（不像 RNN 必须按时间步串行）、注意力权重还能可视化出来当作模型的"自我解释"。本章我会从"用注意力增强 LSTM"这个最初的应用切入，再讲到完整的 Query/Key/Value 框架——这正是下一章 Transformer 的入口。我会用一个股票预测的小例子展示注意力权重热力图，你会直观看到模型把注意力放在哪几天上。

![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/illustration_1.png)

---

## 本文要点

- 循环模型在处理长程依赖时为何遭遇瓶颈，而注意力机制又是如何彻底打破这一限制的。
- Query / Key / Value 机制、缩放点积注意力（scaled dot-product attention）的原理，以及为何要除以 $\sqrt{d_k}$。
- 两种经典打分函数：**Bahdanau**（加性）与 **Luong**（乘性）。
- 如何将 **注意力机制嵌入 LSTM 编码器/解码器**，用于时间序列预测。
- **多头注意力**在时序任务中的专业化分工：不同注意力头分别聚焦于近期性、周期性或异常事件。
- $O(n^2)$ 的显存墙问题，以及稀疏注意力、线性注意力等方法如何绕过它。
- 一个完整的 **股价预测案例**，附带注意力权重热力图，直观揭示模型决策依据。

**前置知识**：熟悉 RNN/LSTM/GRU 的基本原理（第 2–3 篇）、线性代数基础、PyTorch 基本操作。

---
## 为何需要注意力？循环结构的瓶颈

在长度为 $n$ 的循环模型中，两个相距 $k$ 个时间步的位置之间，信息传递路径长达 **$O(k)$ 步**。每一步都需将全部信息压缩进单一的隐藏向量，不仅造成信息损失，还容易导致梯度在反向传播中逐步衰减。

然而，真实世界的时间序列往往不遵循这种“近邻优先”的假设：

- 几分钟前的心电图（ECG）异常，可能比最近 200 个正常样本更重要；
- 今天的电力负荷，通常最像“上周三同一时刻”的负荷；
- 股价仍在对几周前的 **财报发布事件** 做出反应。

注意力机制提出了一种截然不同的信息流动方式：每个时间步都能通过**可学习的直接连接**，与其他所有时间步建立关联。任意两点间的路径长度缩短至 $O(1)$，而连接强度——即 *注意力权重*——本身也具备可解释性。

![24 步窗口下的注意力权重热图：明亮的对角线表示近期偏置，偏离对角线的部分显示 12 步周期性，第 5 列的纵向亮带是对该位置异常的持续记忆。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig1_attention_heatmap.png)
*图 1. 因果注意力图天然编码了三种有用的先验：近期偏好、周期性模式，以及对异常事件的长期记忆，全程无需手工设计特征。*

---
## 从第一性原理理解缩放点积注意力

将输入序列堆叠为矩阵 $X \in \mathbb{R}^{n \times d}$，每行对应一个时间步。通过三个可学习的线性变换，生成对同一数据的三种“视角”：
$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V,$$
其中 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$。

- **Query** $Q$ —— “当前时间步在寻找什么？”
- **Key** $K$ —— “当前时间步提供了哪些线索？”
- **Value** $V$ —— “当前时间步实际携带了什么信息？”

查询 $i$ 与键 $j$ 的兼容性由点积衡量。整体公式为：
$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$
### 为何要除以 $\sqrt{d_k}$？

若 $Q$ 和 $K$ 的元素独立同分布且方差为 1，则每个点积 $q_i^\top k_j$ 的方差为 $d_k$。当 $d_k$ 较大时，softmax 输入值幅度过大，导致输出饱和——几乎所有梯度坍缩至零，仅剩一个位置有响应。除以 $\sqrt{d_k}$ 可将方差重新归一化为 1，从而维持健康的梯度流。

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
```text

整个机制仅包含两次矩阵乘法夹一个 softmax。其表达能力完全源于可学习的投影矩阵 $W^Q, W^K, W^V$。

---
## Bahdanau 与 Luong：两种经典打分函数

在 Transformer 出现之前，Bahdanau 等人（2015）提出了用于序列到序列翻译的 **加性注意力**，随后 Luong 等人（2015）提出了 **乘性（点积）注意力** 变体。即便今日，当你将注意力嵌入 RNN 时，这两种方法仍具实用价值。

![Bahdanau（加性）和 Luong（乘性）注意力打分函数对比。加性用一个小 MLP，乘性用点积；Transformer 选择了乘性并加入了 0 缩放因子。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig2_bahdanau_vs_luong.png)
*图 2. 两种计算 Query-Key 兼容度的方式：加性使用小型 MLP，乘性使用点积。Transformer 采用后者，并加入 $1/\sqrt{d_k}$ 缩放因子。*

| 特性 | Bahdanau（加性） | Luong（乘性） |
|---|---|---|
| 打分公式 | $v^\top \tanh(W_1 h_i + W_2 s_{t-1})$ | $s_t^\top W h_i$ |
| 单对计算成本 | 一次 MLP 前向传播 | 一次点积运算 |
| 参数 | $v, W_1, W_2$ | $W$（常设为单位阵） |
| 适用场景 | Query 与 Key 处于不同空间 | $Q$ 与 $K$ 共享同一空间 |
| 现代使用情况 | 在纯 Transformer 中罕见 | 标准选择（含 $1/\sqrt{d_k}$ 缩放） |

两者均输出 softmax 前的分数，最终通过 softmax 加权求和完成聚合。Transformer 仅选择了**计算更高效**的乘性形式，并补充了缩放因子。

---
## 自注意力在时间序列中的应用

在 seq2seq 模型中，Query 来自解码器，Key/Value 来自编码器——这是两个不同序列。**自注意力（Self-attention）** 则取消这一区分：同一序列同时充当 $Q$、$K$ 和 $V$。每个时间步都能关注窗口内所有其他时间步。

这正是时间序列预测所需的能力。例如，若用 12 步历史预测下一步，当前时刻的注意力权重会明确告诉我们：模型究竟依赖哪些历史时刻做判断。

![从最新时间步出发的自注意力，弧线粗细表示权重大小；下方柱状图显示相同权重分布。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig3_self_attention_ts.png)
*图 3. 在 $t = 11$ 处的预测 Query 不仅关注 $t = 10$，还强烈关注 $t = 5$（六步前），因为底层信号周期约为 6。注意力机制自主发现了季节性，无需任何先验提示。*

### 因果掩码

预测任务中，必须禁止时间步 $i$ 查看未来信息。标准做法是引入 **因果掩码**：在得分矩阵上叠加一个下三角掩码，将上三角部分设为 $-\infty$，使 softmax 将其置零：

```python
def causal_mask(n, device):
    return torch.tril(torch.ones(n, n, device=device)).bool()  # 对角线及以下为 1

scores = scores.masked_fill(~causal_mask(n, scores.device), float("-inf"))
```text

这正是预测型 Transformer 与序列分类 Transformer 的唯一区别。

---
## 多头注意力：为时间序列量身定制的“分工协作”

单头注意力会将多种模式混合在同一张注意力图中。多头注意力则并行运行 $h$ 个独立的注意力机制，每个作用于嵌入维度的一个子空间，最后拼接并通过线性投影融合：
$$
\text{MultiHead}(X) = [\text{head}_1; \dots; \text{head}_h] \, W^O,
\qquad
\text{head}_j = \text{Attention}(X W^{Q}_j, X W^{K}_j, X W^{V}_j).
$$
每个头拥有独立的 $W^Q_j, W^K_j, W^V_j \in \mathbb{R}^{d \times (d/h)}$，可自由专业化。在时间序列任务中，训练后通常观察到四类典型头：

![四个注意力头在同一个 18 步窗口上的权重图：分别对应近期、长程趋势、周期 7、以及对 t=4 异常事件的持续记忆。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig6_multihead_for_time.png)
*图 4. 同一 18 步窗口经四个头处理，呈现出不同结构。多头注意力本质上是一个可学习的时序核函数集合。*

| 注意力头类型 | 学到的模式 | 时序意义 |
|---|---|---|
| Local（局部） | 锐利对角线 | 短期动量 |
| Long-range（长程） | 弥漫三角形 | 缓慢漂移、状态切换 |
| Periodic（周期） | 偏移对角条纹 | 日/周循环 |
| Anomaly（异常） | 垂直亮列 | “记住第 $k$ 步的尖峰” |

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
```text

**该用多少头？** 当 $d_\text{model} = 64\!-\!128$ 时，建议从 4 个头起步。训练后可视化各头：若多个头高度相似，应减少头数；若单个头试图编码多种模式，则应增加头数。

---
## 位置编码：把“时间”重新注入模型

自注意力具有**置换不变性**——打乱输入顺序，输出也会相应重排。这对时间序列而言是灾难性的，因为它直接丢弃了最关键的变量：时间顺序。因此，必须显式注入位置信息。

### 正弦位置编码

原始 Transformer 使用几何间隔频率的正弦与余弦函数：
$$
PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),
\qquad
PE_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$
为何选择此形式？

- **有界性**：所有值落在 $[-1, 1]$ 内，与位置 $p$ 无关；
- **线性位移等变性**：$PE_{p+\Delta}$ 是 $PE_p$ 的固定线性变换，模型可通过单一线性层学会“回溯 7 步”这类相对偏移；
- **多尺度表示**：低维分量变化缓慢（表征长期位置），高维分量变化迅速（表征精细位置）。

### 时间感知编码（适用于非均匀采样）

当采样不规则（如传感器数据、交易记录），应使用 **实际时间戳差值** 而非索引。常见做法如下：

```python
def time_features(timestamps, d_model):
    """timestamps: (B, n)，单位秒；返回 (B, n, d_model)。"""
    deltas = timestamps - timestamps[:, :1]              # 距窗口起点的秒数
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    args = deltas.unsqueeze(-1) * freqs                  # (B, n, d_model/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```text

该方法可统一处理 1 Hz 的 IoT 数据、不规则交易 tick 以及缺失样本。

---
## Attention + LSTM：实用的混合架构

纯 Transformer 在长序列上表现卓越，但需大量数据支撑。对于 **50–500 步** 的窗口长度，混合架构往往是更强的基线：LSTM 高效提取局部时序特征，注意力机制则动态选择每个预测步所依赖的历史状态。

![LSTM 编码器 + Attention + LSTM 解码器：编码器生成隐藏状态 h1...h5，注意力对它们打分后生成上下文向量，用于解码器预测。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig5_attention_lstm_hybrid.png)
*图 5. 混合架构保留了 LSTM 对局部序列结构的强大归纳偏置，同时利用注意力作为内容寻址的“指针”，灵活回溯历史。*

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
```text

实证表明，此类架构（如 DA-RNN、双阶段注意力等）在 M-competition 类基准测试中表现优异，尤其适用于预测视野较短、数据有限的场景。

---
## $O(n^2)$ 显存墙及其突破方案

注意力矩阵包含 $n^2$ 个元素，每个都需计算与存储。以 4096 步窗口、float32 精度为例，单头单层单样本即占用 64 MB 显存——瓶颈真实存在。

![计算与显存复杂度：RNN 是 O(n)，全注意力是 O(n^2)；稀疏注意力 O(n log n)，线性注意力 O(n)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig4_complexity_vs_length.png)
*图 6. RNN 显存友好，并行性弱；注意力并行性强，但显存开销大。计算交叉点约在 $n \approx d$。当 $n \gg d$ 时，需采用次平方复杂度变体。*

| 变体 | 时间复杂度 | 空间复杂度 | 核心思想 |
|---|---|---|---|
| 全注意力 | $O(n^2 d)$ | $O(n^2)$ | 计算所有位置对 |
| 稀疏 / 跨步 | $O(n \log n \cdot d)$ | $O(n \log n)$ | 局部窗口 + 膨胀跳跃（Longformer, BigBird） |
| 线性注意力 | $O(n d^2)$ | $O(n d)$ | 用核特征映射替代 softmax（Linformer, Performer） |
| Informer ProbSparse | $O(n \log n \cdot d)$ | $O(n \log n)$ | 仅保留 top-$\log n$ 查询（见第 8 篇） |

多数时序问题中，$n$ 为数百，$d$ 为数十至数百，此时标准注意力仍优于 RNN。仅当显存不足时，才需转向次平方变体。

---
## 案例研究：股票价格预测

为使流程具体化，我们构造一个合成股价序列：包含缓慢趋势、30 天周期波动，以及第 60 天的财报事件。目标是预测未来 10 天价格，对比 LSTM+Attention 与无注意力基线。

![股价预测：LSTM+Attention（橙色）成功捕捉财报后的新趋势；无注意力基线（灰色虚线）低估了变化。下方柱状图显示预测 query 对过去30天的注意力权重，财报当天用红色标出。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/04-Attention机制/fig7_stock_attention_app.png)
*图 7. 注意力权重并非黑箱。图中清晰显示模型高度关注财报日及最近一周数据。*

三点关键观察：

1. **财报日权重显著偏高**——模型未被告知“财报”概念，却自主识别出该事件记忆；
2. **周期峰值得以保留**——橙色预测曲线跟随 30 天振荡，而基线退化为近似线性外推；
3. **可解释性“免费”获得**——驱动预测的矩阵同时提供解释。LSTM 需依赖事后工具（如积分梯度、SHAP），而注意力直接输出 softmax 行作为解释。

需谨慎的是：注意力权重反映**相关性而非因果性**。在高风险部署中，应通过扰动测试验证解释（如将关键时间步置零，观察预测变化），而非将热图视为绝对真理。

---
## 时间序列注意力的实用指南

1. **标准化输入**：注意力基于点积，未归一化的大尺度特征会主导结果。
2. **加入位置编码**：规则采样用正弦编码，非规则采样用时间感知编码。
3. **预测任务务必使用因果掩码**，训练与推理阶段均不可省略。
4. **初始配置：4 个头，$d_\text{model} \in [64, 128]$**，仅当验证损失要求时才扩展。
5. **注意力前加 LayerNorm**，并在注意力权重与前馈网络上施加 dropout。
6. **学习率低于 RNN**：建议 $10^{-4}$ 至 $5 \cdot 10^{-4}$，配合数百步 warm-up。
7. **尽早可视化注意力头**：若多头趋同，减少头数或引入多样性正则化。
8. **警惕 $O(n^2)$ 瓶颈**：若需处理 $n > 1024$，直接选用次平方变体或 Informer（见第 8 篇）。

---
## 常见陷阱

- **遗漏 $\sqrt{d_k}$ 缩放**：训练几步后损失停滞。
- **掩码错误**：隐蔽的数据泄露导致训练指标虚高，部署时崩溃。
- **注意力被填充符污染**：未屏蔽 padding token，使其信号扩散至全序列。
- **将权重误作因果解释**：权重是证据，非结论。
- **窗口过短**：若有效历史仅需 10 步，LSTM 往往更快且效果不输。

---
## 总结

注意力机制以 **直接、按内容寻址的查找**，取代了 RNN 的 **顺序、有损信息通道**。其数学形式仅为两次矩阵乘法加一个 softmax，却带来深远影响：

- 任意两时间步间路径长度降至 $O(1)$；
- 训练完全并行，所有位置同步计算；
- 注意力矩阵天然提供可解释性；
- 多头机制为多尺度时序模式提供清晰抽象。

代价是 $O(n^2)$ 显存开销及需显式注入位置信息。但对大多数时序任务而言，这些成本完全值得。本系列第 5、6、8 篇将进一步探讨 Transformer、TCN 与 Informer 如何拓展这一思想。

> **记忆口诀** —— *Q 提问，K 回答，V 携带；除以 $\sqrt{d_k}$，softmax 转权重，乘 V 得结果；多个头，多个视角。*

---

## 下一步

注意力机制做的事很简单——让任意两个时间步直接计算关系——但带来的影响是革命性的。RNN 时代那种"长程依赖必须靠门控艰难维持"的问题，被它一次性化解了。同时它解锁了并行训练（不用再按时间步串行）、还附送了可视化解释（直接看注意力权重热图）。

但本章我们用注意力的方式还很"温和"——把它作为 LSTM 的辅助。下一章 [Transformer](/zh/time-series/05-transformer架构) 把这件事推到极致：**完全抛弃 RNN**，整个模型只用注意力堆出来。这种纯 attention 架构在 NLP 里一统天下，但搬到时序场景需要解决两个新问题：怎么把"时间顺序"信息注入进去（attention 本身是顺序无关的），以及怎么应对 O(n²) 复杂度（一个月的小时序列就 720 步）。下一章会把这两个问题的四种主流解法——稀疏、线性、分块、decoder-only——以及对应的明星模型 Autoformer / FEDformer / Informer / PatchTST 一次讲清楚。

在那之前，建议你先把本章的注意力权重热图代码跑通，挑几个已知的"应该被注意到"的时间点（比如已知的事件、已知的周期性峰值），看看模型有没有真的把权重放在那里。这个习惯——把模型的"自我解释"和你已知的领域知识对齐——在后面所有 Transformer 类模型上都会用得到。

## 参考文献

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
3. Luong, Pham, Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP 2015.
4. Qin et al., *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*, IJCAI 2017.
5. Kitaev, Kaiser, Levskaya, *Reformer: The Efficient Transformer*, ICLR 2020.
6. Beltagy, Peters, Cohan, *Longformer: The Long-Document Transformer*, 2020.
7. Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. — 第 8 篇将详细介绍。
