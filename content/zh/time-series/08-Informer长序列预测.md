---
title: "时间序列模型（八）：Informer——高效长序列预测"
date: 2024-12-15 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Informer
  - Transformer
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "Informer 用 ProbSparse 注意力、编码器蒸馏、生成式解码器把 Transformer 复杂度从 O(L^2) 降到 O(L log L)。完整数学推导、PyTorch 代码与 ETT/气象 benchmark。"
disableNunjucks: true
series_order: 8
translationKey: "time-series-8"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/illustration_1.png)

Transformer 在序列建模上确实很强大，但只要序列一长，问题就来了。普通自注意力的计算和显存开销都是 $\mathcal{O}(L^2)$。比如，一周小时级窗口（168 步）还能轻松搞定，一个月窗口（720 步）就开始吃力，三个月窗口（2160 步）在单张 GPU 上基本跑不动。而实际应用中的长 horizon 预测——像气象、能源、金融、IoT——偏偏就在这个范围。

**Informer**（Zhou 等人，AAAI 2021 最佳论文）是让 Transformer 在这类场景中变得实用的关键架构。它做了三件事，每一件单独拿出来都算得上重要贡献：

1. **ProbSparse 自注意力**只保留 $\mathcal{O}(\log L)$ 个最有价值的查询，把每层复杂度从 $\mathcal{O}(L^2)$ 降到 $\mathcal{O}(L \log L)$。
2. **自注意力蒸馏**在编码器层之间将序列长度减半，显存占用随深度几何级下降。
3. **生成式解码器**一次前向传播就能预测整个 horizon，不再需要分 $H$ 步自回归。

这三项改进合在一起，在长 horizon 的 ETT、气象、电力 benchmark 上，比普通 Transformer 快 6-10 倍，MSE 还能提升 5-10%。本章会详细拆解背后的数学原理，并一步步实现代码。
## 这一篇你会学到

- 原始自注意力处理长序列时，$\mathcal{O}(L^2)$ 的性能瓶颈具体在哪里。
- ProbSparse 怎么用 KL 散度衡量稀疏性，以及 $\max - \mathrm{mean}$ 近似方法的原理。
- 编码器蒸馏如何通过减少序列长度来增加模型深度，同时保留关键模式。
- 生成式解码器为什么比自回归解码快得多，精度还稍微更高。
- 完整的 PyTorch 实现代码，以及 Informer 在 ETT 和气象数据集 benchmark 上的结果。

**前置知识**：第 5 部分（Transformer 架构）。熟悉 Big-O 分析，了解基础信息论概念（熵、KL 散度）。
## 长序列为何让原始 Transformer 崩溃

自注意力机制对每个查询 $q_i$ 的计算公式如下：

$$\mathrm{Attn}(q_i, K, V) = \sum_{j=1}^{L} \mathrm{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j.$$

要为所有查询计算，就得生成完整的 $L \times L$ 分数矩阵。这里有三个主要开销，都和 $\mathcal{O}(L^2)$ 相关：

- $L$ 个维度为 $d$ 的 query-key 点积，总计 $L^2 d$ FLOPs。
- $L^2$ 次 softmax 操作。
- 反向传播时需要存储 $L^2$ 个浮点数的注意力矩阵。

举个具体例子：预测长度 $L = 720$，维度 $d = 64$，8 个头，单样本情况下：

- 注意力分数：每层每头有 $720 \times 720 = 518\text{K}$ 个条目。
- 显存占用：batch size 为 32 时，仅注意力权重就占 ~16 MB（float32，8 头，3 层）。反向传播时激活值会让显存需求再增加一个数量级。
- 计算量：每层每头约 ~33 M FLOPs，主要集中在 $L^2 d$ 的矩阵乘法上。

如果把 $L$ 提高到 2160，每头的注意力条目接近 5 M。这一规模下，单张 24 GB 显存 GPU 在常规训练 batch size 下即会触发显存溢出（OOM）。

针对这个问题，之前的研究尝试了不同方法。Longformer 和 BigBird 引入结构化稀疏性，比如局部窗口 + 全局窗口，或者随机 + 全局窗口。Linformer 和 Performer 则采用低秩近似。Informer 的思路完全不同：**让数据决定哪些查询值得完整注意力**。
## ProbSparse：哪些查询重要，哪些不重要

### 直觉

画出一个典型查询 $q_i$ 在所有键上的注意力分布，会看到两种完全不同的形状：

- **尖峰型**：少数几个键占据了大部分概率。这种查询很“挑剔”，它具有明确的注意力聚焦倾向。
- **均匀型**：概率均匀分布在所有键上。这种查询很“模糊”，需要看全量数据才能发挥作用。

尖峰型查询可以通过只计算 top 几个键的注意力来高效近似；均匀型查询则不行。关键在于，如何在**不计算完整注意力矩阵的情况下**区分这两类查询。

### 基于 KL 的稀疏性度量

衡量分布是否尖锐，最自然的方法是用 KL 散度与均匀分布比较。设

$$p(k_j \mid q_i) = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right).$$

那么

$$\mathrm{KL}(q_i \,\|\, U) = \log L + \frac{1}{L}\sum_{j=1}^{L} \log p(k_j \mid q_i).$$

去掉常数项并代入 softmax 后，Zhou 等人证明了

$$\mathrm{KL}(q_i \,\|\, U) \;\propto\; \log\!\left(\sum_{j=1}^{L} e^{q_i^\top k_j / \sqrt{d}}\right) - \frac{1}{L}\sum_{j=1}^{L} \frac{q_i^\top k_j}{\sqrt{d}}.$$

把这个量记为 $M(q_i, K)$。$M$ 大表示分布尖锐——挑剔查询，值得完整注意力；$M$ 小表示分布均匀——模糊查询，可以跳过。

但精确计算 $M$ 仍然需要 $L$ 次内积，这又回到了原点。Informer 的第二个技巧是从 $u = c \log L$ 个键中随机采样来近似 $M$（$c$ 是常数，通常取 5）：

$$\bar{M}(q_i, K) \;=\; \max_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}} \;-\; \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}}.$$

这里用 $\max$ 替代 LogSumExp 的依据是高维近高斯向量的测度集中性：LogSumExp 主要由最大项主导。实验证明，$\bar{M}$ 对查询的排序和精确 $M$ 几乎一致，但代价小得多。

### ProbSparse 实际计算的内容

单个注意力头的流程如下：

1. 均匀随机采样 $u = c \log L$ 个键。
2. 对每个查询 $q_i$ 计算 $\bar{M}(q_i, K)$——复杂度 $\mathcal{O}(L \log L)$。
3. 按 $\bar{M}$ 选出 top $u$ 个查询。
4. 对这 $u$ 个查询，在**全部** $L$ 个键上计算注意力；对剩下的 $L - u$ 个查询，用 $V$ 的均值填充输出。

总复杂度：$\mathcal{O}(L \log L)$。显存占用也是 $\mathcal{O}(L \log L)$。

![ProbSparse 注意力 vs 完整注意力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig1_probsparse_vs_full.png)

图中右侧只保留了高 $M$ 查询对应的行。其他行实际上不是零——它们填的是 $V$ 的均值，这对均匀注意力分布来说是一个合理近似。

参考实现如下：

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_size, n_top):
        # Q, K: (B, H, L, d_k)
        B, H, L_K, _ = K.shape
        L_Q = Q.size(2)

        # 1) 从 K 中均匀采样 u 个键
        idx = torch.randint(0, L_K, (sample_size,), device=K.device)
        K_sample = K[:, :, idx, :]                                  # (B, H, u, d_k)

        # 2) 稀疏性度量 M_bar(q, K_sample) = max - mean
        Q_K_s = torch.matmul(Q, K_sample.transpose(-2, -1))         # (B, H, L_Q, u)
        M_bar = Q_K_s.max(dim=-1).values - Q_K_s.mean(dim=-1)       # (B, H, L_Q)

        # 3) 选 top-n_top 查询
        top_idx = M_bar.topk(n_top, dim=-1).indices                 # (B, H, n_top)

        # 4) 仅对 top 查询计算完整注意力
        Q_top = torch.gather(
            Q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        )                                                           # (B, H, n_top, d_k)
        scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return scores, top_idx

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        L_K = keys.size(1)

        Q = self.W_q(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        u = max(1, min(L_K, self.factor * int(np.ceil(np.log(L_K)))))

        scores, top_idx = self._prob_QK(Q, K, sample_size=u, n_top=u)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out_top = torch.matmul(attn, V)                              # (B, H, n_top, d_k)

        # 未选中的查询输出用 V 的均值初始化
        ctx = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        ctx.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
            out_top,
        )

        ctx = ctx.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.W_o(ctx)
```

几个细节：

- $u = c \ln L$ 是自然对数；用 `numpy` 的 `np.log` 并向上取整。
- “未选中查询用 $V$ 的均值填充”是数学上正确的处理方式，而不是一种 hack。它对应在“未选查询是均匀注意力”约束下熵最大的唯一分布。
- 解码器的 masked self-attention 需在 softmax 计算前屏蔽未来位置的键。上面的实现用了 `-1e9` 这一标准技巧。
## 编码器蒸馏：金字塔式序列压缩

即使有 ProbSparse，三层编码器每层处理 $L = 720$ 的序列依然很耗资源。Informer 在编码器层之间加了一个**蒸馏**操作，把序列长度砍半：

$$X_{\ell+1} = \mathrm{MaxPool}_{k=3, s=2}\!\Big(\mathrm{ELU}\big(\mathrm{Conv1d}_{k=3, s=2}(X_\ell)\big)\Big).$$

stride=2 的 Conv1d 是一个可学习的下采样器；MaxPool 在每对相邻位置保留主导值；中间的 ELU 非线性激活让这个操作比纯池化更有表达能力。

![编码器蒸馏金字塔](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig3_encoder_distilling.png)

效果是累积的：一个 3 层编码器把 720 步输入逐步压缩成 $720 \to 360 \to 180 \to 90$。显存随深度几何级减少，而不是线性增长。底层能看到更长的历史信息，顶层的感受野轻松覆盖几千个原始时间步。

需要注意两点：

- **最后一层不蒸馏**。解码器的交叉注意力读取编码器输出；如果最后一层也蒸馏，分辨率再减半，会丢信息。标准做法是“除最后一层外，每层都蒸馏”。
- **用两个并行编码器提升鲁棒性**。原论文中跑两个编码器，一个处理完整输入，另一个处理半长输入，最后拼接输出。这种冗余设计能避免特定序列上蒸馏决策不佳的问题。

```python
class DistillingLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                              stride=2, padding=1)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = self.conv(x.transpose(1, 2))
        x = self.act(x)
        x = self.pool(x)
        return x.transpose(1, 2)
```
## 生成式解码器：一次搞定整个预测范围

普通的 Transformer 解码器是自回归的。它先预测 $\hat{y}_1$，把结果喂回去，再预测 $\hat{y}_2$，依此类推。如果预测范围 $H = 168$，就要跑 168 次前向计算。延迟不说，错误会逐级累积：第 5 步的预测误差将作为输入影响第 6 步的预测。

Informer 的生成式解码器换了种思路。它的输入构造如下：

$$X_\text{dec} = \big[\, X_\text{token} \;;\; X_0 \,\big],$$

其中 $X_\text{token}$ 是编码器输入的最后 `label_len` 个时间步，相当于“提示”；$X_0$ 是 `out_len` 个占位 token，通常是对应维度的零向量。解码器只需要跑一次，处理整个 $\text{label\_len} + \text{out\_len}$ 序列，最后 `out_len` 个输出就是预测结果。

![自回归解码器 vs 生成式解码器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig2_generative_decoder.png)

它具备三大优势：

- **速度快**：从 $H$ 次前向计算变成 1 次，推理延迟直接减少 $H$ 倍。
- **错误不累积**：所有预测都基于同一份编码器上下文，不会依赖前面的预测结果。
- **长预测更准**：反直觉的是，生成式解码器在长预测范围上往往比自回归解码器更准。原因在于，自回归解码器被迫“短视”优化——每一步训练都假设前面的预测完全正确。而生成式解码器直接联合优化整个预测序列。

label tokens 很关键。它们给解码器提供了一些“真实”的数据点作为起点，帮助后面的占位 token 找到方向。经验上，`label_len = out_len / 2` 的效果不错。
## 拼起来：Informer 完整模型

完整模型由编码器和解码器组成，嵌入部分结合了数值、位置和时间特征。

```python
class TemporalEmbedding(nn.Module):
    """将 (hour, day_of_week, month) 嵌入到 d_model。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.hour = nn.Embedding(24, d_model)
        self.dow = nn.Embedding(7, d_model)
        self.month = nn.Embedding(12, d_model)

    def forward(self, time_feat):
        # time_feat: (B, L, 3) 整数
        return (self.hour(time_feat[..., 0])
                + self.dow(time_feat[..., 1])
                + self.month(time_feat[..., 2]))

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distil=True):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distil = DistillingLayer(d_model) if distil else None

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x, x, x)))
        x = self.norm2(x + self.ffn(x))
        if self.distil is not None:
            x = self.distil(x)
        return x

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask=self_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out)))
        x = self.norm3(x + self.ffn(x))
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                 seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len

        self.enc_value = nn.Linear(enc_in, d_model)
        self.dec_value = nn.Linear(dec_in, d_model)
        self.pos_enc = nn.Embedding(seq_len, d_model)
        self.pos_dec = nn.Embedding(label_len + out_len, d_model)
        self.t_emb = TemporalEmbedding(d_model)

        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                 distil=(i < e_layers - 1))
            for i in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])
        self.head = nn.Linear(d_model, c_out)

    def _embed(self, x_value, time_feat, embed_value, pos_emb):
        B, L, _ = x_value.shape
        positions = torch.arange(L, device=x_value.device).unsqueeze(0).expand(B, L)
        return embed_value(x_value) + pos_emb(positions) + self.t_emb(time_feat)

    def forward(self, x_enc, t_enc, x_dec, t_dec):
        enc = self._embed(x_enc, t_enc, self.enc_value, self.pos_enc)
        for layer in self.encoder:
            enc = layer(enc)

        dec = self._embed(x_dec, t_dec, self.dec_value, self.pos_dec)
        L_dec = dec.size(1)
        # 解码器 self-attention 的因果掩码
        causal = torch.tril(torch.ones(L_dec, L_dec, device=dec.device)).bool()
        for layer in self.decoder:
            dec = layer(dec, enc, self_mask=causal)

        return self.head(dec[:, -self.out_len:, :])  # (B, out_len, c_out)
```

构造训练数据时，解码器输入由最后 `label_len` 个真实值和 `out_len` 个零占位符拼接而成：

```python
def build_decoder_input(x_enc, label_len, out_len):
    # x_enc: (B, seq_len, F)
    start = x_enc[:, -label_len:, :]
    placeholder = torch.zeros(
        x_enc.size(0), out_len, x_enc.size(-1), device=x_enc.device
    )
    return torch.cat([start, placeholder], dim=1)
```
## 长 horizon 表现

主图展示了真实值、原始 Transformer 和 Informer 在 480 步长的预测对比。

![长 horizon 预测：原始 Transformer 漂移，Informer 贴合真实值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig4_long_sequence_forecast.png)

原始 Transformer 的自回归误差从第 100 步开始累积，逐渐偏离真实值。而 Informer 使用一次性生成解码器，所有输出 token 联合优化，因此在整个窗口内都能给出连贯的预测。

经典 ETT（Electricity Transformer Temperature）基准测试中的数据如下：

![ETTh1 单变量 MSE 与 L = 720 时的资源消耗](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig5_ett_benchmark.png)

两个关键点：

- **长 horizon 精度**：在 horizon 为 720 时，Informer 的 MSE 是 0.235，而原始 Transformer 是 0.269。虽然绝对差距不大，但原始 Transformer 在 horizon 720 时已经接近可训练的极限。
- **资源消耗**：当 $L = 720$ 时，Informer 在单张 V100 上的峰值显存占用为 1.8 GB，每轮训练耗时 9.5 秒；而原始 Transformer 需要 10.5 GB 显存，每轮耗时 104 秒。这种差距正是 Informer 存在的核心原因。

---
## 超参速查表

| 超参 | 默认值 | 备注 |
|---|---|---|
| `d_model` | 512 | 标准配置，显存不足时用 256。 |
| `n_heads` | 8 | $d_k = d_\text{model} / n_\text{heads}$，所以这里取 64。 |
| `e_layers` | 3 | 层数越多，蒸馏越激进，但增加层数通常不划算。 |
| `d_layers` | 2 | 非对称设计，编码器承担主要计算任务。 |
| `d_ff` | 2048 | 通常是 `d_model` 的 4 倍。 |
| `factor`（$u = c \log L$ 中的 $c$） | 5 | 求速度用 3，求精度用 7，一般不用改。 |
| `seq_len` | 96 到 720 | 大致等于数据的周或月周期长度。 |
| `label_len` | `out_len / 2` | 解码器的锚点，不要设为 0。 |
| `out_len` | 任务驱动 | 这是你实际要预测的步数。 |
| `dropout` | 0.05-0.1 | 数据集较小时可以适当调大。 |
| 优化器 | Adam，lr 1e-4 | 学习率是普通 Transformer 的一半。 |
| LR 调度 | StepLR，每 30 epoch γ=0.5 | 也可以用 cosine annealing。 |
| 损失函数 | MSE | L1 (MAE) 对异常值更鲁棒。 |

---
## 常见问题

- **解码器的 self-attention 没加 mask**。没有因果 mask，解码器在训练时会偷看未来的占位 token。训练时效果看起来很好，测试时却一团糟。
- **对太短的输入做蒸馏**。如果 `seq_len` 很短（比如 24），三层蒸馏会让编码器输出压缩到长度为 3，丢掉大部分上下文。`seq_len < 96` 时应该关掉蒸馏。
- **忽略逐窗口标准化**。和 N-BEATS 一样的问题：输入模型前要对每个窗口标准化，输出时再逆变换回来。
- **没加入时间特征编码**。时间嵌入（hour-of-day、day-of-week、month）对性能提升很重要；少了它，模型只能从原始数据中推测时间信息。
- **`label_len` 太小**。有些实现默认 `label_len = 0`，这会让解码器失去锚定能力。论文建议设置为 `label_len = out_len / 2`。
## 什么时候不该用 Informer

- **短序列（$L < 96$）**。ProbSparse 和蒸馏操作有固定开销。处理短序列时，普通 Transformer 或 LSTM 更简单，速度也完全够用。
- **跨特征交互是主要信号**（多变量数据，特征间依赖性强）。Informer 的注意力机制沿时间轴工作。如果需要跨特征的注意力，可以看看 TFT 或 TimesNet。
- **需要精确的注意力图来解释模型**。ProbSparse 会丢弃未选查询的逐行注意力信息。如果必须完整可视化注意力分布，那就用普通 Transformer。
- **高频流式推理**（kHz、MHz 级别）。Informer 是为批量预测设计的，流式推理需要更专用的架构。
- **极小数据集（<1k 样本）**。Informer 参数量高达几千万，容易过拟合。这种情况下，选择更小、表达能力更低的模型更合适。
## Q&A

### 为什么是 $u = c \log L$？

这个公式来自“挑剔查询”中最强键被采样到的概率预期。当 $u = c \log L$ 且 $c = 5$ 时，任何查询漏掉 top-1 键的概率不超过 $1/L^4$。实际用下来，$c = 3$ 也完全够用。

### ProbSparse 能选出正确的查询吗？

实验结果表明可以。$\max - \mathrm{mean}$ 近似值和训练过程中观察到的注意力分布的精确 KL 散度之间的 Spearman 相关系数超过 0.95。论文里有完整的消融实验。

### 为什么不选的查询要用 $V$ 的均值，而不是零？

因为均匀注意力分布的计算结果就是 $\frac{1}{L}\sum_j v_j$。对于被判定为“均匀注意力”的查询，用均值填充是最准确的选择。

### Informer 和 Reformer / Performer / Linformer 有什么不同？

- **Reformer**：用 LSH 桶化注意力，复杂度是 $\mathcal{O}(L \log L)$，但桶化过程和数据无关。
- **Performer**：用随机特征核近似，复杂度是 $\mathcal{O}(L)$，但在长序列上，注意力分布陡峭时精度会下降。
- **Linformer**：把键和值投影到固定低秩维度，复杂度是 $\mathcal{O}(L)$，但投影在训练时就固定了。
- **Informer**：根据数据自适应选择查询，复杂度是 $\mathcal{O}(L \log L)$，在时间序列基准测试中精度最高。

### 多变量输入时，编码器和解码器的特征维度可以不同吗？

可以。`enc_in` 和 `dec_in` 是独立的。常见做法是把所有变量输入编码器，而只把目标变量输入解码器。

### Autoformer 和 FEDformer 呢？

它们都是 Informer 的改进版。Autoformer（2021）用序列自相关替换了 self-attention，并增加了一个显式的分解层。FEDformer（2022）引入了频域注意力机制。两者在相同基准测试中表现优于 Informer，但实现更复杂。如果刚开始，我建议从 Informer 入手。

### 是否需要先在大规模多序列数据集上预训练？

有帮助，但不是必须的。和 NLP 不同，时间序列数据集之间的领域差异很大，简单的预训练往往适得其反。针对具体领域从头微调通常是更好的选择。
## 小结

Informer 是一种让 Transformer 在长周期时间序列预测中变得实用的架构。它有三个核心设计：ProbSparse 自注意力、编码器蒸馏和生成式解码器。这三者组合成一个端到端的 $\mathcal{O}(L \log L)$ 系统，在所有长周期基准测试中，精度和实际运行时间都超过了原始的 $\mathcal{O}(L^2)$ Transformer。

如果预测任务的周期 $L > 96$，而且只用一块 GPU，那么 Informer 是最合适的起点。后来的架构（比如 Autoformer、FEDformer 和 PatchTST）进一步优化了这个思路，但它们都基于 Informer 的两个关键洞察：**不是每个查询都需要完整的 attention**，**自回归解码是自己给自己挖的坑**。

到这里，时间序列预测系列就结束了。八章内容里，我从经典的 ARIMA 讲到 LSTM，再到 Transformer、TCN、N-BEATS，最后到 Informer。选择适合你数据的架构，关键时刻做集成，记住一点：在小问题上，简单的基线模型往往能打败那些时髦的复杂模型。
## 参考资料

- Zhou, H. et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI Best Paper.
- Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.
- Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.* ICML.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR.
- Wang, S. et al. (2020). *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768).
- Choromanski, K. et al. (2021). *Rethinking Attention with Performers.* ICLR.

---

> 
>
> 本文是时间序列模型系列的**第 8 篇**，共 8 篇（系列完结）。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - [第 2 篇：LSTM —— 门控机制与长期依赖](/zh/time-series/02-lstm/)
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru/)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构/)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn/)
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - **第 8 篇：Informer —— 高效长序列预测**（当前）
