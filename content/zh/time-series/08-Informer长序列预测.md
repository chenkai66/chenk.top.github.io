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
\nTransformer 在序列建模上确实很强大，但只要序列一变长，问题就来了。普通自注意力机制在计算和显存上的开销都是 $\mathcal{O}(L^2)$ 级别——一周的小时级窗口（168 步）还能轻松处理，一个月窗口（720 步）就已经吃力，而三个月窗口（2160 步）在单张 GPU 上基本无法运行。偏偏现实世界中的长 horizon 预测任务，比如气象、能源、金融和 IoT，恰恰就落在这个区间。

**Informer**（Zhou 等人，AAAI 2021 最佳论文）正是让 Transformer 在这类场景中变得实用的关键架构。它做了三件事，每一件单独拿出来都足以成为一项重要贡献：

1. **ProbSparse 自注意力**仅保留 $\mathcal{O}(\log L)$ 个最具信息量的查询，将每层的时间和空间复杂度从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(L \log L)$。
2. **自注意力蒸馏**在编码器各层之间将序列长度减半，使显存占用随网络深度呈几何级数下降。
3. **生成式解码器**只需一次前向传播即可预测整个 forecast horizon，无需像传统方式那样执行 $H$ 次自回归步骤。

这三项改进组合起来，在 ETT、气象和电力等长 horizon 基准测试中，比原始 Transformer 快 6–10 倍，MSE 还能提升 5–10%。本章将深入剖析每一项技术背后的数学原理，并逐步实现其核心逻辑。

## 这一篇你会学到

- 原始自注意力在处理长序列时，$\mathcal{O}(L^2)$ 瓶颈具体体现在哪些环节。
- ProbSparse 如何利用 KL 散度衡量注意力分布的“尖锐程度”，以及为何可用 $\max - \mathrm{mean}$ 来高效近似。
- 编码器蒸馏如何在压缩序列长度的同时保留主导模式，实现“以深度换长度”。
- 为什么生成式解码器不仅更快，而且在长 horizon 上精度反而略高于自回归解码。
- 完整的 PyTorch 实现思路，以及 Informer 在 ETT 和气象数据集上的实际表现。

**前置知识**：第 5 篇（Transformer 架构）、熟悉 Big-O 复杂度分析、了解基础信息论概念（如熵、KL 散度）。

---

## 长序列为何让原始 Transformer 崩溃

自注意力对每个查询 $q_i$ 的计算为：

$$
\mathrm{Attn}(q_i, K, V) = \sum_{j=1}^{L} \mathrm{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j.
$$

要为所有查询完成这一计算，必须先构建完整的 $L \times L$ 注意力分数矩阵。这里有三个主要开销，均随 $L^2$ 增长：

- $L$ 个维度为 $d$ 的 query-key 点积，总计 $L^2 d$ FLOPs；
- $L^2$ 次 softmax 运算；
- 反向传播时需存储 $L^2$ 个浮点数的注意力权重矩阵。

以 $L = 720$、$d = 64$、8 个注意力头、单样本为例：

- 每层每头的注意力分数条目数为 $720 \times 720 = 518\text{K}$；
- batch size 为 32 时，仅注意力权重就占用约 16 MB 显存（float32，8 头，3 层），反向传播中的激活值还会使显存需求再增加一个数量级；
- 每层每头的 FLOPs 约为 33 M，主要来自 $L^2 d$ 的矩阵乘法。

若将 $L$ 提升至 2160，每头的注意力条目数接近 500 万，这足以让一张 24 GB 显存的 GPU 在常规训练 batch size 下发生显存溢出（OOM）。

此前已有多种尝试缓解此问题：Longformer 和 BigBird 引入结构化稀疏（如局部+全局窗口或随机+全局连接），Linformer 和 Performer 则采用低秩近似。而 Informer 的思路截然不同：**让数据自己告诉我们哪些查询值得分配完整注意力资源**。

---

## ProbSparse：哪些查询重要，哪些不重要

### 直觉

若绘制典型查询 $q_i$ 在所有键上的注意力分布，通常会看到两种截然不同的形态：

- **尖峰型**：少数几个键占据了绝大部分概率质量。这类查询“目标明确”，知道该关注什么。
- **均匀型**：概率几乎均匀分布在所有键上。这类查询“模糊不清”，需要参考全部信息。

尖峰型查询可通过仅计算其 top 几个键的注意力来高效近似；均匀型则不行。关键挑战在于：**如何在不预先计算完整注意力矩阵的前提下，区分这两类查询？**

### 基于 KL 的稀疏性度量

衡量一个分布是否“尖锐”的自然方式，是将其与均匀分布通过 KL 散度进行比较。设

$$
p(k_j \mid q_i) = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right),
$$

则

$$
\mathrm{KL}(q_i \,\|\, U) = \log L + \frac{1}{L}\sum_{j=1}^{L} \log p(k_j \mid q_i).
$$

去掉常数项并代入 softmax 表达式后，Zhou 等人证明：

$$
\mathrm{KL}(q_i \,\|\, U) \;\propto\; \log\!\left(\sum_{j=1}^{L} e^{q_i^\top k_j / \sqrt{d}}\right) - \frac{1}{L}\sum_{j=1}^{L} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

将该量记为 $M(q_i, K)$。$M$ 越大，说明分布越尖锐——属于“选择性强”的查询，值得完整计算；$M$ 越小，则越接近均匀——可安全跳过。

但精确计算 $M$ 仍需 $L$ 次内积，违背了初衷。Informer 的第二项技巧是从 $u = c \log L$ 个键中**随机采样**（$c$ 为常数，通常取 5）来近似：

$$
\bar{M}(q_i, K) \;=\; \max_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}} \; - \; \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

此处用 $\max$ 替代 LogSumExp，依据是高维近高斯向量的“测度集中性”：LogSumExp 主要由最大项主导。实验表明，$\bar{M}$ 对查询的排序与精确 $M$ 几乎完全一致，但计算成本大幅降低。

### ProbSparse 实际计算的内容

单个注意力头的操作流程如下：

1. 从全部键中均匀随机采样 $u = c \log L$ 个；
2. 对每个查询 $q_i$ 计算 $\bar{M}(q_i, K)$，复杂度为 $\mathcal{O}(L \log L)$；
3. 按 $\bar{M}$ 值选出 top $u$ 个查询；
4. 对这 $u$ 个查询，在**全部** $L$ 个键上计算完整注意力；对其余 $L - u$ 个查询，输出直接设为 $V$ 的均值。

总计算和显存复杂度均为 $\mathcal{O}(L \log L)$。

![ProbSparse 注意力 vs 完整注意力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig1_probsparse_vs_full.png)

如图所示，右侧仅保留高 $M$ 查询对应的行。其余行并非置零，而是填充 $V$ 的均值——这对均匀注意力分布而言是一个合理且数学上正确的近似。

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

几个细节需注意：

- 公式 $u = c \ln L$ 使用自然对数；在 `numpy` 中应使用 `np.log` 并向上取整。
- “未选中查询用 $V$ 均值填充”并非 hack，而是数学上唯一满足“未选查询具有均匀注意力”约束且最大化熵的分布。
- 对于解码器的 masked self-attention，必须在 softmax 前屏蔽未来位置的键。上述实现采用 `-1e9` 是标准做法。

---

## 编码器蒸馏：金字塔式序列压缩

即便使用 ProbSparse，三层编码器每层处理 $L = 720$ 的序列依然昂贵。Informer 在编码器层间引入**蒸馏**操作，将序列长度减半：

$$
X_{\ell+1} = \mathrm{MaxPool}_{k=3, s=2}\!\Big(\mathrm{ELU}\big(\mathrm{Conv1d}_{k=3, s=2}(X_\ell)\big)\Big).
$$
\nstride=2 的 Conv1d 作为可学习的下采样器，MaxPool 保留相邻位置中的主导值，中间的 ELU 非线性激活则赋予该操作超越纯池化的表达能力。

![编码器蒸馏金字塔](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig3_encoder_distilling.png)

效果是累积的：一个 3 层编码器将 720 步输入依次压缩为 $720 \to 360 \to 180 \to 90$。显存占用随深度呈几何级下降，而非线性增长。由于底层能看到更长的历史，顶层的感受野轻松覆盖数千个原始时间步。

需注意两点：

- **最后一层不应蒸馏**。解码器的交叉注意力直接读取编码器输出；若最后一层也蒸馏，分辨率再次减半，会丢失关键信息。标准做法是“除最后一层外，每层后都进行蒸馏”。
- **并行双编码器提升鲁棒性**。原论文同时运行两个编码器：一个处理完整输入，另一个处理半长输入，最后拼接输出。这种冗余设计可避免因特定序列的蒸馏决策失误导致性能下降。

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

---

## 生成式解码器：一次搞定整个预测范围

标准 Transformer 解码器采用自回归方式：先预测 $\hat{y}_1$，将其作为输入再预测 $\hat{y}_2$，依此类推。若预测 horizon 为 $H = 168$，则需 168 次顺序前向传播。这不仅带来高延迟，还会导致误差累积——第 5 步的错误会作为输入影响第 6 步的预测。
\nInformer 的生成式解码器另辟蹊径。其输入构造为：

$$
X_\text{dec} = \big[\, X_\text{token} \;;\; X_0 \,\big],
$$

其中 $X_\text{token}$ 是编码器输入的最后 `label_len` 个时间步（作为“提示”），$X_0$ 是 `out_len` 个占位符（通常为零向量）。解码器**仅需一次前向传播**处理整个 $\text{label\_len} + \text{out\_len}$ 序列，最后 `out_len` 个输出即为预测结果。

![自回归解码器 vs 生成式解码器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig2_generative_decoder.png)

该设计带来三大优势：

- **速度极快**：从 $H$ 次前向传播降至 1 次，推理延迟减少 $H$ 倍。
- **无误差累积**：所有预测均基于同一编码器上下文，彼此独立。
- **长 horizon 更准**：反直觉的是，生成式解码器在长预测任务上往往比自回归更准确。原因在于，自回归被迫进行“短视”优化——每步训练都假设前序预测完美无缺；而生成式直接联合优化整个预测序列。

`label_len` 至关重要：它为解码器提供了若干“真实”数据点作为锚点，帮助占位符找到正确方向。经验表明，`label_len = out_len / 2` 效果最佳。

---

## 拼起来：Informer 完整模型

完整模型由编码器和解码器组成，嵌入层融合了数值、位置和时间特征信息。

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

训练时，解码器输入由最后 `label_len` 个真实值与 `out_len` 个零占位符拼接而成：

```python
def build_decoder_input(x_enc, label_len, out_len):
    # x_enc: (B, seq_len, F)
    start = x_enc[:, -label_len:, :]
    placeholder = torch.zeros(
        x_enc.size(0), out_len, x_enc.size(-1), device=x_enc.device
    )
    return torch.cat([start, placeholder], dim=1)
```

---

## 长 horizon 表现

主图展示了真实值、原始 Transformer 与 Informer 在 480 步预测上的对比。

![长 horizon 预测：原始 Transformer 漂移，Informer 贴合真实值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig4_long_sequence_forecast.png)

原始 Transformer 的自回归误差从第 100 步左右开始显著累积，预测逐渐偏离真实轨迹。而 Informer 的生成式解码器因所有输出 token 联合优化，能在整个窗口内保持连贯预测。

在经典的 ETT（Electricity Transformer Temperature）基准测试中，论文报告了如下结果：

![ETTh1 单变量 MSE 与 L = 720 时的资源消耗](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer长序列预测/fig5_ett_benchmark.png)

两个关键结论：

- **长 horizon 精度优势**：在 horizon=720 时，Informer 的 MSE 为 0.235，优于原始 Transformer 的 0.269。虽然绝对差距不大，但此时原始 Transformer 已接近可训练极限。
- **资源效率碾压**：当 $L = 720$ 时，Informer 在单张 V100 上峰值显存仅 1.8 GB，每轮训练耗时 9.5 秒；而原始 Transformer 需 10.5 GB 显存和 104 秒。正是这一差距，让 Informer 成为长序列预测的实用之选。

---

## 超参速查表

| 超参 | 默认值 | 备注 |
|---|---|---|
| `d_model` | 512 | 标准配置；显存紧张时可降至 256 |
| `n_heads` | 8 | $d_k = d_\text{model} / n_\text{heads}$，此处为 64 |
| `e_layers` | 3 | 层数越多蒸馏越激进，但收益递减 |
| `d_layers` | 2 | 非对称设计，编码器承担主要计算 |
| `d_ff` | 2048 | 通常为 `d_model` 的 4 倍 |
| `factor`（$u = c \log L$ 中的 $c$） | 5 | 求速度用 3，求精度用 7，一般无需调整 |
| `seq_len` | 96 到 720 | 大致对应数据的周/月周期长度 |
| `label_len` | `out_len / 2` | 解码器锚点，切勿设为 0 |
| `out_len` | 任务驱动 | 实际需预测的步数 |
| `dropout` | 0.05–0.1 | 小数据集可适当调高 |
| 优化器 | Adam，lr 1e-4 | 学习率约为标准 Transformer 的一半 |
| LR 调度 | StepLR，每 30 epoch γ=0.5 | 或使用 cosine annealing |
| 损失函数 | MSE | L1 (MAE) 对异常值更鲁棒 |

---

## 常见问题

- **忘记为解码器 self-attention 添加因果掩码**：若无掩码，训练时解码器会“偷看”未来的占位符，导致训练表现虚高、测试彻底失效。
- **对过短输入启用蒸馏**：若 `seq_len` 很小（如 24），三层蒸馏会将序列压缩至长度 3，丢失大部分上下文。建议 `seq_len < 96` 时关闭蒸馏。
- **忽略逐窗口标准化**：与 N-BEATS 类似，应在输入模型前对每个窗口标准化，输出后再逆变换。
- **未编码时间特征**：hour-of-day、day-of-week、month 等时间嵌入对性能至关重要；缺少它们，模型只能从原始数值中艰难推断时间信息。
- **`label_len` 过小**：某些实现默认 `label_len = 0`，这会破坏解码器的锚定机制。论文推荐 `label_len = out_len / 2`。

---

## 什么时候不该用 Informer

- **短序列（$L < 96$）**：ProbSparse 和蒸馏有固定开销，此时原始 Transformer 或 LSTM 更简单高效。
- **跨特征交互是主要信号**（多变量且特征间强依赖）：Informer 的注意力沿时间轴展开；若需跨特征注意力，应考虑 TFT 或 TimesNet。
- **需完整注意力图用于解释**：ProbSparse 会丢弃非选中查询的注意力行；若必须可视化全图，请用原始 Transformer。
- **高频流式推理**（kHz/MHz 级）：Informer 面向批量预测，流式场景需专用架构。
- **极小数据集（<1k 样本）**：Informer 参数量达数千万，极易过拟合；此时应选用更轻量模型。

---

## Q&A

### 为什么是 $u = c \log L$？

该设定源于概率分析：当 $u = c \log L$ 且 $c = 5$ 时，任意查询漏掉其 top-1 键的概率不超过 $1/L^4$。实践中 $c = 3$ 也足够有效。

### ProbSparse 能选出正确的查询吗？

实验证明可以。$\max - \mathrm{mean}$ 近似值与精确 KL 散度的 Spearman 相关系数超过 0.95。论文提供了完整消融研究。

### 为何非选中查询用 $V$ 均值而非零？

因为均匀注意力分布的期望输出正是 $\frac{1}{L}\sum_j v_j$。对被判定为“均匀”的查询，均值是数学上最合理的填充值。

### Informer 与 Reformer / Performer / Linformer 有何不同？

- **Reformer**：使用 LSH 桶化注意力，复杂度 $\mathcal{O}(L \log L)$，但桶化与数据无关。
- **Performer**：采用随机特征核近似，复杂度 $\mathcal{O}(L)$，但在注意力尖锐的长序列上精度下降。
- **Linformer**：将键/值投影至固定低秩空间，复杂度 $\mathcal{O}(L)$，但投影在训练时即固定。
- **Informer**：根据数据自适应选择查询，复杂度 $\mathcal{O}(L \log L)$，在时序基准上精度最优。

### 多变量输入时，编码器和解码器特征维度能否不同？

可以。`enc_in` 与 `dec_in` 独立。常见做法是将所有变量输入编码器，仅目标变量输入解码器。

### Autoformer 和 FEDformer 呢？

二者均为 Informer 的直接后继。Autoformer（2021）用自相关替代 self-attention 并引入显式分解；FEDformer（2022）加入频域注意力。两者在相同基准上表现更优，但实现更复杂——Informer 仍是理想的入门起点。

### 是否需在大规模多序列数据上预训练？

有帮助但非必需。与 NLP 不同，时序数据集间领域差异大，盲目预训练常适得其反。针对具体任务从头训练通常是更优默认选择。

---

## 小结
\nInformer 是首个让 Transformer 在长 horizon 时间序列预测中真正实用的架构。其三大核心创新——ProbSparse 自注意力、编码器蒸馏和生成式解码器——共同构成一个端到端 $\mathcal{O}(L \log L)$ 系统，在所有长序列基准上，无论精度还是实际运行速度，均全面超越原始 $\mathcal{O}(L^2)$ Transformer。

对于 $L > 96$ 且仅使用单 GPU 的预测任务，Informer 是当之无愧的首选起点。后续架构（如 Autoformer、FEDformer、PatchTST）虽进一步优化了方案，但都建立在 Informer 的两大洞见之上：**并非每个查询都需要完整注意力**，以及**自回归解码实为自我设限的瓶颈**。

至此，时间序列预测系列正式完结。八章内容从经典 ARIMA 出发，历经 LSTM、Transformer、TCN、N-BEATS，最终抵达 Informer。请根据你的数据特性选择合适架构，关键任务可考虑集成，并始终铭记：面对小规模问题，简洁的基线模型往往胜过时髦的复杂方案。

---

## 参考资料

- Zhou, H. et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI Best Paper.
- Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.
- Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.* ICML.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR.
- Wang, S. et al. (2020). *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768).
- Choromanski, K. et al. (2021). *Rethinking Attention with Performers.* ICLR.

>
>
> 本文是时间序列模型系列的**第 8 篇**，共 8 篇（系列完结）。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - [第 2 篇：LSTM —— 门控机制与长期依赖](/zh/time-series/02-lstm)
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn)
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - **第 8 篇：Informer —— 高效长序列预测**（当前）
