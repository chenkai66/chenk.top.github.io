---
title: "时间序列模型（八）：Informer -- 高效长序列预测"
date: 2024-11-08 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Informer
  - Transformer
categories: 时间序列
series:
  name: "时间序列模型"
  part: 8
  total: 8
lang: zh-CN
mathjax: true
description: "Informer 用 ProbSparse 注意力、编码器蒸馏、生成式解码器把 Transformer 复杂度从 O(L^2) 降到 O(L log L)。完整数学推导、PyTorch 代码与 ETT/气象 benchmark。"
disableNunjucks: true
---

> **系列**：时间序列模型 -- 第 8 部分，共 8 部分
> [<-- 上一篇：N-BEATS](/zh/时间序列模型-七-N-BEATS深度架构/)

Transformer 做序列建模非常好用——直到序列变长。原始自注意力的算力和显存都是 $\mathcal{O}(L^2)$，所以一周小时级窗口（168 步）还行，一个月窗口（720 步）就开始痛苦，三个月窗口（2160 步）在单张 GPU 上基本不可能。而真实的长 horizon 预测——气象、能源、金融、IoT——恰好都在这个区间。

**Informer**（Zhou 等人，AAAI 2021 最佳论文）就是终于让 Transformer 在这种场景下可用的架构。它做了三件事，每一件都值得算一个独立贡献：

1. **ProbSparse 自注意力**只保留 $\mathcal{O}(\log L)$ 个最有信息量的查询，每层代价从 $\mathcal{O}(L^2)$ 降到 $\mathcal{O}(L \log L)$。
2. **自注意力蒸馏**在编码器层之间把序列长度减半，显存随深度几何级缩减。
3. **生成式解码器**一次前向就预测整个 horizon，不再走 $H$ 步自回归。

三者合在一起，长 horizon ETT/气象/电力 benchmark 上能带来 6-10 倍提速、5-10% 的 MSE 改进。本章把每一个机制的数学拆开，并走通实现。

## 这一篇你会学到

- 原始自注意力在长序列上 $\mathcal{O}(L^2)$ 的具体痛点。
- ProbSparse 用 KL 散度衡量稀疏度，再用 $\max - \mathrm{mean}$ 近似它。
- 编码器蒸馏怎么用"序列长度换深度"而不丢主导模式。
- 为什么生成式解码器既比自回归快**又**更准。
- 一份完整的 PyTorch 实现，外加 Informer 在 ETT 和气象 benchmark 上的成绩单。

**前置知识**：第 5 部分（Transformer 架构）。会用 Big-O 推理，了解基本信息论（熵、KL 散度）。

---

## 长序列为什么把原始 Transformer 拖死

自注意力对每个查询 $q_i$ 做：

$$
\mathrm{Attn}(q_i, K, V) = \sum_{j=1}^{L} \mathrm{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j.
$$

要对所有查询都算，就需要完整的 $L \times L$ 分数矩阵。三个代价都是 $\mathcal{O}(L^2)$：

- $L$ 个 $d$ 维查询-键点积：$L^2 d$ FLOPs。
- $L^2$ 次 softmax。
- 反向传播时存储 $L^2$ 个浮点的注意力矩阵。

具体到 horizon $L = 720$、$d = 64$、8 个头、单样本：

- 注意力分数：每头每层 $720 \times 720 = 518\text{K}$ 项。
- 显存：batch 32 时仅注意力权重就 ~16 MB（float32、8 头、3 层）；反向传播时活值要再大一个数量级。
- FLOPs：每头每层 ~33 M，主要在 $L^2 d$ 矩阵乘上。

推到 $L = 2160$，每头近 5 M 项，24 GB 卡在常用 batch 下直接 OOM。

之前一些工作用结构化稀疏（Longformer 的局部 + 全局窗口、BigBird 的随机 + 全局）或低秩近似（Linformer、Performer）来攻这个问题。Informer 的思路不同：**让数据告诉你哪些查询配得上完整注意力**。

---

## ProbSparse：哪些查询要紧、哪些不要紧

### 直觉

把一个典型查询 $q_i$ 在所有键上的注意力分布画出来，会看到两种性质完全不同的形状：

- **尖**：少数几个键拿走绝大部分概率。这个查询很"挑"——它知道自己在找什么。
- **平**：概率均匀分布在所有键上。这个查询很"模糊"——它需要看全部。

尖的查询可以只算 top 几个键就近似得很好；平的不行。诀窍是要在**不算完整注意力矩阵的前提下**就把这两类区分开。

### 基于 KL 的稀疏度度量

衡量分布有多尖，自然的做法是和均匀分布比 KL 散度。设

$$
p(k_j \mid q_i) = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right).
$$

那么

$$
\mathrm{KL}(q_i \,\|\, U) = \log L + \frac{1}{L}\sum_{j=1}^{L} \log p(k_j \mid q_i).
$$

去掉常数项、代入 softmax 之后，Zhou 等人证明

$$
\mathrm{KL}(q_i \,\|\, U) \;\propto\; \log\!\left(\sum_{j=1}^{L} e^{q_i^\top k_j / \sqrt{d}}\right) - \frac{1}{L}\sum_{j=1}^{L} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

记这个量为 $M(q_i, K)$。$M$ 大说明分布尖——挑剔查询，配得上完整注意力；$M$ 小说明分布平——模糊查询，可以跳过。

但精确算 $M$ 仍要 $L$ 个内积，绕回原点了。Informer 的第二招是**从 $u = c \log L$ 个键随机采样**来近似 $M$（$c$ 是常数，通常 5）：

$$
\bar{M}(q_i, K) \;=\; \max_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}} \;-\; \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

这里 $\max$ 替代 LogSumExp 的依据是测度集中：高维近高斯向量上 LogSumExp 由最大项主导。实证上 $\bar{M}$ 给查询的排序和精确 $M$ 几乎一致，代价却小得多。

### ProbSparse 实际算什么

单个注意力头的流程：

1. 均匀随机采样 $u = c \log L$ 个键。
2. 对每个查询 $q_i$ 算 $\bar{M}(q_i, K)$——$\mathcal{O}(L \log L)$。
3. 按 $\bar{M}$ 选出 top $u$ 个查询。
4. 对这 $u$ 个查询，在**全部** $L$ 个键上算注意力。剩下 $L - u$ 个查询用 $V$ 的均值填充输出。

总成本：$\mathcal{O}(L \log L)$。显存：也是 $\mathcal{O}(L \log L)$。

![ProbSparse 注意力 vs 完整注意力](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/fig1_probsparse_vs_full.png)

图里右侧只保留高 $M$ 查询对应的行。其他行实际上不是零——它们填的是 $V$ 的均值，对均匀注意力分布来说这是合理近似。

干净的实现：

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

        # 2) 稀疏度 M_bar(q, K_sample) = max - mean
        Q_K_s = torch.matmul(Q, K_sample.transpose(-2, -1))         # (B, H, L_Q, u)
        M_bar = Q_K_s.max(dim=-1).values - Q_K_s.mean(dim=-1)       # (B, H, L_Q)

        # 3) 选 top-n_top 查询
        top_idx = M_bar.topk(n_top, dim=-1).indices                 # (B, H, n_top)

        # 4) 仅对 top 查询算完整注意力
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

- $u = c \ln L$ 是自然对数；用 `numpy` 的 `np.log` 然后向上取整。
- "未选中查询用 V 的均值填充"是数学上正确的处理而不是 hack。它对应在"未选查询是均匀注意力"约束下熵最大的唯一分布。
- 解码器的 masked self-attention 必须在 softmax **之前**屏蔽未来键。上面的实现用 `-1e9` 这一标准技巧。

---

## 编码器蒸馏：金字塔式序列压缩

即便有了 ProbSparse，三层都对 $L = 720$ 操作的编码器仍然贵。Informer 在编码器层之间加了一个**蒸馏**操作，把序列长度减半：

$$
X_{\ell+1} = \mathrm{MaxPool}_{k=3, s=2}\!\Big(\mathrm{ELU}\big(\mathrm{Conv1d}_{k=3, s=2}(X_\ell)\big)\Big).
$$

stride=2 的 Conv1d 是一个学到的下采样器；MaxPool 在每对相邻位置上保留主导值；中间的 ELU 给这个算子加上比纯池化更多的表达力。

![编码器蒸馏金字塔](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/fig3_encoder_distilling.png)

效果会复利：3 层编码器把 720 步输入按 $720 \to 360 \to 180 \to 90$ 缩。显存对深度几何衰减而非线性增长。同时由于底层看到更长的历史，顶层的有效感受野能轻松覆盖几千个原始时间步。

两个要小心：

- **最后一层不蒸馏**。解码器交叉注意力读编码器输出；如果最后一层也蒸馏，分辨率再砍一半，丢信息。标准做法是"除最后一层外每层都蒸馏"。
- **两个并行编码器更鲁棒**。原论文跑两个编码器，一个全长输入，一个半长输入，最后拼接输出。这种冗余能对冲个别序列上不走运的蒸馏决策。

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

## 生成式解码器：horizon 一次出

原始 Transformer 解码器自回归生成：先预测 $\hat{y}_1$，喂回去，再预测 $\hat{y}_2$……horizon $H = 168$ 就是 168 次顺序前向。先不说延迟，错误也会复利：第 5 步错了会作为第 6 步的输入。

Informer 的生成式解码器换了思路。解码器输入构造为

$$
X_\text{dec} = \big[\, X_\text{token} \;;\; X_0 \,\big],
$$

其中 $X_\text{token}$ 是编码器输入的最后 `label_len` 个时间步（充当"提示"），$X_0$ 是 `out_len` 个占位 token（通常是适当维度的零向量）。解码器在整个 $\text{label\_len} + \text{out\_len}$ 序列上跑**一次**，最后 `out_len` 个输出就是预测。

![自回归解码器 vs 生成式解码器](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/fig2_generative_decoder.png)

三个收益：

- **速度**：$H$ 次顺序前向 $\to$ 1 次前向。推理延迟降低 $H$ 倍。
- **不复利错误**：所有 horizon 预测都从同一份编码器上下文产生，没有谁依赖前一步预测。
- **长 horizon 反而更准**。直觉上反，但确实如此：自回归解码器被迫"短视"地优化——每步训练都假设前面预测都完美。生成式直接联合优化整个预测向量。

label tokens 至关重要：它们给解码器一段"真实"数据点作为开端，把后面的占位 token 锚定住。经验上 `label_len = out_len / 2` 效果好。

---

## 拼起来：Informer 完整模型

完整模型 = 编码器 + 解码器，加上把数值、位置、时间特征结合起来的嵌入。

```python
class TemporalEmbedding(nn.Module):
    """把 (hour, day_of_week, month) 嵌入到 d_model。"""

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

构造训练数据时，解码器输入 = 最后 `label_len` 个真实值 + `out_len` 个零占位：

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

主图：真实值、原始 Transformer、Informer 在 480 步 horizon 上的对比。

![长 horizon 预测：原始 Transformer 漂走，Informer 贴住真值](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/fig4_long_sequence_forecast.png)

原始 Transformer 的自回归误差在 100 步左右开始复利，越走越偏。Informer 一次出全部 horizon，所有输出 token 联合优化，整个窗口都给出连贯预测。

经典 ETT（Electricity Transformer Temperature）benchmark 上的论文数字：

![ETTh1 单变量 MSE 与 L = 720 时的资源开销](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/08-Informer%E9%95%BF%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B/fig5_ett_benchmark.png)

两个要点：

- **长 horizon 精度**：horizon 720 时 Informer MSE 0.235 vs 原始 Transformer 0.269。绝对差距不算大，但原始 Transformer 在 720 已经在能训的边缘。
- **资源开销**：$L = 720$ 时 Informer 单 V100 上峰值显存 1.8 GB、9.5 s/epoch；原始 Transformer 要 10.5 GB、104 s/epoch。这就是 Informer 存在的全部理由。

---

## 超参 cheat sheet

| 超参 | 默认 | 备注 |
|---|---|---|
| `d_model` | 512 | 标准；显存紧用 256。 |
| `n_heads` | 8 | $d_k = d_\text{model} / n_\text{heads}$，所以这里 64。 |
| `e_layers` | 3 | 更多层 = 更激进蒸馏；很少值得加。 |
| `d_layers` | 2 | 非对称：编码器干重活。 |
| `d_ff` | 2048 | 4 × `d_model` 是标准。 |
| `factor`（$u = c \log L$ 中的 $c$） | 5 | 3 求快，7 求准；很少改。 |
| `seq_len` | 96-720 | 大约等于你的周/月周期长度。 |
| `label_len` | `out_len / 2` | 锚定解码器；不要置零。 |
| `out_len` | 任务驱动 | 你真正要预测的步数。 |
| `dropout` | 0.05-0.1 | 小数据集调大。 |
| 优化器 | Adam，lr 1e-4 | 比"普通" Transformer 学习率小一半。 |
| LR 调度 | StepLR，每 30 epoch γ=0.5 | 或 cosine annealing。 |
| 损失 | MSE | L1 (MAE) 对离群点更鲁棒。 |

---

## 常见坑

- **忘了给解码器 self-attention 加 mask**。没有因果 mask，解码器训练时能偷看后面的占位 token。训练时表现"奇迹"，测试时垃圾。
- **太短的输入也蒸馏**。`seq_len` 短（比如 24）时，三层蒸馏把编码器输出压到 3 步，丢光上下文。`seq_len < 96` 时关掉蒸馏。
- **跳过逐窗口标准化**。和 N-BEATS 同样的问题：每窗口标准化后再喂模型，输出端逆变换。
- **不编码时间特征**。时间嵌入（hour-of-day、day-of-week、month）对性能贡献很大；没它模型还要从原始数据里推时刻。
- **`label_len` 太小**。某些实现默认 `label_len = 0`，把解码器锚定彻底破坏。论文用 `label_len = out_len / 2`。

---

## 什么时候**不**用 Informer

- **短序列（$L < 96$）**。ProbSparse 和蒸馏有常数开销；短序列下原始 Transformer 或 LSTM 更简单也更快。
- **跨特征交互才是主信号**（多变量、强特征间依赖）。Informer 沿时间轴 attend；要跨特征 attention 看 TFT 或 TimesNet。
- **要求精确注意力图做可解释性**。ProbSparse 把未选查询的逐行 attention map 丢了。要可视化完整 attention 就用原始 Transformer。
- **高频流式推理**（kHz、MHz）。Informer 是为批预测设计的；流式要更专门的架构。
- **极小数据集（<1k 样本）**。Informer 几千万参数会过拟合，用更小的模型。

---

## Q&A

**为什么偏要 $u = c \log L$？**
界来自"挑剔查询的最强键被采样到"的概率。$u = c \log L$、$c = 5$ 时，对任何给定查询漏掉 top-1 键的概率 $\leq 1/L^4$。实际 $c = 3$ 也行。

**ProbSparse 真的能挑对查询吗？**
经验上能——$\max - \mathrm{mean}$ 近似与精确 KL 在训练中观察到的注意力分布上 Spearman 相关 >0.95。论文有完整消融。

**为什么未选查询要用 $V$ 的均值而不是零？**
因为均匀注意力分布的输出正好是 $\frac{1}{L}\sum_j v_j$。"判定为均匀注意力"的查询用均值填，是解析上正确的。

**Informer 与 Reformer / Performer / Linformer 的区别？**
- **Reformer**：LSH 桶化注意力。$\mathcal{O}(L \log L)$，但桶化与数据无关。
- **Performer**：随机特征核近似。$\mathcal{O}(L)$，但长序列上注意力陡峭时精度退化。
- **Linformer**：把键/值投到固定低秩维。$\mathcal{O}(L)$，但投影在训练时定死。
- **Informer**：基于数据自适应选查询。$\mathcal{O}(L \log L)$，在时间序列 benchmark 上精度保留最好。

**多变量时编码器和解码器特征维度可以不同吗？**
可以——`enc_in` 和 `dec_in` 独立。常见模式：所有变量喂编码器，仅目标变量喂解码器。

**Autoformer / FEDformer 呢？**
都是直接后继。Autoformer（2021）把 self-attention 换成沿序列的自相关并显式加一层分解。FEDformer（2022）加了频域 attention。两者在同样 benchmark 上都比 Informer 好但实现更复杂；Informer 仍是合适的起点。

**要不要先在多序列大数据集上预训练？**
有用但非必须。和 NLP 不同，时间序列数据集之间的领域差距很大，朴素预训练经常起反作用。从零按领域微调通常是更好的默认。

---

## 小结

Informer 是让 Transformer 真正能用于长 horizon 时间序列预测的架构。三个核心机制——ProbSparse 自注意力、编码器蒸馏、生成式解码器——组合成端到端 $\mathcal{O}(L \log L)$ 的系统，在所有长 horizon benchmark 上都比原始 $\mathcal{O}(L^2)$ Transformer 在精度和墙钟时间上都赢。

单卡上 horizon $L > 96$ 的预测任务，Informer 是显然的起点。后来的架构（Autoformer、FEDformer、PatchTST）继续优化这个配方，但每一个都建立在 Informer 的核心观察之上：**不是每个查询都需要完整 attention**、**自回归解码是自找的瓶颈**。

时间序列预测系列到此完结。八章里我们从经典 ARIMA 走到 LSTM、Transformer、TCN、N-BEATS、Informer；选择匹配你数据的架构、关键时刻做集成，并记住一件事——简单 baseline 经常在小问题上击败时髦模型。

---

## 参考资料

- Zhou, H. et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI Best Paper.
- Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.
- Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.* ICML.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR.
- Wang, S. et al. (2020). *Linformer: Self-Attention with Linear Complexity.* arXiv:2006.04768.
- Choromanski, K. et al. (2021). *Rethinking Attention with Performers.* ICLR.

---

**系列导航**

| | |
|---|---|
| **上一篇** | [N-BEATS](/zh/时间序列模型-七-N-BEATS深度架构/) |
| **当前** | 第八部分：Informer（系列完结） |
