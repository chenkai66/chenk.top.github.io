---
title: "时间序列模型（五）：时间序列的 Transformer 架构"
date: 2024-11-05 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Transformer
categories: 时间序列
series:
  name: "时间序列模型"
  part: 5
  total: 8
lang: zh-CN
mathjax: true
description: "时间序列的 Transformer 全景：编码器-解码器结构、时序位置编码、O(n^2) 注意力瓶颈、Decoder-only 自回归预测与 Patching 策略。含 Autoformer / FEDformer / Informer / PatchTST 选型与可直接运行的实现。"
disableNunjucks: true
---
> **系列**：时间序列模型 -- 第 5 部分，共 8 部分
> [<-- 上一篇：Attention 机制](/zh/时间序列模型-四-Attention机制/) | [下一篇：TCN -->](/zh/时间序列模型-六-时序卷积网络TCN/)

## 本章要点

- 把完整的 encoder-decoder Transformer 拆给时间序列重新讲一遍
- 为什么必须注入位置信息，正弦 / 学习式 / 时间感知三种编码的差异
- 多头注意力在时间序列上到底学到了什么
- 朴素 attention 在哪儿撞墙（O(n²)），以及四类解决方案：稀疏 / 线性 / Patching / Decoder-only
- 一份干净的 PyTorch 参考实现，附 Autoformer / FEDformer / Informer / PatchTST 的选型建议

## 前置知识

- 自注意力与多头注意力（第 4 篇）
- 编码器-解码器结构与 teacher forcing
- PyTorch 基础（`nn.Module`、训练循环）

---

## 1. 为什么时间序列要用 Transformer

LSTM / GRU 一步一步地处理序列，由此带来三个问题：

1. **路径长度是 O(L)**。从 $t-L$ 步的信息要传到 $t$ 步，必须穿过 $L$ 次循环——这正是梯度消失的根源。
2. **训练是串行的**。第 $t+1$ 步必须等第 $t$ 步算完，GPU 一半算力闲着。
3. **隐状态是瓶颈**。模型必须把所有可能用到的历史压进一个固定大小的向量里。

自注意力一口气把这三件事都解决了：每个位置在 **一次矩阵乘法** 里就能看见其它所有位置，任意两步之间的路径长度是 $O(1)$，整条序列并行处理。代价是显存：要存 $n \times n$ 的注意力矩阵，开销是 $O(n^2)$，第 5、第 7 节会专门处理。

![时间序列适配版的编码器-解码器 Transformer。编码器并行读取 lookback 窗口，解码器生成预测窗口，并通过 cross-attention 关注编码器的 memory。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig1_architecture.png)
*图 1. 时间序列适配版的编码器-解码器 Transformer。编码器并行读取 lookback 窗口，解码器生成预测窗口，并通过 cross-attention 关注编码器的 memory。*

## 2. 架构逐块拆解

时间序列的 Transformer 就是 2017 年的原版，外加三处不大但重要的改动：

| 模块         | 原版 NLP            | 时间序列                                |
|--------------|---------------------|-----------------------------------------|
| 输入嵌入     | token embedding 查表 | 连续特征的**线性投影**                  |
| 位置信息     | 基于 token 索引的正弦 | **时间感知**编码（日历特征 / 不等间隔） |
| 输出头       | 词表 softmax        | 输出实数预测向量的**线性层**            |

其余结构——多头自注意力、前馈、残差、LayerNorm、解码器 cross-attention、因果掩码——一字不改。每个 block 的四个子层是：

$$
\begin{aligned}
h_1 &= \text{LayerNorm}(x + \text{MHSA}(x)) \\
h_2 &= \text{LayerNorm}(h_1 + \text{FFN}(h_1))
\end{aligned}
$$

注意力依然是第 4 篇里的缩放点积：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

### 2.1 编码器

读 lookback 窗口 $x_{t-L+1:t}$，输出上下文向量 $M \in \mathbb{R}^{L \times d_{\text{model}}}$。**不带 mask**，每个位置看所有位置。

### 2.2 解码器

输入是**标签窗口**（历史末尾的 $L_{\text{label}}$ 步）拼上 $H$ 个零占位，输出是预测 $\hat{y}_{t+1:t+H}$。每个 block 用两层 attention：

- **Masked self-attention**，加因果 mask，第 $t+k$ 步只能看到 $\le t+k-1$ 的位置。
- **Cross-attention**，Query 来自解码器，Key / Value 来自编码器 memory $M$。这是解码器唯一能看见编码器的地方。

### 2.3 标签窗口：一个小技巧

纯 encoder-decoder 经常在历史 / 预测交界处崩。Informer / Autoformer 的解法是给解码器 $L_{\text{label}}$ 步**已知历史** + $H$ 个零占位，让解码器从一个已知状态出发，平滑地推进到未知。

## 3. 时间序列的位置编码

自注意力是排列不变的——把输入打乱，输出不变。NLP 里这是 bug，时间序列里这是灾难。我们用正弦编码注入位置：

$$
\text{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
\text{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$

每个位置 $p$ 拿到一个由几何级数频率构成的**唯一签名**。低维分量震荡快（编码短程位置），高维分量震荡慢（编码长程位置）。

![正弦位置编码。左：完整编码矩阵，每一行都是唯一签名。右：四个代表性维度，频率各不相同。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig2_positional_encoding.png)
*图 2. 正弦位置编码。左：完整编码矩阵，每一行都是唯一签名。右：四个代表性维度，频率各不相同。*

时间序列通常需要比"步索引"**更丰富**的位置信息：

- **日历特征**：小时、星期、月份、节假日标志，每个都给独立的可学习 embedding，加到输入上。
- **不等间隔采样**：把位置 $p$ 替换成实际时间戳 $\tau_p$ 并归一化。Time2Vec、Continuous-Time Transformer 走的是这条路。
- **相对位置**：把 $\tau_q - \tau_k$ 直接加到 attention score 里（T5 / TUPE 风格），更适合超长上下文。

```python
import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    """正弦 PE + 可选的日历特征 embedding。"""

    def __init__(self, d_model: int, max_len: int = 5000,
                 calendar_sizes=(24, 7, 31, 12)):
        super().__init__()
        # ---- 正弦部分 ------------------------------------------------------
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

        # ---- 日历 embedding（小时 / 星期 / 日 / 月）-----------------------
        self.cal_embeds = nn.ModuleList(
            nn.Embedding(n, d_model) for n in calendar_sizes
        )

    def forward(self, x: torch.Tensor, cal: torch.Tensor | None = None):
        # x: (B, L, d). cal: (B, L, 4) 整型日历特征。
        out = x + self.pe[:, : x.size(1)]
        if cal is not None:
            for i, emb in enumerate(self.cal_embeds):
                out = out + emb(cal[..., i])
        return out
```

## 4. 多头注意力到底学到了什么

单头只能建模一种关系。多头把模型切成 $h$ 路并行注意力，每路在 $d_k = d_{\text{model}} / h$ 维上算完再拼起来。在时间序列上，不同的头通常会专门化：

![训练好的 Transformer 在 48 步窗口上的四个头，每个学到不同的时间模式。注意因果 mask：对角线之上没有任何权重。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig3_multihead_patterns.png)
*图 3. 训练好的 Transformer 在 48 步窗口上的四个头，每个学到不同的时间模式。注意因果 mask：对角线之上没有任何权重。*

| 头的模式            | 模型在做什么                                |
|---------------------|---------------------------------------------|
| **局部**（对角线）  | 基本就是自回归滑动平均                      |
| **周期条纹**        | 锁定某个已知周期（24 小时 / 周）            |
| **长程弥散**        | 拉取慢速趋势                                |
| **锚定带**          | 抓住某个具体的过去事件（峰值、状态切换）    |

这也是 Transformer **可解释性** 的来源：把最后一层的 attention 在头维度上平均，就能看出预测到底依赖于哪几步历史。

## 5. O(n²) 瓶颈

朴素 attention 每层每个头都要存一张 $n \times n$ 的 score 矩阵。fp16 + 8 头时，每层注意力的显存是

$$
M_{\text{attn}} = h \cdot n^2 \cdot 2 \;\text{bytes}.
$$

$n=512$ 时 4 MB（无所谓）；$n=4096$ 时 256 MB（开始难受）；$n=16384$ 时单层就要 4 GB+ 仅放注意力矩阵。计算量同样是 $O(n^2 d_{\text{model}})$ FLOPs。

![注意力的显存与 FLOPs 随序列长度的变化。朴素 O(n²) 在几千步之后就跑不动了；稀疏 / 线性 / Patching 三类方案能把开销压回可接受范围。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig5_quadratic_bottleneck.png)
*图 5. 注意力的显存与 FLOPs 随序列长度的变化。朴素 O(n²) 在几千步之后就跑不动了；稀疏 / 线性 / Patching 三类方案能把开销压回可接受范围。*

四类解法，按"对模型改动多少"递增排序：

1. **稀疏注意力**（Longformer、BigBird、Informer 的 ProbSparse）：只算 $(q, k)$ 对的稀疏子集。开销 $O(n \cdot w)$，$w$ 是窗口或选中的 key 数。
2. **线性注意力**（Performer、Linformer、Nystromformer）：把 softmax 换成可分解的核函数，attention 降到 $O(n \cdot d^2)$。
3. **Patching**（PatchTST、Autoformer 风格的序列分解）：直接把序列**变短**，把连续的几步合成一个 patch。第 7 节专讲。
4. **Decoder-only + KV cache**（第 6 节）：训练仍是 $O(n^2)$，但推理可以增量化。

实战经验：lookback 在 2k 以内、horizon 在几百步以内时，朴素 attention 完全够用。再长就上 patching——**性价比最高的一招**，往往**还能顺手提精度**。

## 6. Decoder-only 自回归预测

GPT 风格的 decoder-only 在 NLP 里基本赢麻了。同样的套路也能用于预测：去掉编码器，只训练一个带因果 mask 的 stack，然后一步一步把预测 roll 出来。

![Decoder-only 自回归预测与因果 mask。每一步把目前为止生成的全部内容喂回模型，问"下一步是什么"。右图是 mask：蓝色可见，灰色被屏蔽。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig6_decoder_only_forecast.png)
*图 6. Decoder-only 自回归预测与因果 mask。每一步把目前为止生成的全部内容喂回模型，问"下一步是什么"。右图是 mask：蓝色可见，灰色被屏蔽。*

```python
@torch.no_grad()
def autoregressive_forecast(model, history: torch.Tensor, horizon: int):
    """history: (B, L, d)，返回 (B, horizon, d)。"""
    seq = history
    out = []
    for _ in range(horizon):
        pred = model(seq)[:, -1:, :]   # 最后一步即"下一步"的预测
        out.append(pred)
        seq = torch.cat([seq, pred], dim=1)
    return torch.cat(out, dim=1)
```

**与 encoder-decoder 的取舍**：

| 维度             | Encoder-decoder              | Decoder-only                            |
|------------------|------------------------------|-----------------------------------------|
| 训练成本         | 两个 stack                    | 一个 stack                              |
| 推理延迟         | 一次前向出 $H$ 步             | $H$ 次前向（带 KV cache 时便宜很多）    |
| 暴露偏差         | teacher forcing 缓解         | 必须配 scheduled sampling 才能压住       |
| 预训练迁移       | 不顺手                       | 天然——TimesFM、Lag-Llama、Chronos 都是这套 |

要做"一个基础模型打天下"，decoder-only 现在是主流。

## 7. Patching：性价比最高的提速

PatchTST（Nie et al., ICLR 2023）一句话戳破了真相：**时间步不是合适的 token**。一段长度 512 的小时级序列，token 数远超普通 NLP 句子，但每个"token"几乎不携带信息。把它们按 $P$ 步一组打 patch，token 数就降到 $\lceil L / P \rceil$，每个都是一段有意义的小波形。

![Patching 策略。上：把长度 96 的序列切成 8 个 size 12 的 patch。下：每个 patch 通过线性投影变成一个 token。右：patch size 越大，attention 相对开销越小（O(n²) 的好处）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig7_patching.png)
*图 7. Patching 策略。上：把长度 96 的序列切成 8 个 size 12 的 patch。下：每个 patch 通过线性投影变成一个 token。右：patch size 越大，attention 相对开销越小（O(n²) 的好处）。*

为什么 patching 这么有用：

- **注意力开销直接除以 $P^2$**。$P=16$、$L=512$ 时，每头注意力条目从 26 万降到约 1 千。
- **每个 token 都有意义**。12 个小时合一个 patch 是半天的有效单位；单个小时不是。
- **天然引入局部性偏置**。patch 内部的局部模式由线性投影负责，attention 只用建模 patch 之间的（更长程的）关系。
- **通道独立**。PatchTST 把每个变量当独立序列、共享权重，避免了训练初期的伪通道关联。

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x):                # x: (B, L, C)
        B, L, C = x.shape
        P = self.patch_size
        L_trim = (L // P) * P
        x = x[:, :L_trim, :].reshape(B, L_trim // P, P * C)
        return self.proj(x)              # (B, L/P, d_model)
```

## 8. 参考实现

用 `nn.Transformer` 把上面所有要素拼起来：

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=8,
                 n_enc=3, n_dec=2, d_ff=512, dropout=0.1,
                 lookback=512, horizon=96, patch=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch, d_model, n_features)
        n_tokens = lookback // patch
        self.pos = TemporalPositionalEncoding(d_model, max_len=n_tokens + horizon)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.head = nn.Linear(d_model, n_features)
        self.horizon = horizon

    def forward(self, src, tgt):
        # src: (B, L, C). tgt: (B, H, C) —— 训练时用 teacher forcing。
        memory = self.encoder(self.pos(self.patch_embed(src)))
        tgt_emb = self.pos(self.patch_embed(tgt))
        L_tgt = tgt_emb.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L_tgt).to(src.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=causal)
        return self.head(out)            # (B, L_tgt, C)
```

几条工程经验：

- **`norm_first=True`**（pre-LN）对深层 stack 更稳；原版 post-LN 通常需要 warm-up 才肯收敛。
- FFN 用 **GELU** 而不是 ReLU——BERT 之后就是标配，时间序列上也一致更好。
- **永远先按序列做 z-score 归一化**，输出端再反归一化。漏掉这一步是 Transformer "训不出来" 最常见的原因。

## 9. 变体与选型

| 变体         | 核心想法                                        | 何时选它                       | 年份 |
|--------------|------------------------------------------------|--------------------------------|------|
| **Vanilla**  | 编码器-解码器 + 正弦 PE                         | lookback < 1k，先打个基线      | 2017 |
| **Informer** | ProbSparse 注意力 + 标签窗口                    | 很长的 lookback（5k-10k）       | 2021 |
| **Autoformer** | 序列分解 + 用 auto-correlation 替代 self-attn | 周期清晰强烈                   | 2021 |
| **FEDformer** | 频域注意力                                      | 周期数据，长 horizon            | 2022 |
| **PatchTST** | Patching + 通道独立                             | 大多数多元预测任务              | 2023 |
| **iTransformer** | 把每个变量当一个 token，跨变量做 attention   | 多个相关通道                    | 2024 |

如果在 2024-2025 年从零开始，我们的默认推荐是 **PatchTST 或 iTransformer**：在 ETT / Electricity / Traffic 等标准 benchmark 上都打过老变体，而且实现更简单、训练更快。

## 10. 性能与工程

### 10.1 预测质量

我们在一个带日 / 周双周期 + 随机峰值的合成信号上做 96 步预测。Transformer 把两个周期都干净地抓住，LSTM 锁住了主导的日周期，但在周周期上漂移。

![日 + 周双周期信号上的预测质量。Transformer 锁住两个周期，LSTM 抓住主导日周期但周周期漂移。右：各架构 MAE 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer%E6%9E%B6%E6%9E%84/fig4_lstm_vs_transformer.png)
*图 4. 日 + 周双周期信号上的预测质量。Transformer 锁住两个周期，LSTM 抓住主导日周期但周周期漂移。右：各架构 MAE 对比。*

### 10.2 训练食谱（决定能不能训出来的细节）

- **优化器**：AdamW，$\beta = (0.9, 0.95)$（GPT-3 用的设置——默认的 0.999 对时间序列偏迟钝）。
- **学习率调度**：前 5-10% 步线性 warm-up，之后余弦退火到 0。没有 warm-up，深层 Transformer 会发散。
- **学习率**：$d_{\text{model}}=128$ 时从 $1\text{e-}4$ 起步，模型更大就调小。
- **梯度裁剪**：$\|g\| \le 1.0$，没得商量。
- **batch size**：能放多大放多大，Transformer 对大 batch 的稳定性收益巨大。
- **混合精度**（`torch.cuda.amp` 或 `bfloat16`）：2-3 倍速，几乎无精度损失。
- **耐心**：预测用 Transformer 通常要 100-300 epoch；语言模型那种 3-10 epoch 在这里不成立。

### 10.3 上线：服务成本与 RevIN

- 用 **`torch.compile`**（PyTorch 2.x）：1.5-2 倍延迟优化，免费。
- decoder-only 部署一定要 **缓存 K、V**：每多一步从 $O(n^2)$ 降到 $O(n)$。
- **可逆实例归一化**（RevIN，ICLR 2022）：推理时按序列做归一化，输出端反归一化。一行代码改动，专治"训练用的历史和上线后数据漂移"那个老问题。

## 11. 常见踩坑

| 现象                                    | 大概率原因                                         | 修复                                       |
|-----------------------------------------|----------------------------------------------------|--------------------------------------------|
| Loss 平在数据方差附近                    | 输入没归一化                                       | 按序列 z-score，输出反归一化               |
| 几百步后 loss 发散                       | 没 warm-up；post-LN 配大 LR                         | 线性 warm-up + `norm_first=True`           |
| 验证集塌成常数                          | 解码器 mask 错了，未来漏了进来                       | 确认 `tgt_mask` 是严格上三角                |
| lookback > 1024 就 OOM                   | 朴素 attention                                     | 先上 patching，仍不够再上稀疏 / 线性        |
| 预测只跟着最近值，趋势忽略不计           | 位置没注入，或 PE 被特征数值规模盖过                | 缩放 PE 到特征量级；加上日历特征            |
| "Transformer 还不如 LSTM"               | 数据集不到 1 万样本，正则不够                       | 缩小模型、dropout 0.2-0.3、加 weight decay |

## 12. 总结

Transformer 不是魔法——它是**让每个时间步都能并行直接看到其它任何时间步**的最简单架构。时间序列上有三件事最重要：

1. **位置就是输入**——没有好的位置信息，Transformer 分不清星期一和星期五。用正弦 PE + 日历特征（不等间隔就用相对位置）。
2. **朴素 attention 是 O(n²)**——但只有几千步以上才是真问题。最便宜的修复是 **Patching**，而且通常**还顺手提精度**。
3. **按数据挑变体**——大多数多元预测用 PatchTST 或 iTransformer；周期清晰的用 FEDformer / Autoformer；要做"基础模型迁移"风格则用 decoder-only。

不管怎么变，注意力公式不变：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

本文剩下的全部内容，都是它上面的工程。

## 延伸阅读

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021
- Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation*, NeurIPS 2021
- Zhou et al., *FEDformer: Frequency Enhanced Decomposed Transformer*, ICML 2022
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*, ICLR 2023
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022

---

**系列导航**

| | |
|---|---|
| **上一篇** | [Attention 机制](/zh/时间序列模型-四-Attention机制/) |
| **当前** | 第五部分：Transformer 架构 |
| **下一篇** | [时序卷积网络 TCN](/zh/时间序列模型-六-时序卷积网络TCN/) |
