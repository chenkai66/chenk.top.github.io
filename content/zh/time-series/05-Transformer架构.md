---
title: "时间序列模型（五）：时间序列的 Transformer 架构"
date: 2024-10-31 09:00:00
tags:
  - 时间序列
  - 深度学习
  - Transformer
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "时间序列的 Transformer 全景：编码器-解码器结构、时序位置编码、O(n^2) 注意力瓶颈、Decoder-only 自回归预测与 Patching 策略。含 Autoformer / FEDformer / Informer / PatchTST 选型与可直接运行的实现。"
disableNunjucks: true
series_order: 5
translationKey: "time-series-5"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/illustration_1.png)
## 本章要点

- 详细解析完整的 encoder-decoder Transformer，重新设计用于时间序列
- 为什么必须加入位置信息，正弦编码、学习式编码和时间感知编码各自有什么不同
- 多头注意力在时间序列上到底学到了什么
- 朴素 attention 的瓶颈在哪里（O(n²)），以及四种改进方向：稀疏、线性、分块、仅解码器
- 提供一份清晰的 PyTorch 参考实现，教你如何选择 Autoformer、FEDformer、Informer 或 PatchTST
## 前置知识

- 自注意力和多头注意力（第 4 篇）
- 编码器-解码器架构与 teacher forcing
- PyTorch 基础（`nn.Module`、训练循环）

---
## 1. 为什么时间序列要用 Transformer

LSTM 和 GRU 处理序列时是一步步来的，这带来了三个问题：

1. **路径长度是 O(L)**。信息从第 $t-L$ 步传到第 $t$ 步，需要经过 $L$ 次递归。这就是梯度消失的根源。
2. **训练只能串行**。第 $t+1$ 步必须等第 $t$ 步跑完才能开始，GPU 很多时候都闲着。
3. **隐状态是个瓶颈**。模型要把所有可能用到的历史信息压缩到一个固定大小的向量里。

自注意力机制一次性解决了这三个问题：每个位置通过 **一次矩阵乘法** 就能看到其他所有位置，任意两步之间的路径长度缩短为 $O(1)$，整个序列可以并行处理。代价是显存占用：存储 $n \times n$ 的注意力权重需要 $O(n^2)$ 的空间，这个问题我会在第 5 节和第 7 节详细讨论。

![时间序列适配版的编码器-解码器 Transformer。编码器并行读取 lookback 窗口，解码器生成预测窗口，并通过 cross-attention 关注编码器的 memory。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig1_architecture.png)
*图 1. 时间序列适配版的编码器-解码器 Transformer。编码器并行读取 lookback 窗口，解码器生成预测窗口，并通过 cross-attention 关注编码器的 memory。*
## 2. 架构逐块拆解

时间序列 Transformer 就是 2017 年的原始架构，但做了三处小而关键的改动：

| 模块         | 原版 NLP                          | 时间序列                                              |
|--------------|-----------------------------------|-------------------------------------------------------|
| 输入嵌入     | Token embedding 查表              | 对连续特征做**线性投影**                              |
| 位置信息     | 基于 token 索引的正弦编码          | **时间感知**编码（日历特征、不等间隔时间差）          |
| 输出头       | 对词汇表做 softmax                | 使用**线性层**输出实值预测向量                        |

其他部分完全不变：多头自注意力、前馈网络、残差连接、LayerNorm、解码器 cross-attention、因果掩码。每个 block 包含四个子层：

$$\begin{aligned}
h_1 &= \text{LayerNorm}(x + \text{MHSA}(x)) \\
h_2 &= \text{LayerNorm}(h_1 + \text{FFN}(h_1))
\end{aligned}$$

注意力机制依然是第 4 篇提到的标准缩放点积：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

### 2.1 编码器

编码器读取 lookback 窗口 $x_{t-L+1:t}$，生成上下文向量 $M \in \mathbb{R}^{L \times d_{\text{model}}}$。这里**没有 mask**，每个位置都能看到所有其他位置。

### 2.2 解码器

解码器的输入是**标签窗口**（历史数据最后 $L_{\text{label}}$ 步）拼接上占位零（用于预测范围），输出是预测值 $\hat{y}_{t+1:t+H}$。每个 block 包含两层 attention：

- **Masked self-attention**，带因果 mask，确保第 $t+k$ 步只能看到 $\le t+k-1$ 的位置。
- **Cross-attention**，Query 来自解码器，Key 和 Value 来自编码器的 memory $M$。这是解码器唯一能访问编码器输出的地方。

### 2.3 标签窗口：一个小技巧

纯 encoder-decoder 模型在历史和预测交界处容易出问题。Informer 和 Autoformer 的解决办法是给解码器喂 $L_{\text{label}}$ 步**已知历史**加上 $H$ 个零占位符。这样解码器从一个已知状态开始，逐步推进到未知区域。
## 3. 时间序列的位置编码

自注意力机制是排列不变的——打乱输入，输出不会变。在 NLP 中这是个问题，在时间序列中则是灾难。我用正弦编码来注入位置信息：

$$\text{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
\text{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).$$

每个位置 $p$ 都会生成一个由几何级数频率构成的**唯一标识**。低维分量震荡快，用来编码短程位置；高维分量震荡慢，用来编码长程位置。

![正弦位置编码。左：完整编码矩阵，每一行都是唯一标识。右：四个代表性维度，频率各不相同。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig2_positional_encoding.png)
*图 2. 正弦位置编码。左：完整编码矩阵，每一行都是唯一标识。右：四个代表性维度，频率各不相同。*

时间序列通常需要比“步索引”更丰富的**位置信息**：

- **日历特征**：小时、星期、月份、节假日标志。每个特征都有独立的可学习 embedding，并加到输入上。
- **非均匀采样**：用实际时间戳 $\tau_p$ 替换位置 $p$，并归一化。Time2Vec 和 Continuous-Time Transformer 就是这么做的。
- **相对位置**：直接在 attention score 中编码 $\tau_q - \tau_k$（T5 / TUPE 风格），更适合超长上下文。

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
## 4. 多头注意力学到了什么

单个注意力头只能建模一种关系。多头注意力将模型分成 $h$ 路并行计算，每路在 $d_k = d_{\text{model}} / h$ 维的投影上独立运行，最后拼接结果。在时间序列任务中，不同的头会专注于不同的模式：

![训练好的 Transformer 在 48 步窗口上的四个头，每个学到不同的时间模式。注意因果 mask：对角线之上没有任何权重。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig3_multihead_patterns.png)
*图 3. 训练好的 Transformer 在 48 步窗口上的四个头，每个学到不同的时间模式。注意因果 mask：对角线之上没有任何权重。*

| 头的模式            | 模型在做什么                                |
|---------------------|---------------------------------------------|
| **局部**（对角线）  | 主要是在做自回归滑动平均                    |
| **周期条纹**        | 锁定已知周期（比如 24 小时或一周）          |
| **长程弥散**        | 提取缓慢的趋势信息                          |
| **锚定带**          | 抓住特定的历史事件（比如峰值或状态切换）    |

Transformer 的 **可解释性** 就来源于此：把最后一层的注意力头平均一下，就能知道预测依赖哪些历史步数。
## 5. O(n²) 瓶颈

朴素 attention 每层每个头都需要存一个 $n \times n$ 的 score 矩阵。用 fp16，8 个头时，每层注意力的显存占用是：

$$M_{\text{attn}} = h \cdot n^2 \cdot 2 \;\text{bytes}.$$

$n=512$ 时占 4 MB，完全没问题；$n=4096$ 时占 256 MB，开始吃力；$n=16384$ 时单层光注意力矩阵就要 4 GB 以上。计算量也一样，每层是 $O(n^2 d_{\text{model}})$ FLOPs。

![注意力的显存与 FLOPs 随序列长度的变化。朴素 O(n²) 在几千步后就跑不动了；稀疏、线性、Patching 三类方案能把开销压回可控范围。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig5_quadratic_bottleneck.png)
*图 5. 注意力的显存与 FLOPs 随序列长度的变化。朴素 O(n²) 在几千步后就跑不动了；稀疏、线性、Patching 三类方案能把开销压回可控范围。*

解决这个问题有四类方法，按对模型改动的大小排序：

1. **稀疏注意力**（Longformer、BigBird、Informer 的 ProbSparse）：只计算 $(q, k)$ 对的一个稀疏子集。复杂度降到 $O(n \cdot w)$，其中 $w$ 是窗口大小或选中的 key 数量。
2. **线性注意力**（Performer、Linformer、Nystromformer）：用可分解的核函数替换 softmax，把复杂度降到 $O(n \cdot d^2)$。
3. **Patching**（PatchTST、Autoformer 风格的序列分解）：直接缩短序列，把连续几步合并成一个 patch。第 7 节会详细讲。
4. **Decoder-only + KV cache**（第 6 节）：训练时还是 $O(n^2)$，但推理可以增量处理。

实战中，如果 lookback 窗口在 2k 以内，预测步长在几百以内，朴素 attention 完全够用。再长的话，**Patching 是最划算的选择**——不仅大幅降低计算量，通常还能提升精度。
## 6. Decoder-only 自回归预测

GPT 风格的 decoder-only 架构在 NLP 领域已经大获成功。这个思路同样适用于时间序列预测：去掉编码器，只用一个带因果 mask 的解码器堆栈，逐步生成预测结果。

![Decoder-only 自回归预测与因果 mask。每一步将模型已生成的内容全部输入，要求预测下一步的值。右图展示了可见位置的 mask。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig6_decoder_only_forecast.png)
*图 6. Decoder-only 自回归预测与因果 mask。每一步将模型已生成的内容全部输入，要求预测下一步的值。右图展示了可见位置的 mask。*

```python
@torch.no_grad()
def autoregressive_forecast(model, history: torch.Tensor, horizon: int):
    """history: (B, L, d)，返回 (B, horizon, d)。"""
    seq = history
    out = []
    for _ in range(horizon):
        pred = model(seq)[:, -1:, :]   # 最后一个位置是下一步的预测值
        out.append(pred)
        seq = torch.cat([seq, pred], dim=1)
    return torch.cat(out, dim=1)
```

**与 encoder-decoder 的对比**：

| 特性               | Encoder-decoder              | Decoder-only                            |
|------------------|------------------------------|-----------------------------------------|
| 训练成本         | 两个堆栈                     | 一个堆栈                                |
| 推理延迟         | 一次前向计算出 $H$ 步         | $H$ 次前向计算（带 KV cache 时成本低很多） |
| 暴露偏差         | teacher forcing 可缓解       | 不用 scheduled sampling 就会有暴露偏差   |
| 预训练迁移       | 不够自然                     | 天然适合——TimesFM、Lag-Llama、Chronos 都是这种架构 |

如果要用一个基础模型应对多种任务，decoder-only 已经成为主流选择。
## 7. Patching：最有效的提速方法

PatchTST（Nie 等，ICLR 2023）提出了一个颠覆性的观点：**时间步并不是合适的 token**。一段长度为 512 的小时级序列，token 数量远超普通 NLP 句子，但每个“token”几乎没什么信息量。如果按 $P$ 步一组分成 patch，token 数量会减少到 $\lceil L / P \rceil$，每个 token 都能概括一小段波形。

![Patching 策略。上：将长度 96 的序列分成 8 个大小为 12 的 patch。下：每个 patch 通过线性投影变成一个 token。右：随着 patch size 增大，attention 开销快速下降（O(n²) 的优势）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig7_patching.png)
*图 7. Patching 策略。上：将长度 96 的序列分成 8 个大小为 12 的 patch。下：每个 patch 通过线性投影变成一个 token。右：随着 patch size 增大，attention 开销快速下降（O(n²) 的优势）。*

为什么 patching 这么有效？

- **注意力开销直接减少 $P^2$ 倍**。比如 $P=16$、$L=512$ 时，每头的注意力条目从 26 万降到约 1 千。
- **每个 token 都有实际意义**。12 个小时组成一个 patch，正好是半天的有效单位；单个小时则没有这种意义。
- **天然具备局部性偏置**。patch 内部的局部模式由线性投影处理，attention 只需建模 patch 之间的长程关系。
- **通道独立性**。PatchTST 把每个变量当作独立序列，共享权重，避免了训练初期出现伪通道关联。

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

用 `nn.Transformer` 把所有模块整合起来：

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
        # src: (B, L, C). tgt: (B, H, C) —— 训练时使用 teacher forcing。
        memory = self.encoder(self.pos(self.patch_embed(src)))
        tgt_emb = self.pos(self.patch_embed(tgt))
        L_tgt = tgt_emb.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L_tgt).to(src.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=causal)
        return self.head(out)            # (B, L_tgt, C)
```

几点工程实践心得：

- **`norm_first=True`**（pre-LN）在深层网络中更稳定；原版 post-LN 通常需要 warm-up 才能收敛。
- FFN 中用 **GELU** 替代 ReLU——从 BERT 开始就是标配，实际效果也确实更好。
- **永远先对序列做 z-score 归一化**，输出时再反归一化。漏掉这一步是 Transformer 在时间序列任务上“训不动”的主要原因。
## 9. 模型变体与选择建议

| 变体         | 核心思想                                      | 最佳适用场景                   | 年份 |
|--------------|-----------------------------------------------|--------------------------------|------|
| **Vanilla**  | 编码器-解码器 + 正弦 PE                       | lookback < 1k，需要基线模型     | 2017 |
| **Informer** | ProbSparse 注意力 + 标签窗口                  | 超长 lookback（5k-10k）         | 2021 |
| **Autoformer** | 序列分解 + auto-correlation 替代 self-attention | 数据周期性强、规律清晰          | 2021 |
| **FEDformer** | 频域注意力                                    | 周期性数据，预测跨度大          | 2022 |
| **PatchTST** | Patching + 通道独立                           | 多变量预测任务                 | 2023 |
| **iTransformer** | 将每个变量视为 token，跨变量做 attention    | 多个相关通道                   | 2024 |

如果从 2024-2025 年开始新项目，我建议优先选择 **PatchTST 或 iTransformer**。这两个模型在 ETT / Electricity / Traffic 等标准 benchmark 上表现优于早期模型，同时实现更简单，训练速度更快。
## 10. 性能与工程

### 10.1 预测质量

我用一个带日周期和周周期的合成信号，加上随机尖峰，做了 96 步预测。Transformer 干净利落地捕捉到了两个周期，LSTM 能跟上主要的日周期，但在周周期上开始漂移。

![日 + 周双周期信号上的预测质量。Transformer 锁住两个周期，LSTM 抓住主导日周期但周周期漂移。右：各架构 MAE 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig4_lstm_vs_transformer.png)
*图 4. 日 + 周双周期信号上的预测质量。Transformer 锁住两个周期，LSTM 抓住主导日周期但周周期漂移。右：各架构 MAE 对比。*

### 10.2 训练技巧（那些决定成败的小细节）

- **优化器**：AdamW，$\beta = (0.9, 0.95)$。这是 GPT-3 的设置，默认的 0.999 对时间序列来说太慢了。
- **学习率调度**：前 5%-10% 的步数做线性 warm-up，然后用余弦退火降到 0。没有 warm-up，深层 Transformer 很容易发散。
- **学习率**：$d_{\text{model}}=128$ 时从 $1\text{e-}4$ 开始，模型更大就适当调低。
- **梯度裁剪**：$\|g\| \le 1.0$，这一点没得商量。
- **batch size**：能多大就多大，Transformer 在大 batch 下稳定性提升非常明显。
- **混合精度**（`torch.cuda.amp` 或 `bfloat16`）：速度提升 2-3 倍，精度几乎没有损失。
- **训练轮数**：预测任务的 Transformer 通常需要 100-300 个 epoch，语言模型那种 3-10 个 epoch 的经验在这里不适用。

### 10.3 上线：服务成本与 RevIN

- 用 **`torch.compile`**（PyTorch 2.x）：延迟直接降低 1.5-2 倍，完全免费。
- decoder-only 部署时，一定要 **缓存 K 和 V**：这样每新增一步的复杂度从 $O(n^2)$ 降到 $O(n)$。
- **可逆实例归一化**（RevIN，ICLR 2022）：推理时对每个输入序列做归一化，输出时再反归一化。一行代码的改动，彻底解决“训练数据和线上数据分布漂移”的老问题。
## 11. 常见问题与解决方法

| 现象                                    | 可能原因                                           | 解决办法                                   |
|-----------------------------------------|----------------------------------------------------|--------------------------------------------|
| Loss 停在数据方差附近                   | 忘了归一化输入                                     | 按序列做 z-score，输出时反归一化           |
| 几百步后 Loss 开始发散                  | 没有 warm-up；post-LN 配合学习率过高               | 加上线性 warm-up，设置 `norm_first=True`   |
| 验证集结果变成常数                      | 解码器的 mask 错误导致未来信息泄露                 | 确保 `tgt_mask` 是严格的上三角矩阵         |
| lookback > 1024 时内存不足              | 使用了普通的 attention                             | 先尝试 patching，必要时改用稀疏或线性方法  |
| 预测只跟随最近值，忽略趋势              | 位置信息未注入，或者 PE 被特征数值范围压制         | 将 PE 缩放到特征量级，添加日历特征         |
| "Transformer 效果不如 LSTM"            | 数据集小于 1 万样本，模型正则化不足                | 缩小模型规模，设置 dropout 为 0.2-0.3，启用 weight decay |
## 12. 总结

Transformer 并不是什么魔法，它只是让每个时间步都能直接访问其他所有时间步的最简单架构，而且是并行处理。在时间序列问题中，有三点最关键：

1. **位置信息就是输入**——如果位置信息不够好，Transformer 就分不清周一和周五。可以用正弦位置编码（PE）加上日历特征，或者对不规则数据使用相对位置。
2. **标准注意力机制复杂度是 O(n²)**——但只有当时间步超过几千时才会成为问题。最简单的解决办法是 **Patching**，而且通常还能顺便提升精度。
3. **根据数据选择合适的变体**——大多数多元预测问题用 PatchTST 或 iTransformer；周期性明显的场景用 FEDformer 或 Autoformer；如果是类似基础模型迁移的任务，则用 decoder-only 架构。

不管怎么变化，注意力的核心公式始终不变：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

本文讲的所有内容，本质上都是在这个公式基础上的工程实践。
