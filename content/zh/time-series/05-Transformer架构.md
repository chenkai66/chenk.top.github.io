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

---

## 本文要点

- 完整解析专为时间序列设计的 encoder-decoder Transformer 架构
- 为什么必须注入位置信息，以及正弦编码、可学习编码与时间感知编码的区别
- 多头注意力在时间序列上实际学到的内容
- 朴素 attention 的瓶颈（O(n²)）及四大改进方向：稀疏、线性、分块（patching）、仅解码器
- 提供一份清晰的 PyTorch 参考实现，并指导何时选用 Autoformer / FEDformer / Informer / PatchTST

## 前置知识

- 自注意力与多头注意力（第 4 篇）
- 编码器-解码器架构与 teacher forcing
- PyTorch 基础（`nn.Module`、训练循环）

---

## 为什么时间序列要用 Transformer

LSTM 和 GRU 按时间步顺序处理序列，由此带来三个根本限制：

1. **路径长度为 O(L)**：从第 $t-L$ 步传递到第 $t$ 步的信息必须经过 $L$ 次递归，这正是梯度消失的根源。
2. **训练无法并行**：第 $t+1$ 步必须等第 $t$ 步完成后才能开始，导致 GPU 大量时间处于闲置状态。
3. **隐状态是信息瓶颈**：模型必须将所有可能用到的历史信息压缩进一个固定维度的向量中。

自注意力机制一次性打破这三重限制：每个位置通过**一次矩阵乘法**即可访问所有其他位置，任意两步间的路径长度缩短至 $O(1)$，整个序列得以并行处理。代价是内存开销——存储 $n \times n$ 的注意力权重需要 $O(n^2)$ 空间，我们将在[第 5 节](#时间序列的位置编码)和第 7 节详细讨论这一问题。

![时间序列适配版的编码器-解码器 Transformer。编码器并行读取 lookback 窗口，解码器生成预测窗口，并通过 cross-attention 关注编码器的 memory。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig1_architecture.png)
*图 1. 适配时间序列的编码器-解码器 Transformer。编码器并行读取 lookback 窗口；解码器生成预测范围，并通过 cross-attention 访问编码器的记忆。*

## 架构逐块拆解

时间序列 Transformer 基于 2017 年原始架构，仅在三个关键组件上做了针对性调整：

| 组件         | 原始 NLP                          | 时间序列                                              |
|--------------|-----------------------------------|-------------------------------------------------------|
| 输入嵌入     | Token embedding 查表              | 对连续特征进行**线性投影**                            |
| 位置信息     | 基于 token 索引的正弦编码          | **时间感知**编码（日历特征、不规则采样间隔）          |
| 输出头       | 对词汇表做 softmax                | 使用**线性层**输出实值预测向量                        |

其余部分——多头自注意力、前馈网络、残差连接、LayerNorm、解码器 cross-attention、因果掩码——均保持不变。每个 block 包含四个子层：
$$
\begin{aligned}
h_1 &= \text{LayerNorm}(x + \text{MHSA}(x)) \\
h_2 &= \text{LayerNorm}(h_1 + \text{FFN}(h_1))
\end{aligned}
$$
注意力机制仍采用第 4 篇介绍的标准缩放点积形式：
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$
### 编码器

编码器读取 lookback 窗口 $x_{t-L+1:t}$，输出上下文矩阵 $M \in \mathbb{R}^{L \times d_{\text{model}}}$。此处不使用掩码，即每个位置可关注窗口内所有其他位置。

### 解码器

解码器输入由**标签窗口**（历史数据最后 $L_{\text{label}}$ 步）与 $H$ 个零占位符拼接而成，输出预测值 $\hat{y}_{t+1:t+H}$。每个 block 包含两类注意力子层：

- **带因果掩码的自注意力**：确保第 $t+k$ 步只能看到 $t+k-1$ 及之前的位置。
- **交叉注意力**：Query 来自解码器，Key/Value 来自编码器记忆 $M$。这是解码器获取编码器信息的唯一通道。

### 标签窗口：一个小而有效的技巧

标准 encoder-decoder 模型常在历史与预测的交界处表现不佳。Informer 和 Autoformer 的解决方案是：向解码器输入 $L_{\text{label}}$ 步**已知历史**加 $H$ 个零占位符，使其始终从确定状态出发，逐步推进至未知区域。

## 时间序列的位置编码

自注意力具有排列不变性——打乱输入顺序，输出不变。这对语言建模已是问题，对时间序列更是灾难。为此，我们注入位置信息，最经典的方式是正弦编码：
$$
\text{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \quad
\text{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$
每个位置 $p$ 获得一个由几何级数频率构成的**唯一签名**：低维分量高频振荡（编码短程位置），高维分量低频振荡（编码长程位置）。

![正弦位置编码。左：完整编码矩阵，每一行都是唯一标识。右：四个代表性维度，频率各不相同。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig2_positional_encoding.png)
*图 2. 正弦位置编码。左：完整编码矩阵，每行对应一个唯一签名。右：四个代表性维度，振荡频率各异。*

但时间序列通常需要比“步索引”更丰富的**位置信息**：

- **日历特征**：如小时、星期、月份、节假日标志。每项独立嵌入后加到输入上。
- **非均匀采样**：用实际时间戳 $\tau_p$ 替代位置索引 $p$ 并归一化，Time2Vec 和 Continuous-Time Transformer 即采用此策略。
- **相对位置**：在注意力得分中直接编码 $\tau_q - \tau_k$（T5 / TUPE 风格），更适合超长上下文。

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

## 多头注意力学到了什么

单个注意力头只能建模一种关系。多头机制将模型拆分为 $h$ 路并行计算，每路在 $d_k = d_{\text{model}} / h$ 维投影上独立运行，最终拼接结果。在时间序列任务中，不同头往往各司其职：

![训练好的 Transformer 在 48 步窗口上的四个头，每个学到不同的时间模式。注意因果 mask：对角线之上没有任何权重。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig3_multihead_patterns.png)
*图 3. 在 48 步窗口上训练好的 Transformer 的四个注意力头，各自聚焦不同时间模式。注意因果掩码：对角线上方无任何权重。*

| 注意力头模式      | 模型行为                                      |
|------------------|---------------------------------------------|
| **局部**（对角线）| 近似自回归滑动平均                           |
| **周期条纹**      | 锁定已知周期（如 24 小时或每周）             |
| **长程弥散**      | 提取缓慢趋势信息                             |
| **锚定带**        | 关注特定历史事件（如尖峰或状态切换）         |

这也构成了 Transformer 的**可解释性基础**：对最后一层各头注意力取平均，即可看出预测实际依赖哪些历史时刻。

## O(n²) 瓶颈

朴素注意力每层每头需存储 $n \times n$ 的得分矩阵。以 fp16 精度、8 个头计算，单层注意力内存占用为：
$$
M_{\text{attn}} = h \cdot n^2 \cdot 2 \;\text{bytes}.
$$
当 $n=512$ 时仅占 4 MB，尚可接受；$n=4096$ 时达 256 MB，已显吃力；$n=16384$ 时单层注意力矩阵就超过 4 GB。计算复杂度同样为 $O(n^2 d_{\text{model}})$ FLOPs。

![注意力的显存与 FLOPs 随序列长度的变化。朴素 O(n²) 在几千步后就跑不动了；稀疏、线性、Patching 三类方案能把开销压回可控范围。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig5_quadratic_bottleneck.png)
*图 5. 注意力内存与 FLOPs 随序列长度的变化。朴素 O(n²) 在数千步后变得不可行；稀疏、线性、分块等替代方案能有效控制成本。*

目前有四类主流改进方案，按对模型改动程度递增排序：

1. **稀疏注意力**（Longformer、BigBird、Informer 的 ProbSparse）：仅计算 $(q, k)$ 对的稀疏子集，复杂度降至 $O(n \cdot w)$，其中 $w$ 为窗口大小或选中 key 数。
2. **线性注意力**（Performer、Linformer、Nystromformer）：用可分解核函数替代 softmax，使复杂度变为 $O(n \cdot d^2)$。
3. **分块（Patching）**（PatchTST、Autoformer 式序列分解）：**直接缩短序列长度**，将连续时间步聚合成 patch。[第 7 节](#On²-瓶颈)将详述。
4. **仅解码器 + KV 缓存**（见[第 6 节](#多头注意力学到了什么)）：训练仍为 $O(n^2)$，但推理可增量进行。

实践中，若 lookback 窗口小于 2k、预测步长仅数百，朴素注意力完全够用。超出此范围，**分块是最具性价比的改进**——通常既能显著降低计算开销，又能提升精度。

## Decoder-only 自回归预测

GPT 风格的仅解码器架构已在 NLP 领域胜出，同样适用于时间序列预测：舍弃编码器，仅用带因果掩码的解码器堆栈，逐步生成预测结果。

![Decoder-only 自回归预测与因果 mask。每一步将模型已生成的内容全部输入，要求预测下一步的值。右图展示了可见位置的 mask。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig6_decoder_only_forecast.png)
*图 6. 仅解码器自回归预测与因果掩码。每一步将模型已生成的所有内容作为输入，预测下一时刻值。右侧掩码显示各位置可见范围。*

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

**与 encoder-decoder 的权衡对比**：

| 特性               | Encoder-decoder              | Decoder-only                 |
|--------------------|------------------------------|------------------------------|
| 训练成本           | 需训练两个堆栈               | 仅一个堆栈                   |
| 推理延迟           | 一次前向计算输出全部 $H$ 步   | 需 $H$ 次前向（启用 KV 缓存后成本大降） |
| 暴露偏差           | 可通过 teacher forcing 缓解  | 若不采用 scheduled sampling 则存在 |
| 预训练迁移         | 不够自然                     | 天然契合——TimesFM、Lag-Llama、Chronos 等基础时序模型均采用此架构 |

若需构建一个可泛化至多任务的基础模型，decoder-only 已成主流选择。

## 分块（Patching）：最有效的提速手段

PatchTST（Nie et al., ICLR 2023）提出一个颠覆性观点：**时间步并非合适的 token**。一段 512 步的小时级序列，token 数远超典型 NLP 句子，但每个“token”信息量极低。若将其按 $P$ 步一组聚合成 patch，则 token 数降至 $\lceil L / P \rceil$，每个新 token 都能概括一段短时波形。

![Patching 策略。上：将长度 96 的序列分成 8 个大小为 12 的 patch。下：每个 patch 通过线性投影变成一个 token。右：随着 patch size 增大，attention 开销快速下降（O(n²) 的优势）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig7_patching.png)
*图 7. 分块策略。上：将长度 96 的序列划分为 8 个大小为 12 的 patch。下：每个 patch 通过线性投影变为一个 token。右：随 patch size 增大，注意力开销（O(n²)）迅速下降。*

分块之所以高效，原因有四：

- **注意力开销降低 $P^2$ 倍**：例如 $P=16$、$L=512$ 时，每头注意力条目从 26 万骤减至约 1 千。
- **每个 token 更具语义**：12 小时组成的 patch 可代表半天的有效模式，单个小时则缺乏意义。
- **天然引入局部性偏置**：patch 内部局部模式由线性投影捕获，注意力只需建模跨 patch 的长程依赖。
- **通道独立性**：PatchTST 将每个变量视为独立序列、共享权重，避免训练初期出现虚假的跨通道注意力。

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

## 参考实现

使用 `nn.Transformer` 整合上述组件：

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

几点工程实践建议：

- **`norm_first=True`**（pre-LN）在深层网络中更稳定；原始 post-LN 通常需 warm-up 才能收敛。
- FFN 中使用 **GELU** 而非 ReLU——自 BERT 起已成为标配，实测效果更优。
- **务必对每个序列单独做 z-score 归一化**，并在输出端反归一化。忽略此步是 Transformer 在时序任务上“无法训练”的最常见原因。

## 模型变体与选择指南

| 变体            | 核心思想                                      | 最佳适用场景                   | 年份 |
|-----------------|-----------------------------------------------|--------------------------------|------|
| **Vanilla**     | 编码器-解码器 + 正弦 PE                       | lookback < 1k，需基线模型       | 2017 |
| **Informer**    | ProbSparse 注意力 + 标签窗口                  | 超长 lookback（5k–10k）         | 2021 |
| **Autoformer**  | 序列分解 + 自相关替代自注意力                 | 周期性强且规律清晰的数据        | 2021 |
| **FEDformer**   | 频域注意力                                    | 周期性数据，长预测范围          | 2022 |
| **PatchTST**    | 分块 + 通道独立                               | 多变量预测任务                 | 2023 |
| **iTransformer**| 将每个变量视为 token，跨变量做 attention      | 多个高度相关通道               | 2024 |

若在 2024–2025 年启动新项目，我们推荐优先尝试 **PatchTST 或 iTransformer**——二者在 ETT / Electricity / Traffic 等标准 benchmark 上普遍优于早期模型，且实现更简洁、训练更快。

## 性能与工程实践

### 预测质量

我们在含日周期、周周期及随机尖峰的合成信号上预测 96 步。Transformer 清晰捕捉到双重周期性；LSTM 能跟踪主导的日周期，但在周周期上明显漂移。

![日 + 周双周期信号上的预测质量。Transformer 锁住两个周期，LSTM 抓住主导日周期但周周期漂移。右：各架构 MAE 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/05-Transformer架构/fig4_lstm_vs_transformer.png)
*图 4. 日+周双周期信号上的预测效果。Transformer 同时锁定两个周期；LSTM 仅捕获日周期，周周期发生漂移。右：各架构 MAE 对比。*

### 训练配方（那些决定成败的细节）

- **优化器**：AdamW，$\beta = (0.9, 0.95)$（GPT-3 设置；默认 0.999 对时序任务过慢）。
- **学习率调度**：前 5%–10% 步数线性 warm-up，随后余弦退火至零。无 warm-up 时，深层 Transformer 易发散。
- **初始学习率**：$d_{\text{model}}=128$ 时设为 $1\text{e-}4$，模型更大则适当调低。
- **梯度裁剪**：$\|g\| \le 1.0$，不可或缺。
- **批大小**：尽可能大——大 batch 能显著提升 Transformer 训练稳定性。
- **混合精度**（`torch.cuda.amp` 或 `bfloat16`）：提速 2–3 倍，精度几乎无损。
- **训练轮数**：时序预测通常需 100–300 轮；语言模型的 3–10 轮经验不适用。

### 生产部署：服务成本与 RevIN 技巧

- 启用 **`torch.compile`**（PyTorch 2.x）：免费获得 1.5–2 倍延迟降低。
- decoder-only 部署时，**缓存 K 和 V**：使每步新增预测的复杂度从 $O(n^2)$ 降至 $O(n)$。
- **可逆实例归一化**（RevIN, ICLR 2022）：推理时对每个输入序列归一化，输出时反归一化。一行代码即可解决“训练与线上数据分布漂移”问题。

## 常见陷阱与对策

| 现象                                    | 可能原因                                           | 解决方案                                   |
|-----------------------------------------|----------------------------------------------------|--------------------------------------------|
| Loss 停滞在数据方差附近                 | 未对输入归一化                                     | 按序列 z-score 归一化，输出端反归一化      |
| 训练几百步后 Loss 发散                  | 无 warm-up；post-LN 配高学习率                     | 添加线性 warm-up，启用 `norm_first=True`   |
| 验证集输出退化为常数                    | 解码器掩码错误导致未来信息泄露                     | 确保 `tgt_mask` 为严格上三角矩阵           |
| lookback > 1024 时内存溢出              | 使用朴素注意力                                     | 优先尝试分块，必要时改用稀疏或线性方法     |
| 预测仅跟随最近值，忽略长期趋势          | 位置信息缺失，或 PE 被特征尺度压制                 | 将 PE 缩放至特征量级，添加日历特征         |
| “Transformer 效果不如 LSTM”            | 数据集 < 1 万样本，模型正则化不足                  | 缩小模型，设置 dropout 0.2–0.3，启用 weight decay |

## 总结

Transformer 并非魔法——它只是让每个时间步都能**并行地、直接地**访问其他所有时间步的最简架构。在时间序列任务中，三点至关重要：

1. **位置即输入**：若位置信息不足，Transformer 无法区分周一与周五。应结合正弦 PE 与日历特征（或对不规则数据使用相对位置）。
2. **朴素注意力为 O(n²)**：仅当序列超数千步时才成问题。**分块（Patching）是最经济的解决方案**，且常能同步提升精度。
3. **按数据特性选模型**：多元预测首选 PatchTST 或 iTransformer；强周期性数据适用 FEDformer / Autoformer；基础模型迁移任务则倾向 decoder-only 架构。

无论何种变体，核心注意力公式始终如一：
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$
本文所有内容，本质上都是在此公式之上的工程实践。
