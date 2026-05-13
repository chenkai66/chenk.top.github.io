---
title: "时间序列模型（二）：LSTM——门控机制与长期依赖"
date: 2024-09-16 09:00:00
tags:
  - 时间序列
  - 深度学习
  - LSTM
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "LSTM 的遗忘门、输入门和输出门如何解决梯度消失问题。完整的 PyTorch 时间序列预测代码和实用调参技巧。"
disableNunjucks: true
series_order: 2
translationKey: "time-series-2"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/illustration_1.png)
## 本章要点

- 普通 RNN 为何难以处理长序列，以及 LSTM 如何解决梯度消失与爆炸问题
- 遗忘门、输入门和输出门的直观作用，以及细胞状态这条“高速公路”如何维持长期记忆
- 在单步与多步时间序列预测中，如何合理设计 LSTM 的输入输出结构
- 实用技巧：正则化策略、序列长度选择、双向 vs 堆叠 LSTM 的适用场景，以及何时应优先选用 LSTM 而非 GRU

## 准备工作

- 掌握神经网络基础知识（前向传播、反向传播）
- 熟悉 PyTorch 核心组件（`nn.Module`、张量、优化器）
- 建议阅读本系列第 1 篇（非必需）

---

## 1. LSTM 要解决的问题

普通 RNN 的隐藏状态按如下方式递归更新：
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b).$$\n当从第 $T$ 步将损失反向传播至更早的第 $k$ 步时，梯度会累积一连串雅可比矩阵的乘积：
$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \mathrm{diag}\!\left(1 - h_t^2\right) W_h.$$\n这会导致两种极端情况：

- 若 $W_h$ 的主奇异值小于 1，梯度会**指数级衰减**，模型几乎无法学习超过约 10 步的历史信息；
- 若大于 1，梯度则会**爆炸式增长**，导致训练发散。
\nLSTM（Hochreiter & Schmidhuber, 1997）通过引入**两个状态**（细胞状态 $C_t$ 和隐藏状态 $h_t$）以及三个可学习的门控机制——遗忘门、输入门和输出门——来决定保留什么、覆盖什么、暴露什么。这种近似加性的更新方式使梯度能在数百个时间步中稳定回传。

## 2. LSTM 单元的内部结构

每个 LSTM 单元包含四个共享输入 $[h_{t-1}, x_t]$ 的门控单元，分别输出三个 sigmoid 门和一个 $\tanh$ 候选值：
$$
\begin{aligned}\nf_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{遗忘门} \\\ni_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{输入门} \\
\tilde C_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{候选值} \\\no_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{输出门}
\end{aligned}
$$\n这些信号共同完成细胞状态更新与隐藏状态输出：
$$\nC_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t, \qquad\nh_t = o_t \odot \tanh(C_t).
$$\n其中 $\odot$ 表示逐元素乘法。**通俗理解**：遗忘门 $f_t$ 决定擦除多少旧记忆，输入门 $i_t$ 控制写入多少新候选值，最后通过输出门 $o_t$ 决定对外暴露多少当前记忆。

![LSTM 单元结构：三个 sigmoid 门加一个 tanh 候选值位于细胞状态高速公路下方。遗忘门控制旧记忆的擦除，输入门调节新记忆的写入，输出门将过滤后的状态作为隐藏状态输出。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/fig1_lstm_cell.png)
*LSTM 单元——三个门作用在一条细胞状态高速公路上。*

### 为什么细胞状态如此特殊

隐藏状态 $h_t$ 是网络其他部分可见的输出，但真正的长期记忆存储在**细胞状态** $C_t$ 中。它像一条贯穿时间的水平通道，仅通过逐元素乘法（$f_t$）和加法（$i_t \odot \tilde C_t$）进行更新，**从未经过新的矩阵乘法**。正是这一设计让梯度得以跨越数百步稳定传播。

![两条并行的状态流：绿色的细胞状态高速公路以近似恒等的方式更新，承载长期记忆；紫色虚线表示经过门控过滤的隐藏状态，是下游层实际看到的内容。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/fig2_state_highway.png)
*细胞状态 vs 隐藏状态——两条并行的信息流。*

### 梯度流动的显式表达

对细胞状态求导可得：
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t,$$\n因此长程梯度是**遗忘门的连乘积**，而非 $\tanh$ 导数与循环权重矩阵的乘积：
$$\frac{\partial C_T}{\partial C_k} = \prod_{t=k+1}^{T} f_t.$$\n当模型需要记住某段信息时，只需将对应维度的 $f_t$ 学习为接近 1，相应梯度也会保持接近 1。这便是 LSTM 的核心奥秘。

## 3. 一个最简的 PyTorch 实现

无论是单变量还是多变量预测，以下实现已足够应对大多数场景：

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)              # (batch, seq_len, hidden_size)
        return self.head(out[:, -1, :])     # 取最后一个时间步
```

几点容易忽略的细节：

- `batch_first=True` 将输入形状设为 `(batch, seq_len, features)`，这是业界通用约定，而非 PyTorch 示例中的默认格式。
- 内置 `dropout` 参数**仅作用于层间**，不会在时间步之间丢弃激活值。若需实现循环 dropout，可使用 `nn.LSTMCell` 并手动应用固定掩码，或采用 AWD-LSTM 中的 `weight_drop` 技巧。
- **建议将遗忘门偏置初始化为 +1**，使网络初始处于“记忆模式”。PyTorch 默认不启用此设置：

```python
for name, p in model.lstm.named_parameters():
    if "bias" in name:
        n = p.size(0)
        p.data[n // 4 : n // 2].fill_(1.0)   # 遗忘门偏置
```

## 4. 从单元到预测器

时间序列预测的标准流程如下：

1. **划分窗口**：将原始序列切分为长度为 $L$ 的重叠片段（即*回望窗口*）。
2. **标准化**：使用训练集的均值与标准差对每个特征进行归一化。
3. **训练目标**：预测下一个时间点（单步）或未来 $H$ 个点（多步）。
4. **验证方式**：使用**按时间顺序保留的尾部数据**进行验证，严禁打乱时序。

在一个带噪声的季节性信号上，干净的单步预测效果大致如下：

![单步 LSTM 预测与真实序列对比。蓝色阴影区域是测试窗口残差标准差计算出的 95% 置信区间。模型能够同时捕捉 24 步和 75 步的季节性成分，但会有一小步的滞后。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/fig3_forecast.png)
*带噪季节信号上的单步 LSTM 预测 vs 实际值。*

### 多步预测：递归 vs 直接

当预测视野 $H > 1$ 时，常用两种策略：

| 策略 | 方法 | 权衡 |
| --- | --- | --- |
| **递归** | 训练一个单步模型，将其预测结果作为下一步输入递归使用。 | 实现简单，但误差会累积——方差随 $\sqrt{H}$ 增长。 |
| **直接** | 训练 $H$ 个独立输出头（或单模型输出 $H$ 维），直接预测各未来步。 | 参数更多，但避免了误差反馈循环。 |

![多步预测：琥珀色递归预测的不确定性带随视野扩大呈扇形扩散；绿色直接预测的置信区间基本保持不变。虚线标记预测起点。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/fig4_multistep.png)
*多步预测——递归预测误差累积明显；直接预测参数开销大，但置信区间更紧。*

实践中常用一种折中方案：**seq2seq + teacher forcing**。LSTM 编码器读取回望窗口生成最终 $(h, C)$ 状态对，解码器逐步生成 $H$ 个输出。训练时，解码器以一定概率接收**真实历史值**（而非自身预测）作为输入，该技术称为 scheduled sampling。这也是当前生产环境中的主流做法。

## 5. 架构变体

### 双向 LSTM（BiLSTM）
\nBiLSTM 同时运行一个前向 LSTM 和一个后向 LSTM，并在每一步拼接两者隐藏状态：
$$y_t = [\,\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\,].$$

![双向 LSTM：紫色正向链从左到右读取，琥珀色反向链从右到左读取，每一步的输出是两个隐藏状态的拼接。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig5_bilstm.png)
*双向 LSTM——每一步结合过去和未来的上下文信息。*

**适用场景**：序列标注、分类、缺失值填补等——只要推理时能获取完整序列即可。**禁用于实时预测**：训练时偷看 $x_{t+1}, x_{t+2}, \dots$ 而推理时却要预测 $x_{t+1}$，属于典型的数据泄漏，上线必崩。

### 堆叠（深层）LSTM

堆叠多层可使高层提取更平滑、更抽象的时序特征：第 1 层处理原始输入，第 2 层处理第 1 层的隐藏状态，依此类推。

![三层堆叠 LSTM：每一层从左到右递归，并将隐藏状态传递给上一层。底层捕捉短时局部结构，高层提取长程、更抽象的模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig6_stacked_lstm.png)
*堆叠 LSTM——时间维度上的层级特征提取。*

实践中，**2～3 层**通常是最佳选择。层数更深不仅收益有限，还会显著增加**深度方向**（非时间方向）的梯度消失风险，除非引入残差连接。

## 6. 真正有效的训练方法

以下默认配置适用于大多数单变量或中等规模多变量预测任务（训练窗口数在几千至几十万之间）：

| 参数 | 默认值 | 调整时机 |
| --- | --- | --- |
| 回望长度 $L$ | 序列主周期的 2～3 倍 | 通过自相关分析确定（见下文） |
| `hidden_size` | 64 | 训练窗口 ≥ 5 万时可增至 128～256 |
| `num_layers` | 2 | 数据量小时用 1 层；3 层仅在有残差连接时考虑 |
| `dropout` | 0.2 | 出现过拟合时可增至 0.5 |
| 优化器 | Adam，lr = 1e-3 | 长期训练建议改用 AdamW + 余弦学习率调度 |
| Batch size | 32～64 | 增大时需按 $\sqrt{B/32}$ 同步调整学习率 |
| 损失函数 | MSE 或 Huber | 目标含厚尾或离群点时优先选 Huber |
| 梯度裁剪 | `clip_grad_norm_(..., 1.0)` | **务必开启**——防止梯度爆炸的低成本保险 |
| 早停 | patience = 8～10 | 基于验证损失触发，并恢复最优权重 |

### 如何选择回望长度

绘制自相关函数（ACF），找到自相关系数 $|\rho_k|$ 仍高于小阈值（如 0.1）的最大滞后 $k$，再向上取整至最近的主季节周期。例如，对含日/周双重周期的小时级数据，168（一周）是自然上限。

### 一次健康训练的表现

健康的 LSTM 训练曲线中，验证损失会紧贴训练损失下降，直至达到最低点后开始回升（表明过拟合）。早停机制会在最优验证损失出现后继续等待若干 epoch，随后恢复该时刻的权重：

![训练与验证损失经过 60 个 epoch。绿色虚线标记最优验证 epoch（约 35），紫色点划线标记 patience=8 后早停触发的位置。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/02-LSTM/fig7_training_curves.png)
*带早停的 LSTM 训练曲线——恢复绿色虚线处的权重，而不是紫色点划线处。*

## 7. LSTM 和 GRU——到底该选哪个？

| 维度 | LSTM | GRU |
| --- | --- | --- |
| 门数量 | 3（遗忘、输入、输出） | 2（重置、更新） |
| 独立细胞状态 $C_t$ | 有 | 无 |
| 单元参数量 | 4 个权重矩阵 | 3 个权重矩阵 |
| 推理速度 | 基准 | 快约 25%，精度相当 |
| 典型适用场景 | 超长序列、大数据集、需最大建模容量 | 小数据集、实时推理、移动端/边缘设备 |

实证表明，在多数预测任务中，LSTM 与 GRU 的性能差距通常在噪声范围内。**建议默认使用 GRU** 以加速迭代；仅当数据量极大且依赖跨度达数百步时，才考虑切换至 LSTM。对于超过 500 步的超长序列，时序卷积网络（TCN，第 6 篇）或 Informer 类稀疏 Transformer（第 8 篇）往往表现更优。

## 8. 常见坑点

- **未对每个特征单独标准化**：LSTM 对尺度敏感，混合原始股价与百分比回报会导致训练失败。
- **跨训练/测试边界打乱时序窗口**：必须使用 `TimeSeriesSplit` 或严格按时间切分。
- **误将最后隐藏状态当作预测结果**：它只是中间特征，仍需线性输出层，且目标值也需标准化。
- **用 BiLSTM 做实时预测**：Notebook 中看似效果惊艳（因偷看未来），上线即崩。
- **未定回望长度就调大 hidden_size**：窗口太短时，扩大 cell 宽度毫无意义。
- **仅凭单次随机种子下结论**：RNN 训练噪声大，至少运行 3 次种子，报告均值 ± 标准差。

## 9. 给 LSTM 找问题

当 LSTM 预测效果不佳时，先别急着换架构。以下五类症状覆盖了绝大多数失败案例，且均有明确诊断方法。

### 症状：训练损失高，验证损失同步高

模型**欠拟合**。可能因回望窗口太短或 cell 容量不足。快速验证：固定其他参数，将 `hidden_size` 加倍（64 → 128）。若训练损失毫无变化，说明瓶颈在窗口长度——优先扩展回望窗口。

### 症状：训练损失下降，验证损失早早停滞

典型**小数据过拟合**。可尝试：增大 dropout（0.2 → 0.4）、缩小 cell、或改用 `AdamW` 并设 `weight_decay=1e-4`。**切勿盲目增加特征**——小数据下这通常适得其反。对窗口数 < 5,000 的序列，最有效手段往往是将 `hidden_size` 降至 32。

### 症状：验证损失在 epoch 间剧烈波动

常见原因：学习率过高（试 3e-4），或 batch 中含过长序列导致填充零主导损失。若窗口长度可变，务必使用 `pack_padded_sequence` 掩码，并通过 `collate_fn` 在 batch 内按长度排序。

### 症状：预测结果总是滞后真实值一步

模型退化为**恒等预测器**（即“下一步 = 当前值”）。这表明输入特征缺乏超越简单自回归的预测信号。建议：显式加入滞后特征（如 `y[t-7]`、`y[t-30]`）、构造日历变量（小时、星期、节假日标志），并先计算恒等基线 RMSE 以确认目标是否可预测。

### 症状：回测效果好，上线即崩

几乎必是**数据泄漏**。常见来源：scaler 在全序列上拟合（而非仅训练集）、target encoding 使用未来统计量、或线上有延迟的特征（如“昨日结算价”实际当晚才更新）。重新运行严格 point-in-time 的回测，问题将立即暴露。

一个实用诊断技巧是记录门控激活值：

```python
@torch.no_grad()
def gate_stats(model, batch):
    cell = model.lstm
    h, c = None, None
    for t in range(batch.size(1)):
        x = batch[:, t:t+1, :]
        out, (h, c) = cell(x, (h, c)) if h is not None else cell(x)
    # 把 bias 缓冲区分成四块，分别查看四个门的偏置。
    for name, p in cell.named_parameters():
        if "bias_ih_l0" in name:
            chunks = p.chunk(4)
            print({"i": chunks[0].mean().item(),
                   "f": chunks[1].mean().item(),
                   "g": chunks[2].mean().item(),
                   "o": chunks[3].mean().item()})
```

若训练后遗忘门偏置低于 0，说明网络学会了“每步清空记忆”——通常因序列太短，记忆无价值。此时可进一步缩短序列，或直接改用无状态前馈模型。

## 10. 上线注意事项
\nLSTM 原型易写，部署却暗藏玄机。以下是我在项目中踩过的坑。

### 跨调用的状态管理

生产环境中通常每次仅接收一个新观测值。此时有两种选择：

1. **无状态滚动窗口**：每次调用时重新编码最近 $L$ 个观测值。逻辑清晰，延迟与 $L$ 成正比。
2. **有状态流式处理**：缓存 `(h_t, C_t)` 并仅输入新观测值。延迟为 $O(1)$，但需确保缓存状态与训练一致——这意味着训练也需采用 TBPTT 等流式兼容方式。

对 $L < 200$ 的多数场景，**无状态是更稳妥的默认选择**。仅当需亚秒级高频预测且上下文极长时，才值得引入流式复杂度。

### 量化和 ONNX 导出
\nLSTM 可顺利导出至 ONNX，但细胞状态对 int8 量化极为敏感。若必须量化，建议**仅对线性投影层使用动态量化**，门控部分保留 fp16。据我实测，全量化 LSTM 在多数预测任务上会损失 5%～15% 精度。相比之下，TCN 或 Transformer 的量化鲁棒性更强，这也是它们逐渐取代 LSTM 成为低延迟流水线首选的原因之一。

### 给 scaler 打版本

我见过最多的 LSTM 线上事故，源于模型与 scaler 版本漂移。**逐特征的均值与标准差就是模型的一部分**。务必将其作为同版本工件存储并同步加载。若使用 TorchScript，可直接将标准化操作嵌入计算图：

```python
class WrappedForecaster(nn.Module):
    def __init__(self, base, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self.base((x - self.mean) / self.std)
```

这一改动已帮我避免三次线上事故。

### 监控漂移
\nLSTM 不会主动报告性能退化。仪表盘中至少监控三项指标：hold-out 尾部的滚动 RMSE、输入特征分布漂移（今日直方图 vs 训练直方图的 KL 散度）、以及残差的自相关函数（ACF）。一旦残差在 lag=1 处出现显著自相关，说明模型已失效——该重新训练了。

## 小结
\nLSTM 通过一条近似加性的**细胞状态高速公路**解决了梯度消失问题，由三个乘法门控机制协同管理：遗忘门决定丢弃什么，输入门决定写入什么，输出门决定暴露什么。由于长程梯度是遗忘门的乘积而非循环雅可比矩阵的乘积，模型得以捕捉数百步的依赖关系。

在时间序列任务中，这转化为一套实用准则：使用合理回望窗口、堆叠 1～3 层中等宽度 LSTM、结合 dropout 与早停进行正则化、长视野预测优先采用直接多步而非递归方式、BiLSTM 仅用于离线任务。下一篇将介绍 GRU——LSTM 的轻量表亲，它用更少参数实现了几乎同等的性能。

## 参考资料

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation (1997)
- Gers, Schmidhuber & Cummins, *Learning to Forget: Continual Prediction with LSTM* (2000)——遗忘门偏置 +1 技巧的出处
- Olah, *Understanding LSTM Networks*, colah.github.io (2015)——经典图示讲解
- Greff et al., *LSTM: A Search Space Odyssey*, IEEE TNNLS (2017)——LSTM 变体的实证研究

---

> 
>
> 本文是时间序列模型系列的**第 2 篇**，共 8 篇。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - **第 2 篇：LSTM —— 门控机制与长期依赖**（当前）
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn)
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - [第 8 篇：Informer —— 高效长序列预测](/zh/time-series/08-informer长序列预测)
