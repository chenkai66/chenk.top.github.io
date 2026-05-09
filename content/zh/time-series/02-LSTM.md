---
title: "时间序列模型（二）：LSTM -- 门控机制与长期依赖"
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
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/illustration_1.jpg)
## 本章要点

- 普通 RNN 为什么搞不定长序列，LSTM 又是如何解决梯度问题的
- 遗忘门、输入门、输出门的作用逻辑，以及细胞状态这条“高速公路”的工作原理
- 单步和多步时间序列预测中，如何设计 LSTM 的输入和输出
- 实用技巧：正则化方法、序列长度的选择、双向 LSTM 和堆叠 LSTM 的对比、什么时候该选 LSTM 而不是 GRU
## 准备工作

- 了解神经网络的基本概念，比如前向传播和反向传播。
- 熟悉 PyTorch 的核心内容，包括 `nn.Module`、张量和优化器。
- 最好读过本系列的第 1 部分，但这不是硬性要求。

---
## 1. LSTM 要解决的问题

普通 RNN 的隐藏状态是这样递归更新的：
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b).$$
当我从第 $T$ 步反向传播损失到更早的第 $k$ 步时，梯度会累积一长串雅可比矩阵的乘积：$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \mathrm{diag}\!\left(1 - h_t^2\right) W_h.$$
这里会出现两种情况：

- 如果 $W_h$ 的最大奇异值小于 1，梯度会快速衰减。网络根本学不到超过 ~10 步之前的信息。
- 如果大于 1，梯度会迅速爆炸，训练直接发散。

LSTM（Hochreiter & Schmidhuber, 1997）用两个状态代替单一的循环状态，同时引入三个可学习的门：一个决定记住什么，一个决定覆盖什么，还有一个决定输出什么。这样一来，状态更新变得近乎加性，梯度就能顺利走过几百步的反向传播路径。
## 2. LSTM 单元的内部结构

一个 LSTM 单元包含四个门控单元，它们共享相同的输入 $[h_{t-1}, x_t]$，输出三个 sigmoid 门和一个 $\tanh$ 候选值：
$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{遗忘门} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{输入门} \\
\tilde C_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{候选值} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{输出门}
\end{aligned}
$$
这四个信号共同决定了细胞状态的更新和隐藏状态的输出：
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t, \qquad
h_t = o_t \odot \tanh(C_t).
$$
$\odot$ 表示逐元素乘法。用通俗的话说：遗忘门 $f_t$ 决定擦除多少旧记忆，输入门 $i_t$ 控制写入多少新信息，最后通过输出门 $o_t$ 提取结果。

![LSTM 单元结构：三个 sigmoid 门加一个 tanh 候选值位于细胞状态高速公路下方。遗忘门控制旧记忆的擦除，输入门调节新记忆的写入，输出门将过滤后的状态作为隐藏状态输出。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig1_lstm_cell.png)
*LSTM 单元——三个门作用在一条细胞状态高速公路上。*

### 细胞状态的重要性

隐藏状态 $h_t$ 是网络其他部分能看到的内容，但真正的记忆存储在细胞状态 $C_t$ 中。细胞状态像一条贯穿时间的水平线，只通过逐元素乘法（$f_t$）和加法（$i_t \odot \tilde C_t$）更新，不会被新的矩阵乘法干扰。这个设计让梯度能够跨越数百步传播。

![两条并行的状态流：绿色的细胞状态高速公路以近似恒等的方式更新，承载长期记忆；紫色虚线表示经过门控过滤的隐藏状态，是下游层实际看到的内容。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig2_state_highway.png)
*细胞状态 vs 隐藏状态——两条并行的信息流。*

### 梯度流动的直观解释

对更早的细胞状态求导时，公式如下：$$\frac{\partial C_t}{\partial C_{t-1}} = f_t,$$
因此长程梯度是**遗忘门的乘积**，而不是 $\tanh$ 导数和循环矩阵的乘积：$$\frac{\partial C_T}{\partial C_k} = \prod_{t=k+1}^{T} f_t.$$
当模型需要记住某些信息时，它会学习让对应的 $f_t$ 接近 1，这样梯度也会保持接近 1。这就是 LSTM 的核心技巧。
## 3. 一个最简的 PyTorch 实现

无论是单变量还是多变量预测，下面这段代码就够了：

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

有几点需要特别注意：

- `batch_first=True` 把输入形状调整为 `(batch, seq_len, features)`，这是大多数人的习惯用法。
- 内置的 `dropout` 参数只在层间生效，不会影响时间步之间的激活值。如果需要实现 recurrent dropout，可以用 `nn.LSTMCell` 并手动应用固定掩码，或者参考 AWD-LSTM 的 `weight_drop` 方法。
- 初始化遗忘门偏置为 **+1**，让网络一开始就倾向于记住信息。PyTorch 默认不会这样设置：

```python
for name, p in model.lstm.named_parameters():
    if "bias" in name:
        n = p.size(0)
        p.data[n // 4 : n // 2].fill_(1.0)   # 遗忘门偏置
```
## 4. 从单元到预测器

时间序列预测的标准流程如下：

1. **划分窗口**：将序列切分成长度为 $L$ 的重叠片段，这就是*回望长度*。
2. **标准化**：用训练集的均值和标准差对每个特征进行标准化。
3. **训练模型**：让模型预测下一个值（单步预测），或者预测接下来的 $H$ 个值（多步预测）。
4. **验证模型**：使用按时间顺序保留的尾部数据测试，绝对不能打乱数据顺序。

对于一个带噪声的季节性信号，干净的单步预测结果大致如下：

![单步 LSTM 预测与真实序列对比。蓝色阴影区域是测试窗口残差标准差计算出的 95% 置信区间。模型能够同时捕捉 24 步和 75 步的季节性成分，但会有一小步的滞后。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig3_forecast.png)
*带噪季节信号上的单步 LSTM 预测 vs 实际值。*

### 多步预测：递归 vs 直接

当预测视野 $H > 1$ 时，通常有两种方法：

| 策略 | 方法 | 权衡 |
| --- | --- | --- |
| **递归** | 训练一个单步模型，把预测结果作为下一步输入再喂回去。 | 简单好用，但误差会累积——方差随 $\sqrt{H}$ 增长。 |
| **直接** | 训练 $H$ 个独立的预测头（或一个输出维度为 $H$ 的模型），直接预测每一步的未来值。 | 参数更多，但避免了误差反馈问题。 |

![多步预测：琥珀色递归预测的不确定性带随视野扩大呈扇形扩散；绿色直接预测的置信区间基本保持不变。虚线标记预测起点。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig4_multistep.png)
*多步预测——递归预测误差累积明显；直接预测参数开销大，但置信区间更紧。*

实际应用中，常用一种折中方案：**seq2seq + teacher forcing**。LSTM 编码器读取回望窗口并生成最终的 $(h, C)$ 状态对，LSTM 解码器逐步生成 $H$ 个输出。在训练阶段，解码器以一定概率接收*真实值*（而非自己的预测值）作为输入，这种技术称为 scheduled sampling。这是目前生产环境中最常用的预测方法。
## 5. 架构变体

### 双向 LSTM（BiLSTM）

双向 LSTM 的工作方式很简单：一个 LSTM 正向运行，另一个 LSTM 反向运行。每一步将两个隐藏状态拼接起来：$$y_t = [\,\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\,].$$

![双向 LSTM：紫色正向链从左到右读取，琥珀色反向链从右到左读取，每一步的输出是两个隐藏状态的拼接。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig5_bilstm.png)
*双向 LSTM——每一步结合过去和未来的上下文信息。*

适合用在序列标注、分类、缺失值填补等任务上。只要推理时能拿到完整序列，就可以考虑它。但千万别用在实时预测场景：训练时偷看 $x_{t+1}, x_{t+2}, \dots$，推理时却要预测 $x_{t+1}$，这就是数据泄漏，上线肯定出问题。

### 堆叠（深层）LSTM

堆叠多层 LSTM 的好处是让高层处理更平滑、更抽象的特征。第 1 层接收原始输入，第 2 层接收第 1 层的隐藏状态，依此类推。

![三层堆叠 LSTM：每一层从左到右递归，并将隐藏状态传递给上一层。底层捕捉短时局部结构，高层提取长程、更抽象的模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig6_stacked_lstm.png)
*堆叠 LSTM——时间维度上的层级特征提取。*

实际使用中，**2～3 层**效果最佳。再深的话，如果没有残差连接，不仅提升有限，还会显著增加深度方向（不是时间方向）梯度消失的风险。
## 6. 真正有效的训练方法

以下默认设置适用于大多数单变量或中等多变量预测问题，训练窗口数量在几千到几十万之间：

| 参数 | 默认值 | 调整时机 |
| --- | --- | --- |
| 回望长度 $L$ | 序列主要周期的 2~3 倍 | 用自相关分析选择——见下文 |
| `hidden_size` | 64 | 训练窗口 $\geq$ 5 万时调到 128~256 |
| `num_layers` | 2 | 数据量少用 1 层；3 层仅在有残差连接时考虑 |
| `dropout` | 0.2 | 出现过拟合时调到 0.5 |
| 优化器 | Adam，lr = 1e-3 | 长时间训练改用 AdamW + 余弦学习率调度 |
| Batch size | 32~64 | 增大时按 $\sqrt{B/32}$ 调整学习率 |
| 损失函数 | MSE 或 Huber | 目标分布有厚尾或离群点时用 Huber |
| 梯度裁剪 | `clip_grad_norm_(..., 1.0)` | 必须开启——防止梯度爆炸的低成本保险 |
| 早停 | patience = 8~10 | 根据验证集损失触发，并恢复最优权重 |

### 如何选择回望长度

画出自相关函数（ACF），找到自相关系数 $|\rho_k|$ 仍高于某个小阈值（比如 0.1）的最大滞后 $k$，然后向上取整到最近的主季节周期。对于包含日和周双重周期的小时级数据，168（一周）是一个自然上限。

### 一次健康训练的表现

健康的 LSTM 训练曲线中，验证损失会紧贴训练损失下降，直到达到最低点后开始回升——这说明模型开始过拟合训练数据。早停机制会在最优验证损失出现后再等待若干个 epoch，随后恢复最优权重：

![训练与验证损失经过 60 个 epoch。绿色虚线标记最优验证 epoch（约 35），紫色点划线标记 patience=8 后早停触发的位置。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig7_training_curves.png)
*带早停的 LSTM 训练曲线——恢复绿色虚线处的权重，而不是紫色点划线处。*
## 7. LSTM 和 GRU——到底该选哪个？

| 维度 | LSTM | GRU |
| --- | --- | --- |
| 门数 | 3 个（遗忘、输入、输出） | 2 个（重置、更新） |
| 独立细胞状态 $C_t$ | 有 | 无 |
| 单元参数 | 4 个权重矩阵 | 3 个权重矩阵 |
| 速度 | 基线 | 快约 25%，精度相当 |
| 典型适用 | 长序列、超大数据集、需要最大容量 | 小数据集、实时推理、移动/边缘部署 |

实际测试中，大多数预测任务里 LSTM 和 GRU 的性能差距很小，基本在误差范围内。我建议**优先用 GRU**，因为它训练更快；如果数据量特别大且依赖关系跨越数百步，再换成 LSTM。对于超过 500 步的长序列，时序卷积网络（第 6 部分）或 Informer 类稀疏 Transformer（第 8 部分）通常表现更好。
## 8. 常见坑点

- **忘记对每个特征单独标准化**。LSTM 对尺度非常敏感，如果把原始股价和百分比回报混在一起输入，模型训练效果会很差。
- **跨训练/测试边界打乱时间序列窗口**。要用 `TimeSeriesSplit` 或者按时间顺序固定切分。
- **直接把最后的隐藏状态当成预测结果**。它只是个特征向量，后面还得加一个线性输出层，目标值也需要标准化。
- **用 BiLSTM 做预测**。在 notebook 里看起来效果很好，因为它偷看了未来，但一上线就崩了。
- **没定好回望长度就开始调 hidden size**。回望长度决定了 cell 能看到什么信息；如果窗口太短，加宽 cell 也没用。
- **只跑一个随机种子就下结论**。RNN 训练噪声很大，至少跑 3 个种子，取均值 ± 标准差。
## 9. 给 LSTM 找问题

LSTM 预测不准时，别急着换架构。五个常见问题几乎能解释所有失败原因，而且每个问题都有明确的排查方法。

### 症状：训练损失高，验证损失跟着高

模型**欠拟合**了。可能是回看窗口太短，或者隐藏单元太少。快速测试一下容量：其他参数不动，把 `hidden_size` 加倍（64 → 128）。如果训练损失一点没变，说明瓶颈是回看长度——先扩展窗口，再考虑别的。

### 症状：训练损失下降，验证损失早早停滞

这是小数据集上的典型**过拟合**。可以加大 dropout（0.2 → 0.4），缩小隐藏单元，或者用 `AdamW` 加上 `weight_decay=1e-4`。千万别加更多特征——在小数据集上，这通常会让问题更糟。对于窗口数不到 5,000 的序列，最有效的办法通常是把 `hidden_size` 降到 32。

### 症状：验证损失在 epoch 间剧烈波动

两个常见原因：学习率太高（试试 3e-4），或者 batch 中包含过长的序列，填充的零主导了损失计算。如果窗口长度不固定，一定要传入 `pack_padded_sequence` 的 mask，并且用 `collate_fn` 按长度对 batch 排序。

### 症状：预测结果总是比真实值滞后一步

模型退化成了**恒等预测器**——它学会了“下一步等于当前值”是最优策略。这说明输入特征除了简单的自回归外，没有提供任何额外的预测信息。显式加入滞后特征（比如 `y[t-7]`、`y[t-30]`），构造日历变量（小时、星期、节假日标志），并先计算恒等基线的 RMSE，确认目标是否真的可预测。

### 症状：回测效果很好，上线就崩

几乎可以肯定是**数据泄漏**。常见来源包括：在全序列上拟合 scaler 而不是仅用训练窗口；用未来统计量做 target encoding；离线计算时即时可用但线上有延迟的特征（比如昨日结算价要到晚上 8 点才更新）。重新跑回测，严格使用 point-in-time 特征存储，问题就会暴露。

一个有用的诊断方法是记录门控激活值：

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

如果训练后遗忘门偏置低于 0，说明网络学会了“每一步都清空记忆”——这通常是因为序列太短，记忆没有意义。要么进一步缩短序列，要么直接换成无状态的前馈模型。
## 10. 上线注意事项

LSTM 模型看起来容易上手，但部署时会踩不少坑。以下是我在项目中遇到的一些教训。

### 跨调用的状态管理

线上环境通常每次只会收到一个新观测值，而不是一整个窗口。这时有两种选择：

1. **无状态滚动窗口**：每次调用时重新编码最近的 $L$ 个观测值。简单直接，延迟与 $L$ 成正比，逻辑清晰。
2. **有状态流式处理**：在调用之间缓存 `(h_t, C_t)`，每次只传入新观测值。延迟是 $O(1)$，但必须确保缓存的状态和模型训练时一致。这意味着训练时也要采用流式方法，比如 TBPTT。

对于大多数预测任务，如果 $L < 200$，无状态是更好的默认选择。只有在长上下文、亚秒级高频预测场景下，才值得为流式处理增加复杂度。

### 量化和 ONNX 导出

LSTM 导出到 ONNX 很顺利，但细胞状态对 int8 量化非常敏感。如果非得量化，建议**只对线性投影部分使用动态量化**，门控部分保持 fp16。根据我的经验，完全量化的 LSTM 在大多数预测任务上会损失 5%~15% 的精度。相比之下，TCN 和 Transformer 的量化损失要小得多，这也是它们逐渐取代 LSTM 成为延迟敏感流水线首选架构的原因之一。

### 给 scaler 打版本

我见过最多的 LSTM 线上问题，是模型和 scaler 不同步漂移。逐特征的均值和方差**就是模型的一部分**。把它们作为同版本的姊妹工件存储，并一起加载。如果用 TorchScript 导出，可以直接将标准化操作嵌入计算图中：

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

这个改动帮我避免了 3 次线上事故。

### 监控漂移

LSTM 预测器不会主动告诉你它失效了。仪表盘里至少盯住三件事：滚动尾部 hold-out 数据上的 RMSE、输入特征分布的漂移（今天特征直方图与训练直方图的 KL 散度）、每步残差的 ACF。一旦残差在滞后 1 步出现自相关，说明模型已经无法捕捉结构了——该重新训练了。
## 总结

LSTM 解决了梯度消失问题，靠的是一条加性的**细胞状态通道**来传递记忆。网络通过三个乘法门控机制管理这条通道：遗忘门决定丢弃哪些信息，输入门决定写入哪些内容，输出门决定暴露哪些结果。长程梯度是遗忘门的乘积，而不是循环雅可比矩阵的乘积，这让模型能够捕捉几百步长的依赖关系。

在时间序列任务中，这总结出了一些实用经验：用合理的回溯窗口处理序列，堆叠 1 到 3 层中等宽度的 LSTM，用 dropout 和早停做正则化，长预测范围时优先选择直接多步预测而非递归方式，BiLSTM 只适合离线任务。接下来我会讲 GRU——LSTM 的简化版，它用更少参数实现了几乎相同的效果。
## 参考资料

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation (1997)
- Gers, Schmidhuber & Cummins, *Learning to Forget: Continual Prediction with LSTM* (2000)——遗忘门偏置 +1 这一招的出处
- Olah, *Understanding LSTM Networks*, colah.github.io (2015)——经典图示讲解
- Greff et al., *LSTM: A Search Space Odyssey*, IEEE TNNLS (2017)——LSTM 各种变体的实证研究

---

> 
>
> 本文是时间序列模型系列的**第 2 篇**，共 8 篇。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - **第 2 篇：LSTM —— 门控机制与长期依赖**（当前）
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru/)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构/)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn/)
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - [第 8 篇：Informer —— 高效长序列预测](/zh/time-series/08-informer长序列预测/)
