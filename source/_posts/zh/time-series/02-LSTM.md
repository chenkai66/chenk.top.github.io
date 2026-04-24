---
title: "时间序列模型（二）：LSTM -- 门控机制与长期依赖"
date: 2024-11-20 09:00:00
tags:
  - 时间序列
  - 深度学习
  - LSTM
categories: 时间序列
series:
  name: "时间序列模型"
  part: 2
  total: 8
lang: zh-CN
mathjax: true
description: "LSTM 的遗忘门、输入门和输出门如何解决梯度消失问题。完整的 PyTorch 时间序列预测代码和实用调参技巧。"
disableNunjucks: true
series_order: 2
---

## 本章要点

- 为什么普通 RNN 在长序列上失败，LSTM 如何修复梯度问题
- 每个门（遗忘门、输入门、输出门）的直觉与"细胞状态高速公路"
- 如何为单步与多步时间序列预测构建 LSTM 的输入/输出
- 实战配方：正则化、回望长度选择、双向 vs 堆叠 LSTM、LSTM vs GRU 的取舍

## 前置知识

- 神经网络基础（前向传播、反向传播）
- 熟悉 PyTorch（`nn.Module`、张量、优化器）
- 本系列第一部分（推荐但非必需）

---

## 1. LSTM 要解决的问题

普通 RNN 的隐藏状态是递归更新的：
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b).$$

把第 $T$ 步的损失反传到很早的第 $k$ 步，梯度会乘上一长串雅可比矩阵：
$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \mathrm{diag}\!\left(1 - h_t^2\right) W_h.$$

两种坏情况都会发生：

- 如果 $W_h$ 的最大奇异值小于 1，乘积**指数衰减**——网络无法从超过 ~10 步的历史中学到任何东西。
- 如果大于 1，乘积**指数爆炸**——训练直接发散。

LSTM（Hochreiter & Schmidhuber, 1997）的做法是把单一的循环状态拆成**两个**状态、外加三个可学习的门，由这些门来决定记什么、覆盖什么、输出什么。结果是一个沿时间轴近乎加性的更新方式，让梯度有机会"走"完几百步的反向路径。

## 2. LSTM 单元的解剖

在一个 LSTM cell 内部，四个子单元共享同一份输入 $[h_{t-1}, x_t]$，给出三个 sigmoid 门和一个 $\tanh$ 候选：

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{遗忘门} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{输入门} \\
\tilde C_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{候选} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{输出门}
\end{aligned}
$$

四个量合成为细胞状态更新和隐藏输出：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t, \qquad
h_t = o_t \odot \tanh(C_t).
$$

$\odot$ 是逐元素乘法。**用大白话翻译一下**：擦掉 $1 - f_t$ 比例的旧记忆，写入 $i_t$ 比例的新候选，最后通过 $o_t$ 这副"滤镜"看一眼结果。

![LSTM 单元结构：三个 sigmoid 门加一个 tanh 候选位于细胞状态高速公路下方。遗忘门与输入流相乘擦除旧记忆，输入门缩放候选写入新记忆，输出门把过滤后的状态作为隐藏状态暴露出去。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig1_lstm_cell.png)
*LSTM 单元——三个门作用在一条细胞状态高速公路上。*

### 为什么细胞状态如此关键

隐藏状态 $h_t$ 是网络其它部分能看到的东西，但**真正的记忆**住在细胞状态 $C_t$ 里。$C_t$ 沿着时间轴形成一条不间断的水平线，只被逐元素乘法（$f_t$）和加法（$i_t \odot \tilde C_t$）触碰——**从不**被新的矩阵乘法重写。这一个设计选择，就是梯度能够穿越数百步的根本原因。

![两条并行的状态流：绿色的细胞状态高速公路用近似恒等的更新承载长期记忆；紫色虚线是被门过滤后的隐藏状态，是下游层真正看到的视图。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig2_state_highway.png)
*细胞状态 vs 隐藏状态——两条并行的信息流。*

### 把梯度流写出来

把细胞更新对更早的细胞状态求导：
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t,$$

于是长程梯度就是**遗忘门的乘积**，而不是 $\tanh$ 导数和循环矩阵的乘积：
$$\frac{\partial C_T}{\partial C_k} = \prod_{t=k+1}^{T} f_t.$$

只要模型想记住某个分量，它就能学会让对应的 $f_t$ 接近 1，于是该分量的梯度也保持接近 1。整个魔法就这一招。

## 3. 一个最小可用的 PyTorch 实现

不论是单变量还是多变量预测，下面这段代码就够用了：

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

几个不那么显然的细节：

- `batch_first=True` 让输入形状变成 `(batch, seq_len, features)`，几乎所有人都希望是这种约定。
- 内置的 `dropout` 参数**只作用于层与层之间**，不会在时间步之间丢弃激活。要想做 recurrent dropout，得用 `nn.LSTMCell` 自己实现，或者借鉴 AWD-LSTM 的 `weight_drop` 技巧。
- 把遗忘门偏置初始化为 **+1**，让网络从"倾向记住"开始。PyTorch 默认不会这么做：

```python
for name, p in model.lstm.named_parameters():
    if "bias" in name:
        n = p.size(0)
        p.data[n // 4 : n // 2].fill_(1.0)   # 遗忘门偏置
```

## 4. 从单元到预测器

时间序列预测的标准流程：

1. **窗口化序列**：把序列切成长度为 $L$ 的重叠窗口——这就是*回望长度*。
2. **每个特征单独标准化**：用训练集的均值和标准差。
3. 训练模型预测下一个值（单步），或下 $H$ 个值（多步）。
4. 在**按时间顺序**留出的尾部上验证——绝不打乱。

在含噪的季节性信号上，一个干净的单步预测大致是这样的：

![单步 LSTM 预测与真实序列的对比。蓝色阴影带是测试窗口残差标准差给出的 95% 区间。模型同时跟住了 24 步与 75 步两种季节分量，只有约一步的特征性滞后。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig3_forecast.png)
*在带噪季节信号上的单步 LSTM 预测 vs 实际值。*

### 多步预测：递归 vs 直接

当预测视野 $H > 1$ 时，常见两种策略：

| 策略 | 做法 | 取舍 |
| --- | --- | --- |
| **递归（recursive）** | 训练一个单步模型，把它的预测当作下一步输入再喂回去。 | 简单，但误差累积——方差按 $\sqrt{H}$ 增长。 |
| **直接（direct）** | 训练 $H$ 个独立的预测头（或一个 $H$ 维输出的模型），每个头预测某一未来步。 | 参数更多，但没有误差反馈环。 |

![多步预测：黄色递归预测的不确定性带随视野扩大形成扇形展开；绿色直接预测保持基本不变的带宽。点划竖线标示预测原点。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig4_multistep.png)
*多步预测——递归会累积误差；直接预测以参数为代价换取更紧的置信区间。*

实战中常见的折中是 **seq2seq + teacher forcing**：LSTM 编码器把回望窗口压缩成一对 $(h, C)$，LSTM 解码器逐步生成 $H$ 个输出；训练时解码器以一定概率喂入*真实*的上一步值（scheduled sampling），而非自己的预测。这是当前生产环境里最常见的范式。

## 5. 架构变体

### 双向 LSTM（BiLSTM）

BiLSTM 同时运行一个正向 LSTM 和一个反向 LSTM，并把两者的隐藏状态拼接起来：
$$y_t = [\,\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\,].$$

![双向 LSTM：紫色正向链从左到右扫描，黄色反向链从右到左扫描，每个时间步的输出是两条隐藏状态的拼接。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig5_bilstm.png)
*双向 LSTM——每一步都同时利用过去与未来上下文。*

**适用于**序列标注、分类、缺失值插补——任何在推理时**整段序列已知**的场景。**不要**用于实时预测：训练时让模型偷看 $x_{t+1}, x_{t+2}, \dots$ 再预测 $x_{t+1}$，本质上是数据泄漏，上线必崩。

### 堆叠（深层）LSTM

堆叠多层让上层处理更平滑、更慢的特征：第 1 层看原始输入，第 2 层看第 1 层的隐藏状态，依此类推。

![三层堆叠 LSTM：每层在时间方向递归，并把隐藏状态向上传给下一层。底层捕捉短的局部结构，高层组合出更长程、更抽象的模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig6_stacked_lstm.png)
*堆叠 LSTM——时间维度上的层级特征提取。*

实战中**2~3 层**是预测任务的甜点。再深一般没什么收益，且在没有残差连接时会显著加重**深度方向**（不是时间方向）的梯度消失。

## 6. 一套真能跑出效果的训练配方

下面这套默认值，对几千到几十万训练窗口的单/多变量预测问题基本都管用：

| 超参 | 默认 | 何时偏离默认 |
| --- | --- | --- |
| 回望长度 $L$ | 序列主导周期的 2~3 倍 | 用自相关分析挑选——下文有方法 |
| `hidden_size` | 64 | 训练样本 $\geq$ 5 万时可上 128~256 |
| `num_layers` | 2 | 数据少时 1 层；3 层只在加残差时考虑 |
| `dropout` | 0.2 | 看到过拟合就上调到 0.5 |
| 优化器 | Adam，lr = 1e-3 | 长训练用 AdamW + 余弦退火 |
| Batch size | 32~64 | 增大时学习率按 $\sqrt{B/32}$ 放缩 |
| 损失 | MSE 或 Huber | 目标分布厚尾/有离群值时用 Huber |
| 梯度裁剪 | `clip_grad_norm_(..., 1.0)` | 永远开——便宜的爆炸梯度保险 |
| 早停 | patience = 8~10 | 监控验证集损失，并恢复最优权重 |

### 怎么挑回望长度

画自相关函数（ACF），找到自相关系数 $|\rho_k|$ 仍然超过某个小阈值（比如 0.1）的最大滞后 $k$，再向上取整到下一个主季节周期。对于含日和周双重周期的小时级数据，168（一周）是个自然上限。

### 一次健康训练的样子

健康的 LSTM 训练曲线里，验证损失会紧贴训练损失下降，到达谷底后开始上抬——这就是模型开始记忆训练集的信号。早停在最优验证损失之后再等若干个 epoch，然后回滚到最优权重：

![训练与验证损失走过 60 个 epoch。绿色虚线标出最优验证 epoch（约 35），紫色点划线标出 patience=8 后早停触发的位置。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/02-LSTM/fig7_training_curves.png)
*带早停的 LSTM 训练曲线——恢复绿色那一刻的权重，不是紫色那一刻。*

## 7. LSTM vs GRU——该选哪个？

| 维度 | LSTM | GRU |
| --- | --- | --- |
| 门数 | 3 个（遗忘、输入、输出） | 2 个（重置、更新） |
| 独立细胞状态 $C_t$ | 有 | 无 |
| 单元参数 | 4 个权重矩阵 | 3 个权重矩阵 |
| 速度 | 基线 | 快约 25%，精度相当 |
| 典型适用 | 长序列、超大数据集、需要最大容量 | 小数据集、实时推理、移动/边缘部署 |

经验上，大多数预测任务里 LSTM 与 GRU 的差距都在噪声范围内。**默认从 GRU 起步**，迭代更快；当数据量足够大且依赖跨数百步时再换 LSTM。一旦序列超过约 500 步，两者通常都会被时序卷积网络（第 6 部分）或 Informer 类稀疏 Transformer（第 8 部分）超越。

## 8. 常见踩坑

- **忘了按特征独立标准化**。LSTM 对量纲敏感，把原始股价和百分比收益率混在一份输入里，肯定训不动。
- **跨训练/测试边界打乱时间序列窗口**。用 `TimeSeriesSplit` 或固定的时间切分。
- **把最后一步隐藏状态当作预测**。它只是一个特征向量，仍然需要线性输出头，并且**目标值也要标准化**。
- **拿 BiLSTM 做预测**。在 notebook 里看上去无敌（因为它在偷看未来），上线就崩。
- **没定好回望长度就先调 hidden size**。回望长度决定了 cell 能看到什么；窗口太短，把 cell 加宽也没意义。
- **只跑一个随机种子就下结论**。RNN 训练噪声大，至少跑 3 个种子取均值±方差。

## 总结

LSTM 通过一条带有近似加性更新的**细胞状态高速公路**解决了梯度消失问题，并用三个乘性门控制这条高速公路：遗忘门决定擦除什么，输入门决定写入什么，输出门决定暴露什么。因为长程梯度变成了**遗忘门的乘积**而不是循环雅可比的乘积，模型可以学到几百步长的依赖。

落到时间序列，这就翻译成一套实战配方：用合理的回望长度对序列窗口化，堆 1~3 层中等宽度的 LSTM，用 dropout + 早停做正则化，长视野优先选直接多步而非递归，BiLSTM 仅用于离线任务。下一部分进入 GRU——LSTM 的轻量近亲，用更少的参数做几乎同样的事。

## 参考资料

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation (1997)
- Gers, Schmidhuber & Cummins, *Learning to Forget: Continual Prediction with LSTM* (2000)——遗忘门偏置 +1 这一招的出处
- Olah, *Understanding LSTM Networks*, colah.github.io (2015)——经典图示讲解
- Greff et al., *LSTM: A Search Space Odyssey*, IEEE TNNLS (2017)——LSTM 各种变体的实证研究

---

**系列导航**

