---
title: "时间序列模型（六）：时序卷积网络 (TCN)"
date: 2024-11-15 09:00:00
tags:
  - 时间序列
  - 深度学习
  - TCN
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "TCN 用因果膨胀卷积换取并行训练和指数级感受野。完整 PyTorch 实现，附交通流和多变量传感器两个实战案例。"
disableNunjucks: true
series_order: 6
translationKey: "time-series-6"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/illustration_1.png)

2010 年代的大部分时间里，提到深度学习处理时间序列，默认就是 LSTM。直到 2018 年，Bai、Kolter 和 Koltun 发表了一篇题为 *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* 的论文。结论简单得让人有些意外：将多个 1D 卷积堆叠起来，确保它们是因果卷积（不偷看未来数据），让卷积核的步距按指数增长（dilation），再加上残差连接，直接训练即可。结果表明，这种称为 **Temporal Convolutional Network**（TCN）的模型在各个任务中要么与 LSTM/GRU 打平，要么直接胜出——而且训练速度提高了好几倍，因为前向传播的每个时间步可以并行计算。

这一章我将解释这种方法为何如此有效：首先推导 dilation 的感受野公式，了解其作用；然后逐步拆解残差块的设计细节；最后通过两个工业级案例——交通流量预测和多变量传感器预测——进行总结。代码基于 PyTorch 实现，可以直接使用。
## 你将学到的内容

- 为什么要做“诚实的预测”，因果一维卷积为什么必不可少，左侧 padding 是如何实现这一点的。
- 膨胀卷积如何让感受野以 $\mathcal{O}(2^L)$ 的速度增长，而不是 $\mathcal{O}(L)$。
- TCN 残差块的具体结构：两个膨胀因果卷积 + 权重归一化 + dropout + 1x1 残差连接。
- TCN 和 LSTM/GRU/Transformer 的直接对比：训练时间、显存占用和预测精度。
- 两个实际案例：每小时交通流量预测和多变量 IoT 传感器数据预测。

**前置要求**：熟悉第 2 部分（LSTM）和第 5 部分（Transformer）。会用 PyTorch 的 `nn.Conv1d`，并能理解基本的复杂度分析。  

---
## 为什么当年用 LSTM 做时间序列那么痛苦

在 TCN 出现之前，深度学习处理时间序列的方法基本固定：堆两层 LSTM，想花哨点就加个 attention，然后慢慢训练。虽然能跑通，但每个环节都让人头疼：

- **前向传播必须按顺序来**。要计算 $h_t$ 就得先知道 $h_{t-1}$， GPU 只能干等着上一步完成。就算硬件性能无限强，序列长度翻倍，实际耗时也跟着翻倍。
- **梯度消失或爆炸问题**。反向传播需要经过 $L$ 次乘法操作。 LSTM 的门机制确实缓解了一些问题，但一旦序列长度超过 200 步，模型就变得很不稳定。为了稳住训练，大家不得不折腾梯度裁剪、 layer norm 和各种初始化技巧。
- **隐状态是个黑箱**。问“模型为什么会这样预测？”通常找不到答案，因为隐状态将所有信息混在一起，难以理清。
- **超参数组合复杂**。层数、隐藏维度、门的变体、 dropout 类型和 recurrent dropout 的位置等因素互相影响。选错组合可能会浪费一天的训练时间。

TCN 的思路很简单：用可以并行计算的卷积代替递归结构，用明确的感受野代替隐状态那种模糊的记忆，再通过残差连接让梯度更稳定。表达能力一样强，但复杂度大大降低。
## 一维卷积：但必须是因果的

标准的一维卷积用一个长度为 $k$ 的滤波器 $f$ 在输入序列 $x$ 上滑动：

$$y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i+\lfloor k/2 \rfloor}.$$

这种居中形式让 $t$ 时刻的输出同时读取了过去和未来的输入。在预测任务里，这就是**信息泄露**——你不能靠明天的交通数据来预测明天的交通流量。

**因果卷积**把滤波器向右移动，确保 $t$ 时刻的输出只依赖于 $1, \ldots, t$ 的输入：

$$y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i}.$$

实现时，在输入左侧填充 $k - 1$ 个零，然后调用普通的 `nn.Conv1d`。卷积完成后，去掉右侧多余的填充部分，保证输出长度和输入一致。

![因果 vs 非因果一维卷积，t = 6](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/fig2_causal_convolution.png)

图中绿色的输出 $y_6$ 在两侧是一样的，但它依赖的输入（橙色）不同。左边的非因果卷积读到了 $x_7$，这部分属于阴影标注的"未来"区域——预测任务绝对不允许这样。右边的因果卷积则始终只看左侧。

PyTorch 实现如下：

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """带左填充和右侧裁剪的一维卷积。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, : -self.padding]
        return y
```

两个需要注意的细节：

1. 填充量是 $(k-1) \cdot d$，取决于 dilation $d$（后面会讲到）。
2. 卷积后要裁剪的是**右侧**。常见错误是裁剪了左侧，结果悄无声息地破坏了序列开头的部分。
## 膨胀：用线性深度实现指数级感受野

核大小 $k = 3$ 的因果卷积堆叠 $L$ 层，感受野是 $1 + 2L$。这是线性增长。如果想看到 200 步之前的数据，需要堆 100 层网络，这显然不现实。

**膨胀卷积**通过在卷积核的 tap 之间插入 $d$ 的间隔来扩展感受野：

$$y_t = \sum_{i=0}^{k-1} f_i \, x_{t-d \cdot i}.$$

如果每层的 dilation 翻倍（$d_\ell = 2^{\ell-1}$），那么 $L$ 层网络的感受野会变成：

$$\text{RF}(L) = 1 + (k - 1)\sum_{\ell=1}^{L} d_\ell = 1 + (k - 1)(2^L - 1).$$

当 $k = 3$、$L = 8$ 时，感受野达到 **511 步**——足够覆盖一周的小时级数据。参数量和 8 层普通卷积一样，但覆盖范围却是指数级增长。

![膨胀因果卷积的感受野](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/fig1_dilated_convolution.png)

图中展示了所有对顶层绿色输出节点有贡献的输入。 dilation 分别为 1、 2、 4、 8，让四层网络看起来像一棵稀疏的树。这种稀疏结构正是它能覆盖远距离的原因。

一个小工具，帮你快速计算需要的层数：

```python
import math

def required_layers(receptive_field: int, kernel_size: int = 3) -> int:
    """返回最小的 L，满足 1 + (k-1)(2**L - 1) >= receptive_field。"""
    L = (receptive_field - 1) / (kernel_size - 1) + 1
    return max(1, math.ceil(math.log2(L)))
```

调用 `required_layers(168, kernel_size=3)` 返回 `7`——正好适合需要回看一周小时级数据的场景。
## TCN 残差块

堆叠膨胀因果卷积只是 TCN 的一半，另一半是包裹这些卷积的残差块。 Bai 等人最终采用了以下结构（几乎和 Oord 等人在 WaveNet 中的设计完全一致，除了激活函数的选择）：

![TCN 残差块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/fig3_residual_block.png)

用数学公式表示，给定输入 $x$：

$$F(x) = \mathrm{Dropout}\!\big(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_2 \, \mathrm{Dropout}(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_1 \, x))))) \big),$$

$$o = \mathrm{ReLU}\!\big( F(x) + W_{\text{skip}} \, x \big).$$

这里有三个关键设计点：

- **每块两个卷积**。单个卷积几乎没什么效果，两个卷积能让模块学到非平凡的变换，同时避免网络深度过大。
- **权重归一化**。我发现 batch norm 在长序列上表现不好（统计量会随位置漂移）。 weight norm 把每个滤波器的方向和大小解耦，不干扰激活值，训练更稳定。
- **1x1 残差投影**。如果输入和输出通道数一致，残差连接就是恒等映射；如果不一致，用一个 1x1 卷积进行投影，代价可以忽略。

以下是 PyTorch 实现：

```python
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation,
        ))
        conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation,
        ))
        self.padding = (kernel_size - 1) * dilation
        self.conv1, self.conv2 = conv1, conv2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in (self.conv1, self.conv2):
            nn.init.normal_(layer.weight, 0.0, 0.01)

    def _causal(self, conv, x):
        y = conv(x)
        return y[:, :, : -self.padding] if self.padding > 0 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self._causal(self.conv1, x)))
        out = self.dropout(self.relu(self._causal(self.conv2, out)))
        return self.relu(out + self.skip(x))
```

这个模块简单到很多人直接内联使用。但把它封装成独立模块有两个好处：一是感受野计算更直观，二是必要时可以轻松把 weight norm 替换成 layer norm。
## 搭建网络

一个完整的 TCN 由多个残差块堆叠而成，这些残差块的 dilation 参数呈指数增长。最后可以选择加一个 1x1 卷积，将输出映射到目标维度。

```python
class TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 channels: list[int], kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_size
        for i, c in enumerate(channels):
            layers.append(TCNResidualBlock(
                prev, c, kernel_size, dilation=2 ** i, dropout=dropout,
            ))
            prev = c
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev, output_size, kernel_size=1)
        self._channels = channels
        self._k = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self._k - 1) * (2 ** len(self._channels) - 1)
```

配置时需要注意以下几点：

- **通道数**。大多数论文使用固定宽度，比如 `[64] * 8`。如果输出维度远大于输入，可以在靠近 head 的层增加宽度。
- **核大小**。$k = 3$ 是标准选择。$k = 5$ 或 $7$ 会让参数翻倍，但很少提升精度；想要更大的感受野，增加 dilation 几乎总是更划算。
- **Dropout**。 0.2 是稳妥的默认值。在小数据集上可以调到 0.3 到 0.5。
## TCN vs RNN：架构层面的对比

这张信息流图比任何性能测试表都更能直观说明速度差异：

![RNN 顺序依赖 vs TCN 并行前向](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/fig4_tcn_vs_rnn.png)

在 RNN 的部分，每条红色箭头都表示严格的顺序依赖。 GPU 能并行计算单个 cell 内部的操作，但必须等第 $t$ 步完成后才能开始第 $t+1$ 步。即使硬件支持无限并行，前向传播的实际耗时仍然会随着序列长度线性增长。

TCN 就不一样了。每个输出节点只依赖固定的一组输入节点。同一个卷积核在整个序列上滑动，整层操作本质上是一次大规模矩阵乘法， GPU 可以通过一次 kernel launch 完成。

下面是单 GPU 上每 epoch 的实际训练时间对比：

![训练时间和 TCN 相对 LSTM 的加速比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-时序卷积网络TCN/fig5_parallel_training.png)

两个重点：

1. **训练时间扩展性很重要**。当序列长度 $L = 128$ 时，四种架构的速度差不多；但到了 $L = 1024$， TCN 比 LSTM 快 3-4 倍，比朴素 Transformer （attention 复杂度为 $L^2$）快约 6 倍。这正是大多数实际时间序列问题的典型区间。
2. **推理性能基本持平**。推理时， RNN 和 TCN 的延迟通常在 1.5 倍以内。这个差距主要体现在训练阶段，而不是推理阶段。如果只关心单样本延迟，两者都能满足需求。

什么时候该用哪种模型？这里有一个简单的决策表：

| 场景 | 推荐 | 理由 |
|---|---|---|
| 固定窗口、有 GPU | TCN | 训练并行化，感受野可控 |
| 序列长度变化大、 padding 不可接受 | LSTM/GRU | 原生支持变长序列，没有 padding 开销 |
| 流式 / 在线推理，逐步进数据 | LSTM/GRU | 隐状态天然适合逐步更新 |
| 多变量、跨特征交互需要 attention | Transformer / Informer | attention 显式建模两两关系 |
| 不确定 | 先试 TCN | 训练速度快，超参数少 |
## PyTorch 实现：完整的训练循环

前面提到的两个类是核心部分，训练循环也很常规：

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_tcn(model, train_loader, val_loader,
              num_epochs=50, lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 factor=0.5, patience=5)
    crit = nn.MSELoss()
    best = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "tcn_best.pt")
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: train {train_loss:.4f} val {val_loss:.4f}")
```

这里有两个重点：梯度裁剪对 TCN 来说不是必须的（残差连接和 weight norm 已经让梯度表现良好），但加上也无妨。`ReduceLROnPlateau` 比固定学习率调度更稳健，因为合适的学习率取决于数据集和感受野。

一个用于构造单变量窗口的小工具函数：

```python
import numpy as np

def make_windows(series: np.ndarray, history: int, horizon: int):
    """将一维序列转换为 (X, y) 张量，用于直接多步预测。"""
    n = len(series) - history - horizon + 1
    X = np.stack([series[i : i + history] for i in range(n)])
    y = np.stack([series[i + history : i + history + horizon] for i in range(n)])
    X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, history)
    y = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1, horizon)
    return X, y
```
## 案例 1：每小时交通流量预测

**任务**：用单个高速公路传感器过去一周（168 小时）的车流量数据，预测未来 24 小时的车流量。这是单变量问题，有明显的日周期和周周期，偶尔会出现事件驱动的流量高峰。

**感受野需求**：我希望模型输出时能看到至少一周的历史数据。当 $k = 3$、$L = 7$ 时，$\text{RF} = 1 + 2 \cdot 2 \cdot 127 = 509$，完全够用。

```python
def synthetic_traffic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 1000 + 500 * np.sin(2 * np.pi * t / 24)
    weekly = 200 * np.sin(2 * np.pi * t / (24 * 7))
    trend = 0.05 * t
    noise = rng.normal(0, 50, n)
    return daily + weekly + trend + noise

from sklearn.preprocessing import StandardScaler

raw = synthetic_traffic()
scaler = StandardScaler()
series = scaler.fit_transform(raw.reshape(-1, 1)).flatten()

X, y = make_windows(series, history=168, horizon=24)
split = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X[:split], y[:split]),
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X[split:], y[split:]),
                        batch_size=64)

model = TCN(input_size=1, output_size=1,
            channels=[64] * 7, kernel_size=3, dropout=0.2)
print("感受野：", model.receptive_field)  # 509

train_tcn(model, train_loader, val_loader, num_epochs=80)
```

`output_size=1` 表示模型输出一个单通道序列。在直接多步预测中，通常希望网络一次性输出整个预测区间。有两种实现方式：

1. **序列到序列头**：保持 `output_size=1`，取输出序列的最后 $H$ 步。简单直接，但预测区间和历史长度会绑定在一起。
2. **Flatten + 线性头**：把最后的 `nn.Conv1d(C, 1, 1)` 替换为 `nn.Linear(C * history, horizon)`，让模型直接输出 $H$ 维向量。这种方式更灵活。

两种方法都可行。第一种参数更少，我在这里就用了这种方法。

**预期效果**：在合成数据上，模型训练 30 轮后，能将日峰值预测误差控制在 ~10% MAPE 以内。在真实的 Caltrans 风格数据上，不调参的情况下， MAPE 通常在 8%-15% 之间，明显优于朴素季节性基线模型。
## 案例 2：多变量传感器预测

**任务**：四个 IoT 传感器（温度、湿度、气压、光照）相互关联，每 5 分钟采样一次。根据过去 6 小时（72 步）的数据，预测未来 1 小时（12 步）的温度。

```python
def synthetic_sensors(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    temp = 20 + 5 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 0.5, n)
    hum = 60 - 0.8 * (temp - 20) + rng.normal(0, 2, n)
    pres = 1013 + 2 * np.sin(2 * np.pi * t / 1000) + rng.normal(0, 0.3, n)
    light = 100 * np.maximum(0, np.sin(2 * np.pi * t / 288)) + rng.normal(0, 5, n)
    return np.column_stack([temp, hum, pres, light])

sensors = synthetic_sensors()
scaler = StandardScaler()
sensors_s = scaler.fit_transform(sensors)

def make_multivariate_windows(arr, target_idx, history, horizon):
    n = len(arr) - history - horizon + 1
    X = np.stack([arr[i : i + history].T for i in range(n)])  # (N, F, T)
    y = np.stack([arr[i + history : i + history + horizon, target_idx]
                  for i in range(n)])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)

Xm, ym = make_multivariate_windows(sensors_s, target_idx=0,
                                   history=72, horizon=12)
# ... 构造数据加载器 ...

model = TCN(input_size=4, output_size=1,
            channels=[64, 64, 128, 128, 128], kernel_size=3, dropout=0.2)
print("感受野：", model.receptive_field)  # 253
```

**为什么多变量输入在 TCN 中直接生效**： TCN 的第一层卷积会在每个时间步上处理所有 4 个输入通道，跨特征交互天然融入其中，完全不需要额外设计融合模块。

**快速验证特征重要性**：逐一将某个通道置零，观察验证集 MAE 的变化：

```python
def feature_ablation(model, X_val, y_val, names):
    model.eval()
    base = ((model(X_val) - y_val) ** 2).mean().item()
    out = {}
    for i, name in enumerate(names):
        Xz = X_val.clone()
        Xz[:, i, :] = 0.0
        out[name] = ((model(Xz) - y_val) ** 2).mean().item() - base
    return out

print(feature_ablation(model, Xm[:200], ym[:200],
                       ["temp", "hum", "pres", "light"]))
```

在上述合成数据中，湿度的影响最显著（因为构造数据时它与温度强相关）。在真实传感器数据中，结果会更复杂，但仍然可以作为一项有效的 sanity check。
## 超参数与设计速查表

根据问题特点，优先选择的默认值如下：

| 超参数 | 默认值 | 何时调整 |
|---|---|---|
| 核大小 $k$ | 3 | 几乎不用改。需要扩大感受野时用 dilation。 |
| dilation 序列 | $2^i$（第 $i$ 层） | 几乎不用改。 2 的幂次是最佳选择。 |
| 通道数 | 固定宽度 32-128 | 欠拟合就增加，过拟合就减少。 |
| 层数 $L$ | 满足 $\text{RF}(L) \geq$ 上下文的最小层数 | 按公式计算，别堆叠过多。 |
| dropout | 0.2 | 小数据集用 0.3-0.5，超大数据集用 0.1。 |
| 归一化 | weight norm | batch 很小时用 layer norm，避免使用 batch norm。 |
| 优化器 | Adam， lr 1e-3 | 超大数据集上， SGD + momentum 偶尔表现更好。 |
| 学习率调度 | `ReduceLROnPlateau`， factor 0.5 | 训练轮数很多时改用 cosine annealing。 |
| 梯度裁剪 | 1.0 | 别去掉，这是低成本的保障措施。 |

---
## 常见问题与调试方法

- **输出整体右移**：忘了去掉右侧的 padding。检查代码里有没有 `y[:, :, : -self.padding]`，确保因果卷积写对了。
- **训练 loss 下降但验证 loss 不降**：感受野比数据的主要周期还小。重新跑一下 `required_layers`，调整到合适的预测范围。
- **loss 很快卡住不动**：通道数可能太窄，或者学习率太低。试试把通道数翻倍，或者把学习率调成 `lr=3e-3`。
- **验证 loss 突然爆炸**：基本可以肯定是 batch norm 加上小 batch size 的问题，或者是小数据集没加 dropout。换成 weight norm，并加上 0.3 的 dropout。
- **预测忽略近期值**：网络完全依赖长程结构了。去掉几层（缩小感受野），或者从输入到输出直接加一条 1 步 skip 连接。
## 什么时候不该用 TCN

TCN 确实有它的局限性，下面这些情况就别用了：

- **序列长度变化大，又不能用 padding**。直接上 LSTM 或 GRU 更合适。
- **需要真正的在线流式推理**，比如每次只来一个新样本，要求微秒级响应。因果 CNN 虽然也能做流式处理，但实现起来比跑一个 LSTM cell 复杂多了。
- **目标序列远比窗口长**，比如 100k 步的生理信号，但需要 50k 的上下文。这种场景更适合 N-BEATS-X 或 Informer 这类层级化模型。
- **需要类似 attention 的可解释性**。 TCN 的滤波器可以可视化，但解释性是局部的； attention map 就直观得多。

除此之外，几乎所有预测任务都可以先试试 TCN。它没什么花哨，但速度快、性能稳，是个靠谱的起点。
## Q&A

### TCN 和 WaveNet 有什么不同？

WaveNet （2016）本质上是一个 TCN，只不过它用的是门控激活函数 $\tanh(W_f x) \odot \sigma(W_g x)$，而不是 ReLU。此外， WaveNet 还为音频生成设计了更复杂的条件机制。 TCN 则简化为 ReLU 加残差结构，专注于通用序列建模。

### BatchNorm 和 WeightNorm 哪个更好？

推荐用 WeightNorm。 BatchNorm 的 running statistics 在长序列上容易漂移，噪声也大； WeightNorm 完全避免了这个问题。 LayerNorm 也可以考虑，但处理 1D 卷积数据布局时需要额外加一个 transpose 操作。

### 需要像 Transformer 那样加位置编码吗？

不需要。卷积本身具有平移等变性，位置信息已经隐含在感受野的结构中。

### 多步预测该选直接法还是递归法？

直接法（一次性输出整个 horizon）更准确，因为误差不会累积，但参数量更大，且训练时 horizon 是固定的。递归法（一步步预测，再把结果喂回去）更灵活，但误差会逐步累积。默认建议用直接法。

### 如果想要分位数预测怎么办？

把 L2 损失换成多个分位的 pinball loss， head 输出每个分位对应一个通道。 TCN 的主体结构保持不变即可。
## 小结

TCN 把序列建模的核心归结为一个简单的思路：用因果膨胀卷积加残差连接，就能实现长记忆、并行训练和稳定梯度，完全不需要递归结构。数学公式只有一个 $\text{RF}(L) = 1 + (k-1)(2^L - 1)$ 需要记住， PyTorch 实现只要 60 行代码，在大多数固定长度的基准测试中，性能至少能和调参后的 LSTM 打平。

把它当作预测任务的第一个基线模型。如果它被更复杂的模型打败，说明那些复杂设计确实有价值；如果它赢了——这种情况很常见——你就得到了一个又快又简单的模型。

下一章我们从卷积转向 **N-BEATS**：这个模型彻底抛弃卷积和递归，只用全连接块加基函数展开，不仅赢得了 M4 预测竞赛冠军，还保持了可解释性。
## 参考资料

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* [arXiv:1803.01271](https://arxiv.org/abs/1803.01271).
- van den Oord, A. et al. (2016). *WaveNet: A Generative Model for Raw Audio.* [arXiv:1609.03499](https://arxiv.org/abs/1609.03499).
- Lea, C. et al. (2017). *Temporal Convolutional Networks for Action Segmentation and Detection.* CVPR.
- Salimans, T., & Kingma, D. P. (2016). *Weight Normalization.* NeurIPS.

---

> 
>
> 本文是时间序列模型系列的**第 6 篇**，共 8 篇。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - [第 2 篇：LSTM —— 门控机制与长期依赖](/zh/time-series/02-lstm/)
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru/)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构/)
> - **第 6 篇：时序卷积网络 TCN**（当前）
> - [第 7 篇：N-BEATS —— 可解释的深度架构](/zh/time-series/07-n-beats深度架构)
> - [第 8 篇：Informer —— 高效长序列预测](/zh/time-series/08-informer长序列预测/)
