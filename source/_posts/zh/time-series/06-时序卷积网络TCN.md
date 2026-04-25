---
title: "时间序列模型（六）：时序卷积网络 (TCN)"
date: 2024-12-10 09:00:00
tags:
  - 时间序列
  - 深度学习
  - TCN
categories: 时间序列
series:
  name: "时间序列模型"
  part: 6
  total: 8
lang: zh-CN
mathjax: true
description: "TCN 用因果膨胀卷积换取并行训练和指数级感受野。完整 PyTorch 实现，附交通流和多变量传感器两个实战案例。"
disableNunjucks: true
series_order: 6
---


整个 2010 年代，"用深度学习做时间序列"基本上等价于"上 LSTM"。这件事在 2018 年被 Bai、Kolter、Koltun 的 *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* 改变。结论简单到令人有些不爽：把若干个 1D 卷积叠起来，让它们都是因果的（不偷看���来���，让卷积核的步距按指数膨胀（dilation），整个块外面套一层残差连接，然后训练。在一个又一个任务上，这个**时序卷积网络**（TCN）的表现都和 LSTM/GRU 持平甚至更好——而且训练快好几倍，因为它的前向传播在每一个时间步上都是并行的。

本章把这个配方拆开来讲：先推那条让 dilation 值得用的感受野公式，再一步步看清残差块的内部，最后用一份你可以直接拿走的 PyTorch 实现做两个实战（交通流预测、多变量 IoT 传感器预测）。

## 这一篇你会学到

- 为什么"诚实的预测"必须用因果一维卷积，左侧 padding 是怎么做到这件事的。
- 膨胀卷积如何让感受野按 $\mathcal{O}(2^L)$ 而不是 $\mathcal{O}(L)$ 增长。
- TCN 残差块的精确构造（两个膨胀因果卷积 + 权重归一化 + dropout + 1x1 残差投影）。
- TCN vs LSTM/GRU/Transformer 在训练速度、显存、精度上的正面对比。
- 两个案例：每小时交通流预测和多变量 IoT 传感器预测。

**前置知识**：第 2 部分（LSTM）和第 5 部分（Transformer）。会用 PyTorch 的 `nn.Conv1d` 并理解基本复杂度记号。

---

## 当年用 LSTM 的疼点

在 TCN 之前，时间序列的深度学习剧本是这样的：叠两层 LSTM，需要的话再加点 attention，慢慢训。能用，但每一环都在拖你后腿：

- **前向传播是顺序的**。算 $h_t$ 要用到 $h_{t-1}$，GPU 要在那干等。哪怕硬件并行能力无限，序列长度翻倍，墙钟时间也是翻倍。
- **梯度沿时间方向消失或爆炸**。反向传播要走 $L$ 个乘法步。LSTM 有门可以缓解，但超过 ~200 步就开始变脆。梯度裁剪、layer norm、各种小心翼翼的初始化只是为了让训练别炸。
- **隐状态是黑盒**。"模型为什么这么预测？"通常没有好答案，因为隐状态把所有信息都搅在一起。
- **超参税**。层数、隐藏维度、门变种、dropout 类型、recurrent dropout 的位置之间互相影响。组合错了往往要白训一天。

TCN 的卖点：用可以并行运行的卷积代替递归，用显式的感受野代替隐状态那种暧昧的"记忆"，用残差连接保住梯度。表达能力相当，可动的零件少得多。

---

## 一维卷积：但要是"因果的"

标准的一维卷积把长度为 $k$ 的核 $f$ 滑过输入序列 $x$：

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i+\lfloor k/2 \rfloor}.
$$

这种"居中"形式让 $t$ 时刻的输出同时看到了过去和未来。对于预测来说这是**信息泄漏**——你不能用明天的交通流去预测明天的交通流。

**因果**卷积把核往左推一格，让 $t$ 时刻的输出只用到 $1, \ldots, t$ 的输入：

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i}.
$$

实现上，给输入左侧 pad $k - 1$ 个零，跑一个普通的 `nn.Conv1d`，再把右侧多出来的 padding 切掉，输出长度就和输入一样了。

![因果 vs 非因果一维卷积，t = 6](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-%E6%97%B6%E5%BA%8F%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9CTCN/fig2_causal_convolution.png)

图里两侧的绿色输出 $y_6$ 是同一个，但它读到的输入（橙色）不同：左边的非因果卷积读到了 $x_7$，落在了阴影标出的"未来"区——预测里绝对不能这么干。右边的因果版本永远只看左侧。

PyTorch 实现：

```python
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """左 padding + 右侧裁剪 的一维卷积。"""

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

两个细节：

1. padding 量是 $(k-1) \cdot d$，依赖于 dilation $d$（下文马上会讲）。
2. 卷积之后要切掉的是**右侧**。常见的 bug 是切了左侧，结果默默地把序列开头那几步毁掉了。

---

## 膨胀：用线性深度换指数感受野

核大小 $k = 3$ 的因果卷积叠 $L$ 层，感受野是 $1 + 2L$。线性增长。要看到 200 步以前，得叠 100 层，根本没法用。

**膨胀卷积**让卷积核的相邻 tap 之间留出 $d$ 的间隔：

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-d \cdot i}.
$$

如果每层都把 dilation 翻倍（$d_\ell = 2^{\ell-1}$），$L$ 层网络的感受野就变成

$$
\text{RF}(L) = 1 + (k - 1)\sum_{\ell=1}^{L} d_\ell = 1 + (k - 1)(2^L - 1).
$$

$k = 3$、$L = 8$ 时，感受野是 **511 步**——一周的小时级数据绰绰有余。参数量和 8 层普通卷积一样，覆盖范围却是指数级的。

![膨胀因果卷积的感受野](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-%E6%97%B6%E5%BA%8F%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9CTCN/fig1_dilated_convolution.png)

图里追踪了所有对最顶层那个绿色输出节点有贡献的输入。dilation 1、2、4、8 让这四层网络看上去像一棵稀疏的树——这种稀疏性正是它能覆盖那么远的原因。

一个常用的小工具，帮你定层数：

```python
import math


def required_layers(receptive_field: int, kernel_size: int = 3) -> int:
    """让 1 + (k-1)(2**L - 1) >= receptive_field 的最小 L。"""
    L = (receptive_field - 1) / (kernel_size - 1) + 1
    return max(1, math.ceil(math.log2(L)))
```

`required_layers(168, kernel_size=3)` 返回 `7`——对应小时级数据需要回看一周的场景。

---

## TCN 残差块

把膨胀因果卷积叠起来只是一半，另一半是包在外面的残差块。Bai 等人定下来的结构如下（和 WaveNet 几乎一样，差别只在激活的选择）：

![TCN 残差块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-%E6%97%B6%E5%BA%8F%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9CTCN/fig3_residual_block.png)

数学形式，给定输入 $x$：

$$
F(x) = \mathrm{Dropout}\!\big(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_2 \, \mathrm{Dropout}(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_1 \, x))))) \big),
$$

$$
o = \mathrm{ReLU}\!\big( F(x) + W_{\text{skip}} \, x \big).
$$

三个有意思的设计选择：

- **每块两个卷积**。一个卷积太弱；两个让块本身有学非平凡映射的能力，又不至于让总深度膨胀。
- **权重归一化**。Bai 等人发现 batch norm 在长序列上会拖后腿（统计量随位置漂）。weight norm 把每个滤波器的方向和模长解耦，不去碰激活值，训练更稳。
- **1x1 残差投影**。当输入输出通道数相同时残差就是恒等；不同时用一个 1x1 卷积投影一下，代价几乎为零。

PyTorch 实现：

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

这个块简单到大家经常直接 inline，但拆成模块的好处是感受野算起来一目了然，必要时也能把 weight norm 换成 layer norm。

---

## 把网络搭起来

完整的 TCN 就是按指数 dilation 叠的残差块，最后接一个 1x1 卷积投影到你想要的输出维度。

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

配置时几个建议：

- **通道数**。多数论文用恒定宽度（比如 `[64] * 8`）。如果输出维度比输入大很多，可以让靠近 head 的几层加宽。
- **核大小**。$k = 3$ 是默认。$k = 5/7$ 把参数翻倍但精度提升很小；想要更大感受野，几乎永远是用 dilation 更划算。
- **dropout**。0.2 是安全默认；小数据集推到 0.3-0.5。

---

## TCN vs RNN：架构层面的对比

下面这张信息流图比任何 benchmark 表都更直观地说明了速度差距：

![RNN 顺序依赖 vs TCN 并行前向](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-%E6%97%B6%E5%BA%8F%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9CTCN/fig4_tcn_vs_rnn.png)

RNN 那一列里每条红色箭头都是硬性的顺序依赖。GPU 可以并行算一个 cell 内部的工作，但没法跳过 $t$ 直接开始 $t+1$。前向传播的墙钟时间因此随序列长度线性增长，硬件并行能力再强都救不了。

TCN 那一列里每个输出节点只是固定输入集合的函数。同一个卷积核滑遍全长，整层就是一次大矩阵乘法，GPU 一次 kernel launch 就完事。

具体到单卡的每 epoch 墙钟：

![训练时间和 TCN 相对 LSTM 的加速比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/06-%E6%97%B6%E5%BA%8F%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9CTCN/fig5_parallel_training.png)

两个要点：

1. **训练时间的扩展性才是关键**。$L = 128$ 时四种架构差别不大；到了 $L = 1024$，TCN 比 LSTM 快 3-4 倍，比朴素 Transformer（attention 是 $L^2$）快约 6 倍。这正是大多数实际预测问题所处的区间。
2. **推理基本持平**。推理时 RNN 和 TCN 通常在 1.5 倍以内，差距只是训练时的事。如果你只关心单样本延迟，两者都可以。

什么时候用什么？给一个朴素的决策表：

| 场景 | 推荐 | 理由 |
|---|---|---|
| 固定窗口、有 GPU | TCN | 训练并行，感受野可预测 |
| 序列长度变化大、padding 不可接受 | LSTM/GRU | 原生支持，没有 padding 浪费 |
| 流式 / 在线推理，逐步进数据 | LSTM/GRU | 隐状态就是天然的状态 |
| 多变量、跨特征交互值得用 attention | Transformer / Informer | attention 显式建模两两关系 |
| 不确定 | 先试 TCN | 训练快、超参少 |

---

## PyTorch 训练循环

上面那两个类是核心，训练循环很普通：

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

两个点：梯度裁剪对 TCN **不是必须的**（残差 + weight norm 已经把梯度管得很好），但加上不亏。`ReduceLROnPlateau` 比固定 schedule 鲁棒，因为合适的学习率取决于数据集和感受野。

构造单变量窗口的小帮手：

```python
import numpy as np


def make_windows(series: np.ndarray, history: int, horizon: int):
    """单变量序列 -> (X, y) 张量，用于直接多步预测。"""
    n = len(series) - history - horizon + 1
    X = np.stack([series[i : i + history] for i in range(n)])
    y = np.stack([series[i + history : i + history + horizon] for i in range(n)])
    X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, history)
    y = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1, horizon)
    return X, y
```

---

## 案例 1：每小时交通流预测

**任务**：给定单个高速公路传感器过去一周（168 小时）的车流量，预测未来 24 小时。单变量、强日季节性 + 周季节性，偶尔有事件性尖峰。

**感受野预算**：希望输出端能看到至少一周的历史。$k = 3$、$L = 7$ 时 $\text{RF} = 1 + 2 \cdot 2 \cdot 127 = 509$，宽裕。

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

`output_size=1` 意味着模型输出一通道的序列。直接多步预测里你通常想让网络一次把整个 horizon 输出来。两种做法：

1. **序列到序列 head**：保留 `output_size=1`，取输出序列的最后 $H$ 步。简单，但 horizon 和历史长度耦合。
2. **flatten + 线性 head**：把最后的 `nn.Conv1d(C, 1, 1)` 换成 `nn.Linear(C * history, horizon)`，模型直接输出 $H$ 维向量。更灵活。

两种都能用，做法 1 参数更少，本案例就用这个。

**预期表现**：合成数据上模型训 30 epoch 之后能把日内峰值打到 ~10% MAPE 以内。真实 Caltrans 风格数据上没什么特别调参，MAPE 落在 8-15% 是合理区间，比朴素季节基线明显好。

---

## 案例 2：多变量传感器预测

**任务**：四个相关 IoT 传感器（温度、湿度、气压、光照），5 分钟采样。给定过去 6 小时（72 步）预测未来 1 小时（12 步）的温度。

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
# ... 构造 loader ...

model = TCN(input_size=4, output_size=1,
            channels=[64, 64, 128, 128, 128], kernel_size=3, dropout=0.2)
print("感受野：", model.receptive_field)  # 253
```

**为什么多变量"自然就好用"**：第一层卷积在每个时间步上就跨越所有 4 个输入通道做加权，跨特征交互天然内嵌。不需要额外的融合模块。

**简易特征重要性**：把每个通道置零，看验证 MAE 上升多少：

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

合成数据上湿度会主导（按构造它就和温度强相关）。真实传感器数据上图景更复杂，但作为 sanity check 还是有价值的。

---

## 超参与设计 cheat sheet

按"问题特征 -> 默认值"列：

| 超参 | 默认 | 何时改 |
|---|---|---|
| 核大小 $k$ | 3 | 几乎不动。要扩感受野就用 dilation。 |
| dilation 序列 | $2^i$ | 几乎不动。2 的幂就是答案。 |
| 通道数 | 恒定 32-128 | 欠拟合就加宽，过拟合就缩窄。 |
| 层数 $L$ | 满足 $\text{RF}(L) \geq$ 上下文的最小值 | 用公式算，别多叠。 |
| dropout | 0.2 | 小数据 0.3-0.5；大数据 0.1。 |
| 归一化 | weight norm | batch 极小时 layer norm；避免 batch norm。 |
| 优化器 | Adam，lr 1e-3 | 大数据上 SGD + momentum 偶尔更优。 |
| LR 调度 | `ReduceLROnPlateau`，factor 0.5 | 长训练换 cosine annealing。 |
| 梯度裁剪 | 1.0 | 留着，便宜的保险。 |

---

## 常见坑和排查

- **输出整体右移**：忘了切右侧 padding。检查 `y[:, :, : -self.padding]`。
- **训练降但验证不降**：感受野小于数据中的主导周期。重新算 `required_layers`。
- **loss 很早就走平**：通道太窄或学习率太低。试着把通道翻倍或把 lr 调到 `3e-3`。
- **验证 loss 突然爆炸**：八成是 batch norm + 小 batch，或者小数据没加 dropout。换 weight norm，dropout 调到 0.3。
- **预测无视近期值**：网络完全靠长程结构。少叠几层，或者从输入到输出加一条 1 步 skip。

---

## 什么时候**不**用 TCN

TCN 不是万能：

- **序列长度差异大且不能 padding** → 用 LSTM/GRU。
- **需要真正流式推理**（一次进一个新点，微秒级响应）→ 因果 CNN 也能做流式，但比直接跑一个 LSTM cell 麻烦。
- **目标比窗口长得多**（比如要 50k 上下文的 100k 步生理信号）→ 用 N-BEATS-X、Informer 这类层级化模型。
- **需要 attention 风格的可解释性** → TCN 的滤波器能可视化但意思偏局部，attention map 直观得多。

除上面这些情况，TCN 几乎永远是值得先试的"无聊但快"的方案。

---

## Q&A

### TCN 和 WaveNet 是什么关系？

WaveNet（2016）本质上是一个 TCN，激活换成了门控的 $\tanh(W_f x) \odot \sigma(W_g x)$，并且为音频生成加了更复杂的条件机制。TCN 把这些去掉，留下 ReLU + 残差，作为通用序列模型。

### 用 BatchNorm 还是 WeightNorm？

WeightNorm。BatchNorm 的 running statistics 在长序列上漂得厉害；WeightNorm 直接绕开这个问题。LayerNorm 也行，但 1D 卷积的张量布局要 transpose 一下。

### 需要像 Transformer 那样加位置编码吗？

不需要。卷积本身就是位置等变的，位置信息隐含在感受野结构里。

### 直接多步预测还是递归多步？

直接预测（一次输出整个 horizon）更准，因为误差不会累积，但参数多一点而且训练时就锁定了 horizon。递归（预测一步、喂回去、再预测）更灵活但会累积误差。默认选直接。

### 想要分位数预测怎么办？

把 L2 损失换成多个分位的 pinball loss，head 输出每个分位一通道。TCN 主体不变。

---

## 小结

TCN 把序列建模的核心收敛到一句话：因果膨胀卷积加残差连接就能给你长记忆、并行训练、稳定梯度，整套递归机制都不需要。数学只有 $\text{RF}(L) = 1 + (k-1)(2^L - 1)$ 这一条值得记，PyTorch 实现 60 行能写完，在大多数固定窗口的 benchmark 上对调过参的 LSTM 至少打平。

把它当你做预测任务的第一基线。被更复杂的模型干掉，说明那些复杂度真的赚到了；它赢——经常会赢——你就送出了一个又快又简单的模型。

下一章我们离开卷积，进入 **N-BEATS**：连卷积和递归都不要，只用全连接块加基函数展开，就拿了 M4 预测竞赛冠军，而且保持可解释。

---

## 参考资料

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
- van den Oord, A. et al. (2016). *WaveNet: A Generative Model for Raw Audio.* arXiv:1609.03499.
- Lea, C. et al. (2017). *Temporal Convolutional Networks for Action Segmentation and Detection.* CVPR.
- Salimans, T., & Kingma, D. P. (2016). *Weight Normalization.* NeurIPS.

---

**系列导航**

