---
title: "时间序列模型（七）：N-BEATS -- 可解释的深度架构"
date: 2024-11-30 09:00:00
tags:
  - 时间序列
  - 深度学习
  - N-BEATS
categories: 时间序列
series: time-series
lang: zh
mathjax: true
description: "N-BEATS 把深度学习的表达力和经典分解的可解释性合二为一：基函数展开、双重残差堆叠、M4 竞赛分析，以及完整的 PyTorch 代码。"
disableNunjucks: true
series_order: 7
translationKey: "time-series-7"
---
![章节概念图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/illustration_1.jpg)

2018 年 M4 预测竞赛放出了 100,000 条时间序列，覆盖六种频率，作为统一的基准测试。排行榜上，名列前茅的几乎全是基于几十年统计预测经验手工调优的集成模型。但最后夺冠的是一个**纯神经网络**，完全不需要统计预处理、特征工程，也没有递归结构。这个模型就是 Oreshkin 等人提出的 **N-BEATS**，由一堆全连接块组成，带有两条残差路径。它的可解释版本还把预测结果拆分为多项式趋势和 Fourier 季节性，正好满足了经典统计学家一直追求的目标——可读的分解。

这一章我会详细分析，为什么这种极简架构能同时超越 LSTMs 和 ARIMA 风格的集成模型。我还会教你如何实现和调优，用在自己的数据上。
## 你将学到的内容

- 双重残差堆叠如何把普通的 MLP 变成一个层级分解器。
- 基函数展开：用多项式基处理趋势，用 Fourier 基处理季节性，用学习到的基应对“通用”版本。
- N-BEATS 为什么能做到既最精准又最容易解释。
- M4 比赛结果表：N-BEATS 到底赢了哪些模型，赢了多少。
- 完整的 PyTorch 实现代码，以及可以复用的零售销售和能源需求案例。

**前置要求**：熟悉前馈网络和 PyTorch。了解经典分解（趋势/季节性/残差）会有帮助，但不是必须的。
## 为什么全连接堆栈就够用了

大多数时间序列的深度学习模型都会引入结构先验：卷积假设平移等变性，RNN 假设有序的隐藏状态，attention 假设两两相关性。N-BEATS 的思路完全不同：它直接把整个输入窗口丢给 MLP，让网络自己学习最有用的分解方式。关键不在于层的类型，而在于信息流动的路径。

具体来说，N-BEATS 做了三个大胆的设计选择：

1. **堆叠相同结构的模块**。每个模块同时预测 *backcast*（输入窗口的重构）和 *forecast*（未来的预测）。
2. **双残差流**。每个模块会从当前输入残差中减去自己的 backcast，同时把自己的 forecast 加到预测累加器里。下一个模块只会看到上一个模块无法解释的部分。
3. **基函数输出头**。模块不直接输出预测值，而是生成一个小的系数向量 $\theta$，再将其与一个固定的（可解释的）或学习得到的（通用的）基矩阵相乘。

这三点结合起来，就足以让 N-BEATS 登顶 M4 榜首。
## 架构：双重残差流

想象一下，网络中有两条从上到下并行的管道。左边是**残差流**，它从输入窗口 $x \in \mathbb{R}^{H}$ 开始，每经过一个模块，就会减去该模块的 backcast，逐渐变小。右边是**预测累加器**，它从零开始，每经过一个模块，就会加上该模块的 forecast，逐渐变大。

![N-BEATS 的双重残差堆叠](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig1_stacked_residual_blocks.png)

数学上看，对于第 $b = 1, \ldots, B$ 个模块：

$$r^{(b)} = r^{(b-1)} - \hat{x}^{(b)}, \qquad \hat{y} = \sum_{b=1}^{B} \hat{y}^{(b)},$$

其中 $r^{(0)} = x$。每个模块看到的残差越来越小，因此会专注于处理剩余未被解释的频率或形状。粗略模式（比如整体趋势、主要季节周期）由前面的模块捕捉；细微修正则交给后面的模块。

这和梯度提升的思想类似，只不过实现方式是一个端到端可微的网络。顺序同样重要：第一个模块任务最简单（完整信号都在），最后一个模块最难（只剩下噪声和微妙结构）。
## 块的内部

每个 N-BEATS 块的结构都相同。假设残差输入是 $r \in \mathbb{R}^{H}$：

1. **特征提取**——四层全连接 ReLU 层，宽度在 256 到 512 之间：
   $$   h_1 = \mathrm{ReLU}(W_1 r + b_1), \quad \ldots, \quad h_4 = \mathrm{ReLU}(W_4 h_3 + b_4).
   $$
2. **系数投影**——两个线性头分别生成 backcast 和 forecast 的系数：
   $$
   \theta^{b} = W_b h_4, \qquad \theta^{f} = W_f h_4.
   $$
3. **基函数映射**——固定或可学习的矩阵 $V$ 将系数映射到时间域输出：
   $$
   \hat{x} = V_b \, \theta^{b}, \qquad \hat{y} = V_f \, \theta^{f}.
   $$

两种变体的区别仅在于 $V$ 的定义。

### 可解释：趋势 + 季节性基

趋势块使用低次多项式基。设次数为 $p$，时间索引 $\tau / H \in [0, 1]$：
$$
V_{\text{trend}} = \begin{pmatrix} 1 & \tau & \tau^{2} & \cdots & \tau^{p} \end{pmatrix}, \qquad
\hat{y}_{\text{trend}} = \sum_{i=0}^{p} \theta_i \, \tau^{i}.
$$
通常选择 $p = 2$ 或 $3$。这样既能拟合“先平稳上升后加速”的趋势，又不会过拟合抖动细节。

季节性块使用 Fourier 基：
$$
V_{\text{seas}} = \begin{pmatrix} \sin(2\pi \cdot 1 \cdot \tau / T) & \cos(2\pi \cdot 1 \cdot \tau / T) & \cdots & \sin(2\pi K \tau / T) & \cos(2\pi K \tau / T) \end{pmatrix}.
$$

$K = 1, 2, 3$ 表示谐波阶数，$T$ 是数据已知周期（月级 12，时级 24）。这种设计可以捕捉任意形状的周期信号。

可解释架构先堆叠一个趋势栈（几个趋势块），再接一个季节性栈（几个季节性块）。训练完成后，我可以画出每个栈的贡献，向业务方解释：“这部分来自底层趋势，那部分来自周周期。”

![可解释 N-BEATS 给出的趋势 + 季节性分解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig2_basis_decomposition.png)

### Generic：学习到的基

Generic 版本让 $V_b$ 和 $V_f$ 成为可学习矩阵。块不再局限于趋势或季节性的语义，而是根据梯度信号学出有用的基。精度会稍微提升，但代价是失去了直观的分解图。

![可解释 vs generic 的栈结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig3_interpretable_vs_generic.png)

论文里提到一条实用经验：M4 数据集上**最佳**结果来自可解释模型和 generic 模型的集成，且使用了不同的回看长度。这个话题我会在“集成策略”一节详细展开。
## 为什么基函数输出头很重要

模型块其实可以直接预测结果：$\hat{y} = W_y h_4$。但为什么要绕一圈，通过 $\theta$ 向量和基矩阵来实现？

原因有三点：

- **归纳偏置**。强制模型用少量平滑基的线性组合表达预测值，能有效避免过拟合噪声。比如，720 步输出配上 3 次多项式，趋势分量只有 4 个自由度，根本不可能出现振荡。这种正则化是可解释版本泛化的关键。
- **天然可解释**。趋势栈的 $\theta_0, \theta_1, \theta_2, \theta_3$ 分别对应基线、斜率、曲率和加速度变化率（jerk）。季节性栈的系数则是特定谐波的振幅。这些参数可以画图分析，直观易懂。
- **参数高效**。直接从 512 维隐藏状态映射到 720 步预测，需要一个 $512 \times 720 = 369K$ 参数的线性层。而基函数头只需要两个小线性层（$512 \to p$，再通过固定基 $p \to 720$），每个输出头的参数通常不到 10K。

这个思路现在已经被很多新模型采用。PatchTST、N-HiTS 和 TSMixer 都用了某种分解头的设计。这套方法最早由 N-BEATS 推广开来。
## PyTorch 实现

下面是一个简洁且完整的实现，整个模型大约 120 行代码。

```python
import torch
import torch.nn as nn
import numpy as np

class TrendBasis(nn.Module):
    """多项式基：V[i, t] = (t / horizon) ** i"""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.degree = degree
        tb = torch.stack([torch.linspace(0, 1, backcast_size) ** i
                          for i in range(degree + 1)], dim=0)
        tf = torch.stack([torch.linspace(0, 1, forecast_size) ** i
                          for i in range(degree + 1)], dim=0)
        self.register_buffer("V_b", tb)  # (degree+1, H)
        self.register_buffer("V_f", tf)  # (degree+1, F)

    @property
    def theta_size(self) -> int:
        return self.degree + 1

    def forward(self, theta_b, theta_f):
        return theta_b @ self.V_b, theta_f @ self.V_f

class SeasonalityBasis(nn.Module):
    """Fourier 基：取前 floor((H 或 F) / 2) 阶谐波"""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        K = forecast_size // 2 + 1
        tb = torch.linspace(0, 1, backcast_size)
        tf = torch.linspace(0, 1, forecast_size)
        ks = torch.arange(K).unsqueeze(1).float()
        Vb = torch.cat([torch.cos(2 * np.pi * ks * tb),
                        torch.sin(2 * np.pi * ks * tb)], dim=0)
        Vf = torch.cat([torch.cos(2 * np.pi * ks * tf),
                        torch.sin(2 * np.pi * ks * tf)], dim=0)
        self.register_buffer("V_b", Vb)  # (2K, H)
        self.register_buffer("V_f", Vf)  # (2K, F)

    @property
    def theta_size(self) -> int:
        return self.V_b.shape[0]

    def forward(self, theta_b, theta_f):
        return theta_b @ self.V_b, theta_f @ self.V_f

class GenericBasis(nn.Module):
    """学习基：backcast/forecast 关于 theta 是线性的"""

    def __init__(self, theta_size: int, backcast_size: int,
                 forecast_size: int):
        super().__init__()
        self._theta_size = theta_size
        self.linear_b = nn.Linear(theta_size, backcast_size, bias=False)
        self.linear_f = nn.Linear(theta_size, forecast_size, bias=False)

    @property
    def theta_size(self) -> int:
        return self._theta_size

    def forward(self, theta_b, theta_f):
        return self.linear_b(theta_b), self.linear_f(theta_f)

class NBeatsBlock(nn.Module):
    def __init__(self, basis: nn.Module, backcast_size: int,
                 hidden: int = 256, layers: int = 4):
        super().__init__()
        self.basis = basis
        units = [backcast_size] + [hidden] * layers
        fcs = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            fcs.append(nn.Linear(in_dim, out_dim))
            fcs.append(nn.ReLU())
        self.fc = nn.Sequential(*fcs)
        self.head_b = nn.Linear(hidden, basis.theta_size)
        self.head_f = nn.Linear(hidden, basis.theta_size)

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.head_b(h)
        theta_f = self.head_f(h)
        return self.basis(theta_b, theta_f)  # (backcast, forecast)

class NBeats(nn.Module):
    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x
        forecast = 0.0
        for blk in self.blocks:
            backcast, fc = blk(residual)
            residual = residual - backcast
            forecast = forecast + fc
        return forecast

def make_interpretable(history: int, horizon: int,
                       trend_blocks: int = 3, seasonal_blocks: int = 3,
                       trend_degree: int = 3,
                       hidden: int = 256, layers: int = 4) -> NBeats:
    blocks = []
    trend_basis = TrendBasis(trend_degree, history, horizon)
    for _ in range(trend_blocks):
        blocks.append(NBeatsBlock(trend_basis, history, hidden, layers))
    seas_basis = SeasonalityBasis(history, horizon)
    for _ in range(seasonal_blocks):
        blocks.append(NBeatsBlock(seas_basis, history, hidden, layers))
    return NBeats(blocks)

def make_generic(history: int, horizon: int,
                 num_blocks: int = 30, theta_size: int = 32,
                 hidden: int = 512, layers: int = 4) -> NBeats:
    blocks = []
    for _ in range(num_blocks):
        basis = GenericBasis(theta_size, history, horizon)
        blocks.append(NBeatsBlock(basis, history, hidden, layers))
    return NBeats(blocks)
```

一个小细节但很重要：在可解释版本中，`TrendBasis` 和 `SeasonalityBasis` 实例会在同一个栈内的所有块之间共享。每个块有自己的 MLP 和系数头，但它们都乘以同一个固定的基矩阵。这样既保留了归纳偏置，又节省了一些参数。
## 训练方法

Oreshkin 等人提出的训练方法非常简洁：

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_nbeats(model, train_loader, val_loader, epochs=100,
                 lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.L1Loss()  # MAE；论文根据数据集用 sMAPE/MASE/MAPE，
                        # L1 是个稳妥的默认选择
    best = float("inf")
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        sched.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "nbeats.pt")
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}: train {train_loss:.4f} val {val_loss:.4f}")
```

几点实际建议：

- **损失函数**：M4 数据集用的是 sMAPE；在真实场景中，MAE（L1）比 MSE 更稳定，因为它不会对离群点过度惩罚。选一个和评估指标匹配的损失函数。
- **逐窗口标准化**：对每个输入窗口，先减去均值再除以标准差，然后送入网络，预测结果需要逆变换回来。这一步比损失函数的选择更重要。如果不做标准化，网络不仅要学习序列的形状，还得额外学习每条序列的尺度，浪费了模型容量。
- **早停策略**：N-BEATS 需要较长的训练时间，但在大多数数据集上，50 个 epoch 后验证损失基本就不再下降了。盯着验证损失，一旦不再改善就可以停止训练。
## N-BEATS 在 M4 竞赛中的表现

M4 竞赛涵盖了多种统计方法（ARIMA、ETS、Theta）、冠军 Smyl 的 ES-RNN 混合模型，以及基于特征元学习的第二名模型 FFORMA。N-BEATS 没有任何统计预处理，却在整体 sMAPE 和六个频率分组中的五个上击败了所有对手。

![N-BEATS 在 M4 竞赛中的表现](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig4_m4_results.png)

论文中的数据如下：

- **N-BEATS（可解释 + 通用集成）：sMAPE 11.135**
- N-BEATS（仅通用）：sMAPE 11.168
- Smyl 的 ES-RNN（M4 冠军）：sMAPE 11.374
- FFORMA：sMAPE 11.720
- 最佳经典方法（Theta）：sMAPE 12.309

从绝对值来看，sMAPE 的差距不大，但在年、季、月、周、日级别的数据上，N-BEATS 始终保持领先。唯一例外是小时级数据，Smyl 的 ES-RNN 以 0.4 sMAPE 的微弱优势胜出。

更深层次的启示是：一个表达能力足够强且具备合适归纳偏置的深度模型，可以重新学到统计学家几十年来手工设计的成果。
## 集成：隐藏在配方中的另一半

仔细读 M4 论文，你会发现一个脚注：表头的 N-BEATS 数字其实是 **180 个模型实例预测结果的中位数**。每个模型实例有三个地方不同：回看长度（2H、3H、...、7H）、训练损失函数（sMAPE、MASE、MAPE）以及随机种子。单个模型的表现明显不如集成模型。

![为什么 N-BEATS 使用集成：原理与实证 sMAPE 提升](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig5_ensemble_strategy.png)

右图的实证曲线显示收益递减：大部分提升来自前 30 个集成成员。实际生产中，几乎不需要训练 180 个模型——用 10 到 30 个模型，混合不同的回看长度和随机种子，就能拿到绝大部分性能提升。

一个简单的集成工具类：

```python
class EnsembleNBeats:
    def __init__(self, models: list[NBeats]):
        self.models = models

    def predict(self, x: torch.Tensor, aggregator="median") -> torch.Tensor:
        outs = torch.stack([m(x) for m in self.models], dim=0)
        return outs.median(dim=0).values if aggregator == "median" else outs.mean(0)
```

对于 sMAPE 这类损失函数，中位数比均值更适合做聚合方法。它能有效避免某个模型在某个窗口上表现异常的情况。
## 案例 1：月度零售销量预测

**任务**  
给定过去 36 个月的销量数据，预测一个多门店零售连锁未来 12 个月的销售量。数据有明显的假日季节性（12 月高峰），整体呈上升趋势，偶尔会有促销活动。涉及约 200 条独立的产品-门店序列。

**架构选择**  
我选择了可解释性强的模型。业务团队需要能清楚地看到，预测中的某部分来自长期趋势，某部分来自每年 12 月的季节性提升，剩下的则是残差。`[3 个趋势块] + [3 个季节性块]` 的设计正好满足这个需求。

```python
model = make_interpretable(
    history=36, horizon=12,
    trend_blocks=3, seasonal_blocks=3,
    trend_degree=2,                     # 平滑的多年趋势
    hidden=256, layers=4,
)
```

训练完成后，我可以提取每个栈的贡献，看看每一部分学到了什么：

```python
def stack_contributions(model: NBeats, x: torch.Tensor) -> dict:
    """针对单个窗口，按栈拆分预测贡献。"""
    residual = x.clone()
    out = {}
    cur_stack = "trend"
    cumulative = torch.zeros(x.size(0), model.blocks[0].basis.V_f.shape[1])
    for i, blk in enumerate(model.blocks):
        backcast, fc = blk(residual)
        residual = residual - backcast
        cumulative = cumulative + fc
        next_stack = "seasonality" if isinstance(blk.basis, SeasonalityBasis) else "trend"
        if i + 1 == len(model.blocks) or not isinstance(model.blocks[i + 1].basis, type(blk.basis)):
            out[cur_stack] = cumulative.clone()
            cumulative = torch.zeros_like(cumulative)
            cur_stack = next_stack
    return out

contribs = stack_contributions(model, x_val[:1])
# contribs["trend"]        -> 12 个月的趋势分量
# contribs["seasonality"]  -> 12 个月的季节性分量
```

**典型结果**  
在一个有 5 年历史的真实零售数据集上，可解释版 N-BEATS 在 12 个月预测范围内的 MAPE 达到 7%-12%，与最好的梯度提升加特征工程方法相当，但完全不需要手动做特征工程。真正的优势在于可解释性：一个能让业务团队用已知促销活动覆盖 12 月季节性的模型，远比在回测中精度高出 0.5% 的黑盒模型实用得多。
## 案例 2：小时级电力需求预测

**任务**：根据过去 168 小时（一周）的电网需求，预测未来 24 小时的需求。数据有明显的日周期和周周期，受天气影响显著，热浪期间会出现需求高峰。

**架构选择**：我选用了 generic 架构。需求模式复杂，包含多分辨率的日周期、周周期以及天气驱动因素。业务团队最看重的是预测精度，因为每 1 MW 的误差都会增加运营储备成本。所以我用了一个更深的 generic 栈，并加大了隐状态的宽度。

```python
model = make_generic(
    history=168, horizon=24,
    num_blocks=30, theta_size=32,
    hidden=512, layers=4,
)
```

**逐窗口标准化是关键**。电力需求随季节波动很大，如果不做标准化，模型会浪费容量去学习“冬天比夏天需求高”这种常识。做了标准化后，模型只需关注需求的变化形状。

```python
def normalise_window(x: torch.Tensor) -> tuple:
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return (x - mu) / sd, mu, sd

# 前向计算：
x_norm, mu, sd = normalise_window(x)
y_norm = model(x_norm)
y_hat = y_norm * sd + mu  # 广播到 (B, H)
```

**典型结果**：在公开的 ETT（Electricity Transformer Temperature）数据集上，针对 24 小时预测任务，使用 10 个 generic N-BEATS 模型组成的集成方法，MSE 能达到 0.31 左右。这比 LSTM（约 0.42）表现更好，同时实现成本远低于 Informer，但精度相当。
## 超参速查表

新数据集的实用起点：

| 超参 | 默认（可解释） | 默认（通用） | 调参建议 |
|---|---|---|---|
| 回看 `history` | 预测范围 × 4 到 7 | 预测范围 × 4 到 7 | 范围越大，覆盖的季节越多；按 2 倍步长调整。 |
| 隐藏层宽度 | 256 | 512 | 如果验证损失停滞在高位，就增大宽度。 |
| 每块 MLP 层数 | 4 | 4 | 一般不需要改动。 |
| 趋势阶数 | 2 或 3 | -- | 如果趋势有曲率，用 3；否则用 2。 |
| 趋势块数 | 3 | -- | 如果趋势 RMSE 是主要误差来源，就增加块数。 |
| 季节性块数 | 3 | -- | 多分辨率季节性时，增加块数。 |
| Generic 块数 | -- | 20 到 30 | 增加块数通常还有帮助，但效果会逐渐减弱。 |
| Generic $\theta$ 尺寸 | -- | 32 | 16 容易欠拟合，64 以上很少带来额外收益。 |
| 损失函数 | MAE / sMAPE | MAE / sMAPE | 和评估指标保持一致。 |
| 优化器 | Adam，lr 1e-3 | Adam，lr 1e-3 | 使用 cosine annealing，不需要 warmup。 |
| 集成大小 | 10 到 30 | 10 到 30 | 使用中位数聚合。 |

---
## 什么时候不该用 N-BEATS

- **预测步数特别长（>1000 步）**。输出投影 $\theta \to \hat{y}$ 会卡在参数瓶颈上。换成 **N-HiTS**（N-BEATS 的改进版）或者 PatchTST，它们用多速率降采样来扩展能力。
- **多变量时间序列且特征交互强**。N-BEATS 本质上是单变量模型。要么每个变量单独训练一个模型，要么试试 TFT、Informer 或 DeepAR。
- **极短序列（约 50 个观测点）**。MLP 主干有几万个参数，在这种小数据上跑不过 Theta 或 ETS 模型。
- **在线流式预测**。N-BEATS 处理的是固定窗口，而不是流式的隐藏状态。LSTM 或 TCN 更适合这种场景。
- **预测结果必须满足硬约束**（非负值、整数值）。N-BEATS 输出的是无约束的实数；你可以后处理，或者用带输出裁剪的分位损失，但直接用支持原生分布输出的模型（比如 DeepAR）可能更简单。
## Q&A

### 为什么选多项式和 Fourier？

多项式在紧区间上能逼近任意连续函数（Stone-Weierstrass 定理），Fourier 则擅长处理周期函数。两者结合正好对应“平滑趋势 + 周期成分”的模式，这正是大多数实际时间序列的特点。

### 3 次多项式的趋势块能捕捉高阶行为吗？

局部可以，全局不行。每个趋势块处理的是前面块留下的**残差**，所以堆叠三个趋势块相当于组合了三个三次函数。对于绝大多数预测范围内的趋势，这种能力已经够用。

### 为什么用 Adam 而不是 SGD？

实验表明，N-BEATS 用 Adam 收敛速度快得多。论文提到，SGD 需要精细调整学习率才能达到 Adam 默认参数的效果。

### N-BEATS 需要位置编码吗？

不需要。MLP 把整个窗口当作固定长度的向量处理，输入的位置信息隐含在列索引中，输出端的基矩阵则负责编码每一步的时间索引。

### 如何扩展 N-BEATS 来支持外生变量（比如天气、节假日）？

官方扩展方法叫 **N-BEATS-X**：把外生协变量拼接到输入窗口，并为每个块添加一个辅助输入头。实际生产中，直接将外生时间序列沿输入向量拼接，再稍微加宽第一层 MLP，通常就能满足需求。

### N-HiTS 是什么？我是不是应该直接用它？

N-HiTS（2023）是原作者团队的新作。它在输出端引入了多速率下采样和插值，能够处理更长的预测范围（720+ 步），并且运行速度更快。不过，对于短到中等预测范围（<100 步），原版 N-BEATS 依然表现不错，而且更简单。

**为什么要集成模型？我的单模型效果很好啊。**

你可能在这个数据集上运气不错。论文分析了 10 万条 M4 序列，发现单模型的方差很大——即使是 5 个模型的集成，也能降低 1-2 个 sMAPE 点。如果你只需要单点估计且数据量小，可以不用集成；但如果要报告结果，还是建议集成。
## 小结

N-BEATS 是个“架构平淡无奇”的模型，但它靠把合适的模块按正确顺序堆起来取胜。双重残差流让它有了类似 boosting 的行为；基函数输出头不仅提供了强大的归纳偏置，还在可解释版本中实现了免费的分解能力。M4 排行榜的结果证明，这种组合既能打败经典方法，也能胜过递归深度模型。

对于大多数采样均匀、趋势和季节性结构清晰的单变量预测问题，N-BEATS 是最强的开箱即用基线之一。建议从可解释版本入手，方便争取利益相关方的认可。如果需要榨干最后一丝精度，可以切换到通用版本，或者将两者集成。记得在不同回看长度和随机种子上做集成。

下一章我们用 **Informer** 收尾，它解决的是另一个问题：如何让 Transformer 在预测上千步 horizon 时，避免 $\mathcal{O}(L^2)$ 的注意力计算成本拖垮性能。
## 参考资料

- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 Competition: 100,000 Time Series and 61 Forecasting Methods.* International Journal of Forecasting, 36(1).
- Smyl, S. (2020). *A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting.* International Journal of Forecasting, 36(1).
- Challu, C. et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI.
- Olivares, K. et al. (2023). *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with N-BEATS-X.* International Journal of Forecasting.

---

> 
>
> 本文是时间序列模型系列的**第 7 篇**，共 8 篇。
>
> - [第 1 篇：传统统计模型](/zh/time-series/01-传统模型/)
> - [第 2 篇：LSTM —— 门控机制与长期依赖](/zh/time-series/02-lstm/)
> - [第 3 篇：GRU —— 轻量门控与效率权衡](/zh/time-series/03-gru/)
> - [第 4 篇：Attention 机制 —— 直接的长程依赖](/zh/time-series/04-attention机制)
> - [第 5 篇：时间序列的 Transformer 架构](/zh/time-series/05-transformer架构/)
> - [第 6 篇：时序卷积网络 TCN](/zh/time-series/06-时序卷积网络tcn/)
> - **第 7 篇：N-BEATS —— 可解释的深度架构**（当前）
> - [第 8 篇：Informer —— 高效长序列预测](/zh/time-series/08-informer长序列预测/)
