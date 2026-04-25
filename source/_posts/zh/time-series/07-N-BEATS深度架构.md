---
title: "时间序列模型（七）：N-BEATS -- 可解释的深度架构"
date: 2024-12-15 09:00:00
tags:
  - 时间序列
  - 深度学习
  - N-BEATS
categories: 时间序列
series:
  name: "时间序列模型"
  part: 7
  total: 8
lang: zh-CN
mathjax: true
description: "N-BEATS 把深度学习的表达力和经典分解的可解释性合二为一：基函数展开、双重残差堆叠、M4 竞赛分析，以及完整的 PyTorch 代码。"
disableNunjucks: true
series_order: 7
---


2018 年 M4 预测竞赛把 100,000 条覆盖六种频率的序列摆到一个统一榜单上。占据榜单前几位的是几十年统计预测手艺打磨出来的精调集成。然后一个**纯神经网络**——没有统计预处理、没有特征工程、没有递归——直接拿了第一名。这个网络就是 Oreshkin 等人的 **N-BEATS**：若干全连接块沿着两条残差路径堆叠在一起。它的可解释版本还把预测拆成多项式趋势和 Fourier 季节性，连统计学家最在意的"可读分解"也免费送了。

本章把这套精简到极致的架构拆开来讲：它为什么能在同一个 benchmark 上既最准又最可解释，怎么实现，怎么调。

## 这一篇你会学到

- 双重残差堆叠如何把一个普通的 MLP 变成层级化的分解器。
- 基函数展开：趋势用多项式基、季节性用 Fourier 基、generic 版本用学习到的基。
- 为什么 N-BEATS 能同时拿"全场最准"和"全场最可解释"。
- M4 的实际成绩单：N-BEATS 到底打赢了谁，赢了多少。
- 一份完整的 PyTorch 实现，外加可以直接迁移的零售销量、电力需求两个案例。

**前置知识**：会用 PyTorch 写前馈网络。理解经典分解（趋势/季节性/残差）有帮助但不必须。

---

## 为什么纯全连接堆栈就够用

时间序列上的深度模型大多带结构先验：卷积假定平移等变、RNN 假定顺序隐状态、attention 假定两两相关。N-BEATS 反其道而行：把整个输入窗口塞进 MLP，让网络自己学最有用的分解是什么。聪明的不是层类型，是信息**走的路径**。

具体来说 N-BEATS 做了三个有主张的选择：

1. **若干个相同结构的块叠起来**，每块同时输出一个 *backcast*（输入窗口的重构）和一个 *forecast*（对未来的预测）。
2. **双重残差通路**：每块把自己的 backcast 从输入残差里减掉，把自己的 forecast 加到 forecast 累加器里。下一块只看到上一块**没解释掉的部分**。
3. **基函数输出头**。块不直接输出预测值，而是输出一个小的系数向量 $\theta$，再乘以一个固定（可解释）或学习（generic）的基矩阵。

这三点合在一起就足以拿下 M4 榜首。

---

## 架构：双重残差通路

想象两条管道从顶向下并排穿过网络。左边那条是**残差通路**：起点是输入窗口 $x \in \mathbb{R}^{H}$，每一块减掉自己的 backcast 之后变小一点。右边那条是**预测累加器**：起点是 0，每一块把自己的 forecast 加进去之后变大一点。

![N-BEATS 的双重残差堆叠](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig1_stacked_residual_blocks.png)

数学上，对块 $b = 1, \ldots, B$：

$$
r^{(b)} = r^{(b-1)} - \hat{x}^{(b)}, \qquad \hat{y} = \sum_{b=1}^{B} \hat{y}^{(b)},
$$

其中 $r^{(0)} = x$。每块看到的残差越来越小，所以会自然地专注于剩下还没被解释掉的频率/形状。粗模式（整体趋势、主导季节周期）被前几块吃掉；细修正交给后几块。

这就是梯度提升的思路，只不过装在一个端到端可微的网络里。和提升一样，**顺序很重要**：第一块工作最轻松（信号全在），最后一块最难（剩下的多是噪声 + 微妙结构）。

---

## 块的内部

每个 N-BEATS 块都长一个样。给定残差输入 $r \in \mathbb{R}^{H}$：

1. **特征提取**——四层 256-512 宽的全连接 + ReLU：
   $$
   h_1 = \mathrm{ReLU}(W_1 r + b_1), \quad \ldots, \quad h_4 = \mathrm{ReLU}(W_4 h_3 + b_4).
   $$
2. **系数投影**——两个线性头分别输出 backcast 和 forecast 的系数：
   $$
   \theta^{b} = W_b h_4, \qquad \theta^{f} = W_f h_4.
   $$
3. **基函数乘法**——固定或学习的矩阵 $V$ 把系数映射到时间域：
   $$
   \hat{x} = V_b \, \theta^{b}, \qquad \hat{y} = V_f \, \theta^{f}.
   $$

两个变体的差别只在 $V$ 是什么。

### 可解释：趋势 + 季节性基

趋势块用低次多项式基。设次数 $p$，时间索引 $\tau / H \in [0, 1]$：

$$
V_{\text{trend}} = \begin{pmatrix} 1 & \tau & \tau^{2} & \cdots & \tau^{p} \end{pmatrix}, \qquad
\hat{y}_{\text{trend}} = \sum_{i=0}^{p} \theta_i \, \tau^{i}.
$$

典型选 $p = 2$ 或 $3$。够拟合"先平稳上升后加速"这种形状，又不会拟合出多余抖动。

季节性块用 Fourier 基：

$$
V_{\text{seas}} = \begin{pmatrix} \sin(2\pi \cdot 1 \cdot \tau / T) & \cos(2\pi \cdot 1 \cdot \tau / T) & \cdots & \sin(2\pi K \tau / T) & \cos(2\pi K \tau / T) \end{pmatrix}.
$$

$K = 1, 2, 3$ 阶谐波，$T$ 是数据已知周期（月级 12，时级 24），任意形状的周期信号都能逼近。

可解释架构把一个趋势栈（几个趋势块）后面接一个季节性栈（几个季节性块）。训完之后你可以画出每个栈的贡献，对业务方解释："这一部分是底层趋势，这一部分是周周期。"

![可解释 N-BEATS 给出的趋势 + 季节性分解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig2_basis_decomposition.png)

### Generic：学习到的基

generic 版本让 $V_b$ 和 $V_f$ 也变成可学习矩阵。块不再被强制走"趋势/季节性"的语义，而是学梯度告诉它什么基有用。换来一点精度提升，代价是丢掉那张可读的分解图。

![可解释 vs generic 的栈结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig3_interpretable_vs_generic.png)

论文里有一条实用结论：M4 上**最好**的成绩来自可解释模型 + generic 模型 + 不同回看长度的**集成**。下文"集成策略"一节再展开。

---

## 为什么要走基函数输出头

块完全可以直接预测：$\hat{y} = W_y h_4$。为什么要绕一道 $\theta$ 向量 + 基矩阵？

三个理由：

- **归纳偏置**。把 forecast 强制写成"少量系数 × 平滑基"的线性组合，物理上限制了它去拟合噪声的能力。720 步输出 + 3 次多项式 = 趋势分量只有 4 个自由度，不可能产生振荡。这种正则化就是可解释版本能泛化的根本原因。
- **可解释性免费送**。趋势栈的 $\theta_0, \theta_1, \theta_2, \theta_3$ 直接对应基线、斜率、曲率、jerk。季节性栈的系数对应特定谐波的振幅。能画出来，能讲出来。
- **参数效率**。从 512 维隐状态直接连到 720 步预测，是一个 $512 \times 720 = 369K$ 的线性层。基函数头是两个小线性层（$512 \to p$，再 $p \to 720$ 走固定基），每个输出头通常远小于 10K 参数。

后续的 PatchTST、N-HiTS、TSMixer 也都用了某种分解头——这套思路是 N-BEATS 推广出来的。

---

## PyTorch 实现

下面是一份干净完整的实现。整个模型大概 120 行。

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
    """Fourier 基：取前 floor((H 或 F) / 2) 阶谐波。"""

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
    """学习基：backcast/forecast 关于 theta 是线性的。"""

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

一个小但重要的细节：可解释版本里，`TrendBasis` 和 `SeasonalityBasis` 实例**在同一个栈内的所有块之间共享**。每个块有自己的 MLP 和系数头，但它们都乘同一个固定基矩阵——既保住了归纳偏置，也省下一点参数。

---

## 训练配方

Oreshkin 等人的配方很朴素：

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_nbeats(model, train_loader, val_loader, epochs=100,
                 lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.L1Loss()  # MAE；论文按数据集用 sMAPE/MASE/MAPE，
                        # L1 是个稳妥默认
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

实操上的几个点：

- **损失函数**：M4 用 sMAPE；很多真实数据上 MAE（L1）比 MSE 稳，因为它不会过度惩罚离群点。选和你的评估指标一致的损失。
- **逐窗口标准化**：把**每个输入窗口**减去自身均值除以自身标准差再喂网络，预测结果再逆变换回去。这一步比损失函数选择重要得多，否则网络要额外去学每条序列的尺度，浪费容量。
- **Early stopping**：N-BEATS 喜欢长训练，但大多数数据集大约 50 epoch 后就走平了。盯着 val loss，不动了就停。

---

## N-BEATS 在 M4 上赢了什么

M4 比赛包含统计方法（ARIMA、ETS、Theta）、冠军 Smyl 的 ES-RNN 混合模型、第二名 FFORMA 这种基于特征元学习的方法。N-BEATS 没做任何统计预处理，就在总体 sMAPE 和六个频率桶里的五个上都赢了。

![N-BEATS 在 M4 上的成绩](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig4_m4_results.png)

论文里的数字：

- **N-BEATS（可解释 + generic 集成）：sMAPE 11.135**
- N-BEATS（仅 generic）：sMAPE 11.168
- Smyl ES-RNN（M4 冠军）：sMAPE 11.374
- FFORMA：sMAPE 11.720
- 最好的经典方法（Theta）：sMAPE 12.309

绝对差距在 sMAPE 单位上看不算大，但跨年/季/月/周/日的桶里都稳定领先。小时级是唯一一个 Smyl ES-RNN 反超的桶，差距 0.4 sMAPE。

更深的结论是：足够表达力 + 合适归纳偏置的深度模型，可以从零学出统计学家用几十年手工打磨的东西。

---

## 集成：配方里没大声说的另一半

仔细读 M4 论文你会注意到一行脚注：表头那个 N-BEATS 数字是 **180 个模型实例的中位数**预测。每个实例在三件事上有差异：回看长度（2H, 3H, ..., 7H）、训练损失（sMAPE vs MASE vs MAPE）、随机种子。单模型表现明显比集成差。

![为什么 N-BEATS 论文都报集成结果](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/time-series/07-N-BEATS%E6%B7%B1%E5%BA%A6%E6%9E%B6%E6%9E%84/fig5_ensemble_strategy.png)

右图的实证曲线显示边际效益递减：大部分收益来自前 ~30 个成员。生产环境几乎不用真训 180 个——10 到 30 个，混合不同回看长度和种子，基本就把提升吃掉了。

简易集成工具：

```python
class EnsembleNBeats:
    def __init__(self, models: list[NBeats]):
        self.models = models

    def predict(self, x: torch.Tensor, aggregator="median") -> torch.Tensor:
        outs = torch.stack([m(x) for m in self.models], dim=0)
        return outs.median(dim=0).values if aggregator == "median" else outs.mean(0)
```

中位数比均值更适合 sMAPE 这类损失，因为它对"某个模型在某窗口上抽风"更鲁棒。

---

## 案例 1：月度零售销量

**任务**：给定过去 36 个月销量，预测多店连锁的下 12 个月单量。强假日季节性（12 月峰）、向上趋势 + 偶发促销，约 200 条不同的产品-门店序列。

**架构选择**：可解释版本。业务团队需要能够指着月度预测说"$X$ 来自底层趋势，$Y$ 来自循环的 12 月节日抬升，$Z$ 是残差。"`[3 个趋势块] + [3 个季节性块]` 的可解释栈直接给你这个。

```python
model = make_interpretable(
    history=36, horizon=12,
    trend_blocks=3, seasonal_blocks=3,
    trend_degree=2,                     # 平滑的多年趋势
    hidden=256, layers=4,
)
```

训完之后可以提取每个栈的贡献，看看每一部分到底学了什么：

```python
def stack_contributions(model: NBeats, x: torch.Tensor) -> dict:
    """单窗口下，按栈拆出 forecast 贡献。"""
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

**典型数字**：5 年历史的真实零售序列上，可解释 N-BEATS 能在 12 个月 horizon 上拿到 7-12% MAPE，与最好的梯度提升 + 特征工程管道相当——但完全不用做特征工程。可解释性才是真正的赢点：让业务团队能用已知促销活动覆盖 12 月季节性的模型，比在回测上多 0.5% 精度的黑盒有用得多。

---

## 案例 2：小时级电力需求

**任务**：给定过去 168 小时（一周）的电网总需求，预测未来 24 小时。强日周期 + 周周期，对天气敏感，热浪期间会有需求尖峰。

**架构选择**：generic。模式复杂（多分辨率日 + 周 + 天气驱动），业务团队主要关心精度——预测每多 1 MW 误差就意味着多备一份运营储备。换更深的 generic 栈，宽隐状态。

```python
model = make_generic(
    history=168, horizon=24,
    num_blocks=30, theta_size=32,
    hidden=512, layers=4,
)
```

**这里逐窗口标准化最关键**。需求水平随季节大幅变化；不做标准化的话网络会浪费容量去学"冬天比夏天大"。做了之后网络只需学形状。

```python
def normalise_window(x: torch.Tensor) -> tuple:
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return (x - mu) / sd, mu, sd

# 前向：
x_norm, mu, sd = normalise_window(x)
y_norm = model(x_norm)
y_hat = y_norm * sd + mu  # 广播到 (B, H)
```

**典型数字**：公开 ETT（Electricity Transformer Temperature）数据集上 24 小时 horizon，10 个 generic N-BEATS 集成的 MSE 大约 0.31，明显领先 LSTM（约 0.42），实现成本远低于 Informer 但精度可比。

---

## 超参 cheat sheet

新数据集上的实用起点：

| 超参 | 默认（可解释） | 默认（generic） | 调参提示 |
|---|---|---|---|
| 回看 `history` | horizon × 4-7 | horizon × 4-7 | 越大覆盖越多季节；以 2 倍步长试。 |
| 隐藏宽度 | 256 | 512 | 验证 loss 高位停就调大。 |
| 块内 MLP 层数 | 4 | 4 | 极少需要改。 |
| 趋势次数 | 2 或 3 | -- | 看到曲率就 3，否则 2。 |
| 趋势块数 | 3 | -- | 趋势 RMSE 主导误差时加。 |
| 季节性块数 | 3 | -- | 多分辨率季节性时加。 |
| Generic 块数 | -- | 20-30 | 一般继续加还能涨，慢慢涨。 |
| Generic $\theta$ | -- | 32 | 16 欠拟合，64+ 一般无用。 |
| 损失 | MAE / sMAPE | MAE / sMAPE | 与评估指标对齐。 |
| 优化器 | Adam，lr 1e-3 | Adam，lr 1e-3 | cosine annealing，不需要 warmup。 |
| 集成大小 | 10-30 | 10-30 | 中位数聚合。 |

---

## 什么时候**不**用 N-BEATS

- **超长 horizon（>1000 步）**。$\theta \to \hat{y}$ 的输出投影成为参数瓶颈。换 **N-HiTS**（N-BEATS 的后继）或 PatchTST，它们用多采样率分层来扩展。
- **跨特征强相关的多变量序列**。N-BEATS 设计时是单变量。要么每个变量训一份，要么换 TFT / Informer / DeepAR。
- **极短序列（~50 个观测）**。MLP 主干有几万参数；这种数据量上 Theta 或 ETS 会赢。
- **在线流式预测**。N-BEATS 处理固定窗口而不是流式隐状态。LSTM 或 TCN 更合适。
- **预测必须满足硬约束**（非负、整数）。N-BEATS 输出无约束实数；你可以后处理或用带输出 clip 的分位损失，但带原生分布输出的模型（DeepAR）可能更省事。

---

## Q&A

### 为什么偏要用多项式和 Fourier？

多项式在紧区间上对连续函数稠密（Stone-Weierstrass），Fourier 对周期函数稠密。两者合在一起对"平滑趋势 + 周期分量"是非常强的先验，恰好匹配大多数真实序列的样子。

### 3 次多项式的趋势块能学高阶行为吗？

局部能，整体不行。每个块看到的是上一批块之后的**残差**，所以三块趋势栈相当于复合三个三次函数，对任何合理 horizon 上的趋势都够。

### 为什么用 Adam 不用 SGD？

经验上 N-BEATS 用 Adam 收敛快很多。论文报告 SGD 需要激进的学习率调参才能匹配 Adam 的默认。

### N-BEATS 需要位置编码吗？

不需要。MLP 把整个窗口当固定大小向量看；位置隐含在输入向量的列索引里，输出端的基矩阵则把每步时间索引编码进去了。

### 怎么把 N-BEATS 扩展到带外生变量（天气、节假日）？

官方扩展 **N-BEATS-X**：把外生协变量拼到输入窗口，再给每块加一个辅助输入头。生产里很多场景，简单地把外生序列沿输入向量 concat、然后稍微加宽第一层 MLP 就够用。

### N-HiTS 是什么，是不是直接上它就行？

N-HiTS（2023）是同一批作者的后继。它在输出端加了多采样率下采样和插值，能扩展到更长 horizon（720+ 步）且跑得更快。短到中等 horizon（<100 步）上，原版 N-BEATS 仍然有竞争力且更简单。

**为啥一定要集成？我单模型挺好。**
你这个数据集上幸运而已。论文显示在 10 万条 M4 序列上单模型方差很大——哪怕 5 个成员的集成也能再省 1-2 sMAPE。如果只是单点估计 + 数据小，可以略；如果要报数字，一定要集成。

---

## 小结

N-BEATS 是一个"架构无聊到极致"的模型，赢就赢在把对的块按对的顺序堆起来。双重残差通路给了它类似 boosting 的行为；基函数输出头给了它强归纳偏置和（可解释版本里）免费分解；M4 榜单证明这个组合既打得过经典方法也打得过递归深度模型。

对于大多数采样规整、有明显趋势/季节性的单变量预测问题，N-BEATS 是开箱可用的最强基线之一。从可解释版本起步以争取 stakeholder 认可，需要每一分精度时切换到（或与）generic 版本集成，记得跨回看长度和种子做集成。

下一章我们用 **Informer** 收尾，它解决的是另一个问题：怎么让 Transformer 跑到几千步 horizon，而 $\mathcal{O}(L^2)$ 的注意力代价不会把你压垮。

---

## 参考资料

- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 Competition: 100,000 Time Series and 61 Forecasting Methods.* International Journal of Forecasting, 36(1).
- Smyl, S. (2020). *A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting.* International Journal of Forecasting, 36(1).
- Challu, C. et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI.
- Olivares, K. et al. (2023). *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with N-BEATS-X.* International Journal of Forecasting.

---

**系列导航**

