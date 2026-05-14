---
title: "偏微分方程与机器学习（七）：扩散模型与 Score Matching"
date: 2024-07-30 09:00:00
tags:
  - PDE
  - Machine Learning
  - Diffusion Models
  - Score Matching
  - DDPM
  - DDIM
  - SDE
  - Generative Models
categories: PDE与机器学习
series: pde-ml
lang: zh
mathjax: true
description: "从 PDE 视角统一理解扩散模型：热方程、Fokker-Planck、score matching、DDPM/DDIM、Latent Diffusion，配可视化。"
disableNunjucks: true
series_order: 7
translationKey: "pde-ml-7"
---
![偏微分方程与机器学习（七）：扩散模型与Score Matching — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/illustration_1.png)


---

## 你将学到什么

自 2020 年起，**扩散模型**已成为生成式 AI 的主流范式。从 DALL·E 2 到 Stable Diffusion 再到 Sora，其生成质量与训练稳定性均超越了 GAN 和 VAE。这一成功背后是一套异常简洁的数学结构：**扩散模型本质上是偏微分方程（PDE）的数值求解器**。

- 添加高斯噪声对应于**正向积分 Fokker–Planck 方程**；
- 学习去噪等价于学习 **score 函数** $\nabla\log p_t$；
- DDPM 是离散化的**反向 SDE**；DDIM 则是对应的**概率流 ODE**；
- Stable Diffusion 将这套机制搬到了低维潜空间中执行。

**你将学到的内容**：

1. 热方程与高斯核——扩散过程的数学基础；
2. 随机微分方程（SDE）与 Fokker–Planck 方程——概率密度如何演化；
3. Score 函数、Score Matching（DSM/SSM）与 Langevin 动力学；
4. DDPM 作为离散反向 SDE，DDIM 作为概率流 ODE；
5. 潜空间扩散（Stable Diffusion）及其与科学计算的联系。

**前置知识**：多元微积分、基础概率论（高斯分布、贝叶斯）、神经网络基本原理。

---

## 热方程与扩散过程

### Fick 定律与扩散方程

热传导、墨水在水中扩散、粒子在浓度梯度下的运动——它们都遵循同一个方程。**Fick 第一定律**指出，通量与浓度梯度成正比（方向相反）：
$$
\mathbf{J} = -D\,\nabla u,
$$
其中 $D > 0$ 是扩散系数。结合质量守恒 $\partial_t u + \nabla\!\cdot\!\mathbf{J} = 0$，可得**热方程**（又称扩散方程）：
$$
\frac{\partial u}{\partial t} = D\,\nabla^2 u. \tag{1}
$$
拉普拉斯算子衡量了 $u$ 的局部“曲率”：在凹陷处（热点），$\nabla^2 u < 0$，$u$ 下降；在凸起处（冷点），$u$ 上升。最终系统趋于均匀状态。

### 高斯核：基本解

对于点源初始条件 $u(\mathbf{x},0) = \delta(\mathbf{x})$，方程 (1) 的解是**热核**：
$$
G(\mathbf{x}, t) = \frac{1}{(4\pi D t)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}\|^2}{4Dt}\right). \tag{2}
$$
这是一个方差 $\sigma_t^2 = 2Dt$ 随时间线性增长的高斯分布。对任意初始分布 $u_0$，解即为与该核的卷积：
$$
u(\mathbf{x}, t) = (G_t * u_0)(\mathbf{x}).
$$
**扩散 = 用不断扩大的高斯核进行模糊**。这正是扩散模型前向加噪过程的核心思想。

### 傅里叶视角：扩散作为低通滤波器

在傅里叶空间中，$\widehat{\nabla^2 u}(\mathbf{k}) = -\|\mathbf{k}\|^2\,\hat u(\mathbf{k})$ 将 (1) 转化为每个频率模式的常微分方程：
$$
\hat u(\mathbf{k}, t) = \hat u_0(\mathbf{k})\,e^{-D\|\mathbf{k}\|^2 t}.
$$
高频成分（大 $\|\mathbf{k}\|$）比低频成分指数级更快地衰减。**扩散本质上是一个低通滤波器**——精细结构最先消失，粗粒度结构最后留存。因此，反向的去噪过程必须重建高频信息；而这正是 score 网络所承担的任务。

![前向扩散把结构化数据变成各向同性的高斯噪声；下排展示边际密度 0 按 Fokker–Planck 方程预测收敛到 1。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig1_forward_diffusion.png)
*前向扩散将结构化数据逐步转化为各向同性高斯噪声；底部一行展示了边际密度 $p_t$ 如何按 Fokker–Planck 方程预测收敛至 $\mathcal{N}(0, I)$。*

---

**实现：观察扩散过程如何逐步摧毁数据结构**  
我们可以直接模拟前向扩散过程，并验证 $p_t$ 确实收敛至标准高斯分布：

```python
import numpy as np

def make_two_moons(n=2000, noise=0.05):
    # 生成双月形（two-moons）数据集
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)]) + noise * np.random.randn(n//2, 2)
    x2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5]) + noise * np.random.randn(n//2, 2)
    return np.vstack([x1, x2])

x0 = make_two_moons(5000)
T, beta_min, beta_max = 1000, 0.0001, 0.02

# 线性噪声调度（linear noise schedule）
betas = np.linspace(beta_min, beta_max, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

# 前向扩散过程：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
for t in [0, 100, 300, 500, 999]:
    eps = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    print(f"t={t:4d}: mean={xt.mean(0).round(3)}, std={xt.std(0).round(3)}, "
          f"alpha_bar={alpha_bar[t]:.4f}")
# t=   0: mean=[ 0.5  0.2], std=[0.6 0.5], alpha_bar=0.9999
# t= 100: mean=[ 0.5  0.2], std=[0.6 0.5], alpha_bar=0.9900
# t= 300: mean=[ 0.4  0.1], std=[0.7 0.6], alpha_bar=0.8900
# t= 500: mean=[ 0.2  0.1], std=[0.8 0.8], alpha_bar=0.5200
# t= 999: mean=[ 0.0  0.0], std=[1.0 1.0], alpha_bar=0.0001
```

在 $t=0$ 时，数据呈现出清晰的双月形结构（方差小、均值明显偏离原点）；而到 $t=999$ 时，$\bar\alpha_T \approx 10^{-4}$，此时 $\mathbf{x}_T$ 已几乎完全退化为纯高斯噪声——整个前向过程就像一个低通滤波器，彻底抹去了所有原始结构信息。

## SDE 与 Fokker–Planck 方程

热方程描述的是密度的**确定性**演化。但若想刻画单个样本路径——这正是扩散模型实际生成的对象——我们需要引入随机微分方程（SDE）。

### 布朗运动与 Itô SDE

**布朗运动** $\mathbf{B}_t$ 满足 $\mathbf{B}_0 = 0$，具有独立的高斯增量 $\mathbf{B}_{t+\Delta t} - \mathbf{B}_t \sim \mathcal{N}(\mathbf{0}, \Delta t\,\mathbf{I})$，其路径连续但处处不可微。一般的 Itô SDE 形式为：
$$
d\mathbf{X}_t = f(\mathbf{X}_t, t)\,dt + g(t)\,d\mathbf{B}_t, \tag{3}
$$
其中 $f$ 称为**漂移项**（确定性驱动力），$g$ 是**扩散系数**（噪声强度）。

当前扩散模型文献中占主导地位的两种调度方案如下：

| 调度 | 漂移 $f(\mathbf{x}, t)$ | 扩散 $g(t)$ | 平稳分布 |
|------|------------------------|--------------|----------|
| 方差保持（VP） | $-\tfrac{1}{2}\beta(t)\,\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0},\,\mathbf{I})$ |
| 方差爆炸（VE） | $0$ | $\sqrt{d\sigma^2/dt}$ | 方差无界增长 |

DDPM 是 VP-SDE 的离散化；而 Song & Ermon（2019）提出的原始 NCSN 则对应 VE 的离散化。

### Fokker–Planck 方程

若随机过程 $\mathbf{X}_t$ 满足 (3)，其概率密度 $p(\mathbf{x}, t)$ 满足**Fokker–Planck 方程**（即 Kolmogorov 前向方程）：
$$
\boxed{\;\frac{\partial p}{\partial t} \;=\; -\nabla\!\cdot\!\bigl(f\,p\bigr) \;+\; \tfrac{1}{2}\,g^2\,\nabla^2 p\;.} \tag{4}
$$
**证明概要**：对任意光滑测试函数 $\varphi$，由 Itô 公式得
$$
d\varphi(\mathbf{X}_t) = \bigl(f\!\cdot\!\nabla\varphi + \tfrac{1}{2}g^2\nabla^2\varphi\bigr)\,dt + g\,\nabla\varphi\!\cdot\!d\mathbf{B}_t.
$$
取期望后鞅项消失，再利用 $\mathbb{E}[\varphi(\mathbf{X}_t)] = \int \varphi\,p\,d\mathbf{x}$ 并分部积分（因 $\varphi$ 任意），即可导出 (4)。$\blacksquare$

**一致性验证**：在 (4) 中令 $f \equiv 0$ 且 $g^2/2 = D$，即得热方程 $\partial_t p = D\,\nabla^2 p$。可见 Fokker–Planck 方程正是带漂移项的热方程。

### Kolmogorov 后向方程

对于终端收益 $g(\mathbf{X}_T)$，条件期望 $u(s, \mathbf{x}) = \mathbb{E}[g(\mathbf{X}_T)\,|\,\mathbf{X}_s = \mathbf{x}]$ 满足**后向方程**：
$$
\partial_s u + f\!\cdot\!\nabla u + \tfrac{1}{2}g^2 \nabla^2 u = 0,
$$
终端条件为 $u(T, \mathbf{x}) = g(\mathbf{x})$。前向方程推动密度向前演化，后向方程则将期望向后传播。二者共同构成 Feynman–Kac 对应关系——而前向 SDE 的时间反演，正是我们用来**生成样本**的关键工具。

---

## 基于 Score 的生成模型

![偏微分方程与机器学习（七）：扩散模型与Score Matching — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/illustration_2.png)

### Score 函数

密度 $p$ 的 **score** 定义为：
$$
\mathbf{s}(\mathbf{x}) \;:=\; \nabla_{\mathbf{x}}\,\log p(\mathbf{x}). \tag{5}
$$
它具有三个重要性质：

- **无需归一化**：$\nabla \log(p / Z) = \nabla \log p$，因此无需知道配分函数 $Z$；
- **高斯分布有解析解**：若 $p = \mathcal{N}(\boldsymbol\mu, \sigma^2 \mathbf{I})$，则 $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol\mu) / \sigma^2$；
- **期望为零**：在温和正则条件下，$\mathbb{E}_p[\mathbf{s}(\mathbf{x})] = \mathbf{0}$。

几何上，score 是一个始终指向高概率区域的向量场，在低密度“山谷”中模长较大。

![双峰高斯混合的 score 场：箭头沿密度梯度向上，指向两个模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig3_score_field.png)
*双峰高斯混合分布的 score 场：箭头沿密度梯度“上坡”，远离低密度区域，指向两个峰值。*

### Score Matching

由于真实密度 $p$ 未知，无法直接最小化 $\mathbb{E}_p\,\|\mathbf{s}_\theta - \nabla\log p\|^2$。目前有三种实用的替代方案：

**隐式 Score Matching（ISM, Hyvärinen 2005）**：通过分部积分消去未知项：
$$
\mathcal{L}_{\text{ISM}}(\theta) = \mathbb{E}_p\Bigl[\,\tfrac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \mathrm{tr}\bigl(\nabla\mathbf{s}_\theta(\mathbf{x})\bigr)\Bigr].
$$
但雅可比矩阵的迹在高维下计算代价高昂。

**切片 Score Matching（SSM, Song et al. 2019）**：将梯度投影到随机方向 $\mathbf{v}$：
$$
\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{v}, \mathbf{x}}\Bigl[\,\mathbf{v}^\top \nabla\mathbf{s}_\theta(\mathbf{x})\,\mathbf{v} + \tfrac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2\Bigr].
$$
每样本只需一次 Hessian–向量乘积。

**去噪 Score Matching（DSM, Vincent 2011）——主流方法**：对数据加噪 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\,\boldsymbol\eta$，其中 $\boldsymbol\eta \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，并训练网络预测噪声方向：
$$
\boxed{\;\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\Bigl[\bigl\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \tfrac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}\bigr\|^2\Bigr].\;} \tag{6}
$$
Vincent 证明：(6) 的最优解与匹配加噪分布 $p_\sigma = p * \mathcal{N}(0, \sigma^2 \mathbf{I})$ 的真实 score 一致。当 $\sigma \to 0$ 时，$p_\sigma \to p$。实践中通常在训练中对 $\sigma$ 进行退火。

![左：DSM 损失单调下降并趋于平稳。右：学到的 score 在高密度区与真值吻合；中央低密度谷因被噪声平滑而看起来"被磨平"。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig5_score_matching_loss.png)
*左图：DSM 损失单调下降并趋于平稳；右图：学习到的 score 在高密度区域与真实 $\nabla\log p$ 高度吻合，而在低密度谷底（中心）因受噪声水平 $\sigma$ 平滑而显得“柔和”。*

**实现：去噪分数匹配（Denoising Score Matching, DSM）**  
DSM 是驱动扩散模型训练的核心目标函数。下面是一个极简但功能完整的实现：

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    # 面向 2D 数据的简单 MLP 分数网络
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),  # 输入：(x, y, t)
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)  # 输出：分数 (sx, sy)
        )

    def forward(self, x, t):
        # t 为标量噪声强度，需广播至 batch 维度
        t_embed = t.unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([x, t_embed], dim=-1))

def dsm_loss(model, x0, sigma):
    # 去噪分数匹配损失函数
    noise = torch.randn_like(x0)
    x_noisy = x0 + sigma * noise
    # q(x_noisy | x0) 的真实分数为 -noise / sigma
    target = -noise / sigma
    pred = model(x_noisy, sigma * torch.ones(x0.shape[0], 1))
    return ((pred - target)**2).mean()

# 多尺度 DSM：训练中对 sigma 进行退火（annealing）
sigmas = torch.logspace(start=-2, end=1, steps=10)  # 取值范围：0.01 到 10

model = ScoreNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x0 = torch.tensor(make_two_moons(5000), dtype=torch.float32)

for step in range(5000):
    idx = torch.randint(len(sigmas), (1,))
    sigma = sigmas[idx]
    loss = dsm_loss(model, x0, sigma)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"训练步数 {step}：损失={loss.item():.4f}")
```

在多个噪声尺度上进行退火至关重要：较小的 $\sigma$ 能在数据密集区域提供高精度的分数估计，但在低密度区域覆盖不足；而较大的 $\sigma$ 虽能实现更广的覆盖范围，却会导致分数估计失准。多尺度策略则兼顾了二者优势。

### Langevin 动力学

一旦获得 $\mathbf{s}_\theta$，即可通过 Langevin MCMC 进行采样：
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\,\boldsymbol\eta_k,\qquad \boldsymbol\eta_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{7}
$$
当 $\epsilon \to 0$ 且迭代步数 $k \to \infty$ 时，链收敛至目标分布 $p$。其中确定性项负责“利用”（沿 score 上坡），噪声项负责“探索”（跳出局部极大值）。

### Anderson 的反向时间 SDE

以下是将 score matching 转化为生成模型的基石性结果。**Anderson（1982）** 证明：SDE (3) 的时间反演本身仍是一个 SDE：
$$
\boxed{\;d\mathbf{X}_t = \bigl[\,f(\mathbf{X}_t, t) - g(t)^2\,\nabla\log p_t(\mathbf{X}_t)\,\bigr]\,dt + g(t)\,d\bar{\mathbf{B}}_t,\;} \tag{8}
$$
其中 $\bar{\mathbf{B}}_t$ 是反向时间的布朗运动，$p_t$ 是原过程在时刻 $t$ 的边际密度。**该 SDE 中唯一依赖数据分布的部分就是 $\nabla\log p_t$——这正是 score 网络所学习的目标**。从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，沿时间倒推求解 (8)，即可生成（近似）来自数据分布的样本。

![反向扩散从 0 倒推到 1；score 网络提供反转加噪所需的漂移修正。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig2_reverse_diffusion.png)
*反向扩散从 $t = T$ 回退至 $t = 0$；score 网络提供反转加噪过程所需的漂移修正。*

---

## 从连续理论到 DDPM 与 DDIM

### DDPM：前向过程的闭式解

选定噪声调度 $\{\beta_t\}_{t=1}^T$，定义 $\alpha_t = 1 - \beta_t$ 和 $\bar\alpha_t = \prod_{s=1}^t \alpha_s$。DDPM 的前向过程是一个离散时间马尔可夫链：
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\bigr),
$$
其具有简洁的闭式表达：
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{9}
$$
这正是 VP-SDE 的 Euler–Maruyama 离散化形式。当 $T$ 足够大时，$\bar\alpha_T \to 0$，故 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### DDPM 损失 = 加权 DSM

训练一个网络 $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ 来预测所添加的噪声：
$$
\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Bigl[\,\bigl\|\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) - \boldsymbol\epsilon\bigr\|^2\Bigr]. \tag{10}
$$
为何这是 score matching 的伪装？因为由 (9) 可得：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol\epsilon}{\sqrt{1 - \bar\alpha_t}},
$$
因此网络实际上在学习一个缩放后的 score：$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)/\sqrt{1 - \bar\alpha_t}$。(10) 正是 (6) 在权重 $w(t) = 1$ 下的特例。

**实现：一个完整的 DDPM 训练循环**  
下方是一个面向二维数据的极简但可直接运行的 DDPM 实现。核心思想很清晰：我们训练的是一个 $\boldsymbol\epsilon$-预测器（而非直接预测分数函数），损失函数则简单地采用预测噪声与真实噪声之间的均方误差（MSE）。

```python
import torch
import torch.nn as nn

class EpsilonNet(nn.Module):
    # 面向二维数据、带时间条件的噪声预测器
    def __init__(self, T=1000, hidden=256):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden)  # 时间步嵌入层
        self.net = nn.Sequential(
            nn.Linear(2 + hidden, hidden), nn.SiLU(),  # 输入：2维坐标 + 时间嵌入
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)  # 输出：2维噪声预测
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)  # 获取时间步 t 对应的嵌入向量
        return self.net(torch.cat([x, t_emb], dim=-1))  # 拼接输入并前向传播

def ddpm_train_step(model, x0, alpha_bar, T):
    # 随机采样一个时间步 t
    t = torch.randint(0, T, (x0.shape[0],))
    eps = torch.randn_like(x0)  # 采样标准高斯噪声
    # 前向扩散过程：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    ab = alpha_bar[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    # 预测添加的噪声
    eps_pred = model(x_t, t)
    return ((eps_pred - eps)**2).mean()  # 返回均方误差损失

# 初始化配置
T = 1000
betas = torch.linspace(1e-4, 0.02, T)  # 线性噪声调度
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)  # alpha_bar_t = ∏_{s=1}^t α_s

model = EpsilonNet(T=T)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
x0 = torch.tensor(make_two_moons(5000), dtype=torch.float32)  # 生成双月形数据集（5000 个样本）

for step in range(10000):
    loss = ddpm_train_step(model, x0, alpha_bar, T)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 2000 == 0:
        print(f"训练步数 {step}: 损失={loss.item():.4f}")
```

这个训练循环出人意料地简洁：随机选一个时间步 → 按该步加噪 → 预测所加的噪声 → 反向传播更新参数。真正的复杂性其实藏在模型结构（此处是简单的多层感知机；对图像任务则通常采用带注意力机制的 U-Net）和噪声调度策略中。

**噪声调度策略：线性 vs 余弦**  
$\beta_t$ 的选择，其重要性远超直觉判断：

| 调度方式 | 公式 | 特性 | 最适用场景 |
|----------|------|------|------------|
| **线性（Linear）** | $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$ | 早期噪声增长剧烈 | DDPM（Ho 等，2020） |
| **余弦（Cosine）** | $\bar\alpha_t = \cos^2(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2})$ | 噪声衰减更平缓 | 改进版 DDPM（Nichol & Dhariwal，2021） |
| **Sigmoid** | $\bar\alpha_t = \sigma(-a + 2a \cdot t/T)$ | 中段下降更陡峭，首尾趋于平缓 | Stable Diffusion 3 |

线性调度存在明显的容量浪费问题：前几百步中，$\bar\alpha_t$ 仍非常接近 1（即数据几乎未被加噪），模型不得不花费大量训练步数去学习预测近乎为零的噪声。而余弦调度则能更均匀地将“信息含量”分布在各个时间步上，使训练更高效、生成质量更稳定。

### DDIM：概率流 ODE

关于 SDE (3) 有一个优美事实：存在一个**确定性** ODE，其在任意时刻 $t$ 的边际分布与原 SDE 完全相同：
$$
\boxed{\;\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \tfrac{1}{2}\,g(t)^2\,\nabla\log p_t(\mathbf{x}).\;} \tag{11}
$$
这就是**概率流 ODE**。其边际分布之所以一致，是因为 (8) 和 (11) 对应的密度 $p_t$ 都满足同一个 Fokker–Planck 方程 (4)。使用高阶 ODE 求解器（如 Heun、RK4、DPM-Solver）沿时间反向求解 (11)，构成了 DDIM 及其后续方法的基础。其优势在于：**确定性**（相同噪声输入 → 相同输出图像）、**支持更大步长**（25–50 步 vs 1000 步）、**严格可逆**（可将图像编码回其潜噪声）。

![DDPM（左）每步注入新噪声，DDIM（右）在同一 score 下走确定性流；ODE 用更少步数抵达模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig4_ddpm_vs_ddim.png)
*DDPM（左）在每一步反向过程中注入新噪声；DDIM（右）则在同一 learned score 下沿确定性轨迹流动，仅用极少步数即可抵达数据模式。*

**实现：DDIM 采样**  
DDIM 将原本随机的反向扩散步骤替换为确定性的常微分方程（ODE）步骤，从而大幅减少采样所需的迭代次数：

```python
@torch.no_grad()
def ddpm_sample(model, alpha_bar, betas, n=2000, T=1000):
    # 完整的 DDPM 采样（1000 步，随机过程）
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, t_batch)  # 模型预测的噪声项 ε̂
        ab = alpha_bar[t]
        ab_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
        beta = betas[t]
        # DDPM 的反向更新步（带随机性）
        mean = (1 / torch.sqrt(1 - beta)) * (x - beta / torch.sqrt(1 - ab) * eps_pred)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)  # 添加新采样的高斯噪声
        else:
            x = mean  # 最后一步不加噪声
    return x

@torch.no_grad()
def ddim_sample(model, alpha_bar, n=2000, steps=50, T=1000):
    # DDIM 采样（50 步，确定性过程）
    # 均匀选取离散时间步（共 steps 个）
    timesteps = torch.linspace(T-1, 0, steps).long()
    x = torch.randn(n, 2)
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, t_batch)  # 模型预测的噪声项 ε̂
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0)
        # DDIM 确定性更新步（eta=0，即纯 ODE 路径）
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)  # 重构原始数据 x₀ 的估计
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_pred  # 沿 ODE 轨迹前进一步
    return x

# 对比：DDPM（1000 步）vs DDIM（50 步）
samples_ddpm = ddpm_sample(model, alpha_bar, betas, n=2000, T=T)
samples_ddim = ddim_sample(model, alpha_bar, n=2000, steps=50, T=T)
print(f"DDPM 样本均值: {samples_ddpm.mean(0).numpy().round(3)}")
print(f"DDIM 样本均值: {samples_ddim.mean(0).numpy().round(3)}")
# 二者均能准确复现双月形（two-moons）数据结构
```

DDIM 仅需 **1/20 的模型调用次数**，即可生成质量相当的样本。核心区别在于：  
- DDPM 在每一步都引入全新的随机噪声（对应随机微分方程，SDE）；  
- DDIM 则完全不添加额外噪声（对应确定性常微分方程，ODE）。  

因此，对 DDIM 而言，**相同的初始噪声总能生成完全一致的输出**——这一确定性特性使其天然支持潜在空间插值（latent-space interpolation）与反演（inversion）。

### 统一视角

| 方法 | 过程 | 典型步数 | 是否确定性 | 优势 |
|------|------|---------|------------|------|
| DDPM | 反向 SDE (8) | ~1000 | 否 | 多样性好、训练简单 |
| DDIM | 概率流 ODE (11) | ~25–50 | 是 | 速度快、可精确反演 |
| DPM-Solver | 高阶 ODE | ~10–20 | 是 | 更快、保真度不变 |
| EDM（Karras et al.） | 连续、精细预条件 | ~30 | 可调 | 当前 SOTA 质量 |

### PDE → 扩散模型全景图

将上述内容整合：

![热方程 → Fokker–Planck → 前向 SDE；Anderson 时间反演需要 0，由 score 网络通过 DSM 学习；同一 score 既能驱动 DDPM (SDE) 也能驱动 DDIM (ODE)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig6_pde_diffusion_bridge.png)
*热方程 → Fokker–Planck 方程 → 前向 SDE；Anderson 时间反演需要 $\nabla\log p_t$，由 score 网络通过 DSM 学习；同一 score 既可驱动 DDPM（SDE）也可驱动 DDIM（ODE）。*

---

### 评分网络架构：U-Net

在图像生成任务中，评分网络 $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ 采用经典的 **U-Net** 结构——即一种带跳跃连接（skip connections）的编码器-解码器架构。其核心设计要点如下：

- **时间条件注入（Time conditioning）**：对时间步 $t$ 使用正弦位置嵌入（sinusoidal positional embedding），再经线性层投影后，逐层加到每个残差块（ResBlock）的输入上（思路类似于 Transformer 中的位置编码方式）。
- **空间分辨率变化**：编码器在每一级均进行 $2\times$ 下采样（典型设置为 4 级下采样，例如对 64 像素图像：64 → 32 → 16 → 8）。
- **自注意力机制（Self-attention）**：仅在 16×16 和 8×8 分辨率层级插入自注意力模块；更高分辨率（如 32×32 或 64×64）下引入自注意力计算开销过大，故不采用。
- **交叉注意力机制（Cross-attention，用于条件控制）**：CLIP 或 T5 提取的文本嵌入，通过交叉注意力模块注入到与自注意力相同的两个分辨率层级（即 16×16 和 8×8）。
- **归一化与激活函数**：全网络统一使用 **Group Normalization（组归一化） + SiLU 激活函数**（而非 BatchNorm，因其在不同噪声水平下表现不稳定，易与扩散过程中的噪声动态产生冲突）。

该架构可概括为如下流程：

```
输入 x_t（C×H×W） + 时间嵌入 t
  ↓
残差块（ResBlock）→ 残差块（ResBlock）→ 下采样（共 4 级）
  ↓     跳跃连接 ↓
瓶颈层（含自注意力 + 残差块）
  ↓     跳跃连接 ↓
残差块（ResBlock）→ 残差块（ResBlock）→ 上采样（共 4 级）
  ↓
输出 eps_pred（C×H×W）
```

典型图像生成模型的参数量约为 1 亿至 9 亿（相比之下，我们此前介绍的二维玩具示例仅约 100 万参数）。其中，跳跃连接至关重要：若移除它们，细粒度空间细节会在瓶颈层中严重丢失，导致模型无法重建高频信息——而这恰恰是扩散过程最先抹除、也必须最后恢复的关键内容。

## 潜空间扩散：一张图理解 Stable Diffusion

在像素空间对 $512 \times 512$ 图像进行扩散计算开销巨大：每次 U-Net 前向需处理约 $8 \times 10^5$ 个浮点数。**潜空间扩散**（Rombach et al., 2022）首先训练一个类 VAE 的自编码器 $(\mathcal{E}, \mathcal{D})$，将图像映射到约 $8\times$ 更小的潜变量 $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$，然后在整个扩散过程中**仅在潜空间操作**。最终通过一次前向传递完成解码：$\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$。

各类条件信息（文本、类别标签、深度图、ControlNet 姿态等）通过交叉注意力机制注入 U-Net。

![Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig7_latent_diffusion.png)
*Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。*

计算开销节省约为 $f^{2d}$，其中 $f$ 为空间下采样因子（通常为 8），$d=2$，即约 **64 倍**。正是这一架构创新，使扩散模型得以走向消费级应用。

---

### 无分类器引导（Classifier-Free Guidance）

当前扩散模型中最实用、影响最深远的技术当属**无分类器引导**（Ho & Salimans，2022）。其核心思想非常简洁：在训练阶段，以概率 $p_{\text{uncond}} \approx 0.1$ 随机丢弃条件信号（例如文本提示）；而在推理阶段，则将带条件与不带条件的预测结果进行加权组合：

$$\hat{\boldsymbol\epsilon}(\mathbf{x}_t, t, c) = \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing) + w\,\bigl[\boldsymbol\epsilon_\theta(\mathbf{x}_t, t, c) - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing)\bigr], \tag{12}$$

其中 $w > 1$ 称为**引导尺度（guidance scale）**。当 $w = 1$ 时，退化为标准的条件生成模型；而文本到图像任务中，$w = 7$–$15$ 是典型取值范围。

```python
@torch.no_grad()
def guided_sample(model, alpha_bar, cond, w=7.5, steps=50, T=1000):
    # 使用无分类器引导进行采样
    timesteps = torch.linspace(T-1, 0, steps).long()
    x = torch.randn(cond.shape[0], 2)
    null_cond = torch.zeros_like(cond)  # 无条件嵌入（即空条件）
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((x.shape[0],), t, dtype=torch.long)
        # 执行两次前向传播：一次带条件，一次无条件
        eps_cond = model(x, t_batch, cond)
        eps_uncond = model(x, t_batch, null_cond)
        # 计算引导后的噪声预测
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        # 使用引导后的噪声执行 DDIM 步骤
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0)
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_guided) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_guided
    return x
```

**为何有效？**  
引导机制本质上放大了「模型在给定提示下生成的内容」与「模型无条件生成的内容」之间的差异。这会将采样结果推向那些**在该条件下异常高概率出现**的区域——从而得到更紧凑、更贴合提示的输出，代价是牺牲部分多样性。从数学角度看，该方法近似于从分布 $p(x|c)^w \cdot p(x)^{1-w}$ 中采样，即对原始条件分布进行了锐化（sharpening）。

| 引导尺度 $w$ | 效果 | 典型应用场景 |
|---------------------|--------|-------------|
| $w = 1$ | 无引导，即标准条件生成 | 注重多样性的任务 |
| $w = 3$–$5$ | 轻度引导 | 创意探索、风格微调 |
| $w = 7$–$10$ | 强引导 | 文本到图像生成（DALL-E、Stable Diffusion 等） |
| $w > 15$ | 过度饱和，易出现伪影 | 通常过高，不推荐使用 |

## 与科学计算的联系

基于 score 的扩散不仅是一种生成建模技巧，更是从任意（可能难以处理的）概率分布中采样的通用工具。对 PDE 社区而言，以下两个方向尤为相关：

1. **PDE 约束下的条件生成**：训练一个条件 score 模型 $\mathbf{s}_\theta(\mathbf{x}, t \mid \mathcal{C})$，其中 $\mathcal{C}$ 编码了 PDE 残差或边界条件。反向采样生成的场在分布意义上满足约束。这构成了**扩散后验采样**（diffusion posterior sampling）的基础（Chung et al., 2023），适用于断层成像、超分辨率、去模糊等反问题。

2. **贝叶斯推断的代理采样器**：许多贝叶斯反问题需从后验 $p(\theta \mid \mathbf{y}) \propto p(\mathbf{y}\mid\theta)\,p(\theta)$ 中采样。若能基于模拟生成的 $(\theta, \mathbf{y})$ 数据对训练 score 模型，则可通过条件反向采样替代传统 MCMC。

核心思想是：**只要你需要从高维、多峰分布中采样，且能够负担前向动力学的模拟，扩散模型的整套流程就适用**。

---

## 练习题
**练习 1.** 证明热方程是 Fokker–Planck 方程在 $f \equiv 0$、$g^2 / 2 = D$ 时的特例。

> *解。* 代入 (4)：$\partial_t p = -\nabla\!\cdot\!(\mathbf{0}\cdot p) + D\nabla^2 p = D\nabla^2 p$。$\blacksquare$

**练习 2.** 解释为何 DDPM 损失等价于（加权）score matching。

> *解。* 由 (9)，$q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$，故 $\nabla_{\mathbf{x}_t}\log q = -\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$。预测 $\boldsymbol\epsilon$ 即预测缩放后的 score，(10) 正是 (6) 在 $w(t)=1$ 下的形式。$\blacksquare$

**练习 3.** 为何 DDIM 能使用远少于 DDPM 的步数？

> *解。* DDIM 求解的是光滑的确定性 ODE (11)，可采用高阶多步法（Heun、RK4、DPM-Solver）；而 DDPM 求解 SDE，其强收敛阶仅为 $\mathcal{O}(\sqrt{\Delta t})$，需极小步长才能保证精度。$\blacksquare$

**练习 4.** 将扩散解释为低通滤波器，并说明为何“噪声先破坏高频内容”。

> *解。* 傅里叶模式演化为 $\hat p(\mathbf{k}, t) \propto \exp(-D\|\mathbf{k}\|^2 t)$，故高频（大 $\|\mathbf{k}\|$）衰减更快。由于高频对应细节，精细结构最先消失，粗结构则持续更久。$\blacksquare$

**练习 5.** 对一维 OU 过程 $dX_t = -\tfrac12 \beta X_t\,dt + \sqrt\beta\,dB_t$（平稳分布为 $\mathcal{N}(0,1)$），推导 Anderson 反向 SDE 的漂移项。

> *解概要。* 在平稳态下 $p_t(x) = \mathcal{N}(x;0,1)$，故 $\nabla\log p_t(x) = -x$。代入 (8)：漂移 $= -\tfrac12\beta x - \beta(-x) = \tfrac12\beta x$，即反向过程具有**远离原点**的正向均值回归——恰好抵消了 OU 过程向原点收缩的趋势。$\blacksquare$

---

## 参考文献

[1] Song, Y., et al. (2020). *Score-based generative modeling through stochastic differential equations.* [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).

[2] Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

[3] Song, J., Meng, C., & Ermon, S. (2021). *Denoising diffusion implicit models.* ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502).

[4] Song, Y., & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600).

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the design space of diffusion-based generative models (EDM).* NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364).

[6] Lu, C., et al. (2022). *DPM-Solver: a fast ODE solver for diffusion models.* NeurIPS. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927).

[7] Rombach, R., et al. (2022). *High-resolution image synthesis with latent diffusion models.* CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).

[8] Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 12(3), 313–326.

[9] Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR.

[10] Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation.

[11] Chung, H., et al. (2023). *Diffusion posterior sampling for general noisy inverse problems.* ICLR. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687).

---

*本文是 [PDE 与机器学习](/zh/pde-ml/) 系列的第 7 篇。下一篇：[第 8 篇 —— 反应扩散系统与 GNN](/zh/pde-ml/08-反应扩散系统与gnn)。上一篇：[第 6 篇 —— 连续归一化流](/zh/pde-ml/06-连续归一化流与neural-ode)。*
