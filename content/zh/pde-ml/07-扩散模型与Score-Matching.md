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

### 1 Fick 定律与扩散方程

热传导、墨水在水中扩散、粒子在浓度梯度下的运动——它们都遵循同一个方程。**Fick 第一定律**指出，通量与浓度梯度成正比（方向相反）：
$$
\mathbf{J} = -D\,\nabla u,
$$
其中 $D > 0$ 是扩散系数。结合质量守恒 $\partial_t u + \nabla\!\cdot\!\mathbf{J} = 0$，可得**热方程**（又称扩散方程）：
$$
\frac{\partial u}{\partial t} = D\,\nabla^2 u. \tag{1}
$$
拉普拉斯算子衡量了 $u$ 的局部“曲率”：在凹陷处（热点），$\nabla^2 u < 0$，$u$ 下降；在凸起处（冷点），$u$ 上升。最终系统趋于均匀状态。

### 2 高斯核：基本解

对于点源初始条件 $u(\mathbf{x},0) = \delta(\mathbf{x})$，方程 (1) 的解是**热核**：
$$
G(\mathbf{x}, t) = \frac{1}{(4\pi D t)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}\|^2}{4Dt}\right). \tag{2}
$$
这是一个方差 $\sigma_t^2 = 2Dt$ 随时间线性增长的高斯分布。对任意初始分布 $u_0$，解即为与该核的卷积：
$$
u(\mathbf{x}, t) = (G_t * u_0)(\mathbf{x}).
$$
**扩散 = 用不断扩大的高斯核进行模糊**。这正是扩散模型前向加噪过程的核心思想。

### 3 傅里叶视角：扩散作为低通滤波器

在傅里叶空间中，$\widehat{\nabla^2 u}(\mathbf{k}) = -\|\mathbf{k}\|^2\,\hat u(\mathbf{k})$ 将 (1) 转化为每个频率模式的常微分方程：
$$
\hat u(\mathbf{k}, t) = \hat u_0(\mathbf{k})\,e^{-D\|\mathbf{k}\|^2 t}.
$$
高频成分（大 $\|\mathbf{k}\|$）比低频成分指数级更快地衰减。**扩散本质上是一个低通滤波器**——精细结构最先消失，粗粒度结构最后留存。因此，反向的去噪过程必须重建高频信息；而这正是 score 网络所承担的任务。

![前向扩散把结构化数据变成各向同性的高斯噪声；下排展示边际密度 0 按 Fokker–Planck 方程预测收敛到 1。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig1_forward_diffusion.png)
*前向扩散将结构化数据逐步转化为各向同性高斯噪声；底部一行展示了边际密度 $p_t$ 如何按 Fokker–Planck 方程预测收敛至 $\mathcal{N}(0, I)$。*

---

## SDE 与 Fokker–Planck 方程

热方程描述的是密度的**确定性**演化。但若想刻画单个样本路径——这正是扩散模型实际生成的对象——我们需要引入随机微分方程（SDE）。

### 1 布朗运动与 Itô SDE

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

### 2 Fokker–Planck 方程

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

### 3 Kolmogorov 后向方程

对于终端收益 $g(\mathbf{X}_T)$，条件期望 $u(s, \mathbf{x}) = \mathbb{E}[g(\mathbf{X}_T)\,|\,\mathbf{X}_s = \mathbf{x}]$ 满足**后向方程**：
$$
\partial_s u + f\!\cdot\!\nabla u + \tfrac{1}{2}g^2 \nabla^2 u = 0,
$$
终端条件为 $u(T, \mathbf{x}) = g(\mathbf{x})$。前向方程推动密度向前演化，后向方程则将期望向后传播。二者共同构成 Feynman–Kac 对应关系——而前向 SDE 的时间反演，正是我们用来**生成样本**的关键工具。

---

## 基于 Score 的生成模型

![偏微分方程与机器学习（七）：扩散模型与Score Matching — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/illustration_2.png)

### 1 Score 函数

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

### 2 Score Matching

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

### 3 Langevin 动力学

一旦获得 $\mathbf{s}_\theta$，即可通过 Langevin MCMC 进行采样：
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\,\boldsymbol\eta_k,\qquad \boldsymbol\eta_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{7}
$$
当 $\epsilon \to 0$ 且迭代步数 $k \to \infty$ 时，链收敛至目标分布 $p$。其中确定性项负责“利用”（沿 score 上坡），噪声项负责“探索”（跳出局部极大值）。

### 4 Anderson 的反向时间 SDE

以下是将 score matching 转化为生成模型的基石性结果。**Anderson（1982）** 证明：SDE (3) 的时间反演本身仍是一个 SDE：
$$
\boxed{\;d\mathbf{X}_t = \bigl[\,f(\mathbf{X}_t, t) - g(t)^2\,\nabla\log p_t(\mathbf{X}_t)\,\bigr]\,dt + g(t)\,d\bar{\mathbf{B}}_t,\;} \tag{8}
$$
其中 $\bar{\mathbf{B}}_t$ 是反向时间的布朗运动，$p_t$ 是原过程在时刻 $t$ 的边际密度。**该 SDE 中唯一依赖数据分布的部分就是 $\nabla\log p_t$——这正是 score 网络所学习的目标**。从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，沿时间倒推求解 (8)，即可生成（近似）来自数据分布的样本。

![反向扩散从 0 倒推到 1；score 网络提供反转加噪所需的漂移修正。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig2_reverse_diffusion.png)
*反向扩散从 $t = T$ 回退至 $t = 0$；score 网络提供反转加噪过程所需的漂移修正。*

---

## 从连续理论到 DDPM 与 DDIM

### 1 DDPM：前向过程的闭式解

选定噪声调度 $\{\beta_t\}_{t=1}^T$，定义 $\alpha_t = 1 - \beta_t$ 和 $\bar\alpha_t = \prod_{s=1}^t \alpha_s$。DDPM 的前向过程是一个离散时间马尔可夫链：
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\bigr),
$$
其具有简洁的闭式表达：
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{9}
$$
这正是 VP-SDE 的 Euler–Maruyama 离散化形式。当 $T$ 足够大时，$\bar\alpha_T \to 0$，故 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 2 DDPM 损失 = 加权 DSM

训练一个网络 $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ 来预测所添加的噪声：
$$
\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Bigl[\,\bigl\|\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) - \boldsymbol\epsilon\bigr\|^2\Bigr]. \tag{10}
$$
为何这是 score matching 的伪装？因为由 (9) 可得：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol\epsilon}{\sqrt{1 - \bar\alpha_t}},
$$
因此网络实际上在学习一个缩放后的 score：$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)/\sqrt{1 - \bar\alpha_t}$。(10) 正是 (6) 在权重 $w(t) = 1$ 下的特例。

### 3 DDIM：概率流 ODE

关于 SDE (3) 有一个优美事实：存在一个**确定性** ODE，其在任意时刻 $t$ 的边际分布与原 SDE 完全相同：
$$
\boxed{\;\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \tfrac{1}{2}\,g(t)^2\,\nabla\log p_t(\mathbf{x}).\;} \tag{11}
$$
这就是**概率流 ODE**。其边际分布之所以一致，是因为 (8) 和 (11) 对应的密度 $p_t$ 都满足同一个 Fokker–Planck 方程 (4)。使用高阶 ODE 求解器（如 Heun、RK4、DPM-Solver）沿时间反向求解 (11)，构成了 DDIM 及其后续方法的基础。其优势在于：**确定性**（相同噪声输入 → 相同输出图像）、**支持更大步长**（25–50 步 vs 1000 步）、**严格可逆**（可将图像编码回其潜噪声）。

![DDPM（左）每步注入新噪声，DDIM（右）在同一 score 下走确定性流；ODE 用更少步数抵达模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig4_ddpm_vs_ddim.png)
*DDPM（左）在每一步反向过程中注入新噪声；DDIM（右）则在同一 learned score 下沿确定性轨迹流动，仅用极少步数即可抵达数据模式。*

### 4 统一视角

| 方法 | 过程 | 典型步数 | 是否确定性 | 优势 |
|------|------|---------|------------|------|
| DDPM | 反向 SDE (8) | ~1000 | 否 | 多样性好、训练简单 |
| DDIM | 概率流 ODE (11) | ~25–50 | 是 | 速度快、可精确反演 |
| DPM-Solver | 高阶 ODE | ~10–20 | 是 | 更快、保真度不变 |
| EDM（Karras et al.） | 连续、精细预条件 | ~30 | 可调 | 当前 SOTA 质量 |

### 5 PDE → 扩散模型全景图

将上述内容整合：

![热方程 → Fokker–Planck → 前向 SDE；Anderson 时间反演需要 0，由 score 网络通过 DSM 学习；同一 score 既能驱动 DDPM (SDE) 也能驱动 DDIM (ODE)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig6_pde_diffusion_bridge.png)
*热方程 → Fokker–Planck 方程 → 前向 SDE；Anderson 时间反演需要 $\nabla\log p_t$，由 score 网络通过 DSM 学习；同一 score 既可驱动 DDPM（SDE）也可驱动 DDIM（ODE）。*

---

## 潜空间扩散：一张图理解 Stable Diffusion

在像素空间对 $512 \times 512$ 图像进行扩散计算开销巨大：每次 U-Net 前向需处理约 $8 \times 10^5$ 个浮点数。**潜空间扩散**（Rombach et al., 2022）首先训练一个类 VAE 的自编码器 $(\mathcal{E}, \mathcal{D})$，将图像映射到约 $8\times$ 更小的潜变量 $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$，然后在整个扩散过程中**仅在潜空间操作**。最终通过一次前向传递完成解码：$\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$。

各类条件信息（文本、类别标签、深度图、ControlNet 姿态等）通过交叉注意力机制注入 U-Net。

![Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-扩散模型与Score-Matching/fig7_latent_diffusion.png)
*Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。*

计算开销节省约为 $f^{2d}$，其中 $f$ 为空间下采样因子（通常为 8），$d=2$，即约 **64 倍**。正是这一架构创新，使扩散模型得以走向消费级应用。

---

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
