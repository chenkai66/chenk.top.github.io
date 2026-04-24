---
title: "PDE与机器学习（七）：扩散模型与Score Matching"
date: 2024-10-31 09:00:00
tags:
  - 扩散模型
  - Score Matching
  - PDE
  - 生成模型
  - SDE
categories:
  - PDE与机器学习
series:
  name: "PDE与机器学习"
  part: 7
  total: 8
lang: zh-CN
mathjax: true
description: "从 PDE 视角统一理解扩散模型：热方程、Fokker-Planck、score matching、DDPM/DDIM、Latent Diffusion，配可视化。"
disableNunjucks: true
---

## 本文你会学到

2020 年以来，**扩散模型**（Diffusion Models）已经成为生成式 AI 的主流：DALL·E 2、Stable Diffusion、Sora 都是它的变种。在它惊人的工程效果背后，是一套异常清爽的数学结构——**扩散模型本质上就是偏微分方程（PDE）的数值求解器**：

- 向数据加高斯噪声，等价于沿时间正向求解 **Fokker–Planck 方程**；
- 训练去噪网络，等价于学习 **score 函数** $\nabla\log p_t$；
- DDPM 是反向 SDE 的离散化；DDIM 是对应的**概率流 ODE**；
- Stable Diffusion 把同一套机器搬到低维潜空间执行。

**本文目录**

1. 热方程与高斯核——扩散的基本数学；
2. SDE 与 Fokker–Planck 方程——概率密度的演化；
3. Score 函数、Score Matching（DSM/SSM）、Langevin 动力学；
4. DDPM 是离散反向 SDE，DDIM 是概率流 ODE；
5. Latent Diffusion（Stable Diffusion）和它与科学计算的联系。

**前置知识**：多变量微积分、基本概率（高斯、贝叶斯）、神经网络基础。

---

## 1. 热方程与扩散过程

### 1.1 Fick 定律和扩散方程

热在金属棒里传导、墨水在水里扩散、粒子沿浓度梯度运动——它们都遵循同一个方程。**Fick 第一定律**说，通量与浓度梯度成正比、方向相反：
$$
\mathbf{J} = -D\,\nabla u,
$$
其中 $D > 0$ 是扩散系数。结合质量守恒 $\partial_t u + \nabla\!\cdot\!\mathbf{J} = 0$，立刻得到**热方程**：
$$
\frac{\partial u}{\partial t} = D\,\nabla^2 u. \tag{1}
$$
拉普拉斯算子衡量 $u$ 的"局部弯曲程度"：在凹陷的山谷里 $\nabla^2 u > 0$，$u$ 增长；在隆起的山顶 $\nabla^2 u < 0$，$u$ 下降。终态是均匀场。

### 1.2 高斯核：基本解

对点源初值 $u(\mathbf{x},0) = \delta(\mathbf{x})$，方程 (1) 的解是**热核**：
$$
G(\mathbf{x}, t) = \frac{1}{(4\pi D t)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}\|^2}{4Dt}\right). \tag{2}
$$
这是一个方差 $\sigma_t^2 = 2Dt$ 随时间线性增长的高斯。对一般初值 $u_0$，解就是与热核的卷积：
$$
u(\mathbf{x}, t) = (G_t * u_0)(\mathbf{x}).
$$
**扩散 = 用越来越宽的高斯核做模糊**——这正是扩散模型前向加噪的几何图像。

### 1.3 傅里叶视角：扩散是低通滤波

在傅里叶域，$\widehat{\nabla^2 u}(\mathbf{k}) = -\|\mathbf{k}\|^2\,\hat u(\mathbf{k})$，把 (1) 变成每个频率上的常微分方程：
$$
\hat u(\mathbf{k}, t) = \hat u_0(\mathbf{k})\,e^{-D\|\mathbf{k}\|^2 t}.
$$
高频（大 $\|\mathbf{k}\|$）指数级衰减得更快。**扩散是一台低通滤波器**——细节先死，粗结构最后死。反过来，去噪过程必须把高频细节"补"回来；这就是 score 网络的任务。

![前向扩散把结构化数据变成各向同性的高斯噪声；下排展示边际密度 $p_t$ 按 Fokker–Planck 方程预测收敛到 $\mathcal{N}(0, I)$。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig1_forward_diffusion.png)
*前向扩散把结构化数据变成各向同性的高斯噪声；下排展示边际密度 $p_t$ 按 Fokker–Planck 方程预测收敛到 $\mathcal{N}(0, I)$。*

---

## 2. 随机微分方程与 Fokker–Planck 方程

热方程描述的是密度的**确定性**演化。要谈论"单个样本路径"——而这正是扩散模型实际上生成的对象——我们需要随机微分方程。

### 2.1 布朗运动与 Itô SDE

**布朗运动** $\mathbf{B}_t$ 满足 $\mathbf{B}_0 = \mathbf{0}$，独立的高斯增量 $\mathbf{B}_{t+\Delta t} - \mathbf{B}_t \sim \mathcal{N}(\mathbf{0}, \Delta t\,\mathbf{I})$，路径连续但几乎处处不可微。一般 Itô SDE 为：
$$
d\mathbf{X}_t = f(\mathbf{X}_t, t)\,dt + g(t)\,d\mathbf{B}_t, \tag{3}
$$
其中 $f$ 是**漂移**（确定性拉力），$g$ 是**扩散系数**（噪声幅度）。

扩散模型文献里两种主流调度：

| 调度 | 漂移 $f(\mathbf{x}, t)$ | 扩散 $g(t)$ | 平稳分布 |
|------|------------------------|--------------|----------|
| 方差保持 (VP) | $-\tfrac{1}{2}\beta(t)\,\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0},\,\mathbf{I})$ |
| 方差爆炸 (VE) | $0$ | $\sqrt{d\sigma^2/dt}$ | 方差无界发散 |

DDPM 是 VP 的离散化；Song & Ermon (2019) 的 NCSN 是 VE 的离散化。

### 2.2 Fokker–Planck 方程

若 $\mathbf{X}_t$ 满足 (3)，其密度 $p(\mathbf{x}, t)$ 满足**Fokker–Planck 方程**（Kolmogorov 前向方程）：
$$
\boxed{\;\frac{\partial p}{\partial t} \;=\; -\nabla\!\cdot\!\bigl(f\,p\bigr) \;+\; \tfrac{1}{2}\,g^2\,\nabla^2 p\;.\;} \tag{4}
$$

**证明思路。** 对任意光滑测试函数 $\varphi$，由 Itô 公式：
$$
d\varphi(\mathbf{X}_t) = \bigl(f\!\cdot\!\nabla\varphi + \tfrac{1}{2}g^2\nabla^2\varphi\bigr)\,dt + g\,\nabla\varphi\!\cdot\!d\mathbf{B}_t.
$$
取期望消掉鞅项，再写 $\mathbb{E}[\varphi(\mathbf{X}_t)] = \int \varphi\,p\,d\mathbf{x}$，分部积分（利用 $\varphi$ 的任意性）就得到 (4)。$\blacksquare$

**自洽性检查。** 在 (4) 里令 $f \equiv 0$、$g^2/2 = D$，立即退化为热方程 $\partial_t p = D\,\nabla^2 p$。**Fokker–Planck = 热方程 + 漂移项**。

### 2.3 Kolmogorov 后向方程

对终端收益 $g(\mathbf{X}_T)$，条件期望 $u(s, \mathbf{x}) = \mathbb{E}[g(\mathbf{X}_T)\,|\,\mathbf{X}_s = \mathbf{x}]$ 满足**后向方程**：
$$
\partial_s u + f\!\cdot\!\nabla u + \tfrac{1}{2}g^2 \nabla^2 u = 0,
$$
终端条件 $u(T, \mathbf{x}) = g(\mathbf{x})$。前向方程把密度向前推，后向方程把期望向后拉；它们就是 Feynman–Kac 对偶——而**前向 SDE 的时间反演**正是我们用来"生成样本"的 SDE。

---

## 3. 基于 Score 的生成模型

### 3.1 Score 函数

密度 $p$ 的 **score** 定义为：
$$
\mathbf{s}(\mathbf{x}) \;:=\; \nabla_{\mathbf{x}}\,\log p(\mathbf{x}). \tag{5}
$$
三个关键性质：

- **不依赖归一化**：$\nabla \log(p / Z) = \nabla \log p$，永远不需要算配分函数。
- **高斯有闭式**：若 $p = \mathcal{N}(\boldsymbol\mu, \sigma^2 \mathbf{I})$，则 $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol\mu) / \sigma^2$。
- **零均值**：$\mathbb{E}_p[\mathbf{s}(\mathbf{x})] = \mathbf{0}$（在温和正则性下）。

几何上，score 是一个永远指向高密度区域的向量场，在低密度的山谷里模长会变得很大。

![双峰高斯混合的 score 场：箭头沿密度梯度向上，指向两个模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig3_score_field.png)
*双峰高斯混合的 score 场：箭头沿密度梯度向上，指向两个模式。*

### 3.2 Score Matching

由于 $p$ 未知，无法直接最小化 $\mathbb{E}_p\,\|\mathbf{s}_\theta - \nabla\log p\|^2$。有三种可行的代理：

**Implicit Score Matching (ISM, Hyvärinen 2005)。** 分部积分掉未知项：
$$
\mathcal{L}_{\text{ISM}}(\theta) = \mathbb{E}_p\Bigl[\,\tfrac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \mathrm{tr}\bigl(\nabla\mathbf{s}_\theta(\mathbf{x})\bigr)\Bigr].
$$
雅可比的迹在高维下计算昂贵。

**Sliced Score Matching (SSM, Song et al. 2019)。** 在随机方向 $\mathbf{v}$ 上投影：
$$
\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{v}, \mathbf{x}}\Bigl[\,\mathbf{v}^\top \nabla\mathbf{s}_\theta(\mathbf{x})\,\mathbf{v} + \tfrac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2\Bigr].
$$
每样本只需一次 Hessian–向量积。

**Denoising Score Matching (DSM, Vincent 2011)——主力。** 加噪 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol\eta$，$\boldsymbol\eta\sim\mathcal{N}(\mathbf{0},\mathbf{I})$，让网络预测噪声方向：
$$
\boxed{\;\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\Bigl[\bigl\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \tfrac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}\bigr\|^2\Bigr].\;} \tag{6}
$$
Vincent 证明 (6) 与匹配加噪分布 $p_\sigma = p * \mathcal{N}(0, \sigma^2 \mathbf{I})$ 的真实 score 同最优解。当 $\sigma\to 0$ 时 $p_\sigma\to p$。实际训练里我们对 $\sigma$ 退火。

![左：DSM 损失单调下降并趋于平稳。右：学到的 score 在高密度区与真值吻合；中央低密度谷因被噪声平滑而看起来"被磨平"。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig5_score_matching_loss.png)
*左：DSM 损失单调下降并趋于平稳。右：学到的 score 在高密度区与真值吻合；中央低密度谷因被噪声平滑而看起来"被磨平"。*

### 3.3 Langevin 动力学

有了 $\mathbf{s}_\theta$ 就能用 Langevin MCMC 采样：
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\,\boldsymbol\eta_k,\qquad \boldsymbol\eta_k\sim\mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{7}
$$
当 $\epsilon\to 0$、$k\to\infty$ 时收敛到 $p$。确定项是**利用**（沿 score 上山），噪声项是**探索**（避免卡在局部最大）。

### 3.4 Anderson 反向 SDE

下面是把 score matching 变成生成模型的**关键定理**。**Anderson (1982)** 证明，(3) 的时间反演本身仍是一个 SDE：
$$
\boxed{\;d\mathbf{X}_t = \bigl[\,f(\mathbf{X}_t, t) - g(t)^2\,\nabla\log p_t(\mathbf{X}_t)\,\bigr]\,dt + g(t)\,d\bar{\mathbf{B}}_t,\;} \tag{8}
$$
其中 $\bar{\mathbf{B}}_t$ 是反向时间布朗运动，$p_t$ 是 (3) 在时刻 $t$ 的边际。**这个 SDE 里唯一依赖数据分布的是 $\nabla\log p_t$——正好是 score 网络学习的对象**。从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，沿时间倒退求解 (8)，就能从（近似的）数据分布里采样。

![反向扩散从 $t=T$ 倒推到 $t=0$；score 网络提供反转加噪所需的漂移修正。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig2_reverse_diffusion.png)
*反向扩散从 $t=T$ 倒推到 $t=0$；score 网络提供反转加噪所需的漂移修正。*

---

## 4. 从连续理论到 DDPM 与 DDIM

### 4.1 DDPM：前向过程的闭式

选噪声调度 $\{\beta_t\}_{t=1}^T$，定义 $\alpha_t = 1 - \beta_t$、$\bar\alpha_t = \prod_{s=1}^t \alpha_s$。DDPM 的前向 Markov 链：
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\bigr),
$$
有方便的闭式：
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}). \tag{9}
$$
这恰好是 VP-SDE 的 Euler–Maruyama 离散化，所以 $\bar\alpha_T \to 0$、$\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 4.2 DDPM 损失 = 加权 DSM

训练网络 $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ 预测被加进去的噪声：
$$
\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Bigl[\,\bigl\|\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) - \boldsymbol\epsilon\bigr\|^2\Bigr]. \tag{10}
$$
为什么这是 score matching？因为 (9) 立刻给出
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol\epsilon}{\sqrt{1 - \bar\alpha_t}},
$$
所以网络学到的实际是缩放过的 score：$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)/\sqrt{1 - \bar\alpha_t}$，(10) 正是 (6) 在 $w(t)=1$ 下的特例。

### 4.3 DDIM：概率流 ODE

(3) 有一个漂亮性质：存在一个**确定性** ODE，与之共享所有时间的边际：
$$
\boxed{\;\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \tfrac{1}{2}\,g(t)^2\,\nabla\log p_t(\mathbf{x}).\;} \tag{11}
$$
这就是**概率流 ODE**。边际相同的原因是：(8) 与 (11) 关于 $p_t$ 都给出同一个 Fokker–Planck 方程 (4)。沿时间倒推求解 (11)，配合高阶 ODE 求解器（Heun、RK4、DPM-Solver），就是 DDIM 及其后裔的根基。结果：**确定性**（同一噪声→同一图）、**步长大得多**（25–50 步 vs 1000 步）、**严格可逆**（图像可被精确编码回潜噪声）。

![DDPM（左）每步注入新噪声，DDIM（右）在同一 score 下走确定性流；ODE 用更少的步数就抵达模式。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig4_ddpm_vs_ddim.png)
*DDPM（左）每步注入新噪声，DDIM（右）在同一 score 下走确定性流；ODE 用更少的步数就抵达模式。*

### 4.4 统一图景

| 方法 | 过程 | 典型步数 | 是否确定性 | 优势 |
|------|------|---------|------------|------|
| DDPM | 反向 SDE (8) | ~1000 | 否 | 多样性、训练稳定 |
| DDIM | 概率流 ODE (11) | ~25–50 | 是 | 速度、可精确反演 |
| DPM-Solver | 高阶 ODE | ~10–20 | 是 | 进一步加速、保真度不变 |
| EDM (Karras et al.) | 连续、精细预条件 | ~30 | 可调 | 当前 SOTA 质量 |

### 4.5 PDE → 扩散模型 全图

把所有部件拼起来：

![热方程 → Fokker–Planck → 前向 SDE；Anderson 时间反演需要 $\nabla\log p_t$，由 score 网络通过 DSM 学习；同一 score 既能驱动 DDPM (SDE) 也能驱动 DDIM (ODE)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig6_pde_diffusion_bridge.png)
*热方程 → Fokker–Planck → 前向 SDE；Anderson 时间反演需要 $\nabla\log p_t$，由 score 网络通过 DSM 学习；同一 score 既能驱动 DDPM (SDE) 也能驱动 DDIM (ODE)。*

---

## 5. Latent Diffusion：一张图理解 Stable Diffusion

像素空间上跑 $512\times 512$ 的扩散非常昂贵：每次 U-Net 前向都要处理约 $8\times 10^5$ 个浮点数。**Latent Diffusion**（Rombach et al., 2022）先训练一个 VAE 风格的自编码器 $(\mathcal{E}, \mathcal{D})$，把图像映射到约 $8\times$ 更小的潜变量 $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$，然后**整个扩散过程都在潜空间里跑**。最后 $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$ 一次解码完成。

条件信息（文本、类标签、深度图、ControlNet 姿态……）通过**交叉注意力**进入 U-Net。

![Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/07-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8EScore-Matching/fig7_latent_diffusion.png)
*Stable Diffusion = 自编码器 + 潜空间扩散 + 交叉注意力条件输入。*

计算量节省大致是 $f^{2d}$，其中 $f$ 是空间下采样因子（一般 8）、$d=2$，即约 $64\times$。这是把扩散模型推向消费级的关键架构 trick。

---

## 6. 与科学计算的连接

基于 score 的扩散不仅是生成图像的小聪明，更是**从任意（可能无解析形式的）概率分布中采样的工具**。两个对 PDE 社区特别有意思的方向：

1. **PDE 约束的条件生成。** 训练 $\mathbf{s}_\theta(\mathbf{x}, t \mid \mathcal{C})$，其中 $\mathcal{C}$ 编码 PDE 残差或边界数据；反向采样会产出（在分布上）满足约束的场。这是**diffusion posterior sampling**（Chung et al., 2023）的基础：成像、超分、去模糊、断层重建等反问题。

2. **贝叶斯推断的代理采样器。** 许多贝叶斯反问题需要从 $p(\theta\mid\mathbf{y}) \propto p(\mathbf{y}\mid\theta)p(\theta)$ 采样。如果能在 $(\theta, \mathbf{y})$ 模拟对上训练 score 模型，条件反向采样就能替代 MCMC。

一句话：**只要你需要从高维多峰分布里采样、又能负担前向动力学模拟，扩散模型这套配方就适用**。

---

## 7. 练习

**练习 1.** 证明热方程是 Fokker–Planck 在 $f \equiv 0$、$g^2/2 = D$ 下的特例。

> *解。* 代入 (4)：$\partial_t p = -\nabla\!\cdot\!(\mathbf{0}\cdot p) + D\nabla^2 p = D\nabla^2 p$。$\blacksquare$

**练习 2.** 解释为什么 DDPM 损失等价于（加权）score matching。

> *解。* 由 (9)，$q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$，故 $\nabla_{\mathbf{x}_t}\log q = -\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$。预测 $\boldsymbol\epsilon$ 即在预测一个缩放的 score，(10) 就是 $w(t)=1$ 下的 (6)。$\blacksquare$

**练习 3.** 为什么 DDIM 能用比 DDPM 少得多的步数？

> *解。* DDIM 解的是光滑的确定性 ODE (11)，可用高阶多步法（Heun、RK4、DPM-Solver）；DDPM 解的是 SDE，强阶收敛速率为 $\mathcal{O}(\sqrt{\Delta t})$，需要小步长保证精度。$\blacksquare$

**练习 4.** 把扩散解释成低通滤波，并说明为何与"噪声先抹掉高频"一致。

> *解。* 频域里 $\hat p(\mathbf{k}, t) \propto \exp(-D\|\mathbf{k}\|^2 t)$，高 $\|\mathbf{k}\|$ 指数衰减更快。高频对应细节，所以细节最先消失，粗结构最后才被抹平。$\blacksquare$

**练习 5.** 对一维 OU 过程 $dX_t = -\tfrac12\beta X_t\,dt + \sqrt\beta\,dB_t$（平稳分布 $\mathcal{N}(0,1)$）写出 Anderson 反向 SDE 的漂移。

> *解概要。* 平稳态下 $p_t(x) = \mathcal{N}(x;0,1)$，故 $\nabla\log p_t(x) = -x$。代入 (8)：漂移 $= -\tfrac12\beta x - \beta(-x) = +\tfrac12\beta x$——反向时间内变成**远离原点**的均值反演，这正是"撑开"被 OU 收缩到原点的样本所需的力。$\blacksquare$

---

## 参考文献

[1] Song, Y., et al. (2020). *Score-based generative modeling through stochastic differential equations.* [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

[2] Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[3] Song, J., Meng, C., & Ermon, S. (2021). *Denoising diffusion implicit models.* ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

[4] Song, Y., & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the design space of diffusion-based generative models (EDM).* NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)

[6] Lu, C., et al. (2022). *DPM-Solver: a fast ODE solver for diffusion models.* NeurIPS. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)

[7] Rombach, R., et al. (2022). *High-resolution image synthesis with latent diffusion models.* CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

[8] Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 12(3), 313–326.

[9] Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR.

[10] Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation.

[11] Chung, H., et al. (2023). *Diffusion posterior sampling for general noisy inverse problems.* ICLR. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

---

## 系列导航

| 部分 | 主题 |
|------|------|
| [1](/zh/PDE与机器学习-一-物理信息神经网络/) | 物理信息神经网络 |
| [2](/zh/PDE与机器学习-二-神经算子理论/) | 神经算子理论 |
| [3](/zh/PDE与机器学习-三-变分原理与优化/) | 变分原理与优化 |
| [4](/zh/PDE与机器学习-四-变分推断与Fokker-Planck方程/) | 变分推断与Fokker-Planck方程 |
| [5](/zh/PDE与机器学习-五-辛几何与保结构网络/) | 辛几何与保结构网络 |
| [6](/zh/PDE与机器学习-六-连续归一化流与Neural-ODE/) | 连续归一化流与Neural ODE |
| **7** | **扩散模型与Score Matching（本文）** |
| [8](/zh/PDE与机器学习-八-反应扩散系统与GNN/) | 反应扩散系统与GNN |
