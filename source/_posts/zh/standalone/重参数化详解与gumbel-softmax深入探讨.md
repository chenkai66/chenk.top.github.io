---
title: "重参数化技巧与 Gumbel-Softmax 详解"
date: 2025-01-08 09:00:00
tags:
  - ML
  - Deep Learning
  - Generative Models
categories: Algorithm
lang: zh-CN
mathjax: true
description: "讲清楚连续重参数化与 Gumbel-Softmax 的推导、直觉与实现：为什么梯度能穿过采样节点，温度参数如何权衡偏差-方差，以及离散变量端到端训练的常见坑。"
disableNunjucks: true
---

一旦模型里出现"采样"，训练立刻就会撞上一个硬问题：**梯度怎么穿过随机节点？**

重参数化（reparameterization）给出的答案非常直接——把 $z\sim p_\theta(z)$ 改写成 $z=g_\theta(\epsilon)$，把随机性隔离到与参数无关的噪声 $\epsilon$ 里，于是反向传播可以顺着 $g_\theta$ 走下去。麻烦在于离散变量：$\arg\max$ 一类操作不可导，梯度会断掉。**Gumbel-Softmax**（也叫 Concrete 分布）用"带温度的 softmax + Gumbel 噪声"把离散采样变成可微近似，让你在保留离散结构的同时仍能端到端训练。

本文把推导、直觉、实现细节、温度参数下的偏差-方差权衡，以及训练里最常见的坑都讲清楚。

## 你将学到

- 为什么梯度无法穿过 $z\sim\mathcal N(\mu,\sigma^2)$，而 $z=\mu+\sigma\epsilon$ 就可以
- Gumbel 分布从何而来，以及"加 Gumbel 噪声后取 argmax = 从 softmax 中采样"为什么严格成立
- Gumbel-Softmax 的温度 $\tau$ 如何在偏差与方差之间权衡，以及退火策略
- Straight-Through 估计器（ST-GS）：前向硬采样、反向用软梯度
- 与 REINFORCE / Score Function 估计器对比：方差差几个数量级
- VAE 与离散 VAE 的完整 PyTorch 实现要点，以及训练时的常见坑

## 前置知识

- 期望、概率密度、变量替换公式
- PyTorch 基本用法、自动微分链式法则
- 变分自编码器 VAE 的基本概念（可参考站内 [《变分自编码器 (VAE)：从直觉到实现与调试》](../变分自编码器-vae-详解/)）

---

# 一、为什么需要重参数化？

考虑一个非常一般的训练目标：在某个分布 $q_\theta(z)$ 上对函数 $f(z)$ 求期望，并对参数 $\theta$ 求导：

$$
\mathcal L(\theta) \;=\; \mathbb E_{z\sim q_\theta(z)}\,[\,f(z)\,].
$$

VAE 里 $f$ 是重建对数似然，强化学习里 $f$ 是回报，离散结构学习里 $f$ 是下游模型的损失。**问题在于**：$\theta$ 同时藏在被积函数和分布里，朴素地"先采一个 $z$，再算 $f(z)$，再 backward"——计算图在采样那一步就断了。

直接看一段 PyTorch：

```python
# 错误写法 1：直接采样
mu, logvar = encoder(x)
sigma = torch.exp(0.5 * logvar)
z = torch.normal(mu, sigma)        # <- 采样节点不可微
recon = decoder(z)
loss = (recon - x).pow(2).sum()
loss.backward()                    # mu, sigma 拿不到任何梯度
```

`torch.normal(mu, sigma)` 是一个**随机操作**，它的输出 $z$ 与输入 $\mu,\sigma$ 之间没有可微关系——你换一个 $\mu$，同一个 $z$ 不会"平滑地变"，而是整个分布换了一个，梯度无从定义。

> **核心问题**：训练过程要求 $\nabla_\theta \mathbb E_{q_\theta}[f(z)]$，但采样这一步切断了反向传播。

可行的解法有两类：

1. **Score Function / REINFORCE**：用对数似然的导数把期望写成 $\mathbb E[f(z)\nabla_\theta\log q_\theta(z)]$。通用、不要求 $z$ 可微，但**方差极大**。
2. **重参数化**：把随机性从参数里"挪出去"，换成与参数无关的噪声，让计算图保持可微。**方差小、可端到端训练，但要求分布形式可重参数化**。

# 二、连续分布的重参数化

## 2.1 数学表达

重参数化的基本思想：用一个**确定性、可微**的函数 $g_\theta$ 把简单分布的样本 $\epsilon\sim p(\epsilon)$ 变成目标分布的样本：

$$
z \;=\; g_\theta(\epsilon),\qquad \epsilon \sim p(\epsilon),
$$

其中 $p(\epsilon)$ 是一个**与 $\theta$ 无关**的"基准分布"（如 $\mathcal N(0,I)$、$\mathrm{Uniform}(0,1)$）。代回目标：

$$
\mathcal L(\theta) \;=\; \mathbb E_{\epsilon\sim p(\epsilon)}\,[\,f(g_\theta(\epsilon))\,].
$$

期望算子里**没有 $\theta$ 了**，求导可以直接进期望里：

$$
\nabla_\theta\,\mathcal L(\theta) \;=\; \mathbb E_{\epsilon\sim p(\epsilon)}\,[\,\nabla_\theta\, f(g_\theta(\epsilon))\,].
$$

蒙特卡洛估计就是采一个或几个 $\epsilon$，对 $f(g_\theta(\epsilon))$ 反向传播即可。

![Reparameterization trick](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E8%AF%A6%E8%A7%A3%E4%B8%8Egumbel-softmax%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%AE%A8/fig1_reparam_trick.png)

> **图示对比**：左图 $z$ 从 $\mathcal N(\mu,\sigma^2)$ 直接采样，是计算图里的随机节点，梯度无法穿过；右图 $z=\mu+\sigma\epsilon$ 是关于 $\mu,\sigma$ 的确定性函数，梯度顺利沿绿色路径回到 encoder。

## 2.2 正态分布的重参数化

最常见的例子是 $z\sim\mathcal N(\mu,\sigma^2)$。重参数化为：

$$
z \;=\; \mu \;+\; \sigma \odot \epsilon,\qquad \epsilon \sim \mathcal N(0,I).
$$

可以从两侧验证：(i) $\mathbb E[z]=\mu$，$\mathrm{Var}(z)=\sigma^2$；(ii) 由于线性变换保持正态性，$z$ 的边际分布严格是 $\mathcal N(\mu,\sigma^2)$。**关键是**：$\frac{\partial z}{\partial\mu}=1$、$\frac{\partial z}{\partial\sigma}=\epsilon$，都是常数级表达式，梯度通畅。

PyTorch 实现里我们一般直接学 $\log\sigma^2$（保数值稳定，避免 $\sigma$ 为负）：

```python
def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """z = mu + sigma * eps,  eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)        # sigma >= 0 by construction
    eps = torch.randn_like(std)          # 唯一的随机源，与 mu/logvar 无关
    return mu + std * eps
```

## 2.3 在 VAE 中的应用

VAE 优化证据下界（ELBO）：

$$
\mathcal L_{\text{ELBO}}(\theta,\phi;x)
\;=\; \mathbb E_{q_\phi(z|x)}\!\bigl[\log p_\theta(x|z)\bigr]
\;-\;\mathrm{KL}\!\bigl(q_\phi(z|x)\,\Vert\,p(z)\bigr).
$$

第一项的期望是关于 $q_\phi(z|x)$ 的，参数 $\phi$ 在分布里。**重参数化就是让这一项可微的关键**：写成 $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$ 之后，

$$
\nabla_\phi\,\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
\;=\;\mathbb E_{\epsilon\sim\mathcal N(0,I)}\!\bigl[\,\nabla_\phi\log p_\theta(x\mid \mu_\phi+\sigma_\phi\epsilon)\,\bigr].
$$

对 $\phi$ 的梯度可以顺着 decoder 一路传回 encoder。第二项 KL 在 $q_\phi=\mathcal N(\mu,\sigma^2)$、$p=\mathcal N(0,I)$ 时有闭式：

$$
\mathrm{KL}=\tfrac12\sum_j\!\bigl(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\bigr),
$$

不需要采样。

## 2.4 哪些连续分布"天然可重参数化"？

凡是能写成"位置-尺度族"或"基础噪声 + 可微变换"的都可以：

| 分布 | 重参数化形式 |
|------|--------------|
| $\mathcal N(\mu,\sigma^2)$ | $z=\mu+\sigma\epsilon$, $\epsilon\sim\mathcal N(0,1)$ |
| $\mathrm{Logistic}(\mu,s)$ | $z=\mu+s\,\log\frac{u}{1-u}$, $u\sim\mathrm U(0,1)$ |
| $\mathrm{Laplace}(\mu,b)$ | $z=\mu-b\,\mathrm{sign}(u)\log(1-2|u|)$, $u\sim\mathrm U(-\tfrac12,\tfrac12)$ |
| $\mathrm{Exp}(\lambda)$ | $z=-\tfrac1\lambda\log(1-u)$, $u\sim\mathrm U(0,1)$ |
| $\mathrm{Gumbel}(\mu,\beta)$ | $z=\mu-\beta\log(-\log u)$, $u\sim\mathrm U(0,1)$ |

**反例**：Gamma、Beta、Dirichlet、Student-t 这些没有简单的位置-尺度形式，需要"隐式重参数化（implicit reparameterization gradients）"或路径求导技巧——这是 NeurIPS 2018 Figurnov 等人的工作（详见 §7）。

# 三、离散分布的重参数化：Gumbel-Max 技巧

## 3.1 难点

设 $K$ 类的分类分布 $\mathrm{Cat}(\pi_1,\dots,\pi_K)$，其中 $\pi_i=\mathrm{softmax}(\alpha)_i=\frac{\exp(\alpha_i)}{\sum_j\exp(\alpha_j)}$。直接的"采样"是：先算概率 $\pi$，再做多项式采样得到一个类别索引 $k$。这一步**完全不可导**——它的输出是离散的 one-hot，没有平滑变化的概念。

朴素估计器只能依赖 REINFORCE：

$$
\nabla_\alpha\,\mathbb E_{k\sim\mathrm{Cat}(\pi)}[f(k)]
\;=\;\mathbb E_{k}\bigl[f(k)\,\nabla_\alpha\log\pi_k\bigr],
$$

通用但方差大到没法稳定训练。能不能像正态分布那样，把"采样"分解成"噪声 + 确定性变换"？答案就是 Gumbel-Max 技巧。

## 3.2 Gumbel 分布速览

标准 Gumbel 分布 $\mathrm{Gumbel}(0,1)$ 的 CDF/PDF：

$$
F(g)=\exp(-e^{-g}),\qquad f(g)=\exp\!\bigl(-(g+e^{-g})\bigr).
$$

它的众数在 $0$，均值在 $\gamma\approx 0.5772$（欧拉-马歇罗尼常数）。重要的是它是**极值分布**——独立同分布样本的最大值（在合适的标准化下）服从 Gumbel。Gumbel-Max 正是利用这一点。

利用逆 CDF 采样很方便：

$$
u\sim\mathrm U(0,1) \;\Rightarrow\; g=-\log(-\log u)\sim\mathrm{Gumbel}(0,1).
$$

![Gumbel distribution PDF/CDF/empirical](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E8%AF%A6%E8%A7%A3%E4%B8%8Egumbel-softmax%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%AE%A8/fig2_gumbel_pdf.png)

> **左**：Gumbel(0,1) 的 PDF；**中**：通过 inverse-CDF 采样的几何示意；**右**：用 $-\log(-\log u)$ 生成 2 万样本，直方图与解析 PDF 完美吻合。

## 3.3 Gumbel-Max 技巧

**核心命题**：设 $g_1,\dots,g_K\overset{iid}{\sim}\mathrm{Gumbel}(0,1)$，令

$$
k^\star \;=\; \arg\max_i\,(\alpha_i + g_i),
$$

则 $k^\star\sim\mathrm{Cat}(\mathrm{softmax}(\alpha))$。也就是说，**给 logits 加 Gumbel 噪声再取 argmax，等价于从 softmax 分布中采样**。

### 推导（以采到类别 1 为例）

$\Pr[k^\star=1] = \Pr\bigl[\,\forall i\neq 1:\;\alpha_1+g_1>\alpha_i+g_i\,\bigr] = \Pr\bigl[\,\forall i\neq 1:\;g_i<\alpha_1-\alpha_i+g_1\,\bigr]$.

给定 $g_1=t$：

$$
\Pr[k^\star=1\mid g_1=t]=\prod_{i\neq 1}F(\alpha_1-\alpha_i+t)
=\prod_{i\neq 1}\exp\!\bigl(-e^{-(\alpha_1-\alpha_i+t)}\bigr)
=\exp\!\Bigl(-e^{-t}\!\!\sum_{i\neq 1}e^{\alpha_i-\alpha_1}\Bigr).
$$

记 $S=\sum_{i\neq 1}e^{\alpha_i-\alpha_1}$，对 $g_1$ 取期望：

$$
\Pr[k^\star=1]=\int e^{-(t+e^{-t})}\cdot\exp(-S e^{-t})\,dt
=\int e^{-t}\exp\!\bigl(-(1+S)e^{-t}\bigr)\,dt.
$$

令 $u=(1+S)e^{-t}$，$du=-(1+S)e^{-t}dt$，积分化为 $\frac{1}{1+S}\int_0^\infty e^{-u}du=\frac{1}{1+S}$。

而 $1+S=1+\sum_{i\neq 1}e^{\alpha_i-\alpha_1}=e^{-\alpha_1}\sum_i e^{\alpha_i}$，所以

$$
\Pr[k^\star=1]=\frac{e^{\alpha_1}}{\sum_i e^{\alpha_i}}=\mathrm{softmax}(\alpha)_1. \quad\blacksquare
$$

![Gumbel-Max trick](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E8%AF%A6%E8%A7%A3%E4%B8%8Egumbel-softmax%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%AE%A8/fig3_gumbel_max_trick.png)

> **左**：目标 categorical 分布；**中**：单次实验，加 Gumbel 噪声后的 argmax 选中 c2；**右**：8000 次重复采样的经验频率与目标 softmax 高度吻合，验证了 Gumbel-Max 在期望意义下严格等价。

### 为什么这一招有用？

- **采样高效**：只需 $K$ 个独立 Uniform 样本，一次 argmax，没有显式归一化、没有累积分布查找。
- **数值稳定**：在 logits 上做加法后取 argmax，对常数平移免疫——softmax 中的 $\max$ 减法稳定性同理可得。
- **更重要的**：这个表达把"采样的随机性"全部装进 $g$ 里，logits $\alpha$ 出现在确定性的 $\alpha+g$ 上——离"可微"只剩最后一步：把 $\arg\max$ 换成可微近似。

# 四、Gumbel-Softmax：把 argmax 软化

## 4.1 定义

把 $\arg\max$ 替换为带温度 $\tau$ 的 softmax，得到 **Gumbel-Softmax / Concrete** 分布的样本：

$$
y_i \;=\; \frac{\exp\!\bigl((\alpha_i + g_i)/\tau\bigr)}{\sum_{j=1}^K\exp\!\bigl((\alpha_j + g_j)/\tau\bigr)},
\qquad g_i\overset{iid}{\sim}\mathrm{Gumbel}(0,1).
$$

其中 $y\in\Delta^{K-1}$（$K-1$ 维概率单纯形），是一个连续向量。两端极限给出直观理解：

- $\tau\to 0^+$：softmax 退化为 argmax 的指示向量，$y$ 趋于 one-hot，**接近真离散采样**；
- $\tau\to\infty$：$(\alpha_i+g_i)/\tau\to 0$，$y$ 趋于均匀向量 $(1/K,\dots,1/K)$。

由于 $g$ 与 $\alpha$ 无关、并且 softmax 是处处可微的，$y$ 关于 $\alpha$（也就是关于上游网络参数 $\theta$）**可微**。这就是 Gumbel-Softmax 的核心：用一个连续可微的 $y$ 当作离散 one-hot 的"代理"。

## 4.2 温度的偏差-方差权衡

![Gumbel-Softmax temperature effect](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E8%AF%A6%E8%A7%A3%E4%B8%8Egumbel-softmax%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%AE%A8/fig4_gumbel_softmax_temp.png)

> **四个温度下的样本**：每个面板里浅蓝柱是目标 softmax，橙色折线是 5 次独立的 Gumbel-Softmax 采样。
>
> - $\tau=5$：极平滑、几乎均匀，**梯度方差小**但**严重偏离**真实离散采样；
> - $\tau=1$：折中；
> - $\tau=0.5$：开始接近 one-hot；
> - $\tau=0.1$：几乎是 one-hot（看起来像柱状），**偏差小**但**方差大**（且 softmax 内部数值容易上下溢）。

可以从估计器层面写出这件事：用 Gumbel-Softmax 做 $\nabla_\alpha\,\mathbb E[f(z)]$ 的估计时，温度 $\tau$ 越小，$y$ 越接近真实 one-hot $z$，偏差越小；但 softmax 在 logit 差别大时的雅可比矩阵 $\partial y/\partial\alpha$ 元素呈 $1/\tau$ 量级，方差按 $\tau^{-2}$ 放大。

**实践退火（annealing）建议**：

- 起始 $\tau=1.0\sim 2.0$，end $\tau=0.1\sim 0.5$；
- 指数退火 $\tau_t=\max(\tau_{\min},\tau_0\,e^{-rt})$，每 $\sim 1000$ 步衰减一次；
- **不要**直接训到 $\tau\to 0$——softmax 数值会爆掉，且梯度方差会主导；让模型早期"看清楚"分布形状，后期再"硬化"。

## 4.3 Straight-Through Gumbel-Softmax (ST-GS)

很多任务（例如 hard attention、离散 token 选择、稀疏 routing）**前向需要严格 one-hot**——比如下游是一个 lookup table，必须是整数索引。这时用 **Straight-Through 估计器**：

$$
\boxed{
\;\;y_{\text{hard}}=\mathrm{onehot}(\arg\max_i y_i),
\qquad
\tilde y \;=\; y_{\text{hard}}\;-\;\mathrm{stop\_grad}(y_{\text{soft}})\;+\;y_{\text{soft}}.\;\;
}
$$

在前向，$\tilde y=y_{\text{hard}}$（严格 one-hot）；在反向，$\partial\tilde y/\partial\alpha=\partial y_{\text{soft}}/\partial\alpha$（用软的 Jacobian）。本质是**一种有偏的、低方差的近似**：用 hard 样本去算损失，用 soft 样本去算梯度。

PyTorch 一行就能写：

```python
def gumbel_softmax(logits: Tensor, tau: float = 1.0,
                   hard: bool = False) -> Tensor:
    """y = softmax((logits + g) / tau); 可选 ST-hard。"""
    # 1) 采 Gumbel 噪声，clamp 防止 log(0)
    u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
    g = -torch.log(-torch.log(u))
    # 2) 软样本
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    if not hard:
        return y_soft
    # 3) Straight-Through:前向硬,反向软
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard - y_soft.detach() + y_soft
```

PyTorch 官方有 `torch.nn.functional.gumbel_softmax(logits, tau, hard)`，实现等价；自己写一遍是为了理解每一步。

# 五、与 REINFORCE 的对比：方差差几个数量级

## 5.1 Score Function / REINFORCE 估计器

通用形式：

$$
\nabla_\theta\,\mathbb E_{z\sim q_\theta}[f(z)]
=\mathbb E_{z\sim q_\theta}\!\bigl[f(z)\,\nabla_\theta\log q_\theta(z)\bigr].
$$

它的最大优点是**完全不要求** $z$ 与 $\theta$ 的可微关系——离散、流程控制、调用外部黑盒都行。代价是**方差非常大**：直觉上，$f(z)$ 的整个数值都进了梯度，没有可微路径帮忙抵消符号。

降方差的常见手段：

- **Baseline / Control variate**：把 $f(z)-b$ 代入，$b$ 是与 $z$ 无关的基线（如运行平均）；
- **Rao-Blackwellization、antithetic、importance sampling**……

但即使各种 trick 上身，REINFORCE 的方差通常仍比重参数化高 $1\sim 3$ 个数量级。

## 5.2 实测对比

我们在合成 8 类 categorical 上估计 $\nabla_{\alpha_0}\,\mathbb E_z[r^\top z]$（其中 $r$ 是固定 reward 向量）。每条曲线对 200 次重复试验取梯度方差：

![Variance: REINFORCE vs Gumbel-Softmax](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E8%AF%A6%E8%A7%A3%E4%B8%8Egumbel-softmax%E6%B7%B1%E5%85%A5%E6%8E%A2%E8%AE%A8/fig5_discrete_pipeline.png)

> **左侧管线**：logits → +Gumbel 噪声 → /τ → softmax → $y_{\text{soft}}$；可选 argmax → $y_{\text{hard}}$，结合 STE，即可前向硬、反向软。**右侧曲线**：横轴是每次梯度估计使用的样本数，纵轴是估计的方差（log-log）。Gumbel-Softmax（$\tau=0.5$）始终低 REINFORCE 一个数量级以上，且两者方差都按 $1/n$ 下降（斜率 $-1$）。

这就是为什么有了重参数化，离散变量的端到端训练才真正可用。

# 六、完整 PyTorch 实现：连续 VAE 与离散 VAE

## 6.1 连续 VAE（重参数化）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim: int = 784, h_dim: int = 400,
                 z_dim: int = 20):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_lv = nn.Linear(h_dim, z_dim)
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(logits_x, x, mu, logvar):
    # 重建损失:用 logits + BCE 比 sigmoid + BCE 数值更稳
    recon = F.binary_cross_entropy_with_logits(
        logits_x, x, reduction="sum")
    # 闭式 KL: KL(N(mu,sigma^2) || N(0,I))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```

**实现要点**：

- 用 `binary_cross_entropy_with_logits` 而不是 `BCELoss + sigmoid`，避免饱和区数值不稳；
- KL 项闭式而非采样，减少梯度方差；
- 可选 KL 退火（$\beta$-VAE 的 $\beta$ 从 0 慢慢升到 1）以缓解后验坍塌。

## 6.2 离散隐变量 VAE（Categorical + Gumbel-Softmax）

```python
class CategoricalVAE(nn.Module):
    def __init__(self, x_dim=784, h_dim=400, n_cat=10, n_dim=20):
        """n_cat 个独立 categorical, 每个 K = n_dim 类。"""
        super().__init__()
        self.n_cat, self.n_dim = n_cat, n_dim
        self.enc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, n_cat * n_dim))
        self.dec = nn.Sequential(nn.Linear(n_cat * n_dim, h_dim),
                                 nn.ReLU(), nn.Linear(h_dim, x_dim))

    def forward(self, x, tau: float, hard: bool = False):
        logits = self.enc(x).view(-1, self.n_cat, self.n_dim)
        # 在最后一维(类别维)做 Gumbel-Softmax
        z = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        z_flat = z.view(-1, self.n_cat * self.n_dim)
        return self.dec(z_flat), logits


def cat_vae_loss(logits_x, x, q_logits):
    """KL 用解析形式: KL(q || Uniform(K)) = log K - H(q)。"""
    recon = F.binary_cross_entropy_with_logits(
        logits_x, x, reduction="sum")
    K = q_logits.size(-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    q = log_q.exp()
    kl = (q * (log_q + torch.log(torch.tensor(float(K))))).sum()
    return recon + kl


# 训练循环里温度退火:
# tau = max(tau_min, tau0 * exp(-r * step))
```

**实现要点**：

- 把每个隐变量看作 $K$-类分类，logits 形状 `[B, n_cat, K]`，**最后一维** softmax；
- KL 用 categorical 闭式 $\mathrm{KL}(q\|\mathrm{Uniform})=\log K - H(q)$，不用采样；
- `hard=True` 走 ST-GS，能保证下游真离散；`hard=False` 走纯 soft，梯度更稳；
- 退火：`tau0=1.0`，`tau_min=0.5`，`r=1e-5`（每 1000 步检查一次）。

# 七、训练里的常见坑

| 现象 | 可能原因 | 修复 |
|------|----------|------|
| **NaN loss** | $\tau$ 退到太小，softmax 上下溢；或 $\log(-\log u)$ 中 $u=0$ | clamp $u\in[\epsilon, 1-\epsilon]$；$\tau_{\min}\ge 0.1$ |
| **梯度方差炸** | 一次只采 1 个样本 + $\tau$ 过小 | 增加每 batch 的 MC 样本数；$\tau$ 退火节奏放慢 |
| **离散结构不"硬"** | 一直用 soft 输出做评估 | 推断时 `hard=True` 或 `argmax`；训练后期切 ST-GS |
| **后验坍塌**（连续 VAE） | KL 一开始就太强 | $\beta$ 从 0 退到 1；用 free-bits |
| **离散 VAE 不学结构** | 解码器太强、KL 太弱 | 弱化 decoder 容量；降 KL 权重再升回来 |
| **梯度对 logit 不敏感** | $\tau$ 过大，输出趋均匀 | 降 $\tau$ 或加大学习率 |
| **forward/backward 不一致** | 用了 ST-GS 却没把 `.detach()` 写对 | 复查 `y_hard - y_soft.detach() + y_soft` 这一行 |

# 八、最新研究进展（Recent Work）

- **Implicit Reparameterization Gradients**（Figurnov et al., NeurIPS 2018）：用隐函数定理把 Gamma、Beta、Dirichlet、Student-t 等非位置-尺度族变成可重参数化，公式简洁、误差低。
- **Rebar / Relax**（Tucker et al., NeurIPS 2017；Grathwohl et al., ICLR 2018）：把 Gumbel-Softmax 与 REINFORCE 结合，用神经网络学习控制变量，方差更低、估计无偏。
- **Hard Concrete Gates**（Louizos et al., ICLR 2018）：把 Concrete/Gumbel-Softmax 拉伸 + 截断到 $[0,1]$，得到带"严格 0"的可微门控，用于 $L_0$ 正则化与稀疏化。
- **Top-$k$ Gumbel** 与 **Plackett-Luce**：把 Gumbel-Max 推广到"无放回采样 $k$ 个类别"，用于稀疏注意力与 routing。
- **Permutation-equivariant relaxations**（Mena et al., ICLR 2018）：Sinkhorn 算子 + Gumbel 噪声，对**置换矩阵**做可微近似。

# 总结

- 连续场景下，重参数化把"随机变量 $z$"改写为"噪声 $\epsilon$ + 可微变换 $g_\theta$"，让梯度顺着确定性路径回到参数；这是 VAE 等深度生成模型可以用 SGD 稳定训练的关键。
- 离散场景下，Gumbel-Max 给出"加 Gumbel 噪声后取 argmax = softmax 采样"的精确等价；Gumbel-Softmax 把 argmax 软化成温度 $\tau$ 的 softmax，让整个采样可微。
- 温度 $\tau$ 是核心超参：小 $\tau$ 偏差小但方差大；大 $\tau$ 反之。退火策略（先大后小）是训练稳定的关键。
- 需要严格离散输出时用 ST-GS：前向 hard、反向 soft，是工程上最常用的折中。
- 与 REINFORCE 相比，重参数化路径估计的梯度方差小 $1\sim 3$ 个数量级，是离散结构能端到端训练的物质基础。

# 参考文献

- E. Jang, S. Gu, B. Poole. "Categorical Reparameterization with Gumbel-Softmax." *ICLR*, 2017.
- C. Maddison, A. Mnih, Y. W. Teh. "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables." *ICLR*, 2017.
- D. P. Kingma, M. Welling. "Auto-Encoding Variational Bayes." *ICLR*, 2014.
- M. Figurnov, S. Mohamed, A. Mnih. "Implicit Reparameterization Gradients." *NeurIPS*, 2018.
- G. Tucker et al. "REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models." *NeurIPS*, 2017.
- C. Louizos, M. Welling, D. P. Kingma. "Learning Sparse Neural Networks through $L_0$ Regularization." *ICLR*, 2018.
- W. Kool, H. van Hoof, M. Welling. "Stochastic Beams and Where to Find Them: The Gumbel-Top-$k$ Trick for Sampling Sequences Without Replacement." *ICML*, 2019.
