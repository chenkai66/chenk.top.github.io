---
title: "概率与统计（二）：随机变量及关键分布"
date: 2024-08-20 09:00:00
tags:
  - Probability
  - Statistics
  - Distributions
  - Random Variables
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "一次严谨的巡礼：涵盖随机变量、概率质量函数（PMF）、概率密度函数（PDF）、累积分布函数（CDF），以及所有在实践中至关重要的分布——Bernoulli、Binomial、Poisson、Gaussian、Exponential、Gamma 和 Beta——含推导、证明与 Python 可视化。"
disableNunjucks: true
series_order: 2
series_total: 8
translationKey: "probability-statistics-2"
---
在上一篇文章中，我们构建了概率论的公理化基础，你或许会觉得花了太多时间讨论集合与子集——事实的确如此。事件与 σ-代数这套机制虽必不可少，却略显枯燥，无法自然地支持均值计算、离散度度量或数据拟合。

连接抽象概率与应用统计的桥梁正是**随机变量（Random Variable）**。一旦为样本空间中的结果赋予数值，整个微积分工具箱——导数、积分、级数——便随之启用，使我们能用一组命名分布来刻画随机性。每种分布都编码了关于数据生成机制的特定假设。

本文系统梳理你在实践中最常遇到的分布，并精确揭示其来源。

---

## 随机变量作为函数

**定义。** 一个**随机变量** $X$ 是从样本空间到实数集的函数：

![离散与连续](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/02-discrete-vs-continuous.png)
$$X: \Omega \to \mathbb{R}$$
且对任意实数 $x$，集合 $\{\omega \in \Omega : X(\omega) \leq x\}$ 是 $\mathcal{F}$ 中的一个事件。

可测性条件（定义的第二部分）确保诸如“$X$ 不超过 3 的概率是多少？”这类问题有明确定义的答案。对于有限或可数的样本空间，该条件自动满足。

**示例。** 同时掷两枚骰子，令 $X$ 表示点数之和。样本空间为 $\Omega = \{(i,j) : 1 \leq i,j \leq 6\}$，共 36 个等可能结果。随机变量 $X(i,j) = i + j$ 将每个有序对映射为 2 到 12 之间的整数。

关键转变在于：我们不再追踪完整结果 $\omega$，而是处理数值 $X(\omega)$。这会损失信息（例如 $X = 7$ 并不能告诉我们是 $(1,6)$ 还是 $(3,4)$），但细节上的损失换来了强大的数学表达力。

## 离散型随机变量

若一个随机变量取值于可数集（有限或可数无限），则称其为**离散型随机变量（Discrete Random Variable）**。

![分布连接](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/02-distribution-connections.png)

### 概率质量函数（PMF）

离散型随机变量 $X$ 的**概率质量函数（Probability Mass Function, PMF）** 定义为：
$$p_X(x) = P(X = x)$$
其中 $x$ 取遍 $X$ 支撑集（support）中的所有值。其性质如下：

1. 对所有 $x$，有 $p_X(x) \geq 0$
2. $\sum_{x} p_X(x) = 1$（对支撑集内所有值求和）

### 累积分布函数（CDF）

任意随机变量（离散或连续）的**累积分布函数（Cumulative Distribution Function, CDF）** 定义为：

![CDF 比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/02-cdf-comparison.png)
$$
F_X(x) = P(X \leq x) = \sum_{t \leq x} p_X(t) \quad \text{(离散情形)}.
$$
CDF 是右连续、非减函数，且满足 $\lim_{x \to -\infty} F(x) = 0$ 与 $\lim_{x \to \infty} F(x) = 1$。

## 关键离散分布

![分位数函数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/02-quantile-function.png)

![概率分布如山地景观：正态分布和指数分布](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/02-probability-distributions-as-mountain-landscapes-normal-expo.jpg)

### Bernoulli 分布

最简单的随机变量：单次试验，仅两种结果。
$$X \sim \text{Bernoulli}(p), \quad p_X(x) = p^x (1-p)^{1-x} \text{ for } x \in \{0, 1\}.$$
- **均值（Mean）：** $E[X] = p$
- **方差（Variance）：** $\text{Var}(X) = p(1-p)$

所有二元结果——抛硬币、点击/未点击、垃圾邮件/非垃圾邮件——均可建模为 Bernoulli 试验。

### Binomial 分布

$n$ 次独立 Bernoulli 试验中成功的次数。
$$X \sim \text{Binomial}(n, p), \quad p_X(k) = \binom{n}{k} p^k (1-p)^{n-k} \text{ for } k = 0, 1, \ldots, n.$$
*推导。* 恰好含 $k$ 次成功的某一特定序列的概率为 $p^k(1-p)^{n-k}$；此类序列总数为 $\binom{n}{k}$。

- **均值：** $E[X] = np$
- **方差：** $\text{Var}(X) = np(1-p)$

*均值证明。* 令 $X = X_1 + X_2 + \cdots + X_n$，其中每个 $X_i \sim \text{Bernoulli}(p)$。由期望的线性性：$E[X] = \sum E[X_i] = np$。$\blacksquare$

### Geometric 分布

首次成功所需的试验次数。
$$X \sim \text{Geometric}(p), \quad p_X(k) = (1-p)^{k-1} p \text{ for } k = 1, 2, 3, \ldots$$
*归一化验证：*
$$\sum_{k=1}^{\infty} (1-p)^{k-1} p = p \sum_{j=0}^{\infty} (1-p)^j = p \cdot \frac{1}{1-(1-p)} = p \cdot \frac{1}{p} = 1. \quad \checkmark$$
- **均值：** $E[X] = 1/p$
- **方差：** $\text{Var}(X) = (1-p)/p^2$Geometric 分布具有**无记忆性（memoryless property）**：$P(X > s + t \mid X > s) = P(X > t)$。也就是说，即使你已经等待了 $s$ 次仍未成功，剩余等待时间的分布仍与重新开始时完全相同。

*证明。* $P(X > n) = (1-p)^n$（前 $n$ 次全失败）。于是：
$$P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X > t). \quad \blacksquare$$
**负二项分布（Negative Binomial Distribution）。** 其推广形式：第 $r$ 次成功所需的试验次数。
$$
X \sim \text{NegBin}(r, p), \quad p_X(k) = \binom{k-1}{r-1} p^r (1-p)^{k-r} \text{ for } k = r, r+1, \ldots
$$
Geometric 分布是 $r = 1$ 的特例。当建模过离散计数数据（方差大于均值）时，负二项分布自然出现，因此在实践中常作为 Poisson 的替代选择。

- **均值：** $E[X] = r/p$
- **方差：** $\text{Var}(X) = r(1-p)/p^2$

### Poisson 分布

单位时间内事件发生次数，假设事件以恒定平均速率发生。
$$X \sim \text{Poisson}(\lambda), \quad p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!} \text{ for } k = 0, 1, 2, \ldots$$
- **均值：** $E[X] = \lambda$
- **方差：** $\text{Var}(X) = \lambda$（均值等于方差——Poisson 的标志性特征）

*均值证明。*
$$E[X] = \sum_{k=0}^{\infty} k \frac{\lambda^k e^{-\lambda}}{k!} = \lambda e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!} = \lambda e^{-\lambda} \cdot e^{\lambda} = \lambda. \quad \blacksquare$$
### Poisson 对 Binomial 的近似

当 $n$ 很大、$p$ 很小、且 $\lambda = np$ 适中时，有 $\text{Binomial}(n, p) \approx \text{Poisson}(\lambda)$。

*证明概要。* 对固定 $k$：
$$\binom{n}{k} p^k (1-p)^{n-k} = \frac{n!}{k!(n-k)!} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}.$$
当 $n \to \infty$ 且 $\lambda = np$ 固定时：
- $\frac{n!}{(n-k)! \cdot n^k} \to 1$
- $(1 - \lambda/n)^n \to e^{-\lambda}$
- $(1 - \lambda/n)^{-k} \to 1$

故整个表达式收敛至 $\frac{\lambda^k e^{-\lambda}}{k!}$。$\blacksquare$

**经验法则：** 当 $n \geq 20$ 且 $p \leq 0.05$ 时，该近似效果良好。

## 连续型随机变量

若存在一个非负函数 $f_X$（称为**概率密度函数（Probability Density Function, PDF）**），使得
$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx,$$
则称该随机变量为**连续型随机变量（Continuous Random Variable）**。

其性质如下：
1. 对所有 $x$，有 $f_X(x) \geq 0$
2. $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$

**关键区别：** 对连续型随机变量，任一单点 $x$ 的概率 $P(X = x) = 0$。这并非矛盾——密度 $f(x)$ 可为正值，而单点概率却为零。概率存在于区间上，而非点上。

CDF 定义为：
$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) \, dt$$
且（在可导处）PDF 是 CDF 的导数：
$$f_X(x) = F_X'(x).$$
## 关键连续分布

### 均匀分布（Uniform Distribution）
$$X \sim \text{Uniform}(a, b), \quad f_X(x) = \frac{1}{b-a} \text{ for } x \in [a, b].$$
![主要分布图库](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/02-distribution-gallery.png)

- **均值：** $E[X] = (a+b)/2$
- **方差：** $\text{Var}(X) = (b-a)^2/12$
- **CDF：** $F_X(x) = (x-a)/(b-a)$，当 $x \in [a,b]$

这是“最大无知”分布——区间 $[a,b]$ 内每个值出现概率均等。

### 指数分布（Exponential Distribution）
$$X \sim \text{Exponential}(\lambda), \quad f_X(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0.$$
- **均值：** $E[X] = 1/\lambda$
- **方差：** $\text{Var}(X) = 1/\lambda^2$
- **CDF：** $F_X(x) = 1 - e^{-\lambda x}$

它是 Geometric 分布的连续类比，也是**唯一**具备无记忆性的连续分布：
$$P(X > s + t \mid X > s) = P(X > t).$$
*证明。* $P(X > x) = e^{-\lambda x}$。于是：
$$P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t). \quad \blacksquare$$
这使得指数分布在建模无记忆过程的等待时间时极为自然——如放射性衰变、泊松过程中的事件到达间隔、服务器请求间的时间间隔。

### 高斯（正态）分布（Gaussian / Normal Distribution）

统计学中最重要的分布。
$$X \sim \mathcal{N}(\mu, \sigma^2), \quad f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).$$
- **均值：** $E[X] = \mu$
- **方差：** $\text{Var}(X) = \sigma^2$

**标准正态分布（standard normal）** $Z \sim \mathcal{N}(0, 1)$ 满足 $\mu = 0$, $\sigma = 1$。任意正态变量均可标准化：
$$Z = \frac{X - \mu}{\sigma}.$$
**68–95–99.7 法则（经验法则）：**

| 区间 | 概率 |
|---|---|
| $\mu \pm \sigma$ | 0.6827 |
| $\mu \pm 2\sigma$ | 0.9545 |
| $\mu \pm 3\sigma$ | 0.9973 |

**为何正态分布如此重要？** 三大原因：

1. **中心极限定理（Central Limit Theorem, CLT）（第 5 篇）：** 大量独立随机变量的和或均值，无论原分布如何，均收敛于正态分布。
2. **最大熵（Maximum entropy）：** 在给定均值与方差的所有分布中，正态分布具有最高熵（即“最随机”或“最少信息”）。当你只知道均值与方差时，采用正态分布是最保守的选择。
3. **数学便利性（Mathematical convenience）：** 正态分布在仿射变换（线性组合）、条件分布与边缘分布下封闭——使其成为线性回归、卡尔曼滤波器（Kalman filters）与高斯过程（Gaussian processes）的基石。

### 对数正态分布（Log-Normal Distribution）

若 $X \sim \mathcal{N}(\mu, \sigma^2)$，则 $Y = e^X$ 服从**对数正态分布（log-normal）**。其 PDF 为：
$$f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right) \quad \text{for } y > 0.$$
- **均值：** $E[Y] = e^{\mu + \sigma^2/2}$
- **方差：** $\text{Var}(Y) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$

对数正态分布用于建模多个正因子乘积构成的量（如收入、股价、颗粒尺寸），正如正态分布建模多个加性因子之和。它恒为右偏且取值恒为正。

*PDF 归一化证明。* 令 $I = \int_{-\infty}^{\infty} e^{-x^2/2} dx$。则：
$$I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2 + y^2)/2} dx \, dy.$$
转换为极坐标：$x^2 + y^2 = r^2$, $dx \, dy = r \, dr \, d\theta$：
$$I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-r^2/2} r \, dr \, d\theta = 2\pi \int_0^{\infty} r e^{-r^2/2} dr = 2\pi \left[-e^{-r^2/2}\right]_0^{\infty} = 2\pi.$$
故 $I = \sqrt{2\pi}$，从而确认 $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ 积分为 1。$\blacksquare$

### Gamma 分布

指数分布的推广：$\alpha$ 个独立同分布 $\text{Exponential}(\beta)$ 随机变量之和。
$$X \sim \text{Gamma}(\alpha, \beta), \quad f_X(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} \text{ for } x > 0$$
其中 $\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1} e^{-t} dt$ 为伽马函数。当 $\alpha$ 为正整数时，$\Gamma(\alpha) = (\alpha - 1)!$。

- **均值：** $E[X] = \alpha/\beta$
- **方差：** $\text{Var}(X) = \alpha/\beta^2$

特例：$\text{Gamma}(1, \lambda) = \text{Exponential}(\lambda)$；$\text{Gamma}(n/2, 1/2) = \chi^2(n)$（自由度为 $n$ 的卡方分布）。

### Beta 分布
$$X \sim \text{Beta}(\alpha, \beta), \quad f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } x \in (0, 1)$$
其中 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ 为贝塔函数。

- **均值：** $E[X] = \frac{\alpha}{\alpha + \beta}$
- **方差：** $\text{Var}(X) = \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$Beta 分布定义域为 $[0,1]$，天然适用于对概率本身建模。它是 Bernoulli 与 Binomial 似然函数的共轭先验（conjugate prior）——这一性质将在第 8 篇（贝叶斯统计）中被大量使用。

特例：$\text{Beta}(1,1) = \text{Uniform}(0,1)$。

## 分布参考表

| 分布 | 类型 | PMF/PDF | 均值 | 方差 | 典型用途 |
|---|---|---|---|---|---|
| Bernoulli($p$) | 离散 | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | 二元结果 |
| Binomial($n,p$) | 离散 | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | 成功次数计数 |
| Geometric($p$) | 离散 | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | 首次成功所需试验次数 |
| Poisson($\lambda$) | 离散 | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 固定区间内事件计数 |
| Uniform($a,b$) | 连续 | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 最大无知假设 |
| Exponential($\lambda$) | 连续 | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | 等待时间建模 |
| Normal($\mu,\sigma^2$) | 连续 | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | 万能分布（CLT） |
| Gamma($\alpha,\beta$) | 连续 | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | 等待时间之和 |
| Beta($\alpha,\beta$) | 连续 | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | 概率/比例建模 |

## Python：可视化所有主要分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

ax = axes[0, 0]
for p in [0.2, 0.5, 0.8]:
    ax.bar([0, 1], [1-p, p], alpha=0.5, width=0.3, label=f'p={p}')
ax.set_title('Bernoulli(p)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.legend()

ax = axes[0, 1]
n = 20
for p in [0.2, 0.5, 0.8]:
    k = np.arange(0, n+1)
    ax.plot(k, stats.binom.pmf(k, n, p), 'o-', markersize=4, label=f'n={n}, p={p}')
ax.set_title('Binomial(n, p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

ax = axes[0, 2]
for p in [0.2, 0.5, 0.8]:
    k = np.arange(1, 15)
    ax.plot(k, stats.geom.pmf(k, p), 'o-', markersize=4, label=f'p={p}')
ax.set_title('Geometric(p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

ax = axes[1, 0]
for lam in [1, 4, 10]:
    k = np.arange(0, 20)
    ax.plot(k, stats.poisson.pmf(k, lam), 'o-', markersize=4, label=f'$\\lambda$={lam}')
ax.set_title('Poisson($\\lambda$)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

ax = axes[1, 1]
x = np.linspace(-0.5, 1.5, 300)
ax.plot(x, stats.uniform.pdf(x, 0, 1), 'b-', linewidth=2, label='Uniform(0,1)')
ax.fill_between(x, stats.uniform.pdf(x, 0, 1), alpha=0.3)
ax.set_title('Uniform(a, b)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

ax = axes[1, 2]
x = np.linspace(0, 5, 300)
for lam in [0.5, 1, 2]:
    ax.plot(x, stats.expon.pdf(x, scale=1/lam), linewidth=2, label=f'$\\lambda$={lam}')
ax.set_title('Exponential($\\lambda$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

ax = axes[2, 0]
x = np.linspace(-5, 8, 300)
for mu, sigma in [(0,1), (2,0.5), (0,2)]:
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2,
            label=f'$\\mu$={mu}, $\\sigma$={sigma}')
ax.set_title('Normal($\\mu$, $\\sigma^2$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

ax = axes[2, 1]
x = np.linspace(0, 15, 300)
for a, b in [(1, 1), (2, 1), (5, 1), (5, 2)]:
    ax.plot(x, stats.gamma.pdf(x, a, scale=1/b), linewidth=2,
            label=f'$\\alpha$={a}, $\\beta$={b}')
ax.set_title('Gamma($\\alpha$, $\\beta$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

ax = axes[2, 2]
x = np.linspace(0.001, 0.999, 300)
for a, b in [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (5, 5)]:
    ax.plot(x, stats.beta.pdf(x, a, b), linewidth=2,
            label=f'$\\alpha$={a}, $\\beta$={b}')
ax.set_title('Beta($\\alpha$, $\\beta$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

plt.suptitle('概率分布图鉴', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('distribution_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

此图鉴让你一览所有分布的形状。注意以下模式：

- **Binomial** 随 $p$ 趋近 0.5 而趋于对称；
- **Poisson** 随 $\lambda$ 增大而向右平移并趋于对称（依 CLT，渐近正态）；
- **Exponential** 恒为右偏——多数等待时间短，少数极长；
- **Beta** 极其灵活：U 形、均匀、左/右偏，皆取决于参数；
- **Gamma** 推广了 Exponential，新增形状参数控制“峰形”。

## 分布间的联系

![随机变量变换机：输入结果输出](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/02-random-variable-transformation-machine-input-outcomes-output.jpg)

上述分布并非彼此孤立，而是一个具有深层关联的家族：

1. **Bernoulli 是 $n=1$ 的 Binomial**：$\text{Binomial}(1, p) = \text{Bernoulli}(p)$
2. **Binomial 是 Bernoulli 的和**：若 $X_i \sim \text{Bernoulli}(p)$ 独立同分布，则 $\sum X_i \sim \text{Binomial}(n, p)$
3. **Poisson 近似 Binomial**：$\text{Binomial}(n, \lambda/n) \to \text{Poisson}(\lambda)$ 当 $n \to \infty$
4. **Geometric 是离散版 Exponential**：二者均具无记忆性
5. **Gamma 是 Exponential 的和**：若 $X_i \sim \text{Exp}(\lambda)$ 独立同分布，则 $\sum X_i \sim \text{Gamma}(n, \lambda)$
6. **卡方分布是特殊 Gamma**：$\chi^2(n) = \text{Gamma}(n/2, 1/2)$
7. **Beta(1,1) = Uniform(0,1)**：均匀分布是 Beta 的特例

这些联系绝非偶然，它们反映了随机过程生成数据时内在的结构性关系。

## 分位函数与逆 CDF

**分位函数（quantile function）**（或称**逆 CDF （inverse CDF）**）$F^{-1}(p)$ 对 $p \in (0, 1)$ 定义为：
$$F^{-1}(p) = \inf\{x : F(x) \geq p\}.$$
对 CDF 严格递增的连续分布，此式简化为：$F^{-1}(p)$ 是唯一满足 $F(x) = p$ 的 $x$。

关键分位点有专有名称：
- $F^{-1}(0.5)$：**中位数（median）**
- $F^{-1}(0.25)$ 与 $F^{-1}(0.75)$：**四分位数（quartiles）**
- $F^{-1}(0.01), \ldots, F^{-1}(0.99)$：**百分位数（percentiles）**

分位函数对从任意分布生成随机样本至关重要。若 $U \sim \text{Uniform}(0, 1)$，则 $X = F^{-1}(U)$ 的 CDF 即为 $F$。此即**逆 CDF 方法（inverse CDF method）**（亦称**概率积分变换（probability integral transform）**）。

*证明。* $P(X \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)$，因 $U$ 在 $(0,1)$ 上均匀分布。$\blacksquare$

**示例。** 生成 Exponential($\lambda$) 样本：$X = -\frac{1}{\lambda}\ln(1-U)$，其中 $U \sim \text{Uniform}(0,1)$。

*验证。* $F(x) = 1 - e^{-\lambda x}$，故 $F^{-1}(p) = -\frac{1}{\lambda}\ln(1-p)$。$\checkmark$

## 混合分布（Mixtures of Distributions）

并非所有分布都能完美契合单一命名族。**混合分布（mixture distribution）** 组合多个成分：
$$f(x) = \sum_{k=1}^{K} w_k f_k(x), \qquad \sum_{k=1}^K w_k = 1, \quad w_k \geq 0$$
其中 $f_k$ 为各成分密度，$w_k$ 为混合权重。

**示例。** 人口由两组构成：70% 收入 $\sim \mathcal{N}(50000, 10000^2)$，30% 收入 $\sim \mathcal{N}(90000, 15000^2)$。整体收入分布即为双成分高斯混合——呈双峰，非正态。

高斯混合模型（Gaussian Mixture Models, GMMs）是无监督学习的主力：它将复杂、多峰数据建模为高斯分布的加权和，参数通过期望最大化（Expectation-Maximization, EM）算法拟合。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(10000, 140000, 500)
component1 = 0.7 * stats.norm.pdf(x, 50000, 10000)
component2 = 0.3 * stats.norm.pdf(x, 90000, 15000)
mixture = component1 + component2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, component1, 'b--', linewidth=1.5, label='Component 1 (70%)')
ax.plot(x, component2, 'r--', linewidth=1.5, label='Component 2 (30%)')
ax.plot(x, mixture, 'k-', linewidth=2.5, label='Mixture')
ax.fill_between(x, mixture, alpha=0.15, color='gray')
ax.set_xlabel('Income ($)')
ax.set_ylabel('Density')
ax.set_title('高斯混合模型：双峰收入分布')
ax.legend()
plt.tight_layout()
plt.savefig('gaussian_mixture.png', dpi=150)
plt.show()
```

## 如何选择合适分布：决策指南

对真实数据建模时，选择恰当分布至关重要。以下是实用决策树：

**变量是离散还是连续？**

若为**离散**：
- 二元结果（是/否）：**Bernoulli**
- 固定 $n$ 次试验中成功次数：**Binomial**
- 首次成功所需试验次数：**Geometric**
- 固定区间内稀有事件发生次数：**Poisson**
- 第 $r$ 次成功前失败次数：**Negative Binomial**

若为**连续**：
- 区间内所有值等可能：**Uniform**
- 等待时间，具无记忆性：**Exponential**
- 等待时间之和：**Gamma**
- 对称钟形，多因素之和：**Normal**
- 概率/比例（取值于 $[0,1]$）：**Beta**
- 重尾、极端事件：**t-distribution** 或 **Cauchy**
- 正值、右偏：**Log-Normal** 或 **Gamma**

**经验法则：** 从简单入手。对连续数据，默认使用 Normal（CLT 为均值与和提供了理论依据）。仅当数据明显违背正态性时（如重尾、偏斜、有界支撑、离散计数），才选用更复杂的分布。

## 随机变量的函数：预览

已知 $X$ 的分布，那么 $Y = g(X)$ 的分布是什么？该问题频繁出现——特征变换、衍生量计算、或不确定性在模型中传播。我们将在第 4 篇中完整展开该工具（雅可比行列式、卷积）。此处先以简单案例一窥端倪。

**示例。** 若 $X \sim \mathcal{N}(\mu, \sigma^2)$，则 $Y = aX + b$ 的分布为何？

因正态分布经线性变换后仍为正态（第 3 篇中将用矩生成函数 MGFs 证明）：
$$Y = aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2).$$
这正是标准化成立的原因：$Z = (X - \mu)/\sigma$ 满足 $\mu_Z = 0$ 且 $\sigma_Z^2 = 1$。

**示例。** 若 $X \sim \text{Uniform}(0, 1)$，则 $Y = X^2$ 的分布为何？

使用 CDF 法：$F_Y(y) = P(X^2 \leq y) = P(X \leq \sqrt{y}) = \sqrt{y}$，其中 $0 \leq y \leq 1$。

求导得：$f_Y(y) = \frac{1}{2\sqrt{y}}$，其中 $0 < y < 1$。此即 $\text{Beta}(1/2, 1)$ 分布。

## 下一步

至此，我们已能描述单个随机变量的概率分布。但分布是一个完整对象——它包含的信息远超我们日常易处理的范畴。下一篇将引入压缩该信息的**汇总统计量（summary statistics）**：期望（“中心”）、方差（“离散度”）与矩生成函数（MGF，“指纹”，可唯一标识分布）。
