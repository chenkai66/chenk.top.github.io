---
title: "概率与统计（五）：大数定律与中心极限定理"
date: 2024-08-24 09:00:00
tags:
  - Probability
  - Statistics
  - Central Limit Theorem
  - Convergence
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "概率论的两大支柱：大数定律保证样本均值收敛，中心极限定理解释为何万物趋近高斯分布——含严格证明、各类收敛概念及 Python 仿真。"
disableNunjucks: true
series_order: 5
translationKey: "probability-statistics-5"
---
若你必须从全部概率论中仅挑选两个定理，那必然是**大数定律（LLN）**与**中心极限定理（CLT）**——它们共同回答了两个根本性问题：LLN 断言“样本均值将收敛至真实均值”，CLT 则进一步指出“这些波动的精确形态”。若无这两个定理，民意调查将失去理论依据，临床试验结果无法令人信服，随机梯度下降（SGD）的收敛性也无从解释。

本文将严谨推导这两个定理，并首先厘清各类“收敛”概念，以使结论更加精确。

## 收敛模式

在讨论“收敛”之前，必须先明确定义收敛类型。共有四种主要类型，按强度由弱到强排列如下。

![大数定律模拟](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-lln-simulation.png)

### 依分布收敛

$X_n \xrightarrow{d} X$ 当且仅当对 $F_X$ 的每一个连续点 $x$，均有  
$$F_{X_n}(x) \to F_X(x).$$
![收敛模式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-convergence-modes.png)

这是最弱的收敛形式。我们仅要求累积分布函数（CDF）逐点收敛；随机变量 $X_n$ 甚至无需与 $X$ 定义在同一概率空间上。

### 依概率收敛

$X_n \xrightarrow{P} X$ 当且仅当对任意 $\varepsilon > 0$，有  
$$\lim_{n \to \infty} P(|X_n - X| > \varepsilon) = 0.$$
它比依分布收敛更强：此处 $X_n$ 偏离 $X$ 的概率趋于零，但对任一具体试验，$X_n$ 仍可能与 $X$ 不同。

### 均方收敛（L² 收敛）

$X_n \xrightarrow{L^2} X$ 当且仅当  
$$\lim_{n \to \infty} E[(X_n - X)^2] = 0.$$
该收敛蕴含依概率收敛（由 Markov 不等式作用于 $(X_n - X)^2$ 可得）。

*均方收敛蕴含依概率收敛的证明。* 对任意 $\varepsilon > 0$：  
$$P(|X_n - X| > \varepsilon) = P((X_n - X)^2 > \varepsilon^2) \leq \frac{E[(X_n - X)^2]}{\varepsilon^2} \to 0. \quad \blacksquare$$
### 几乎必然收敛

$X_n \xrightarrow{a.s.} X$ 当且仅当  
$$P\left(\lim_{n \to \infty} X_n = X\right) = 1.$$
这是最强的收敛形式：序列对“几乎所有”样本点 $\omega$ 均收敛。

### 收敛层级关系
$$\text{a.s.} \implies \text{in probability} \implies \text{in distribution}$$  
$$\text{L}^2 \implies \text{in probability} \implies \text{in distribution}$$
几乎必然收敛与 $L^2$ 收敛不可直接比较——一般情形下，二者互不蕴含。

## 弱大数定律（WLLN）

**定理（WLLN）。** 设 $X_1, X_2, \ldots$ 是独立同分布（i.i.d.）随机变量，满足 $E[X_i] = \mu$ 且 $\text{Var}(X_i) = \sigma^2 < \infty$。定义样本均值：  
$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i.$$
则 $\bar{X}_n \xrightarrow{P} \mu$。即，对任意 $\varepsilon > 0$：  
$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \varepsilon) = 0.$$
![中心极限定理收敛](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-clt-convergence.png)

### 利用切比雪夫不等式的证明

这是概率论中最优美的证明之一——简洁、清晰、富有启发性。

**步骤 1.** 计算 $\bar{X}_n$ 的均值与方差：  
$$E[\bar{X}_n] = \frac{1}{n} \sum_{i=1}^n E[X_i] = \frac{n\mu}{n} = \mu.$$  
$$\text{Var}(\bar{X}_n) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}.$$
（利用独立性得 $\text{Var}(\sum X_i) = \sum \text{Var}(X_i)$。）

**步骤 2.** 对 $\bar{X}_n$ 应用切比雪夫不等式：  
$$P(|\bar{X}_n - \mu| > \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}.$$
**步骤 3.** 令 $n \to \infty$：  
$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \varepsilon) \leq \lim_{n \to \infty} \frac{\sigma^2}{n\varepsilon^2} = 0. \quad \blacksquare$$
该证明还给出了收敛速率：偏离 $\mu$ 超过 $\varepsilon$ 的概率至多为 $O(1/n)$。虽非最紧界（常可获得指数级集中），但具普适性。

### WLLN 的含义

拥有足够数据时，样本均值以高概率接近真实均值。这为以下实践提供了理论基础：  
- **民意调查**：询问足够多的人，样本比例即可逼近总体比例。  
- **蒙特卡洛方法**：平均足够多的随机样本，即可逼近积分值。  
- **机器学习**：平均足够多的随机梯度，即可逼近真实梯度。

## 强大数定律（SLLN）

![多个随机流合并成一个钟形曲线的中心极限定理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/05-central-limit-theorem-many-random-streams-merging-into-a-bel.jpg)

**定理（SLLN）。** 在与 WLLN 相同（甚至更弱）的条件下（仅需有限均值）：  
$$\bar{X}_n \xrightarrow{a.s.} \mu.$$
即，$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$。

![正态近似](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-normal-approximation.png)

SLLN 严格强于 WLLN：它不仅断言偏离概率趋于零，更声称在几乎所有样本路径上，均值序列都以普通微积分意义收敛至 $\mu$。其证明难度显著更高（通常需 Borel-Cantelli 引理或截断论证），故此处仅陈述而不予证明。

**关键区别**： WLLN 表述为“对任意容差，大多数实验成功”； SLLN 则表述为“在单次无限长实验中，收敛必然发生”。

## 中心极限定理（CLT）

LLN 指出样本均值收敛至 $\mu$； CLT 则揭示其**如何收敛**——通过刻画围绕 $\mu$ 的波动形态。

**定理（CLT）。** 设 $X_1, X_2, \ldots$ 是 i.i.d. 随机变量，满足 $E[X_i] = \mu$ 且 $\text{Var}(X_i) = \sigma^2 \in (0, \infty)$。则：  
$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$
等价地，记 $S_n = X_1 + \cdots + X_n$，则：  
$$\frac{S_n - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$
换言之：无论 $X_i$ 来自何种分布（只要具有有限均值与方差），标准化后的和均依分布收敛至标准正态分布。

这正是高斯分布在各处涌现的原因。身高是众多遗传与环境因素之和；测量误差是诸多微小扰动之和；股票收益（粗略而言）是大量小额交易之和。 CLT 断言：大量微小独立效应之和呈现正态性。

### 利用矩生成函数（MGF）的证明概要

我们采用稍弱版本的证明（要求 MGF 在 0 的邻域内存在）。

**步骤 1.** 令 $Z_i = (X_i - \mu)/\sigma$ 为标准化变量，则 $E[Z_i] = 0$，$\text{Var}(Z_i) = 1$。定义：  
$$W_n = \frac{S_n - n\mu}{\sigma\sqrt{n}} = \frac{1}{\sqrt{n}} \sum_{i=1}^n Z_i.$$
**步骤 2.** 计算 $W_n$ 的 MGF。因 $Z_i$ 独立同分布：  
$$M_{W_n}(t) = E\left[e^{tW_n}\right] = E\left[\prod_{i=1}^n e^{tZ_i/\sqrt{n}}\right] = \left[M_Z\left(\frac{t}{\sqrt{n}}\right)\right]^n.$$
**步骤 3.** 在 $s = 0$ 处对 $M_Z(s)$ 进行泰勒展开。由 $E[Z] = 0$、$E[Z^2] = 1$ 得：  
$$M_Z(s) = 1 + sE[Z] + \frac{s^2}{2}E[Z^2] + O(s^3) = 1 + \frac{s^2}{2} + O(s^3).$$
代入 $s = t/\sqrt{n}$：  
$$M_Z\left(\frac{t}{\sqrt{n}}\right) = 1 + \frac{t^2}{2n} + O\left(\frac{t^3}{n^{3/2}}\right).$$
**步骤 4.** 取 $n$ 次幂：  
$$M_{W_n}(t) = \left[1 + \frac{t^2}{2n} + O(n^{-3/2})\right]^n \to e^{t^2/2} \quad \text{当 } n \to \infty,$$
利用极限 $(1 + a/n)^n \to e^a$。

**步骤 5.** 函数 $e^{t^2/2}$ 正是 $\mathcal{N}(0, 1)$ 的 MGF。由 MGF 唯一性定理：  
$$W_n \xrightarrow{d} \mathcal{N}(0, 1). \quad \blacksquare$$
## 二项分布的正态近似

经典应用之一。若 $X \sim \text{Binomial}(n, p)$，则 $X = \sum_{i=1}^n X_i$，其中 $X_i \sim \text{Bernoulli}(p)$ 独立同分布。由 CLT：  
$$\frac{X - np}{\sqrt{np(1-p)}} \approx \mathcal{N}(0, 1) \quad \text{当 } n \text{ 充分大}.$$
### 连续性校正

由于 $X$ 是离散的而正态分布是连续的，为提升精度，需应用**连续性校正**：  
$$P(X \leq k) \approx \Phi\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right),$$
其中 $\Phi$ 为标准正态 CDF。

**示例。** 抛掷一枚均匀硬币 100 次，求出现至少 60 次正面的概率？  
精确解：$P(X \geq 60)$，其中 $X \sim \text{Binomial}(100, 0.5)$。  
带连续性校正的正态近似：  
$$P(X \geq 60) = P(X \geq 59.5) \approx 1 - \Phi\left(\frac{59.5 - 50}{5}\right) = 1 - \Phi(1.9) \approx 1 - 0.9713 = 0.0287.$$
精确值（scipy 计算）为 $0.0284$。近似效果极佳。

## Berry-Esseen 定理

CLT 的收敛速度如何？ Berry-Esseen 定理给出了定量界。

![Berry-Esseen 界限](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-berry-esseen.png)

**定理（Berry-Esseen）。** 在 CLT 条件下，若 $E[|Z_i|^3] = \rho < \infty$，则：  
$$\sup_x \left|P(W_n \leq x) - \Phi(x)\right| \leq \frac{C \rho}{\sqrt{n}},$$
其中 $C$ 为绝对常数（目前最优已知值为 $C \leq 0.4748$）。

该定理表明收敛速率为 $O(1/\sqrt{n})$：样本量翻倍， CDF 最大误差约减小 $\sqrt{2}$ 倍。

## CLT 对和与均值的应用

CLT 同时适用于和与均值，但标准化方式不同：

**对和：** $S_n = \sum X_i$ 的均值为 $n\mu$，标准差为 $\sigma\sqrt{n}$（随 $n$ 增长）：  
$$\frac{S_n - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$
**对均值：** $\bar{X}_n = S_n/n$ 的均值为 $\mu$，标准差为 $\sigma/\sqrt{n}$（随 $n$ 缩小）：  
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$
等价地：对充分大的 $n$，$\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2/n)$。

分母中的 $\sqrt{n}$ 是根本缩放因子：“将标准误减半，需四倍数据量”。这体现了抽样的“边际收益递减”。

## CLT 失效的情形

CLT 要求**有限方差**。若方差无穷，该定理可能彻底失效。

### 重尾分布：柯西分布

$\text{Cauchy}(0,1)$ 分布的概率密度函数为 $f(x) = \frac{1}{\pi(1+x^2)}$。其均值与方差均不存在（积分发散）。

对 i.i.d. 柯西随机变量，样本均值 $\bar{X}_n$ 与任一单个 $X_i$ 具有**相同分布**：$\bar{X}_n \sim \text{Cauchy}(0, 1)$。取平均毫无帮助：无任何收敛， CLT 完全失效。

**实践中如何检测重尾？** 若数据存在极端离群值、峰度极高（$\gamma_2 \gg 3$），或样本均值随数据量增加而无法稳定，很可能面临重尾分布。此时应：  
- 使用**中位数**替代均值（更稳健）  
- 使用**截尾均值**（剔除极端观测）  
- 应用**Winsorization**（对极端值设限）  
- 考虑专为重尾设计的分布（t 分布、 Pareto 分布、稳定分布）

柯西例子虽极端，但许多现实世界量（金融收益、保险损失、社交网络节点度分布）的尾部足够厚重，导致 CLT 收敛极慢或需极大样本量。

### 独立性缺失

CLT 同样要求独立性（或至多弱依赖）。对强相关变量（如 $X_1 = X_2 = \cdots = X_n$），样本均值无法集中：  
$$\bar{X}_n = X_1, \quad \text{Var}(\bar{X}_n) = \text{Var}(X_1),$$
故无法收敛至一点。

### 针对相依数据的 CLT 扩展

尽管经典 CLT 要求 i.i.d. 数据，若干扩展可处理相依观测：

**鞅 CLT。** 若 $\{S_n\}$ 是满足适当矩条件的鞅，则经缩放的鞅收敛至正态分布。应用于序贯分析与金融数学。

**混合序列 CLT。** 若 $X_i$ 与 $X_j$ 的依赖性随 $|i - j| \to \infty$ 足够快衰减（如 $\alpha$-混合或 $\phi$-混合条件），则仍成立 CLT，但需调整方差缩放。适用于诸多时间序列模型。

**Lindeberg-Feller CLT。** 针对独立（但未必同分布）变量的最一般版本。若 Lindeberg 条件成立（直观而言：无单一变量主导总和），则标准化和收敛至 $\mathcal{N}(0,1)$。

核心启示： CLT 具鲁棒性。你只需依赖性随时间衰减，且无单项主导，而无需严格的 i.i.d.。

## Glivenko-Cantelli 定理： CDF 版的大数定律

LLN 断言 $\bar{X}_n \to \mu$；**Glivenko-Cantelli 定理**则是关于整个分布函数的更强结果。

定义**经验 CDF**：  
$$\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{X_i \leq x}.$$
**定理（Glivenko-Cantelli）。** 若 $X_1, X_2, \ldots$ 是 i.i.d.，其 CDF 为 $F$，则：  
$$\sup_x |\hat{F}_n(x) - F(x)| \xrightarrow{a.s.} 0.$$
经验 CDF 一致收敛至真实 CDF。这正是直方图（经验分布）能可靠逼近底层分布的原因。 Kolmogorov-Smirnov 检验即基于此：通过测度经验 CDF 与理论 CDF 的最大偏差，检验数据是否服从指定分布。

**Dvoretzky-Kiefer-Wolfowitz （DKW）不等式**给出非渐近界：  
$$P\left(\sup_x |\hat{F}_n(x) - F(x)| > \varepsilon\right) \leq 2e^{-2n\varepsilon^2}.$$
这是针对整个 CDF 的 Hoeffding 型指数集中不等式（而非仅针对均值）。它给出经验 CDF 的置信带：以至少 $1 - \alpha$ 的概率，真实 CDF 在处处距经验 CDF 不超过 $\pm\sqrt{\frac{\ln(2/\alpha)}{2n}}$。

## Python：模拟 CLT 收敛过程

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Source distributions
distributions = {
    'Uniform(0,1)': lambda size: np.random.uniform(0, 1, size),
    'Exponential(1)': lambda size: np.random.exponential(1, size),
    'Bernoulli(0.3)': lambda size: np.random.binomial(1, 0.3, size),
    'Poisson(3)': lambda size: np.random.poisson(3, size),
}

sample_sizes = [1, 2, 5, 30]
n_simulations = 10000

fig, axes = plt.subplots(len(distributions), len(sample_sizes),
                         figsize=(16, 12))

for row, (dist_name, sampler) in enumerate(distributions.items()):
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col]

        # Generate n_simulations sample means, each from n observations
        sample_means = np.array([
            sampler(n).mean() for _ in range(n_simulations)
        ])

        # Standardize
        mu = sample_means.mean()
        sigma = sample_means.std()
        standardized = (sample_means - mu) / sigma if sigma > 0 else sample_means

        # Histogram
        ax.hist(standardized, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='white', linewidth=0.5)

        # Overlay standard normal
        x = np.linspace(-4, 4, 200)
        ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2)

        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.6)

        if row == 0:
            ax.set_title(f'n = {n}', fontsize=13)
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=11)
        if row == len(distributions) - 1:
            ax.set_xlabel('Standardized mean')

plt.suptitle('Central Limit Theorem: Standardized Sample Means vs N(0,1)',
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

该模拟展示了 CLT 的实际运作。每行对应一种源分布——有的偏斜（指数分布）、有的离散（伯努利分布）、有的对称（均匀分布）。每列增大样本量 $n$。红色曲线为标准正态 $\mathcal{N}(0,1)$。

当 $n = 1$ 时，直方图反映原始分布形状；至 $n = 5$，各分布已呈钟形；当 $n = 30$ 时，即使高度偏斜的指数分布，其样本均值也几乎无法与高斯分布区分。这正是 CLT 的普适性：源分布无关紧要（只要方差有限）。

## Delta 方法：变换量的 CLT

![大数定律硬币翻转收敛到百分之五十](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/05-law-of-large-numbers-coin-flips-converging-to-fifty-percent.jpg)

CLT 告诉我们 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$。但若关注均值的函数（如 $g(\bar{X}_n)$）呢？**Delta 方法**将 CLT 推广至此。

**定理（Delta 方法）。** 若 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$，且 $g$ 在 $\mu$ 处可导且 $g'(\mu) \neq 0$，则：  
$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2).$$
*证明。* 在 $\mu$ 处作一阶泰勒展开：  
$$g(\bar{X}_n) \approx g(\mu) + g'(\mu)(\bar{X}_n - \mu).$$
因此：  
$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \approx g'(\mu) \cdot \sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} g'(\mu) \cdot \mathcal{N}(0, \sigma^2) = \mathcal{N}(0, [g'(\mu)]^2\sigma^2). \quad \blacksquare$$
**示例。** 估计 $\theta = \mu^2$。取 $g(x) = x^2$，则 $g'(x) = 2x$：  
$$\sqrt{n}(\bar{X}_n^2 - \mu^2) \xrightarrow{d} \mathcal{N}(0, 4\mu^2\sigma^2).$$
故 $\text{Var}(\bar{X}_n^2) \approx 4\mu^2\sigma^2/n$。

**示例。** 估计 $\theta = \ln \mu$（$\mu > 0$）。取 $g(x) = \ln x$，则 $g'(x) = 1/x$：  
$$\sqrt{n}(\ln \bar{X}_n - \ln \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2/\mu^2).$$
Delta 方法对构造变换参数的置信区间至关重要——方差稳定化变换、优势比、对数风险比，以及应用统计学中的诸多其他量。

## 超越切比雪夫的集中不等式

切比雪夫不等式具普适性但较松。对特定分布族，存在更紧的界。

### Hoeffding 不等式

**定理（Hoeffding）。** 若 $X_1, \ldots, X_n$ 独立，且 $a_i \leq X_i \leq b_i$ 几乎必然成立，则：  
$$P\left(\bar{X}_n - E[\bar{X}_n] \geq t\right) \leq \exp\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i - a_i)^2}\right).$$
对同分布有界变量（$a_i = a$, $b_i = b$）：  
$$P\left(|\bar{X}_n - \mu| \geq t\right) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right).$$
该界在 $n$ 上呈**指数衰减**，远优于切比雪夫的 $O(1/n)$。

**示例。** 抛掷一枚均匀硬币 $n = 100$ 次，观察到 60% 或更多正面的概率是多少？  
切比雪夫：$P(|\bar{X} - 0.5| \geq 0.1) \leq \frac{0.25}{100 \times 0.01} = 0.25$。  
Hoeffding：$P(|\bar{X} - 0.5| \geq 0.1) \leq 2\exp(-2 \times 100 \times 0.01/1) = 2e^{-2} \approx 0.27$。  
本例中两者相近，但当 $t$ 增大或对亚高斯变量， Hoeffding 显著更紧。

### Chernoff 界

对任意 $t > 0$ 及任意随机变量 $X$：  
$$P(X \geq a) = P(e^{tX} \geq e^{ta}) \leq \frac{E[e^{tX}]}{e^{ta}} = \frac{M_X(t)}{e^{ta}}.$$
对 $t$ 优化可得最紧界。此即 **Chernoff 界**， Hoeffding 不等式即由此导出。

## 连续映射定理

**定理。** 若 $X_n \xrightarrow{d} X$ 且 $g$ 连续，则 $g(X_n) \xrightarrow{d} g(X)$。

该简单定理出人意料地强大。结合 CLT，可用于推导检验统计量的渐近分布。

**示例。** 若 $Z_n \xrightarrow{d} \mathcal{N}(0,1)$，则 $Z_n^2 \xrightarrow{d} \chi^2(1)$，因 $g(x) = x^2$ 连续。

## Slutsky 定理

**定理（Slutsky）。** 若 $X_n \xrightarrow{d} X$ 且 $Y_n \xrightarrow{P} c$（$c$ 为常数），则：  
- $X_n + Y_n \xrightarrow{d} X + c$  
- $X_n Y_n \xrightarrow{d} cX$  
- $X_n / Y_n \xrightarrow{d} X/c$（若 $c \neq 0$）

**应用于 t 统计量。** 在原假设 $H_0: \mu = \mu_0$ 下：  
$$T_n = \frac{\bar{X}_n - \mu_0}{S_n/\sqrt{n}} = \frac{\sqrt{n}(\bar{X}_n - \mu_0)/\sigma}{S_n/\sigma}.$$
分子依分布收敛至 $\mathcal{N}(0, 1)$（CLT）；分母 $S_n/\sigma \xrightarrow{P} 1$（LLN 作用于样本方差）。由 Slutsky 定理，$T_n \xrightarrow{d} \mathcal{N}(0, 1)$。  
这说明：即使 $\sigma$ 未知，对大样本仍可用 Z 检验——t 分布收敛至正态分布。

## 为何机器学习有效： CLT 的关联

随机梯度下降（SGD）通过一个大小为 $B$ 的 mini-batch 样本估计真实梯度 $\nabla L(\theta)$：  
$$\hat{g} = \frac{1}{B} \sum_{i=1}^B \nabla \ell(\theta; x_i).$$
由 LLN，当 $B \to \infty$ 时，$\hat{g} \to \nabla L(\theta)$；由 CLT：  
$$\hat{g} \approx \mathcal{N}\left(\nabla L(\theta), \frac{\Sigma}{B}\right),$$
其中 $\Sigma$ 为单个梯度的协方差矩阵。

该高斯近似具有实际意义：  
- **噪声随 $O(1/\sqrt{B})$ 减小**：批量大小加倍，噪声仅减小 $\sqrt{2}$ 倍。  
- **梯度噪声起正则化作用**： CLT 预测的高斯噪声有助于 SGD 逃离尖锐极小值，找到更平坦、泛化性更好的解。  
- **学习率与批量大小耦合**：“线性缩放规则”（$\text{lr} \propto B$）即源于 CLT 的缩放特性。

CLT 不仅解释样本均值为何有效，更解释了为何含噪声的优化算法能收敛并泛化。

## Python：演示 LLN 收敛过程

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LLN convergence for different distributions
distributions = {
    'Exponential(1) [$\\mu=1$]': (np.random.exponential, {'scale': 1.0}, 1.0),
    'Bernoulli(0.7) [$\\mu=0.7$]': (np.random.binomial, {'n': 1, 'p': 0.7}, 0.7),
    'Poisson(3) [$\\mu=3$]': (np.random.poisson, {'lam': 3}, 3.0),
    'Uniform(0,1) [$\\mu=0.5$]': (np.random.uniform, {'low': 0, 'high': 1}, 0.5),
}

for idx, (name, (sampler, params, mu)) in enumerate(distributions.items()):
    ax = axes[idx // 2, idx % 2]
    n_max = 2000

    # Multiple sample paths
    for trial in range(10):
        data = sampler(size=n_max, **params)
        running_means = np.cumsum(data) / np.arange(1, n_max + 1)
        ax.plot(running_means, alpha=0.4, linewidth=0.8)

    ax.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'$\\mu = {mu}$')
    ax.set_xlabel('n (number of samples)')
    ax.set_ylabel('Running mean $\\bar{X}_n$')
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Law of Large Numbers: Running Averages Converge to True Mean',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('lln_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

每幅图展示 10 条独立的运行均值 $\bar{X}_n$ 轨迹。初期轨迹嘈杂且远离真实均值（红色虚线）；随着 $n$ 增大，所有轨迹均收敛——这正是 LLN 的直观体现。指数分布初始波动最大（因其右尾厚重），但即便如此，$n \approx 500$ 时亦趋于稳定。

## 总结
| 定理 | 表述 | 要求 | 速率 |
|---|---|---|---|
| WLLN | $\bar{X}_n \xrightarrow{P} \mu$ | i.i.d.，有限均值与方差 | Chebyshev 给出 $O(1/n)$ |
| SLLN | $\bar{X}_n \xrightarrow{a.s.} \mu$ | i.i.d.，有限均值 | — |
| CLT | $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0,1)$ | i.i.d.，有限方差 | Berry-Esseen 给出 $O(1/\sqrt{n})$ |
| Delta 方法 | $\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2\sigma^2)$ | CLT 成立 + $g$ 在 $\mu$ 可导 | $O(1/\sqrt{n})$ |
| Hoeffding | $P(\|\bar{X}_n - \mu\| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$ | 独立、有界 | 指数级 |

## 下一步

LLN 与 CLT 告诉我们样本均值收敛至总体参数。但如何从数据中实际估计这些参数？下一篇文章将发展估计理论——矩估计法、最大似然估计、偏差-方差权衡——并将这些思想与机器学习中广泛应用的正则化技术联系起来。
