---
title: "概率与统计（八）：贝叶斯统计——先验、后验，以及频率学派为何争论不休"
date: 2024-08-30 09:00:00
tags:
  - Probability
  - Statistics
  - Bayesian Inference
  - MCMC
  - Machine Learning
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "从第一性原理出发的贝叶斯推断：后验分布、共轭先验、Beta-二项模型与正态-正态模型、可信区间、预测分布、MCMC 直观理解，以及与机器学习正则化的深层联系。"
disableNunjucks: true
series_order: 8
series_total: 8
translationKey: "probability-statistics-8"
---
两位统计学家走进一家酒吧。一人说：“明天下雨的概率是 30%。”另一人反驳道：“概率指的是长期频率；而明天只发生一次，这种说法毫无意义。”第一个人回应：“它量化的是我对这一独特事件的不确定性。”两人就此争论了一整晚。

这大致就是贝叶斯学派与频率学派之争。争论的焦点并非谁对谁错——两种框架在数学上都是自洽的。关键在于“概率”究竟意味着什么，以及这种解释如何塑造你所使用的工具。在前六篇文章中，我们主要采用了频率学派的推理方式；现在，我们将转向贝叶斯视角：参数是随机变量，数据用于更新我们的信念，而不确定性通过概率分布（而非置信区间）来刻画。

---

## 贝叶斯 vs 频率学派：核心差异


![先验到后验更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-prior-posterior.png)

### 频率学派观点

- **参数** 是固定但未知的常数。
- **概率** 指的是长期相对频率。
- **推断** 基于估计量的抽样分布（即 $\hat{\theta}$ 在重复抽样下的变异性）。
- 95% 置信区间的意思是：“如果我重复这个实验很多次，那么我计算出的区间中有 95% 会包含真实参数 $\theta$。”

### 贝叶斯观点

- **参数** 是具有概率分布的随机变量。
- **概率** 用于量化主观不确定性（即信念的程度）。
- **推断** 通过将先验分布与观测数据结合，得到后验分布。
- 95% 可信区间的意思是：“在给定数据的前提下，$\theta$ 落在该区间内的概率为 95%。”

对于具体问题，贝叶斯框架通常更直观：你从已有信念出发，观察数据，然后更新信念。而频率学派框架则更适合方法验证：它无需指定先验，就能保证在重复使用时控制错误率。

实践中，两者的结果常常一致——尤其是在大样本情况下，数据会压倒先验的影响。

## 分布形式的贝叶斯公式

我们之前已经见过针对事件的贝叶斯定理。贝叶斯推断的核心正是将同样的逻辑应用到概率分布上。

![共轭先验族](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-conjugate-priors.png)

给定数据 $\mathbf{x} = (x_1, \ldots, x_n)$ 和参数 $\theta$：
$$\underbrace{p(\theta | \mathbf{x})}_{\text{后验}} = \frac{\underbrace{p(\mathbf{x} | \theta)}_{\text{似然}} \cdot \underbrace{p(\theta)}_{\text{先验}}}{\underbrace{p(\mathbf{x})}_{\text{边缘似然}}}$$
其中边缘似然（也称为 **证据**）为：
$$p(\mathbf{x}) = \int p(\mathbf{x} | \theta) \, p(\theta) \, d\theta.$$
由于 $p(\mathbf{x})$ 对于 $\theta$ 来说是一个常数，我们通常写作：
$$\boxed{p(\theta | \mathbf{x}) \propto p(\mathbf{x} | \theta) \cdot p(\theta)}$$
**后验正比于似然乘以先验。**

这就是贝叶斯统计的基本方程，其余一切皆由此衍生。

### 各成分解读

- **先验** $p(\theta)$：你在看到数据之前对 $\theta$ 的信念。这里可以融入领域知识、以往实验结果，或者“合理的默认假设”。
- **似然** $p(\mathbf{x} | \theta)$：在给定特定 $\theta$ 值的情况下，观测到当前数据的概率。这与极大似然估计（MLE）中使用的似然函数完全相同。
- **后验** $p(\theta | \mathbf{x})$：你在看到数据之后更新的信念。这是一个完整的概率分布，而不是一个点估计。

## 共轭先验

如果某个先验与似然结合后，所得的后验分布与先验属于同一分布族，那么这个先验就被称为该似然的 **共轭先验**。共轭性使得数学推导变得简单——后验拥有闭式解。

![可信区间与置信区间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-credible-vs-confidence.png)

| 似然 | 共轭先验 | 后验 |
|---|---|---|
| 伯努利 / 二项 | Beta | Beta |
| 泊松 | Gamma | Gamma |
| 正态（已知 $\sigma^2$） | 正态 | 正态 |
| 正态（已知 $\mu$） | 逆伽马 | 逆伽马 |
| 多项 | Dirichlet | Dirichlet |
| 指数 | Gamma | Gamma |

如果没有共轭性，后验积分 $p(\mathbf{x}) = \int p(\mathbf{x}|\theta)p(\theta)d\theta$ 可能没有闭式解，这时就需要借助数值方法（如 MCMC、变分推断）。

## Beta-二项模型


![贝叶斯更新：先验到后验信念转换](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/08-bayesian-updating-prior-to-posterior-belief-transformation-t.jpg)

这是贝叶斯推断的经典范例，我们将完整地推导一遍。

![贝叶斯更新动画](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/probstat-08-bayesian-updating.gif)


![Beta-二项分布顺序更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-beta-binomial.png)

### 设定

你正在抛一枚硬币，想估计正面朝上的概率 $\theta$。

- **似然：** $X | \theta \sim \text{Binomial}(n, \theta)$，因此 $p(x | \theta) = \binom{n}{x} \theta^x (1-\theta)^{n-x}$。
- **先验：** $\theta \sim \text{Beta}(\alpha, \beta)$，因此 $p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$。

### 推导后验
$$p(\theta | x) \propto p(x | \theta) \cdot p(\theta) = \binom{n}{x} \theta^x (1-\theta)^{n-x} \cdot \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

$$\propto \theta^{x + \alpha - 1} (1-\theta)^{n - x + \beta - 1}.$$
这正是 Beta 分布的核函数：
$$\boxed{\theta | x \sim \text{Beta}(\alpha + x, \beta + n - x)}$$
### 解读

先验参数 $\alpha$ 和 $\beta$ 起到了“伪计数”的作用——仿佛你在实验前已经观察到了 $\alpha - 1$ 次正面和 $\beta - 1$ 次反面。而实际数据则增加了 $x$ 次正面和 $n - x$ 次反面。

后验均值为：
$$E[\theta | x] = \frac{\alpha + x}{\alpha + \beta + n}.$$
这实际上是先验均值 $\frac{\alpha}{\alpha + \beta}$ 与样本比例 $\frac{x}{n}$ 的加权平均：
$$E[\theta | x] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{先验均值}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{x}{n}}_{\text{样本比例}}.$$
当 $n \to \infty$ 时，先验的权重趋近于零，后验均值收敛到 MLE $x/n$。**只要有足够多的数据，先验就变得无关紧要。**

### 实例演算

你怀疑一枚硬币可能有偏。你设定的先验是 $\theta \sim \text{Beta}(2, 2)$（对称分布，略微倾向于公平）。你抛了 10 次，得到 7 次正面。

**先验：** $\text{Beta}(2, 2)$，均值 = 0.5。

**后验：** $\text{Beta}(2 + 7, 2 + 3) = \text{Beta}(9, 5)$，均值 = $9/14 \approx 0.643$。

**MLE：** $7/10 = 0.7$。

后验均值（0.643）相比 MLE（0.7）更靠近先验均值（0.5）。先验起到了正则化的作用，将估计值向中心拉近。随着数据量增加，这种收缩效应会逐渐减弱。

### 先验选择

| 先验参数 | 解释 | 强度 |
|---|---|---|
| $\text{Beta}(1, 1)$ | 均匀分布——无任何偏好 | 弱（相当于 0 次伪观测） |
| $\text{Beta}(0.5, 0.5)$ | Jeffreys 先验——更关注极端值 | 无信息 |
| $\text{Beta}(2, 2)$ | 略微偏好公平性 | 弱 |
| $\text{Beta}(10, 10)$ | 强烈相信硬币是公平的 | 中等 |
| $\text{Beta}(100, 100)$ | 极其确信硬币是公平的 | 强 |

先验的强度由 $\alpha + \beta$ 控制。该值越大，代表“先验数据”越多，对新数据的更新就越不敏感。

## 正态-正态模型

### 设定

观测 $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$，其中 $\sigma^2$ 已知。先验为 $\mu \sim \mathcal{N}(\mu_0, \tau^2)$。

### 后验
$$p(\mu | \mathbf{x}) \propto \exp\left(-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2\right) \cdot \exp\left(-\frac{(\mu - \mu_0)^2}{2\tau^2}\right).$$
对 $\mu$ 完成平方（合并指数项）后可得：
$$\mu | \mathbf{x} \sim \mathcal{N}\left(\mu_n, \sigma_n^2\right)$$
其中：
$$\mu_n = \frac{\frac{n}{\sigma^2}\bar{x} + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}, \qquad \sigma_n^2 = \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}.$$
*推导.* 忽略与 $\mu$ 无关的常数项，对数后验为：
$$-\frac{1}{2}\left[\frac{n(\mu - \bar{x})^2}{\sigma^2} + \frac{(\mu - \mu_0)^2}{\tau^2}\right]$$

$$= -\frac{1}{2}\left[\left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right)\mu^2 - 2\left(\frac{n\bar{x}}{\sigma^2} + \frac{\mu_0}{\tau^2}\right)\mu + \text{const}\right].$$
这是一个关于 $\mu$ 的二次函数，因此后验服从正态分布，其精度（方差的倒数）为 $\frac{n}{\sigma^2} + \frac{1}{\tau^2}$，均值即为上述的精度加权平均。$\blacksquare$

### 精度形式

定义 **精度** 为方差的倒数：$\lambda = 1/\sigma^2$，$\lambda_0 = 1/\tau^2$。
$$\text{后验精度} = n\lambda + \lambda_0 \qquad \text{(精度相加)}$$

$$\text{后验均值} = \frac{n\lambda \bar{x} + \lambda_0 \mu_0}{n\lambda + \lambda_0} \qquad \text{(精度加权平均)}$$
由于精度具有可加性，它成为高斯分布下贝叶斯更新的自然参数化方式。

## 可信区间 vs 置信区间

**可信区间** 是贝叶斯框架中对应于置信区间的概念。

**定义.** 参数 $\theta$ 的 $100(1-\alpha)\%$ 可信区间 $[a, b]$ 满足：
$$P(\theta \in [a, b] | \mathbf{x}) = 1 - \alpha.$$
**最高后验密度（HPD）区间** 是满足该条件的最短区间：它包含所有后验密度高于某一阈值的点。

### 关键差异

| | 置信区间 | 可信区间 |
|---|---|---|
| 什么是随机的？ | 区间（依赖于数据） | $\theta$（参数是随机的） |
| 固定的是什么？ | $\theta$（未知常数） | 数据（已观测） |
| 解释 | 95% 的区间覆盖 $\theta$ | $\theta$ 落在区间内的后验概率为 95% |
| 所需前提 | 抽样分布 | 先验分布 |

对于先验弥散（$\tau \to \infty$）的正态-正态模型，95% 可信区间等于 95% 置信区间。但在先验信息较强或样本量较小时，两者会有所不同。

## 后验的点估计

后验 $p(\theta | \mathbf{x})$ 是一个完整的分布。若需要一个单一数值，可以选择以下几种估计量：

| 估计量 | 定义 | 最优于 |
|---|---|---|
| 后验均值 | $E[\theta \mid\mathbf{x}]$ | 最小化 $E[(\hat\theta - \theta)^2 \mid\mathbf{x}]$（平方误差） |
| 后验中位数 | $p(\theta \mid\mathbf{x})$ 的中位数 | 最小化 $E[\mid\hat\theta - \theta\mid\, \mid\mathbf{x}]$（绝对误差） |
| MAP | $\arg\max_\theta p(\theta \mid\mathbf{x})$ | 后验众数（= 带惩罚的 MLE） |

对于对称且单峰的后验（如正态分布），这三种估计量重合。对于偏斜的后验，它们会有所不同，具体选择取决于你所采用的损失函数。

## 预测分布

很多时候，我们并不关心参数 $\theta$ 本身，而是希望预测未来的观测值 $\tilde{X}$。

**后验预测分布** 通过对参数积分来实现：
$$p(\tilde{x} | \mathbf{x}) = \int p(\tilde{x} | \theta) \, p(\theta | \mathbf{x}) \, d\theta.$$
这考虑了 **参数的不确定性**：我们不是代入 $\theta$ 的某个点估计，而是对所有可能的 $\theta$ 值进行加权平均，权重即为后验概率。

### Beta-二项预测

在观测到 $n$ 次抛掷中有 $x$ 次正面，且后验为 $\theta | x \sim \text{Beta}(\alpha + x, \beta + n - x)$ 的情况下：
$$P(\tilde{X} = 1 | x) = E[\theta | x] = \frac{\alpha + x}{\alpha + \beta + n}.$$
这就是 **拉普拉斯继承法则（Laplace's rule of succession）**。若采用均匀先验（$\alpha = \beta = 1$），且在 $n$ 次试验中有 $x$ 次成功，则：
$$P(\text{下次成功}) = \frac{x + 1}{n + 2}.$$
例如，如果你在 10 次抛掷中看到 7 次正面，那么下一次抛出正面的预测概率是 $8/12 = 2/3$，而不是 MLE 给出的 $7/10$。

### 正态预测

若后验为 $\mu | \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma_n^2)$，且 $\tilde{X} | \mu \sim \mathcal{N}(\mu, \sigma^2)$，则：
$$\tilde{X} | \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2).$$
预测方差 $\sigma^2 + \sigma_n^2$ **大于** 单纯的抽样方差 $\sigma^2$，因为它额外包含了对 $\mu$ 的不确定性。如果直接用点估计 $\hat\mu$ 代替真实 $\mu$ 进行预测（即“插值预测”），就会低估整体的不确定性。

## MCMC：当共轭性不足时

大多数现实中的模型并没有共轭先验。此时，后验 $p(\theta | \mathbf{x})$ 仅能以未归一化的形式给出：我们可以对任意 $\theta$ 计算 $p(\mathbf{x} | \theta) p(\theta)$，但计算归一化常数 $p(\mathbf{x}) = \int p(\mathbf{x}|\theta)p(\theta)d\theta$ 是不可行的。

![MCMC轨迹图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-mcmc-trace.png)

**马尔可夫链蒙特卡洛（MCMC）** 方法可以在不计算归一化常数的情况下，生成来自后验分布的样本 $\theta^{(1)}, \theta^{(2)}, \ldots$。

### Metropolis-Hastings：核心思想

1. 从某个初始值 $\theta^{(0)}$ 开始。
2. **提议**：从提议分布 $q(\theta^* | \theta^{(t)})$ 中生成一个新值 $\theta^*$。
3. **接受**：以概率
$$\alpha = \min\left(1, \frac{p(\theta^* | \mathbf{x}) \, q(\theta^{(t)} | \theta^*)}{p(\theta^{(t)} | \mathbf{x}) \, q(\theta^* | \theta^{(t)})}\right) = \min\left(1, \frac{p(\mathbf{x} | \theta^*) p(\theta^*) \, q(\theta^{(t)} | \theta^*)}{p(\mathbf{x} | \theta^{(t)}) p(\theta^{(t)}) \, q(\theta^* | \theta^{(t)})}\right)$$
接受该提议。

4. 如果接受，则 $\theta^{(t+1)} = \theta^*$；否则，$\theta^{(t+1)} = \theta^{(t)}$。

归一化常数 $p(\mathbf{x})$ 在比值中被约去。由此生成的马尔可夫链以 $p(\theta | \mathbf{x})$ 为平稳分布，因此经过预烧期（burn-in）后，样本就近似来自后验分布。

如果提议分布 **对称**（即 $q(\theta^*|\theta) = q(\theta|\theta^*)$），那么接受概率简化为 $p(\theta^* | \mathbf{x}) / p(\theta^{(t)} | \mathbf{x})$ —— 算法会倾向于向后验密度更高的区域移动，同时偶尔接受“下坡”移动以充分探索整个参数空间。

## 贝叶斯与机器学习的联系


![MCMC随机游走探索概率景观路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/08-mcmc-random-walk-exploring-probability-landscape-pathfinding.jpg)

### 正则化即先验

我们在第 6 篇文章中提到，带高斯先验的 MAP 估计等价于 L2 正则化。现在我们来明确这一点。

**神经网络权重先验：** 对权重向量 $w$ 施加先验 $w \sim \mathcal{N}(0, \tau^2 I)$。此时 MAP 目标变为：
$$\hat{w}_{\text{MAP}} = \arg\max_w \left[\sum_{i=1}^n \ln p(y_i | x_i, w) - \frac{\|w\|^2}{2\tau^2}\right]$$

$$= \arg\min_w \left[-\sum_{i=1}^n \ln p(y_i | x_i, w) + \frac{1}{2\tau^2}\|w\|^2\right].$$
这正是带有 L2 正则化（权重衰减）的损失函数，其中正则化系数 $\lambda = 1/\tau^2$。

模糊先验（$\tau \to \infty$）对应无正则化（即 MLE）。强先验（$\tau$ 较小）则对应强正则化，将权重拉向零。

### Dropout 作为近似贝叶斯推断

Gal 和 Ghahramani（2016）证明，使用 dropout 进行训练等价于在深度高斯过程中进行近似贝叶斯推断。在测试时，通过多次启用 dropout 的前向传播所得到的预测分布，近似于后验预测分布。

### 贝叶斯神经网络

与其寻找单一的权重向量 $\hat{w}$，不如维护完整的后验分布 $p(w | \mathbf{x}, \mathbf{y})$，并通过积分进行预测：
$$p(y^* | x^*, \mathbf{x}, \mathbf{y}) = \int p(y^* | x^*, w) \, p(w | \mathbf{x}, \mathbf{y}) \, dw.$$
这是不确定性量化的黄金标准，但对于大型网络来说，该积分难以计算。实际应用中通常采用变分推断或 MCMC 近似。

## 贝叶斯与频率学派何时一致？

在大样本且模型设定正确的情况下：

1. 后验会集中在真实参数 $\theta$ 附近（Bernstein-von Mises 定理）。
2. 后验均值趋近于 MLE。
3. 可信区间与置信区间趋于一致。
4. 先验的影响变得可以忽略。

两者的分歧主要出现在以下情况：
- 样本量较小，先验对结果有显著影响；
- 参数维度远高于数据量；
- 问题的关注点不同：贝叶斯回答的是“基于当前数据，我现在应该相信什么？”，而频率学派关注的是“这个推断程序在长期重复使用中表现如何？”

## Python：贝叶斯更新可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Beta-Binomial updating
ax = axes[0, 0]
theta = np.linspace(0, 1, 500)
alpha_prior, beta_prior = 2, 2
n_obs, x_obs = 10, 7

# Prior
prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
# Likelihood (unnormalized)
likelihood = theta**x_obs * (1-theta)**(n_obs - x_obs)
likelihood = likelihood / likelihood.max() * prior.max()  # scale for plotting
# Posterior
posterior = stats.beta.pdf(theta, alpha_prior + x_obs, beta_prior + n_obs - x_obs)

ax.plot(theta, prior, 'b-', linewidth=2, label=f'Prior: Beta({alpha_prior},{beta_prior})')
ax.plot(theta, likelihood, 'g--', linewidth=2, label=f'Likelihood (scaled)')
ax.plot(theta, posterior, 'r-', linewidth=2.5,
        label=f'Posterior: Beta({alpha_prior+x_obs},{beta_prior+n_obs-x_obs})')
ax.axvline(x_obs/n_obs, color='gray', linestyle=':', alpha=0.7, label=f'MLE = {x_obs/n_obs}')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('Beta-Binomial: 7 heads in 10 flips', fontsize=13)
ax.legend(fontsize=9)

# Panel 2: Sequential updating
ax = axes[0, 1]
observations = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
alpha, beta_param = 1, 1  # Start with uniform prior
colors = plt.cm.viridis(np.linspace(0, 1, len(observations) + 1))

ax.plot(theta, stats.beta.pdf(theta, alpha, beta_param), color=colors[0],
        linewidth=1.5, label='Prior (n=0)')

for i, obs in enumerate(observations):
    alpha += obs
    beta_param += 1 - obs
    if i in [0, 4, 9, 19]:
        ax.plot(theta, stats.beta.pdf(theta, alpha, beta_param),
                color=colors[i+1], linewidth=1.5, label=f'n={i+1}')

ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('Sequential Bayesian Updating', fontsize=13)
ax.legend(fontsize=9)

# Panel 3: Prior sensitivity
ax = axes[1, 0]
n_obs, x_obs = 5, 4
priors = [(1, 1, 'Uniform'), (2, 2, 'Beta(2,2)'), (10, 10, 'Beta(10,10)'),
          (0.5, 0.5, 'Jeffreys')]

for a, b, name in priors:
    post = stats.beta.pdf(theta, a + x_obs, b + n_obs - x_obs)
    ax.plot(theta, post, linewidth=2, label=f'Prior: {name}')

ax.axvline(x_obs/n_obs, color='gray', linestyle=':', alpha=0.7, label='MLE')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Posterior density')
ax.set_title(f'Prior Sensitivity (n={n_obs}, x={x_obs})', fontsize=13)
ax.legend(fontsize=9)

# Panel 4: Credible interval vs confidence interval
ax = axes[1, 1]
# Normal-Normal model
mu_0, tau = 0, 2  # prior
sigma = 1  # known
data = np.array([1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4])
n = len(data)
xbar = data.mean()

# Posterior
precision_post = n/sigma**2 + 1/tau**2
sigma_post = 1/np.sqrt(precision_post)
mu_post = (n/sigma**2 * xbar + 1/tau**2 * mu_0) / precision_post

x = np.linspace(-1, 3, 300)
prior_dist = stats.norm.pdf(x, mu_0, tau)
post_dist = stats.norm.pdf(x, mu_post, sigma_post)

ax.plot(x, prior_dist, 'b--', linewidth=1.5, label=f'Prior: N({mu_0}, {tau}$^2$)')
ax.plot(x, post_dist, 'r-', linewidth=2.5,
        label=f'Posterior: N({mu_post:.2f}, {sigma_post:.3f}$^2$)')
ax.axvline(xbar, color='green', linestyle=':', label=f'$\\bar{{x}}$ = {xbar:.2f}')

# 95% credible interval
ci_low = mu_post - 1.96 * sigma_post
ci_high = mu_post + 1.96 * sigma_post
ax.axvspan(ci_low, ci_high, alpha=0.15, color='red', label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')

ax.set_xlabel('$\\mu$')
ax.set_ylabel('Density')
ax.set_title('Normal-Normal Model', fontsize=13)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('bayesian_inference.png', dpi=150)
plt.show()
```

```python
# Simple Metropolis-Hastings sampler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Target: posterior for Beta-Binomial (we know the answer, so we can verify)
n_obs, x_obs = 20, 14
alpha_prior, beta_prior = 2, 2

def log_posterior(theta):
    """Log posterior (up to a constant) for Beta-Binomial."""
    if theta <= 0 or theta >= 1:
        return -np.inf
    log_lik = x_obs * np.log(theta) + (n_obs - x_obs) * np.log(1 - theta)
    log_prior = (alpha_prior - 1) * np.log(theta) + (beta_prior - 1) * np.log(1 - theta)
    return log_lik + log_prior

# Metropolis-Hastings
n_samples = 50000
samples = np.zeros(n_samples)
samples[0] = 0.5
accepted = 0

for i in range(1, n_samples):
    # Propose from normal centered at current value
    proposal = samples[i-1] + np.random.normal(0, 0.1)

    # Acceptance ratio
    log_alpha = log_posterior(proposal) - log_posterior(samples[i-1])

    if np.log(np.random.uniform()) < log_alpha:
        samples[i] = proposal
        accepted += 1
    else:
        samples[i] = samples[i-1]

burn_in = 5000
posterior_samples = samples[burn_in:]

print(f"Acceptance rate: {accepted/n_samples:.3f}")
print(f"MCMC posterior mean: {posterior_samples.mean():.4f}")
print(f"Exact posterior mean: {(alpha_prior + x_obs)/(alpha_prior + beta_prior + n_obs):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Trace plot
ax = axes[0]
ax.plot(samples[:2000], 'b-', alpha=0.7, linewidth=0.5)
ax.axhline(y=(alpha_prior + x_obs)/(alpha_prior + beta_prior + n_obs),
           color='red', linestyle='--', label='True mean')
ax.set_xlabel('Iteration')
ax.set_ylabel('$\\theta$')
ax.set_title('MCMC Trace Plot', fontsize=13)
ax.legend()

# Histogram vs exact posterior
ax = axes[1]
theta = np.linspace(0, 1, 200)
exact = stats.beta.pdf(theta, alpha_prior + x_obs, beta_prior + n_obs - x_obs)
ax.hist(posterior_samples, bins=60, density=True, alpha=0.7,
        color='steelblue', edgecolor='white', label='MCMC samples')
ax.plot(theta, exact, 'r-', linewidth=2.5, label='Exact Beta posterior')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('MCMC vs Exact Posterior', fontsize=13)
ax.legend()

# Autocorrelation
ax = axes[2]
max_lag = 50
acf = np.correlate(posterior_samples - posterior_samples.mean(),
                    posterior_samples - posterior_samples.mean(), mode='full')
acf = acf[len(acf)//2:]
acf = acf / acf[0]
ax.bar(range(max_lag), acf[:max_lag], color='steelblue', alpha=0.7)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('MCMC Autocorrelation', fontsize=13)
ax.axhline(0, color='gray', linestyle='-')

plt.tight_layout()
plt.savefig('mcmc_demo.png', dpi=150)
plt.show()
```

MCMC 生成的直方图与精确的 Beta 后验高度吻合，验证了采样器的有效性。迹线图展示了马尔可夫链在参数空间中的探索过程，而自相关图则揭示了连续样本之间何时趋于独立（自相关越短，采样效率越高）。

## 系列回顾

历经八篇文章，我们从零开始构建了概率与统计的完整体系：

1. **公理体系** 为我们提供了严谨的不确定性度量基础。
2. **随机变量** 将抽象结果转化为可计算的数值。
3. **期望与矩** 将复杂的分布压缩为简洁的摘要统计量。
4. **联合分布** 处理多个变量及其相互依赖关系。
5. **极限定理** 解释了为何样本均值有效，以及高斯分布为何如此普遍。
6. **估计理论** 展示了如何从数据中最优地提取参数。
7. **假设检验** 提供了在不确定性下做出决策的工具。
8. **贝叶斯推断** 提供了一个用数据持续更新信念的连贯框架。

每一篇文章都建立在前文的基础之上，共同构成了现代数据科学与机器学习的数学主干。文中涵盖的分布、定理与技术绝非历史遗迹——它们是每一位构建数据驱动系统者的日常工具。

前方的道路通向诸多方向：多元分析、时间序列、因果推断、信息论、统计学习理论。但本系列所奠定的基础，足以让你自信地深入研读上述任何一个领域。

## 总结
| 概念 | 公式 | 作用 |
|---|---|---|
| 贝叶斯公式 | $p(\theta\mid\mathbf{x}) \propto p(\mathbf{x}\mid\theta)p(\theta)$ | 基本更新规则 |
| 共轭先验 | 先验与后验同属一族 | 后验具闭式解 |
| Beta-二项 | $\text{Beta}(\alpha+x, \beta+n-x)$ | 比例估计 |
| 正态-正态 | 精度加权平均 | 均值估计 |
| 后验均值 | $E[\theta\mid\mathbf{x}]$ | 平方损失下最优 |
| MAP | $\arg\max p(\theta\mid\mathbf{x})$ | = MLE + 正则化 |
| 可信区间 | $P(\theta \in [a,b]\mid\mathbf{x}) = 0.95$ | 直接概率陈述 |
| 预测分布 | $p(\tilde{x}\mid\mathbf{x}) = \int p(\tilde{x}\mid\theta)p(\theta\mid\mathbf{x})d\theta$ | 带不确定性的预测 |
| MCMC | 从后验采样 | 无闭式解时的替代方案 |
