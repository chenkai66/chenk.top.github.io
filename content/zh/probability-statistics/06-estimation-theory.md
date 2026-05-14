---
title: "概率与统计（六）：参数估计——极大似然估计、最大后验估计与偏差-方差分解"
date: 2024-08-26 09:00:00
tags:
  - Probability
  - Statistics
  - Maximum Likelihood
  - Bias-Variance
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "从矩估计到极大似然估计与最大后验估计的点估计方法，涵盖费希尔信息量、克拉美–罗下界，以及解释过拟合与欠拟合的偏差-方差分解。"
disableNunjucks: true
series_order: 6
translationKey: "probability-statistics-6"
---
我们此前构建的所有内容——分布、期望、极限定理——都假设参数已知。高斯分布有均值 $\mu$ 和方差 $\sigma^2$；二项分布有 $n$ 次试验和成功概率 $p$。但在实际中，你并不知道 $\mu$ 或 $p$，只能通过观测数据来推断它们。

这就是**估计理论**：它架起了概率论（参数给定）与统计学（参数待推断）之间的桥梁，也是机器学习的理论根基。每次训练模型，本质上都是从数据中估计参数，而估计质量直接决定了模型是泛化良好还是陷入过拟合。

---

## 设定：估计量 vs. 估计值

假设我们观测到独立同分布（i.i.d.）样本 $x_1, x_2, \ldots, x_n$，其来自某个分布 $p(x|\theta)$，其中 $\theta$ 是未知参数（或参数向量）。

![置信区间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-confidence-interval.png)

一个**估计量** $\hat{\theta}$ 是数据的函数：$\hat{\theta} = g(X_1, \ldots, X_n)$。由于数据本身是随机的，估计量也是一个随机变量。而一个**估计值**则是该函数在某组具体观测数据上计算出的数值：$\hat{\theta}(x_1, \ldots, x_n)$。

**记号惯例：** 带帽子的 $\hat{\theta}$ 表示估计量或估计值；不带帽子的 $\theta$ 表示真实参数。

## 估计量的性质

### 偏差（Bias）

估计量的**偏差**定义为：

![偏差-方差权衡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-bias-variance.png)
$$
\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta.
$$
若对所有 $\theta$ 都有 $E[\hat{\theta}] = \theta$，即 $\text{Bias}(\hat{\theta}) = 0$，则称该估计量是**无偏的**。

**例：** 样本均值 $\bar{X} = \frac{1}{n}\sum X_i$ 对 $\mu$ 是无偏的：
$$
E[\bar{X}] = \frac{1}{n}\sum E[X_i] = \mu. \quad \checkmark
$$
而样本方差 $S^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ 则是**有偏的**：
$$
E\left[\frac{1}{n}\sum(X_i - \bar{X})^2\right] = \frac{n-1}{n}\sigma^2 \neq \sigma^2.
$$
因此，无偏样本方差使用 $n-1$ 作为分母：
$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2.
$$
*偏差证明：* 展开：
$$
\sum(X_i - \bar{X})^2 = \sum X_i^2 - n\bar{X}^2.
$$
取期望：
$$
E\left[\sum X_i^2\right] = n(\sigma^2 + \mu^2), \quad E[n\bar{X}^2] = n\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2.
$$
于是 $E\left[\sum(X_i - \bar{X})^2\right] = n\sigma^2 + n\mu^2 - \sigma^2 - n\mu^2 = (n-1)\sigma^2$。除以 $n$ 得 $\frac{n-1}{n}\sigma^2$；除以 $n-1$ 则得 $\sigma^2$。$\blacksquare$

### 相合性（Consistency）

若当 $n \to \infty$ 时 $\hat{\theta}_n \xrightarrow{P} \theta$，则称估计量是**相合的**。

由弱大数定律（WLLN），样本均值 $\bar{X}$ 对 $\mu$ 是相合的。同样，无论是用 $1/n$ 还是 $1/(n-1)$ 作分母的样本方差，对 $\sigma^2$ 也都是相合的。

### 有效性（Efficiency）

在所有关于 $\theta$ 的无偏估计量中，方差最小者称为最**有效**的。克拉美–罗下界（Cramér–Rao lower bound，见下文推导）给出了方差可能达到的最小值。

### 充分性（Sufficiency）

若统计量 $T(X_1, \ldots, X_n)$ 的条件分布 $p(\mathbf{x}|T)$ 不依赖于 $\theta$，则称 $T$ 是 $\theta$ 的**充分统计量**。直观地说，$T$ 已捕获了数据中关于 $\theta$ 的全部信息——一旦知道 $T$，原始数据即可丢弃而不损失任何信息。

**例：** 若 $X_i \sim \text{Bernoulli}(p)$，则和 $T = \sum X_i$ 是 $p$ 的充分统计量。知道“10 次试验中成功 7 次”就足够了；具体序列（如 1101110100 或 1110101100）不提供关于 $p$ 的额外信息。

## 矩估计法（Method of Moments）

![最大似然估计：登山者寻找参数p](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/06-maximum-likelihood-estimation-mountain-climber-finding-the-p.jpg)

最简单的估计方法：将样本矩与总体矩匹配。

**步骤：**
1. 将参数表示为总体矩的函数：$\theta = h(\mu_1, \mu_2, \ldots)$，其中 $\mu_k = E[X^k]$；
2. 用样本矩替代总体矩：$\hat{\mu}_k = \frac{1}{n}\sum X_i^k$；
3. 代入：$\hat{\theta}_{\text{MoM}} = h(\hat{\mu}_1, \hat{\mu}_2, \ldots)$。

**例：伽马分布。** $X \sim \text{Gamma}(\alpha, \beta)$，满足 $E[X] = \alpha/\beta$，$E[X^2] = \alpha(\alpha+1)/\beta^2$。

由 $\mu_1 = \alpha/\beta$ 及 $\mu_2 - \mu_1^2 = \alpha/\beta^2$（即方差）可得：
$$
\hat{\beta}_{\text{MoM}} = \frac{\bar{X}}{S^2}, \qquad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}.
$$
矩估计量易于计算，但通常不是最有效的。它常被用作初始化起点或合理性检验。

## 极大似然估计（Maximum Likelihood Estimation）

![估计方法比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-estimation-comparison.png)

![偏差-方差权衡：箭靶散射与集中](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/06-bias-variance-tradeoff-archery-target-scattered-vs-centered-.jpg)

### 似然函数（Likelihood Function）

给定 i.i.d. 观测数据 $x_1, \ldots, x_n$ 来自 $p(x|\theta)$，其**似然函数**为：
$$
L(\theta) = \prod_{i=1}^n p(x_i | \theta).
$$
其**对数似然函数**为：
$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln p(x_i | \theta).
$$
对数似然更易处理（求和比连乘简单），且因 $\ln$ 单调递增，最大化 $\ell$ 等价于最大化 $L$。

### 极大似然估计量（MLE）

**极大似然估计量**是使似然函数最大的 $\theta$ 值：

![最大似然估计的似然曲面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-mle-likelihood.png)
$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta).
$$
在常规情形下，MLE 满足**得分方程（score equation）**：
$$
\frac{\partial \ell}{\partial \theta} = 0.
$$
### 例 1：伯努利分布

$X_i \sim \text{Bernoulli}(p)$，观测到 $n$ 次试验中 $k$ 次成功。
$$
\ell(p) = \sum_{i=1}^n [x_i \ln p + (1-x_i) \ln(1-p)] = k \ln p + (n-k) \ln(1-p).
$$

$$
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0.
$$

$$
k(1-p) = (n-k)p \implies k = np \implies \hat{p}_{\text{MLE}} = \frac{k}{n} = \bar{X}.
$$
MLE 即样本比例，自然且直观。

### 例 2：高斯分布（两参数）

$X_i \sim \mathcal{N}(\mu, \sigma^2)$。
$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2.
$$
令 $\partial\ell/\partial\mu = 0$：
$$
\frac{1}{\sigma^2}\sum(x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \bar{X}.
$$
令 $\partial\ell/\partial\sigma^2 = 0$：
$$
-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \bar{X})^2.
$$
注意：$\sigma^2$ 的 MLE 分母为 $n$ 而非 $n-1$。它是有偏的（偏差因子为 $(n-1)/n$），但相合且有效。当 $n$ 很大时，差异可忽略。

### 例 3：泊松分布

$X_i \sim \text{Poisson}(\lambda)$。
$$
\ell(\lambda) = \sum_{i=1}^n [x_i \ln\lambda - \lambda - \ln(x_i!)] = \left(\sum x_i\right)\ln\lambda - n\lambda - \sum\ln(x_i!).
$$

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{\text{MLE}} = \bar{X}.
$$
再次得到样本均值——这很合理，因为泊松分布满足 $E[X] = \lambda$。

### 例 4：均匀分布

$X_i \sim \text{Uniform}(0, \theta)$，$\theta > 0$ 未知。
$$
L(\theta) = \prod_{i=1}^n \frac{1}{\theta} \cdot \mathbf{1}_{0 \leq x_i \leq \theta} = \frac{1}{\theta^n} \cdot \mathbf{1}_{\theta \geq x_{(n)}}
$$
其中 $x_{(n)} = \max(x_1, \ldots, x_n)$。当 $\theta \geq x_{(n)}$ 时，$L(\theta) = 1/\theta^n$ 关于 $\theta$ 单调递减，故 $L$ 在 $\hat{\theta}_{\text{MLE}} = x_{(n)} = \max_i x_i$ 处取得最大值。

这是一个有趣案例：MLE 是**有偏的**：$E[X_{(n)}] = \frac{n}{n+1}\theta < \theta$。MLE 系统性低估，因为 $n$ 个样本的最大值永远不能超过 $\theta$，却可能远小于 $\theta$。无偏估计量为 $\frac{n+1}{n} X_{(n)}$。

此例还说明：MLE 并不总满足光滑的得分方程——此处似然函数在 $\theta = x_{(n)}$ 处不连续。

## MLE 的性质

在正则性条件（模型光滑、可识别）下：

1. **相合性：** $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$（收敛于真实参数）。
2. **渐近正态性：** $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$，其中 $I(\theta_0)$ 是费希尔信息量。
3. **渐近有效性：** MLE 渐近达到克拉美–罗下界。
4. **不变性（Invariance）：** 若 $\hat{\theta}$ 是 $\theta$ 的 MLE，则 $g(\hat{\theta})$ 是 $g(\theta)$ 的 MLE。

*不变性证明：* 若 $\hat{\theta} = \arg\max_\theta L(\theta)$，且 $\phi = g(\theta)$ 是单射，则 $\hat{\phi} = g(\hat{\theta}) = \arg\max_\phi L(g^{-1}(\phi))$。对非单射 $g$，直接定义 $\hat{\phi} = g(\hat{\theta})$；轮廓似然在此处取最大值。$\blacksquare$

**不变性示例：** 若 $\hat{\sigma}^2_{\text{MLE}}$ 是 $\sigma^2$ 的 MLE，则 $\hat{\sigma}_{\text{MLE}} = \sqrt{\hat{\sigma}^2_{\text{MLE}}}$ 是 $\sigma$ 的 MLE。无需重新推导，只需应用变换。

**警告：** 不变性是 MLE 特有性质，一般不适用于无偏估计量。若 $\hat{\theta}$ 对 $\theta$ 无偏，则 $g(\hat{\theta})$ 通常**不是** $g(\theta)$ 的无偏估计量（由 Jensen 不等式，除非 $g$ 是线性的）。

## 费希尔信息量与克拉美–罗下界（CRLB）

### 费希尔信息量（Fisher Information）

**得分函数（score function）** 定义为 $s(\theta) = \frac{\partial}{\partial\theta} \ln p(X|\theta)$。

![费希尔信息量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-fisher-information.png)

在正则性条件下，$E[s(\theta)] = 0$。

**费希尔信息量** 是得分的方差：
$$
I(\theta) = E\left[\left(\frac{\partial \ln p(X|\theta)}{\partial\theta}\right)^2\right] = -E\left[\frac{\partial^2 \ln p(X|\theta)}{\partial\theta^2}\right].
$$
第二个等式（通过交换微分与期望得到）提供了更便捷的计算公式。对 $n$ 个 i.i.d. 观测，$I_n(\theta) = n \cdot I_1(\theta)$。

**例：伯努利分布。** $\ln p(x|p) = x\ln p + (1-x)\ln(1-p)$。
$$
\frac{\partial^2}{\partial p^2}\ln p(x|p) = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}.
$$

$$
I_1(p) = -E\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}.
$$
### 克拉美–罗下界（Cramér–Rao Lower Bound）

**定理（克拉美–罗）：** 对任意 $\theta$ 的无偏估计量 $\hat{\theta}$，有：
$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I_1(\theta)}.
$$
*证明概要：* 对协方差 $\text{Cov}(\hat{\theta}, s(\theta))$ 应用柯西–施瓦茨不等式，其中 $s$ 是总得分。因 $E[s] = 0$，协方差等于 $E[\hat{\theta} \cdot s]$，再利用无偏性及积分下微分可得其值为 1。于是柯西–施瓦茨给出 $1 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(s) = \text{Var}(\hat{\theta}) \cdot I_n(\theta)$。$\blacksquare$

**例：** 对伯努利分布，CRLB 为 $\frac{p(1-p)}{n}$。MLE $\hat{p} = \bar{X}$ 的方差恰为 $\frac{p(1-p)}{n}$，精确达到该下界。故 MLE 是有效的。

## 最大后验估计（MAP）

### 从 MLE 到 MAP

MLE 将 $\theta$ 视为固定未知量。**最大后验（MAP）估计**则将 $\theta$ 视为具有**先验分布** $p(\theta)$ 的随机变量。

由贝叶斯定理：
$$
p(\theta | x_1, \ldots, x_n) \propto p(x_1, \ldots, x_n | \theta) \cdot p(\theta) = L(\theta) \cdot p(\theta).
$$
MAP 估计量最大化后验分布：
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\ell(\theta) + \ln p(\theta)\right].
$$
### 与正则化的联系

MAP 估计等价于带惩罚项的 MLE —— 对数先验起正则化作用。

| 先验分布 | 惩罚项 | 对应的 ML 正则化 |
|---|---|---|
| $\theta \sim \mathcal{N}(0, \tau^2)$ | $-\frac{\theta^2}{2\tau^2}$ | L2 正则化（岭回归） |
| $\theta \sim \text{Laplace}(0, b)$ | $-\frac{\mid\theta\mid}{b}$ | L1 正则化（Lasso） |

**例：** 对已知方差 $\sigma^2$ 的高斯分布，对其均值 $\mu$ 施加高斯先验 $\mu \sim \mathcal{N}(\mu_0, \tau^2)$。

数据：$X_i \sim \mathcal{N}(\mu, \sigma^2)$；先验：$\mu \sim \mathcal{N}(\mu_0, \tau^2)$。
$$
\hat{\mu}_{\text{MAP}} = \arg\max_\mu \left[-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2 - \frac{(\mu - \mu_0)^2}{2\tau^2}\right].
$$
求导并令其为零：
$$
\frac{\sum(x_i - \mu)}{\sigma^2} - \frac{\mu - \mu_0}{\tau^2} = 0.
$$

$$
\hat{\mu}_{\text{MAP}} = \frac{\frac{n}{\sigma^2}\bar{X} + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}} = \frac{n\tau^2\bar{X} + \sigma^2\mu_0}{n\tau^2 + \sigma^2}.
$$
这是样本均值 $\bar{X}$ 与先验均值 $\mu_0$ 的**加权平均**。当 $n$ 很大时，数据主导，MAP 收敛至 MLE；当 $n$ 很小时，先验将估计拉向 $\mu_0$。

## 偏差-方差分解（Bias-Variance Decomposition）

### 均方误差（Mean Squared Error）

估计量的**均方误差（MSE）** 定义为：
$$
\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2].
$$
**定理（偏差-方差分解）：**
$$
\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta}).
$$
*证明：* 记 $b = E[\hat{\theta}] - \theta$ 为偏差。则：
$$
\text{MSE} = E[(\hat{\theta} - \theta)^2] = E[(\hat{\theta} - E[\hat{\theta}] + E[\hat{\theta}] - \theta)^2]
$$

$$
= E[(\hat{\theta} - E[\hat{\theta}])^2 + 2(\hat{\theta} - E[\hat{\theta}])(E[\hat{\theta}] - \theta) + (E[\hat{\theta}] - \theta)^2]
$$

$$
= \text{Var}(\hat{\theta}) + 2(E[\hat{\theta}] - E[\hat{\theta}]) \cdot b + b^2 = \text{Var}(\hat{\theta}) + b^2. \quad \blacksquare
$$
### 与机器学习的联系

在监督学习中，设预测器为 $\hat{f}$，对测试点 $x$，真值为 $y = f(x) + \varepsilon$，其中 $E[\varepsilon] = 0$，$\text{Var}(\varepsilon) = \sigma^2$：
$$
E[(\hat{f}(x) - y)^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}.
$$
- **高偏差** → 欠拟合：模型过于简单，无法捕捉真实模式；
- **高方差** → 过拟合：模型过度拟合训练数据中的噪声；
- **权衡（Tradeoff）：** 增加模型复杂度会降低偏差但提高方差；最优模型需平衡二者。

正则化（即带先验的 MAP）有意引入偏差以降低方差，从而常能降低总 MSE。

## Python：偏差-方差可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 真实函数
def f_true(x):
    return np.sin(2 * np.pi * x)

# 生成数据集并拟合不同阶数的多项式
n_train = 20
n_datasets = 200
x_test = np.linspace(0, 1, 100)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

degrees = [1, 3, 5, 10, 15, 19]

for idx, degree in enumerate(degrees):
    ax = axes[idx // 3, idx % 3]
    predictions = np.zeros((n_datasets, len(x_test)))

    for d in range(n_datasets):
        x_train = np.random.uniform(0, 1, n_train)
        y_train = f_true(x_train) + np.random.normal(0, 0.3, n_train)
        coeffs = np.polyfit(x_train, y_train, degree)
        predictions[d] = np.polyval(coeffs, x_test)

    # 绘制部分拟合曲线（前 20 条）
    for d in range(min(20, n_datasets)):
        ax.plot(x_test, predictions[d], 'b-', alpha=0.1, linewidth=0.5)

    # 绘制平均预测
    mean_pred = predictions.mean(axis=0)
    ax.plot(x_test, mean_pred, 'r-', linewidth=2, label='E[$\\hat{f}$]')

    # 绘制真实函数
    ax.plot(x_test, f_true(x_test), 'k--', linewidth=2, label='f(x)')

    # 计算偏差平方与方差
    bias_sq = np.mean((mean_pred - f_true(x_test))**2)
    variance = np.mean(predictions.var(axis=0))

    ax.set_title(f'Degree {degree}\nBias$^2$={bias_sq:.3f}, Var={variance:.3f}',
                 fontsize=11)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=9)

plt.suptitle('Bias-Variance Tradeoff: Polynomial Regression', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('bias_variance.png', dpi=150, bbox_inches='tight')
plt.show()
```

该可视化直接展示了权衡关系。在阶数 1（欠拟合）时，平均预测（红色）远离真实函数（黑色虚线）——高偏差；但各条拟合曲线（蓝色）紧密聚集——低方差。在阶数 19（过拟合）时，平均预测与真实函数吻合良好——低偏差；但各条拟合曲线差异巨大——高方差。最佳折中点介于两者之间。

```python
# 计算偏差-方差权衡曲线
degrees = range(1, 20)
biases = []
variances = []

for degree in degrees:
    predictions = np.zeros((n_datasets, len(x_test)))
    for d in range(n_datasets):
        x_train = np.random.uniform(0, 1, n_train)
        y_train = f_true(x_train) + np.random.normal(0, 0.3, n_train)
        coeffs = np.polyfit(x_train, y_train, degree)
        predictions[d] = np.polyval(coeffs, x_test)
    mean_pred = predictions.mean(axis=0)
    biases.append(np.mean((mean_pred - f_true(x_test))**2))
    variances.append(np.mean(predictions.var(axis=0)))

mse = np.array(biases) + np.array(variances) + 0.3**2  # + 噪声方差

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(degrees), biases, 'b-o', label='Bias$^2$', markersize=5)
ax.plot(list(degrees), variances, 'r-o', label='Variance', markersize=5)
ax.plot(list(degrees), mse, 'k-o', label='Total MSE', markersize=5)
ax.axhline(y=0.3**2, color='gray', linestyle=':', label='Noise ($\\sigma^2$)')
ax.set_xlabel('Polynomial Degree', fontsize=13)
ax.set_ylabel('Error', fontsize=13)
ax.set_title('Bias-Variance Tradeoff', fontsize=14)
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bias_variance_curve.png', dpi=150)
plt.show()
```

## 充分统计量与数据降维

### 因子分解定理（Factorization Theorem）

统计量 $T(\mathbf{X})$ 是 $\theta$ 的**充分统计量**，当且仅当似然函数可分解为：
$$
p(\mathbf{x} | \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})
$$
其中 $g$ 仅通过 $T$ 依赖于数据，而 $h$ 不依赖于 $\theta$。

**例：** 对 $X_i \sim \text{Poisson}(\lambda)$：
$$
p(\mathbf{x}|\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} e^{-n\lambda}}{\prod x_i!} = \underbrace{\lambda^{\sum x_i} e^{-n\lambda}}_{g(\sum x_i, \lambda)} \cdot \underbrace{\frac{1}{\prod x_i!}}_{h(\mathbf{x})}.
$$
故 $T = \sum X_i$ 是 $\lambda$ 的充分统计量。一旦知道总频数，单个观测不再提供关于 $\lambda$ 的额外信息。

### 拉奥–布莱克韦尔定理（Rao–Blackwell Theorem）

**定理：** 若 $\hat{\theta}$ 是任意无偏估计量，$T$ 是充分统计量，则 $\tilde{\theta} = E[\hat{\theta} | T]$ 同样无偏，且方差更小（或相等）：
$$
\text{Var}(\tilde{\theta}) \leq \text{Var}(\hat{\theta}).
$$
*证明：* 无偏性：$E[\tilde{\theta}] = E[E[\hat{\theta}|T]] = E[\hat{\theta}] = \theta$（塔性质）。对方差，使用全方差公式：
$$
\text{Var}(\hat{\theta}) = E[\text{Var}(\hat{\theta}|T)] + \text{Var}(E[\hat{\theta}|T]) = E[\text{Var}(\hat{\theta}|T)] + \text{Var}(\tilde{\theta}) \geq \text{Var}(\tilde{\theta}). \quad \blacksquare
$$
该定理指出：**务必对充分统计量做条件化。** 它只会带来好处，绝无坏处。

## 交叉验证：实用的偏差-方差工具

偏差-方差分解是理论框架。实践中，你无法直接计算偏差或方差（它们依赖于未知的真实函数）。**交叉验证（Cross-validation）** 是实用替代方案。

**k 折交叉验证：**
1. 将数据分为 $k$ 个大小相等的子集（fold）；
2. 对每个子集 $i$：在其余 $k-1$ 个子集上训练，在第 $i$ 个子集上评估；
3. 对 $k$ 次测试误差取平均。

它在无需独立测试集的前提下估计测试误差（含偏差与方差效应）。常用选择：$k = 5$ 或 $k = 10$。

**留一法交叉验证（LOOCV）：** $k = n$。每次留出一个观测。偏差低（训练集几乎为全数据集），但方差高（$n$ 个训练集高度相似，导致 $n$ 个误差估计相关）。

偏差-方差权衡亦适用于交叉验证自身：
- $k$ 小：偏差高（训练集小），方差低；
- $k$ 大：偏差低，方差高（估计相关）；
- $k = 5$ 或 $10$ 在多数场景下是良好折中。

## 指数族：统一框架

我们研究的大多数分布属于**指数族（exponential family）**，其通式为：
$$
p(x|\theta) = h(x) \exp\left(\eta(\theta)^T T(x) - A(\theta)\right)
$$
其中 $T(x)$ 是充分统计量，$\eta(\theta)$ 是自然参数，$A(\theta)$ 是对数配分函数。

| 分布 | $T(x)$ | $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Bernoulli($p$) | $x$ | $\ln\frac{p}{1-p}$ | $\ln(1+e^\eta)$ |
| Poisson($\lambda$) | $x$ | $\ln\lambda$ | $e^\eta$ |
| Normal($\mu$, known $\sigma^2$) | $x$ | $\mu/\sigma^2$ | $\eta^2\sigma^2/2$ |
| Exponential($\lambda$) | $x$ | $-\lambda$ | $-\ln(-\eta)$ |

对数配分函数 $A(\theta)$ 生成矩：
$$
E[T(X)] = A'(\eta), \qquad \text{Var}(T(X)) = A''(\eta).
$$
对指数族，MLE 由充分统计量唯一确定，且等价于基于 $T(X)$ 的矩估计。共轭先验也有自然形式，且 MLE 总存在且唯一（在温和条件下）。

## 数值 MLE：闭式解不存在时

许多模型无闭式 MLE 解。此时需数值优化对数似然。

### 牛顿–拉弗森法（Newton-Raphson）

求解得分方程 $\ell'(\theta) = 0$ 的迭代格式为：
$$
\theta^{(t+1)} = \theta^{(t)} - \frac{\ell'(\theta^{(t)})}{\ell''(\theta^{(t)})}.
$$
多元情形下：
$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - [\nabla^2 \ell(\boldsymbol{\theta}^{(t)})]^{-1} \nabla \ell(\boldsymbol{\theta}^{(t)}).
$$
### 费希尔打分法（Fisher Scoring）

将观测 Hessian $\nabla^2 \ell$ 替换为其期望 $-I(\theta)$（费希尔信息矩阵）：
$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + [I(\boldsymbol{\theta}^{(t)})]^{-1} \nabla \ell(\boldsymbol{\theta}^{(t)}).
$$
费希尔打分法比牛顿–拉弗森更稳定，因费希尔信息矩阵必为半正定（保证沿上升方向移动）。对指数族，二者重合。

### EM 算法（Expectation-Maximization）

对含隐变量（latent variables）的模型，直接 MLE 常不可行。**EM 算法**交替执行：

1. **E 步：** 计算 $Q(\theta | \theta^{(t)}) = E_{\mathbf{Z}|\mathbf{X}, \theta^{(t)}}[\ell(\theta; \mathbf{X}, \mathbf{Z})]$ —— 完整数据对数似然的期望；
2. **M 步：** $\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})$ —— 最大化期望对数似然。

每步保证对数似然不减：$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$。算法收敛至局部极大值。

EM 用于高斯混合模型、隐马尔可夫模型、因子分析等——凡边际化隐变量使似然难解，但条件化使其易解的模型。

### MLE 的梯度下降法

在高维模型（如神经网络）中，计算 Hessian 或费希尔信息代价过高。此时采用一阶方法：
$$
\theta^{(t+1)} = \theta^{(t)} + \eta \nabla \ell(\theta^{(t)})
$$
其中 $\eta$ 是学习率。这是对数似然的**梯度上升**——与深度学习所称“训练”完全一致。MLE 框架提供了理论依据：我们在最大化模型对观测数据的似然。

在随机设置（mini-batch）下，使用随机梯度上升；第 5 篇文章中的中心极限定理（CLT）保证梯度估计的噪声近似服从方差为 $O(1/B)$ 的高斯分布。

## 总结

| 方法 | 公式 | 关键性质 |
|---|---|---|
| 矩估计法 | 匹配 $\hat{\mu}_k$ 与理论矩 | 简单，但未必最有效 |
| MLE | $\arg\max_\theta \sum \ln p(x_i\mid\theta)$ | 相合，渐近有效 |
| MAP | $\arg\max_\theta [\ell(\theta) + \ln p(\theta)]$ | MLE + 正则化 |
| 费希尔信息量 | $I(\theta) = -E[\partial^2 \ell/\partial\theta^2]$ | 衡量数据所含信息量 |
| CRLB | $\text{Var}(\hat{\theta}) \geq 1/(nI_1(\theta))$ | 方差的下界 |
| 拉奥–布莱克韦尔 | $E[\hat{\theta}\midT]$ 改进 $\hat{\theta}$ | 对充分统计量做条件化 |
| 偏差-方差 | $\text{MSE} = \text{Bias}^2 + \text{Var}$ | 解释过拟合/欠拟合 |

## 下一步

参数估计给出的是**点估计**——参数的单一最优猜测。但我们应有多大的信心？下一篇文章将探讨**假设检验与置信区间**：量化不确定性的框架、控制错误率的方法，以及基于数据做出原则性决策的工具。我们还将揭示为何 p 值常被误解，以及如何规避最常见的统计陷阱。
