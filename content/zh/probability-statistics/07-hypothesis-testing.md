---
title: "概率与统计（七）：假设检验——p 值、置信区间及其全部陷阱"
date: 2024-08-28 09:00:00
tags:
  - Probability
  - Statistics
  - Hypothesis Testing
  - Confidence Intervals
  - A/B Testing
categories:
  - Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "对假设检验、p 值、I 类/II 类错误、置信区间及多重检验校正的严谨讲解——包括连资深实践者都会陷入的常见误读，并附 Python 实现代码。"
disableNunjucks: true
series_order: 7
translationKey: "probability-statistics-7"
---

你已估计了一个参数，也量化了偏差-方差权衡。现在，驱动绝大多数应用统计学的核心问题浮现出来：“这个效应是真实的，还是仅仅是噪声？”

假设检验正是回答这一问题的形式化框架。它同时也是统计学中最常被误解的部分。大量论文专门探讨研究者如何误读 p 值、显著性阈值为何是任意设定的，以及多重检验问题如何推高假阳性发现率。对理论原理与常见陷阱的双重理解，对任何从事数据分析工作的人来说都至关重要。

## 框架


![Rejection regions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-rejection-region.png)

### 设定假设

假设检验始于两个相互竞争的主张：

- **零假设** $H_0$：即“默认”或“并无特殊事件发生”的主张，通常代表现状。
- **备择假设** $H_1$（或 $H_a$）：我们希望证实为真的主张。

**示例：**
- 药物试验：$H_0: \mu_{\text{drug}} = \mu_{\text{placebo}}$ vs $H_1: \mu_{\text{drug}} > \mu_{\text{placebo}}$
- A/B 测试：$H_0: p_A = p_B$ vs $H_1: p_A \neq p_B$
- 质量控制：$H_0: \mu = 10.0$ vs $H_1: \mu \neq 10.0$

备择假设可以是**单侧的**（$H_1: \theta > \theta_0$）或**双侧的**（$H_1: \theta \neq \theta_0$）。

### 检验统计量与拒绝域

一个**检验统计量** $T = T(X_1, \ldots, X_n)$ 将数据汇总为一个单一数值，用以度量对 $H_0$ 的反证强度。

**拒绝域** $R$ 是 $T$ 的一组取值，当 $T$ 落入其中时，我们拒绝 $H_0$：

- 若 $T \in R$：拒绝 $H_0$
- 若 $T \notin R$：不拒绝 $H_0$（这**不等于**接受 $H_0$）

### 显著性水平

**显著性水平** $\alpha$ 是在 $H_0$ 实际为真时仍拒绝 $H_0$ 的最大概率：

$$\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true}) = P(T \in R \mid H_0).$$

常用取值：$\alpha = 0.05, 0.01, 0.001$。拒绝域需据此约束选定。

## 假设检验中的错误

| | $H_0$ 为真 | $H_0$ 为假 |
|---|---|---|
| **拒绝 $H_0$** | I 类错误（假阳性） | 正确（真阳性） |
| **不拒绝 $H_0$** | 正确（真阴性） | II 类错误（假阴性） |

![Statistical power curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-power-curve.png)


- **I 类错误率** = $\alpha$ = P(拒绝 $H_0$ | $H_0$ 为真)
- **II 类错误率** = $\beta$ = P(不拒绝 $H_0$ | $H_0$ 为假)
- **统计功效（Power）** = $1 - \beta$ = P(拒绝 $H_0$ | $H_0$ 为假)

我们通过选择 $\alpha$ 直接控制 I 类错误。而功效则取决于：
- 真实效应量（效应越大越易检测）
- 样本量 $n$（数据越多，功效越高）
- 显著性水平 $\alpha$（阈值越宽松，功效越高，但假阳性也越多）
- 方差 $\sigma^2$（噪声越小，功效越高）

### 权衡关系

降低 $\alpha$ 可减少假阳性，但会增加假阴性（$\beta$ 上升，功效下降）。在样本量固定的前提下，无法同时最小化二者。唯一“免费午餐”是获取更多数据。

## p 值


![Type I and Type II errors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-error-types.png)


![Ab testing laboratory two beakers being compared scientific](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/07-ab-testing-laboratory-two-beakers-being-compared-scientific-.jpg)

### 定义

**p 值**是在 $H_0$ 成立的前提下，观察到当前检验统计量或更极端值的概率：

$$p\text{-value} = P(T \geq t_{\text{obs}} \mid H_0) \quad \text{(单侧)}$$

$$p\text{-value} = P(|T| \geq |t_{\text{obs}}| \mid H_0) \quad \text{(双侧)}$$

**决策规则：** 若 $p\text{-value} \leq \alpha$，则拒绝 $H_0$。

p 值是一个随机变量（依赖于数据）。在 $H_0$ 下，p 值服从 $\text{Uniform}(0, 1)$ 分布——这正是 $P(p\text{-value} \leq 0.05 \mid H_0) = 0.05$ 的原因。

### p 值**是**什么

p 值是：在**假设 $H_0$ 为真**的前提下，观察到如此极端或更极端数据的概率。

### p 值**不是**什么

以下均为常见且危险的误读：

1. **不是** $H_0$ 为真的概率。（$P(H_0 | \text{data})$ 需借助贝叶斯定理和先验分布。）
2. **不是**结果由偶然性导致的概率。（p 值的计算**前提是**结果由偶然性导致。）
3. **不是**犯错的概率。（错误率是 $\alpha$，而非 p 值。）
4. **不是**效应量的度量。（一个微小、无实际意义的效应，在足够大数据下也可能有 $p < 0.001$。）
5. **不是**可重复性的度量。（$p = 0.04$ 的结果并不意味着其有 96% 的概率可复现。）

## 常见检验方法


![Effect size visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-effect-size.png)

### Z 检验

当 $\sigma$ 已知，且总体服从正态分布（或 $n$ 足够大满足中心极限定理）时：

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1) \quad \text{under } H_0: \mu = \mu_0.$$

**示例。** 某工厂声称螺栓平均长度为 10.0 mm，已知 $\sigma = 0.5$ mm。抽取 $n = 25$ 个样本，得 $\bar{x} = 10.2$。在 $\alpha = 0.05$ 水平下检验 $H_0: \mu = 10$ vs $H_1: \mu \neq 10$。

$$Z = \frac{10.2 - 10.0}{0.5/\sqrt{25}} = \frac{0.2}{0.1} = 2.0.$$

$p\text{-value} = 2 \cdot P(Z > 2.0) = 2 \times 0.0228 = 0.0456 < 0.05$。

拒绝 $H_0$。存在统计上显著的证据表明平均长度不同于 10.0 mm。

### t 检验

当 $\sigma$ 未知（通常情形），用样本标准差 $S$ 替代：

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1} \quad \text{under } H_0$$

其中 $t_{n-1}$ 是自由度为 $n-1$ 的学生 t 分布。

t 分布比正态分布具有更重的尾部，反映了因估计 $\sigma$ 而引入的额外不确定性。当 $n \to \infty$ 时，$t_{n-1} \to \mathcal{N}(0, 1)$。

### 双样本 t 检验

比较两组均值。若 $X_1, \ldots, X_{n_1} \sim \mathcal{N}(\mu_1, \sigma^2)$ 且 $Y_1, \ldots, Y_{n_2} \sim \mathcal{N}(\mu_2, \sigma^2)$（假设方差相等）：

$$T = \frac{\bar{X} - \bar{Y}}{S_p \sqrt{1/n_1 + 1/n_2}}, \quad S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$$

其中 $S_p$ 为合并标准差。在 $H_0: \mu_1 = \mu_2$ 下，$T \sim t_{n_1 + n_2 - 2}$。

**Welch’s t 检验** 通过调整自由度来处理方差不等的情形——实践中应将其作为默认选项。

### 配对 t 检验

当观测值天然成对出现（如前后对比、左右对比）时，计算差值 $D_i = X_i - Y_i$，再对 $D_i$ 进行单样本 t 检验：

$$T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.$$

### 卡方独立性检验

针对列联表中的分类数据，检验两个变量是否独立。

**检验统计量：**

$$\chi^2 = \sum_{\text{cells}} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

其中 $O_{ij}$ 是观测频数，$E_{ij} = \frac{(\text{第 } i \text{ 行合计})(\text{第 } j \text{ 列合计})}{n}$ 是在独立性假设下的期望频数。

在 $H_0$ 下，$\chi^2 \sim \chi^2_{(r-1)(c-1)}$，其中 $r$ 和 $c$ 分别为行列数。

## 置信区间

### 构造方法

一个**置信区间**（CI）在置信水平 $1 - \alpha$ 下，是一个随机区间 $[L, U]$（数据的函数），满足：

$$P(\theta \in [L, U]) = 1 - \alpha.$$

对于已知 $\sigma$ 的正态均值：

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

其中 $z_{\alpha/2}$ 是 $\mathcal{N}(0, 1)$ 的 $1 - \alpha/2$ 分位数。对于 95% CI：$z_{0.025} = 1.96$。

对于未知 $\sigma$（使用 t 分布）：

$$\bar{X} \pm t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}.$$

### 解释（频率学派）

“若我们多次重复该实验并每次计算一个 95% 置信区间，则约 95% 的这些区间将包含真实参数 $\theta$。”

**常见误读：** “$\theta$ 落在此区间内的概率为 95%。” 这在频率学派框架下是错误的——$\theta$ 是一个固定的（未知）常数，而非随机变量。它要么在区间内，要么不在；我们只是不知道是哪一种情况。

### 置信区间与假设检验的关系

二者存在精确对偶性：

$$\text{Reject } H_0: \theta = \theta_0 \text{ at level } \alpha \iff \theta_0 \notin \text{CI at level } 1-\alpha.$$

一个 95% 置信区间恰好包含所有在 $\alpha = 0.05$ 水平下**不会被拒绝**的 $\theta_0$ 值。

*证明。* 置信区间为 $\{\theta_0 : |T(\theta_0)| \leq z_{\alpha/2}\}$，而检验在 $|T(\theta_0)| > z_{\alpha/2}$ 时拒绝。二者互为补集。$\blacksquare$

## 多重检验问题

### 问题所在

若你在 $\alpha = 0.05$ 水平下检验 20 个独立假设，且所有零假设均为真，则至少出现一次假阳性的概率为：

$$P(\text{at least one Type I error}) = 1 - (1 - 0.05)^{20} = 1 - 0.95^{20} \approx 0.64.$$

即有 64% 的概率出现假发现。若进行 100 次检验：$1 - 0.95^{100} \approx 0.994$。

这就是**多重检验问题**或**四处寻找效应（look-elsewhere effect）**。它解释了科学各领域中诸多“可重复性危机”。

### Bonferroni 校正

最简单的修正方法：对 $m$ 个检验，每个均采用显著性水平 $\alpha/m$。

**定理（Bonferroni）。** 若 $m$ 个检验各自使用水平 $\alpha/m$，则**族系误差率**（FWER）——即任意一次假拒绝的概率——至多为 $\alpha$。

*证明。* 由并集界（union bound）：

$$P(\text{any false rejection}) = P\left(\bigcup_{i=1}^{m_0} \{p_i \leq \alpha/m\}\right) \leq \sum_{i=1}^{m_0} P(p_i \leq \alpha/m) = m_0 \cdot \frac{\alpha}{m} \leq \alpha$$

其中 $m_0 \leq m$ 是真实零假设的数量。$\blacksquare$

Bonferroni 方法偏保守——它控制 FWER，但在 $m$ 很大时功效极低。

### 错误发现率（FDR）：Benjamini-Hochberg 方法

**错误发现率**定义为 $\text{FDR} = E\left[\frac{\text{false positives}}{\text{total rejections}}\right]$。

**Benjamini-Hochberg 过程：**
1. 将 $m$ 个 p 值排序：$p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$。
2. 找到最大的 $k$，使得 $p_{(k)} \leq \frac{k}{m} \alpha$。
3. 拒绝对应于 $p_{(1)}, \ldots, p_{(k)}$ 的假设。

该方法控制 $\text{FDR} \leq \alpha$，且在进行大量检验时比 Bonferroni 更具功效。

## A/B 测试


![A/B test design](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-ab-test.png)

### 设置方式

比较网页、广告或产品功能的两个版本（A = 对照组，B = 处理组）。观测转化率 $\hat{p}_A = k_A/n_A$ 和 $\hat{p}_B = k_B/n_B$。

### 检验统计量

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}$$

其中 $\hat{p} = (k_A + k_B)/(n_A + n_B)$ 是合并比例。

### 样本量计算

为以功效 $1 - \beta$ 在显著性水平 $\alpha$（双侧）下检测最小可检测效应 $\delta = p_B - p_A$：

$$n \geq \left(\frac{z_{\alpha/2} + z_\beta}{\delta}\right)^2 \cdot 2\bar{p}(1-\bar{p})$$

其中 $\bar{p} = (p_A + p_B)/2$ 是平均比例。

**示例。** 基线转化率 $p_A = 0.10$，最小可检测效应 $\delta = 0.02$（即检测提升至 $p_B = 0.12$），$\alpha = 0.05$，功效 $= 0.80$。

$$n \geq \left(\frac{1.96 + 0.84}{0.02}\right)^2 \cdot 2 \times 0.11 \times 0.89 \approx \frac{19600}{1} \times 0.1958 \approx 3838$$

每组需约 3,838 名用户——总计约 7,700 名。

### 实际显著性 vs 统计显著性

一个结果可能具有统计显著性（p 值小），但实际毫无意义。例如，一项拥有 100 万用户的 A/B 测试显示 $p_B - p_A = 0.0001$ 且 $p < 0.001$，该效应虽真实却微不足道——很可能不值得投入工程资源上线。

务必报告**效应量和置信区间**，而不仅是 p 值。

## Python：实践中的假设检验

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Example 1: One-sample t-test
data = np.random.normal(loc=10.2, scale=0.5, size=25)
t_stat, p_value = stats.ttest_1samp(data, popmean=10.0)
print(f"One-sample t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

# Example 2: Two-sample t-test (Welch's)
group_a = np.random.normal(loc=5.0, scale=1.0, size=30)
group_b = np.random.normal(loc=5.5, scale=1.2, size=35)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"Two-sample t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

# Example 3: Chi-squared test
observed = np.array([[30, 10], [15, 25]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-squared test: chi2 = {chi2:.4f}, p = {p_value:.4f}, dof = {dof}")

# Example 4: A/B test
n_A, n_B = 5000, 5000
conversions_A, conversions_B = 500, 560
p_A = conversions_A / n_A
p_B = conversions_B / n_B
p_pool = (conversions_A + conversions_B) / (n_A + n_B)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z = (p_B - p_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"A/B test: z = {z:.4f}, p = {p_value:.4f}")
print(f"  Lift: {p_B - p_A:.4f} ({(p_B - p_A)/p_A*100:.1f}%)")
print(f"  95% CI for lift: [{p_B-p_A-1.96*se:.4f}, {p_B-p_A+1.96*se:.4f}]")

# Visualization: p-value under null hypothesis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: p-value illustration
ax = axes[0]
x = np.linspace(-4, 4, 300)
ax.plot(x, stats.norm.pdf(x), 'b-', linewidth=2)
t_obs = 2.0
ax.fill_between(x[x >= t_obs], stats.norm.pdf(x[x >= t_obs]),
                alpha=0.4, color='red', label=f'p-value/2 (right tail)')
ax.fill_between(x[x <= -t_obs], stats.norm.pdf(x[x <= -t_obs]),
                alpha=0.4, color='red', label=f'p-value/2 (left tail)')
ax.axvline(t_obs, color='red', linestyle='--', alpha=0.7)
ax.axvline(-t_obs, color='red', linestyle='--', alpha=0.7)
ax.set_title(f'Two-sided p-value (z = {t_obs})', fontsize=13)
ax.set_xlabel('Test statistic')
ax.set_ylabel('Density under $H_0$')
ax.legend(fontsize=10)

# Panel 2: Power illustration
ax = axes[1]
x = np.linspace(-4, 6, 300)
# Null distribution
ax.plot(x, stats.norm.pdf(x, 0, 1), 'b-', linewidth=2, label='$H_0$: $\\mu = 0$')
# Alternative distribution
delta = 2.0
ax.plot(x, stats.norm.pdf(x, delta, 1), 'r-', linewidth=2,
        label=f'$H_1$: $\\mu = {delta}$')
# Critical value
z_crit = 1.645  # one-sided alpha = 0.05
ax.axvline(z_crit, color='gray', linestyle='--')
ax.fill_between(x[x >= z_crit], stats.norm.pdf(x[x >= z_crit], delta, 1),
                alpha=0.3, color='green', label=f'Power = {1-stats.norm.cdf(z_crit-delta):.3f}')
ax.fill_between(x[x >= z_crit], stats.norm.pdf(x[x >= z_crit], 0, 1),
                alpha=0.3, color='red', label=f'$\\alpha$ = {1-stats.norm.cdf(z_crit):.3f}')
ax.set_title('Power of a Test', fontsize=13)
ax.set_xlabel('Test statistic')
ax.legend(fontsize=9)

# Panel 3: Multiple testing
ax = axes[2]
m_tests = np.arange(1, 101)
fwer = 1 - (1 - 0.05)**m_tests
ax.plot(m_tests, fwer, 'r-', linewidth=2, label='FWER (no correction)')
ax.axhline(0.05, color='gray', linestyle=':', label='$\\alpha$ = 0.05')
ax.set_xlabel('Number of tests', fontsize=12)
ax.set_ylabel('P(at least one false positive)', fontsize=12)
ax.set_title('Multiple Testing Problem', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis_testing.png', dpi=150)
plt.show()
```

```python
# Demonstrate Benjamini-Hochberg FDR control
from scipy.stats import false_discovery_control

# Simulate: 100 tests, 10 true effects (mu=3), 90 null (mu=0)
np.random.seed(42)
n_tests = 100
n_true_effects = 10
n = 30  # samples per test

p_values = []
truth = []

for i in range(n_tests):
    if i < n_true_effects:
        data = np.random.normal(loc=2.0, scale=1, size=n)
        truth.append(True)
    else:
        data = np.random.normal(loc=0.0, scale=1, size=n)
        truth.append(False)
    _, pval = stats.ttest_1samp(data, 0)
    p_values.append(pval)

p_values = np.array(p_values)
truth = np.array(truth)

# Bonferroni
bonferroni_reject = p_values < 0.05 / n_tests

# Benjamini-Hochberg
sorted_idx = np.argsort(p_values)
bh_threshold = np.arange(1, n_tests + 1) / n_tests * 0.05
bh_reject_sorted = p_values[sorted_idx] <= bh_threshold
# Find largest k where p_(k) <= k/m * alpha
if bh_reject_sorted.any():
    k = np.max(np.where(bh_reject_sorted)[0]) + 1
    bh_reject = np.zeros(n_tests, dtype=bool)
    bh_reject[sorted_idx[:k]] = True
else:
    bh_reject = np.zeros(n_tests, dtype=bool)

print(f"{'Method':<25} {'Rejections':>10} {'True Pos':>10} {'False Pos':>10}")
print("-" * 55)
print(f"{'No correction':<25} {(p_values < 0.05).sum():>10} "
      f"{((p_values < 0.05) & truth).sum():>10} "
      f"{((p_values < 0.05) & ~truth).sum():>10}")
print(f"{'Bonferroni':<25} {bonferroni_reject.sum():>10} "
      f"{(bonferroni_reject & truth).sum():>10} "
      f"{(bonferroni_reject & ~truth).sum():>10}")
print(f"{'Benjamini-Hochberg':<25} {bh_reject.sum():>10} "
      f"{(bh_reject & truth).sum():>10} "
      f"{(bh_reject & ~truth).sum():>10}")
```

输出通常显示：未校正时，约 5–6 个假阳性混入 90 个零假设中。Bonferroni 消除了所有假阳性，但可能遗漏部分真实效应。Benjamini-Hochberg 则在控制假发现的同时捕获了大部分真实效应。这体现了 FDR 控制相比 FWER 控制的功效优势。

## 效应量：p 值无法告诉你的信息


![Hypothesis testing courtroom trial null hypothesis on trial](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/07-hypothesis-testing-courtroom-trial-null-hypothesis-on-trial.jpg)

p 值仅说明效应是否在统计上可与零区分。它**无法**告诉你效应有多大，或是否具有实际意义。

### Cohen’s d

用于比较两组均值：

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$

其中 $S_p$ 是合并标准差。这是一个**标准化效应量**——以标准差为单位衡量差异。

| $d$ | 解释 |
|---|---|
| 0.2 | 小效应 |
| 0.5 | 中等效应 |
| 0.8 | 大效应 |

### 为何效应量至关重要

当每组 $n = 1{,}000{,}000$ 时，你可在 $p < 0.001$ 水平下检测到 $d = 0.01$。该效应“高度显著”，但实际毫无意义——0.01 个标准差的差异对任何个体而言都不可察觉。

应对之策：**始终报告效应量的置信区间，而非仅报告 p 值。** 均值差异的置信区间既可判断效应是否显著（CI 是否排除零？），也能揭示其可能的实际大小。

## Neyman-Pearson 引理

**定理（Neyman-Pearson）。** 对于检验 $H_0: \theta = \theta_0$ vs $H_1: \theta = \theta_1$（简单零假设 vs 简单备择假设），在水平 $\alpha$ 下最具功效的检验在如下条件下拒绝 $H_0$：

$$\Lambda(\mathbf{x}) = \frac{L(\theta_1; \mathbf{x})}{L(\theta_0; \mathbf{x})} > c$$

其中 $c$ 被选定为使 $P(\Lambda > c | H_0) = \alpha$。

**似然比** $\Lambda$ 是最优检验统计量：在相同显著性水平下，没有任何其他检验能获得更高功效。这一理论结果为参数统计中广泛使用的似然比检验提供了正当性依据。

### 广义似然比检验

对于复合假设（$H_0: \theta \in \Theta_0$ vs $H_1: \theta \notin \Theta_0$）：

$$\Lambda = \frac{\max_{\theta \in \Theta_0} L(\theta)}{\max_{\theta \in \Theta} L(\theta)} = \frac{L(\hat{\theta}_0)}{L(\hat{\theta}_{\text{MLE}})}.$$

在 $H_0$ 及正则性条件下：

$$-2 \ln \Lambda \xrightarrow{d} \chi^2_r$$

其中 $r = \dim(\Theta) - \dim(\Theta_0)$ 是自由参数数量之差。

此即**Wilks 定理**，也是嵌套模型比较的基础——从检验回归系数是否为零，到比较分层贝叶斯模型。

## 置换检验：免分布推断

当分布假设（正态性、方差齐性）存疑时，**置换检验**提供了一种精确替代方案。

**思想：** 在 $H_0$（组间无差异）下，组标签是任意的。多次随机置换标签，每次重新计算检验统计量，并观察观测统计量在该置换分布中的位置。

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Observed data
group_a = np.array([5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 5.4, 4.7])
group_b = np.array([5.5, 5.8, 5.3, 5.7, 5.9, 5.6, 5.4, 5.8])

observed_diff = group_b.mean() - group_a.mean()
combined = np.concatenate([group_a, group_b])
n_a = len(group_a)
n_perms = 100_000

# Permutation test
perm_diffs = np.zeros(n_perms)
for i in range(n_perms):
    np.random.shuffle(combined)
    perm_diffs[i] = combined[n_a:].mean() - combined[:n_a].mean()

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"Observed difference: {observed_diff:.4f}")
print(f"Permutation p-value: {p_value:.4f}")

# Compare with t-test
t_stat, t_pval = stats.ttest_ind(group_a, group_b)
print(f"t-test p-value: {t_pval:.4f}")
```

置换检验对任意检验统计量都是精确的（非近似），无需分布假设，且易于实现。其主要局限在于：它仅检验可交换性（在 $H_0$ 下组是否可互换？），而这未必涵盖所有感兴趣的零假设。

## 可重复性危机及应对之道

自 2010 年左右起，系统性重复研究揭示，心理学与生物医学等领域中许多已发表成果——或许高达 50% 或更多——无法复现。常见诱因包括：

1. **p 值操纵（p-hacking）：** 尝试多种分析直至 $p < 0.05$，然后仅报告“显著”结果。
2. **事后假设（HARKing）：** 结果已知后提出假设——将事后的发现包装成预先设定的假设。
3. **发表偏倚（Publication bias）：** 期刊偏好发表“显著”结果，导致文献偏向假阳性。
4. **统计功效不足的研究：** 功效低时，显著结果更可能高估真实效应。

**推荐实践：**
- 提前注册假设与分析计划
- 报告效应量和置信区间，而非仅 p 值
- 使用适当的多重检验校正方法
- 考虑贝叶斯方法以累积证据
- 关注重复验证，而非单次研究

## 快速参考表

| 场景 | 检验方法 | 统计量 | $H_0$ 下分布 |
|---|---|---|---|
| 单均值，$\sigma$ 已知 | Z 检验 | $(\bar{X}-\mu_0)/(\sigma/\sqrt{n})$ | $\mathcal{N}(0,1)$ |
| 单均值，$\sigma$ 未知 | t 检验 | $(\bar{X}-\mu_0)/(S/\sqrt{n})$ | $t_{n-1}$ |
| 双均值，独立样本 | 双样本 t 检验 | $(\bar{X}-\bar{Y})/(S_p\sqrt{1/n_1+1/n_2})$ | $t_{n_1+n_2-2}$ |
| 双均值，配对样本 | 配对 t 检验 | $\bar{D}/(S_D/\sqrt{n})$ | $t_{n-1}$ |
| 分类变量独立性 | 卡方检验 | $\sum (O-E)^2/E$ | $\chi^2_{(r-1)(c-1)}$ |
| 双比例 | Z 检验 | $(p_B-p_A)/\text{SE}$ | $\mathcal{N}(0,1)$ |
| 模型比较 | 似然比检验 | $-2\ln\Lambda$ | $\chi^2_r$（Wilks 定理） |
| 非参数检验 | 置换检验 | 任意统计量 | 置换分布 |

## 下一步

假设检验通过控制假阳性率来回答“该效应是否真实？”这一问题。但它将参数视为固定未知量，而将数据视为随机变量。贝叶斯统计则反转这一视角：数据是固定的（你已观测到），而参数是随机的（由某个分布描述）。下一篇文章将展开贝叶斯思维——先验、后验、共轭族，以及塑造统计学百年发展的哲学论争。