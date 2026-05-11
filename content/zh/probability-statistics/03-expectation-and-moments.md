---
title: "概率与统计（三）：期望、方差与矩生成函数技巧"
date: 2024-08-21 09:00:00
tags:
  - Probability
  - Statistics
  - Expectation
  - Moment Generating Functions
categories:
  - Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "从期望与方差，到协方差、相关系数与矩生成函数，再到切比雪夫不等式——涵盖随机变量总结性刻画的完整工具箱，所有结论均附严格证明。"
disableNunjucks: true
series_order: 3
translationKey: "probability-statistics-3"
---

概率分布是对随机变量的**完整描述**——它告诉你每个可能结果出现的概率。但“完整”往往意味着繁琐。当有人问：“这座城市的人平均身高是多少？”，你不会递给他一个密度函数；你会说：“大约 170 厘米，上下浮动约 10 厘米。” 平均值与离散程度（spread）就已捕捉了实践中最核心的信息。

本文构建用于**概括分布特征**的数学框架：我们从刻画“中心位置”的**期望**出发，进阶至刻画“离散程度”的**方差**，最后引入**矩生成函数（MGF）**——一个简洁公式，不仅能编码分布的所有矩（moments），更惊人地——**唯一确定该分布本身**。

## 期望


![Expectation as balance point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-expectation.png)

### 定义

对于具有概率质量函数（PMF）$p(x)$ 的**离散型**随机变量 $X$：

$$E[X] = \sum_{x} x \, p(x)$$

其中求和遍历 $X$ 支撑集（support）中的所有取值，且要求该级数**绝对收敛**。

对于具有概率密度函数（PDF）$f(x)$ 的**连续型**随机变量 $X$：

$$E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx$$

同样要求该积分**绝对收敛**。

期望可直观理解为分布的“质心”。若在数轴上每个点 $x$ 处放置重量为 $p(x)$ 的质点，则 $E[X]$ 就是整个系统的平衡点。

### 期望的线性性

这或许是整个概率论中**最有用的性质**。

**定理。** 对任意随机变量 $X$ 和 $Y$（未必独立）及常数 $a, b, c$，有：

$$E[aX + bY + c] = aE[X] + bE[Y] + c.$$

*证明（离散情形）。* 设 $(X, Y)$ 的联合 PMF 为 $p(x, y)$。

$$E[aX + bY + c] = \sum_x \sum_y (ax + by + c) \, p(x, y)$$

$$= a \sum_x \sum_y x \, p(x, y) + b \sum_x \sum_y y \, p(x, y) + c \sum_x \sum_y p(x, y)$$

$$= a \sum_x x \sum_y p(x, y) + b \sum_y y \sum_x p(x, y) + c \cdot 1$$

$$= a \sum_x x \, p_X(x) + b \sum_y y \, p_Y(y) + c = aE[X] + bE[Y] + c. \quad \blacksquare$$

注意：**我们从未假设独立性。** 线性性恒成立——这正是其威力所在。

**例。** 洗匀 $n$ 张牌后，固定点（fixed point）的期望个数是多少？（固定点指某张牌最终仍在其原始位置）

令 $X_i = 1$ 若第 $i$ 张牌在位置 $i$，否则 $X_i = 0$。则 $X = \sum_{i=1}^n X_i$ 表示固定点总数。$E[X_i] = P(\text{第 } i \text{ 张牌不动}) = 1/n$。由线性性：

$$E[X] = \sum_{i=1}^n E[X_i] = n \cdot \frac{1}{n} = 1.$$

平均而言，**恰好有一张牌保持原位**——且该结论与 $n$ 无关。我们甚至无需知道 $X$ 的分布，就计算出了这个复杂随机变量的期望。

**例。** 随机排列 $\{1, 2, \ldots, n\}$ 中逆序对（inversion）的期望个数。

一个**逆序对**是指满足 $i < j$ 但 $\sigma(i) > \sigma(j)$ 的数对 $(i, j)$。令 $X_{ij} = 1$ 若 $(i,j)$ 是逆序对。由对称性，对任意数对，$P(X_{ij} = 1) = 1/2$（两元素以任一顺序出现的概率相等）。共有 $\binom{n}{2}$ 个数对，故：

$$E[\text{逆序对个数}] = \binom{n}{2} \cdot \frac{1}{2} = \frac{n(n-1)}{4}.$$

再次，线性性让我们绕开了逆序对之间复杂的依赖关系，直接得出答案。注意：这些 $X_{ij}$ **并非相互独立**（交换两个元素会影响多个数对），但线性性对此毫不在意。

## LOTUS：无意识统计学家法则

要计算 $E[g(X)]$（$g$ 为某函数），你或许会认为必须先求出 $Y = g(X)$ 的分布。其实不必。

![Variance visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-variance.png)


**定理（LOTUS）。** 若 $X$ 是离散型随机变量，其 PMF 为 $p(x)$，则：

$$E[g(X)] = \sum_x g(x) \, p(x).$$

若 $X$ 是连续型随机变量，其 PDF 为 $f(x)$，则：

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f(x) \, dx.$$

“无意识统计学家”这一名称略带戏谑：该公式如此自然，人们常在未意识到自己正使用一个定理的情况下就直接应用它。

**例。** 设 $X \sim \text{Uniform}(0, 1)$。求 $E[X^2]$。

$$E[X^2] = \int_0^1 x^2 \cdot 1 \, dx = \frac{x^3}{3}\bigg|_0^1 = \frac{1}{3}.$$

我们在此应用了 LOTUS，其中 $g(x) = x^2$，$f(x) = 1$（定义域为 $[0,1]$）。无需推导 $X^2$ 的分布。

## 方差


![Covariance scatter plots](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-covariance-scatter.png)


![Covariance correlation dance partners moving together or ind](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/03-covariance-correlation-dance-partners-moving-together-or-ind.jpg)

### 定义

随机变量 $X$ 的**方差**衡量其值偏离均值的平均平方距离：

$$\text{Var}(X) = E\left[(X - E[X])^2\right].$$

**标准差**定义为 $\text{SD}(X) = \sigma_X = \sqrt{\text{Var}(X)}$，其单位与 $X$ 相同。

### 计算公式

展开平方项：

$$\text{Var}(X) = E\left[X^2 - 2XE[X] + (E[X])^2\right] = E[X^2] - 2E[X]E[X] + (E[X])^2$$

$$= E[X^2] - (E[X])^2.$$

这是你 90% 时间都会用到的公式：

$$\boxed{\text{Var}(X) = E[X^2] - (E[X])^2}$$

**例。** $X \sim \text{Uniform}(0,1)$：$E[X] = 1/2$，$E[X^2] = 1/3$（如前所算）。故 $\text{Var}(X) = 1/3 - 1/4 = 1/12$。这与公式 $(b-a)^2/12 = 1/12$ 一致。

### 缩放与平移

**定理。** 对常数 $a$ 和 $b$，有：

$$\text{Var}(aX + b) = a^2 \text{Var}(X).$$

*证明。*

$$\text{Var}(aX + b) = E[(aX + b)^2] - (E[aX + b])^2$$

$$= E[a^2X^2 + 2abX + b^2] - (aE[X] + b)^2$$

$$= a^2E[X^2] + 2abE[X] + b^2 - a^2(E[X])^2 - 2abE[X] - b^2$$

$$= a^2(E[X^2] - (E[X])^2) = a^2\text{Var}(X). \quad \blacksquare$$

加常数 $b$ 仅平移分布，不改变其离散程度；乘以 $a$ 则将离散程度缩放 $|a|$ 倍（方差缩放 $a^2$ 倍）。

### 和的方差

若 $X$ 和 $Y$ **相互独立**：

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y).$$

更一般地（无需独立性）：

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

这便引出了我们的下一个主题。

## 协方差


![Chebyshev inequality](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-chebyshev-bound.png)

### 定义

$X$ 与 $Y$ 的**协方差**衡量二者如何共同变动：

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y].$$

*计算公式的证明：*

$$\text{Cov}(X,Y) = E[XY - XE[Y] - YE[X] + E[X]E[Y]]$$

$$= E[XY] - E[X]E[Y] - E[Y]E[X] + E[X]E[Y] = E[XY] - E[X]E[Y]. \quad \blacksquare$$

### 性质

1. $\text{Cov}(X, X) = \text{Var}(X)$  
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$（对称性）  
3. $\text{Cov}(aX + b, Y) = a \, \text{Cov}(X, Y)$（对每个参数的线性性）  
4. $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$（双线性性）  
5. 若 $X$ 与 $Y$ 独立，则 $\text{Cov}(X, Y) = 0$。

**警告：** 性质 5 的逆命题**不成立**。$\text{Cov}(X, Y) = 0$ **不能推出** $X$ 与 $Y$ 独立。

**反例。** 设 $X \sim \text{Uniform}(-1, 1)$，$Y = X^2$。此时 $Y$ 完全由 $X$ 决定（完全依赖），但：

$$E[XY] = E[X^3] = \int_{-1}^{1} x^3 \cdot \frac{1}{2} dx = 0$$

因为 $x^3$ 在对称区间 $[-1,1]$ 上是奇函数。故 $\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 0 - 0 \cdot E[Y] = 0$。

协方差仅能探测**线性关系**，对非线性关系无能为力。

## 相关系数

**皮尔逊相关系数**将协方差标准化，使其取值范围落在 $[-1, 1]$ 内：

![Moment generating function](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-mgf.png)


$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}.$$

### 性质

- $-1 \leq \rho \leq 1$（由柯西-施瓦茨不等式证明）  
- $\rho = 1$ 当且仅当 $Y = aX + b$ 且 $a > 0$（完美正向线性关系）  
- $\rho = -1$ 当且仅当 $Y = aX + b$ 且 $a < 0$（完美负向线性关系）  
- $\rho = 0$ 表示“不相关”，但**不一定独立**

*$|\rho| \leq 1$ 的证明。* 由期望形式的柯西-施瓦茨不等式：

$$(E[UV])^2 \leq E[U^2] E[V^2].$$

令 $U = X - E[X]$，$V = Y - E[Y]$：

$$(\text{Cov}(X,Y))^2 \leq \text{Var}(X) \text{Var}(Y)$$

$$\rho^2 = \frac{(\text{Cov}(X,Y))^2}{\text{Var}(X)\text{Var}(Y)} \leq 1. \quad \blacksquare$$

### 相关系数的含义与局限

相关系数度量两个变量间**线性关系**的强度与方向。它**无法**刻画：

- 非线性关系（例如 $X$ 与 $X^2$ 可能有 $\rho = 0$）  
- 因果关系（可能存在第三个变量同时驱动两者）  
- 关系的斜率（那是回归系数，而非 $\rho$）

## 高阶矩

$X$ 关于原点的 $k$ 阶**矩**定义为 $E[X^k]$；关于均值 $\mu$ 的 $k$ 阶**中心矩**定义为 $E[(X - \mu)^k]$。

- 一阶矩：$E[X] = \mu$（位置）  
- 二阶中心矩：$\text{Var}(X) = \sigma^2$（离散程度）  
- 三阶中心矩（标准化）：**偏度（Skewness）**

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}$$

偏度衡量分布的不对称性。$\gamma_1 > 0$：右尾更长（右偏）；$\gamma_1 < 0$：左尾更长（左偏）；$\gamma_1 = 0$：对称（如正态分布）。

**例。** 指数分布 $\text{Exp}(\lambda)$ 的 $\gamma_1 = 2$ —— 恒为正偏。收入分布通常 $\gamma_1 > 0$（高收入者构成长长的右尾）。金融资产收益常 $\gamma_1 < 0$（“崩盘”尾部比“繁荣”尾部更重）。

- 四阶中心矩（标准化）：**峰度（Kurtosis）**

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4}$$

正态分布的 $\gamma_2 = 3$。**超额峰度（Excess kurtosis）** 定义为 $\gamma_2 - 3$，衡量尾部相对于正态分布的厚重程度。正值表示尾部更厚（极端事件发生概率高于高斯分布预测）。

**例。** 自由度为 $\nu$ 的学生 $t$ 分布，其超额峰度为 $6/(\nu - 4)$（当 $\nu > 4$）。当 $\nu \to \infty$ 时，它趋近于 0（即正态分布）。对较小的 $\nu$，超额峰度很大，反映其厚重尾部。这正是 $t$ 分布在稳健统计中被采用的原因。

**例。** 均匀分布的 $\gamma_2 = 1.8$，故其超额峰度为 $-1.2$（尾部比正态分布更轻）。在有界区间上的连续分布中，它是最“低峰态（platykurtic）”的。

## 矩生成函数

### 定义

随机变量 $X$ 的**矩生成函数（MGF）** 定义为：

$$M_X(t) = E[e^{tX}]$$

其定义域为包含 $0$ 的某个开区间。

名称源于泰勒展开 $e^{tX} = 1 + tX + \frac{t^2 X^2}{2!} + \cdots$，故：

$$M_X(t) = 1 + tE[X] + \frac{t^2}{2!}E[X^2] + \frac{t^3}{3!}E[X^3] + \cdots$$

通过求导可提取各阶矩：

$$E[X^k] = M_X^{(k)}(0) = \frac{d^k}{dt^k} M_X(t) \bigg|_{t=0}.$$

### 唯一性定理

**定理。** 若 $M_X(t) = M_Y(t)$ 对所有包含 $0$ 的开区间内的 $t$ 成立，则 $X$ 与 $Y$ 具有相同的分布。

这正是 MGF 强大的原因：它们是分布的**指纹**。识别出 MGF，就等于识别出该分布。

### 泊松分布的 MGF

设 $X \sim \text{Poisson}(\lambda)$：

$$M_X(t) = E[e^{tX}] = \sum_{k=0}^{\infty} e^{tk} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} \cdot e^{\lambda e^t} = e^{\lambda(e^t - 1)}.$$

*验证：* $M_X'(t) = \lambda e^t \cdot e^{\lambda(e^t - 1)}$，故 $M_X'(0) = \lambda \cdot 1 = \lambda = E[X]$。$\checkmark$

$M_X''(t) = (\lambda e^t)^2 e^{\lambda(e^t-1)} + \lambda e^t e^{\lambda(e^t-1)}$，故 $M_X''(0) = \lambda^2 + \lambda = E[X^2]$。进而 $\text{Var}(X) = \lambda^2 + \lambda - \lambda^2 = \lambda$。$\checkmark$

### 正态分布的 MGF

设 $Z \sim \mathcal{N}(0, 1)$：

$$M_Z(t) = E[e^{tZ}] = \int_{-\infty}^{\infty} e^{tz} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-(z^2 - 2tz)/2} dz.$$

配方：$z^2 - 2tz = (z - t)^2 - t^2$。

$$M_Z(t) = e^{t^2/2} \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-(z-t)^2/2} dz = e^{t^2/2} \cdot 1 = e^{t^2/2}.$$

对 $X \sim \mathcal{N}(\mu, \sigma^2)$，写作 $X = \mu + \sigma Z$：

$$M_X(t) = E[e^{t(\mu + \sigma Z)}] = e^{\mu t} M_Z(\sigma t) = e^{\mu t + \sigma^2 t^2/2}.$$

### 指数分布的 MGF

设 $X \sim \text{Exp}(\lambda)$：

$$M_X(t) = \int_0^{\infty} e^{tx} \lambda e^{-\lambda x} dx = \lambda \int_0^{\infty} e^{-(\lambda - t)x} dx = \frac{\lambda}{\lambda - t} \quad \text{for } t < \lambda.$$

### 使用 MGF 证明分布性质

**断言：** 独立泊松变量之和仍为泊松分布。若 $X \sim \text{Poisson}(\lambda)$ 且 $Y \sim \text{Poisson}(\mu)$ 独立，则 $X + Y \sim \text{Poisson}(\lambda + \mu)$。

*证明。* $M_{X+Y}(t) = M_X(t) M_Y(t) = e^{\lambda(e^t - 1)} e^{\mu(e^t - 1)} = e^{(\lambda + \mu)(e^t - 1)}$，这正是 $\text{Poisson}(\lambda + \mu)$ 的 MGF。由唯一性定理，$X + Y \sim \text{Poisson}(\lambda + \mu)$。$\blacksquare$

## 马尔可夫不等式

**定理（马尔可夫）。** 若 $X \geq 0$ 且 $a > 0$，则

$$P(X \geq a) \leq \frac{E[X]}{a}.$$

*证明。* 注意 $a \cdot \mathbf{1}_{X \geq a} \leq X$（若 $X \geq a$，则 $a \leq X$；若 $X < a$，则 $0 \leq X$，而 $X \geq 0$ 故成立）。两边取期望：

$$a \cdot P(X \geq a) \leq E[X].$$

两边除以 $a$ 即得。$\blacksquare$

马尔可夫不等式通常较宽松，但它所需前提极少——仅需 $X \geq 0$ 且期望有限。

## 切比雪夫不等式

**定理（切比雪夫）。** 对任意均值为 $\mu$、方差为 $\sigma^2$（有限）的随机变量 $X$，及任意 $k > 0$，有：

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}.$$

*证明。* 对非负随机变量 $(X - \mu)^2$ 应用马尔可夫不等式，取 $a = k^2 \sigma^2$：

$$P((X - \mu)^2 \geq k^2 \sigma^2) \leq \frac{E[(X - \mu)^2]}{k^2 \sigma^2} = \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}. \quad \blacksquare$$

| $k$ | 切比雪夫上界 $P(|X - \mu| \geq k\sigma) \leq$ | 正态分布精确值 $P(|Z| \geq k)$ |
|---|---|---|
| 1 | 1.000 | 0.317 |
| 2 | 0.250 | 0.046 |
| 3 | 0.111 | 0.003 |
| 4 | 0.063 | 0.00006 |

切比雪夫不等式是**分布无关的（distribution-free）**——只要方差有限，它对任何分布都成立。这种普适性的代价是上界较宽松（对比上表两列）。但“宽松却恒成立”，往往比“紧致却仅适用于高斯分布”更有实用价值。

## Python：可视化矩与切比雪夫上界


![Expectation as center of mass balance point on probability d](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/03-expectation-as-center-of-mass-balance-point-on-probability-d.jpg)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. 期望作为质心
ax = axes[0]
x = np.linspace(-4, 6, 300)
for mu, sigma in [(0, 1), (2, 0.7), (-1, 1.5)]:
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, linewidth=2, label=f'$\\mu$={mu}, $\\sigma$={sigma}')
    ax.axvline(mu, linestyle=':', alpha=0.5)
ax.set_title('期望作为位置', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 2. 方差作为离散程度
ax = axes[1]
x = np.linspace(-6, 6, 300)
for sigma in [0.5, 1, 2]:
    y = stats.norm.pdf(x, 0, sigma)
    ax.plot(x, y, linewidth=2, label=f'$\\sigma$={sigma}')
    ax.axvspan(-sigma, sigma, alpha=0.05)
ax.set_title('方差作为离散程度', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 3. 切比雪夫上界 vs 实际值
ax = axes[2]
ks = np.linspace(1, 5, 100)
chebyshev_bound = 1 / ks**2
normal_exact = 2 * (1 - stats.norm.cdf(ks))
uniform_exact = np.maximum(0, 1 - ks * np.sqrt(3) / 3)  # 近似

ax.plot(ks, chebyshev_bound, 'r-', linewidth=2, label='切比雪夫上界')
ax.plot(ks, normal_exact, 'b--', linewidth=2, label='正态分布（精确）')
ax.set_title("切比雪夫不等式", fontsize=13)
ax.set_xlabel('k（标准差倍数）')
ax.set_ylabel('P(|X - $\\mu$| $\\geq$ k$\\sigma$)')
ax.legend()
ax.set_yscale('log')
ax.set_ylim(1e-4, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('moments_and_chebyshev.png', dpi=150)
plt.show()
```

右图以对数尺度展示了切比雪夫上界与正态分布实际尾部概率之间的差距。对正态分布，真实概率随 $k^2$ 指数衰减，而切比雪夫仅保证 $1/k^2$ 的衰减速率。该上界虽保守，却适用于**所有**分布——包括那些尾部远比高斯分布厚重的分布。

## 条件期望

### 定义

给定 $Y = y$ 时，$X$ 的**条件期望**定义为：

$$E[X | Y = y] = \sum_x x \, p_{X|Y}(x | y) \quad \text{(离散型)}$$

$$E[X | Y = y] = \int_{-\infty}^{\infty} x \, f_{X|Y}(x | y) \, dx \quad \text{(连续型)}$$

将 $E[X|Y]$ 视为 $Y$ 的函数，它本身就是一个随机变量。

### 塔式性质（迭代期望定律）

**定理。** $E[E[X | Y]] = E[X]$。

*证明（离散情形）。*

$$E[E[X|Y]] = \sum_y E[X|Y=y] \cdot p_Y(y) = \sum_y \left(\sum_x x \, p_{X|Y}(x|y)\right) p_Y(y)$$

$$= \sum_y \sum_x x \, p_{X,Y}(x,y) = \sum_x x \sum_y p_{X,Y}(x,y) = \sum_x x \, p_X(x) = E[X]. \quad \blacksquare$$

此性质极为有用。为计算 $E[X]$，可先对某个能简化问题的变量 $Y$ 取条件，分别计算各 $Y$ 值下的 $E[X|Y]$，再对其加权平均。

**例。** 投掷一枚骰子，结果为 $Y$；然后抛掷 $Y$ 枚硬币。令 $X$ 为正面朝上的次数。求 $E[X]$。

给定 $Y$ 时，$X | Y \sim \text{Binomial}(Y, 1/2)$，故 $E[X|Y] = Y/2$。

$$E[X] = E[E[X|Y]] = E[Y/2] = E[Y]/2 = 3.5/2 = 1.75.$$

### 条件方差

$$\text{Var}(X | Y) = E[X^2 | Y] - (E[X|Y])^2.$$

**全方差公式（Eve's Law）：**

$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y]).$$

总方差分解为：
- **组内方差：** $E[\text{Var}(X|Y)]$ —— 各 $Y$ 水平内部的平均变异性。
- **组间方差：** $\text{Var}(E[X|Y])$ —— 各组均值的变异性。

*证明。* 由方差的计算公式：

$$\text{Var}(X) = E[X^2] - (E[X])^2.$$

利用塔式性质：$E[X^2] = E[E[X^2|Y]]$ 且 $E[X] = E[E[X|Y]]$。

$$E[E[X^2|Y]] = E[\text{Var}(X|Y) + (E[X|Y])^2] = E[\text{Var}(X|Y)] + E[(E[X|Y])^2].$$

$$\text{Var}(X) = E[\text{Var}(X|Y)] + E[(E[X|Y])^2] - (E[E[X|Y]])^2.$$

后两项即为 $\text{Var}(E[X|Y])$。$\blacksquare$

## 詹森不等式

**定理（詹森）。** 若 $g$ 是凸函数，且随机变量 $X$ 的期望 $E[X]$ 有限，则：

$$g(E[X]) \leq E[g(X)].$$

若 $g$ 是凹函数，不等号反向：$g(E[X]) \geq E[g(X)]$。

*证明概要。* 凸函数位于其任意切线之上。对 $E[X]$ 处的切线：$g(x) \geq g(E[X]) + g'(E[X])(x - E[X])$。两边取期望，并注意到 $E[x - E[X]] = 0$，即得结论。$\blacksquare$

**应用：**

1. **方差非负：** 取 $g(x) = x^2$（凸函数），则 $E[X^2] \geq (E[X])^2$，故 $\text{Var}(X) \geq 0$。

2. **算术-几何平均不等式（AM-GM）：** 对 $X > 0$，因 $\ln$ 是凹函数，有 $E[\ln X] \leq \ln E[X]$，即正数的算术平均不小于其几何平均。

3. **KL 散度非负：** 对 $\ln$ 函数应用詹森不等式，可证 KL 散度 $D_{\text{KL}}(P \| Q) \geq 0$——信息论与机器学习的基础性结论。

## 特征函数（简述）

并非所有分布都存在矩生成函数（MGF 可能在 $0$ 附近对所有 $t$ 都不存在）。而**特征函数**总是存在：

$$\varphi_X(t) = E[e^{itX}] = E[\cos(tX)] + i \, E[\sin(tX)]$$

其中 $i = \sqrt{-1}$。特征函数对**任何分布**都存在（因 $|e^{itX}| = 1$，期望恒收敛）。它也唯一确定分布，且与 MGF 类似，对独立变量 $X$ 和 $Y$，有 $\varphi_{X+Y}(t) = \varphi_X(t) \varphi_Y(t)$。

中心极限定理（CLT）的一般性证明即基于特征函数——你证明 $\varphi_{\bar{X}_n}(t) \to e^{-t^2/2}$（即 $\mathcal{N}(0,1)$ 的特征函数），再应用 Levy 连续性定理。

## 超越切比雪夫的不等式

### 切尔诺夫界（Chernoff Bound）

对任意 $t > 0$：

$$P(X \geq a) = P(e^{tX} \geq e^{ta}) \leq \frac{E[e^{tX}]}{e^{ta}} = \frac{M_X(t)}{e^{ta}}$$

此即对 $e^{tX}$ 应用马尔可夫不等式所得。对 $t$ 进行优化可得到最紧的上界。这就是**切尔诺夫界**，它对具有良好数学性质 MGF 的分布，提供指数衰减的尾部概率。

**例。** 标准正态分布尾部：$P(Z \geq a) \leq e^{-a^2/2}$，此式由在切尔诺夫界中取 $t = a$ 并代入 $M_Z(t) = e^{t^2/2}$ 得到。

### 单侧切比雪夫不等式（坎泰利不等式）

对均值为 $\mu$、方差为 $\sigma^2$ 的随机变量：

$$P(X - \mu \geq t) \leq \frac{\sigma^2}{\sigma^2 + t^2}.$$

此式对单侧偏差的估计比切比雪夫更紧，且无需对称性假设。

## 关键公式汇总

| 量 | 公式 | 说明 |
|---|---|---|
| $E[X]$ | $\sum x \, p(x)$ 或 $\int x \, f(x) \, dx$ | 分布的中心位置 |
| $E[g(X)]$ | $\sum g(x) \, p(x)$ 或 $\int g(x) \, f(x) \, dx$ | LOTUS |
| $\text{Var}(X)$ | $E[X^2] - (E[X])^2$ | 分布的离散程度 |
| $\text{Var}(aX + b)$ | $a^2 \text{Var}(X)$ | 平移不影响离散程度 |
| $\text{Cov}(X,Y)$ | $E[XY] - E[X]E[Y]$ | 线性协同变化 |
| $\rho(X,Y)$ | $\text{Cov}(X,Y)/(\sigma_X \sigma_Y)$ | 标准化，取值范围 $[-1, 1]$ |
| $M_X(t)$ | $E[e^{tX}]$ | 生成所有矩 |
| 塔式性质 | $E[E[X|Y]] = E[X]$ | 迭代期望 |
| 全方差公式 | $\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$ | 组内 + 组间 |
| 詹森不等式 | $g(E[X]) \leq E[g(X)]$（$g$ 凸） | 基础不等式 |
| 马尔可夫不等式 | $P(X \geq a) \leq E[X]/a$ | 要求 $X \geq 0$ |
| 切比雪夫不等式 | $P(|X-\mu| \geq k\sigma) \leq 1/k^2$ | 分布无关 |

## 下一步

至此，我们一直只处理单个随机变量。但真实数据是**多维的**：身高与体重相关联，数据集中的特征相互作用，误差在函数中传播。下一篇文章将探讨**联合分布**——多个随机变量共存的数学，内容包括边缘分布、条件分布、变量变换以及二元正态分布。