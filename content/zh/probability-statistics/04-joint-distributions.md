---
title: "概率与统计（四）：联合分布、边缘化与独立性"
date: 2024-08-23 09:00:00
tags:
  - Probability
  - Statistics
  - Joint Distributions
  - Transformations
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "联合概率质量函数（PMF）与概率密度函数（PDF）、边缘分布与条件分布、二元正态分布、基于雅可比行列式的变量变换、卷积、次序统计量——含严格证明与等高线图可视化。"
disableNunjucks: true
series_order: 4
series_total: 8
translationKey: "probability-statistics-4"
---
到目前为止，我们研究过的每一种分布都只描述**单个**随机量：一次骰子投掷、一个等待时间、一次测量值。但真正有趣的问题往往涉及**多个变量之间的关系**：学习时长是否能预测考试成绩？不同行业的股票收益率是否相关？两个随机变量之和的行为如何？

要回答这些问题，我们需要 **联合分布（Joint Distributions）** —— 这是同时刻画多个随机变量的数学框架。也正是从这里开始，概率论直接连接到回归分析、多元统计学，以及机器学习所依赖的高维空间。

---

## 联合分布：离散情形

### 联合概率质量函数（Joint PMF）

若 $X$ 和 $Y$ 是定义在同一概率空间上的离散型随机变量，则其 **联合 PMF** 定义为：

![联合概率质量函数表](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-joint-pmf.png)
$$p_{X,Y}(x, y) = P(X = x, Y = y)$$
对所有 $(x, y)$ 成立。

性质：
1. $p_{X,Y}(x, y) \geq 0$，对所有 $(x, y)$ 成立；
2. $\sum_x \sum_y p_{X,Y}(x, y) = 1$。

**示例。** 投掷两枚均匀骰子。令 $X$ 表示第一枚骰子的点数，$Y$ 表示两枚骰子点数之和。则联合 PMF 为：当 $x \in \{1,\dots,6\}$ 且 $y = x + j$（其中 $j \in \{1,\dots,6\}$）时，$p_{X,Y}(x, y) = 1/36$。

### 边缘分布（Marginal Distributions）

$X$ 的 **边缘 PMF** 通过对 $Y$ 求和得到：

![边缘投影](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-marginal-projection.png)
$$p_X(x) = \sum_y p_{X,Y}(x, y).$$
同理：
$$p_Y(y) = \sum_x p_{X,Y}(x, y).$$
这是离散情形下“对某变量积分消去（integrating out）”的对应操作：通过遍历另一变量的所有可能取值求和，将联合分布“坍缩”为单变量分布。

**关键点：** 仅知道边缘分布 **并不能唯一确定** 联合分布。许多不同的联合分布可以产生完全相同的边缘分布。联合分布所包含的信息，严格多于各边缘分布信息之和。

### 条件分布（离散情形）

给定 $Y = y$ 时，$X$ 的 **条件 PMF** 定义为：
$$p_{X|Y}(x | y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}, \quad \text{要求 } p_Y(y) > 0.$$
这正是条件概率公式 $P(A|B) = P(A \cap B)/P(B)$ 在随机变量上的自然推广。

对每个固定的 $y$，函数 $p_{X|Y}(\cdot | y)$ 构成一个合法的 PMF（非负、总和为 1）。

## 联合分布：连续情形

### 联合概率密度函数（Joint PDF）

对于连续型随机变量，其 **联合 PDF** $f_{X,Y}(x, y)$ 满足：
$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$
对任意“合理”的集合 $A \subseteq \mathbb{R}^2$ 成立。

性质：
1. $f_{X,Y}(x, y) \geq 0$；
2. $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$。

联合累积分布函数（CDF）为：
$$F_{X,Y}(x, y) = P(X \leq x, Y \leq y) = \int_{-\infty}^x \int_{-\infty}^y f_{X,Y}(s, t) \, dt \, ds.$$
### 边缘 PDF
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy, \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx.$$
### 条件 PDF
$$f_{X|Y}(x | y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}, \quad \text{要求 } f_Y(y) > 0.$$
**示例。** 设 $(X, Y)$ 的联合 PDF 为  
$$f(x, y) = 6(1 - y), \quad \text{当 } 0 < x < y < 1,$$
其余区域 $f(x,y) = 0$。

*验证其为合法 PDF：*
$$\int_0^1 \int_0^y 6(1-y) \, dx \, dy = \int_0^1 6y(1-y) \, dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6 \cdot \frac{1}{6} = 1. \quad \checkmark$$
*求 $f_X(x)$：*
$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[y - \frac{y^2}{2}\right]_x^1 = 6\left[\left(1 - \frac{1}{2}\right) - \left(x - \frac{x^2}{2}\right)\right] = 6\left[\frac{1}{2} - x + \frac{x^2}{2}\right] = 3(1-x)^2 \quad \text{对 } 0 < x < 1.$$
*求 $f_Y(y)$：*
$$f_Y(y) = \int_0^y 6(1 - y) \, dx = 6y(1 - y) \quad \text{对 } 0 < y < 1.$$
*验证：* $\int_0^1 6y(1-y) \, dy = 6 \left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6 \cdot \frac{1}{6} = 1$. $\checkmark$

*求 $f_{X|Y}(x | y)$：*
$$f_{X|Y}(x | y) = \frac{6(1-y)}{6y(1-y)} = \frac{1}{y} \quad \text{对 } 0 < x < y.$$
即在给定 $Y = y$ 的条件下，$X$ 在区间 $(0, y)$ 上服从均匀分布。这很直观：条件密度不依赖于 $x$（平坦），且支撑集为 $(0, y)$。

## 独立性（Independence）

![独立与依赖链 vs 自由浮动变量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/04-independence-vs-dependence-linked-chains-vs-free-floating-ev.jpg)

随机变量 $X$ 和 $Y$ **相互独立**，当且仅当其联合分布可分解为边缘分布的乘积：
$$f_{X,Y}(x, y) = f_X(x) \, f_Y(y) \quad \text{对所有 } (x, y) \text{ 成立}.$$
离散情形下等价地：$p_{X,Y}(x, y) = p_X(x) \, p_Y(y)$。

**检验独立性：** 尝试将联合 PDF/PMF 分解。若可写为 $f_{X,Y}(x, y) = g(x) h(y)$（其中 $g,h$ 为某函数），且其支撑集（support）为矩形区域，则 $X$ 与 $Y$ 独立。

上例中，$f(x, y) = 6(1-y)$ 的支撑集为三角形区域 $\{0 < x < y < 1\}$。非矩形支撑集立即表明 $X$ 与 $Y$ **不独立**：已知 $Y = 0.3$ 将 $X$ 限制在 $(0, 0.3)$ 内，即 $Y$ 对 $X$ 施加了约束。

**独立性的第二重检验。** 即使支撑集为矩形，联合 PDF 仍必须能分解为 $g(x)h(y)$ 形式。例如，在 $[0,1]^2$ 上，$f(x,y) = 4xy$ 可分解为 $(2x)(2y)$，故 $X$ 与 $Y$ 独立，且 $f_X(x) = 2x$, $f_Y(y) = 2y$；但 $f(x,y) = 2(x+y)$ 在 $[0,1]^2$ 上无法分解（含交叉项），因此 $X$ 与 $Y$ 依赖，尽管支撑集是矩形。

## 二元正态分布（The Bivariate Normal Distribution）

![作为3D地形图的联合概率分布及其边缘](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/04-joint-probability-distribution-as-3d-terrain-map-with-margin.jpg)

最重要的多元分布。若 $(X, Y)$ 服从二元正态分布，则其联合 PDF 为：

![二元正态分布等高线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-bivariate-normal.png)
$$f(x, y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1 - \rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)} \left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)$$
其中 $\mu_X, \mu_Y$ 为均值，$\sigma_X, \sigma_Y$ 为标准差，$\rho = \text{Corr}(X, Y)$ 为相关系数。

关键性质：
1. 两个边缘分布均为正态：$X \sim \mathcal{N}(\mu_X, \sigma_X^2)$，$Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$；
2. 所有条件分布均为正态：  
   
$$
X | Y = y \sim \mathcal{N}\left(\mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \sigma_X^2(1 - \rho^2)\right)
$$
3. $X$ 与 $Y$ 独立 **当且仅当** $\rho = 0$。

性质 3 是二元正态分布特有的。一般情况下，$\rho = 0$ **并不意味着** 独立；但对于联合正态变量，它确实等价于独立。

条件均值 $E[X | Y = y] = \mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y)$ 是 $y$ 的线性函数。这正是 **回归直线（regression line）** —— 概率论与现代机器学习之间最早期的联系之一。

条件方差 $\sigma_X^2(1 - \rho^2)$ 不依赖于 $y$。这意味着 $X$ 围绕其条件均值的“离散程度”对所有 $Y$ 的取值都相同 —— 这一性质称为 **同方差性（homoscedasticity）**。因子 $(1 - \rho^2)$ 刻画了已知 $Y$ 后对 $X$ 不确定性的削减程度：

- $\rho = 0$：无削减（$\text{Var}(X|Y) = \sigma_X^2$）；
- $|\rho| = 0.5$：削减 25% （$\text{Var}(X|Y) = 0.75\sigma_X^2$）；
- $|\rho| = 0.9$：削减 81% （$\text{Var}(X|Y) = 0.19\sigma_X^2$）；
- $|\rho| = 1$：完全消除（$\text{Var}(X|Y) = 0$，即 $X$ 完全由 $Y$ 决定）。

量 $R^2 = \rho^2$ 即为 **决定系数（coefficient of determination）** —— 表示 $Y$ 解释的 $X$ 方差比例。这正是线性回归中计算的 $R^2$。

## 随机变量的变换（Transformations of Random Variables）

已知 $X$ 的 PDF $f_X(x)$，求 $Y = g(X)$ 的 PDF。

### CDF 方法（最通用）

1. 计算 $F_Y(y) = P(Y \leq y) = P(g(X) \leq y)$；
2. 将其用 $X$ 表达，并利用 $f_X$；
3. 求导：$f_Y(y) = F_Y'(y)$。

**示例。** 设 $X \sim \text{Uniform}(0, 1)$，$Y = -\frac{1}{\lambda} \ln(X)$，$\lambda > 0$。
$$F_Y(y) = P\left(-\frac{1}{\lambda}\ln X \leq y\right) = P(\ln X \geq -\lambda y) = P(X \geq e^{-\lambda y}) = 1 - e^{-\lambda y}$$
对 $y \geq 0$ 成立。求导得：$f_Y(y) = \lambda e^{-\lambda y}$，即 $\text{Exponential}(\lambda)$ 的 PDF。

这就是生成随机样本的 **逆 CDF 法（inverse CDF method）** —— 蒙特卡洛模拟的基石。

### 雅可比方法（Jacobian Method，变量替换）

对一一映射（one-to-one）变换 $Y = g(X)$，设其可微反函数为 $X = g^{-1}(Y)$，则：
$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{dg^{-1}}{dy}\right|.$$
导数的绝对值（即 **雅可比行列式**）用于校正因拉伸/压缩导致的密度变化。

*推导。* 若 $g$ 严格递增：$F_Y(y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y))$。链式法则求导：$f_Y(y) = f_X(g^{-1}(y)) \cdot (g^{-1})'(y)$。若 $g$ 递减，会引入负号，故取绝对值。$\blacksquare$

### 多元雅可比方法（Multivariate Jacobian）

对变换 $(X_1, X_2) \to (Y_1, Y_2)$，其中 $y_1 = g_1(x_1, x_2)$，$y_2 = g_2(x_1, x_2)$，反函数为 $(x_1, x_2) = h(y_1, y_2)$，则：
$$f_{Y_1, Y_2}(y_1, y_2) = f_{X_1, X_2}(h_1(y_1,y_2), h_2(y_1,y_2)) \left|J\right|$$
其中雅可比行列式为：
$$J = \det \begin{pmatrix} \frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\ \frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} \end{pmatrix}.$$
## 随机变量之和：卷积（Convolution）

若 $X$ 与 $Y$ 独立， PDF 分别为 $f_X$ 和 $f_Y$，则 $Z = X + Y$ 的 PDF 为 **卷积**：

![随机变量的卷积](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-convolution.png)
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, dx = (f_X * f_Y)(z).$$
*推导。* 令 $Z = X + Y$，并引入辅助变量 $W = X$：
$$f_{Z,W}(z, w) = f_{X,Y}(w, z - w) |J| = f_X(w) f_Y(z - w) \cdot 1.$$
对 $W$ 边缘化：$f_Z(z) = \int f_X(w) f_Y(z - w) dw$. $\blacksquare$

**示例。** 两个独立 $\text{Exponential}(\lambda)$ 随机变量之和。
$$f_Z(z) = \int_0^z \lambda e^{-\lambda x} \lambda e^{-\lambda(z-x)} dx = \lambda^2 e^{-\lambda z} \int_0^z dx = \lambda^2 z e^{-\lambda z}$$
对 $z > 0$ 成立。此即 $\text{Gamma}(2, \lambda)$，印证了 $n$ 个独立指数分布之和为 Gamma 分布。

### 矩母函数法（MGF Approach，常更简便）

因对独立 $X,Y$ 有 $M_{X+Y}(t) = M_X(t) M_Y(t)$，可通过相乘 MGF 并识别结果来确定和的分布。

**示例。** 独立正态变量之和。若 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$，$Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$：
$$M_{X+Y}(t) = e^{\mu_1 t + \sigma_1^2 t^2/2} \cdot e^{\mu_2 t + \sigma_2^2 t^2/2} = e^{(\mu_1 + \mu_2)t + (\sigma_1^2 + \sigma_2^2)t^2/2}$$
这正是 $\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ 的 MGF。故独立正态变量之和仍为正态，且均值与方差分别相加。$\blacksquare$

## 多项分布（The Multinomial Distribution）

二项分布的多元推广。在 $n$ 次独立试验中，每次试验落入 $k$ 个类别之一，概率分别为 $p_1, \ldots, p_k$（$\sum p_i = 1$）。计数向量 $(X_1, \ldots, X_k)$ 的分布为：
$$P(X_1 = n_1, \ldots, X_k = n_k) = \frac{n!}{n_1! \cdots n_k!} p_1^{n_1} \cdots p_k^{n_k}$$
其中 $\sum n_i = n$。

性质：
- 每个 $X_i$ 边缘上服从 $\text{Binomial}(n, p_i)$；
- $\text{Cov}(X_i, X_j) = -np_i p_j$（$i \neq j$）（计数负相关：若更多试验落入类别 $i$，则留给类别 $j$ 的试验数减少）。

## 次序统计量（Order Statistics）

给定 $n$ 个 i.i.d. 随机变量 $X_1, \ldots, X_n$，其 **次序统计量** 为排序后的值 $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$。

![顺序统计量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-order-statistics.png)

- $X_{(1)} = \min(X_1, \ldots, X_n)$；
- $X_{(n)} = \max(X_1, \ldots, X_n)$；
- $X_{(k)}$ 为第 $k$ 小的值。

### 最大值的 CDF
$$F_{X_{(n)}}(x) = P(\max_i X_i \leq x) = P(X_1 \leq x, \ldots, X_n \leq x) = [F_X(x)]^n.$$
求导得：$f_{X_{(n)}}(x) = n [F_X(x)]^{n-1} f_X(x)$。

### 最小值的 CDF
$$P(X_{(1)} > x) = P(\min_i X_i > x) = [1 - F_X(x)]^n.$$
故 $F_{X_{(1)}}(x) = 1 - [1 - F_X(x)]^n$，且 $f_{X_{(1)}}(x) = n [1 - F_X(x)]^{n-1} f_X(x)$。

**示例。** 若 $X_i \sim \text{Exp}(\lambda)$，则 $n$ 个独立副本的最小值满足：
$$P(X_{(1)} > x) = [e^{-\lambda x}]^n = e^{-n\lambda x}$$
故 $X_{(1)} \sim \text{Exp}(n\lambda)$。$n$ 个指数分布的最小值仍是指数分布，速率变为 $n\lambda$ —— 观察更多独立过程，首次事件的等待时间缩短。

### 一般第 $k$ 个次序统计量

$X_{(k)}$ 的 PDF 为：
$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F_X(x)]^{k-1} [1 - F_X(x)]^{n-k} f_X(x).$$
该式统计了：恰好 $k-1$ 个值小于 $x$、一个值恰在 $x$、$n-k$ 个值大于 $x$ 的方式数。

## Python：联合分布可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Bivariate normals with different correlations
rhos = [-0.8, 0, 0.8]
for idx, rho in enumerate(rhos):
    ax = axes[0, idx]
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    # Generate samples
    np.random.seed(42)
    samples = np.random.multivariate_normal(mean, cov, 1000)

    # Scatter plot
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='steelblue')

    # Contour plot
    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean, cov)
    ax.contour(X, Y, rv.pdf(pos), levels=5, colors='darkred', linewidths=1.5)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Bivariate Normal, $\\rho$ = {rho}', fontsize=13)
    ax.set_aspect('equal')

# Marginal and conditional illustration
ax = axes[1, 0]
rho = 0.7
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
np.random.seed(42)
samples = np.random.multivariate_normal(mean, cov, 2000)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=8, c='gray')

# Highlight conditional: X | Y = 1
y_val = 1.0
mask = np.abs(samples[:, 1] - y_val) < 0.15
ax.scatter(samples[mask, 0], samples[mask, 1], c='red', s=15, alpha=0.7,
           label=f'X | Y $\\approx$ {y_val}')
cond_mean = rho * y_val
ax.axvline(cond_mean, color='red', linestyle='--', alpha=0.7,
           label=f'E[X|Y={y_val}] = {cond_mean:.1f}')
ax.axhline(y_val, color='orange', linestyle=':', alpha=0.5)
ax.set_title('Conditional Distribution', fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(fontsize=9)

# Order statistics
ax = axes[1, 1]
np.random.seed(42)
n_values = [2, 5, 10, 20]
x = np.linspace(0, 1, 200)
for n in n_values:
    # PDF of max of n Uniform(0,1)
    pdf_max = n * x**(n-1)
    ax.plot(x, pdf_max, linewidth=2, label=f'max of {n}')
ax.set_title('Order Statistics: Max of Uniform(0,1)', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# Convolution: sum of exponentials
ax = axes[1, 2]
x = np.linspace(0, 10, 300)
lam = 1.0
for n in [1, 2, 3, 5]:
    # Gamma(n, lambda) PDF
    pdf = stats.gamma.pdf(x, n, scale=1/lam)
    ax.plot(x, pdf, linewidth=2, label=f'Sum of {n} Exp({lam})')
ax.set_title('Sum of Exponentials = Gamma', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

plt.tight_layout()
plt.savefig('joint_distributions.png', dpi=150)
plt.show()
```

顶行展示了相关系数 $\rho$ 如何塑造二元正态分布：负相关使椭圆沿某一方向倾斜，零相关给出圆形（独立），正相关则朝另一方向倾斜。底行展示了条件分布（红色切片）、次序统计量（最大值随 $n$ 增大而右移）、以及卷积（指数分布之和趋近 Gamma 分布）。

## 多元正态分布（The Multivariate Normal Distribution）

二元正态可推广至 $d$ 维。随机向量 $\mathbf{X} = (X_1, \ldots, X_d)^T$ 服从 **多元正态分布** $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，当且仅当其 PDF 为：
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$
其中 $\boldsymbol{\mu} \in \mathbb{R}^d$ 为均值向量，$\boldsymbol{\Sigma}$ 为 $d \times d$ 正定协方差矩阵。

### 关键性质

1. **边缘分布为正态。** 任意分量子集服从多元正态分布（只需取 $\boldsymbol{\mu}$ 的对应子向量及 $\boldsymbol{\Sigma}$ 的对应子矩阵）。

2. **线性变换保持正态性。** 若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，且 $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$，则 $\mathbf{Y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$。

3. **不相关蕴含独立。** 对联合正态变量，$\text{Cov}(X_i, X_j) = 0$ 意味着 $X_i$ 与 $X_j$ 独立。这是正态分布特有的性质，对一般分布不成立。

4. **条件分布为正态。** 将 $\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}$ 分块，对应均值与协方差亦分块。则：
$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1\mid 2}, \boldsymbol{\Sigma}_{1\mid 2})$$
其中 $\boldsymbol{\mu}_{1\mid 2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$，$\boldsymbol{\Sigma}_{1\mid 2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$。

条件均值是条件变量的线性函数 —— 这正是线性回归的基础。条件协方差完全不依赖于 $\mathbf{x}_2$ —— 不确定性的“形状”不变，仅中心发生平移。

## 协方差矩阵性质（Covariance Matrix Properties）

协方差矩阵 $\boldsymbol{\Sigma}$ 具有：
- **对称性：** $\boldsymbol{\Sigma}^T = \boldsymbol{\Sigma}$；
- **半正定性：** $\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a} \geq 0$，对所有向量 $\mathbf{a}$ 成立。

*半正定性证明。* $\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \geq 0$（方差恒非负）。$\blacksquare$

**特征分解** $\boldsymbol{\Sigma} = Q\Lambda Q^T$ 揭示了分布的主轴方向。特征向量给出最大/最小方差的方向，特征值给出沿这些方向的方差大小。这正是 **主成分分析（PCA）** 所计算的内容 —— 它寻找能捕获数据最多方差的方向。

### 马氏距离（The Mahalanobis Distance）

多元正态 PDF 的指数部分定义了一种距离度量：
$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}.$$
此即 **马氏距离（Mahalanobis distance）** —— 它衡量某点距均值有多少个“标准差”，同时考虑了变量间的相关性及不同尺度。位于马氏距离为常数的椭球面上的点，在多元正态下具有相同密度。

马氏距离退化为：
- 当 $\boldsymbol{\Sigma} = \sigma^2 I$（各向同性协方差）时，即为欧氏距离；
- 在一维情形下，即为标准化距离 $|x - \mu|/\sigma$。

在机器学习中，马氏距离被用于异常检测、聚类，以及高斯判别分析（Gaussian discriminant analysis）的核函数。

### 卡方分布关联（Chi-Squared Connection）

若 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I_d)$，则 $\|\mathbf{Z}\|^2 = Z_1^2 + \cdots + Z_d^2 \sim \chi^2(d)$。更一般地，若 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则：
$$(\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(d).$$
这解释了为何卡方分布在拟合优度检验、似然比检验及多元参数的置信域中频繁出现。

## Copula：分离边缘分布与相依结构（Copulas: Separating Marginals from Dependence）

**Sklar 定理** 指出：任何联合分布均可分解为：
1. 各变量的边缘分布；
2. 一个刻画相依结构的 **Copula 函数**。
$$F(x, y) = C(F_X(x), F_Y(y))$$
其中 $C: [0,1]^2 \to [0,1]$ 为 Copula 函数。

这意味着你可以**分别建模边缘分布与相依结构**。两组数据可拥有相同的边缘分布（如均为高斯），却具有截然不同的相依结构（如高斯 Copula vs 重尾的 $t$-Copula）。

Copula 广泛应用于金融（建模违约相关性）、气候科学（温度与降水的联合建模），以及任何变量间关系超越线性相关的领域。

## 贝叶斯网络与条件独立性（Bayesian Networks and Conditional Independence）

在高维情形下，指定完整联合分布不现实 —— $d$ 个二元变量的联合 PMF 有 $2^d - 1$ 个自由参数。**贝叶斯网络**（有向图模型）利用条件独立性高效分解联合分布。

贝叶斯网络编码如下分解：
$$p(x_1, x_2, \ldots, x_d) = \prod_{i=1}^d p(x_i \mid \text{parents}(x_i))$$
其中每个变量仅依赖于其在有向无环图（DAG）中的父节点。这可将 $2^d - 1$ 个参数大幅缩减。

**条件独立性** $X \perp Y \mid Z$ 意味着 $p(x, y \mid z) = p(x \mid z) p(y \mid z)$。已知 $Z$ “屏蔽”了 $X$ 与 $Y$ 间的依赖。

**示例。** $X$ = “洒水器开启”，$Y$ = “下雨”，$Z$ = “草地湿润”。若已知草地湿润，则洒水器与下雨成为条件依赖（若草地湿且未下雨，则洒水器很可能开启了）。但边际上，洒水器与下雨可能独立。这是 **解释消解（explaining away）** 的经典例子 —— 概率推理中的核心概念。

理解 **边际独立性**（$X \perp Y$）与 **条件独立性**（$X \perp Y \mid Z$）的区别，对因果推断与图模型至关重要。两变量可边际独立但条件依赖（碰撞偏差， collider bias），或边际依赖但条件独立（中介效应， mediation）。 DAG 的结构决定了哪些独立性成立。

对数学感兴趣的读者： DAG 中的 **d-分离（d-separation）准则** 提供了一套完整的图形规则，可直接从网络结构读出条件独立性，无需显式计算任何概率。图论与概率论的这一深刻联系，构成了现代统计学（尤其是 Judea Pearl 发展的因果推断）的基石。

## 多个随机变量的函数（Functions of Multiple Random Variables）

除求和外，我们常需其他函数的分布。

### 独立随机变量的乘积

若 $X$ 与 $Y$ 独立且为正，$Z = XY$ 的 PDF 为：
$$f_Z(z) = \int_0^{\infty} f_X(x) f_Y(z/x) \frac{1}{x} dx.$$
### 独立随机变量的比值

$Z = X/Y$（$Y > 0$）的 PDF 为：
$$f_Z(z) = \int_0^{\infty} y \, f_X(zy) f_Y(y) \, dy.$$
**示例。** 若 $X \sim \mathcal{N}(0,1)$ 且 $Y \sim \chi^2(n)/n$ 独立，则 $X/\sqrt{Y} \sim t_n$（自由度为 $n$ 的学生 $t$ 分布）。这正是 $t$ 检验（第 7 篇文章）中自然出现的分布。

### 随机变量的最大值与最小值

对独立随机变量，最大值与最小值的分布可由 CDF 推导：

**最大值：** $F_{\max}(z) = P(\max(X, Y) \leq z) = P(X \leq z)P(Y \leq z) = F_X(z)F_Y(z)$。

**最小值：** $F_{\min}(z) = P(\min(X, Y) \leq z) = 1 - P(X > z)P(Y > z) = 1 - (1-F_X(z))(1-F_Y(z))$。

**示例。** 串联的两个部件（系统在任一部件失效时即失效）：寿命 = $\min(X, Y)$；并联的两个部件（系统仅在两部件均失效时才失效）：寿命 = $\max(X, Y)$。若两者寿命均服从 $\text{Exponential}(\lambda)$：

- 串联：$\min(X,Y) \sim \text{Exponential}(2\lambda)$ —— 失效速度翻倍；
- 并联：$F_{\max}(z) = (1-e^{-\lambda z})^2$，非指数分布；平均寿命 = $3/(2\lambda)$ —— 比单个部件提升 50%。

## 总结
| 概念 | 关键公式 | 解释 |
|---|---|---|
| 联合 PMF/PDF | $p(x,y)$ 或 $f(x,y)$ | $(X,Y)$ 的完整概率描述 |
| 边缘分布 | $f_X(x) = \int f(x,y) dy$ | “遗忘”另一变量 |
| 条件分布 | $f(x\midy) = f(x,y)/f_Y(y)$ | 已知 $Y=y$ 时 $X$ 的分布 |
| 独立性 | $f(x,y) = f_X(x) f_Y(y)$ | 联合分布可分解为边缘分布乘积 |
| 雅可比行列式 | $f_Y(y) = f_X(g^{-1}(y)) \middg^{-1}/dy\mid$ | 正确变换概率密度 |
| 卷积 | $f_{X+Y} = f_X * f_Y$ | 独立变量之和的 PDF |
| 次序统计量 | $f_{X_{(k)}}$ 含 $F^{k-1}(1-F)^{n-k}f$ | 第 $k$ 小值的分布 |
| 多元正态条件分布 | $\boldsymbol{\mu}_{1\mid2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ | 线性回归 |

## 下一步（What's Next）

我们现已掌握描述多个随机变量协同行为的工具。下一篇文章将探讨概率论的皇冠明珠：**大数定律（Law of Large Numbers）** 与 **中心极限定理（Central Limit Theorem）**。这两个定理解释了为何样本均值可靠，为何正态分布在各处涌现 —— 并直接关联到随机梯度下降（SGD）等算法的收敛性。
