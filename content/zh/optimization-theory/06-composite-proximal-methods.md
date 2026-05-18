---
title: '优化理论（六）：复合优化与近端方法'
date: 2022-09-21 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
lang: zh
mathjax: true
description: "系统讲解近端算子的理论与应用：凸分析基础、Moreau 包络、常见近端闭式解，以及 ISTA/FISTA、ADMM 等算法中的实际用法。"
disableNunjucks: true
translationKey: "optim-06"
series: optimization-theory
series_order: 6
series_total: 12
aliases:
  - /zh/standalone/近端算子/
---
当目标函数包含不可导项（如稀疏正则、TV 正则或约束集的指示函数），又或者约束难以直接处理时，“直接上梯度下降”往往会卡住——要么在不可导点处没有梯度可用，要么每一步都破坏可行性。**近端算子（proximal operator）** 提供了一种精巧而优美的解决方案：把每次更新理解为“先对光滑部分走一步，再通过一个带二次惩罚的小规模优化，将当前点拉回具有特定结构的解空间”。

本文将从凸分析所需的最小工具集出发，推导 Moreau 包络与近端映射的核心性质，列举实际中高频使用的闭式近端算子，并将其嵌入 ISTA、FISTA、ADMM、SVM 和稀疏优化等具体算法中——重点解释每个组件为何有效、何时一种方法优于另一种，以及实现中最容易踩的坑。

---

一旦目标函数里出现一项不可导的东西——L1 正则、TV 正则、或者一个集合的指示函数——梯度下降就开始打嗝：要么在尖点处没有梯度可用，要么每走一步都把约束破坏掉。

**近端算子（proximal operator）** 是处理这种情况的标准答案，但第一次接触它的人很容易被一堆术语吓到：Moreau 包络、共轭、ISTA、ADMM……听上去全是新名词。

这篇文章我想做的事情是：先用一句话告诉你它在干嘛——

> *“先对光滑那部分走一小步，再用一个带二次惩罚的小优化把当前点拉回到具有特定结构的区域。”*

然后再去推那些性质和闭式解。等你看到 ISTA、FISTA、ADMM 这些算法的骨架时，会发现它们其实都是同一个动作的不同包装。

## 你将学到什么
- 凸分析最小工具集：凸集、凸函数、子梯度
- 近端算子：定义、几何直觉、四条核心性质
- 三个日常高频闭式近端：软阈值、投影、二次收缩
- Moreau 包络：如何将非光滑函数光滑化，及其驱动 ISTA 的梯度恒等式
- ISTA：最简形式的近端梯度算法
- FISTA：通过动量加速达到 $O(1/k^2)$ 收敛率
- LASSO 端到端求解：从理论到清晰的 Python 实现
- 一页讲清 ADMM，及其与近端梯度族的关系
- 常见实现陷阱：步长选择、Lipschitz 常数估计、收敛性判断

## 前置知识

- 多元微积分（梯度、链式法则）
- 线性代数基础（范数、内积、特征值）
- 一点凸优化常识（梯度下降、强凸性）

---

## 凸分析基础

在讨论近端算子前，需先厘清三个基本概念：凸集、凸函数和子梯度。后续所有关键性质（如非扩张性、闭式解存在性、收敛速率）都建立在这三者之上。

### 凸集与凸函数

**凸集**：集合 $C \subseteq \mathbb{R}^n$ 是凸的，当且仅当对任意 $x, y \in C$ 和 $\theta \in [0, 1]$，
$$\theta x + (1 - \theta) y \in C.$$
即任意两点间的线段完全包含在 $C$ 中。

**凸函数**：函数 $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是凸的，当且仅当其有效域 $\mathrm{dom}\,f$ 是凸集，且对任意 $x, y \in \mathrm{dom}\,f$ 和 $\theta \in [0, 1]$，
$$f\!\left(\theta x + (1 - \theta) y\right) \le \theta f(x) + (1 - \theta) f(y).$$
几何上，函数图像上任意两点间的弦始终位于图像上方（“碗状”）。

**两条关键事实**：

- **局部最小即全局最小**：凸函数的任一局部极小点都是全局最小点。
- **支撑超平面存在性**：凸集的每个边界点、凸函数的每个定义点处都存在支撑超平面——这正是子梯度的来源。

### 子梯度

可微凸函数处处有唯一梯度 $\nabla f(x)$。但像 $|x|$ 或 hinge 损失 $\max(0, 1 - t)$ 这类函数在“折点”处不可导，此时需引入**子梯度**。

**定义**：向量 $g \in \mathbb{R}^n$ 是凸函数 $f$ 在点 $x$ 处的一个子梯度，当且仅当对任意 $y$，
$$f(y) \ge f(x) + \langle g,\, y - x \rangle.$$
这意味着 $g$ 定义了一个位于函数图像下方、且在 $(x, f(x))$ 处接触图像的支撑超平面。所有这样的 $g$ 构成的集合称为**次微分** $\partial f(x)$。

**示例（绝对值函数）**：设 $f(t) = |t|$，则
$$
\partial |t| =
\begin{cases}
\{+1\}, & t > 0,\
\{-1\}, & t < 0,\
[-1, +1], & t = 0.
\end{cases}
$$
在 $t = 0$ 处，次微分是一个区间，这正是后文软阈值算子产生“死区”（输出为零的区间）的根本原因。

**性质**：

- 若 $f$ 在 $x$ 处可微，则 $\partial f(x) = \{\nabla f(x)\}$。
- $\partial f(x)$ 总是凸集（且当 $f$ 为闭真凸函数时，在 $\mathrm{dom}\,f$ 内部非空）。
- **最优性条件**：$x^\star$ 是 $f$ 的全局最小点当且仅当 $0 \in \partial f(x^\star)$。这一条件推广了“梯度为零”，是后续所有推导的核心工具。

---

## 近端算子

### 定义与几何直觉

对闭真凸函数 $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 和 $\lambda > 0$，**近端算子**定义为
$$
\mathrm{prox}_{\lambda f}(v) \;=\; \arg\min_{x \in \mathbb{R}^n}\left\{\, f(x) + \frac{1}{2\lambda} \|x - v\|_2^2 \,\right\}.
$$
当 $f$ 闭真凸时，该最小化问题有**唯一解**（目标函数强凸）。

**直观理解**：$\mathrm{prox}_{\lambda f}(v)$ 在“使 $f$ 尽可能小”和“尽量靠近 $v$”之间做权衡，$\lambda$ 控制这一权衡：

- $\lambda \to 0$：远离 $v$ 的惩罚极大，故 $\mathrm{prox}_{\lambda f}(v) \to v$；
- $\lambda \to \infty$：可自由最小化 $f$，故 $\mathrm{prox}_{\lambda f}(v) \to \arg\min f$。

![图1：近端算子和Moreau包络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/proximal-operator/fig1_prox_definition.png)

左图展示了这一权衡：蓝色为 $f$，灰色为二次锚定项，紫色为其和；橙色点即为 $\mathrm{prox}_{\lambda f}(v)$。右图暂且按下不表，待介绍 Moreau 包络时再回看。

### 四条核心性质

以下四条性质是后续所有算法分析的基石。

**(1) 存在性、唯一性与最优性条件**：$\mathrm{prox}_{\lambda f}(v)$ 存在且唯一，且 $x^\star = \mathrm{prox}_{\lambda f}(v)$ 当且仅当
$$\frac{1}{\lambda}(v - x^\star) \in \partial f(x^\star) \;\;\Longleftrightarrow\;\; v \in x^\star + \lambda\, \partial f(x^\star).$$
**(2) 不动点刻画**：$x^\star$ 最小化 $f$ 当且仅当 $x^\star = \mathrm{prox}_{\lambda f}(x^\star)$。这使得最小化问题可转化为不动点迭代。

**(3) 强非扩张性（firmly non-expansive）**：对任意 $u, v$，
$$\|\mathrm{prox}_{\lambda f}(u) - \mathrm{prox}_{\lambda f}(v)\|_2 \le \|u - v\|_2.$$
更强的“强”版本表明 $\mathrm{prox}_{\lambda f}$ 是 $\tfrac{1}{2}$-平均映射——这是 ISTA 收敛性的主要工具。

**(4) 可分离性**：若 $f(x) = \sum_i f_i(x_i)$，则
$$\bigl[\mathrm{prox}_{\lambda f}(v)\bigr]_i = \mathrm{prox}_{\lambda f_i}(v_i).$$
对于 $\ell_1$ 范数、盒约束等逐坐标函数，其近端算子可**完全并行**计算——这正是 LASSO 能扩展至百万维特征的关键原因。

---

## 三个常用闭式近端算子

### $\ell_1$ 范数：软阈值

设 $f(x) = \|x\|_1 = \sum_i |x_i|$。由可分离性，问题退化为一维：
$$\min_{x_i} |x_i| + \frac{1}{2\lambda}(x_i - v_i)^2.$$
通过对符号分类并应用 $0 \in \partial(\cdot)$，可得**软阈值算子**：
$$
\bigl[\mathrm{prox}_{\lambda \|\cdot\|_1}(v)\bigr]_i \;=\; \mathrm{soft}_\lambda(v_i) \;=\; \mathrm{sign}(v_i) \cdot \max\!\bigl(|v_i| - \lambda,\, 0\bigr).
$$
![图2：软阈值：L1范数的近端算子](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/proximal-operator/fig2_soft_threshold.png)

- 左图：当 $|v| \le \lambda$（“死区”）时输出精确为零；超出此范围时，$v$ 被**收缩** $\lambda$ 向零（注意是收缩而非截断）。
- 右图：对含噪信号应用一次软阈值，小幅噪声被压至零，尖峰则保留并轻微收缩——这正是 LASSO 将无关系数**精确归零**的机制。

**实现提示**：NumPy 中一行即可向量化实现：`np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)`。

### 指示函数：投影

对凸集 $C$，其指示函数
$$
\iota_C(x) = \begin{cases} 0, & x \in C, \ +\infty, & x
otin C, \end{cases}
$$
是凸函数。其近端算子即为**欧氏投影**：
$$\mathrm{prox}_{\lambda \iota_C}(v) = \arg\min_{x \in C} \tfrac{1}{2}\|x - v\|_2^2 = P_C(v).$$
注意结果**与 $\lambda$ 无关**（因指示函数取值仅为 $0$ 或 $+\infty$）。这表明“投影梯度法”是“近端梯度法”的特例——硬约束等价于无穷大的惩罚。

常见投影：

- $\ell_2$ 球 $\{x : \|x\|_2 \le r\}$：$P(v) = v \cdot \min(1, r / \|v\|_2)$。
- $\ell_\infty$ 球 $\{x : \|x\|_\infty \le 1\}$：$P(v)_i = \mathrm{clip}(v_i, -1, 1)$。
- 非负象限 $\mathbb{R}^n_{+}$：$P(v) = \max(v, 0)$（即 ReLU）。
- 单纯形 / $\ell_1$ 球：需排序，复杂度 $O(n \log n)$，但已有成熟算法。

### 平方范数：线性收缩

设 $f(x) = \tfrac{1}{2}\|x\|_2^2$。由一阶条件 $x + \tfrac{1}{\lambda}(x - v) = 0$ 得
$$\mathrm{prox}_{\lambda f}(v) = \frac{v}{1 + \lambda}.$$
这是朝原点的纯线性收缩——对应岭回归的正则形式。

**更一般的二次型** $f(x) = \tfrac{1}{2}x^\top Q x + b^\top x$（$Q \succeq 0$）：
$$\mathrm{prox}_{\lambda f}(v) = (I + \lambda Q)^{-1}(v - \lambda b),$$
需解一个线性系统——当 $Q$ 稀疏或具特殊结构时仍很实用。

### 无闭式解时的应对策略

并非所有 $f$ 都有闭式近端。常用备选方案包括：

- 对一阶最优条件使用**半光滑牛顿法**或 **Newton-CG**；
- **求解对偶问题**——许多复合近端在对偶空间更易计算（见下文 Moreau 分解）；
- **嵌入 ADMM**，将难解的近端拆分为两个易解子问题。

---

## Moreau 包络：光滑化非光滑函数

### 定义与图像

对闭真凸函数 $f$ 和 $\lambda > 0$，**Moreau 包络**定义为
$$\widehat{f}_\lambda(x) \;=\; \min_{y \in \mathbb{R}^n}\left\{ f(y) + \frac{1}{2\lambda}\|y - x\|_2^2 \right\}.$$
包络给出的是**最小值**（标量），而近端给出的是**最小化点**（向量）。二者源于同一优化问题，故关系紧密。

回到 Figure 1 右图：紫色和绿色曲线分别是 $f(x) = |x|$ 在 $\lambda = 0.5$ 和 $\lambda = 1.5$ 下的 Moreau 包络（即 Huber 函数）。原点处的“尖角”被光滑化为可微弧线，且**最小值与最小化点保持不变**。

### 三条关键性质

**(1) 最小值与最小化点不变**：
$$\inf_x f(x) = \inf_x \widehat{f}_\lambda(x), \qquad \arg\min f = \arg\min \widehat{f}_\lambda.$$
**(2) $\widehat{f}_\lambda$ 是凸且 $\tfrac{1}{\lambda}$-光滑的**。即使 $f$ 处处不可导，$\widehat{f}_\lambda$ 也处处可微，且其梯度为 $\tfrac{1}{\lambda}$-Lipschitz。

**(3) 梯度恒等式（核心工具）**：
$$\nabla \widehat{f}_\lambda(x) \;=\; \frac{1}{\lambda}\bigl(x - \mathrm{prox}_{\lambda f}(x)\bigr).$$
**为何重要**：它将“对包络做梯度下降”转化为“计算一次近端”——这正是 ISTA 的算法本质。

**简要推导**：令 $y^\star = \mathrm{prox}_{\lambda f}(x)$。一阶最优性给出 $0 \in \partial f(y^\star) + \tfrac{1}{\lambda}(y^\star - x)$，即 $\tfrac{1}{\lambda}(x - y^\star) \in \partial f(y^\star)$。对 $\widehat{f}_\lambda(x) = f(y^\star) + \tfrac{1}{2\lambda}\|y^\star - x\|^2$ 应用包络定理——因内层关于 $y$ 的偏导在最优处为零，仅剩 $\nabla_x \tfrac{1}{2\lambda}\|y - x\|^2 \big|_{y = y^\star} = \tfrac{1}{\lambda}(x - y^\star)$。

### Moreau 分解

一个有用的对偶恒等式：对闭真凸函数 $f$ 及其共轭 $f^*$，
$$v = \mathrm{prox}_{\lambda f}(v) + \lambda \cdot \mathrm{prox}_{f^* / \lambda}(v / \lambda).$$
实践中，若 $f$ 的近端难算但 $f^*$ 的近端易算（或反之），可在易算侧进行计算。经典应用如核范数近端（SVD 软阈值）与谱范数投影之间的转换。

---

## 近端梯度：ISTA

### 问题设定

考虑**复合优化**问题：
$$\min_{x \in \mathbb{R}^n} F(x) \;=\; g(x) + h(x),$$
其中

- $g$ 凸、可微，且 $\nabla g$ 为 $L$-Lipschitz（“光滑部分”），
- $h$ 凸、可能不可微，但 $\mathrm{prox}_{\lambda h}$ **易计算**（“非光滑部分”）。

LASSO 是典型例子：$g(x) = \tfrac{1}{2}\|Ax - y\|_2^2$ 光滑，$h(x) = \mu \|x\|_1$ 可通过软阈值计算。

### ISTA 迭代

**ISTA（Iterative Shrinkage-Thresholding Algorithm）** 将“对 $g$ 做一步梯度”与“对 $h$ 做一次近端”结合：
$$
\boxed{\;x_{k+1} \;=\; \mathrm{prox}_{\eta h}\!\bigl(x_k - \eta 
\nabla g(x_k)\bigr).\;}
$$
**主化视角**：用二次上界 $\widetilde{g}(x; x_k) = g(x_k) + \langle\nabla g(x_k), x - x_k \rangle + \tfrac{1}{2\eta}\|x - x_k\|_2^2$ 替代 $g$，然后最小化 $\widetilde{g}(x; x_k) + h(x)$——这恰好就是上述近端步骤。因此 ISTA 是 MM（主化-最小化）方法的一个实例。

**步长选择**：$\eta \le 1 / L$，其中 $L$ 是 $\nabla g$ 的 Lipschitz 常数。对 LASSO，$L = \|A\|_2^2$（最大奇异值平方），实践中两三次幂迭代即可足够准确。

**收敛速率**：对凸 $F$，
$$F(x_k) - F^\star \le \frac{\|x_0 - x^\star\|_2^2}{2\eta k} = O(1 / k).$$
![图3：二维LASSO上的ISTA](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/proximal-operator/fig3_ista_iterations.png)

Figure 3 展示了二维 LASSO 上的 ISTA 运行过程：灰色等高线为目标函数，橙线为稀疏轴（$x_2 = 0$）。从右上角紫星出发，ISTA 迭代点（蓝色折线）逐步逼近最优解，并**精确落在 $x_2 = 0$ 上**——软阈值诱导稀疏性的效果一目了然。

---

## 加速：FISTA

### 算法

ISTA 的 $O(1/k)$ 速率在大规模问题上较慢。**FISTA**（Beck & Teboulle, 2009）借鉴 Nesterov 动量，在外推点而非当前点计算梯度：
$$
\begin{aligned}
y_k &= x_k + \frac{t_{k-1} - 1}{t_k}\bigl(x_k - x_{k-1}\bigr), \
x_{k+1} &= \mathrm{prox}_{\eta h}\!\bigl(y_k - \eta 
\nabla g(y_k)\bigr), \
t_{k+1} &= \frac{1 + \sqrt{1 + 4 t_k^2}}{2}.
\end{aligned}
$$
初始化 $t_0 = 1$，$x_0 = x_{-1}$。

**收敛速率**：$F(x_k) - F^\star \le \dfrac{2 \|x_0 - x^\star\|_2^2}{\eta (k + 1)^2} = O(1/k^2)$。

![图4：FISTA加速与ISTA对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/proximal-operator/fig4_fista_acceleration.png)

Figure 4 在双对数坐标下绘制了 60 维 LASSO 上 ISTA 与 FISTA 的次优间隙。两条参考虚线（$1/k$ 和 $1/k^2$）几乎与实测曲线平行——加速效果真实存在，且在前 50 次迭代内，FISTA 已比 ISTA 快约一个数量级。

### 实现要点

- **重启策略**：$t_k$ 单调递增可能导致过度外推。简单有效的修复是**函数值重启**：若 $F(x_{k+1}) > F(x_k)$，则重置 $t_k = 1$。实践中常带来 2–3 倍额外加速。
- **强凸情形**：若 $g$ 还是 $\mu$-强凸的，APGD 等变体可实现线性收敛 $(1 - \sqrt{\mu/L})^k$。
- **近端近似**：即使近端仅近似求解，只要残差以 $1/k^{3/2}$ 衰减，FISTA 仍保持加速率。

---

## 应用：求解 LASSO

### 问题与解的几何特性

LASSO 问题：
$$\min_x \;\tfrac{1}{2}\|Ax - y\|_2^2 + \mu \|x\|_1.$$
**关键现象**：随着 $\mu$ 增大，越来越多系数被**精确推至零**——这使 LASSO 同时具备拟合与特征选择能力。

![图5：LASSO解路径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/proximal-operator/fig5_lasso_path.png)

Figure 5 展示经典的**LASSO 解路径**：横轴为 $\mu$（对数刻度），纵轴为各系数值。紫色实线对应 4 个真实非零特征，灰色虚线对应 8 个真实零特征。

- 小 $\mu$：所有系数非零（接近 OLS 解）；
- 大 $\mu$：灰色（无关）特征**率先**归零，紫色（相关）特征最后消失；
- 通过交叉验证选择 $\mu$，可获得稀疏且预测性能良好的模型。

### 清晰的 ISTA / FISTA 实现

```python
import numpy as np

def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def lasso_fista(A, y, mu, n_iter=500, tol=1e-8):
    """求解 min  0.5 ||Ax - y||^2 + mu ||x||_1。"""
    n, d = A.shape
    L = np.linalg.norm(A, 2) ** 2          # 1/eta = L
    eta = 1.0 / L

    x = np.zeros(d)
    x_prev = x.copy()
    t = 1.0

    for k in range(n_iter):
        # 外推点
        y_k = x + ((t - 1.0) / t) * (x - x_prev) if k > 0 else x
        # 在外推点处做一次近端梯度
        grad = A.T @ (A @ y_k - y)
        x_new = soft_threshold(y_k - eta * grad, eta * mu)

        # 收敛诊断
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            return x_new

        x_prev, x = x, x_new
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0

    return x
```

**实践建议**：

- $L = \|A\|_2^2$ 最好用幂迭代估计（避免完整 SVD）；
- 收敛判断用相对变化量而非梯度范数（折点处梯度不存在）；
- 若需计算多个 $\mu$ 的解路径，按 $\mu$ 从大到小顺序并使用**热启动**（warm start）——上一解作为下一初值，可大幅提速。

---

## 子梯度法 vs 近端法

子梯度法是处理非光滑问题的“原始”工具：
$$x_{k+1} = x_k - \eta_k g_k, \quad g_k \in \partial F(x_k),$$
但其收敛率仅为 $O(1/\sqrt{k})$，且需递减步长 $\eta_k = O(1/\sqrt{k})$ 才能保证收敛。

**方法对比**：

| 方法 | 所需结构 | 收敛率（凸） | 实现难度 | 备注 |
|---|---|---|---|---|
| 子梯度法 | 任意凸 $F$ | $O(1/\sqrt{k})$ | 低 | 通用但慢 |
| ISTA | $g$ 光滑 + $h$ 易 prox | $O(1/k)$ | 低 | LASSO 默认选择 |
| FISTA | 同 ISTA | $O(1/k^2)$ | 中 | 大规模首选 |
| ADMM | $\min g(x) + h(z)$ s.t. $Ax + Bz = c$ | $O(1/k)$（一般凸） | 中 | 适用于可拆分复合问题 |

**实践建议**：只要非光滑部分可分离且近端易算，优先使用近端方法——实际问题中通常能获得数量级的速度提升。

---

## 一页讲清 ADMM

当问题包含**两个非光滑项**或**线性等式约束**时，仅用 ISTA/FISTA 不够。**ADMM（Alternating Direction Method of Multipliers）** 将问题写为
$$\min_{x, z}\; g(x) + h(z) \quad \text{s.t.}\quad Ax + Bz = c,$$
并交替更新：
$$
\begin{aligned}
x_{k+1} &= \arg\min_x\; g(x) + \tfrac{\rho}{2}\|Ax + Bz_k - c + u_k\|_2^2, \
z_{k+1} &= \arg\min_z\; h(z) + \tfrac{\rho}{2}\|Ax_{k+1} + Bz - c + u_k\|_2^2, \
u_{k+1} &= u_k + Ax_{k+1} + Bz_{k+1} - c.
\end{aligned}
$$
每个子问题仅含**一个**非光滑项，故可用单次近端求解。

**LASSO 的 ADMM 形式**：将约束写为 $x = z$。$x$-更新为闭式岭回归解，$z$-更新为软阈值——简洁明了。

![图 6 - ADMM 迭代结构图与 LASSO 收敛对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/06-composite-proximal-methods/fig6_admm.png)

图 6 左侧展示一次完整 ADMM 迭代：$x$-更新处理光滑项 $g$ 与二次惩罚（LASSO 中为闭式岭回归），$z$-更新对 $h$ 做单次近端（LASSO 中为软阈值），对偶变量 $u$ 累积残差 $Ax+Bz-c$。右侧在同一 60 维 LASSO 实例上比较 ISTA、FISTA 与 ADMM（$\rho=1$）：ADMM 初期与 FISTA 相当，并最先达到机器精度——当子问题本身易解时，拆分策略往往更优。

ADMM 的优势：

- 适用于**两个非光滑项之和**（如 $\ell_1$ + 全变分）；
- 天然支持**分布式计算**（共识 ADMM）；
- 对一般凸问题为 $O(1/k)$，且常数常小于 ISTA。

ADMM 的代价：需选择 $\rho$，且 $x$-更新需解线性系统。

---

## 收敛性：实战考量

### ISTA / FISTA 的监控指标

1. **目标函数单调下降（ISTA）** 或 **重启后严格下降（FISTA）**；
2. **$\|x_{k+1} - x_k\|$ 衰减至容差**——非光滑情形下最可靠的收敛判据；
3. **活跃集稳定性**：一旦 $x_k$ 的非零位置不再变化，基本已接近最优。

### 常见陷阱

- **步长过大**：$\eta > 1/L$ 会导致发散。若 $L$ 不确定，使用**回溯线搜索**：尝试 $\eta \cdot \beta$，若不满足充分下降条件则减半。
- **解路径无热启动**：对每个 $\mu$ 从零开始是巨大浪费——务必热启动。
- **混淆 $\lambda$ 与 $\eta$**：注意 $\mathrm{prox}_{\eta \cdot \mu \|\cdot\|_1}$ 的阈值是 $\eta \mu$，而非 $\mu$ 或 $\eta$。
- **对 hinge / $\ell_1$ 直接用梯度下降**：会在折点附近震荡。应使用近端或子梯度法。

---

## 练习题

### 习题 1：闭式近端计算

计算下列函数的 $\mathrm{prox}_{\lambda f}$：

(a) $f(x) = \|x\|_1$。

**解**：由可分离性及一维子梯度分析，
$$\bigl[\mathrm{prox}_{\lambda f}(v)\bigr]_i = \mathrm{sign}(v_i)\max(|v_i| - \lambda, 0).$$
(b) $f(x) = \iota_{B_\infty}(x)$，其中 $B_\infty = \{x : \|x\|_\infty \le 1\}$。

**解**：投影至 $\ell_\infty$ 球，逐坐标截断：
$$\bigl[\mathrm{prox}_{\lambda f}(v)\bigr]_i = \min\bigl(\max(v_i, -1),\, 1\bigr).$$
注意结果与 $\lambda$ 无关。

(c) $f(x) = \tfrac{\beta}{3}\|x\|_3^3$（$\beta > 0$）。

**解**：可分离。对 $v_i \ge 0$，最小化点 $x_i \ge 0$ 满足 $\beta x_i^2 + \tfrac{1}{\lambda}(x_i - v_i) = 0$，即 $\lambda \beta x_i^2 + x_i - v_i = 0$：
$$\bigl[\mathrm{prox}_{\lambda f}(v)\bigr]_i = \mathrm{sign}(v_i) \cdot \frac{-1 + \sqrt{1 + 4\lambda\beta |v_i|}}{2\lambda\beta}.$$
这是少数 $\ell_p$ 范数（$p > 2$）存在闭式近端的例子。

### 习题 2：Moreau 包络的可微性

证明闭真凸函数 $f$ 的 Moreau 包络 $\widehat{f}_\lambda$ 处处可微，且
$$\nabla \widehat{f}_\lambda(x) = \frac{1}{\lambda}\bigl(x - \mathrm{prox}_{\lambda f}(x)\bigr).$$
**思路**：

1. 最小化点唯一，记 $y(x) := \mathrm{prox}_{\lambda f}(x)$。由非扩张性，$y(x)$ 关于 $x$ 是 1-Lipschitz 的。
2. 一阶条件给出 $\tfrac{1}{\lambda}(x - y(x)) \in \partial f(y(x))$。
3. 对 $\widehat{f}_\lambda(x) = f(y(x)) + \tfrac{1}{2\lambda}\|y(x) - x\|^2$ 应用包络定理：内层关于 $y$ 的偏导因最优性为零，仅剩 $\nabla_x \tfrac{1}{2\lambda}\|y - x\|^2 \big|_{y = y(x)} = \tfrac{1}{\lambda}(x - y(x))$。

因 $y(x)$ 1-Lipschitz，$\nabla \widehat{f}_\lambda$ 为 $\tfrac{1}{\lambda}$-Lipschitz——故包络自动光滑。

### 习题 3：为何 SVM 近端“无用”

考虑线性 SVM：$f(w) = \sum_i \max(0, 1 - y_i x_i^\top w) + \tfrac{\lambda}{2}\|w\|_2^2$。

(a) 给出 $f$ 在 $w$ 处的一个子梯度。

**解**：对 hinge 损失 $\ell_i(w) = \max(0, 1 - y_i x_i^\top w)$，
$$
\partial \ell_i(w) =
\begin{cases}
\{0\}, & y_i x_i^\top w > 1, \
\{- y_i x_i\}, & y_i x_i^\top w < 1, \
[-y_i x_i, 0], & y_i x_i^\top w = 1.
\end{cases}
$$
总子梯度：$\partial f(w)i \sum_i g_i + \lambda w$，其中 $g_i \in \partial \ell_i(w)$。

(b) 证明计算 $\mathrm{prox}_{\alpha f}(0)$ 与求解 SVM 本身难度相当。

**解**：由定义，
$$\mathrm{prox}_{\alpha f}(0) = \arg\min_w \;\sum_i \max(0, 1 - y_i x_i^\top w) + \tfrac{1}{2}\!\left(\lambda + \tfrac{1}{\alpha}\right)\|w\|_2^2.$$
这本身就是一个 SVM 问题，仅正则强度变为 $\lambda + 1/\alpha$。结论：**不要对整个复杂目标计算近端**——近端方法的优势仅在能干净分离出易处理的非光滑部分时才显现。

### 习题 4：投影梯度是 ISTA 特例

证明约束优化 $\min_{x \in C} g(x)$（$g$ 光滑，$C$ 闭凸）等价于复合问题 $\min_x g(x) + \iota_C(x)$，并写出 ISTA 迭代。

**解**：令 $h = \iota_C$，则 $\mathrm{prox}_{\eta h}(v) = P_C(v)$。代入 ISTA：
$$
x_{k+1} = P_C\!\bigl(x_k - \eta
\nabla g(x_k)\bigr).
$$
这正是**投影梯度法**——即 $h = \iota_C$ 时的 ISTA。加入动量即得加速投影梯度法。

---

## 总结

近端算子的核心价值在于一个具体目标：**将“非光滑”或“带约束”部分从主问题中隔离，转化为小型子问题**。具体而言：

- $\ell_1$ 的不可微性 → 软阈值；
- 凸约束 → 投影；
- 整体非光滑函数 → 可微的 Moreau 包络。

最小操作清单：

1. **ISTA**：$x_{k+1} = \mathrm{prox}_{\eta h}(x_k - \eta\nabla g(x_k))$，$\eta \le 1/L$，$O(1/k)$；
2. **FISTA**：在外推点执行 ISTA 步，更新 $t_k$；$O(1/k^2)$，实践中常用函数值重启；
3. **LASSO**：FISTA + 软阈值 + $\mu$ 路径热启动——工业界 $\ell_1$ 求解标准方案；
4. **ADMM**：当存在两个非光滑项或等式约束时，按变量拆分并交替更新。

掌握这些工具后，下次再遇到目标函数中的 $\|\cdot\|_1$、$\iota_C$、全变分或核范数，便无需畏惧——它们都只是“一次近端之遥”。

## 延伸阅读

- N. Parikh, S. Boyd. *Proximal Algorithms*. Foundations and Trends in Optimization, 2014.（权威综述）
- A. Beck, M. Teboulle. *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*. SIAM J. Imaging Sciences, 2009.（FISTA 原文）
- S. Boyd 等. *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. FnTML, 2011.（ADMM 综述）
- L. Condat 等. *Proximal Splitting Algorithms for Convex Optimization: A Tour of Recent Advances*. SIAM Review, 2023.（最新综述，含 PDHG / Condat-Vu）
