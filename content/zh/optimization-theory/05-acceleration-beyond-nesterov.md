---
title: '优化理论（五）：Nesterov 之外的加速'
date: 2022-09-20 09:00:00
tags:
  - ML
  - Optimization
  - Acceleration
categories: Algorithm
series: optimization-theory
series_order: 5
series_total: 12
lang: zh
mathjax: true
description: "一阶优化方法“最优”究竟意味着什么？我们证明了与 Nesterov 速率相匹配的紧致下界，将 Polyak 重球法导出为其连续时间极限，构建了统一的 Lyapunov 分析框架以同时涵盖二者，并揭示 Catalyst 元算法如何将任意求解器提升至加速收敛速率。"
disableNunjucks: true
translationKey: "optim-05"
---
[文章 02](../02-smoothness-strong-convexity-nesterov/) 介绍了 Nesterov 加速，并展示了它将每次迭代的复杂度从 $\kappa$ 改进到 $\sqrt{\kappa}$。本文探讨更深层次的问题：

- **为什么是 $\sqrt{\kappa}$ 而不是更快？** 我们证明了一个匹配的下界——没有任何一阶方法能做得更好。
- **Nesterov 是唯一的方式吗？** Polyak 的 Heavy-Ball 方法通过完全不同的更新规则达到了相同的速率。
- **我们能加速任意求解器吗？** Catalyst 框架通过包装一个黑盒优化器来获得加速速率，代价是求解一个正则化的子问题。

统一的工具是一个 **Lyapunov 势函数（Lyapunov potential）** —— 一种非负量，算法在每一步都会使其减小。Nesterov 和 Heavy-Ball 都有 Lyapunov 证明，而下界本质上说明了 Lyapunov 减小的速度不可能更快。

---

上一篇我们见过 Nesterov 加速：把梯度下降里每条件数的代价从 $\kappa$ 压到 $\sqrt{\kappa}$。但读完之后我自己留了几个问号：

- **为什么是 $\sqrt{\kappa}$，而不是更快？** 这个数字是哪里来的？
- **Nesterov 是唯一的方式吗？** Polyak 的 Heavy-Ball 法长得完全不一样，速率却也是 $\sqrt{\kappa}$。
- **能不能把任意一个慢求解器“包装”一下，让它也加速？** Catalyst 框架告诉我们可以。

本文围绕这三个问题展开。统一工具是 **Lyapunov 势函数**——一个非负的“监控量”，每一步算法都让它变小。Nesterov 和 Heavy-Ball 都能写成这种监控的形式，连下界证明本质上都是“你不可能让这个监控减得更快”。

如果你之前只是把 Nesterov 当成一个公式记住，这篇文章会帮你看到它背后那一类“能量在变小”的几何图像。

## 你将学到什么

1. Nemirovski--Yudin 下界：在最坏情况的光滑强凸问题上，任何一阶方法至少需要 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 次迭代。
2. Polyak 的 Heavy-Ball 方法、其连续时间极限（一个阻尼二阶 ODE）及其 Lyapunov 分析。
3. estimate-sequence 框架 —— Nesterov 最初的记账工具 —— 以及更简洁的 Lyapunov 风格推导。
4. 自适应重启：为什么固定动量加速会过冲，以及重启如何修复它。
5. Catalyst 元算法：通过内部正则化求解实现黑盒加速。
6. 一个具体示例，比较 GD、Heavy-Ball、Nesterov 和 Catalyst 在病态条件二次函数上的表现。

## 前置知识
[文章 01](../01-convex-analysis-foundations/)（凸分析基础），[文章 02](../02-smoothness-strong-convexity-nesterov/)（Lipschitz 光滑性、强凸性以及基本的 Nesterov 更新）。熟悉二次型的操作，并能理解“对于任意 $L, \mu$，存在一个函数使得……”这类证明风格。

---

## 下界：为什么 $\sqrt{\kappa}$ 是速度极限

“一阶方法”是指任何其迭代点 $x_k$ 满足以下条件的算法：
$$
x_0 + \mathrm{span}\{\nabla f(x_0), \nabla f(x_1), \ldots, \nabla f(x_{k-1})\}.
$$
这涵盖了 GD、Heavy-Ball、Nesterov、共轭梯度，以及基本上所有仅在访问过的点上查询 $\nabla f$ 的方法。

> **定理（Nesterov, 1983）**。对于任意 $L \geq \mu > 0$ 以及任意 $k \leq (n-1)/2$（其中 $n$ 是维度），存在一个 $L$-光滑 $\mu$-强凸函数 $f$，使得对任意一阶方法，
>
> 
$$
f(x_k) - f^\star \geq \frac{\mu (\sqrt{\kappa} - 1)^{2k}}{2 (\sqrt{\kappa} + 1)^{2k}} \|x_0 - x^\star\|_2^2 \cdot \text{（常数因子）}.
$$
> 特别地，达到 $\epsilon$-精度至少需要 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 次迭代。

### 最坏情况函数

该构造使用一个带状二次函数。令 $A \in \mathbb{R}^{n \times n}$ 为三对角矩阵
$$
A = \begin{pmatrix} 2 & -1 & & & \\ -1 & 2 & -1 & & \\ & -1 & 2 & -1 & \\ & & \ddots & \ddots & \ddots \\ & & & -1 & 2 \end{pmatrix},
$$
并令 $f(x) = \frac{L - \mu}{8} (x^\top A x - 2 e_1^\top x) + \frac{\mu}{2} \|x\|_2^2$，其中 $e_1$ 是第一个标准基向量。

Hessian 矩阵为 $\nabla^2 f = \frac{L-\mu}{4} A + \mu I$。计算 $A$ 的特征值——它们是 $4 \sin^2 \frac{j \pi}{2(n+1)}$（$j = 1, \ldots, n$）——表明 $\nabla^2 f$ 的特征值位于 $[\mu, L]$ 区间内，因此 $f$ 是 $L$-光滑且 $\mu$-强凸的。

### 为什么这个函数很难

从 $x_0 = 0$ 开始，梯度 $\nabla f(x_0) = -\frac{L - \mu}{4} e_1 + \mu \cdot 0 = -\frac{L-\mu}{4} e_1$ 仅在第一个坐标上非零。经过任意一阶方法的 $k$ 次迭代后，$x_k$ 位于 $\mathrm{span}\{e_1, e_2, \ldots, e_k\}$ 中——即前 $k$ 个坐标。为什么？因为 $A$ 是三对角矩阵，将一个仅在前 $j$ 个坐标上有支撑的向量乘以 $A$，结果向量的支撑最多扩展到前 $j+1$ 个坐标。

因此，在 $k$ 次迭代后，$x_k$ 的最后 $n - k$ 个坐标仍为零，残差 $f(x_k) - f^\star$ 由一个 $(n-k)$ 维子问题决定，该子问题具有相同的条件数。仔细追踪这一过程可得到 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 的速率；我们省略代数细节（Nesterov 2004 §2.1.2 提供了完整推导）。

关键结论：**任何仅在访问点上看到 $\nabla f$ 的一阶方法，在信息论意义上被限制在 $\sqrt{\kappa}$**。要更快，要么 (a) 需要关于 $f$ 的更多信息（二阶方法，见[文章 07](../07-second-order-methods/)），要么 (b) 利用超出光滑性 + 强凸性的结构。

![Lower bound vs upper bounds](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig3_lowerbound.png "Nemirovski-Yudin lower bound matches Nesterov's upper bound")

上图中的阴影带表示最优速率区域：任何一阶方法的最坏情况误差必须位于或高于 Nemirovski-Yudin 曲线，而 Nesterov 方法实际上达到了该速率（相差常数因子）。普通 GD 位于高得多的区域——其 $(1 - 1/\kappa)^{2k}$ 包络的衰减速率大约比加速界慢 $\sqrt{\kappa}$ 倍。

---

## Polyak 的 Heavy-Ball 方法

### 物理类比

想象一个质量为 $m$ 的球在粘性介质中沿曲面 $f$ 滚动，摩擦系数为 $\gamma$。牛顿定律给出
$$
m \ddot x(t) + \gamma \dot x(t) + \nabla f(x(t)) = 0.
$$
重球会积累动量：它不会在每一时刻都简单地沿着 $-\nabla f$ 移动，而是从前几步继承速度。这种惯性带来了加速效果。

使用 leapfrog 格式以步长 $h$ 离散化：
$$
x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1}),
$$
其中 $\alpha = h^2 / m$ 且 $\beta = 1 - \gamma h / m$。这就是 **Polyak 的 Heavy-Ball** 更新：一个梯度步加上一个与前一步成比例的动量项。

### 最优参数

对于二次函数 $f(x) = \frac{1}{2} x^\top Q x - b^\top x$，其中 $\mu I \preceq Q \preceq L I$，迭代线性化为
$$
\begin{pmatrix} x_{k+1} - x^\star \\ x_k - x^\star \end{pmatrix} = M \begin{pmatrix} x_k - x^\star \\ x_{k-1} - x^\star \end{pmatrix}, \quad M = \begin{pmatrix} (1 + \beta) I - \alpha Q & -\beta I \\ I & 0 \end{pmatrix}.
$$
收敛速率是 $\rho(M)$（谱半径）。对 $Q$ 对角化后，问题简化为标量情形；在最坏特征值上对 $(\alpha, \beta)$ 最小化 $\rho$ 得到：
$$
\alpha^\star = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \quad \beta^\star = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2,
$$
对应的速率为 $\rho(M) = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$。这与 **Nesterov 的加速速率相同**，但使用了看起来简单得多的更新规则。

### 缺陷：Heavy-Ball 不是全局收敛的

Polyak 证明了其在二次函数上的速率。但对于一般的光滑强凸函数，相同的参数可能导致不收敛——Lessard、Recht 和 Packard（2016）给出了一个明确的光滑强凸反例，其中使用最优二次参数的 Heavy-Ball 会永远循环。这与 Nesterov 方法形成鲜明对比，后者在每个 $L$-光滑 $\mu$-强凸函数上都收敛。

解决方法是使用稍保守的参数，或通过实验验证收敛性；实践中，Heavy-Ball 是 PyTorch 中 `momentum=0.9` 的核心机制，即使理论保证仅适用于二次函数，它在神经网络中也表现良好。

![2D trajectories on ellipse contours](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig2_trajectory.png "GD zig-zags along the steep direction; momentum methods overshoot but reach $x^\star$ much faster")

GD 沿陡峭的 $x_1$ 方向来回震荡，并在平坦的 $x_2$ 方向上缓慢爬行。Heavy-Ball 和 Nesterov 都因惯性而 *过冲* 最优点，然后回摆——但每次振荡都是朝正确方向的粗略修正，因此它们能在少得多的迭代次数内达到最优点。

---

## 统一的 Lyapunov 框架

### estimate-sequence（Nesterov 的原始工具）

Nesterov 1983 年的论文使用了一组“模型函数” $\phi_k$ 来下界 $f$。定义
$$
\phi_{k+1}(x) = (1 - \alpha_k) \phi_k(x) + \alpha_k \big[ f(y_k) + \langle \nabla f(y_k), x - y_k \rangle + \tfrac{\mu}{2} \|x - y_k\|_2^2 \big],
$$
其中 $\phi_0(x) = f(x_0) + \frac{\mu}{2} \|x - x_0\|_2^2$，$y_k$ 是前瞻点。序列 $\phi_k$ 是凸二次函数；追踪其最小值点 $v_k$ 和最小值 $\phi_k^\star$。归纳法
$$
f(x_k) \leq \phi_k^\star
$$
结合 $\phi_k(x^\star) \to f(x^\star)$ 的速率 $(1 - \sqrt{\mu/L})^k$，即可得到加速收敛。

estimate-sequence 很强大但设置繁琐。现代表述采用 Lyapunov 函数，我们接下来展开。

### Lyapunov 方法

**Lyapunov 函数** 是一个非负量 $V_k$，算法在每一步都使其减小：
$$
V_{k+1} \leq (1 - \sqrt{\mu/L}) V_k.
$$
关键是找到合适的 $V_k$。对于 $L$-光滑 $\mu$-强凸函数 $f$ 上的 Nesterov 方法，取
$$
V_k = (f(x_k) - f^\star) + \frac{\mu}{2} \|z_k - x^\star\|_2^2,
$$
其中 $z_k$ 是算法维护的辅助“外推点”（即上文提到的 $v_k$）。使用标准 Nesterov 参数可以证明——参见 Bansal & Gupta (2019)，Wilson、Recht、Jordan (2021)——
$$
V_{k+1} \leq \big( 1 - \sqrt{\mu/L} \big) V_k.
$$
迭代得到 $V_k \leq (1 - \sqrt{\mu/L})^k V_0$，因此 $f(x_k) - f^\star \leq V_k = O((1 - 1/\sqrt{\kappa})^k)$，即加速速率。

Lyapunov 论证具有普适性：对于 **任意** 可写为阻尼二阶 ODE $\ddot x + \gamma(t) \dot x + \nabla f(x) = 0$ 离散化的算法（只要 $\gamma(t)$ 合适），都存在 Lyapunov 函数并导出加速速率。

### Mirror descent 与加速的差距

对于凸但 **非强凸** 问题，下界变为 $\Omega(\sqrt{L/\epsilon})$——即 $\Omega(1/k^2)$ 速率。Nesterov 方法能达到该速率；FISTA（[文章 06](../06-composite-proximal-methods/)）也能。仅凸情形下的 Lyapunov 函数为：
$$
V_k = \tfrac{k(k+1)}{4 L} (f(x_k) - f^\star) + \tfrac{1}{2} \|z_k - x^\star\|_2^2.
$$
证明 $V_{k+1} \leq V_k$ 即可得到 $f(x_k) - f^\star \leq O(1/k^2)$。同一 Lyapunov 模板处理两种情形——只需使用不同的具体函数形式。

---

## 自适应重启：何时中断动量

加速有一个讨厌的特性：凸情形下的动量系数 $\beta_k = (k-1)/(k+2)$ 会趋近于 1。如果你从远离 $x^\star$ 的地方开始并积累了过多动量，迭代点会过冲，导致函数值振荡而非单调下降。

**重启启发式（O'Donoghue & Candès, 2015）**：每次迭代检查是否满足以下任一条件：
- **函数值重启**：$f(x_{k+1}) > f(x_k)$，或
- **梯度重启**：$\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$（该步在上坡），

若满足，则重置动量：设 $z_k \leftarrow x_k$ 并将内部计数器归零。

梯度重启准则通常更受青睐——它不需要计算 $f$，在某些场景下计算 $f$ 可能很昂贵。

**为什么重启有效**。重启将单个 $O(1/k^2)$ 阶段转换为一系列阶段。在每个阶段内，迭代点的行为就像从一个新的、更接近 $x^\star$ 的 $x_0$ 开始。总迭代次数仍为 $O(\sqrt{\kappa} \log(1/\epsilon))$，但常数更好，且实践中无振荡。在强凸问题上，重启能自动适应未知的 $\mu$——你可以运行仅凸加速方案，无需提前知道 $\mu$ 即可达到强凸速率。

![Adaptive restart vs no restart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig4_restart.png "Without restart, the convex Nesterov scheme oscillates; gradient restart yields clean monotone descent")

无重启时，函数值会反弹（橙色曲线）——动量系数 $\beta_k \to 1$ 持续推动迭代点沿平坦方向越过 $x^\star$。自适应梯度重启（蓝色）通过廉价测试 $\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$ 检测到这一点并重启动量，产生一系列干净的加速阶段（竖线标记重启事件）。

---

## Catalyst：黑盒加速

如果问题过于复杂而无法直接应用 Nesterov——例如梯度难以精确计算，或你想使用任意内部求解器？**Catalyst** 框架（Lin、Mairal、Harchaoui, 2015）通过正则化内部子问题加速任何线性收敛的内部求解器。

### 元算法

对于最小化 $L$-光滑凸函数 $f$，选择 $\kappa > 0$ 并定义正则化目标
$$
g_y(x) := f(x) + \frac{\kappa}{2} \|x - y\|_2^2.
$$
注意即使 $f$ 非强凸，$g_y$ 也是 $\kappa$-强凸的。Catalyst 迭代如下：

1. 设 $y_0 = x_0$。
2. 对 $k = 0, 1, \ldots$：使用任意内部求解器近似最小化 $g_{y_k}(x)$，得到 $x_{k+1}$，满足 $g_{y_k}(x_{k+1}) - g_{y_k}^\star \leq \epsilon_k$。
3. 更新 $y_{k+1} = x_{k+1} + \beta_k (x_{k+1} - x_k)$，其中动量 $\beta_k$ 按 Nesterov 方式选择。

如果内部求解器在 $\kappa$-强凸问题上线性收敛速率为 $\rho$，则 Catalyst 以 **加速** 速率 $O(\sqrt{L/\kappa} \cdot \rho^{-1} \log(1/\epsilon))$ 收敛。

![Catalyst meta-algorithm flowchart](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig5_catalyst.png "Outer Nesterov loop drives an inner regularized solver")

外层循环在 *锚点* $y_k$ 上运行 Nesterov 风格的动量；内层循环对正则化子问题 $g_{y_k}(x) = f(x) + \frac{\kappa}{2}\|x - y_k\|^2$ 调用任意线性收敛求解器。由于 $g_{y_k}$ 通过构造是 $\kappa$-强凸的，即使原问题非强凸，也能使用线性收敛的内部求解器。

### 应用示例

假设你的内部问题是有限和 $f(x) = \frac{1}{n} \sum_i f_i(x)$，并使用 SVRG（[文章 10](../10-stochastic-variance-reduction/)）作为内部求解器。SVRG 在 $\kappa$-强凸问题上的速率为 $O((n + L/\kappa) \log(1/\epsilon))$。Catalyst-SVRG 在原始 $\mu$-强凸问题上的总复杂度为 $O((n + \sqrt{n L / \mu}) \log(1/\epsilon))$——当 $L \gg \mu$ 时严格优于普通 SVRG。

这就是加速那些无法干净纳入 Nesterov 框架的算法的方法。

---

## 具体比较：一个病态条件二次函数

考虑 $f(x) = \frac{1}{2} x^\top Q x$，其中 $Q = \mathrm{diag}(1, 1/\kappa)$，$\kappa = 10^4$。最优点：$x^\star = 0$。初始点：$x_0 = (1, 1)$。

```python
import numpy as np

L, mu = 1.0, 1e-4
kappa = L / mu
Q = np.diag([L, mu])
x0 = np.ones(2)

def gd(steps):
    eta = 1 / L
    x = x0.copy(); hist = []
    for _ in range(steps):
        x = x - eta * Q @ x
        hist.append(0.5 * x @ Q @ x)
    return hist

def heavy_ball(steps):
    alpha = 4 / (np.sqrt(L) + np.sqrt(mu))**2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)))**2
    x_prev = x0.copy(); x = x0.copy(); hist = []
    for _ in range(steps):
        x_new = x - alpha * Q @ x + beta * (x - x_prev)
        x_prev = x; x = x_new
        hist.append(0.5 * x @ Q @ x)
    return hist

def nesterov(steps):
    eta = 1 / L
    q = mu / L
    a = (1 - np.sqrt(q)) / (1 + np.sqrt(q))
    x = x0.copy(); y = x0.copy(); hist = []
    for _ in range(steps):
        x_new = y - eta * Q @ y
        y = x_new + a * (x_new - x)
        x = x_new
        hist.append(0.5 * x @ Q @ x)
    return hist

steps = 500
gd_hist = gd(steps)
hb_hist = heavy_ball(steps)
nag_hist = nesterov(steps)
```text

![GD vs Heavy-Ball vs Nesterov on a kappa=100 quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig1_convergence.png "Convergence curves: sqrt(kappa) speedup of momentum")

两种加速方法（Heavy-Ball、Nesterov）基本位于同一条线上，都紧贴 $(1 - 1/\sqrt{\kappa})^{2k}$ 包络；GD 则位于更平缓的 $(1 - 1/\kappa)^{2k}$ 包络上。读取达到 $10^{-6}$ 所需的迭代次数：GD 大约需要多 $\sqrt{\kappa} = 10$ 倍的迭代。

典型输出（第 500 次迭代的函数值）：
- 普通 GD：$\sim 6 \times 10^{-3}$ —— $\kappa = 10^4$ 非常严酷。
- Heavy-Ball：$\sim 4 \times 10^{-19}$ —— 实质上为零。
- Nesterov：$\sim 4 \times 10^{-19}$ —— 相同的加速速率。

GD 与加速方法在达到相同精度所需的迭代次数上的差距约为 $\sqrt{\kappa} = 100$。这就是加速的实际价值。

---

## 故事的延续

- [文章 06](../06-composite-proximal-methods/) 使用上述 Lyapunov 模板推导 FISTA（加速近端梯度方法）。
- [文章 10](../10-stochastic-variance-reduction/) 将 Catalyst 与随机内部求解器（SVRG、SAGA）结合用于有限和问题。
- [文章 07](../07-second-order-methods/) 探索二阶方法，它们通过使用更多信息突破了 $\sqrt{\kappa}$ 壁垒。

## 总结

| 问题                                  | 答案                                                                  |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| 能否比 $\sqrt{\kappa}$ 更好？    | 不能 —— Nemirovski--Yudin 下界禁止了这一点。                         |
| Heavy-Ball vs Nesterov？                   | 二次函数上速率相同；Heavy-Ball 在一般 SC 问题上可能失败。     |
| 何时重启？                          | 当动量过冲时（梯度或函数值准则）。         |
| 如何加速黑盒求解器？     | 使用正则化内部子问题的 Catalyst 元算法。              |
| 统一理论是什么？                | 阻尼二阶 ODE 及其离散化的 Lyapunov 函数。 |

## 参考文献

- Nesterov, *Lectures on Convex Optimization* (2nd ed.), §2 —— 权威论述。
- d'Aspremont, Scieur & Taylor, *Acceleration Methods*, FnT-OPT 5(1-2), 2021 —— 现代综述，包含 Lyapunov 框架。
- O'Donoghue & Candès, *Adaptive Restart for Accelerated Gradient Schemes*, FoCM 15, 2015 —— 重启论文。
- Lin, Mairal & Harchaoui, *Catalyst Acceleration for First-Order Convex Optimization*, JMLR 18, 2018 —— 元算法。
- Wilson, Recht & Jordan, *A Lyapunov Analysis of Accelerated Methods in Optimization*, JMLR 22, 2021 —— 统一框架。