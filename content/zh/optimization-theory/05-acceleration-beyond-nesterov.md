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
lang: zh
mathjax: true
description: "一阶优化方法“最优”究竟意味着什么？我们证明了与Nesterov速率相匹配的紧致下界，将Polyak重球法导出为其连续时间极限，构建了统一的Lyapunov分析框架以同时涵盖二者，并揭示Catalyst元算法如何将任意求解器提升至加速收敛速率。"
disableNunjucks: true
translationKey: "optim-05"
---
第二篇文章介绍了 Nesterov 加速法，并指出它将每步迭代的收敛代价从 $\kappa$ 改善至 $\sqrt{\kappa}$。本文则深入探讨以下根本性问题：

- **为何是 $\sqrt{\kappa}$，而不能更快？** 我们将证明一个匹配的下界——任何一阶方法均无法超越该速率。
- **Nesterov 是唯一途径吗？** Polyak 的重球法（Heavy-Ball method）采用完全不同的更新规则，却达到了相同的收敛速率。
- **能否对任意求解器进行加速？** Catalyst 框架通过封装一个黑箱优化器，在每次迭代中求解一个正则化子问题，即可获得加速速率。

贯穿这一切的统一工具是 **Lyapunov 势函数（Lyapunov potential）**——一种非负量，其值在算法每一步严格递减。 Nesterov 方法与重球法均可借助 Lyapunov 函数完成分析；而上述下界本质上表明：不存在任何 Lyapunov 势函数能实现比 $\sqrt{\kappa}$ 更快的衰减速率。

## 你将学到的内容

1. Nemirovski–Yudin 下界：对于最坏情形下的光滑强凸问题，任意一阶方法至少需要 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 次迭代才能达到 $\epsilon$ 精度。
2. Polyak 重球法、其连续时间极限（一个阻尼二阶常微分方程），及其 Lyapunov 分析。
3. 估计序列（estimate-sequence）框架——Nesterov 原始的“簿记技巧”，以及一种更简洁、基于 Lyapunov 风格的推导方式。
4. 自适应重启（adaptive restart）：为何固定动量的加速策略容易过冲，以及重启机制如何修正这一缺陷。
5. Catalyst 元算法（meta-algorithm）：通过内层正则化子问题求解，实现对黑箱优化器的通用加速。
6. 一个完整算例：在病态二次函数上对比梯度下降（GD）、重球法、 Nesterov 加速法与 Catalyst 的实际表现。

## 前置知识

第一篇文章（凸分析基础）、第二篇文章（Lipschitz 光滑性、强凸性及基本 Nesterov 更新公式）。需熟悉二次型运算，并能理解形如“对任意 $L, \mu$，存在某个函数使得……”的构造性证明。

---

## 1. 下界：为何 $\sqrt{\kappa}$ 是速度极限？

所谓“一阶方法”，是指其第 $k$ 步迭代点 $x_k$ 必然属于如下子空间：
$$
x_0 + \mathrm{span}\{\nabla f(x_0), \nabla f(x_1), \ldots, \nabla f(x_{k-1})\}.
$$
该定义涵盖了梯度下降（GD）、重球法、 Nesterov 加速法、共轭梯度法等几乎所有仅依赖于已访问点处梯度 $\nabla f$ 的算法。

> **定理（Nesterov, 1983）**：对任意 $L \geq \mu > 0$ 及任意 $k \leq (n-1)/2$（其中 $n$ 为变量维数），存在一个 $L$-光滑且 $\mu$-强凸的函数 $f$，使得对任意一阶方法均有  
> $$
> f(x_k) - f^\star \geq \frac{\mu (\sqrt{\kappa} - 1)^{2k}}{2 (\sqrt{\kappa} + 1)^{2k}} \|x_0 - x^\star\|_2^2 \cdot \text{（常数因子）}.
> $$
> 特别地，要达到 $\epsilon$ 精度，至少需要 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 次迭代。

### 1.1 最坏情形函数

该构造使用一个带状二次函数。令 $A \in \mathbb{R}^{n \times n}$ 为三对角矩阵：
$$
A = \begin{pmatrix} 2 & -1 & & & \\ -1 & 2 & -1 & & \\ & -1 & 2 & -1 & \\ & & \ddots & \ddots & \ddots \\ & & & -1 & 2 \end{pmatrix},
$$
并定义  
$$
f(x) = \frac{L - \mu}{8} (x^\top A x - 2 e_1^\top x) + \frac{\mu}{2} \|x\|_2^2,
$$
其中 $e_1$ 是第一个标准基向量。

其 Hessian 矩阵为 $\nabla^2 f = \frac{L-\mu}{4} A + \mu I$。由于 $A$ 的特征值为 $4 \sin^2 \frac{j \pi}{2(n+1)}$（$j = 1,\dots,n$），可得 $\nabla^2 f$ 的特征值落在区间 $[\mu, L]$ 内，因此 $f$ 确实是 $L$-光滑且 $\mu$-强凸的。

### 1.2 为何该函数难以优化

取初始点 $x_0 = 0$，则梯度  
$$
\nabla f(x_0) = -\frac{L - \mu}{4} e_1 + \mu \cdot 0 = -\frac{L-\mu}{4} e_1
$$  
仅在第一个坐标方向非零。又因 $A$ 是三对角矩阵，对任一支撑在前 $j$ 个坐标的向量 $v$，乘积 $Av$ 的支撑集至多扩展至前 $j+1$ 个坐标。因此，对任意一阶方法，经 $k$ 步迭代后，$x_k$ 必然位于 $\mathrm{span}\{e_1, e_2, \ldots, e_k\}$ 中——即仅前 $k$ 个坐标可能非零。

换言之，$x_k$ 的后 $n-k$ 个坐标恒为零；此时剩余误差 $f(x_k) - f^\star$ 实际由一个 $(n-k)$ 维子问题决定，且该子问题具有相同的条件数 $\kappa$。细致追踪这一维度衰减过程，即可导出 $\Omega(\sqrt{\kappa} \log(1/\epsilon))$ 的迭代复杂度下界；此处略去代数细节（完整推导见 Nesterov 2004 年著作 §2.1.2）。

核心结论：**任何仅通过已访问点处梯度 $\nabla f$ 获取信息的一阶方法，在信息论意义上注定受限于 $\sqrt{\kappa}$ 的收敛速率。** 若想突破此极限，则必须引入额外信息：（a）关于 $f$ 的二阶信息（参见第七篇文章），或（b）超出光滑性与强凸性之外的问题结构。

![下界与上界对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig3_lowerbound.png "Nemirovski-Yudin 下界与 Nesterov 上界完美匹配")

上图阴影区域即为最优收敛速率带：任何一阶方法在最坏情形下的误差都不可能低于 Nemirovski-Yudin 曲线，而 Nesterov 加速法恰好能取到这一速率（仅相差常数因子）。普通梯度下降则被压制在远高于此带的区域——其 $(1 - 1/\kappa)^{2k}$ 包络的衰减速率比加速上界慢了大约 $\sqrt{\kappa}$ 倍。

## 2. Polyak 重球法（Heavy-Ball Method）

### 2.1 物理类比

设想一个质量为 $m$ 的小球，在粘性介质中沿曲面 $f$ 向下滚动，介质的摩擦系数为 $\gamma$。根据牛顿第二定律，其运动方程为：
$$
m \ddot x(t) + \gamma \dot x(t) + \nabla f(x(t)) = 0.
$$
“重球”具有惯性：它并非在每一时刻都简单地沿负梯度方向 $-\nabla f$ 移动，而是继承了前几步的运动速度；加速度正源于这种惯性效应。

采用步长 $h$，并以蛙跳（leapfrog）格式进行离散化，可得：
$$
x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta (x_k - x_{k-1}),
$$
其中 $\alpha = h^2 / m$，$\beta = 1 - \gamma h / m$。此即 **Polyak 重球法** 的迭代更新式：它由一个标准梯度步与一个正比于前一步位移的动量项组成。

### 2.2 最优参数选择

考虑二次函数 $f(x) = \frac{1}{2} x^\top Q x - b^\top x$，其中 $Q$ 满足 $\mu I \preceq Q \preceq L I$。此时迭代可线性化为：
$$
\begin{pmatrix} x_{k+1} - x^\star \\ x_k - x^\star \end{pmatrix} = M \begin{pmatrix} x_k - x^\star \\ x_{k-1} - x^\star \end{pmatrix}, \quad M = \begin{pmatrix} (1 + \beta) I - \alpha Q & -\beta I \\ I & 0 \end{pmatrix}.
$$
收敛速率由矩阵 $M$ 的谱半径 $\rho(M)$ 决定。由于 $Q$ 可对角化，该问题可约化为一系列标量子问题；在最坏特征值上对 $(\alpha, \beta)$ 最小化 $\rho$，可得最优参数：
$$
\alpha^\star = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \quad \beta^\star = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2,
$$
对应收敛速率为 $\rho(M) = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$，其中 $\kappa = L/\mu$。这一加速速率 **与 Nesterov 加速法完全相同**，却仅需一个形式上更简洁的更新公式。

### 2.3 关键限制：重球法不具备全局收敛性

Polyak 所证明的收敛速率仅适用于二次函数。对于一般的光滑强凸函数，若直接套用上述针对二次情形优化所得的参数，算法可能根本无法收敛——Lessard、 Recht 与 Packard （2016）构造了一个显式的光滑强凸反例，表明在此类函数上，采用最优二次参数的重球法将永远循环振荡，永不收敛。这与 Nesterov 方法形成鲜明对比：后者对任意 $L$-光滑且 $\mu$-强凸的函数均能保证收敛。

实际应用中的补救措施包括：采用略为保守（即更小）的参数，或通过实验验证收敛性。事实上，重球法正是 PyTorch 中 `SGD` 优化器默认 `momentum=0.9` 的理论原型，在训练神经网络时表现稳健；尽管其严格理论保障目前仅限于二次情形，工程实践中仍被广泛视为可靠高效的动量方案。

![椭圆等高线上的二维轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig2_trajectory.png "GD 在陡方向上反复折返；动量法虽过冲但更快到达  0 ")

GD 沿陡峭的 $x_1$ 方向反复"之"字形折返，沿平缓的 $x_2$ 方向爬行极慢。重球法与 Nesterov 法因惯性会**过冲**最优点，再回摆——但每次震荡都是朝正确方向的粗略修正，因此能以远少的迭代次数抵达最优解。

## 3. 统一的李雅普诺夫（Lyapunov）分析框架

### 3.1 估计序列（Nesterov 的原始技巧）

Nesterov 在 1983 年的论文中引入了一组“模型函数”$\phi_k$，它们对目标函数 $f$ 构成下界。定义如下递推关系：
$$
\phi_{k+1}(x) = (1 - \alpha_k) \phi_k(x) + \alpha_k \big[ f(y_k) + \langle \nabla f(y_k), x - y_k \rangle + \tfrac{\mu}{2} \|x - y_k\|_2^2 \big],
$$
其中初始函数为 $\phi_0(x) = f(x_0) + \frac{\mu}{2} \|x - x_0\|_2^2$，而 $y_k$ 是算法中使用的前视点（lookahead point）。该序列 $\phi_k$ 恒为凸二次函数；记其最小值点为 $v_k$，最小值为 $\phi_k^\star$。通过归纳可得关键不等式：
$$
f(x_k) \leq \phi_k^\star,
$$
再结合 $\phi_k(x^\star) \to f(x^\star)$ 的收敛速率 $(1 - \sqrt{\mu/L})^k$，即可导出加速收敛性。

估计序列虽功能强大，但构造繁琐。现代分析更倾向于采用李雅普诺夫函数方法，我们将在下文展开。

### 3.2 李雅普诺夫方法

一个**李雅普诺夫函数**是一个非负量 $V_k$，满足算法每步均使其衰减：
$$
V_{k+1} \leq (1 - \sqrt{\mu/L}) V_k.
$$
难点在于构造恰当的 $V_k$。对于 $L$-光滑且 $\mu$-强凸的目标函数 $f$， Nesterov 方法对应的李雅普诺夫函数取为：
$$
V_k = (f(x_k) - f^\star) + \frac{\mu}{2} \|z_k - x^\star\|_2^2,
$$
其中 $z_k$ 是算法所维护的辅助“外推点”（即前述的 $v_k$）。在标准 Nesterov 参数设置下，可以证明——参见 Bansal & Gupta (2019)， Wilson, Recht, Jordan (2021)——有：
$$
V_{k+1} \leq \big( 1 - \sqrt{\mu/L} \big) V_k.
$$
迭代后得 $V_k \leq (1 - \sqrt{\mu/L})^k V_0$，从而 $f(x_k) - f^\star \leq V_k = O((1 - 1/\sqrt{\kappa})^k)$，即达到加速收敛速率。

该李雅普诺夫分析具有普适性：对**任意**可表示为阻尼二阶常微分方程  
$$
\ddot x + \gamma(t) \dot x + \nabla f(x) = 0
$$  
之适当离散化的算法（其中 $\gamma(t)$ 选取合适），均存在对应的李雅普诺夫函数，并能导出加速收敛速率。

### 3.3 镜像下降与加速的间隙

对于仅凸（convex）、但**非强凸**的问题，最优下界退化为 $\Omega(\sqrt{L/\epsilon})$ —— 即对应于 $O(1/k^2)$ 的收敛速率。 Nesterov 方法可达此界； FISTA （见第 06 篇文章）亦然。此时对应的李雅普诺夫函数为：
$$
V_k = \tfrac{k(k+1)}{4 L} (f(x_k) - f^\star) + \tfrac{1}{2} \|z_k - x^\star\|_2^2.
$$
验证 $V_{k+1} \leq V_k$ 即可推出 $f(x_k) - f^\star \leq O(1/k^2)$。同一李雅普诺夫模板可统一处理强凸与仅凸两种情形——区别仅在于所采用的具体函数形式。

---

## 4. 自适应重启：何时中断动量

加速方法存在一个棘手问题：在仅凸情形下，动量系数 $\beta_k = (k-1)/(k+2)$ 随 $k$ 增大而趋近于 1。若初始点 $x_0$ 远离最优解 $x^\star$，持续累积动量将导致迭代点严重过冲，函数值反而震荡而非单调下降。

**重启启发式规则**（O'Donoghue & Candès, 2015）：在每一步迭代中，检查以下任一条件是否成立：
- **函数值重启**：$f(x_{k+1}) > f(x_k)$，或  
- **梯度重启**：$\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$（即当前步沿梯度方向上行），  

若任一条件满足，则重置动量：令 $z_k \leftarrow x_k$，并清零内部计数器。

梯度重启准则通常更受青睐——它无需计算函数值 $f$，而后者在某些场景中开销高昂。

**重启为何有效？**  
重启将原本单一的 $O(1/k^2)$ 收敛阶段，分解为若干连续子阶段。在每个子阶段内，迭代行为等价于从一个更靠近 $x^\star$ 的新起点 $x_0$ 重新开始。总迭代复杂度仍为 $O(\sqrt{\kappa} \log(1/\epsilon))$，但实际常数显著改善，且实践中完全消除震荡现象。在强凸问题上，重启机制还能自动适配未知的强凸模 $\mu$：即使仅使用仅凸情形下的加速方案，也能自动实现强凸速率，而无需预先知晓 $\mu$。

![自适应重启 vs 无重启](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig4_restart.png "无重启时凸 Nesterov 方案震荡不已；梯度重启则给出干净的单调下降")

无重启时函数值剧烈震荡（橙色曲线）——动量系数 $\beta_k \to 1$ 不断把迭代点沿平缓方向推过 $x^\star$。自适应梯度重启（蓝色）以廉价的判据 $\langle \nabla f(y_k), x_{k+1} - x_k \rangle > 0$ 检测到这种过冲，并即刻重置动量，从而产生一连串干净的加速段（竖线标示重启时刻）。

## 5. Catalyst：黑箱加速框架

倘若问题过于复杂，无法直接应用 Nesterov 加速——例如梯度难以精确计算，或你希望使用任意的内层求解器（inner solver）？此时，**Catalyst** 框架（Lin, Mairal, Harchaoui, 2015）提供了一种通用方案：它通过求解一系列带正则项的内层子问题，对**任意线性收敛的内层求解器**进行加速。

### 5.1 元算法（meta-algorithm）

设目标函数 $f$ 是 $L$-光滑凸函数。任选 $\kappa > 0$，定义关于参考点 $y$ 的正则化目标函数：
$$
g_y(x) := f(x) + \frac{\kappa}{2} \|x - y\|_2^2.
$$
注意：即使 $f$ 非强凸，$g_y$ 也必为 $\kappa$-强凸。 Catalyst 迭代如下：

1. 初始化 $y_0 = x_0$；
2. 对 $k = 0, 1, \ldots$：使用任意内层求解器，近似最小化 $g_{y_k}(x)$，得到 $x_{k+1}$，满足精度条件  
   $$
   g_{y_k}(x_{k+1}) - g_{y_k}^\star \leq \epsilon_k；
   $$
3. 以类 Nesterov 动量更新 $y$：  
   $$
   y_{k+1} = x_{k+1} + \beta_k (x_{k+1} - x_k),
   $$
   其中 $\beta_k$ 的选取方式与 Nesterov 方法一致。

若内层求解器在 $\kappa$-强凸问题上具有线性收敛率 $\rho$，则 Catalyst 整体达到**加速收敛率**：  
$$
O\!\left(\sqrt{L/\kappa} \cdot \rho^{-1} \log(1/\epsilon)\right).
$$

![Catalyst 元算法流程图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig5_catalyst.png "外层 Nesterov 循环驱动内层正则化求解器")

外层循环以 Nesterov 风格的动量更新**锚点** $y_k$；内层循环则调用任意线性收敛求解器去近似优化正则化子问题 $g_{y_k}(x) = f(x) + \frac{\kappa}{2}\|x - y_k\|^2$。由于 $g_{y_k}$ 在构造上恒为 $\kappa$-强凸，即便原问题非强凸，也能借助内层线性收敛求解器获得加速速率。


### 5.2 示例应用

假设原始问题是有限和形式 $f(x) = \frac{1}{n} \sum_i f_i(x)$，且你选用 SVRG （见第 10 篇文章）作为内层求解器。 SVRG 在 $\kappa$-强凸问题上的收敛率为 $O\!\left((n + L/\kappa) \log(1/\epsilon)\right)$。经 Catalyst 加速后， Catalyst-SVRG 在原始 $\mu$-强凸问题上的总复杂度为  
$$
O\!\left((n + \sqrt{n L / \mu}) \log(1/\epsilon)\right),
$$  
当 $L \gg \mu$ 时，该结果严格优于标准 SVRG。  

这正是对**无法自然嵌入 Nesterov 框架**的算法实施加速的系统性方法。

---

## 6. 实例对比：病态二次函数

考虑 $f(x) = \frac{1}{2} x^\top Q x$，其中 $Q = \mathrm{diag}(1, 1/\kappa)$，取 $\kappa = 10^4$。最优解为 $x^\star = 0$，初始点为 $x_0 = (1, 1)$。

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

# Run
steps = 500
gd_hist = gd(steps)
hb_hist = heavy_ball(steps)
nag_hist = nesterov(steps)
```

![梯度下降 vs 重球法 vs Nesterov 在  0  二次函数上的收敛对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/05-acceleration-beyond-nesterov/fig1_convergence.png "收敛曲线：动量带来的  1  倍加速")

两条加速方法（Heavy-Ball、 Nesterov）几乎重合，紧贴 $(1 - 1/\sqrt{\kappa})^{2k}$ 包络； GD 则贴着平缓得多的 $(1 - 1/\kappa)^{2k}$ 包络。从达到 $10^{-6}$ 精度所需的迭代次数读出： GD 需要的迭代次数大约是加速方法的 $\sqrt{\kappa} = 10$ 倍。

典型输出（第 500 次迭代处的函数值）：
- 标准梯度下降（GD）：$\sim 6 \times 10^{-3}$ —— $\kappa = 10^4$ 下表现极差；
- 重球法（Heavy-Ball）：$\sim 4 \times 10^{-19}$ —— 实质已达机器精度零点；
- Nesterov：$\sim 4 \times 10^{-19}$ —— 同样达到加速收敛速率。

为达到相同精度， GD 所需迭代次数比加速方法多出约 $\sqrt{\kappa} = 100$ 倍。这正是加速方法在实践中的核心价值。

---

## 7. 总结

| 问题                                      | 回答                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------- |
| 能否突破 $\sqrt{\kappa}$ 瓶颈？           | 否——Nemirovski–Yudin 下界已证明其不可能性。                            |
| 重球法 vs Nesterov？                      | 在二次函数上收敛率相同；但在一般强凸（SC）问题上，重球法可能发散。         |
| 何时应重启（restart）？                   | 当动量导致过冲时（依据梯度符号变化或函数值上升等判据）。                    |
| 如何加速一个黑箱求解器？                  | 使用 Catalyst 元算法，辅以带正则项的内层子问题。                          |
| 统一理论基础是什么？                      | 阻尼二阶常微分方程（ODE）及其离散化所对应的 Lyapunov 函数分析框架。         |
## 故事的延续

- 第 06 篇文章使用上述 Lyapunov 模板，严格推导出 FISTA——即加速近端梯度法。
- 第 10 篇文章将 Catalyst 框架与随机内层求解器（SVRG、 SAGA）结合，用于有限和优化问题。
- 第 07 篇文章探讨二阶方法，这类方法通过利用更多曲率信息，突破了 $\sqrt{\kappa}$ 的收敛速率瓶颈。

## 延伸阅读

- Nesterov，《凸优化讲义》（第二版），§2 —— 该领域的经典权威论述。  
- d'Aspremont、 Scieur & Taylor，《加速方法》，*Foundations and Trends® in Optimization* 5(1–2), 2021 —— 当代综述，涵盖 Lyapunov 分析框架。  
- O'Donoghue & Candès，《加速梯度算法的自适应重启策略》，*Foundations of Computational Mathematics* 15, 2015 —— 提出重启机制的奠基性论文。  
- Lin、 Mairal & Harchaoui，《Catalyst：一阶凸优化的加速元算法》，*Journal of Machine Learning Research* 18, 2018 —— Catalyst 元算法的原始文献。  
- Wilson、 Recht & Jordan，《优化中加速方法的 Lyapunov 分析》，*Journal of Machine Learning Research* 22, 2021 —— 提出统一分析框架的重要工作。