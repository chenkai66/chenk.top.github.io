---
title: '优化理论（九）：内点法与自和谐障碍函数'
date: 2022-09-26 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 9
lang: zh
mathjax: true
description: '内点法何以成为凸规划默认求解器：以对数障碍函数替代不等式约束，参数化中心路径，并应用牛顿法；涵盖自协调性理论及著名的 $O(\sqrt{n} \log(1/\varepsilon))$ 迭代复杂度证明。'
disableNunjucks: true
translationKey: "optim-09"
---
1984 年，Karmarkar 证明了线性规划（LP）问题不仅在理论上（椭球法早已在纸面上实现这一点），更在**实际中**可于多项式时间内求解。他的**内点法**始终停留在可行多面体内部，并以 $O(n L)$ 次迭代收敛，远优于单纯形法的指数级最坏时间复杂度。短短十年之内，Nesterov 与 Nemirovski 利用**自协调障碍函数**（self-concordant barrier）框架，将该思想推广至**全部凸规划问题**。其标志性成果——对 $n$ 维问题仅需 $O(\sqrt{n} \log(1/\epsilon))$ 次牛顿迭代——至今仍是中等规模凸优化的**黄金标准**。

本文将这一技术体系拆解为三层结构，逐层展开：

1. **障碍法**（barrier method）：用对数惩罚项替代不等式约束，并随惩罚权重递增追踪**中心路径**（central path）；
2. **自协调性**（self-concordance）：障碍函数所满足的一类解析性质，它保障牛顿法具有良好行为——收敛域大小为 $O(1)$，而非通常的 $O(\mu / L)$；
3. **原始-对偶内点法**（primal-dual interior-point）：现代主流变体，同步求解原始与对偶变量，被几乎所有商用 LP/QP/SDP 求解器所采用。

我们完整给出中心路径复杂度的严格证明。

## 你将学到的内容

1. 对数障碍函数与中心路径的定义与几何意义；
2. 自协调性的定义、三个关键推论，以及参数 $\sqrt{\nu}$ 的来源与作用；
3. 在自协调函数上执行**阻尼牛顿法**（damped Newton）——可在常数大小区域内实现二次收敛；
4. 障碍法的整体迭代复杂度：外层牛顿步数为 $O(\sqrt{\nu} \log(\nu / \epsilon))$；
5. 原始-对偶内点法的构造原理，及其在实践中占据主导地位的根本原因。

## 先修知识

第 02 篇（光滑性）、第 07 篇（牛顿法）、第 08 篇（拉格朗日函数与 KKT 条件）。需熟悉方向导数，并能理解涉及“Hessian 度量”（Hessian metric）的论证。

---

## 1. 障碍法

考虑如下凸优化问题：
$$
\min_x f_0(x) \quad \text{s.t. } f_i(x) \leq 0, \ i = 1, \ldots, m, \quad Ax = b.
$$
我们将不等式约束替换为**对数障碍函数**（logarithmic barrier）：
$$
\phi(x) = -\sum_{i=1}^m \log(-f_i(x)),
$$
该函数在严格可行域 $\{x : f_i(x) < 0\}$ 上取有限值，且在边界处趋于 $+\infty$。对每个 $t > 0$，求解带等式约束的无约束化子问题：
$$
\min_x \quad t f_0(x) + \phi(x), \quad Ax = b. \tag{$P_t$}
$$
这是一个等式约束凸优化问题；其唯一最优解记为 $x^\star(t)$，当 $t$ 变化时，$x^\star(t)$ 的轨迹即为**中心路径**（central path）。

### 1.1 中心路径的性质

对每个 $t > 0$，有如下结论（在温和正则性条件下成立）：

- $x^\star(t)$ 是 ($P_t$) 的唯一极小点；
- ($P_t$) 的 KKT 条件给出对偶变量 $\lambda_i(t) := 1/(t \cdot (-f_i(x^\star(t)))) \geq 0$，并满足**扰动互补松弛条件**（perturbed complementary slackness）：
  $$
  \lambda_i(t) (-f_i(x^\star(t))) = 1/t.
  $$
  这正是经典 KKT 系统，只是互补松弛条件由精确为零变为偏移 $1/t$。

- 中心路径上的对偶间隙（duality gap）**恰好等于** $m/t$：
  $$
  f_0(x^\star(t)) - p^\star \leq m / t.
  $$
  为什么？令 $\nu(t) = (\nu_1(t), \ldots, \nu_p(t))$ 为对应等式约束 $Ax = b$ 的拉格朗日乘子向量，则 $(\lambda(t), \nu(t))$ 是一个对偶可行点；直接计算可知，拉格朗日函数在此对偶点处的取值恰为 $f_0(x^\star(t)) - m/t$。

因此，**当 $t \to \infty$ 时，$x^\star(t) \to x^\star$**，且对偶间隙以 $1/t$ 的速率衰减。这便导出了最基础的内点算法：取 $t = m/\epsilon$，求解 ($P_t$)，即可获得 $\epsilon$-次优解。

### 1.2 天真算法及其缺陷

当 $t$ 极大时，子问题 ($P_t$) 会严重病态——目标中 $t f_0$ 项在边界附近起主导作用，导致牛顿法难以稳定收敛。解决方法是：**沿递增的 $t$ 序列依次求解多个 ($P_t$)**，并将前一问题的解作为后一问题的初始点（即**热启动**，warm-start）：

```
算法：障碍法
输入：严格可行初值 x_0，初始 t_0 > 0，目标精度 ε
      更新因子 μ > 1（典型取值：μ = 10 或 100）

for k = 0, 1, 2, ...:
    以 x_k 为初值，用阻尼牛顿法求解 (P_{t_k})，得 x_{k+1}
    if m / t_k < ε: 停止
    t_{k+1} = μ * t_k
返回 x_{k+1}
```

每次外层迭代将 $t$ 放大 $\mu$ 倍。外层迭代总数为 $\log(m / (t_0 \epsilon)) / \log \mu$，典型值为 20–50 次。核心问题是：**每次内层牛顿求解需要多少步？**——而这正是自协调性发挥作用的关键所在。
## 2. 自协调性（Self-concordance）

凸函数 $\phi : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 称为**自协调的**，若对任意 $x \in \mathrm{dom}(\phi)$ 及任一方向 $u$，均有  
$$
\Big| \frac{d^3}{dt^3} \phi(x + tu) \Big|_{t=0} \Big| \leq 2 \big(u^\top \nabla^2 \phi(x) u \big)^{3/2}.
$$  
即：三阶方向导数受 Hessian 度量控制。

最重要的例子包括：

- $\phi(x) = -\log x$ 定义在 $(0, \infty)$ 上：$\phi'(x) = -1/x$，$\phi''(x) = 1/x^2$，$\phi'''(x) = -2/x^3$。验证：$|\phi'''| / (\phi'')^{3/2} = 2 / x^3 / x^{-3} = 2$，等号成立，紧界。
- $\phi(X) = -\log \det X$ 定义在 $\mathbf{S}^n_{++}$（$n$ 阶正定对称矩阵锥）上：自协调。
- 对仿射函数 $f_i$ 构造的障碍函数 $-\sum_i \log(-f_i(x))$：自协调。（更一般地，当各 $f_i$ 本身“性质良好”时亦成立。）

### 2.1 自协调性为何重要

自协调性带来三大关键性质（常被称作“魔法后果”）：

**1. 牛顿减量（Newton decrement）具有明确意义。** 定义  
$$
\lambda_\phi(x) := \sqrt{\nabla \phi(x)^\top [\nabla^2 \phi(x)]^{-1} \nabla \phi(x)}.
$$  
这是 $\phi$ 的自然尺度不变的“距最优解距离”度量。若 $\phi$ 自协调，则当 $\lambda_\phi(x) \leq 0.68$ 时，有  
$$
\phi(x) - \phi^\star \leq \lambda_\phi(x)^2.
$$

**2. 阻尼牛顿法（damped Newton）以 $O(1)$ 量级常数保证进展。** 考虑阻尼牛顿迭代：  
$$
x_+ = x - \frac{1}{1 + \lambda} \, [\nabla^2 \phi(x)]^{-1} \nabla \phi(x), \quad \text{其中 } \lambda = \lambda_\phi(x).
$$  
若 $\phi$ 自协调，则  
$$
\phi(x_+) \leq \phi(x) - \omega(\lambda),
$$  
其中 $\omega(\lambda) = \lambda - \log(1 + \lambda)$。只要 $\lambda \geq \frac{1}{4}$，就有 $\omega(\lambda) \geq 0.02$，故每步牛顿迭代使 $\phi$ 至少下降一个绝对常数。

**3. 在常数半径区域内实现二次收敛。** 若 $\lambda_\phi(x) \leq \frac{1}{4}$，则执行**完整牛顿步**可得  
$$
\lambda_\phi(x_+) \leq 2 \lambda_\phi(x)^2.
$$  
此即二次收敛——且该收敛半径 $\frac{1}{4}$ **与问题条件数无关**。

正是上述三条性质，保障了牛顿法在自协调函数上的鲁棒性：先经“阻尼牛顿阶段”，以常数降幅快速逼近，直至 $\lambda \leq 1/4$；再进入“二次收敛阶段”，仅需 $O(\log \log(1/\epsilon))$ 步即可达到精度 $\epsilon$。

### 2.2 将其整合应用于障碍法（barrier method）

在障碍法的每一外层迭代中，我们求解 $\min_x t f_0(x) + \phi(x)$，并采用牛顿法。若 $f_0$ 自协调（或更一般地，若 $t f_0 + \phi$ 自协调），且从上一轮解 $x^\star(t/\mu)$（即前一参数 $t/\mu$ 对应的最优解）出发热启动（warm-start），则该初值已落在新目标函数的二次收敛区域内。因此，**每次外层迭代仅需 $O(1)$ 次牛顿迭代**，其代价与 $t$ 及问题规模均无关。

总牛顿迭代次数 = （外层迭代次数）$\times$ $O(1)$ = $O(\log(m / \epsilon))$。

但此处存在一个微妙之处：为精确刻画**障碍参数** $\nu$ 的影响，我们需要更精细的界。
## 3. 障碍参数与 $\sqrt{\nu}$ 收敛速率

若一个自协调函数 $\phi$ 满足：对所有 $x \in \mathrm{dom}(\phi)$，有  
$$
\nabla \phi(x)^\top [\nabla^2 \phi(x)]^{-1} \nabla \phi(x) \leq \nu,
$$  
则称其具有**障碍参数** $\nu \geq 1$。  
等价地：$\lambda_\phi(x)^2 \leq \nu$ 在定义域内处处成立。

对于仿射约束下的对数障碍函数 $\phi = -\sum_{i=1}^m \log(-f_i(x))$，有 $\nu = m$；  
对于半正定矩阵锥上的障碍函数 $\phi = -\log \det X$（定义在 $n \times n$ 半正定矩阵上），有 $\nu = n$。

障碍参数控制中心路径随参数 $t$ 变化的“剧烈程度”：  
$$
\|x^\star(t) - x^\star(t')\|_{\nabla^2 \phi} \leq O(\sqrt{\nu} \log(t'/t)).
$$  
若将 $t$ 按因子 $\mu = 1 + 1/\sqrt{\nu}$（远小于 $\mu = 10$）更新，则热启动点仍位于新中心路径点的二次收敛区域内。此时外层迭代总次数为  
$$
\frac{\log(m / (t_0 \epsilon))}{\log(1 + 1/\sqrt{\nu})} = O(\sqrt{\nu} \log(\nu / \epsilon)),
$$  
而每次内层求解仅需 $O(1)$ 次牛顿迭代。这即著名的**短步长内点算法**，具备最优复杂度保证。

> **定理（Nesterov–Nemirovski, 1994）**：求解具有障碍参数 $\nu$ 的自协调障碍函数所刻画的凸规划问题，达到精度 $\epsilon$ 所需的牛顿迭代次数为 $O(\sqrt{\nu} \log(\nu/\epsilon))$。

对于含 $m$ 个不等式约束的线性规划（LP），$\nu = m$，复杂度为 $O(\sqrt{m} \log(m/\epsilon))$ —— 即 Karmarkar 界；  
对于 $n \times n$ 矩阵上的半定规划（SDP），$\nu = n$，复杂度为 $O(\sqrt{n} \log(n/\epsilon))$。

### 3.1 长步长 vs 短步长

教科书式的短步长方法（$\mu = 1 + 1/\sqrt{\nu}$，采用完整牛顿步）理论最简洁，但实践中较慢——需执行大量细粒度的外层迭代。**长步长**算法（$\mu = 10$ 或 $100$，采用阻尼牛顿法）虽偏离严格理论保证，却在实际中收敛快得多。其最坏情况理论复杂度退化为 $O(\nu \log(\nu/\epsilon))$，但实际性能常与短步长方法相当。

现代求解器普遍采用 **预测-校正（predictor-corrector）方案**（Mehrotra, 1992）：先用一次牛顿步预测中心路径方向，再引入二阶修正项进行校正。该框架构成了所有商用 LP/QP/SDP 求解器的基础。

---

## 4. 原始-对偶内点法

障碍法显式更新原始变量 $x$，并将对偶变量 $(\lambda, \nu)$ 作为副产品恢复。**原始-对偶法**则同步更新原始与对偶变量。

### 4.1 原始-对偶方程组

考虑带不等式约束的线性规划问题：$\min c^\top x$，满足 $Ax = b,\, x \geq 0$。其中心路径条件为：
$$
\begin{aligned}
A x &= b \quad \text{（原始可行性）} \\
A^\top \nu + \lambda &= c \quad \text{（对偶可行性）} \\
x_i \lambda_i &= 1/t \quad \text{（扰动互补松弛条件，perturbed CS）} \\
x, \lambda &\geq 0
\end{aligned}
$$
这是一个关于 $(x, \nu, \lambda)$ 的非线性方程组，以 $1/t$ 为扰动参数。对该系统应用牛顿法，并令 $1/t \to 0$，即可收敛至原始-对偶最优解对。

### 4.2 为何原始-对偶法更受青睐

- **无需严格可行初始点**：原始-对偶法可从不可行点出发，在迭代过程中同步趋近原始与对偶可行性；
- **数值条件更优**：牛顿方程组具有块结构，便于高效求解；且当 $t \to \infty$ 时，其病态程度显著低于纯原始牛顿系统；
- **自校正性（self-correcting）**：$x$ 中的误差可通过 $\lambda$ 得到修正，反之亦然。

目前绝大多数主流求解器（如 Mosek、Gurobi（QP）、SDPT3（SDP）、OSQP（QP））均实现基于预测-校正机制与自适应步长策略的原始-对偶内点法。

---

## 5. 内点法的优势场景与局限场景

**内点法占优的情形：**

- 规模适中的线性规划（LP）与二次规划（QP），变量数 $n \lesssim 10^5$；
- 半定规划（SDP）、二阶锥规划（SOCP）；
- 具有结构化约束的凸问题（如弦图稀疏性等）；
- 高精度需求场景：数十次迭代内即可获得 12 位以上有效数字精度。

**一阶方法占优的情形：**

- 极大规模问题（$n > 10^7$）且具有强稀疏结构：牛顿步最坏为 $O(n^3)$，而一阶方法具有更优的可扩展性；
- 随机优化或流式数据场景：内点法需将整个问题载入内存；
- 低精度需求问题（如 $\epsilon = 10^{-3}$ 即可）：随机梯度下降（SGD）与 Adam 等方法更具优势。

凸优化建模的经典工作流为：使用 CVXPY 描述问题，建模层自动将其转化为标准 SDP/SOCP 形式，再由内点法求解器在数秒内返回高达 10 位精度的最优解。这一端到端流水线已使凸优化成为一项常规化的工程工具。
## 6. 实例演算：通过障碍函数求解线性规划（LP）

考虑最小化 $c^\top x$，满足约束 $Ax \leq b$，其中 $A \in \mathbb{R}^{m \times n}$。对应参数 $t$ 的障碍目标函数为：
$$
B_t(x) = t c^\top x - \sum_{i=1}^m \log(b_i - a_i^\top x).
$$
其梯度与 Hessian 矩阵为：
$$
\nabla B_t(x) = t c + A^\top \frac{1}{b - Ax}, \quad \nabla^2 B_t(x) = A^\top \mathrm{diag}\big( (b - Ax)^{-2} \big) A.
$$
（此处除法为按分量进行。）

牛顿方向 $\Delta x$ 是如下线性方程组的解：
$$
\nabla^2 B_t(x) \, \Delta x = -\nabla B_t(x).
$$
当 $A$ 稀疏时，该系统通常采用稀疏 Cholesky 分解求解；最坏时间复杂度为 $O(n^3)$，但实践中往往显著更低。

以一个小型实例为例：$n = 100$、$m = 200$，要求高精度解，通常仅需约 30 次外层迭代、总计约 100 步牛顿迭代——其效率比一阶方法高出若干数量级。

---

## 7. 总结

| 概念                          | 作用                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| 对数障碍函数 $\phi$           | 用光滑罚函数替代不等式约束。                                          |
| 中心路径 $x^\star(t)$         | 一条光滑曲线，从解析中心出发，随 $t \to \infty$ 趋向最优解。             |
| 自协调性（self-concordance）  | 一种解析性质，保证牛顿法在半径为 $O(1)$ 的区域内收敛。                    |
| 障碍参数 $\nu$                | 控制中心路径上相邻点之间的步长；决定了 $\sqrt{\nu}$ 收敛速率。              |
| 原-对偶内点法（Primal-dual IPM） | 现代变体，所有商用凸优化求解器的基础。                                   |

至此，我们完成了**确定性、精确信息**下的优化理论主线。第 10 和 11 篇文章将转向 **随机方法**（含噪声的梯度）与 **非凸景观**（不再具备全局收敛保证）——这正是深度学习所处的典型优化范式。

## 延伸阅读

- Nesterov & Nemirovski，《Convex Programming 中的内点多项式算法》（*Interior-Point Polynomial Algorithms in Convex Programming*），SIAM，1994 —— 奠基性专著。  
- Boyd & Vandenberghe，《凸优化》（*Convex Optimization*），第 11 章 —— 面向工程师最清晰的讲解。  
- Renegar，《内点方法的数学视角》（*A Mathematical View of Interior-Point Methods*），SIAM，2001 —— 从几何角度阐释自协调性。  
- Wright，《原-对偶内点法》（*Primal-Dual Interior-Point Methods*），SIAM，1997 —— 面向实际算法的系统论述。  
- Mehrotra，《原-对偶内点法的实现》（*On the Implementation of a Primal-Dual Interior Point Method*），SIOPT 2, 1992 —— 所有现代求解器所采用的预估-校正（predictor-corrector）方案。