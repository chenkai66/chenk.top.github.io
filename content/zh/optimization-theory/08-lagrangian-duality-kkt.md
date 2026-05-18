---
title: '优化理论（八）：Lagrangian 对偶与 KKT 条件'
date: 2022-09-24 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 8
series_total: 12
lang: zh
mathjax: true
description: "约束如何转化为价格：拉格朗日函数、弱对偶性、保证强对偶性的 Slater 条件、KKT 条件作为最优性的充要条件，以及为何 SVM 的对偶问题远小于其原始问题；包含完整证明与鞍点表征。"
disableNunjucks: true
translationKey: "optim-08"
---
约束优化中最具深远意义的思想是：**约束具有价格**。拉格朗日函数通过为每个不等式约束赋予一个非负乘子、为每个等式约束赋予一个自由（无符号限制）乘子，将带约束的问题转化为无约束问题。由此得到的无约束问题可能更易求解（如支持向量机 SVM 的对偶问题），也可能提供一个可验证的下界（如线性规划 LP 对偶性用于整数规划的可行性认证）。

本文系统阐述以下核心内容：

- **弱对偶性（Weak duality）**：对偶问题的最优值恒为原始问题最优值的下界——无需任何假设；
- **强对偶性（Strong duality）**：在 Slater 条件（或凸问题 + 线性约束）成立时，对偶间隙为零；
- **KKT 条件（Karush–Kuhn–Tucker conditions）**：原始驻点性 + 对偶可行性 + 互补松弛性，构成实用的最优性系统；
- **鞍点刻画（Saddle-point characterization）**：拉格朗日函数的鞍点恰好对应最优的原始–对偶变量对。

每一项结论均给出严格证明或明确引用。最后以 SVM 为例收尾：其对偶问题将问题维度从 $d$（特征维数）降至 $n$（训练样本数）——这正是核方法（kernel method）最初展现的“魔法”。

---

约束优化里我觉得最深刻的一个想法是这一句话：**约束是有价格的**。

每个不等式约束配一个非负乘子，每个等式约束配一个自由乘子，原本带约束的问题就变成了一个无约束问题。这个无约束问题有时候比原问题好解（SVM 的对偶问题就是这种情况），有时候只是给你一个有用的下界（整数规划里 LP 松弛就是干这个的）。

这篇文章我把整个理论按“一定成立 → 通常成立 → 直接可用”的顺序讲：

- **弱对偶性**：不需要任何假设，对偶最优值一定 $\leq$ 原问题最优值。
- **强对偶性**：在 Slater 条件（凸 + 内点存在）下，两边相等。
- **KKT 条件**：把上面两条翻译成可以直接验证的等式/不等式系统。
- **鞍点刻画**：拉格朗日的鞍点就是最优原始-对偶对——这是后面所有 minimax 算法的源头。

最后用 SVM 收尾：你会看到对偶变量数从特征数 $d$ 降到样本数 $n$，正是核方法最早展示的“魔法”。

## 你将学到什么
1. 如何构造拉格朗日函数与对偶函数；  
2. 弱对偶性的证明（仅需一行推导）；  
3. Slater 条件及其在凸问题中导出强对偶性的简洁证明；  
4. KKT 系统的必要性与充分性条件；  
5. 拉格朗日函数的鞍点视角，及其与博弈论的内在联系；  
6. 完整演算示例：SVM 的对偶问题。

## 前置知识

[第 01 篇](../01-convex-analysis-foundations/)–[第 02 篇](../02-smoothness-strong-convexity-nesterov/)（凸集、凸函数、次梯度、光滑性、强凸性）。

---

## 问题设定

考虑如下**原始问题（primal problem）**
$$
\begin{aligned}
\text{最小化}\quad & f_0(x) \\
\text{满足}\quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p,
\end{aligned} \tag{P}
$$
其最优值记为 $p^\star$。定义**拉格朗日函数（Lagrangian）**
$$
L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x),
$$
其中**对偶变量（dual variables）** $\lambda \in \mathbb{R}^m_+$（不等式约束对应非负乘子），$\nu \in \mathbb{R}^p$（等式约束对应自由乘子）。

定义**对偶函数（dual function）**
$$
g(\lambda, \nu) = \inf_x L(x, \lambda, \nu).
$$
该对偶函数关于 $(\lambda, \nu)$ **恒为凹函数**——无论 $f_0, f_i, h_j$ 是否为凸函数，它都是关于 $(\lambda, \nu)$ 的一族仿射函数的逐点下确界。

**对偶问题（dual problem）** 为：
$$
\text{最大化}\quad g(\lambda, \nu) \quad \text{满足 } \lambda \geq 0, \tag{D}
$$
其最优值记为 $d^\star$。

---

## 弱对偶性

> **定理（弱对偶性）**：$d^\star \leq p^\star$。

**证明**：对任意原始可行解 $x$（即满足 (P) 中所有约束）及任意满足 $\lambda \geq 0$ 的 $(\lambda, \nu)$，有  
$$
L(x, \lambda, \nu) = f_0(x) + \underbrace{\sum_i \lambda_i f_i(x)}_{\leq 0} + \underbrace{\sum_j \nu_j h_j(x)}_{= 0} \leq f_0(x).
$$
对左侧关于 $x$ 取下确界，得  
$$
g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) \leq L(x, \lambda, \nu) \leq f_0(x).
$$
再对右侧关于所有原始可行 $x$ 取下确界，即得 $g(\lambda, \nu) \leq p^\star$。最后对左侧关于所有 $\lambda \geq 0, \nu$ 取上确界，即得 $d^\star \leq p^\star$。$\blacksquare$

该定理**不依赖任何凸性假设**——对任意（无论多么病态的）约束优化问题均成立。对偶问题提供了最优性的认证机制：若能找到原始可行解 $x$ 和对偶可行解 $(\lambda, \nu)$，使得 $f_0(x) = g(\lambda, \nu)$，则 $x$ 必为原始问题的最优解。这正是整数规划分支定界法（branch-and-bound）的理论基础。

差值 $p^\star - d^\star \geq 0$ 称为**对偶间隙（duality gap）**。当该间隙为零时，称**强对偶性（strong duality）** 成立。

![弱对偶性与对偶间隙](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig1.png)
*图 1 —— 弱对偶性。对偶函数 $g(\lambda)$ 始终不超过原始最优值 $p^\star$，其上确界 $d^\star$ 是最佳下界；阴影区域即为对偶间隙，在强对偶性成立时为零。*

## 强对偶性

### 斯莱特条件（Slater’s condition）

> **斯莱特条件。** 存在一点 $x \in \mathrm{relint}(\mathrm{dom}(f_0))$，使得对所有非仿射的约束指标 $i$ 满足 $f_i(x) < 0$，且对所有等式约束指标 $j$ 满足 $h_j(x) = 0$。（等价地：存在一个严格可行点；仿射不等式约束允许取等号而无需严格性。）

> **定理（强对偶性，凸情形）。** 若原问题（P）是凸的（即目标函数 $f_0$ 和不等式约束函数 $f_i$ 均为凸函数，等式约束函数 $h_j$ 均为仿射函数），且斯莱特条件成立，则对偶最优值等于原问题最优值，即 $d^\star = p^\star$，且对偶最优解可达。

**证明概要。** 该证明采用分离超平面方法，作用于如下**值函数**：
$$
V(u, v) = \inf\{f_0(x) : f_i(x) \leq u_i,\; h_j(x) = v_j\}.
$$
由于（P）是凸的，故 $V$ 在其定义域上是凸函数。斯莱特条件保证了 $0 \in \mathrm{relint}(\mathrm{dom}(V))$。对偶函数即为 $V$ 在非负乘子上的共轭函数之负：
$$
g(\lambda, \nu) = -V^*(-\lambda, -\nu) \quad \text{其中 } \lambda \geq 0.
$$
借助共轭函数性质，以及 $V$ 在 $0$ 附近是凸且下半连续的事实，可得 $V(0) = -V^{**}(0)$，进而推出 $p^\star = d^\star$。

完整证明见 Boyd 与 Vandenberghe 《凸优化》第 5.3.2 节；关键步骤是在凸集 $\{(u, t) : t \geq V(u)\}$ 的边界点 $(0, V(0))$ 处应用支撑超平面定理。斯莱特条件确保所得到的支撑超平面非竖直，从而导出有限的拉格朗日乘子。$\blacksquare$

![值函数与支撑超平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig4.png)
*图 4 —— 通过值函数解释强对偶性。$V(u)$ 是扰动约束后的最优值函数；凸性加斯莱特条件保证 $u=0$ 处存在非竖直的支撑超平面，其斜率恰好为 $-\lambda^\star$。*

### 斯莱特条件不成立的情形

若斯莱特条件不成立，强对偶性仍可能成立（例如线性规划 LP 总是满足强对偶性，无需斯莱特条件），但也可能出现正的对偶间隙（duality gap）。典型病态例子包括：

- 在 $y > 0$ 上极小化 $x^2/y$ —— 该问题是凸的，$p^\star = 0$，但其对偶问题满足 $d^\star = -\infty$（不满足斯莱特条件，亦无相对内部可行点）；
- 非凸问题可呈现任意大小的对偶间隙；二次约束二次规划（QCQP）的半定规划（SDP）松弛是著名实例，其对偶间隙恰好等于整数性间隙（integrality gap）。

对于线性规划（LP）和凸二次规划（QP），只要原问题（P）与对偶问题（D）均可行，则强对偶性一定成立。而对于半定规划（SDP），斯莱特条件（即存在一个严格正定的可行点）通常是保证强对偶性的标准假设。

## KKT 条件

若 $x^\star$ 是原始问题的最优解，且 $(\lambda^\star, \nu^\star)$ 是对偶问题的最优解，并满足**强对偶性**（strong duality），则 **Karush–Kuhn–Tucker （KKT）条件** 成立：

| 条件                                                                 | 名称                     |
| -------------------------------------------------------------------- | ------------------------ |
| $\nabla f_0(x^\star) + \sum_i \lambda_i^\star \nabla f_i(x^\star) + \sum_j \nu_j^\star \nabla h_j(x^\star) = 0$ | 原始平稳性（stationarity） |
| $f_i(x^\star) \leq 0,\ h_j(x^\star) = 0$                             | 原始可行性（primal feasibility） |
| $\lambda_i^\star \geq 0$                                             | 对偶可行性（dual feasibility）   |
| $\lambda_i^\star f_i(x^\star) = 0$（对所有 $i$）                      | 互补松弛性（complementary slackness） |

**为何在强对偶下这些条件成立？**  
强对偶性给出 $f_0(x^\star) = L(x^\star, \lambda^\star, \nu^\star) \leq L(x, \lambda^\star, \nu^\star)$ 对所有 $x$ 成立。因此 $x^\star$ 是拉格朗日函数 $L(\cdot, \lambda^\star, \nu^\star)$ 在 $\mathbb{R}^n$ 上的全局最小值点，从而导出平稳性条件。原始/对偶可行性由定义直接保证。唯一非平凡的步骤是互补松弛性：由 $L(x^\star, \lambda^\star, \nu^\star) = f_0(x^\star)$ 可得 $\sum_i \lambda_i^\star f_i(x^\star) = 0$；又因每一项 $\lambda_i^\star f_i(x^\star) \leq 0$，故各项必须各自为零。

![KKT 条件的几何图景](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig3.png)
*图 3 —— 二维下的 KKT 平稳性。在最优点 $x^\star$ 处，目标函数负梯度 $-\nabla f_0$ 落在活跃约束梯度所张成的凸锥中，权重 $\lambda_i^\star$ 非负。*

### KKT 作为充分最优性条件（凸问题）

> **定理。** 设原始问题 (P) 是凸的，且 $(x^\star, \lambda^\star, \nu^\star)$ 满足 KKT 条件，则 $x^\star$ 是原始最优解，$(\lambda^\star, \nu^\star)$ 是对偶最优解。

**证明。** 平稳性条件表明 $\nabla_x L(x^\star, \lambda^\star, \nu^\star) = 0$。由于 $L(\cdot, \lambda^\star, \nu^\star)$ 是凸函数（它是凸函数 $f_0$、非负加权的凸函数 $\lambda_i^\star f_i$（因 $\lambda_i^\star \geq 0$）以及仿射函数 $\nu_j^\star h_j$ 的和），平稳性即意味着 $x^\star$ 全局最小化 $L(\cdot, \lambda^\star, \nu^\star)$。因此  
$$
g(\lambda^\star, \nu^\star) = L(x^\star, \lambda^\star, \nu^\star) = f_0(x^\star) + \underbrace{\sum_i \lambda_i^\star f_i(x^\star)}_{= 0 \text{ 由互补松弛性}} + \underbrace{\sum_j \nu_j^\star h_j(x^\star)}_{= 0} = f_0(x^\star).
$$
于是弱对偶性取等：$f_0(x^\star) = g(\lambda^\star, \nu^\star) \leq p^\star \leq f_0(x^\star)$，故 $x^\star$ 为最优解。$\blacksquare$

这正是使 KKT 成为实用核心工具的关键结论：对凸问题而言，KKT 给出一个有限的方程与不等式系统，其解即为最优解。

### KKT 失效的情形

KKT 条件在最优解处成立**仅当满足某种约束规范（constraint qualification）**。Slater 条件是其中一种；LICQ（活跃约束梯度的线性无关性，linear independence of active constraint gradients）是另一种。若缺乏任一约束规范，最优解可能不存在拉格朗日乘子，而依赖 KKT 系统的基于梯度的方法亦可能停滞。

对非凸问题，KKT（在约束规范下）是必要条件，但**非充分条件**——KKT 点包括局部最优解、拉格朗日函数的鞍点，甚至某些非平稳点。

---

## 鞍点刻画

拉格朗日函数定义了一个**极小-极大博弈**（min-max game）：原始玩家 $x$ 力求最小化，对偶玩家 $(\lambda, \nu)$ 力求最大化。

> **定理（鞍点原理）。** 强对偶性成立，且 $(x^\star, \lambda^\star, \nu^\star)$ 是原始-对偶最优解，**当且仅当** $(x^\star, \lambda^\star, \nu^\star)$ 是 $L$ 的一个鞍点：
> 
$$
L(x^\star, \lambda, \nu) \leq L(x^\star, \lambda^\star, \nu^\star) \leq L(x, \lambda^\star, \nu^\star) \quad \forall x,\ \forall \lambda \geq 0,\ \nu.
$$
右侧不等式表示：给定乘子 $(\lambda^\star, \nu^\star)$，$x^\star$ 是原始最优；左侧不等式表示：$(\lambda^\star, \nu^\star)$ 是对偶最优。

![拉格朗日函数的鞍点曲面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig2.png)
*图 2 —— 拉格朗日函数的鞍点曲面。对每个 $\lambda$ 关于 $x$ 取最小，得到对偶函数（绿色）；对每个原始可行 $x$ 关于 $\lambda$ 取最大，得到原始值（橙色）。强对偶下两者相交于鞍点。*

鞍点刻画是以下方法的理论基础：

- **增广拉格朗日法（Augmented Lagrangian methods）**：交替进行原始与对偶更新，并引入二次惩罚项以增强稳定性；
- **ADMM（见第 06 篇）**：将原始变量拆分，然后在一类特殊拉格朗日函数上执行原始-对偶上升；
- **GAN 训练**：本质上即是一场鞍点博弈（尽管非凸，故上述理论不能直接适用）；
- **在线原始-对偶算法**：对双方玩家分别给出遗憾界（regret bounds），可保证近似最优性。

## 实例详解：SVM 对偶问题

线性可分数据 $\{(x_i, y_i)\}_{i=1}^n$（其中 $y_i \in \{-1, +1\}$）上的硬间隔支持向量机（Hard-margin SVM）：
$$
\begin{aligned}
\min_{w, b} \quad & \tfrac{1}{2} \|w\|_2^2 \\
\text{s.t.} \quad & y_i (w^\top x_i + b) \geq 1, \quad i = 1, \ldots, n.
\end{aligned}
$$
其拉格朗日函数为：
$$
L(w, b, \alpha) = \tfrac{1}{2} \|w\|_2^2 - \sum_i \alpha_i [y_i (w^\top x_i + b) - 1].
$$
令 $\nabla_w L = 0$，得最优解 $w^\star = \sum_i \alpha_i y_i x_i$；令 $\partial_b L = 0$，得约束 $\sum_i \alpha_i y_i = 0$。代回原式可得对偶函数：
$$
g(\alpha) = -\tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i,
$$
从而得到对偶问题：
$$
\begin{aligned}
\max_\alpha \quad & \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0.
\end{aligned}
$$
![SVM 对偶与支持向量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/08-lagrangian-duality-kkt/fig5.png)
*图 5 —— SVM 对偶问题。最大间隔分离超平面（黑色）完全由支持向量（紫色圆圈）决定；互补松弛性使得所有内部样本 $\alpha_i^\star = 0$，而 $w^\star = \sum_i \alpha_i^\star y_i x_i$ 是一个稀疏加权和。*

**为何这意义重大？**

1. **变量规模**：对偶问题含 $n$ 个变量（训练样本数），原始问题含 $d + 1$ 个变量（特征维数加偏置项）。当 $d \gg n$ 时，对偶问题显著更小。
2. **核技巧（Kernel trick）**：对偶形式仅通过内积 $x_i^\top x_j$ 依赖数据；将其替换为任意正定核 $K(x_i, x_j)$，即可在不显式映射到高维特征空间的前提下获得非线性分类器。
3. **稀疏性**：由互补松弛性可知，仅当 $y_i(w^\top x_i + b) = 1$（即样本恰位于间隔边界上）时，$\alpha_i > 0$ —— 这些样本即为**支持向量（support vectors）**；绝大多数 $\alpha_i$ 为零。

这一结构同样支撑着核岭回归（kernel ridge regression）、核主成分分析（kernel PCA）与高斯过程（Gaussian processes）：它们的对偶形式均只通过数据点之间的内积“观测”数据，而该内积可被任意正定核替代。

---

## 对偶性失效的场景：含噪声与大规模机器学习

当训练样本数达 $n = 10^9$（如大规模广告点击率预测 CTR）时，SVM 对偶问题拥有 $10^9$ 个变量 —— 规模反而超过原始问题。此时对偶性的优雅荡然无存；这也是深度学习完全绕开对偶理论、直接在原始问题上使用随机梯度下降（SGD）的重要原因之一。

对于满足强对偶性的凸问题，现代实践策略如下：

- **中小规模 $n$**（$\leq 10^4$）：显式求解对偶问题，通常借助二次规划（QP）求解器（如 libsvm）；
- **大规模 $n$**：采用**随机对偶坐标上升法（Stochastic Dual Coordinate Ascent, SDCA）**（Shalev-Shwartz & Zhang, 2013）—— 每次仅选取一个 $\alpha_i$ 进行优化，在保持对偶理论保证的同时将内存开销控制在 $O(n)$；
- **凸–凹鞍点问题**：采用**原始–对偶方法（primal-dual methods）**（Chambolle–Pock, 2011），广泛应用于图像去噪与稀疏信号恢复等任务。

---

## 总结

| 概念                 | 它为你提供                                                                 |
| --------------------- | -------------------------------------------------------------------------- |
| 弱对偶性（Weak duality）   | 原始问题最优值的一个下界；恒成立。                                          |
| 斯莱特条件（Slater's condition） | 凸优化问题中强对偶性成立的一个充分条件。                                       |
| 强对偶性（Strong duality）   | 原始与对偶最优值相等（零对偶间隙）；拉格朗日乘子存在。                              |
| KKT 条件系统             | 在满足约束规范（CQ）下为必要条件；对凸问题亦为充分最优性条件。                        |
| 鞍点视角（Saddle-point view） | 最小–最大刻画；为 ADMM、增广拉格朗日法及生成对抗网络（GANs）等提供理论桥梁。              |
| 对偶问题               | 往往变量更少、解更稀疏、或可核化；是 SVM 等经典算法的理论基石。                         |

第 09 篇文章将从带约束优化问题出发，介绍**内点法（interior-point methods）**：通过引入障碍函数（barrier）将不等式约束光滑化，并应用牛顿法求解。我们将看到，中心路径（central path）的迭代复杂度为 $O(\sqrt{n} \log(1/\epsilon))$，这使得内点法成为中等规模凸规划的黄金标准。

## 参考文献

- Boyd & Vandenberghe, *Convex Optimization*, 第 5 章 —— 经典教材，所有例子均有详尽推导；
- Bertsekas, *Convex Optimization Theory*, 第 5 章 —— 从更抽象的对偶理论视角给出严格证明；
- Rockafellar, *Conjugate Duality and Optimization*, 1974 —— 基于共轭函数的深层对偶理论；
- Shalev-Shwartz & Zhang, *Stochastic Dual Coordinate Ascent for Regularized Loss Minimization*, JMLR 14, 2013 —— 大规模机器学习中的对偶优化方法。
