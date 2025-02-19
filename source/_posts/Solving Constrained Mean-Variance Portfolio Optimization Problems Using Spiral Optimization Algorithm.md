---
title: Solving Constrained Mean-Variance Portfolio Optimization Problems Using Spiral Optimization Algorithm
tags: 
  - Portfolio optimization
categories: Paper
date: 2024-02-03 19:00:00
mathjax: true
---

本文讨论了在投资组合优化中如何平衡收益与风险，并提出了一种使用修改后的螺旋优化算法（SOA）解决带有买入门槛和基数约束的均值-方差投资组合优化问题。通过将该问题建模为混合整数非线性规划问题（MINLP），验证了该算法在投资组合优化问题上的有效性。

<!-- more -->

# 背景介绍

投资组合是为了实现投资目标而维护的各种资产的组合。这些资产的选择决定了投资组合的风险与收益，投资组合可以包含股票、债券、房地产和现金等多种资产。投资者一般希望在风险最小化的前提下，追求最高的投资回报。然而，高收益往往伴随着高风险，因此需要一种方法来最小化风险。马克维茨（1959）提出的均值-方差优化模型（Mean-Variance Model）是解决这一问题的理论基础。

在均值-方差模型中，资产的回报用期望值表示，风险用回报的方差来衡量。通过组合不同资产，投资者可以在给定的风险水平下最大化预期回报，或者在目标回报下最小化风险。本文采用了马克维茨均值-方差模型，并加入了买入门槛和基数约束来模拟实际投资情况。

# 投资组合优化问题的混合整数非线性规划模型

## 投资组合优化问题

投资组合理论涉及投资者对风险和回报的估计，这些估计可以通过统计手段来构建投资组合。马克维茨（1959年）提出的均值-方差模型描述了如何将资产组合成一个有效且多样化的投资组合。在实际中，证券投资者通常通过组合多种证券或股票来实现分散化投资，形成投资组合。

首先假设未来的资产回报可以通过统计方法进行估计，风险则可以通过回报分布的方差来衡量。令 $y_i$ 表示投资于第 $i$ 项资产的资金比例（其中 $0 \leq y_i \leq 1$），则 $\mathbf{y}^T = (y_1, y_2, \dots, y_n)$ 为投资比例的向量。对于 $n$ 种资产，平均回报向量为 $\overline{\mathbf{r}}^T = (\bar{r}_1, \bar{r}_2, \dots, \bar{r}_n)$，协方差矩阵 $Q$ 是一个 $n \times n$ 的正半定矩阵。

投资组合的优化目标是最小化总风险，即最小化投资组合的总方差 $V(y)$。其公式为：

$$
\min V(y) = \mathbf{y}^T Q \mathbf{y}
$$

约束条件为：

$$
\overline{\mathbf{r}}^T \mathbf{y} = R_p
$$

其中 $R_p$ 表示投资组合的预期回报。此外，由于总投资金额为 1，因此：

$$
\sum_{i=1}^{n} y_i = 1 \quad \text{或} \quad \mathbf{e}^T \mathbf{y} = 1
$$

其中 $\mathbf{e}^T = (1, 1, \dots, 1)$。

上述模型的约束条件还包括：投资比例 $y_i \geq 0$，即不允许卖空交易。因此模型变为：

$$
\begin{aligned}
\min \quad & V(y) = \mathbf{y}^T Q \mathbf{y} \\
\text{subject to:} \quad & \overline{\mathbf{r}}^T \mathbf{y} = R_p \\
& \sum_{i=1}^{n} y_i = 1 \\
& y_i \geq 0, \quad i = 1, 2, \dots, n
\end{aligned}
$$

## 买入门槛约束

买入门槛约束定义为每个资产的最小投资比例 $l_i$，即投资比例 $y_i$ 不能低于门槛 $l_i$。买入门槛消除了在优化投资组合中可能出现的投资比例过小的情况。带有买入门槛的投资组合优化问题如下：

$$
\begin{aligned}
\min \quad & V(y) = \mathbf{y}^T Q \mathbf{y} \\
\text{subject to:} \quad & \overline{\mathbf{r}}^T \mathbf{y} = R_p \\
& \mathbf{e}^T \mathbf{y} = 1 \\
& l_i z_i \leq y_i \leq u_i z_i, \quad i = 1, 2, \dots, n \\
& 0 < l_i < u_i \leq 1 \\
& z_i \in \{0, 1\}
\end{aligned}
$$

其中 $z_i$ 是 0 或 1 的二值变量，表示是否投资于第 $i$ 项资产。

## 基数约束

基数约束是指投资组合中允许的最大资产数量，即只能选择 $K$ 种资产进行投资。带有基数约束的投资组合优化问题可以表示为：

$$
\begin{aligned}
\min \quad & V(y) = \mathbf{y}^T Q \mathbf{y} \\
\text{subject to:} \quad & \overline{\mathbf{r}}^T \mathbf{y} = R_p \\
& \mathbf{e}^T \mathbf{y} = 1 \\
& \sum_{i=1}^{n} z_i = K \\
& l_i z_i \leq y_i \leq u_i z_i, \quad i = 1, 2, \dots, n \\
& 0 < l_i < u_i \leq 1 \\
& z_i \in \{0, 1\}
\end{aligned}
$$

其中 $K$ 表示投资组合中选择的资产数量。

# 混合整数非线性规划

基于 Kania 和 Sidarto（2016）的研究，混合整数非线性规划问题可以表示为：

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

其中，约束条件如下：

$$
g_i(\mathbf{x}) = 0, \quad i = 1, 2, \dots, M
$$

$$
h_j(\mathbf{x}) \leq 0, \quad j = 1, 2, \dots, N
$$

变量 $\mathbf{x} = (x_1, x_2, \dots, x_q, x_{q+1}, \dots, x_n)^T$ 中，前 $q$ 个变量是整数。

# 数值示例

本文使用 Bartholomew-Biggs 和 Kane（2009）提供的五个资产的数据进行了数值验证。资产的平均回报向量为：

$$
\overline{r} = (-0.056, 0.324, 0.343, 0.132, 0.108)^T
$$

方差协方差矩阵为：

$$
Q = \begin{pmatrix}
2.4037 & -0.0222 & 0.5230 & 0.2612 & 0.6126 \\
-0.0222 & 1.8912 & 0.0442 & 0.0020 & 0.4272 \\
0.5230 & 0.0442 & 1.7704 & 0.2283 & 0.3103 \\
0.2612 & 0.0020 & 0.2283 & 4.4812 & -0.1134 \\
0.6126 & 0.4272 & 0.3103 & -0.1134 & 7.7490
\end{pmatrix}
$$

目标回报率为 $R_p = 0.25 \%$，投资比例的下限为 $y_{\min} = 0.05$，惩罚参数为 $\rho = 10^8$。在进行50次迭代后，优化得到的最小风险为0.6969，总投资比例为1，活跃资产数量为5。

# 结论

本文提出了一种修改后的螺旋优化算法（SOA MINLP），用于解决带有买入门槛和基数约束的均值-方差投资组合优化问题。通过数值示例验证了该算法的有效性，并与现有的Quasi-Newton和DIRECT方法进行了比较，结果表明 SOA MINLP 是求解该类优化问题的有效工具。