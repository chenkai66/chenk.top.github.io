---
title: "微分几何（六）：Gauss-Bonnet 定理——几何遇上拓扑"
date: 2021-06-05 09:00:00
tags:
  - differential-geometry
  - gauss-bonnet
  - topology
  - euler-characteristic
  - mathematics
categories: Mathematics
series: differential-geometry
lang: zh
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "differential-geometry-6"
description: "Gauss-Bonnet 定理将总曲率与 Euler 示性数联系，几何约束拓扑。"
---

Gauss-Bonnet 定理是数学中最美的结果之一。它说的是：把 Gauss 曲率在整个闭曲面上积分，得到一个拓扑不变量——一个只依赖曲面"橡皮膜意义上的形状"的数，与具体的几何无关。

## 局部 Gauss-Bonnet 定理

设 $R$ 是曲面 $S$ 上由简单闭合分段光滑曲线 $\partial R$ 围成的区域。$\partial R$ 由光滑弧段组成，在顶点处的外角为 $\theta_1,\ldots,\theta_k$。则：

$$\iint_R K\,dA + \int_{\partial R}\kappa_g\,ds + \sum_{i=1}^k\theta_i = 2\pi$$

其中 $\kappa_g$ 是边界的测地曲率，积分方向使区域在左边。

**例子。** 测地三角形（每边 $\kappa_g = 0$，3 个顶点外角 $\theta_i = \pi - \alpha_i$）：

$$\iint_T K\,dA + 0 + \sum(\pi-\alpha_i) = 2\pi$$

化简得 $\alpha_1+\alpha_2+\alpha_3 = \pi + \iint_T K\,dA$。球面上内角和大于 $\pi$，鞍面上小于 $\pi$。

## 整体 Gauss-Bonnet 定理

**定理。** 设 $S$ 是紧致有向无边界曲面。则：

$$\iint_S K\,dA = 2\pi\chi(S)$$

其中 $\chi(S)$ 是 $S$ 的 **Euler 示性数**。

Euler 示性数是拓扑不变量：$\chi = 2-2g$，$g$ 是亏格（把手数）。
- 球面（$g=0$）：$\chi=2$，总曲率 $= 4\pi$。
- 环面（$g=1$）：$\chi=0$，总曲率 $= 0$。
- 双环面（$g=2$）：$\chi=-2$，总曲率 $= -4\pi$。

*证明思路。* 对 $S$ 做测地三角剖分。对每个三角形 $T_j$ 用局部 Gauss-Bonnet：$\iint_{T_j}K\,dA = (\alpha_1^j+\alpha_2^j+\alpha_3^j)-\pi$。对所有三角形求和：内部边被相邻三角形共享，测地曲率贡献抵消；内部顶点的角度和为 $2\pi$。仔细簿记后得到：

$$\iint_S K\,dA = 2\pi(V-E+F) = 2\pi\chi(S)$$

其中 $V$、$E$、$F$ 是三角剖分的顶点、边、面数。

## 验证：球面

半径 $R$ 的球面：$K = 1/R^2$，$A = 4\pi R^2$。$\iint K\,dA = 4\pi R^2/(R^2) = 4\pi = 2\pi\cdot 2 = 2\pi\chi(S^2)$。正确。

## 验证：环面

环面有正曲率区域（外侧）和负曲率区域（内侧）。Gauss-Bonnet 要求它们精确抵消：$\iint K\,dA = 0 = 2\pi\chi(T^2)$。第 3 章直接计算已验证——无论 $R$ 和 $r$ 的具体比例如何，环面总曲率为零。

## 为什么这很了不起

等号左边 $\iint K\,dA$ 依赖 Riemann 度量。改变度量（拉伸、压缩、弯曲曲面），$K$ 在每个点都会变。

等号右边 $2\pi\chi(S)$ 是纯拓扑量。在任何光滑形变下不变。

Gauss-Bonnet 说：无论怎样改变度量，总曲率被拓扑锁定。一处的正曲率迫使别处出现负曲率来补偿（除非曲面是球面）。积分是刚性的。

## 应用

### 拓扑约束几何

紧致曲面若处处 $K > 0$，则 $\chi > 0$，只能是球面（或 $\mathbb{RP}^2$）。亏格 $\geq 1$ 的闭曲面不可能处处正 Gauss 曲率。

### 毛球定理

$S^2$ 上每个光滑向量场都有零点。由 Poincare-Hopf 定理（Gauss-Bonnet 对向量场的推广）：向量场零点指标之和等于 $\chi(S^2) = 2 \neq 0$。

直觉：你不可能把球面上的毛梳平而不留下旋涡。

### 几何约束拓扑

若曲面容许平坦度量（$K \equiv 0$），则 $\chi = 0$，只能是环面或 Klein 瓶。负曲率迫使 $\chi < 0$，即亏格 $\geq 2$。

## Chern-Gauss-Bonnet 定理

Gauss-Bonnet 推广到高维。对紧致有向偶数维 Riemann 流形 $M^{2n}$：

$$\int_M\text{Pf}(\Omega) = (2\pi)^n\chi(M)$$

其中 $\text{Pf}(\Omega)$ 是曲率形式的 Pfaffian。2 维时 Pfaffian 退化为 $K\,dA/(2\pi)$，回到经典定理。

这是 Atiyah-Singer 指标定理的基石——20 世纪最深刻的数学成果之一，连接了分析、几何与拓扑。

## 例：Mobius 带

Mobius 带不可定向，闭曲面的整体 Gauss-Bonnet 不直接适用。但带边界版本可以。Mobius 带的 $\chi = 0$（它形变回缩到圆）。对平坦 Mobius 带（纸条做的），$K = 0$，故

$$0 + \int_{\partial M}\kappa_g\,ds + 0 = 2\pi\chi(M) = 0.$$

Mobius 带的边界是单条闭曲线，其总测地曲率为零——它绕了两圈，贡献相互抵消。

## 总结

这个系列覆盖了经典路径：曲线、曲面、内蕴几何，以及几何通向拓扑的桥梁。自然的后续方向是 Riemann 几何本身（曲率张量、比较定理、Ricci 流）、Lie 群与对称空间，或支撑现代物理的规范理论与纤维丛。

Gauss-Bonnet 定理不是终点。它是一个原型——指标定理家族中的第一个成员，这些定理连接局部微分数据与整体拓扑不变量。局部分析约束整体结构——也许这是几何中最深刻的主题。
