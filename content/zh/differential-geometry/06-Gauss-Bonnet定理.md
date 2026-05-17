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

Gauss-Bonnet 定理是数学中最美的结果之一。它说：把 Gauss 曲率在整个闭曲面上积分，得到一个拓扑不变量——只依赖曲面"橡皮膜意义上的形状"的数，与具体几何无关。随便怎么拉伸、弯曲、压缩曲面，总曲率不变。这是局部微分几何通向整体拓扑的桥梁。

## 局部 Gauss-Bonnet 定理

设 $R$ 是有向曲面 $S$ 上的单连通区域，边界 $\partial R$ 是简单闭合分段光滑曲线，由光滑弧段 $C_1,\ldots,C_k$ 组成，顶点处外角为 $\theta_1,\ldots,\theta_k$。则：

$$\iint_R K\,dA + \int_{\partial R}\kappa_g\,ds + \sum_{i=1}^k\theta_i = 2\pi$$

![Gauss-Bonnet 定理：曲率积分等于 Euler 示性数的 2pi 倍](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/06-gauss-bonnet/dg_fig6_gauss_bonnet.png)


其中 $\kappa_g$ 是边界的测地曲率（曲线左转为正），边界沿逆时针方向（区域在左边）绕行。

公式统一了三类"转动"：Gauss 曲率 $K$ 度量曲面本身弯曲；测地曲率 $\kappa_g$ 度量边界在曲面上弯曲；外角 $\theta_i$ 是顶点处的离散转弯。三者之和恒为 $2\pi$——转满一圈。

**例（测地三角形）。** 三边是测地线（$\kappa_g = 0$），外角 $\theta_i = \pi-\alpha_i$（$\alpha_i$ 为内角）。局部 Gauss-Bonnet 给出：

$$\iint_T K\,dA + 0 + \sum(\pi-\alpha_i) = 2\pi$$

化简：

$$\alpha_1+\alpha_2+\alpha_3 = \pi + \iint_T K\,dA.$$

球面上（$K > 0$）：内角和大于 $\pi$。三角形"鼓起来"——边向外弓，内部角度因而增大。

双曲平面上（$K < 0$）：内角和小于 $\pi$。三角形面积有上界 $\pi/|K|$，无论边长多大。

平坦面上（$K = 0$）：内角和恰好 $\pi$。Euclid 几何。

**例（测地圆盘）。** 半径 $r$ 的测地圆盘（到中心测地距离 $\leq r$ 的点集），边界测地曲率 $\kappa_g\approx 1/r$（$r$ 小时），无顶点。局部 Gauss-Bonnet 给出 $K(p)\approx 3(2\pi r - L)/(\pi r^3)$（$L$ 是圆盘周长）。Gauss 曲率可以通过测量小圆周长偏离 $2\pi r$ 的程度来探测。

## 整体 Gauss-Bonnet 定理

**定理。** 设 $S$ 是紧致有向无边界曲面。则：

$$\iint_S K\,dA = 2\pi\chi(S)$$

其中 $\chi(S)$ 是 $S$ 的 **Euler 示性数**。

Euler 示性数由任意三角剖分定义：$\chi = V-E+F$（顶点减边加面）。有向曲面上 $\chi = 2-2g$（$g$ 为亏格/把手数）：

- 球面（$g=0$）：$\chi=2$，总曲率 $= 4\pi$。
- 环面（$g=1$）：$\chi=0$，总曲率 $= 0$。
- 双环面（$g=2$）：$\chi=-2$，总曲率 $= -4\pi$。
- 亏格 $g$ 曲面：$\chi = 2-2g$，总曲率 $= 2\pi(2-2g)$。

*证明思路。* 对 $S$ 做测地三角剖分。对每个三角形 $T_j$，局部 Gauss-Bonnet 给出 $\iint_{T_j}K\,dA = (\alpha_1^j+\alpha_2^j+\alpha_3^j)-\pi$。对所有 $F$ 个面求和。右边：每个内部顶点贡献 $2\pi$（绕一圈），总共 $2\pi V$ 的顶点角度，减去 $\pi F$。共享边的测地曲率贡献相消（相邻三角形沿反方向走过）。仔细簿记得：

$$\iint_S K\,dA = 2\pi V - \pi F + \text{（边修正）} = 2\pi(V-E+F) = 2\pi\chi(S).$$

边修正：每条边被 2 个三角形共享，且 $3F = 2E$，代入即得。

## 验证：球面

半径 $R$ 的球面：$K = 1/R^2$，面积 $= 4\pi R^2$。总曲率：

$$\iint_{S^2}K\,dA = \frac{1}{R^2}\cdot 4\pi R^2 = 4\pi = 2\pi\cdot 2 = 2\pi\chi(S^2).$$

$\chi(S^2) = 2$。正确。

## 验证：环面

环面 Gauss 曲率 $K = \cos u/(r(R+r\cos u))$。总曲率：

$$\iint_{T^2}K\,dA = \int_0^{2\pi}\int_0^{2\pi}\frac{\cos u}{r(R+r\cos u)}\cdot r(R+r\cos u)\,du\,dv = \int_0^{2\pi}\int_0^{2\pi}\cos u\,du\,dv = 0.$$

外半部正曲率与内半部负曲率精确抵消。Gauss-Bonnet 要求如此，因为 $\chi(T^2) = 0$，与 $R$、$r$ 的具体值无关。

## 为什么这很了不起

等号左边 $\iint K\,dA$ 是几何量，依赖 Riemann 度量。形变曲面时 $K$ 在每点都变。

等号右边 $2\pi\chi(S)$ 是拓扑不变量。任何连续形变都不改变它。你可以把球面膨胀到任意大小、压成椭球、捏出凹坑——只要不撕裂或粘贴把手，$\chi$ 恒为 2，总曲率恒为 $4\pi$。

Gauss-Bonnet 说：无论怎样在曲面上重新分配曲率，积分被拓扑锁死。一处推走正曲率，别处必然冒出来以维持全局约束。

## 应用与推论

### 拓扑约束几何

紧致曲面若处处 $K > 0$，则 $\chi > 0$，只能同胚于球面（或 $\mathbb{RP}^2$）。亏格 $\geq 1$ 的闭有向曲面不可能处处正 Gauss 曲率。

反过来，容许平坦度量（$K\equiv 0$）的紧致曲面必须 $\chi = 0$——只能是环面或 Klein 瓶。

### 毛球定理

$S^2$ 上每个光滑切向量场至少有一个零点。由 Poincare-Hopf 指标定理（Gauss-Bonnet 对向量场的推广）：向量场零点指标之和等于 $\chi(S)$。$\chi(S^2) = 2\neq 0$，故零点必须存在。

物理直觉：你不可能把椰子上的毛梳平而不留下旋涡或光点。

### 多面体的角度亏损（Descartes 定理）

凸多面体（"多面体曲面"）上 Gauss 曲率集中在顶点，以离散角度亏损的形式出现：$K_v = 2\pi - \sum\theta_i$（$\theta_i$ 是汇聚到顶点 $v$ 的面角）。Gauss-Bonnet 变成 Descartes 定理：$\sum_v K_v = 4\pi$。任何凸多面体（正方体、二十面体、任何形状），总角度亏损为 $4\pi$。

## 例：Mobius 带（有边界曲面）

Mobius 带不可定向且有边界，闭曲面的整体定理不直接适用。但带边界版本可以。对紧致有边界曲面 $M$：

$$\iint_M K\,dA + \int_{\partial M}\kappa_g\,ds = 2\pi\chi(M).$$

平坦 Mobius 带（纸条做的）$K = 0$，$\chi = 0$，故 $\int_{\partial M}\kappa_g\,ds = 0$。边界是单条闭曲线，绕了两圈，正负测地曲率贡献相消。

## 例：双环面与双曲几何

亏格 2 曲面 $\chi = -2$，总曲率 $\iint K\,dA = -4\pi$。它不可能处处 $K\geq 0$。事实上由单值化定理，每个亏格 $\geq 2$ 的紧致曲面都容许常负曲率 $K = -1$ 的度量，面积为 $4\pi(g-1)$。双环面取 $K = -1$ 时面积为 $4\pi$。

这连接到双曲几何：亏格 $g$ 曲面（配常负曲率度量）的万有覆盖是双曲平面 $\mathbb{H}^2$。

## Chern-Gauss-Bonnet 定理

Gauss-Bonnet 推广到高偶数维。紧致有向偶数维 Riemann 流形 $M^{2n}$：

$$\int_M\text{Pf}(\Omega) = (2\pi)^n\chi(M)$$

$\text{Pf}(\Omega)$ 是曲率 2-形式的 Pfaffian。$n=1$（2维）时退化为 $K\,dA/(2\pi)$，回到经典定理。4 维时被积函数涉及 Riemann 张量及其缩并。

这是 Atiyah-Singer 指标定理的基石——20 世纪最深刻的数学结果之一，连接分析（椭圆微分算子）、几何（曲率）与拓扑（示性类）。

## 结语

这个系列覆盖了经典的微分几何路径：曲线、曲面、内蕴几何，以及几何通向拓扑的桥梁。自然的后续方向包括：

- **Riemann 几何**：曲率张量、Jacobi 场、比较定理（Rauch、Toponogov）、证明了 Poincare 猜想的 Ricci 流。
- **Lie 群与对称空间**：带代数结构的流形，几何与代数交织。
- **规范理论与纤维丛**：现代粒子物理的数学框架，联络推广为非 Abel 规范场。

Gauss-Bonnet 定理不是终点。它是一个原型——指标定理家族中最简单的第一个成员。局部微分数据约束整体拓扑结构——也许这是几何中最深刻的主题。

---

*本文是 [微分几何](/zh/series/differential-geometry/) 系列的第 6 篇，共 6 篇。
上一篇: [第 5 篇 — 光滑流形、切丛与联络](/zh/differential-geometry/05-流形与联络/)*
