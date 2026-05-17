---
title: "微分几何（十）：Riemann 几何——度量、联络与平行移动"
date: 2021-11-19 09:00:00
tags:
  - differential-geometry
  - riemannian-geometry
  - connections
  - mathematics
categories: Mathematics
series: differential-geometry
lang: zh
mathjax: true
description: "Riemann 度量让我们在任意光滑流形上度量长度、角度和体积——Levi-Civita 联络则提供了典范的平行移动与测地线概念。"
disableNunjucks: true
series_order: 10
series_total: 12
translationKey: "differential-geometry-10"
---

在本系列的前几篇文章中，我们研究的是配备了微分结构的光滑流形——坐标卡、切向量、微分形式、外微积分。这些都不需要"距离"的概念。我们可以讨论光滑函数及其导数，但无法谈论曲线的长度、两个切向量之间的夹角，或者一条路径是否"笔直"。要在经典意义上做几何——度量、比较、讨论曲率——我们需要额外的结构。

这个结构就是 **Riemann 度量**：在每个切空间上光滑变化的内积。有了它，任何光滑流形都成为一个几何空间。本文将介绍 Riemann 度量、Levi-Civita 联络（在曲线上微分向量场的典范方式）、平行移动和测地线。

---

## Riemann 度量：定义、存在性与例子

### 定义

光滑流形 $M$ 上的 **Riemann 度量**是对每个切空间光滑地赋予一个内积。形式地，它是一个光滑的、对称的、正定的 $(0,2)$ 型张量场 $g$：对每个 $p \in M$，

![球面上的平行移动展示和乐](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_fig10_parallel_transport.png)


$$g_p : T_pM \times T_pM \to \mathbb{R}$$

是双线性的、对称的（$g_p(X, Y) = g_p(Y, X)$）、正定的（当 $X \ne 0$ 时 $g_p(X, X) > 0$）。在局部坐标 $(x^1, \ldots, x^n)$ 中，度量由其分量确定：

$$g_{ij}(x) = g\left(\frac{\partial}{\partial x^i}, \frac{\partial}{\partial x^j}\right),$$

写作 $g = g_{ij}\, dx^i \otimes dx^j$（Einstein 求和约定）。矩阵 $(g_{ij})$ 在每个点上对称且正定。

配备 Riemann 度量的光滑流形称为 **Riemann 流形**，记为 $(M, g)$。

### 存在性

每个光滑流形都承认 Riemann 度量。证明用到单位分解：用坐标卡覆盖 $M$，在每个卡上定义任意内积（例如拉回 Euclid 度量），然后用单位分解粘合。正定性因其为凸性条件而得以保持。这个存在性结果令人安心——但度量的*选择*极为重要，因为它决定了全部几何。

### 例子

**Euclid 空间。** $\mathbb{R}^n$ 配 $g = \delta_{ij}\,dx^i \otimes dx^j$ 是平坦的 Euclid 度量。测地线是直线，曲率处处为零。

**标准球面。** $S^2 \subset \mathbb{R}^3$ 从环境 Euclid 度量继承一个度量（**诱导度量**）。在球坐标 $(\theta, \varphi)$ 中（$\theta \in (0, \pi)$ 为极角，$\varphi \in (0, 2\pi)$ 为方位角）：

$$g_{S^2} = d\theta \otimes d\theta + \sin^2\theta\, d\varphi \otimes d\varphi.$$

度量分量为 $g_{\theta\theta} = 1$，$g_{\varphi\varphi} = \sin^2\theta$，$g_{\theta\varphi} = 0$。$S^2$ 上的测地线是大圆。

**双曲空间。** 双曲平面 $\mathbb{H}^2 = \{(x,y) \in \mathbb{R}^2 : y > 0\}$ 的上半平面模型配度量

$$g_{\mathbb{H}^2} = \frac{dx \otimes dx + dy \otimes dy}{y^2}.$$

此度量具有常负 Gauss 曲率 $K = -1$。测地线是竖直线和圆心在 $x$ 轴上的半圆。靠近 $y = 0$ 时距离增长：这个"边界"距离内部无穷远。

**积度量。** 若 $(M_1, g_1)$ 和 $(M_2, g_2)$ 是 Riemann 流形，乘积 $M_1 \times M_2$ 上的积度量为 $g = g_1 + g_2$。例如，平坦环面 $T^2 = S^1 \times S^1$ 配每个圆因子上标准度量的积是一个曲率处处为零的平坦 Riemann 流形——尽管它在拓扑上是环面。

**共形变换。** 给定度量 $g$ 和正光滑函数 $e^{2f}$，共形缩放度量 $\tilde{g} = e^{2f}g$ 定义了保持角度但改变长度的新 Riemann 结构。共形几何在复分析和弦理论中都是核心课题。

---

## Levi-Civita 联络：存在性与唯一性

### 微分的困难

在 $\mathbb{R}^n$ 上，微分向量场很直接：将邻近点的向量平移到同一原点进行比较。但在一般流形上，不同点处的切空间是不同的向量空间——没有典范的方式将 $p$ 点和 $q$ 点的向量进行比较。**联络**正提供了这个缺失的结构。

### 联络（协变导数）

$M$ 上的一个**仿射联络**（或**协变导数**）是映射 $\nabla : \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M)$，记 $(X, Y) \mapsto \nabla_X Y$，满足：

1. **$X$ 的 $C^\infty(M)$-线性：** $\nabla_{fX + gZ} Y = f\nabla_X Y + g\nabla_Z Y$。
2. **$Y$ 的 $\mathbb{R}$-线性：** $\nabla_X(Y + Z) = \nabla_X Y + \nabla_X Z$。
3. **$Y$ 的 Leibniz 律：** $\nabla_X(fY) = (Xf)Y + f\nabla_X Y$。

在局部坐标中，联络由 **Christoffel 符号** $\Gamma^k_{ij}$ 决定：

$$\nabla_{\partial_i} \partial_j = \Gamma^k_{ij}\, \partial_k.$$

任何流形上存在无穷多个联络。Riemann 度量选出了一个唯一的、首选的联络。

### Riemann 几何基本定理

> **定理（Levi-Civita）。** 任何 Riemann 流形 $(M, g)$ 上存在唯一的联络 $\nabla$，同时满足：
>
> 1. **度量相容：** $Xg(Y,Z) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$。
> 2. **无挠：** $\nabla_X Y - \nabla_Y X = [X, Y]$。

这个唯一的联络称为 **Levi-Civita 联络**。

度量相容意味着平行移动保持内积（长度和角度不变）。无挠意味着联络没有"扭曲"——平行四边形能够闭合。

### Christoffel 符号公式

Levi-Civita 联络的 Christoffel 符号由度量显式确定：

$$\Gamma^k_{ij} = \frac{1}{2} g^{k\ell} \left(\frac{\partial g_{j\ell}}{\partial x^i} + \frac{\partial g_{i\ell}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^\ell}\right),$$

其中 $g^{k\ell}$ 是逆度量（$g^{k\ell}g_{\ell m} = \delta^k_m$）。

**例：球面 $S^2$。** 坐标 $(\theta, \varphi)$，度量 $g = d\theta^2 + \sin^2\theta\, d\varphi^2$，非零 Christoffel 符号为

$$\Gamma^\theta_{\varphi\varphi} = -\sin\theta\cos\theta, \qquad \Gamma^\varphi_{\theta\varphi} = \Gamma^\varphi_{\varphi\theta} = \cot\theta.$$

其余所有符号为零。这些符号编码了坐标基向量在球面上移动时如何变化。

---

## 沿曲线的平行移动

### 定义

设 $\gamma : [0,1] \to M$ 是一条光滑曲线。沿 $\gamma$ 的向量场 $V(t)$ 称为**平行的**（关于 $\nabla$），如果

$$\nabla_{\dot{\gamma}(t)} V(t) = 0 \quad \forall\, t \in [0,1].$$

在坐标中，若 $\gamma(t) = (x^1(t), \ldots, x^n(t))$，$V(t) = V^k(t)\,\partial_k$，平行移动方程变为

$$\frac{dV^k}{dt} + \Gamma^k_{ij}(\gamma(t))\, \dot{x}^i(t)\, V^j(t) = 0.$$

这是分量 $V^k(t)$ 的一阶线性常微分方程组。由标准 ODE 理论，给定任何初始向量 $V(0) \in T_{\gamma(0)}M$，存在唯一的沿 $\gamma$ 的平行向量场。映射 $V(0) \mapsto V(1)$ 称为沿 $\gamma$ 从 $\gamma(0)$ 到 $\gamma(1)$ 的**平行移动**。

### 性质

- 平行移动是 $T_{\gamma(0)}M$ 到 $T_{\gamma(1)}M$ 的**线性同构**。
- 因为 Levi-Civita 联络度量相容，平行移动是**等距映射**：保持内积，因而保持长度和角度。
- 平行移动**依赖于路径**——沿两条不同路径从 $p$ 到 $q$ 移动同一个向量，结果一般不同。

### 例：球面上的平行移动

这是建立几何直觉的经典例子。在 $S^2$ 的北极放一个指向东方的切向量。沿子午线将它平行移动到赤道，然后沿赤道移动角度 $\alpha$，再沿子午线回到北极。向量回来时，相对于原方向旋转了角度 $\alpha$。

对于一个有三个直角的三角形（$\alpha = \pi/2$），向量旋转了 $90°$。这个旋转直接体现了曲率——在平坦表面上，沿任何闭合回路的平行移动都将向量送回原方向。旋转的量等于围住区域上 Gauss 曲率的积分（这就是 Gauss-Bonnet 定理的体现）。

### 路径依赖性等于曲率

平行移动的路径依赖性恰由**曲率张量**度量。对于由向量 $X$ 和 $Y$ 张成的无穷小平行四边形，绕回路平行移动累积的和乐群（holonomy，即旋转）为

$$R(X, Y)V = \nabla_X \nabla_Y V - \nabla_Y \nabla_X V - \nabla_{[X,Y]} V.$$

若 $R = 0$ 处处成立，则平行移动与路径无关，流形是**平坦的**。我们将在第十一篇中全面展开这一主题。

---

## Riemann 流形上的测地线

### 通过联络定义

曲线 $\gamma : [a,b] \to M$ 是**测地线**，如果其切向量沿自身平行：

$$\nabla_{\dot{\gamma}} \dot{\gamma} = 0.$$

在坐标中，测地线方程是二阶常微分方程组：

$$\ddot{x}^k + \Gamma^k_{ij}\, \dot{x}^i \dot{x}^j = 0.$$

测地线是"直线"在流形上的推广——它们是加速度为零的曲线（加速度由联络度量）。在配平坦度量的 $\mathbb{R}^n$ 上，$\Gamma^k_{ij} = 0$，测地线就是通常的直线。在球面上，测地线是大圆。

### 指数映射

在点 $p \in M$ 和切向量 $v \in T_pM$，满足 $\gamma_v(0) = p$、$\dot{\gamma}_v(0) = v$ 的测地线 $\gamma_v(t)$ 在小 $t$ 时由 ODE 理论保证存在。**指数映射**定义为

$$\exp_p(v) = \gamma_v(1),$$

前提是测地线存在到 $t = 1$。对小的 $v$，$\exp_p$ 是从 $T_pM$ 中 $0$ 的邻域到 $M$ 中 $p$ 的邻域的微分同胚。这给出**法坐标**（也称测地坐标或 Riemann 法坐标），在其中 $\Gamma^k_{ij}(p) = 0$——联络在中心点"看起来像平坦的"。

### 测地线作为长度极小化者

测地线局部极小化长度泛函 $L(\gamma) = \int_a^b \|\dot{\gamma}(t)\|\,dt$，其中 $\|\dot{\gamma}\| = \sqrt{g(\dot{\gamma}, \dot{\gamma})}$。更精确地说，测地线是能量泛函 $E(\gamma) = \frac{1}{2}\int_a^b g(\dot{\gamma}, \dot{\gamma})\,dt$ 的临界点，测地线方程是 $E$ 的 Euler-Lagrange 方程。

短测地线是长度极小化的。但长测地线可能不是全局最短的：在球面上，大于半圆的大圆弧是测地线但不是端点间的最短路径。

---

## Riemann 距离函数与完备性

### 距离

Riemann 度量诱导 $M$ 上的距离函数：

$$d(p, q) = \inf_\gamma \int_0^1 \|\dot{\gamma}(t)\|\, dt,$$

下确界取遍所有从 $p$ 到 $q$ 的分段光滑曲线 $\gamma$。这满足度量空间公理（正性、对称性、三角不等式），且诱导与流形拓扑相同的拓扑。

### 完备性

Riemann 流形是**测地完备的**，如果每条测地线都可以延伸到全部时间——等价地，$\exp_p(v)$ 对所有 $v \in T_pM$、所有 $p$ 都有定义。$\mathbb{R}^2$ 中的开圆盘配平坦度量不是完备的（测地线在有限时间内到达边界），而 $\mathbb{R}^n$ 和 $S^n$ 是完备的。

### Hopf-Rinow 定理

> **定理（Hopf-Rinow）。** 对连通 Riemann 流形 $(M, g)$，以下条件等价：
>
> 1. $(M, d)$ 是完备度量空间（每个 Cauchy 列收敛）。
> 2. $M$ 是测地完备的。
> 3. 对某个 $p \in M$，$\exp_p$ 在全部 $T_pM$ 上有定义。
> 4. $M$ 的每个有界闭子集是紧的。
>
> 此外，若以上任一条件成立，则 $M$ 中任意两点可由长度极小化测地线连接。

Hopf-Rinow 定理是 Heine-Borel 定理的 Riemann 类比。它保证在完备流形上最短路径总是存在的。紧致流形自动完备。

**例。** 双曲空间 $\mathbb{H}^n$ 是完备的：测地线在两个方向上都可以延伸到无穷长（尽管在 Poincare 模型中它们看起来有限长，但双曲度量在边界附近拉伸距离）。

---

## 等距与 Killing 场

### 等距

Riemann 流形 $(M, g)$ 和 $(N, h)$ 之间的**等距**是满足 $\phi^*h = g$ 的微分同胚 $\phi : M \to N$。当 $M = N$ 时，$\phi$ 称为 $(M, g)$ 的等距。所有等距 $M \to M$ 构成群 $\text{Isom}(M, g)$。

等距保持一切几何量：距离、角度、测地线、曲率。Myers-Steenrod 定理指出 $\text{Isom}(M, g)$ 是 Lie 群。

**例子。**
- $\text{Isom}(\mathbb{R}^n, g_{\text{flat}}) = \mathbb{R}^n \rtimes O(n)$：平移和正交变换。
- $\text{Isom}(S^n, g_{\text{round}}) = O(n+1)$：全正交群。
- $\text{Isom}(\mathbb{H}^2, g_{\text{hyp}}) = \text{PSL}(2, \mathbb{R})$：保持上半平面的 Mobius 变换。

### Killing 向量场

$(M, g)$ 上的 **Killing 向量场** $K$ 是无穷小等距：其流保持度量。形式地，$g$ 沿 $K$ 的 Lie 导数为零：

$$\mathcal{L}_K g = 0.$$

在坐标中等价于 **Killing 方程**：

$$\nabla_i K_j + \nabla_j K_i = 0.$$

Killing 场在 Lie 括号下构成 Lie 代数，即等距群的 Lie 代数。

**例。** 在 $S^2$ 上，绕三个正交轴的旋转给出三个线性无关的 Killing 场——Lie 代数 $\mathfrak{so}(3)$。在 $\mathbb{R}^n$ 上，平移和无穷小旋转给出 $\frac{n(n+1)}{2}$ 个 Killing 场。

具有最大数目 $\frac{n(n+1)}{2}$ 个独立 Killing 场的 Riemann 流形称为**极大对称空间**，它们恰好是常截面曲率空间：$\mathbb{R}^n$、$S^n$ 和 $\mathbb{H}^n$。

---

## 下一步

我们现在拥有了做几何的工具：度量用来度量，联络用来微分，平行移动用来比较。自然的下一个问题是：**Riemann 流形有多弯曲？** 在下一篇文章中，我们将引入 Riemann 曲率张量及其缩并——Ricci 曲率和标量曲率——它们以各种形式量化曲率。这些正是出现在 Einstein 场方程和 Riemann 几何诸大定理中的对象。

---

*本文是 [微分几何](/zh/series/differential-geometry/) 系列的第 10 篇（共 12 篇）。*

*上一篇：[第 9 篇 —— 积分与 Stokes 定理](/zh/differential-geometry/09-流形上的积分与Stokes定理/)*

*下一篇：[第 11 篇 —— 流形上的曲率](/zh/differential-geometry/11-流形上的曲率/)*
