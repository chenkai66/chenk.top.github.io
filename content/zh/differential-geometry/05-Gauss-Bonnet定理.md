---
title: "微分几何（五）：Gauss-Bonnet 定理——几何遇上拓扑"
date: 2021-11-09 09:00:00
tags:
  - differential-geometry
  - gauss-bonnet
  - topology
  - mathematics
categories: Mathematics
series: differential-geometry
lang: zh
mathjax: true
description: "Gauss-Bonnet 定理将总 Gauss 曲率与 Euler 示性数相连——一座令人震撼的桥梁，沟通了局部微分几何与全局拓扑。"
disableNunjucks: true
series_order: 5
series_total: 12
translationKey: "differential-geometry-5"
---

到目前为止，我们建立的一切——第一和第二基本形式、Gauss 曲率、测地线、绝妙定理——都是**局部的**：在一个点或沿一条曲线定义的性质，通过导数来计算。Gauss-Bonnet 定理打破了局部与全局之间的壁垒。它宣称：一个闭曲面的总 Gauss 曲率——将 $K$ 在整个曲面上积分所得——等于 $2\pi$ 乘以 **Euler 示性数** $\chi$，而 $\chi$ 仅取决于曲面的拓扑（"橡皮膜"意义下的形状），与几何无关。

这是非凡的。无论你如何弯曲、拉伸或变形一个曲面（只要不撕裂或粘合），积分 $\int K\,dA$ 始终不变。曲率，一个典型的几何量，竟然受到拓扑的约束。

---

## 先说结论：曲率决定拓扑

让我们先把定理摆出来，明确目标所在。

**全局 Gauss-Bonnet 定理.** 设 $M$ 是紧致定向无边曲面，则

$$\iint_M K\,dA = 2\pi\chi(M),$$

其中 $K$ 是 Gauss 曲率，$dA$ 是面积元，$\chi(M)$ 是 $M$ 的 Euler 示性数。

对球面：$K = 1/R^2$ 处处成立，$\text{Area} = 4\pi R^2$，故 $\int K\,dA = 4\pi = 2\pi \cdot 2$，得 $\chi(S^2) = 2$。对环面：$\chi(T^2) = 0$，故总曲率必为零——外侧的正曲率恰好抵消内侧的负曲率。对亏格 $g$ 曲面：$\chi = 2-2g$，故 $\int K\,dA = 2\pi(2-2g)$。

我们将逐步建立这一结果，从最简单的情形开始。

---


![Gauss-Bonnet 定理：曲率积分等于 Euler 示性数的 2pi 倍](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/05-gauss-bonnet/dg_fig5_gauss_bonnet.png)

## 局部 Gauss-Bonnet 定理：测地三角形

局部版本将测地三角形的角度盈余与 Gauss 曲率在三角形上的积分联系起来。

**定理（测地三角形的局部 Gauss-Bonnet）.** 设 $T$ 是正则曲面 $S$ 上的测地三角形——由三段测地线弧围成的区域——内角为 $\alpha_1, \alpha_2, \alpha_3$。则

$$\iint_T K\,dA = (\alpha_1 + \alpha_2 + \alpha_3) - \pi.$$

### 完整证明

我们使用分段光滑边界区域的 Gauss-Bonnet 公式。设 $R$ 是曲面 $S$ 上一个单连通区域，其边界 $\partial R$ 由光滑弧 $C_1, \ldots, C_n$ 组成（正向定向），在顶点处的外角为 $\theta_1, \ldots, \theta_n$。**局部 Gauss-Bonnet 公式**为：

$$\iint_R K\,dA + \sum_{i=1}^{n} \int_{C_i} \kappa_g\,ds + \sum_{i=1}^{n} \theta_i = 2\pi,$$

其中 $\kappa_g$ 是各光滑弧的测地曲率，$\theta_i$ 是第 $i$ 个顶点处的外角。

**一般公式的证明.** 选取区域 $R$ 的正交参数化 $(u,v)$（$F = 0$）。定义角度函数 $\varphi(t)$：边界曲线切向量 $\gamma'(t)$ 与坐标方向 $\mathbf{r}_u/|\mathbf{r}_u|$ 的夹角。通过仔细计算测地曲率的定义，可以证明沿光滑弧有：

$$\kappa_g = \frac{d\varphi}{ds} + \frac{1}{2\sqrt{EG}}\left(E_v \frac{du}{ds} - G_u \frac{dv}{ds}\right).$$

第二项与联络形式有关。定义 1-形式

$$\omega = \frac{1}{2\sqrt{EG}}\left(E_v\,du - G_u\,dv\right).$$

则 $\kappa_g\,ds = d\varphi + \omega$。沿整个边界 $\partial R$ 积分：

$$\sum_i \int_{C_i} \kappa_g\,ds + \sum_i \theta_i = \oint_{\partial R} d\varphi + \sum_i \theta_i + \oint_{\partial R} \omega.$$

现在，$\oint_{\partial R} d\varphi + \sum_i \theta_i = 2\pi$——这是**切线旋转定理**（Umlaufsatz）：切向量沿简单闭曲线的总旋转角（包括在拐角处的跳跃）等于 $2\pi$。这是平面曲线的拓扑事实（通过局部坐标卡应用）。

因此：

$$\sum_i \int_{C_i} \kappa_g\,ds + \sum_i \theta_i = 2\pi + \oint_{\partial R} \omega.$$

由 Green 定理（适用因为 $R$ 单连通且参数化正则）：

$$\oint_{\partial R} \omega = \iint_R d\omega.$$

直接计算 $\omega$ 的外微分：

$$d\omega = -K\sqrt{EG}\,du \wedge dv = -K\,dA.$$

这是关键步骤——Gauss 曲率在此出现：它是联络形式的曲率。由 Gauss 公式：

$$K = -\frac{1}{2\sqrt{EG}}\left[\frac{\partial}{\partial u}\left(\frac{G_u}{\sqrt{EG}}\right) + \frac{\partial}{\partial v}\left(\frac{E_v}{\sqrt{EG}}\right)\right].$$

将 $d\omega = -K\,dA$ 代回：

$$\sum_i \int_{C_i} \kappa_g\,ds + \sum_i \theta_i = 2\pi - \iint_R K\,dA.$$

移项得到局部 Gauss-Bonnet 公式：

$$\iint_R K\,dA + \sum_i \int_{C_i} \kappa_g\,ds + \sum_i \theta_i = 2\pi. \quad \square$$

**特化到测地三角形.** 对顶点为 $P_1, P_2, P_3$ 的测地三角形 $T$：

- 每条边是测地线，故沿每条弧 $\kappa_g = 0$。
- 第 $i$ 个顶点处的外角为 $\theta_i = \pi - \alpha_i$，其中 $\alpha_i$ 是内角。

代入公式：

$$\iint_T K\,dA + 0 + \sum_{i=1}^{3}(\pi - \alpha_i) = 2\pi,$$

$$\iint_T K\,dA + 3\pi - (\alpha_1 + \alpha_2 + \alpha_3) = 2\pi,$$

$$\iint_T K\,dA = (\alpha_1 + \alpha_2 + \alpha_3) - \pi. \quad \square$$

### 在球面上的验证

球面 $S^2(R)$ 上 $K = 1/R^2$ 处处成立。内角为 $\alpha_1, \alpha_2, \alpha_3$ 的测地三角形面积为

$$\text{Area}(T) = R^2(\alpha_1 + \alpha_2 + \alpha_3 - \pi),$$

于是

$$\iint_T K\,dA = \frac{1}{R^2} \cdot R^2(\alpha_1 + \alpha_2 + \alpha_3 - \pi) = (\alpha_1 + \alpha_2 + \alpha_3) - \pi. \quad \checkmark$$

作为具体例子，考虑由三条互相垂直的大圆弧构成的球面三角形（球面的八分之一）。每个角为 $\pi/2$，角度盈余为 $3\pi/2 - \pi = \pi/2$，面积为 $R^2 \cdot \pi/2 = 4\pi R^2/8$，确实是球面面积的八分之一。

---

## 局部 Gauss-Bonnet 定理：测地多边形

推广到 $n$ 边测地多边形是直接的。

**定理.** 设 $P$ 是具有 $n$ 条边和内角 $\alpha_1, \ldots, \alpha_n$ 的测地多边形。则

$$\iint_P K\,dA = \left(\sum_{i=1}^{n} \alpha_i\right) - (n-2)\pi.$$

**证明.** 将局部 Gauss-Bonnet 公式应用于 $\kappa_g = 0$（每条测地边）和外角 $\theta_i = \pi - \alpha_i$：

$$\iint_P K\,dA + \sum_{i=1}^{n}(\pi - \alpha_i) = 2\pi,$$

$$\iint_P K\,dA = 2\pi - n\pi + \sum \alpha_i = \sum \alpha_i - (n-2)\pi. \quad \square$$

对 $n = 3$，恢复三角形公式。对 $n = 4$（常曲率 $K$ 曲面上的测地四边形）：

$$\text{内角和} = 2\pi + K \cdot \text{面积}.$$

在欧氏平面（$K = 0$）上，内角和为 $2\pi$，与四边形内角和 $360°$ 的熟知结果一致。

---

## 全局 Gauss-Bonnet 定理

现在我们从局部跨越到全局。

**定理（全局 Gauss-Bonnet）.** 设 $M$ 是紧致定向无边曲面。则

$$\iint_M K\,dA = 2\pi\chi(M),$$

其中 $\chi(M) = V - E + F$ 是 $M$ 的任意三角剖分的 Euler 示性数（$V$ 个顶点、$E$ 条边、$F$ 个面）。

### 通过三角剖分的证明

选取 $M$ 的一个测地三角剖分：将 $M$ 分解为测地三角形 $T_1, \ldots, T_F$，有 $V$ 个顶点和 $E$ 条边。（对任何紧致正则曲面，这样的三角剖分存在，这是微分拓扑学的一个定理。）

对每个三角形 $T_j$ 应用局部 Gauss-Bonnet 公式：

$$\iint_{T_j} K\,dA = (\alpha_1^{(j)} + \alpha_2^{(j)} + \alpha_3^{(j)}) - \pi.$$

对所有 $F$ 个面求和：

$$\iint_M K\,dA = \sum_{j=1}^{F}\left(\sum_{k=1}^3 \alpha_k^{(j)}\right) - F\pi.$$

关键是求 $\sum_{j,k} \alpha_k^{(j)}$——所有三角形中所有角的总和。

**断言：** $\displaystyle\sum_{\text{所有角}} \alpha = 2\pi V.$

在每个内部顶点处，围绕它的角度之和为 $2\pi$（因为三角形铺满了整个邻域）。由于 $M$ 无边界，每个顶点都是内部顶点，故总角度和为 $2\pi V$。

因此：

$$\iint_M K\,dA = 2\pi V - F\pi.$$

现在利用三角剖分的 Euler 关系。每个三角形有 3 条边，每条边恰被 2 个三角形共享（因为 $M$ 是闭曲面），所以 $3F = 2E$，即 $E = 3F/2$。Euler 示性数为

$$\chi = V - E + F = V - \frac{3F}{2} + F = V - \frac{F}{2},$$

故 $F = 2V - 2\chi$，代入得

$$\iint_M K\,dA = 2\pi V - \pi(2V - 2\chi) = 2\pi V - 2\pi V + 2\pi\chi = 2\pi\chi(M). \quad \square$$

这个证明的精妙之处在于其直接性：局部的角度盈余公式逐三角形叠加后，因为顶点处的角度和由拓扑决定，最终折叠成了一个全局陈述。

---

## Euler 示性数 $\chi$ 与紧致曲面的分类

Euler 示性数 $\chi(M)$ 是一个**拓扑不变量**——它仅取决于 $M$ 的同胚类型，与三角剖分的选取无关。这可以通过证明任意两个三角剖分有公共加细来建立。

### 标准曲面的 $\chi$ 值

| 曲面 | 亏格 $g$ | $\chi = 2-2g$ | 总曲率 $\int K\,dA$ |
|:-----|:---------|:--------------|:-------------------|
| 球面 $S^2$ | 0 | 2 | $4\pi$ |
| 环面 $T^2$ | 1 | 0 | $0$ |
| 双环面 | 2 | $-2$ | $-4\pi$ |
| 亏格-$g$ 曲面 | $g$ | $2-2g$ | $2\pi(2-2g)$ |

### 分类定理

**定理.** 每个紧致连通定向无边曲面同胚于一个附加了 $g$ 个"把手"的球面，$g \geq 0$。整数 $g$（**亏格**）是完整的拓扑不变量：两个这样的曲面同胚当且仅当它们有相同的亏格。

Euler 示性数 $\chi = 2-2g$ 提供了等价的不变量。结合 Gauss-Bonnet，这给出了一个从几何读取拓扑的方法：

$$g = 1 - \frac{1}{4\pi}\iint_M K\,dA.$$

如果有人递给你一个曲面，你能够在其上处处计算 Gauss 曲率，那么通过积分就可以确定它的亏格——根本无需"看到"把手。

### 不可定向曲面

对不可定向曲面（如 Klein 瓶或实射影平面），Gauss-Bonnet 定理经适当修改后仍然成立。实射影平面 $\mathbb{R}P^2$ 的 $\chi = 1$，故对其上任何度量都有 $\int K\,dA = 2\pi$。Klein 瓶的 $\chi = 0$（与环面相同）。

---

## 应用

### 闭曲面的总曲率

Gauss-Bonnet 定理对闭曲面的几何施加了直接约束：

1. **处处正曲率 $\Rightarrow$ 球面.** 若紧致定向曲面 $M$ 的每一点处 $K > 0$，则 $\int K\,dA > 0$，故 $\chi(M) > 0$。对可定向曲面，$\chi > 0$ 的唯一可能是 $\chi = 2$，因此 $M$ 同胚于球面。

2. **处处零曲率 $\Rightarrow$ 环面.** 若 $K \equiv 0$ 且 $M$ 紧致定向，则 $\chi(M) = 0$，故 $M$ 同胚于环面。（平坦环面确实存在——标准的方环面 $\mathbb{R}^2/\mathbb{Z}^2$ 就是一个例子，尽管它不能等距嵌入 $\mathbb{R}^3$。）

3. **处处负曲率 $\Rightarrow$ 亏格 $\geq 2$.** 若 $K < 0$ 处处成立，则 $\chi(M) < 0$，故 $g \geq 2$。

### 为什么不能把球面压平

绝妙定理告诉我们等距映射保持 $K$。球面的 $K = 1/R^2 > 0$；平面的 $K = 0$。因为 $K$ 是等距不变量，球面的任何一块都不能等距映射到平面上。这就是为什么地球的每幅平面地图都必然扭曲距离。

但 Gauss-Bonnet 给出了更强的陈述：即使我们允许 $K$ 的扭曲（即不要求等距，只要求光滑映射），总曲率仍被拓扑锁定。任何度量下的球面都满足 $\int K\,dA = 4\pi$。你可以重新分配曲率（集中在极点、均匀分布、或使某些区域负曲率），但总量永远是 $4\pi$。

### 毛球定理（联系）

一个相关的拓扑结果是**毛球定理**：$S^2$ 上不存在连续的处处不为零的切向量场。虽然这通常通过代数拓扑来证明，但它与 Gauss-Bonnet 通过 **Poincare-Hopf 指标定理**紧密相连：$M$ 上任何切向量场零点的指标之和等于 $\chi(M)$。对 $S^2$，$\chi = 2 \neq 0$，因此每个向量场都必须有零点。

### 曲率与面积估计

Gauss-Bonnet 提供了定量的界。对亏格 $g$ 的紧致曲面，若 $K \leq K_0 < 0$：

$$\text{Area}(M) \geq \frac{2\pi|2-2g|}{|K_0|} = \frac{4\pi(g-1)}{|K_0|}.$$

这给出了面积关于拓扑和曲率界的下界——一个在欧氏几何中没有类比的结果。

### Descartes 角度亏损公式

作为一个离散应用，考虑一个凸多面体，有 $V$ 个顶点、$E$ 条边和 $F$ 个面。在每个顶点 $v_i$ 处定义**角度亏损** $\delta_i = 2\pi - \sum(\text{该顶点处的面角})$。Descartes 定理说：

$$\sum_{i=1}^{V} \delta_i = 4\pi.$$

这是 Gauss-Bonnet 的离散版本：角度亏损扮演了集中 Gauss 曲率的角色，而 $4\pi = 2\pi\chi(S^2)$。光滑凸曲面可以被凸多面体逼近，角度亏损收敛到 $K$ 的积分。

### 带边界的 Gauss-Bonnet

对有光滑边界 $\partial M$ 的紧致曲面 $M$：

$$\iint_M K\,dA + \int_{\partial M} \kappa_g\,ds = 2\pi\chi(M).$$

边界项 $\int \kappa_g\,ds$ 解释了通过边界"泄漏"的曲率。对平坦圆盘（$K = 0$），$\chi = 1$，公式给出 $\int \kappa_g\,ds = 2\pi$——边界圆的总测地曲率为 $2\pi$，与其曲率 $1/r$ 和周长 $2\pi r$ 一致。

---

## 下一步

Gauss-Bonnet 定理是经典曲面理论的巅峰成果。它证明了几何与拓扑并非独立——它们以深刻的方式相互制约。

展望未来，这种联系将极大地加深。在高维中，**Chern-Gauss-Bonnet 定理**将 Gauss-Bonnet 推广到 $2n$ 维流形，用曲率形式的 **Pfaffian** 替代 Gauss 曲率。**Atiyah-Singer 指标定理**进一步拓展这一范式，将微分算子的解析指标与拓扑不变量相连。这些是现代数学中最深刻的结果。

但在我们能够接近这些推广之前，需要先发展抽象流形、张量场和联络的语言——这一现代框架将微分几何从嵌入环境欧氏空间的需要中解放出来。这是下一篇文章的主题。

---

*本文是 [微分几何](/zh/series/differential-geometry/) 系列的第 5 篇（共 12 篇）。*

*上一篇：[第 4 篇 —— 内蕴几何](/zh/differential-geometry/04-内蕴几何/)*

*下一篇：[第 6 篇 —— 光滑流形](/zh/differential-geometry/06-光滑流形/)*
