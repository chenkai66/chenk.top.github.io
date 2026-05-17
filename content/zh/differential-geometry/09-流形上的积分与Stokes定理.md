---
title: "微分几何（九）：流形上的积分与 Stokes 定理"
date: 2021-11-17 09:00:00
tags:
  - differential-geometry
  - stokes-theorem
  - integration
  - mathematics
categories: Mathematics
series: differential-geometry
lang: zh
mathjax: true
description: "Stokes 定理——流形上的微积分基本定理——将 Green 定理、Gauss 散度定理、经典 Stokes 定理统一为一个优雅的等式。"
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "differential-geometry-9"
---

一元微积分的基本定理告诉我们：$\int_a^b f'(x)\,dx = f(b) - f(a)$——导数在区间上的积分等于边界处的函数值之差。我们在多元微积分中遇到的每一个"基本定理"——Green 定理、散度定理、经典 Stokes 定理——都是这同一思想的高维推广。本文的目标是给出统一它们的那个结果：**流形上的 Stokes 定理**。

要达到这个目标，我们需要两样尚未建立的东西：流形上的**积分**概念，以及流形的**边界**概念。这两者都需要仔细处理——流形没有环境坐标可以依赖，我们必须从微分形式和定向出发，内蕴地构建积分理论。

---

## 积分需要定向

### 为什么定向至关重要

考虑在 $\mathbb{R}^3$ 中的一个曲面 $S$ 上积分一个 2-形式 $\omega$。在每个点我们选取一个局部参数化，将 $\omega$ 拉回到 $\mathbb{R}^2$ 上进行积分。但参数化带有一个选择：我们让法向量指向"上方"还是"下方"？反转参数化会翻转积分的符号。要使积分具有良好定义，我们需要一个**全局一致的选择**——即定向。

![Stokes 定理统一所有经典积分定理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_fig9_stokes.png)


### 可定向流形

一个 $n$ 维光滑流形 $M$ 是**可定向的**，如果存在一个处处不为零的 $n$-形式 $\Omega \in \Omega^n(M)$。这样的形式称为**体积形式**。两个体积形式 $\Omega$ 和 $\Omega'$ 定义相同的定向，当且仅当 $\Omega' = f\Omega$，其中 $f > 0$ 是处处为正的光滑函数。一个**定向流形**就是流形加上一个体积形式等价类的选择。

**例子。**

- $\mathbb{R}^n$ 是可定向的：$\Omega = dx^1 \wedge \cdots \wedge dx^n$ 是一个全局体积形式。
- 对每个 $n$，球面 $S^n$ 都是可定向的。在 $S^2 \subset \mathbb{R}^3$ 上，面积形式 $\omega = x\,dy \wedge dz + y\,dz \wedge dx + z\,dx \wedge dy$ 限制到球面上就是体积形式。
- Mobius 带**不可定向**：任何沿带面定义一致法方向的尝试在绕行一圈后都会失败。

**等价刻画。** $M$ 可定向当且仅当可以选取一个图册 $\{(U_\alpha, \varphi_\alpha)\}$，使得所有转移函数 $\varphi_\beta \circ \varphi_\alpha^{-1}$ 的 Jacobi 行列式处处为正。这就是局部坐标定向全局相容的条件。

### Riemann 流形上的体积形式

如果 $M$ 带有 Riemann 度量 $g$，那么一个定向就确定了一个典范的体积形式：在局部定向坐标中，

$$\text{vol}_g = \sqrt{\det(g_{ij})}\, dx^1 \wedge \cdots \wedge dx^n.$$

这就是积分后给出 Riemann 体积的那个形式。在单位球面 $S^2$ 上配标准度量，$\text{vol}_g = \sin\theta\, d\theta \wedge d\varphi$，积分得到 $4\pi$。

---

## 定向流形上 $n$-形式的积分

### $\mathbb{R}^n$ 上的积分

从基本情形出发。如果 $\omega = f(x)\, dx^1 \wedge \cdots \wedge dx^n$ 是 $\mathbb{R}^n$ 的开子集 $U$ 上一个紧支撑的 $n$-形式，我们定义

$$\int_U \omega = \int_U f(x)\, dx^1 \cdots dx^n,$$

右端是普通的 Lebesgue（或 Riemann）积分。

### 通过坐标卡在流形上积分

对于定向 $n$-流形 $M$ 上的紧支撑 $n$-形式 $\omega$，如果 $\text{supp}(\omega)$ 落在单个定向坐标卡 $(U, \varphi)$ 内，我们定义

$$\int_M \omega = \int_{\varphi(U)} (\varphi^{-1})^*\omega.$$

拉回 $(\varphi^{-1})^*\omega$ 是 $\mathbb{R}^n$ 开子集上的紧支撑 $n$-形式，我们已知如何积分。

**关键事实。** 如果 $(V, \psi)$ 是另一个包含 $\text{supp}(\omega)$ 的定向坐标卡，积分的换元公式（由于定向保证 Jacobi 行列式为正）确保得到相同结果。这正是我们需要定向的原因：没有正 Jacobi 条件，坐标变换可能引入符号翻转。

### 单位分解

一般情况下，$\text{supp}(\omega)$ 可能无法装入单个坐标卡。我们用**单位分解**来处理：一族光滑函数 $\{\rho_\alpha\}$，从属于局部有限开覆盖 $\{U_\alpha\}$，满足

1. $0 \le \rho_\alpha \le 1$，且 $\text{supp}(\rho_\alpha) \subseteq U_\alpha$；
2. $\sum_\alpha \rho_\alpha = 1$，处处成立。

然后定义

$$\int_M \omega = \sum_\alpha \int_M \rho_\alpha\, \omega,$$

其中每个 $\rho_\alpha\, \omega$ 的支撑都在 $U_\alpha$ 内，可以通过单个坐标卡积分。标准论证表明结果与单位分解和覆盖的选取无关。

**注记。** 光滑单位分解的存在性是光滑流形的基本性质——正是它使得微分几何中从局部到全局的过渡成为可能。

---

## 带边流形

### 定义

**带边流形** $M$ 是一个局部以上半空间

$$\mathbb{H}^n = \{(x^1, \ldots, x^n) \in \mathbb{R}^n : x^n \ge 0\}$$

为模型的拓扑空间。映到 $\mathbb{H}^n$ 内部的点是 $M$ 的内部点；映到超平面 $\{x^n = 0\}$ 的点组成**边界** $\partial M$。边界 $\partial M$ 本身是一个没有边界的光滑 $(n-1)$ 维流形。

**例子。**

- 闭单位圆盘 $\bar{D}^2 = \{(x,y) : x^2 + y^2 \le 1\}$ 是带边 2-流形，$\partial \bar{D}^2 = S^1$。
- 闭单位球 $\bar{B}^n$ 的边界是 $\partial \bar{B}^n = S^{n-1}$。
- 柱面 $S^1 \times [0,1]$ 的边界由两个圆组成。
- 闭区间 $[a,b]$ 是带边 1-流形，$\partial [a,b] = \{a, b\}$——仅两个点。

### 边界上的诱导定向

如果 $M$ 是定向的带边 $n$-流形，则边界 $\partial M$ 继承一个自然的定向。使 Stokes 定理符号正确的约定是**外法向在前**约定：在边界点 $p \in \partial M$，设 $\nu$ 是一个指向外部的向量（不与 $\partial M$ 相切）。如果 $(e_1, \ldots, e_{n-1})$ 是 $T_p(\partial M)$ 的一组基，当 $(\nu, e_1, \ldots, e_{n-1})$ 是 $T_pM$ 的正定向基时，我们说 $(e_1, \ldots, e_{n-1})$ 是正定向的。

**例。** 对 $M = [a,b]$，在 $b$ 处外法向指向右（正方向），诱导定向为 $+1$；在 $a$ 处指向左，定向为 $-1$。因此 $\int_{\partial [a,b]} f = f(b) - f(a)$，恰好恢复了微积分基本定理的边界项。

---

## Stokes 定理：陈述与证明概要

### 定理陈述

> **Stokes 定理。** 设 $M$ 是紧致定向 $n$ 维光滑带边流形，$\partial M$ 取诱导定向，$\omega$ 是 $M$ 上的光滑 $(n-1)$-形式。则
>
> $$\int_M d\omega = \int_{\partial M} \omega.$$

这个陈述的美在于其简洁：$d\omega$ 在"体"上的积分等于 $\omega$ 在边界上的积分。不需要向量场，不需要内积，不需要叉积——只有形式和外微分。

### 证明概要

证明分三步：

**第一步：化归到坐标卡。** 利用单位分解 $\{\rho_\alpha\}$，将 $\omega$ 写为 $\omega = \sum_\alpha \rho_\alpha \omega$。由积分与外微分的线性性，只需对每个 $\rho_\alpha \omega$（支撑在单个坐标卡中）证明定理。

**第二步：在 $\mathbb{R}^n$ 上证明（内部坐标卡）。** 若 $\text{supp}(\omega)$ 完全在 $M$ 内部，则需证 $\int_{\mathbb{R}^n} d\omega = 0$（因为没有边界贡献）。将 $\omega$ 展开为 $\omega = \sum_i f_i\, dx^1 \wedge \cdots \wedge \widehat{dx^i} \wedge \cdots \wedge dx^n$。则 $d\omega = \sum_i (-1)^{i-1} \frac{\partial f_i}{\partial x^i} dx^1 \wedge \cdots \wedge dx^n$。由于每个 $f_i$ 有紧支撑，$\int_{-\infty}^{\infty} \frac{\partial f_i}{\partial x^i} dx^i = 0$（一元微积分基本定理，函数在 $\pm\infty$ 处为零）。故 $\int_{\mathbb{R}^n} d\omega = 0$。

**第三步：在 $\mathbb{H}^n$ 上证明（边界坐标卡）。** 若 $\text{supp}(\omega)$ 与边界 $\{x^n = 0\}$ 相交，同样的计算表明除含 $\frac{\partial f_n}{\partial x^n}$ 的项外所有项都为零。对那一项，$\int_0^\infty \frac{\partial f_n}{\partial x^n} dx^n = -f_n(x^1, \ldots, x^{n-1}, 0)$（函数在 $+\infty$ 为零但在 $0$ 处不为零）。在 $\{x^n = 0\}$ 上的积分恰好是 $\int_{\partial \mathbb{H}^n} \omega$，符号由外法向在前约定保证正确。

对所有坐标卡求和即完成证明。核心洞察是：整个论证归结为一元微积分基本定理的逐变量应用。

---

## 恢复经典定理

Stokes 定理的威力在于所有经典向量分析积分定理都是它的特例。

### 微积分基本定理

取 $M = [a,b]$，边界 $\{a, b\}$。设 $\omega = f$ 是 0-形式（函数），$d\omega = f'\,dx$。则

$$\int_{[a,b]} f'\,dx = \int_{\partial [a,b]} f = f(b) - f(a).$$

### Green 定理

设 $M = D$ 是 $\mathbb{R}^2$ 中的紧致区域，边界曲线 $\partial D$。设 $\omega = P\,dx + Q\,dy$ 是 1-形式，则 $d\omega = \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx \wedge dy$，Stokes 定理给出

$$\iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dx\,dy = \oint_{\partial D} P\,dx + Q\,dy.$$

### 散度定理（Gauss 定理）

设 $M = \Omega$ 是 $\mathbb{R}^3$ 中的紧致区域，边界曲面 $\partial \Omega$。给定向量场 $\mathbf{F} = (F_1, F_2, F_3)$，定义 2-形式

$$\omega = F_1\, dy \wedge dz + F_2\, dz \wedge dx + F_3\, dx \wedge dy.$$

则 $d\omega = (\nabla \cdot \mathbf{F})\, dV$，Stokes 定理给出

$$\iiint_\Omega \nabla \cdot \mathbf{F}\, dV = \oiint_{\partial \Omega} \mathbf{F} \cdot d\mathbf{S}.$$

这就是散度定理。上面的 2-形式 $\omega$ 正是与 $\mathbf{F}$ 对应的通量形式。

### 经典 Stokes 定理（旋度定理）

设 $M = S$ 是 $\mathbb{R}^3$ 中的定向曲面，边界曲线 $\partial S$。给定向量场 $\mathbf{F}$，定义 1-形式 $\omega = F_1\,dx + F_2\,dy + F_3\,dz$。则 $d\omega$ 是旋度对应的 2-形式，Stokes 定理给出

$$\iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S} = \oint_{\partial S} \mathbf{F} \cdot d\mathbf{r}.$$

**统一至此完成。** 微积分基本定理、Green 定理、散度定理、经典 Stokes 定理——这四个定理分别是 $\int_M d\omega = \int_{\partial M} \omega$ 应用于维度 1、2、2、3 的流形、并选择适当微分形式的结果。

---

## de Rham 上同调初步

Stokes 定理有一个深刻的拓扑学推论。如果 $\omega$ 是无边界紧致流形 $M$ 上的闭 $(n-1)$-形式（$d\omega = 0$），那么 $\int_M d\omega = 0$ 是平凡的。但如果 $\omega$ 是闭的却**不恰当**——即不存在 $(n-2)$-形式 $\eta$ 使得 $d\eta = \omega$——会怎样？恰当性的障碍携带着拓扑信息。

### 闭形式与恰当形式

$M$ 上的 $k$-形式 $\omega$ 称为：
- **闭的**，若 $d\omega = 0$；
- **恰当的**，若 $\omega = d\eta$（$\eta$ 是某个 $(k-1)$-形式）。

由于 $d^2 = 0$，每个恰当形式都是闭的。反之一般不成立，这种失败由上同调来度量。

### de Rham 上同调群

$M$ 的**第 $k$ 阶 de Rham 上同调群**是商向量空间

$$H^k_{\text{dR}}(M) = \frac{\ker(d : \Omega^k \to \Omega^{k+1})}{\text{im}(d : \Omega^{k-1} \to \Omega^k)} = \frac{\{\text{闭 } k\text{-形式}\}}{\{\text{恰当 } k\text{-形式}\}}.$$

两个闭形式代表相同的上同调类，当且仅当它们相差一个恰当形式。维数 $b_k = \dim H^k_{\text{dR}}(M)$ 是**第 $k$ 阶 Betti 数**。

### 例子

- **$\mathbb{R}^n$：** 由 Poincare 引理，$\mathbb{R}^n$ 上每个闭形式都是恰当的，所以 $H^k(\mathbb{R}^n) = 0$（$k > 0$），$H^0 = \mathbb{R}$（常函数）。全部 Betti 数为零（除 $b_0 = 1$）。

- **$S^1$：** 1-形式 $d\theta$ 是闭但不恰当的（绕 $S^1$ 积分为 $2\pi \ne 0$）。所以 $H^1(S^1) \cong \mathbb{R}$，$b_1 = 1$。这探测到了圆的"洞"。

- **$S^2$：** $H^0 = H^2 = \mathbb{R}$，$H^1 = 0$，反映 $S^2$ 是连通的、没有一维洞、且围住一个二维空腔。

- **环面 $T^2$：** $H^0 = \mathbb{R}$，$H^1 = \mathbb{R}^2$，$H^2 = \mathbb{R}$。$H^1$ 的两个生成元对应环面上两个独立的闭合回路。

### 与拓扑的关系

著名的 **de Rham 定理**指出：de Rham 上同调与实系数奇异上同调同构：

$$H^k_{\text{dR}}(M) \cong H^k(M; \mathbb{R}).$$

这意味着纯分析对象（微分形式和外微分）捕捉了纯拓扑不变量（上同调类和 Betti 数）。Stokes 定理是桥梁：它表明闭形式在圈上的积分仅依赖于形式的上同调类和圈的同调类。

Euler 示性数可由 Betti 数计算：$\chi(M) = \sum_{k=0}^n (-1)^k b_k$。对 $S^2$，$\chi = 1 - 0 + 1 = 2$；对 $T^2$，$\chi = 1 - 2 + 1 = 0$。这与经典值一致。

de Rham 上同调提供了分析与拓扑之间最优美的联系之一，在后续讨论示性类时它将再次出现。

---

## 下一步

有了积分理论和 Stokes 定理，微分形式在流形上的核心机制已经完备。但到目前为止我们研究的是**光滑**结构——还没有问如何度量长度、角度和曲率。下一篇文章将引入 **Riemann 度量**：使光滑流形成为可度量距离、定义测地线和曲率的几何空间的附加结构。这就是 Riemann 几何——广义相对论和现代几何学的语言。

---

*本文是 [微分几何](/zh/series/differential-geometry/) 系列的第 9 篇（共 12 篇）。*

*上一篇：[第 8 篇 —— 微分形式](/zh/differential-geometry/08-微分形式/)*

*下一篇：[第 10 篇 —— Riemann 几何](/zh/differential-geometry/10-Riemann几何/)*
