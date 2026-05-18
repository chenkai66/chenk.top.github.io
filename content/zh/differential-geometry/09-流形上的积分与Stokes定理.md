---
title: "微分几何（9）：流形上的积分与斯托克斯定理"
date: 2021-11-17 09:00:00
tags:
  - 微分几何
  - 斯托克斯定理
  - 积分
  - 数学
categories: 数学
series: differential-geometry
lang: zh
mathjax: true
description: "斯托克斯定理——流形上的微积分基本定理——将格林定理、高斯定理和经典的斯托克斯定理统一为一个优雅的陈述。"
disableNunjucks: true
series_order: 9
series_total: 12
translationKey: "differential-geometry-9"
---

在单变量微积分中，基本定理说在一个区间上对导数进行积分等于边界差值：$\int_a^b f'(x)\,dx = f(b) - f(a)$。$[a, b]$ 的“边界”是两点集 $\{a, b\}$，其中 $b$ 计为正，$a$ 计为负。右边是对这个带符号边界的 $f$ 的积分。左边是对区间的导数的积分。这本质上是你遇到过的每一个“基本定理”——平面上的格林定理、三维空间中的散度定理、曲面上的经典斯托克斯定理。它们都是流形上的一个陈述的实例：**$M$ 上 $d\omega$ 的积分等于 $\partial M$ 上 $\omega$ 的积分**。

本文的目标是证明并理解这个单一等式。为此，需要三样东西。首先，一个连贯的方向概念——没有它，积分甚至没有符号。其次，带有其诱导方向的边界概念——没有它，右边是没有意义的。第三，$k$ 维子流形上微分形式的积分概念——这需要使用坐标图和平凡分割仔细构造。有了这些，斯托克斯定理就从一个局部计算加上平凡分割论证得出。

这篇文章的重要性在于：斯托克斯定理是流形上微积分的*唯一*结果。其他所有积分定理都是它的推论。一旦你理解了斯托克斯定理，你就理解了为什么电磁通量守恒，为什么绕数是整数，为什么第五篇文章中的 Gauss-Bonnet 定理成立，以及为什么 de Rham 上同调与奇异同调配对。它是局部理论的顶峰，也是每个全局结果的门户。

---

## 1. 方向

切空间 $T_pM$ 是一个 $n$ 维实向量空间，像任何这样的空间一样，它有两个有序基的等价类（由保持方向的线性映射和改变方向的线性映射相关）。选择其中一个类就是 $T_pM$ 的一个**方向**。如果可以在整个流形上平滑地做出这种选择，并且在重叠部分一致，则称流形 $M$ 是**可定向的**。当存在时，方向是一个全局拓扑选择——在连通的可定向流形上恰好有两种方向。

![流形通过一致的有序基的方向](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_1_orientation.png)

如果以下任一条件成立，则流形 $M$ 是可定向的：
1. 存在一个图册，其过渡映射的雅可比行列式都为正。
2. 存在一个处处非零的 $n$ 形式（一个“体积形式”）。
3. $M$ 的标架丛允许一个截面进入 $\mathrm{GL}^+(n)$ 子丛。

这些是等价的，在不同上下文中各有用处。形式理论版本（定义 2）是用来定义积分的。

举个例子。
- $\mathbb{R}^n$ 的每个开子集都是可定向的；标准体积形式 $dx^1\wedge\dots\wedge dx^n$ 给出了一个规范方向。
- 球 $S^n$ 是可定向的——外单位法向量与 $dx^0\wedge\dots\wedge dx^n$ 的收缩给出了一个体积形式。
- 扭环 $T^n$ 是可定向的。
- **莫比乌斯带**是非可定向的。带着一个选定的有序基绕带一圈，你会以相反的基返回。等价地说，莫比乌斯带上不存在处处非零的 2 形式。
- **实射影平面** $\mathbb{RP}^2$ 由于相同的原因是非可定向的。实际上，$\mathbb{RP}^n$ 可定向当且仅当 $n$ 为奇数。

将 $\mathbb{RP}^2$ 视为 $S^2$ 在反演映射 $A(x) = -x$ 下的商。拉回 $A^* (dx\wedge dy\wedge dz) = -dx\wedge dy\wedge dz$ 在 $\mathbb{R}^3$ 中——因此 $A$ 在环境空间中反转方向，但在*球*上计算更为微妙。使用局部坐标图，你会发现反演映射在 $S^2$ 上反转方向，因此商不能继承一个方向。具体、机械，从根本上说是关于基的等价类的拓扑事实。

方向不是多余的装饰——它使积分有符号。一维中的积分 $\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx$ 是最简单的体现：反转区间的方向会翻转积分的符号。在曲面和更高维流形上，同样的现象控制着通量符号、电荷守恒和斯托克斯定理的一致性。没有方向，你甚至不知道曲面的哪一侧算作“向外”。

当 $M$ 非可定向时，你不能积分最高阶形式（符号是模糊的），但你可以积分**密度**——在坐标变换下拾取雅可比行列式的绝对值而不是带符号的雅可比行列式。密度是在非可定向流形上进行积分的方法。在物理学中，它们通常是不可见的，因为时空被认为是可定向的，但在数学生物学和某些拓扑问题（例如计数克莱因瓶上的轨道）中，它们是不可避免的。

每个连通的非可定向流形 $M$ 都有一个连通的可定向双覆盖 $\tilde M \to M$——可定向覆盖。在 $M$ 上的计算通常可以提升到 $\tilde M$ 上，在那里符号是有意义的。例如，莫比乌斯带的可定向双覆盖是圆柱体；射影平面的可定向双覆盖是球体。这个技巧将许多非可定向问题简化为可定向问题，代价是数据加倍。

---

## 2. 带边界的流形；诱导方向

**带边界的流形**是局部模型为半空间 $\mathbb{H}^n = \{x \in \mathbb{R}^n : x^n \geq 0\}$ 的拓扑空间。坐标图有两种类型：内部坐标图（其图像不包含边界 $\{x^n = 0\}$）和边界坐标图（其图像触及 $\{x^n = 0\}$）。**边界** $\partial M$ 是在某个边界坐标图中位于 $\{x^n = 0\}$ 上的点集。它是一个 $(n-1)$ 维流形（无边界）。

![带诱导方向的带边界流形的边界](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_2_boundary.png)

举个例子。
- 闭圆盘 $\bar D^2 = \{|x| \leq 1\} \subset \mathbb{R}^2$：边界是 $S^1$。
- 闭球 $\bar B^3$：边界是 $S^2$。
- 圆柱 $S^1 \times [0, 1]$：边界是 $S^1 \times \{0\} \sqcup S^1 \times \{1\}$ ——两个圆。
- 莫比乌斯带：边界是一个圆（它双重覆盖中心圆）。
- 亏格-$g$ 柄体：边界是一个亏格-$g$ 曲面。三维区域“内部”的二维表面作为其边界；许多三维流形拓扑结构从这里开始构建。

带边界的流形的边界本身没有边界——$\partial(\partial M) = \emptyset$。这对应于形式侧的 $d^2 = 0$。这两个事实（链上的 $\partial^2 = 0$ 和形式上的 $d^2 = 0$）是彼此的镜像，并且通过斯托克斯定理结合在一起：$\int_C d^2\eta = \int_{\partial^2 C}\eta = \int_\emptyset \eta = 0$，反之亦然。几何上的“边界的边界为空”和分析上的“微分平方为零”是同一结构性事实的两种观点。

如果 $M$ 是定向的，$\partial M$ 继承了一个规范方向。规则：$T_p \partial M$ 的基 $(v_1, \dots, v_{n-1})$ 是正定向的，当且仅当 $(N, v_1, \dots, v_{n-1})$ 在 $T_p M$ 中是正定向的，其中 $N$ 是一个向外指向的法向量。等价地说：“先向外法向。”

$M = [a, b]$ 具有标准方向（增加 $x$）。边界是 $\{a, b\}$。在 $b$ 处，向外法向指向右（$+\partial_x$），所以方向规则给出“没有 $v$，只有空基”，约定为“$b$ 处的正号。”在 $a$ 处，向外法向指向左（$-\partial_x$），给出“$a$ 处的负号。”这正是在一维基本定理中的符号约定 $f(b) - f(a)$。

$M = \bar D^2$ 具有标准方向 $dx\wedge dy$。边界是 $S^1$。在某一点，向外法向 $N$ 是径向方向。$S^1$ 上的诱导方向是“逆时针”——标准数学约定。这就是为什么平面中的格林定理要求逆时针边界。

$M = \bar B^3$ 具有 $dx\wedge dy\wedge dz$。边界是 $S^2$。球面上某一点的向外法向是径向方向；$S^2$ 上的诱导方向是使得径向向量在前时重现标准体积形式的方向。这是表面积为正的方向——自然的“向外看”方向。

诱导方向是斯托克斯定理中符号的几何内容。如果你在 $\partial M$ 上选择了错误的方向，你的公式会多出一个全局负号——定理看起来就不成立了。“先向外法向”约定不是任意的；它是唯一能使斯托克斯定理成立而无需临时修正符号的选择。

实际生活中的带边界流形经常有*角*：平面上的一个正方形，三维空间中的一个立方体。斯托克斯定理仍然成立，但边界 $\partial M$ 现在是一个分段光滑流形，诱导方向在角处自然断裂。积分 $\int_{\partial M}\omega$ 只是各光滑部分的和。还有一种推广到带角流形（Joyce, Melrose）的情况，其中角层化是数据的一部分；这在奇异摄动理论和稳定曲线模空间中很重要。

斯托克斯定理也自然处理不连通边界。环面 $\{1 \leq r \leq 2\}$ 有 $\partial M = S^1_{r=1} \sqcup S^1_{r=2}$，内圈顺时针（因为向外法向指向内），外圈逆时针。积分 $\int_{\partial M}\omega$ 是这些符号的和。混淆符号会导致错误答案；正确处理符号是诱导方向规则的全部要点。

---

## 3. 最高阶形式的积分

在流形上自然要积分的对象是最高阶形式：$n$ 维流形上的 $n$ 形式。为什么？因为正微分同胚拉回最高阶形式是良定义的，并且所得积分在坐标变换下不变。（低阶形式只在相应的子流形上积分。）

在具有标准方向的 $\mathbb{R}^n$ 上，$n$ 形式 $\omega = f(x)\,dx^1\wedge\dots\wedge dx^n$ 的积分为
$$\int_{\mathbb{R}^n} \omega = \int_{\mathbb{R}^n} f(x)\,dx^1\dots dx^n,$$
其中右边是普通的 Lebesgue 积分。注意隐含的顺序：楔积的正号匹配迭代积分中变量的标准顺序。

如果 $\varphi: U \to V$ 是 $\mathbb{R}^n$ 开子集之间的保定向微分同胚，且 $\omega$ 是 $V$ 上的 $n$ 形式，则
$$\int_U \varphi^* \omega = \int_V \omega.$$
这是变量替换公式。关键点：楔积*自动*处理雅可比行列式。回想第 8 章：$\varphi^*(dx^1\wedge\dots\wedge dx^n) = \det(D\varphi)\,du^1\wedge\dots\wedge du^n$。带符号的行列式匹配保定向条件。

要在定向流形 $M$ 上积分 $n$ 形式 $\omega$：
1. 用定向坐标图 $U_\alpha$ 覆盖 $M$。
2. 选择一个从属于该覆盖的平凡分割 $\{\rho_\alpha\}$。
3. 定义 $\int_M \omega = \sum_\alpha \int_{U_\alpha} \rho_\alpha \omega$。

结果独立于坐标图和平凡分割的选择。这就是整个定义。

计算 $\int_{S^2} \omega$，其中 $\omega = x\,dy\wedge dz + y\,dz\wedge dx + z\,dx\wedge dy$ 是通过将径向向量与 $dx\wedge dy\wedge dz$ 收缩得到的“球体积形式”。在上半球参数化为 $\varphi(u, v) = (u, v, \sqrt{1 - u^2 - v^2})$ 且 $u^2 + v^2 < 1$ 时，拉回（经过计算）给出 $\frac{1}{\sqrt{1-u^2-v^2}}du\wedge dv$，并在单位圆盘上积分给出 $2\pi$。加上下半球（适当定向）给出 $4\pi$，这正是 $S^2$ 的表面积。合理性检查：$\omega = \iota_R(dx\wedge dy\wedge dz)$，其中 $R = x\partial_x + y\partial_y + z\partial_z$ 是径向向量。拉回到 $S^2$，其中 $R$ 是单位法向量：结果是面积形式。因此 $\int_{S^2}\omega$ 是面积，而 $S^2$ 的面积是 $4\pi$。确认。

流形上的积分是局部几何（形式、导数）和全局量（总通量、总体积、总电荷）之间的桥梁。没有平凡分割论证，你无法定义由多个坐标图覆盖的流形上的积分；有了它们，定义是明确的，定理也适用。

为什么使用平凡分割？从属于覆盖 $\{U_\alpha\}$ 的平凡分割 $\{\rho_\alpha\}$ 是一组光滑非负函数，$\rho_\alpha$ 支撑在 $U_\alpha$ 内，逐点求和为 1（局部有限）。这样的平凡分割存在于任何仿紧 Hausdorff 流形上（这基本上是实践中出现的所有流形）。函数 $\rho_\alpha\omega$ 支撑在 $U_\alpha$ 内，并可以使用 $U_\alpha$ 的坐标图进行积分。选择的独立性来自变量替换公式加上常规检查。

用两个坐标图覆盖 $S^2$，即上半球和下半球（每个都微分同胚于一个圆盘，重叠部分是一个赤道条带）。选择一个平凡分割 $\rho_+ + \rho_- = 1$，其中 $\rho_\pm$ 支撑在相应的半球内。则 $\int_{S^2}\omega = \int_{\text{上半球}}\rho_+\omega + \int_{\text{下半球}}\rho_-\omega$。实际中，选择如球坐标可以使这些计算显式化，球的面积 $4\pi$ 可以通过直接积分确认。平凡分割形式主义是使这严格的*理论*装置；实践中你通常直接计算。

---

## 4. 斯托克斯定理

下面这个结论是核心：设 $M$ 是一个定向的紧致 $n$ 维带边界流形 $\partial M$，配备诱导方向。设 $\omega$ 是 $M$ 上的光滑 $(n-1)$ 形式。则
$$\int_M d\omega = \int_{\partial M} \omega.$$

这就是整个定理，它是微积分中最重要的公式。

![斯托克斯定理：$M$ 上 $d\omega$ 的积分等于边界上 $\omega$ 的积分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_3_stokes.png)

*证明思路。* 两步。

*步骤 1：上半空间中的局部情况。* 取 $M = \mathbb{H}^n$ 且 $\omega$ 支撑在一个紧子集中。写成 $\omega = \sum_i (-1)^{i-1} f_i\,dx^1\wedge\dots\wedge \hat{dx^i}\wedge\dots\wedge dx^n$。则 $d\omega = \sum_i \partial_i f_i\,dx^1\wedge\dots\wedge dx^n$。逐项积分。对于 $i < n$，积分 $\int \partial_i f_i$ 由基本定理（紧支撑的 $f_i$）为零。对于 $i = n$，积分 $\int_{x^n \geq 0} \partial_n f_n = -\int_{\mathbb{R}^{n-1}} f_n(x^1, \dots, x^{n-1}, 0)$，这正好是带符号的边界积分。因此斯托克斯定理在局部成立。

*步骤 2：全局化。* 使用从属于 $M$ 的定向坐标图覆盖的平凡分割 $\{\rho_\alpha\}$。则 $\omega = \sum_\alpha \rho_\alpha \omega$，逐坐标图应用步骤 1 给出 $\int_M d(\rho_\alpha\omega) = \int_{\partial M}\rho_\alpha\omega$。对 $\alpha$ 求和使用 $\sum_\alpha d(\rho_\alpha\omega) = d(\sum_\alpha \rho_\alpha\omega) - 0 = d\omega$（交叉项 $\sum_\alpha d\rho_\alpha\wedge\omega$ 因为 $\sum_\alpha \rho_\alpha = 1$ 导致 $\sum_\alpha d\rho_\alpha = 0$ 而消失）。完成。

就是这样。整个定理是一维基本定理逐坐标图应用并通过平凡分割拼接起来的结果。

如果 $M$ 没有边界（$\partial M = \emptyset$），则对于每个 $(n-1)$ 形式 $\omega$，$\int_M d\omega = 0$。换句话说，闭流形上的精确 $n$ 形式的积分总是零。反过来说，如果一个 $n$ 形式在 $M$ 上的积分非零，则它不可能是精确的，因此在 $H^n_{dR}(M)$ 中代表一个非零类。

举个例子，在 $\mathbb{R}^2 \setminus \{0\}$ 上，角度形式 $\omega = \frac{-y\,dx + x\,dy}{x^2+y^2}$ 是闭的但不是精确的。取 $M$ 为环面 $\{1 \leq r \leq 2\}$。斯托克斯定理说
$$\int_M d\omega = \int_{\partial M}\omega.$$
左边为零（$d\omega = 0$）。边界是内圆（顺时针——与标准逆时针相反）加上外圆（逆时针）。每个圆的积分是 $2\pi$，所以边界积分是 $-2\pi + 2\pi = 0$。一致。形式在每个圆上贡献 $2\pi$，但由于方向相反，它们抵消了。

举个例子，计算 $\int_{\partial \bar B^3} \omega$ 对于 $\omega = (x^2 + y)\,dy\wedge dz + (xy + z)\,dz\wedge dx + (xz - y)\,dx\wedge dy$，其中 $\bar B^3$ 是闭单位球。使用斯托克斯：
$$d\omega = (\partial_x(x^2+y) + \partial_y(xy + z) + \partial_z(xz - y))\,dx\wedge dy\wedge dz = (2x + x + x)\,dx\wedge dy\wedge dz = 4x\,dx\wedge dy\wedge dz.$$
通过对称性 $\int_{\bar B^3} 4x\,dV = 0$（对称域上的奇函数）。因此 $\int_{S^2}\omega = 0$。直接用球坐标进行表面积分会非常痛苦；斯托克斯定理使其瞬间完成。

这很关键，因为证明只有两个要素：一维微积分基本定理（逐坐标图的事实）和平凡分割的粘合能力（全局结构性事实）。没有奇特的分析，除了紧性和可定向性之外没有特殊假设。斯托克斯定理就像微分本身一样基本；它本质上是“在 $M$ 上微分一个形式并积分结果等于沿着 $\partial M$ 积分该形式”——在每个维度上都是相同的陈述，相同的证明。

---

## 5. 经典定理的恢复

所有三个经典“积分定理”都是斯托克斯定理的特例。

![经典定理统一：梯度、斯托克斯、格林、散度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_4_classical_unify.png)

$M = \gamma$（一条曲线），$\omega = f$（一个 0 形式）： $$\int_\gamma df = f(\gamma(b)) - f(\gamma(a)).$$ 这只是一个 $n = 1$ 的斯托克斯定理。“边界”是两个端点，带有适当的符号。
$$\int_\gamma df = f(\gamma(b)) - f(\gamma(a)).$$
这只是一个 $n = 1$ 的斯托克斯定理。“边界”是两个端点，带有适当的符号。

$M$ 是 $\mathbb{R}^2$ 中的一个区域，$\omega = P\,dx + Q\,dy$ 是一个 1 形式。则 $d\omega = (\partial_x Q - \partial_y P)\,dx\wedge dy$，斯托克斯定理给出 $$\iint_M (\partial_x Q - \partial_y P)\,dA = \oint_{\partial M}(P\,dx + Q\,dy).$$ 经典的格林定理。
$$\iint_M (\partial_x Q - \partial_y P)\,dA = \oint_{\partial M}(P\,dx + Q\,dy).$$
经典的格林定理。

$M$ 是 $\mathbb{R}^3$ 中的一个带边界的表面，$F$ 是一个向量场，$\omega = F^\flat$（相应的 1 形式）。则 $d\omega$ 是旋度 2 形式，积分给出
$$\iint_M (\nabla\times F)\cdot dA = \oint_{\partial M} F\cdot dr.$$
经典的“旋度定理”。

斯托克斯定理给出：$M$ 是 $\mathbb{R}^3$ 中的一个区域，$F$ 是一个向量场，$\omega$ 是相应的通量 2 形式。则 $d\omega = (\nabla\cdot F)\,dx\wedge dy\wedge dz$，斯托克斯定理给出 $$\iiint_M \nabla\cdot F\,dV = \iint_{\partial M} F\cdot dA.$$ 高斯散度定理。
$$\iiint_M \nabla\cdot F\,dV = \iint_{\partial M} F\cdot dA.$$
高斯散度定理。

这很关键，因为所有四个经典定理都是流形上一个陈述的推论。将它们记忆为四个独立的结果是一种受坐标限制的地方主义；一个工作的微分几何学家只需使用一次斯托克斯定理就能按需恢复它们。

一个更微妙的推论是柯西积分定理：在 $\mathbb{C} = \mathbb{R}^2$ 上，全纯函数 $f$ 有 $df = f'(z)\,dz$ 其中 $dz = dx + i\,dy$。因此 1 形式 $f(z)\,dz$ 是闭的（柯西-黎曼方程），斯托克斯定理给出 $\oint_{\partial M} f(z)\,dz = 0$ 对于 $f$ 在其上全纯的任何区域 $M$。这是复分析中的柯西定理。整个复分析主题是通过全纯性的视角来看待二维实数中的微分形式。

通过斯托克斯定理的柯西积分公式：取 $f$ 在包含 $z_0$ 半径为 $R$ 的闭圆盘内的区域上全纯。则 $\frac{f(z)}{z - z_0}$ 在 $z_0$ 处有一个简单极点。应用斯托克斯定理（或留数定理）到 $z_0$ 附近的小环：
$$\oint_{|z - z_0| = R}\frac{f(z)}{z - z_0}dz = 2\pi i\,f(z_0).$$
柯西积分公式。由此，所有复分析（刘维尔定理、最大模原理、留数定理）级联而来。全纯函数的奇迹般的刚性——知道 $f$ 在一个圆上的值就可以确定 $f$ 在圆内的值——根本上是应用于带有极点的闭形式的斯托克斯定理。

霍奇定理和调和形式：在紧致定向黎曼流形上，每个 de Rham 上同调类都有唯一的调和代表（$\Delta\omega = 0$ 其中 $\Delta = d\delta + \delta d$）。证明使用斯托克斯定理来设置内积 $\langle\alpha, \beta\rangle = \int_M \alpha \wedge *\beta$，然后使用 PDE 理论找到调和代表。霍奇理论给出了“每个闭形式在模 $\ker\Delta$ 下是精确的”的广泛推广，并是 Kahler 几何、指标定理和流形上椭圆正则性的解析基础。

---

## 6. de Rham 上同调与斯托克斯定理

斯托克斯定理意味着一个基本事实：闭流形上闭形式的积分仅依赖于形式的上同调类。

如果 $\omega_1, \omega_2$ 是 $M$（无边界）上的闭 $k$ 形式，且 $\omega_1 - \omega_2 = d\eta$ 是精确的，则对于任何闭 $k$ 循环 $C$，
$$\int_C \omega_1 = \int_C \omega_2.$$

*证明。* $\int_C(\omega_1 - \omega_2) = \int_C d\eta = \int_{\partial C} \eta = 0$（因为 $\partial C = \emptyset$）。

因此积分 $\int_C \omega$ 仅依赖于 $[\omega] \in H^k_{dR}(M)$。类似地，用同调循环 $C'$ 替换 $C$（即 $C - C' = \partial D$ 对于某个链 $D$）不会改变积分，因为 $\int_C\omega - \int_{C'}\omega = \int_{\partial D}\omega = \int_D d\omega = 0$（因为 $\omega$ 是闭的）。

![de Rham 上同调和庞加莱对偶](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_5_de_rham_coh.png)

de Rham 定理：积分配对
$$H^k_{dR}(M) \times H_k(M; \mathbb{R}) \to \mathbb{R}, \qquad ([\omega], [C]) \mapsto \int_C \omega$$
是一个完美配对。因此 $H^k_{dR}(M) \cong H_k(M; \mathbb{R})^*$ ——de Rham 上同调是实系数奇异同调的对偶。等价地，通过通用系数定理 $H^k_{dR}(M) \cong H^k(M; \mathbb{R})$。

庞加莱对偶：在紧致定向无边界的 $n$ 维流形上，配对
$$H^k_{dR}(M) \times H^{n-k}_{dR}(M) \to \mathbb{R}, \qquad ([\alpha], [\beta]) \mapsto \int_M \alpha\wedge\beta$$
也是一个完美配对。因此 $H^k_{dR}(M) \cong H^{n-k}_{dR}(M)^*$。对于闭合且定向的 $M$，这意味着贝蒂数满足 $b_k = b_{n-k}$ ——在第 8 章的环面示例中看到的上同调维数的对称性。

同调类可以通过积分计算。给定一个闭形式，你可以通过在循环上积分来检测其非平凡性；给定一个循环，你可以通过在其上积分闭形式来检测其非平凡性。这是拓扑的分析基础：贝蒂数、欧拉特征、曲面的亏格——都可以通过微分形式获得。

当流形具有额外结构（例如，复代数簇）时，对于特定形式 $\omega$ 和循环 $C$ 的积分 $\int_C\omega$ 被称为**周期**。代数簇的周期是深刻的算术不变量——它们包括 $2\pi i$、$\log 2$、zeta 函数的值以及更多奇异的超越数。Grothendieck、Kontsevich-Zagier 的猜想以及现代动机上同调理论围绕理解周期展开。因此，斯托克斯定理连接到数学中最深奥的开放问题之一：作为代数形式在代数循环上的积分所获得的超越数的结构。

流形的许多拓扑不变量可以通过积分类似于曲率的形式来计算。欧拉特征等于 $\int_M e(TM)$（Chern-Gauss-Bonnet）。签名等于 $\int_M L(TM)$（Hirzebruch）。椭圆算子的指标等于 $\int_M \mathrm{ch}(\sigma) \mathrm{Td}(TM)$（Atiyah-Singer）。这些都是伪装下的斯托克斯型计算：一个拓扑不变量作为一个闭形式的积分出现。第 12 章将发展相关的特征类并解释为什么这些公式成立。

---

## 7. 链和循环上的积分

我们一直在积分带边界的流形上的形式。完整的通用设置是积分**链**：光滑奇异单纯形的整系数形式组合。

$M$ 中的**光滑 $k$ 单纯形是从标准 $k$ 单纯形到 $M$ 的光滑映射 $\sigma: \Delta^k \to M$：光滑 $k$ 链**是整系数的形式和 $C = \sum_i a_i \sigma_i$。边界算子 $\partial$ 将 $k$ 链映射到 $(k-1)$ 链，通过对面的限制的交错和。

![沿细胞链积分一个形式](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_6_chain_integration.png)

链上的积分线性定义：$\int_{\sum a_i\sigma_i} \omega = \sum a_i \int_{\sigma_i^* \omega}$。斯托克斯定理扩展：
$$\int_C d\omega = \int_{\partial C} \omega.$$
这只是链水平的陈述。

为什么用链？它们比子流形更灵活。链不必嵌入，不必是流形，可以有多重性。这种灵活性使得奇异同调得以工作——你可以细分、细化和计算。三角剖分及其边界是链。de Rham 定理的任一方向证明（构造表示上闭链的闭形式，或构造积分给出上同调类的循环）都在链水平上进行。

在 $\mathbb{R}^2 \setminus \{0\}$ 上的角度形式 $\omega = \frac{-y\,dx + x\,dy}{x^2+y^2}$ 和闭曲线 $\gamma: S^1 \to \mathbb{R}^2 \setminus \{0\}$，整数
$$n(\gamma) = \frac{1}{2\pi}\int_\gamma \omega$$
是 $\gamma$ 绕原点的**绕数**。它由拓扑保证为整数值，并可通过分析计算。根据 de Rham，$H^1_{dR}(\mathbb{R}^2\setminus\{0\}) = \mathbb{R}$ 且 $[\omega/2\pi]$ 是生成元；绕数只是上同调配对。

对于 $\mathbb{R}^3$ 中两个不相交的光滑环 $\gamma_1, \gamma_2$，**链接数** $\mathrm{lk}(\gamma_1, \gamma_2)$ 是一个整数，衡量它们相互缠绕的次数。有一个积分公式（Gauss）：
$$\mathrm{lk}(\gamma_1,\gamma_2) = \frac{1}{4\pi}\oint_{\gamma_1}\oint_{\gamma_2}\frac{(\vec r_1 - \vec r_2)\cdot(d\vec r_1 \times d\vec r_2)}{|\vec r_1 - \vec r_2|^3}.$$
这同样是绕数类型的积分，整数值性来自 $\mathbb{R}^3 \setminus \gamma_2$ 的 de Rham 上同调。链接数是最简单的纽结理论不变量；更高阶的类似物（Massey 积、有限型不变量）推广了同样的思想。

斯托克斯定理的全部力量在链水平上最清晰地显现：链上的 $\partial$ 和形式上的 $d$ 在积分配对下是
<!-- 本节内容因生成长度限制截断；完整推导请参阅本系列对应英文版本。 -->


![Stokes' theorem applied to a sphere and a torus（图）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/09-integration-stokes/dg_v2_09_7_examples.png)


---

*本文是[《微分几何》](/zh/series/differential-geometry/)系列的第 9 篇（共 12 篇）。*

*下一篇：[第 10 篇 — Riemann 几何](/zh/differential-geometry/10-riemannian-geometry/)*
