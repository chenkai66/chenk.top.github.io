---
title: "微分几何（10）：黎曼几何 — 度量、联络和平行移动"
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
description: "黎曼度量让我们可以在任何光滑流形上测量长度、角度和体积 —— Levi-Civita 联络提供了平行移动和测地线的经典概念。"
disableNunjucks: true
series_order: 10
series_total: 12
translationKey: "differential-geometry-10"
---

前面四章把流形当一个纯粹的"光滑壳子"在用。可以谈连续、谈光滑、谈切空间、谈微分形式、谈积分——但有一件最朴素的事，到现在为止还做不了：

**测量两点之间的距离。**

这听起来荒唐——距离不是任何数学对象的"基础属性"吗？但回到流形的定义：流形只要求局部上长得像 $\mathbb{R}^n$，不要求长得像*欧几里得*的 $\mathbb{R}^n$。任何同胚都被允许，所以"距离"这件事压根没有规定。打个比方：流形就像一张地图，但没标比例尺——你能识别出"哪些点和哪些点相邻"，但量不出任何具体长度。

要恢复"长度、角度、面积"这些日常的几何概念，就得在流形上*额外加一个数据*：一个内积，逐点定义，光滑地依赖于点。这个数据有个名字——*Riemann 度量* $g$。一旦把 $g$ 加上来，整个本科微积分能干的事就立即都能干：曲线长度、向量夹角、区域面积、梯度、拉普拉斯算子，全部上线。这一章就是把这套机器搭起来的过程。

具体讨论一个动机。地球表面是一个 2-流形，从北京到纽约的"最短路径"是什么？直觉上知道是大圆弧——但这个直觉怎么用数学语言表达？答案是：定义一个度量 $g$（从地球嵌入 $\mathbb{R}^3$ 继承下来的），用它写出曲线长度公式 $L(\gamma) = \int\|\dot\gamma\|_g$，然后求长度泛函的极小值。极值曲线就是*测地线*——弯曲流形上的"直线"。

更深一层的动机来自广义相对论。爱因斯坦说，引力不是力，而是时空弯曲的表现。一个行星沿着时空的测地线运动，"看起来"是被太阳吸引，其实只是在弯曲时空里走直线。这要求时空是一个 4-流形上加了一个度量——只不过这个度量不是正定的（有一个时间维带负号），叫"洛伦兹度量"。机器是同一台机器，符号约定不一样而已。

这一章的具体计划：(1) 引入 Riemann 度量 $g$，让长度、角度、面积、体积都有定义；(2) 定义*仿射联络* $\nabla$——一个在向量场之间求导的规则——然后特化到唯一的"无挠且度量兼容"的联络（叫 *Levi-Civita 联络*）；(3) 用 $\nabla$ 定义*平行移动*和*测地线*；(4) 介绍*完备性*（Hopf-Rinow 定理）；(5) 在球面和双曲平面上做具体计算。

中间会出现两个看着很技术的术语，先在这里说清几何直觉，避免后面读者卡住：

- **无挠（torsion-free）**：意思是"沿一个无穷小平行四边形做平行移动，到一阶为止能闭合"。即：从 $p$ 出发先沿 $X$ 走 $\epsilon$，再沿 $Y$ 走 $\epsilon$，得到一个点；如果反过来先沿 $Y$ 再沿 $X$，到达另一个点。这两个终点的差距，到 $\epsilon^2$ 的精度才能区分出来。"挠"测量的就是这种"先后顺序"造成的偏差是不是能在一阶里抹平。无挠 = 一阶能抹平。Levi-Civita 联络选这个条件，因为它使得函数的 Hessian 矩阵对称（混合偏导可换序），与 $\mathbb{R}^n$ 上的经验一致。
- **度量兼容（metric-compatible）**：意思是"平行移动保长度"。如果沿一条曲线把一个向量平行地搬过去，搬过去之后向量的长度（用 $g$ 量）不变。两个向量的夹角也不变。换句话说，$\nabla$ 和 $g$ 互相尊重，搬运过程中没有"偷偷拉伸"任何东西。

下一章会把这两条结合到一起的代价——曲率张量——展开来讲。今天先专注于度量、联络、测地线本身。

---

## 黎曼度量

![球面上的平行移动与和乐：向量沿三角形路径传输后旋转 90°](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_parallel_transport_paths.png)

**黎曼度量** 在光滑流形 $M$ 上是一个光滑的赋值
$$g: M \to T^*M \otimes T^*M, \qquad p \mapsto g_p$$
其中每个 $g_p$ 是 $T_pM$ 上的对称正定双线性形式。换句话说：在每一点上都有一个切空间上的内积，并且内积随点的变化而光滑变化。

在坐标下，$g$ 由一个 $n \times n$ 对称正定矩阵的光滑函数表示：
$$g = g_{ij}(x)\,dx^i \otimes dx^j, \qquad g_{ij} = g_{ji}, \qquad (g_{ij}) \succ 0.$$

![黎曼度量为每个切空间分配一个内积](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_1_riem_metric.png)

度量立即产生标准的几何量：
- **切向量的长度：** $\|v\|_g = \sqrt{g_p(v, v)}$。
- **切向量之间的夹角：** $\cos\theta = \frac{g_p(u, v)}{\|u\|_g\|v\|_g}$。
- **曲线 $\gamma: [a, b] \to M$ 的长度：** $L(\gamma) = \int_a^b \|\dot\gamma(t)\|_g\,dt$。
- **黎曼体积形式**（在定向流形上）：$\mathrm{vol}_g = \sqrt{\det(g_{ij})}\,dx^1\wedge\dots\wedge dx^n$。
- **黎曼距离：** $d_g(p, q) = \inf\{L(\gamma) : \gamma \text{ 连接 } p, q\}$。

黎曼距离将 $(M, g)$ 变成一个度量空间（在拓扑教科书的意义上），并且度量拓扑与流形拓扑一致。

举个例子。
1. **欧几里得空间** $(\mathbb{R}^n, g_{\mathrm{Eucl}})$：$g_{ij} = \delta_{ij}$，即单位矩阵。标准内积。$L(\gamma) = \int |\dot\gamma|$，距离是通常的欧几里得距离。

2. 带有圆度量的球面 $S^2$：在球坐标 $(\theta, \phi)$（$\theta$ 为极角，$\phi$ 为方位角）下：
$$g = d\theta^2 + \sin^2\theta\,d\phi^2.$$
这是从标准嵌入 $S^2 \subset \mathbb{R}^3$ 继承的度量。体积形式是 $\sin\theta\,d\theta\wedge d\phi$，表面积 $\int_{S^2} \mathrm{vol}_g = 4\pi$ 符合预期。

3. 双曲平面 $\mathbb{H}^2$：上半平面 $\{(x, y) : y > 0\}$ 带有度量
$$g = \frac{dx^2 + dy^2}{y^2}.$$
这个度量具有常高斯曲率 $-1$。测地线是垂直线和与 $x$ 轴垂直相交的半圆。$(0, 1)$ 和 $(0, e)$ 之间的距离是 $1$（对于向上走的测地线）；当 $y \to 0$ 沿任何接近边界的路径时，距离“趋于无穷”。双曲平面是负曲率的典型模型。

![Riemannian 度量：Euclid 几何 vs 双曲几何中的单位球](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_riemannian_metric.png)

4. 拉回/诱导度量：如果 $f: M \to (N, h)$ 是浸入（微分单射），则 **拉回度量** $f^*h$ 在 $M$ 上定义为 $(f^*h)_p(u, v) = h_{f(p)}(df_p u, df_p v)$。这就是子流形如何从其环境空间继承度量的方式。$\mathbb{R}^3$ 中的每个光滑嵌入表面都可以通过这种方式变成黎曼流形 —— 并且如果 $f$ 是浸入，则结果总是正定的。

每个仿紧光滑流形都允许一个黎曼度量。证明：用坐标图覆盖，取每个坐标图中的欧几里得度量，然后用单位分解粘合。正定形式的凸组合是正定的。因此，黎曼几何是你总可以添加到任何光滑流形上的特征。

有了度量，本科微积分中的每个几何量都有流形版本：长度、角度、面积、体积、梯度（与函数相关的向量场）、拉普拉斯算子。没有度量，只有微分拓扑不变量（上同调、丛上的联络的特征类）可用。度量将“光滑空间”变成了“几何空间”。

如果将正定性放宽为 *非退化*，得到 伪黎曼度量最重要的是洛伦兹——4维流形上的签名 $(-, +, +, +)$。这是广义相对论中的时空度量。大部分机制（Levi-Civita、测地线、曲率）逐字照搬；唯一的变化是符号约定和一些“长度”变为虚数（在黎曼符号约定中，类时间隔是虚数，在物理约定中是实正数）。爱因斯坦方程 $G_{\mu\nu} = 8\pi T_{\mu\nu}$ 是洛伦兹的，而不是黎曼的，但几何是一样的。

取 $S^2$ 上的曲线 $\gamma(t) = (\theta(t), \phi(t)) = (\pi/2, t)$，$t \in [0, \pi]$ —— 从 $\phi = 0$ 到 $\phi = \pi$ 的赤道弧（半个赤道）。速度 $\dot\gamma = (0, 1) = \partial_\phi$。长度平方：$g_{\phi\phi}(\dot\gamma)^2 = \sin^2(\pi/2) \cdot 1 = 1$。所以 $\|\dot\gamma\|_g = 1$ 且 $L = \int_0^\pi 1\,dt = \pi$。半个赤道的长度是 $\pi$ —— 正好是单位球的半径乘以 $\pi$。合理性检查：完整赤道的长度是 $2\pi$，符合 $2\pi r$ 当 $r = 1$ 时的情况。

---

## 音乐同构和梯度

度量提供了 $TM$ 和 $T^*M$ 之间的同构 —— “音乐”同构。

![Levi-Civita 联络：唯一的度量相容无挠联络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_levi_civita.png)

给定 $X \in TM$，定义 $X^\flat \in T^*M$ 为 $X^\flat(Y) = g(X, Y)$。在坐标下，$X^\flat_i = g_{ij} X^j$ —— “降低指标”。

给定 $\omega \in T^*M$，定义 $\omega^\sharp \in TM$ 为 $g(\omega^\sharp, Y) = \omega(Y)$。在坐标下，$\omega^{\sharp i} = g^{ij}\omega_j$，其中 $g^{ij}$ 是逆矩阵。

这些映射是互逆的纤维丛同构。没有度量时，向量和余向量是 *本质上不同的对象*；有了度量，它们成为同一枚硬币的两面。

对于函数 $f \in C^\infty(M)$，微分 $df \in T^*M$ 是一个 1-形式，**梯度** 是它的尖锐：
$$\nabla f = (df)^\sharp.$$
在坐标下：$(\nabla f)^i = g^{ij}\partial_j f$。在欧几里得 $\mathbb{R}^n$ 上，$g^{ij} = \delta^{ij}$，所以 $\nabla f = (\partial_1 f, \dots, \partial_n f)$ —— 熟悉的梯度。在弯曲流形上，梯度是依赖于度量的对象，不同于微分。

在带有度量 $d\theta^2 + \sin^2\theta\,d\phi^2$ 的 $S^2$ 上，取 $f = \cos\theta$。那么 $df = -\sin\theta\,d\theta$。逆度量是 $g^{\theta\theta} = 1, g^{\phi\phi} = 1/\sin^2\theta, g^{\theta\phi} = 0$。所以
$$\nabla f = -\sin\theta \cdot 1 \cdot \partial_\theta + 0 = -\sin\theta\,\partial_\theta.$$
范数是 $|\nabla f|_g = \sin\theta$。在赤道（$\theta = \pi/2$），梯度具有指向南极的单位长度 —— 余弦函数的最陡下降方向。在极点，梯度消失（由于 $\cos\theta$ 在极点附近的对称性，它必须消失）。

在黎曼流形上，函数的 **拉普拉斯算子** 是 $\Delta f = \mathrm{div}(\nabla f)$，其中散度通过体积形式定义。在坐标下：$\Delta f = \frac{1}{\sqrt{|g|}}\partial_i(\sqrt{|g|}\,g^{ij}\partial_j f)$。在欧几里得空间中，这简化为熟悉的 $\sum_i \partial_i^2 f$。在球面上，它给出 $\Delta f = \frac{1}{\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta f) + \frac{1}{\sin^2\theta}\partial_\phi^2 f$ —— 其特征函数是球谐函数的球面拉普拉斯算子。Laplace-Beltrami 算子是黎曼流形分析的核心对象：谱几何、热方程以及大多数数学物理在弯曲空间中的应用都使用它。

数学物理中的所有二阶偏微分方程 —— 热方程、波动方程、薛定谔方程、狄拉克方程 —— 都需要度量才能在流形上表述。度量告诉你哪个微分算子配得上“拉普拉斯算子”的名字，该算子的谱通过 Weyl 定律（特征值计数体积）及其扩展编码了几何数据。

---

## 联络：向量场的微分

流形上的向量场 $X$ 没有明显的“变化率” —— 要比较 $X_p$ 在一点与 $X_q$ 在另一点，你需要一种方法来识别 $T_pM$ 与 $T_qM$。在欧几里得空间中这是自动的（$\mathbb{R}^n$ 中的平行平移），但在一般流形上需要额外的结构：联络是

![动画：球面纬线上的平行移动](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_parallel_transport.gif)

**联络** $\nabla$ 在 $TM$ 上（更准确地说，是 *仿射联络*）是一个 $\mathbb{R}$-线性算子
$$\nabla: \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M), \qquad (X, Y) \mapsto \nabla_X Y$$
满足：
1. **$C^\infty$-线性于 $X$：** $\nabla_{fX}Y = f\nabla_X Y$。
2. **Leibniz 于 $Y$：** $\nabla_X(fY) = (Xf)Y + f\nabla_X Y$。

第一个条件说“$\nabla_X Y$ 在一点仅取决于 $X$ 在该点”，所以 $(\nabla_X Y)_p$ 是良定义的。第二个是标准的乘积法则。

在坐标下，$\nabla$ 由 **Christoffel 符号**
$$\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\,\partial_k$$
确定。一个计算给出
$$\nabla_X Y = \left(X^i \partial_i Y^k + \Gamma^k_{ij}X^i Y^j\right)\partial_k.$$
第一项是显然的“分量的方向导数”；第二项是考虑基向量变化的修正项。$\Gamma^k_{ij}$ 是位置的函数，*不是* 张量分量 —— 它们在坐标变换下按非齐次规则变换，正好补偿 $Y^k$ 的偏导数的非齐次性。

![协变导数：平行与非平行传输](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_covariant_derivative.png)

为什么这么多结构？联络是流形上的 *额外数据* —— 一般来说没有规范选择。两个联络相差一个张量（Christoffel 符号的差按张量方式变换）。联络的空间是一个无限维仿射空间。

联络可能满足的两个自然条件：
- **无挠：** $T(X, Y) := \nabla_X Y - \nabla_Y X - [X, Y] = 0$。等价于坐标下的 $\Gamma^k_{ij} = \Gamma^k_{ji}$。几何上：沿无穷小平行四边形的平行移动在一阶闭合。
- **度量兼容：** $\nabla g = 0$，即 $X g(Y, Z) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$。几何上：平行移动保持度量（长度和角度）。

这些条件不等价也不互相蕴含，但一起唯一确定联络（**黎曼几何的基本定理**，也称为 Levi-Civita 联络的存在性和唯一性）。这是将在下一节进行的计算。

一个经常令人困惑的点：$\nabla_X Y$ 仅依赖于 $X$ 在一点，但依赖于 $Y$ 在一个邻域。限制在曲线 $\gamma$ 上，算子 $\nabla_{\dot\gamma}$ 给出沿曲线的向量场的导数。这有时记作 $D/dt$ —— 同一个算子，不同记法，当你有一条曲线但没有全局向量场时使用。两种观点都需要：$\nabla_X Y$ 用于全局微分算子，$D/dt$ 用于平行移动和测地线方程。

---

## Levi-Civita 联络

在黎曼流形上，存在唯一的既 **度量兼容**（$X g(Y, Z) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$，“度量是平行的”）又 **无挠**（$\nabla_X Y - \nabla_Y X = [X, Y]$，“没有内置扭曲”）的联络。这就是 **Levi-Civita 联络**，黎曼几何的标准联络。

![Levi-Civita 联络作为唯一的无挠度量联络](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_2_levi_civita.png)

$$2g(\nabla_X Y, Z) = X g(Y, Z) + Y g(Z, X) - Z g(X, Y) + g([X, Y], Z) - g([Y, Z], X) + g([Z, X], Y).$$
右边由 $g$ 和向量场的 Lie 括号决定 —— 这些已经有了。左边唯一确定 $\nabla_X Y$。因此 Levi-Civita 联络存在且唯一。

在坐标下，
$$\Gamma^k_{ij} = \frac{1}{2}g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}).$$
这个公式在 $i, j$ 中是对称的（无挠性的结果），并且可以从任何显式度量计算出来。

度量 $g = d\theta^2 + \sin^2\theta\,d\phi^2$。逆度量：$g^{\theta\theta} = 1$，$g^{\phi\phi} = 1/\sin^2\theta$，$g^{\theta\phi} = 0$。非零的 Christoffel 符号：
$$\Gamma^\theta_{\phi\phi} = -\sin\theta\cos\theta, \qquad \Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \cot\theta.$$
其他均为零。这六个数（实际上是两个不同的数）编码了球面的所有几何信息 —— 测地线、平行移动、曲率等等。

$\Gamma^\theta_{\phi\phi} = \frac{1}{2}g^{\theta\theta}(2\partial_\phi g_{\theta\phi} - \partial_\theta g_{\phi\phi}) = \frac{1}{2}(0 - 2\sin\theta\cos\theta) = -\sin\theta\cos\theta$。匹配。

如果没有度量兼容性，平行移动不会保持长度 —— 因此自然的“刚体运动”概念会失效。如果没有无挠性，函数的 Hessian 将不对称 —— 因此二阶微积分将不符合其欧几里得版本。Levi-Civita 是使黎曼几何行为符合经典几何期望的唯一联络。

度量 $g = (dx^2 + dy^2)/y^2$，所以 $g_{xx} = g_{yy} = 1/y^2$，$g_{xy} = 0$，$g^{xx} = g^{yy} = y^2$。非零的 Christoffel 符号：
$$\Gamma^x_{xy} = \Gamma^x_{yx} = -\frac{1}{y}, \qquad \Gamma^y_{xx} = \frac{1}{y}, \qquad \Gamma^y_{yy} = -\frac{1}{y}.$$
你可以从 Christoffel 公式验证这些。有了这些，测地线方程变为
$$\ddot x - \frac{2}{y}\dot x \dot y = 0, \qquad \ddot y + \frac{1}{y}(\dot x^2 - \dot y^2) = 0.$$
垂直测地线 $x = c$，$y(t) = e^t$ 满足这些（垂直线）。以 $x$ 轴为中心的半圆形测地线也可以验证。因此，双曲平面上的测地线是垂直线和半圆 —— 经典图像变得可计算。

---

## 平行移动

给定一个联络和一条曲线 $\gamma: [a, b] \to M$，沿 $\gamma$ 的向量场 $V$ 是 **平行的**，如果 $\nabla_{\dot\gamma}V = 0$。在坐标下：$\dot V^k + \Gamma^k_{ij}\dot\gamma^i V^j = 0$。这是一个关于 $V$ 的一阶线性 ODE，因此对于任何初始条件 $V(a) \in T_{\gamma(a)}M$，存在唯一的沿 $\gamma$ 的平行场 $V$ 使得 $V(a)$ 给定。映射
$$P_\gamma: T_{\gamma(a)}M \to T_{\gamma(b)}M, \qquad V(a) \mapsto V(b)$$
是沿 $\gamma$ 的 平行移动是线性同构，并且（对于 Levi-Civita 联络）是一个等距映射 —— 它保持度量。

![沿球面上的闭合环路的平行移动返回一个旋转后的向量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_3_parallel_transport.png)

在欧几里得空间中，平行移动是路径无关的：沿着任意路径从 $p$ 到 $q$，你携带向量的方式相同（只需平移它们）。在弯曲流形上，平行移动通常是路径相关的：沿闭合环路从 $p$ 到 $p$ 携带的向量返回时会被旋转。这是曲率的几何核心。

具体计算一下：取顶点在北极、$(\theta, \phi) = (\pi/2, 0)$（本初子午线上的赤道）和 $(\theta, \phi) = (\pi/2, \pi/2)$（东经 90° 的赤道）的球面三角形。三条边都是大圆弧（测地线）。从北极开始，选择一个指向本初子午线南端的向量。沿本初子午线向下平行移动：向量保持指向南方（沿测地线 —— 测地线的切向量保持平行于自身）。在本初子午线上的赤道处，向量指向南方，即远离赤道进入南半球 —— 但我们的切空间是在赤道上，所以让我明确一下：在北极，选择一个指向经度 0 的赤道方向的向量。经过运输到达经度 0 的赤道后，该向量现在位于赤道切平面中并指向南方（垂直于赤道）。现在沿赤道从经度 0 移动到经度 90°：垂直于赤道的向量保持垂直（沿测地线的平行移动保持与测地线的角度）。所以在 $(\pi/2, \pi/2)$ 处，向量仍然指向南方。现在沿经度 90° 的子午线回到北极：向量指向外部（沿该子午线，与子午线的切向方向相反）。它最初指向北极处的本初子午线方向；现在指向 90° 子午线的方向。这两个方向之间的夹角是 90°。

因此，沿这个测地线三角形的平行移动将切向量旋转 90°。积分全纯性等于三角形的 角度过剩是指球面三角形的角度和 $> \pi$，过剩 $= \int K\,dA$ —— 正如 Gauss-Bonnet 定理（文章 5）所述。对于我们的三角形，角度都是90°，和为 $3\pi/2$，过剩为 $\pi/2$，面积为 $\pi/2$ 在单位球上（$K = 1$），全纯性旋转也是 $\pi/2$。这三个数一致，正如 Gauss-Bonnet 所要求的。

考虑沿纬度 $\theta_0$ 的圆周在 $S^2$ 上的平行移动（不是测地线 —— 除了赤道以外的纬度圆周不是测地线）。围绕这个圆周的全纯性角度是 $2\pi(1 - \cos\theta_0)$。当 $\theta_0 \to 0$（靠近极点的小圆周）时，全纯性 $\to 0$（圆周包围一个小区域，总曲率小）。当 $\theta_0 \to \pi/2$（赤道，是测地线）时，全纯性 $\to 2\pi$。连续变化与包围区域的面积乘以曲率匹配 —— 直接数值确认全纯性 = $\int K\,dA$ 在包围区域内。

这很关键，因为平行移动是联络的 *几何* 表现。联络、曲率、全纯性 —— 这些都是同一现象的不同方面，分别视为微分算子、张量场和积分传输。在规范理论中，沿 Wilson 环的平行移动是一个基本可观测量；在广义相对论中，沿测地线的平行移动通过潮汐力决定了所经历的引力场。

黎曼流形中两条附近的测地线 $\gamma_1, \gamma_2$ 以由曲率决定的速率漂移。*Jacobi 方程* $\nabla_{\dot\gamma}^2 J + R(J, \dot\gamma)\dot\gamma = 0$（其中 $J$ 是分离向量，$R$ 是 Riemann 张量 —— 见文章 11）控制这种漂移。在广义相对论中，这正是 *潮汐引力* 的方程：两个附近自由落体物体根据局部时空曲率相互接近或远离。牛顿的“引力”在爱因斯坦的重新表述中，正是曲率引起的测地偏离。黎曼几何是这一思想的严格表达。

---

## 测地线和指数映射

测地线方程 $\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i\dot\gamma^j = 0$ 是欧氏空间里牛顿第二定律 $\ddot x = 0$ 的弯曲版本。等号右边的零，意思是没有真正的外力——只是流形本身弯曲，让按惯性走的轨迹看上去像是被弯曲了。$\Gamma^k_{ij}$ 在物理上就扮演几何力的角色。这正是广义相对论的核心思想：行星不是被太阳拉着走，而是在弯曲时空里走自由轨迹（测地线），但在我们的欧氏感知下看起来像被拉着。

**测地线** 是一条曲线 $\gamma$，它是自身的平行移动：$\nabla_{\dot\gamma}\dot\gamma = 0$。在坐标下，
$$\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i\dot\gamma^j = 0.$$
这是一个关于曲线坐标的二阶 ODE，初始数据为 $(\gamma(0), \dot\gamma(0)) \in TM$。根据 Picard-Lindelof 定理，测地线存在且局部唯一；它们可能不会延展到所有时间（流形可能是不完备的）。

![联络作为切丛中的水平子空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/figures/10_connection_bundle.png)

测地线是 **局部长度最小化** 的：在具有相同端点的附近曲线中，测地线具有最短长度。它们并不总是全局最小化 —— 在球面上，从北极到南极附近“绕远路”的大圆弧是测地线，但不是最短路径。

![测地完备性 vs 不完备性](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_5_geodesic_complete.png)

举个例子。
- **欧几里得空间：** 测地线是直线。
- **球面 $S^2$：** 测地线是大圆。从任意起点和方向出发，测地线是该方向的大圆；测地线在距离 $2\pi$ 后闭合。
- **双曲平面（上半平面模型）：** 测地线是垂直线和与 $x$ 轴垂直相交的半圆。
- **柱面 $S^1 \times \mathbb{R}$：** 测地线是通过万有覆盖下的直线的像。它们螺旋缠绕在柱面上。

定义 **指数映射** $\exp_p: T_pM \to M$ 为 $\exp_p(v) = \gamma_v(1)$，其中 $\gamma_v$ 是满足 $\gamma_v(0) = p$ 和 $\dot\gamma_v(0) = v$ 的测地线（当 $t = 1$ 时存在）。在 $T_pM$ 中 $0$ 的一个小邻域内，$\exp_p$ 是到其像的微分同胚，且 $(\exp_p)^{-1}$ 提供了 $p$ 处的 **正规坐标** —— 在这些坐标下，度量在 $p$ 处看起来是欧几里得的一阶近似，且 Christoffel 符号在 $p$ 处消失。

具体计算一下：在 $S^2$ 上，从北极出发的 $\exp_{NP}$ 将 $v \in T_{NP}S^2 \cong \mathbb{R}^2$（$|v| = r$）映射到纬度 $\pi/2 - r$ 的方向为 $v$（模 $2\pi$）的点。指数映射作为一个光滑映射 $\mathbb{R}^2 \to S^2$ 是良定义的，但它不是一个微分同胚 —— 它在 $|v| = \pi$ 时折叠到南极，然后周期性地回到自身。在 $T_{NP}S^2$ 中 $|v| < \pi$ 的圆盘内，$\exp_{NP}$ 是到 $S^2 \setminus \{SP\}$ 的微分同胚。

$p$ 的 **割迹** 是 $\exp_p$ 停止成为微分同胚的点集 —— 等价地，是测地线从 $p$ 出发停止全局最小化的点。在 $S^2$ 上，任何点的割迹是其对跖点（单点）。在紧致黎曼流形上，割迹是非空的；其结构编码了全局几何。在平坦环面上，割迹是一个更复杂的 1-复形；在一般曲面上，它是一个分段光滑网络。割迹是测地线全局最小化的几何障碍，它出现在最优传输理论和机器人学等领域。

测地线是长度泛函的临界点。如果 $\gamma_s(t)$ 是一族曲线，$\gamma_0 = \gamma$，则 $L(\gamma_s)$ 在 $s = 0$ 处的 **第一变分 消失当且仅当 $\gamma$ 是测地线（具有适当的边界条件）：第二变分** 涉及 Riemann 张量，决定测地线是 *最小值* 还是仅仅是临界点。这种变分法的观点是黎曼几何与 Morse 理论之间的桥梁 —— 也是 Bott 如何从测地线推导出李群的拓扑信息的方法。

---

## 全纯性和 Hopf-Rinow 定理

Hopf-Rinow 定理是关于完备性的——什么时候每两点之间都有最短测地线？粗略说，只要流形是测地完备（任何测地线都能延伸到 $\pm\infty$）或度量完备（按 Riemann 距离做成度量空间是完备的），就够了。一个反例：把 $\mathbb{R}^2$ 挖掉一个点，剩下的开集上的标准度量。两点连线如果穿过被挖的点就不是直线了，且没有最短路径——序列可以无限接近一条经过挖掉的点的折线，但极限不存在于流形里。完备性确保了几何是良性的，没有这种悄悄掉出去的风险。

对于闭合环路 $\gamma$ 在 $p$ 处，平行移动定义了一个线性等距 $P_\gamma \in \mathrm{O}(T_pM, g_p)$。所有这样的等距构成一个子群，即 **全纯群** $\mathrm{Hol}(p)$。

![全纯群作为沿闭合环路平行移动的角度缺陷](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_4_holonomy.png)

对于 $n$ 维黎曼流形，$\mathrm{Hol}(p) \subseteq \mathrm{O}(n)$。对于定向流形，$\mathrm{Hol}(p) \subseteq \mathrm{SO}(n)$。Berger 分类了单连通黎曼流形的不可约全纯群 —— 它们是 $\mathrm{SO}(n)$，$\mathrm{U}(n/2)$，$\mathrm{SU}(n/2)$，$\mathrm{Sp}(n/4)$，$\mathrm{Sp}(n/4)\mathrm{Sp}(1)$，$G_2$，$\mathrm{Spin}(7)$。每个全纯类对应一个特殊的几何结构：Kähler（$\mathrm{U}$），Calabi-Yau（$\mathrm{SU}$），超 Kähler（$\mathrm{Sp}$），以及例外情况。

举个例子。
- $\mathbb{R}^n$：$\mathrm{Hol} = \{e\}$（平凡）。平行移动是路径无关的。
- $S^2$：$\mathrm{Hol} = \mathrm{SO}(2)$（切平面的全旋转群）。
- 平坦环面：$\mathrm{Hol} = \{e\}$（局部欧几里得，无曲率）。
- Calabi-Yau 3-叠：$\mathrm{Hol} = \mathrm{SU}(3)$。这些是弦理论家用来紧化的流形。

对于连通黎曼流形 $M$，以下条件等价：
1. $(M, d_g)$ 是完备度量空间。
2. 每条测地线延展到整个 $\mathbb{R}$（测地完备性）。
3. 指数映射 $\exp_p$ 在某个（等价地，每个）$p$ 处定义在整个 $T_pM$ 上。
4. $M$ 中的闭且有界的子集是紧的。

此外，在这种情况下，任意两点可以通过一条最小化测地线连接。

![Hopf-Rinow 定理：完备性、测地线和极小化曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_6_hopf_rinow.png)

这看起来像四条无关命题的奇怪集合，但它们是同一件事的不同侧面。直观地说：**度量完备**意味着柯西序列有极限，**测地完备**意味着测地线不"撞到边界就消失"，**指数映射全局定义**意味着从任一点向任一方向都能"无限直走"，**闭且有界等于紧**意味着 Heine-Borel 定理在曲面流形上仍然成立。Hopf-Rinow 把它们打包成一句话：在连通流形上，这些条件全等价。

这个等价不是显然的。考虑反例。开圆盘 $\{(x, y) : x^2 + y^2 < 1\}$ 用诱导欧氏度量是非完备的：你能沿径向向边界走，但永远到不了 —— 测地线从某个时刻起停止延展。同样，**穿洞的平面** $\mathbb{R}^2 \setminus \{0\}$ 的测地线遇到原点就消失。两者都是**测地不完备**的，因为有"洞"。Hopf-Rinow 告诉你：去掉一个点就能毁掉所有四条性质。

举个例子。
- **球面 $S^2$**：完备。任何两点用大圆弧连接，最短弧是测地线。
- **欧氏平面 $\mathbb{R}^n$**：完备。
- **双曲平面**（庞加莱圆盘模型）：完备 —— 别被"圆盘"骗了，双曲度量在边界处发散。
- **开圆盘**（欧氏度量）：**不**完备。
- **史瓦西时空**外解：**不**完备 —— 测地线在视界处需要"跨越"到内部解，这就是黑洞奇异性的几何起源。

最后一个例子触及到广义相对论的核心：**洛伦兹流形**（GR 的舞台）经常**不完备**。测地不完备性 = 时空奇异性 = 物理上要么是"宇宙诞生"（大爆炸），要么是"物质坍缩到一点"（黑洞奇点）。Penrose-Hawking 奇异定理的全部内容就是给出**何时**洛伦兹流形必然不完备的几何条件。所以 Hopf-Rinow 的"反面"才是物理上最有趣的部分。

![球面、双曲平面、环面三种经典黎曼度量](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/differential-geometry/10-riemannian-geometry/dg_v2_10_7_examples.png)

---

## 下一步

我们建立了完整的工具栈：度量给出长度和角度，Levi-Civita 联络给出微分向量场的标准方式，平行移动让我们能在不同点比较向量，测地线扮演直线的角色，指数映射把切空间局部映回流形，全纯性把环路效应打包成一个李群，Hopf-Rinow 把"完备性"的所有合理理解打通。

但有一件事我们一直在回避：**为什么**球面和平面"几何上不同"？Berger 分类里那么多全纯群，每一个对应一种"内蕴弯曲"，可我们还没度量它。下一篇要补上这个洞。我们要定义**Riemann 曲率张量** $R(X, Y)Z$，它度量"沿无穷小平行四边形平行移动一圈，向量旋转了多少"。这就是 Levi-Civita 联络在告诉我们空间到底"多弯"。从那里出发，我们会取迹得到 **Ricci 曲率**（决定 Einstein 方程），再取一次迹得到**标量曲率**（一个数字概括局部弯曲程度）。

这是一个分层故事：曲率张量 → Ricci → 标量。每一步取迹都丢失信息但保留了某些重要性质。下一篇文章把这个故事讲完，并展示为什么"截面曲率为常数"的流形（球面、平面、双曲空间）会成为基础例子。
