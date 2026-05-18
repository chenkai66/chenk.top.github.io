---
title: "泛函分析 (3)：Hilbert 空间 —— 无限维空间中的几何"
date: 2021-10-05 09:00:00
tags:
  - functional-analysis
  - hilbert-spaces
  - inner-product
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
description: "内积赋予无限维空间几何结构——正交性、投影和 Riesz 表示定理使 Hilbert 空间成为分析学家的天堂。"
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "functional-analysis-3"
---

# Hilbert 空间 —— 无限维空间中的几何

## 为什么我更喜欢 Hilbert 空间而不是 Banach 空间

如果一个 Banach 空间是一个同意完备的赋范空间，那么一个 Hilbert 空间就是一个进一步同意引入角度的 Banach 空间。这个额外的约定几乎恢复了所有有限维几何——正交性、投影、直角勾股定理——到无限维设置中。作为回报，这种结构足够刚性，使得每个可分的 Hilbert 空间看起来都像一个模型 $\ell^2$。本质上只有一个无限维的 Hilbert 空间（同构意义下），我们在理论中所做的一切归结为选择一个基并计算坐标。

入场费只是一个额外的公理：内积。几何和计算上的回报是巨大的。线性回归、傅里叶级数、量子力学、信号处理、偏微分方程中的能量方法——每一个都是 Hilbert 空间的论证。内积为它们提供了一个统一的语法。

## 内积和 Hilbert 空间

设 $\mathcal{H}$ 是 $\mathbb{C}$ 上的一个向量空间（实数情况类似，去掉共轭符号）。**内积** 是一个函数 $\langle \cdot, \cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ 满足对于所有 $x, y, z \in \mathcal{H}$ 和 $\alpha \in \mathbb{C}$：

1. $\langle x, x \rangle \geq 0$，当且仅当 $x = 0$ 时取等号（正定性）。
2. $\langle x, y \rangle = \overline{\langle y, x \rangle}$（共轭对称性）。
3. $\langle \alpha x + z, y \rangle = \alpha \langle x, y \rangle + \langle z, y \rangle$（在第一个变量上线性）。

共轭对称性则迫使第二个变量上是共轭线性的：$\langle x, \alpha y \rangle = \overline{\alpha} \langle x, y \rangle$。（有些作者将线性放在第二个位置；一旦大家达成一致，选择无关紧要。）

内积通过 $\|x\| = \langle x, x \rangle^{1/2}$ 诱导出一个范数。它由公理 1 正定，齐次因为 $\langle \alpha x, \alpha x \rangle = |\alpha|^2 \langle x, x \rangle$，三角不等式由 Cauchy-Schwarz 不等式得出，这是该主题的核心引理。

对于所有 $x, y \in \mathcal{H}$，$|\langle x, y \rangle| \leq \|x\| \|y\|$，当且仅当 $x, y$ 线性相关时取等号。

*证明.* 如果 $y = 0$，两边都消失。否则，设 $\lambda = \langle x, y \rangle / \|y\|^2$。展开 $0 \leq \|x - \lambda y\|^2 = \|x\|^2 - 2 \mathrm{Re}(\overline{\lambda} \langle x, y \rangle) + |\lambda|^2 \|y\|^2$，用 $\lambda$ 的值简化，不等式就出来了。$\square$

**预 Hilbert 空间** 是带有内积的向量空间。**Hilbert 空间** 是完备的预 Hilbert 空间。

![内积几何：角度和长度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_1_inner_product.png)

### 例子

- $\mathbb{C}^n$ 带有 $\langle x, y \rangle = \sum_i x_i \overline{y_i}$（标准 Hermitian 内积）。
- $\ell^2$ 带有 $\langle x, y \rangle = \sum_n x_n \overline{y_n}$。$\mathbb{C}^n$ 上的 Cauchy-Schwarz 不等式传递到极限，给出求和的收敛性。
- $L^2(\Omega, \mu)$ 带有 $\langle f, g \rangle = \int_\Omega f \overline{g}\, d\mu$ 对于任何测度空间 $(\Omega, \mu)$。这是每个量子力学和偏微分方程分析师的 Hilbert 空间。
- Sobolev 空间 $H^1(\Omega) = W^{1,2}(\Omega)$ 带有 $\langle f, g \rangle_{H^1} = \int (f \overline{g} + \nabla f \cdot \overline{\nabla g})$。

### 数值例子

在 $L^2[0,1]$ 中，取 $f(t) = 1$ 和 $g(t) = t$。则 $\|f\|_2^2 = 1$，$\|g\|_2^2 = \int_0^1 t^2\,dt = 1/3$，$\langle f, g \rangle = \int_0^1 t\,dt = 1/2$。Cauchy-Schwarz：$|\langle f, g \rangle| = 1/2 \leq \|f\| \cdot \|g\| = 1 \cdot 1/\sqrt{3} \approx 0.577$。$f$ 和 $g$ 之间的角度是 $\theta$，$\cos \theta = (1/2)/(1/\sqrt{3}) = \sqrt{3}/2$，所以 $\theta = \pi/6 = 30°$。所以在 $L^2$ 的几何中，常数函数和恒等函数以 $30$ 度的角度相交。这就是我所说的“函数空间中的角度”：一旦有了内积，几何语言就可以直接移植。

## 平行四边形恒等式

区分内积范数与一般范数的单一代数恒等式是**平行四边形法则**：
$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2.$$
几何上：平行四边形对角线平方和等于边长平方和。证明：利用 $\|u\|^2 = \langle u, u \rangle$ 展开两边。

令人惊讶的逆命题是**Jordan-von Neumann 定理**：一个范数满足平行四边形法则当且仅当它来自一个内积。内积可以通过**极化恒等式**恢复：
$$\langle x, y \rangle = \tfrac{1}{4}\big( \|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2 \big).$$
所以我可以通过检查一个代数恒等式来检测 Banach 空间的“Hilbert 性”。检查是具体的：取 $x = (1,0)$ 和 $y = (0,1)$ 在 $\ell^p_2$ 中。则 $\|x+y\|_p^2 + \|x-y\|_p^2 = 2 \cdot 2^{2/p}$，而 $2\|x\|^2 + 2\|y\|^2 = 4$。等式成立要求 $2^{2/p} = 2$，即 $p = 2$。因此，在 $\ell^p$ 家族中，只有 $\ell^2$ 是 Hilbert 空间。$\ell^p$ 家族中 $p \neq 2$ 的永远只是 Banach 空间。

### 为什么这很重要

平行四边形法则是单位球“圆度”的代数阴影。任何来自内积的范数给出一个严格凸、光滑的球——没有角，没有平的部分。这解锁了唯一的最佳逼近：在 Hilbert 空间中，每个闭凸集都有一个唯一的最近点，无一例外。在没有严格凸性的 Banach 空间中，这种唯一性会失败。整个 Hilbert 空间理论的力量追溯到这一单个几何性质。

## 正交性和勾股定理

两个向量 $x, y \in \mathcal{H}$ **正交** 如果 $\langle x, y \rangle = 0$，记作 $x \perp y$。子集 $S \subseteq \mathcal{H}$ 是正交的如果它的成员两两正交，并且**标准正交** 如果此外每个都有范数 $1$。

勾股定理在一般情况下成立：如果 $x \perp y$，则 $\|x + y\|^2 = \|x\|^2 + \|y\|^2$。通过归纳，对于正交族 $\{x_1, \ldots, x_n\}$，$\|\sum x_i\|^2 = \sum \|x_i\|^2$。

给定子空间 $M \subseteq \mathcal{H}$，其**正交补** 是 $M^\perp = \{ y : \langle y, x \rangle = 0 \text{ 对所有 } x \in M \}$。正交补总是闭的（因为它是一些连续泛函 $x \mapsto \langle x, y \rangle$ 的零集的交集）。

## 正交投影

设 $M \subseteq \mathcal{H}$ 是一个闭子空间。**正交投影** $P_M : \mathcal{H} \to M$ 定义为：对于每个 $x \in \mathcal{H}$，$P_M x$ 是 $M$ 中最接近 $x$ 的唯一元素。这种接近性来自平行四边形法则：取序列 $(y_n) \subseteq M$ 使得 $\|x - y_n\| \to d(x, M)$，对 $x - y_n$ 和 $x - y_m$ 应用平行四边形法则，得出 $(y_n)$ 是 Cauchy 列。完备性给出极限 $y^* \in M$（因为 $M$ 是闭的），唯一性来自严格凸性。

投影的定义特征是正交关系 $x - P_M x \perp M$。$M$ 中任何更接近 $x$ 的元素都会违背最小性，差值因此正交于 $M$。

![向量到闭子空间的正交投影](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_2_orthogonal_proj.png)

这个投影具有所有预期的性质：它是有界的（当 $M \neq 0$ 时 $\|P_M\| = 1$），幂等的（$P_M^2 = P_M$），自伴的（$\langle P_M x, y \rangle = \langle x, P_M y \rangle$）。反之，任何在 $\mathcal{H}$ 上的有界算子，如果自伴且幂等，则是其值域上的正交投影。

正交分解定理随之而来：$\mathcal{H} = M \oplus M^\perp$ 对于任何闭子空间 $M$。每个 $x$ 可以唯一地写成 $x = P_M x + (x - P_M x)$，其中两个部分正交。作为一个推论，$(M^\perp)^\perp = M$ 对于闭的 $M$ 成立——这是一个远非显而易见的事实，其 Banach 空间类比需要 Hahn-Banach 定理。

### 数值例子：最小二乘

取 $\mathcal{H} = \mathbb{R}^3$ 并设 $M$ 是由 $u = (1, 0, 0)$ 和 $v = (0, 1, 0)$ 张成的平面。将 $x = (3, 4, 5)$ 投影到 $M$ 上：$P_M x = \langle x, u \rangle u + \langle x, v \rangle v = 3 \cdot u + 4 \cdot v = (3, 4, 0)$。残差 $x - P_M x = (0, 0, 5)$ 正交于 $M$，如预期。

这正是线性回归公式。给定数据点 $(x_i, y_i)$，在最小二乘意义上最佳拟合直线 $y = ax + b$ 是向量 $y \in \mathbb{R}^n$ 在由 $(1, 1, \ldots, 1)$ 和 $(x_1, \ldots, x_n)$ 张成的二维子空间上的正交投影。最小二乘“解”实际上就是某种 Hilbert 空间中的正交投影。同样的机制推广到 $L^2$ 中的最小二乘拟合（最佳多项式逼近、最佳傅里叶逼近、最佳小波逼近）：这些都是某些 Hilbert 空间中的正交投影。

### 为什么这很重要

在没有内积的任何 Banach 空间中，最近点投影可能不存在或不唯一。即使存在，它也不一定是线性的。Hilbert 空间中投影是*线性和连续*的这一事实正是使其成为变分方法、优化以及任何涉及“具有性质 P 的最近函数”的算法的正确设置的原因。没有严格凸性，这样的“最近函数”论证就会失效。

## 标准正交基和 Fourier 系数

设 $\mathcal{H}$ 是一个可分的 Hilbert 空间。$\mathcal{H}$ 中的**标准正交序列** $(e_n)_{n \geq 1}$ 是**完备的**（或**Hilbert 基**）如果 $\overline{\mathrm{span}}\{e_n\} = \mathcal{H}$。通过 Gram-Schmidt 过程，每个可分的 Hilbert 空间都有一个标准正交基（如下所示）。

对于标准正交基 $(e_n)$ 和任意 $x \in \mathcal{H}$，定义**Fourier 系数** $c_n = \langle x, e_n \rangle$。基本结果：

- （Bessel 不等式）$\sum_n |c_n|^2 \leq \|x\|^2$。
- （Parseval）标准正交序列是 Hilbert 基当且仅当对于每个 $x$ 有 $\sum_n |c_n|^2 = \|x\|^2$。在这种情况下，$x = \sum_n c_n e_n$ 在范数意义下成立。
- （Plancherel）$\langle x, y \rangle = \sum_n c_n(x) \overline{c_n(y)}$。

![标准正交基和 Fourier 系数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_3_orthonormal_basis.png)

映射 $x \mapsto (c_n)_{n \geq 1}$ 是一个酉（即保范数且满射）同构 $\mathcal{H} \to \ell^2$。因此每个可分的无限维 Hilbert 空间都同构于 $\ell^2$。在同构意义下，恰好有一个这样的空间。

### $L^2[0, 2\pi]$ 的三角基

函数 $e_n(t) = e^{int}/\sqrt{2\pi}$ 对于 $n \in \mathbb{Z}$ 形成 $L^2[0, 2\pi]$ 的一个标准正交基。Fourier 系数是 $c_n = \frac{1}{\sqrt{2\pi}}\int_0^{2\pi} f(t) e^{-int}\,dt$，并且 Parseval 读作
$$\|f\|_2^2 = \sum_{n \in \mathbb{Z}} |c_n|^2.$$
经典的 Fourier 级数收敛问题是：$\sum c_n e_n$ 在什么意义上收敛于 $f$？在 $L^2$ 中，答案是*总是*，根据 Parseval。在 $L^p$ 中 $p \neq 2$，问题很难（Carleson 定理对于 $p=2$ 几乎处处逐点收敛是一个深刻的结果，而对于 $p=1$ 几乎处处收敛可能会在一个正测度集上严重失败）。Hilbert 结构使收敛问题变得简单。

![L^2[0,1] 的三角基](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_6_l2_basis.png)

### 数值例子

考虑 $f(t) = t$ 在 $[0, 2\pi]$ 上，视为 $L^2[0, 2\pi]$ 中的函数。Fourier 系数（以 $e_n = e^{int}/\sqrt{2\pi}$ 为基）是 $c_n = -\sqrt{2\pi}/(in)$ 对于 $n \neq 0$，以及 $c_0 = \sqrt{2\pi} \cdot \pi / \sqrt{2\pi} = \pi\sqrt{2\pi}$。Parseval：$\|f\|_2^2 = \int_0^{2\pi} t^2 \,dt = 8\pi^3/3$。求和 $\sum |c_n|^2 = 2\pi^3 + \sum_{n \neq 0} 2\pi/n^2 = 2\pi^3 + 4\pi \cdot \pi^2/6 \cdot 2/2$。计算：$2\pi^3 + 2\pi (\pi^2/3) = 2\pi^3 + 2\pi^3/3 = 8\pi^3/3$。等式成立。

这是一个数值确认，虽然没有实际工作但令人放心：你经典计算的相同公式（Fourier 级数的 Parseval 恒等式）与作为 Hilbert 空间定理的 Parseval 相同。

![Parseval 恒等式：范数等于系数平方和](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_5_parseval.png)

## Gram-Schmidt 和构造基

给定 $\mathcal{H}$ 中任意线性独立的向量序列 $(v_n)_{n \geq 1}$，**Gram-Schmidt 过程** 构造一个具有相同跨度的标准正交序列 $(e_n)$：
$$u_1 = v_1,\quad e_1 = u_1/\|u_1\|;\quad u_{n+1} = v_{n+1} - \sum_{k=1}^{n} \langle v_{n+1}, e_k \rangle e_k,\quad e_{n+1} = u_{n+1} / \|u_{n+1}\|.$$
每一步，$u_{n+1}$ 是 $v_{n+1}$ 在投影到 $\mathrm{span}\{e_1, \ldots, e_n\}$ 后的残差，然后标准化。只要 $v_n$ 是线性独立的，过程就能进行。

![Gram-Schmidt 正交化过程应用于有限集](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_7_gram_schmidt.png)

### 数值例子

在 $L^2[-1, 1]$ 中，取 $v_n(t) = t^{n-1}$。Gram-Schmidt 产生（经过标准化后）**Legendre 多项式**：$P_0(t) = 1/\sqrt{2}$，$P_1(t) = \sqrt{3/2}\, t$，$P_2(t) = \sqrt{5/8}(3t^2 - 1)$，$\ldots$。每个 $P_n$ 是 $t^n$ 在 $\mathrm{span}\{1, t, \ldots, t^{n-1}\}$ 的正交补上的正交投影，然后标准化。Legendre 多项式是 $L^2[-1, 1]$ 的一个标准正交基，因此 $L^2[-1, 1]$ 中的任何函数都可以进行“Legendre 展开”——与 Fourier 级数使用不同的基。

不同的基有不同的收敛行为：Legendre 多项式非常适合 $[-1, 1]$ 上的光滑函数但在 $[0, 2\pi]$ 上没有什么特别之处；Fourier 级数自然适合周期函数；Haar 小波自然适合分段定义的函数。基的选择是 Hilbert 空间中应用分析的一半艺术。

### 为什么这很重要

Gram-Schmidt 是构造性的——给定任何可数生成集，它会产生一个标准正交基，从而立即得到 Fourier 系数和 Parseval。因此，可分性（可数稠密集）加上内积自动给出标准正交基。许多 Hilbert 空间理论中的存在性证明可以简化为“对可数稠密集应用 Gram-Schmidt”而无需进一步论证。

## Riesz 表示定理

Hilbert 空间理论中最实用的定理：

设 $\varphi: \mathcal{H} \to \mathbb{C}$ 是 Hilbert 空间上的连续线性泛函。则存在唯一的 $y \in \mathcal{H}$ 使得对于所有 $x \in \mathcal{H}$ 有 $\varphi(x) = \langle x, y \rangle$。此外，$\|\varphi\| = \|y\|$。

换句话说：Hilbert 空间上的每个连续线性泛函都是某个向量的内积。$\mathcal{H}$ 的对偶自然地与 $\mathcal{H}$ 本身同构。

![Riesz 表示定理：每个连续泛函都来自内积](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_4_riesz.png)

*证明草图。* 如果 $\varphi = 0$，取 $y = 0$。否则 $\ker \varphi$ 是一个闭的真子空间，因此 $(\ker \varphi)^\perp$ 包含一个单位向量 $z$（正交补是非平凡的，因为 $\ker \varphi$ 是一个真闭子空间，余维数为 $1$）。设 $y = \overline{\varphi(z)} z$。对于任意 $x \in \mathcal{H}$，写 $x = (x - \alpha z) + \alpha z$ 其中 $\alpha = \langle x, z \rangle$ 使第一部分在 $\ker \varphi$ 中（一个计算）。则 $\varphi(x) = \alpha \varphi(z)$ 且 $\langle x, y \rangle = \alpha \overline{\varphi(z)} \cdot \overline{\overline{1}} = \alpha \varphi(z)$。它们匹配。$\square$

### 数值例子

在 $\ell^2$ 中，考虑泛函 $\varphi(x) = x_1 + x_2/2 + x_3/4 + \cdots = \sum_n x_n / 2^{n-1}$。根据 Riesz，$\varphi(x) = \langle x, y \rangle$ 对于某个 $y \in \ell^2$。读出，$y = (1, 1/2, 1/4, \ldots)$，且 $\|y\|_2^2 = \sum 1/4^{n-1} = 4/3$。因此 $\|\varphi\| = \|y\|_2 = 2/\sqrt{3}$。合理性检查：由 Cauchy-Schwarz，$|\varphi(x)| \leq \|x\|_2 \|y\|_2 = (2/\sqrt{3}) \|x\|_2$，且在 $x = y$ 时达到等号，确认 $\|\varphi\| = 2/\sqrt{3}$。

### 为什么这很重要

Riesz 说 Hilbert 空间的对偶是*自身*。因此，对偶性在一般的 Banach 空间中是一个微妙且抽象的构造（第四篇文章将花费大部分时间在 Hahn-Banach 定理上，只是为了*产生*足够的泛函），而在 Hilbert 空间中却变得非常简单。每个线性泛函都是内积。这就是为什么许多偏微分方程和量子力学中的计算如此流畅：当你需要一个线性泛函时，只需写下向量并与之配对即可。

另一个后果：$\mathcal{H} \times \mathcal{H}$ 上的每个有界双线性形式 $b(x, y)$ 都有形式 $b(x, y) = \langle T x, y \rangle$ 对于某个唯一的有界算子 $T$。这就是如何将偏微分方程的变分形式（Lax-Milgram 定理，有限元方法的核心）转化为算子理论问题——第十二篇文章将对此进行扩展。

## 伴随和自伴算子

设 $T: \mathcal{H} \to \mathcal{H}$ 是有界的。**伴随** $T^*$ 由 $\langle T x, y \rangle = \langle x, T^* y \rangle$ 对所有 $x, y$ 定义。Riesz 保证 $T^*$ 存在且有界，且 $\|T^*\| = \|T\|$。性质：$(S+T)^* = S^* + T^*$，$(\alpha T)^* = \overline{\alpha} T^*$，$(ST)^* = T^* S^*$，$(T^*)^* = T$。

算子 $T$ 是**自伴的**（或 Hermitian）如果 $T = T^*$。Hilbert 空间中的自伴算子扮演着有限维中实对称矩阵的角色。它们的谱是实的，它们允许谱分解（第八篇文章），并且它们生成酉群（第十篇文章）。量子力学中的可观测量是自伴算子。

**酉**算子满足 $T^* T = T T^* = I$，等价地说是等距同构。$L^2(\mathbb{R})$ 上的 Fourier 变换是一个酉算子；它对角化平移，这就是为什么它能解决热方程、波动方程和 Schrödinger 方程的基本情况。

**正规**算子满足 $T^* T = T T^*$。正规算子包括自伴和酉作为特殊情况，并且它们是谱定理成立的最大类。

### 数值例子

移位 $S: \ell^2 \to \ell^2$，$S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$，其伴随 $S^*(x_1, x_2, \ldots) = (x_2, x_3, \ldots)$（反向移位）。验证：$\langle S x, y \rangle = \sum_{n \geq 2} x_{n-1} \overline{y_n} = \sum_{n \geq 1} x_n \overline{y_{n+1}} = \langle x, S^* y \rangle$。移位不是自伴或正规的：$S^* S = I$（右移然后左移给出单位），但 $S S^* x = (0, x_2, x_3, \ldots) \neq x$ 通常。因此移位表现出在有限维中不可能的不对称性，因为在有限维中 $S^* S = I$ 意味着 $S S^* = I$。

## Hilbert 空间中的弱收敛（预览）

第五篇文章将花大量时间讨论弱拓扑，但 Hilbert 空间的情况非常清晰，值得在这里预览一下。序列 $(x_n) \subset \mathcal{H}$ **弱收敛** 到 $x$，记作 $x_n \rightharpoonup x$，如果对于每个 $y \in \mathcal{H}$ 有 $\langle x_n, y \rangle \to \langle x, y \rangle$。根据 Riesz，这等价于：对于每个连续线性泛函 $\varphi$ 有 $\varphi(x_n) \to \varphi(x)$。

弱收敛严格弱于范数收敛。标准基 $(e_n) \subset \ell^2$ 有 $\langle e_n, y \rangle = y_n \to 0$ 对于任何固定的 $y \in \ell^2$（因为 $y$ 的系数是平方可和的，因此趋于零），所以 $e_n \rightharpoonup 0$。但 $\|e_n\| = 1$ 对于每个 $n$，所以 $(e_n)$ 不收敛到 $0$ 在范数意义下。

**Banach-Alaoglu 定理**（第五篇文章）在 Hilbert 空间设置中意味着每个有界序列都有一个弱收敛的子序列。这是闭单位球紧凑性的替代品——回想在无限维中，范数闭单位球*不是*紧的，但在弱拓扑中是紧的。有界集的“弱紧性”是使变分方法工作的杠杆：能量泛函的最小化序列可以假设具有弱极限，这些极限就是所需的最小化器。

一个微妙但重要的点：在 $\mathcal{H}$ 中，弱收敛加上范数收敛意味着范数收敛。也就是说，$x_n \rightharpoonup x$ 和 $\|x_n\| \to \|x\|$ 意味着 $\|x_n - x\| \to 0$。这有时称为*Radon-Riesz 性质*或*Kadec-Klee 性质*。证明：$\|x_n - x\|^2 = \|x_n\|^2 - 2\mathrm{Re}\langle x_n, x \rangle + \|x\|^2 \to \|x\|^2 - 2\|x\|^2 + \|x\|^2 = 0$。所以在 Hilbert 空间中，当范数行为正确时，范数和弱收敛之间的差距精确地消失。

## 极化恒等式的应用

极化恒等式将内积计算简化为范数计算。一个我认为真正有用的推论：复 Hilbert 空间上的有界算子由对角双线性形式 $x \mapsto \langle T x, x \rangle$ 确定。如果对于所有 $x$ 有 $\langle T x, x \rangle = \langle S x, x \rangle$，则 $T = S$。证明：通过极化，$\langle T x, y \rangle$ 由 $\langle T(x \pm y), (x \pm y) \rangle$ 和 $\langle T(x \pm i y), (x \pm i y) \rangle$ 确定，所有这些都等于 $S$ 的相应对角表达式。

这在实 Hilbert 空间中是*错误的*：$\mathbb{R}^2$ 中旋转 $90°$ 有 $\langle T x, x \rangle = 0$ 对于所有 $x$（因为在这种情况下 $T x \perp x$），但 $T \neq 0$。复数情况有更多的冗余，这种冗余固定了算子。

具体来说，这意味着复算子是**自伴的**当且仅当对于每个 $x$ 有 $\langle T x, x \rangle$ 是实数。（取方程 $\langle T x, x \rangle = \overline{\langle T x, x \rangle}$ 的伴随并应用极化。）这个特征在检查自伴性而不显式写出伴随时非常有用。例如，$L^2(\Omega)$ 中适当区域上的 Laplace 算子 $-\Delta$ 有 $\langle -\Delta f, f \rangle = \int |\nabla f|^2 \geq 0$，是实数，所以 $-\Delta$ 是自伴的（关于域的问题将在第九篇文章中讨论）。

## 极化恒等式的应用

极化恒等式将内积计算简化为范数计算。一个我认为真正有用的推论：复 Hilbert 空间上的有界算子由对角双线性形式 $x \mapsto \langle T x, x \rangle$ 确定。如果对于所有 $x$ 有 $\langle T x, x \rangle = \langle S x, x \rangle$，则 $T = S$。证明：通过极化，$\langle T x, y \rangle$ 由 $\langle T(x \pm y), (x \pm y) \rangle$ 和 $\langle T(x \pm i y), (x \pm i y) \rangle$ 确定，所有这些都等于 $S$ 的相应对角表达式。

这在实 Hilbert 空间中是*错误的*：$\mathbb{R}^2$ 中旋转 $90°$ 有 $\langle T x, x \rangle = 0$ 对于所有 $x$（因为在这种情况下 $T x \perp x$），但 $T \neq 0$。复数情况有更多的冗余，这种冗余固定了算子。

具体来说，这意味着复算子是**自伴的**当且仅当对于每个 $x$ 有 $\langle T x, x \rangle$ 是实数。（取方程 $\langle T x, x \rangle = \overline{\langle T x, x \rangle}$ 的伴随并应用极化。）这个特征在检查自伴性而不显式写出伴随时非常有用。例如，$L^2(\Omega)$ 中适当区域上的 Laplace 算子 $-\Delta$ 有 $\langle -\Delta f, f \rangle = \int |\nabla f|^2 \geq 0$，是实数，所以 $-\Delta$ 是自伴的（关于域的问题将在第九篇文章中讨论）。

## 直和与张量积

两个 Hilbert 空间 $\mathcal{H}_1, \mathcal{H}_2$ 允许一个**直和** $\mathcal{H}_1 \oplus \mathcal{H}_2$——对 $(x_1, x_2)$ 有内积 $\langle (x_1, x_2), (y_1, y_2) \rangle = \langle x_1, y_1 \rangle_{\mathcal{H}_1} + \langle x_2, y_2 \rangle_{\mathcal{H}_2}$。直和是一个 Hilbert 空间（分量的完备性意味着对的完备性）。闭子空间的分解 $\mathcal{H} = M \oplus M
<!-- 本节内容因生成长度限制截断；完整推导请参阅本系列对应英文版本。 -->

---

*本文是[《泛函分析》](/zh/series/functional-analysis/)系列的第 3 篇（共 12 篇）。*

*下一篇：[第 4 篇 — 对偶空间与 Hahn-Banach](/zh/functional-analysis/04-dual-spaces-hahn-banach/)*
