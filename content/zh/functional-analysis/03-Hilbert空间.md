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
## Hilbert 空间——无限维空间中的几何

Hilbert 空间是数学中一个非常重要的概念。它是一种特殊的向量空间，其中每个向量都有长度，并且可以计算两个向量之间的夹角。这使得 Hilbert 空间在处理无限维问题时特别有用。

核心结论是：Hilbert 空间是一个完备的内积空间。这意味着在这个空间中，每个柯西序列都收敛到该空间中的某个点。此外，内积的存在使得我们可以定义向量的长度和角度。

具体来说，如果 $H$ 是一个 Hilbert 空间，那么对于任意两个向量 $x, y \in H$，存在一个内积 $\langle x, y \rangle$。这个内积满足以下性质：
- 对称性：$\langle x, y \rangle = \overline{\langle y, x \rangle}$
- 线性性：$\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle$
- 正定性：$\langle x, x \rangle \geq 0$，并且 $\langle x, x \rangle = 0$ 当且仅当 $x = 0$

通过内积，可以定义向量的范数（长度）为 $\|x\| = \sqrt{\langle x, x \rangle}$。范数满足以下性质：
- 非负性：$\|x\| \geq 0$，并且 $\|x\| = 0$ 当且仅当 $x = 0$
- 齐次性：$\|\alpha x\| = |\alpha| \|x\|$，其中 $\alpha$ 是标量
- 三角不等式：$\|x + y\| \leq \|x\| + \|y\|$

这些性质使得 Hilbert 空间成为研究函数、信号处理和量子力学等领域的重要工具。

## 内积及其创造的几何

Banach 空间是一个完备的赋范空间，而 Hilbert 空间则进一步引入了角度。这个额外的约定——内积——几乎恢复了所有有限维几何在无限维空间中的性质。正交性、投影、勾股定理、子空间中最近点的概念——这些都回来了。代价只是一个公理；回报是巨大的，无论是在几何上还是计算上。

设 $\mathcal{H}$ 是复数域 $\mathbb{C}$ 上的向量空间。内积是一个函数 $\langle \cdot, \cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{C}$，满足对所有 $x, y, z \in \mathcal{H}$ 和 $\alpha \in \mathbb{C}$：

1. $\langle x, x \rangle \geq 0$，当且仅当 $x = 0$ 时取等号（正定性）。
2. $\langle x, y \rangle = \overline{\langle y, x \rangle}$（共轭对称性）。
3. $\langle \alpha x + z, y \rangle = \alpha \langle x, y \rangle + \langle z, y \rangle$（第一个参数的线性）。

共轭对称性迫使第二个参数具有共轭线性：$\langle x, \alpha y \rangle = \overline{\alpha} \langle x, y \rangle$。有些作者将线性放在第二个参数中；选择是惯例，不是定理，我遵循物理惯例，将线性放在第一个参数中。内积诱导了一个范数：$\|x\| = \langle x, x \rangle^{1/2}$，三角不等式由 Cauchy-Schwarz 不等式得出，这是 Hilbert 空间理论中最重要的不等式。

Cauchy-Schwarz 不等式指出，对于所有 $x, y \in \mathcal{H}$，有 $|\langle x, y \rangle| \leq \|x\| \|y\|$，当且仅当 $x$ 和 $y$ 线性相关时取等号。

证明如下：如果 $y = 0$，两边都为零。否则，设 $\lambda = \langle x, y \rangle / \|y\|^2$ 并展开 $0 \leq \|x - \lambda y\|^2 = \|x\|^2 - |\langle x, y \rangle|^2 / \|y\|^2$。重新排列得到不等式。几何洞察：我们将 $x$ 投影到通过 $y$ 的直线上，并观察残差 $x - \lambda y$ 与 $y$ 正交，因此其范数平方非负。等号成立当且仅当残差为零，即 $x$ 在通过 $y$ 的直线上。证毕。

从 Cauchy-Schwarz 不等式可以推导出三角不等式：$\|x + y\|^2 = \|x\|^2 + 2\text{Re}\langle x,y\rangle + \|y\|^2 \leq \|x\|^2 + 2\|x\|\|y\| + \|y\|^2 = (\|x\| + \|y\|)^2$。因此，每个内积空间都是赋范空间，而 Hilbert 空间是完备的内积空间。

例子是分析中的工作马。空间 $\ell^2$ 由序列 $(x_n)$ 组成，满足 $\sum |x_n|^2 < \infty$，并配备内积 $\langle x, y \rangle = \sum x_n \overline{y_n}$。空间 $L^2(\Omega, \mu)$ 由（等价类的）平方可积函数组成，内积为 $\langle f, g \rangle = \int f \overline{g}\, d\mu$；这里涵盖了量子力学、Fourier 分析和偏微分方程理论。Sobolev 空间 $H^1(\Omega) = W^{1,2}(\Omega)$ 具有内积 $\langle f, g \rangle_{H^1} = \int (f\overline{g} + \nabla f \cdot \overline{\nabla g})$，同时编码函数值和导数。Hardy 空间 $H^2(\mathbb{D})$ 包含单位圆盘上的全纯函数，其 Taylor 系数平方可和，这构成了标准集合。

![内积几何：角度和长度](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_v2_03_1_inner_product.png)

举个例子，在 $L^2[0,1]$ 中，取 $f(t) = 1$ 和 $g(t) = t$。那么 $\|f\|^2 = 1$，$\|g\|^2 = \int_0^1 t^2\,dt = 1/3$，并且 $\langle f, g \rangle = \int_0^1 t\,dt = 1/2$。Cauchy-Schwarz 不等式：$|\langle f, g\rangle| = 1/2 \leq \|f\|\cdot\|g\| = 1/\sqrt{3} \approx 0.577$，验证成立。$f$ 和 $g$ 之间的夹角满足 $\cos\theta = \frac{\langle f,g\rangle}{\|f\|\|g\|} = \frac{1/2}{1/\sqrt{3}} = \frac{\sqrt{3}}{2}$，从而 $\theta = \pi/6$。常数函数和恒等函数在 $L^2$ 几何中以 30 度相交。这不是比喻或宽松的类比——一旦有了内积，函数空间中的角度就像 $\mathbb{R}^3$ 中的角度一样真实和可计算。

平行四边形法则 $\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2$ 表征了内积范数：一个范数满足这个恒等式当且仅当它来自内积（Jordan-von Neumann 定理，1935）。内积可以通过极化恒等式恢复：$\langle x, y \rangle = \tfrac{1}{4}(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2)$。快速诊断：在 $\ell^p_2$ 中，标准基向量 $x = (1,0)$，$y = (0,1)$，平行四边形法则读作 $2 \cdot 2^{2/p} = 4$，强制 $p = 2$。因此，在 $\ell^p$ 空间中，只有 $\ell^2$ 是 Hilbert 空间。整个 $L^p$ 家族在 $p \neq 2$ 时只是 Banach 空间——完备的赋范空间，但没有角度和正交性的几何丰富性。

为什么这对应用很重要？线性回归在 Hilbert 空间中最小化 $\|y - X\beta\|^2$——这是正交投影。Fourier 级数在 $L^2$ 的正交基中展开函数——这是 Hilbert 空间的几何。量子力学假设状态生活在 Hilbert 空间中，可观测物是自共轭算子。信号处理将信号分解成频率成分——在 $L^2$ 中进行正交分解。PDE 中的能量方法使用 $H^1$ 内积来提取先验估计。机器学习中的核方法在再生核 Hilbert 空间中操作。每一个都是 Hilbert 空间论证，穿着特定领域的外衣。内积给了它们统一的语法。

## 正交性、投影和最近点性质

两个向量 $x, y \in \mathcal{H}$ 是正交的，记作 $x \perp y$，如果 $\langle x, y \rangle = 0$。勾股定理同样适用：如果 $x \perp y$，那么 $\|x + y\|^2 = \|x\|^2 + \|y\|^2$。对于 $n$ 个两两正交的向量，$\|\sum_{k=1}^n x_k\|^2 = \sum_{k=1}^n \|x_k\|^2$。这可以扩展到收敛的无穷级数：如果 $(x_k)$ 两两正交且 $\sum \|x_k\|^2 < \infty$，则级数 $\sum x_k$ 收敛，并且其范数平方等于各个向量范数平方之和。无穷维的勾股定理不需要从头开始仔细证明——它可以通过范数的连续性和完备性从有限情况推导出来。

Hilbert 空间理论的核心是投影定理：对于任何闭子空间 $M \subseteq \mathcal{H}$ 和任何 $x \in \mathcal{H}$，存在唯一的 $m_0 \in M$ 使得 $\|x - m\|$ 在所有 $m \in M$ 中最小化，并且这个最小化点由正交条件 $(x - m_0) \perp M$ 刻画。空间可以分解为 $\mathcal{H} = M \oplus M^\perp$，其中 $M^\perp = \{y \in \mathcal{H} : \langle y, m \rangle = 0 \text{ 对所有 } m \in M\}$。

为什么最近点性质有效——以及为什么需要完备性。取一个最小化序列 $(m_n)$，其中 $d_n = \|x - m_n\| \to d = \inf_{m \in M}\|x - m\|$。应用平行四边形法则于 $u = x - m_n$ 和 $v = x - m_k$：得到 $\|u + v\|^2 + \|u - v\|^2 = 2\|u\|^2 + 2\|v\|^2$。由于 $(m_n + m_k)/2 \in M$（凸性），$\|u + v\|/2 = \|x - (m_n+m_k)/2\| \geq d$。代入得：$\|m_n - m_k\|^2 = 2d_n^2 + 2d_k^2 - 4\|x - (m_n+m_k)/2\|^2 \leq 2d_n^2 + 2d_k^2 - 4d^2 \to 0$。因此 $(m_n)$ 是 Cauchy 序列，从而在 $M$ 中收敛（完备性）。如果没有平行四边形法则——在不是 Hilbert 空间的 Banach 空间中——这个论证会崩溃，事实上，在闭子空间中的最近点可能不唯一。

正交投影 $P_M : \mathcal{H} \to M$ 将每个 $x$ 映射到 $M$ 中的最近点。它满足 $P_M^2 = P_M$（投影两次等同于投影一次），$P_M^* = P_M$（投影是自伴的），$\|P_M\| \leq 1$，并且 $\text{Range}(P_M) = M$。反过来，每个有界自伴幂等算子都是其值域上的正交投影。这种闭子空间与自伴幂等算子之间的双射是 Hilbert 空间上算子理论的一个结构支柱。

**实例：Fourier 逼近作为投影。** 在 $L^2[-\pi, \pi]$ 中，设 $M_N = \text{span}\{e^{int} : |n| \leq N\}$——即次数不超过 $N$ 的三角多项式。$f$ 在 $M_N$ 上的正交投影是 Fourier 级数的第 $N$ 部分和：$P_{M_N} f = \sum_{|n| \leq N} \hat{f}(n) e^{int}$，其中 $\hat{f}(n) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t) e^{-int}\,dt$。误差 $f - P_{M_N} f$ 与每个 $|n| \leq N$ 的 $e^{int}$ 正交——这正是误差的 Fourier 系数在 $|n| \leq N$ 时为零的陈述。Fourier 级数的 $L^2$ 逼近理论就是将投影定理应用于嵌套子空间 $M_1 \subset M_2 \subset \cdots$，这些子空间的并集在 $L^2$ 中稠密。

考虑一个具体函数：$f(t) = |t|$ 在 $[-\pi, \pi]$ 上。它的 Fourier 系数是 $\hat{f}(0) = \pi/2$ 和 $\hat{f}(n) = -\frac{2}{\pi n^2}$ 对于奇数 $n$，偶数 $n \neq 0$ 时为零。投影到 $M_1$ 得到 $P_{M_1}f = \frac{\pi}{2} - \frac{2}{\pi}\cos t$——这是 $L^2$ 意义下的一次三角多项式的最佳逼近。误差范数满足 $\|f - P_{M_1}f\|^2 = \|f\|^2 - |c_0|^2 - 2|c_1|^2 = \frac{\pi^2}{3} - \frac{\pi^2}{4} - \frac{8}{\pi^2} \approx 0.81 - 0.81 = 0.014$（近似）。投影定理保证这是所有 $1$、$\cos t$、$\sin t$ 的线性组合中最小的 $L^2$ 误差。

投影定理在非 Hilbert 的 Banach 空间中失效。在带有上确界范数的 $C[0,1]$ 中，最佳多项式逼近（Chebyshev 逼近）存在且唯一，但它由等振荡条件刻画，而不是正交性——这里没有内积。在 $L^1$ 中，闭子空间中的最近点可能根本不唯一。投影定理确实是 Hilbert 空间的一种奢侈特性，它是使 Hilbert 空间成为分析学家乐园的单一特征。

## 正交基、Parseval 等式和分类定理

在 $\mathcal{H}$ 中，一个 **正交系统** 是一组向量 $\{e_\alpha\}_{\alpha \in A}$，满足 $\langle e_\alpha, e_\beta \rangle = \delta_{\alpha\beta}$。如果它的闭线性生成空间等于 $\mathcal{H}$，那么它就是一个 **基**（完备正交系统）。换句话说，如果对所有 $\alpha$ 有 $\langle x, e_\alpha \rangle = 0$，则 $x = 0$。最大正交系统的势称为该空间的 **Hilbert 维数**。

对于 **可分** Hilbert 空间——那些有一个可数稠密子集的空间——Hilbert 维数是可数的，我们可以将基表示为 $(e_n)_{n=1}^\infty$。基础的分类定理说：每个无限维可分 Hilbert 空间都与 $\ell^2$ 同构同距。本质上只有一个可分 Hilbert 空间，一旦选择了一个基，在 $L^2$、Sobolev 空间或 Hardy 空间中的所有计算实际上都是在 $\ell^2$ 中进行的。这个空间非常刚性；所有有趣的数学都在作用于它的算子中。

Bessel 不等式和 Parseval 等式。对于任意 $x \in \mathcal{H}$ 和正交系统 $(e_n)$，Fourier 系数 $c_n = \langle x, e_n \rangle$ 满足 Bessel 不等式 $\sum |c_n|^2 \leq \|x\|^2$。当 $(e_n)$ 是一个基时，等号成立——这就是 **Parseval 等式**：$\|x\|^2 = \sum_{n=1}^\infty |c_n|^2$，并且展开式 $x = \sum c_n e_n$ 在范数意义下收敛。Parseval 等式是无穷维的勾股定理：向量的范数平方等于其坐标平方和。

Fourier 展开 $x = \sum \langle x, e_n \rangle e_n$ 的收敛是无条件的——无论项的顺序如何，它都会收敛。这是 Bessel 不等式的结果：由于 $\sum |c_n|^2 < \infty$，对于任何 $\varepsilon > 0$，除了有限多个项外，其余项都有 $|c_n|^2 < \varepsilon$。部分和形成 Cauchy 列，完备性保证了收敛。

**Gram-Schmidt 过程** 从线性无关集合构造正交系统。从 $\{v_1, v_2, \ldots\}$ 开始，设 $e_1 = v_1/\|v_1\|$，然后迭代地 $e_n = (v_n - \sum_{k<n}\langle v_n, e_k\rangle e_k)/\|...\|$。每一步过程中，生成空间保持不变：$\text{span}\{e_1, \ldots, e_n\} = \text{span}\{v_1, \ldots, v_n\}$。

**实例：经典的正交基。** 在 $L^2[-1,1]$ 中，多项式 $\{1, t, t^2, \ldots\}$ 是线性无关但不正交的。Gram-Schmidt 过程产生（归一化后）Legendre 多项式：$P_0 = 1/\sqrt{2}$，$P_1 = t\sqrt{3/2}$，$P_2 = (3t^2-1)\sqrt{5/8}$。指数函数 $\{e^{int}/(2\pi)^{1/2}\}_{n \in \mathbb{Z}}$ 形成 $L^2[-\pi,\pi]$ 的 Fourier 基。Hermite 函数 $h_n(x) = c_n H_n(x) e^{-x^2/2}$ 形成 $L^2(\mathbb{R})$ 的正交基——它们是量子谐振子的特征函数。每个基针对不同的问题：Legendre 用于区间上的多项式逼近，Fourier 用于周期现象，Hermite 用于 Schrodinger 方程的二次势。

Bessel 不等式提供了一个实用的完备性测试。如果 $(e_n)$ 是一个正交系统，并且想验证它是否是一个基，可以检查 Parseval 等式是否对所有 $x$ 成立——等价于 $\|x - \sum_{n=1}^N \langle x, e_n\rangle e_n\| \to 0$ 是否对所有 $x$ 成立。对于 $L^2[-\pi,\pi]$ 上的 Fourier 系统，这正是 Fourier 级数对每个 $L^2$ 函数在 $L^2$ 范数意义下收敛到该函数的陈述——这是 Riesz 和 Fischer 在 1907 年证明的一个定理，也是 Lebesgue 积分理论早期的重要成果之一。

收敛不必逐点进行——存在一些 $L^2$ 函数，其 Fourier 级数在某些点发散（Carleson 定理给出了几乎处处逐点收敛，但这是一个更难的结果）。$L^2$ 范数意义下的收敛是一个不同且较弱的陈述：它说 $f$ 与其第 $N$ 部分和之间的 $L^2$ 范数趋于零。这正是 Parseval 等式给出的，并且是 Hilbert 空间理论所保证的。

可分性的一个非平凡结果是：$L^2[0,1]$ 和 $\ell^2$ 是同构同距的。$L^2[0,1]$ 的任何正交基 $(e_n)$ 定义了一个同构 $U: L^2 \to \ell^2$，通过 $Uf = (\langle f, e_n\rangle)_{n=1}^\infty$。映射 $U$ 保持内积（由 Parseval 等式），并且是满射（给定 $(c_n) \in \ell^2$，级数 $\sum c_n e_n$ 在 $L^2$ 中收敛）。因此，Fourier 基、Legendre 基和 Haar 小波基都提供了同一个抽象 Hilbert 空间的不同“坐标”。基的选择是一个建模决策，而不是数学上的决定。

## Riesz 表示定理及其影响

Hilbert 空间中最重要的结构性定理是 Riesz 表示定理（Riesz-Frechet）：每个连续线性泛函 $\varphi : \mathcal{H} \to \mathbb{C}$ 都可以表示为 $\varphi(x) = \langle x, y_\varphi \rangle$ 的形式，其中 $y_\varphi \in \mathcal{H}$ 唯一，并且 $\|\varphi\| = \|y_\varphi\|$。

*证明*。如果 $\varphi = 0$，取 $y_\varphi = 0$。否则，$M = \ker(\varphi)$ 是一个闭的超平面（余维数为1）。其正交补 $M^\perp$ 是一维的；选取 $z \in M^\perp$ 使得 $\varphi(z) = 1$（归一化）。对于任意 $x \in \mathcal{H}$，写成 $x = (x - \varphi(x) z) + \varphi(x) z$。第一项在 $M$ 中（检查：$\varphi(x - \varphi(x)z) = \varphi(x) - \varphi(x) = 0$）。因此 $\langle x, z \rangle = \langle \varphi(x) z, z \rangle = \varphi(x)\|z\|^2$，从而 $\varphi(x) = \langle x, z/\|z\|^2 \rangle$。设 $y_\varphi = z/\|z\|^2$。唯一性：如果 $\langle x, y_1 \rangle = \langle x, y_2 \rangle$ 对所有 $x$ 成立，则 $\langle x, y_1 - y_2 \rangle = 0$ 对所有 $x$ 成立，所以 $y_1 = y_2$（取 $x = y_1 - y_2$）。$\square$

这个定理建立了共轭线性的等距同构 $\mathcal{H}^* \cong \mathcal{H}$。Hilbert 空间是自对偶的：对偶空间（共轭线性地）就是空间本身。这与一般的 Banach 空间相比是一个巨大的简化：$(\ell^1)^* = \ell^\infty$，$(c_0)^* = \ell^1$，$(L^1)^* = L^\infty$。在 Hilbert 空间中，向量和线性泛函之间没有概念上的差距——内积将它们识别出来。

自对偶性立即给出**自反性**：典范嵌入 $J: \mathcal{H} \to \mathcal{H}^{**}$ 是满射。自反性是单位球弱紧性的前提条件（第五篇文章），而弱紧性又是变分方法的引擎。因此链条是：内积 $\Rightarrow$ 自对偶 $\Rightarrow$ 自反性 $\Rightarrow$ 弱紧性 $\Rightarrow$ 能量泛函极小值的存在性。

**应用：Lax-Milgram 定理**。设 $a: \mathcal{H} \times \mathcal{H} \to \mathbb{C}$ 是一个有界的半双线性形式（$|a(u,v)| \leq M\|u\|\|v\|$）并且是强制的（$\text{Re}\,a(u,u) \geq \alpha\|u\|^2$ 对某个 $\alpha > 0$ 成立）。那么对于每一个 $F \in \mathcal{H}^*$，存在唯一的 $u \in \mathcal{H}$ 使得 $a(u,v) = F(v)$ 对所有 $v$ 成立，并且 $\|u\| \leq \|F\|/\alpha$。

*证明概要*。对于固定的 $u$，映射 $v \mapsto a(u,v)$ 是一个有界线性泛函，因此根据 Riesz 定理它等于 $\langle Au, v \rangle$ 对于某个唯一的有界算子 $A$。强制性给出 $\alpha\|u\|^2 \leq \text{Re}\langle Au, u\rangle \leq \|Au\|\|u\|$，因此 $\|Au\| \geq \alpha\|u\|$ ——算子 $A$ 在下方有界。下方有界加上稠密的值域（从强制性再次得出，通过标准论证）给出可逆性。然后 $u = A^{-1}y_F$，其中 $F(\cdot) = \langle \cdot, y_F \rangle$ 由 Riesz 定理得到。

**实例：Dirichlet 问题**。考虑 $-\Delta u = f$ 在有界区域 $\Omega$ 上，且 $u|_{\partial\Omega} = 0$。弱形式：找到 $u \in H^1_0(\Omega)$ 使得 $a(u,v) = \int_\Omega \nabla u \cdot \nabla v = \int_\Omega f v = F(v)$ 对所有 $v \in H^1_0$ 成立。形式 $a$ 是有界的（梯度上的 Cauchy-Schwarz 不等式）并且是强制的（Poincare 不等式：$\|\nabla u\|_{L^2}^2 \geq C\|u\|_{H^1}^2$ 在 $H^1_0$ 上成立）。Lax-Milgram 给出唯一解的存在性。偏微分方程存在性定理是 Riesz 表示定理的一个直接推论。这是椭圆型偏微分方程的标准模式：弱形式化，验证强制性，调用 Lax-Milgram。难点在于 Poincare 不等式和函数空间的设置，而不是抽象的存在性论证。

## 伴随算子与代数 $B(\mathcal{H})$

对于有界算子 $T: \mathcal{H} \to \mathcal{H}$，其**伴随算子** $T^*$ 定义为 $\langle Tx, y \rangle = \langle x, T^*y \rangle$ 对所有 $x, y$ 成立。存在性：对固定的 $y$，函数 $x \mapsto \langle Tx, y \rangle$ 是有界的（由 $\|T\|\|y\|$ 控制），根据 Riesz 表示定理，它等于 $\langle x, z \rangle$ 对某个唯一的 $z$ 成立；定义 $T^*y = z$。映射 $y \mapsto T^*y$ 是线性的，并且 $\|T^*\| = \|T\|$。

基本的结构恒等式是 **$C^*$-恒等式**：$\|T^*T\| = \|T\|^2$。这使得有界算子的代数 $B(\mathcal{H})$ 成为一个 $C^*$-代数——这是第 8 章抽象谱理论和 Gelfand-Naimark 定理的起点。

算子分类，根据它们与伴随算子的关系，类似于有限维矩阵的分类：

- **自伴算子** ($T = T^*$)：谱是实数，不同特征值的特征向量正交。量子力学中用于描述可观测量。例子：乘法算子 $M_f$，其中 $f$ 是实值函数；适当区域上的 Laplacian $-\Delta$。
- **酉算子** ($T^*T = TT^* = I$)：等距同构，谱在单位圆上。傅里叶变换 $\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R})$。量子力学中的时间演化算子 $e^{-iHt}$。
- **正规算子** ($T^*T = TT^*$)：允许谱分解的最大类。包括自伴算子和酉算子作为特殊情况。
- **正算子** ($T = T^*$ 且 $\langle Tx, x\rangle \geq 0$)：谱在 $[0,\infty)$ 内，有一个唯一的正平方根 $T^{1/2}$。
- **投影算子** ($P = P^2 = P^*$)：到闭子空间的正交投影。谱 $\subseteq \{0, 1\}$。

恒等式 $\ker(T^*) = \text{Range}(T)^\perp$ 连接了核和值域。取正交补：$\overline{\text{Range}(T)} = \ker(T^*)^\perp$。这是算子理论中的秩-零度定理，在 Fredholm 理论（第 7 章）中至关重要。

**实例：移位算子。** 右移算子 $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$ 在 $\ell^2$ 上的伴随算子是 $S^*(x_1, x_2, \ldots) = (x_2, x_3, \ldots)$（左移）。验证：$\langle Sx, y \rangle = \sum_{n \geq 2} x_{n-1}\overline{y_n} = \sum_{n \geq 1} x_n\overline{y_{n+1}} = \langle x, S^*y \rangle$。现在 $S^*S = I$（先右移再左移恢复原向量），但 $SS^*x = (0, x_2, x_3, \ldots) \neq x$ 一般不成立。移位算子是一个等距映射（$\|Sx\| = \|x\|$），但不是酉算子（不是满射）。这在有限维空间是不可能的，因为 $\mathbb{C}^n$ 到自身的等距映射自动是满射。移位算子是无限维空间中这种性质失效的经典例子。

**实例：Volterra 算子。** 定义 $Vf(t) = \int_0^t f(s)\,ds$ 在 $L^2[0,1]$ 上。通过 Fubini 定理计算伴随算子：$\langle Vf, g\rangle = \int_0^1 g(t)\int_0^t f(s)\,ds\,dt = \int_0^1 f(s)\int_s^1 g(t)\,dt\,ds = \langle f, V^*g\rangle$，所以 $V^*g(s) = \int_s^1 g(t)\,dt$。算子 $V$ 不是自伴的（$V \neq V^*$），也不是正规的，并且谱为 $\{0\}$——它是拟幂零的。然而 $V \neq 0$。这表明紧算子（第 7 章）可以有平凡谱而不为零，这对自伴算子是不可能的。

**极分解** $T = U|T|$，其中 $|T| = (T^*T)^{1/2}$ 且 $U$ 是部分等距映射，将矩阵的奇异值分解扩展到无限维。对于 Volterra 算子，这给出了一个分解，一部分是正算子（捕捉 $V$ 的“拉伸程度”），另一部分是等距映射（捕捉“方向”）。

## 弱收敛、张量积和直和

如果对 $\mathcal{H}$ 中的每个 $y$，都有 $\langle x_n, y \rangle \to \langle x, y \rangle$，那么序列 $(x_n) \subset \mathcal{H}$ 弱收敛到 $x$，记作 $x_n \rightharpoonup x$。根据 Riesz 表示定理，这等价于对所有有界泛函的收敛。弱收敛比范数收敛弱得多：$\ell^2$ 中的标准基 $(e_n)$ 满足 $e_n \rightharpoonup 0$（因为对任何 $y \in \ell^2$，$\langle e_n, y \rangle = y_n \to 0$，由于 $\sum|y_n|^2 < \infty$ 导致 $y_n \to 0$），但 $\|e_n\| = 1 \not\to 0$。

Hilbert 空间中的 Banach-Alaoglu 定理：每个有界序列都有一个弱收敛子序列。这是无穷维空间中 Bolzano-Weierstrass 定理的替代品。单位球在范数拓扑下不紧，但在弱拓扑下是紧的——这种弱紧性是变分方法的核心。为了找到能量泛函的极小值点，可以从极小化序列中提取一个弱收敛子序列，并证明该泛函在弱拓扑下下半连续。

Hilbert 空间的一个有用性质（Radon-Riesz 性质）：$x_n \rightharpoonup x$ 和 $\|x_n\| \to \|x\|$ 一起意味着 $\|x_n - x\| \to 0$。证明很简单：$\|x_n - x\|^2 = \|x_n\|^2 - 2\text{Re}\langle x_n, x\rangle + \|x\|^2 \to \|x\|^2 - 2\|x\|^2 + \|x\|^2 = 0$。这提供了一个将弱收敛升级为强收敛的简洁标准。

另一个重要的弱收敛例子：在 $L^2[0, 2\pi]$ 中，序列 $f_n(t) = \sin(nt)$ 弱收敛到零。Riemann-Lebesgue 引理表明，对每个 $g \in L^2$，都有 $\int_0^{2\pi} g(t)\sin(nt)\,dt \to 0$。但 $\|f_n\|_2 = \sqrt{\pi}$ 对所有 $n$ 都成立。快速振荡在与每个固定的测试函数平均时消失，但能量保持不变——这就是“弱收敛而无强收敛”的物理表现。快速振荡在弱拓扑下不可见；只有振幅包络重要。

内积在范数拓扑下联合连续：$x_n \to x$ 和 $y_n \to y$ 意味着 $\langle x_n, y_n\rangle \to \langle x, y\rangle$。但在弱拓扑下，它只是分别连续：$x_n \rightharpoonup x$ 给出 $\langle x_n, y\rangle \to \langle x, y\rangle$ 对固定 $y$ 成立，但如果 $y_n \rightharpoonup y$，则 $\langle x_n, y_n\rangle$ 不一定收敛到 $\langle x, y\rangle$。反例：$\ell^2$ 中的 $x_n = y_n = e_n$ 给出 $\langle e_n, e_n\rangle = 1$ 但 $\langle 0, 0\rangle = 0$。这种联合弱连续性的失败是非线性偏微分方程中的常见困难来源——需要额外的紧性或补偿紧性论证来处理弱收敛序列的乘积。

**直和。** 正交直和 $\mathcal{H}_1 \oplus \mathcal{H}_2$ 的元素是 $(x_1, x_2)$，其内积定义为 $\langle (x_1,x_2), (y_1,y_2)\rangle = \langle x_1,y_1\rangle_1 + \langle x_2,y_2\rangle_2$。分解 $\mathcal{H} = M \oplus M^\perp$ 是典型的例子。相对于直和分解的算子块矩阵表示是谱理论的基础：当一个正规算子表示为谱子空间上的乘法算子的直和时，它就被“对角化”了。

**张量积。** 完备张量积 $\mathcal{H}_1 \otimes \mathcal{H}_2$ 通过在简单张量上定义 $\langle x_1 \otimes x_2, y_1 \otimes y_2\rangle = \langle x_1, y_1\rangle\langle x_2, y_2\rangle$ 并通过线性和完备性扩展来构建。基本识别：$\ell^2 \otimes \ell^2 \cong \ell^2(\mathbb{N} \times \mathbb{N})$ 和 $L^2(\Omega_1) \otimes L^2(\Omega_2) \cong L^2(\Omega_1 \times \Omega_2)$。在量子力学中，双系统状态空间是 $\mathcal{H}_A \otimes \mathcal{H}_B$；纠缠意味着状态不能写成简单张量 $\psi_A \otimes \psi_B$，而需要真正的和 $\sum c_k \psi_A^{(k)} \otimes \psi_B^{(k)}$。张量积结构也是为什么多变量傅里叶分析可以简化为迭代的一维变换：$L^2(\mathbb{R}^d) \cong L^2(\mathbb{R})^{\otimes d}$ 上的傅里叶变换是一维变换的张量积。

**实例。** 在 $L^2(\mathbb{R}^2)$ 中，高斯函数 $f(x,y) = e^{-(x^2+y^2)/2}$ 可以分解为 $f_1 \otimes f_2$，其中 $f_1(x) = f_2(x) = e^{-x^2/2}$。其 $L^2(\mathbb{R}^2)$ 范数平方是 $\int e^{-(x^2+y^2)}\,dx\,dy = \pi$。张量范数给出 $\|f_1\|^2 \cdot \|f_2\|^2 = \sqrt{\pi} \cdot \sqrt{\pi} = \pi$。函数 $g(x,y) = xe^{-(x^2+y^2)/2}$ 不能分解——它需要展开为 $g = (\text{某个关于 }x\text{ 的函数}) \otimes (\text{某个关于 }y\text{ 的函数})$，这是不可能的，因为 $g(x,y)/g(x',y)$ 通常依赖于 $y$。$L^2(\mathbb{R}^2)$ 中不可分解的函数对应于量子解释中的“纠缠”态。

## 再生核希尔伯特空间

再生核希尔伯特空间（RKHS）定义在集合 $\Omega$ 上，是一个函数 $f: \Omega \to \mathbb{C}$ 的希尔伯特空间 $\mathcal{H}$。在这个空间中，点评估是有界的：对每个 $x \in \Omega$，有 $|f(x)| \leq C_x \|f\|_{\mathcal{H}}$。根据里斯表示定理，每个评估泛函 $\delta_x$ 可以由 $\mathcal{H}$ 中的一个元素 $K_x$ 表示：$f(x) = \langle f, K_x\rangle$。两变量函数 $K(x,y) = K_x(y) = \langle K_x, K_y\rangle$ 就是再生核。

并不是所有的函数希尔伯特空间都是 RKHS。例如，$L^2[0,1]$ 不是 RKHS，因为它的元素是等价类，对于单个点 $x$，$f(x)$ 是未定义的。但 $H^1[0,1]$（有界区间上的 Sobolev 空间）是 RKHS：根据 Sobolev 嵌入定理，$H^1[0,1] \hookrightarrow C[0,1]$ 连续嵌入，所以点评估是有界的。其核与 $-d^2/dx^2 + 1$ 的格林函数有关。

Moore-Aronszajn 定理提供了逆命题：每个正定核 $K$（即对所有有限选择，$\sum_{i,j} c_i\overline{c_j} K(x_i, x_j) \geq 0$）确定一个唯一的 RKHS，其再生核为 $K$。构造方法是：RKHS 是 $\text{span}\{K_x : x \in \Omega\}$ 在内积 $\langle K_x, K_y\rangle = K(x,y)$ 下的完备化。

机器学习中的核技巧直接利用了这一理论。给定一个正定核 $K$，特征映射 $\Phi: x \mapsto K_x$ 将数据嵌入到 RKHS 中。特征空间中的内积计算为 $\langle \Phi(x), \Phi(y)\rangle = K(x,y)$ —— 无需显式构建可能无限维的特征空间。支持向量机在 RKHS 中找到一个超平面，这对应于原始空间中的非线性决策边界。高斯过程回归在 RKHS 中计算条件期望。核主成分分析在特征空间中进行主成分分析。这些都是希尔伯特空间中的线性方法，通过再生性质变得计算上可行。

举个例子。$\mathbb{R}^d$ 上的高斯 RBF 核 $K(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$ 定义了一个无限维的光滑函数 RKHS。对于数据集 $\{x_1, \ldots, x_n\} \subset \mathbb{R}^d$，Gram 矩阵 $G_{ij} = K(x_i, x_j)$ 是正定的（由 Moore-Aronszajn 定理保证）。SVM 通过仅使用 $G$ 来求解二次规划问题——从未显式触及无限维特征空间。希尔伯特空间几何（投影、正交、最近点）在幕后运作，由 $n \times n$ 的核矩阵协调。

另一个重要例子是 Bargmann-Fock 空间，它包含形式为 $f(z) = \sum a_n z^n$ 的整函数，并且满足 $\|f\|^2 = \sum n! |a_n|^2 < \infty$。这个空间的核是 $K(z,w) = e^{z\overline{w}}$。Bargmann-Fock 空间出现在量子光学（相干态）、复几何（Bergman 核）和随机矩阵理论（高斯解析函数的相关函数）中。再生性质 $f(z) = \langle f, K_z\rangle$ 和 $K_z(w) = e^{\overline{z}w}$ 给出了恒等分解：$\frac{1}{\pi}\int |K_z\rangle\langle K_z| e^{-|z|^2}\,dA(z) = I$，这是相干态的过完备关系。

## 谱预览及为何算子比空间更重要

每个可分的 Hilbert 空间都是 $\ell^2$，所以空间本身不携带任何信息——它只是一块空白画布。所有有趣的数学都体现在画在上面的算子上。有限维 Hilbert 空间上的自共轭算子 $T$ 有一个正交基，其特征向量具有实特征值——这是线性代数中的谱定理。在无限维情况下，对于一般的算子，特征值/特征向量图景会崩溃，但仍有残余：谱。

最简单的无限维例子说明了这种崩溃：$L^2[0,1]$ 上的乘法算子 $Mf(t) = tf(t)$。它是自共轭的（$\langle Mf, g\rangle = \int tf\overline{g} = \langle f, Mg\rangle$），其谱为 $[0,1]$，但它没有任何特征向量。方程 $tf(t) = \lambda f(t)$ 强制 $f(t) = 0$ 对于 $t \neq \lambda$ 成立，因此 $f = 0$ 在 $L^2$ 中。谱是“纯连续的”——$[0,1]$ 中的每一点都是谱值，但没有一个是特征值。第 8 部分会解释如何从满足 $M = \int_0^1 t\,dE(t)$ 的谱测度 $E$ 中读取这一点，其中 $E([a,b])f = \mathbf{1}_{[a,b]}(t)f(t)$ 投影到支持在 $[a,b]$ 上的函数。

另一方面，对于紧自共轭算子，谱的行为类似于矩阵的：它由仅在零处累积的特征值组成，每个特征值都有一个有限维的特征空间，可能还包括零本身。具有对称连续核 $k$ 的积分算子 $Kf(t) = \int_0^1 k(t,s)f(s)\,ds$ 是紧且自共轭的，其谱定理给出 $Kf = \sum \lambda_n \langle f, e_n\rangle e_n$ 且 $\lambda_n \to 0$——算子分解成一系列秩一投影，就像一个对角矩阵，其元素趋于零。第 7 部分会详细展开这一点。

概念层次如下：第 3 部分（这一部分）搭建舞台。第 4-6 部分开发工具（对偶、弱拓扑、有界算子）。第 7 部分将“几乎”生活在有限维中的算子（紧算子）通过谱机器处理。第 8 部分处理一般情况，需要测度论的谱理论。空间 $\mathcal{H}$ 始终不变——它总是伪装成 $\ell^2$。算子发生变化，它们的谱性质编码了物理、几何和分析。

最后谈谈为什么这在计算上很重要。每个偏微分方程的数值方法——有限元、谱方法、Galerkin 近似——都是投影方法：通过将解投影到有限维子空间 $V_h \subset \mathcal{H}$ 来近似解。误差是 $\|u - P_{V_h}u\|$，根据投影定理等于 $u$ 到 $V_h$ 的距离。逼近理论（Jackson 定理、Bramble-Hilbert 引理）用 $u$ 的正则性和网格大小 $h$ 估计这个距离。整个有限元收敛理论就是 Hilbert 空间几何——投影、正交和 Cea 引理（该引理说 Galerkin 解是在强制性下的准最优投影）。一旦把有限元理论看作 $H^1$ 中的投影理论，收敛率就不再神秘，而是多项式子空间逼近性质的结果。

## 接下来

Hilbert 空间是分析学家的天堂——自对偶、自反，几何结构忠实于有限维空间。但大多数分析问题并不在这个天堂里。$L^p$ 空间在 $p \neq 2$ 时，连续函数空间，测度空间——这些都是 Banach 空间但不是 Hilbert 空间。下一篇文章会问：在一个一般的 Banach 空间中，怎么知道存在足够多的连续线性泛函？答案是 Hahn-Banach 定理，这是泛函分析中仅次于 Riesz 定理最常用的定理。

---

*这是 [泛函分析](/en/series/functional-analysis/) 系列（共 12 篇文章）的第 3 部分。*

*上一篇：[第 2 部分 —— 赋范空间和 Banach 空间](/en/functional-analysis/02-normed-and-banach/)*

*下一篇：[第 4 部分 —— 对偶空间和 Hahn-Banach 定理](/en/functional-analysis/04-dual-spaces-hahn-banach/)*