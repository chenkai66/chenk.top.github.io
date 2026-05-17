---
title: "泛函分析（四）：对偶空间与 Hahn-Banach 定理"
date: 2021-10-07 09:00:00
tags:
  - functional-analysis
  - dual-spaces
  - hahn-banach
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "functional-analysis-4"
description: "对偶空间的定义与经典例子，Hahn-Banach 定理的完整证明（解析形式与几何形式），自反空间与 James 定理。"
---

线性泛函是泛函分析中最基本的对象之一。一个赋范空间上的连续线性泛函——即从空间到标量域的连续线性映射——看起来平淡无奇，但所有这些泛函集合在一起构成的空间，即**对偶空间**，却蕴含着极为丰富的结构。对偶空间不仅为我们提供了"测量"原空间元素的工具，还为弱拓扑、自反性、以及整个凸分析理论奠定了基础。

本文的核心是 **Hahn-Banach 定理**。这个定理告诉我们：赋范空间上的连续线性泛函总是"足够多"的。具体地说，定义在子空间上的连续线性泛函可以延拓到全空间而不增加范数。这个看似简单的延拓性质，其推论却触及泛函分析的每一个角落。

## 对偶空间的定义

设 $X$ 是 $\mathbb{K}$（$\mathbb{R}$ 或 $\mathbb{C}$）上的赋范空间。$X$ 的**对偶空间** $X^*$ 定义为所有连续线性泛函 $f: X \to \mathbb{K}$ 的集合，赋以算子范数：
$$
\|f\| = \sup_{\|x\| \leq 1} |f(x)| = \sup_{\|x\|=1} |f(x)| = \sup_{x \neq 0} \frac{|f(x)|}{\|x\|}.
$$

由于 $\mathbb{K}$ 是完备的，$X^*$ 总是 Banach 空间——即使 $X$ 本身不完备。这是对偶空间的一个重要优势：无论原空间多么"粗糙"，对偶空间总是完备的。

**对偶空间为什么重要？** 考虑一个类比：实数域 $\mathbb{R}$ 上的线性函数 $f(x) = ax$ 完全由斜率 $a$ 决定。类似地，赋范空间上的连续线性泛函可以看作"广义坐标"——它们从不同角度"投影"空间中的元素。对偶空间收集了所有可能的投影方向，如果这些投影方向足够丰富（这正是 Hahn-Banach 定理所保证的），我们就能通过对偶空间来完全理解原空间。


![赋范空间中的 Hahn-Banach 分离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_fig2_projection.png)

## 经典例子：$\ell^p$ 的对偶

最干净的例子是序列空间。设 $1 \leq p < \infty$，$1/p + 1/q = 1$（当 $p=1$ 时约定 $q=\infty$）。

**定理.** $(\ell^p)^* \cong \ell^q$，即 $\ell^p$ 的对偶空间等距同构于 $\ell^q$。

**证明思路.** 分两步进行。

*第一步：每个 $y \in \ell^q$ 定义一个 $\ell^p$ 上的连续线性泛函。* 对 $y = (y_n) \in \ell^q$，定义 $f_y: \ell^p \to \mathbb{K}$：
$$
f_y(x) = \sum_{n=1}^\infty x_n y_n.
$$
Holder 不等式保证该级数绝对收敛，且 $|f_y(x)| \leq \|x\|_p \|y\|_q$，因此 $f_y \in (\ell^p)^*$ 且 $\|f_y\| \leq \|y\|_q$。

取 $x_n = |y_n|^{q-2} \overline{y_n}$（当 $y_n \neq 0$ 时），可验证 $\|x\|_p^p = \|y\|_q^q$，且 $f_y(x) = \|y\|_q^q$。由此得 $\|f_y\| = \|y\|_q$。

*第二步：每个 $f \in (\ell^p)^*$ 都有这种形式。* 设 $e_n = (0,\ldots,0,1,0,\ldots)$（第 $n$ 位为 1）。令 $y_n = f(e_n)$。对任意有限支撑的 $x = \sum_{n=1}^N x_n e_n$，有 $f(x) = \sum x_n y_n$。需要证明 $y = (y_n) \in \ell^q$。

当 $1 < p < \infty$ 时：取 $x_n^{(N)} = |y_n|^{q-2}\overline{y_n}$ 对 $n \leq N$，$x_n^{(N)} = 0$ 对 $n > N$。则
$$
\sum_{n=1}^N |y_n|^q = f(x^{(N)}) \leq \|f\| \cdot \|x^{(N)}\|_p = \|f\| \cdot \left(\sum_{n=1}^N |y_n|^q\right)^{1/p}.
$$
化简得 $\left(\sum_{n=1}^N |y_n|^q\right)^{1/q} \leq \|f\|$。令 $N \to \infty$，得 $\|y\|_q \leq \|f\|$。

当 $p = 1$ 时：$|y_n| = |f(e_n)| \leq \|f\| \cdot \|e_n\|_1 = \|f\|$，因此 $\|y\|_\infty \leq \|f\|$。结合密度论证可得 $f = f_y$。

**其他经典例子：**
- $(c_0)^* \cong \ell^1$：$c_0$ 的对偶是 $\ell^1$，证明与上面类似。
- $(L^p(\mu))^* \cong L^q(\mu)$（$1 \leq p < \infty$）：这是 Radon-Nikodym 定理的推论。
- $(C[a,b])^*$：由 Riesz 表示定理，$C[a,b]$ 的对偶等距同构于 $[a,b]$ 上的有符号正则 Borel 测度空间。这个结果远比序列空间的情形深刻。

## Hahn-Banach 定理：解析形式

Hahn-Banach 定理是泛函分析三大基本定理之一（另两个是一致有界性原理和开映射定理），也可以说是最基本的一个，因为它不需要完备性假设。

**定理（Hahn-Banach，实情形）.** 设 $X$ 是实线性空间，$p: X \to \mathbb{R}$ 是次线性泛函（即满足 $p(\alpha x) = \alpha p(x)$ 对 $\alpha > 0$ 以及 $p(x+y) \leq p(x) + p(y)$）。设 $M \subseteq X$ 是线性子空间，$f: M \to \mathbb{R}$ 是线性泛函，满足 $f(x) \leq p(x)$ 对所有 $x \in M$。则存在线性泛函 $F: X \to \mathbb{R}$ 使得 $F|_M = f$ 且 $F(x) \leq p(x)$ 对所有 $x \in X$。

**完整证明.** 分两步。

*第一步：一维延拓。* 设 $M \subsetneq X$，取 $x_0 \in X \setminus M$。令 $M_1 = M + \mathbb{R}x_0 = \\{x + tx_0 : x \in M, t \in \mathbb{R}\\}$。我们要找 $c \in \mathbb{R}$ 使得 $F_1(x + tx_0) = f(x) + tc$ 满足 $F_1 \leq p$ on $M_1$。

$F_1 \leq p$ 要求：对所有 $x \in M$ 和 $t > 0$，
$$
f(x) + tc \leq p(x + tx_0) \implies c \leq p(x/t + x_0) - f(x/t);
$$
对 $t < 0$（令 $t = -s$，$s > 0$），
$$
f(x) - sc \leq p(x - sx_0) \implies c \geq f(x/s) - p(x/s - x_0).
$$

因此需要
$$
\sup_{y \in M} [f(y) - p(y - x_0)] \leq c \leq \inf_{z \in M} [p(z + x_0) - f(z)].
$$

这个区间非空吗？对任意 $y, z \in M$：
$$
f(y) + f(z) = f(y+z) \leq p(y+z) = p((y - x_0) + (z + x_0)) \leq p(y-x_0) + p(z+x_0),
$$
因此 $f(y) - p(y-x_0) \leq p(z+x_0) - f(z)$。取上确界和下确界，区间确实非空。

*第二步：Zorn 引理。* 考虑所有 $(M', f')$ 的集合 $\mathcal{F}$，其中 $M \subseteq M' \subseteq X$ 是线性子空间，$f': M' \to \mathbb{R}$ 是线性的，$f'|_M = f$，且 $f' \leq p$ on $M'$。以 $(M_1, f_1) \leq (M_2, f_2)$ 当且仅当 $M_1 \subseteq M_2$ 且 $f_2|_{M_1} = f_1$ 为偏序。

$\mathcal{F}$ 非空（$(M, f) \in \mathcal{F}$）。每条链 $\\{(M_\alpha, f_\alpha)\\}$ 有上界：取 $M_\infty = \bigcup M_\alpha$，$f_\infty(x) = f_\alpha(x)$（当 $x \in M_\alpha$），容易验证良定义性和条件。由 Zorn 引理，$\mathcal{F}$ 有极大元 $(M^*, F)$。

若 $M^* \neq X$，由第一步可做一维延拓，矛盾极大性。故 $M^* = X$，$F$ 即为所求延拓。$\square$

**复情形.** 设 $X$ 是复线性空间。若 $f: M \to \mathbb{C}$ 是复线性泛函，令 $u = \operatorname{Re} f$。则 $u$ 是实线性的，且 $f(x) = u(x) - iu(ix)$（这个恢复公式是因为 $f(ix) = if(x)$，取实部即得 $\operatorname{Re}(if(x)) = -\operatorname{Im} f(x)$）。

将实 Hahn-Banach 定理应用于 $u$ 和 $p(x) = \|x\|$（或更一般的次线性泛函），得到实线性延拓 $U$，再定义 $F(x) = U(x) - iU(ix)$。可验证 $F$ 是复线性的且 $|F(x)| \leq p(x)$。

对范数的控制需要一个小技巧：对任意 $x$，取 $\theta$ 使得 $e^{i\theta}F(x) = |F(x)|$，则 $|F(x)| = F(e^{i\theta}x) = U(e^{i\theta}x) \leq p(e^{i\theta}x) = p(x)$。

## Hahn-Banach 定理：几何形式

解析形式的 Hahn-Banach 定理有一个等价的几何表述，它用超平面分离凸集。这个形式在凸分析和优化理论中更为直接。

**Minkowski 泛函.** 设 $X$ 是实线性空间，$C \subseteq X$ 是凸集且 $0 \in \operatorname{int}(C)$。**Minkowski 泛函**（又称规范泛函）定义为
$$
p_C(x) = \inf\\{t > 0 : x/t \in C\\} = \inf\\{t > 0 : x \in tC\\}.
$$

Minkowski 泛函具有次线性性质：$p_C(\alpha x) = \alpha p_C(x)$（$\alpha > 0$）且 $p_C(x+y) \leq p_C(x) + p_C(y)$。如果 $C$ 还是平衡的（$\alpha C \subseteq C$ 对 $|\alpha| \leq 1$），则 $p_C$ 是半范数。如果进一步 $C$ 是有界的，$p_C$ 就是范数。

Minkowski 泛函是连接分析形式和几何形式的桥梁。

**定理（Hahn-Banach，几何形式/分离定理）.** 设 $X$ 是实赋范空间。
1. *（开凸集分离）* 若 $A$ 是非空开凸集，$B$ 是非空凸集，$A \cap B = \emptyset$，则存在 $f \in X^*$ 和 $c \in \mathbb{R}$ 使得 $f(a) < c \leq f(b)$ 对所有 $a \in A$，$b \in B$。
2. *（闭凸集严格分离）* 若 $A$ 是紧凸集，$B$ 是闭凸集，$A \cap B = \emptyset$，则存在 $f \in X^*$ 和 $c_1 < c_2$ 使得 $f(a) \leq c_1 < c_2 \leq f(b)$ 对所有 $a \in A$，$b \in B$。

**证明思路（开凸集情形）.** 取 $a_0 \in A$，$b_0 \in B$。令 $x_0 = b_0 - a_0$ 和 $C = A - B + x_0$。则 $C$ 是开凸集且 $0 \in C$（因为 $a_0 - b_0 + x_0 = 0$），而 $x_0 \notin C$（否则 $A \cap B \neq \emptyset$）。

考虑 Minkowski 泛函 $p_C$。因为 $x_0 \notin C$，有 $p_C(x_0) \geq 1$。在一维子空间 $\mathbb{R}x_0$ 上定义 $g(tx_0) = t$，则 $g(tx_0) \leq p_C(tx_0)$（对 $t \geq 0$，$g = t \leq tp_C(x_0) = p_C(tx_0)$；对 $t < 0$，$g = t < 0 \leq p_C(tx_0)$）。

由 Hahn-Banach 解析形式，$g$ 延拓为 $f: X \to \mathbb{R}$，$f \leq p_C$。由 $C$ 是开集可推出 $f$ 连续（$f(x) \leq p_C(x) < 1$ 对 $x \in C$，故 $f$ 在 $0$ 处有界，线性泛函在一点有界则连续）。最后验证分离性质。

## 重要推论

Hahn-Banach 定理的推论遍布泛函分析。这里列举最核心的几个。

**推论 1（范数实现）.** 对任意赋范空间 $X$ 和任意 $x_0 \in X$，$x_0 \neq 0$，存在 $f \in X^*$ 使得 $\|f\| = 1$ 且 $f(x_0) = \|x_0\|$。

*证明：* 在子空间 $M = \mathbb{K}x_0$ 上定义 $g(\alpha x_0) = \alpha \|x_0\|$。则 $\|g\| = 1$。由 Hahn-Banach 定理延拓到全空间。

这个推论告诉我们 $X^*$ 足够大：它能"看到" $X$ 中的每一个非零元素。

**推论 2（对偶公式）.** 对任意 $x \in X$：
$$
\|x\| = \sup_{\substack{f \in X^* \\\ \|f\| \leq 1}} |f(x)| = \max_{\substack{f \in X^* \\\ \|f\| \leq 1}} |f(x)|.
$$

这意味着一个元素的范数完全由对偶空间中的泛函决定。"max"而非仅仅"sup"是因为推论 1 保证了上确界可以达到。

**推论 3（子空间的泛函延拓保范数）.** 若 $M$ 是 $X$ 的闭子空间，$f \in M^*$，则存在 $F \in X^*$ 使得 $F|_M = f$ 且 $\|F\| = \|f\|$。

**推论 4（对偶空间分离点）.** 若 $x \neq y$，则存在 $f \in X^*$ 使得 $f(x) \neq f(y)$。即 $X^*$ 分离 $X$ 的点。

## Riesz-Markov 表示定理

对偶空间的结构在具体空间上往往有深刻的几何或测度论意义。最著名的例子之一是 Riesz-Markov 表示定理。

**定理（Riesz-Markov）.** 设 $K$ 是紧 Hausdorff 空间。则 $C(K)^*$ 等距同构于 $K$ 上的有符号正则 Borel 测度空间 $M(K)$：每个 $\Lambda \in C(K)^*$ 唯一对应一个 $\mu \in M(K)$，使得
$$
\Lambda(f) = \int_K f \, d\mu \quad \text{对所有 } f \in C(K),
$$
且 $\|\Lambda\| = |\mu|(K)$（总变差）。

这个定理的证明是测度论的经典内容，需要用到 Daniell 积分或 Caratheodory 延拓。它的意义在于：连续函数空间上的线性泛函本质上就是积分——这将抽象的泛函分析与具体的测度论联系了起来。

## 自反空间

对偶空间 $X^*$ 本身也是 Banach 空间，因此有它自己的对偶 $X^{**} = (X^*)^*$，称为**二次对偶**或**双对偶**。

每个 $x \in X$ 自然地定义一个 $X^{**}$ 中的元素：$\hat{x}(f) = f(x)$ 对 $f \in X^*$。映射 $J: X \to X^{**}$，$J(x) = \hat{x}$，称为**典范嵌入**。

**性质：**
- $J$ 是线性的。
- $\|J(x)\| = \|x\|$（由对偶公式，$\|J(x)\| = \sup_{\|f\|\leq 1}|f(x)| = \|x\|$）。因此 $J$ 是等距嵌入。
- $J$ 不一定是满射。

**定义.** 如果典范嵌入 $J: X \to X^{**}$ 是满射（从而是等距同构），则称 $X$ 是**自反的**。

**自反空间的例子：**
- $\ell^p$（$1 < p < \infty$）是自反的：$(\ell^p)^* \cong \ell^q$，$(\ell^q)^* \cong \ell^p$，典范嵌入恰好是这些等距同构的复合。
- $L^p(\mu)$（$1 < p < \infty$）是自反的，理由类似。
- 每个 Hilbert 空间都是自反的（由 Riesz 表示定理，$H^* \cong H$）。

**非自反空间的例子：**
- $\ell^1$ 不是自反的：$(\ell^1)^* \cong \ell^\infty$，但 $(\ell^\infty)^* \supsetneq \ell^1$（$(\ell^\infty)^*$ 包含不能由 $\ell^1$ 序列表示的泛函，例如 Banach 极限）。
- $c_0$ 不是自反的：$(c_0)^* \cong \ell^1$，$(\ell^1)^* \cong \ell^\infty \neq c_0$。
- $L^1$ 和 $L^\infty$ 不是自反的。
- $C[0,1]$ 不是自反的。

## James 定理与非自反性的刻画

自反性有一个优美的等价刻画，这就是 James 定理。

**James 定理.** Banach 空间 $X$ 是自反的当且仅当 $X^*$ 中的每个连续线性泛函都在 $X$ 的闭单位球上取到最大值。换言之，$X$ 自反当且仅当：对每个 $f \in X^*$，存在 $x_0 \in X$，$\|x_0\| \leq 1$，使得 $f(x_0) = \|f\|$。

James 定理的证明相当精巧，是 R.C. James 在 1964 年给出的。其深刻之处在于：它将一个拓扑/代数条件（典范嵌入是满射）转化为一个极值条件（泛函在单位球上取到最大值）。

**$c_0$ 非自反性的直接验证.** 考虑 $f = (1, 1/2, 1/3, \ldots) \in \ell^1 = (c_0)^*$。对任意 $x = (x_n) \in c_0$，$\|x\|_\infty \leq 1$：
$$
|f(x)| = \left|\sum_{n=1}^\infty \frac{x_n}{n}\right| \leq \sum_{n=1}^\infty \frac{|x_n|}{n} \leq \sum_{n=1}^\infty \frac{1}{n} = \infty?
$$

不对——我们需要更仔细地分析。实际上 $\|f\| = \sum 1/n = \infty$，所以这个 $f \notin \ell^1$。让我们换一个更好的例子。

取 $f = (1, 0, 0, \ldots) \in \ell^1$。则 $\|f\| = 1$，且 $f(x) = x_1$。这个泛函当然在 $x = e_1 = (1,0,0,\ldots)$ 处取到最大值，所以这个例子不能说明非自反性。

非自反性的正确论证如下：$(c_0)^{**} = (\ell^1)^* = \ell^\infty$，而典范嵌入 $J: c_0 \to \ell^\infty$ 就是包含映射。由于 $c_0 \subsetneq \ell^\infty$（例如常值序列 $(1,1,1,\ldots) \in \ell^\infty \setminus c_0$），$J$ 不是满射，故 $c_0$ 不自反。

用 James 定理的语言：存在 $g \in (c_0)^* = \ell^1$ 使得 $\sup_{\|x\|_\infty \leq 1} g(x)$ 不能取到。取 $g = (1/2, 1/4, 1/8, \ldots)$，$\|g\|_1 = 1$。对任意 $x \in c_0$，$\|x\|_\infty \leq 1$：
$$
g(x) = \sum_{n=1}^\infty \frac{x_n}{2^n}.
$$
若 $g(x_0) = 1$，则需要 $x_n^{(0)} = 1$ 对所有 $n$，即 $x_0 = (1,1,1,\ldots)$。但这不属于 $c_0$。因此 $\|g\| = 1$ 但 $g$ 不在 $c_0$ 的单位球上取到最大值。

## 自反性的意义

为什么自反性如此重要？因为它保证了许多强大的弱紧性结果。

**定理（Eberlein-Smulian）.** Banach 空间 $X$ 是自反的当且仅当 $X$ 的闭单位球在弱拓扑下是序列紧的。

在自反空间中，每个有界序列都有弱收敛子列。这对变分法（寻找泛函的极值点）和偏微分方程理论（弱解的存在性）至关重要。

**定理（Milman-Pettis）.** 一致凸的 Banach 空间是自反的。

特别地，$L^p$（$1 < p < \infty$）的自反性可以从其一致凸性直接推出（Clarkson 不等式给出一致凸性）。

## 接下来

对偶空间和 Hahn-Banach 定理为我们提供了泛函分析最基本的"软分析"工具。下一篇文章（第五篇）将转向**内积空间与 Hilbert 空间**。Hilbert 空间是所有 Banach 空间中结构最丰富的——内积赋予的几何结构允许我们谈论正交性、投影和 Fourier 展开。Riesz 表示定理将告诉我们 Hilbert 空间的对偶空间就是它自身（因此自然自反），而正交分解定理将成为我们分析 Hilbert 空间上算子的起点。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 4 篇（共 12 篇）。*

*上一篇：[第 3 篇 —— Hilbert 空间](/zh/functional-analysis/03-Hilbert空间/)*

*下一篇：[第 5 篇 —— 弱拓扑](/zh/functional-analysis/05-弱拓扑/)*
