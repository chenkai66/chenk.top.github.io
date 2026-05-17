---
title: "泛函分析（二）：内积空间与 Hilbert 空间"
date: 2021-03-08 09:00:00
tags:
  - functional-analysis
  - hilbert-spaces
  - inner-products
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 6
translationKey: "functional-analysis-2"
description: "Hilbert 空间为无穷维带来几何——正交性、投影、Fourier 级数都住在这里。"
---

## 无穷维空间中的几何

Banach 空间有距离和线性结构，但没有角度的概念。在 $\ell^1$ 或 $C[0,1]$（配上确界范数）里，你没有自然的方式谈论正交性。要获得几何——垂直、投影、最小二乘逼近——你需要内积。

$\mathbb{R}^n$ 的直觉是对的：内积让你定义两个向量的夹角为 $\cos\theta = \langle x, y \rangle / (\|x\|\|y\|)$。当它为零时两个向量正交。把 $x$ 投影到子空间上就是找最近点，而误差垂直于子空间。所有这些都完美推广到无穷维——前提是空间完备。

我觉得从 Banach 到 Hilbert 的跨越很奇妙：只加一条结构公理（平行四边形等式），就获得了如此多的额外力量：正交分解、Riesz 表示、自对偶。内积提供的几何不是奢侈品，它让理论变得可操控。

![正交投影示意图](/images/functional-analysis/fig02_projection.png)

## 内积空间

复向量空间 $X$ 上的**内积**是映射 $\langle \cdot, \cdot \rangle: X \times X \to \mathbb{C}$，满足：

$$\langle x, x \rangle \ge 0, \quad \langle x, x \rangle = 0 \iff x = 0,$$

$$\langle x, y \rangle = \overline{\langle y, x \rangle},$$

$$\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle.$$

每个内积诱导范数：$\|x\| = \sqrt{\langle x, x \rangle}$。但不是每个范数都来自内积——这里故事变得有趣。

**平行四边形等式判据。** 范数 $\|\cdot\|$ 来自某个内积，当且仅当：

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2 \quad \forall\, x, y.$$

这个等式说：平行四边形两条对角线的平方和等于四条边的平方和。如果范数满足它，**极化恒等式**可以恢复内积：

$$\langle x, y \rangle = \frac{1}{4}\left(\|x+y\|^2 - \|x-y\|^2 + i\|x+iy\|^2 - i\|x-iy\|^2\right).$$

**例 1（哪个 $\ell^p$ 有内积？）。** $\ell^2$ 范数满足平行四边形等式。$\ell^1$ 和 $\ell^\infty$ 不满足。取 $x = (1,0,0,\ldots)$，$y = (0,1,0,\ldots)$，在 $\ell^1$ 中：$\|x+y\|_1^2 + \|x-y\|_1^2 = 4 + 4 = 8$，但 $2\|x\|_1^2 + 2\|y\|_1^2 = 2 + 2 = 4$。等式不成立。所有 $\ell^p$ 空间中，只有 $p = 2$ 给出内积空间。$\ell^2$ 在几何上是特殊的。

**例 2。** $L^2[0,1]$ 配内积：

$$\langle f, g \rangle = \int_0^1 f(t)\overline{g(t)}\, dt.$$

这个内积让 Fourier 分析成立。函数 $e^{2\pi i n t}$ 在这个配对下正交。

**例 3。** $H^1[0,1]$ 上的 Sobolev 内积：

$$\langle f, g \rangle_{H^1} = \int_0^1 f\bar{g}\, dt + \int_0^1 f'\bar{g}'\, dt.$$

这在 $L^2$ 的子空间上给出了不同的 Hilbert 空间结构，融入了导数信息。

## Cauchy-Schwarz 不等式

分析学中最重要的不等式：

$$|\langle x, y \rangle| \le \|x\|\,\|y\|,$$

等号成立当且仅当 $x$ 和 $y$ 线性相关。

*证明。* $y = 0$ 时两边都是零。$y \ne 0$ 时，令 $\alpha = \langle x, y \rangle / \|y\|^2$，则：

$$0 \le \|x - \alpha y\|^2 = \|x\|^2 - \frac{|\langle x, y \rangle|^2}{\|y\|^2}.$$

整理得 $|\langle x, y \rangle|^2 \le \|x\|^2\|y\|^2$。等号在 $x = \alpha y$ 时成立。$\square$

这一个不等式就推出：
1. 诱导范数满足三角不等式（所以确实是范数）。
2. 内积关于两个变量连续。
3. 可以定义角度：$\cos\theta = \text{Re}\,\langle x, y \rangle / (\|x\|\|y\|)$。

在 $L^2$ 里，Cauchy-Schwarz 变成 $|\int fg| \le (\int |f|^2)^{1/2}(\int |g|^2)^{1/2}$。在概率论里，它给出 $|\text{Cov}(X,Y)| \le \text{SD}(X)\text{SD}(Y)$。

## Hilbert 空间

**Hilbert 空间**是完备的内积空间：

$$\text{Hilbert 空间} = \text{有内积的 Banach 空间}.$$

$\ell^2$ 和 $L^2[0,1]$ 都是 Hilbert 空间。有限序列空间配 $\ell^2$ 范数是内积空间但不是 Hilbert 空间（不完备）。

**重要的反例。** $\ell^p$（$p \ne 2$）是 Banach 但不是 Hilbert。$C[0,1]$ 配上确界范数是 Banach 但不是 Hilbert（平行四边形等式不成立）。有内积是很特殊的性质。

## 正交性与投影

向量 $x, y$ **正交**（记作 $x \perp y$）如果 $\langle x, y \rangle = 0$。集合 $S$ 是**标准正交**的，如果其元素两两正交且都是单位向量。

Hilbert 空间理论的核心几何定理：

**定理（正交投影）。** 设 $M$ 是 Hilbert 空间 $H$ 的闭子空间。对每个 $x \in H$，存在唯一的 $m_0 \in M$ 使距离最小：

$$\|x - m_0\| = \inf_{m \in M} \|x - m\| =: d(x, M).$$

而且 $m_0$ 由正交条件刻画：$(x - m_0) \perp M$，即 $\langle x - m_0, m \rangle = 0$ 对所有 $m \in M$。

*证明。* 设 $d = \inf_{m \in M} \|x - m\|$。取极小化序列 $(m_n)$，$\|x - m_n\| \to d$。对 $u = m_n - x$，$v = m_k - x$ 用平行四边形等式：

$$\|m_n - m_k\|^2 = 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - \|m_n + m_k - 2x\|^2.$$

因为 $(m_n + m_k)/2 \in M$（子空间！），$\|m_n + m_k - 2x\|^2 \ge 4d^2$。所以：

$$\|m_n - m_k\|^2 \le 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4d^2 \to 0.$$

序列是 Cauchy 的。完备性给出 $m_0 = \lim m_n \in M$（闭子空间）。正交刻画：展开 $\|x - m_0 - tv\|^2 \ge d^2$ 对所有 $t$ 并关于 $t$ 优化。$\square$

这在一般 Banach 空间中失败。$\ell^1$ 中子空间的最近点不一定唯一。平行四边形等式——即内积——对唯一性是本质的。

**推论（正交分解）。** $H = M \oplus M^\perp$，其中 $M^\perp = \{x \in H : \langle x, m \rangle = 0 \ \forall m \in M\}$。

每个 $x$ 唯一分解为 $x = P_M x + (x - P_M x)$，$P_M x \in M$，$(x - P_M x) \in M^\perp$。映射 $P_M: H \to H$ 是 $M$ 上的**正交投影**。它满足 $P_M^2 = P_M$，$P_M^* = P_M$，$\|P_M\| = 1$。

**例 4（$L^2$ 中的最小二乘）。** 要找 $f \in L^2[0,1]$ 的最佳次数 $\le n$ 的多项式逼近，把 $f$ 投影到闭子空间 $M = \{$次数 $\le n$ 的多项式$\}$。投影 $P_M f$ 是唯一使 $\int_0^1 |f - p|^2 dt$ 最小的多项式，误差 $f - P_M f$ 正交于所有次数 $\le n$ 的多项式。

## 标准正交基与 Fourier 级数

$H$ 中的标准正交集 $\{e_\alpha\}_{\alpha \in A}$ 是**标准正交基**（或完备标准正交系），如果与所有 $e_\alpha$ 正交的唯一向量是零向量。等价地：$\{e_\alpha\}$ 的闭线性张成是整个 $H$。

**定理（Bessel 不等式）。** 对任何标准正交集 $\{e_n\}$ 和任何 $x \in H$：

$$\sum_n |\langle x, e_n \rangle|^2 \le \|x\|^2.$$

**定理（Parseval 恒等式）。** 若 $\{e_n\}_{n=1}^\infty$ 是 $H$ 的标准正交基，则对每个 $x \in H$：

$$x = \sum_{n=1}^\infty \langle x, e_n \rangle\, e_n, \quad \|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.$$

系数 $c_n = \langle x, e_n \rangle$ 是 $x$ 关于这组基的 **Fourier 系数**。第一个等式说每个元素由其 Fourier 系数决定；第二个说范数分解为系数平方之和。

*证明思路。* 令 $S_N = \sum_{n=1}^N c_n e_n$。则 $S_N = P_M x$，$M = \text{span}(e_1, \ldots, e_N)$，且 $\|x - S_N\|^2 = \|x\|^2 - \sum_{n=1}^N |c_n|^2$（勾股定理）。$N \to \infty$ 时，基的完备性迫使 $\|x - S_N\| \to 0$。$\square$

**例 5（经典 Fourier 级数）。** $L^2[0, 2\pi]$ 中，$e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$ 构成标准正交基。Parseval 恒等式变为：

$$\frac{1}{2\pi}\int_0^{2\pi} |f(t)|^2\, dt = \sum_{n=-\infty}^{\infty} |\hat{f}(n)|^2.$$

这就是经典的 Parseval 定理。这组系统的完备性（指数函数构成基）是一个深刻的定理——等价于三角多项式在 $L^2$ 中稠密。

**例 6（Haar 小波）。** $[0,1]$ 上的 Haar 系统是 $L^2[0,1]$ 的另一组标准正交基，有不同的逼近性质：Haar 部分和对有跳跃的函数收敛好，而 Fourier 部分和会出现 Gibbs 现象。同样的抽象框架，不同的具体基，不同的应用。

## Riesz 表示定理

**定理（Riesz-Frechet）。** Hilbert 空间上每个连续线性泛函 $f: H \to \mathbb{C}$ 都有形式 $f(x) = \langle x, y \rangle$，$y \in H$ 唯一，且 $\|f\| = \|y\|$。

*证明。* $f = 0$ 时取 $y = 0$。否则 $\ker f$ 是余维为 1 的闭子空间（因为 $f$ 连续所以核闭）。正交分解给出 $H = \ker f \oplus (\ker f)^\perp$，$(\ker f)^\perp$ 是一维的。取 $z \in (\ker f)^\perp$，$f(z) = 1$。对任何 $x \in H$：

$$x = \underbrace{(x - f(x)z)}_{\in \ker f} + f(x)z.$$

和 $z$ 做内积：$\langle x, z \rangle = f(x)\|z\|^2$，所以 $f(x) = \langle x, z/\|z\|^2 \rangle$。令 $y = z/\|z\|^2$。由 Cauchy-Schwarz，$\|f\| = \|y\|$（在 $x = y$ 处等号成立）。$\square$

这建立了 $H^* \cong H$（反线性同构）。Hilbert 空间的对偶就是自身。没有其他 Banach 空间有这种自对偶性质（有限维除外）。这就是为什么 Hilbert 空间是最"好"的无穷维空间——你不需要离开空间就能谈论泛函。

**推论。** 每个有界半双线性形式 $a: H \times H \to \mathbb{C}$ 都可以写成 $a(x,y) = \langle Ax, y \rangle$，$A$ 是唯一的有界算子。这把形式和算子联系起来——是 Lax-Milgram 定理的起点（第 6 篇）。

## 可分性与分类

Hilbert 空间是**可分**的，如果它有可数标准正交基（等价地：有可数稠密子集）。

**定理。** 所有可分的无穷维 Hilbert 空间都等距同构于 $\ell^2$。

*证明。* 如果 $\{e_n\}$ 是可数标准正交基，映射 $x \mapsto (\langle x, e_n \rangle)_{n=1}^\infty$ 就是从 $H$ 到 $\ell^2$ 的等距同构（由 Parseval）。$\square$

这很惊人。$L^2[0,1]$、$L^2(\mathbb{R})$、Hardy 空间 $H^2(\mathbb{D})$、Sobolev 空间 $H^1[0,1]$——作为 Hilbert 空间它们都和 $\ell^2$ 同构。差异体现在哪些算子自然作用于其上、哪些子空间对应有意义的函数类、哪些基在计算上有用。

不可分 Hilbert 空间存在（如不可数 $A$ 上的 $\ell^2(A)$），但在应用中很少出现。本系列中"Hilbert 空间"默认指可分的。

## 接下来的内容

有了 Hilbert 空间几何，下一步是研究空间之间的映射：有界线性算子。它们是矩阵的无穷维推广，行为方式会让习惯有限维线性代数的人震惊。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 2 篇，共 6 篇。
上一篇: [第 1 篇 — 度量空间、赋范空间与 Banach 空间](/zh/functional-analysis/01-度量空间与赋范空间/) · 下一篇: [第 3 篇 — 有界线性算子与泛函](/zh/functional-analysis/03-有界算子/)*
