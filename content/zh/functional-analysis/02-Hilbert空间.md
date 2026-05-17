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
description: "Hilbert 空间把几何带入无穷维——正交性、投影和 Fourier 级数都住在这里。"
---

## 从距离到角度

Banach 空间有距离和线性结构，但没有角度的概念。在 $\ell^1$ 或 $C[0,1]$（上确界范数）中谈正交毫无意义。要获得几何——垂直、投影、最小二乘——需要内积。

## 内积空间的定义

复向量空间 $X$ 上的**内积**是映射 $\langle \cdot, \cdot \rangle: X \times X \to \mathbb{C}$，满足：

$$\langle x, x \rangle \ge 0, \quad \langle x, x \rangle = 0 \iff x = 0,$$
$$\langle x, y \rangle = \overline{\langle y, x \rangle},$$
$$\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle.$$

内积诱导范数 $\|x\| = \sqrt{\langle x, x \rangle}$。但不是每个范数都来自内积。

**平行四边形恒等式判别法。** 范数 $\|\cdot\|$ 来自内积当且仅当：

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2 \quad \forall\, x, y.$$

$\ell^2$ 范数满足这个等式，$\ell^1$ 和 $\ell^\infty$ 范数不满足。所以在所有 $\ell^p$ 空间中，只有 $\ell^2$ 是内积空间。这就是 $\ell^2$ 特殊的原因。

**例 1。** $\ell^2$ 配内积 $\langle x, y \rangle = \sum_{n=1}^\infty x_n \overline{y_n}$。

**例 2。** $L^2[0,1]$ 配内积 $\langle f, g \rangle = \int_0^1 f(t)\overline{g(t)}\, dt$。

## Cauchy-Schwarz 不等式

分析中最重要的单个不等式：

$$|\langle x, y \rangle| \le \|x\|\,\|y\|,$$

等号成立当且仅当 $x, y$ 线性相关。

*证明。* 对 $y \ne 0$，令 $\alpha = \langle x, y \rangle / \|y\|^2$。展开 $0 \le \|x - \alpha y\|^2 = \|x\|^2 - |\langle x, y \rangle|^2 / \|y\|^2$，整理即得。$\square$

这个不等式保证了诱导范数确实满足三角不等式，也给了我们角度：$\cos\theta = \text{Re}\,\langle x, y \rangle / (\|x\|\,\|y\|)$。

## Hilbert 空间

**Hilbert 空间**是完备的内积空间：

$$\text{Hilbert 空间} = \text{有内积的 Banach 空间}。$$

$\ell^2$ 和 $L^2[0,1]$ 都是 Hilbert 空间。有限序列空间配 $\ell^2$ 范数是内积空间但不是 Hilbert 空间（不完备）。

## 正交与投影

$x \perp y$ 表示 $\langle x, y \rangle = 0$。正交规范集的元素两两正交且都是单位向量。

**定理（正交投影）。** 设 $M$ 是 Hilbert 空间 $H$ 的闭子空间。对任意 $x \in H$，存在唯一 $m_0 \in M$ 使得：

$$\|x - m_0\| = \inf_{m \in M} \|x - m\|,$$

且 $(x - m_0) \perp M$。

*证明思路。* 取极小化序列 $(m_n)$，用平行四边形恒等式证明它是 Cauchy 列：

$$\|m_n - m_k\|^2 = 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4\left\|\frac{m_n + m_k}{2} - x\right\|^2 \le 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4d^2.$$

完备性给出极限 $m_0 \in M$。正交性通过展开 $\|x - m_0 - tv\|^2 \ge \|x - m_0\|^2$（对 $v \in M$）得到。$\square$

这在一般 Banach 空间中失败。$\ell^1$ 中最近点可能不唯一。内积是本质的。

**推论（正交分解）。** $H = M \oplus M^\perp$，其中 $M^\perp = \{x \in H : x \perp m, \forall m \in M\}$。

每个 $x$ 唯一分解为 $x = m + m^\perp$。这是 $\mathbb{R}^n$ 中向子空间投影的无穷维推广。

## 正交规范基与 Fourier 级数

正交规范集 $\{e_n\}$ 如果满足"与所有 $e_n$ 正交的唯一向量是零向量"，就称为**正交规范基**（完备正交系）。

**定理（Parseval）。** 若 $\{e_n\}_{n=1}^\infty$ 是 $H$ 的正交规范基，则对任意 $x \in H$：

$$x = \sum_{n=1}^\infty \langle x, e_n \rangle\, e_n, \quad \|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.$$

系数 $\langle x, e_n \rangle$ 是 $x$ 关于基的 **Fourier 系数**。

**例 3。** 在 $L^2[0, 2\pi]$ 中，$e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$ 构成正交规范基。Parseval 恒等式变成经典结论：

$$\frac{1}{2\pi}\int_0^{2\pi} |f(t)|^2\, dt = \sum_{n=-\infty}^{\infty} |\hat{f}(n)|^2.$$

**例 4。** 在 $\ell^2$ 中标准基 $e_n = (0, \ldots, 0, 1, 0, \ldots)$ 是正交规范基，Fourier 展开平凡：$x = \sum x_n e_n$。

## Riesz 表示定理

**定理（Riesz-Frechet）。** Hilbert 空间 $H$ 上每个连续线性泛函 $f: H \to \mathbb{C}$ 都有形式 $f(x) = \langle x, y \rangle$（唯一的 $y \in H$），且 $\|f\| = \|y\|$。

*证明思路。* $\ker f$ 是闭超平面。取 $z \perp \ker f$，$\|z\| = 1$，$f(z) \ne 0$。对任意 $x$，$x - \frac{f(x)}{f(z)}z \in \ker f$，所以 $\langle x - \frac{f(x)}{f(z)}z, z \rangle = 0$，解出 $f(x) = \frac{\overline{f(z)}}{\|z\|^2} \langle x, z \rangle$。取 $y = \frac{f(z)}{\|z\|^2} z$ 即可。$\square$

这个定理给出 $H^* \cong H$（反线性同构）。没有其他 Banach 空间有这种自对偶性。这是 Hilbert 空间最好的原因。

## 可分性

可分 Hilbert 空间有可数的正交规范基。所有可分无穷维 Hilbert 空间彼此等距同构于 $\ell^2$。

这意味着：$L^2[0,1]$、$L^2(\mathbb{R})$（适当测度下）、Sobolev 空间 $H^1[0,1]$——作为 Hilbert 空间它们都"一样"。差别体现在哪些算子作用其上、哪些子空间是自然的。

## 接下来的内容

有了 Hilbert 空间的几何，下一步研究空间之间的映射：有界线性算子。这些是矩阵的无穷维推广，但它们的行为方式会让习惯有限维线性代数的人震惊。
