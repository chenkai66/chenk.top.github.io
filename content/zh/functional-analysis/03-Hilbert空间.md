---
title: "泛函分析（三）：Hilbert 空间——无穷维的几何"
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
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "functional-analysis-3"
description: "内积赋予无穷维空间几何结构——投影、正交基与 Riesz 表示定理。"
---


## 内积与平行四边形律

在泛函分析中，Hilbert 空间是具有内积结构的完备赋范线性空间。内积不仅定义了向量之间的夹角，还提供了长度的概念。这使得 Hilbert 空间成为研究无穷维几何的理想场所。

![Hilbert 空间中向闭子空间的正交投影](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/03-hilbert-spaces/fa_fig2_projection.png)


**定义**：设 $X$ 是一个线性空间，如果存在一个映射 $\langle \cdot, \cdot \rangle: X \times X \to \mathbb{C}$ 满足以下条件，则称该映射为内积：
1. $\langle x, y \rangle = \overline{\langle y, x \rangle}$
2. $\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle$
3. $\langle x, x \rangle \geq 0$ 且 $\langle x, x \rangle = 0$ 当且仅当 $x = 0$

由内积可以诱导出一个范数 $\|x\| = \sqrt{\langle x, x \rangle}$，从而使得 $X$ 成为一个赋范线性空间。

**平行四边形律**：对于任意 $x, y \in X$，有
$$
\|x + y\|^2 + \|x - y\|^2 = 2(\|x\|^2 + \|y\|^2).
$$

**证明思路**：利用内积的性质直接展开即可得到上述等式。具体来说，
$$
\begin{aligned}
\|x + y\|^2 &= \langle x + y, x + y \rangle = \langle x, x \rangle + \langle x, y \rangle + \langle y, x \rangle + \langle y, y \rangle \\
&= \|x\|^2 + 2\Re\langle x, y \rangle + \|y\|^2,
\end{aligned}
$$
类似地，
$$
\begin{aligned}
\|x - y\|^2 &= \langle x - y, x - y \rangle = \langle x, x \rangle - \langle x, y \rangle - \langle y, x \rangle + \langle y, y \rangle \\
&= \|x\|^2 - 2\Re\langle x, y \rangle + \|y\|^2.
\end{aligned}
$$
将两式相加即得平行四边形律。

**例题**：考虑 $\mathbb{R}^2$ 中的向量 $x = (1, 0)$ 和 $y = (0, 1)$，验证平行四边形律。
$$
\begin{aligned}
\|x + y\|^2 &= \|(1, 1)\|^2 = 1^2 + 1^2 = 2, \\
\|x - y\|^2 &= \|(1, -1)\|^2 = 1^2 + (-1)^2 = 2, \\
\|x\|^2 + \|y\|^2 &= 1^2 + 0^2 + 0^2 + 1^2 = 2.
\end{aligned}
$$
因此，
$$
\|x + y\|^2 + \|x - y\|^2 = 2 + 2 = 4 = 2(\|x\|^2 + \|y\|^2).
$$

## 正交性与投影定理

正交性是 Hilbert 空间中的一个重要概念。两个向量 $x, y \in X$ 称为正交的，如果 $\langle x, y \rangle = 0$。正交性的引入使得我们可以讨论子空间的正交补和投影问题。

**投影定理**：设 $M$ 是 Hilbert 空间 $X$ 的闭子空间，对于任意 $x \in X$，存在唯一的 $y \in M$ 使得 $\|x - y\|$ 最小。此外，$x - y \perp M$。

**证明思路**：首先，通过构造方法找到这样的 $y$。假设 $y_1, y_2 \in M$ 都满足条件，那么
$$
\|x - y_1\| \leq \|x - y_2\| \quad \text{且} \quad \|x - y_2\| \leq \|x - y_1\|,
$$
从而 $\|x - y_1\| = \|x - y_2\|$。由此可推出 $y_1 = y_2$，唯一性得证。其次，若 $x - y \not\perp M$，则存在 $z \in M$ 使得 $\langle x - y, z \rangle \neq 0$，这与 $y$ 的极小性矛盾。

**例题**：考虑 $L^2[0, 1]$ 中的函数 $f(x) = x$ 和子空间 $M = \{g \in L^2[0, 1] : g(0) = 0\}$，求 $f$ 在 $M$ 上的投影。

解：设 $g \in M$ 为 $f$ 在 $M$ 上的投影，则 $f - g \perp M$。记 $g(x) = ax$，则
$$
\langle f - g, h \rangle = 0 \quad \forall h \in M.
$$
取 $h(x) = x$，则
$$
\int_0^1 (x - ax)x \, dx = 0 \implies \int_0^1 x^2 \, dx - a \int_0^1 x^2 \, dx = 0 \implies \frac{1}{3} - a \cdot \frac{1}{3} = 0 \implies a = 1.
$$
因此，$g(x) = x$，即 $f$ 在 $M$ 上的投影就是 $f$ 本身。

## 正交系与 Bessel 不等式

正交系是 Hilbert 空间中的一个重要概念。一组向量 $\{e_n\}_{n=1}^\infty$ 称为正交系，如果对于任意 $m \neq n$，有 $\langle e_m, e_n \rangle = 0$。正交系的一个重要性质是 Bessel 不等式。

**Bessel 不等式**：设 $\{e_n\}_{n=1}^\infty$ 是 Hilbert 空间 $X$ 中的正交系，对于任意 $x \in X$，有
$$
\sum_{n=1}^\infty |\langle x, e_n \rangle|^2 \leq \|x\|^2.
$$

**证明思路**：考虑部分和
$$
S_N = \sum_{n=1}^N \langle x, e_n \rangle e_n.
$$
由于 $\{e_n\}$ 是正交系，我们有
$$
\|S_N\|^2 = \sum_{n=1}^N |\langle x, e_n \rangle|^2.
$$
另一方面，$\|x - S_N\|^2 \geq 0$，因此
$$
\|x\|^2 \geq \sum_{n=1}^N |\langle x, e_n \rangle|^2.
$$
令 $N \to \infty$ 即得 Bessel 不等式。

**例题**：考虑 $L^2[0, 1]$ 中的标准正交基 $\{e_n(x) = \sqrt{2} \sin(n\pi x)\}_{n=1}^\infty$，验证 Bessel 不等式对于 $f(x) = x$ 成立。

解：计算
$$
\langle f, e_n \rangle = \int_0^1 x \sqrt{2} \sin(n\pi x) \, dx.
$$
使用分部积分法，令 $u = x$，$dv = \sqrt{2} \sin(n\pi x) \, dx$，则
$$
\begin{aligned}
\langle f, e_n \rangle &= \left[ -\frac{x \cos(n\pi x)}{n\pi} \right]_0^1 + \frac{1}{n\pi} \int_0^1 \cos(n\pi x) \, dx \\
&= -\frac{\cos(n\pi)}{n\pi} + \frac{1}{n\pi} \left[ \frac{\sin(n\pi x)}{n\pi} \right]_0^1 \\
&= -\frac{(-1)^n}{n\pi}.
\end{aligned}
$$
因此，
$$
|\langle f, e_n \rangle|^2 = \frac{1}{n^2 \pi^2}.
$$
根据 Bessel 不等式，
$$
\sum_{n=1}^\infty \frac{1}{n^2 \pi^2} \leq \int_0^1 x^2 \, dx = \frac{1}{3}.
$$
实际上，
$$
\sum_{n=1}^\infty \frac{1}{n^2 \pi^2} = \frac{1}{6},
$$
确实满足 Bessel 不等式。

## 正交基、Parseval 恒等式、Fourier 级数

正交基是 Hilbert 空间中的一个重要概念。一组向量 $\{e_n\}_{n=1}^\infty$ 称为正交基，如果它是正交系且其张成的空间稠密于整个 Hilbert 空间。正交基的一个重要性质是 Parseval 恒等式。

**Parseval 恒等式**：设 $\{e_n\}_{n=1}^\infty$ 是 Hilbert 空间 $X$ 的正交基，对于任意 $x \in X$，有
$$
\|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.
$$

**证明思路**：由于 $\{e_n\}$ 是正交基，对于任意 $x \in X$，可以表示为
$$
x = \sum_{n=1}^\infty \langle x, e_n \rangle e_n.
$$
因此，
$$
\|x\|^2 = \left\| \sum_{n=1}^\infty \langle x, e_n \rangle e_n \right\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.
$$

**Fourier 级数**：设 $\{e_n\}_{n=1}^\infty$ 是 Hilbert 空间 $X$ 的正交基，对于任意 $x \in X$，其 Fourier 级数展开为
$$
x = \sum_{n=1}^\infty \langle x, e_n \rangle e_n.
$$

**例题**：考虑 $L^2[-\pi, \pi]$ 中的函数 $f(x) = x$，用标准正交基 $\{e_n(x) = \frac{1}{\sqrt{2\pi}} e^{inx}\}_{n=-\infty}^\infty$ 展开 $f$ 的 Fourier 级数，并验证 Parseval 恒等式。

解：计算
$$
\langle f, e_n \rangle = \int_{-\pi}^\pi x \frac{1}{\sqrt{2\pi}} e^{-inx} \, dx.
$$
使用分部积分法，令 $u = x$，$dv = \frac{1}{\sqrt{2\pi}} e^{-inx} \, dx$，则
$$
\begin{aligned}
\langle f, e_n \rangle &= \left[ -\frac{x e^{-inx}}{in\sqrt{2\pi}} \right]_{-\pi}^\pi + \frac{1}{in\sqrt{2\pi}} \int_{-\pi}^\pi e^{-inx} \, dx \\
&= -\frac{\pi e^{-in\pi} - (-\pi) e^{in\pi}}{in\sqrt{2\pi}} + 0 \\
&= \frac{2\pi i (-1)^n}{in\sqrt{2\pi}} \\
&= \frac{2(-1)^n}{n\sqrt{2\pi}}.
\end{aligned}
$$
因此，
$$
\langle f, e_n \rangle = \begin{cases}
0 & \text{if } n = 0, \\
\frac{2(-1)^n}{in\sqrt{2\pi}} & \text{if } n \neq 0.
\end{cases}
$$
$f$ 的 Fourier 级数展开为
$$
f(x) = \sum_{n=-\infty}^\infty \langle f, e_n \rangle e_n(x) = \sum_{n \neq 0} \frac{2(-1)^n}{in\sqrt{2\pi}} \frac{1}{\sqrt{2\pi}} e^{inx} = \sum_{n \neq 0} \frac{2(-1)^n}{in} e^{inx}.
$$
验证 Parseval 恒等式：
$$
\|f\|^2 = \int_{-\pi}^\pi x^2 \, dx = \frac{2\pi^3}{3},
$$
而
$$
\sum_{n=-\infty}^\infty |\langle f, e_n \rangle|^2 = 2 \sum_{n=1}^\infty \left| \frac{2(-1)^n}{in\sqrt{2\pi}} \right|^2 = 2 \sum_{n=1}^\infty \frac{4}{n^2 2\pi} = \frac{4}{\pi} \sum_{n=1}^\infty \frac{1}{n^2} = \frac{4}{\pi} \cdot \frac{\pi^2}{6} = \frac{2\pi^3}{3}.
$$
因此，Parseval 恒等式成立。

## Riesz 表示定理（证明）

Riesz 表示定理是 Hilbert 空间理论中的一个基本定理，它表明每个有界线性泛函都可以表示为内积的形式。

**Riesz 表示定理**：设 $X$ 是 Hilbert 空间，$f \in X^*$ 是有界线性泛函，则存在唯一的 $y \in X$ 使得
$$
f(x) = \langle x, y \rangle \quad \forall x \in X.
$$
此外，$\|f\| = \|y\|$。

**证明思路**：首先，考虑 $f \equiv 0$ 的情况，此时 $y = 0$ 显然满足条件。接下来，假设 $f \not\equiv 0$。设 $M = \ker(f)$，则 $M$ 是 $X$ 的闭子空间。由于 $f \not\equiv 0$，$M \neq X$。根据投影定理，存在 $z \notin M$ 使得 $\|z\| = 1$ 且 $z \perp M$。定义
$$
y = \overline{f(z)} z.
$$
对于任意 $x \in X$，存在 $m \in M$ 和 $\alpha \in \mathbb{C}$ 使得 $x = m + \alpha z$。因此，
$$
f(x) = f(m) + \alpha f(z) = \alpha f(z).
$$
另一方面，
$$
\langle x, y \rangle = \langle m + \alpha z, \overline{f(z)} z \rangle = \alpha \overline{f(z)} \langle z, z \rangle = \alpha f(z).
$$
因此，$f(x) = \langle x, y \rangle$。最后，$\|f\| = \|y\|$ 可以通过计算验证。

**例题**：考虑 $L^2[0, 1]$ 中的线性泛函 $f(g) = \int_0^1 g(x) \, dx$，求对应的 $y \in L^2[0, 1]$ 使得 $f(g) = \langle g, y \rangle$。

解：根据 Riesz 表示定理，存在唯一的 $y \in L^2[0, 1]$ 使得
$$
\int_0^1 g(x) \, dx = \int_0^1 g(x) \overline{y(x)} \, dx.
$$
比较两边，得到
$$
\overline{y(x)} = 1 \quad \text{a.e.}
$$
因此，$y(x) = 1$ 几乎处处成立。

## $l^2$ 作为万能可分 Hilbert 空间

$l^2$ 空间是所有平方可和序列的集合，即
$$
l^2 = \left\{ (x_n)_{n=1}^\infty : \sum_{n=1}^\infty |x_n|^2 < \infty \right\}.
$$
$l^2$ 空间是一个可分的 Hilbert 空间，它在泛函分析中扮演着重要的角色。

**定理**：每个可分的无限维 Hilbert 空间都同构于 $l^2$。

**证明思路**：设 $X$ 是一个可分的无限维 Hilbert 空间，存在一个可数稠密子集 $\{e_n\}_{n=1}^\infty$。通过 Gram-Schmidt 正交化过程，可以得到一个标准正交基 $\{f_n\}_{n=1}^\infty$。定义映射
$$
T: X \to l^2, \quad T(x) = (\langle x, f_n \rangle)_{n=1}^\infty.
$$
可以验证 $T$ 是一个线性同构映射，且保持内积不变。因此，$X \cong l^2$。

**例题**：考虑 $L^2[0, 1]$，验证它同构于 $l^2$。

解：考虑 $L^2[0, 1]$ 中的标准正交基 $\{e_n(x) = \sqrt{2} \sin(n\pi x)\}_{n=1}^\infty$。定义映射
$$
T: L^2[0, 1] \to l^2, \quad T(f) = (\langle f, e_n \rangle)_{n=1}^\infty.
$$
可以验证 $T$ 是一个线性同构映射，且保持内积不变。因此，$L^2[0, 1] \cong l^2$。

## 接下来

在下一节中，我们将继续探讨 Hilbert 空间中的算子理论，包括自伴算子、紧算子以及谱理论等内容。这些内容将进一步深化我们对 Hilbert 空间的理解，并为后续的应用打下坚实的基础。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 3 篇（共 12 篇）。*

*上一篇：[第 2 篇 —— 赋范空间与 Banach 空间](/zh/functional-analysis/02-赋范空间与Banach空间/)*

*下一篇：[第 4 篇 —— 对偶空间与 Hahn-Banach](/zh/functional-analysis/04-对偶空间与Hahn-Banach/)*
