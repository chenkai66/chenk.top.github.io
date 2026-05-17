---
title: "泛函分析（六）：分布与 Sobolev 空间——分析遇上 PDE"
date: 2021-04-05 09:00:00
tags:
  - functional-analysis
  - distributions
  - sobolev-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "functional-analysis-6"
description: "分布把微分运算推广到非光滑函数；Sobolev 空间让 PDE 理论严格化。"
---

## 经典导数的困境

考虑 PDE $-u'' = f$。若 $f$ 连续，经典解存在且是 $C^2$ 的。但如果 $f$ 是 $L^2$ 函数，甚至是点质量 $\delta(x - x_0)$？经典导数不适用，但物理上有意义的解确实存在。

解决方案：不再追问 $u'(x)$ 在每一点的值。改为通过对光滑试验函数的作用来定义导数。

## 试验函数与分布

设 $\Omega \subseteq \mathbb{R}^n$ 是开集。**试验函数空间**：

$$\mathcal{D}(\Omega) = C_c^\infty(\Omega) = \{f \in C^\infty(\Omega) : \text{supp}(f) \text{ 是 } \Omega \text{ 中的紧集}\}.$$

**分布**是 $\mathcal{D}(\Omega)$ 上的连续线性泛函：

$$u: \mathcal{D}(\Omega) \to \mathbb{R}, \quad \varphi \mapsto \langle u, \varphi \rangle.$$

连续性意味着：若 $\varphi_n \to 0$（支集在固定紧集内，所有阶导数一致收敛），则 $\langle u, \varphi_n \rangle \to 0$。分布空间记作 $\mathcal{D}'(\Omega)$。

**每个局部可积函数都是分布。** 若 $f \in L^1_{\text{loc}}(\Omega)$，定义：

$$\langle f, \varphi \rangle = \int_\Omega f(x)\varphi(x)\, dx.$$

分布推广了函数。但有些分布不是函数。

**例 1：Dirac delta。** $\langle \delta, \varphi \rangle = \varphi(0)$。这是分布（线性且连续），但不存在函数 $f$ 使得 $\int f\varphi\, dx = \varphi(0)$ 对所有试验函数成立。

## 分布意义的导数

$u \in \mathcal{D}'$ 的**分布导数**定义为：

$$\langle u', \varphi \rangle = -\langle u, \varphi' \rangle \quad \forall \varphi \in \mathcal{D}.$$

负号来自分部积分（边界项为零因为 $\varphi$ 有紧支集）。高阶导数：

$$\langle D^\alpha u, \varphi \rangle = (-1)^{|\alpha|} \langle u, D^\alpha \varphi \rangle.$$

每个分布无穷次可微（作为分布）。这是理论的力量：微分永远不会失败。

**例 2：Heaviside 函数的导数。** $H(x) = \mathbf{1}_{x > 0}$。则：

$$\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x)\, dx = \varphi(0) = \langle \delta, \varphi \rangle.$$

所以 $H' = \delta$。"阶跃函数的导数是 delta 函数"成为严格的数学命题。

**例 3。** $f(x) = |x|$ 的分布导数是 $f'(x) = \text{sgn}(x)$，$f'' = 2\delta$。非光滑函数的反复微分产生越来越奇异的分布。

## Sobolev 空间

分布允许微分任何东西，但做 PDE 需要同时具备可积性和可微性的空间。这就是 Sobolev 空间。

**定义。** 对 $k \in \mathbb{N}$，$1 \le p \le \infty$：

$$W^{k,p}(\Omega) = \{u \in L^p(\Omega) : D^\alpha u \in L^p(\Omega), \ \forall |\alpha| \le k\},$$

范数：

$$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}.$$

$D^\alpha u$ 是分布导数。$p = 2$ 时记作 $H^k(\Omega) = W^{k,2}(\Omega)$——Hilbert 空间。

**定理。** $W^{k,p}(\Omega)$ 是 Banach 空间。$H^k(\Omega)$ 是 Hilbert 空间，内积为：

$$\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \int_\Omega D^\alpha u \cdot D^\alpha v\, dx.$$

*证明思路。* $(u_n)$ 在 $W^{k,p}$ 中 Cauchy $\Rightarrow$ $(D^\alpha u_n)$ 在 $L^p$ 中 Cauchy $\Rightarrow$ 收敛到 $v_\alpha$。验证 $v_\alpha$ 确实是 $v_0 = \lim u_n$ 的分布导数：对试验函数分部积分传递极限即可。$\square$

## Sobolev 嵌入

核心结构性结果：用导数换取可积性（甚至连续性）。

**定理（1维 Sobolev 嵌入）。** 若 $u \in H^1(0,1)$，则 $u$ 几乎处处等于一个连续函数，且：

$$\|u\|_{L^\infty} \le C\|u\|_{H^1}.$$

一维里一阶 $L^2$ 正则性就买到了连续性。高维更精细：

**定理（一般 Sobolev 嵌入）。** 若 $\Omega \subseteq \mathbb{R}^n$ 有界光滑边界，$kp > n$，则 $W^{k,p}(\Omega) \hookrightarrow C(\overline{\Omega})$。

若 $kp < n$，嵌入到 $L^q$，$1/q = 1/p - k/n$（Sobolev 共轭指数）。

**例 4。** $\mathbb{R}^3$ 中 $H^1 = W^{1,2}$ 嵌入 $L^6$（$1/6 = 1/2 - 1/3$），但不嵌入 $L^\infty$。3维需要 $H^2$ 才能嵌入 $C^0$。临界指数取决于维数。

## PDE 的弱解

真正的回报来了。考虑边值问题：

$$-\Delta u = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial\Omega.$$

两边乘 $v \in H^1_0(\Omega)$（边界为零的 $H^1$ 函数）并分部积分：

$$\int_\Omega \nabla u \cdot \nabla v\, dx = \int_\Omega f v\, dx \quad \forall v \in H^1_0(\Omega).$$

**弱解**是满足此等式的 $u \in H^1_0(\Omega)$。不需要 $u$ 的二阶导数——只需 $u \in H^1$。

**定理（Lax-Milgram）。** 设 $H$ 是 Hilbert 空间，$a: H \times H \to \mathbb{R}$ 双线性、有界且强制（$a(u,u) \ge \alpha\|u\|^2$）。则对每个 $f \in H^*$，存在唯一 $u \in H$ 满足 $a(u,v) = f(v)$，$\forall v$。

*证明。* 由 Riesz 表示定理 $a(u, \cdot) = \langle Au, \cdot \rangle$，$A$ 有界。强制性给出 $\|Au\| \ge \alpha\|u\|$（单射+闭值域）。满射：若 $w \perp \text{range}(A)$，则 $0 = \langle Aw, w \rangle = a(w,w) \ge \alpha\|w\|^2$，$w = 0$。$\square$

应用到 Dirichlet 问题：$a(u,v) = \int \nabla u \cdot \nabla v$ 在 $H^1_0$ 上强制（Poincare 不等式），所以对每个 $f \in L^2$ 弱解唯一存在。

## 思想链条

1. **分布**将微分推广到非光滑对象。
2. **Sobolev 空间**给"弱可微"函数配备 Banach/Hilbert 结构。
3. **嵌入定理**把弱可微性和经典正则性联系起来。
4. **变分公式**把 PDE 变成 Hilbert 空间中的方程。
5. **Lax-Milgram/Riesz**（抽象泛函分析）给出存在唯一性。
6. **正则性理论**（椭圆正则性）把弱解提升回经典解。

泛函分析提供第4-5步。整个现代 PDE 存在性理论建立在这个系列构建的基础之上。

## 总结

这个系列覆盖了泛函分析的核心：空间（度量、赋范、Banach、Hilbert），映射（有界算子、泛函、紧算子），结构定理（四大定理），谱论，以及通过分布和 Sobolev 空间与 PDE 的连接。

从这里向外分支的方向很多：无界算子和半群（量子力学、发展方程），算子代数（$C^*$-代数、von Neumann 代数），非线性泛函分析（不动点理论、度理论），微局部分析（波前集、拟微分算子）。每个方向用的都是同一套核心机器——完备性、对偶、紧性——施加到越来越复杂的对象上。

贯穿始终的主线：无穷维分析需要用结构性论证替代逐点论证。$L^2$ 中的元素是等价类，不是函数——无法逐点计算。只能用范数、内积、对偶配对、算子估计来工作。抽象不是为了美观——是因为在需要的细节层次上，具体对象根本不存在。
