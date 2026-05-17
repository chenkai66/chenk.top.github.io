---
title: "泛函分析（六）：分布与 Sobolev 空间——分析遇见 PDE"
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
description: "分布把微分推广到非光滑函数；Sobolev 空间让 PDE 理论严格化。"
---

## 经典导数的困境

考虑 PDE $-u'' = f$ 在 $[0,1]$ 上。$f$ 连续时，经典解存在且是 $C^2$ 的。但若 $f$ 只是 $L^2$ 函数——更极端地，若 $f$ 是点质量 $\delta(x - x_0)$？经典导数不适用，但物理上有意义的解存在（一根绳子受集中力作用产生折角，而不是位移不连续）。我们需要一个框架，让"导数"对非光滑对象有意义。

核心想法：不再问 $u'(x)$ 在每个点上是什么。转而通过对光滑试验函数的作用来定义导数。这个从逐点求值到对偶性的转变，是解锁现代 PDE 理论的概念飞跃。

我觉得这个转变很美，因为它逆转了通常的视角。不再说"$f$ 有导数 $f'$"，而说"$f$ 对试验函数的作用方式与有导数一致"。$f'$ 作为函数可能不存在，但作用 $\varphi \mapsto -\int f\varphi'$ 总有意义。导数变成了 $f$ 与光滑对象互动方式的属性，而不是逐点变化率。

![函数空间层级图](/images/functional-analysis/fig06_hierarchy.png)

## 试验函数与分布

设 $\Omega \subseteq \mathbb{R}^n$ 开。**试验函数空间**是：

$$\mathcal{D}(\Omega) = C_c^\infty(\Omega) = \{\varphi \in C^\infty(\Omega) : \text{supp}(\varphi) \text{ 是 } \Omega \text{ 中紧集}\}.$$

这些函数无穷次可微，且在某个紧集之外为零。它们的存在不是显然的——构造单个非零紧支撑 $C^\infty$ 函数需要技巧——但它们构成丰富的类。

**定义。** **分布**是 $\mathcal{D}(\Omega)$ 上的连续线性泛函：

$$u: \mathcal{D}(\Omega) \to \mathbb{R}, \quad \varphi \mapsto \langle u, \varphi \rangle,$$

连续性指：若 $\varphi_k \to 0$ 在 $\mathcal{D}$ 中（所有支撑包含在固定紧集 $K$ 中，且所有阶导数一致收敛到 0），则 $\langle u, \varphi_k \rangle \to 0$。

分布空间记作 $\mathcal{D}'(\Omega)$。

**每个局部可积函数定义一个分布。** 若 $f \in L^1_{\text{loc}}(\Omega)$（在每个紧子集上可积），定义：

$$\langle T_f, \varphi \rangle = \int_\Omega f(x)\varphi(x)\, dx.$$

线性且连续（控制收敛定理）。映射 $f \mapsto T_f$ 是单射（若对所有 $\varphi \in C_c^\infty$ 有 $\int f\varphi = 0$，则 $f = 0$ a.e.）。所以 $L^1_{\text{loc}} \hookrightarrow \mathcal{D}'$——分布推广了函数。

**例 1：Dirac delta。** 定义 $\langle \delta, \varphi \rangle = \varphi(0)$。线性且对 $\mathcal{D}$ 连续。但不存在局部可积函数 $f$ 使 $\int f\varphi = \varphi(0)$ 对所有试验函数成立。Delta 真正是一个不是函数的分布。

更一般地，$\langle \delta_a, \varphi \rangle = \varphi(a)$ 和 $\langle \delta_a^{(k)}, \varphi \rangle = (-1)^k \varphi^{(k)}(a)$ 都是分布。

**例 2（主值）。** $1/x$ 在 0 附近不是局部可积的。但**主值**分布：

$$\langle \text{p.v.}(1/x), \varphi \rangle = \lim_{\varepsilon \to 0^+} \int_{|x| > \varepsilon} \frac{\varphi(x)}{x}\, dx$$

是良定义的分布（极限存在因为 $\varphi(x)/x$ 的奇数部分在 0 附近可积）。

## Schwartz 空间与缓增分布

$\mathcal{D}$ 和 $L^2$ 之间是**速降函数的 Schwartz 空间**：

$$\mathcal{S}(\mathbb{R}^n) = \{f \in C^\infty : \sup_x |x^\alpha D^\beta f(x)| < \infty, \ \forall \alpha, \beta\}.$$

这些函数（及所有导数）衰减快于任何多项式。Fourier 变换把 $\mathcal{S}$ 映到自身——所以 $\mathcal{S}$ 是 Fourier 分析的自然定义域。

**缓增分布** $\mathcal{S}'$ 是 $\mathcal{S}$ 上的连续线性泛函。有包含关系 $\mathcal{D} \subset \mathcal{S} \subset L^2 \subset \mathcal{S}' \subset \mathcal{D}'$。Fourier 变换通过对偶从 $\mathcal{S}$ 延拓到 $\mathcal{S}'$：$\langle \hat{u}, \varphi \rangle = \langle u, \hat{\varphi} \rangle$。

## 分布导数

$u \in \mathcal{D}'$ 的**分布导数**定义为：

$$\langle D^\alpha u, \varphi \rangle = (-1)^{|\alpha|} \langle u, D^\alpha \varphi \rangle \quad \forall \varphi \in \mathcal{D}.$$

一阶：$\langle u', \varphi \rangle = -\langle u, \varphi' \rangle$。负号来自分部积分（边界项消失因为 $\varphi$ 紧支撑）。这个定义是强制的：若 $u$ 是经典 $C^1$ 函数，分部积分给出 $\int u'\varphi = -\int u\varphi'$，所以分布导数与经典导数一致。

**每个分布都无穷次可微**（作为分布）。在 $\mathcal{D}'$ 中微分永远不会失败。这是理论的力量。

**例 3：Heaviside 函数的导数。** 令 $H(x) = \mathbf{1}_{x > 0}$。则：

$$\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x)\, dx = \varphi(0) = \langle \delta, \varphi \rangle.$$

所以 $H' = \delta$（分布意义）。"阶梯函数的导数是 delta 函数"现在是严格陈述。

**例 4：迭代导数。** $|x|$ 的分布导数是 $\text{sgn}(x)$（符号函数），$(\text{sgn})' = 2\delta$。更一般地 $|x|'' = 2\delta$。每次微分产生更奇异的分布。

**例 5：处处不可微的函数。** Weierstrass 函数 $W(x) = \sum a^n \cos(b^n \pi x)$（$0 < a < 1$，$ab > 1$）连续但处处不可微。但作为分布，$W'$ 存在（级数在 $\mathcal{D}'$ 中收敛）。分布框架处理经典分析无法触及的对象。

## Sobolev 空间

分布让我们能微分任何东西，但 PDE 需要同时内建可积性和可微性的函数空间。需要是 Banach 空间（让抽象理论适用），且包含"弱可微"函数。

**定义。** 对 $k \in \mathbb{N}_0$，$1 \le p \le \infty$，**Sobolev 空间**是：

$$W^{k,p}(\Omega) = \{u \in L^p(\Omega) : D^\alpha u \in L^p(\Omega), \ |\alpha| \le k\},$$

范数：

$$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|D^\alpha u\|_{L^p}^p\right)^{1/p}.$$

$D^\alpha u$ 是分布导数。条件 $D^\alpha u \in L^p$ 意味着分布 $D^\alpha u$ 实际由一个 $L^p$ 函数表示。

$p = 2$ 的特例记作 $H^k(\Omega) = W^{k,2}(\Omega)$——Hilbert 空间。

**定理（完备性）。** $W^{k,p}(\Omega)$ 是 Banach 空间。$H^k(\Omega)$ 是 Hilbert 空间，内积：

$$\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \int_\Omega D^\alpha u \cdot \overline{D^\alpha v}\, dx.$$

*证明。* 若 $(u_n)$ 在 $W^{k,p}$ 中 Cauchy，则对每个 $|\alpha| \le k$，$(D^\alpha u_n)$ 在 $L^p$ 中 Cauchy。$L^p$ 完备，$D^\alpha u_n \to v_\alpha$ 在 $L^p$ 中。验证 $v_\alpha$ 是 $v_0 := \lim u_n$ 的分布 $\alpha$ 阶导数：

$$\langle v_\alpha, \varphi \rangle = \lim_n \langle D^\alpha u_n, \varphi \rangle = \lim_n (-1)^{|\alpha|}\langle u_n, D^\alpha \varphi \rangle = (-1)^{|\alpha|}\langle v_0, D^\alpha \varphi \rangle.$$

所以 $v_\alpha = D^\alpha v_0$，$v_0 \in W^{k,p}$，$u_n \to v_0$。$\square$

**例 6。** $f(x) = |x|$ 在 $(-1,1)$ 上。分布导数 $f' = \text{sgn}(x) \in L^p$（所有 $p$）。所以 $|x| \in W^{1,p}(-1,1)$。但 $f'' = 2\delta \notin L^p$，所以 $|x| \notin W^{2,p}$。Sobolev 正则性是光滑度的精确定量度量。

**例 7。** 一维中 $W^{1,p}(0,1) \subset C[0,1]$（一阶弱导数在 $L^p$ 蕴含连续性）。高维中失败：$\mathbb{R}^3$ 中 $H^1$ 函数不一定连续。

## Sobolev 嵌入

核心结构结果：用导数换可积性（甚至连续性）。

**定理（Sobolev 嵌入，临界情况）。** 设 $\Omega \subseteq \mathbb{R}^n$ 有界，Lipschitz 边界。若 $kp > n$，则：

$$W^{k,p}(\Omega) \hookrightarrow C(\overline{\Omega}) \quad \text{（连续嵌入）}.$$

若 $kp < n$，则 $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$，$1/q = 1/p - k/n$（**Sobolev 共轭指数**）。

若 $kp = n$，临界情况：$W^{k,p}$ 嵌入所有有限 $q$ 的 $L^q$，但不嵌入 $L^\infty$（Trudinger 不等式给出指数可积性）。

**例 8。** $\mathbb{R}^3$（$n=3$）中：
- $H^1 = W^{1,2}$：$kp = 2 < 3$，所以 $H^1 \hookrightarrow L^6$（$1/6 = 1/2 - 1/3$）。不连续。
- $H^2 = W^{2,2}$：$kp = 4 > 3$，所以 $H^2 \hookrightarrow C(\overline{\Omega})$。三维中两阶 $L^2$ 弱导数给你连续函数。
- $W^{1,4}$：$kp = 4 > 3$，$W^{1,4} \hookrightarrow C(\overline{\Omega})$。

临界维数 $n$ 决定了连续性需要多少阶导数。一维中一阶够了；三维需要两阶（在 $L^2$ 中）。

**定理（Rellich-Kondrachov 紧嵌入）。** 若 $kp < n$，嵌入 $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$ 对 $q < np/(n-kp)$（严格小于 Sobolev 共轭）是**紧**的。若 $kp > n$，到 $C(\overline{\Omega})$ 的嵌入紧。

紧性对存在性理论至关重要：它让你从 Sobolev 空间的有界序列中提取收敛子列——正是变分方法所需要的。

## PDE 的弱解

整个系列的收获。考虑边值问题：

$$-\Delta u = f \text{ 在 } \Omega \text{ 中}, \quad u = 0 \text{ 在 } \partial\Omega \text{ 上}.$$

**第一步：弱形式化。** 乘以试验函数 $v \in H^1_0(\Omega)$（$H^1$ 中边界迹为零的函数），分部积分：

$$\int_\Omega \nabla u \cdot \nabla v\, dx = \int_\Omega f v\, dx \quad \forall v \in H^1_0(\Omega).$$

**弱解**是满足此等式的 $u \in H^1_0(\Omega)$。$u$ 的二阶导数不出现——只需 $u \in H^1$。

**第二步：应用 Lax-Milgram。**

**定理（Lax-Milgram）。** 设 $H$ 是 Hilbert 空间，$a: H \times H \to \mathbb{R}$ 是双线性形式，满足：
- 有界：$|a(u,v)| \le M\|u\|\|v\|$
- 强制：$a(u,u) \ge \alpha\|u\|^2$，某个 $\alpha > 0$

则对每个 $F \in H^*$，存在唯一 $u \in H$ 使 $a(u,v) = F(v)$ 对所有 $v$。

*证明。* Riesz 给出 $F(v) = \langle w, v \rangle$。有界双线性形式表示给 $a(u,v) = \langle Au, v \rangle$，$A$ 有界。强制性给 $\langle Au, u \rangle \ge \alpha\|u\|^2$，$A$ 单射且像闭。满射：若 $z \perp \text{range}(A)$，则 $0 = \langle Az, z \rangle \ge \alpha\|z\|^2$，迫使 $z = 0$。所以 $Au = w$ 有唯一解。$\square$

**第三步：验证假设。** 对 $a(u,v) = \int_\Omega \nabla u \cdot \nabla v\, dx$ 在 $H^1_0(\Omega)$ 上：
- 有界：Cauchy-Schwarz。
- 强制：$a(u,u) = \|\nabla u\|_{L^2}^2 \ge c\|u\|_{H^1}^2$，由 **Poincare 不等式**（有界 $\Omega$ 上 $H^1_0$ 中 $\|u\|_{L^2} \le C_\Omega\|\nabla u\|_{L^2}$）。

Lax-Milgram 给出：对每个 $f \in L^2(\Omega)$，存在唯一弱解 $u \in H^1_0(\Omega)$。

**第四步：正则性提升。** 若 $\partial\Omega$ 光滑且 $f \in L^2$，椭圆正则性理论给 $u \in H^2$。$f$ 更光滑则 $u$ 更光滑。特别地，$f \in C^\infty$ 且 $\Omega$ 光滑时，$u \in C^\infty$——弱解实际上是经典解。

## 全景

连接抽象泛函分析和具体 PDE 的思路链：

1. **分布**（$\mathcal{D}'$）把微分推广到非光滑对象。
2. **Sobolev 空间**（$W^{k,p}$）为弱可微函数提供 Banach/Hilbert 结构。
3. **嵌入定理**把弱可微性联系到经典正则性。
4. **紧性**（Rellich-Kondrachov）提供存在性论证所需的序列紧性。
5. **变分形式化**把 PDE 转化为 Hilbert 空间上的双线性形式方程。
6. **Lax-Milgram/Riesz**（抽象泛函分析）给出存在唯一性。
7. **正则性理论**把弱解提升回经典解。

第 5-6 步是纯泛函分析——在任何 Hilbert 空间、任何强制双线性形式上都成立。PDE 特有的内容在第 1-4 步（建立正确空间）和第 7 步（正则性估计）。抽象机器是通用引擎，一旦空间正确选定就驱动存在性理论。

## 后续方向

本系列涵盖了泛函分析核心：空间（度量、赋范、Banach、Hilbert），映射（有界算子、泛函、紧算子），结构定理（四大定理），谱论，以及通过分布和 Sobolev 空间连接 PDE。

从这里分支出：无界算子与半群（量子力学、演化方程），算子代数（$C^*$-代数、von Neumann 代数），非线性泛函分析（不动点理论、变分方法、度理论），微局部分析（拟微分算子、波前集），随机 PDE（Hilbert 空间上的柱 Brownian 运动）。每个方向用相同核心机器——完备性、对偶性、紧性——应用于更精密的设定。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 6 篇，共 6 篇。
上一篇: [第 5 篇 — 紧算子的谱论](/zh/functional-analysis/05-谱论/)*
