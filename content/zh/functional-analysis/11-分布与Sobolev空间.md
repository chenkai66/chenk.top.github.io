---
title: "泛函分析（十一）：分布与 Sobolev 空间"
date: 2021-10-21 09:00:00
tags:
  - functional-analysis
  - distributions
  - sobolev-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
description: "分布将函数的概念推广到能处理经典不存在的导数——Sobolev 空间为 PDE 弱解提供了合适的框架。"
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "functional-analysis-11"
---

考虑实数轴上的波动方程。行波 $u(x,t) = f(x-ct)$ 在 $f$ 二次可微时求解 $u_{tt} = c^2 u_{xx}$。但物理波可以有尖锐的前沿——例如阶跃函数轮廓，它在任何地方都不可微。我们还能说这样的函数"求解"了波动方程吗？经典微积分给出否定回答。分布理论给出肯定回答。

Dirac delta "函数" $\delta(x)$，由性质 $\int \delta(x)\varphi(x)\,dx = \varphi(0)$ 定义，是另一个抵抗经典处理的基本对象。它在传统意义上不是函数——没有任何可测函数能满足这一积分恒等式。然而它在物理和工程中无处不在：点源、点质量、瞬时脉冲的理想化。

Laurent Schwartz 的分布理论（1950）为这两种情形提供了严格框架。核心思想优雅而简洁：不再试图给广义"函数"赋予逐点值，而是通过它们对光滑检验函数的作用来定义。这一视角转变——从逐点求值到对偶——是泛函分析的伟大胜利之一。

Sobolev 空间由 Sergei Sobolev 在 1930 年代引入，在此基础上更进一步。它们是由具有规定阶弱导数（属于 $L^p$）的函数构成的 Banach 空间（通常还是 Hilbert 空间）。这些空间是微分算子的自然定义域，也是 PDE 存在性与正则性理论的正确舞台。

---

## 为什么经典导数不够

### 驱动问题

**问题 1：PDE 的弱解.** 考虑 $-u'' = f$ 在 $(0,1)$ 上，$u(0) = u(1) = 0$。若 $f$ 连续，经典解存在且是 $C^2$ 的。但若 $f \in L^2(0,1)$——比如 $f$ 是某区间的示性函数呢？经典（$C^2$）解不存在。然而用检验函数 $\varphi \in C_c^\infty(0,1)$ 乘两边并分部积分得

![Sobolev 空间的嵌入链](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_fig6_sobolev.png)


$$
\int_0^1 u'\varphi'\,dx = \int_0^1 f\varphi\,dx \quad \forall\, \varphi \in C_c^\infty(0,1).
$$

满足此积分恒等式的函数 $u$ 是**弱解**。它不需要二次可微；$u \in H_0^1(0,1)$（一阶弱导数在 $L^2$ 中，边界值为零）即可。使这一切精确化需要 Sobolev 空间。

**问题 2：Dirac delta.** 点电荷在原点的静电势满足 $-\Delta \phi = \delta$。右端不是函数。为使此方程严格化，需要一个包含 $\delta$ 的空间——分布空间。

**问题 3：Fourier 分析.** 常函数 $1$ 的 Fourier 变换（形式上）是 $\delta$。$L^2$ 上的 Fourier 变换严格理论（Plancherel）不覆盖分布，但将 Fourier 分析推广到缓增分布对 PDE 和信号处理至关重要。

### 概念转变

经典方法：函数 $f$ 由逐点值 $f(x)$ 定义。

分布方法：广义函数 $f$ 由它对检验函数的作用定义：$\varphi \mapsto \langle f, \varphi \rangle = \int f\varphi\,dx$。

两个在零测集上不同的局部可积函数定义相同的分布。这意味着分布自动模掉了零测集——这是优点而非缺陷，因为 $L^p$ 空间本来就做同样的事。

---

## 检验函数 D(Omega) 与分布空间 D'(Omega)

### 检验函数空间

设 $\Omega \subseteq \mathbb{R}^n$ 为开集。**检验函数空间**为

$$
\mathcal{D}(\Omega) = C_c^\infty(\Omega),
$$

即紧支撑在 $\Omega$ 中的无穷次可微函数空间。

**$\mathcal{D}(\Omega)$ 上的拓扑.** 序列 $\varphi_j \to 0$ 在 $\mathcal{D}(\Omega)$ 中意味着：
1. 存在紧集 $K \subset \Omega$ 使得 $\text{supp}(\varphi_j) \subset K$，$\forall\, j$。
2. 对每个多重指标 $\alpha$，$\partial^\alpha \varphi_j$ 在 $K$ 上一致收敛到零。

此拓扑不是范数拓扑——它是归纳极限拓扑。精确细节在技术上要求较高，但在应用中很少需要；重要的是上面的序列刻画。

**检验函数的存在性.** 标准构造：令 $\rho(x) = Ce^{-1/(1-|x|^2)}$（$|x| < 1$），$\rho(x) = 0$（$|x| \ge 1$），$C$ 选取使 $\int\rho = 1$。则 $\rho \in C_c^\infty(\mathbb{R}^n)$，$\rho_\epsilon(x) = \epsilon^{-n}\rho(x/\epsilon)$ 是支撑在 $B(0,\epsilon)$ 中的光滑磨光子。

### 分布

**定义.** $\Omega$ 上的**分布**是 $\mathcal{D}(\Omega)$ 上的连续线性泛函。所有分布的空间记为 $\mathcal{D}'(\Omega)$。

连续性意味着：若 $\varphi_j \to 0$ 在 $\mathcal{D}(\Omega)$ 中，则 $\langle u, \varphi_j \rangle \to 0$。等价地，对每个紧集 $K \subset \Omega$，存在 $C > 0$ 和 $N \in \mathbb{N}$ 使得

$$
|\langle u, \varphi \rangle| \le C\sum_{|\alpha|\le N}\sup_K |\partial^\alpha \varphi| \quad \forall\, \varphi \in C_c^\infty(K).
$$

最小的这样的 $N$（若全局存在）称为分布的**阶**。

### 函数嵌入分布

每个局部可积函数 $f \in L^1_{\text{loc}}(\Omega)$ 定义一个分布：

$$
\langle T_f, \varphi \rangle = \int_\Omega f(x)\varphi(x)\,dx.
$$

映射 $f \mapsto T_f$ 单射且连续。我们将 $f$ 与 $T_f$ 等同。

### 关键例子

**Dirac delta.** $\langle \delta, \varphi \rangle = \varphi(0)$。阶为 0 的分布，不能由任何局部可积函数表示。

**主值分布.** $\langle \text{p.v.}\frac{1}{x}, \varphi \rangle = \lim_{\epsilon\to 0^+}\int_{|x|>\epsilon}\frac{\varphi(x)}{x}\,dx$。阶为 1。

**Delta 的导数.** $\langle \delta^{(k)}, \varphi \rangle = (-1)^k \varphi^{(k)}(0)$。阶为 $k$。

---

## 分布上的运算：导数、乘法、卷积

### 分布导数

这是最关键的运算。对光滑函数 $f$，分部积分给出 $\int f'\varphi\,dx = -\int f\varphi'\,dx$。这启发了：

**定义.** 分布 $u \in \mathcal{D}'(\Omega)$ 的**分布导数** $\partial^\alpha u$ 定义为

$$
\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha \varphi \rangle \quad \forall\, \varphi \in \mathcal{D}(\Omega).
$$

**每个分布在分布意义下都无穷次可微**——这与经典分析中可微性是限制条件的情形形成鲜明对比。

**例子.**

1. **Heaviside 函数.** $H(x) = \mathbf{1}_{[0,\infty)}(x)$。则 $H' = \delta$（分布意义）。阶跃函数的导数是 delta 函数。

2. **绝对值.** $|x|' = \text{sgn}(x)$，$|x|'' = 2\delta$。经典不存在的原点处导数在分布意义下完美定义。

3. **$\log|x|$（一维）.** 分布导数为主值分布 $\text{p.v.}\frac{1}{x}$。

### 与光滑函数的乘法

若 $a \in C^\infty(\Omega)$, $u \in \mathcal{D}'(\Omega)$，定义 $\langle au, \varphi \rangle = \langle u, a\varphi \rangle$。

**警告.** 两个任意分布的乘积*一般没有定义*。$\delta \cdot \delta$ 没有典范含义，试图定义它会导致量子场论中重正化的困难。Schwartz 不可能性定理表明：$\mathcal{D}'$ 上不存在同时结合的、交换的、延拓连续函数逐点积并满足 Leibniz 律的乘法。

### 卷积

若 $u \in \mathcal{D}'(\mathbb{R}^n)$ 紧支撑，$\varphi \in C^\infty(\mathbb{R}^n)$，则卷积 $u * \varphi$ 有定义且属于 $C^\infty$。关键性质：

- $\delta * f = f$（delta 函数是卷积恒等元）。
- $\partial^\alpha(u * v) = (\partial^\alpha u) * v = u * (\partial^\alpha v)$（导数可在因子间转移）。
- 磨光：$f * \rho_\epsilon \in C^\infty$，且 $f * \rho_\epsilon \to f$（在 $\mathcal{D}'$ 中）。

### 缓增分布与 Fourier 变换

**Schwartz 空间** $\mathcal{S}(\mathbb{R}^n)$ 由所有光滑函数组成，其导数全都比任意多项式衰减更快。其对偶 $\mathcal{S}'(\mathbb{R}^n)$ 是**缓增分布**空间。

Fourier 变换通过对偶延拓为同构 $\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$：$\langle \hat{u}, \varphi \rangle = \langle u, \hat{\varphi} \rangle$。这赋予了多项式、$\delta$ 等超出 $L^1$ 或 $L^2$ Fourier 分析范围的对象以 Fourier 变换的严格含义。

---

## Sobolev 空间 W^{k,p} 与 H^k

### 定义

**定义.** 对 $k \in \mathbb{N}_0$, $1 \le p \le \infty$, 开集 $\Omega \subseteq \mathbb{R}^n$，**Sobolev 空间**

$$
W^{k,p}(\Omega) = \{u \in L^p(\Omega) : \partial^\alpha u \in L^p(\Omega),\; \forall\, |\alpha| \le k\},
$$

范数

$$
\|u\|_{W^{k,p}} = \left(\sum_{|\alpha|\le k}\|\partial^\alpha u\|_{L^p}^p\right)^{1/p}.
$$

**记号.** $H^k(\Omega) = W^{k,2}(\Omega)$，配备内积 $\langle u, v \rangle_{H^k} = \sum_{|\alpha|\le k}\langle \partial^\alpha u, \partial^\alpha v \rangle_{L^2}$，是 Hilbert 空间。

**定义.** $W_0^{k,p}(\Omega)$ 是 $C_c^\infty(\Omega)$ 在 $W^{k,p}(\Omega)$ 中的闭包。直觉上，这些是广义意义下"在边界上为零"的 Sobolev 函数。$H_0^k(\Omega) = W_0^{k,2}(\Omega)$。

### 完备性

**定理.** $W^{k,p}(\Omega)$ 是 Banach 空间。$p = 2$ 时 $H^k(\Omega)$ 是 Hilbert 空间。

*证明.* 设 $(u_j)$ 是 $W^{k,p}$ 中的 Cauchy 列。对每个 $|\alpha| \le k$，$(\partial^\alpha u_j)$ 在 $L^p$ 中 Cauchy。由 $L^p$ 完备性，存在 $u_\alpha \in L^p$ 使得 $\partial^\alpha u_j \to u_\alpha$。令 $u = u_0$。对任何检验函数 $\varphi$：

$$
\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha\varphi \rangle = (-1)^{|\alpha|}\lim_j \langle u_j, \partial^\alpha\varphi \rangle = \lim_j \langle \partial^\alpha u_j, \varphi \rangle = \langle u_\alpha, \varphi \rangle.
$$

故 $u \in W^{k,p}$ 且 $u_j \to u$。$\square$

### 分数阶与负阶 Sobolev 空间

对 $s \in \mathbb{R}$（不必是整数），可通过 Fourier 变换定义 $H^s(\mathbb{R}^n)$：

$$
H^s(\mathbb{R}^n) = \{u \in \mathcal{S}' : (1 + |\xi|^2)^{s/2}\hat{u} \in L^2\},
$$

范数 $\|u\|_{H^s} = \|(1 + |\xi|^2)^{s/2}\hat{u}\|_{L^2}$。$s < 0$ 时 $H^s$ 包含不是函数的分布。$H^{-k}(\Omega)$ 自然等同于 $H_0^k(\Omega)$ 的对偶。

Dirac delta $\delta \in H^s(\mathbb{R}^n)$ 当且仅当 $s < -n/2$。

---

## Sobolev 嵌入定理

Sobolev 嵌入定理回答一个基本问题：若函数有 $k$ 阶 $L^p$ 导数，关于其逐点正则性能说什么？

### Sobolev 不等式（Gagliardo-Nirenberg-Sobolev）

**定理.** 设 $1 \le p < n$，定义 Sobolev 共轭指数 $p^* = np/(n-p)$。存在常数 $C = C(n,p)$ 使得对所有 $u \in W^{1,p}(\mathbb{R}^n)$，

$$
\|u\|_{L^{p^*}} \le C\|\nabla u\|_{L^p}.
$$

从而 $W^{1,p}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$（连续嵌入）。

**解读.** 有一阶 $L^p$ 导数意味着属于*更好的* $L^q$ 空间（$q = p^* > p$）。可积性的提升幅度取决于维数 $n$：维数越高，改善越少。

### Morrey 不等式

**定理.** 设 $p > n$。存在 $C = C(n,p)$ 使得对所有 $u \in W^{1,p}(\mathbb{R}^n)$，

$$
\|u\|_{C^{0,\gamma}} \le C\|u\|_{W^{1,p}},
$$

其中 $\gamma = 1 - n/p$。

**解读.** 当 $p > n$ 时，一阶 $L^p$ 导数保证*连续性*（甚至 Holder 连续性）。$p < n$ 给可积性改善；$p > n$ 给正则性。这是临界阈值。

### 一般嵌入

- $kp < n$：$W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$，$q \le np/(n-kp)$。
- $kp = n$：$W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$，$\forall\, q < \infty$。
- $kp > n$：$W^{k,p}(\Omega) \hookrightarrow C^{m,\gamma}(\overline{\Omega})$。

### Rellich-Kondrachov 紧嵌入定理

**定理.** 设 $\Omega \subset \mathbb{R}^n$ 有界且边界 Lipschitz，$1 \le p < n$。则嵌入 $W^{1,p}(\Omega) \hookrightarrow L^q(\Omega)$ 对 $1 \le q < p^*$ 是**紧的**。

**意义.** 这是驱动 PDE 变分方法的紧性结果。思想：$W^{1,p}$ 中的有界序列在 $L^q$（次临界 $q$）中有收敛子列。这是 Bolzano-Weierstrass 定理的无穷维类比。

**应用.** 证明有界区域上 Laplacian 特征值的存在性，可归结为（通过 Rellich-Kondrachov）证明 $-\Delta$ 的预解是紧算子，再应用本系列前面的紧算子谱定理。

---

## 迹定理与边界值

$W^{1,p}(\Omega)$ 中的函数仅在零测集意义下确定。边界 $\partial\Omega$ 在 $\mathbb{R}^n$ 中是零测集，因此逐点限制到边界在 Lebesgue 意义下无法定义。然而 Dirichlet 边界条件 $u|_{\partial\Omega} = 0$ 或 Neumann 边界条件 $\partial u/\partial n|_{\partial\Omega} = g$ 对 PDE 至关重要。

**迹定理**表明限制到边界可以延拓为 Sobolev 空间上的连续运算。

**定理（迹定理）.** 设 $\Omega \subset \mathbb{R}^n$ 有界，边界 Lipschitz。存在唯一的有界线性算子

$$
\gamma_0: W^{1,p}(\Omega) \to L^p(\partial\Omega),
$$

使得 $\gamma_0 u = u|_{\partial\Omega}$（$\forall\, u \in C^\infty(\overline{\Omega})$）。且：

1. $\gamma_0$ 满射到 $W^{1-1/p,\,p}(\partial\Omega)$（边界上的分数阶 Sobolev 空间）。
2. $\ker \gamma_0 = W_0^{1,p}(\Omega)$。

**解读.** 性质 2 给出 $W_0^{1,p}$ 的精确刻画：恰好是那些迹（边界值）为零的 Sobolev 函数。"$H_0^1(\Omega)$ 中的函数在 $\partial\Omega$ 上为零"这一说法由此获得严格含义。

对**高阶 Sobolev 空间**，有高阶迹算子 $\gamma_j u = \partial^j u/\partial n^j|_{\partial\Omega}$（$j$ 阶法向导数），且 $(\gamma_0, \ldots, \gamma_{k-1})$ 的核恰是 $W_0^{k,p}(\Omega)$。

---

## 下一步

我们构建了现代 PDE 理论所需的分布与 Sobolev 空间基础：分布赋予经典不存在的导数以意义，Sobolev 空间提供弱公式化的 Banach/Hilbert 框架，嵌入定理将可积性与正则性相连，迹定理严格处理边界条件。

在本系列的最后一篇文章中，一切将汇聚起来。**Lax-Milgram 定理**给出椭圆边值问题的存在唯一性。**变分方法**将 PDE 化为极小化问题。**Stone 定理**将自伴算子与量子动力学联系在一起。泛函分析的抽象机器在这里与最重要的应用相遇。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 11 篇（共 12 篇）。*

*上一篇：[第 10 篇 —— 算子半群](/zh/functional-analysis/10-算子半群/)*

*下一篇：[第 12 篇 —— 应用](/zh/functional-analysis/12-应用/)*
