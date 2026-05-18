---
title: "泛函分析 (11)：分布与Sobolev空间 — 广义解"
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
description: "分布扩展了函数的概念，以处理经典上不存在的导数 — Sobolev空间为PDE的弱解提供了合适的框架。"
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "functional-analysis-11"
---

我想从一个坦白开始。多年来，我像一个本科生物理学家那样对待Dirac delta：它在原点以外处处为零，在原点处无穷大，并且其积分等于一。这种描述当然是数学上的无稽之谈。没有可测函数具有这些性质。然而，每本量子力学教科书都在第一页使用$\delta$，每个信号处理课程都用$\delta(t)$表示脉冲，每本PDE书都调用满足$\Delta E = \delta$的Green函数$E$。要么整个科学界在过去一个世纪里犯了一个根本性的错误，要么有一种方法可以使这个对象变得严格。显然是后者——而这种方法就是分布理论。

让我具体说说这个尴尬到底有多大。我教过一个本科生求线上 $-u'' = \delta$ 的解，他写下 $u(x) = -|x|/2 + Ax + B$，然后向我求证。这个答案是对的——它出现在每本电动力学的教科书里。但如果我把 $u'' = -\delta$ 当成普通微分方程，它根本没有“点态”意义：在 $x \neq 0$ 处 $u''(x) = 0$，在 $x = 0$ 处 $u''$ 不存在，整个等式形式上不成立。然而我们日常用这条方程做计算。这中间一定有某种数学机制，让“分布意义下的导数”和经典导数能在它们都有意义时一致，而在经典导数失效的地方依然能给出可预测的答案。

这个问题比$\delta$本身还要古老。考虑线上的波动方程$u_{tt} = c^2 u_{xx}$。任何二阶可微的轮廓$f$给出一个行波解$u(x,t) = f(x - ct)$。但物理波携带冲击波：浅水方程的解可以发展成一个台阶，声波可以有尖锐的前沿，光脉冲可以是方形包络。“函数”$f$在这种情况下甚至不连续。称这样的不连续$u$为$u_{tt} = c^2 u_{xx}$的“解”要求我们对它求两次导数，而经典的二阶导数在冲击波处或其两侧都不存在，因为对指示函数求导会得到一个delta。

![Delta 分布作为凸函数的极限](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_delta_distribution.png)

Laurent Schwartz的分布理论（1944-1950）用一个技巧解决了这两个问题。停止尝试将点值赋给广义“函数”。相反，通过它们如何作用于光滑测试函数来定义它们：$f$是线性映射$\varphi \mapsto \int f\varphi$。几乎处处相等的两个函数给出相同的映射，因此$L^1_{\text{loc}}$嵌入到测试函数的对偶空间中。但该对偶空间远大于$L^1_{\text{loc}}$；它包含$\delta$、$\delta$的导数、主值分布以及许多其他结构。一旦有了对偶空间，想要的所有操作——导数、傅里叶变换、卷积——都可以通过形式对偶从光滑函数扩展到所有分布。

Sobolev空间是故事的第二部分。Schwartz的分布太大，不能成为有用的Banach空间（$C_c^\infty$的对偶没有自然的范数拓扑）。对于PDE，需要具体的Hilbert空间，带有范数、嵌入和紧性。Sergei Sobolev在1930年代的构造正好做到了这一点：$W^{k,p}(\Omega)$包含那些直到$k$阶的分布导数属于$L^p$的函数。这些是微分算子的自然域，是弱解的合适设置，并且它们附带三个关键工具——嵌入定理、迹定理、Rellich-Kondrachov紧性——没有这些，下一篇文章中的Lax-Milgram定理就无法发挥作用。

---

## 为什么经典导数不够

### 三个动机问题

![动画：磨光算子对函数的光滑化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/gif11_mollification.gif)

考虑$(0,1)$上的$-u'' = f$，边界条件为$u(0) = u(1) = 0$。如果$f$是连续的，经典解存在且是$C^2$的。但如果$f \in L^2(0,1)$——比如$f$是$[1/3, 2/3]$的指示函数呢？没有经典的$C^2$解存在；$u''$必须跳跃。然而，乘以一个测试函数$\varphi \in C_c^\infty(0,1)$并两次分部积分（第二次积分不会产生边界项，因为$\varphi$有紧支集）给出

$$
\int_0^1 u'\varphi' \, dx = \int_0^1 f\varphi \, dx \quad \text{对于所有 } \varphi \in C_c^\infty(0,1).
$$

满足这个积分恒等式的函数$u$称为弱解不必是$C^2$的；$u \in H^1_0(0,1)$——一个弱导数在$L^2$中，边界处消失——就足够了。这里的问题实际上有一个显式答案：两次积分$f$并调整常数。你得到一个$C^1$的分段二次函数，但它不是$C^2$的（二阶导数在$x = 1/3$和$x = 2/3$处跳跃），这正是Sobolev理论预测的正则性：$f \in L^2$并且一次应用$-d^2/dx^2$反演后获得两个导数，所以$u \in H^2$，从而$u \in C^{1,1/2}$但不是$C^2$。经典理论会简单地宣布这个问题不可解。

![测试函数：紧支集的光滑凸起](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_1_test_functions.png)

在静电学中，单位点电荷在原点的势满足$-\Delta\phi = \delta$。右边不是一个函数。更糟的是，解$\phi(x) = 1/(4\pi|x|)$在原点处奇异，因此在那里也不是经典意义上的二阶可微。整个方程生活在一个“导数”必须重新解释的世界中。

常数函数$1$的Fourier变换，形式上是$(2\pi)^n\delta$。$|x|$的Fourier变换涉及$1/|\xi|^{n+1}$（一个非局部可积函数，读作主值）。如果没有包含$\delta$及其导数的框架，大量调和分析就会崩溃或需要临时修补。缓增分布$\mathcal{S}'$提供了正确的框架：每个缓增分布都有Fourier变换，并且变换是一个拓扑同构$\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$。

### 概念转变

经典方法：函数$f$由其点值$f(x)$定义。分布方法：广义函数$f$由线性泛函$\varphi \mapsto \int f\varphi\,dx$定义。

两个局部可积函数如果在零测度集上不同，则定义相同的分布；它们对测试函数的积分相同。这意味着分布自动商出零集，正如$L^p$空间所做的那样。点值在$L^p$中始终是虚构的；分布只是公开承认这一虚构，并不再假装可以在某一点上进行评估。

### $\mathcal{D}'(\Omega)$上的拓扑

分布空间带有**弱-*拓扑**：$u_j \to u$在$\mathcal{D}'(\Omega)$中当且仅当$\langle u_j, \varphi \rangle \to \langle u, \varphi \rangle$对于每个$\varphi \in \mathcal{D}(\Omega)$。这是测试函数上的逐点收敛，比有界集上的均匀收敛弱，但对于大部分PDE理论已经足够。

一个重要结果：每个分布都是光滑函数序列在$\mathcal{D}'$中的极限。如果$u \in \mathcal{D}'(\Omega)$且$\rho_\epsilon$是标准磨光器，则$u * \rho_\epsilon \to u$在$\mathcal{D}'$中当$\epsilon \to 0$（其中卷积通过转置定义）。因此，分布是“光滑函数的极限”，这是一个严格的含义——正如实数是有限小数的极限，$L^p$函数是简单函数的极限。

---

## 测试函数$\mathcal{D}(\Omega)$和分布$\mathcal{D}'(\Omega)$

### 测试函数空间

我刚开始读分布理论时被一个问题困扰：为什么测试函数偏要选 $C_c^\infty$（光滑且紧支撑），不能用 $C^\infty$ 或 $C_c^k$？答案分两块。**支撑紧**让分布意义下的分部积分可以无忧无虑地把导数搬到测试函数上——边界项消失，所以 $\langle \partial^\alpha u, \varphi\rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha \varphi\rangle$ 这条定义在整个 $\mathcal{D}'$ 上没有歧义。**$C^\infty$ 光滑**让“分布有任意阶导数”成立——只要测试函数能任意求导，分布在对偶意义下就可以被任意求导。这两条性质合起来让测试函数空间“尽可能小、性质尽可能多”，从而对偶空间 $\mathcal{D}'$ 尽可能大。

实际写下一个具体的测试函数也并不平凡。$C^\infty$ 光滑加紧支撑听起来矛盾——多项式光滑但不紧支撑，指示函数紧支撑但不光滑。经典构造是 $\rho(x) = \exp(-1/(1-|x|^2)) \cdot \mathbf{1}_{|x|<1}$：在 $|x|<1$ 时是 $\exp$ 复合多项式，光滑；在 $|x|=1$ 时所有导数都从两侧匹配为零（因为 $\exp(-1/0^+) = 0$ 比任何多项式衰减得快）；在 $|x|>1$ 时恒为零。这个看似魔术的拼接是分布理论一切构造的起点：磨光器、单位分解、cut-off 函数都是这个 bump 的变形。

![试验函数：光滑且紧支](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_test_functions.png)

设$\Omega \subseteq \mathbb{R}^n$是开集。**测试函数**空间是

$$
\mathcal{D}(\Omega) = C_c^\infty(\Omega),
$$

即在$\Omega$中有紧支集的无限可微函数空间。标准例子是凸起$\varphi(x) = \exp(-1/(1-|x|^2))$对于$|x| < 1$，其余部分为零。它在各处都是$C^\infty$的（在$|x|=1$处所有导数的匹配是一个精细的微积分练习），非负，并且支集在闭单位球内。

序列$\varphi_j \to 0$在$\mathcal{D}(\Omega)$中如果：
1. 存在一个紧集$K \subset \Omega$使得$\text{supp}(\varphi_j) \subset K$对于所有$j$。
2. 对于每个多重指标$\alpha$，$\partial^\alpha \varphi_j \to 0$在$K$上一致收敛。

这不是*范数拓扑*；它是归纳极限，最细的局部凸拓扑使每个紧集$K \subset \Omega$的包含$C_c^\infty(K) \hookrightarrow C_c^\infty(\Omega)$连续。细节很重要，因为它保证了对偶$\mathcal{D}'(\Omega)$足够大，包含想要的对象。

### 分布

$\Omega$上的**分布**是连续线性泛函$u: \mathcal{D}(\Omega) \to \mathbb{C}$——等价地，线性$u$使得对于每个紧集$K \subset \Omega$，存在$C, N \ge 0$使得

“分布是连续线性泛函”这一定义看似抽象，但它实际上把分布完全刻画了：分布就是“可以与测试函数配对得出一个数”的对象。任何具体分布都可以通过指定它如何作用在每个测试函数上来定义——比如 $\delta$ 通过 $\langle\delta,\varphi\rangle = \varphi(0)$ 定义，$\delta'$ 通过 $\langle\delta',\varphi\rangle = -\varphi'(0)$ 定义。分布之间的运算也通过对偶性继承：每个测试函数上的连续线性运算都自动诱导分布上的运算。

这种“通过测试函数定义”的方法是泛函分析的核心模式。第四篇里的对偶空间是同样的思路——通过线性泛函刻画向量；第五篇里的弱拓扑是同样的思路——通过对偶配对定义收敛；这里的分布是把这种思路推到极致：把广义函数完全用它们对测试函数的作用来定义。一旦适应这种思路，$\delta$ 就不再是“无穷大点函数”这种数学谎言，而是一个良定义的对偶对象。

$$
|\langle u, \varphi \rangle| \le C \sum_{|\alpha| \le N} \sup_K |\partial^\alpha \varphi| \quad \text{对于所有 } \varphi \in C_c^\infty(K).
$$

最小的这样的$N$是$u$在$K$上的**阶**。局部可积函数是阶为$0$的分布；Dirac delta是阶为$0$的；它的导数是更高阶的。

关键例子：

1. 局部可积函数：每个$f \in L^1_{\text{loc}}(\Omega)$定义一个分布$\langle f, \varphi \rangle = \int_\Omega f\varphi\,dx$。

2. Dirac delta：$\langle \delta_a, \varphi \rangle = \varphi(a)$。这*不是*上述形式；没有$L^1_{\text{loc}}$函数能满足定义恒等式（取$\varphi$为收缩到$a$的一系列凸起显示$f$必须以不允许的方式“集中”）。

3. Heaviside：$H(x) = 1$对于$x \ge 0$，否则为$0$。它的分布导数是$\delta$——一个分部积分：$\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x)\,dx = \varphi(0)$。

4. 主值$1/x$：函数$1/x$在$\mathbb{R}$上在$0$附近不是局部可积的，但主值$\langle \mathrm{p.v.} \tfrac{1}{x}, \varphi \rangle = \lim_{\epsilon \to 0}\int_{|x|>\epsilon}\varphi(x)/x\,dx$定义了一个阶为$1$的分布。

![像Dirac delta这样的分布在测试函数上的作用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_2_distributions.png)

### 分布上的运算

统一原则：通过形式对偶定义每个运算。如果$T$是测试函数上的连续线性映射，伴随为$T^*$，则通过$\langle Tu, \varphi \rangle = \langle u, T^*\varphi \rangle$定义$Tu$在分布上的作用。

这条“通过对偶定义运算”的原则是分布理论里最简洁的设计选择。看似抽象，但它把所有想要的运算自动给出了具体定义：导数 $\partial^\alpha$、与光滑函数的乘法、平移、缩放、Fourier 变换——每一个都是把测试函数上对应的伴随运算搬到分布上。这种设计的好处是“向后兼容”：在测试函数本身就是合法的分布（通过 $\varphi \mapsto \int \varphi \psi$）这一身份下，新定义的运算与经典运算一致——所以分布意义下的导数对 $C^1$ 函数等于经典导数，分布意义下的 Fourier 变换对 $L^1$ 函数等于经典 Fourier 变换。

代价是某些经典运算无法搬过来。两个分布的乘法没有典范定义——$\delta \cdot \delta$ 没意义、$1/x \cdot \delta$ 没意义、一般两个非零分布的乘积都没有典范定义。这是 Schwartz 不可能性定理的内容：在 $\mathcal{D}'$ 上不存在结合的、交换的、扩展点态乘法且满足 Leibniz 法则的乘法。这个否定性结果是非线性 PDE 困难的来源——非线性 PDE 形式上要把分布的乘积取意义，但这件事一般做不到，必须依赖额外的正则性（比如解在某些 Sobolev 空间中）才能让乘积合法。所以非线性 PDE 的弱解理论本质上是“在哪些情形下可以定义乘积”这个问题。

$\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha \varphi \rangle$。符号来自分部积分：$\int \partial f \cdot \varphi = -\int f \cdot \partial\varphi$对于光滑的$f$和紧支集的$\varphi$。这个公式将链式法则扩展到整个对偶空间；*每个*分布都是分布意义下的无限可微的，并且微分在弱-*拓扑中是连续的。与经典情况相比，微分是无界的，并且不与极限交换。

如果$a \in C^\infty(\Omega)$，$\langle au, \varphi \rangle = \langle u, a\varphi \rangle$——测试函数$a\varphi$仍然是$C_c^\infty$。

任意两个分布的乘法*未*定义。$\delta \cdot \delta$没有规范意义；试图定义它会导致QFT中的重整化。Schwartz的不可能性定理使这一点精确：在$\mathcal{D}'$上没有关联的、交换的乘法，它扩展连续函数上的点乘法并满足Leibniz规则。非线性PDE和量子场论中最深的困难源于这个单一的否定结果。

### 当经典导数存在时，为什么分布导数是唯一的

一个自然的担忧：当两者都定义时，新导数是否与旧导数一致？是的。如果$f \in C^1$，那么对于任何$\varphi \in C_c^\infty$，分部积分给出$\int f'\varphi = -\int f\varphi'$，这正是分布的定义。对于$C^1$函数，分布导数与经典导数一致，对于$W^{1,p}$函数，与几乎处处导数一致，而对于真正奇异的对象如$H'$（因为经典导数在那里根本不存在）则不一致。

关于$\mathbb{R}$上的$|x|$的一个具体例子：它在原点外处处可微，但在原点处经典导数未定义。分布导数是符号函数$\mathrm{sgn}(x) = H(x) - H(-x)$，它几乎处处定义，并且在经典导数存在的地方与其一致。再微分一次：$|x|'' = (\mathrm{sgn})' = 2\delta$，因为符号函数在原点处跳跃2。这个具体的计算——$|x|$的第二个分布导数是$2\delta$——是我每隔几个月就要重新推导一次的标准检查。它确保你的分布记账是正确的。

### 卷积和平滑化

如果$u \in \mathcal{D}'(\mathbb{R}^n)$有紧支集且$\varphi \in C^\infty(\mathbb{R}^n)$，卷积$u * \varphi$被定义且是$C^\infty$的：

$$
(u * \varphi)(x) = \langle u_y, \varphi(x - y)\rangle.
$$

更一般地，与紧支集分布的卷积对于任何分布都是良好定义的。关键性质：

- $\delta * f = f$对于任何分布$f$——delta是卷积单位。
- $\partial^\alpha(u * v) = (\partial^\alpha u) * v = u * (\partial^\alpha v)$——导数可以在因子之间转移。
- **平滑化：** 如果$\rho_\epsilon(x) = \epsilon^{-n}\rho(x/\epsilon)$是标准磨光器（正的，$C_c^\infty$，积分1，支集在单位球内），则对于任何分布$u$，$u * \rho_\epsilon \in C^\infty$，并且$u * \rho_\epsilon \to u$在$\mathcal{D}'$中当$\epsilon \to 0$。

平滑化事实是分布理论的工作马。它说每个分布都可以用光滑函数逼近，逼近是明确且可计算的。大多数PDE证明遵循以下模式：先对光滑$u$证明结果，然后平滑化并通过极限传递。

### 缓增分布和Fourier变换

**Schwartz空间** $\mathcal{S}(\mathbb{R}^n)$由所有导数都比任何多项式衰减得更快的光滑函数组成：$\sup_x |x^\alpha \partial^\beta \varphi(x)| < \infty$对于所有多重指标。其对偶$\mathcal{S}'(\mathbb{R}^n)$是**缓增分布**空间——严格小于$\mathcal{D}'$但足够大，包括多项式、所有$p$的$L^p$、$\delta$及其导数，以及调和分析中关心的大多数东西。

Fourier变换通过$\langle \hat{u}, \varphi \rangle = \langle u, \hat{\varphi} \rangle$扩展为$\mathcal{S}'$上的同构$\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$，利用$\mathcal{F}$已经将$\mathcal{S}$自身双连续映射的事实。关键公式：

- $\hat{\delta} = 1$，
- $\hat{1} = (2\pi)^n \delta$，
- $\widehat{\partial^\alpha u} = (i\xi)^\alpha \hat{u}$——微分变为多项式乘法，对*所有*缓增分布有效。

这些恒等式为Fourier分析PDE提供动力：解$-\Delta u = f$变成$|\xi|^2\hat{u} = \hat{f}$，所以$\hat{u} = \hat{f}/|\xi|^2$——难点纯粹在于逆变换和$\xi = 0$处的行为，而不是代数步骤。

### 基本解

线性微分算子$L$的**基本解**是满足$LE = \delta$的分布$E$。对于$\mathbb{R}^n$上的Laplacian ($n \ge 3$)：

$$
E(x) = \frac{1}{n(n-2)\omega_n|x|^{n-2}},
$$

其中$\omega_n$是单位球的体积。在$\mathbb{R}^2$中，$E(x) = -\frac{1}{2\pi}\log|x|$；在$\mathbb{R}^1$中，$E(x) = -\frac{1}{2}|x|$。解$-\Delta u = f$（对于适当的$f$）则是$u = E * f$，与基本解的卷积。这是经典位势理论的分布实现。

对于热方程$(\partial_t - \Delta)u = 0$，基本解是热核$K(x, t) = (4\pi t)^{-n/2}e^{-|x|^2/(4t)}$对于$t > 0$，与半群文章中出现的相同。分布视角澄清了为什么会出现这个核：它是唯一满足$(\partial_t - \Delta)K = \delta(x)\delta(t)$的缓增分布。

---

## 弱导数和Sobolev空间

### 从分布导数到弱导数

![Sobolev 空间层次](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_sobolev_spaces.png)

分布导数总是存在的；它们是$\mathcal{D}'$中的抽象对象。对于PDE，我们希望具体的导数生活在$L^p$中。

如果$u \in L^1_{\text{loc}}(\Omega)$在方向$\partial^\alpha$上有**弱导数**$g \in L^1_{\text{loc}}(\Omega)$，则

$$
\int_\Omega u\,\partial^\alpha\varphi\,dx = (-1)^{|\alpha|}\int_\Omega g\,\varphi\,dx \quad \text{对于所有 } \varphi \in C_c^\infty(\Omega).
$$

弱导数，当它存在时，在几乎处处是唯一的，并且在两者都有意义时与经典导数一致。许多在经典意义上不可微的函数具有弱导数：$|x|$的弱导数是$\mathrm{sgn}(x)$；$\max(u, 0)$的弱导数是$u'\cdot\mathbf{1}_{u>0}$；绝对连续函数在线上具有与其经典几乎处处导数相等的弱导数。

![通过与测试函数积分定义的弱导数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_3_weak_deriv.png)

### Sobolev空间$W^{k,p}$

引入 Sobolev 空间之前先讲一下它解决什么问题。分布空间 $\mathcal{D}'$ 太大——它甚至没有一个有用的范数拓扑，无法做精细的存在性证明。$L^p$ 空间太小——它装得下函数但装不下“函数的导数”，差分算子在 $L^p$ 上无界。Sobolev 空间正好填这个缝：要求函数和它的（弱）导数都在 $L^p$ 中，把“正则性”量化成范数大小。这一来微分算子在 $W^{k,p} \to W^{k-1,p}$ 上有界，PDE 弱形式的双线性形式有自然定义域，连续性和强制性可以用具体范数验证。

$W^{k,p}$ 还有第二层意义：它把“正则性”和“可积性”绑在一起，并通过 Sobolev 嵌入定理把这两者互换。如果一个函数在 $L^p$ 中有 $k$ 个导数，那么它本身可能就在更大的 $L^q$（$q > p$）中，甚至是连续 Hölder 函数（如果 $kp > n$）。这一类“正则性 → 可积性” 或 “正则性 → 逐点正则性”的兑换是 PDE 解的正则性理论的全部内容。

对于$k \in \mathbb{N}_0$，$1 \le p \le \infty$，和$\Omega \subseteq \mathbb{R}^n$开集，**Sobolev空间** $W^{k,p}(\Omega)$是

$$
W^{k,p}(\Omega) = \{u \in L^p(\Omega) : \partial^\alpha u \in L^p(\Omega) \text{ 对于所有 } |\alpha| \le k\},
$$

其中$\partial^\alpha u$表示弱（等价地，分布）导数。范数是

$$
$\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|\partial^\alpha u\|_{L^p}^p\right)^{1/p} \quad (1 \le p < \infty)$
$$

对于$p = \infty$的情况做相应修改。

$H^k(\Omega) = W^{k,2}(\Omega)$。这些是带有内积$\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \langle \partial^\alpha u, \partial^\alpha v \rangle_{L^2}$的Hilbert空间。Hilbert结构使$H^k$成为Lax-Milgram定理和变分方法的自然设置。

$W_0^{k,p}(\Omega)$是$C_c^\infty(\Omega)$在$W^{k,p}(\Omega)$中的闭包。直观上，这些是在广义意义上“在边界上消失”的Sobolev函数；下面的迹定理使其精确。

### 完备性

可以证明，$W^{k,p}(\Omega)$是Banach空间；$H^k(\Omega)$是Hilbert空间。

*证明概要。* 设$(u_j)$是$W^{k,p}$中的Cauchy序列。对于每个$|\alpha| \le k$，序列$(\partial^\alpha u_j)$是$L^p(\Omega)$中的Cauchy序列。由$L^p$的完备性，存在$u_\alpha \in L^p$使得$\partial^\alpha u_j \to u_\alpha$在$L^p$中。设$u = u_0$。我们声称$\partial^\alpha u = u_\alpha$在分布意义下成立：

$$
$\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha\varphi \rangle = (-1)^{|\alpha|}\lim_j \langle u_j, \partial^\alpha\varphi \rangle = \lim_j \langle \partial^\alpha u_j, \varphi \rangle = \langle u_\alpha, \varphi \rangle$。
$$

因此$u \in W^{k,p}$且$u_j \to u$。$\square$

### 数值例子：$|x|^\alpha$的正则性

取$u(x) = |x|^\alpha$在单位球$B \subset \mathbb{R}^n$上。何时$u \in W^{1,p}(B)$？弱梯度是$\nabla u = \alpha|x|^{\alpha-2}x$（扩展经典公式），并且

$$
$\int_B |\nabla u|^p\,dx = |\alpha|^p \int_B |x|^{p(\alpha-1)}\,dx = |\alpha|^p \omega_{n-1}\int_0^1 r^{p(\alpha-1)+n-1}\,dr$
$$

这个积分收敛当且仅当$p(\alpha-1) + n - 1 > -1$，即$\alpha > 1 - n/p$。因此$|x|^\alpha \in W^{1,p}(B)$恰好当$\alpha > 1 - n/p$。在$n = 3$，$p = 2$的情况下，这给出$\alpha > -1/2$，所以$|x|^{-1/2}$勉强*不在*$H^1$中，但$|x|^{-1/4}$在。这种明确的阈值指导了奇异PDE解的预期正则性。

第二个数值例子：三维中的Sobolev嵌入边界。对于$n = 3$和$p = 2$，$p^* = 6$，因此$H^1(\mathbb{R}^3) \hookrightarrow L^6(\mathbb{R}^3)$。具体来说，Sobolev不等式说$\|u\|_{L^6} \le C\|\nabla u\|_{L^2}$对于任何紧支集光滑$u$。最优常数$C$由Talenti在1976年计算得出：$C = \frac{1}{\pi}\sqrt[3]{\frac{1}{4}\Gamma(3/2)/\Gamma(3)}$，极值函数恰好是Aubin-Talenti气泡$u_\epsilon(x) = c_n(\epsilon^2 + |x|^2)^{-(n-2)/2}$（这里$c_n$是归一化）。插入$\epsilon = 1$，$n = 3$：$u_1(x) = c_3(1 + |x|^2)^{-1/2}$，并且$\|u_1\|_{L^6}/\|\nabla u_1\|_{L^2} = C$恰好成立。极值函数的存在（Aubin，Talenti）是微妙的；它们在缩放$u \mapsto \epsilon^{1/2}u(\epsilon x)$下形成非紧轨道是非线性分析中每个集中现象的来源。

### 分数阶和负Sobolev空间

对于$s \in \mathbb{R}$（不一定为整数），通过Fourier变换定义$H^s(\mathbb{R}^n)$：

$$
H^s(\mathbb{R}^n) = \{u \in \mathcal{S}'(\mathbb{R}^n) : (1 + |\xi|^2)^{s/2}\hat{u} \in L^2(\mathbb{R}^n)\},
$$

范数为$\|u\|_{H^s} = \|(1 + |\xi|^2)^{s/2}\hat{u}\|_{L^2}$。对于$s < 0$，$H^s$包含不是函数的分布。空间$H^{-k}(\Omega)$是对偶空间$H_0^k(\Omega)$的对偶。

Dirac delta在$H^s(\mathbb{R}^n)$中当且仅当$s < -n/2$（因为$\hat{\delta} = 1$且$(1 + |\xi|^2)^{s/2} \in L^2$当且仅当$s < -n/2$）。因此$\delta \in H^{-1-\epsilon}(\mathbb{R}) \setminus H^{-1/2}(\mathbb{R})$，维度越高，$\delta$越“奇异”。

![不同正则性的Sobolev嵌入链](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_4_sobolev_chain.png)

### 密度和逼近

Sobolev空间的一个基本性质是光滑函数是稠密的：

可以证明，$C^\infty(\Omega) \cap W^{k,p}(\Omega)$在$W^{k,p}(\Omega)$中是稠密的，对于$1 \le p < \infty$。

当$\Omega$有Lipschitz边界时，甚至$C^\infty(\overline{\Omega})$在$W^{k,p}(\Omega)$中也是稠密的。这种逼近性质对于证明定理至关重要：首先对光滑函数建立结果（此时经典微积分适用），然后通过密度推广。

### Poincare不等式

可以证明，设 $\Omega \subset \mathbb{R}^n$ 是有界的且连通的。存在 $C = C(\Omega, p)$ 使得对于所有 $u \in W_0^{1,p}(\Omega)$，

Poincare 不等式是 PDE 弱解理论里出现频率最高的具体不等式之一。它说的是：在边界上消失的函数（$W^{1,p}_0$）的 $L^p$ 范数被它的梯度的 $L^p$ 范数控制。这条结论看似无关紧要，但它是 Lax-Milgram 应用于 Dirichlet 问题时验证强制性的关键步骤——双线性形式 $a(u,v) = \int \nabla u \cdot \nabla v$ 满足 $a(u,u) = \|\nabla u\|^2$，要让它强制控制 $\|u\|_{H^1}^2 = \|u\|^2 + \|\nabla u\|^2$，必须有 $\|u\| \leq C\|\nabla u\|$，这正是 Poincare。

Poincare 不等式之所以成立靠的是边界条件 $u|_{\partial\Omega} = 0$。没有这个条件，$u \equiv 1$ 是合法的 $W^{1,p}$ 函数，$\|\nabla u\| = 0$ 但 $\|u\| > 0$，不等式失效。$W^{1,p}_0$ 强制函数在边界上消失，这意味着 $u$ 内部的“质量”可以通过它的“坡度”反推出来——质量不能凭空出现。这种几何直觉让 Poincare 常数 $C$ 与区域几何相关：$C$ 大约等于 $\Omega$ 的直径，所以小区域有小的 Poincare 常数（强制性更强），大区域有大的 Poincare 常数（强制性更弱）。

$$
\|u\|_{L^p(\Omega)} \le C\|\nabla u\|_{L^p(\Omega)}.
$$

对于在边界上消失的函数，$L^p$ 范数由梯度单独控制。这是在 $W_0^{1,p}(\Omega)$ 上 $\|\nabla u\|_{L^p}$ 是等价范数的关键步骤，并且它是下一章中 Lax-Milgram 应用于椭圆 PDE 的动力。Poincare 常数按 $\Omega$ 的直径缩放：对于半径为 $R$ 的球，$C \sim R$。

一个变体，**Poincare-Wirtinger不等式**，适用于没有边界条件的函数：$\|u - \bar{u}\|_{L^p} \le C\|\nabla u\|_{L^p}$，其中$\bar{u}$是$\Omega$上$u$的平均值。这对于Neumann问题相关，其中解仅确定到一个加性常数。

---

## Sobolev嵌入定理

嵌入定理回答了一个基本问题：如果一个函数在$L^p$中有$k$个导数，我们能对其逐点正则性说什么？

写出嵌入定理之前先讲一下它要算什么。直觉上“多一个导数”意味着函数“更光滑”，但“光滑”这个词在 PDE 里有两种具体含义：可以是“在更高 $L^q$ 中”（更可积），也可以是“逐点连续甚至 Hölder 连续”。Sobolev 嵌入定理告诉我两件事的兑换汇率取决于维数 $n$ 和指数 $p$：低维或大 $p$ 时，一个导数能直接换来逐点连续性；高维或小 $p$ 时，一个导数只能换来更高 $L^q$ 可积性，要拿到逐点连续需要更多导数。

具体的兑换由 Sobolev 共轭指数 $p^* = np/(n-p)$ 控制（当 $p < n$ 时）。$W^{1,p} \hookrightarrow L^{p^*}$ 是说一个导数把可积性从 $L^p$ 提升到 $L^{p^*}$；当 $kp = n$ 时正好达到嵌入到所有 $L^q$ 的临界（$q < \infty$）；当 $kp > n$ 时跨过阈值，得到逐点连续性甚至 Hölder 连续性。这条阈值在三维情形特别有意义：$H^1(\mathbb{R}^3) \hookrightarrow L^6$ 但 $H^1$ 函数不必连续，需要 $H^2$ 才有连续性。这就是为什么在三维 PDE 中“弱解的连续性”是个非平凡问题。

![一维 Sobolev 嵌入](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_sobolev_embedding.png)

### Sobolev不等式（Gagliardo-Nirenberg-Sobolev）

可以证明，设$1 \le p < n$并定义$p^* = np/(n-p)$（**Sobolev共轭指数**）。则存在$C = C(n, p)$使得对于所有$u \in W^{1,p}(\mathbb{R}^n)$，

$$
\|u\|_{L^{p^*}} \le C\|\nabla u\|_{L^p}.
$$

因此$W^{1,p}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$连续嵌入。

在$L^p$中有一个导数会使你进入*更好*的$L^q$空间，其中$q = p^* > p$。增益由维度决定：在$n = 3$，$p = 2$的情况下，$p^* = 6$，因此$H^1(\mathbb{R}^3) \hookrightarrow L^6(\mathbb{R}^3)$——每个梯度平方可积的函数都是六次幂可积的。随着$n$增长，增益缩小；当$n \to \infty$时，$p^* \to p$且嵌入变得平凡。高维给出较少的可积性改进。

### Morrey不等式

可以证明，设$p > n$。则存在$C = C(n, p)$使得对于所有$u \in W^{1,p}(\mathbb{R}^n)$，

$$
\|u\|_{C^{0,\gamma}(\mathbb{R}^n)} \le C\|u\|_{W^{1,p}},
$$

其中$\gamma = 1 - n/p$且$C^{0,\gamma}$是Holder连续函数空间，指数为$\gamma$。

当$p > n$时，在$L^p$中有一个导数保证*连续性*甚至Holder连续性。跨越阈值$p = n$是关键过渡：$p < n$给出可积性改进，$p > n$给出逐点正则性。这就是为什么$H^1(\mathbb{R})$函数是连续的（因为$1 < 2 = p$对于迹维度），但$H^1(\mathbb{R}^2)$和$H^1(\mathbb{R}^3)$函数不必如此——一个导数增益不足以逃脱$n \ge 2$时的可积性。

### 一般嵌入

对于$kp < n$：$W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$对于$q \le np/(n - kp)$。

对于$kp = n$：$W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$对于所有$q < \infty$——临界情况，其中出现对数修正。

对于$kp > n$：$W^{k,p}(\Omega) \hookrightarrow C^{m,\gamma}(\overline{\Omega})$其中$m = k - \lfloor n/p \rfloor - 1$且$\gamma$取决于分数部分。在这种情况下，Sobolev函数是经典可微的。

### Rellich-Kondrachov紧性定理

可以证明，设$\Omega \subset \mathbb{R}^n$是有界的且有Lipschitz边界，$1 \le p < n$。则$W^{1,p}(\Omega) \hookrightarrow L^q(\Omega)$对于$1 \le q < p^*$是**紧**的。

这是推动变分方法的紧性结果。$W^{1,p}$中的有界序列有一个子序列在任何次临界$q$中收敛于$L^q$。这是无穷维的Bolzano-Weierstrass定理：用“更强范数下的有界闭集”代替“有界闭集”。有界域上Laplacian的特征值存在性（通过Rellich-Kondrachov）简化为$-\Delta$的预解算子的紧性——之前章节中的紧算子谱定理随后给出谱。

### 临界指数处的紧性失效

Rellich-Kondrachov在$q = p^*$处失败：嵌入$W^{1,p}(\Omega) \hookrightarrow L^{p^*}(\Omega)$是连续的但*不*是紧的。这种失败对非线性PDE有深远影响。Sobolev不等式的极值（Aubin，Talenti，1976）和相关的变分问题表现出集中是指极小化序列可以坍缩到一点，失去紧性。浓度-紧性原理（Lions，1984）列举了紧性可能失效的方式，并通过附加结构恢复它。大多数临界指数非线性问题——Yamabe问题、规定标量曲率、共形几何——完全处于这种状态。

### 扩展定理

嵌入从$\mathbb{R}^n$扩展到边界正则的区域$\Omega$：

可以证明，设$\Omega \subset \mathbb{R}^n$是有界的且有Lipschitz边界。存在有界线性扩展算子 $E: W^{k,p}(\Omega) \to W^{k,p}(\mathbb{R}^n)$，使得 $Eu|_\Omega = u$。

扩展定理说明，$\Omega$ 上的 Sobolev 函数总可以被延拓到整个 $\mathbb{R}^n$ 上而不损失正则性。这把所有 $\mathbb{R}^n$ 上证明的嵌入定理（Sobolev 不等式、Morrey 不等式、Rellich-Kondrachov）自动转移到 $\Omega$ 上：先扩展到 $\mathbb{R}^n$、应用 $\mathbb{R}^n$ 版本、再限制回 $\Omega$。Lipschitz 边界正则性是扩展算子存在的标准条件；更粗糙的边界（带有内尖点等）可能让扩展失效，对应的嵌入定理也会出问题。

![$H^s$ 与 $H^{-s}$ 的对偶](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_6_dual_sobolev.png)

### 迹定理

Dirichlet 边界条件 $u|_{\partial\Omega} = 0$ 形式上要求把 $u$ 限制到一个 Lebesgue 测度为零的集合上。对一般 $L^2$ 函数这一限制无意义。但对 $H^1(\Omega)$ 函数，迹定理给出严格意义：存在唯一的有界线性算子 $\gamma_0: H^1(\Omega) \to H^{1/2}(\partial\Omega)$，对 $C^1(\bar\Omega)$ 函数与点态限制一致。这个算子叫**迹算子**，它损失半个导数：$H^1$ 函数的迹只在 $H^{1/2}$ 中，不会更光滑。

子空间 $H^1_0(\Omega) = \ker(\gamma_0)$ 正是“边界上消失的 $H^1$ 函数”，等价地，$C_c^\infty(\Omega)$ 在 $H^1$ 范数下的闭包。下一篇 Lax-Milgram 用到的就是这个空间——Dirichlet 问题的弱形式是在 $H^1_0(\Omega)$ 上找极小值点，迹定理保证“边界为零”这件事即便对粗糙的弱解也有意义。

## 为什么这很重要

回到本篇开头的尴尬：$\delta$、冲击波解、$|x|$ 在原点的二阶导数——这些经典分析里的“病态对象”全都被分布理论收编。每个分布都可以被光滑函数 $C^\infty$ 序列在 $\mathcal{D}'$ 中逼近（磨光），每个分布都有任意阶的导数（通过对偶定义），每个微分算子的基本解都是分布。整个 PDE 理论从“假设解是 $C^k$ 的”这一过强假设里解放出来。

Sobolev 空间是这套理论里能做定量估计的层次。$W^{k,p}$ 的范数控制了 $k$ 阶导数的 $L^p$ 大小；嵌入定理把 $k$ 阶 $L^p$ 正则性翻译成更弱意义下的逐点正则性（Morrey）或更高 $L^q$ 可积性（Sobolev 不等式）；Rellich-Kondrachov 的紧嵌入把 $H^1$ 弱收敛升级到 $L^2$ 强收敛——这是非线性 PDE 中“弱+紧→强”模板的关键步骤。Poincare 不等式、迹定理、扩展定理把抽象空间和具体边值问题接起来。这一整套语言下一篇 Lax-Milgram 会反复用到。

## 反例：Sobolev 嵌入失效的临界情形

Sobolev 嵌入 $W^{k, p}(\Omega) \hookrightarrow L^q(\Omega)$ 在 $\Omega \subset \mathbb{R}^n$ 有界的条件是 $1/q \geq 1/p - k/n$。等号情形（临界 Sobolev 指数 $q^* = np/(n - kp)$）边缘成立，但**不是紧嵌入**。这条区别决定了非线性 PDE 的存在性论证能不能跑。

具体反例：$n = 3, k = 1, p = 2$。临界指数 $q^* = 6$。Sobolev 嵌入 $H^1(B_1) \hookrightarrow L^6(B_1)$ 成立但不紧。具体的非紧序列：$u_\epsilon(x) = \epsilon^{-1/2} \varphi(x/\epsilon)$，其中 $\varphi$ 是固定的 $C_c^\infty(B_1)$ 函数。计算：
- $\|u_\epsilon\|_{L^6}^6 = \epsilon^{-3} \int |\varphi(x/\epsilon)|^6 dx = \epsilon^{-3} \cdot \epsilon^3 \int |\varphi|^6 = \int |\varphi|^6$（不变）。
- $\|\nabla u_\epsilon\|_{L^2}^2 = \epsilon^{-3} \cdot \epsilon^{-2} \int |\nabla \varphi(x/\epsilon)|^2 dx = \epsilon^{-3} \cdot \epsilon \int |\nabla \varphi|^2 = \epsilon^{-2} \int |\nabla \varphi|^2$？计算错——重做：$\nabla u_\epsilon(x) = \epsilon^{-3/2} \nabla\varphi(x/\epsilon)$，$|\nabla u_\epsilon|^2 = \epsilon^{-3} |\nabla\varphi|^2(x/\epsilon)$，积分给 $\epsilon^{-3} \cdot \epsilon^3 \int |\nabla \varphi|^2 = \int |\nabla \varphi|^2$。

正确的浓缩反例：$u_\epsilon(x) = \epsilon^{(n-2)/2} (\epsilon^2 + |x|^2)^{-(n-2)/2}$（Talenti 函数）。$\|u_\epsilon\|_{H^1}$ 守恒、$\|u_\epsilon\|_{L^{q^*}}$ 守恒，但 $u_\epsilon$ 在 $\epsilon \to 0$ 浓缩到一个点——质量集中，没有强收敛子列。这条反例正是 Brezis-Lieb 引理和集中紧致原理（Lions）要解决的现象。

教训：临界 Sobolev 指数下嵌入是连续但不紧的。任何依赖紧嵌入的论证（变分法直接方法、PDE 弱解的存在）在临界情形需要额外工具——集中紧致、profile decomposition、Talenti 极小化。这是为什么 Yamabe 问题（$n \geq 3$ 流形上找常数标量曲率度量）的存在性证明在 Aubin 和 Trudinger 之后还需要 Schoen 的额外工作。

## 反例：分布乘积一般无意义

光滑函数可以相乘，但**两个分布一般不能相乘**。这条限制让分布理论在非线性 PDE 上有明显短板。

具体反例：考虑 $\mathbb{R}$ 上的 $\delta$ 函数。$\delta \cdot \delta$ 没有意义。证明：若 $\delta^2$ 是分布，对任何 $\varphi \in C_c^\infty$ 应该有 $\langle \delta^2, \varphi\rangle = $ 某个数。用磨光近似 $\delta_\epsilon \to \delta$，$\delta_\epsilon^2 \to ?$。计算 $\int \delta_\epsilon^2 \varphi = \epsilon^{-1} \int \rho(x/\epsilon)^2 \varphi(x)/\epsilon \,dx \approx \varphi(0) \cdot \epsilon^{-1} \int \rho^2 \to \infty$ 当 $\epsilon \to 0$（这里 $\rho$ 是磨光核）。极限发散——$\delta^2$ 不存在。

第二个反例：$H(x) \cdot \delta(x)$，其中 $H$ 是 Heaviside。$H \delta$ 在不同正则化下给出不同答案：$H_\epsilon \delta_\epsilon \to (1/2) \delta$（如果 $H_\epsilon$ 和 $\delta_\epsilon$ 都关于 $0$ 对称）但 $H_\epsilon (\delta_\epsilon \star \rho) \to \delta$（如果 $\delta_\epsilon$ 偏向 $0^+$）。乘积本身不良定义。

教训：分布的线性运算（导数、平移、Fourier 变换、卷积与紧支撑分布）都良定义，但乘积一般不行。非线性 PDE 用 Sobolev 空间替代分布部分原因正在此——$W^{k, p}$ 中的元素是函数（不是一般分布），可以逐点相乘，乘积估计由 Sobolev 嵌入控制。这种"放弃部分分布的一般性，换取乘积的合法性"是 Sobolev 框架的核心权衡。

## 常见陷阱：弱导数 ≠ 经典导数 ≠ 几乎处处导数

三个不同的"导数"概念，在足够正则的函数上一致，在一般函数上分裂。

具体例子：$f(x) = |x|$ 在 $[-1, 1]$ 上。
- **经典导数：** 在 $x = 0$ 不存在。
- **几乎处处导数：** $f'(x) = \text{sgn}(x)$ 在 $x \neq 0$（零测集排除原点）。
- **弱导数（分布意义）：** $f' = \text{sgn}$，与几乎处处导数一致——因为 $|x|$ 是 Lipschitz，绝对连续，弱导数等于其经典导数（哪里存在）。

但换 $f$ 是 Cantor 函数（在 $[0, 1]$ 上单调升从 $0$ 到 $1$，但几乎处处导数为 $0$）：
- **经典导数：** 在 Cantor 集上不存在，其余处为 $0$。
- **几乎处处导数：** $0$ 几乎处处。
- **弱导数：** Cantor 测度（不是函数，是奇异测度），$f'$ 作为分布给 $\langle f', \varphi\rangle = -\int f \varphi'\,dx$，由于 $f$ 不绝对连续，这个分布不能用 $L^1$ 函数表示。

陷阱：常以为"$f' = 0$ 几乎处处 ⇒ $f$ 常数"。这条对 $f \in W^{1, 1}$（弱导数为 $L^1$ 函数）成立，对一般连续函数不成立——Cantor 函数就是反例。Sobolev 空间 $W^{k, p}$ 的元素自动绝对连续（通过其在 $L^p$ 中的弱导数），所以"导数为零意味着常数"在 Sobolev 空间里恢复。但出 Sobolev 空间，这条本科直觉就失效。

## 下一步

到此为止，泛函分析的工具箱已经基本搭好：度量、范数、内积、对偶、弱拓扑、有界算子、谱、半群、分布、Sobolev 空间。接下来的最后一篇是把这些工具接到具体应用上。

下一篇会展示三类典型应用。**椭圆 PDE** 的 Lax-Milgram 模板：把 $-\Delta u = f$ 写成 $H^1_0$ 上的变分恒等式，用第三篇的 Riesz 表示和强制双线性形式直接得到唯一弱解，再用第七篇的紧嵌入引出 Galerkin 方法和有限元的收敛理论。**变分极小化**的直接方法：能量泛函的极小值点通过第五篇的弱紧性 + 弱下半连续性自动构造，下游覆盖 Hilbert 第十九/二十问题、Yamabe 问题、最小曲面、机器学习中的核回归。**量子力学**的 Stone 定理：自共轭算子 $H$ 通过第八/九篇的谱测度生成强连续幺正群 $e^{-itH/\hbar}$，Schrödinger 方程的解、能量谱、对称性与守恒律全部从这一条对应关系自动得出。十二篇文章就在这里收尾。

---
