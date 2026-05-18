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

![|x| 的弱导数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_weak_derivative.png)

分布导数总是存在的；它们是$\mathcal{D}'$中的抽象对象。对于PDE，我们希望具体的导数生活在$L^p$中。

如果$u \in L^1_{\text{loc}}(\Omega)$在方向$\partial^\alpha$上有**弱导数**$g \in L^1_{\text{loc}}(\Omega)$，则

$$
\int_\Omega u\,\partial^\alpha\varphi\,dx = (-1)^{|\alpha|}\int_\Omega g\,\varphi\,dx \quad \text{对于所有 } \varphi \in C_c^\infty(\Omega).
$$

弱导数，当它存在时，在几乎处处是唯一的，并且在两者都有意义时与经典导数一致。许多在经典意义上不可微的函数具有弱导数：$|x|$的弱导数是$\mathrm{sgn}(x)$；$\max(u, 0)$的弱导数是$u'\cdot\mathbf{1}_{u>0}$；绝对连续函数在线上具有与其经典几乎处处导数相等的弱导数。

![通过与测试函数积分定义的弱导数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_3_weak_deriv.png)

### Sobolev空间$W^{k,p}$

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

可以证明，设$\Omega \subset \mathbb{R}^n$是有界的且连通的。存在$C = C(\Omega, p)$使得对于所有$u \in W_0^{1,p}(\Omega)$，

$$
\|u\|_{L^p(\Omega)} \le C\|\nabla u\|_{L^p(\Omega)}.
$$

对于在边界上消失的函数，$L^p$范数由梯度单独控制。这是在$W_0^{1,p}(\Omega)$上$\|\nabla u\|_{L^p}$是等价范数的关键步骤，并且它是下一章中Lax-Milgram应用于椭圆PDE的动力。Poincare常数按$\Omega$的直径缩放：对于半径为$R$的球，$C \sim R$。

一个变体，**Poincare-Wirtinger不等式**，适用于没有边界条件的函数：$\|u - \bar{u}\|_{L^p} \le C\|\nabla u\|_{L^p}$，其中$\bar{u}$是$\Omega$上$u$的平均值。这对于Neumann问题相关，其中解仅确定到一个加性常数。

---

## Sobolev嵌入定理

嵌入定理回答了一个基本问题：如果一个函数在$L^p$中有$k$个导数，我们能对其逐点正则性说什么？

![迹定理：限制到边界](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/fig11_trace_theorem.png)

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

可以证明，设$\Omega \subset \mathbb{R}^n$是有界的且有Lipschitz边界。存在有界线性$E: W^{

![迹定理：把 Sobolev 函数限制到边界](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_5_trace.png)

![$H^s$ 与 $H^{-s}$ 的对偶](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_6_dual_sobolev.png)

![分布及其导数的例子](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/11-distributions-sobolev/fa_v2_11_7_examples.png)

---
