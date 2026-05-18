---
title: "泛函分析 (4)：对偶空间与 Hahn-Banach 定理 —— 线性泛函的驯服"
date: 2021-10-07 09:00:00
tags:
  - functional-analysis
  - hahn-banach
  - dual-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: zh
mathjax: true
description: "Hahn-Banach 定理保证了足够多的连续线性泛函存在，以区分点——这是泛函分析中对偶理论的基础。"
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "functional-analysis-4"
---

# 对偶空间与 Hahn-Banach 定理 —— 线性泛函的驯服

## 为什么你不能跳过这篇文章

到目前为止，理论主要讨论的是空间及其元素。本文改变了视角：它询问通过一组测试泛函来“测量”向量 $x$ 时，你能对 $x$ 说些什么。从“向量”到“向量加上泛函”的转变使 Banach 空间成为有限维线性代数的有效类比。在有限维中，每个线性泛函都是连续的，对偶空间与原空间维度相同——因此无需证明。在无限维中，连续性是一个真正的约束，而存在足够的连续泛函来区分点或扩展部分数据并不明显。Hahn-Banach 定理正是保证这一点的结果，它使得泛函分析成为可能。

![对偶空间：泛函作为超平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_dual_space.png)

一个实际工作的分析师使用 Hahn-Banach 的方式就像一个实际工作的代数学家使用 Zorn 引理一样：无形地，每天几十次，从未从头开始证明。本文的目的是清晰地产生这个定理，并检查其一些标准结果——几何形式、支撑超平面的存在性、典范嵌入到二次对偶。第 5 篇文章将利用弱拓扑来让对偶发挥作用。

## 对偶空间

设 $X$ 是 $\mathbb{R}$ 或 $\mathbb{C}$ 上的赋范空间。**对偶空间** $X^*$ 是有界（等价于连续）线性泛函 $\varphi: X \to \mathbb{C}$ 的空间，配备**对偶范数**
$$\|\varphi\|_{X^*} = \sup_{\|x\| \leq 1} |\varphi(x)|.$$
在这个范数下，$X^*$ 是 Banach 空间——即使 $X$ 本身不是（泛函的 Cauchy 序列逐点 Cauchy，极限定义了一个线性泛函，且有界性通过一致 Cauchy 性传递给极限）。

![动画：寻找分离超平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_separation.gif)

![Hahn-Banach 分离定理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_hahn_banach_separation.png)

因此，对偶空间自动是 Banach 空间，无论原始空间是否完备。这是对偶构造如此受欢迎的一个结构上的优点：形成对偶会将不完备的赋范空间升级为完备的空间。

### 经典对偶

你应该知道的一些对偶识别：

- $(\ell^p)^* = \ell^q$ 对于 $1 < p < \infty$，$1/p + 1/q = 1$，通过 $y \mapsto \varphi_y(x) = \sum y_n x_n$。
- $(\ell^1)^* = \ell^\infty$，同样的公式。
- $(\ell^\infty)^* \supsetneq \ell^1$ —— $\ell^\infty$ 的对偶包含 $\mathbb{N}$ 上的有限可加测度，严格大于 $\ell^1$（Banach 极限是经典的非-$\ell^1$ 例子）。
- $(c_0)^* = \ell^1$。
- $(L^p[\Omega])^* = L^q[\Omega]$ 对于 $1 \leq p < \infty$，$1/p + 1/q = 1$（约定 $1/\infty = 0$，所以 $(L^1)^* = L^\infty$）。
- $(C[K])^* = M[K]$，紧度量空间 $K$ 上的有限符号 Borel 测度空间（Riesz-Markov 定理）。
- $(\mathcal{H})^* = \mathcal{H}$ 对于 Hilbert 空间，由 Riesz（第 3 篇文章）。

![对偶空间的几何解释为超平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_1_dual_geom.png)

模式 $(\ell^p)^* = \ell^q$ 如此干净，几乎看起来像是巧合，但它是由 Hölder 不等式强制的。$\ell^p$ 和 $\ell^q$ 之间的配对 $\langle x, y \rangle = \sum x_n y_n$ 有界，$|\langle x, y \rangle| \leq \|x\|_p \|y\|_q$，且 Hölder 不等式在适当选择的向量上是尖锐的。这个论证逐字推广到任何测度空间上的 $L^p$。

### 数值示例

在 $\ell^2$ 中，取 $y = (1, 1/2, 1/3, \ldots, 1/n, 0, 0, \ldots)$ 对于 $n = 4$，所以 $y = (1, 1/2, 1/3, 1/4, 0, \ldots)$。对偶泛函 $\varphi_y(x) = \sum y_k x_k$ 的范数 $\|\varphi_y\|_{(\ell^2)^*} = \|y\|_2 = \sqrt{1 + 1/4 + 1/9 + 1/16} = \sqrt{205/144} \approx 1.193$。根据 Cauchy-Schwarz 不等式，这个范数在 $x = y / \|y\|_2$ 处达到，这给出 $\varphi_y(x) = \|y\|_2$。对偶是紧密的——Cauchy-Schwarz 是饱和情况。

## Hahn-Banach 定理（解析形式）

可以证明，设 $X$ 是实向量空间，$p: X \to \mathbb{R}$ 是次线性泛函（$p(x + y) \leq p(x) + p(y)$ 且 $p(\alpha x) = \alpha p(x)$ 对于 $\alpha \geq 0$），$\varphi_0: M \to \mathbb{R}$ 是子空间 $M \subseteq X$ 上的线性泛函，且 $\varphi_0(x) \leq p(x)$ 对所有 $x \in M$ 成立。则 $\varphi_0$ 可以延拓为 $X$ 上的线性泛函 $\varphi: X \to \mathbb{R}$，满足 $\varphi(x) \leq p(x)$ 对所有 $x \in X$ 成立。

![Hahn-Banach 延拓](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_hahn_banach_extension.png)

复版本：将次线性替换为半范数（即 $p(\alpha x) = |\alpha| p(x)$），要求 $|\varphi_0(x)| \leq p(x)$ 在 $M$ 上成立，且延拓满足 $|\varphi(x)| \leq p(x)$ 在 $X$ 上成立。

最常引用的版本：赋范空间的子空间上的任何有界线性泛函可以延拓到整个空间而不增加其范数。这只是半范数情况下的 $p(x) = \|\varphi_0\|_M \cdot \|x\|$。

![Hahn-Banach 延拓有界泛函从子空间到整个空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_2_hb_extension.png)

### 证明概要

步骤 1（单步延拓）。给定 $M$ 上的 $\varphi_0$ 和 $x_0 \notin M$，设 $M' = M + \mathbb{R} x_0$。任何延拓都由值 $c = \varphi(x_0)$ 确定。约束 $\varphi(m + tx_0) \leq p(m + tx_0)$ 对所有 $m \in M$ 和 $t \in \mathbb{R}$ 成立（经过一些 $t > 0, t < 0$ 情况的操作）变成两个不等式 $A \leq c \leq B$，其中 $A = \sup_{m \in M} (\varphi_0(m) - p(m - x_0))$ 和 $B = \inf_{m \in M} (p(m + x_0) - \varphi_0(m))$。次线性性保证 $A \leq B$（一个计算），因此存在有效的 $c \in [A, B]$。

步骤 2（Zorn 引理）。按域和图的包含关系对部分延拓进行排序，取一个最大的。最大延拓必须定义在整个 $X$ 上，否则步骤 1 会产生一个更大的延拓，与最大性矛盾。$\square$

一般情况下，Zorn 引理的使用是不可避免的；该定理在没有选择公理的 ZF 中失败。然而，对于*可分*赋范空间，Hahn-Banach 可以在没有选择的情况下证明——选取一个可数稠密子集并逐方向延拓，仅使用步骤 1 的可数步版本。这种细微差别在实践中很少重要。

### 为什么这很重要

Hahn-Banach 让我做三件本来不可能的事情。（i）**延拓**线性泛函从子空间到整个空间——任何时候我在子集上有数据并希望得到一个全局对象时都需要这样做。（ii）**分离**点：存在一个连续泛函 $\varphi$ 使得 $\varphi(x_0) = \|x_0\|$ 且 $\|\varphi\| = 1$，通过延拓 $\mathbb{R} x_0$ 上的泛函 $\alpha x_0 \mapsto \alpha \|x_0\|$。因此 $X^*$ 有足够的泛函来检测 $X$ 中的每个非零元素。（iii）**计算范数** 作为 $\|x\| = \sup_{\|\varphi\| \leq 1} |\varphi(x)|$，即对偶范数的对偶范数，恢复原始范数。

## 几何 Hahn-Banach：凸集的分离

“几何”或“分离”形式的 Hahn-Banach 在优化和概率中更有用。

![自反空间：X = X**](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_reflexive.png)

可以证明，设 $X$ 是实赋范空间，$A, B \subseteq X$ 是不相交的非空凸集。（i）如果 $A$ 是开集，则存在 $\varphi \in X^*$ 和 $\alpha \in \mathbb{R}$ 使得 $\varphi(a) < \alpha \leq \varphi(b)$ 对所有 $a \in A$，$b \in B$ 成立。（ii）如果 $A$ 是闭集且 $B$ 是紧集，则存在 $\varphi \in X^*$ 和 $\alpha < \beta$ 使得 $\varphi(a) \leq \alpha < \beta \leq \varphi(b)$ 对所有 $a \in A$，$b \in B$ 成立——*严格*分离。

换句话说：任何两个不相交的凸集都可以被一个超平面分离，如果其中一个集合是闭集且另一个是紧集，则可以*严格*分离（具体来说，由一个板条分离）。

![Hahn-Banach 几何形式：用超平面分离两个凸集](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_3_separation.png)

### 证明概要

使用包含 $0$ 的开凸集 $A$ 的 Minkowski 泛函 $p_A(x) = \inf\{t > 0 : x \in tA\}$（如果不是，平移）。$p_A$ 是次线性的，且 $p_A(x) \leq 1$ 当且仅当 $x \in A$（开球测试）。取任意 $b \in B$ 并考虑直线 $\mathbb{R}(b - a_0)$ 对于 $a_0 \in A$；在线上定义一个线性泛函，使其在 $b - a_0$ 处为 $1$ 并由 $p_A$ 限制。Hahn-Banach 将其延拓到整个 $X$。延拓后的泛函将 $A$ 与 $\{b\}$ 分离，稍微细化（使用 $A$ 与闭包之间的间隙加上紧性如果需要）给出严格分离。$\square$

### 为什么这很重要

几何形式是优化中每个对偶论证的基础。凸规划依赖于不可行系统 $A x = b, x \geq 0$ 对应于一个分离超平面的事实，而超平面给出了不可行性的“Farkas 类型”证书。博弈论中的极小极大定理是关于分离凸集的定理（游戏的鞍点是两个凸包的交点）。Banach 空间中紧凸集的 Choquet 积分表示——Banach 空间中紧凸集的每个点都是极端点的概率测度的积分——是对分离的深刻应用。

## 支撑超平面

几何 Hahn-Banach 的一个特殊情况：Banach 空间 $X$ 中的闭凸集 $C$ 在每个边界点都有一个支撑超平面。也就是说，对于每个 $x_0 \in \partial C$，存在 $\varphi \in X^*$ 使得 $\varphi(x_0) = \sup_{c \in C} \varphi(c)$。

![对偶空间辨认](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_dual_lp.png)

![凸集在边界点处的支撑超平面](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_7_supporting.png)

在凸分析的语言中，$\varphi$ 属于 $C$ 在 $x_0$ 处的指示函数的次微分是记录该点所有“支撑方向”的对偶对象。

### 数值示例

取 $C = \{(x_1, x_2) \in \mathbb{R}^2 : x_1^2 + x_2^2 \leq 1\}$，单位圆盘。在边界点 $(\cos\theta, \sin\theta)$ 处，唯一的支撑超平面是切线，法向量为 $\varphi(x) = \cos\theta \cdot x_1 + \sin\theta \cdot x_2$。因此每个边界点的支撑超平面是唯一的——圆盘是*光滑*的。

现在取 $C = \{ x : |x_1| + |x_2| \leq 1 \}$，单位 $\ell^1$ 球。在顶点如 $(1, 0)$ 处，*无穷多个*支撑超平面存在：任何 $\varphi(x) = a x_1 + b x_2$ 且 $a = 1$ 和 $|b| \leq 1$ 都在 $(1,0)$ 处支撑 $C$，因为 $\sup_{c \in C}(c_1 + b c_2) = 1$ 对于 $|b| \leq 1$（在 $(1,0)$ 处达到）。因此顶点有非唯一的支撑超平面——角有一个“扇形”支撑。

这种非唯一性正是 $\ell^1$ 最小化可以有非唯一解的几何原因；LASSO 回归和压缩感知文献花费大量精力诊断何时解*是*唯一的。

## 二次对偶与自反性

对偶空间 $X^*$ 本身是 Banach 空间，因此它有自己的对偶 $X^{**} = (X^*)^*$，称为二次对偶有一个典范嵌入 $J: X \to X^{**}$ 定义为 $(Jx)(\varphi) = \varphi(x)$ 对于 $\varphi \in X^*$。这个嵌入是良定义的（线性映射 $\varphi \mapsto \varphi(x)$ 有界且范数 $\leq \|x\|$），线性的，且等距——后者使用 Hahn-Banach 找到一个 $\varphi$ 使得 $|\varphi(x)| = \|x\|$ 且 $\|\varphi\| = 1$。

![子空间的零化子](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/figures/04_annihilator.png)

Banach 空间是**自反的**，如果 $J$ 是满射，即 $X = X^{**}$ 典范地。自反性是一个强性质；它在取闭子空间、商空间和有限积时保持，并且它蕴含许多紧性和正则性结果。

![典范嵌入 V 到 V** 和自反性的含义](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_4_reflexive.png)

### 示例

- 所有有限维空间都是自反的（显然）。
- Hilbert 空间是自反的：$\mathcal{H}^* = \mathcal{H}$ 由 Riesz 得出，所以 $\mathcal{H}^{**} = \mathcal{H}^* = \mathcal{H}$。
- $\ell^p$ 和 $L^p$ 对于 $1 < p < \infty$ 是自反的：$((L^p)^*)^* = (L^q)^* = L^p$。
- $\ell^1, L^1, c_0, C[K]$ 不是自反的。$c_0$ 的二次对偶是 $\ell^\infty$，且 $\ell^\infty$ 严格大于 $c_0$。

### 为什么自反性重要

自反性等价于闭单位球的弱紧性——Eberlein 和 Šmulian 的一个重要定理（第 5 篇文章中会有提示）。因此，在自反空间中，每个有界序列都有一个弱收敛子序列——仅次于范数紧性的最强紧性。这就是为什么在 $L^p$ 中 $1 < p < \infty$ 的最小化论证有效：取一个最小化序列，提取一个弱收敛子序列（通过自反性），使用范数的下半连续性传递到极限。在 $L^1$ 或 $L^\infty$ 中同样的方法失败是因为非自反性，需要更精细的论证（测度紧性，$C_0$ 对偶中的弱-* 极限）。

## 有界算子的伴随（对偶）

给定 Banach 空间之间的一个有界线性算子 $T: X \to Y$，**伴随**（或对偶）算子 $T^*: Y^* \to X^*$ 定义为 $T^*\varphi = \varphi \circ T$，即 $(T^*\varphi)(x) = \varphi(T x)$。伴随是有界的，且 $\|T^*\| = \|T\|$——上界是直接的，匹配的下界使用 Hahn-Banach 找到几乎达到 $T x$ 范数的泛函。

一般 Banach 空间中的伴随具有线性代数期望的形式性质：$(S+T)^* = S^* + T^*$，$(\lambda T)^* = \lambda T^*$，$(ST)^* = T^* S^*$，$(T^*)^* = T^{**}$（当两个空间都是自反的时，识别 $X^{**}$ 与 $X$）。核与值域的关系：
$$\ker(T^*) = \mathrm{Range}(T)^\perp,\quad \overline{\mathrm{Range}(T)} = \ker(T^*)^\perp,$$
其中 $\perp$ 在适当的对偶或预对偶中取消没子。闭值域定理给出更细的关系：$T$ 的值域在 $Y$ 中闭当且仅当 $T^*$ 的值域在 $X^*$ 中闭，且在这种情况下两者等于对方核的消没子。

### 数值示例

取 $T: \ell^1 \to \ell^\infty$，$T x = (x_1, x_1 + x_2, x_1 + x_2 + x_3, \ldots)$——序列的部分和，视为有界算子。泛函 $\varphi_n \in (\ell^\infty)^*$ 由 $\varphi_n(y) = y_n$ 给出，范数为 $1$。则 $T^* \varphi_n \in (\ell^1)^* = \ell^\infty$ 是泛函 $x \mapsto x_1 + \cdots + x_n$，由有界序列 $(1, 1, \ldots, 1, 0, 0, \ldots)$ 表示，前 $n$ 个为 $1$。当 $n \to \infty$ 时，$\|T^*\varphi_n\|_{\ell^\infty} = 1$ 保持有界，说明 $\|T^*\| \leq \|T\|$——事实上 $\|T\| = \|T^*\| = 1$（因为 $\ell^1$-范数最多为 $1$ 的序列的部分和的 $\ell^\infty$-范数最多为 $1$）。

## 消没子与预消没子

给定子集 $A \subseteq X$，**消没子**是
$$A^\perp = \{ \varphi \in X^* : \varphi(a) = 0 \text{ 对所有 } a \in A \}.$$
这是 $X^*$ 的闭子空间。对偶地，给定 $B \subseteq X^*$，
$$^\perp B = \{ x \in X : \varphi(x) = 0 \text{ 对所有 } \varphi \in B \}$$
是 $X$ 的闭子空间。这两个操作满足 $^\perp(A^\perp) = \overline{\mathrm{span}(A)}$ 对于 $A \subseteq X$，由 Hahn-Banach 得出：$X$ 中不在 $\overline{\mathrm{span}(A)}$ 中的任何元素可以通过消失在 $A$ 上的连续泛函与 $\mathrm{span}(A)$ 分离。

子空间与其消没子之间的这种对偶性是每种“Fredholm 替代”类型定理的基础。经典陈述：$T x = y$ 有解当且仅当 $\varphi(y) = 0$ 对于每个 $\varphi$ 使得 $T^* \varphi = 0$。对于闭值域算子（例如 Fredholm 算子，第 7 篇文章），特征是精确且可计算的。

### 为什么这很重要

消没子对偶将“在哪里可以解 $T x = y$？”简化为“$\ker(T^*)$ 的元素是什么？”——这是一个关于不同空间上不同算子的问题。在 PDE 中，这是日常的：非齐次方程 $L u = f$ 有解当且仅当 $f$ 与 $\ker(L^*)$ 正交，其中 $L^*$ 是微分算子的形式伴随（分部积分中的边界项给出正确的伴随概念）。这有时被称为“可解条件”或“相容条件”。

## $L^p$ 对偶详细说明

为了完整性，$1 \leq p < \infty$，$1/p + 1/q = 1$ 时的对偶 $(L^p)^* = L^q$ 值得仔细陈述。

可以证明，对于每个 $g \in L^q$，泛函 $\varphi_g(f) = \int f g$ 属于 $(L^p)^*$，且 $\|\varphi_g\|_{(L^p)^*} = \|g\|_{L^q}$。映射 $g \mapsto \varphi_g$ 是 $L^q \to (L^p)^*$ 的等距同构。

Hölder 不等式给出 $\|\varphi_g\| \leq \|g\|_{L^q}$。反向不等式使用显式的 $f$ 选择：取 $f = |g|^{q-1} \mathrm{sgn}(g) / \|g\|_{L^q}^{q/p}$，标准化使得 $\|f\|_{L^p} = 1$。则 $\varphi_g(f) = \|g\|_{L^q}$，展示等式。

![$(l^p)* = l^q$ 对于共轭指数的对偶](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_6_lp_dual.png)

对于满射性，给定 $\varphi \in (L^p)^*$，需要构造 $g \in L^q$ 来表示它。Radon-Nikodym 定理提供 $g$ 作为从 $\varphi$ 构建的绝对连续测度的密度。$p = 1$ 的情况需要测度的 $\sigma$-有限性；否则对偶可能失败。

$p = \infty$ 的情况打破了模式。$(L^\infty)^*$ *严格大于* $L^1$——它包含不是测度的有限可加集函数。这是 $L^\infty$ 的一个结构性麻烦，实际后果是 $L^\infty$ 不是自反的，许多紧性论证在那里失败。

### 数值示例

在 $L^p[0,1]$ 中，取 $f(t) = t$ 并考虑泛函 $\varphi(g) = \int_0^1 g(t) f(t)\,dt = \int_0^1 t \, g(t)\,dt$ 作用于 $g \in L^p$（因此 $f$ 在对偶中扮演 $g$ 的角色，但符号交换——$\varphi \in (L^p)^*$ 由 $f \in L^q$ 确定）。对于 $p = 2$，$q = 2$，$\|\varphi\|_{(L^2)^*} = \|f\|_{L^2} = \big(\int_0^1 t^2\,dt\big)^{1/2} = 1/\sqrt{3} \approx 0.577$。合理性检查：由 Cauchy-Schwarz，$|\varphi(g)| \leq \|g\|_{L^2} \cdot 1/\sqrt{3}$ 对于 $\|g\|_{L^2} = 1$。在 $g(t) = t \sqrt{3}$ 处等号成立，其 $L^2$ 范数为 $1$ 且 $\varphi(g) = \sqrt{3} \int_0^1 t^2\,dt = 1/\sqrt{3}$。对偶是紧密的。

## Hahn-Banach 的微妙应用：Banach 极限

Hahn-Banach 的一个经典且反直觉的应用：存在一个有界线性泛函 $L: \ell^\infty(\mathbb{N}) \to \mathbb{R}$——一个**Banach 极限**——在收敛序列的子空间上扩展 $\lim$，且 $\|L\| = 1$，且平移不变：$L((x_2, x_3, \ldots)) = L((x_1, x_2, \ldots))$。

构造：在 $\ell^\infty$ 上定义一个次线性泛函 $p(x) = \limsup_{n} \frac{1}{n} \sum_{k=1}^n x_k$（上 Cesàro 平均）。在收敛序列的子空间上，$p$ 与 $\lim$ 一致。Hahn-Banach 将 $\lim$ 延拓为泛函 $L: \ell^\infty \to \mathbb{R}$ 使得 $L(x) \leq p(x)$。稍作工作显示 $L$ 是平移不变的，且 $\|L\| = 1$。

Banach 极限*不是*唯一的（不同的 Hahn-Banach 延拓给出不同的 Banach 极限）且*无法*明确定义（没有 $L$ 的公式；构造需要通过 Zorn 的选择公理）。在有界序列 $(0, 1, 0, 1, \ldots)$ 上——在经典意义上不收敛——每个 Banach 极限给出 $L = 1/2$，通过平均和平移不变性。因此，Banach 极限为所有有界序列提供了收敛值，代价是值取决于选择了哪个延拓。

这是一个奇怪但有用的对象。它填充了 $(\ell^\infty)^* \setminus \ell^1$——$\ell^\infty$ 对偶中不来自 $\ell^1$ 向量的部分。它也是泛函分析中没有构造性类比的存在性证明的一个干净示例。

## 弱拓扑 vs 强拓扑（第 5 篇文章的预览）

对偶空间引入了原始空间的新拓扑：**弱拓扑**，使所有对偶泛函连续的最粗拓扑。网 $x_\alpha \to x$ 弱收敛当且仅当 $\varphi(x_\alpha) \to \varphi(x)$ 对每个 $\varphi \in X^*$ 成立。范数收敛蕴含弱收敛；在无限维中，逆命题是假的。

![弱拓扑 vs 强拓扑比较](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_v2_04_5_weak_strong.png)

弱收敛是你应该认为是“矩收敛”或“平均收敛”。一个典型例子：在 $L^2[0, 2\pi]$ 中，序列 $f_n(t) = \sin(n t)$ 在范数上*不*收敛（其范数为 $\sqrt{\pi}$ 对每个 $n$ 成立），但在弱拓扑下收敛到 $0$（由 Riemann-Lebesgue 引理：$\int g(t) \sin(nt)\,dt \to 0$ 对每个 $g \in L^2$ 成立）。高频振荡在积分下抵消，但在范数下不抵消。弱收敛看到抵消；范数收敛看不到。

使弱拓扑有用的是**Banach-Alaoglu** 定理：$X^*$ 的闭单位球在弱-* 拓扑下是紧的（对偶类似物）。当 $X$ 是自反的时，$X$ 的闭单位球在弱拓扑下是紧的，由 Eberlein-Šmulian 得出。这些是变分分析的工作马紧性结果。第 5 篇文章将证明它们。

## 一个具体应用：$C[K]$ 中的最佳逼近

设 $K$ 是紧度量空间，考虑以下问题：给定 $f \in C[K]$ 和闭子空间 $M \subset C[K]$，找到 $M$ 中最接近 $f$ 的元素（在 sup 范数下）。

直接的最小化序列方法在 $C[K]$ 中很微妙，因为单位球不是弱紧的（空间不是自反的）。但 Hahn-Banach 提供了一个优雅的替代方案——最佳逼近的*对偶*。

可以证明，$d(f, M) = \sup\{ |\varphi(f)| : \varphi \in M^\perp, \|\varphi\| \leq 1 \}$，其中 $M^\perp \subseteq (C[K])^*$ 是消没子。

右边是在对偶空间 $(M^\perp)$ 的闭单位球上的最大化，视作 $(C[K])^*$ 的子集。根据 Banach-Alaoglu（第 5 篇文章），$(C[K])^*$ 的闭单位球在弱-* 拓扑下是紧的，其闭子集 $M^\perp \cap \overline{B}(0, 1)$ 也是。紧集上的连续函数达到其上确界，因此上确界由某个泛函 $\varphi^* \in M^\perp$ 达到。对偶已将“找到最佳逼近”转换为“找到最优认证泛函”，这通常更容易。

这个技巧是 Chebyshev 逼近理论的基础。泛函 $\varphi^*$ 根据 Riesz-Markov 是 $K$ 上的有限符号测度——Markov 的一个定理说它最多支持 $\dim M + 1$ 个点（Chebyshev 交替定理的伪装）。对于 $[a, b]$ 上的多项式逼近，这给出了经典的 Chebyshev 等振荡：最佳多项式逼近在 $\geq n+2$ 个点上围绕 $f$ 振荡。

### 数值示例

在 $[-1, 1]$ 上用次数 $\leq 2$ 的多项式逼近 $f(t) = t^4$（在 sup 范数下）。最佳逼近是 $p^*(t) = t^2 - 1/8$（这可以从 Chebyshev 多项式理论推导出来：$[-1, 1]$ 上 $t^4$ 的最佳均匀逼近是 Chebyshev 基中的截断）。误差 $f - p^* = t^4 - t^2 + 1/8$ 在 $[-1, 1]$ 的 $5$ 个点上等振荡：$\pm 1, \pm 1/\sqrt{2}, 0$，交替符号和幅度 $1/8$。因此 $d(f, M) = 1/8$，由显式极小极大配对达到——对偶泛函 $\varphi^*$ 是支持在这 $5$ 个点上的离散测度，带有适当的符号。

## 双极定理与闭凸包

包含原点的 Banach 空间 $X$ 中的闭凸集 $C$ 由其**极集**确定：$C^\circ = \{ \varphi \in X^* : \varphi(x) \leq 1 \text{ 对所有 } x \in C \}$。对称地，$(C^\circ)^\circ = \{ x \in X : \varphi(x) \leq 1 \text{ 对所有 } \varphi \in C^\circ \}$，且**双极定理**说 $(C^\circ)^\circ = C$ 对于包含 $0$ 的闭凸集 $C$ 成立（其中双极是相对于 $X$ 和 $X^*$ 的典范配对）。

双极定理是 Hahn-Banach 几何形式的直接结果：$C$ 外的任何点可以通过一个连续泛函与 $C$ 分离，该泛函属于 $C^\circ$ 并见证双极包含的失败。因此，极集/双极对偶忠实表示闭凸集为其“支撑超平面数据”。

凸分析和优化主要是处理这种对偶。凸函数的 Fenchel-Legendre 变换正好是应用于 epigraph 的极集，由此产生的 Fenchel 对偶定理将 $f + g$ 的最小化简化为对偶变量上 $-f^* - g^*$ 的最大化，其中 $f^*$ 是共轭。现代优化中最干净的算法（近端方法、ADMM、镜
<!-- 本节内容因生成长度限制截断；完整推导请参阅本系列对应英文版本。 -->

---
