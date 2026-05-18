---
title: "抽象代数 (8)：Galois 理论 —— 域与群之间的桥梁"
date: 2021-09-15 09:00:00
tags:
  - abstract-algebra
  - galois-theory
  - field-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: zh
mathjax: true
description: "Galois 理论的基本定理建立了中间域和子群之间的完美对应关系，并解决了古老的根式可解性问题。"
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "abstract-algebra-8"
---
1832 年 5 月 29 日的深夜，巴黎的一间阁楼里，20 岁的 Évariste Galois 知道自己活不过明天。决斗的约定已经写下，对手是当时法国枪法最好的军官之一。在蜡烛快要燃尽的时候，他没有写遗书，而是抓起一叠草稿纸，疯狂地把自己脑子里关于多项式根的所有想法往外倒。他在页边匆匆写下“我没有时间了”，然后把这叠纸寄给了朋友。这些手稿在抽屉里躺了十几年才被数学界真正读懂，但它们彻底改写了代数的走向。

我第一次读到这段历史时，心里冒出的第一个问题不是“他证明了什么”，而是“他到底在躲什么”。为什么一个研究多项式求根的人，会突然抛开根的具体数值，跑去研究一堆抽象的置换？后来我自己推导了几次才猛然醒悟：Galois 根本不是在做计算，他是在做“对称性普查”。他意识到，死磕根的数值是一条死胡同，但如果你把目光从“根等于几”转移到“根之间有哪些代数关系”，整个问题就会从一团乱麻变成一张清晰的地图。这就是 Galois 理论（Galois theory）的核心直觉：不要盯着数字看，去盯着数字背后的对称结构看。

这篇文章是我自学这段内容时的完整复盘。我会先问“为什么需要这个概念”，再用日常类比铺路，最后才给出正式定义。每一个公式后面，我都会紧跟一句大白话解释。如果你曾经被“正规子群（normal subgroup）”或“Galois 对应（Galois correspondence）”这些词绕晕过，这篇就是为你写的。

![分裂域的对称性示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_splitting_symmetry.png)

---

## Galois 群：固定基域的自同构

![Galois 对应：域扩张与子群的双射](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/aa08_galois_correspondence.png)

![Galois 群：固定基域的自同构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_galois_group.png)

我们先从一个最朴素的困惑开始。假设你有一个域扩张（field extension）$L/K$，比如把有理数域 $\mathbb{Q}$ 扩大成 $\mathbb{Q}(\sqrt{2})$。在这个更大的域里，多了很多新元素。如果我们想研究这个扩张的“内部结构”，最直接的办法是什么？不是去列元素清单，而是去找那些“保持结构不变的变换”。在域的世界里，这种变换叫自同构（automorphism）。

但这里有个关键限制：我们只关心那些**不动基域 $K$** 的自同构。为什么？因为 $K$ 是我们的“地基”，地基要是跟着一起动，我们就失去了参照系。正式写出来就是：

$$\mathrm{Gal}(L/K) = \mathrm{Aut}_K(L) = \{\sigma : L \to L \mid \sigma \text{ 是一个域自同构},\ \sigma|_K = \mathrm{id}\}.$$

这句话的意思是，Galois 群（Galois group）专门收集那些把 $K$ 里每个数原封不动留下，只在上层域 $L$ 里做“合法洗牌”的映射，并且这些映射复合起来刚好构成一个群。

这个定义的威力在哪里？在于它自动抓住了多项式的根。假设 $f(x) \in K[x]$ 是一个系数全在 $K$ 里的多项式，$\alpha \in L$ 是它的一个根。如果你把群里的任意一个映射 $\sigma$ 作用在等式 $f(\alpha)=0$ 两边，会发生什么？

$$0 = \sigma(0) = \sigma(f(\alpha)) = f(\sigma(\alpha)).$$

因为 $\sigma$ 不动 $f$ 的系数，它只能把根 $\alpha$ 挪到另一个根的位置，等式依然成立。这句话翻译成人话就是：Galois 群里的每一个操作，都必然把多项式的根置换成另一个根，绝不可能把根变成非根。

这就引出了群作用（group action）的直觉。想象你在玩一个三阶魔方。群作用就像你转动魔方的某一面：色块的位置变了，但魔方“每个面九宫格、颜色互斥”的底层规则纹丝不动。Galois 群作用在多项式的根上，逻辑完全一样：根的位置被置换，但它们满足的代数方程和运算关系绝对不破裂。当 $L$ 恰好是 $f$ 的分裂域（splitting field）时，这种作用是忠实的（faithful）——只要你知道了 $\sigma$ 怎么置换根，你就知道了 $\sigma$ 在整个域 $L$ 上怎么作用，没有任何盲区。

![Q(√2)/Q 的 Galois 群通过符号翻转作用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_1_galois_group.png)

**例子 1：$\mathbb{Q}(\sqrt{2})/\mathbb{Q}$**
任何 $\sigma \in \mathrm{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$ 必须把 $\sqrt{2}$ 送到 $x^2 - 2$ 的根上，也就是 $\pm\sqrt{2}$。
$$\sigma(\sqrt{2}) \in \{\sqrt{2}, -\sqrt{2}\}.$$
这句话的意思是，自同构的选择被多项式的根死死限制住了，只有两条路可走。
于是恰好只有两个自同构：恒等映射 $\mathrm{id}$，以及把符号翻过来的 $\sigma(\sqrt{2}) = -\sqrt{2}$。Galois 群同构于 $\mathbb{Z}/2\mathbb{Z}$。
注意一个细节：$\sigma$ 完全由它在 $\sqrt{2}$ 上的表现决定。因为 $\mathbb{Q}(\sqrt{2})$ 里任何元素都能写成 $a + b\sqrt{2}$（$a,b \in \mathbb{Q}$），而 $\sigma$ 是线性的且不动有理数：
$$\sigma(a + b\sqrt{2}) = a + b\sigma(\sqrt{2}).$$
这句话的意思是，只要定好了生成元 $\sqrt{2}$ 的去向，整个域里所有元素的去向就自动确定了。
在符号翻转下，什么量会保持不变？算一下范数（norm）：
$$(a+b\sqrt{2})(a-b\sqrt{2}) = a^2 - 2b^2.$$
这句话的意思是，两个共轭元素相乘会消掉无理部分，结果自动落回 $\mathbb{Q}$，这正是 Galois 不变量的典型构造。

**例子 2：$\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$**
设 $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$。固定 $\mathbb{Q}$ 的自同构 $\sigma$ 可以独立决定 $\sqrt{2}$ 和 $\sqrt{3}$ 的符号：
$$\sigma(\sqrt{2}) = \pm\sqrt{2}, \quad \sigma(\sqrt{3}) = \pm\sqrt{3}.$$
这句话的意思是，两个平方根的最小多项式互不干扰，自同构可以像拨开关一样独立翻转它们的正负号。
组合起来给出四个自同构：恒等、只翻 $\sqrt{2}$、只翻 $\sqrt{3}$、两个都翻。Galois 群是 $\mathbb{Z}/2 \times \mathbb{Z}/2$，也就是 Klein 四元群 $V_4$。
$$|\mathrm{Gal}(L/\mathbb{Q})| = 4 = [L:\mathbb{Q}].$$
这句话的意思是，自同构的个数刚好等于域扩张的次数，这是 Galois 扩张（Galois extension）的标志性特征。

**例子 3（反例）：$\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$**
最小多项式是 $x^3 - 2$。它有三个根：$\sqrt[3]{2}$（实数），$\sqrt[3]{2}\omega$，$\sqrt[3]{2}\omega^2$（复数，其中 $\omega = e^{2\pi i/3}$）。
但我们的域 $L = \mathbb{Q}(\sqrt[3]{2})$ 只包含实数。任何自同构 $\sigma: L \to L$ 必须把 $\sqrt[3]{2}$ 映到 $L$ 内部的根上：
$$\sigma(\sqrt[3]{2}) \in L \cap \{\text{roots of } x^3-2\} = \{\sqrt[3]{2}\}.$$
这句话的意思是，因为域里只装得下一个实根，自同构根本无处可去，只能原地不动。
所以 $\sigma = \mathrm{id}$，Galois 群是平凡群 $\{e\}$。
$$|\mathrm{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})| = 1 \neq 3 = [\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}].$$
这句话的意思是，自同构个数严格小于扩张次数，因为扩张“漏掉”了另外两个复根，破坏了正规性（normality）。

这就是上一篇可分性与正规性铺垫的伏笔：等式 $|\mathrm{Gal}(L/K)| = [L:K]$ 只在 Galois 扩张里成立。$\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ 不是分裂域，群直接坍缩了。

$$L/K \text{ 是有限 Galois 扩张} \iff |\mathrm{Gal}(L/K)| = [L:K].$$
这句话的意思是，你可以用“数自同构个数”这种纯操作性的办法，来检验一个扩张是不是 Galois 的，两边完全等价。

这个等价命题的两半都极其好用。正向：数出自同构个数等于次数，立刻断定是 Galois 扩张。反向：已知是 Galois 扩张，如果你只找到了 $[L:K]$ 个自同构，就可以停手了，绝对没有漏网之鱼。它也和“可分+正规”的抽象定义严丝合缝：可分保证最多有 $[L:K]$ 个嵌入，正规保证这些嵌入全部落回 $L$ 内部，加起来就是恰好 $[L:K]$ 个自同构。“嵌入数等于次数”只是 $|\mathrm{Gal}(L/K)| = [L:K]$ 的另一种说法。

Galois 群虽然是个有限的小群，但它把整个域扩张的骨架全抽出来了。群论已经积累了两百年的结构定理（Sylow 定理、单群分类、表示论），Galois 理论等于开了一条隧道，把这些现成的武器全部搬进多项式求根的问题里。我们在第 7 篇搭的所有脚手架，就是为了换这张门票。

这里还有一个极其重要的视角转换。Galois 群本质上是多项式 $f$ 的自同构群，精确体现在它对根的作用上。我们可以等价地把 $\mathrm{Gal}(f)$ 定义为对称群 $S_n$（$n = \deg f$）的一个子群：
$$\mathrm{Gal}(f) = \{\pi \in S_n \mid \pi \text{ 尊重根之间的所有代数关系}\}.$$
这句话的意思是，不是所有根的排列都合法，只有那些不破坏根之间加减乘除恒等式的排列，才能升级成域自同构。
“尊重代数关系”这个条件是整个定义的命门。如果没有它，任何置换都合法，Galois 群永远是 $S_n$，理论就废了。有些置换之所以不能扩张成域自同构，正是因为它们会撕碎某些隐藏的代数等式。你在计算群的时候，利用的正是这些“被撕碎的等式”。

**小验证：$f(x) = x^4 + 1$**
这是第 8 分圆多项式。四个根是 $\zeta_8, \zeta_8^3, \zeta_8^5, \zeta_8^7$（8 次本原单位根）。
根之间有硬性的代数关系，比如：
$$\zeta_8^3 = \zeta_8 \cdot \zeta_8^2 = \zeta_8 \cdot i, \quad \zeta_8 + \zeta_8^7 = \sqrt{2}.$$
这句话的意思是，根不是独立漂浮的四个点，它们被乘法和加法关系紧紧绑在一起。
如果你随便挑一个 $S_4$ 里的置换（共 24 种），大概率会把 $\zeta_8 + \zeta_8^7 = \sqrt{2}$ 这种等式拆坏。只有 4 个置换能同时保住所有关系，它们构成一个 4 阶群，同构于 $(\mathbb{Z}/8)^\times$。
$$|\mathrm{Gal}(x^4+1)| = 4 \ll 24 = 4!.$$
这句话的意思是，代数关系像滤网一样筛掉了 20 个非法置换，剩下的 4 个才是真正合法的 Galois 群。

![Galois 群置换多项式根的动画演示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_root_permutation.gif)

---

## 固定域和 Galois 对应

![子群的固定域示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_fixed_field.png)

在正式翻开固定域的字典之前，我想先补一个极其接地气的直觉：为什么“不动点”能拼成一个域？你可以把自同构想象成一面镜子。如果你手里拿着一个不对称的物件（比如 $\sqrt{2}$），照镜子它会变成 $-\sqrt{2}$；但如果你把物件和它的镜像绑在一起（比如算和 $\sqrt{2} + (-\sqrt{2}) = 0$，或者算积 $\sqrt{2} \cdot (-\sqrt{2}) = -2$），结果就彻底对称了，镜子再也照不出区别。在代数里，这叫“对称化操作”。随便取 $L$ 里的一个元素 $\alpha$，把它在所有 $\sigma \in G$ 作用下的像加起来得到迹（trace）$\mathrm{Tr}_{L/K}(\alpha) = \sum_{\sigma \in G} \sigma(\alpha)$，乘起来得到范数（norm）$\mathrm{N}_{L/K}(\alpha) = \prod_{\sigma \in G} \sigma(\alpha)$。因为群作用只是把求和/求积的顺序打乱，结果纹丝不动，所以迹和范数自动掉进固定域 $K$ 里。拿 $\mathbb{Q}(\sqrt{2})$ 试一下：$\alpha = 3 + 5\sqrt{2}$，$\sigma(\alpha) = 3 - 5\sqrt{2}$。迹是 $6$，范数是 $9 - 50 = -41$，全是有理数。这不是巧合，而是群作用强行“榨干”了无理部分。固定域的本质，就是收集所有被这种对称化操作保护下来的量。

群拿到了，怎么用它反推域的结构？这里有一个极其漂亮的“反向字典”。

给定扩张 $L/K$，我们得到群 $G = \mathrm{Gal}(L/K)$。反过来，如果给你 $G$ 的一个子群 $H$，你能得到什么？你能得到一个域：
$$L^H = \{\alpha \in L : \sigma(\alpha) = \alpha \text{ 对所有 } \sigma \in H\}.$$
这句话的意思是，$L^H$ 专门收集那些在 $H$ 里所有映射折腾下都“岿然不动”的元素，它们自己刚好能拼成 $L$ 的一个子域，叫作 $H$ 的固定域（fixed field）。

固定域一定包含 $K$，因为 $K$ 里的元素被整个 $G$ 固定，自然也被子群 $H$ 固定。于是我们手里有了两张地图，可以来回翻译：
- $\Phi$：子群 $\to$ 中间域，$H \mapsto L^H$。
- $\Psi$：中间域 $\to$ 子群，$M \mapsto \mathrm{Gal}(L/M)$。

$$L/K \text{ 有限 Galois} \implies \Phi, \Psi \text{ 是互逆、序反向的双射。}$$
这句话的意思是，子群和中间域是一一对应的，但方向是反的：子群越大，施加的“不动”约束越多，能幸存下来的元素就越少，固定域就越小。
符号写出来就是 $H_1 \leq H_2 \implies L^{H_1} \supseteq L^{H_2}$。第一次见这个“序反向”的人都会愣一下，但想通“约束越多，地盘越小”就极其自然。

两张地图的刻度由两个公式死死钉住：
$$[L : L^H] = |H|.$$
这句话的意思是，从大域 $L$ 往下走到固定域 $L^H$ 的扩张次数，刚好等于负责固定的子群 $H$ 的元素个数。
$$[L^H : K] = [\mathrm{Gal}(L/K) : H].$$
这句话的意思是，从地基 $K$ 往上走到固定域 $L^H$ 的次数，刚好等于子群 $H$ 在全群里的指数（陪集个数）。
把两式相乘，直接得到 $|\mathrm{Gal}(L/K)| = [L:K]$，完美闭环。

对应关系的证明靠在两块基石上：
1. **Artin 引理（Artin's Lemma）**：如果 $G$ 是 $L$ 的有限自同构群，则 $[L : L^G] = |G|$。
$$[L : L^G] = |G| \text{ 说明任何有限群作用在域上，都能把该域变成其固定域的 Galois 扩张。}$$
这句话的意思是，群作用不是被动观察，它能主动“生成”域扩张结构。
2. **Galois 定理**：如果 $L/K$ 有限 Galois，则 $L^{\mathrm{Gal}(L/K)} = K$。
$$L^{\mathrm{Gal}(L/K)} = K \text{ 说明 } K \text{ 之外的任何元素，都至少会被某个自同构挪动。}$$
这句话的意思是，自同构的“视力”极好，没有任何非基域元素能躲过所有对称变换的审查。

把两块石头拼起来：从 Galois 扩张出发，全群的固定域是 $K$；对任意子群 $H$ 用 Artin 引理，得到 $[L:L^H]=|H|$。双射的齿轮就此咬合。

**为什么这很重要？** 这是把域论难题翻译成群论计算的跨海大桥。想知道 $K$ 和 $L$ 之间藏了多少个中间域？去数 $\mathrm{Gal}(L/K)$ 的子群。想知道哪些中间域自己也是 Galois 扩张？去挑正规子群。域里需要硬算元素关系的问题，在群这边全变成了画格子、数指数的组合游戏。

随身带一个实用引理，画格子时极其顺手。设 $H_1, H_2 \leq G$，固定域分别是 $M_1, M_2$：
$$L^{H_1 \cap H_2} = M_1 \cdot M_2.$$
这句话的意思是，两个子群的交集对应的固定域，是原来两个域的合成域（包含两者的最小域），因为要同时满足两拨人的“不动”要求，地盘只能扩大。
$$L^{\langle H_1, H_2 \rangle} = M_1 \cap M_2.$$
这句话的意思是，两个子群生成的更大子群对应的固定域，是原来两个域的交集，因为约束变多了，幸存元素自然变少。
子群格上的交与并，完美反转为子域格上的并与交。内化这一点后，画任何具体扩张的子域图，都变成纯粹的群论填空题。

对应里还藏着一个强唯一性结论：
$$\sigma \in \mathrm{Gal}(L/K) \text{ 在每个中间域上作用平凡} \implies \sigma = \mathrm{id}.$$
这句话的意思是，如果一个自同构在所有子域上都装死，那它在整个大域上也只能是恒等映射，子域格加上群作用已经榨干了扩张的全部信息。

![Galois 对应：Galois 群的子群匹配中间域](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_2_correspondence.png)

---

## Galois 理论基本定理

把前面的齿轮全部组装起来，就得到整个理论的核心引擎。

设 $L/K$ 是有限 Galois 扩张，Galois 群为 $G = \mathrm{Gal}(L/K)$。则：

1. **（双射）** 映射 $H \mapsto L^H$ 和 $M \mapsto \mathrm{Gal}(L/M)$ 是子群集 $G$ 和中间域集 $K \subseteq M \subseteq L$ 之间的互逆、序反向双射。
$$H \leftrightarrow M \text{ 是一一对应且方向相反的，子群越大，对应的中间域越小。}$$

2. **（次数匹配）** 对每个子群 $H \leq G$，$[L : L^H] = |H|$ 且 $[L^H : K] = [G : H]$。
$$\text{扩张次数被精准拆分成子群阶数与指数，群的大小直接量出了域的厚度。}$$

3. **（正规性）** 中间扩张 $M/K$ 是 Galois（等价于正规）当且仅当 $\mathrm{Gal}(L/M)$ 是 $G$ 的正规子群（normal subgroup）。此时 $\mathrm{Gal}(M/K) \cong G/\mathrm{Gal}(L/M)$。
$$M/K \text{ 正规} \iff \mathrm{Gal}(L/M) \trianglelefteq G, \text{ 且商群正好是下层扩张的 Galois 群。}$$

第三部分是整条定理最锋利的地方。“正规”这个词在域和群两边同时出现，但初学时总觉得是巧合。其实根本不是。域那边的“正规”指“某个多项式的分裂域”，群那边的“正规”指“对共轭封闭（$gHg^{-1}=H$）”。定理说这是一回事。

**为什么不是所有子群都正规？** 直觉在于“对称性是否均匀”。想象一个长方形和一个正方形。正方形的旋转对称群是正规的，因为你从哪个角度切入，旋转操作都彼此兼容；但长方形的某些翻转操作就不正规，因为“先翻转再旋转”和“先旋转再翻转”会落到不同的状态上。在群论里，正规意味着子群的结构在整体群的“视角切换”（共轭）下保持不变。如果子群不正规，说明它在某些共轭操作下会“歪掉”，对应的中间域也就无法在基域上保持对称（不是分裂域）。

![正规子群对应于正规域扩张](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_7_normal_field.png)

**第 (3) 部分的直觉推导：**
如果 $H \trianglelefteq G$，我们可以把 $G$ 里的每个 $\sigma$ 限制到 $L^H$ 上，得到一个同态 $G \to \mathrm{Aut}_K(L^H)$。
$$\text{限制映射良定义，因为正规性保证 } \sigma(L^H) = L^H \text{，元素不会跑出固定域。}$$
核恰好是 $H$，所以 $G/H$ 嵌入到 $\mathrm{Aut}_K(L^H)$ 里。数一下大小：$|G/H| = [L^H : K]$。由 Galois 条件直接推出 $G/H \cong \mathrm{Gal}(L^H/K)$，且 $L^H/K$ 是 Galois 扩张。
$$G/H \cong \mathrm{Gal}(L^H/K) \text{ 说明把大群模掉固定子群，剩下的商群完美描述了底层域的对称性。}$$

反过来，如果 $M/K$ 是 Galois 扩张，$M$ 是某个多项式的分裂域。任何 $\sigma \in G$ 都会把 $M$ 的生成根映成同多项式的其他根，而这些根全在 $M$ 里（分裂域的定义）。
$$\sigma(M) \subseteq M \implies \sigma(M) = M \text{，说明 } M \text{ 在所有自同构下整体稳定。}$$
限制映射 $G \to \mathrm{Aut}_K(M)$ 良定义，核是 $\mathrm{Gal}(L/M)$，核自动正规。

“在 $\sigma$ 下稳定”这一步值得多看两眼。设 $M = K(\alpha_1, \ldots, \alpha_r)$ 是 $f \in K[x]$ 的分裂域。$\sigma$ 固定 $f$ 的系数，所以必然把根 $\alpha_i$ 送到另一个根 $\alpha_j$。其他根本来就在 $M$ 里，所以 $\sigma(\alpha_i) \in M$。生成元跑不出 $M$，整个域就跑不出 $M$。维数一卡，$\sigma(M)=M$。限制映射顺理成章。

基本定理是 Galois 理论里所有具体计算的发动机。想找 $L$ 的子域？画 $G$ 的子群格，两张图一模一样。想知道哪个子域是 Galois 的？在子群格上圈出正规子群。整个问题从“必须死算元素”的域论，平移到了“可以画图数格子”的群论。

FTGT 还有三个极其实用的结构红利，值得单列：
1. **数中间域**：$L$ 里包含 $K$ 的子域个数 $=$ $G$ 的子群个数。$G \cong S_3$ 有 6 个子群，所以有 6 个中间域。$G \cong (\mathbb{Z}/p)^n$ 时，子域个数等于有限向量空间的子空间个数，直接套高斯二项式系数。
$$\#\{\text{中间域}\} = \#\{\text{子群}\} \text{，域的结构复杂度被群的组合结构完全接管。}$$
2. **找 Galois 闭包**：包含给定 $M \subseteq L$ 的最小 Galois 扩张，对应 $G$ 中包含 $\mathrm{Gal}(L/M)$ 的最大正规子群（正规核）。当你手里只有一个非 Galois 扩张却硬要做 Galois 理论时，这招能一键补全缺失的对称性。
$$\text{Galois 闭包} \leftrightarrow \text{正规核} \text{，补全域的过程等价于在群里找最大的正规碎片。}$$
3. **商群耦合**：只要 $H \trianglelefteq G$，商群 $G/H$ 就会在 $L^H$ 上诱导一个作用，这个作用就是 $\mathrm{Gal}(L^H/K)$。大群的计算经常被拆成“算 $H$”和“算 $G/H$”两步，分而治之。
$$\mathrm{Gal}(L^H/K) \cong G/H \text{，高层对称性模掉底层对称性，刚好露出中间层的对称结构。}$$

---

## 计算 Galois 群：具体例子

理论再漂亮，不动手算一遍永远是空中楼阁。我们直接下场地，把几个经典扩张的 Galois 群和对应关系完整扒开。

### 例子 1：$\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$ 的完整对应

前面已经看出 $G \cong V_4 = \mathbb{Z}/2 \times \mathbb{Z}/2$。我们把子群和固定域一对一算清楚。
$V_4$ 有 5 个子群，对应 5 个中间域：

- $\{e\}$：不施加任何约束，固定域是 $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$。次数 $[L:\mathbb{Q}] = 4$，匹配 $|G|/|\{e\}| = 4/1 = 4$。
$$[L:L^{\{e\}}] = 4 = |\{e\}|^{-1}|G| \text{，恒等子群对应最大的域。}$$
- $\langle \sigma_2 \rangle$（$\sigma_2: \sqrt{2} \mapsto -\sqrt{2}, \sqrt{3} \mapsto \sqrt{3}$）：只允许 $\sqrt{3}$ 存活。固定域 $\mathbb{Q}(\sqrt{3})$。次数 2。
$$\sigma_2(a+b\sqrt{2}+c\sqrt{3}+d\sqrt{6}) = a-b\sqrt{2}+c\sqrt{3}-d\sqrt{6} \text{，令其不变推出 } b=d=0。$$
这句话的意思是，把一般元素展开作用一遍，系数对比直接筛出固定域基底。
- $\langle \sigma_3 \rangle$（$\sigma_3: \sqrt{2} \mapsto \sqrt{2}, \sqrt{3} \mapsto -\sqrt{3}$）：固定域 $\mathbb{Q}(\sqrt{2})$。次数 2。
- $\langle \sigma_2 \sigma_3 \rangle$（同时翻转 $\sqrt{2}, \sqrt{3}$）：注意 $\sqrt{6} = \sqrt{2}\sqrt{3}$ 的符号是 $(-1)\times(-1)=1$，不变！固定域 $\mathbb{Q}(\sqrt{6})$。次数 2。
$$\sigma_2\sigma_3(\sqrt{6}) = (-\sqrt{2})(-\sqrt{3}) = \sqrt{6} \text{，双负得正让乘积意外幸存。}$$
- $G$：所有自同构一起上，只有有理数能活下来。固定域 $\mathbb{Q}$。次数 1。

五个子群，五个中间域，次数全部严丝合缝。$V_4$ 是交换群，所有子群自动正规，所以每个中间域都是 $\mathbb{Q}$ 的 Galois 扩张。直接验证也很容易：$\mathbb{Q}(\sqrt{2})$ 是 $x^2-2$ 的分裂域，$\mathbb{Q}(\sqrt{3})$ 是 $x^2-3$ 的分裂域，$\mathbb{Q}(\sqrt{6})$ 是 $x^2-6$ 的分裂域。
这是最标准的“双二次扩张”。它也解释了为什么 $\sqrt{6}$ 会凭空冒出来：对角自同构同时翻转两个根，乘积的符号抵消，$\sqrt{6}$ 成为新的不变量。这个技巧在代数数论里反复出现，比如 Pell 方程的整数环 $\mathbb{Z}[\sqrt{6}]$ 就干净地嵌在 $\mathbb{Z}[\sqrt{2}, \sqrt{3}]$ 里。

### 例子 2：$x^3 - 2$ 在 $\mathbb{Q}$ 上的分裂域

设 $L = \mathbb{Q}(\sqrt[3]{2}, \omega)$，其中 $\omega = e^{2\pi i/3} = \frac{-1+\sqrt{-3}}{2}$。上一篇算过 $[L:\mathbb{Q}] = 6$。
$x^3 - 2$ 的三个根是 $\alpha_1 = \sqrt[3]{2}$，$\alpha_2 = \sqrt[3]{2}\omega$，$\alpha_3 = \sqrt[3]{2}\omega^2$。
Galois 群置换这三个根，自然嵌入 $S_3$。因为阶数是 6，直接锁定 $G \cong S_3$。
$$G \hookrightarrow S_3 \text{ 且 } |G|=6 \implies G \cong S_3 \text{，群的大小填满了对称群，说明所有置换都合法。}$$

写出生成元的具体作用：
- $\sigma$（3 阶）：$\sqrt[3]{2} \mapsto \sqrt[3]{2}\omega$，$\omega \mapsto \omega$。它循环置换三个根 $(\alpha_1 \alpha_2 \alpha_3)$。
- $\tau$（2 阶）：$\sqrt[3]{2} \mapsto \sqrt[3]{2}$，$\omega \mapsto \omega^2$（复共轭）。它交换 $\alpha_2, \alpha_3$，固定 $\alpha_1$。对应置换 $(23)$。

$S_3$ 的子群格非常经典，我们逐个算固定域：
- $\{e\}$ → $L$。
- $\langle \tau \rangle$（固定 $\sqrt[3]{2}$）→ $\mathbb{Q}(\sqrt[3]{2})$。次数 3。
- $\langle \sigma\tau \rangle$ → 固定 $\sqrt[3]{2}\omega^2$ → $\mathbb{Q}(\sqrt[3]{2}\omega^2)$。次数 3。
- $\langle \sigma^2\tau \rangle$ → 固定 $\sqrt[3]{2}\omega$ → $\mathbb{Q}(\sqrt[3]{2}\omega)$。次数 3。
$$\text{三个 2 阶子群彼此共轭，对应的三个三次域同构但互不相等。}$$
这句话的意思是，群里的共轭关系完美翻译成了域的“同构嵌入不同位置”现象。
- $\langle \sigma \rangle = A_3$（3 阶循环）：固定 $\omega$。因为 $\sigma$ 不动 $\omega$，而 $\tau$ 会把 $\omega$ 翻成 $\omega^2$。固定域 $\mathbb{Q}(\omega)$。次数 2。
- $S_3$ → $\mathbb{Q}$。

关键观察：三个 2 阶子群在 $S_3$ 里**不正规**（比如 $\sigma \langle \tau \rangle \sigma^{-1} = \langle \sigma\tau\sigma^{-1} \rangle \neq \langle \tau \rangle$）。对应地，$\mathbb{Q}(\sqrt[3]{2})$ 等三个域在 $\mathbb{Q}$ 上**不是 Galois 扩张**。它们都不是任何多项式的分裂域（漏了复根），也不在所有自同构下封闭。
而 $A_3$ 指数为 2，自动正规。对应地，$\mathbb{Q}(\omega)/\mathbb{Q}$ 是 Galois 扩张（它是 $x^2+x+1$ 的分裂域，Galois 群 $\mathbb{Z}/2$）。
$$H \trianglelefteq G \iff M/K \text{ 是 Galois 扩张，正规性在群和域两侧同步出现或同步消失。}$$

数值检查 FTGT：取 $\langle \tau \rangle$，阶 2，固定域次数 $6/2=3$，匹配。取 $A_3$，阶 3，固定域次数 $6/3=2$，匹配。数字本身很平凡，震撼的是 $S_3$ 的每一个子群都在域那边有一个专属的“影子域”，一一对应，绝无错漏。

### 例子 3：$x^4 - 2$ 在 $\mathbb{Q}$ 上的分裂域

设 $L = \mathbb{Q}(\sqrt[4]{2}, i)$。$[L:\mathbb{Q}] = 8$。四个根是 $\pm\sqrt[4]{2}, \pm i\sqrt[4]{2}$。
生成元：
- $r$（4 阶）：$\sqrt[4]{2} \mapsto i\sqrt[4]{2}$，$i \mapsto i$。循环四个根。
- $s$（2 阶）：$\sqrt[4]{2} \mapsto \sqrt[4]{2}$，$i \mapsto -i$。复共轭。
关系式：$r^4 = s^2 = 1$，$srs = r^{-1}$。
$$G \cong D_4 \text{，八阶二面体群，正是正方形的对称群。}$$
这句话的意思是，四个根在复平面上刚好构成正方形，Galois 群就是旋转和翻转正方形的所有操作。

![x^4 - 2 的分裂域的完整 Galois 对应](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_3_x4_minus_2.png)

$D_4$ 有 10 个子群，所以有 10 个中间域。
中心 $\langle r^2 \rangle$ 正规（$r^2$ 把 $\sqrt[4]{2} \mapsto -\sqrt[4]{2}$，$i \mapsto i$，固定域 $\mathbb{Q}(i, \sqrt{2})$）。
三个指数 2 的 4 阶子群（$\langle r \rangle$, $\langle r^2, s \rangle$, $\langle r^2, rs \rangle$）都正规，对应三个 Galois 子扩张：$\mathbb{Q}(i)$, $\mathbb{Q}(\sqrt{2})$, $\mathbb{Q}(i\sqrt{2})$。
两个非正规的 2 阶子群 $\langle s \rangle$ 和 $\langle r^2s \rangle$ 共轭，固定域分别是 $\mathbb{Q}(\sqrt[4]{2})$ 和 $\mathbb{Q}(i\sqrt[4]{2})$。它们不是 Galois 扩张，但彼此同构。
$$\text{群论共轭类} \leftrightarrow \text{域的同构嵌入类，FTGT 连“长得一样但位置不同”的域都能精准分类。}$$
每次看到这个对应我都觉得极其舒适：Galois 对应不只是匹配集合，它连结构的“对称等价类”都一并翻译了。

### 例子 4：分圆域（Cyclotomic Fields）

$n$ 阶分圆域是 $\mathbb{Q}(\zeta_n)$，$\zeta_n = e^{2\pi i/n}$。极小多项式是 $n$ 阶分圆多项式 $\Phi_n(x)$，次数 $\varphi(n)$。
$$\mathrm{Gal}(\mathbb{Q}(\zeta_n)/\mathbb{Q}) \cong (\mathbb{Z}/n\mathbb{Z})^\times.$$
这句话的意思是，分圆扩张的 Galois 群同构于模 $n$ 的乘法单位群，结构完全由数论决定。
同构映射把 $\sigma$ 送到唯一的 $a \in (\mathbb{Z}/n)^\times$，满足 $\sigma(\zeta_n) = \zeta_n^a$。证明核心是两点：(i) $\Phi_n$ 在 $\mathbb{Q}$ 上不可约（Gauss 定理）；(ii) $\varphi(n)$ 个本原根正好是 $\zeta_n^a$（$\gcd(a,n)=1$），每个自同构就是选一个合法的 $a$。

![分圆扩张及其 Abel Galois 群](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_6_cyclotomic.png)

因为 $(\mathbb{Z}/n)^\times$ 是交换群，所有分圆扩张都是 Abel 扩张。Kronecker-Weber 定理（1853 提出，1886 证毕）说逆命题也成立：
$$\mathbb{Q} \text{ 的每个有限 Abel 扩张都包含在某个分圆域里。}$$
这句话的意思是，分圆域是 $\mathbb{Q}$ 上所有交换对称性的“万能容器”，类域论（class field theory）的雏形就在这里。

算个具体的：$n=8$。$\mathrm{Gal}(\mathbb{Q}(\zeta_8)/\mathbb{Q}) \cong (\mathbb{Z}/8)^\times \cong \mathbb{Z}/2 \times \mathbb{Z}/2$。
中间域是 $\mathbb{Q}(\sqrt{2})$, $\mathbb{Q}(i)$, $\mathbb{Q}(i\sqrt{2})$。全是二次域，干净地嵌在 8 阶分圆域里。
高斯和恒等式不再神秘：
$$\zeta_8 + \zeta_8^{-1} = \sqrt{2}.$$
这句话的意思是，左边是分圆域元素在复共轭下的迹（trace），自动掉进指数 2 的固定域 $\mathbb{Q}(\sqrt{2})$ 里。

### 例子 5：有限域（Finite Fields）

有限域的 Galois 理论干净到令人发指。
$$\mathrm{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p) = \langle \mathrm{Frob}_p \rangle \cong \mathbb{Z}/n\mathbb{Z}, \quad \mathrm{Frob}_p(x) = x^p.$$
这句话的意思是，有限域扩张的 Galois 群是由 Frobenius 自同构生成的循环群，结构完全由次数 $n$ 决定。
每个有限扩张都是 Galois 的，子群格就是 $n$ 的因数格。$\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n} \iff d \mid n$。$\mathbb{F}_{p^n}/\mathbb{F}_{p^d}$ 的 Galois 群由 $\mathrm{Frob}_p^d$ 生成，阶数 $n/d$。
$$x \mapsto x^p \text{ 保持加法和乘法，且不动 } \mathbb{F}_p \text{，是有限域世界里唯一且最强的对称操作。}$$

**为什么算 Galois 群不是纸上谈兵？** 它是代数数论和现代密码学的算法底座。判断多项式根能否闭式表达、计算类数、构造互反律、椭圆曲线密码学，底层全在跑 Galois 群计算。任何计算机代数系统（SageMath, PARI/GP）的数域模块，本质都是一个披着因式分解外衣的 Galois 群计算器。

实战技巧三条：
1. 对 $\mathbb{Q}[x]$ 里 $n$ 次不可约多项式，Galois 群是 $S_n$ 的传递子群。$S_4$ 有 5 个，$S_5$ 有 5 个，$S_6$ 有 16 个。用 `polgalois` 或 `galois_group()` 一键识别。
$$\text{传递性保证群能把任意根送到任意根，对应多项式不可约。}$$
2. 模 $p$ 分解模式暴露 Frobenius 的循环结构。Chebotarev 密度定理保证，跑遍素数 $p$，群里的每种循环型都会按正确比例出现。因式分解 $f \bmod p$ 是猜 Galois 群的最强启发式。
$$f \bmod p \text{ 的不可约因子次数} = \mathrm{Frob}_p \text{ 在 } \mathrm{Gal}(f) \text{ 中的轮换长度。}$$
3. 判别式 $\mathrm{disc}(f)$ 是 $\mathbb{Q}$ 中的平方 $\iff \mathrm{Gal}(f) \subseteq A_n$。看判别式直接砍掉一半候选群。
$$\sqrt{\mathrm{disc}(f)} \text{ 在奇置换下变号，平方性正好检测群是否全为偶置换。}$$

---

## 根式可解性与可解群

现在回到那个折磨了数学家三百年的原始问题：这个多项式到底能不能用根号解出来？
Galois 的绝杀在于，他把“能不能开根号”翻译成了“群能不能一层层剥开”。

先严格定义什么叫“根式可解（solvable by radicals）”。多项式 $f(x) \in K[x]$ 在 $K$ 上根式可解，如果存在一个域塔：
$$K = K_0 \subseteq K_1 \subseteq \cdots \subseteq K_r$$
$$K_{i+1} = K_i(\sqrt[n_i]{a_i}) \text{，且 } f \text{ 的分裂域包含在 } K_r \text{ 中。}$$
这句话的意思是，你可以从基域出发，通过有限次“添加 $n$ 次根”的操作，最终把多项式的所有根都装进来。
换句话说：只用 $+,-,\times,\div$ 和 $\sqrt[n]{\cdot}$，你能写出根的表达式。

再看群那边的对应概念。群 $G$ 叫可解群（solvable group），如果存在一条子群链：
$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_n = G$$
$$G_{i+1}/G_i \text{ 是 Abel 群。}$$
这句话的意思是，复杂的群可以像剥洋葱一样一层层拆开，每一层剥下来的“商”都是交换群，也就是结构最简单、最听话的那类群。

![可解群与根式扩张的对应关系](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_solvable_groups.png)

为了把“根式塔”和“可解链”真正焊死，我们亲手算一层最典型的根式扩张（radical extension）。假设基域 $K$ 已经包含了 $n$ 次单位根 $\zeta_n$（比如 $K=\mathbb{C}$ 或适当扩域），我们添加一个 $n$ 次根 $\alpha = \sqrt[n]{a}$ 得到 $L = K(\alpha)$。$L/K$ 的 Galois 群长什么样？任何 $\sigma \in \mathrm{Gal}(L/K)$ 必须把 $\alpha$ 送到 $x^n - a$ 的另一个根上，也就是 $\zeta_n^k \alpha$。所以 $\sigma$ 完全由指数 $k \in \mathbb{Z}/n\mathbb{Z}$ 决定。复合两个自同构 $\sigma_k \circ \sigma_m$ 作用在 $\alpha$ 上：$\sigma_k(\zeta_n^m \alpha) = \zeta_n^m \sigma_k(\alpha) = \zeta_n^m \zeta_n^k \alpha = \zeta_n^{k+m} \alpha$。指数直接相加！这说明 $\mathrm{Gal}(L/K)$ 同构于加法群 $\mathbb{Z}/n\mathbb{Z}$，是绝对的交换群（Abelian group）。现在回头看可解群的定义 $G_{i+1}/G_i$ 是 Abel 群，它翻译回域语言就是：每一层扩张 $K_{i+1}/K_i$ 的对称群都是交换的。开根号之所以能“解”方程，正是因为根号扩张的对称性足够简单（循环/交换），允许我们把高维的纠缠一层层拆解成低维的线性叠加。当对称群复杂到像 $A_5$ 那样没有正规子群可切时，这根“拆解链条”就断了，根号也就无能为力了。

为什么叫“可解”？因为历史上人们解方程就是一层层降次：二次配方、三次卡丹公式、四次费拉里法，每一步都在做根式扩张，而根式扩张对应的 Galois 群商恰好是交换的（循环群）。可解群的定义就是把这个操作流程抽象成了纯群论语言。

哪些群可解？
- 所有 Abel 群（链长 1）。
- 所有阶 $<60$ 的群。
- $S_n$ 对 $n \leq 4$。$S_3$ 有 $\{e\} \trianglelefteq A_3 \trianglelefteq S_3$，商是 $\mathbb{Z}/3, \mathbb{Z}/2$。$S_4$ 有 $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$，商是 $V_4, \mathbb{Z}/3, \mathbb{Z}/2$。全交换。
$$S_4 \text{ 的可解链解释了为什么四次方程还有求根公式，对称性还没复杂到失控。}$$
- 所有 $p$-群（中心非平凡，归纳即得）。

但 $S_n$ 对 $n \geq 5$ **不可解**。瓶颈在交错群 $A_n$。
$$n \geq 5 \implies A_n \text{ 是单群（simple group）且非 Abel。}$$
这句话的意思是，$A_5$ 就像一块实心的铁球，你找不到任何非平凡的正规子群把它切开。既然切不开，就没法构造出交换商，可解链在第一步就彻底断裂。

把这两块拼图合起来，就是 Galois 真正的定理：
$$f \in \mathbb{Q}[x] \text{ 根式可解} \iff \mathrm{Gal}(f/\mathbb{Q}) \text{ 是可解群。}$$
这句话的意思是，“能不能用根号写出来”的代数难题，被精准等价成了“对应的对称群能不能一层层剥成交换群”的结构问题。

![一般五次方程不可解的群论障碍](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/08_quintic.png)

构造一个 Galois 群恰好是 $S_5$ 的具体五次多项式并不困难。经典例子：
$$f(x) = x^5 - 4x + 2.$$
验证步骤完整走一遍：
1. **不可约**：用 Eisenstein 准则（素数 $p=2$ 整除 $-4, 2$，不整除首项 $1$，$2^2=4$ 不整除常数项 $2$）。$f$ 在 $\mathbb{Q}$ 上不可约。
$$\text{不可约} \implies \mathrm{Gal}(f) \text{ 在 5 个根上作用是传递的。}$$
2. **实根个数**：求导 $f'(x) = 5x^4 - 4$。令导数为 0 得 $x = \pm \sqrt[4]{4/5} \approx \pm 0.945$。代入原函数算极值：$f(-0.945) > 0$，$f(0.945) < 0$。函数图像穿 x 轴三次。所以 $f$ 有 3 个实根，2 个共轭复根。
$$\text{复共轭置换恰好交换那 2 个复根，在 } S_5 \text{ 中是一个对换（transposition）。}$$
3. **群生成**：传递子群包含一个 5-循环（由不可约性保证 Frobenius 元素或 Cauchy 定理）和一个对换。在 $S_5$ 里，一个 5-循环加一个对换直接生成整个 $S_5$。
$$\langle (12345), (ij) \rangle = S_5 \implies \mathrm{Gal}(f) \cong S_5.$$
这句话的意思是，多项式的具体分析直接锁定了最大可能的对称群，没有任何缩小余地。
4. **结论**：$S_5$ 不可解 $\implies x^5 - 4x + 2 = 0$ 没有根式求根公式。

![A_5 的单性与非交换性阻挡了五次方程的根式解](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_5_quintic.png)

“五次方程没有求根公式”从来不是一句模糊的哲学感叹，而是写在纸上的某一个具体方程 $x^5 - 4x + 2 = 0$ 绝对无法用根号表达。Abel 在 1824 年证明了不可解性，Galois 在 1832 年把它重写成了群的陈述。这是数学史上最早、也最漂亮的“把计算问题翻译成结构问题”的范例。对称性太复杂，公式就写不出来。结构决定命运。

---

![Solvable group: chain of normal subgroups with abelian quotients](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_4_solvable_chain.png)

## 下一步

Galois 理论把“多项式有没有根式解”换成了“它的 Galois 群是不是可解”，把“求出所有中间域”换成了“列出 Galois 群的子群格”，把“三等分角、化圆为方”换成了“扩张次数能不能凑出 $2^k$”。它用对称性这把手术刀，把三百年的代数难题解剖得清清楚楚。但这只是抽象代数结构观的起点。下一篇，我们要走出“系数必须能除”的舒适区，进入**模（module）**的世界：把向量空间定义里的“域”换成“环”。这个看似只改了一个词的推广，会同时把 Abel 群分类、$\mathbb{Z}$-模结构、线性算子的 Jordan 标准形、群表示论全部装进同一个框架里。在 Galois 理论里，我们看到群作用在域上；在模理论里，我们会看到环作用在 Abel 群上。“作用”这条线索将贯穿始终。当你发现矩阵对角化和整数分解其实是同一套语言的不同方言时，抽象代数的第二层大门就真正打开了。准备好纸笔，我们下一篇见。
