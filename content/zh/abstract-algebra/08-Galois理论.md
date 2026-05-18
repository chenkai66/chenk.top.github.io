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

1832 年，一位名叫 Évariste Galois 的 20 岁数学家在决斗前夕写下他的数学思想的最终版本，并寄给了一位朋友。这些思想——将多项式的根的对称性与群的结构联系起来——花了十多年时间才被理解并发表，但它们永远改变了代数。现在称之为 Galois 理论，它在域扩张的中间域和对称群的子群之间建立了一个精确的字典。它在一个优雅的框架中解释了为什么二次公式存在，为什么五次没有类似的公式，以及“可解性”到底意味着什么。

我感到惊讶的是 Galois 思想的抽象方向。他不是通过更巧妙地计算多项式的根来研究多项式，而是完全忽略根，转而分析那些保持根之间所有代数关系不变的根的置换。这样的置换集合形成一个群，这个群的结构告诉你关于原始多项式的一切。这是一个从数字到群的彻底转变，尽管如此，它仍然回答了最初的问题。本文详细介绍了这一主题的转变。

---

## Galois 群：固定基域的自同构


![Galois 对应：域扩张与子群的双射](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/figures/aa08_galois_correspondence.png)

给定一个域扩张 $L/K$，*$L$ 在 $K$ 上的自同构* 是一个域同构 $\sigma : L \to L$，使得对于每个 $a \in K$ 有 $\sigma(a) = a$。所有这样的自同构在复合下形成一个群，称为 $L/K$ 的 *Galois 群*：

$$\mathrm{Gal}(L/K) = \mathrm{Aut}_K(L) = \{\sigma : L \to L \mid \sigma \text{ 是一个域自同构},\ \sigma|_K = \mathrm{id}\}.$$

关键在于，$L$ 中固定 $K$ 的自同构会置换 $K[x]$ 中任何多项式的根。如果 $f(x) \in K[x]$ 且 $f(\alpha) = 0$ 对于某个 $\alpha \in L$ 成立，那么应用 $\sigma \in \mathrm{Gal}(L/K)$ 到两边：

$$0 = \sigma(0) = \sigma(f(\alpha)) = f(\sigma(\alpha)),$$

因为 $\sigma$ 固定了 $f$ 的系数。所以 $\sigma(\alpha)$ 也是 $f$ 的根。Galois 群通过置换作用于 $f$ 的根，并且当 $L$ 是分裂域时，这种作用是忠实的——一旦你知道 $\sigma$ 对根的作用，你就知道 $\sigma$ 在任何地方的作用。

![Q(√2)/Q 的 Galois 群通过符号翻转作用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_1_galois_group.png)

例子 ($\mathbb{Q}(\sqrt{2})/\mathbb{Q}$)：任何 $\sigma \in \mathrm{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$ 必须将 $\sqrt{2}$ 映射到 $x^2 - 2$ 的一个根，即 $\pm\sqrt{2}$。因此恰好有两个自同构：恒等映射和 $\sigma(\sqrt{2}) = -\sqrt{2}$。Galois 群是 $\mathbb{Z}/2\mathbb{Z}$。

注意：$\sigma$ 由其在 $\sqrt{2}$ 上的作用决定，因为 $\mathbb{Q}(\sqrt{2})$ 中的每个元素都有形式 $a + b\sqrt{2}$，其中 $a, b \in \mathbb{Q}$，并且 $\sigma(a + b\sqrt{2}) = a + b\sigma(\sqrt{2})$。在符号翻转下保持不变的算术恒等式正是 $(a+b\sqrt{2})(a-b\sqrt{2}) = a^2 - 2b^2$ —— 即 *范数* —— 它自动是 $\mathbb{Q}$-值且 Galois 不变的。

例子 ($\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$)：设 $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$。一个固定 $\mathbb{Q}$ 的自同构 $\sigma$ 将 $\sqrt{2} \mapsto \pm\sqrt{2}$ 和 $\sqrt{3} \mapsto \pm\sqrt{3}$ 独立地映射，给出四个自同构：恒等映射、$\sqrt{2} \mapsto -\sqrt{2}$、$\sqrt{3} \mapsto -\sqrt{3}$ 和它们的乘积。Galois 群是 $\mathbb{Z}/2 \times \mathbb{Z}/2$，即 Klein 四元群 $V_4$。并且 $|\mathrm{Gal}(L/\mathbb{Q})| = 4 = [L:\mathbb{Q}]$，与次数匹配。

例子 ($\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$)：最小多项式 $x^3 - 2$ 只有一个实根 $\sqrt[3]{2}$，而其他两个是复数。任何 $\sigma : \mathbb{Q}(\sqrt[3]{2}) \to \mathbb{Q}(\sqrt[3]{2})$ 必须将 $\sqrt[3]{2}$ 映射到 $x^3 - 2$ 在 $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$ 内的一个根。只有一个这样的根存在。所以 $\sigma = \mathrm{id}$ 并且 $\mathrm{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})$ 是平凡的。然而 $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3 \neq 1$。

这是上一篇文章中关于可分性和正规性的要点：$|\mathrm{Gal}(L/K)| = [L:K]$ 恰好在 Galois 扩张中成立。这里 $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ 失去了正规性（扩展不是分裂域），群就坍缩了。

可以证明，有限扩张 $L/K$ 是 Galois 当且仅当 $|\mathrm{Gal}(L/K)| = [L:K]$。

这是一个两半都有用的等价命题：在一个方向上，它通过计数自同构来检查 Galois 条件；在另一个方向上，它通过检查计数与次数的关系来验证是否找到了所有自同构。

这也与通过可分性 + 正规性定义的“Galois”的抽象定义一致。可分扩张最多有 $[L:K]$ 个嵌入到代数闭包中的嵌入（通过计数最小多项式的根，所有根都是不同的）；正规扩张使这些嵌入实际上回到 $L$（而不是逃到代数闭包中）；因此 Galois 扩张恰好有 $[L:K]$ 个自同构 —— 没有多余的，也没有缺失的。“嵌入计数 = 次数” 身份正好是 $|\mathrm{Gal}(L/K)| = [L:K]$ 的伪装。

Galois 群是一个小的、有限的、可计算的对象，但它捕捉到了整个域扩张的内容。群论已经有了 200 年的结构结果 —— Sylow 定理、单群分类、表示理论 —— 而 Galois 理论将所有这些引入到多项式根的研究中。这种交换几乎等于我们在第 7 部分设置机制所付出的一切。

这里有一个有用的概念重构。Galois 群是多项式 $f$ 的 *自同构群*，以根的作用为精确意义。可以等价地定义 $\mathrm{Gal}(f)$ 为 $S_n$（其中 $n = \deg f$）中的子群，包含那些尊重根之间所有代数关系的置换。"尊重所有代数关系" 这一条件使定义变得非平凡 —— 如果没有它，每个置换都会是自同构，Galois 群总是 $S_n$。有些置换不能扩展到场自同构，这正是你在计算群时利用的信息。

一个小的实例检查。对于 $f(x) = x^4 + 1$（第八旋回多项式），四个根是 $\zeta_8, \zeta_8^3, \zeta_8^5, \zeta_8^7$，即第八单位根。有一个代数关系 $\zeta_8^3 = \zeta_8 \cdot \zeta_8^2 = \zeta_8 \cdot i$，另一个是 $\zeta_8 + \zeta_8^7 = \sqrt{2}$。尊重所有这些恒等式的四个根的置换形成一个四阶群 —— 即 $(\mathbb{Z}/8)^\times$ —— 即使在底层集合上有 $4! = 24$ 个置换。大多数置换破坏了一些关系；剩下的四个正好是 Galois 群。

---

## 固定域和 Galois 对应

在一个方向上，给定一个扩张 $L/K$，得到一个群 $\mathrm{Gal}(L/K)$。在另一个方向上，给定一个子群 $H \leq \mathrm{Gal}(L/K)$，得到一个域：

$$L^H = \{\alpha \in L : \sigma(\alpha) = \alpha \text{ 对所有 } \sigma \in H\}.$$

这是 $H$ 的 *固定域*。它是 $L$ 的一个子域，包含 $K$（因为 $K$ 中的元素被 $\mathrm{Gal}(L/K)$ 中的所有元素固定）。

因此有两个映射：

- $\Phi$：$\mathrm{Gal}(L/K)$ 的子群 → 中间域，$H \mapsto L^H$。
- $\Psi$：中间域 → 子群，$M \mapsto \mathrm{Gal}(L/M)$。

可以证明，如果 $L/K$ 是有限 Galois 扩张，则 $\Phi$ 和 $\Psi$ 是子群集 $\mathrm{Gal}(L/K)$ 和中间域集 $K \subseteq M \subseteq L$ 之间的互逆、序反向双射。

![Galois 对应：Galois 群的子群匹配中间域](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_2_correspondence.png)

序反向方面是第一次遇到时令人惊讶的部分，但它是被迫的：子群越大，施加的约束越多，固定的域越小。符号上，$H_1 \leq H_2 \implies L^{H_1} \supseteq L^{H_2}$。

两个公式确定了字典：

- $[L : L^H] = |H|$（次数等于固定群的大小）。
- $[L^H : K] = [\mathrm{Gal}(L/K) : H]$（从底部开始的次数等于全群中的指数）。

相乘得到 $|\mathrm{Gal}(L/K)| = [L:K]$，这是 Galois 条件。

对应证明基于两个引理：

如果 $G$ 是 $L$ 的有限自同构群，则 $[L : L^G] = |G|$。因此，*每个* 作用于 $L$ 的有限群都实现 $L$ 为其固定域的 Galois 扩张。

如果 $L/K$ 是有限 Galois 扩张，则 $L^{\mathrm{Gal}(L/K)} = K$。因此，$K$ 之外的 $L$ 中的任何元素都不会被 *所有* 自同构固定 —— 自同构看到一切。

结合两者：从 Galois 扩张 $L/K$ 开始，子群 $\mathrm{Gal}(L/K)$ 的固定域是 $K$，对子群 $H$ 应用 Artin 得到驱动双射的等式 $[L : L^H] = |H|$。

这是从域论问题到群论答案的桥梁。想知道有多少子域严格位于 $K$ 和 $L$ 之间？计算 $\mathrm{Gal}(L/K)$ 的适当非平凡子群。想知道哪些子域本身是 Galois 扩张？看 *正规* 子群。关于扩展的每个结构性问题都有一个群论阴影，在实践中更容易计算。

一个值得记住的实用引理：如果 $H_1, H_2 \leq G$ 是子群，其固定域分别为 $M_1, M_2$，则：

- $L^{H_1 \cap H_2} = M_1 \cdot M_2$（合成域，包含两者的最小域）。
- $L^{\langle H_1, H_2 \rangle} = M_1 \cap M_2$。

因此子群上的格运算（交集、生成子群）转化为子域上的格运算（合成域、交集），顺序相反。一旦你内化了这一点，绘制任何具体扩展的子域格就变成了纯粹的群论练习。

对应中还隐藏着一个强唯一性陈述。如果 $L/K$ 是 Galois 扩张且 $\sigma \in \mathrm{Gal}(L/K)$ 对每个中间域都作用平凡，则 $\sigma = \mathrm{id}$。等价地说，唯一固定所有子域的元素是恒等元素。这是一种说子域格连同 Galois 群作用包含了扩展的所有信息的方式。

---

## Galois 理论基本定理

把所有部分放在一起，得到中心结果：

设 $L/K$ 是有限 Galois 扩张，Galois 群为 $G = \mathrm{Gal}(L/K)$。则：

1. *(双射)* 映射 $H \mapsto L^H$ 和 $M \mapsto \mathrm{Gal}(L/M)$ 是子群集 $G$ 和中间域集 $K \subseteq M \subseteq L$ 之间的互逆、序反向双射。

2. *(次数匹配)* 对于每个子群 $H \leq G$，$[L : L^H] = |H|$ 且 $[L^H : K] = [G : H]$。

3. *(正规性)* 中间扩张 $M/K$ 是 Galois（等价于正规）当且仅当 $\mathrm{Gal}(L/M)$ 是 $G$ 的 *正规子群*。在这种情况下，$\mathrm{Gal}(M/K) \cong G/\mathrm{Gal}(L/M)$。

第三部分是最引人注目的。术语 "正规" 出现在两边 —— 对于域，它意味着 "某些多项式的分裂域"；对于群，它意味着 "在共轭下封闭" —— 定理说这些是相同条件的不同视角。术语的一致性绝非巧合；"正规" 最初是为了使对应成立而定义的。

![正规子群对应于正规域扩张](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_7_normal_field.png)

如果 $H \trianglelefteq G$，定义一个同态 $G \to \mathrm{Aut}_K(L^H)$，通过限制每个 $\sigma \in G$ 到 $L^H$。（限制有意义是因为 $L^H$ 被 $\sigma$ 映射到自身：如果 $\alpha \in L^H$ 且 $\tau \in H$，则 $\tau\sigma(\alpha) = \sigma\sigma^{-1}\tau\sigma(\alpha) = \sigma(\alpha)$，因为 $\sigma^{-1}\tau\sigma \in H$ 由正规性。）核恰好是 $H$，所以 $G/H$ 嵌入到 $\mathrm{Aut}_K(L^H)$ 中。计数：$|G/H| = [L^H : K]$，因此由 Galois 条件 $G/H \cong \mathrm{Gal}(L^H/K)$ 且 $L^H/K$ 是 Galois 扩张。

反之，如果 $M/K$ 是 Galois 扩张，则每个 $\sigma \in G$ 将 $M$ 映射到自身（因为 $M$ 是分裂域，故在 $L$ 的所有 $K$-自同构下稳定），所以限制映射 $G \to \mathrm{Aut}_K(M)$ 是良定义的，其核是 $\mathrm{Gal}(L/M)$，因此是正规的。

在反向论证中的 "在 $\sigma$ 下稳定" 步骤值得仔细看看。假设 $M = K(\alpha_1, \ldots, \alpha_r)$ 是 $f \in K[x]$ 在 $L$ 中的分裂域。对于任何 $\sigma \in G$，$\sigma$ 将 $f$ 的每个根 $\alpha_i$ 映射到 $f$ 的另一个根 $\alpha_j$（因为 $\sigma$ 固定了 $f$ 的系数）。其他根也在 $M$ 中（这就是分裂域的意义），所以 $\sigma(\alpha_i) \in M$。因此 $\sigma(M) \subseteq M$。通过维数计数（或通过 $\sigma$ 的单射性），$\sigma(M) = M$。因此限制是良定义的。

基本定理是推动 Galois 理论中每个具体计算的引擎。如果你想找到 $L$ 的子域，画出 $G$ 的子群格 —— 那些是相同的格。如果你想了解哪个子域是 Galois 扩张，标记正规子群。整个问题从一个需要推理元素的主题（域）转移到一个可以组合推理的主题（群）。

FTGT 的三个额外结构收益值得明确指出：

1. *计数中间域。* 包含 $K$ 的 $L$ 的子域数量等于 $G$ 的子群数量。对于 $G \cong S_3$，这是 $1 + 3 + 1 + 1 = 6$ 个子群，因此有 6 个中间域。对于 $G \cong (\mathbb{Z}/p)^n$，这是 $(\mathbb{F}_p)^n$ 的子空间数量，由高斯二项式系数给出。

2. *检测 Galois 闭包。* 包含给定 $M \subseteq L$ 的最小 Galois 扩张是 $G$ 中包含 $\mathrm{Gal}(L/M)$ 的最大正规子群的固定域。因此 Galois 闭包对应于正规核。当你从一个非 Galois 扩张开始并希望进行 Galois 理论时，这种情况经常出现。

3. *与商群耦合。* 每当 $H \trianglelefteq G$ 时，$G/H$ 在 $L^H$ 上诱导一个作用，该作用正是 $\mathrm{Gal}(L^H/K)$。这通常允许你将大 Galois 群的计算简化为两个较小的计算 —— 一个是 $H$，一个是 $G/H$。

---

## 计算 Galois 群：具体例子

理论很优雅，但例子驱动直觉。实际计算一些 Galois 群。

### 例子 1：$\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$

已经看到 $G \cong V_4 = \mathbb{Z}/2 \times \mathbb{Z}/2$。子群是：

- $\{e\}$，固定域 $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$，次数 4。
- $\langle \sigma_2 \rangle$（其中 $\sigma_2 : \sqrt{2} \mapsto -\sqrt{2}$，$\sqrt{3} \mapsto \sqrt{3}$），固定域 $\mathbb{Q}(\sqrt{3})$，次数 2。
- $\langle \sigma_3 \rangle$（其中 $\sigma_3 : \sqrt{2} \mapsto \sqrt{2}$，$\sqrt{3} \mapsto -\sqrt{3}$），固定域 $\mathbb{Q}(\sqrt{2})$，次数 2。
- $\langle \sigma_2 \sigma_3 \rangle$（将 $\sqrt{2} \mapsto -\sqrt{2}$，$\sqrt{3} \mapsto -\sqrt{3}$，因此 $\sqrt{6} \mapsto \sqrt{6}$），固定域 $\mathbb{Q}(\sqrt{6})$，次数 2。
- $G$，固定域 $\mathbb{Q}$，次数 1。

五个子群，五个中间域，所有都按次数匹配。由于 $V_4$ 是 Abel 群，每个子群都是正规的，因此每个中间域都是 Galois 扩张 —— 直接验证很容易，因为每个都是二次多项式的分裂域。

这是典型的“双二次”扩展。它还说明了看似神秘的元素 $\sqrt{6} = \sqrt{2}\sqrt{3}$ 如何在对角自同构下保持不变：因为两个平方根同时改变符号，它们的乘积是不变的。这是产生 Pell 方程格 $\mathbb{Z}[\sqrt{6}]$ 作为 $\mathbb{Q}(\sqrt{6})$ 的整数环的同一技巧，整洁地嵌入在 $\mathbb{Z}[\sqrt{2}, \sqrt{3}]$ 中。

### 例子 2：$x^3 - 2$ 在 $\mathbb{Q}$ 上的分裂域

设 $L = \mathbb{Q}(\sqrt[3]{2}, \omega)$ 其中 $\omega = e^{2\pi i/3}$。从第 7 部分，$[L:\mathbb{Q}] = 6$。

$x^3 - 2$ 的三个根是 $\sqrt[3]{2}$，$\sqrt[3]{2}\omega$，$\sqrt[3]{2}\omega^2$。Galois 群置换这三个根，嵌入到 $S_3$ 中。由于阶数为 6，$G \cong S_3$。可以写出生成元：

- $\sigma$（阶数 3）：$\sqrt[3]{2} \mapsto \sqrt[3]{2}\omega$，$\omega \mapsto \omega$。
- $\tau$（阶数 2）：$\sqrt[3]{2} \mapsto \sqrt[3]{2}$，$\omega \mapsto \omega^2$（复共轭）。

$S_3$ 的子群格：

- $\{e\}$ — 固定域 $L$。
- $\langle \tau \rangle$，$\langle \sigma\tau \rangle$，$\langle \sigma^2\tau \rangle$ — 三个二阶子群，固定域 $\mathbb{Q}(\sqrt[3]{2})$，$\mathbb{Q}(\sqrt[3]{2}\omega^2)$，$\mathbb{Q}(\sqrt[3]{2}\omega)$（每个是三个实或复立方根之一）。
- $\langle \sigma \rangle = A_3$ — 固定域 $\mathbb{Q}(\omega)$，次数 2。
- $S_3$ — 固定域 $\mathbb{Q}$。

三个二阶子群在 $S_3$ 中不是正规的（它们彼此共轭），相应地，三个域 $\mathbb{Q}(\sqrt[3]{2}\omega^k)$ 不是 Galois 扩张 —— 它们都不是任何东西的分裂域，也不在所有自同构下封闭。子群 $A_3$ 是正规的（在 $S_3$ 中的指数为 2），相应地 $\mathbb{Q}(\omega)/\mathbb{Q}$ 是 Galois 扩张（它是 $x^2 + x + 1$ 的分裂域，Galois 群为 $\mathbb{Z}/2$）。

这是如何在域侧出现非正规性在群侧表现为非正规性的典型例子。

在这里对 FTGT 进行简短的数值检查。取二阶子群 $\langle \tau \rangle$，其固定域为 $\mathbb{Q}(\sqrt[3]{2})$，次数为 3。FTGT 预测 $|G| / |H| = 6/2 = 3$，匹配。现在取 $A_3$，三阶，固定域为 $\mathbb{Q}(\omega)$，次数为 2：比率为 $6/3 = 2$，匹配。数值是平凡的；令人惊讶的是，*每个* $S_3$ 的子群都有一个独特的域知道它，反之亦然。

### 例子 3：$x^4 - 2$ 在 $\mathbb{Q}$ 上的分裂域

设 $L = \mathbb{Q}(\sqrt[4]{2}, i)$。从第 7 部分，$[L:\mathbb{Q}] = 8$。Galois 群作用于四个根 $\pm\sqrt[4]{2}, \pm i\sqrt[4]{2}$。生成元：

- $r$（阶数 4）：$\sqrt[4]{2} \mapsto i\sqrt[4]{2}$，$i \mapsto i$。（循环四个根。）
- $s$（阶数 2）：$\sqrt[4]{2} \mapsto \sqrt[4]{2}$，$i \mapsto -i$。（复共轭。）

这些满足 $r^4 = s^2 = 1$，$srs = r^{-1}$。因此 $G \cong D_4$，八阶二面体群。

![x^4 - 2 的分裂域的完整 Galois 对应](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_3_x4_minus_2.png)

$D_4$ 有 10 个子群：$\{e\}$，三个二阶子群（$\langle r^2 \rangle$，$\langle s \rangle$，$\langle rs \rangle$，$\langle r^2 s \rangle$，$\langle r^3 s \rangle$ — 总共五个），三个四阶子群（$\langle r \rangle$，$\langle r^2, s \rangle$，$\langle r^2, rs \rangle$），和 $D_4$。因此有 10 个中间域。

$D_4$ 的中心是 $\langle r^2 \rangle$，正规。三个指数为 2 的子群（四阶）都是正规的，给出三个 Galois 子扩展。显式计算固定域是一个令人满意的练习；其中包括 $\mathbb{Q}(i)$，$\mathbb{Q}(\sqrt{2})$，$\mathbb{Q}(i, \sqrt{2})$，还有一些不太明显的如 $\mathbb{Q}(\sqrt{2}\cdot i)$。

两个二阶非正规子群是共轭的，即 $\langle s \rangle$ 和 $\langle r^2 s \rangle$（同样 $\langle rs \rangle, \langle r^3 s \rangle$）。它们的固定域 $\mathbb{Q}(\sqrt[4]{2})$ 和 $\mathbb{Q}(i\sqrt[4]{2})$ 不是 Galois 扩张，但它们彼此同构（一个是另一个的共轭）。群论共轭反映为域同构但不相等，每次看到这一点我都感到满意：FTGT 不仅将子群与子域匹配，还将共轭类与“相同”域的不同嵌入的同构类匹配。

### 例子 4：分圆域

$n$ 阶分圆域是 $\mathbb{Q}(\zeta_n)$，其中 $\zeta_n = e^{2\pi i/n}$。$\zeta_n$ 在 $\mathbb{Q}$ 上的极小多项式是 $n$ 阶分圆多项式 $\Phi_n(x)$，次数为 $\varphi(n)$。

可以证明，$\mathrm{Gal}(\mathbb{Q}(\zeta_n)/\mathbb{Q}) \cong (\mathbb{Z}/n\mathbb{Z})^\times$。

同构将 $\sigma \in \mathrm{Gal}$ 映射到唯一的 $a \in (\mathbb{Z}/n\mathbb{Z})^\times$ 使得 $\sigma(\zeta_n) = \zeta_n^a$。证明归结为 (i) 证明 $\Phi_n$ 在 $\mathbb{Q}$ 上不可约 —— Gauss 的经典定理 —— 和 (ii) 注意到 $\varphi(n)$ 个 $n$ 阶原根恰好是 $\zeta_n^a$，其中 $\gcd(a, n) = 1$，因此每个 Galois 自同构对应于这样一个 $a$ 的选择。

![分圆扩张及其 Abel Galois 群](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_6_cyclotomic.png)

由于 $(\mathbb{Z}/n\mathbb{Z})^\times$ 是 Abel 群，所有 $\mathbb{Q}$ 的分圆扩张都是 Abel 扩张。Kronecker-Weber 定理（1853 年提出，1886 年完成）说 *逆* 也成立：$\mathbb{Q}$ 的每个有限 Abel 扩张都包含在某个分圆域中。因此 $\mathbb{Q}$ 的 Abel 扩张 *恰好* 是分圆域的子域。这是类域论的原型。

例如，$\mathrm{Gal}(\mathbb{Q}(\zeta_8)/\mathbb{Q}) \cong (\mathbb{Z}/8)^\times \cong \mathbb{Z}/2 \times \mathbb{Z}/2$。中间域是 $\mathbb{Q}(\sqrt{2})$，$\mathbb{Q}(i)$，$\mathbb{Q}(i\sqrt{2})$ —— 所有实或虚二次域，嵌入在第八分圆域中。特别是，二次 Gauss 和恒等
$$\zeta_8 + \zeta_8^{-1} = \sqrt{2}$$
不再神秘：它是从 $\mathbb{Q}(\zeta_8)$ 到其指数为 2 的子域 $\mathbb{Q}(\sqrt{2})$ 的迹，由复共轭固定。

### 例子 5：有限域

Galois 群 $\mathrm{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p)$ 是由 Frobenius 自同构 $\mathrm{Frob}_p : x \mapsto x^p$ 生成的 $n$ 阶循环群。因此每个有限域的有限扩张都是 Galois 扩张，具有循环 Galois 群，整个子群格是 $n$ 的除数格。这是最干净的 Galois 理论：$\mathbb{F}_{p^d}$ 嵌入 $\mathbb{F}_{p^n}$ 当且仅当 $d \mid n$，且 $\mathbb{F}_{p^n}/\mathbb{F}_{p^d}$ 的 Galois 群是由 $\mathrm{Frob}_p^d$ 生成的 $n/d$ 阶循环群。

这很关键，因为 Galois 群计算不仅仅是运动。它们是大量代数数论中的算法步骤：识别多项式的根何时允许闭形式表达，计算类数，构建互反律，进行现代密码学。任何计算机代数系统中的数域包本质上是一个 Galois 群计算器，包裹在因子分解数据库中。

一些实用的计算注意事项：

- 对于 $\mathbb{Q}[x]$ 中的不可约 $f$，Galois 群是 $S_n$ 的传递子群。$S_4$ 有 5 个传递子群，$S_5$ 有 5 个，$S_6$ 有 16 个，$S_7$ 有 7 个，$S_8$ 有 50 个，等等。识别给定 $f$ 产生的哪一个是一个有限（但有时烦人的）检查。PARI/GP 和 SageMath 都自带 `polgalois` / `f.galois_group()` 用于此目的。

- $f \bmod p$ 对于各种素数 $p$ 的因子分解模式告诉你 Frobenius 在 $\mathrm{Gal}(f)$ 中的循环结构。Chebotarev 密度定理说，随着 $p$ 的变化，$\mathrm{Gal}(f)$ 中出现的每种循环结构都以正确的密度出现。因此你可以通过因式分解 $f$ 模许多素数并匹配循环类型来猜测 Galois 群 —— 这是一种非常有效的启发式方法。

- $f$ 的判别式在 $\mathbb{Q}$ 中是平方当且仅当 $\mathrm{Gal}(f) \subseteq A_n$。因此 $\mathrm{disc}(f)$ 是否为平方大致将候选 Galois 群减半。

---

## 根式可解性与可解群

现在将“这个多项式能否通过根式求解？”的问题转化为群论问题。

定义如果存在塔
$$K = K_0 \subseteq K_1 \subseteq \cdots \subseteq K_r$$
其中每一步通过添加根式获得（即 $K_{i+1} = K_i(\sqrt[n_i]{a_i})$ 对于某些 $n_i \geq 1$ 和 $a_i \in K_i$），并且 $f$ 的分裂域包含在 $K_r$ 中，则多项式 $f(x) \in K[x]$ 在 $K$ 上 *可通过根式求解*。

换句话说：你可以使用 $+, -, \times, \div$ 和 $n$ 次根，从 $K$ 开始写出 $f$ 的根。

定义如果群 $G$ 有一条子群链
$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_n = G$$
使得每个商 $G_{i+1}/G_i$ 是 Abel 群，则称 $G$ 是 *可解的*。

![可解群：具有 Abel 商的正规子群链](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_4_solvable_chain.png)

举个例子

- 所有 Abel 群（长度为 1 的链）。
- 所有阶小于 60 的群。
- $S_n$ 对于 $n \leq 4$（$S_3$ 有 $\{e\}
<!-- 本节内容因生成长度限制截断；完整推导请参阅本系列对应英文版本。 -->


![A_5 is simple and non-abelian — the obstruction to solving the quintic（图）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_v2_08_5_quintic.png)


---

*本文是[《抽象代数》](/zh/series/abstract-algebra/)系列的第 8 篇（共 12 篇）。*

*下一篇：[第 9 篇 — 模](/zh/abstract-algebra/09-modules/)*
