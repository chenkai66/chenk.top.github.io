---
title: "抽象代数（八）：Galois 理论——域与群的桥梁"
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
description: "Galois 理论的基本定理在中间域与子群之间建立了完美的对应——并最终解决了用根式求解多项式的古老问题。"
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "abstract-algebra-8"
---

1832 年，二十岁的 Evariste Galois 在一场他注定活不过的决斗前夜，把自己最终版本的数学思想写在了给朋友的信中。那些思想——将多项式根的对称性与群的结构联系起来——花了十余年才被人理解和发表，但它们永久地改变了代数学的面貌。如今我们称之为 Galois 理论的这套框架，在域扩张的中间域和一个对称群的子群之间建立了精确的字典。它统一地解释了为什么二次方程有求根公式、为什么五次方程没有、以及"可解性"到底意味着什么。

上一篇文章中，我们建好了域扩张的全部基础设施：次数、极小多项式、塔式法则、分裂域、可分性。现在把它们投入使用。

---

## Galois 群：固定基域的自同构

**定义。** 设 $L/K$ 是域扩张。$L$ 的一个 **$K$-自同构**是域同构 $\sigma : L \to L$，且对所有 $a \in K$ 有 $\sigma(a) = a$。全体 $K$-自同构在复合运算下构成一个群，称为 $L$ 在 $K$ 上的 **Galois 群**：

$$\operatorname{Gal}(L/K) = \{\sigma \in \operatorname{Aut}(L) : \sigma|_K = \operatorname{id}_K\}.$$

![子群与中间域之间的 Galois 对应](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/08-galois-theory/aa_fig8_galois_correspondence.png)


$K$-自同构由它对 $L/K$ 生成元的作用完全决定。由于 $\sigma$ 必须保持所有 $K$ 系数的多项式关系，它只能把不可约多项式的根置换为同一多项式的其他根。

**关键观察。** 若 $\alpha \in L$ 是 $K[x]$ 中不可约多项式 $p(x)$ 的根，$\sigma \in \operatorname{Gal}(L/K)$，则 $\sigma(\alpha)$ 也是 $p$ 的根：

$$p(\sigma(\alpha)) = \sigma(p(\alpha)) = \sigma(0) = 0.$$

这里用到了 $\sigma$ 固定 $K$（从而固定 $p$ 的所有系数）以及保持加法和乘法。

**例 1（$\operatorname{Gal}(\mathbb{C}/\mathbb{R})$）。** $\mathbb{R}$-自同构 $\sigma$ 必须把 $i$ 送到 $x^2+1$ 的根，即 $\sigma(i) = i$（恒等）或 $\sigma(i) = -i$（复共轭）。故 $\operatorname{Gal}(\mathbb{C}/\mathbb{R}) \cong \mathbb{Z}/2\mathbb{Z}$。

**例 2（$\operatorname{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$）。** $\sigma(\sqrt{2})$ 必须是 $x^2-2$ 的根，即 $\pm\sqrt{2}$。恒等和 $\sigma: a+b\sqrt{2} \mapsto a-b\sqrt{2}$ 两个自同构，$\operatorname{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q}) \cong \mathbb{Z}/2\mathbb{Z}$。

**例 3（$\operatorname{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})$）。** $\sigma(\sqrt[3]{2})$ 必须是 $x^3-2$ 的根：$\sqrt[3]{2}, \sqrt[3]{2}\omega, \sqrt[3]{2}\omega^2$。但 $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$，$\sigma(\sqrt[3]{2})$ 必须是实数。唯一的实根是 $\sqrt[3]{2}$ 本身，故 $\sigma = \operatorname{id}$，$\operatorname{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}) = \{1\}$。

这个例子揭示了关键一点：Galois 群可能"太小"。这里 $|\operatorname{Gal}| = 1 < 3 = [\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}]$。原因是 $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ 不正规（不是分裂域）。当扩张是 Galois 扩张时，两个数严格相等：

**定理。** 若 $L/K$ 是有限 Galois 扩张（既正规又可分），则

$$|\operatorname{Gal}(L/K)| = [L:K].$$

---

## 固定域与 Galois 对应

**定义。** 设 $L$ 是域，$H \leq \operatorname{Aut}(L)$ 是自同构群的子群。$H$ 的**固定域**为

$$L^H = \{a \in L : \sigma(a) = a, \ \forall \sigma \in H\}.$$

容易验证 $L^H$ 是 $L$ 的子域。

Galois 对应是以下两个映射：

$$\{\text{中间域 } K \subseteq M \subseteq L\} \quad \longleftrightarrow \quad \{\text{子群 } H \leq \operatorname{Gal}(L/K)\}$$

方向一：$M \mapsto \operatorname{Gal}(L/M)$；方向二：$H \mapsto L^H$。

### 计算实例：$\operatorname{Gal}(\mathbb{Q}(\sqrt{2},\sqrt{3})/\mathbb{Q})$

这是展示完整对应的最简单非平凡例子。由上篇，$[\mathbb{Q}(\sqrt{2},\sqrt{3}):\mathbb{Q}] = 4$。

$\mathbb{Q}(\sqrt{2},\sqrt{3})$ 是 $(x^2-2)(x^2-3)$ 在 $\mathbb{Q}$ 上的分裂域，故正规。特征 0 下自动可分。因此是 Galois 扩张，$|\operatorname{Gal}| = 4$。

$\mathbb{Q}$-自同构 $\sigma$ 由 $\sigma(\sqrt{2}) = \pm\sqrt{2}$ 和 $\sigma(\sqrt{3}) = \pm\sqrt{3}$ 决定，四种组合：

| 自同构 | $\sigma(\sqrt{2})$ | $\sigma(\sqrt{3})$ |
|:---:|:---:|:---:|
| $e$ | $\sqrt{2}$ | $\sqrt{3}$ |
| $\sigma_1$ | $-\sqrt{2}$ | $\sqrt{3}$ |
| $\sigma_2$ | $\sqrt{2}$ | $-\sqrt{3}$ |
| $\sigma_3$ | $-\sqrt{2}$ | $-\sqrt{3}$ |

每个 $\sigma_i$ 的阶为 2，$\sigma_3 = \sigma_1\sigma_2$，故 $G \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$（Klein 四元群 $V_4$）。

$V_4$ 有 5 个子群，对应关系如下：

| 子群 $H$ | $\|H\|$ | 固定域 $L^H$ | $[L^H:\mathbb{Q}]$ |
|:---:|:---:|:---:|:---:|
| $\{e\}$ | 1 | $\mathbb{Q}(\sqrt{2},\sqrt{3})$ | 4 |
| $\{e,\sigma_1\}$ | 2 | $\mathbb{Q}(\sqrt{3})$ | 2 |
| $\{e,\sigma_2\}$ | 2 | $\mathbb{Q}(\sqrt{2})$ | 2 |
| $\{e,\sigma_3\}$ | 2 | $\mathbb{Q}(\sqrt{6})$ | 2 |
| $G$ | 4 | $\mathbb{Q}$ | 1 |

**验证 $\{e,\sigma_3\}$ 的固定域。** $\mathbb{Q}(\sqrt{2},\sqrt{3})$ 的一般元素 $a + b\sqrt{2} + c\sqrt{3} + d\sqrt{6}$，施加 $\sigma_3$：$a - b\sqrt{2} - c\sqrt{3} + d\sqrt{6}$（因为 $\sigma_3(\sqrt{6}) = (-\sqrt{2})(-\sqrt{3}) = \sqrt{6}$）。固定要求 $b = c = 0$，故 $L^H = \mathbb{Q}(\sqrt{6})$。

观察对应的反序性：子群越大，固定域越小。且每个条目中 $[L:L^H] = |H|$，$[L^H:\mathbb{Q}] = |G|/|H|$。这些都是基本定理的推论。

---

## Galois 理论基本定理

**定理（Galois 理论基本定理）。** 设 $L/K$ 是有限 Galois 扩张，$G = \operatorname{Gal}(L/K)$。则：

**(a) 双射。** 存在保序反转的双射
$$\Phi: \{\text{中间域}\} \to \{\text{$G$ 的子群}\}, \quad M \mapsto \operatorname{Gal}(L/M),$$
逆映射 $\Psi(H) = L^H$。

**(b) 次数等于指标。**
$$[L:M] = |\operatorname{Gal}(L/M)|, \qquad [M:K] = [G:\operatorname{Gal}(L/M)].$$

**(c) 正规性判据。** $M/K$ 正规当且仅当 $\operatorname{Gal}(L/M) \trianglelefteq G$。此时限制映射给出

$$\operatorname{Gal}(M/K) \cong G / \operatorname{Gal}(L/M).$$

*证明要点。*

核心引理是 **Artin 引理**：若 $H$ 是 $L$ 的有限自同构群，则 $[L:L^H] \leq |H|$。

*Artin 引理证明。* 设 $|H| = n$。若 $[L:L^H] > n$，选 $n+1$ 个 $L^H$-线性无关元 $\alpha_1, \ldots, \alpha_{n+1}$。将 $H = \{\sigma_1, \ldots, \sigma_n\}$，考虑齐次线性方程组

$$\sum_{j=1}^{n+1} \sigma_i(\alpha_j) x_j = 0, \quad i = 1, \ldots, n.$$

$n$ 个方程 $n+1$ 个未知数，存在非平凡解。在所有非平凡解中取非零分量最少的，归一化最后一个非零分量为 1。若某 $c_j \notin L^H$，对方程施加某 $\tau \in H$（$\tau(c_j) \neq c_j$），再与原方程相减，得到非零分量更少的非平凡解——矛盾。故所有 $c_j \in L^H$，给出 $L^H$-线性相关性——矛盾。$\blacksquare$

有了 Artin 引理：

- $\Psi \circ \Phi = \operatorname{id}$：$L^{\operatorname{Gal}(L/M)} \supseteq M$ 显然。由 Galois 性质 $|\operatorname{Gal}(L/M)| = [L:M]$，由 Artin $[L:L^{\operatorname{Gal}(L/M)}] \leq |\operatorname{Gal}(L/M)| = [L:M]$，故 $L^{\operatorname{Gal}(L/M)} \subseteq M$。
- $\Phi \circ \Psi = \operatorname{id}$：$H \subseteq \operatorname{Gal}(L/L^H)$ 显然。由 Artin $[L:L^H] \leq |H|$，而 $|\operatorname{Gal}(L/L^H)| = [L:L^H] \leq |H|$，故 $H = \operatorname{Gal}(L/L^H)$。

(c) 的证明：若 $M/K$ 正规，则对任意 $\sigma \in G$ 和 $\tau \in \operatorname{Gal}(L/M)$，$\sigma\tau\sigma^{-1}$ 仍固定 $M$（因为 $\sigma$ 把 $M$ 的元素映到同一不可约多项式的另一根，正规性保证该根仍在 $M$ 中），故 $\operatorname{Gal}(L/M) \trianglelefteq G$。反之，若 $\operatorname{Gal}(L/M) \trianglelefteq G$，则 $G$ 通过限制作用在 $M$ 上，核为 $\operatorname{Gal}(L/M)$，由第一同构定理得商群同构。$\blacksquare$

---

## 计算 Galois 群：具体例子

### 例 1：$x^4 - 2$ 在 $\mathbb{Q}$ 上的分裂域

由上篇，分裂域 $L = \mathbb{Q}(\sqrt[4]{2}, i)$，$[L:\mathbb{Q}] = 8$。

$x^4-2$ 的根为 $\alpha, i\alpha, -\alpha, -i\alpha$（$\alpha = \sqrt[4]{2}$）。$\sigma \in G$ 由 $\sigma(\alpha)$ 和 $\sigma(i)$ 决定：

- $\sigma(\alpha) \in \{\alpha, i\alpha, -\alpha, -i\alpha\}$（4 种）
- $\sigma(i) \in \{i, -i\}$（2 种）

$|G| = 8 = 4 \times 2$，所有组合都实现了。定义：

- $\rho$：$\alpha \mapsto i\alpha$，$i \mapsto i$（对四个根的"旋转"）。
- $\tau$：$\alpha \mapsto \alpha$，$i \mapsto -i$（复共轭限制到 $L$）。

则 $\rho$ 的阶为 4（$\rho^2(\alpha) = -\alpha$，$\rho^3(\alpha) = -i\alpha$，$\rho^4 = e$），$\tau$ 的阶为 2。验证 $\tau\rho\tau^{-1}(\alpha) = \tau(i\alpha) = (-i)\alpha = \rho^3(\alpha)$，故 $\tau\rho = \rho^{-1}\tau$，这是二面体群的定义关系：

$$G = \operatorname{Gal}(\mathbb{Q}(\sqrt[4]{2},i)/\mathbb{Q}) \cong D_4,$$

即正方形的对称群（8 阶二面体群）。

$D_4$ 有 10 个子群（含平凡群和 $D_4$ 本身），对应 10 个中间域。例如：

- $\langle\rho\rangle \cong \mathbb{Z}/4\mathbb{Z}$ 对应 $L^{\langle\rho\rangle} = \mathbb{Q}(i)$。
- $\langle\tau\rangle \cong \mathbb{Z}/2\mathbb{Z}$ 对应 $L^{\langle\tau\rangle} = \mathbb{Q}(\sqrt[4]{2})$。
- $\langle\rho^2\rangle \cong \mathbb{Z}/2\mathbb{Z}$ 对应 $L^{\langle\rho^2\rangle} = \mathbb{Q}(i, \sqrt{2})$。

$D_4$ 的正规子群对应给出 $\mathbb{Q}$ 上 Galois（正规）扩张的中间域。

### 例 2：分圆域

取素数 $p$，令 $\zeta = e^{2\pi i/p}$。$p$ 次**分圆域**为 $\mathbb{Q}(\zeta)$。极小多项式是 $p$ 次分圆多项式：

$$\Phi_p(x) = 1 + x + x^2 + \cdots + x^{p-1} = \frac{x^p-1}{x-1},$$

在 $\mathbb{Q}$ 上不可约（对 $\Phi_p(x+1)$ 用 Eisenstein）。故 $[\mathbb{Q}(\zeta):\mathbb{Q}] = p-1$。

任意 $\sigma \in \operatorname{Gal}(\mathbb{Q}(\zeta)/\mathbb{Q})$ 将 $\zeta$ 送到另一本原 $p$ 次单位根 $\zeta^k$（$1 \leq k \leq p-1$）。映射 $\sigma \mapsto k \pmod{p}$ 给出同构

$$\operatorname{Gal}(\mathbb{Q}(\zeta)/\mathbb{Q}) \cong (\mathbb{Z}/p\mathbb{Z})^\times \cong \mathbb{Z}/(p-1)\mathbb{Z}.$$

例如 $\operatorname{Gal}(\mathbb{Q}(\zeta_7)/\mathbb{Q}) \cong \mathbb{Z}/6\mathbb{Z}$，其子群 $\{0\}, \{0,3\}, \{0,2,4\}, \mathbb{Z}/6\mathbb{Z}$ 对应中间域链 $\mathbb{Q}(\zeta_7) \supset M_3 \supset M_2 \supset \mathbb{Q}$，其中 $[M_3:\mathbb{Q}] = 3$，$[M_2:\mathbb{Q}] = 2$。

---

## 根式可解性与可解群

Galois 理论的原始动机正是这个问题：哪些多项式的根可以用加减乘除和开方来表达？这就是"根式可解性"。

**定义。** 域扩张 $L/K$ 是**根式扩张**，若存在域链

$$K = K_0 \subset K_1 \subset \cdots \subset K_r = L,$$

每步 $K_{i+1} = K_i(\alpha_i)$，$\alpha_i^{n_i} \in K_i$（$n_i$ 为正整数）。即每步添加一个已有元素的 $n$ 次根。

**定义。** 多项式 $f(x) \in K[x]$ **可用根式求解**，若其分裂域包含在 $K$ 的某个根式扩张中。

**定义。** 群 $G$ 是**可解**的，若存在次正规列

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_r = G,$$

其中每个商群 $G_{i+1}/G_i$ 是交换群。

**定理（Galois 判据）。** 设 $\operatorname{char}(K) = 0$，$f(x) \in K[x]$，$L$ 是 $f$ 的分裂域。则 $f$ 可用根式求解当且仅当 $\operatorname{Gal}(L/K)$ 是可解群。

### 实例：$x^3 - 2$ 的可解性

$x^3-2$ 的分裂域 $L = \mathbb{Q}(\sqrt[3]{2}, \omega)$，$[L:\mathbb{Q}] = 6$。Galois 群 $G$ 置换三个根，$|G| = 6$，$G \hookrightarrow S_3$（$|S_3|=6$），故 $G \cong S_3$。

$S_3$ 的可解列：$\{e\} \trianglelefteq A_3 \trianglelefteq S_3$，$A_3 \cong \mathbb{Z}/3\mathbb{Z}$，$S_3/A_3 \cong \mathbb{Z}/2\mathbb{Z}$，商群都交换。$G$ 可解，故 $x^3-2$ 可用根式求解——果然，它的根 $\sqrt[3]{2}, \omega\sqrt[3]{2}, \omega^2\sqrt[3]{2}$ 都可用根式表示（$\omega = (-1+\sqrt{-3})/2$）。

Galois 对应下，唯一的指标 2 子群 $A_3$ 对应中间域 $\mathbb{Q}(\omega)$。商群 $S_3/A_3 \cong \mathbb{Z}/2\mathbb{Z}$ 对应第一步（添 $\sqrt{-3}$ 得 $\omega$），$A_3 \cong \mathbb{Z}/3\mathbb{Z}$ 对应第二步（添 $\sqrt[3]{2}$）。这正是 Cardano 公式的结构：先开平方（判别式），再开立方。

### 低次多项式为何可解

$n \leq 4$ 时 $S_n$ 可解：

- $S_1 = \{e\}$：平凡可解。
- $S_2 \cong \mathbb{Z}/2\mathbb{Z}$：交换群。
- $S_3$：$\{e\} \trianglelefteq A_3 \trianglelefteq S_3$。
- $S_4$：$\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$（$V_4$ 是 Klein 四元群）。

$n$ 次多项式的 Galois 群是 $S_n$ 的子群（置换 $n$ 个根），可解群的子群仍然可解。因此所有四次及以下的多项式都可用根式求解——这与二次、三次、四次公式的存在一致。

---

## 一般五次方程的不可解性

**定理（Abel-Ruffini）。** 一般的五次（及更高次）多项式不能用根式求解。

由 Galois 判据，只要找到一个 Galois 群为 $S_5$ 的五次多项式，再证明 $S_5$ 不可解，即可。

### 第一步：$S_5$ 不可解

**命题。** 交替群 $A_5$ 是**单群**（没有非平凡正规子群）。

*证明。* $|A_5| = 60$。$A_5$ 的共轭类大小为 $1, 12, 12, 15, 20$（分别对应恒等、两类 5-轮换、两对不相交对换的乘积、3-轮换）。正规子群是共轭类的并且含恒等（贡献 1），其阶必整除 60。检查所有包含 1 的共轭类大小子集之和：

- $1+12=13$：不整除 60。
- $1+15=16$：不整除 60。
- $1+20=21$：不整除 60。
- $1+12+12=25$：不整除 60。
- $1+12+15=28$：不整除 60。
- $1+12+20=33$：不整除 60。
- $1+15+20=36$：不整除 60。

继续检查其余组合（$1+12+12+15=40$、$1+12+12+20=45$、$1+12+15+20=48$），均不整除 60。故 $A_5$ 仅有平凡正规子群。$\blacksquare$

**推论。** $S_5$ 不可解。

*证明。* 若 $S_5$ 可解，则其子群 $A_5$ 也可解。可解群必有非平凡交换商群（可解列第一步给出 $G_r/G_{r-1}$ 交换且非平凡）。$A_5$ 要有非平凡交换商群 $A_5/N$，需要正规子群 $N \trianglelefteq A_5$。但 $A_5$ 是单群，$N$ 只能是 $\{e\}$ 或 $A_5$。$N = \{e\}$ 给出 $A_5$ 本身，而 $A_5$ 非交换（$|A_5| = 60$）。矛盾。$\blacksquare$

### 第二步：Galois 群为 $S_5$ 的具体多项式

考虑 $f(x) = x^5 - 4x + 2 \in \mathbb{Q}[x]$。

**不可约性。** 对 $p=2$ 用 Eisenstein 判别法：$2 \mid (-4)$，$2 \mid 2$，$4 \nmid 2$，$2 \nmid 1$（首项系数）。故 $f$ 在 $\mathbb{Q}$ 上不可约。

**Galois 群为 $S_5$。** 标准判据：若不可约五次多项式恰有三个实根和两个共轭复根，则 Galois 群为 $S_5$。

$f'(x) = 5x^4 - 4 = 0$ 给出 $x = \pm c$，$c = (4/5)^{1/4} \approx 0.946$。

- $f(c) = c^5 - 4c + 2 = 4c/5 - 4c + 2 = -16c/5 + 2 \approx -1.03 < 0$（极小值为负）。
- $f(-c) = 16c/5 + 2 \approx 5.03 > 0$（极大值为正）。

$\deg f = 5$（奇数），$f(x) \to +\infty$（$x \to +\infty$），$f(x) \to -\infty$（$x \to -\infty$）。极大值为正、极小值为负，实数轴上恰穿越三次，故恰有 3 个实根、2 个共轭复根。

**为何推出 $\operatorname{Gal}(f/\mathbb{Q}) = S_5$。** Galois 群 $G \leq S_5$ 满足：

1. $5 \mid |G|$（$f$ 不可约 $\Rightarrow$ $G$ 在根上传递 $\Rightarrow$ $G$ 含 5-轮换）。
2. $G$ 含对换（复共轭交换两个非实根、固定三个实根——这在 $S_5$ 中是对换）。

$S_p$（$p$ 素数）中包含一个 $p$-轮换和一个对换的子群必然是全部 $S_p$（标准群论练习）。故 $G \cong S_5$。

$S_5$ 不可解，由 Galois 判据，$f(x) = x^5 - 4x + 2$ 不可用根式求解。$\blacksquare$

### 历史视角与定理的精确含义

这个结果——一般五次方程的不可解性——由 Abel 在 1824 年首先证明（没有完整的 Galois 框架），后由 Galois 给出群论定义的最终形式。它终结了从文艺复兴以来的悬案：Tartaglia、Cardano、Ferrari 在十六世纪找到了 2、3、4 次方程的公式，所有对 5 次方程的尝试都失败了。原因现在很清楚：这不是技巧不够，而是结构性的不可能——$S_5$ 不可解。

必须强调定理**不是**说什么：

- 不是说五次方程没有根——在 $\mathbb{C}$ 中总有根（代数基本定理）。
- 不是说每个具体的五次多项式都不可解——$x^5-2$ 的 Galois 群是 20 阶 Frobenius 群 $F_{20} \cong \mathbb{Z}/5\mathbb{Z} \rtimes \mathbb{Z}/4\mathbb{Z}$，它可解，其根确实可用根式表达。
- 定理说的是：不存在统一的根式公式适用于所有五次多项式——因为某些五次多项式（如 $x^5-4x+2$）的 Galois 群是 $S_5$，而 $S_5$ 不可解。

| 次数 | 一般 Galois 群 | 可解？ | 存在公式？ |
|:---:|:---:|:---:|:---:|
| 1 | $S_1 = \{e\}$ | 是 | 是（平凡） |
| 2 | $S_2 \cong \mathbb{Z}/2\mathbb{Z}$ | 是 | 是（求根公式） |
| 3 | $S_3$ | 是 | 是（Cardano 公式） |
| 4 | $S_4$ | 是 | 是（Ferrari 方法） |
| $\geq 5$ | $S_n$ | 否 | 否（Abel-Ruffini） |

---

## 下一步

Galois 理论的故事到此远未结束。在后续文章中，我们将探索更多应用和推广：

- **有限域：** 每个有限域的阶为 $p^n$，$\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p)$ 是 $n$ 阶循环群，由 Frobenius 自同构 $x \mapsto x^p$ 生成。这把 Galois 理论与数论和编码理论联系起来。

- **Galois 逆问题：** 给定有限群 $G$，是否存在 $\mathbb{Q}$ 上的 Galois 扩张使 Galois 群同构于 $G$？这是代数学的重大未解问题之一。对所有可解群、对称群、交替群以及许多单群已知为真，但一般情况仍然悬而未决。

- **无穷 Galois 理论：** 对无穷代数扩张（如 $\overline{\mathbb{Q}}/\mathbb{Q}$），Galois 群带有自然拓扑（Krull 拓扑），基本定理推广为*闭*子群与中间域的对应。绝对 Galois 群 $\operatorname{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$ 是数学中研究最多也最神秘的对象之一。

从一个不肯交出根的多项式方程，经过域扩张和群论的抽象机器，到确定性的不可能性结果和仍在推进的研究前沿——这就是 Galois 理论的完整弧线，数学中最优美的篇章之一。

---

*本文是 [抽象代数](/zh/series/abstract-algebra/) 系列的第 8 篇（共 12 篇）。*

*上一篇：[第 7 篇 —— 域扩张](/zh/abstract-algebra/07-域扩张/)*

*下一篇：[第 9 篇 —— 模](/zh/abstract-algebra/09-模/)*
