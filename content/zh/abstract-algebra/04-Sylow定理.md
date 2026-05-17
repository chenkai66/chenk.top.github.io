---
title: "抽象代数（四）：Sylow 定理——有限群的精细解剖"
date: 2021-09-07 09:00:00
tags:
  - abstract-algebra
  - group-theory
  - sylow-theorems
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: zh
mathjax: true
description: "Sylow 定理为我们提供了系统寻找和计数素数幂阶子群的方法——这是分类有限群最锋利的工具。"
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "abstract-algebra-4"
---

有限群的结构理论建立在一个基本观察之上：$|G|$ 的素因数分解从根本上约束了 $G$ 可能拥有的子群。Lagrange 定理告诉我们子群的阶必须整除 $|G|$，但反过来不成立——$|G|$ 的因数不一定对应一个子群。Sylow 定理大幅收紧了这个约束：对于整除 $|G|$ 的每个素数幂，不仅相应阶数的子群一定存在，而且它们的个数和相互关系都受到严格限制。

Ludwig Sylow 于 1872 年证明了这组定理，一百五十多年后它仍然是分析有限群最有效的工具——没有之一。

---

## 有限群的分类问题

给定正整数 $n$，$n$ 阶群有多少个（在同构意义下）？

- $n = 1$：唯一（平凡群）。
- $n = p$（素数）：唯一（循环群 $\mathbb{Z}/p\mathbb{Z}$）。
- $n = 4$：两个（$\mathbb{Z}/4\mathbb{Z}$ 和 $\mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$）。
- $n = 6$：两个（$\mathbb{Z}/6\mathbb{Z}$ 和 $S_3$）。
- $n = 8$：五个。$n = 12$：五个。$n = 64$：267 个。

数量增长极不规则，完全取决于 $n$ 的算术结构。要系统地解决分类问题，我们需要三类工具：
1. **存在性：** 证明特定阶数的子群必须存在。
2. **共轭性/唯一性：** 判断这些子群之间的关系。
3. **计数：** 确定这类子群的精确个数或范围。

Lagrange 定理提供了必要条件（子群阶整除群阶），Cauchy 定理对素因子给出了部分逆命题，Sylow 定理则对素数幂因子给出了完整答案。

---


![S_3 的子群格与 Sylow 子群](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/abstract-algebra/04-sylow-theorems/aa_fig4_sylow_lattice.png)

## p-群与 Cauchy 定理

**定义.** 设 $p$ 为素数。**$p$-群**是指每个元素的阶都是 $p$ 的幂的群。对有限群而言，这等价于 $|G| = p^k$（$k \geq 0$）。

$p$-群是构成所有有限群的"原子"，正如素数是构成所有正整数的原子。

**$p$-群的关键性质.** 有限 $p$-群的中心非平凡：$|Z(G)| > 1$。

**证明（类方程）.** 有限群 $G$ 的类方程为：
$$|G| = |Z(G)| + \sum_{i} [G : C_G(g_i)]$$
其中求和遍历所有大小 $> 1$ 的共轭类的代表元 $g_i$，$C_G(g_i)$ 为 $g_i$ 的中心化子。

每个 $[G : C_G(g_i)]$ 整除 $|G| = p^k$ 且 $> 1$，故被 $p$ 整除。左端 $|G| = p^k$ 被 $p$ 整除。因此 $|Z(G)|$ 也被 $p$ 整除，即 $|Z(G)| \geq p > 1$。$\blacksquare$

这个简洁的论证有深远推论。

**推论.** $p^2$ 阶群都是 Abel 群。

*证明.* 设 $|G| = p^2$。由上述性质，$|Z(G)| \in \{p, p^2\}$。若 $|Z(G)| = p^2$，则 $G = Z(G)$ 为 Abel 群。若 $|Z(G)| = p$，则 $|G/Z(G)| = p$，$G/Z(G)$ 为循环群。但 $G/Z(G)$ 为循环群蕴含 $G$ 为 Abel 群（标准练习题），与 $|Z(G)| < |G|$ 矛盾。$\blacksquare$

**Cauchy 定理.** 若素数 $p$ 整除 $|G|$，则 $G$ 包含 $p$ 阶元素。

**证明（McKay 的精巧论证）.** 考虑集合
$$S = \{(g_1, g_2, \ldots, g_p) \in G^p : g_1 g_2 \cdots g_p = e\}.$$

任取 $g_1, \ldots, g_{p-1}$，$g_p = (g_1 \cdots g_{p-1})^{-1}$ 唯一确定，故 $|S| = |G|^{p-1}$。

循环群 $\mathbb{Z}/p\mathbb{Z}$ 通过循环置换作用在 $S$ 上：将 $(g_1, \ldots, g_p)$ 映到 $(g_{2}, \ldots, g_p, g_1)$。这个作用是良定义的，因为若 $g_1 \cdots g_p = e$，则 $g_2 \cdots g_p \cdot g_1 = g_1^{-1} e \cdot g_1 = e$。

由于 $p$ 是素数，每个轨道的大小为 $1$ 或 $p$。不动点恰为形如 $(g, g, \ldots, g)$ 的 $p$-元组（$g^p = e$），其中包括 $(e, \ldots, e)$。

由轨道计数公式：$|S| \equiv (\text{不动点个数}) \pmod{p}$。因 $|S| = |G|^{p-1}$ 且 $p \mid |G|$，有 $p \mid |S|$，故不动点个数被 $p$ 整除。已知至少有一个不动点（恒等元组），因此不动点个数 $\geq p$，即存在 $g \neq e$ 使 $g^p = e$。这个 $g$ 的阶恰为 $p$。$\blacksquare$

McKay 的证明是"群作用计数"技术的典范——同样的思路贯穿整个 Sylow 理论。

---

## Sylow 第一定理：存在性

**定义.** 设 $|G| = p^a m$，其中 $\gcd(p, m) = 1$（即 $p^a$ 是整除 $|G|$ 的 $p$ 的最高幂次）。$G$ 的一个 **Sylow $p$-子群**是 $G$ 的 $p^a$ 阶子群。

**定理（Sylow 第一定理）.** 设 $G$ 为有限群，$p$ 为整除 $|G|$ 的素数。写 $|G| = p^a m$（$\gcd(p,m) = 1$，$a \geq 1$）。则 $G$ 包含 $p^a$ 阶子群（Sylow $p$-子群）。更一般地，对每个 $0 \leq k \leq a$，$G$ 包含 $p^k$ 阶子群。

**证明（通过群作用，对 $|G|$ 归纳）.**

*归纳基础：* $|G| = 1$ 平凡。

*归纳步骤：* 假设对所有阶小于 $|G|$ 的群定理成立。考虑类方程：
$$|G| = |Z(G)| + \sum_i [G : C_G(g_i)]$$

**情形 1：** 某个 $[G : C_G(g_i)]$ 不被 $p^a$ 整除。由 $|G| = [G : C_G(g_i)] \cdot |C_G(g_i)|$，可知 $p^a$ 整除 $|C_G(g_i)|$。又 $|C_G(g_i)| < |G|$（因为 $g_i \notin Z(G)$），由归纳假设，$C_G(g_i)$ 有 Sylow $p$-子群，这也是 $G$ 的 $p^a$ 阶子群。

**情形 2：** 每个 $[G : C_G(g_i)]$ 都被 $p$ 整除。则从类方程看，$p \mid |Z(G)|$。对 Abel 群 $Z(G)$ 用 Cauchy 定理，得到 $z \in Z(G)$，$|z| = p$。子群 $N = \langle z \rangle$ 正规于 $G$（因 $z$ 在中心）。商群 $G/N$ 的阶为 $p^{a-1}m$。由归纳假设，$G/N$ 有 $p^{a-1}$ 阶子群。由格对应定理，这对应 $G$ 中包含 $N$ 的子群 $H$，且 $|H| = p \cdot p^{a-1} = p^a$。$\blacksquare$

---

## Sylow 第二、第三定理：共轭与计数

**定理（Sylow 第二定理）.** $G$ 的任意两个 Sylow $p$-子群共轭。即若 $P, Q$ 都是 Sylow $p$-子群，则存在 $g \in G$ 使 $gPg^{-1} = Q$。

**证明思路.** 令 $P$ 通过左乘作用在陪集空间 $G/Q$ 上。用轨道计数分析不动点。陪集 $gQ$ 是 $P$ 的不动点当且仅当 $P \subseteq gQg^{-1}$。不动点个数 $\equiv |G/Q| = m \pmod{p}$（非不动轨道大小被 $p$ 整除）。因 $\gcd(m, p) = 1$，至少有一个不动点 $gQ$，于是 $P \subseteq gQg^{-1}$。两边阶相等，故 $P = gQg^{-1}$。$\blacksquare$

**定理（Sylow 第三定理）.** 设 $n_p$ 为 $G$ 的 Sylow $p$-子群个数。则：
1. $n_p \mid m$（其中 $|G| = p^a m$，$\gcd(p, m) = 1$）。
2. $n_p \equiv 1 \pmod{p}$。

**证明思路.** 记 $\text{Syl}_p(G)$ 为所有 Sylow $p$-子群的集合。由 Sylow 第二定理，$G$ 通过共轭传递地作用在 $\text{Syl}_p(G)$ 上。固定 $P \in \text{Syl}_p(G)$，其稳定子为正规化子 $N_G(P)$。由轨道-稳定子定理，$n_p = [G : N_G(P)]$。因 $P \leq N_G(P)$，$p^a \mid |N_G(P)|$，故 $n_p$ 整除 $m$。

对于条件 (2)，令 $P$ 通过共轭作用在 $\text{Syl}_p(G)$ 上。细致分析表明唯一的不动点是 $P$ 自身（若 $Q$ 也是不动点，则 $PQ$ 成群且阶的分析迫使 $P = Q$）。非不动轨道大小被 $p$ 整除，故 $n_p \equiv 1 \pmod{p}$。$\blacksquare$

**汇总.** 写 $|G| = p^a m$，$\gcd(p,m) = 1$：
- **Sylow I：** Sylow $p$-子群存在（阶为 $p^a$ 的子群）。
- **Sylow II：** 所有 Sylow $p$-子群彼此共轭。
- **Sylow III：** Sylow $p$-子群的个数 $n_p$ 满足 $n_p \mid m$ 且 $n_p \equiv 1 \pmod{p}$。

**关键推论.** Sylow $p$-子群 $P$ 正规于 $G$ 当且仅当 $n_p = 1$。（因为所有 Sylow $p$-子群共轭，$P$ 正规等价于它是唯一的。）

---

## 小阶群的分类

Sylow 定理与半直积等结构工具结合，可以系统地分类小阶群。

### 6 阶群

$|G| = 6 = 2 \cdot 3$。由 Sylow 第三定理：
- $n_3 \mid 2$ 且 $n_3 \equiv 1 \pmod{3}$，故 $n_3 = 1$。
- $n_2 \mid 3$ 且 $n_2 \equiv 1 \pmod{2}$，故 $n_2 \in \{1, 3\}$。

$n_3 = 1$ 意味着唯一的 Sylow 3-子群 $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ 正规于 $G$。

**若 $n_2 = 1$：** $P_2, P_3$ 均正规，$|P_2 \cap P_3| = 1$（因为阶互素），$G = P_2 \times P_3 \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}/6\mathbb{Z}$。

**若 $n_2 = 3$：** 群非 Abel（否则 $n_2 = 1$）。$P_3$ 正规，$P_2$ 中的 2 阶元素通过共轭作用在 $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ 上。$\mathbb{Z}/3\mathbb{Z}$ 的自同构群为 $\mathbb{Z}/2\mathbb{Z}$，唯一的非平凡自同构是求逆 $x \mapsto x^{-1}$。这给出 $G \cong S_3$——唯一的 6 阶非 Abel 群。

**结论.** 6 阶群在同构意义下恰有两个：$\mathbb{Z}/6\mathbb{Z}$ 和 $S_3$。

### 8 阶群

$|G| = 8 = 2^3$ 是 $p$-群，Sylow 定理不直接约束内部结构。利用 $|Z(G)| > 1$：

- $|Z(G)| = 8$：$G$ 为 Abel 群。有限生成 Abel 群基本定理给出三种：$\mathbb{Z}/8\mathbb{Z}$，$\mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$，$(\mathbb{Z}/2\mathbb{Z})^3$。
- $|Z(G)| = 4$：$|G/Z(G)| = 2$，循环，迫使 $G$ 为 Abel 群——矛盾。
- $|Z(G)| = 2$：$|G/Z(G)| = 4$。若 $G/Z(G)$ 循环则 $G$ Abel（矛盾），故 $G/Z(G) \cong (\mathbb{Z}/2\mathbb{Z})^2$。此情形产生两个非 Abel 群：**二面体群** $D_4$（正方形对称群）和**四元数群** $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$。

**结论.** 8 阶群恰有五个：三个 Abel 群（$\mathbb{Z}/8\mathbb{Z}$, $\mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$, $(\mathbb{Z}/2\mathbb{Z})^3$）和两个非 Abel 群（$D_4$, $Q_8$）。

### 12 阶群

$|G| = 12 = 2^2 \cdot 3$。由 Sylow 第三定理：
- $n_3 \mid 4$，$n_3 \equiv 1 \pmod{3}$，故 $n_3 \in \{1, 4\}$。
- $n_2 \mid 3$，$n_2 \equiv 1 \pmod{2}$，故 $n_2 \in \{1, 3\}$。

**$n_3 = 1$ 的情形：** 唯一的 Sylow 3-子群 $P_3 \cong \mathbb{Z}/3\mathbb{Z}$ 正规。$G$ 为 $P_3$ 与 4 阶 Sylow 2-子群 $P_2$ 的半直积 $P_3 \rtimes P_2$，由 $P_2$ 对 $P_3$ 的作用分类。

- 平凡作用给出直积：$\mathbb{Z}/12\mathbb{Z}$ 或 $\mathbb{Z}/6\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$。
- $P_2 \cong \mathbb{Z}/4\mathbb{Z}$ 的非平凡作用：$\text{Aut}(\mathbb{Z}/3\mathbb{Z}) \cong \mathbb{Z}/2\mathbb{Z}$，作用通过商映射 $\mathbb{Z}/4\mathbb{Z} \to \mathbb{Z}/2\mathbb{Z}$ 给出二循环群 $\text{Dic}_3$。
- $P_2 \cong (\mathbb{Z}/2\mathbb{Z})^2$ 的非平凡作用：给出二面体群 $D_6$（12 阶）。

**$n_3 = 4$ 的情形：** 4 个 Sylow 3-子群贡献 $4 \times 2 = 8$ 个 3 阶元素，剩余 4 个元素构成唯一的 Sylow 2-子群（$n_2 = 1$）。$G$ 通过共轭作用于 4 个 Sylow 3-子群上，细致分析给出 $A_4$。

**结论.** 12 阶群恰有五个：$\mathbb{Z}/12\mathbb{Z}$，$\mathbb{Z}/6\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}$，$D_6$，$\text{Dic}_3$，$A_4$。

---

## 应用与非 Abel 实例

**应用 1: $pq$ 阶群（$p < q$ 为素数）.** 由 Sylow 第三定理，$n_q \mid p$ 且 $n_q \equiv 1 \pmod{q}$。因 $p < q$，唯一的可能是 $n_q = 1$。故 Sylow $q$-子群唯一且正规。若还有 $q \not\equiv 1 \pmod{p}$，则 $n_p = 1$，$G \cong \mathbb{Z}/pq\mathbb{Z}$。

**实例.** 15 阶群（$15 = 3 \times 5$）。$n_5 \mid 3$，$n_5 \equiv 1 \pmod{5}$，故 $n_5 = 1$；$n_3 \mid 5$，$n_3 \equiv 1 \pmod{3}$，故 $n_3 = 1$。两个 Sylow 子群均正规，$G \cong \mathbb{Z}/15\mathbb{Z}$。15 阶群只有一个。

**应用 2: 不存在 12 阶单群.** 反证：设 $G$ 单群，$|G| = 12$。则 $n_3 = 4$（否则 Sylow 3-子群正规，与单群矛盾）。共轭作用给出单射 $\varphi: G \hookrightarrow S_4$（核平凡因 $G$ 单），故 $G$ 同构于 $S_4$ 的 12 阶子群，即 $A_4$。但 $A_4$ 有正规子群 $V_4$，不是单群——矛盾。

**应用 3: 30 阶群一定有正规 Sylow 5-子群.** $|G| = 30 = 2 \cdot 3 \cdot 5$。

$n_5 \mid 6$，$n_5 \equiv 1 \pmod{5}$，故 $n_5 \in \{1, 6\}$。$n_3 \mid 10$，$n_3 \equiv 1 \pmod{3}$，故 $n_3 \in \{1, 10\}$。

若 $n_5 = 6$：有 $6 \times 4 = 24$ 个 5 阶元素。若同时 $n_3 = 10$：有 $10 \times 2 = 20$ 个 3 阶元素。总计至少 $24 + 20 = 44 > 30$ 个元素——矛盾。故 $n_5 = 6$ 时必有 $n_3 = 1$。

$n_3 = 1$ 给出正规 Sylow 3-子群 $P_3$，商群 $G/P_3$ 的阶为 10。在 $G/P_3$ 中，$n_5 \mid 2$ 且 $n_5 \equiv 1 \pmod 5$，所以 $n_5(G/P_3) = 1$。回溯到 $G$，得到一个 15 阶正规子群。15 阶群是循环群（如上所述），其 Sylow 5-子群是特征子群，从而正规于 $G$，即 $n_5 = 1$——与假设矛盾。

因此 $n_5 = 1$：每个 30 阶群都有正规 Sylow 5-子群。

---

## 下一步

Sylow 定理为素数幂阶子群提供了存在性、共轭性和计数三重保证。结合上一篇的商群与同态理论，我们已经能够对有限群进行精细的结构分析。下一篇文章将从子群转向群本身的"外部行为"——**群作用**。轨道-稳定子定理、Burnside 引理及其在组合计数和几何中的应用，让抽象的群论真正"落地"到具体的数学问题中。群作用也是我们证明 Sylow 定理本身所用的核心技术——现在是时候系统地掌握它了。

---

*本文是 [抽象代数](/zh/series/abstract-algebra/) 系列的第 4 篇（共 12 篇）。*

*上一篇：[第 3 篇 —— 商群与同态](/zh/abstract-algebra/03-商群与同态/)*

*下一篇：[第 5 篇 —— 环与理想](/zh/abstract-algebra/05-环与理想/)*
