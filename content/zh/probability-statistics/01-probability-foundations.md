---
title: "概率与统计（一）：概率空间——为何需要公理化（但不过度深究）"
date: 2024-08-18 09:00:00
tags:
  - Probability
  - Statistics
  - Combinatorics
  - Bayes Theorem
categories: Probability and Statistics
series: probability-statistics
lang: zh
mathjax: true
description: "从零构建概率论：样本空间、柯尔莫哥洛夫公理、条件概率、贝叶斯定理与生日问题——含严格证明与 Python 模拟。"
disableNunjucks: true
series_order: 1
translationKey: "probability-statistics-1"
---
每次查看天气预报、运行 A/B 测试或训练神经网络，其底层基础都可追溯至 1933 年安德雷·柯尔莫哥洛夫（Andrey Kolmogorov）建立的概率公理化框架——此前概率论只是赌徒与精算师手边的经验技巧，此后则成为与微积分、代数同等严谨的数学分支。

好消息是：掌握现代概率论核心思想无需精通测度论。三条公理本身极为简洁，真正关键的是建立可靠直觉，并清醒识别其失效边界。

本文为你夯实基础：精确定义“概率”，推导关键工具——条件概率、贝叶斯定理与独立性，并以一个几乎颠覆所有初学者直觉的经典问题收尾。

## 样本空间、事件与 σ-代数


![样本空间可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/01-sample-space.png)

### 样本空间

一个**样本空间** $\Omega$ 是某个随机试验所有可能结果构成的集合。

| 实验 | 样本空间 $\Omega$ |
|---|---|
| 抛一枚硬币 | $\{H, T\}$ |
| 掷一颗骰子 | $\{1, 2, 3, 4, 5, 6\}$ |
| 测量电压值 | $\mathbb{R}$（或 $[0, \infty)$） |
| 统计一小时内网站访问次数 | $\{0, 1, 2, \ldots\}$ |

样本空间必须是**穷尽的**（包含所有可能结果）且**互斥的**（每个结果都是 $\Omega$ 中的一个单独元素）。

### 事件

一个**事件**是 $\Omega$ 的一个子集。例如掷骰子时，“掷出偶数”这一事件即为 $A = \{2, 4, 6\}$；“掷出某个数”即为 $\Omega$ 本身（称为**必然事件**）；而“在标准骰子上掷出 7”则是 $\emptyset$（称为**不可能事件**）。

### σ-代数（温和引入）

对于有限样本空间，我们可以为 $\Omega$ 的每一个子集赋予概率。但当 $\Omega$ 是不可数集（如 $\mathbb{R}$）时，会出现一些病态子集，无法被一致地赋予概率。一个**σ-代数** $\mathcal{F}$ 就是一类经过精心挑选的 $\Omega$ 的子集族——即我们“允许测量”的那些集合。

一个 σ-代数 $\mathcal{F}$ 必须满足以下三条性质：

1. $\Omega \in \mathcal{F}$
2. 若 $A \in \mathcal{F}$，则其补集 $A^c \in \mathcal{F}$（对补集封闭）
3. 若 $A_1, A_2, \ldots \in \mathcal{F}$，则其可数并 $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$（对可数并封闭）

在本系列中，当 $\Omega$ 为有限集时，我们采用**幂集**（即所有子集）；当 $\Omega = \mathbb{R}$ 时，则采用由开区间生成的**博雷尔 σ-代数**。你可以放心地将“事件”与“子集”视为后续内容中的同义词。

## 柯尔莫哥洛夫三大公理

定义在 $(\Omega, \mathcal{F})$ 上的**概率测度** $P$ 是一个函数 $P: \mathcal{F} \to \mathbb{R}$，满足：

![柯尔莫哥洛夫公理](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/01-kolmogorov-axioms.png)


**公理 1 （非负性）**：对任意事件 $A \in \mathcal{F}$，

$$P(A) \geq 0.$$

**公理 2 （归一化）**：整个样本空间的概率为 1：

$$P(\Omega) = 1.$$

**公理 3 （可数可加性）**：若 $A_1, A_2, \ldots$ 是两两不相交的事件（即当 $i \neq j$ 时 $A_i \cap A_j = \emptyset$），则

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i).$$

仅此而已。三条公理。其余一切——贝叶斯定理、大数定律、中心极限定理——皆为其逻辑推论。

### 直接推论

仅从这三条公理出发，即可推出：

**补集法则**：由于 $A$ 与 $A^c$ 不相交，且 $A \cup A^c = \Omega$，

$$P(A) + P(A^c) = P(\Omega) = 1 \implies P(A^c) = 1 - P(A).$$

**不可能事件**：$P(\emptyset) = 1 - P(\Omega) = 0$。

**容斥原理（两个事件）**：

$$P(A \cup B) = P(A) + P(B) - P(A \cap B).$$

*证明*：将 $A \cup B$ 写作 $A \cup (B \setminus A)$，其中 $A$ 与 $B \setminus A$ 不相交。于是 $P(A \cup B) = P(A) + P(B \setminus A)$。又因 $B = (A \cap B) \cup (B \setminus A)$（不相交），故 $P(B) = P(A \cap B) + P(B \setminus A)$，从而 $P(B \setminus A) = P(B) - P(A \cap B)$。代入即得结论。$\blacksquare$

**单调性**：若 $A \subseteq B$，则 $P(A) \leq P(B)$。

*证明*：$B = A \cup (B \setminus A)$（不相交），因此 $P(B) = P(A) + P(B \setminus A) \geq P(A)$（由公理 1）。$\blacksquare$

## 条件概率


![概率样本空间作为所有可能结果的宇宙](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/01-probability-sample-space-as-a-cosmic-universe-of-possible-ou.jpg)

当我们得知某事件 $B$ 已发生时，我们对其他事件的信念会发生变化。这种更新由**条件概率**刻画。

![条件概率](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/01-conditional-probability.png)


**定义**：若 $P(B) > 0$，则在 $B$ 发生条件下 $A$ 的条件概率为

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}.$$

直观上，我们“聚焦”于 $B$ 已发生的那个世界：$B$ 成为新的样本空间，并重新归一化。

### 乘法法则

重排上述定义可得：

$$P(A \cap B) = P(A \mid B) \, P(B) = P(B \mid A) \, P(A).$$

该法则可推广至链式情形：

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \, P(A_2 \mid A_1) \, P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1}).$$

### 示例：抽牌

一副标准扑克牌共 52 张。连续无放回地抽出两张 A 的概率是多少？

$$P(\text{Ace}_1 \cap \text{Ace}_2) = P(\text{Ace}_1) \cdot P(\text{Ace}_2 \mid \text{Ace}_1) = \frac{4}{52} \cdot \frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.00452.$$

## 全概率公式


![贝叶斯定理：侦探根据新证据更新信念](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/01-bayes-theorem-detective-updating-beliefs-with-new-evidence.jpg)

设 $B_1, B_2, \ldots, B_n$ 是 $\Omega$ 的一个**划分**——即各 $B_i$ 两两不相交，且 $\bigcup_i B_i = \Omega$，同时对所有 $i$ 都有 $P(B_i) > 0$。那么对任意事件 $A$，有：

![贝叶斯定理决策树](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/01-bayes-tree.png)


$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \, P(B_i).$$

*证明*：$A = A \cap \Omega = A \cap \left(\bigcup_i B_i\right) = \bigcup_i (A \cap B_i)$。由于 $A \cap B_i$ 两两不相交，由公理 3 得：

$$P(A) = \sum_i P(A \cap B_i) = \sum_i P(A \mid B_i) P(B_i). \quad \blacksquare$$

该公式极为实用。当你无法直接计算 $P(A)$ 时，可将整个世界按情形拆分，再分别处理每一部分。

## 贝叶斯定理

**定理（贝叶斯）**：若 $P(A) > 0$ 且 $P(B) > 0$，则

![生日问题曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/01-birthday-problem.png)


$$P(B \mid A) = \frac{P(A \mid B) \, P(B)}{P(A)}.$$

*证明*：由乘法法则，$P(A \mid B) P(B) = P(A \cap B) = P(B \mid A) P(A)$。两边同除以 $P(A)$ 即得。$\blacksquare$

结合全概率公式（使用划分 $B_1, \ldots, B_n$）：

$$P(B_k \mid A) = \frac{P(A \mid B_k) \, P(B_k)}{\sum_{i=1}^{n} P(A \mid B_i) \, P(B_i)}.$$

### 医学检测示例

某种疾病在人群中发病率为千分之一。一种检测方法的灵敏度（真阳性率）为 99%，特异度（真阴性率）为 95%。你检测结果为阳性，那么你实际患病的概率是多少？

定义：
- $D$：患病，$P(D) = 0.001$
- $D^c$：健康，$P(D^c) = 0.999$
- $T^+$：检测呈阳性
- $P(T^+ \mid D) = 0.99$（灵敏度）
- $P(T^+ \mid D^c) = 0.05$（1 − 特异度 = 假阳性率）

由全概率公式：

$$P(T^+) = P(T^+ \mid D)P(D) + P(T^+ \mid D^c)P(D^c) = 0.99 \times 0.001 + 0.05 \times 0.999 = 0.00099 + 0.04995 = 0.05094.$$

由贝叶斯定理：

$$P(D \mid T^+) = \frac{P(T^+ \mid D) P(D)}{P(T^+)} = \frac{0.99 \times 0.001}{0.05094} \approx 0.0194.$$

**不足 2%。** 即便面对一个“准确率达 99%”的检测，当疾病罕见时，阳性结果极大概率是假阳性。这就是**基础比率谬误**（base rate fallacy）——忽略先验概率 $P(D)$ 会导致严重错误的结论。

关键洞见在于：当疾病罕见时，健康人产生的假阳性（$0.05 \times 999 \approx 50$）远超患者产生的真阳性（$0.99 \times 1 \approx 1$）。

### 序贯检测：若你连续两次检测呈阳性呢？

假设你第一次检测呈阳性后，再次独立接受同一检测，结果仍为阳性。此时：

$$P(D \mid T_1^+, T_2^+) = \frac{P(T_1^+, T_2^+ \mid D) P(D)}{P(T_1^+, T_2^+)}.$$

由于在已知患病状态下两次检测相互独立：

$$P(T_1^+, T_2^+ \mid D) = 0.99^2 = 0.9801$$

$$P(T_1^+, T_2^+ \mid D^c) = 0.05^2 = 0.0025$$

$$P(T_1^+, T_2^+) = 0.9801 \times 0.001 + 0.0025 \times 0.999 = 0.000980 + 0.002498 = 0.003478$$

$$P(D \mid T_1^+, T_2^+) = \frac{0.000980}{0.003478} \approx 0.282.$$

两次阳性将患病概率从 1.9% 提升至 28.2%；第三次阳性将进一步推高至约 88%。这体现了**序贯更新**（sequential updating）的力量——每一份证据都会乘性地更新赔率（odds），即使单次检测较弱，累积起来也极具说服力。

等价地，你可将第一次检测后的后验概率（$P(D|T_1^+) \approx 0.0194$）作为第二次检测的先验——这正是我们在第 8 篇文章中将形式化的序贯贝叶斯更新。

## 独立性

两个事件 $A$ 和 $B$ 称为**独立的**，若

$$P(A \cap B) = P(A) \, P(B).$$

等价地，$P(A \mid B) = P(A)$ —— 已知 $B$ 发生，对 $A$ 的概率毫无影响。

### 两两独立 vs. 相互独立

对三个事件 $A, B, C$，**两两独立**指：

$$P(A \cap B) = P(A)P(B), \quad P(A \cap C) = P(A)P(C), \quad P(B \cap C) = P(B)P(C).$$

**相互独立**则额外要求：

$$P(A \cap B \cap C) = P(A) P(B) P(C).$$

两两独立**不能推出**相互独立。

**反例**：掷两颗均匀骰子。令 $A$ = “第一颗骰子点数为奇数”，$B$ = “第二颗骰子点数为奇数”，$C$ = “点数和为奇数”。则 $P(A) = P(B) = P(C) = 1/2$。三组两两独立（练习：验证之）。但 $A \cap B \cap C = \emptyset$（两个奇数之和必为偶数），故 $P(A \cap B \cap C) = 0 \neq 1/8 = P(A)P(B)P(C)$。

*验证该例中两两独立性*：考虑 $A$ 与 $C$：第一颗骰子为奇数的情形占全部 36 种结果中的 18 种；点数和为奇数的情形亦占 18 种（当且仅当一奇一偶）；事件 $A \cap C$ = “第一颗为奇数且和为奇数” = “第一颗为奇数且第二颗为偶数”，共 $3 \times 3 = 9$ 种结果。故 $P(A \cap C) = 9/36 = 1/4 = P(A)P(C)$。其余两组同理。$\checkmark$

对 $n$ 个事件，相互独立需满足 $2^n - n - 1$ 个条件（所有 2、 3、…、$n$ 个事件的交集概率）。而两两独立仅检验 $\binom{n}{2}$ 个条件。当 $n$ 很大时，两者间的差距呈指数级增长。

## 容斥原理：一般情形

双事件公式可推广至 $n$ 个事件：

$$P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i} P(A_i) - \sum_{i < j} P(A_i \cap A_j) + \sum_{i < j < k} P(A_i \cap A_j \cap A_k) - \cdots + (-1)^{n+1} P(A_1 \cap \cdots \cap A_n).$$

*证明*：用数学归纳法。基础情形 $n = 2$ 已证。假设公式对 $n-1$ 个事件成立。记 $\bigcup_{i=1}^n A_i = \left(\bigcup_{i=1}^{n-1} A_i\right) \cup A_n$，应用双事件公式：

$$P\left(\bigcup_{i=1}^n A_i\right) = P\left(\bigcup_{i=1}^{n-1} A_i\right) + P(A_n) - P\left(\left(\bigcup_{i=1}^{n-1} A_i\right) \cap A_n\right).$$

最后一项等于 $P\left(\bigcup_{i=1}^{n-1} (A_i \cap A_n)\right)$，由归纳假设展开后符号交替。合并同类项即得通式。$\blacksquare$

尽管一般容斥公式含 $2^n - 1$ 项，但它对计算涉及“至少一个”事件的概率至关重要——如下文的生日问题，以及后续的并界（union bound，即弱化版容斥，仅保留首项求和）。

**并界（布尔不等式）**：由于所有减去的项均非负，

$$P\left(\bigcup_{i=1}^n A_i\right) \leq \sum_{i=1}^n P(A_i).$$

这个简单界虽宽松，却极为实用——它支撑了多重检验中的邦费罗尼校正（第 7 篇文章）及学习理论中的诸多结论。

## 条件概率本身是一种概率测度

一个微妙但重要的事实是：对固定事件 $B$（满足 $P(B) > 0$），函数 $Q(A) = P(A|B)$ 本身也是 $(\Omega, \mathcal{F})$ 上的一个概率测度。即：

1. 对所有 $A$，有 $Q(A) \geq 0$（非负性）
2. $Q(\Omega) = P(\Omega | B) = P(\Omega \cap B)/P(B) = P(B)/P(B) = 1$（归一化）
3. 若 $A_1, A_2, \ldots$ 互不相交，则 $Q(\bigcup A_i) = \sum Q(A_i)$（可数可加性）

*（3）的证明*：$P(\bigcup A_i | B) = P((\bigcup A_i) \cap B)/P(B) = P(\bigcup(A_i \cap B))/P(B) = \sum P(A_i \cap B)/P(B) = \sum P(A_i|B)$。$\blacksquare$

这意味着，我们关于概率测度所证明的一切定理，自动适用于条件概率。条件期望、条件方差、条件独立性——它们均遵循相同的公理体系。

## 计数：排列与组合

当有限样本空间 $\Omega$ 中所有结果等可能发生时，$P(A) = |A|/|\Omega|$，概率问题就简化为计数问题。

### 排列

从 $n$ 个不同对象中选出 $k$ 个并排序（顺序重要）的方式数：

$$P(n, k) = \frac{n!}{(n-k)!}.$$

**示例**： 8 名选手参加赛跑，前三名的可能排列数？$P(8, 3) = 8!/5! = 8 \times 7 \times 6 = 336$。

### 组合

从 $n$ 个对象中选出 $k$ 个（顺序无关）的方式数：

$$\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{P(n,k)}{k!}.$$

除以 $k!$ 是为了消除顺序——每个大小为 $k$ 的无序集合对应 $k!$ 种有序排列。

关键性质：
- $\binom{n}{k} = \binom{n}{n-k}$（对称性）
- $\binom{n}{0} = \binom{n}{n} = 1$
- $\sum_{k=0}^{n} \binom{n}{k} = 2^n$（子集总数）
- 帕斯卡法则：$\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$

*帕斯卡法则的证明*：考虑 $n$ 个对象中有一个“特殊”对象。要选 $k$ 个对象：要么包含该特殊对象（再从其余 $n-1$ 个中选 $k-1$ 个，共 $\binom{n-1}{k-1}$ 种），要么排除它（从其余 $n-1$ 个中选 $k$ 个，共 $\binom{n-1}{k}$ 种）。$\blacksquare$

### 二项式定理

帕斯卡法则将组合数与展开式联系起来：

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}.$$

令 $a = b = 1$，得 $\sum \binom{n}{k} = 2^n$（子集总数）；令 $a = 1, b = -1$，得 $\sum (-1)^k \binom{n}{k} = 0$，即偶数大小子集数等于奇数大小子集数。

### 多项式系数

将 $n$ 个对象划分为大小分别为 $n_1, n_2, \ldots, n_r$ 的 $r$ 组（满足 $\sum n_i = n$）的方式数：

$$\binom{n}{n_1, n_2, \ldots, n_r} = \frac{n!}{n_1! \, n_2! \cdots n_r!}.$$

**示例**：“MISSISSIPPI”一词有多少种不同排列？共 11 个字母： 1 个 M、 4 个 I、 4 个 S、 2 个 P。

$$\frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = \frac{39916800}{1 \cdot 24 \cdot 24 \cdot 2} = 34650.$$

### 星与条（Stars and Bars）

将 $k$ 个相同物品分配到 $n$ 个不同容器中有多少种方式？即经典的“星与条”问题：

$$\binom{k + n - 1}{n - 1} = \binom{k + n - 1}{k}.$$

**示例**：将 10 块相同饼干分给 4 个孩子：$\binom{13}{3} = 286$ 种方式。

*证明*：用 $k$ 颗星表示物品，$n-1$ 根竖线表示容器分隔符。有效分配对应任意一个由 $k$ 颗星与 $n-1$ 根竖线组成的序列，此类序列总数为 $\binom{k+n-1}{n-1}$。$\blacksquare$

## 生日问题

**问题**：一个 $n$ 人的房间里，至少有两人同一天生日的概率是多少？（假设一年 365 天，生日等可能，忽略闰年。）

更易通过补集计算：令 $A$ 表示“至少有一对生日相同”，则 $A^c$ 表示“所有人生日均不同”。

$$P(A^c) = \frac{365}{365} \cdot \frac{364}{365} \cdot \frac{363}{365} \cdots \frac{365 - n + 1}{365} = \prod_{k=0}^{n-1} \frac{365 - k}{365}.$$

因此：

$$P(A) = 1 - \prod_{k=0}^{n-1} \left(1 - \frac{k}{365}\right).$$

### 近似解法

利用小 $x$ 下 $\ln(1-x) \approx -x$：

$$\ln P(A^c) = \sum_{k=0}^{n-1} \ln\left(1 - \frac{k}{365}\right) \approx -\sum_{k=0}^{n-1} \frac{k}{365} = -\frac{n(n-1)}{2 \cdot 365}.$$

故：

$$P(A^c) \approx e^{-n(n-1)/730}.$$

令 $P(A) = 0.5$：$e^{-n(n-1)/730} = 0.5$，解得 $n(n-1) = 730 \ln 2 \approx 506$，即 $n \approx 23$。

**仅需 23 人，就有 50% 的概率出现生日匹配。** 大多数人会猜一个高得多的数字（如 183），因为他们混淆了“有人和我同一天生日”与“某两人同一天生日”。配对数量呈二次增长：$\binom{23}{2} = 253$ 对，每对匹配概率虽小，但 253 次机会叠加起来速度极快。

### 广义生日问题

生日问题可推广：若共有 $d$ 种等可能的“生日”（未必是 365），且有 $n$ 人，则碰撞阈值近似为：

$$n \approx \sqrt{2d \ln 2} \approx 1.177 \sqrt{d}.$$

该结论的应用远不止派对游戏：

- **哈希碰撞**：一个输出空间为 $d = 2^{128}$ 的哈希函数，在约 $2^{64}$ 次输入后即大概率发生碰撞——这解释了为何 128 位哈希对数十亿对象安全，但对 $2^{64}$ 个对象则不安全。
- **DNA 分型**：若有 $d$ 种可能基因型，需测试多少人才可能因偶然性出现两人匹配？
- **随机抽样**：从大小为 $d$ 的总体中随机抽取多少样本后，会首次出现重复？

其核心惊奇在于**二次缩放律**（$n \sim \sqrt{d}$）。人们的直觉常是线性的（$n \sim d/2$），误差巨大。

### Python 模拟

```python
import numpy as np
import matplotlib.pyplot as plt

def birthday_exact(n, days=365):
    """n 个人中至少有一对生日相同的精确概率。"""
    p_no_match = 1.0
    for k in range(n):
        p_no_match *= (days - k) / days
    return 1 - p_no_match

def birthday_simulation(n, days=365, trials=100_000):
    """生日碰撞概率的蒙特卡洛估计。"""
    collisions = 0
    for _ in range(trials):
        birthdays = np.random.randint(0, days, size=n)
        if len(set(birthdays)) < n:
            collisions += 1
    return collisions / trials

# 计算 n = 1..80 的精确概率
ns = np.arange(1, 81)
exact = [birthday_exact(n) for n in ns]
approx = [1 - np.exp(-n * (n - 1) / 730) for n in ns]

# 模拟若干关键值
sim_ns = [10, 20, 23, 30, 40, 50, 60, 70]
sim_probs = [birthday_simulation(n) for n in sim_ns]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ns, exact, 'b-', linewidth=2, label='精确解')
ax.plot(ns, approx, 'r--', linewidth=1.5, label='近似解')
ax.scatter(sim_ns, sim_probs, color='green', zorder=5, label='模拟（10 万次试验）')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
ax.axvline(x=23, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('人数', fontsize=13)
ax.set_ylabel('P(至少一对匹配)', fontsize=13)
ax.set_title('生日问题', fontsize=15)
ax.legend(fontsize=12)
ax.set_xlim(1, 80)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('birthday_problem.png', dpi=150)
plt.show()
```

运行该模拟可验证精确计算：当 $n = 23$ 时，概率约为 0.507；当 $n = 50$ 时，概率超过 0.97。蒙特卡洛模拟得到的绿色散点紧密围绕蓝色精确曲线，说明：**即便个体结果随机，整体行为却高度可预测。**

| $n$ | $P(\text{匹配})$ |
|-----|-------------------|
| 10  | 0.117 |
| 20  | 0.411 |
| 23  | 0.507 |
| 30  | 0.706 |
| 40  | 0.891 |
| 50  | 0.970 |
| 60  | 0.994 |
| 70  | 0.999 |

## 三门问题（蒙提霍尔问题）

任何关于概率基础的文章，若不包含这个经典问题，都是不完整的。

**设定**：你参加一个电视竞猜节目。面前有三扇门。其中一扇门后是汽车，另两扇后是山羊。你选择一扇门（比如第 1 扇）。主持人（知道门后情况）打开另一扇门（比如第 3 扇），露出一只山羊。此时，你是应该换到第 2 扇门，还是坚持选第 1 扇？

**直觉（错误）答案**：无所谓——剩下两扇门，概率各半。

**正确答案**：换门。换门获胜概率为 $2/3$。

*用全概率公式证明*：令 $C_i$ 表示汽车在第 $i$ 扇门后，$P(C_i) = 1/3$。令 $H_3$ 表示主持人打开第 3 扇门。我们需求 $P(C_2 | H_3)$。

$$P(H_3 | C_1) = 1/2 \quad \text{（主持人在 2、3 中随机选一扇）}$$
$$P(H_3 | C_2) = 1 \quad \text{（主持人必须开 3，因 2 后是车）}$$
$$P(H_3 | C_3) = 0 \quad \text{（主持人绝不会揭示汽车）}$$

由贝叶斯定理：

$$P(C_2 | H_3) = \frac{P(H_3 | C_2) P(C_2)}{P(H_3)} = \frac{1 \cdot 1/3}{P(H_3)}.$$

$$P(H_3) = P(H_3|C_1)P(C_1) + P(H_3|C_2)P(C_2) + P(H_3|C_3)P(C_3) = \frac{1}{2}\cdot\frac{1}{3} + 1\cdot\frac{1}{3} + 0 = \frac{1}{2}.$$

$$P(C_2 | H_3) = \frac{1/3}{1/2} = \frac{2}{3}. \quad \blacksquare$$

主持人的行为提供了信息。在他开门前，第 1 扇门有 $1/3$ 概率正确。他揭示第 3 扇门后是山羊，并未改变第 1 扇门后的情况——它仍是 $1/3$。但原本分布在第 2、 3 扇门上的 $2/3$ 概率，现在全部集中到了第 2 扇门。

**为何此问题超越游戏秀的意义**：三门问题阐明了一个普适原则——**基于信息进行条件化会以依赖于信息生成机制的方式改变概率**。主持人的选择受约束（绝不揭示汽车），正是这一约束使换门变得有利。在机器学习中，类似情形出现在选择偏差（selection bias）、幸存者偏差（survivorship bias）及因果推断中的混杂器偏差（collider bias）中。

```python
# 三门问题模拟
import numpy as np

def monty_hall_sim(n_trials=100_000):
    """模拟三门问题。"""
    np.random.seed(42)
    car_positions = np.random.randint(0, 3, n_trials)
    initial_choices = np.random.randint(0, 3, n_trials)

    stay_wins = np.sum(car_positions == initial_choices)
    switch_wins = n_trials - stay_wins  # 换门获胜当且仅当坚持失败

    print(f"坚持获胜:   {stay_wins/n_trials:.4f} (理论值: 0.3333)")
    print(f"换门获胜: {switch_wins/n_trials:.4f} (理论值: 0.6667)")

monty_hall_sim()
```

## 概率作为信念的度量

我们一直以公理化方式处理概率——将其视为满足三条规则的函数。但概率**究竟意味着什么**？主要有三种诠释：

**古典（拉普拉斯）诠释**：概率是有利结果数与总等可能结果数之比。这对硬币、骰子有效，但当结果非等可能时失效（例如，“明天下雨的概率”是多少？）。

**频率主义诠释**：概率是事件在大量重复试验中出现的长期相对频率。$P(A) = \lim_{n \to \infty} \frac{\text{事件 } A \text{ 在 } n \text{ 次试验中出现的次数}}{n}$。该定义精确，但要求实验（至少在概念上）可重复。

**贝叶斯（主观）诠释**：概率量化了智能体对不确定命题的信念程度。它无需对应任何物理频率。两位拥有不同先验信息的理性智能体，可对同一事件赋予不同概率，且二者均可“正确”。

这三种诠释均使用相同的公理体系。区别在于哲学立场，但具有实际后果：它决定了你采用频率主义统计（置信区间、 p 值）还是贝叶斯统计（先验、后验、可信区间）。我们将在第 8 篇文章中重返这一争论。

## 概率工具箱速查表

以下是本文推导出的关键公式汇总：

| 规则 | 公式 | 使用场景 |
|---|---|---|
| 补集法则 | $P(A^c) = 1 - P(A)$ | “至少一个”类问题 |
| 容斥原理 | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | 事件并集概率 |
| 并界 | $P(\bigcup A_i) \leq \sum P(A_i)$ | 快速上界估计 |
| 条件概率 | $P(A\|B) = P(A \cap B)/P(B)$ | 基于新信息更新概率 |
| 乘法法则 | $P(A \cap B) = P(A\|B)P(B)$ | 联合事件概率 |
| 全概率公式 | $P(A) = \sum P(A\|B_i)P(B_i)$ | 按情形分解计算 |
| 贝叶斯定理 | $P(B\|A) = \frac{P(A\|B)P(B)}{P(A)}$ | 反向条件化 |
| 独立性 | $P(A \cap B) = P(A)P(B)$ | 简化乘积计算 |
| 组合数 | $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ | 等可能结果情形 |

## 常见概率误区

在继续之前，列出连经验丰富的实践者都可能犯的错误：

**1. 赌徒谬误**：“轮盘连续 5 次出红，黑该出了。”若各次旋转独立，则下次出黑的概率恒定不变。过去结果不影响未来结果。

**2. 混淆 $P(A|B)$ 与 $P(B|A)$**：已知检测阳性时患病的概率（$P(D|T^+)$）不等于已知患病时检测阳性的概率（$P(T^+|D)$）。这在法律语境中称为**检察官谬误**（prosecutor's fallacy），曾导致冤假错案。

**3. 未经论证即假设独立性**：若事件不独立，则 $P(A \cap B) \neq P(A)P(B)$。在不满足独立性时强行假设，会严重低估联合概率（2008 年金融危机部分源于模型错误假设房贷违约相互独立）。

**4. 忽略基础比率**：解读证据时忽视先验概率。如前所述的医学检测示例所示，即使检测高度准确，当基础比率很低时，结果仍以假阳性为主。

**5. 生日问题直觉失效**：预期线性缩放（$n \sim d/2$），而正确缩放是二次的（$n \sim \sqrt{d}$）。这在哈希碰撞、 DNA 匹配及任何涉及两两比较的问题中都会发生。

**6. 混淆“不太可能”与“不可能”**：概率 0.01 意味着 100 次中有 1 次。做 100 次实验，你预计它会发生一次。在全球数十亿人口背景下，百万分之一的事件每天发生数千次。“不太可能”不等于“奇迹”，而是“在大规模下必然发生”。

**7. 辛普森悖论**：某一趋势在多个分组中均成立，但在合并数据后却逆转。例如，治疗 A 在男性和女性中成功率均高于治疗 B，但总体成功率却低于治疗 B——若男女在两种治疗中的分布不均所致。解决之道是控制混杂变量（性别），这又将我们带回条件概率与贝叶斯定理。辛普森悖论表明：因果推理所需远不止概率——你还需知晓因果结构，才能决定是否应进行条件化。

## 下一步

我们已搭建起形式化框架：样本空间、公理、条件概率与独立性。但迄今为止，我们只讨论了**事件**——即关于结果的是/否问题。概率论真正的力量在于将**数值**赋予结果，从而将事件转化为**随机变量**。分布（distributions）由此诞生，而数学也开始与数据科学及机器学习直接接轨。

下一篇中，我们将定义随机变量、概率质量函数（PMF）、概率密度函数（PDF），并梳理实践中无处不在的各类分布：伯努利（Bernoulli）、二项（Binomial）、泊松（Poisson）、高斯（Gaussian）、指数（Exponential）等。我们将看到每种分布如何自然地源于特定建模假设，并构建一张你将在本系列余下部分反复使用的参考表。
