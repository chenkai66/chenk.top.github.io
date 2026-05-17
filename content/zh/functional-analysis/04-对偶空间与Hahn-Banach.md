---
title: "泛函分析（四）：对偶空间与 Hahn-Banach 定理"
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
disableNunjucks: true
series_order: 4
series_total: 12
translationKey: "functional-analysis-4"
description: "Hahn-Banach 定理保证了足够多的泛函来分离点——对偶理论的基础。"
---

## 泛函分析（四）：对偶空间与 Hahn-Banach 定理

### 1. 对偶空间定义和例子

在泛函分析中，对偶空间是一个重要的概念。对于一个赋范线性空间 $X$，其对偶空间 $X^*$ 定义为所有从 $X$ 到 $\mathbb{R}$ 或 $\mathbb{C}$ 的有界线性泛函的集合，赋予算子范数。

![赋范空间中的 Hahn-Banach 分离](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/functional-analysis/04-dual-spaces-hahn-banach/fa_fig2_projection.png)


**定义 1.1**：设 $X$ 是一个赋范线性空间，则
$$ X^* = \{ f: X \to \mathbb{K} \mid f \text{ 是有界的且线性的} \}, $$
其中 $\mathbb{K}$ 可以是 $\mathbb{R}$ 或 $\mathbb{C}$。对偶空间 $X^*$ 也是一个赋范线性空间，其范数定义为
$$ \|f\|_{X^*} = \sup_{\|x\|_X \leq 1} |f(x)|. $$

**例题 1.1**：考虑实数域上的有限维空间 $\mathbb{R}^n$，其对偶空间 $(\mathbb{R}^n)^*$ 由所有形如
$$ f(x) = \sum_{i=1}^n a_i x_i, $$
的线性泛函组成，其中 $a_i \in \mathbb{R}$。容易验证，$(\mathbb{R}^n)^*$ 同构于 $\mathbb{R}^n$ 本身。

**证明思路**：通过直接计算可以验证每个 $f \in (\mathbb{R}^n)^*$ 都可以表示成上述形式，并且这种表示是唯一的。因此，$(\mathbb{R}^n)^*$ 和 $\mathbb{R}^n$ 之间存在一一对应的关系。

### 2. Hahn-Banach 解析形式（用 Zorn 引理证明）

Hahn-Banach 定理是泛函分析中的一个重要定理，它提供了将有界线性泛函从子空间延拓到整个空间的方法。

**定理 2.1 (Hahn-Banach 定理)**：设 $X$ 是一个实或复赋范线性空间，$M$ 是 $X$ 的一个子空间，$f \in M^*$ 是一个有界线性泛函。则存在 $F \in X^*$ 使得
$$ F|_M = f \quad \text{且} \quad \|F\|_{X^*} = \|f\|_{M^*}. $$

**证明思路**：利用 Zorn 引理构造一个极大线性独立集，然后通过逐步延拓来证明定理。具体来说，考虑所有包含 $M$ 的子空间 $N$ 上的有界线性泛函 $g$ 使得 $g|_M = f$ 并且 $\|g\|_{N^*} = \|f\|_{M^*}$。这些泛函构成一个偏序集，根据 Zorn 引理，该偏序集有一个极大元，这个极大元即为所求的 $F$。

### 3. 推论：分离、范数实现

Hahn-Banach 定理有许多重要的推论，其中包括分离定理和范数实现定理。

**推论 3.1 (分离定理)**：设 $X$ 是一个实或复赋范线性空间，$A$ 和 $B$ 是 $X$ 中的两个非空凸集，且 $A \cap B = \emptyset$。则存在一个有界线性泛函 $f \in X^*$ 和常数 $c$ 使得
$$ \sup_{x \in A} \operatorname{Re}(f(x)) < c < \inf_{y \in B} \operatorname{Re}(f(y)). $$

**推论 3.2 (范数实现定理)**：设 $X$ 是一个实或复赋范线性空间，$x \in X$ 且 $x \neq 0$。则存在 $f \in X^*$ 使得
$$ \|f\|_{X^*} = 1 \quad \text{且} \quad f(x) = \|x\|_X. $$

**证明思路**：这两个推论都可以通过 Hahn-Banach 定理直接得到。对于分离定理，可以构造适当的线性泛函并应用 Hahn-Banach 定理；对于范数实现定理，可以在单点生成的子空间上定义合适的泛函并延拓。

### 4. 几何 Hahn-Banach

几何 Hahn-Banach 定理是对 Hahn-Banach 定理的一个几何解释，它涉及凸集的分离性质。

**定理 4.1 (几何 Hahn-Banach 定理)**：设 $X$ 是一个实或复赋范线性空间，$A$ 是 $X$ 中的一个非空开凸集，$x_0 \notin A$。则存在一个超平面 $H$ 使得
$$ x_0 \notin H \quad \text{且} \quad A \subset H. $$

**证明思路**：通过构造适当的线性泛函并应用 Hahn-Banach 定理来分离点 $x_0$ 和凸集 $A$。具体来说，可以找到一个有界线性泛函 $f$ 使得 $f(x_0) > \sup_{x \in A} f(x)$，从而得到所需的超平面。

### 5. $l^p$ 和 $C[a,b]$ 的对偶

对于常见的函数空间，我们可以具体地描述它们的对偶空间。

**定理 5.1**：设 $1 \leq p < \infty$ 且 $\frac{1}{p} + \frac{1}{q} = 1$，则
$$ (l^p)^* \cong l^q. $$

**定理 5.2**：设 $C[a,b]$ 是闭区间 $[a,b]$ 上的连续函数空间，则
$$ (C[a,b])^* \cong M[a,b], $$
其中 $M[a,b]$ 是 $[a,b]$ 上的所有有界变差函数组成的空间。

**证明思路**：对于 $l^p$ 空间，可以通过 Hölder 不等式和 Riesz 表示定理来证明。对于 $C[a,b]$ 空间，可以通过 Riesz 表示定理和 Radon 测度来证明。

### 6. 自反性与双对偶

自反性是泛函分析中的一个重要概念，它涉及到对偶空间的对偶空间。

**定义 6.1**：设 $X$ 是一个赋范线性空间，称 $X$ 是自反的，如果自然映射
$$ J: X \to X^{**}, \quad J(x)(f) = f(x) $$
是满射。

**定理 6.1**：$L^p$ 空间 ($1 < p < \infty$) 是自反的，而 $L^1$ 和 $L^\infty$ 不是自反的。

**证明思路**：对于 $L^p$ 空间 ($1 < p < \infty$)，可以通过 Riesz 表示定理和 Hölder 不等式来证明。对于 $L^1$ 和 $L^\infty$，可以通过具体的反例来说明它们不是自反的。

### 7. 接下来

在接下来的内容中，我们将进一步探讨 Banach 空间的其他重要性质，包括弱收敛、Banach-Steinhaus 定理以及开映射定理。这些内容将进一步深化我们对泛函分析的理解，并为我们解决更复杂的问题提供有力的工具。

---

*本文是 [泛函分析](/zh/series/functional-analysis/) 系列的第 4 篇（共 12 篇）。*

*上一篇：[第 3 篇 —— Hilbert 空间](/zh/functional-analysis/03-Hilbert空间/)*

*下一篇：[第 5 篇 —— 弱拓扑](/zh/functional-analysis/05-弱拓扑/)*
