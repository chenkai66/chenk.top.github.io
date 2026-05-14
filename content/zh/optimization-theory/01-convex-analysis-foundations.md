---
title: '优化理论（一）：凸分析基础'
date: 2022-09-14 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
series: optimization-theory
series_order: 1
lang: zh
mathjax: true
description: "解锁本系列后续内容的几何与分析工具包：凸集、凸函数、共轭（Fenchel）变换、次梯度，以及示性函数/支撑函数对；包含詹森不等式、投影定理及基本范数次微分的完整证明。"
disableNunjucks: true
translationKey: "optim-01"
---
本文是本系列其余所有内容的基石。我们后续将证明的几乎所有结论——梯度下降法的收敛速率、拉格朗日对偶性、近端算子（proximal operator），乃至随机优化方法的分析——都依赖于关于凸集与凸函数的一小套基本事实。我们将从零开始，逐一推导全部结论。

若你仅记住本文中的三点，请务必牢记以下内容：

- 一个集合是**凸集**，当且仅当它包含其中任意两点之间的整条线段；一个函数是**凸函数**，当且仅当它的**上镜图（epigraph）** 是一个凸集。
- **共轭函数（conjugate）** $f^*$ 是广义化的勒让德变换（Legendre transform）：它将逐点不等式转化为线性不等式，是连接原始问题（primal problem）与对偶问题（dual problem）的关键桥梁。
- **次微分（subdifferential）** $\partial f(x)$ 是非光滑凸函数在点 $x$ 处“梯度”的恰当推广；只要 $x$ 位于 $\mathrm{dom}(f)$ 的相对内部（relative interior），$\partial f(x)$ 就非空。

## 你将学到什么
1. 凸集、凸包（convex hull）、投影定理（含严格证明）；
2. 凸函数及其四种等价刻画方式（定义、上镜图、一阶条件、二阶条件）；
3. 保持凸性的运算：逐点上确界（pointwise sup）、复合（composition）、透视变换（perspective）；
4. 共轭函数 $f^*$ 及芬切尔–杨不等式（Fenchel–Young inequality）；
5. 次梯度（subgradients）与次微分演算（subdifferential calculus）；
6. 具体计算示例：$\partial \|x\|_1$、$\partial \|x\|_2$、$\partial \max\{0, x\}$。

## 前置知识

线性代数（内积、范数）、基础实分析（极限、连续性、上确界）、多元微积分（梯度、Hessian 矩阵）。无需任何优化背景。

## 凸集

### 1 定义与基本例子

集合 $C \subseteq \mathbb{R}^n$ 称为**凸集**，若对任意 $x, y \in C$ 及任意 $\lambda \in [0, 1]$，均有  
$$
\lambda x + (1 - \lambda) y \in C.
$$
几何含义：连接 $C$ 中任意两点的线段完全落在 $C$ 内部。

以下例子应熟记于心：

- 仿射子空间 $\{x : Ax = b\}$ 与半空间 $\{x : a^\top x \leq b\}$；  
- 任意范数下的范数球 $\{x : \|x\| \leq r\}$；  
- 半正定锥 $\mathbf{S}^n_+ = \{X \in \mathbb{R}^{n \times n} : X = X^\top,\, X \succeq 0\}$；  
- 概率单形 $\Delta^{n-1} = \{p \in \mathbb{R}^n_+ : \sum_i p_i = 1\}$。

一个反直觉的例子：所有**可逆矩阵**构成的集合**不是凸集**。取 $X = I$ 与 $Y = -I$，其连线中点为零矩阵（不可逆）。

![凸集与非凸集对比：集合是凸集，当且仅当其中任意两点之间的线段都包含在集合内。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig1_convex_set.png)

### 2 保持凸性的运算

下列构造若以凸集为输入，则输出仍为凸集。每条均可直接由定义一步验证；你应能不查阅资料即写出证明。

| 运算                     | 命题                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| 交集                     | 若每个 $C_i$ 是凸集，则 $\bigcap_{i \in I} C_i$ 是凸集。              |
| 仿射像                   | $A C + b = \{Ax + b : x \in C\}$ 是凸集。                             |
| 笛卡尔积                 | $C_1 \times C_2$ 是凸集。                                             |
| 和集                     | $C_1 + C_2 = \{x + y : x \in C_1,\, y \in C_2\}$ 是凸集。             |
| 仿射原像（逆像）         | $\{x : Ax + b \in C\}$ 是凸集。                                       |

其中交集规则在实践中最为常用——它解释了为何多面体 $\{x : Ax \leq b\}$ 是凸集（它是若干半空间的交），也说明了凸优化问题的可行域必为凸集。

### 3 投影定理

我们将反复使用如下重要定理：

> **投影定理**．设 $C \subseteq \mathbb{R}^n$ 是非空闭凸集，$y \in \mathbb{R}^n$。则存在唯一一点 $\pi_C(y) \in C$，使得 $\|x - y\|_2$ 在 $x \in C$ 上取得最小值。进一步地，$z = \pi_C(y)$ 当且仅当  
> $$\langle y - z,\, x - z \rangle \leq 0 \quad \text{对所有 } x \in C \text{ 成立}。$$
**存在性证明**．令 $d = \inf_{x \in C} \|x - y\|_2$，并取序列 $\{x_k\} \subseteq C$ 满足 $\|x_k - y\|_2 \to d$。我们证明 $\{x_k\}$ 是 Cauchy 列。对向量 $x_k - y$ 与 $x_m - y$ 应用平行四边形恒等式：  
$$
\|x_k - x_m\|_2^2 = 2 \|x_k - y\|_2^2 + 2 \|x_m - y\|_2^2 - 4 \left\| \tfrac{x_k + x_m}{2} - y \right\|_2^2.
$$
由于 $C$ 是凸集，故 $\frac{x_k + x_m}{2} \in C$，从而 $\|\frac{x_k + x_m}{2} - y\|_2 \geq d$。因此  
$$
\|x_k - x_m\|_2^2 \leq 2 \|x_k - y\|_2^2 + 2 \|x_m - y\|_2^2 - 4 d^2 \to 0,
$$
当 $k,m \to \infty$ 时成立。于是 $\{x_k\}$ 收敛于某点 $z$；又因 $C$ 是闭集，故 $z \in C$，且满足 $\|z - y\|_2 = d$。

**唯一性证明**．假设 $z_1$ 与 $z_2$ 均达到最小值。在上述平行四边形恒等式中取 $x_k = z_1$、$x_m = z_2$，得  
$$
\|z_1 - z_2\|_2^2 \leq 2 d^2 + 2 d^2 - 4 d^2 = 0,
$$
故 $z_1 = z_2$。

**变分不等式证明**．设 $z = \pi_C(y)$。对任意 $x \in C$ 及 $\lambda \in (0, 1]$，点 $z + \lambda (x - z) = (1 - \lambda) z + \lambda x$ 属于 $C$（由凸性），故  
$$
\|y - z\|_2^2 \leq \|y - z - \lambda (x - z)\|_2^2 = \|y - z\|_2^2 - 2 \lambda \langle y - z,\, x - z \rangle + \lambda^2 \|x - z\|_2^2.
$$
整理并两边除以 $\lambda$ 得：  
$$
2 \langle y - z,\, x - z \rangle \leq \lambda \|x - z\|_2^2.
$$
令 $\lambda \to 0^+$ 即得所需不等式。反之，若该不等式对所有 $x \in C$ 成立，展开 $\|x - y\|_2^2 = \|(x - z) - (y - z)\|_2^2 \geq \|y - z\|_2^2$，即可推出 $z = \pi_C(y)$。$\blacksquare$

投影定理具有优美的几何解释：$\pi_C(y)$ 是 $C$ 中使得线段 $y \to z$ 与 $C$ 内任一方向夹角均不超过 $90^\circ$ 的唯一点。

![点 0 到闭凸集 1 的投影：2 是唯一的最近点，残量 3 与指向 4 内部的任意方向 5 的夹角均不小于 6。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig2_projection.png)

## 凸函数

### 1 四种等价刻画

设 $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$，其**有效定义域**为 $\mathrm{dom}(f) = \{x : f(x) < +\infty\}$，并假设该集合是凸集。以下四条性质彼此等价，后续我们将不加区分地使用它们：

**定义式（凸性原始定义）**：对任意 $x, y \in \mathrm{dom}(f)$ 及 $\lambda \in [0, 1]$，
$$
f(\lambda x + (1 - \lambda) y) \leq \lambda f(x) + (1 - \lambda) f(y). \tag{D}
$$
**上镜图刻画**：集合 $\mathrm{epi}(f) = \{(x, t) \in \mathbb{R}^{n+1} : f(x) \leq t\}$ 是 $\mathbb{R}^{n+1}$ 中的凸集。

**一阶条件（要求 $f$ 可微）**：对任意 $x, y \in \mathrm{dom}(f)$，
$$
f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle. \tag{F1}
$$
**二阶条件（要求 $f$ 二阶可微）**：对任意 $x \in \mathrm{int}(\mathrm{dom}(f))$，
$$
\nabla^2 f(x) \succeq 0. \tag{F2}
$$
**(D) 与上镜图刻画的等价性**：对任意 $(x, s), (y, t) \in \mathrm{epi}(f)$，点 $(\lambda x + (1 - \lambda) y,\, \lambda s + (1 - \lambda) t)$ 属于 $\mathrm{epi}(f)$ 当且仅当不等式 (D) 成立（取 $s = f(x),\, t = f(y)$ 即得）。

**(D) $\Rightarrow$ (F1)**：将 (D) 式改写为  
$$
\frac{f(x + \lambda (y - x)) - f(x)}{\lambda} \leq f(y) - f(x).
$$
令 $\lambda \to 0^+$，左边趋于方向导数 $\langle \nabla f(x), y - x \rangle$。

**(F1) $\Rightarrow$ (F2)**：令 $y = x + t v$，其中 $t > 0$ 很小、$v$ 为任意向量。则 (F1) 给出  
$$
f(x + tv) \geq f(x) + t \langle \nabla f(x), v \rangle.
$$
由 Taylor 展开 $f(x + tv) = f(x) + t \langle \nabla f(x), v \rangle + \tfrac{t^2}{2} v^\top \nabla^2 f(x) v + o(t^2)$，比较两边可得 $v^\top \nabla^2 f(x) v \geq 0$。

**(F2) $\Rightarrow$ (D)**：沿直线积分两次。具体地，定义 $g(\lambda) = f((1 - \lambda) x + \lambda y)$，则  
$$
g''(\lambda) = (y - x)^\top \nabla^2 f((1 - \lambda) x + \lambda y) (y - x) \geq 0,
$$
故 $g$ 在 $[0,1]$ 上是凸函数，这正是 (D)。

![凸性的两种等价视角：（左）一阶条件——任意点处的切线都是 0 的全局下界；（右）上镜图 1 本身是 2 中的凸集。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig3_convex_function.png)

### 2 严格凸性与强凸性

- **严格凸函数**：当 $x \neq y$ 且 $\lambda \in (0,1)$ 时，(D) 中不等式严格成立。
- **$\mu$-强凸函数**（$\mu > 0$）：函数 $f - \frac{\mu}{2} \|x\|_2^2$ 是凸函数。等价地，
  $$
  f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2} \|y - x\|_2^2.
  $$
强凸性之于凸性，正如“一致连续”之于“连续”——它提供了可量化的间隙（quantitative gap），是使优化算法收敛速率具象化的核心条件。我们将在第 02 篇文章中深入剖析其含义与作用。

### 3 典型例子

你应能借助二阶条件（或直接验证）确认下列函数的凸性：

| $f(x)$                        | 是否凸？ | 理由                                                                 |
| ----------------------------- | -------- | -------------------------------------------------------------------- |
| $\|x\|_p$（$p \geq 1$）        | 是       | 三角不等式 + 正齐次性。                                              |
| $\log(1 + e^x)$（softplus）   | 是       | $f''(x) = \dfrac{e^x}{(1 + e^x)^2} > 0$。                            |
| $-\log \det X$（定义在 $\mathbf{S}^n_{++}$ 上） | 是 | Hessian 为 $X^{-1} \otimes X^{-1}$，半正定。                         |
| $x \log x$（定义在 $\mathbb{R}_+$ 上） | 是    | $f''(x) = 1/x > 0$。                                                 |
| $x^4$                         | 严格凸   | $f''(x) = 12 x^2 \geq 0$，仅在 $x = 0$ 处为零（非强凸）。             |
| $\frac{1}{2} x^\top Q x$（$Q \succ 0$） | $\lambda_{\min}(Q)$-强凸 | $\nabla^2 f = Q$。                             |

### 4 保持凸性的运算

| 运算                                     | 是否保持凸性？                                                   |
| ------------------------------------------ | ---------------------------------------------------------------- |
| 非负加权和                                 | 是（$\sum_i w_i f_i$，其中 $w_i \geq 0$）。                        |
| 逐点上确界                                 | $\sup_{i \in I} f_i$ 是凸函数（其上镜图是各 $f_i$ 上镜图的交集）。     |
| 与仿射映射复合                             | $g(x) = f(Ax + b)$ 继承凸性。                                      |
| 复合函数 $g(x) = h(f(x))$                  | 若 $h$ 凸且非减、$f$ 凸，则 $g$ 凸。                                |
| 射影变换（Perspective）$g(x, t) = t f(x/t)$（$t > 0$） | 是。                                                          |
| 偏最小化 $g(x) = \inf_y f(x, y)$           | 若 $f$ 关于 $(x,y)$ 联合凸，则 $g$ 凸。                              |

逐点上确界法则堪称“秘密武器”：它解释了为何**支撑函数** $\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle$ 对任意集合 $C$ 恒为凸函数，也说明了凸函数的最大值仍是凸函数。

## 共轭函数

对任意函数 $f : \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$（未必凸），定义其**共轭函数**（或称**Legendre–Fenchel 变换**）为：
$$
f^*(y) = \sup_{x \in \mathbb{R}^n} \big[ \langle y, x \rangle - f(x) \big].
$$
无论 $f$ 是否凸，其共轭 $f^*$ **恒为凸函数**——因为它是关于 $y$ 的一族仿射函数的逐点上确界。

### 1 几何解释

对固定斜率 $y$，$f^*(y)$ 表示 $\langle y, x \rangle - f(x)$ 关于 $x$ 能取到的最大值。等价地，仿射函数 $x \mapsto \langle y, x \rangle - f^*(y)$ 是斜率为 $y$、且位于 $f$ 下方的最高仿射下界（affine minorant）。因此，$f^*$ 刻画了：对每个可能的斜率 $y$，对应的支持超平面（supporting hyperplane）距离函数图像下方有多远。

![共轭函数的几何含义：对斜率 0，1 下方斜率为 2 的最高仿射下界为 3；4 即该直线与 5 轴交点到原点的纵向距离。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig4_conjugate.png)

### 2 Fenchel–Young 不等式

由定义直接可得：
$$
f(x) + f^*(y) \geq \langle x, y \rangle. \tag{FY}
$$
等号成立当且仅当 $y \in \partial f(x)$（该次梯度概念将在后文引入）。这是凸分析中 AM-GM 不等式与 Young 不等式的统一推广；例如取 $f(x) = \frac{1}{p} |x|^p$（其中 $p > 1$），计算其共轭可精确导出经典 Young 不等式 $\frac{|x|^p}{p} + \frac{|y|^q}{q} \geq xy$，其中 $\frac{1}{p} + \frac{1}{q} = 1$。

### 3 常见共轭函数示例

请熟记以下结果——它们将作为后续各讲中的基本构件反复使用。

| $f(x)$                                            | $f^*(y)$                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| $\frac{1}{2} \|x\|_2^2$                           | $\frac{1}{2} \|y\|_2^2$（自共轭）。                                        |
| $\frac{1}{2} x^\top Q x$, $Q \succ 0$             | $\frac{1}{2} y^\top Q^{-1} y$。                                             |
| 指示函数 $\iota_C$                                | 支撑函数 $\sigma_C(y) = \sup_{x \in C} \langle y, x \rangle$。              |
| $\|x\|$（任一范数）                               | $\iota_{B^\circ}(y) = \begin{cases} 0 & \|y\|_* \leq 1 \\ +\infty & \text{否则} \end{cases}$，其中 $\|\cdot\|_*$ 为对偶范数。 |
| $e^x$                                             | $y \log y - y$（定义域 $y > 0$，具熵型结构）。                              |
| $\log(1 + e^x)$                                   | $y \log y + (1 - y) \log(1 - y)$（定义域 $y \in [0, 1]$）。                 |
| $-\log x$（定义在 $x > 0$ 上）                    | $-1 - \log(-y)$（定义在 $y < 0$ 上）。                                      |

范数与其对偶单位球指示函数构成的共轭对，完整解释了我们在第 06 篇文章中展开的 LASSO 对偶性理论。

### 4 双共轭及“凸性”即对 $(f^*)^*$ 封闭

双共轭函数 $f^{**} = (f^*)^*$ 恒满足逐点不等式 $f^{**} \leq f$。当且仅当 $f$ 是凸函数且下半连续时，二者相等，即 $f^{**} = f$。因此，“取共轭两次”这一运算，恰好给出 $f$ 下方的最小凸闭函数——即 $f$ 的**凸包络**（convex envelope）。正因如此，人们有时将凸松弛（convex relaxation）直接写作 $f^{**}$。

## 次梯度

### 1 次微分

对凸函数 $f$，其在点 $x$ 处的**次微分**定义为  
$$
\partial f(x) = \{g \in \mathbb{R}^n : f(y) \geq f(x) + \langle g, y - x \rangle \text{ 对所有 } y\}.
$$
属于 $\partial f(x)$ 的向量 $g$ 称为 $f$ 在 $x$ 处的一个**次梯度**。与条件 (F1) 对比：当 $f$ 可微时，$\partial f(x) = \{\nabla f(x)\}$，即为单点集。

当 $f$ 不可微时，$\partial f(x)$ 可能是非单点集。次梯度的核心意义在于：*任意* 凸函数在其定义域的每个内点（更准确地说，相对内点）处都存在次梯度——即使该点处无经典导数；且“$g = 0$”这一条件取代了“$\nabla f = 0$”，成为最优性判据：  
$$
x^\star \in \arg\min f \iff 0 \in \partial f(x^\star).
$$
### 2 存在性与基本微分法则

> **定理**．若 $f$ 是凸函数，且 $x \in \mathrm{relint}(\mathrm{dom}(f))$，则 $\partial f(x)$ 非空。

该结论源于凸集 $\mathrm{epi}(f)$ 在点 $(x, f(x))$ 处存在支撑超平面（支撑超平面定理）。当 $x$ 位于定义域边界时，$\partial f(x)$ 可能为空集——例如，$f(x) = -\sqrt{x}$ 定义在 $[0, \infty)$ 上，则 $\partial f(0) = \emptyset$。

以下微分运算法则构成了近端方法（proximal methods）的基石：

| 法则                              | 表述                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| 和法则                            | 若 $\mathrm{relint}\,\mathrm{dom}\,f \cap \mathrm{relint}\,\mathrm{dom}\,g \neq \emptyset$，则 $\partial(f + g)(x) = \partial f(x) + \partial g(x)$。 |
| 仿射复合                          | 若 $\mathrm{relint}\,\mathrm{dom}\,f \cap \mathrm{range}(A) \neq \emptyset$，则 $\partial(f \circ A)(x) = A^\top \partial f(Ax)$。 |
| 逐点最大值                         | $\partial \max_i f_i(x) = \mathrm{conv} \bigcup_{i \in I(x)} \partial f_i(x)$，其中 $I(x) = \{i : f_i(x) = \max_j f_j(x)\}$。 |
| 共轭等价性                        | $g \in \partial f(x) \iff x \in \partial f^*(g) \iff f(x) + f^*(g) = \langle x, g \rangle$。 |

### 3 具体示例

**例 1：$f(x) = |x|$ 在 $\mathbb{R}$ 上。**  
$$
\partial f(x) = \begin{cases} \{1\} & x > 0 \\ \{-1\} & x < 0 \\ [-1, 1] & x = 0. \end{cases}
$$
在 $x = 0$ 处，任意斜率介于 $-1$ 与 $1$ 之间的直线均为 $|x|$ 的支撑线，且整体位于其下方。

![0 在尖点 1 处的次梯度集合：2 中的每个斜率都对应一条过原点且位于 3 下方的支撑直线，故 4。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/01-convex-analysis-foundations/fig5_subgradient.png)

**例 2：$f(x) = \|x\|_1$ 在 $\mathbb{R}^n$ 上。**  

按分量独立处理：$g \in \partial \|\cdot\|_1(x)$ 当且仅当  
$$
g_i = \begin{cases} \mathrm{sign}(x_i) & x_i \neq 0 \\ \in [-1, 1] & x_i = 0. \end{cases}
$$
这正是 ISTA 算法利用以生成稀疏解的结构。

**例 3：$f(x) = \|x\|_2$ 在 $\mathbb{R}^n$ 上。**  
$$
\partial \|\cdot\|_2(x) = \begin{cases} \{x / \|x\|_2\} & x \neq 0 \\ \{g : \|g\|_2 \leq 1\} & x = 0. \end{cases}
$$
**例 4：合页损失（hinge loss）$f(x) = \max\{0, 1 - x\}$。**  
$$
\partial f(x) = \begin{cases} \{0\} & x > 1 \\ \{-1\} & x < 1 \\ [-1, 0] & x = 1. \end{cases}
$$
点 $x = 1$ 是“拐点”（kink）——在此间隔边界上，损失函数可被赋予任意斜率 $[-1, 0]$ 中的值，这正是支撑 SVM 对偶理论成立的关键。

### 4 最优性：从 $\nabla f = 0$ 到 $0 \in \partial f$

我们将最常依赖的核心结论是：若 $f$ 是凸函数，则  
$$
x^\star \in \arg\min f \iff 0 \in \partial f(x^\star).
$$
正向推导直接由次梯度定义 (D) 应用于 $y = x^\star$ 与任意 $x$ 得到；反向推导中，$0 \in \partial f(x^\star)$ 意味着对所有 $y$ 均有 $f(y) \geq f(x^\star)$。

对于带约束的优化问题 $\min_{x \in C} f(x)$，其中 $C$ 是闭凸集、$f$ 是凸函数，最优性条件推广为：  
$$
0 \in \partial f(x^\star) + N_C(x^\star),
$$
其中 $N_C(x^\star) = \{g : \langle g, y - x^\star \rangle \leq 0 \ \forall y \in C\}$ 称为集合 $C$ 在 $x^\star$ 处的**法锥**（normal cone）。我们将在第 08 篇文章中结合 KKT 条件再次深入讨论此式。

## 综合应用：一个完整求解示例

考虑 LASSO 问题：
$$
\min_x F(x) := \tfrac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1,
$$
其中 $A \in \mathbb{R}^{m \times n}$，$b \in \mathbb{R}^m$，$\lambda > 0$。

第一项是凸函数（其 Hessian 矩阵 $A^\top A$ 半正定，故为凸二次函数）；第二项也是凸函数（范数恒为凸）。因此 $F$ 是凸函数。由最优性条件可得：
$$
0 \in A^\top (Ax^\star - b) + \lambda \, \partial \|\cdot\|_1(x^\star).
$$
逐分量展开，记残差 $r = A^\top (Ax^\star - b)$，则有：
$$
r_i = -\lambda \, \mathrm{sign}(x^\star_i) \quad \text{若 } x^\star_i \neq 0, \qquad r_i \in [-\lambda, \lambda] \quad \text{若 } x^\star_i = 0.
$$
这正是著名的 **LASSO 的 KKT 条件**——它表明：对任意坐标 $i$，只要 $|r_i| < \lambda$，就必有 $x^\star_i = 0$。这正是 $\ell_1$ 正则化诱导稀疏性的根本原因。我们在第 06 篇文章中将通过软阈值算子的不动点重新推导该条件。

---

## 总结

| 概念              | 它赋予你的能力                                                                 |
| ----------------- | ------------------------------------------------------------------------------ |
| 凸集              | 清晰地讨论可行性与投影问题。                                                    |
| 凸函数            | 快速验证凸性：任选四种等价刻画之一即可。                                           |
| 共轭函数 $f^*$    | 在原始空间与对偶空间之间自由切换；导出 Young 型不等式；构建对偶问题。                    |
| 次梯度            | 当 $f$ 不可微时替代 $\nabla f$；将最优性条件简洁表述为 $0 \in \partial f$。         |
| 法锥（Normal cone）| 在约束定义域上表述最优性；架起通向 KKT 条件的桥梁。                                 |

下一篇文章将在此基础上引入两个关键假设——Lipschitz 光滑性与强凸性——从而为梯度下降法提供**定量的收敛速率保证**。

## 故事的后续发展

- 第 02 篇文章在上述基础上引入光滑性与强凸性，并在三种不同设定下证明梯度下降法（GD）的收敛性。
- 第 06 篇文章利用次梯度与共轭函数，系统推导 ISTA、FISTA 与 ADMM 等经典算法。
- 第 08 篇文章将最优性条件 $0 \in \partial f$ 推广至带约束的问题，导出完整的 KKT 系统。

## 参考文献

- Boyd & Vandenberghe，《Convex Optimization》，第 2–3 章——关于凸集与凸函数的标准权威参考。
- Rockafellar，《Convex Analysis》——更深入的专著；所有关于次梯度与共轭函数的理论根基尽在于此。
- Bertsekas，《Convex Optimization Theory》，第 1 章——简明扼要、自成体系。
- Nesterov，《Lectures on Convex Optimization》，第 1 章——聚焦现代优化算法视角。
