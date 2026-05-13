---
title: '优化理论（十）：随机优化与方差缩减'
date: 2022-09-27 09:00:00
tags:
  - ML
  - Optimization
  - Stochastic Methods
categories: Algorithm
series: optimization-theory
series_order: 10
lang: zh
mathjax: true
description: 'SGD 为何有效？我们基于梯度噪声预算证明了其在凸函数下的 $O(1/\sqrt{T})$ 收敛率与强凸函数下的 $O(1/(\mu T))$ 收敛率；进而介绍方差缩减方法——SVRG、SAGA、Katyusha，它们利用随机样本达到全梯度下降的线性收敛速率，并完整解析其理论机理。'
disableNunjucks: true
translationKey: "optim-10"
---
对于有限和优化问题  
$$
\min_x f(x) := \frac{1}{n} \sum_{i=1}^n f_i(x),
$$  
**确定性梯度下降**（full GD）每步计算代价为 $O(n)$，但收敛步数为 $O(\kappa \log(1/\epsilon))$。**随机梯度下降**（SGD）每步仅需 $O(1)$ 计算量，但在凸情形下收敛步数为 $O(1/\epsilon^2)$，强凸情形下为 $O(\kappa^2 \log(1/\epsilon))$。究竟哪种更快，取决于 $n$、条件数 $\kappa$ 和精度 $\epsilon$。

一类现代算法——**方差缩减型 SGD**（variance-reduced SGD）——在仅使用随机采样的前提下，达到了与确定性方法相当的收敛速率 $O((n + \kappa) \log(1/\epsilon))$。它们填补了关键空白，使得随机方法在有限和问题上严格优于全梯度下降。

本文内容如下：

1. 从「噪声预算」（noise budget）视角推导基础 SGD 的收敛速率；  
2. 解释小批量（mini-batching）与学习率衰减（learning rate decay）的作用；  
3. 推导 SVRG 算法，并证明其在强凸目标函数上的线性收敛性；  
4. 简述 SAGA 与 Katyusha，并介绍催生这些算法的下界结果。

## 你将学到

1. SGD 的两类收敛行为：凸情形下的 $O(1/\sqrt{T})$ 速率，以及强凸情形下的 $O(1/T)$ 速率；  
2. SGD 的方差控制界，为何步长 $\eta = 1/L$ 过于激进，以及 $\eta_t = 1/(\mu t)$ 这一调度策略的来源；  
3. SVRG 算法及其线性收敛性；  
4. SAGA、Katyusha，以及下界结果 $\Omega((n + \sqrt{n \kappa}) \log(1/\epsilon))$；  
5. 实践考量：小批量、动量（momentum），以及 SGD 与方差缩减方法各自适用的场景。

## 前置知识

第 02 篇（光滑性、强凸性）、第 03 篇（梯度下降与 SGD）。

---

## 1. SGD 框架

在每轮迭代 $t$，SGD 从 $\{1, \ldots, n\}$ 中均匀随机采样一个索引 $i_t$，并执行更新：
$$
x_{t+1} = x_t - \eta_t \nabla f_{i_t}(x_t).
$$
关于随机梯度 $\nabla f_{i_t}(x_t)$，有两个基本事实：

- **无偏性**（Unbiased）：$\mathbb{E}[\nabla f_{i_t}(x_t) \mid x_t] = \nabla f(x_t)$；  
- **有界方差**（Bounded variance）：通常假设  
  $$
  \mathbb{E}\big[ \|\nabla f_{i_t}(x_t) - \nabla f(x_t)\|_2^2 \mid x_t \big] \leq \sigma^2,
  $$  
  其中 $\sigma^2$ 称为**梯度方差预算**（gradient variance budget）。

这两条假设（无偏性 + 有界方差）构成了 SGD 的公理基础。由此导出的收敛界强度，取决于目标函数 $f$ 所具备的额外结构（如凸性、强凸性、光滑性等）。

![病态二次函数上 SGD 与全梯度下降的轨迹对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig1.png)
*全梯度下降沿确定性路径平稳下降；SGD 则在同方向上呈现噪声锯齿轨迹。每步 SGD 的扰动幅度正由噪声预算 $\sigma^2$ 所控制。*

---

## 2. 凸情形收敛率：$O(1/\sqrt{T})$

> **定理**。设 $f$ 是凸函数，且满足上述方差界。取常数步长 $\eta = R / (\sigma \sqrt{T})$，并从满足 $\|x_0 - x^\star\|_2 \leq R$ 的初始点 $x_0$ 出发，则经过 $T$ 轮迭代后，有  
> $$
> \mathbb{E}[f(\bar x_T) - f^\star] \leq \frac{R \sigma}{\sqrt{T}},
> $$  
> 其中 $\bar x_T = \frac{1}{T} \sum_{t=0}^{T-1} x_t$ 为迭代点的运行平均。

**证明**。对 $x_t$ 取条件期望：  
$$
\mathbb{E}\|x_{t+1} - x^\star\|_2^2 = \|x_t - x^\star\|_2^2 - 2 \eta \langle \nabla f(x_t), x_t - x^\star \rangle + \eta^2 \mathbb{E}\|\nabla f_{i_t}(x_t)\|_2^2.
$$  
由凸性，$\langle \nabla f(x_t), x_t - x^\star \rangle \geq f(x_t) - f^\star$；由方差界，  
$$
\mathbb{E}\|\nabla f_{i_t}(x_t)\|_2^2 = \|\nabla f(x_t)\|_2^2 + \mathbb{E}\|\nabla f_{i_t}(x_t) - \nabla f(x_t)\|_2^2 \leq \|\nabla f(x_t)\|_2^2 + \sigma^2.
$$  
暂设 $\|\nabla f\|_2 \leq G$（即 $f$ 是 $G$-Lipschitz 连续的），则 $\mathbb{E}\|\nabla f_{i_t}\|_2^2 \leq G^2 + \sigma^2$。于是  
$$
\mathbb{E}\|x_{t+1} - x^\star\|_2^2 \leq \|x_t - x^\star\|_2^2 - 2 \eta (f(x_t) - f^\star) + \eta^2 (G^2 + \sigma^2).
$$  
整理后对 $t = 0, \ldots, T-1$ 求和（利用望远镜求和）：  
$$
\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f^\star] \leq \frac{R^2}{2 \eta} + \frac{T \eta (G^2 + \sigma^2)}{2}.
$$  
两边同除以 $T$，再对 $\bar x_T$ 应用 Jensen 不等式：  
$$
\mathbb{E}[f(\bar x_T) - f^\star] \leq \frac{R^2}{2 \eta T} + \frac{\eta (G^2 + \sigma^2)}{2}.
$$  
令 $\eta = R / \sqrt{T (G^2 + \sigma^2)}$ 可得最优界：  
$$
\mathbb{E}[f(\bar x_T) - f^\star] \leq R \sqrt{(G^2 + \sigma^2) / T}.
$$  
若将 $G^2 + \sigma^2$ 简化为 $\sigma^2$（或视 $G$ 为噪声的一部分），即得前述形式。$\blacksquare$

该 $O(1/\sqrt{T})$ 速率是**经典 SGD 的标准收敛速率**。注意它只依赖于方差 $\sigma^2$，而与函数的光滑常数 $L$ 或条件数 $\kappa$ 无关。因此，只要步长选取恰当，SGD 对噪声具有鲁棒性，但收敛速度较慢。

---

## 3. 强凸情形下的收敛速率：$O(1/T)$

> **定理**。假设 $f$ 是 $\mu$-强凸函数，且满足方差界 $\mathbb{E}[\|\nabla f_i(x) - \nabla f(x)\|_2^2] \leq \sigma^2$。取步长 $\eta_t = 2 / (\mu (t + 1))$，则经过 $T$ 次迭代后，
> $$
> \mathbb{E}[\|x_T - x^\star\|_2^2] \leq \frac{4 \sigma^2}{\mu^2 T}.
> $$

**证明概要**。令 $a_t = \mathbb{E}[\|x_t - x^\star\|_2^2]$。利用强凸性不等式 $\langle \nabla f(x_t), x_t - x^\star \rangle \geq \mu \|x_t - x^\star\|_2^2$，并结合方差控制，可得如下递推关系：
$$
a_{t+1} \leq (1 - 2 \eta_t \mu) a_t + \eta_t^2 \sigma^2 + \eta_t^2 L^2 a_t,
$$
其中最后一项源于对梯度模长的界：对 $L$-光滑且强凸的 $f$，有 $\|\nabla f(x_t)\|_2 \leq L \|x_t - x^\star\|_2$。当 $\eta_t = 2/(\mu(t+1))$ 足够小时，$L^2$ 项可忽略，通过归纳法即得 $a_t = O(1/(\mu^2 t))$。

最优步长按 $1/t$ 衰减——这正是 Robbins–Monro（1951）提出的经典步长调度方案，也是所有现代自适应 SGD 步长策略（如 AdaGrad、Adam）的理论基础。

### 3.1 为何在强凸函数上 SGD 不能使用常数步长？

若采用常数步长 $\eta_t = \eta$，上述递推式存在不动点 $a^\star = \eta \sigma^2 / (2 \mu)$。此时迭代点 $x_t$ 并不收敛至最优解 $x^\star$，而是收敛到一个以 $x^\star$ 为中心、半径为 $O(\sqrt{\eta \sigma^2 / \mu})$ 的「噪声球」内。若希望将该球半径压缩至 $\epsilon$，需设 $\eta = O(\epsilon \mu / \sigma^2)$，进而总迭代次数为 $T = O(\sigma^2 / (\epsilon \mu^2))$ —— 其对 $\epsilon$ 的依赖（$1/\epsilon$）虽与衰减步长 SGD 相同，但需针对每个目标精度 $\epsilon$ 手动重设 $\eta$。

在深度学习中，我们实际上并不追求精确收敛至 $x^\star$：泛化误差通常在优化间隙仍较大时便已停止下降；因此常数步长完全可行。这是经典优化理论与深度学习实践之间的一个关键结构性差异。

---

## 4. 小批量（Mini-batching）：方差随批大小线性衰减

若每步采样大小为 $B$ 的小批量，并对 $B$ 个随机梯度取平均：
$$
g_t = \frac{1}{B} \sum_{j=1}^B \nabla f_{i_{t,j}}(x_t),
$$
则其方差降为 $\sigma^2 / B$（假设各样本独立）。换言之，小批量线性地降低了噪声预算。

相应地，凸优化速率提升为 $O(\sigma / \sqrt{TB})$，比 $B = 1$ 的标准 SGD 快 $\sqrt{B}$ 倍。但每步计算代价也增加为 $B$ 倍的梯度评估量。为达到精度 $\epsilon$ 所需的总梯度评估次数为：
$$
\text{grads} = TB = O(\sigma^2 B / \epsilon^2).
$$
该量随 $B$ 线性增长——因此小批量本身**并不节省总计算量**！其实际优势在于更大的批次更易于在 GPU 上高效并行化。

**线性缩放律**（Goyal 等，2017）——「批大小扩大 $k$ 倍，学习率同步扩大 $k$ 倍」——正源于此分析：噪声项 $\eta^2 \sigma^2 / B$ 在 $\eta \propto B$ 时保持恒定，故更大批次允许采用更大步长。但该规律仅在「临界批大小」（critical batch size）以内成立；超过该阈值后，噪声不再是最主要瓶颈（McCandlish 等，2018）。

![小批量方差与临界批大小](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig4.png)
*左：梯度方差按 $\sigma^2/B$ 的速率随批大小线性下降（log-log 坐标）。右：线性缩放律允许有效步长随 $B$ 同比例增长——但仅到临界批大小 $B^\star$ 为止，超过后加速比饱和，因为此时梯度信号本身（而非噪声）成为瓶颈。*

---

## 5. 方差缩减：SVRG

只要我们仅用单个 $\nabla f_{i_t}$ 作为梯度估计，SGD 的方差项 $\sigma^2$ 就不可避免。**方差缩减（variance reduction）** 引入额外的控制变量（control variates）——即额外的计算开销，以在极限下将估计方差降至零。

### 5.1 SVRG 算法

（随机方差缩减梯度法，Stochastic Variance-Reduced Gradient，Johnson & Zhang，2013）

```text
SVRG（epoch 长度为 m，学习率为 η）：
初始化 w̃_0
for s = 0, 1, 2, ...:                        # 外层 epoch
    g̃_s = ∇f(w̃_s) = (1/n) Σ ∇f_i(w̃_s)        # 在快照点 w̃_s 处计算全梯度
    x_0 = w̃_s
    for t = 0, ..., m-1:                     # 内层迭代
        均匀采样 i_t ∈ {1, ..., n}
        g_t = ∇f_{i_t}(x_t) - ∇f_{i_t}(w̃_s) + g̃_s
        x_{t+1} = x_t - η g_t
    w̃_{s+1} = x_m  （或在该 epoch 中随机选取某个 x_t）
```

核心在于该**梯度估计器**
$$
g_t = \nabla f_{i_t}(x_t) - \nabla f_{i_t}(\tilde w_s) + \tilde g_s.
$$
其性质如下：

- **无偏性**：$\mathbb{E}[g_t \mid x_t] = \nabla f(x_t) - \nabla f(\tilde w_s) + \nabla f(\tilde w_s) = \nabla f(x_t)$。
- **最优解附近方差趋零**：当 $\tilde w_s, x_t \to x^\star$ 时，有 $\nabla f_{i_t}(x_t) - \nabla f_{i_t}(\tilde w_s) \to 0$，故噪声消失。

正是这一特性赋予了 SVRG 线性收敛速率。

![同一点处 SGD 与 SVRG 的随机梯度样本对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig3.png)
*每条浅色箭头是一次随机梯度采样，加粗蓝色箭头表示真实梯度 $\nabla f(x)$。SGD（橙色）样本在均值附近大幅散布；SVRG（绿色）样本紧密聚集——控制变量 $-\nabla f_{i_t}(\tilde w_s) + \tilde g_s$ 抵消了大部分方差。*

### 5.2 SVRG 收敛性分析

> **定理（Johnson–Zhang，2013）**：假设每个 $f_i$ 是 $L$-光滑的，且 $f$ 是 $\mu$-强凸的。取步长 $\eta = \frac{1}{10 L}$，并令 epoch 长度 $m$ 足够大（具体地，$m \geq 100 L / \mu$），则 SVRG 几何收敛：
> $$
> \mathbb{E}[f(\tilde w_{s+1}) - f^\star] \leq 0.5 \cdot \mathbb{E}[f(\tilde w_s) - f^\star].
> $$

**证明概要**：$g_t$ 的方差满足  
$$
\mathbb{E}\|g_t - \nabla f(x_t)\|_2^2 \leq L (f(x_t) - f^\star) + L (f(\tilde w_s) - f^\star).
$$  
此即**共轭光滑性（co-coercivity）引理**。将其代入标准 SGD 分析框架（参见第 2 节），但将此处所得的 $\sigma^2$ 上界代入，并仔细追踪一个 SVRG epoch 的全过程，即可导出关于 $f(\tilde w_s) - f^\star$ 的收缩不等式。

### 5.3 总计算代价

每个 SVRG epoch 消耗 $n + m$ 次梯度计算（$n$ 次用于快照，$m$ 次用于内层迭代）。达到精度 $\epsilon$ 所需 epoch 数为 $O(\log(1/\epsilon))$。总梯度计算次数为  
$$
O\big((n + L/\mu) \log(1/\epsilon)\big) = O\big((n + \kappa) \log(1/\epsilon)\big).
$$

对比其他方法：

- **全梯度下降（Full GD）**：$O(n \kappa \log(1/\epsilon))$ —— $n$ 与 $\kappa$ 相乘；
- **SGD**：$O(\kappa^2 / \epsilon)$ —— 对于小 $\epsilon$，可能远差于 SVRG；
- **SVRG**：$O\big((n + \kappa) \log(1/\epsilon)\big)$ —— $n$ 与 $\kappa$ 相加。

当 $n \approx \kappa$（典型正则化机器学习场景）时，SVRG 比全梯度下降快约 $\kappa$ 倍；相比 SGD，在小 $\epsilon$ 下快约 $\kappa^2 / (\kappa \log(1/\epsilon)) = \kappa / \log(1/\epsilon)$ 倍。

![SGD、全梯度下降、SVRG 与 Katyusha 的收敛速率](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig2.png)
*log-log 坐标下，次优间隙 $f(x_T) - f^\star$ 随梯度计算次数 $T$ 的变化。SGD 的 $1/\sqrt{T}$ 速率最为缓慢；全梯度下降几何收敛但每步消耗 $n$ 次梯度计算；SVRG 与 Katyusha 在 epoch 数上呈线性收敛，最终全面胜出。*

---

## 6. SAGA、Katyusha 与下界

**SAGA**（Defazio, Bach & Lacoste-Julien, 2014）与 SVRG 类似，但为每个 $i$ 维护一张表，记录最新计算的 $\nabla f_i$，每次迭代仅更新对应条目。它避免了快照开销，但需 $O(nd)$ 额外内存。收敛速率同为 $O\big((n + \kappa) \log(1/\epsilon)\big)$。

**Katyusha**（Allen-Zhu, 2017）将方差缩减与 Nesterov 加速相结合，达到更优速率：  
$$
O\big((n + \sqrt{n \kappa}) \log(1/\epsilon)\big),
$$  
当 $\kappa \gg n$ 时优于 SVRG。

> **定理（下界，Woodworth & Srebro, 2016）**：任意随机一阶有限和（finite-sum）算法，至少需要 $\Omega\big((n + \sqrt{n \kappa}) \log(1/\epsilon)\big)$ 次梯度计算。

因此，Katyusha 在强凸有限和问题中是**最优的**。

![达到给定精度所需的总梯度计算次数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/10-stochastic-variance-reduction/fig5.png)
*在 $n=10^4$、$\kappa=10^3$ 的设定下，全梯度下降需约 $10^{11}$ 次梯度计算才能达到 $\epsilon=10^{-4}$；SGD 的 $O(\kappa^2/\epsilon)$ 复杂度约需 $10^{10}$ 次。SVRG 将其降至 $\sim 10^5$，Katyusha 进一步省下 $\sqrt{n/\kappa}$ 倍。*

---

## 7. 实践启示

| 问题场景                                      | 推荐方法                                 |
| --------------------------------------------- | ---------------------------------------- |
| $n$ 很大、精度要求低（深度学习）               | SGD + 动量 + 余弦学习率调度                |
| $n$ 中等、精度要求高（经典机器学习）             | SVRG、SAGA 或拟牛顿法                      |
| 强凸、病态、有限和问题                           | Katyusha 或加速版 SVRG                   |
| 凸但非强凸                                     | SGD 上的 Polyak 平均；FISTA-SGD           |
| 在线（流式）数据                                | 仅用 SGD；方差缩减依赖有限 $n$，无法适用     |

在深度学习中，朴素 SGD + 动量 + 学习率调度仍普遍优于各类方差缩减方法。其原因尚未完全阐明，主流猜测包括：（a）随机性有助于泛化；（b）损失下降时噪声尺度自然衰减；（c）SGD 的隐式偏差倾向于导向平坦极小值（flat minima）。

---

## 7. 总结

随机优化以每步计算代价的降低为代价，引入了噪声。经典 SGD 的收敛速率（凸函数下为 $O(1/\sqrt{T})$，强凸函数下为 $O(1/T)$）可直接由“噪声预算”分析得出。方差缩减技术则将 SGD 的单步效率提升至确定性优化的速率量级，其中 Katyusha 算法达到了匹配的理论下界。

第 11 篇文章标志着下一前沿：**非凸**优化问题——此时全局收敛不可保证，但仍存在局部性质保障（例如：逃离鞍点、收敛至平坦极小值点）。

## 延伸阅读

- Bottou, Curtis & Nocedal, *Optimization Methods for Large-Scale Machine Learning*, SIAM Review 60, 2018 —— 关于 SGD 及其变体的全面综述。  
- Johnson & Zhang, *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*, NeurIPS 2013 —— SVRG 原始论文。  
- Defazio, Bach & Lacoste-Julien, *SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives*, NeurIPS 2014 —— SAGA 原始论文。  
- Allen-Zhu, *Katyusha: The First Direct Acceleration of Stochastic Gradient Methods*, JMLR 18, 2017 —— 首个直接加速的方差缩减算法。  
- Woodworth & Srebro, *Tight Complexity Bounds for Optimizing Composite Objectives*, NeurIPS 2016 —— 匹配的复杂度下界分析。
