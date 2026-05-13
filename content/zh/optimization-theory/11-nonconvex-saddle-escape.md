---
title: '优化理论（十一）：非凸优化与鞍点逃逸'
date: 2022-09-29 09:00:00
tags:
  - ML
  - Optimization
  - Deep Learning Theory
categories: Algorithm
series: optimization-theory
series_order: 11
lang: zh
mathjax: true
description: "为何 SGD 能在非凸景观下有效训练神经网络？我们证明扰动梯度下降可在多项式时间内逃离严格鞍点，在 Polyak-Łojasiewicz 条件下推导其收敛性，并综述深度学习损失曲面的已知理论结果——过参数化、神经正切核（NTK）及对平坦极小值的隐式偏好。"
disableNunjucks: true
translationKey: "optim-11"
---
对于非凸函数 $f$，梯度下降法（GD）没有全局收敛保证。我们最多只能说 $\nabla f(x_t) \to 0$——即算法会收敛到一个**平稳点（stationary point）**，而该点可能是局部极小值、鞍点，甚至是局部极大值。本文要探讨的问题是：**在什么条件下，我们能得出更强的结论？**

这里有三项积极成果：

1. **鞍点逃逸（Saddle escape）**：在“严格鞍点（strict saddle）”假设下，带扰动的 GD 能在多项式时间内收敛到局部极小值。鞍点本质上是不稳定的；布朗噪声（或仅仅是数值计算中的微小扰动）就能帮助算法逃离。
2. **PL 条件（Polyak–Łojasiewicz condition）**：这是强凸性的一种弱化形式，在过参数化的神经网络中常常成立。在 PL 条件下，即使函数非凸，标准 GD 也能获得线性收敛速率 $O(\log(1/\epsilon))$。
3. **损失景观的事实（Loss landscape facts）**：对于足够宽的神经网络，**所有局部极小值都是全局极小值**；而 SGD 的随机噪声会带来一种**隐式偏差（implicit bias）**，使其倾向于收敛到更平坦的极小值——这类极小值通常泛化性能更好。

上述每项结论在其适用范围内都有严格的理论支撑。但本文也指出当前**尚未解决的问题**：目前并不存在一个普适定理，能断言“SGD 总能找到深度神经网络的全局最优解”。

## 你将学到

1. 平稳点及其分类（依据 Hessian 矩阵 $\nabla^2 f$ 的特征值符号）；
2. 严格鞍点性质，以及 Ge–Huang–Jin–Yuan（2015）关于带扰动 GD 鞍点逃逸的证明；
3. Polyak–Łojasiewicz（PL）条件及其对收敛性的含义；
4. 过参数化、神经正切核（NTK），以及为何所有局部极小值都可能是全局极小值；
5. SGD 对平坦极小值的隐式偏好。

## 前置知识

第 02 篇（光滑性）、第 03 篇（GD 基础）、第 10 篇（随机优化方法）。

---

## 1. 非凸优化景观

对于非凸函数 $f$，一阶最优性条件 $\nabla f(x) = 0$ 的解有多种类型，其分类取决于 Hessian 矩阵 $\nabla^2 f(x^\star)$ 的特征值符号：

| $\nabla^2 f(x^\star)$                          | 类型                              |
| ---------------------------------------------- | --------------------------------- |
| $\succ 0$                                      | 严格局部极小值                    |
| $\preceq 0$                                    | 局部极大值                        |
| 不定（同时存在正、负特征值）                   | 鞍点                              |
| 奇异（含零特征值）                             | 退化点；需高阶信息判断            |

使用小步长的 GD 会收敛到某个平稳点，**但它无法区分上述四类情况**：GD 可能停在鞍点上，而在平坦鞍点（即退化情形）附近，甚至可能永远无法逃逸。

经典担忧在于：在 $d$ 维空间中，鞍点数量可能随维度指数增长。对于随机高斯多项式，几乎所有临界点都是鞍点而非极小值（Auffinger, Ben Arous, Černý 2013）。因此，从先验角度看，我们很容易被困住。

好消息是：**大多数鞍点其实是可逃逸的**。

![按 Hessian 特征值签名分类的驻点](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig2.png)

*图示：按 Hessian 特征值符号划分的四类平稳点。严格鞍点（第三幅）至少有一个严格负特征值，可被逃逸；退化临界点（第四幅）含零特征值，需高阶信息判别。*

## 2. 通过扰动梯度下降实现鞍点逃逸

### 2.1 严格鞍点性质

若函数 $f$ 在每个平稳点 $x^\star$ 处满足
$$
\nabla^2 f(x^\star) \succ 0 \quad \text{或} \quad \lambda_{\min}(\nabla^2 f(x^\star)) < 0,
$$
则称其具有**严格鞍点性质**。也就是说，每个平稳点要么是严格局部极小值，要么其 Hessian 至少有一个严格负特征值（即“严格”鞍点，没有平坦方向）。

许多机器学习问题都满足这一性质，例如正交张量分解、广义相位恢复、低秩矩阵补全和字典学习。对这些问题而言，只要能逃离鞍点，算法就会自动收敛到局部极小值——而该极小值往往就是全局最优解。

### 2.2 扰动梯度下降（Perturbed GD）

```text
算法：PGD（扰动梯度下降）
输入：初始点 x_0，步长 η，扰动半径 r，精度阈值 ε
for t = 0, 1, 2, ...:
    if ||∇f(x_t)|| ≤ ε 且 “近期未施加扰动”：
        x_t ← x_t + ξ_t，其中 ξ_t 在半径为 r 的球面上均匀采样
    x_{t+1} = x_t - η ∇f(x_t)
```

核心思想是：在具有负特征值的平稳点处，随机扰动 $\xi_t$ 以极高概率在对应负特征向量方向上有非零分量；随后的 GD 步骤会沿该负曲率方向指数级放大这一分量（体现为负特征值方向上的矩阵指数效应），从而将迭代点迅速推离鞍点。

![3D 鞍点景观与原始/扰动 GD 轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig1.png)

*图示：典型严格鞍点 $f(x,y) = x^2 - y^2$。标准 GD 沿 $x$ 方向下降后停滞于鞍点（灰色）。加入微小随机扰动后产生非零 $y$ 分量，该分量被负曲率指数放大，迭代点成功逃逸（橙色）。*

> **定理（Jin 等，2017）**：对于 $L$-光滑、$\rho$-Hessian-Lipschitz 且满足严格鞍点性质的函数，扰动 GD 能在
> $$
> O\left( \frac{L \, (f(x_0) - f^\star)}{\epsilon^2} \log^4 d \right) \text{ 次迭代内}
> $$
> 找到一个 $\epsilon$-二阶平稳点（即 $\|\nabla f\| \leq \epsilon$ 且 $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\rho \epsilon}$）。

这一复杂度关于 $\epsilon$ 的依赖与标准 GD 的一阶收敛速率相同，仅多出一个 $\log^4 d$ 的多项式对数因子——这正是为获得二阶保证所付出的唯一代价。

证明虽技术性强，但直觉清晰：定义“停滞期”（stuck epochs）——即函数值几乎不下降的连续迭代段；可以证明，每个停滞期以高概率在 $O(\log^4 d)$ 步扰动 GD 后结束。每个停滞期都发生在近平稳点附近，而只要该点满足 $\lambda_{\min}(\nabla^2 f) < -\sqrt{\rho \epsilon}$，扰动后的轨迹就会以高概率逃逸。

### 2.3 SGD 的隐式扰动

在随机优化设定下，随机梯度 $\nabla f_{i_t}$ 本身已包含噪声，天然构成对迭代点的扰动。上述分析可直接迁移：即使不显式添加扰动，SGD 也能在多项式时间内逃离严格鞍点。这也部分解释了为何深度学习从业者通常不担心鞍点问题——训练过程中的固有噪声已免费提供了逃逸机制。

## 3. Polyak–Łojasiewicz（PL）条件

函数 $f$ 满足参数为 $\mu > 0$ 的 **PL 不等式**，如果对所有 $x$ 有
$$
\tfrac{1}{2} \|\nabla f(x)\|_2^2 \geq \mu (f(x) - f^\star).
$$\nPL 条件**弱于强凸性**：每个 $\mu$-强凸函数都满足相同 $\mu$ 的 PL 不等式，但 PL 函数可以是非凸的。典型例子包括：

- 平方损失 $\|Ax - b\|_2^2$，当 $A$ 行满秩时成立——若 $A$ 是“宽”矩阵（列数多于行数），则函数满足 PL 但不强凸；
- 在合适初始化下的过参数化神经网络（Liu, Zhu & Belkin 2022）——在某些设定下，损失景观在初始化附近满足 PL；
- 数据可分时的逻辑回归——在无穷远处满足 PL，但无有限最小值点。

![PL 条件与强凸性对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig3.png)

*图示：左——强凸性强制唯一极小值，而 PL 允许多个互不相连的全局极小值（如四次函数 $\tfrac12(x^2-1.2)^2$ 有两个）。右——两种函数均满足 $\|\nabla f\|^2 \geq 2\mu(f - f^\star)$（对数-对数坐标，虚线为 $\mu = 0.5$ 时的 PL 边界）。*

> **定理**：若 $f$ 是 $L$-光滑的，且满足参数为 $\mu$ 的 PL 不等式，则取步长 $\eta = 1/L$ 的 GD 线性收敛：
> $$
> f(x_T) - f^\star \leq (1 - \mu / L)^T (f(x_0) - f^\star).
> $$

**证明**：由光滑性，
$$\nf(x_{t+1}) \leq f(x_t) + \nabla f(x_t)^\top (x_{t+1} - x_t) + \tfrac{L}{2} \|x_{t+1} - x_t\|_2^2 = f(x_t) - \tfrac{1}{2 L} \|\nabla f(x_t)\|_2^2.
$$
再由 PL 条件，$\|\nabla f(x_t)\|_2^2 \geq 2 \mu (f(x_t) - f^\star)$。代入得：
$$\nf(x_{t+1}) - f^\star \leq (1 - \mu / L) (f(x_t) - f^\star). \quad \blacksquare
$$

这个证明简洁得令人惊讶——仅需两行。PL 条件正是刻画**非凸情形下快速收敛**的恰当假设；它绕开了强凸性所依赖的“全局最小值唯一”这一强前提。

### 3.1 PL 条件在深度学习中的应用

对于足够宽的神经网络（宽度关于训练样本数呈多项式增长），神经正切核（NTK）理论可在 GD 轨迹上**严格证明 PL 不等式成立**（Du, Lee, Li, Wang, Zhai 2019；Liu 等 2022）。这是少数能从理论上解释“为何 GD 能在非凸神经网络中达到零训练损失”的机制之一。

但需注意：PL 常数 $\mu$ 随网络宽度和深度恶化，因此尽管收敛速率形式上线性，实际速度可能很慢。NTK 理论描述的是“懒惰训练”（lazy training）区域——此时网络行为近似核方法；而真实深度学习通常处于更丰富的特征学习区域，这些理论保证在此情形下显著削弱。

## 4. 神经网络的损失景观

我们对深度网络的损失景观究竟了解多少？

### 4.1 过参数化消除了虚假局部极小值

对于参数量 $\geq n$ 的深度神经网络（$n$ 为训练样本数），在模型容量足以插值数据的前提下，经验损失的全局最小值为 0。在激活函数满足温和假设的条件下，**所有局部极小值都是全局极小值**——在合适设定下，GD 轨迹上的损失景观满足 **PL 性质**。

这正是当代“**过参数化有助于优化**”的核心观点：当参数多于数据点时，零训练损失可达，且梯度方法能实际达到。2017–2020 年间关于神经网络损失景观的理论工作，为自 AlexNet 以来实践者长期观察到的现象提供了坚实基础。

### 4.2 NTK 区域

**神经正切核**（NTK，Jacot, Gabriel & Hongler 2018）指出：足够宽的网络在其初始化 $\theta_0$ 附近，输出关于参数近似线性：
$$\nf(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0).
$$
在此区域中，对平方损失执行 GD，其收敛速率由 NTK 的特征值决定——梯度流退化为线性常微分方程（ODE）。在随机初始化下，NTK 的最小特征值以高概率远离零，从而保证线性收敛。
\nNTK 描述的是“懒训练”现象：宽网络权重几乎不变。而真实网络通常运行于该区域之外，具备显著的**特征学习**能力与权重的**定性变化**。

![NTK 区域：宽网络停留在初始化附近，GD 线性收敛](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig4.png)

*图示：左——在 NTK / 懒训练区域中，宽网络的参数 $\theta$ 始终停留在初始化 $\theta_0$ 的极小邻域内（蓝色），而窄网络的参数则大幅移动（橙色，特征学习）。右——在满足 PL 的 NTK 区域中，GD 享有线性速率 $(1-\mu/L)^t$，与一般非凸景观下的次线性平台形成鲜明对比。*

### 4.3 SGD 的隐式偏差
\nSGD 找到的并非任意全局极小值，而是**平坦的**（flat）极小值。实证表明：SGD 找到的极小值比全批量 GD 在相同损失值下找到的极小值泛化更好。其被广泛接受的机制是：SGD 噪声的方差正比于损失的局部曲率，因此 SGD 在曲率小（即更平坦）的区域停留更久。

一个精确刻画（Mandt 等，2017）：在固定步长 $\eta$ 下，SGD 的平稳分布近似为以 $x^\star$ 为中心、协方差为 $\eta C$ 的高斯分布，其中 $C$ 依赖于 Hessian 的逆与梯度协方差。更平坦的极小值（Hessian 更小）被访问得更频繁。

![平坦极小值 vs 尖锐极小值与泛化差距](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig5.png)

*图示：左——一维损失同时具有平坦（宽）盆地和尖锐（窄）盆地，二者训练损失近似相等。右——当测试分布发生小幅偏移时，平坦极小值的损失几乎不变（差距 $\approx 0.05$），而尖锐极小值的损失急剧上升（差距 $\approx 0.93$）。这正是 SGD 偏好平坦极小值并获得更好泛化能力的几何直觉。*

这属于更宏大的“**隐式偏差**”图景：即使损失存在大量全局极小值，优化算法的选择仍会实质性地塑造最终学到的函数。例如，线性模型在逻辑损失下用 GD 会收敛到最大间隔解；而用 Adam 则会得到不同解。

---

## 5. 立方正则化：一种非凸牛顿法

阻尼牛顿法（见第 07 篇）在非凸情形下的对应物是**立方正则化**（cubic regularization）：
$$\nx_{t+1} = \arg\min_x \, f(x_t) + \nabla f(x_t)^\top (x - x_t) + \tfrac{1}{2} (x - x_t)^\top \nabla^2 f(x_t) (x - x_t) + \tfrac{M}{6} \|x - x_t\|^3.
$$

> **定理（Nesterov & Polyak, 2006）**：若取 $M$ 为 $\nabla^2 f$ 的 Lipschitz 常数，则立方正则化能在 $O(1/\epsilon^{1.5})$ 次迭代内收敛到二阶平稳点（即 $\|\nabla f\| \leq \epsilon$ 且 $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\epsilon}$）。

对比如下：

- 标准 GD：需 $O(1/\epsilon^2)$ 次迭代达到一阶平稳点；
- 扰动 GD：需 $\tilde O(1/\epsilon^2)$ 次迭代达到二阶平稳点；
- 立方正则化：仅需 $O(1/\epsilon^{1.5})$ 次迭代——**严格更快**。

但代价是：立方正则化需要 Hessian（或求解三次子问题）。它已在部分 ML 应用中使用，但在大规模场景下仍被 SGD 家族方法主导。

---

## 6. 什么是被证明为困难的问题

为平衡前述乐观结论，以下问题是已被严格证明困难的：

- **在多项式时间内求一般非凸函数的全局极小值**是 NP-hard（其判定版本“是否 $f^\star \leq c$？”对一般非凸多项式是 NP-hard）；
- **对光滑有界函数 $f$ 求 $\epsilon$-局部极小值**至少需要 $\Omega(1/\epsilon^{1.5})$ 次 Hessian 查询（Carmon, Duchi, Hinder & Sidford 2017）——该下界恰好匹配立方正则化的上界；
- **对深度网络的训练损失求全局极小值**在最坏情形下已被证明困难（Blum & Rivest 1992，针对 ReLU 网络）。

所有关于鞍点逃逸、PL 和 NTK 的成功分析，都依赖于问题的**特定结构**，从而排除了最坏情形。如今学界共识是：“机器学习实践之所以有效，是因为问题本身具有结构，而非因为优化普遍容易”。

## 7. 总结

| 概念                         | 它为你提供                                                                 |
| ---------------------------- | -------------------------------------------------------------------------- |
| 严格鞍点性质                 | 鞍点不稳定；扰动 GD 可在多项式时间内逃离。                                 |
| PL 条件                      | 无需凸性即可实现 GD 的线性收敛；在宽神经网络中成立。                       |
| NTK 范式                     | 宽网络近似线性；GD 以线性速率收敛。                                        |
| SGD 的隐式偏差               | SGD 偏好平坦极小值；解释其优于全批量 GD 的泛化性能。                       |
| 立方正则化                   | 达到二阶平稳点的一阶最优收敛速率。                                         |

至此，本系列关于连续优化理论的四篇文章已全部完成。第 12 篇将收尾本系列，主题是**离散与全局优化**——包括分支定界法、整数规划和各类启发式算法——适用于那些缺乏光滑性等良好结构的问题。

## 延伸阅读

- Jin, Ge, Netrapalli, Kakade, Jordan, *How to Escape Saddle Points Efficiently*, ICML 2017 —— 提出扰动 GD 的经典论文。
- Liu, Zhu & Belkin, *Loss landscapes and optimization in over-parameterized non-linear systems and neural networks*, ACHA 59, 2022 —— PL 与神经网络的现代理论成果。
- Du, Lee, Li, Wang & Zhai, *Gradient Descent Finds Global Minima of Deep Neural Networks*, ICML 2019 —— 将 NTK 推广至深度网络的工作。
- Karimi, Nutini & Schmidt, *Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak–Łojasiewicz Condition*, ECML 2016 —— PL 下梯度方法线性收敛的系统分析。
- Carmon, Duchi, Hinder & Sidford, *Lower Bounds for Finding Stationary Points*, MathProg 184, 2020 —— 非凸一阶方法的匹配下界。
- Auffinger, Ben Arous & Černý, *Random Matrices and Complexity of Spin Glasses*, CPAM 66, 2013 —— 随机景观中鞍点统计的奠基工作。
