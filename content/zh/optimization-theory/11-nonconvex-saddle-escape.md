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
description: "为何SGD能在非凸景观下有效训练神经网络？我们证明扰动梯度下降可在多项式时间内逃离严格鞍点，在Polyak-Łojasiewicz条件下推导其收敛性，并综述深度学习损失曲面的已知理论结果——过参数化、神经正切核（NTK）及对平坦极小值的隐式偏好。"
disableNunjucks: true
translationKey: "optim-11"
---
对于非凸函数 $f$，梯度下降法（GD）不具有全局收敛保证。我们所能断言的最强结论仅为：$\nabla f(x_t) \to 0$ —— 即算法收敛至一个**平稳点（stationary point）**，而该点可能是局部极小值点、鞍点，甚至局部极大值点。本文探讨的问题是：**在何种条件下，我们能得出更强的结论？**

三项积极成果如下：

1. **鞍点逃逸（Saddle escape）**：在“严格鞍点（strict saddle）”假设下，带扰动的梯度下降法（perturbed GD）可在多项式时间内收敛到局部极小值点。鞍点本质上是不稳定的；布朗噪声（或仅需数值计算中固有的微小扰动）即可使其逃逸。
2. **PL 条件（Polyak–Łojasiewicz condition）**：这是强凸性的一种松弛形式，在过参数化的神经网络中常成立。在 PL 条件下，标准 GD 即便在非凸情形下也能获得线性收敛速率 $O(\log(1/\epsilon))$。
3. **损失景观事实（Loss landscape facts）**：对足够宽的神经网络而言，**所有局部极小值点均为全局极小值点**；而 SGD 引入的随机噪声则带来一种**隐式偏差（implicit bias）**，使其偏好平坦极小值点——这类极小值点通常具备更优的泛化性能。

上述每项结论在其各自设定下均具备严格的数学基础。本文亦指出当前**尚未解决的问题**：目前并不存在一个普适定理，能断言“SGD 可找到深度神经网络的全局最优解”。

## 你将学到的内容

1. 平稳点及其分类（依据 Hessian 矩阵 $\nabla^2 f$ 的特征值符号）；
2. 严格鞍点性质，以及 Ge–Huang–Jin–Yuan（2015）关于带扰动 GD 鞍点逃逸的证明；
3. Polyak–Łojasiewicz（PL）条件及其对收敛性的含义；
4. 过参数化、神经正切核（NTK），以及为何所有局部极小值点均可为全局极小值点；
5. SGD 对平坦极小值点的隐式偏差。

## 先修知识

第 02 篇（光滑性）、第 03 篇（GD 基础）、第 10 篇（随机优化方法）。

---

## 1. 非凸优化景观

对于非凸函数 $f$，一阶最优性条件 $\nabla f(x) = 0$ 的解具有多种类型，其分类依赖于 Hessian 矩阵 $\nabla^2 f(x^\star)$ 的正定性：

| $\nabla^2 f(x^\star)$                          | 类型                              |
| ---------------------------------------------- | --------------------------------- |
| $\succ 0$                                      | 严格局部极小值点                  |
| $\preceq 0$                                    | 局部极大值点                      |
| 不定（存在正、负特征值）                       | 鞍点                              |
| 奇异（含零特征值）                             | 退化点；需借助高阶信息判别        |

采用小步长的 GD 必收敛至某个平稳点。**但它无法区分上述四类情况**：GD 可能停驻于鞍点；而在平坦鞍点（即退化情形）上，它甚至可能永远无法逃逸。

经典担忧在于：在 $d$ 维空间中，鞍点数量可能随维度呈指数增长。对随机高斯多项式而言，几乎所有临界点都是鞍点而非极小值点（Auffinger, Ben Arous, Černý, 2013）。因此，从先验角度看，我们极易陷入鞍点。

好消息是：**绝大多数鞍点是可逃逸的**。

![按 Hessian 特征值签名分类的驻点](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig2.png)

*图示：按 Hessian 矩阵特征值签名划分的四类驻点。严格鞍点（第三幅）至少存在一个严格负特征值，可被逃逸；退化临界点（第四幅）含零特征值，需借助高阶信息判别。*

## 2. 通过扰动梯度下降实现鞍点逃逸

### 2.1 严格鞍点性质（Strict Saddle Property）

若函数 $f$ 在每个驻点 $x^\star$ 处均满足  
$$
\nabla^2 f(x^\star) \succ 0 \quad \text{或} \quad \lambda_{\min}(\nabla^2 f(x^\star)) < 0,
$$  
则称其具有**严格鞍点性质**。  
即：每个驻点要么是严格的局部极小值点，要么其 Hessian 矩阵至少有一个严格负的特征值（即“严格”鞍点，不存在平坦方向）。

许多机器学习问题满足严格鞍点性质，例如：正交张量分解、广义相位恢复、低秩矩阵补全、字典学习等。对这些问题而言，只要能有效逃离鞍点，迭代过程便自然收敛至局部极小值点——而该局部极小值点往往就是全局最优解。

### 2.2 扰动梯度下降（Perturbed GD）

```
算法：PGD（扰动梯度下降）
输入：初始点 x_0，步长 η，扰动半径 r，精度阈值 ε
for t = 0, 1, 2, ...:
    if ||∇f(x_t)|| ≤ ε 且 “近期未施加扰动”：
        x_t ← x_t + ξ_t，其中 ξ_t 在半径为 r 的球面上均匀采样
    x_{t+1} = x_t - η ∇f(x_t)
```

核心思想：在具有负特征值的驻点处，随机扰动 $\xi_t$ 以极高概率在对应负特征向量方向上具有正分量；随后的梯度下降步会沿该负曲率方向呈指数级放大该分量（体现为负特征值方向上的矩阵指数效应），从而将迭代点快速推离鞍点。

![3D 鞍点景观与原始/扰动 GD 轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig1.png)

*图示：典型严格鞍点 $f(x,y) = x^2 - y^2$。原始 GD 沿 $x$ 方向下降并停滞于鞍点（灰色）。微小随机扰动产生非零 $y$ 分量，该分量被负曲率指数放大——迭代点成功逃逸（橙色）。*


> **定理（Jin 等，2017）**：对于 $L$-光滑、$\rho$-Hessian-Lipschitz 且满足严格鞍点性质的函数，扰动梯度下降可在  
> $$
> O\left( \frac{L \, (f(x_0) - f^\star)}{\epsilon^2} \log^4 d \right) \text{ 次迭代内}
> $$  
> 找到一个 $\epsilon$-二阶平稳点（即满足 $\|\nabla f\| \leq \epsilon$ 且 $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\rho \epsilon}$）。

该迭代复杂度关于 $\epsilon$ 的依赖关系与标准梯度下降的一阶收敛速率完全一致；仅多出一个 $\log^4 d$ 的多项式对数因子——这正是为获得二阶保证所付出的唯一代价。

证明虽技术性较强，但直觉清晰：定义“停滞期”（stuck epochs）——即函数值几乎不下降的连续迭代段；可证每个停滞期以高概率在 $O(\log^4 d)$ 步扰动 GD 后终止。每个停滞期必然发生在近驻点附近，而一旦该点满足 $\lambda_{\min}(\nabla^2 f) < -\sqrt{\rho \epsilon}$，扰动后的轨迹将以高概率成功逃逸。

### 2.3 SGD 的隐式扰动

在随机优化设定下，随机梯度 $\nabla f_{i_t}$ 自带噪声，本身即构成对迭代点的天然扰动。上述分析可直接迁移：即使不显式添加扰动，SGD 仍能在多项式时间内逃离严格鞍点。这也部分解释了为何深度学习实践者通常无需担忧鞍点问题——训练过程中的固有噪声已免费提供了逃逸机制。
## 3. Polyak–Łojasiewicz（PL）条件

函数 $f$ 满足参数为 $\mu > 0$ 的 **PL 不等式**，若对所有 $x$ 有  
$$
\tfrac{1}{2} \|\nabla f(x)\|_2^2 \geq \mu (f(x) - f^\star).
$$

PL 条件**弱于强凸性**：任一 $\mu$-强凸函数必满足相同 $\mu$ 的 PL 不等式，但 PL 函数可以是非凸的。典型例子包括：

- 平方损失 $\|Ax - b\|_2^2$，当 $A$ 行满秩时成立——若 $A$ 是“宽”矩阵（列数多于行数），则该函数满足 PL，但不强凸；  
- 在合适初始化下过参数化的神经网络（Liu, Zhu & Belkin 2022）——在某些设定下，损失曲面在初始化邻域内满足 PL；  
- 数据可分时的逻辑回归——在无穷远处满足 PL，但不存在有限的最小值点。

![PL 条件与强凸性对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig3.png)

*图示：左——强凸性强制要求唯一极小值，而 PL 允许多个互不相连的全局极小值（四次函数 $\tfrac12(x^2-1.2)^2$ 有两个）。右——两种函数都满足 $\|\nabla f\|^2 \geq 2\mu(f - f^\star)$（对数-对数坐标，虚线为 $\mu = 0.5$ 时的 PL 边界）。*


> **定理**：若 $f$ 是 $L$-光滑的，且满足参数为 $\mu$ 的 PL 不等式，则取步长 $\eta = 1/L$ 的梯度下降法（GD）线性收敛：  
> $$
> f(x_T) - f^\star \leq (1 - \mu / L)^T (f(x_0) - f^\star).
> $$

**证明**：由光滑性，  
$$
f(x_{t+1}) \leq f(x_t) + \nabla f(x_t)^\top (x_{t+1} - x_t) + \tfrac{L}{2} \|x_{t+1} - x_t\|_2^2 = f(x_t) - \tfrac{1}{2 L} \|\nabla f(x_t)\|_2^2.
$$  
再由 PL 条件，$\|\nabla f(x_t)\|_2^2 \geq 2 \mu (f(x_t) - f^\star)$。代入得：  
$$
f(x_{t+1}) - f^\star \leq (1 - \mu / L) (f(x_t) - f^\star). \quad \blacksquare
$$

该证明简洁得令人震惊——仅需两行。PL 条件正是刻画**非凸情形下快速收敛**的恰当条件；它绕开了强凸性所依赖的全局最小值唯一性这一强假设。

### 3.1 PL 条件在深度学习中的应用

对于足够宽的神经网络（宽度关于训练样本数呈多项式增长），神经正切核（NTK）理论框架可在梯度下降轨迹上**严格证明 PL 不等式成立**（Du, Lee, Li, Wang, Zhai 2019；Liu 等 2022）。这是目前为数不多能从理论上解释“为何 GD 能在非凸的神经网络优化中达到零训练损失”的机制之一。

但需注意：PL 常数 $\mu$ 随网络宽度与深度恶化，因此尽管收敛率形式上线性，实际速度可能极慢。NTK 理论刻画的是“懒惰训练”（lazy training） regime，此时网络行为近似于核方法；而真实深度学习通常处于更丰富的特征学习 regime，上述理论保证在此情形下显著削弱。
## 4. 神经网络的损失景观（Loss Landscape）

我们对深度网络的损失景观究竟了解多少？

### 4.1 过参数化消除了虚假局部极小值

对于一个参数量 $\geq n$ 的深度神经网络（其中 $n$ 是训练样本数），在模型容量足以插值数据的前提下，经验损失的全局最小值为 $0$。在激活函数满足温和假设的条件下，**经验损失的所有局部极小值都是全局极小值**——在合适的设定下，梯度下降（GD）轨迹上的损失景观具有 **PL 性质（Polyak–Łojasiewicz 条件）**。

这是当代“**过参数化有助于优化**”的核心命题：当参数数量超过数据点数量时，训练损失可降至零，且梯度类方法能实际达到该值。2017–2020 年间关于神经网络损失景观的理论工作，为自 AlexNet 以来实践者长期观察到的现象提供了坚实的数学基础。

### 4.2 NTK 区域（Neural Tangent Kernel Regime）

**神经正切核**（NTK，Jacot, Gabriel & Hongler 2018）指出：足够宽的网络在其初始参数 $\theta_0$ 附近，其输出关于参数近似线性：
$$
f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0).
$$
在此区域中，对平方损失执行梯度下降，其收敛速率由 NTK 的特征值决定——梯度流退化为一个线性常微分方程（ODE）。在随机初始化下，NTK 的最小特征值以高概率远离零，从而保证线性收敛速率。

NTK 描述的是“懒训练”（lazy training）现象：即网络极宽、权重几乎不更新的情形。而真实世界的网络通常运行于该区域之外，具备显著的**特征学习**能力与权重的**定性变化**。

![NTK 区域：宽网络停留在初始化附近，GD 线性收敛](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig4.png)

*图示：左——在 NTK / 懒训练区域中，宽网络的参数 $\theta$ 始终停留在初始化 $\theta_0$ 的极小邻域内（蓝色），而窄网络的参数则发生显著位移（橙色，特征学习）。右——在 PL 条件成立的 NTK 区域中，GD 享有线性速率 $(1-\mu/L)^t$，与一般非凸景观下的次线性平台形成鲜明对比。*


### 4.3 SGD 的隐式偏差（Implicit Bias）

SGD 所找到的并非任意一个全局极小值，而是**平坦的**（flat）极小值。实证表明：SGD 找到的极小值比全批量 GD 在相同损失值下找到的极小值具有更强的泛化能力。其被广泛猜想的机制是：SGD 噪声的方差正比于损失函数的局部曲率，因此 SGD 在曲率较小（即更平坦）的区域停留时间更长。

一个精确刻画（Mandt et al., 2017）：在固定步长 $\eta$ 下，SGD 的平稳分布近似为以最优解 $x^\star$ 为中心、协方差为 $\eta C$ 的高斯分布，其中 $C$ 依赖于 Hessian 矩阵的逆与梯度的协方差。Hessian 更小（即更平坦）的极小值被采样得更频繁。

![平坦极小值 vs 尖锐极小值与泛化差距](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/11-nonconvex-saddle-escape/fig5.png)

*图示：左——一维损失同时具有平坦（宽）盆地和尖锐（窄）盆地，二者训练损失值近似相等。右——当测试分布相对训练发生小幅平移时，平坦极小值的损失值几乎不变（差距 $\approx 0.05$），而尖锐极小值的损失值急剧增大（差距 $\approx 0.93$）。这正是 SGD 偏好平坦极小值并获得更优泛化能力的几何直觉。*


这属于更宏大的“**隐式偏差**”图景的一部分：即使损失函数存在大量全局极小值，优化算法的选择仍会实质性地塑造最终学到的函数类别。例如，线性模型在逻辑损失下使用 GD 会收敛至最大间隔解；而使用 Adam 优化同一损失，则会得到不同的解。

---

## 5. 立方正则化：一种非凸牛顿法

阻尼牛顿法（见第 07 篇）在非凸情形下的对应物是**立方正则化**（cubic regularization）：
$$
x_{t+1} = \arg\min_x \, f(x_t) + \nabla f(x_t)^\top (x - x_t) + \tfrac{1}{2} (x - x_t)^\top \nabla^2 f(x_t) (x - x_t) + \tfrac{M}{6} \|x - x_t\|^3.
$$

> **定理（Nesterov & Polyak, 2006）**：若取 $M$ 为 $\nabla^2 f$ 的 Lipschitz 常数，则立方正则化可在 $O(1/\epsilon^{1.5})$ 次迭代内收敛至二阶驻点（即满足 $\|\nabla f\| \leq \epsilon$ 且 $\lambda_{\min}(\nabla^2 f) \geq -\sqrt{\epsilon}$）。

对比如下：

- 标准 GD：需 $O(1/\epsilon^2)$ 次迭代达到一阶驻点；
- 扰动 GD：需 $\tilde O(1/\epsilon^2)$ 次迭代达到二阶驻点；
- 立方正则化：仅需 $O(1/\epsilon^{1.5})$ 次迭代达到二阶驻点——**严格更快**。

但其代价在于：立方正则化需要计算 Hessian 矩阵（或求解一个三次子问题）。它已在部分机器学习任务中得到应用，但在大规模场景下仍被 SGD 及其变体所主导。

---

## 6. 什么是被证明为困难的问题

为平衡前述乐观结论，以下问题是已被严格证明为困难的：

- **在多项式时间内求出一般非凸函数的全局极小值**是 NP-hard 问题（其判定版本——“是否 $f^\star \leq c$？”——对一般的非凸多项式而言是 NP-hard）；
- **对光滑有界函数 $f$ 求 $\epsilon$-局部极小值**，至少需要 $\Omega(1/\epsilon^{1.5})$ 次 Hessian 查询（Carmon, Duchi, Hinder & Sidford, 2017）——该下界恰好匹配立方正则化的上界；
- **对深度网络的训练损失求全局极小值**，在最坏情形下已被证明是困难的（Blum & Rivest, 1992，针对 ReLU 网络）。

所有关于鞍点逃逸、PL 性质与 NTK 的成功分析，都严重依赖于问题所具有的**特定结构**，从而排除了最坏情形。如今学界已形成共识：“机器学习实践之所以有效，是因为问题本身具有结构，而非因为优化问题普遍容易”。
## 7. 总结

| 概念                         | 它为你提供                                                                 |
| ---------------------------- | -------------------------------------------------------------------------- |
| 严格鞍点性质（Strict saddle property） | 鞍点是不稳定的；带扰动的梯度下降法（perturbed GD）可在多项式时间内逃离鞍点。     |
| Polyak–Łojasiewicz（PL）条件         | 无需凸性假设即可实现梯度下降法的线性收敛；该条件在宽神经网络中成立。               |
| 神经正切核（NTK）范式                | 宽网络表现出近似线性行为；梯度下降法以线性速率收敛。                              |
| SGD 的隐式偏置（Implicit bias of SGD） | SGD 倾向于收敛到平坦极小值点；可解释其相较于全批量 GD 更优的泛化性能。              |
| 三次正则化（Cubic regularization）     | 达到二阶平稳点（second-order stationary points）的一阶最优收敛速率。                 |

至此，本系列关于连续优化理论的四篇文章已全部完成。第 12 篇文章将为本系列收尾，主题是**离散优化与全局优化**——包括分支定界法（branch-and-bound）、整数规划（integer programming）以及各类启发式算法——适用于那些不具备光滑性等良好结构的优化问题。

## 延伸阅读

- Jin, Ge, Netrapalli, Kakade, Jordan, *How to Escape Saddle Points Efficiently*, ICML 2017 —— 提出带扰动梯度下降法的经典论文。  
- Liu, Zhu & Belkin, *Loss landscapes and optimization in over-parameterized non-linear systems and neural networks*, ACHA 59, 2022 —— 关于 PL 条件与神经网络的现代理论成果。  
- Du, Lee, Li, Wang & Zhai, *Gradient Descent Finds Global Minima of Deep Neural Networks*, ICML 2019 —— 将 NTK 理论推广至深度神经网络的工作。  
- Karimi, Nutini & Schmidt, *Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Łojasiewicz Condition*, ECML 2016 —— PL 条件下梯度类方法线性收敛性的系统性分析。  
- Carmon, Duchi, Hinder & Sidford, *Lower Bounds for Finding Stationary Points*, MathProg 184, 2020 —— 非凸一阶优化方法达到平稳点所需迭代次数的匹配下界。  
- Auffinger, Ben Arous & Černý, *Random Matrices and Complexity of Spin Glasses*, CPAM 66, 2013 —— 在随机能量景观中对鞍点数量进行统计与刻画的奠基性工作。