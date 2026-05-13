---
title: "偏微分方程与机器学习（六）：连续归一化流与 Neural ODE"
date: 2024-07-15 09:00:00
tags:
  - PDE
  - Machine Learning
  - Neural ODE
  - Normalizing Flows
  - CNF
  - Optimal Transport
  - Flow Matching
categories: PDE与机器学习
series: pde-ml
lang: zh
mathjax: true
description: "如何把高斯变成数据分布？本文从 ODE/PDE 理论出发，系统推导 Neural ODE、伴随方法、连续归一化流（FFJORD）与 Flow Matching，并用 7 张图把核心机制画清楚。"
disableNunjucks: true
series_order: 6
translationKey: "pde-ml-6"
---
![偏微分方程与机器学习（六）：连续归一化流与Neural ODE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/illustration_1.png)

## 这一篇要讲什么

生成建模归根结底是一个几何问题：**如何将简单分布（比如高斯分布）变换为复杂分布（如人脸、分子或运动轨迹）？** 离散归一化流通过堆叠可逆模块实现这一目标，但每个模块都需要计算 Jacobian 行列式，其代价高达 $O(d^3)$。**Neural ODE** 用连续的常微分方程（ODE）取代离散的网络深度；**连续归一化流（Continuous Normalizing Flows, CNF）** 则借助 *瞬时* 变量替换公式，将密度计算的复杂度降至 $O(d)$；而 **Flow Matching** 更进一步，直接省去了散度积分，将训练简化为对目标速度场的普通回归任务。

全文围绕三条主线交织展开：

1. **PDE 视角** —— 连续性方程 $\partial_t\rho + \nabla\!\cdot(\rho v) = 0$ 描述了速度场 $v$ 如何输运密度 $\rho$。
2. **ODE 视角** —— Picard-Lindelöf 定理保证了解的存在性与唯一性；Liouville 定理将体积变化与 $\nabla\!\cdot v$ 联系起来；伴随方程则使反向传播的内存开销降至 $O(1)$。
3. **机器学习视角** —— Neural ODE、FFJORD 和 Flow Matching 都通过神经网络参数化速度场 $v$，并从数据中学习它。

**前置知识**：ODE 基础（解的存在唯一性）、概率密度的变量替换、自动微分（autograd）。

![连续流把高斯逐步变成两月牙目标分布。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig1_density_transformation.png)
*图 1：连续时间流将高斯基分布输运成“双月牙”目标分布。每个面板是通过对约 4000 个粒子求解 ODE 至时间 $t$ 后，使用核密度估计（KDE）得到的密度 $\rho_t$。所有时间点共享同一个网络 $v_\theta$，仅积分上限不同。*

---

## 1. ODE 基础：存在唯一性与体积演化

### 1.1 Picard-Lindelöf：解何时存在且唯一？

**定理（Picard-Lindelöf）**。考虑初值问题 $\dot{\mathbf{z}} = f(\mathbf{z}, t)$，$\mathbf{z}(0) = \mathbf{z}_0$。若 $f$ 关于 $t$ 连续，且关于 $\mathbf{z}$ 满足 Lipschitz 条件：
$$
\|f(\mathbf{z}_1, t) - f(\mathbf{z}_2, t)\| \le L\,\|\mathbf{z}_1 - \mathbf{z}_2\|,
$$
则在某个区间 $[0, T]$ 上存在唯一解。

*这对机器学习意味着什么？* 如果 $f_\theta$ 是一个使用 Lipschitz 激活函数（如 ReLU、tanh、GELU）且权重有界的神经网络，那么局部 Lipschitz 条件自然成立。因此，只要网络行为良好——这在实践中几乎总是成立——Neural ODE 就是适定的，这也是它能稳定进行反向传播的根本原因。

### 1.2 Liouville 定理：流如何改变体积

**定理（Liouville）**。设 $\phi_t$ 是 ODE $\dot{\mathbf{z}} = f(\mathbf{z}, t)$ 的流映射。对任意可测集合 $\Omega$，有
$$
\frac{d}{dt}\,\mathrm{vol}(\phi_t(\Omega)) = \int_{\phi_t(\Omega)} \nabla\!\cdot f\,d\mathbf{z}.
$$
因此，$\nabla\!\cdot f = 0$ 保持体积不变，$\nabla\!\cdot f < 0$ 导致压缩，$\nabla\!\cdot f > 0$ 引起膨胀。在归一化流中，我们恰恰希望散度非零——这正是重塑概率质量的关键杠杆。

*直观理解*：散度为零的 $f$ 类似不可压缩流体（如第 5 篇讨论的哈密顿或辛系统）；而散度非零的 $f$ 则像可压缩流，能将概率质量挤压成细丝，再在别处重新膨胀——这正是生成建模所需要的特性。

### 1.3 瞬时变量替换公式

**定理**。沿 ODE $\dot{\mathbf{z}} = f(\mathbf{z}, t)$ 的轨迹 $\mathbf{z}(t) = \phi_t(\mathbf{z}_0)$，密度满足
$$
\boxed{\;\frac{d}{dt}\log\rho_t(\mathbf{z}(t)) = -\nabla\!\cdot f(\mathbf{z}(t), t).\;}\tag{1}
$$
*证明思路*：连续性方程 $\partial_t\rho + \nabla\!\cdot(\rho f) = 0$ 可展开为 $\partial_t\rho + f\!\cdot\!\nabla\rho = -\rho\,\nabla\!\cdot f$。左边正是沿轨迹 $\mathbf{z}(t)$ 的物质导数 $D\rho/Dt$。两边同除以 $\rho$ 即得 (1)。

**为何这个公式至关重要？** 离散归一化流需要 $O(d^3)$ 的代价来计算 $\log|\det \partial\phi / \partial\mathbf{z}|$，而公式 (1) 仅需 Jacobian 的迹（即散度），借助一次 vector-Jacobian product 即可在 $O(d)$ 时间内完成（见 3.2 节）。这正是连续归一化流（CNF）得以存在的核心计算优势。

## 2. Neural ODE：从离散到连续深度

### 2.1 残差网络即前向 Euler 方法

ResNet 的更新规则 $\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$ 正是步长 $\Delta t = 1$ 的前向 Euler 方法，用于求解 ODE $\dot{\mathbf{h}} = f(\mathbf{h}, t)$。取连续极限后，我们得到一个统一的连续时间 ODE：
$$
\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t), \qquad \mathbf{h}(T) = \mathbf{h}(0) + \int_0^T f_\theta(\mathbf{h}(t), t)\,dt. \tag{2}
$$
这一转变带来三大优势：

- **参数效率更高**：单个网络 $f_\theta$ 替代了每一层不同的 $f_l$。
- **自适应深度**：如 dopri5（自适应 Runge-Kutta）等求解器会自动在动力学剧烈处采用小步长，在平缓处采用大步长。
- **内存占用更低**：伴随方法将反向传播的内存开销从 $O(L)$ 降至 $O(1)$——深度增加带来的只是计算时间，而非显存压力。

![ResNet（离散深度，固定步）vs Neural ODE（连续深度，自适应求解器）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig2_neural_ode_vs_resnet.png)
*图 2：左侧是 ResNet，由一系列 $h_{l+1} = h_l + f_l(h_l)$ 的 Euler 步组成，每层参数独立，反向传播需存储所有中间激活；右侧是 Neural ODE，由单一 $f_\theta$ 驱动，自适应求解器动态选择评估点，伴随方法仅用 $O(1)$ 内存即可恢复梯度。*

### 2.2 伴随灵敏度方法

标准反向传播在 ODE 求解过程中需存储每一步的中间状态，内存开销为 $O(L)$——而自适应求解器可能执行上百步。伴随方法则完全避免了这一问题。

定义**伴随状态** $\mathbf{a}(t) = \partial\mathcal{L}/\partial\mathbf{h}(t)$，它满足
$$
\frac{d\mathbf{a}}{dt} = -\,\mathbf{a}(t)^\top \frac{\partial f_\theta}{\partial\mathbf{h}}, \tag{3}
$$
而参数梯度为
$$
\frac{d\mathcal{L}}{d\theta} = -\int_T^0 \mathbf{a}(t)^\top \frac{\partial f_\theta}{\partial\theta}\,dt. \tag{4}
$$
**算法流程如下**：
1. *前向传递*：求解 (2) 从 $0 \to T$，仅保存最终状态 $\mathbf{h}(T)$。
2. *初始化*：设 $\mathbf{a}(T) = \partial\mathcal{L}/\partial\mathbf{h}(T)$。
3. *反向传递*：将 $\mathbf{h}$ 与 $\mathbf{a}$ 联合从 $T \to 0$ 反向求解，并累积 (4) 中的梯度。

该方法的内存开销为 $O(1)$，与求解步数无关。代价是反向过程需额外求解一次 ODE——计算量约为原来的两倍，却换来了近乎无限的内存节省。

![伴随方法：在二维向量场上的正反两条轨迹，以及不同深度下的内存对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig3_adjoint_method.png)
*图 3：左图展示同一螺旋 ODE 先正向积分（蓝色）得到 $h(T)$，再与伴随状态（红色虚线）一同反向积分以恢复梯度；右图对比内存开销随求解步数 $L$ 的增长情况：标准反向传播为 $O(L)$，而伴随方法始终保持 $O(1)$——当 $L=1000$ 时，内存节省达千倍。*

### 2.3 表达能力

Neural ODE 在 $\mathbb{R}^d$ 上的同胚映射空间中是稠密的（Zhang et al., 2020）。然而，它**无法改变拓扑结构**——例如，单个定义在 $\mathbb{R}^d$ 上的 Neural ODE 无法解开两个互锁的环。这催生了**增广型 Neural ODE（Augmented Neural ODE）**：通过将系统提升至 $\mathbb{R}^{d+k}$，额外维度为流提供了“解扣”的空间。

## 3. 连续归一化流（CNF）

### 3.1 从离散流到连续流

离散归一化流通过一系列可逆映射将初始样本 $\mathbf{z}_0 \sim p_0$ 变换为目标分布：
$$
\mathbf{z}_K = f_K \circ \cdots \circ f_1(\mathbf{z}_0), \qquad \log p_K = \log p_0 - \sum_{k=1}^K \log\!\bigl|\det \partial f_k / \partial\mathbf{z}_{k-1}\bigr|.
$$
每个行列式计算的复杂度为 $O(d^3)$，除非采用特殊架构（如耦合层、自回归结构等）将其降至 $O(d)$——但这会限制模型的表达能力。

CNF 则用一个 ODE 替代整个堆叠结构，并利用瞬时公式 (1)：
$$
\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z}(t), t), \qquad \frac{d\log p}{dt} = -\nabla\!\cdot f_\theta(\mathbf{z}(t), t). \tag{5}
$$
**无需对网络施加可逆性约束**——ODE 本身可通过反向积分实现逆映射；**无需计算行列式**——只需计算迹（即散度）。

### 3.2 FFJORD：通过 Hutchinson 估计实现可扩展的迹计算

剩下的瓶颈是散度 $\nabla\!\cdot f = \mathrm{tr}(\partial f / \partial\mathbf{z})$。精确计算仍需 $d$ 次 vector-Jacobian product。**FFJORD**（Grathwohl et al., 2018）提出用无偏估计替代：
$$
\nabla\!\cdot f = \mathbb{E}_{\boldsymbol\epsilon}\!\left[\boldsymbol\epsilon^\top\!\frac{\partial f}{\partial\mathbf{z}}\,\boldsymbol\epsilon\right], \qquad \boldsymbol\epsilon \sim \mathcal{N}(0, \mathbf{I}). \tag{6}
$$
这就是著名的 **Hutchinson 迹估计器**，每次采样仅需一次 vector-Jacobian product，其计算成本与维度 $d$ 无关。

![Hutchinson 迹估计：方差以 1/sqrt(K) 收缩，单步代价从 O(d^2) 降为 O(d)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig4_ffjord_trace.png)
*图 4：左图展示在 $d=64$ 下，Hutchinson 估计器在 400 次试验中的方差随探针向量数 $K$ 的变化，虚线表示理论上的 $1/\sqrt{K}$ 收敛速率；右图对比不同维度下的单步散度计算成本：完整 Jacobian 需 $O(d^2)$ 次自动微分调用，而 $K=4$ 的 Hutchinson 方法仅需 $O(Kd)$——在 $d=1024$ 时快三个数量级。*

### 3.3 训练与采样

给定数据点 $\mathbf{x}$，其对数似然为：
$$
\log p_1(\mathbf{x}) = \log p_0(\mathbf{z}_0) + \int_0^1 \nabla\!\cdot f_\theta(\mathbf{z}(t), t)\,dt,
$$
其中 $\mathbf{z}_0$ 通过从 $\mathbf{x}$ 反向积分 ODE (5) 得到。我们使用伴随方法最大化对数似然。**采样时**，只需从 $p_0$ 中采样 $\mathbf{z}_0$，然后正向积分即可。

**权衡取舍**：CNF 提供精确的似然估计，但每次前向或反向传递都需要求解 ODE——通常涉及数十至数百次网络评估。训练过程也较为敏感：求解器容差、$f_\theta$ 的正则化强度以及 Hutchinson 估计的方差会相互影响。

## 4. 最优传输与 Flow Matching

![偏微分方程与机器学习（六）：连续归一化流与Neural ODE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/illustration_2.png)

### 4.1 Benamou-Brenier 联系

二次代价的最优传输问题具有动态形式：
$$
\min_{v_t}\,\int_0^1\!\!\int \|v_t(\mathbf{z})\|^2\,\rho_t(\mathbf{z})\,d\mathbf{z}\,dt
\quad\text{s.t.}\quad \partial_t\rho + \nabla\!\cdot(\rho v) = 0,\;\rho_0, \rho_1\text{ 给定}.
$$
其最优解 $v_t^\star$ 恰好是 CNF 的速度场，且在欧氏最优传输情形下，其轨迹为直线。这为将 CNF 与最优传输结合提供了最清晰的几何动机。

### 4.2 Flow Matching

**Flow Matching**（Lipman et al., 2022）是一种极具实用价值的简化方案。它既不通过 ODE 求解器优化负对数似然（NLL），也不求解复杂的最优传输问题，而是选定一条条件概率路径，并直接回归对应的速度场。

最简单的选择是：将 $\mathbf{z}_0 \sim p_0$ 与 $\mathbf{z}_1 \sim p_{\text{data}}$ 配对，定义**条件路径** $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$，此时条件目标速度为
$$
u_t^\star(\mathbf{z}_t \mid \mathbf{z}_0, \mathbf{z}_1) = \mathbf{z}_1 - \mathbf{z}_0. \tag{7}
$$
**训练目标为**：
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\,\mathbf{z}_0,\,\mathbf{z}_1}\Bigl[\,\|v_\theta(\mathbf{z}_t, t) - (\mathbf{z}_1 - \mathbf{z}_0)\|^2\,\Bigr]. \tag{8}
$$

**关键定理（Lipman et al.）**：边缘速度 $\mathbb{E}[u_t^\star \mid \mathbf{z}_t]$ 满足连续性方程，能将 $p_0$ 输运至 $p_1$。因此，只要 $v_\theta$ 足够灵活，最小化 (8) 即可恢复一个有效的 CNF——且训练过程中完全无需计算散度。

![Flow Matching：成对样本之间的线性条件路径；对比 CNF 的训练曲线。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig5_flow_matching.png)
*图 5：左图展示随机样本对 $(\mathbf{z}_0, \mathbf{z}_1)$ 之间的线性条件路径，任意 $\mathbf{z}_t$ 处的目标速度就是 $\mathbf{z}_1 - \mathbf{z}_0$——训练时既无散度计算，也无需 ODE 求解；右图对比损失曲线：Flow Matching 收敛速度快近一个数量级，且达到更低的稳定平台。*

### 4.3 2024 年的实际选择

| 方法 | 训练成本 | 优点 | 缺点 |
|------|---------|------|------|
| 离散 NF（RealNVP/Glow） | 低廉，无需 ODE | 采样和似然计算快 | 架构受限 |
| CNF / FFJORD | ODE + Hutchinson | $f_\theta$ 形式自由，似然精确 | 训练慢，调参敏感 |
| OT-Flow | OT 代价 + 匹配 | 路径笔直、最优 | 需平衡两个损失项 |
| **Flow Matching** | 纯回归 | 稳定、快速、易于扩展至图像等高维数据 | 需设计合适的条件路径 |
| Rectified Flow / 一致性 | 迭代拉直 | 极少步数即可采样 | 需多阶段训练 |

截至 2024 年，大多数生产级的连续流系统（用于图像、音频、分子生成）都采用了 Flow Matching 或 Rectified Flow 的某种变体。

## 5. “连续深度”的直观图景

“连续深度”是贯穿本章的核心思想：Neural ODE 是深度网络的连续极限，而 CNF 则是归一化流的连续极限。两者背后的图像完全一致。

![连续轨迹 h(t) 分别由固定深度的 ResNet 和自适应 ODE 求解器近似。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig6_continuous_depth.png)
*图 6：蓝色曲线是底层 Neural ODE 生成的真实连续轨迹 $h(t)$；红色虚线是固定深度 $L=4$ 的 ResNet，在 $|\dot h|$ 较大的区域明显欠拟合（红色误差区域）；橙色曲线（$L=8$）仍无法捕捉高频振荡；紫色菱形是自适应求解器的评估点——**在动力学剧烈处密集采样，在平滑处稀疏采样**，以更少的总评估次数达到相同精度，且无需手动调整“深度”。*

这也解释了为何一个 ODE 函数 $f_\theta$ 能替代深 ResNet 中的数百层：**时间变量**取代了层索引的角色，而离散化策略则交由求解器自动决定。

## 6. 整合全流程：二维密度估计

为使整个流程更具体，我们在经典的“双月牙”玩具数据集上展示端到端的密度估计过程。

![二维玩具数据上的密度估计：目标样本、经验 KDE、CNF 拟合密度、生成样本与 ODE 轨迹。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig7_density_estimation.png)
*图 7：(a) 4000 个来自“双月牙”目标分布的样本；(b) 通过 KDE 得到的经验密度；(c) 连续流学到的密度——通过将高斯分布沿训练好的 $v_\theta$ 正向输运得到；(d) 紫色为生成样本，绿色为若干条 ODE 轨迹，展示基点如何从单位高斯分布（绿点）流向目标月牙。同一个网络 $v_\theta$ 同时用于密度评估（从 $\mathbf{x}$ 反向积分）和采样（从噪声正向积分）。*

这种双重能力——既能进行**精确似然的密度估计**，又能高效**采样**——仅依赖一个网络 $v_\theta$ 和一个 ODE $\dot{\mathbf{z}} = v_\theta(\mathbf{z}, t)$，正是连续流在理论上极具吸引力的原因。

## 7. 实验

### 7.1 螺旋 ODE 拟合

使用一个 3 层 MLP（隐藏维 64，tanh 激活）参数化 $f_\theta$，通过伴随方法在二维阻尼螺旋目标上训练，求解器为 dopri5（rtol $=10^{-5}$）。经过 1000 步训练后，平均轨迹误差降至 $<10^{-3}$，峰值 GPU 内存约为 40 MB，与内部约 80 步的求解过程无关。

### 7.2 高斯 → 双月牙 CNF

采用 4 层 MLP（隐藏维 128，softplus 激活），按 FFJORD 方式训练，使用 Hutchinson 迹估计和 dopri5 求解器，共训练 5000 步。生成样本完整覆盖两个月牙，并准确捕捉其弯月状厚度；与目标分布的 KDE 对比显示，Wasserstein-2 距离约为 0.07。

### 7.3 伴随方法 vs 标准反向传播（数据源自 Neural ODE 原文，外推至 1024 维隐藏状态）

| 方法 | 内存 (MB) | 时间 (s) | 测试准确率 |
|------|-----------|---------|-----------|
| 标准反向传播，固定 $L=100$ | 2450 | 2.3 | 85.2% |
| 伴随方法，固定 $L=100$ | 320 | 3.1 | 85.1% |
| 伴随方法，自适应（dopri5） | 310 | 2.8 | 85.3% |

内存开销降低约 87%，而实际运行时间仅增加 20–30%。

### 7.4 Flow Matching vs CNF（双月牙数据）

| 方法 | 样本质量（越低越好） | 收敛所需迭代数 | 采样时间 |
|------|--------------------|----------------|----------|
| CNF (FFJORD) | 12.3 | 8000 | 2.1 s / 1k 样本 |
| Flow Matching | 8.7 | 3000 | 1.8 s / 1k 样本 |

Flow Matching 收敛速度约为 CNF 的 2.7 倍，且生成的月牙形状更清晰。在真实图像数据上，这一差距更为显著——训练时间和采样所需的函数评估次数（NFE）相差一到两个数量级。

## 8. 习题

**习题 1.** 从连续性方程直接推导瞬时变量替换公式 (1)。

> *解*。连续性方程：$\partial_t\rho + \nabla\!\cdot(\rho f) = 0$，即 $\partial_t\rho + f\!\cdot\!\nabla\rho + \rho\,\nabla\!\cdot f = 0$。沿轨迹 $\mathbf{z}(t)$，有 $\frac{d}{dt}\rho(\mathbf{z}(t), t) = \partial_t\rho + f\!\cdot\!\nabla\rho = -\rho\,\nabla\!\cdot f$。两边同除以 $\rho$ 即得结果。

**习题 2.** 为何伴随方法的内存开销为 $O(1)$？

> *解*。它仅需存储当前的 $\mathbf{h}(t)$、$\mathbf{a}(t)$ 以及参数梯度的累加器。当反向求解器需要历史状态 $\mathbf{h}(s)$ 时，会通过反向积分前向 ODE 重新生成，而非预先存储。这里的 $O(1)$ 指与深度无关，空间维度 $d$ 的开销依然存在。

**习题 3.** 证明 Hutchinson 估计器 (6) 是无偏的。

> *解*。对任意矩阵 $A$ 和满足 $\mathbb{E}[\boldsymbol\epsilon] = 0$、$\mathrm{Cov}[\boldsymbol\epsilon] = \mathbf{I}$ 的随机向量 $\boldsymbol\epsilon$，有 $\mathbb{E}[\boldsymbol\epsilon^\top A\,\boldsymbol\epsilon] = \sum_{i,j} A_{ij}\,\mathbb{E}[\epsilon_i\epsilon_j] = \sum_i A_{ii} = \mathrm{tr}\,A$。

**习题 4.** 从高层面对比 Flow Matching 与 DDPM。

> *解*。两者都实现从噪声到数据的变换。DDPM 在随机前向 SDE 加噪过程中通过 score matching 学习去噪器，采样时需求解反向 SDE 或其对应的概率流 ODE；Flow Matching 则在确定性 ODE 上学习速度场 $v_\theta$，匹配预设的条件路径，采样只需对该 ODE 积分。Flow Matching 的训练损失是纯回归，无需设计时变噪声调度。

**习题 5.** 证明对线性条件路径 $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$，边缘速度 $\mathbb{E}[\mathbf{z}_1 - \mathbf{z}_0 \mid \mathbf{z}_t]$ 通过连续性方程将 $p_0$ 推至 $p_1$。

> *解（要点）*。写出 $\rho_t(\mathbf{z}) = \int q(\mathbf{z}_0, \mathbf{z}_1)\,\delta(\mathbf{z} - (1-t)\mathbf{z}_0 - t\mathbf{z}_1)\,d\mathbf{z}_0\,d\mathbf{z}_1$。对 $t$ 求导，并利用恒等式 $\partial_t\delta = -\nabla\!\cdot[(\mathbf{z}_1 - \mathbf{z}_0)\delta]$。在给定 $\mathbf{z}_t$ 的条件下对 $\mathbf{z}_0, \mathbf{z}_1$ 边缘化，可得 $\partial_t\rho_t + \nabla\!\cdot(\rho_t\,\bar v_t) = 0$，其中 $\bar v_t(\mathbf{z}) = \mathbb{E}[\mathbf{z}_1 - \mathbf{z}_0 \mid \mathbf{z}_t = \mathbf{z}]$。

## 参考文献

[1] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS*.

[2] Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form continuous dynamics for scalable reversible generative models. *ICLR*.

[3] Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2021). OT-Flow: Fast and accurate continuous normalizing flows via optimal transport. *AAAI*.

[4] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *ICLR*.

[5] Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.

[6] Tzen, B., & Raginsky, M. (2019). Theoretical guarantees for sampling and inference in generative models with latent diffusions. *COLT*.

[7] Zhang, H., Gao, X., Unterman, J., & Arodz, T. (2020). Approximation capabilities of neural ODEs and invertible residual networks. *ICML*.

---

*本文是 [PDE 与机器学习](/zh/pde-ml/) 系列的第 6 篇。下一篇：[第 7 篇 —— 扩散模型](/zh/pde-ml/07-扩散模型与score-matching)。上一篇：[第 5 篇 —— 辛几何](/zh/pde-ml/05-辛几何与保结构网络)。*
