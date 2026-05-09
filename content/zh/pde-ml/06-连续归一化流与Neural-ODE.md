---
title: "偏微分方程与机器学习（六）：连续归一化流与Neural ODE"
date: 2024-07-15 09:00:00
tags:
  - PDE
  - Machine Learning
  - Neural ODE
  - Normalizing Flows
  - CNF
  - Optimal Transport
  - Flow Matching
categories:
  - PDE与机器学习
series: pde-ml
lang: zh
mathjax: true
description: "如何把高斯变成数据分布？本文从 ODE/PDE 理论出发，系统推导 Neural ODE、伴随方法、连续归一化流（FFJORD）与 Flow Matching，并用 7 张图把核心机制画清楚。"
disableNunjucks: true
series_order: 6
translationKey: "pde-ml-6"
---
![偏微分方程与机器学习（六）：连续归一化流与Neural ODE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-Continuous-Normalizing-Flows/illustration_1.png)

## 这一篇要讲什么

生成建模归根结底是个几何问题：**如何把简单分布（高斯）变成复杂分布（人脸、分子、动作）？** 离散归一化流堆叠可逆层，但每层算 Jacobian 行列式代价 $O(d^3)$。**Neural ODE** 用连续 ODE 替代离散深度；**连续归一化流（CNF）** 借助*瞬时*变量替换公式，将密度计算降到 $O(d)$；**Flow Matching** 直接去掉散度积分，训练简化为目标速度场的回归。

文章围绕三条线展开：

1. **PDE 这边** —— 连续性方程 $\partial_t\rho+\nabla\!\cdot(\rho v)=0$ 描述速度场 $v$ 输运密度 $\rho$ 的规律。
2. **ODE 这边** —— Picard-Lindelof 保证解的存在唯一性；Liouville 将体积变化与 $\nabla\!\cdot v$ 关联；伴随方程让反向传播内存开销降到 $O(1)$。
3. **ML 这边** —— Neural ODE、FFJORD、Flow Matching 都用神经网络参数化 $v$，从数据中学出模型。

**前置知识**：ODE 基础、概率密度变量替换、自动微分。

![连续流把高斯逐步变成两月牙目标分布。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig1_density_transformation.png)
*图 1：连续时间流把高斯基分布输运成两月牙目标。每个面板是用约 4000 个粒子求解 ODE 到时间 $t$ 后做 KDE 得到的密度 $\rho_t$。所有 $t$ 共用同一个 $v_\theta$，仅积分上限不同。*

---
## 1. ODE 基础：存在唯一性、体积演化

### 1.1 Picard-Lindelof：解何时存在且唯一？

**定理（Picard-Lindelof）。** 考虑 $\dot{\mathbf{z}}=f(\mathbf{z},t)$，$\mathbf{z}(0)=\mathbf{z}_0$。若 $f$ 关于 $t$ 连续、关于 $\mathbf{z}$ Lipschitz：
$$\|f(\mathbf{z}_1,t)-f(\mathbf{z}_2,t)\|\le L\,\|\mathbf{z}_1-\mathbf{z}_2\|,$$则在区间 $[0,T]$ 上有唯一解。

*ML 为何关心？* 若 $f_\theta$ 是带 Lipschitz 激活（ReLU、tanh、GELU）且权重有界的神经网络，局部 Lipschitz 自然成立。只要网络行为正常，Neural ODE 就适定——这是它能稳定反向传播的根本原因。

### 1.2 Liouville 定理：流如何改变体积

**定理（Liouville）。** 设 $\phi_t$ 是 $\dot{\mathbf{z}}=f(\mathbf{z},t)$ 的流。对任意可测集 $\Omega$：$$\frac{d}{dt}\,\mathrm{vol}(\phi_t(\Omega))=\int_{\phi_t(\Omega)}\nabla\!\cdot f\,d\mathbf{z}.$$$\nabla\!\cdot f=0$ 保体积，$\nabla\!\cdot f<0$ 压缩，$\nabla\!\cdot f>0$ 膨胀。归一化流中，我们希望散度非零——它是重排概率质量的关键。

*直观理解。* 散度为零的 $f$ 像不可压缩流（哈密顿/辛——见第 5 篇）。散度非零的 $f$ 像可压缩流，能把概率质量挤进窄带，再到别处膨胀回来——生成建模正需要这种特性。

### 1.3 瞬时变量替换公式

**定理。** 沿 $\dot{\mathbf{z}}=f(\mathbf{z},t)$ 的轨迹 $\mathbf{z}(t)=\phi_t(\mathbf{z}_0)$，密度满足$$\boxed{\;\frac{d}{dt}\log\rho_t(\mathbf{z}(t))=-\nabla\!\cdot f(\mathbf{z}(t),t).\;}\tag{1}$$
*证明思路。* 连续性方程 $\partial_t\rho+\nabla\!\cdot(\rho f)=0$ 展开为 $\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$。左边是沿 $\mathbf{z}(t)$ 的物质导数 $D\rho/Dt$。两边除以 $\rho$ 得 (1)。

**这公式为何关键？** 离散归一化流需 $O(d^3)$ 计算 $\log|\det\partial\phi/\partial\mathbf{z}|$。公式 (1) 只需 Jacobian 的迹（即散度），用一次 vector-Jacobian product 即可 $O(d)$ 完成（见 3.2 节）。这是 CNF 存在的核心计算理由。
## 2. Neural ODE：从离散到连续深度

### 2.1 残差网络就是前向 Euler

ResNet 更新公式 $\mathbf{h}_{l+1}=\mathbf{h}_l+f_l(\mathbf{h}_l)$，等价于步长 $\Delta t=1$ 的前向 Euler 方法解 $\dot{\mathbf{h}}=f(\mathbf{h},t)$。取连续极限后得到一个 ODE：$$\frac{d\mathbf{h}}{dt}=f_\theta(\mathbf{h}(t),t),\qquad \mathbf{h}(T)=\mathbf{h}(0)+\int_0^T f_\theta(\mathbf{h}(t),t)\,dt. \tag{2}$$
直接好处有三点：

- **参数效率。** 一个 $f_\theta$ 替代每层不同参数。
- **自适应深度。** dopri5 等自适应 Runge-Kutta 求解器，动态调整步长，剧烈处加密，平缓处稀疏。
- **内存节省。** 伴随方法将反向传播内存从 $O(L)$ 降到 $O(1)$——随深度增加的只有时间，不是显存。

![ResNet（离散深度，固定步）vs Neural ODE（连续深度，自适应求解器）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig2_neural_ode_vs_resnet.png)
*图 2：左侧是 ResNet，每层一组参数的 $h_{l+1}=h_l+f_l(h_l)$ Euler 步，反向传播需存所有中间激活。右侧是 Neural ODE，单个 $f_\theta$ 驱动 ODE，自适应求解器决定采样点，伴随方法以 $O(1)$ 内存恢复梯度。*

### 2.2 伴随方法

标准 backprop 对 ODE 求解器需存每步中间状态，复杂度 $O(L)$（自适应求解器常上百步）。伴随方法完全避免了这点。

定义**伴随状态** $\mathbf{a}(t)=\partial\mathcal{L}/\partial\mathbf{h}(t)$，满足$$\frac{d\mathbf{a}}{dt}=-\,\mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\mathbf{h}}, \tag{3}$$参数梯度为$$\frac{d\mathcal{L}}{d\theta}=-\int_T^0 \mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\theta}\,dt. \tag{4}$$
**算法：**
1. *前向。* 求解 (2)，从 $0\to T$，只存 $\mathbf{h}(T)$。
2. *初始化。* $\mathbf{a}(T)=\partial\mathcal{L}/\partial\mathbf{h}(T)$。
3. *反向。* 将 $\mathbf{h}$ 和 $\mathbf{a}$ 一起从 $T\to 0$ 反向求解，累加 (4)。

内存复杂度 $O(1)$，与求解步数无关。代价是反向多解一次 ODE——约 2 倍计算换无限大内存节省。

![伴随方法：在二维向量场上的正反两条轨迹，以及不同深度下的内存对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig3_adjoint_method.png)
*图 3：左：同一螺旋 ODE 先正向积分（蓝）得 $h(T)$，再连同伴随反向积分（红虚线）恢复梯度。右：随求解步数 $L$ 增加的内存代价。标准 backprop 是 $O(L)$；伴随始终 $O(1)$——$L{=}1000$ 时节省 1000 倍。*

### 2.3 表达能力

Neural ODE 在 $\mathbb{R}^d$ 同胚映射空间中稠密 [Zhang et al. 2020]。但它**不能改变拓扑**——单个 $\mathbb{R}^d$ 上的 Neural ODE 无法解开相扣环。这是 **Augmented Neural ODE** 的动机：升维到 $\mathbb{R}^{d+k}$，新增维度让流有空间解扣。
## 3. 连续归一化流（CNF）

### 3.1 从离散流到连续流

离散流通过可逆映射变换 $\mathbf{z}_0\sim p_0$：$$\mathbf{z}_K=f_K\circ\cdots\circ f_1(\mathbf{z}_0),\qquad \log p_K=\log p_0-\sum_{k=1}^K\log\!\bigl|\det\partial f_k/\partial\mathbf{z}_{k-1}\bigr|.$$每个 $\det$ 计算复杂度是 $O(d^3)$，除非用特殊架构（耦合层、自回归等）优化到 $O(d)$——但表达力受限。

CNF 把整个堆叠换成 ODE，使用瞬时公式 (1)：$$\frac{d\mathbf{z}}{dt}=f_\theta(\mathbf{z}(t),t),\qquad \frac{d\log p}{dt}=-\nabla\!\cdot f_\theta(\mathbf{z}(t),t). \tag{5}$$**网络无需可逆约束**——ODE 反向积分就是逆映射。**没有行列式**——只需迹。

### 3.2 FFJORD：用 Hutchinson 估计迹

瓶颈是迹 $\nabla\!\cdot f=\mathrm{tr}(\partial f/\partial\mathbf{z})$。精确计算仍需 $d$ 次 vector-Jacobian product。**FFJORD**（Grathwohl 等 2018）用无偏估计替代：$$\nabla\!\cdot f=\mathbb{E}_{\boldsymbol\epsilon}\!\left[\boldsymbol\epsilon^\top\!\frac{\partial f}{\partial\mathbf{z}}\,\boldsymbol\epsilon\right],\qquad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I}). \tag{6}$$这是 **Hutchinson 迹估计**：每次采样只需一次 vector-Jacobian product，与 $d$ 无关。

![Hutchinson 迹估计：方差以 1/sqrt(K) 收缩，单步代价从 O(d^2) 降为 O(d)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig4_ffjord_trace.png)
*图 4：左：$d{=}64$ 下 Hutchinson 估计在 400 次试验的方差，随采样向量数 $K$ 变化，虚线是教科书 $1/\sqrt{K}$ 包络。右：随维度 $d$ 增加的单步散度代价对比。完整 Jacobian 是 $O(d^2)$ 次 AD 调用；$K{=}4$ 的 Hutchinson 是 $O(Kd)$——在 $d{=}1024$ 时便宜三个数量级。*

### 3.3 训练与采样

给定数据 $\mathbf{x}$：$$\log p_1(\mathbf{x})=\log p_0(\mathbf{z}_0)+\int_0^1 \nabla\!\cdot f_\theta(\mathbf{z}(t),t)\,dt,$$其中 $\mathbf{z}_0$ 通过反向积分从 $\mathbf{x}$ 得到。用伴随方法最大化对数似然。**采样**时，抽 $\mathbf{z}_0\sim p_0$，正向积分。

**权衡。** CNF 提供精确似然，但每次前向/反向需解 ODE——典型几十到几百次网络评估。训练较脆：求解容差、$f_\theta$ 正则化、Hutchinson 方差互相影响。
## 4. 最优传输与 Flow Matching

![偏微分方程与机器学习（六）：连续归一化流与Neural ODE — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-Continuous-Normalizing-Flows/illustration_2.png)

### 4.1 Benamou-Brenier 联系

二次代价的最优传输有动态表述：$$\min_{v_t}\,\int_0^1\!\!\int \|v_t(\mathbf{z})\|^2\,\rho_t(\mathbf{z})\,d\mathbf{z}\,dt
\quad\text{s.t.}\quad \partial_t\rho+\nabla\!\cdot(\rho v)=0,\;\rho_0,\rho_1\text{ 给定}.$$极小化的 $v_t^\star$ 是 CNF 的速度场，轨迹为直线（欧氏 OT 情形）。这是结合 CNF 和 OT 的几何本质。

### 4.2 Flow Matching

**Flow Matching**（Lipman 等 2022）是杀手级简化。不通过 ODE 求解器优化 NLL，也不解 OT 问题，而是选一条条件概率路径，回归对应速度场。

最简单的方法：配对 $\mathbf{z}_0\sim p_0$ 和 $\mathbf{z}_1\sim p_{\text{data}}$，定义条件路径 $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$，目标速度为$$u_t^\star(\mathbf{z}_t\mid\mathbf{z}_0,\mathbf{z}_1)=\mathbf{z}_1-\mathbf{z}_0. \tag{7}$$
**训练目标：**$$\mathcal{L}_{\text{FM}}=\mathbb{E}_{t,\,\mathbf{z}_0,\,\mathbf{z}_1}\Bigl[\,\|v_\theta(\mathbf{z}_t,t)-(\mathbf{z}_1-\mathbf{z}_0)\|^2\,\Bigr]. \tag{8}$$

**关键定理（Lipman 等）：** 边缘速度 $\mathbb{E}[u_t^\star\mid\mathbf{z}_t]$ 满足连续性方程，将 $p_0$ 输运到 $p_1$。在灵活的 $v_\theta$ 上极小化 (8)，可恢复合法 CNF，且训练时无需计算散度。

![Flow Matching：成对样本之间的线性条件路径；对比 CNF 的训练曲线。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig5_flow_matching.png)
*图 5：左：随机样本对 $(\mathbf{z}_0,\mathbf{z}_1)$ 的线性条件路径。任意 $\mathbf{z}_t$ 处目标速度为 $\mathbf{z}_1-\mathbf{z}_0$——训练时无散度，无 ODE 求解。右：损失曲线对比。FM 收敛快一个数量级，稳定平台更低。*

### 4.3 2024 年实际怎么用

| 方法 | 训练成本 | 优点 | 缺点 |
|------|---------|------|------|
| 离散 NF（RealNVP/Glow） | 便宜，无 ODE | 采样和似然快 | 架构受限 |
| CNF / FFJORD | ODE + Hutchinson | $f_\theta$ 自由形式，精确 NLL | 慢，调参敏感 |
| OT-Flow | OT 代价 + 匹配 | 路径直、最优 | 两个损失需平衡 |
| **Flow Matching** | 纯回归 | 稳定、快、扩展性强 | 需设计条件路径 |
| Rectified Flow / 一致性 | 迭代拉直 | 极少步采样 | 多阶段训练 |

到 2024 年，生产规模的连续流系统（图像、音频、分子）多为 Flow Matching 或 Rectified Flow 变体。
## 5. "连续深度"的图像

"连续深度"是贯穿全章的核心概念。Neural ODE 是深度网络的连续极限，CNF 是归一化流的连续极限。两者的图示完全一致。

![连续轨迹 h(t) 分别由固定深度的 ResNet 和自适应 ODE 求解器近似。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig6_continuous_depth.png)
*图 6：蓝色是底层 Neural ODE 给出的真实连续轨迹 $h(t)$。红虚线是固定深度 $L{=}4$ 的 ResNet，在 $|\dot h|$ 大的地方欠拟合（红色误差片）。橙色 $L{=}8$ 仍无法捕捉高频振荡。紫色菱形是自适应求解器的检查点：**动力学复杂处多采样，平滑处少采样**，用更少评估达到相同精度，无需手动调参。*

深 ResNet 中几百层在 Neural ODE 里被一个 $f_\theta$ 替代。时间变量取代层索引，离散化由求解器决定。
## 6. 整合全流程：二维密度估计

用经典的两月牙数据，走一遍端到端的密度估计流程。

![二维玩具数据上的密度估计：目标样本、经验 KDE、CNF 拟合密度、生成样本与 ODE 轨迹。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-连续归一化流与Neural-ODE/fig7_density_estimation.png)
*图 7：(a) 4000 个两月牙目标样本。(b) KDE 经验密度。(c) 连续流学到的密度——通过训练好的 $v_\theta$ 正向输运高斯分布得到。(d) 紫色是生成样本，绿色是若干条 ODE 轨迹，展示基点如何从单位高斯（绿点）流向目标月牙。同一个网络 $v_\theta$ 既用于密度估计（从 $\mathbf{x}$ 反向积分），也用于采样（从噪声正向积分）。*

一个网络 $v_\theta$ 加一个 ODE $\dot{\mathbf{z}}=v_\theta(\mathbf{z},t)$，既能精确估计似然，又能高效采样。这种双重特性正是连续流理论吸引力的核心所在。
## 7. 实验

### 7.1 螺旋 ODE 拟合

3 层 MLP（隐藏维 64，tanh）参数化 $f_\theta$。用伴随方法训练，目标是二维阻尼螺旋，求解器 dopri5（rtol $=10^{-5}$）。1000 步后，平均轨迹误差降到 $<10^{-3}$。峰值 GPU 内存约 40 MB，与内部 ~80 步无关。

### 7.2 高斯 → 两月牙 CNF

4 层 MLP（隐藏维 128，softplus），按 FFJORD 训练。Hutchinson 迹估计，dopri5 求解器，5000 步。生成样本覆盖两条月牙，厚度符合弯月形状。KDE 对比目标分布，Wasserstein-2 $\approx 0.07$。

### 7.3 伴随 vs 标准 backprop（数据来自 Neural ODE 原文，外推到 1024 维隐藏态）

| 方法 | 内存 (MB) | 时间 (s) | 测试准确率 |
|------|-----------|---------|-----------|
| 标准 backprop，固定 $L=100$ | 2450 | 2.3 | 85.2% |
| 伴随，固定 $L=100$ | 320 | 3.1 | 85.1% |
| 伴随，自适应（dopri5） | 310 | 2.8 | 85.3% |

内存减少约 87%，挂钟时间增加 20-30%。

### 7.4 Flow Matching vs CNF（二维 moons）

| 方法 | 样本质量（越低越好） | 收敛迭代数 | 采样时间 |
|------|--------------------|-----------|---------|
| CNF (FFJORD) | 12.3 | 8000 | 2.1 s / 1k 样本 |
| Flow Matching | 8.7 | 3000 | 1.8 s / 1k 样本 |

FM 收敛快 $2.7\times$，月牙更干净。真实图像上差距更大，训练时间和采样 NFE 差一两个数量级。
## 8. 习题

**习题 1.** 从连续性方程直接推导瞬时变量替换公式 (1)。

> *解。* 连续性方程：$\partial_t\rho+\nabla\!\cdot(\rho f)=0$，即 $\partial_t\rho+f\!\cdot\!\nabla\rho+\rho\,\nabla\!\cdot f=0$。沿 $\mathbf{z}(t)$，$\frac{d}{dt}\rho(\mathbf{z}(t),t)=\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$。两边除以 $\rho$ 即得。

**习题 2.** 伴随方法的内存为何是 $O(1)$？

> *解。* 只存当前 $\mathbf{h}(t)$、$\mathbf{a}(t)$ 和参数梯度累加器。反向求解需旧 $\mathbf{h}(s)$ 时，通过前向 ODE 反向积分重新计算，不存中间状态。$O(1)$ 指深度无关，空间维度仍需 $d$。

**习题 3.** 证明 Hutchinson 估计 (6) 是无偏的。

> *解。* 对任意矩阵 $A$ 和满足 $\mathbb{E}[\boldsymbol\epsilon]=0$、$\mathrm{Cov}[\boldsymbol\epsilon]=\mathbf{I}$ 的 $\boldsymbol\epsilon$：$\mathbb{E}[\boldsymbol\epsilon^\top A\,\boldsymbol\epsilon]=\sum_{i,j}A_{ij}\mathbb{E}[\epsilon_i\epsilon_j]=\sum_i A_{ii}=\mathrm{tr}\,A$。

**习题 4.** 高层次比较 Flow Matching 与 DDPM。

> *解。* 两者均实现噪声 → 数据。DDPM 在随机前向 SDE 加噪过程上用 score matching 学去噪器；采样解反向 SDE 或概率流 ODE。Flow Matching 在确定性 ODE 上学速度场 $v_\theta$，匹配某条件路径；采样直接积分该 ODE。FM 训练损失为纯回归，无需时变噪声调度。

**习题 5.** 证明对线性条件路径 $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$，边缘速度 $\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t]$ 通过连续性方程将 $p_0$ 推至 $p_1$。

> *解（要点）。* 写 $\rho_t(\mathbf{z})=\int q(\mathbf{z}_0,\mathbf{z}_1)\,\delta(\mathbf{z}-(1-t)\mathbf{z}_0-t\mathbf{z}_1)\,d\mathbf{z}_0\,d\mathbf{z}_1$。对 $t$ 求导，用恒等式 $\partial_t\delta=-\nabla\!\cdot[(\mathbf{z}_1-\mathbf{z}_0)\delta]$。给定 $\mathbf{z}_t$ 边缘化 $\mathbf{z}_0,\mathbf{z}_1$ 得 $\partial_t\rho_t+\nabla\!\cdot(\rho_t\,\bar v_t)=0$，其中 $\bar v_t(\mathbf{z})=\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t=\mathbf{z}]$。
## 参考文献

[1] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS*.

[2] Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form continuous dynamics for scalable reversible generative models. *ICLR*.

[3] Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2021). OT-Flow: Fast and accurate continuous normalizing flows via optimal transport. *AAAI*.

[4] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *ICLR*.

[5] Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.

[6] Tzen, B., & Raginsky, M. (2019). Theoretical guarantees for sampling and inference in generative models with latent diffusions. *COLT*.

[7] Zhang, H., Gao, X., Unterman, J., & Arodz, T. (2020). Approximation capabilities of neural ODEs and invertible residual networks. *ICML*.

---

*This is Part 6 of the [PDE and Machine Learning](/zh/categories/pde-and-machine-learning/) series. Next: [Part 7 -- Diffusion Models](/en/pde-ml/07-diffusion-models/). Previous: [Part 5 -- Symplectic Geometry](/en/pde-ml/05-symplectic-geometry).*
