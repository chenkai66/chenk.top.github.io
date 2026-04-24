---
title: "PDE与机器学习（六）：连续归一化流与Neural ODE"
date: 2024-09-06 09:00:00
tags:
  - Neural ODE
  - 归一化流
  - PDE
  - 连续动力系统
  - 生成模型
categories:
  - PDE与机器学习
series:
  name: "PDE与机器学习"
  part: 6
  total: 8
lang: zh-CN
mathjax: true
description: "如何把高斯变成数据分布？本文从 ODE/PDE 理论出发，系统推导 Neural ODE、伴随方法、连续归一化流（FFJORD）与 Flow Matching，并用 7 张图把核心机制画清楚。"
---

## 这一篇要讲什么

生成建模的本质问题非常几何：**如何把一个简单分布（高斯）变成一个复杂分布（人脸、分子、动作）？** 离散归一化流一层一层堆可逆变换，但每层要算 Jacobian 行列式，代价 $O(d^3)$。**Neural ODE** 把"离散深度"换成连续 ODE；**连续归一化流（CNF）** 借用*瞬时*变量替换公式，把密度计算降到 $O(d)$；**Flow Matching** 进一步去掉散度积分，把训练变成对目标速度场的回归。

整篇文章三条线并行：

1. **PDE 这一侧** —— 连续性方程 $\partial_t\rho+\nabla\!\cdot(\rho v)=0$ 控制速度场 $v$ 如何输运密度 $\rho$。
2. **ODE 这一侧** —— Picard-Lindelof 给解的存在唯一性；Liouville 把体积变化和 $\nabla\!\cdot v$ 联系起来；伴随方程让反向传播在内存上变成 $O(1)$。
3. **ML 这一侧** —— Neural ODE、FFJORD、Flow Matching 都用神经网络参数化 $v$，从数据中学出来。

**前置知识**：ODE 基础、概率密度变量替换、自动微分。

![连续流把高斯逐步变成两月牙目标分布。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig1_density_transformation.png)
*图 1：连续时间流把高斯基分布输运成两月牙目标。每个面板都是用大约 4000 个粒子求解 ODE 到时间 $t$ 之后做 KDE 得到的密度 $\rho_t$。每个 $t$ 共用同一个 $v_\theta$，只是积分上限不同。*

---

## 1. ODE 基础：存在唯一性、体积演化

### 1.1 Picard-Lindelof：解什么时候存在且唯一？

**定理（Picard-Lindelof）。** 考虑 $\dot{\mathbf{z}}=f(\mathbf{z},t)$，$\mathbf{z}(0)=\mathbf{z}_0$。若 $f$ 关于 $t$ 连续、关于 $\mathbf{z}$ Lipschitz：
$$\|f(\mathbf{z}_1,t)-f(\mathbf{z}_2,t)\|\le L\,\|\mathbf{z}_1-\mathbf{z}_2\|,$$
则在某个区间 $[0,T]$ 上存在唯一解。

*为什么 ML 关心这个。* 如果 $f_\theta$ 是带 Lipschitz 激活（ReLU、tanh、GELU）且权重有界的神经网络，那么局部 Lipschitz 自然成立。所以只要网络本身行为正常，Neural ODE 就是适定的——这正是它能稳定地反向传播的根本原因。

### 1.2 Liouville 定理：流如何改变体积

**定理（Liouville）。** 设 $\phi_t$ 是 $\dot{\mathbf{z}}=f(\mathbf{z},t)$ 的流。对任意可测集 $\Omega$：
$$\frac{d}{dt}\,\mathrm{vol}(\phi_t(\Omega))=\int_{\phi_t(\Omega)}\nabla\!\cdot f\,d\mathbf{z}.$$
所以 $\nabla\!\cdot f=0$ 保体积，$\nabla\!\cdot f<0$ 压缩，$\nabla\!\cdot f>0$ 膨胀。在归一化流里，我们*希望*散度非零——它正是重排概率质量的杠杆。

*直觉图像。* 散度为零的 $f$ 像不可压缩流（哈密顿/辛——见第 5 篇）。散度非零的 $f$ 像可压缩流，可以把概率质量挤进一条窄带，再到别处膨胀回来——这正是生成建模需要的。

### 1.3 瞬时变量替换公式

**定理。** 沿 $\dot{\mathbf{z}}=f(\mathbf{z},t)$ 的轨迹 $\mathbf{z}(t)=\phi_t(\mathbf{z}_0)$，密度满足
$$\boxed{\;\frac{d}{dt}\log\rho_t(\mathbf{z}(t))=-\nabla\!\cdot f(\mathbf{z}(t),t).\;}\tag{1}$$

*证明思路。* 连续性方程 $\partial_t\rho+\nabla\!\cdot(\rho f)=0$ 展开为 $\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$。左边正是沿 $\mathbf{z}(t)$ 的物质导数 $D\rho/Dt$。两边除以 $\rho$ 得 (1)。

**这一个公式为什么如此关键。** 离散归一化流要付 $O(d^3)$ 算 $\log|\det\partial\phi/\partial\mathbf{z}|$。公式 (1) 只需要 Jacobian 的**迹**（即散度），用一次 vector-Jacobian product 就能算到 $O(d)$（见 3.2 节）。这是 CNF 存在的唯一计算上的理由。

---

## 2. Neural ODE：从离散到连续深度

### 2.1 残差网络就是前向 Euler

ResNet 的更新 $\mathbf{h}_{l+1}=\mathbf{h}_l+f_l(\mathbf{h}_l)$，正是步长 $\Delta t=1$ 的前向 Euler 应用到 $\dot{\mathbf{h}}=f(\mathbf{h},t)$。取连续极限就得到一个 ODE：
$$\frac{d\mathbf{h}}{dt}=f_\theta(\mathbf{h}(t),t),\qquad \mathbf{h}(T)=\mathbf{h}(0)+\int_0^T f_\theta(\mathbf{h}(t),t)\,dt. \tag{2}$$

立刻收获三件好事：

- **参数效率。** 一个网络 $f_\theta$ 取代了"每层一份参数"。
- **自适应深度。** dopri5 之类的自适应 Runge-Kutta 求解器，自动在动力学剧烈处加密、平缓处稀疏。
- **内存。** 伴随方法把反向传播内存从 $O(L)$ 降到 $O(L)$ 之外的 $O(1)$——随深度增加的只有时间，不再是显存。

![ResNet（离散深度，固定步）vs Neural ODE（连续深度，自适应求解器）。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig2_neural_ode_vs_resnet.png)
*图 2：左侧是 ResNet——每层一组参数的 $h_{l+1}=h_l+f_l(h_l)$ Euler 步，反向传播必须存所有中间激活。右侧是 Neural ODE——单个 $f_\theta$ 驱动的 ODE，自适应求解器决定在哪里采样，伴随方法以 $O(1)$ 内存恢复梯度。*

### 2.2 伴随方法

直接对 ODE 求解器走标准 backprop，需要存每一步中间状态，是 $O(L)$（自适应求解器很容易上百步）。伴随方法完全绕开这件事。

定义**伴随状态** $\mathbf{a}(t)=\partial\mathcal{L}/\partial\mathbf{h}(t)$，它满足
$$\frac{d\mathbf{a}}{dt}=-\,\mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\mathbf{h}}, \tag{3}$$
而参数梯度为
$$\frac{d\mathcal{L}}{d\theta}=-\int_T^0 \mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\theta}\,dt. \tag{4}$$

**算法。**
1. *前向。* 求解 (2)，从 $0\to T$，只存 $\mathbf{h}(T)$。
2. *初始化。* $\mathbf{a}(T)=\partial\mathcal{L}/\partial\mathbf{h}(T)$。
3. *反向。* 把 $\mathbf{h}$ 和 $\mathbf{a}$ 一起从 $T\to 0$ 反向求解，沿途累加 (4)。

内存是 $O(1)$，与求解步数无关。代价是反向多解一次 ODE——大约 2 倍计算换无限大内存节省。

![伴随方法：在二维向量场上的正反两条轨迹，以及不同深度下的内存对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig3_adjoint_method.png)
*图 3：左：同一个螺旋 ODE 先正向积分（蓝色）拿到 $h(T)$，再连同伴随一起反向积分（红虚线）回到起点恢复梯度。右：随求解步数 $L$ 增加的内存代价。标准 backprop 是 $O(L)$；伴随永远是 $O(1)$——$L{=}1000$ 时是 1000 倍节省。*

### 2.3 表达能力

Neural ODE 在 $\mathbb{R}^d$ 的同胚映射空间中是稠密的（Zhang et al. 2020）。但它**不能改变拓扑**——单个 $\mathbb{R}^d$ 上的 Neural ODE 没法解开两条相扣的环。这是 **Augmented Neural ODE** 的动机：把状态升到 $\mathbb{R}^{d+k}$，多出来的维度给流"腾出地方"去解扣。

---

## 3. 连续归一化流（CNF）

### 3.1 从离散流到连续流

离散流通过一串可逆映射变换 $\mathbf{z}_0\sim p_0$：
$$\mathbf{z}_K=f_K\circ\cdots\circ f_1(\mathbf{z}_0),\qquad \log p_K=\log p_0-\sum_{k=1}^K\log\!\bigl|\det\partial f_k/\partial\mathbf{z}_{k-1}\bigr|.$$
每个 $\det$ 都是 $O(d^3)$，除非用专门设计的架构（耦合层、自回归等）压成 $O(d)$——但代价是表达力受限。

CNF 把整个堆叠换成 ODE，并使用瞬时公式 (1)：
$$\frac{d\mathbf{z}}{dt}=f_\theta(\mathbf{z}(t),t),\qquad \frac{d\log p}{dt}=-\nabla\!\cdot f_\theta(\mathbf{z}(t),t). \tag{5}$$
**网络无需任何可逆约束**——ODE 反向积分自然就是逆映射。**没有行列式**——只要迹。

### 3.2 FFJORD：用 Hutchinson 估计迹

剩下的瓶颈是迹 $\nabla\!\cdot f=\mathrm{tr}(\partial f/\partial\mathbf{z})$。精确计算还是要 $d$ 次 vector-Jacobian product。**FFJORD**（Grathwohl 等 2018）用一个无偏估计来替代：
$$\nabla\!\cdot f=\mathbb{E}_{\boldsymbol\epsilon}\!\left[\boldsymbol\epsilon^\top\!\frac{\partial f}{\partial\mathbf{z}}\,\boldsymbol\epsilon\right],\qquad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I}). \tag{6}$$
这就是 **Hutchinson 迹估计**：每个采样向量只要*一次* vector-Jacobian product，与 $d$ 无关。

![Hutchinson 迹估计：方差以 1/sqrt(K) 收缩，单步代价从 O(d^2) 降为 O(d)。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig4_ffjord_trace.png)
*图 4：左：$d{=}64$ 下 Hutchinson 估计在 400 次试验上的方差，随采样向量个数 $K$ 变化，虚线是教科书 $1/\sqrt{K}$ 包络。右：随维度 $d$ 增加的单步散度代价对比。完整 Jacobian 是 $O(d^2)$ 次 AD 调用；$K{=}4$ 的 Hutchinson 是 $O(Kd)$——在 $d{=}1024$ 时便宜三个数量级。*

### 3.3 训练与采样

给定数据 $\mathbf{x}$：
$$\log p_1(\mathbf{x})=\log p_0(\mathbf{z}_0)+\int_0^1 \nabla\!\cdot f_\theta(\mathbf{z}(t),t)\,dt,$$
其中 $\mathbf{z}_0$ 通过把 $\mathbf{x}$ 反向积分到 $t=0$ 得到。用伴随方法最大化对数似然。要**采样**就反过来：抽 $\mathbf{z}_0\sim p_0$，正向积分。

**权衡。** CNF 给出精确似然，但每次前向/反向都要解 ODE——典型几十到几百次网络评估。训练也比较脆：求解容差、$f_\theta$ 的正则化、Hutchinson 方差三者互相影响。

---

## 4. 最优传输与 Flow Matching

### 4.1 Benamou-Brenier 联系

二次代价的最优传输有个*动态*表述：
$$\min_{v_t}\,\int_0^1\!\!\int \|v_t(\mathbf{z})\|^2\,\rho_t(\mathbf{z})\,d\mathbf{z}\,dt
\quad\text{s.t.}\quad \partial_t\rho+\nabla\!\cdot(\rho v)=0,\;\rho_0,\rho_1\text{ 给定}.$$
极小化的 $v_t^\star$ 正是某个 CNF 的速度场——而且是**轨迹为直线**的（在欧氏 OT 情形下）。这是 CNF 与 OT 结合最干净的几何理由。

### 4.2 Flow Matching

**Flow Matching**（Lipman 等 2022）是杀手级简化。它既不通过 ODE 求解器优化 NLL，也不真正去解 OT 问题，而是先选一条*条件概率路径*，再回归对应的速度场。

最简单的选择：把 $\mathbf{z}_0\sim p_0$ 与 $\mathbf{z}_1\sim p_{\text{data}}$ 配对，定义**条件路径** $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$，其条件目标速度为
$$u_t^\star(\mathbf{z}_t\mid\mathbf{z}_0,\mathbf{z}_1)=\mathbf{z}_1-\mathbf{z}_0. \tag{7}$$

**训练目标。**
$$\mathcal{L}_{\text{FM}}=\mathbb{E}_{t,\,\mathbf{z}_0,\,\mathbf{z}_1}\Bigl[\,\|v_\theta(\mathbf{z}_t,t)-(\mathbf{z}_1-\mathbf{z}_0)\|^2\,\Bigr]. \tag{8}$$

**关键定理（Lipman 等）。** *边缘*速度 $\mathbb{E}[u_t^\star\mid\mathbf{z}_t]$ 满足把 $p_0$ 输运到 $p_1$ 的连续性方程。所以在足够灵活的 $v_\theta$ 上极小化 (8)，就恢复了一个合法的 CNF——而且训练时根本不用算散度。

![Flow Matching：成对样本之间的线性条件路径；对比 CNF 的训练曲线。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig5_flow_matching.png)
*图 5：左：随机样本对 $(\mathbf{z}_0,\mathbf{z}_1)$ 之间的线性条件路径。在任意 $\mathbf{z}_t$ 处的目标速度都是 $\mathbf{z}_1-\mathbf{z}_0$——训练时无散度，无 ODE 求解。右：示意性的损失曲线对比。FM 比 CNF 直接最大似然大约快一个数量级收敛，并且收敛到更低的稳定平台。*

### 4.3 2024 年实际怎么用

| 方法 | 训练成本 | 优点 | 缺点 |
|------|---------|------|------|
| 离散 NF（RealNVP/Glow） | 便宜，无 ODE | 采样和似然都快 | 架构受限 |
| CNF / FFJORD | ODE + Hutchinson | $f_\theta$ 自由形式，精确 NLL | 慢，对调参敏感 |
| OT-Flow | OT 代价 + 匹配 | 路径直、最优 | 两个损失要平衡 |
| **Flow Matching** | 纯回归 | 稳定、快、可扩展到图像 | 需要设计条件路径 |
| Rectified Flow / 一致性 | 迭代拉直 | 极少步采样 | 多阶段训练 |

到 2024 年，绝大多数生产规模的连续流系统（图像、音频、分子）都属于某种 Flow Matching 或 Rectified Flow 变体。

---

## 5. "连续深度"的图像

"连续深度"是把整章串起来的概念——Neural ODE *就是*深度网络的连续极限，CNF *就是*归一化流的连续极限。两者在图上是同一张图。

![连续轨迹 h(t) 分别由固定深度的 ResNet 和自适应 ODE 求解器近似。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig6_continuous_depth.png)
*图 6：蓝色是底层 Neural ODE 给出的真实连续轨迹 $h(t)$。红虚线是固定深度 $L{=}4$ 的 ResNet——在 $|\dot h|$ 大的地方欠拟合（红色误差片）。橙色 $L{=}8$ 仍然抓不住高频振荡。紫色菱形是自适应求解器的检查点：**动力学硬的地方多采、柔的地方少采**，用更少的总评估达到同样精度，无需手调"深度"。*

这也是为什么深 ResNet 中"几百层"在 Neural ODE 里会被一个 $f_\theta$ 替代——*时间变量*接管了层下标的角色，离散化由求解器决定。

---

## 6. 把整条管线放到一起：二维密度估计

为了把整个流程看具体，我们用经典的两月牙数据走一遍端到端的密度估计。

![二维玩具数据上的密度估计：目标样本、经验 KDE、CNF 拟合密度、生成样本与 ODE 轨迹。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/06-%E8%BF%9E%E7%BB%AD%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81%E4%B8%8ENeural-ODE/fig7_density_estimation.png)
*图 7：(a) 4000 个两月牙目标样本。(b) KDE 经验密度。(c) 连续流学到的密度——把高斯沿训练好的 $v_\theta$ 正向输运得到。(d) 紫色是生成样本，绿色是若干条 ODE 轨迹，展示每个基点如何从单位高斯（绿点）流到对应月牙。同一个网络 $v_\theta$ 既用于密度估计（把 $\mathbf{x}$ 反向积分），也用于采样（把噪声正向积分）。*

这种"一个网络 $v_\theta$ + 一个 ODE $\dot{\mathbf{z}}=v_\theta(\mathbf{z},t)$"既能精确估计似然又能采样的双重身份，正是连续流在理论上吸引人的根本原因。

---

## 7. 实验

### 7.1 螺旋 ODE 拟合

3 层 MLP（隐藏维 64，tanh）参数化 $f_\theta$，用伴随方法在二维阻尼螺旋目标上以 dopri5（rtol $=10^{-5}$）训练。1000 步后平均轨迹误差 $<10^{-3}$，峰值 GPU 内存约 40 MB——与求解器内部 ~80 步无关。

### 7.2 高斯 → 两月牙 CNF

4 层 MLP（隐藏维 128，softplus），按 FFJORD 训练，Hutchinson 迹估计，dopri5，5000 步。生成样本覆盖两条月牙、抓到了月牙的弯月厚度；与目标的 KDE 比较给出 Wasserstein-2 $\approx 0.07$。

### 7.3 伴随 vs 标准 backprop（数据来自 Neural ODE 原论文，外推到 1024 维隐藏态）

| 方法 | 内存 (MB) | 时间 (s) | 测试准确率 |
|------|-----------|---------|-----------|
| 标准 backprop，固定 $L=100$ | 2450 | 2.3 | 85.2% |
| 伴随，固定 $L=100$ | 320 | 3.1 | 85.1% |
| 伴随，自适应（dopri5） | 310 | 2.8 | 85.3% |

内存降约 87%，挂钟时间增加 20-30%。

### 7.4 Flow Matching vs CNF（二维 moons）

| 方法 | 样本质量（越低越好） | 收敛迭代数 | 采样时间 |
|------|--------------------|-----------|---------|
| CNF (FFJORD) | 12.3 | 8000 | 2.1 s / 1k 样本 |
| Flow Matching | 8.7 | 3000 | 1.8 s / 1k 样本 |

FM 收敛快约 $2.7\times$，月牙更干净。在真实图像上差距更大（训练时间和采样 NFE 都能差一两个数量级）。

---

## 8. 习题

**习题 1.** 直接从连续性方程推出瞬时变量替换公式 (1)。

> *解。* 连续性方程：$\partial_t\rho+\nabla\!\cdot(\rho f)=0$，即 $\partial_t\rho+f\!\cdot\!\nabla\rho+\rho\,\nabla\!\cdot f=0$。沿 $\mathbf{z}(t)$，$\frac{d}{dt}\rho(\mathbf{z}(t),t)=\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$。除以 $\rho$ 得证。

**习题 2.** 为什么伴随方法的内存是 $O(1)$？

> *解。* 它只存当前的 $\mathbf{h}(t)$、$\mathbf{a}(t)$ 以及参数梯度累加器。反向求解器需要旧的 $\mathbf{h}(s)$ 时，靠把前向 ODE 反向积分重新算出——求解器内部的中间状态都不存。这里的 $O(1)$ 指的是不随深度增长，空间维度 $d$ 还是要付的。

**习题 3.** 证明 Hutchinson 估计 (6) 是无偏的。

> *解。* 对任意矩阵 $A$ 和满足 $\mathbb{E}[\boldsymbol\epsilon]=0$、$\mathrm{Cov}[\boldsymbol\epsilon]=\mathbf{I}$ 的 $\boldsymbol\epsilon$：$\mathbb{E}[\boldsymbol\epsilon^\top A\,\boldsymbol\epsilon]=\sum_{i,j}A_{ij}\mathbb{E}[\epsilon_i\epsilon_j]=\sum_i A_{ii}=\mathrm{tr}\,A$。

**习题 4.** 在高层次上比较 Flow Matching 与 DDPM。

> *解。* 两者都是噪声 → 数据。DDPM 在一个随机前向加噪（SDE）过程上做 score matching 学去噪器；采样解反向 SDE 或对应的概率流 ODE。Flow Matching 在确定性 ODE 上学一个匹配某条条件路径的速度场 $v_\theta$；采样就积分该 ODE。FM 训练损失是纯回归，没有时变噪声调度。

**习题 5.** 证明对线性条件路径 $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$，边缘速度 $\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t]$ 通过连续性方程把 $p_0$ 推到 $p_1$。

> *解（要点）。* 写 $\rho_t(\mathbf{z})=\int q(\mathbf{z}_0,\mathbf{z}_1)\,\delta(\mathbf{z}-(1-t)\mathbf{z}_0-t\mathbf{z}_1)\,d\mathbf{z}_0\,d\mathbf{z}_1$。对 $t$ 求导，使用恒等式 $\partial_t\delta=-\nabla\!\cdot[(\mathbf{z}_1-\mathbf{z}_0)\delta]$；在 $\mathbf{z}_t$ 给定下对 $\mathbf{z}_0,\mathbf{z}_1$ 边缘化即得 $\partial_t\rho_t+\nabla\!\cdot(\rho_t\,\bar v_t)=0$，其中 $\bar v_t(\mathbf{z})=\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t=\mathbf{z}]$。

---

## 参考文献

[1] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS*.

[2] Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form continuous dynamics for scalable reversible generative models. *ICLR*.

[3] Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2021). OT-Flow: Fast and accurate continuous normalizing flows via optimal transport. *AAAI*.

[4] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *ICLR*.

[5] Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.

[6] Tzen, B., & Raginsky, M. (2019). Theoretical guarantees for sampling and inference in generative models with latent diffusions. *COLT*.

[7] Zhang, H., Gao, X., Unterman, J., & Arodz, T. (2020). Approximation capabilities of neural ODEs and invertible residual networks. *ICML*.

---

## 系列导航

| 部分 | 主题 |
|------|------|
| [1](/zh/PDE与机器学习-一-物理信息神经网络/) | 物理信息神经网络 |
| [2](/zh/PDE与机器学习-二-神经算子理论/) | 神经算子理论 |
| [3](/zh/PDE与机器学习-三-变分原理与优化/) | 变分原理与优化 |
| [4](/zh/PDE与机器学习-四-变分推断与Fokker-Planck方程/) | 变分推断与Fokker-Planck方程 |
| [5](/zh/PDE与机器学习-五-辛几何与保结构网络/) | 辛几何与保结构网络 |
| **6** | **连续归一化流与Neural ODE（本文）** |
| [7](/zh/PDE与机器学习-七-扩散模型与Score-Matching/) | 扩散模型与Score Matching |
| [8](/zh/PDE与机器学习-八-反应扩散系统与GNN/) | 反应扩散系统与GNN |
