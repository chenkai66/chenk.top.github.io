---
title: "辛几何与结构保持神经网络：让模型学会守恒"
date: 2025-01-30 09:00:00
tags:
  - PDE
  - Machine Learning
  - Hamiltonian Systems
  - Symplectic Geometry
  - Structure-Preserving Learning
categories: PDE and Machine Learning
lang: zh-CN
mathjax: true
description: "理解能保持能量与辛结构的物理感知神经网络。涵盖 HNN、LNN、SympNet、辛积分器，以及四个经典物理系统实验。"
disableNunjucks: true
---

随手训练一个普通 MLP 去拟合一维谐振子的运动。验证集上误差很小，前十步看着也对。然后让它一口气往后推一千步——轨道不再闭合，能量缓慢漂移，本该周期运动的系统变成了一条慢慢张开的螺旋。网络学到了"数据点之间的插值"，没学到"物理"。**结构保持网络**（structure-preserving NN）的做法，是把守恒律——能量守恒、辛 2-形式、欧拉-拉格朗日方程——直接编码进架构里，使得模型从数学结构上就不可能违反这些约束，无论积分多长时间。

## 你将学到

- 为什么短期误差很小的普通 NN，在长时预测里仍然会"漂"
- 哈密顿力学的最少必需品：相空间、Hamilton 方程、Poisson 括号
- 读论文够用的辛几何：闭非退化 2-形式、Darboux 定理、Liouville 定理
- 为什么辛积分器（Verlet、辛 RK）的能量误差是有界振荡，而不是线性增长
- 三种主流架构与选型：Hamiltonian NN（HNN）、Lagrangian NN（LNN）、Symplectic NN（SympNet）
- 四个经典实验：谐振子、双摆（混沌）、Kepler 问题、Lennard-Jones 分子动力学

## 前置知识

- 多元微积分与线性代数
- 用 PyTorch 训过 MLP
- 经典力学（能量、动量）有概念即可，文中按需补全

---

## 1. 为什么需要结构保持？

### 1.1 普通网络的失败模式

最简单的 Hamilton 系统——一维谐振子：

$$
H(q, p) \;=\; \tfrac{1}{2} p^{2} + \tfrac{1}{2}\omega^{2} q^{2},
$$

Hamilton 方程 $\dot q = p$、$\dot p = -\omega^{2} q$，精确解 $q(t) = A\cos(\omega t + \varphi)$，能量恒定。

用一个 MLP $f_{\theta}(q, p) \approx (\dot q, \dot p)$ 在干净的轨迹上训练。训完单步误差也许只有 $10^{-4}$。但只要把它当成动力系统积上一千步，两件事就会发生：

1. **能量漂移**。$H(q_t, p_t)$ 不再围绕 $H(q_0, p_0)$ 振荡，而是单调地走开。走的方向取决于优化器、初始化甚至随机种子——损失函数里没有任何机制把"学到的向量场"和"某个守恒标量"绑在一起。
2. **相位累积**。每步几乎可以忽略的小误差累加起来，几百步之后预测轨道与真实轨道相位差就到了周期的相当一部分。

这两个问题不是统计的，是**结构性的**。MLP 能拟合任意光滑的 $\mathbb{R}^2 \to \mathbb{R}^2$ 映射，而由 Hamilton 量生成的"辛映射"是这个空间里的一个测度零的子流形。SGD 没有任何理由恰好落在上面。

### 1.2 结构性的解法

结构保持网络的思路：把假设空间限制成"已经满足守恒律"的那部分。常见的三种做法，差别在于"网络代表什么"：

- **HNN**——网络代表标量函数 $H_{\theta}(q,p)$。Hamilton 方程是后处理的解析步骤（用 autograd 求梯度），所以只要 $H_{\theta}$ 不显含时间，能量就严格守恒。
- **LNN**——网络代表 Lagrange 函数 $L_{\theta}(q, \dot q)$，加速度从欧拉-拉格朗日方程反解。当动量定义不方便时（比如约束力学）特别合用。
- **SympNet**——网络**直接是**一个辛映射 $\Phi_{\theta} : (q_t, p_t) \mapsto (q_{t+1}, p_{t+1})$，由若干基本"剪切层"复合而成，每一层的 Jacobian 都满足 $J^\top \Omega J = \Omega$。整个过程不显式写下任何 Hamilton 量。

下面几节把后续讨论需要的几何最小集补齐。

## 2. 一张图讲清楚 Hamilton 力学

### 2.1 相空间与 Hamilton 方程

$n$ 自由度系统的**相空间** $M$ 是 $2n$ 维流形，局部坐标 $z = (q, p) = (q_1,\dots,q_n,p_1,\dots,p_n)$。**Hamilton 量** $H : M \times \mathbb{R} \to \mathbb{R}$ 是一个光滑函数，通常代表总能量。**Hamilton 方程**：

$$
\dot{q}_{i} \;=\; \frac{\partial H}{\partial p_{i}}, \qquad \dot{p}_{i} \;=\; -\frac{\partial H}{\partial q_{i}}.
$$

向量形式：

$$
\dot{z} \;=\; J \,\nabla H(z), \qquad
J \;=\; \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}.
$$

矩阵 $J$ 叫做**典范辛矩阵**，满足 $J^\top = -J$，$J^2 = -I_{2n}$。

### 2.2 Poisson 括号

两个光滑可观测量 $f, g$ 的 **Poisson 括号** 定义为

$$
\{f, g\} \;=\; \sum_{i=1}^{n} \left( \frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i} \right) \;=\; (\nabla f)^\top J\, \nabla g.
$$

它双线性、反对称、满足 Leibniz 律和 Jacobi 恒等式。任何观测量沿流的演化是

$$
\frac{d f}{dt} \;=\; \{f, H\} + \frac{\partial f}{\partial t}.
$$

把 $f$ 取成 $H$ 自己，立刻得到 $\dot H = \{H, H\} = 0$——**能量自动守恒**，一行就证完了。

## 3. 够用的辛几何

![辛 2-形式与一般 2-形式的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/fig1_two_forms.png)

### 3.1 辛形式

$2n$ 维流形 $M$ 上的**辛形式** $\omega$ 是一个**闭**（$d\omega = 0$）且**非退化**的 2-形式。非退化的意思是：对任何非零切向量 $u \in T_z M$，都存在 $v$ 使 $\omega(u, v) \ne 0$。在典范坐标下，

$$
\omega \;=\; \sum_{i=1}^{n} dq_{i} \wedge dp_{i}.
$$

几何上，$\omega(u, v)$ 是 $u, v$ 在每个 $(q_i, p_i)$ 平面上张成的平行四边形的**有向面积**之和（图 1 左）。右图给出了非退化失败时的反面教材：$\eta = x\,dx \wedge dy$ 在 $x = 0$ 直线上恒为零，那里就无法定义合理的 Hamilton 流。

**Darboux 定理** 告诉我们这是局部上**唯一**的形式：在任何辛流形上，每个点附近都能找到坐标使 $\omega$ 化为 $\sum_i dq_i \wedge dp_i$。辛流形没有局部不变量（除维数外），所有有趣的结构都是全局的（辛容量、辛上同调……）。但对本文够用，局部图像就足够了。

### 3.2 辛映射与 Liouville 定理

微分同胚 $\Phi : M \to M$ 称为**辛的**（也叫正则变换），如果 $\Phi^{*}\omega = \omega$，等价地，其 Jacobian 满足

$$
J_{\Phi}^{\top}\, \Omega\, J_{\Phi} \;=\; \Omega, \qquad \Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}.
$$

任意 Hamilton 流 $\varphi_{t}^{H}$ 对每个 $t$ 都是辛映射。直接推论是 **Liouville 定理**：体积形式 $\omega^{n}/n!$ 守恒——相空间里任意一块区域被流推移之后，体积不变。

![Liouville 定理在单摆上的演示：方块变形，但面积守恒](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/fig2_phase_conservation.png)

图 2 在单摆 $H = \tfrac{1}{2}p^2 + \frac{g}{L}(1 - \cos q)$ 上把这件事画了出来。一小块方形被流推动，它逐渐被拉伸、剪切，变成一条又细又弯的"小舌头"，但**沿边界用鞋带公式算出来的面积，到第四位小数都没变**。这就是长时数值方法必须保留、而普通网络几乎注定破坏的那条性质。

## 4. 辛积分器

### 4.1 为什么标准方法会漂

显式 Euler、经典 RK4、几乎所有显式 Runge-Kutta 都是**非辛**的。它们生成的离散映射 Jacobian 不满足辛条件，能量误差会随时间近乎线性增长：

$$
|\,H(z_{n}) - H(z_{0})\,| \;\sim\; C\, t \, h^{p} \quad (\text{非辛}),
$$

$h$ 是步长，$p$ 是阶。辛方法的情况则在质上不同：**反向误差分析**告诉我们，辛方法实际上**精确地**积分了一个略微扰动的 Hamilton 量 $\tilde H = H + h^{p} H_{1} + \cdots$。所以**真实的** $H$ 在指数长的时间里都只是在 $H_0$ 附近一个 $O(h^{p})$ 宽的带子里振荡：

$$
|\,H(z_{n}) - H(z_{0})\,| \;\le\; C\, h^{p} \quad (\text{辛、可积}).
$$

这一条不等式就是为什么所有生产环境的分子动力学代码都用辛积分器。

### 4.2 Verlet（Störmer / leapfrog）

对可分 Hamilton 量 $H(q, p) = \tfrac{1}{2}p^{\top} M^{-1} p + V(q)$，**速度 Verlet** 步骤为：

$$
\begin{aligned}
p_{n + \tfrac{1}{2}} &= p_{n} - \tfrac{h}{2}\,\nabla V(q_{n}), \\
q_{n+1} &= q_{n} + h\, M^{-1} p_{n+\tfrac{1}{2}}, \\
p_{n+1} &= p_{n+\tfrac{1}{2}} - \tfrac{h}{2}\,\nabla V(q_{n+1}).
\end{aligned}
$$

二阶、对称、显式（不需要解隐式方程）、辛——离散映射可以拆成两个剪切再加一个平移，每一个都明显保体积。LAMMPS、GROMACS 等大型 MD 包的"主力发动机"都是 Verlet。

### 4.3 辛 Runge-Kutta

非可分 Hamilton 量需要隐式方法。$s$ 阶 Runge-Kutta（系数 $a_{ij}$、$b_i$）是辛的，当且仅当

$$
b_{i}\, a_{ij} + b_{j}\, a_{ji} \;=\; b_{i}\, b_{j}, \quad \forall i, j.
$$

Gauss-Legendre 配点法对所有阶都满足这个条件。最简单的 $s=1$ 即**隐式中点法**：

$$
z_{n+1} \;=\; z_{n} + h\, J\, \nabla H\!\left(\tfrac{z_{n} + z_{n+1}}{2}\right),
$$

二阶、辛、无条件 B-稳定。

## 5. Hamiltonian Neural Networks（HNN）

### 5.1 核心想法

Greydanus、Dzamba 与 Yosinski（2019）的关键观察是：**别去回归 $\dot z$，去回归 $H$**。让网络 $H_{\theta} : \mathbb{R}^{2n} \to \mathbb{R}$ 学习标量 Hamilton 量，然后用解析公式拿到动力学：

$$
\dot z \;=\; J \,\nabla_{z} H_{\theta}(z).
$$

右边的 $\nabla_z H_\theta$ 在训练时由 autograd 计算。因为 $\dot H_{\theta} = (\nabla H_{\theta})^\top J \nabla H_{\theta} = 0$ 对任何标量函数都成立，所以模型在连续时间下严格守恒能量。

### 5.2 架构与损失

普通 MLP：输入 $z \in \mathbb{R}^{2n}$，两到三个隐藏层（激活要 $C^1$，所以 tanh 或 softplus 比 ReLU 合适），输出标量。损失：

$$
\mathcal{L}(\theta) \;=\; \frac{1}{N} \sum_{i=1}^{N} \big\| J \,\nabla H_{\theta}(z_{i}) - \dot z_{i} \big\|^{2}.
$$

如果手头只有状态对 $(z_t, z_{t+1})$ 而拿不到导数，可以把 $\dot z_i$ 替换为有限差分，或者干脆用辛积分器把 $H_\theta$ 滚动几步、最小化多步预测误差。

### 5.3 它换来了什么

- 能量在连续时间下严格守恒（最终离散积分时取决于积分器）。
- 学到的 $H_\theta$ 是可解释的：可以画等高线、找对称性、对参数做敏感性分析。
- 控制方便：$H_\theta$ 是天然的 Lyapunov 候选函数。

### 5.4 它的边界

- 必须有显式的典范坐标 $(q, p)$。如果数据是别的参数化，得先变换。
- 纯 HNN 处理不了耗散。带阻尼或外驱动时用 **port-Hamiltonian** 扩展（Desai 等，2021）。

## 6. Lagrangian Neural Networks（LNN）

Cranmer 等（2020）做了对偶的事情：让网络代表 Lagrange 函数 $L_{\theta}(q, \dot q)$，加速度由欧拉-拉格朗日方程反解：

$$
\ddot q \;=\; \big(\nabla_{\dot q \dot q}^{2} L_{\theta}\big)^{-1} \Big[ \nabla_{q} L_{\theta} - \nabla_{q \dot q}^{2} L_{\theta}\, \dot q \Big].
$$

矩阵求逆在前向传播里完成（系统不大，通常 $n \le 20$）。损失对齐预测加速度与观测加速度：

$$
\mathcal{L}(\theta) \;=\; \frac{1}{N} \sum_{i=1}^{N} \big\| \ddot q_{i} - \ddot q_{\theta}(q_{i}, \dot q_{i}) \big\|^{2}.
$$

LNN 用更高的 autograd 代价（二阶导加上一个矩阵解）换来两个真正的优势：(1) 直接在位形空间工作，约束力学（比如沿轨道滑行的摆）天然方便；(2) 不用做 Legendre 变换——而 Legendre 变换在动力学退化时是会出问题的。

对无约束的保守系统，LNN 与 HNN 通过 Legendre 变换 $H = p^\top \dot q - L$、$p = \nabla_{\dot q} L$ 在形式上等价。

## 7. Symplectic Neural Networks（SympNet）

![SympNet 架构：剪切型辛模块的复合](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/fig3_sympnet_arch.png)

### 7.1 剪切技巧

Jin、Zhang、Zhu、Zhang 与 Karniadakis（2020）走的是另一条路：**不学 Hamilton 量，直接学流映射本身**，但把架构造得每一层都从结构上就是辛的。两种基本模块就够了：

- **G-module（梯度 / 动能剪切）**
$$
(q, p) \;\mapsto\; \big(q,\; p + \nabla V_{\theta}(q)\big),
$$
- **L-module（提升 / 势能剪切）**
$$
(q, p) \;\mapsto\; \big(q + \nabla K_{\phi}(p),\; p\big),
$$

其中 $V_\theta$、$K_\phi$ 是标量网络。每一块的 Jacobian 都是分块上三角或下三角，对角线上是单位矩阵，直接代入就能验证 $J^\top \Omega J = \Omega$。辛映射的复合仍是辛映射，所以任何 G-、L- 块的交替堆叠都是辛映射。

这是 normalizing flow 思路在物理上的对偶：RealNVP 用 coupling 层强制可逆，SympNet 用剪切层强制保辛。

### 7.2 训练

给定固定时间步 $\tau$ 的相邻样本对 $(z_i, z_{i+1})$，定义 $\Phi_{\theta} = \mathrm{block}_K \circ \cdots \circ \mathrm{block}_1$，最小化单步预测误差

$$
\mathcal{L}(\theta) \;=\; \sum_{i} \big\| \Phi_{\theta}(z_i) - z_{i+1} \big\|^{2}.
$$

要更长时段稳定，再加一项 $k$-步 rollout 损失 $\| \Phi_{\theta}^{k}(z_i) - z_{i+k} \|^{2}$。

### 7.3 取舍

- (+) 前向不需要 autograd——单次推断比 HNN/LNN 快很多。
- (+) 能表示**非 Hamilton 的**辛映射（比如未知系统的"学出来的辛积分器"）。
- ($-$) 没有显式 Hamilton 量，可解释性弱；只有辛 2-形式守恒，能量并不严格守恒。
- ($-$) 时间步 $\tau$ 烧死在网络里。变步长预测要重训或换架构。

## 8. 实验：四个经典系统

### 8.1 谐振子（健全性检查）

![单摆上的能量守恒：vanilla NN 漂移，SympNet 稳定振荡](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/fig4_energy_drift.png)

图 4 在略丰富一点的系统——单摆 $H = \tfrac{1}{2}p^{2} + (g/L)(1-\cos q)$——上展示了本文最核心的结论。"Vanilla NN"用显式 Euler 作为代理（它的失败模式与一个不带约束的 MLP 在质上一致）：能量沿一条几乎是直线的路线缓慢漂离 $H_0$。SympNet 这边用同步长的 Verlet 作为代理（受过良好训练的 SympNet **行为相同**）：能量在宽度为 $O(h^2)$ 的窄带里振荡。相图把这件事画得更直观——vanilla 轨道向外旋开，SympNet 轨道闭合。

真正训练出来的网络结果一样，只不过那条带子的宽度是由"训练误差"而非"离散误差"决定的。

### 8.2 双摆（混沌）

双摆是混沌系统：相距 $10^{-6}$ 的两条轨迹会指数发散。光靠能量守恒并不足以——单条轨道在长时间后本来就无法重建。但你**可以**重建的是**统计结构**：能壳上的不变测度、Lyapunov 谱、Poincaré 截面的拓扑。

HNN 与 SympNet 能保住这些统计；普通 MLP 因为引入了"假性耗散"，几个 Lyapunov 时间内就把不变测度坍缩到一个不动点附近——画出来就是"系统停了下来"。

### 8.3 Kepler 问题（多守恒量）

平面二体问题在极坐标下的 Hamilton 量是

$$
H \;=\; \tfrac{1}{2}\!\left( p_{r}^{2} + \frac{p_{\theta}^{2}}{r^{2}} \right) - \frac{k}{r},
$$

有两个守恒量：能量 $H$ 与角动量 $p_\theta$。HNN 自动保住第一个。第二个守恒**当且仅当**学到的 $H_\theta$ 不显式依赖 $\theta$——这个对称性既可以靠等变层强制注入，也可以期望从数据中浮现。Finzi、Wang 与 Wilson（2020）用 Lagrange 乘子的方式把守恒约束"硬"加进去。

值得画的诊断图：$10^4$ 圈轨道下 $H$ 与 $p_\theta$ 的相对漂移。Vanilla NN：两个都漂；HNN：$H$ 严格、$p_\theta$ 慢漂；HNN + 对称约束：两个都严格守恒。

### 8.4 分子动力学（真正的工业战场）

![结构保持学习在分子动力学中的应用](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/symplectic-geometry-and-structure-preserving-neural-networks/fig5_md_application.png)

这是结构保持学习正在改变实际科学的地方。256 粒子的 Lennard-Jones 流体 Hamilton 量为

$$
H \;=\; \sum_{i=1}^{N} \frac{p_{i}^{2}}{2 m} + \sum_{i < j} 4\varepsilon \!\left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^{6} \right].
$$

图 5 给出三件最重要的东西：

(a) LJ 对势曲线，最低点在 $r_{\min} = 2^{1/6}\sigma$ 处取值 $-\varepsilon$；

(b) **径向分布函数** $g(r)$——X 射线散射真正测的就是这个，所有 MD 力场最终都拿它去验证；

(c) 长时能量曲线。Vanilla NN 力场会漂——系统慢慢升温或降温，$g(r)$ 的峰被涂宽，跑 $10^5$ 步之后你模拟的已经是**别的物相**了。HNN/Verlet 在任意长时段里都只在 $E_0$ 附近做窄带振荡。

这正是为什么近年来的机器学习原子间势——ANI、NequIP、MACE——都开始构建在**等变架构 + 显式能量守恒**之上，而不是直接预测力。**几何的代价是值得付的。**

## 9. 工程上的几条经验

- **HNN 适合**：要可解释性、系统保守、有 $(q,p)$ 数据或可靠的有限差分导数。
- **LNN 适合**：动量定义不方便（约束、非笛卡尔坐标），或者想结合变分积分器。
- **SympNet 适合**：只有固定步长 $\tau$ 的 $(z_t, z_{t+1})$ 对，更在意单次推断成本而非可解释性。
- **耗散或受驱系统**：看 port-Hamiltonian NN（Desai 等）或带显式阻尼项的 Neural ODE（Chen 等，2018）。
- **推断时务必用辛积分器**积分训好的模型——否则离散误差会把架构辛苦消除的"漂移"重新带回来。

## 10. 总结

两句话讲完本文。**普通神经网络在长时物理预测里失败，是因为辛映射在所有光滑映射中是一个测度零的薄子流形，SGD 没有理由命中它。结构保持网络通过参数化 Hamilton 量（HNN）、Lagrange 量（LNN）或辛剪切的复合（SympNet）把假设空间限制在这个子流形里，无论 rollout 多长都不会漂。**

值得追的几条线：

- **随机与耗散扩展**——保持**扩展**相空间的辛结构（Langevin、port-Hamiltonian）。
- **高维辛学习**——多体系统里普通 MLP 已经吃不消，需要等变架构。
- **辛基础模型**——能不能在百万级 Hamilton 量上预训一个 SympNet，然后小样本迁移到新动力学？
- **可证泛化**——利用几何归纳偏置，而不是忽视它，给出更紧的泛化界。

更深的一课：当机器学习被告知"问题里哪些约束不能违反"时，它会变得**更便宜、更快、更可靠**。辛几何是最干净的一类此种约束，上面的工作就是它的原型。

## 参考文献

Greydanus, S., Dzamba, M. & Yosinski, J. (2019). **Hamiltonian Neural Networks.** *NeurIPS* 32. [arXiv:1906.01563](https://arxiv.org/abs/1906.01563)

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D. & Ho, S. (2020). **Lagrangian Neural Networks.** *ICLR Workshop on Deep Differential Equations.* [arXiv:2003.04630](https://arxiv.org/abs/2003.04630)

Jin, P., Zhang, Z., Zhu, A., Zhang, Y. & Karniadakis, G. E. (2020). **SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems.** *Neural Networks* 132, 166-179. [arXiv:2001.03750](https://arxiv.org/abs/2001.03750)

Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. K. (2018). **Neural Ordinary Differential Equations.** *NeurIPS* 31. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

Toth, P., Rezende, D. J., Jaegle, A., Racanière, S., Botev, A. & Higgins, I. (2020). **Hamiltonian Generative Networks.** *ICLR.* [arXiv:1909.13789](https://arxiv.org/abs/1909.13789)

Finzi, M., Wang, K. A. & Wilson, A. G. (2020). **Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints.** *NeurIPS* 33. [arXiv:2010.13581](https://arxiv.org/abs/2010.13581)

Zhong, Y. D., Dey, B. & Chakraborty, A. (2020). **Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control.** *ICLR.* [arXiv:1909.12077](https://arxiv.org/abs/1909.12077)

Desai, S., Mattheakis, M., Joy, H., Protopapas, P. & Roberts, S. (2021). **Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems.** *Physical Review E* 104(3), 034312. [arXiv:2107.08024](https://arxiv.org/abs/2107.08024)

Lutter, M., Ritter, C. & Peters, J. (2019). **Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning.** *ICLR.* [arXiv:1907.04490](https://arxiv.org/abs/1907.04490)

Hairer, E., Lubich, C. & Wanner, G. (2006). **Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations.** Springer Series in Computational Mathematics 31. [Springer link](https://link.springer.com/book/10.1007/3-540-30666-8)

Arnold, V. I. (1989). **Mathematical Methods of Classical Mechanics**（第 2 版）. Springer GTM 60. [Springer link](https://link.springer.com/book/10.1007/978-1-4757-2063-1)

Marsden, J. E. & Ratiu, T. S. (1999). **Introduction to Mechanics and Symmetry.** Springer TAM 17. [Springer link](https://link.springer.com/book/10.1007/978-0-387-21792-5)

Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J. & Battaglia, P. (2020). **Learning to Simulate Complex Physics with Graph Networks.** *ICML.* [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)

Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). **Physics-Informed Neural Networks.** *Journal of Computational Physics* 378, 686-707. [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C. & Csányi, G. (2022). **MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.** *NeurIPS* 35. [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)

Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E. & Kozinsky, B. (2022). **E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials.** *Nature Communications* 13, 2453. [arXiv:2101.03164](https://arxiv.org/abs/2101.03164)
