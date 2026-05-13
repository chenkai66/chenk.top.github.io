---
title: "偏微分方程与机器学习（八）：反应扩散系统与 GNN"
date: 2024-08-14 09:00:00
tags:
  - PDE
  - Machine Learning
  - Reaction-Diffusion
  - Graph Neural Networks
  - GNN
  - Turing Instability
  - Over-smoothing
categories: PDE与机器学习
series: pde-ml
lang: zh
mathjax: true
description: "深层 GNN 之所以崩溃，是因为它就是图上的扩散方程；图灵 1952 年的反应扩散理论告诉我们如何修好它——也为整个八章 PDE+ML 系列收尾。"
disableNunjucks: true
series_order: 8
translationKey: "pde-ml-8"
---
![偏微分方程与机器学习（八）：反应扩散系统与GNN — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/illustration_1.png)

## 本文你会学到

在引文图上堆叠 32 层 GCN，准确率会从 81% 骤降至 20%，所有节点特征最终坍缩为同一向量——这就是 GNN 中的“热寂”现象，即**过度平滑**（over-smoothing）。其根源可直接追溯至 PDE 理论：**单层 GCN 实质上是图上热方程的一个显式欧拉步**，而热方程只有一个不动点——常数函数。早在 1952 年，Alan Turing 就提出了解法：若在扩散方程中加入一个**反应项**，原本均匀的状态便能自发分裂出条纹、斑点或迷宫等复杂结构。同样的技巧——引入一个可学习的反应项——能让深层 GNN 免于坍缩，保持表达能力。

这是《PDE + 机器学习》系列的第八章，也是终章。前七章反复论证了一个核心观点：几乎所有现代神经网络架构本质上都是某类偏微分方程的离散化形式。而反应扩散系统与 GNN 的结合，恰恰构成了最显式、最贴近 PDE 原型的架构，也为回溯整个系列提供了一个清晰的终点视角。

**本文目录**

1. 连续空间上的反应扩散方程——Gray-Scott、FitzHugh-Nagumo 及其生成的形态；
2. 图灵不稳定性——线性稳定性分析如何解释扩散“创造”结构；
3. 图拉普拉斯算子——$abla^2$ 的离散对应物，其谱性质决定 GNN 行为；
4. GCN = 离散图扩散——过度平滑的谱证明；
5. 反应扩散 GNN（GRAND、GRAND++、RDGNN）——通过反应项维持节点差异；
6. 对整个 PDE + ML 系列的回顾。

**前置知识**：线性代数（特征分解）、基本 PDE 概念（扩散方程）、消息传递 GNN 的基本原理。

---

![Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 0 参数平面上的位置示意。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig1_turing_patterns.png)
*Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 $(F,k)$ 参数平面上的位置示意。*

## 1. 连续空间上的反应扩散

### 1.1 一般形式

反应扩散（Reaction-Diffusion, RD）方程将空间扩散与局部非线性反应耦合在一起：

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D}\,\nabla^2\mathbf{u} + \mathbf{R}(\mathbf{u}). \tag{1}
$$

- 扩散项 $\mathbf{D}\nabla^2\mathbf{u}$ 是**线性**且**平滑**的——它总是削弱梯度；
- 反应项 $\mathbf{R}(\mathbf{u})$ 是**局部**（不含空间导数）且**非线性**的——它可以强化或对抗平滑效应。

从物理角度看，$\mathbf{u}$ 表示一组化学物质的浓度，扩散遵循菲克定律，反应则由局部速率方程描述。从数学角度看，(1) 是一个半线性抛物型 PDE——即我们在第七章遇到的热方程，加上一个逐点非线性外力项。

真正令人惊叹的是 Turing 的洞见：这两项之间的**竞争**，竟能让一个均匀的初始状态自发演化出稳定且非平凡的空间模式。这被称为**扩散驱动不稳定性**（diffusion-driven instability）。

### 1.2 Gray-Scott 模型

Gray-Scott 是经典的双组分模型：

$$
\partial_t u = D_u \nabla^2 u - u v^2 + F(1-u),\qquad
\partial_t v = D_v \nabla^2 v + u v^2 - (F+k)\,v.
$$

- $u$ 是底物，以速率 $F$ 持续注入；$v$ 是自催化剂，通过反应 $u + 2v \to 3v$ 消耗 $u$，并以速率 $k$ 衰减；
- 当 $D_u > D_v$（底物扩散快于催化剂）时，$v$ 的微小扰动会稳定下来，形成图 1 所示的各种形态。

仅通过调整参数 $(F, k)$，同一方程就能产生**斑点**、**条纹**、**迷宫**、**孔洞**、**移动斑点**，甚至**自复制斑点**——Pearson (1993) 曾系统地划分出十余种不同的动力学区域。

### 1.3 FitzHugh-Nagumo 模型

该模型最初用于简化神经元动力学：

$$
\partial_t v = D \nabla^2 v + v - \tfrac{v^3}{3} - w + I,\qquad
\partial_t w = \varepsilon\,(v + \beta - \gamma w),\quad \varepsilon \ll 1.
$$

- $v$ 是快速变化的膜电位，$w$ 是缓慢的恢复变量；
- 三次非线性使 $v$ 具有“可激发性”：一旦扰动超过阈值，就会触发一个标准脉冲，随后被慢变量 $w$ 复位。

在二维空间中，该模型会产生螺旋波和靶心波——这些图案恰好出现在心律失常的心脏组织和发育中的鸡胚视网膜中（见 §6 图 6）。

## 2. 图灵不稳定性：从均匀中诞生结构

### 2.1 核心问题

假设存在一个均匀稳态 $\bar{\mathbf{u}}$，满足 $\mathbf{R}(\bar{\mathbf{u}}) = \mathbf{0}$，且在**无扩散**（well-mixed）系统中是稳定的。那么，加入扩散后，这个稳态是否可能变得不稳定？

直觉上答案是否定的——扩散只会抹平差异，理应增强稳定性。但 Turing (1952) 证明，这种直觉是错误的。

### 2.2 线性稳定性分析

将方程 (1) 在 $\bar{\mathbf{u}}$ 附近线性化，设扰动为 $\delta\mathbf{u}(\mathbf{x}, t) = \mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}\,e^{\sigma t}$，可得：

$$
\sigma\,\mathbf{q} \;=\; \underbrace{\bigl(\mathbf{J} - |\mathbf{k}|^2\,\mathbf{D}\bigr)}_{\mathbf{A}(|\mathbf{k}|^2)}\,\mathbf{q},\qquad
\mathbf{J} = \nabla_{\mathbf{u}}\mathbf{R}(\bar{\mathbf{u}}). \tag{2}
$$

当 $\mathbf{A}(|\mathbf{k}|^2)$ 存在正实部特征值时，对应模态 $\mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}$ 将增长。完整的**图灵条件**包含以下四条不等式（见图 2 右侧）：

1. $\mathrm{tr}\,\mathbf{J} < 0$ 且 $\det\,\mathbf{J} > 0$ —— **无扩散系统稳定**；
2. Jacobian 具有激活-抑制结构：$f_u > 0$，$g_v < 0$，且 $f_v\,g_u < 0$；
3. **扩散不对称**：$D_v \gg D_u$ —— 抑制剂扩散远快于激活剂；
4. 存在某个 $|\mathbf{k}|^2$ 使得 $\det\,\mathbf{A}(|\mathbf{k}|^2) < 0$，即存在不稳定的波数。

前三条是关于反应动力学的代数条件，第四条则是实际机制：一旦前三条满足，就会打开一个**不稳定波数带**，其中最不稳定的波数 $|\mathbf{k}_*|$ 决定了最终图案的**特征长度尺度** $\ell \sim 2\pi/|\mathbf{k}_*|$。

![左：激活-抑制系统色散关系 0。等扩散（蓝）下处处稳定；抑制剂扩散更快（红），在 1 附近打开不稳定带。右：四条图灵条件一目了然。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig2_turing_instability.png)
*左：激活-抑制系统的色散关系 $\sigma(|\mathbf{k}|^2)$。当扩散系数相等（蓝色）时，系统处处稳定；若抑制剂扩散更快（红色），则在 $|\mathbf{k}_*|^2 \approx 3.4$ 附近打开一个不稳定波数带。右：四条图灵条件一目了然。*

### 2.3 直观机制：短程激活，长程抑制

为何不对称扩散会破坏原本稳定的均匀态？想象局部出现一个微小的激活剂凸起：**在局部**，激活剂通过正反馈自我放大；同时它也产生抑制剂，但由于抑制剂扩散更快，其**局部浓度较低**，无法有效抑制该凸起；而**远处**的抑制剂浓度较高，反而压制了其他潜在凸起的形成。这种**短程激活、长程抑制**机制，正是动物皮毛斑纹、半干旱地区植被条带、沙丘波纹的通用成因，也是我们在 §5 中构建深层 GNN 架构的核心思想。

## 3. 从网格到图

### 3.1 为何使用图？

有限差分法（FDM）和有限元法（FEM）在规则网格或精心设计的网格上离散 PDE，在简单几何域中极为高效。但对于**分子结构、社交网络、引文图、道路网络、脑连接组**等场景，不存在自然的规则网格——**连接关系本身就是几何**。

图 $G = (V, E)$ 提供了一种统一框架：它仅由节点集合及其相互关系构成。GNN 正是在这种“几何”上求解某种“PDE”。本节的目标，就是写出这个 PDE 的具体形式。

![从规则网格到不规则图。两者都在离散 0，但格点 stencil 被节点的邻域代替。连续 PDE 1 同时容许两种离散；图版本的一步 Euler 就是一层 GCN。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig3_grid_to_graph.png)
*从规则网格到不规则图。两者都在离散 $\nabla^2$，但网格 stencil 被节点的邻域所替代。连续 PDE $\partial_t u = D\nabla^2 u + R(u)$ 同时容许这两种离散方式；图版本的一个欧拉步，恰好对应一层 GCN。*

### 3.2 图拉普拉斯算子

对于带权无向图，设邻接矩阵为 $\mathbf{A}$，度矩阵为 $\mathbf{D} = \mathrm{diag}(d_i)$，常见的图拉普拉斯变体如下：

| 变体 | 公式 | 谱范围 |
|------|------|--------|
| 组合型 | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | $[0, 2 d_{\max}]$ |
| 对称归一化 | $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ | $[0, 2]$ |
| 随机游走 | $\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$ | $[0, 2]$ |

三者共享一个关键性质：

$$
\mathbf{x}^{\!\top}\!\mathbf{L}\mathbf{x} \;=\; \tfrac{1}{2}\sum_{(i,j) \in E} w_{ij}\,(x_i - x_j)^2 \;\geq\; 0. \tag{3}
$$

图拉普拉斯是 $-\nabla^2$ 的离散类比——它对梯度的平方进行积分。它是对称半正定矩阵，具有谱分解 $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\!\top}$，特征值满足 $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$。

最小特征值恒为 0，对应的特征向量与常向量 $\mathbf{1}$ 成比例。第二小特征值 $\lambda_2$（即**代数连通度**）衡量了图的整体连通性。

### 3.3 图热方程

考虑如下连续时间动力学：

$$
\frac{d\mathbf{X}}{dt} = -\mathbf{L}\mathbf{X}. \tag{4}
$$

这就是图上的热方程。其解析解为 $\mathbf{X}(t) = e^{-\mathbf{L}t}\mathbf{X}(0)$，在谱坐标下完全解耦：

$$
\hat x_k(t) = e^{-\lambda_k t}\,\hat x_k(0),\qquad \hat x_k = \mathbf{u}_k^{\!\top}\mathbf{X}(0).
$$

每个模态都以自身速率 $\lambda_k$ 指数衰减，唯独 $\lambda_1 = 0$ 对应的常数模态被永久保留。当 $t \to \infty$ 时，所有节点值趋于一致。

![图热方程实战。50 节点小世界图上的随机初始信号被扩散抹平：0 时所有节点取同一值。右图解释原因——第 1 个模按 2 衰减，只有 3 存活。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig4_graph_laplacian.png)
*图热方程的实际效果。在一个 50 节点的小世界图上，随机初始信号被扩散迅速抹平：到 $t = 6$ 时，所有节点取值相同。右图揭示原因——第 $k$ 个模态按 $e^{-\lambda_k t}$ 衰减，只有 $\lambda_1 = 0$ 存活。*

这便是**最纯粹的过度平滑**，而我们甚至尚未引入神经网络。

## 4. GCN 就是热方程

![偏微分方程与机器学习（八）：反应扩散系统与GNN — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/08-Reaction-Diffusion-Systems/illustration_2.png)

### 4.1 等价性

标准 GCN 层（Kipf & Welling, 2017）定义为：

$$
\mathbf{H}^{(\ell+1)} = \sigma\bigl(\tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}\,\mathbf{W}^{(\ell)}\bigr),
\qquad \tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}(\mathbf{A} + \mathbf{I})\tilde{\mathbf{D}}^{-1/2}.
$$

若去掉非线性激活和线性投影（即令 $\sigma = \mathrm{id}$，$\mathbf{W} = \mathbf{I}$），剩余部分为：

$$
\mathbf{H}^{(\ell+1)} = \tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}
\;=\; \bigl(\mathbf{I} - \tilde{\mathbf{L}}_{\text{sym}}\bigr)\mathbf{H}^{(\ell)}.
\tag{5}
$$

这正是图热方程 $\dot{\mathbf{H}} = -\tilde{\mathbf{L}}_{\text{sym}}\mathbf{H}$ 的**显式欧拉步**，步长 $h = 1$。其中“加自环”技巧 $\mathbf{A} + \mathbf{I}$ 是标准的 FDM 稳定化方法，它将 $\tilde{\mathbf{L}}_{\text{sym}}$ 的谱压缩至 $[0, 2)$ 区间，确保显式格式稳定。

### 4.2 过度平滑的谱证明

忽略非线性和权重矩阵，经过 $L$ 层后：

$$
\mathbf{H}^{(L)} = \tilde{\mathbf{A}}^L\,\mathbf{H}^{(0)}.
$$

由于 $\tilde{\mathbf{A}}$ 的特征值位于 $(-1, 1]$ 区间，且特征值 1 对应常向量，取高次幂后所有非常数分量都会消失：

$$
\tilde{\mathbf{A}}^L \xrightarrow[L \to \infty]{} \pi_{\text{const}}.
$$

所有节点特征最终坍缩到同一向量。**这并非 GCN 特有的缺陷，而是任何低通滤波器迭代的必然结果**。加入 ReLU 和可学习权重矩阵只能延缓这一过程：Oono & Suzuki (2020) 证明，只要权重矩阵序列的奇异值有界，GCN 的特征仍会收敛到一个低维子空间。

### 4.3 连续深度 GNN

既然 GCN 对应一个欧拉步，为何不直接求解 ODE？**GRAND**（Chamberlain et al., 2021）提出了连续时间 GNN：

$$
\frac{d\mathbf{X}}{dt} = -\mathcal{L}_\theta(\mathbf{X})\,\mathbf{X},\qquad \mathbf{X}(T) = \text{输出}.
$$

其中 $\mathcal{L}_\theta$ 是一个带注意力机制的可学习拉普拉斯算子，积分通过现成的 ODE 求解器（如 Dormand-Prince）完成。但这**并未解决**过度平滑问题——更精确地求解热方程，终究还是在求解热方程。**GRAND++**（Thorpe et al., 2022）引入了源项，而 **RDGNN**（Eliasof et al., 2024 及前期工作）则加入了完整的**反应项**。我们将在下一节构建后者。

## 5. RDGNN：反应扩散图神经网络

### 5.1 架构设计

连续时间 RD-GNN 是方程 (1) 在图上的自然推广：

$$
\frac{d\mathbf{H}}{dt} = -\epsilon_d\,\mathbf{L}\,\mathbf{H} \;+\; \epsilon_r\,R_\theta(\mathbf{H}, \mathbf{H}^{(0)}).
\tag{6}
$$

采用 Lie-Trotter（算子分裂）方法进行一步离散，得到更新公式：

$$
\boxed{\;\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} \; - \; \epsilon_d\,\mathbf{L}\,\mathbf{H}^{(\ell)} \; + \; \epsilon_r\,R_\theta\bigl(\mathbf{H}^{(\ell)},\,\mathbf{H}^{(0)}\bigr).\;} \tag{7}
$$

该层包含三个模块（见图 5）：

- **扩散项** $-\epsilon_d \mathbf{L}\mathbf{H}^{(\ell)}$：标准的图平滑操作。为保证显式欧拉稳定性，需满足步长约束 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$；
- **反应项** $\epsilon_r R_\theta(\mathbf{H}^{(\ell)}, \mathbf{H}^{(0)})$：一个可学习的、**纯局部**变换——通常为逐节点应用的小型 MLP。以初始特征 $\mathbf{H}^{(0)}$ 为条件，类似于 ResNet 中的输入跳跃连接，可防止特征漂移；
- **跳跃项** $\mathbf{H}^{(\ell)}$：使整体动力学贴近恒等映射，这是实现深层网络数值稳定的关键。

一种常见的反应项设计借鉴了 FitzHugh-Nagumo 的激活-衰减结构：

$$
R_\theta(\mathbf{H}, \mathbf{H}^{(0)}) = \mathrm{MLP}_\theta\bigl([\mathbf{H} \,\Vert\, \mathbf{H}^{(0)}]\bigr) \; - \; \alpha\,\mathbf{H}.
$$

![RDGNN 单层：扩散分支做图拉普拉斯平滑，反应分支是学得的逐节点非线性，输入跳跃避免漂移。重复 0 次得到深层 GNN，不像 GCN 会塌陷。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig5_rdgnn_architecture.png)
*一个反应扩散 GNN 层。扩散分支执行标准的图拉普拉斯平滑；反应分支是一个可学习的逐节点非线性更新；来自 $\mathbf{H}^{(0)}$ 的输入跳跃提供了防止漂移的“锚点”。重复该模块 $L$ 次，即可构建深层 GNN，且不会像 GCN 那样发生坍缩。*

### 5.2 为何有效？

有两种视角可以解释反应项如何克服过度平滑。

**谱视角**：纯扩散以速率 $e^{-\lambda_k t}$ 衰减所有非常数模态；而反应项不依赖于 $\mathbf{L}$，因此可具有任意谱成分——特别是，它能将能量重新注入被扩散抑制的高频模态。最终结果是在不同频率上形成非平凡的能量分布。

**图灵视角**：若 $R_\theta$ 学习到了激活-抑制结构（足够表达力的 MLP 可以做到），且扩散强度 $\epsilon_d$ 使得 Jacobian $\mathbf{J} - \epsilon_d\,\lambda_k$ 在某些 $k$ 上不稳定，网络便会表现出**节点级的图灵图案**——不同节点收敛到不同的特征值，其空间组织由图谱决定。这正是 GNN 中的“鱼纹”现象。

### 5.3 PyTorch 实现

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class RDGNN(nn.Module):
    """反应扩散 GNN。

    H^(l+1) = H^(l) - eps_d * L H^(l) + eps_r * R(H^(l), H^(0))
    扩散步用 GCNConv（含归一化拉普拉斯）；
    反应步是逐节点 MLP，条件化在输入嵌入上。
    """

    def __init__(self, in_dim, hidden, out_dim, n_layers,
                 eps_d=0.1, eps_r=0.1, alpha=0.1):
        super().__init__()
        self.eps_d, self.eps_r, self.alpha = eps_d, eps_r, alpha
        self.encoder = nn.Linear(in_dim, hidden)
        self.diff = GCNConv(hidden, hidden, normalize=True, add_self_loops=True)
        self.react = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden),
                          nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        h0 = self.encoder(x)
        h = h0
        for r in self.react:
            # 扩散步: -eps_d * L h  ~~  eps_d * (GCN(h) - h)
            h_diff = self.diff(h, edge_index) - h
            h_react = r(torch.cat([h, h0], dim=-1)) - self.alpha * h
            h = h + self.eps_d * h_diff + self.eps_r * h_react
        return self.decoder(h)
```

该架构极为简洁：一个共享的 GCN 用于扩散，$L$ 个小型 MLP 用于反应，两端各有一个线性投影。图 6c 显示，仅凭如此简单的结构，就能在 Cora 数据集上将准确率维持至 64 层——深度达到 GCN 崩溃阈值的八倍。

## 6. 反应扩散已胜出的领域

同一个方程，三种应用场景。

![从生物到 GNN。(a) Gray-Scott 模型生成逼真的皮毛斑纹——正是 Turing 提出的生物形态发生机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可见于心室纤颤和早期视皮层发育。(c) 同样的 RD 思想用于图结构，得到不会退化的深层 GNN——RDGNN 在 64 层时仍保持 ~80% Cora 准确率，而 GCN、GAT 和纯扩散 GRAND 都跌破 25%。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig6_applications.png)
*从生物学到 GNN。(a) Gray-Scott 模型生成逼真的皮毛图案——这正是 Turing 为生物形态发生提出的机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可见于心律失常和早期视觉皮层发育。(c) 将相同的 RD 原理应用于图结构，可构建不会坍缩的深层 GNN——RDGNN 在 64 层时仍保持约 80% 的 Cora 准确率，而 GCN、GAT 乃至纯扩散的 GRAND 均跌至 25% 以下。*

**形态发生**：Murray 的《Mathematical Biology》利用图灵机制解释大型猫科动物的斑纹演化：体型较大的美洲虎呈现斑点，中等体型的豹子呈现玫瑰花斑，而尾巴（局部几何约束了 $|\mathbf{k}|$）则呈现条纹——所有这些均可由一套反应参数和胚胎几何共同解释。同样的数学还能预测半干旱地区的植被条带（水为抑制剂，生物量为激活剂）以及 Belousov-Zhabotinsky 化学实验中的螺旋臂。

**神经发育**：在视网膜镶嵌形成过程中，相邻光感受器会**抑制**彼此分化为相同亚型，但通过扩散性形态发生素在**更远距离**上**激活**同类型分化。这在数学上构成一个图灵系统，所得视锥细胞排列具有可测量的波长 $\ell \sim 2\pi/|\mathbf{k}_*|$。皮层电活动中的螺旋波——在癫痫中属病理现象，在发育中则参与神经连接——正是二维可激发介质上的 FitzHugh-Nagumo 解。

**深层 GNN**：在标准基准测试中，深度与准确率的关系极为显著（图 6c，复现自 Eliasof et al. (2024) 的 Cora 实验）。GCN 和 GAT 在超过 8 层后迅速崩溃，与谱理论预测一致；纯扩散的 GRAND 仅能**推迟**崩溃，因其更精确地求解热方程；唯有 RDGNN——显式引入反应项——能在 $L = 64$ 时维持高准确率。这一优势在**异配图**（heterophilic graphs，即邻居倾向于拥有不同标签）上尤为突出，因为纯平滑在此类图上会主动破坏判别性信号：反应项可学习**放大**相连节点间的差异。

## 7. 系列收官：八章一念

至此，《PDE + 机器学习》系列迎来终章。让我们退后一步，纵观全局。

![八章之旅。第 1-2 章用神经网络解 PDE；第 3-4 章把训练重写成变分原理；第 5-6 章构造保结构的网络；第 7-8 章把同一套方法搬到生成模型和图学习上。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig7_series_journey.png)
*八章之旅。第 1–2 章用神经网络求解 PDE；第 3–4 章将训练过程重释为变分原理；第 5–6 章构建保结构的网络；第 7–8 章将同一套方法应用于生成建模与图学习。*

这八章可分为四对：

| 组别 | 章节 | 背后的 PDE 思想 |
|------|------|------------------|
| 用 NN 解 PDE | [1](/zh/pde-ml/01-物理信息神经网络/) PINN，[2](/zh/pde-ml/02-神经算子理论/) 神经算子 | 目标 PDE 本身成为损失函数 |
| 变分视角 | [3](/zh/pde-ml/03-变分原理与优化/) Deep Ritz，[4](/zh/pde-ml/04-变分推断与fokker-planck方程) VI / Fokker-Planck | 损失 = 自由能；梯度流 = 连续性方程 |
| 保结构流 | [5](/zh/pde-ml/05-辛几何与保结构网络/) 辛网络，[6](/zh/pde-ml/06-连续归一化流与neural-ode) Neural ODE / CNF | 网络尊重流的辛/体积/散度结构 |
| 生成 + 图 PDE | [7](/zh/pde-ml/07-扩散模型与score-matching) 扩散模型，**8** RD + GNN | 正反向 Fokker-Planck；图上的反应扩散 |

贯穿始终的核心思想只有一句：

> **一种神经网络架构，就是某个 PDE 的离散化形式。选择架构，即是选择 PDE。**

具体而言：

- **希望在训练分布之外外推？** 选择算子学习的 PDE（第 2 章）；
- **希望网络保留守恒量？** 选择辛积分器（第 5 章）；
- **希望似然可计算？** 选择连续性方程并学习其漂移项（第 6 章）；
- **希望从复杂分布采样？** 选择 Fokker-Planck 方程并学习其 score（第 7 章）；
- **希望深层 GNN 不坍缩？** 选择反应扩散方程，而非仅有扩散（本章）。

PDE 视角并非理解深度学习的唯一途径，但它出奇地具有**生成性**：每当我们追问“对应的数值分析会怎么说？”，总能得到具体回报——一个稳定性边界、一个步长约束、一个结构性修复方案。这正是物理学思维方式为机器学习带来的红利，也是两个领域对话远未终结的原因。

## 8. 习题

**练习 1.** 证明对连通图，$\mathbf{L}\mathbf{x} = \mathbf{0}$ 的唯一解是常向量。由此说明图热方程将任意初值驱动至其均值。

> *解。* 由 (3) 式，$\mathbf{x}^\top\!\mathbf{L}\mathbf{x} = \tfrac{1}{2}\sum w_{ij}(x_i - x_j)^2 = 0$ 意味着每条边两端的值相等；在连通图上，这迫使 $\mathbf{x}$ 为常向量。因此 $\mathbf{L}$ 的核空间为一维。其余特征值严格为正，故 $e^{-\mathbf{L}t}$ 会消除所有非常数分量，仅保留均值。$\blacksquare$

**练习 2.** 推导扩散步显式欧拉格式的稳定性条件 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$。

> *解。* 在谱坐标下，欧拉更新为 $\hat{x}_k^{(\ell+1)} = (1 - \epsilon_d \lambda_k)\,\hat{x}_k^{(\ell)}$。为避免增长，需对所有 $k$ 满足 $|1 - \epsilon_d \lambda_k| \leq 1$，即 $0 \leq \epsilon_d \lambda_k \leq 2$，故 $\epsilon_d \leq 2/\lambda_{\max}$。若要求单调衰减（无振荡），则需更严格的条件 $\epsilon_d < 1/\lambda_{\max}$。$\blacksquare$

**练习 3.** 为何 RDGNN 在异配图上特别有效？

> *解。* 在异配图中，相邻节点往往具有**相反**的标签。纯扩散步骤会对邻居特征取平均，从而主动破坏判别性信号——层数越多，损害越严重。而反应项是**逐节点**操作，且以 $\mathbf{H}^{(0)}$ 为条件，因此可生成节点专属的更新，**放大**邻居间的差异，恢复类别可分性。$\blacksquare$

**练习 4.** 证明离散 RDGNN 更新式 (7) 是连续 RD-GNN (6) 的一阶 Lie-Trotter 算子分裂离散。

> *解。* 算子分裂将 $\dot{\mathbf{H}} = (\mathcal{D} + \mathcal{R})\,\mathbf{H}$ 分解为交替的欧拉步：先 $\mathbf{H}^{1/2} = \mathbf{H} + h\mathcal{D}\mathbf{H}$，再 $\mathbf{H}^{(\ell+1)} = \mathbf{H}^{1/2} + h\mathcal{R}\mathbf{H}^{1/2}$。在标准实现中，两个算子均在 $\mathbf{H}^{(\ell)}$ 处求值，结果恰好为 (7)。其局部截断误差为 $\mathcal{O}(h^2[\mathcal{D}, \mathcal{R}])$，即一阶精度。$\blacksquare$

**练习 5.** 单条图灵不稳定性条件可通过数值验证：选取 Gray-Scott 参数 $(D_u, D_v, F, k)$，在均匀稳态附近线性化，扫描 $|\mathbf{k}|^2$ 并观察 $\det\,\mathbf{A}(|\mathbf{k}|^2)$ 的符号变化。请实现该过程并复现图 1 中的某一形态。

> *解概要。* Gray-Scott 的均匀稳态满足 $u v^2 = F(1 - u)$ 且 $u v^2 = (F + k) v$。计算该点处的 $2\times2$ Jacobian，构造 $\mathbf{A}(|\mathbf{k}|^2) = \mathbf{J} - |\mathbf{k}|^2 \mathrm{diag}(D_u, D_v)$，并绘制 $\det\,\mathbf{A}$ 关于 $|\mathbf{k}|^2$ 的曲线。若出现负值区间，则表明存在不稳定波数带；对应的波长 $2\pi/|\mathbf{k}_*|$ 应与模拟图案的视觉尺度一致。$\blacksquare$

## 参考文献

[1] Turing, A. M. (1952). *The chemical basis of morphogenesis.* Phil. Trans. R. Soc. B, 237(641), 37-72.

[2] Pearson, J. E. (1993). *Complex patterns in a simple system.* Science, 261(5118), 189-192.

[3] Murray, J. D. (2003). *Mathematical biology II: Spatial models and biomedical applications* (3rd ed.). Springer.

[4] Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks.* ICLR. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).

[5] Oono, K., & Suzuki, T. (2020). *Graph neural networks exponentially lose expressive power for node classification.* ICLR. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947).

[6] Chamberlain, B., Rowbottom, J., Gorinova, M., Webb, S., Rossi, E., & Bronstein, M. (2021). *GRAND: Graph neural diffusion.* ICML. [arXiv:2106.10934](https://arxiv.org/abs/2106.10934).

[7] Thorpe, M., Nguyen, T. M., Xia, H., Strohmer, T., Bertozzi, A., Osher, S., & Wang, B. (2022). *GRAND++: Graph neural diffusion with a source term.* ICLR.

[8] Eliasof, M., Haber, E., & Treister, E. (2021). *PDE-GCN: Novel architectures for graph neural networks motivated by partial differential equations.* NeurIPS. [arXiv:2108.01938](https://arxiv.org/abs/2108.01938).

[9] Di Giovanni, F., Rowbottom, J., Chamberlain, B., Markovich, T., & Bronstein, M. (2022). *Graph neural networks as gradient flows.* [arXiv:2206.10991](https://arxiv.org/abs/2206.10991).

[10] Choi, J., Hong, S., Park, N., & Cho, S.-B. (2023). *GREAD: Graph neural reaction-diffusion networks.* ICML. [arXiv:2211.14208](https://arxiv.org/abs/2211.14208).

[11] Eliasof, M., Haber, E., & Treister, E. (2024). *Graph neural reaction-diffusion models.* SIAM J. Sci. Comput. [arXiv:2406.10871](https://arxiv.org/abs/2406.10871).

---

*本文是 [PDE 与机器学习](/zh/pde-ml/) 系列的第 8 篇——也是最后一篇。上一篇：[第 7 篇 —— 扩散模型与 Score Matching](/zh/pde-ml/07-扩散模型与score-matching)。回到开篇：[第 1 篇 —— 物理信息神经网络](/zh/pde-ml/01-物理信息神经网络)。感谢阅读。*
