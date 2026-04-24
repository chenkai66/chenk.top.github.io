---
title: "PDE与机器学习（八）：反应扩散系统与GNN"
date: 2024-09-08 09:00:00
tags:
  - 反应扩散
  - 图神经网络
  - PDE
  - 图上PDE
  - 模式形成
categories:
  - PDE与机器学习
series:
  name: "PDE与机器学习"
  part: 8
  total: 8
lang: zh-CN
mathjax: true
description: "深层 GNN 之所以崩溃，是因为它就是图上的扩散方程；图灵 1952 年的反应扩散理论告诉我们如何修好它——也为整个八章 PDE+ML 系列收尾。"
disableNunjucks: true
---

## 本文你会学到

把 32 层 GCN 堆在一张引文网络上，准确率从 81% 跌到 20%，每个节点的特征向量都收敛到同一个点。这就是**过度平滑**——GNN 版本的"热寂"，而病因来自 PDE 教科书的第一章：**一层 GCN 就是图上热方程的一步显式 Euler**，热方程只有一个不动点：常数。解药 1952 年就有了。Alan Turing 证明，给一个扩散方程加上一个**反应项**，原本均匀的稳态可以自发地长出条纹、斑点、迷宫——同样的把戏（一个**学得到**的反应项）也能让深层 GNN 活下来。

这同时是 *PDE+机器学习* 系列的第 8 章，也是终章。前 7 章我们一直在论证"几乎每一种现代神经网络架构都是某个 PDE 的离散化"。反应扩散 + GNN 把这条线收尾：它是其中**最显式的 PDE 形态**，也让我们可以站在终点回望整套系列。

**本文目录**

1. 连续空间上的反应扩散方程——Gray-Scott、FitzHugh-Nagumo、它们能造出哪些模式；
2. 图灵不稳定性——线性稳定性分析告诉我们扩散为什么能"创造"结构；
3. 图拉普拉斯——$\nabla^2$ 的离散版本，以及它的谱为何决定 GNN 的命运；
4. GCN $=$ 离散图扩散——过度平滑的谱证明；
5. 反应扩散 GNN（GRAND、GRAND++、RDGNN）——加上一个反应项；
6. 整个 PDE+ML 系列的回顾。

**前置知识**：线性代数（特征分解）、基本 PDE 概念（扩散方程）、消息传递 GNN。

---

![Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 $(F,k)$ 参数平面上的位置示意。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig1_turing_patterns.png)
*Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 $(F,k)$ 参数平面上的位置示意。*

## 1. 连续空间上的反应扩散

### 1.1 一般形式

反应扩散方程把空间扩散和局部非线性反应耦合在一起：
$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D}\,\nabla^2\mathbf{u} + \mathbf{R}(\mathbf{u}). \tag{1}
$$
- 扩散项 $\mathbf{D}\nabla^2\mathbf{u}$ 是**线性**且**抹平**的——它永远在缩小梯度。
- 反应项 $\mathbf{R}(\mathbf{u})$ 是**局部**（不带空间导数）且**非线性**的——它可以加强也可以对抗扩散。

两种视角都有用。物理上，$\mathbf{u}$ 是一组浓度，扩散是 Fick 定律，反应是局部速率方程；数学上，(1) 是一个半线性抛物 PDE——就是我们在第 7 章见过的热方程加上一个逐点非线性源项。

Turing 的洞察在于：这两项的**竞争**可以让一个均匀初值自发演化成稳定的、非平凡的空间模式。这种现象被称为**扩散驱动的不稳定性**（diffusion-driven instability）。

### 1.2 Gray-Scott 模型

Gray-Scott 是经典的双组分模型：
$$
\partial_t u = D_u \nabla^2 u - u v^2 + F(1-u),\qquad
\partial_t v = D_v \nabla^2 v + u v^2 - (F+k)\,v.
$$
- $u$ 是按速率 $F$ 注入的底物，$v$ 是消耗 $u$ 的自催化剂（反应 $u + 2v \to 3v$），并以速率 $k$ 衰减。
- 当 $D_u > D_v$（底物比催化剂扩散得快），$v$ 的小斑块就能稳定下来，演化出图 1 中的各种形态。

同一个方程，换不同的 $(F, k)$，可以得到**斑点**、**条纹**、**迷宫**、**孔洞**、**移动斑点**，甚至**自复制斑点**——Pearson (1993) 一口气分类出十几种相区。

### 1.3 FitzHugh-Nagumo 模型

最早是简化的神经元模型：
$$
\partial_t v = D \nabla^2 v + v - \tfrac{v^3}{3} - w + I,\qquad
\partial_t w = \varepsilon\,(v + \beta - \gamma w),\quad \varepsilon \ll 1.
$$
- $v$ 是快变量（膜电位），$w$ 是慢恢复变量。
- 三次非线性让 $v$ 具有**激发性**：超过阈值的扰动会触发一个标志化脉冲，慢变量 $w$ 随后把它复位。

二维下你会看到螺旋波和靶心波——这正是心脏纤颤时的电波形态、也是发育中视网膜上看到的图样（见 §6 图 6）。

---

## 2. 图灵不稳定性：从均匀生出模式

### 2.1 问题

挑一个均匀稳态 $\bar{\mathbf{u}}$，满足 $\mathbf{R}(\bar{\mathbf{u}}) = \mathbf{0}$，并且**在去掉扩散的纯局部系统中是稳定的**。把扩散加回来，能否让这个稳态**失稳**？

直觉上不能——扩散只会"抹平"，怎么会让东西更"乱"？Turing (1952) 证明了直觉是错的。

### 2.2 推导

把 (1) 在 $\bar{\mathbf{u}}$ 附近线性化，假设扰动 $\delta\mathbf{u}(\mathbf{x}, t) = \mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}\,e^{\sigma t}$：
$$
\sigma\,\mathbf{q} \;=\; \underbrace{\bigl(\mathbf{J} - |\mathbf{k}|^2\,\mathbf{D}\bigr)}_{\mathbf{A}(|\mathbf{k}|^2)}\,\mathbf{q},\qquad
\mathbf{J} = \nabla_{\mathbf{u}}\mathbf{R}(\bar{\mathbf{u}}). \tag{2}
$$
当 $\mathbf{A}(|\mathbf{k}|^2)$ 有正实部特征值时，模式 $\mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}$ 增长。完整的**图灵条件**是四条不等式（图 2 右）：

1. $\mathrm{tr}\,\mathbf{J} < 0$ 且 $\det\,\mathbf{J} > 0$——**纯局部系统稳定**；
2. Jacobian 具有激活-抑制结构：$f_u > 0,\;g_v < 0,\;f_v g_u < 0$；
3. **扩散不对称性**：$D_v \gg D_u$——抑制剂扩散得远比激活剂快；
4. 存在某个 $|\mathbf{k}|^2$ 使 $\det\,\mathbf{A}(|\mathbf{k}|^2) < 0$，即一个不稳定波数。

前三条是关于动力学的代数事实，第四条是真正的机制：**一段波数变得不稳定**，其中最不稳定的模 $|\mathbf{k}_*|$ 决定了模式的特征长度 $\ell \sim 2\pi/|\mathbf{k}_*|$。

![左：激活-抑制系统的色散关系 $\sigma(|\mathbf{k}|^2)$。等扩散（蓝）下处处稳定；让抑制剂跑得更快（红），就在 $|\mathbf{k}_*|^2 \approx 3.4$ 附近打开一段不稳定带。右：四条图灵条件一目了然。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig2_turing_instability.png)
*左：激活-抑制系统的色散关系 $\sigma(|\mathbf{k}|^2)$。等扩散（蓝）下处处稳定；让抑制剂跑得更快（红），就在 $|\mathbf{k}_*|^2 \approx 3.4$ 附近打开一段不稳定带。右：四条图灵条件一目了然。*

### 2.3 直觉：短程激活、长程抑制

为什么不对称扩散能"破坏"一个原本稳定的稳态？想象一个微小的局部凸起（多了点激活剂）。**局部上**，激活剂正反馈、自我放大；同时它会生成抑制剂，但抑制剂跑得快，所以局部抑制剂浓度仍然偏低，**远处**抑制剂浓度则迅速堆高，把别处可能形成的凸起摁住。这就是**短程激活、长程抑制**——动物斑纹、植被条带、沙波纹背后的通用配方，也是后面 §5 深层 GNN 架构背后的同一个东西。

---

## 3. 从网格到图

### 3.1 为什么要图

有限差分（FDM）和有限元（FEM）在规则网格或精心剖分的网格上离散 PDE。对简单几何很好用。但**分子结构、社交网络、引文图、路网、脑连接组**没有自然的"网格"——连接关系本身就是几何。

图 $G = (V, E)$ 是统一的：一组节点加上节点间的关系。GNN 在解一个图上的"PDE"。本节就是把这个 PDE 写出来。

![从规则网格到不规则图。两者都在离散 $\nabla^2$，但格点 stencil 被节点的邻域代替。连续 PDE $\partial_t u = D\nabla^2 u + R(u)$ 同时容许两种离散；图版本的一步 Euler 就是一层 GCN。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig3_grid_to_graph.png)
*从规则网格到不规则图。两者都在离散 $\nabla^2$，但格点 stencil 被节点的邻域代替。连续 PDE $\partial_t u = D\nabla^2 u + R(u)$ 同时容许两种离散；图版本的一步 Euler 就是一层 GCN。*

### 3.2 图拉普拉斯

带权无向图，邻接矩阵 $\mathbf{A}$，度矩阵 $\mathbf{D} = \mathrm{diag}(d_i)$：

| 变体 | 公式 | 谱所在 |
|------|------|--------|
| 组合型 | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | $[0, 2 d_{\max}]$ |
| 对称归一 | $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ | $[0, 2]$ |
| 随机游走 | $\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$ | $[0, 2]$ |

三者都满足同一个核心性质：
$$
\mathbf{x}^{\!\top}\!\mathbf{L}\mathbf{x} \;=\; \tfrac{1}{2}\sum_{(i,j) \in E} w_{ij}\,(x_i - x_j)^2 \;\geq\; 0. \tag{3}
$$
图拉普拉斯就是 $-\nabla^2$ 的离散版：它对**梯度的平方做积分**。它对称半正定，谱分解 $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\!\top}$，特征值 $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$。

最小特征值恒为 0，对应常向量 $\mathbf{1}$；第二小的 $\lambda_2$（**代数连通度**）刻画图的连通强度。

### 3.3 图上的热方程

直接写下显然的连续时间动力学：
$$
\frac{d\mathbf{X}}{dt} = -\mathbf{L}\mathbf{X}. \tag{4}
$$
解是闭式的：$\mathbf{X}(t) = e^{-\mathbf{L}t}\mathbf{X}(0)$。在谱坐标下完全解耦：
$$
\hat x_k(t) = e^{-\lambda_k t}\,\hat x_k(0),\qquad \hat x_k = \mathbf{u}_k^{\!\top}\mathbf{X}(0).
$$
每个模都按各自速率 $\lambda_k$ 指数衰减——除了 $\lambda_1 = 0$，那个常数模永远活着。$t \to \infty$ 时只剩常数。

![图热方程的实战。一个 50 节点小世界图上的随机初始信号被扩散摧毁：到 $t = 6$ 时所有节点取同一个值。右图给出原因——第 $k$ 个模按 $e^{-\lambda_k t}$ 衰减，只有 $\lambda_1 = 0$ 不死。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig4_graph_laplacian.png)
*图热方程的实战。一个 50 节点小世界图上的随机初始信号被扩散摧毁：到 $t = 6$ 时所有节点取同一个值。右图给出原因——第 $k$ 个模按 $e^{-\lambda_k t}$ 衰减，只有 $\lambda_1 = 0$ 不死。*

这就是**最纯粹的过度平滑**，我们甚至还没提神经网络。

---

## 4. GCN 就是热方程

### 4.1 等价关系

标准 GCN 层（Kipf & Welling, 2017）：
$$
\mathbf{H}^{(\ell+1)} = \sigma\bigl(\tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}\,\mathbf{W}^{(\ell)}\bigr),
\qquad \tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}(\mathbf{A} + \mathbf{I})\tilde{\mathbf{D}}^{-1/2}.
$$
去掉非线性、把权重换成单位阵（$\sigma = \mathrm{id}$，$\mathbf{W} = \mathbf{I}$），剩下：
$$
\mathbf{H}^{(\ell+1)} = \tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}
\;=\; \bigl(\mathbf{I} - \tilde{\mathbf{L}}_{\text{sym}}\bigr)\mathbf{H}^{(\ell)}.
\tag{5}
$$
这正是图热方程 $\dot{\mathbf{H}} = -\tilde{\mathbf{L}}_{\text{sym}}\mathbf{H}$ 的**显式 Euler**，步长 $h = 1$。"加自环" $\mathbf{A} + \mathbf{I}$ 是标准的 FDM 稳定化技巧——把 $\tilde{\mathbf{L}}_{\text{sym}}$ 的谱压进 $[0, 2)$，让显式格式仍然稳定。

### 4.2 过度平滑的谱证明

叠 $L$ 层（继续忽略非线性和权重）：
$$
\mathbf{H}^{(L)} = \tilde{\mathbf{A}}^L\,\mathbf{H}^{(0)}.
$$
$\tilde{\mathbf{A}}$ 的特征值在 $(-1, 1]$ 内，特征值 $1$ 对应常向量。取幂之后除了首特征空间什么都活不下来：
$$
\tilde{\mathbf{A}}^L \xrightarrow[L \to \infty]{} \pi_{\text{const}}.
$$
所有节点特征坍缩到同一个向量。**这不是 GCN 的怪癖，是任何低通滤波器迭代后的定理**。加上 ReLU 和可学习权重只能延缓这个过程：Oono & Suzuki (2020) 证明，对任意奇异值有界的权重序列，GCN 特征仍然收敛到一个低维子空间。

### 4.3 连续深度 GNN

既然一层 GCN 就是一步 Euler，干嘛不把 ODE 老老实实地解了？**GRAND**（Chamberlain et al., 2021）就是连续时间 GNN：
$$
\frac{d\mathbf{X}}{dt} = -\mathcal{L}_\theta(\mathbf{X})\,\mathbf{X},\qquad \mathbf{X}(T) = \text{输出}.
$$
$\mathcal{L}_\theta$ 是带学习注意力的拉普拉斯，积分用现成的 ODE solver（Dormand-Prince 等）。这并**不能**根治过度平滑——更精确地解一个热方程，依然在解一个热方程。**GRAND++**（Thorpe et al., 2022）加了一个**源**项；**RDGNN**（Eliasof et al., 2024 等）则加上了完整的**反应**项。下一节我们造出后者。

---

## 5. RDGNN：反应扩散图神经网络

### 5.1 架构

连续时间 RD-GNN 就是 (1) 在图上的自然版本：
$$
\frac{d\mathbf{H}}{dt} = -\epsilon_d\,\mathbf{L}\,\mathbf{H} \;+\; \epsilon_r\,R_\theta(\mathbf{H}, \mathbf{H}^{(0)}).
\tag{6}
$$
做一步 Lie-Trotter 算子分裂，得到离散更新：
$$
\boxed{\;\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} \;-\; \epsilon_d\,\mathbf{L}\,\mathbf{H}^{(\ell)} \;+\; \epsilon_r\,R_\theta\bigl(\mathbf{H}^{(\ell)},\,\mathbf{H}^{(0)}\bigr).\;} \tag{7}
$$
三个分支（图 5）：

- **扩散** $-\epsilon_d \mathbf{L}\mathbf{H}^{(\ell)}$：标准的图平滑。步长约束 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$，保证显式 Euler 稳定。
- **反应** $\epsilon_r R_\theta(\mathbf{H}^{(\ell)}, \mathbf{H}^{(0)})$：可学习的、**严格逐节点**的变换——通常是一个小 MLP 在节点上单独跑。条件化在 $\mathbf{H}^{(0)}$ 上相当于 ResNet 的 input-skip，防止漂移。
- **跳过项** $\mathbf{H}^{(\ell)}$：让动力学贴近恒等，正是大 $L$ 时数值稳定的关键。

一种常见反应项设计（FitzHugh 风格的"激活+衰减"）：
$$
R_\theta(\mathbf{H}, \mathbf{H}^{(0)}) = \mathrm{MLP}_\theta\bigl([\mathbf{H} \,\Vert\, \mathbf{H}^{(0)}]\bigr) \;-\; \alpha\,\mathbf{H}.
$$

![RDGNN 一层：扩散分支做图拉普拉斯平滑，反应分支做学得到的逐节点非线性，从 $\mathbf{H}^{(0)}$ 出来的输入跳跃避免漂移。把这块重复 $L$ 次得到的深层 GNN，不像 GCN 一样会塌掉。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig5_rdgnn_architecture.png)
*RDGNN 一层：扩散分支做图拉普拉斯平滑，反应分支做学得到的逐节点非线性，从 $\mathbf{H}^{(0)}$ 出来的输入跳跃避免漂移。把这块重复 $L$ 次得到的深层 GNN，不像 GCN 一样会塌掉。*

### 5.2 为什么有效

两种角度。

**谱视角**。纯扩散把每个非常数模按 $e^{-\lambda_k t}$ 衰减；反应项**不是** $\mathbf{L}$ 的函数，谱内容可以任意，特别可以把扩散正在杀掉的高频能量再灌回去。结果是能量在频率上有个非平凡的稳定分布。

**图灵视角**。如果 $R_\theta$ 学到了激活-抑制结构（一个表达力够的 MLP 完全可以），并且扩散强度 $\epsilon_d$ 选得让 $\mathbf{J} - \epsilon_d \lambda_k$ 在某些 $k$ 上不稳定，整个网络就会出现**节点级别的图灵模式**——不同节点收敛到不同的特征值，按图谱组织起来。这就是 GNN 版本的"鱼身上的条纹"。

### 5.3 PyTorch 实现

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class RDGNN(nn.Module):
    """反应扩散 GNN。

    H^(l+1) = H^(l) - eps_d * L H^(l) + eps_r * R(H^(l), H^(0))
    扩散步借助 GCNConv（内部含归一化拉普拉斯）；
    反应步是逐节点 MLP，并以输入嵌入为条件。
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

整个架构非常精瘦：扩散用一个共享的 GCN，反应用 $L$ 个小 MLP，两端各一个线性投影。图 6c 的精度曲线显示，仅靠这点料就能把 Cora 准确率维持到 64 层——比 GCN 崩溃的那个深度还多了八倍。

---

## 6. 反应扩散已经胜出的地方

同一个方程，三个应用故事。

![从生物到 GNN。(a) Gray-Scott 模型造出逼真的皮毛斑纹——正是 Turing 假设的生物形态发生机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可在心室纤颤和早期视皮层发育中观测到。(c) 同样的 RD 思想用到图上，得到不会塌的深层 GNN——RDGNN 在 64 层时仍维持 ~80% Cora 准确率，而 GCN、GAT、纯扩散的 GRAND 都跌破 25%。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig6_applications.png)
*从生物到 GNN。(a) Gray-Scott 模型造出逼真的皮毛斑纹——正是 Turing 假设的生物形态发生机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可在心室纤颤和早期视皮层发育中观测到。(c) 同样的 RD 思想用到图上，得到不会塌的深层 GNN——RDGNN 在 64 层时仍维持 ~80% Cora 准确率，而 GCN、GAT、纯扩散的 GRAND 都跌破 25%。*

**形态发生**。Murray 的《Mathematical Biology》里用图灵机制拟合了大型猫科动物的斑纹相变：体型大的（美洲虎）出斑点、中等的（豹）出玫瑰花斑、尾巴（局部几何把 $|\mathbf{k}|$ 限住）出条纹——一套反应参数加上胚胎几何就能解释。同样的数学也预测了半干旱地带的植被条带（水当抑制剂、生物量当激活剂）和 Belousov-Zhabotinsky 化学振荡的螺旋臂。

**神经发育**。视网膜镶嵌形成时，相邻光感受器**互斥**分化为同一亚型，但远程通过扩散性形态发生素**协助**同型分化——这从数学上就是图灵系统，得到的视锥排列具有可测量的波长 $\ell \sim 2\pi/|\mathbf{k}_*|$。皮层电活动里的螺旋波（癫痫时是病理，发育时则参与连接组成型）就是 2D 可激发介质上的 FitzHugh-Nagumo 解。

**深层 GNN**。在标准基准上深度-精度的故事戏剧化（图 6c，复刻自 Eliasof et al. (2024) 的 Cora 实验）：GCN/GAT 在 8 层后崩溃，跟谱证明的预测一致；纯扩散 GRAND 只是**推迟**了崩溃，因为它只是更精确地解同一个热方程；只有 RDGNN——显式反应项——能在 $L = 64$ 时维持精度。这种效应在**异配图**（邻居标签更倾向不同）上更明显，因为对异配图来说"平滑"本身就是有害的：反应项可以学着**放大**邻居之间的差异。

---

## 7. 系列收官：八章一念

我们到了 *PDE+机器学习* 系列的终点。退一步看。

![八章之旅。第 1-2 章用神经网络**解** PDE；第 3-4 章把训练重写成变分原理；第 5-6 章构造**保结构**的网络；第 7-8 章把同一套机器搬到生成模型和图学习上。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-%E5%8F%8D%E5%BA%94%E6%89%A9%E6%95%A3%E7%B3%BB%E7%BB%9F%E4%B8%8EGNN/fig7_series_journey.png)
*八章之旅。第 1-2 章用神经网络**解** PDE；第 3-4 章把训练重写成变分原理；第 5-6 章构造**保结构**的网络；第 7-8 章把同一套机器搬到生成模型和图学习上。*

八章可以分成四对：

| 对 | 章节 | 背后的 PDE |
|----|------|------------|
| 用 NN 解 PDE | [1](/zh/PDE与机器学习-一-物理信息神经网络/) PINN，[2](/zh/PDE与机器学习-二-神经算子理论/) 神经算子 | PDE 本身就是损失函数 |
| 变分视角 | [3](/zh/PDE与机器学习-三-变分原理与优化/) Deep Ritz，[4](/zh/PDE与机器学习-四-变分推断与Fokker-Planck方程/) VI / Fokker-Planck | 损失 $=$ 自由能；梯度流 $=$ 连续性方程 |
| 保结构流 | [5](/zh/PDE与机器学习-五-辛几何与保结构网络/) 辛网络，[6](/zh/PDE与机器学习-六-连续归一化流与Neural-ODE/) Neural ODE / CNF | 网络尊重流的辛 / 体积 / 散度结构 |
| 生成 + 图 PDE | [7](/zh/PDE与机器学习-七-扩散模型与Score-Matching/) 扩散模型，**8** RD + GNN | 正反向 Fokker-Planck；图上反应扩散 |

每一章背后都是同一句话：

> **一种神经架构是某个被离散化的 PDE。选架构 = 选 PDE。**

具体来说：

- **想在训练分布外外推？**——选算子学习的 PDE（第 2 章）。
- **想让网络尊重守恒量？**——选辛积分器（第 5 章）。
- **想要可计算的似然？**——选连续性方程，学它的漂移（第 6 章）。
- **想从复杂分布采样？**——选 Fokker-Planck，学它的 score（第 7 章）。
- **想要不会塌的深层 GNN？**——选反应扩散，而不是只有扩散（本章）。

PDE 视角不是看深度学习的唯一有用透镜，但它出乎意料地**有生产力**：每次我们问"对应的数值分析会怎么说？"都能换来一个具体的回报——一个稳定性界、一个步长约束、一个结构修复。这是物理范式给机器学习带来的红利，也是这两个领域之间的对话还远没说完的原因。

---

## 8. 习题

**练习 1.** 证明对连通图，$\mathbf{L}\mathbf{x} = \mathbf{0}$ 的唯一解是常向量。由此说明热方程把任何初值驱动到其均值。

> *解。* 由 (3)，$\mathbf{x}^\top\!\mathbf{L}\mathbf{x} = \tfrac{1}{2}\sum w_{ij}(x_i - x_j)^2 = 0$ 强制每条边两端 $x_i = x_j$；连通图上这意味着 $\mathbf{x}$ 是常向量。所以 $\mathbf{L}$ 的核是一维的，其余特征值严格为正，$e^{-\mathbf{L}t}$ 杀掉所有非常数分量并保住均值。$\blacksquare$

**练习 2.** 推导扩散步的显式 Euler 稳定性界 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$。

> *解。* 谱坐标下 Euler 更新为 $\hat{x}_k^{(\ell+1)} = (1 - \epsilon_d \lambda_k)\,\hat{x}_k^{(\ell)}$。要求每个 $k$ 都不增长 $|1 - \epsilon_d \lambda_k| \leq 1$，即 $0 \leq \epsilon_d \lambda_k \leq 2$，得 $\epsilon_d \leq 2/\lambda_{\max}$。要求单调衰减（不振荡）则更严格地要求 $\epsilon_d < 1/\lambda_{\max}$。$\blacksquare$

**练习 3.** 为什么 RDGNN 在异配图上特别有用？

> *解。* 异配图上邻居标签倾向相反。纯扩散把邻居特征相加平均，这本身就在**摧毁**判别信号——层数越多越糟。反应项**逐节点**操作，并以 $\mathbf{H}^{(0)}$ 为条件，因此可以输出节点专属的更新，**放大**邻居之间的差异，恢复类间可分性。$\blacksquare$

**练习 4.** 证明 RDGNN 的离散更新 (7) 是连续 RD-GNN (6) 的一阶 Lie-Trotter 算子分裂离散。

> *解。* 算子分裂把 $\dot{\mathbf{H}} = (\mathcal{D} + \mathcal{R})\,\mathbf{H}$ 分成两步交替的 Euler：先 $\mathbf{H}^{1/2} = \mathbf{H} + h\mathcal{D}\mathbf{H}$，再 $\mathbf{H}^{(\ell+1)} = \mathbf{H}^{1/2} + h\mathcal{R}\mathbf{H}^{1/2}$。在标准实现中两个算子都在同一个 $\mathbf{H}^{(\ell)}$ 上求值，结果就是 (7)。局部截断误差为 $\mathcal{O}(h^2[\mathcal{D}, \mathcal{R}])$，即一阶。$\blacksquare$

**练习 5.** 单条图灵不稳定性条件可以数值检验：取 Gray-Scott 的参数 $(D_u, D_v, F, k)$，在均匀稳态附近线性化，扫 $|\mathbf{k}|^2$ 看 $\det\,\mathbf{A}(|\mathbf{k}|^2)$ 何时变号。把这个写成代码，复现图 1 中的某一格。

> *解概要。* Gray-Scott 的均匀稳态满足 $u v^2 = F(1 - u)$ 且 $u v^2 = (F + k) v$。算这个稳态处的 $2\times 2$ Jacobian，构造 $\mathbf{A}(|\mathbf{k}|^2) = \mathbf{J} - |\mathbf{k}|^2 \mathrm{diag}(D_u, D_v)$，画 $\det\,\mathbf{A}$ vs $|\mathbf{k}|^2$。负的一段表示不稳定带，对应波长 $2\pi/|\mathbf{k}_*|$ 与模拟出来的模式可视尺度一致。$\blacksquare$

---

## 参考文献

[1] Turing, A. M. (1952). *The chemical basis of morphogenesis.* Phil. Trans. R. Soc. B, 237(641), 37-72.

[2] Pearson, J. E. (1993). *Complex patterns in a simple system.* Science, 261(5118), 189-192.

[3] Murray, J. D. (2003). *Mathematical biology II: Spatial models and biomedical applications* (3rd ed.). Springer.

[4] Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks.* ICLR. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

[5] Oono, K., & Suzuki, T. (2020). *Graph neural networks exponentially lose expressive power for node classification.* ICLR. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947)

[6] Chamberlain, B., Rowbottom, J., Gorinova, M., Webb, S., Rossi, E., & Bronstein, M. (2021). *GRAND: Graph neural diffusion.* ICML. [arXiv:2106.10934](https://arxiv.org/abs/2106.10934)

[7] Thorpe, M., Nguyen, T. M., Xia, H., Strohmer, T., Bertozzi, A., Osher, S., & Wang, B. (2022). *GRAND++: Graph neural diffusion with a source term.* ICLR.

[8] Eliasof, M., Haber, E., & Treister, E. (2021). *PDE-GCN: Novel architectures for graph neural networks motivated by partial differential equations.* NeurIPS. [arXiv:2108.01938](https://arxiv.org/abs/2108.01938)

[9] Di Giovanni, F., Rowbottom, J., Chamberlain, B., Markovich, T., & Bronstein, M. (2022). *Graph neural networks as gradient flows.* [arXiv:2206.10991](https://arxiv.org/abs/2206.10991)

[10] Choi, J., Hong, S., Park, N., & Cho, S.-B. (2023). *GREAD: Graph neural reaction-diffusion networks.* ICML. [arXiv:2211.14208](https://arxiv.org/abs/2211.14208)

[11] Eliasof, M., Haber, E., & Treister, E. (2024). *Graph neural reaction-diffusion models.* SIAM J. Sci. Comput. [arXiv:2406.10871](https://arxiv.org/abs/2406.10871)

---

## 系列导航

| 部分 | 主题 |
|------|------|
| [1](/zh/PDE与机器学习-一-物理信息神经网络/) | 物理信息神经网络 |
| [2](/zh/PDE与机器学习-二-神经算子理论/) | 神经算子理论 |
| [3](/zh/PDE与机器学习-三-变分原理与优化/) | 变分原理与优化 |
| [4](/zh/PDE与机器学习-四-变分推断与Fokker-Planck方程/) | 变分推断与Fokker-Planck方程 |
| [5](/zh/PDE与机器学习-五-辛几何与保结构网络/) | 辛几何与保结构网络 |
| [6](/zh/PDE与机器学习-六-连续归一化流与Neural-ODE/) | 连续归一化流与Neural ODE |
| [7](/zh/PDE与机器学习-七-扩散模型与Score-Matching/) | 扩散模型与Score Matching |
| **8** | **反应扩散系统与GNN（本文，终章）** |

*感谢看到这里。*
