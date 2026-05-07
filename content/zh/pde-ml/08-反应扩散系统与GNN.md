---
title: "偏微分方程与机器学习（八）：反应扩散系统与GNN"
date: 2024-08-14 09:00:00
tags:
  - 反应扩散
  - 图神经网络
  - PDE
  - 图上PDE
  - 模式形成
categories:
  - PDE与机器学习
series: pde-ml
lang: zh-CN
mathjax: true
description: "深层 GNN 之所以崩溃，是因为它就是图上的扩散方程；图灵 1952 年的反应扩散理论告诉我们如何修好它——也为整个八章 PDE+ML 系列收尾。"
disableNunjucks: true
series_order: 8
translationKey: "pde-ml-8"
---
![偏微分方程与机器学习（八）：反应扩散系统与GNN — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-Reaction-Diffusion-Systems/illustration_1.jpg)

## 本文你会学到

堆 32 层 GCN 到引文网络上，准确率从 81% 跌到 20%。节点特征全收敛到同一点。这就是**过度平滑**，GNN 的"热寂"。病因来自 PDE 基础：**一层 GCN 是图上热方程的显式 Euler 一步**。热方程只有一个不动点：常数。解药 1952 年就有了。Alan Turing 证明，给扩散方程加一个**反应项**，均匀稳态能自发长出条纹、斑点、迷宫。同样方法（学得到的反应项）能让深层 GNN 活下来。

这是 *PDE+机器学习* 系列第 8 章，也是最后一章。前 7 章论证了"几乎所有现代神经网络架构都是某个 PDE 的离散化"。反应扩散 + GNN 收尾：它是最显式的 PDE 形态，也让我在终点回顾整套系列。

**本文目录**

1. 连续空间上的反应扩散方程——Gray-Scott、FitzHugh-Nagumo、生成的模式；
2. 图灵不稳定性——线性分析解释扩散如何"创造"结构；
3. 图拉普拉斯——$\nabla^2$ 的离散版本，谱决定 GNN 行为；
4. GCN $=$ 离散图扩散——过度平滑的谱证明；
5. 反应扩散 GNN（GRAND、GRAND++、RDGNN）——加反应项；
6. 回顾整个 PDE+ML 系列。

**前置知识**：线性代数（特征分解）、基本 PDE 概念（扩散方程）、消息传递 GNN。

---

![Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 $(F,k)$ 参数平面上的位置示意。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig1_turing_patterns.png)
*Gray-Scott 模型生成的四种图灵形态——斑点、条纹、迷宫、孔洞——以及它们在 $(F,k)$ 参数平面上的位置示意。*
## 1. 连续空间上的反应扩散

### 1.1 一般形式

反应扩散方程把空间扩散和局部非线性反应结合起来：
$$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D}\,\nabla^2\mathbf{u} + \mathbf{R}(\mathbf{u}). \tag{1}$$
- 扩散项 $\mathbf{D}\nabla^2\mathbf{u}$ 是线性的，抹平梯度。
- 反应项 $\mathbf{R}(\mathbf{u})$ 是局部的，非线性的，能加强或对抗扩散。

两种视角都有用。物理上，$\mathbf{u}$ 是浓度，扩散是 Fick 定律，反应是速率方程。数学上，(1) 是半线性抛物 PDE——热方程加逐点非线性源项。

Turing 的洞见在于：两项竞争能让均匀初值演化出稳定、非平凡的空间模式。这叫**扩散驱动不稳定性**。

### 1.2 Gray-Scott 模型

Gray-Scott 是经典双组分模型：
$$\partial_t u = D_u \nabla^2 u - u v^2 + F(1-u),\qquad
\partial_t v = D_v \nabla^2 v + u v^2 - (F+k)\,v.$$
- $u$ 是底物，按速率 $F$ 注入；$v$ 是自催化剂，消耗 $u$（反应 $u + 2v \to 3v$），以速率 $k$ 衰减。
- $D_u > D_v$ 时，$v$ 的小斑块会稳定下来，形成图 1 的形态。

同一方程，换不同 $(F, k)$，能生成斑点、条纹、迷宫、孔洞、移动斑点甚至自复制斑点——Pearson (1993) 分类出十几种相区。

### 1.3 FitzHugh-Nagumo 模型

最早是简化神经元模型：
$$\partial_t v = D \nabla^2 v + v - \tfrac{v^3}{3} - w + I,\qquad
\partial_t w = \varepsilon\,(v + \beta - \gamma w),\quad \varepsilon \ll 1.$$
- $v$ 是快变量（膜电位），$w$ 是慢恢复变量。
- 三次非线性让 $v$ 具有激发性：超阈值扰动触发脉冲，$w$ 随后复位。

二维下会出现螺旋波和靶心波——正是心脏纤颤和发育中视网膜的图样（见 §6 图 6）。
## 2. 图灵不稳定性：从均匀生出模式

### 2.1 问题

选一个均匀稳态 $\bar{\mathbf{u}}$，满足 $\mathbf{R}(\bar{\mathbf{u}}) = \mathbf{0}$，且在无扩散时稳定。加扩散会让它失稳吗？

直觉上不会——扩散只会抹平，怎么会添乱？Turing (1952) 证明直觉错了。

### 2.2 推导

将 (1) 在 $\bar{\mathbf{u}}$ 附近线性化，设扰动为 $\delta\mathbf{u}(\mathbf{x}, t) = \mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}\,e^{\sigma t}$：
$$\sigma\,\mathbf{q} \;=\; \underbrace{\bigl(\mathbf{J} - |\mathbf{k}|^2\,\mathbf{D}\bigr)}_{\mathbf{A}(|\mathbf{k}|^2)}\,\mathbf{q},\qquad
\mathbf{J} = \nabla_{\mathbf{u}}\mathbf{R}(\bar{\mathbf{u}}). \tag{2}$$
若 $\mathbf{A}(|\mathbf{k}|^2)$ 有正实部特征值，模式 $\mathbf{q}\,e^{i\mathbf{k}\cdot\mathbf{x}}$ 就会增长。完整图灵条件如下（图 2 右）：

1. $\mathrm{tr}\,\mathbf{J} < 0$ 且 $\det\,\mathbf{J} > 0$——无扩散时稳定；
2. Jacobian 具激活-抑制结构：$f_u > 0,\;g_v < 0,\;f_v g_u < 0$；
3. **扩散不对称**：$D_v \gg D_u$——抑制剂扩散远快于激活剂；
4. 存在 $|\mathbf{k}|^2$ 使 $\det\,\mathbf{A}(|\mathbf{k}|^2) < 0$，即不稳定波数。

前三条是动力学代数性质，第四条是机制：一段波数变得不稳定，最不稳定的模 $|\mathbf{k}_*|$ 决定模式特征长度 $\ell \sim 2\pi/|\mathbf{k}_*|$。

![左：激活-抑制系统色散关系 $\sigma(|\mathbf{k}|^2)$。等扩散（蓝）下处处稳定；抑制剂扩散更快（红），在 $|\mathbf{k}_*|^2 \approx 3.4$ 附近打开不稳定带。右：四条图灵条件一目了然。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig2_turing_instability.png)
*左：激活-抑制系统色散关系 $\sigma(|\mathbf{k}|^2)$。等扩散（蓝）下处处稳定；抑制剂扩散更快（红），在 $|\mathbf{k}_*|^2 \approx 3.4$ 附近打开不稳定带。右：四条图灵条件一目了然。*

### 2.3 直觉：短程激活、长程抑制

不对称扩散为何破坏原本稳定的状态？想象局部多了一点激活剂。**局部**，激活剂自我放大；同时生成抑制剂，但抑制剂跑得快，局部浓度低，**远处**浓度高，压制其他凸起。这就是**短程激活、长程抑制**——动物斑纹、植被条带、沙波纹的通用机制，也是 §5 深层 GNN 架构的核心原理。
## 3. 从网格到图

### 3.1 为什么用图

有限差分（FDM）和有限元（FEM）在规则网格或精心设计的网格上离散 PDE。简单几何域没问题。但**分子结构、社交网络、引文图、路网、脑连接组**没有自然的规则网格——连接关系就是几何。

图 $G = (V, E)$ 是统一形式：节点加节点间的关系。GNN 解的就是图上的"PDE"。本节任务是写出这个 PDE。

![从规则网格到不规则图。两者都在离散 $\nabla^2$，但格点 stencil 被节点的邻域代替。连续 PDE $\partial_t u = D\nabla^2 u + R(u)$ 同时容许两种离散；图版本的一步 Euler 就是一层 GCN。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig3_grid_to_graph.png)
*从规则网格到不规则图。两者都在离散 $\nabla^2$，但格点 stencil 被节点的邻域代替。连续 PDE $\partial_t u = D\nabla^2 u + R(u)$ 同时容许两种离散；图版本的一步 Euler 就是一层 GCN。*

### 3.2 图拉普拉斯

带权无向图，邻接矩阵 $\mathbf{A}$，度矩阵 $\mathbf{D} = \mathrm{diag}(d_i)$：

| 变体 | 公式 | 谱范围 |
|------|------|--------|
| 组合型 | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | $[0, 2 d_{\max}]$ |
| 对称归一 | $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ | $[0, 2]$ |
| 随机游走 | $\mathbf{L}_{\text{rw}} = \mathbf{I} - \mathbf{D}^{-1}\mathbf{A}$ | $[0, 2]$ |

三者共享一个核心性质：
$$\mathbf{x}^{\!\top}\!\mathbf{L}\mathbf{x} \;=\; \tfrac{1}{2}\sum_{(i,j) \in E} w_{ij}\,(x_i - x_j)^2 \;\geq\; 0. \tag{3}$$
图拉普拉斯是 $-\nabla^2$ 的离散版：对梯度平方积分。它对称半正定，谱分解为 $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\!\top}$，特征值 $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$。

最小特征值恒为 0，对应常向量 $\mathbf{1}$。第二小特征值 $\lambda_2$（代数连通度）刻画图的连通性。

### 3.3 图热方程

直接写连续时间动力学：
$$\frac{d\mathbf{X}}{dt} = -\mathbf{L}\mathbf{X}. \tag{4}$$
闭式解为 $\mathbf{X}(t) = e^{-\mathbf{L}t}\mathbf{X}(0)$。谱坐标下完全解耦：
$$\hat x_k(t) = e^{-\lambda_k t}\,\hat x_k(0),\qquad \hat x_k = \mathbf{u}_k^{\!\top}\mathbf{X}(0).$$
每个模按速率 $\lambda_k$ 指数衰减，只有 $\lambda_1 = 0$ 不变。$t \to \infty$ 时只剩常数。

![图热方程实战。50 节点小世界图上的随机初始信号被扩散抹平：$t = 6$ 时所有节点取同一值。右图解释原因——第 $k$ 个模按 $e^{-\lambda_k t}$ 衰减，只有 $\lambda_1 = 0$ 存活。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig4_graph_laplacian.png)
*图热方程实战。50 节点小世界图上的随机初始信号被扩散抹平：$t = 6$ 时所有节点取同一值。右图解释原因——第 $k$ 个模按 $e^{-\lambda_k t}$ 衰减，只有 $\lambda_1 = 0$ 存活。*

这是**最纯粹的过度平滑**，我还没提神经网络。
## 4. GCN 就是热方程

![偏微分方程与机器学习（八）：反应扩散系统与GNN — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-Reaction-Diffusion-Systems/illustration_2.jpg)

### 4.1 等价关系

标准 GCN 层（Kipf & Welling, 2017）：
$$\mathbf{H}^{(\ell+1)} = \sigma\bigl(\tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}\,\mathbf{W}^{(\ell)}\bigr),
\qquad \tilde{\mathbf{A}} = \tilde{\mathbf{D}}^{-1/2}(\mathbf{A} + \mathbf{I})\tilde{\mathbf{D}}^{-1/2}.$$
去掉非线性，权重换成单位阵（$\sigma = \mathrm{id}$，$\mathbf{W} = \mathbf{I}$），剩下：
$$\mathbf{H}^{(\ell+1)} = \tilde{\mathbf{A}}\,\mathbf{H}^{(\ell)}
\;=\; \bigl(\mathbf{I} - \tilde{\mathbf{L}}_{\text{sym}}\bigr)\mathbf{H}^{(\ell)}.
\tag{5}$$
这是图热方程 $\dot{\mathbf{H}} = -\tilde{\mathbf{L}}_{\text{sym}}\mathbf{H}$ 的显式 Euler，步长 $h = 1$。"加自环" $\mathbf{A} + \mathbf{I}$ 是 FDM 稳定化技巧，把 $\tilde{\mathbf{L}}_{\text{sym}}$ 的谱压进 $[0, 2)$，确保显式格式稳定。

### 4.2 过度平滑的谱证明

叠 $L$ 层（忽略非线性和权重）：
$$\mathbf{H}^{(L)} = \tilde{\mathbf{A}}^L\,\mathbf{H}^{(0)}.$$
$\tilde{\mathbf{A}}$ 的特征值在 $(-1, 1]$ 内，特征值 $1$ 对应常向量。取幂后，除了首特征空间全死光：
$$\tilde{\mathbf{A}}^L \xrightarrow[L \to \infty]{} \pi_{\text{const}}.$$
所有节点特征坍缩到同一个向量。**这不是 GCN 的问题，而是低通滤波器迭代的必然结果**。加上 ReLU 和可学习权重只能延缓过程：Oono & Suzuki (2020) 证明，对任意奇异值有界的权重序列，GCN 特征仍收敛到低维子空间。

### 4.3 连续深度 GNN

一层 GCN 是一步 Euler，为什么不直接解 ODE？**GRAND**（Chamberlain et al., 2021）是连续时间 GNN：
$$\frac{d\mathbf{X}}{dt} = -\mathcal{L}_\theta(\mathbf{X})\,\mathbf{X},\qquad \mathbf{X}(T) = \text{输出}.$$
$\mathcal{L}_\theta$ 是带学习注意力的拉普拉斯，积分用现成 ODE solver（如 Dormand-Prince）。这不能根治过度平滑——更精确地解热方程，还是在解热方程。**GRAND++**（Thorpe et al., 2022）加了源项；**RDGNN**（Eliasof et al., 2024 等）加了完整反应项。下一节我来构造后者。
## 5. RDGNN：反应扩散图神经网络

### 5.1 架构

连续时间 RD-GNN 是 (1) 的图版本：
$$\frac{d\mathbf{H}}{dt} = -\epsilon_d\,\mathbf{L}\,\mathbf{H} \;+\; \epsilon_r\,R_\theta(\mathbf{H}, \mathbf{H}^{(0)}).
\tag{6}$$
一步 Lie-Trotter 分裂，离散更新公式为：
$$\boxed{\;\mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} \;-\; \epsilon_d\,\mathbf{L}\,\mathbf{H}^{(\ell)} \;+\; \epsilon_r\,R_\theta\bigl(\mathbf{H}^{(\ell)},\,\mathbf{H}^{(0)}\bigr).\;} \tag{7}$$
三个模块（图 5）：

- **扩散** $-\epsilon_d \mathbf{L}\mathbf{H}^{(\ell)}$：标准图平滑。步长约束 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$，保证显式 Euler 稳定。
- **反应** $\epsilon_r R_\theta(\mathbf{H}^{(\ell)}, \mathbf{H}^{(0)})$：可学习的逐节点变换，通常用小 MLP 实现。条件化在 $\mathbf{H}^{(0)}$ 上，类似 ResNet 的输入跳跃，防止漂移。
- **跳过项** $\mathbf{H}^{(\ell)}$：让动力学贴近恒等映射，这是深层网络数值稳定的关键。

常见反应项设计（FitzHugh 风格）：
$$R_\theta(\mathbf{H}, \mathbf{H}^{(0)}) = \mathrm{MLP}_\theta\bigl([\mathbf{H} \,\Vert\, \mathbf{H}^{(0)}]\bigr) \;-\; \alpha\,\mathbf{H}.$$

![RDGNN 单层：扩散分支做图拉普拉斯平滑，反应分支是学得的逐节点非线性，输入跳跃避免漂移。重复 $L$ 次得到深层 GNN，不像 GCN 会塌陷。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig5_rdgnn_architecture.png)
*RDGNN 单层：扩散分支做图拉普拉斯平滑，反应分支是学得的逐节点非线性，输入跳跃避免漂移。重复 $L$ 次得到深层 GNN，不像 GCN 会塌陷。*

### 5.2 为什么有效

两种视角。

**谱视角**。纯扩散按 $e^{-\lambda_k t}$ 衰减非常数模；反应项与 $\mathbf{L}$ 无关，谱内容任意，能把高频能量重新注入。结果是频率上形成非平凡的能量分布。

**图灵视角**。如果 $R_\theta$ 学到激活-抑制结构（表达力强的 MLP 可以做到），且扩散强度 $\epsilon_d$ 让 $\mathbf{J} - \epsilon_d \lambda_k$ 在某些 $k$ 上不稳定，网络会出现**节点级图灵模式**——不同节点收敛到不同特征值，由图谱组织。这就是 GNN 的"鱼纹"。

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

架构极简：扩散用共享 GCN，反应用 $L$ 个小 MLP，两端各一个线性投影。图 6c 显示，仅靠这点料就能把 Cora 准确率维持到 64 层——比 GCN 崩溃深度多八倍。
## 6. 反应扩散已经胜出的地方

同一个方程，三个应用故事。

![从生物到 GNN。(a) Gray-Scott 模型生成逼真的皮毛斑纹——正是 Turing 提出的生物形态发生机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可见于心室纤颤和早期视皮层发育。(c) 同样的 RD 思想用于图结构，得到不会退化的深层 GNN——RDGNN 在 64 层时仍保持 ~80% Cora 准确率，而 GCN、GAT 和纯扩散 GRAND 都跌破 25%。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig6_applications.png)
*从生物到 GNN。(a) Gray-Scott 模型生成逼真的皮毛斑纹——正是 Turing 提出的生物形态发生机制。(b) FitzHugh-Nagumo 动力学产生螺旋波，可见于心室纤颤和早期视皮层发育。(c) 同样的 RD 思想用于图结构，得到不会退化的深层 GNN——RDGNN 在 64 层时仍保持 ~80% Cora 准确率，而 GCN、GAT 和纯扩散 GRAND 都跌破 25%。*

**形态发生**  
Murray 的《Mathematical Biology》用图灵机制解释大型猫科动物的斑纹变化：体型大的（美洲虎）出斑点，中等的（豹）出玫瑰花斑，尾巴（局部几何限制 $|\mathbf{k}|$）出条纹——一套反应参数加胚胎几何就能解释。同样数学预测了半干旱地带的植被条带（水为抑制剂，生物量为激活剂）和 Belousov-Zhabotinsky 化学实验中的螺旋臂。

**神经发育**  
视网膜镶嵌形成时，相邻光感受器互相抑制分化为同一亚型，但通过扩散性形态发生素远程促进同型分化——这在数学上就是图灵系统，得到的视锥排列具有可测波长 $\ell \sim 2\pi/|\mathbf{k}_*|$。皮层电活动中的螺旋波（癫痫时是病理，发育时参与连接）是 2D 可激发介质上的 FitzHugh-Nagumo 解。

**深层 GNN**  
在标准基准上，深度与精度的关系戏剧化（图 6c，复刻自 Eliasof et al. (2024) 的 Cora 实验）。GCN/GAT 超过 8 层后崩溃，符合谱证明预测；纯扩散 GRAND 只是推迟崩溃，因为它更精确解热方程；只有 RDGNN——显式反应项——在 $L = 64$ 时维持精度。这种效应在异配图上更显著，因为平滑对异配图有害：反应项能学习放大邻居差异。
## 7. 系列收官：八章一念

到了 *PDE+机器学习* 系列的终点。退一步看。

![八章之旅。第 1-2 章用神经网络解 PDE；第 3-4 章把训练重写成变分原理；第 5-6 章构造保结构的网络；第 7-8 章把同一套方法搬到生成模型和图学习上。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/08-反应扩散系统与GNN/fig7_series_journey.png)
*八章之旅。第 1-2 章用神经网络解 PDE；第 3-4 章把训练重写成变分原理；第 5-6 章构造保结构的网络；第 7-8 章把同一套方法搬到生成模型和图学习上。*

八章分成四对：

| 对 | 章节 | 背后的 PDE |
|----|------|------------|
| 用 NN 解 PDE | [1](/zh/pde-ml/01-物理信息神经网络/) PINN，[2](/zh/pde-ml/02-神经算子理论/) 神经算子 | PDE 本身就是损失函数 |
| 变分视角 | [3](/zh/pde-ml/03-变分原理与优化/) Deep Ritz，[4](/zh/pde-ml/04-变分推断与fokker-planck方程) VI / Fokker-Planck | 损失 $=$ 自由能；梯度流 $=$ 连续性方程 |
| 保结构流 | [5](/zh/pde-ml/05-辛几何与保结构网络/) 辛网络，[6](/zh/pde-ml/06-连续归一化流与neural-ode/) Neural ODE / CNF | 网络尊重流的辛 / 体积 / 散度结构 |
| 生成 + 图 PDE | [7](/zh/pde-ml/07-扩散模型与score-matching/) 扩散模型，**8** RD + GNN | 正反向 Fokker-Planck；图上反应扩散 |

每一章背后都是同一句话：

> **一种神经架构是某个被离散化的 PDE。选架构 = 选 PDE。**

具体来说：

- **想在训练分布外外推？** 选算子学习的 PDE（第 2 章）。
- **想让网络尊重守恒量？** 选辛积分器（第 5 章）。
- **想要可计算的似然？** 选连续性方程，学它的漂移（第 6 章）。
- **想从复杂分布采样？** 选 Fokker-Planck，学它的 score（第 7 章）。
- **想要不会塌的深层 GNN？** 选反应扩散，而不是只有扩散（本章）。

PDE 视角不是看深度学习的唯一透镜，但它出乎意料地有生产力。每次问“对应的数值分析会怎么说？”都能换来具体回报——一个稳定性界、一个步长约束、一个结构修复。这是物理范式给机器学习带来的红利，也是两个领域对话远未结束的原因。
## 8. 习题

**练习 1.** 证明对连通图，$\mathbf{L}\mathbf{x} = \mathbf{0}$ 的唯一解是常向量。说明热方程把任何初值驱动到均值。

> *解。* 由 (3)，$\mathbf{x}^\top\!\mathbf{L}\mathbf{x} = \tfrac{1}{2}\sum w_{ij}(x_i - x_j)^2 = 0$ 强制每条边两端 $x_i = x_j$。连通图上 $\mathbf{x}$ 必为常向量。$\mathbf{L}$ 的核是一维的，其余特征值严格为正。$e^{-\mathbf{L}t}$ 杀掉非常数分量并保留均值。$\blacksquare$

**练习 2.** 推导扩散步显式 Euler 稳定性界 $\epsilon_d < 1/\lambda_{\max}(\mathbf{L})$。

> *解。* 谱坐标下 Euler 更新为 $\hat{x}_k^{(\ell+1)} = (1 - \epsilon_d \lambda_k)\,\hat{x}_k^{(\ell)}$。要求每个 $k$ 满足 $|1 - \epsilon_d \lambda_k| \leq 1$，即 $0 \leq \epsilon_d \lambda_k \leq 2$，得 $\epsilon_d \leq 2/\lambda_{\max}$。单调衰减要求更严格，$\epsilon_d < 1/\lambda_{\max}$。$\blacksquare$

**练习 3.** 为什么 RDGNN 在异配图上特别有用？

> *解。* 异配图上邻居标签倾向相反。纯扩散平均邻居特征，摧毁判别信号，层数越多越糟。反应项逐节点操作，以 $\mathbf{H}^{(0)}$ 为条件，输出节点专属更新，放大邻居差异，恢复类间可分性。$\blacksquare$

**练习 4.** 证明 RDGNN 离散更新 (7) 是连续 RD-GNN (6) 的一阶 Lie-Trotter 算子分裂离散。

> *解。* 算子分裂将 $\dot{\mathbf{H}} = (\mathcal{D} + \mathcal{R})\,\mathbf{H}$ 分成两步交替 Euler：先 $\mathbf{H}^{1/2} = \mathbf{H} + h\mathcal{D}\mathbf{H}$，再 $\mathbf{H}^{(\ell+1)} = \mathbf{H}^{1/2} + h\mathcal{R}\mathbf{H}^{1/2}$。标准实现中两个算子在同一点 $\mathbf{H}^{(\ell)}$ 上求值，结果就是 (7)。局部截断误差为 $\mathcal{O}(h^2[\mathcal{D}, \mathcal{R}])$，即一阶。$\blacksquare$

**练习 5.** 单条图灵不稳定性条件可以数值检验：取 Gray-Scott 参数 $(D_u, D_v, F, k)$，在均匀稳态附近线性化，扫 $|\mathbf{k}|^2$ 看 $\det\,\mathbf{A}(|\mathbf{k}|^2)$ 何时变号。写代码复现图 1 中某一格。

> *解概要。* Gray-Scott 均匀稳态满足 $u v^2 = F(1 - u)$ 且 $u v^2 = (F + k) v$。计算该稳态处 $2\times 2$ Jacobian，构造 $\mathbf{A}(|\mathbf{k}|^2) = \mathbf{J} - |\mathbf{k}|^2 \mathrm{diag}(D_u, D_v)$，画 $\det\,\mathbf{A}$ vs $|\mathbf{k}|^2$。负段表示不稳定带，对应波长 $2\pi/|\mathbf{k}_*|$ 与模拟模式尺度一致。$\blacksquare$
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

*This is Part 8 — the final part — of the [PDE and Machine Learning](/zh/categories/pde-and-machine-learning/) series. Previous: [Part 7 — Diffusion Models and Score Matching](/en/pde-ml/07-diffusion-models/). Start from the beginning: [Part 1 — Physics-Informed Neural Networks](/en/pde-ml/01-physics-informed-neural-networks/). Thanks for reading.*
