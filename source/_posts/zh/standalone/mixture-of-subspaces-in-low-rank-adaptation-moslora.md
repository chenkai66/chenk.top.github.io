---
date: 2023-04-15 09:00:00
title: "Mixture-of-Subspaces in Low-Rank Adaptation (MoSLoRA)"
tags:
  - PEFT
  - LoRA
  - 参数高效微调
categories: 论文笔记
lang: zh-CN
mathjax: true
disableNunjucks: true
---

LoRA 把"全量微调"压缩成一个低秩更新，在工程上几乎是免费的：参数少、训练稳、能合并回原权重，因此部署时和原模型一样便宜。但只要你的微调数据稍微"杂"一点——把代码、数学、指令跟随、写作放到一起——单一低秩子空间就显得不够用了。直觉上的解法是把 $r$ 调大，可惜代价线性增长，而且本质上依然只有**一个**子空间，只是更"胖"了。

把 LoRA 和 MoE 拼起来当然是另一条路：把若干个 LoRA 当成专家，加一个 router 做选择。但这条路会丢掉 LoRA 最值钱的两个性质——可合并、推理零开销，并且引入路由训练不充分、负载不均衡等一系列工程问题。

[**MoSLoRA**](https://arxiv.org/abs/2406.11909)（Wu, Huang, Wei, 2024）选择了一条更克制的路线：去掉 router，用一个**可学习的混合矩阵** $W$ 把 $k$ 个低秩子空间组合起来，整体可以重写成一个干净的 $B\,W\,A$ 乘积。这样既保留了 LoRA 的可合并性与零推理开销，又把"单一子空间"的容量瓶颈打开。下面按"动机—结构—与 MoE 的边界—参数与算力—经验现象—调参建议"的顺序把这篇工作梳理一遍，重点说清楚 mixer 到底在干什么，以及它和 LoRA / LoRA-MoE 的边界到底在哪里。

## 你将了解到

- 为什么"单纯调大 $r$"不是好答案，瓶颈到底在哪
- MoSLoRA 的核心结构：$k$ 个低秩子空间 + 一个 $k\times k$ 的 mixer
- mixer 的几种实现选择：全局 mixer、输入相关 gate、按层/按投影分组
- 与 LoRA、LoRA-MoE、Full FT 在参数量、推理代价上的对比
- MoSLoRA 真正起作用的场景，以及哪些场景里 LoRA 已经够用
- 一份干净的 PyTorch 实现 + 实战调参建议

## 前置知识

- 熟悉 LoRA 的低秩分解和 Transformer 中线性投影的位置（Q/K/V/O、MLP up/down/gate）
- 了解 PEFT 家族的基本术语（Adapter、Prefix-Tuning、BitFit）
- 对 MoE 的 router、top-k、负载均衡有基本概念

---

## 1. LoRA 回顾：为什么低秩管用，又是在哪里失效

设 Transformer 中任意一个线性投影 $W \in \mathbb{R}^{d_{out}\times d_{in}}$。LoRA 把 $W_0$ 冻住，把更新写成低秩分解：

$$
W \;=\; W_0 + \Delta W,
\qquad
\Delta W \;=\; \frac{\alpha}{r}\, B\, A,
\qquad
B \in \mathbb{R}^{d_{out}\times r},\;
A \in \mathbb{R}^{r\times d_{in}}.
$$

当 $r \ll \min(d_{in}, d_{out})$ 时，可训练参数量从 $d_{in}\!\cdot\!d_{out}$ 降到 $r(d_{in}+d_{out})$，通常只占全量微调的 $0.1\%$ 到 $1\%$。$B$ 初始化为 $0$ 保证微调起步等价于原模型；推理时把 $W_0 + \frac{\alpha}{r} B A$ 合并成同形状的稠密矩阵，所以**部署成本和原模型完全一致**。

![LoRA 结构回顾：冻结基座 + 单一低秩更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/mixture-of-subspaces-in-low-rank-adaptation-moslora/fig1_lora_recap.png)

LoRA 隐含了一个很强的假设：**一个**秩为 $r$ 的子空间就是更新的合理形状。在窄场景下这条假设没问题；一旦数据变得多样，这条假设就开始失效，原因有三：

- 最优的 $\Delta W$ 可能在**单个任务上低秩**，但**跨任务整体高秩**——不同子任务想把权重往不同方向推。
- 单一子空间强行让所有方向共用同一对 $A$、$B$。一个任务的梯度会覆盖另一个任务学到的有用方向，出现"互相拆台"。
- 简单调大 $r$ 只能让子空间"更胖"，但最优解可能想要的是**几个细子空间的结构化组合**，而不是一个胖子空间。

这正是 MoSLoRA 想填的缺口。

## 2. 为什么"调大 $r$"不是答案

调大 $r$ 当然能增加容量，但代价在大模型规模上很容易压垮收益：

- **参数与显存随 $r$ 线性增长**：$r$ 翻倍，每层的 adapter 占用就翻倍。
- **收益边际递减**：经验上 $r$ 过了 $8$–$16$ 之后，下游任务的精度曲线就开始变平。
- **本质仍是一个子空间**：如果损失曲面在不同输入处偏好不同方向（loss surface 是分块低秩的），多出的维度并不能被同时利用，相当于浪费容量。

我们真正想要的是**结构化容量**：几个互相区分的低秩子空间，让模型按需组合。MoSLoRA 给出的正是这种结构，且边际成本几乎为零。

## 3. MoSLoRA 的核心结构：$\Delta W = B\, W\, A$

MoSLoRA 用一个**可学习的 mixer 矩阵**把 $k$ 个低秩子空间组合起来：

$$
\Delta W
\;=\;
\sum_{i=1}^{k} W_{ii}\, B_i\, A_i
\;=\;
B\, W\, A,
$$

其中 $A \in \mathbb{R}^{kr\times d_{in}}$ 把 $k$ 个 down-projection 堆在一起，$B \in \mathbb{R}^{d_{out}\times kr}$ 把 $k$ 个 up-projection 堆在一起，$W \in \mathbb{R}^{kr\times kr}$ 就是 mixer。$W$ 取对角时退化为"$k$ 个独立 LoRA 之和"；非对角项的关键作用是**让子空间 $i$ 的 up-projection 与子空间 $j$ 的 down-projection 配对**——额外的表达能力就来自这里。

![MoSLoRA 架构：$k$ 个低秩子空间 + 一个可学习 mixer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/mixture-of-subspaces-in-low-rank-adaptation-moslora/fig2_subspace_mixing.png)

这个写法直接带来三个性质：

- **没有 router、没有 top-k、没有负载均衡**：每个 token 上所有子空间都会贡献，训练像 LoRA 一样平滑。
- **推理可合并**：$BWA$ 是一次稠密矩阵乘，$W_0 + \frac{\alpha}{r} BWA$ 又能折叠回同形状的稠密矩阵。LoRA-MoE 一类设计通常就是在这一步丢掉了可合并性。
- **mixer 的额外参数可以忽略**：$k$ 个秩为 $r$ 的子空间一共多了 $kr(d_{in}+d_{out}) + (kr)^2$ 个参数。$(kr)^2$ 是 mixer 的全部新增成本，典型设置 $k=4, r=8$ 下也只有 $1024$ 个标量。

可以把每对 $(B_i, A_i)$ 想象成权重空间里的一个"旋钮"，指向某个方向；mixer 学的是**每个旋钮该开多大**，以及**旋钮之间该怎么组合**。$k=1$ 时 MoSLoRA 退化为 LoRA；$k>1$ 且 $W$ 非对角时，更新族严格更丰富，但每 token 的算力开销变化很小。

## 4. 与 LoRA-MoE、经典 MoE 路由的边界

很容易把 MoSLoRA 读成"去掉 gate 的 LoRA-MoE"，但这个对比稍微有点偏。经典 MoE 的代价主要来自：

- 每个 token 上由 **router** 显式做专家选择
- 用 **top-k** 稀疏化控制算力
- 用**负载均衡损失**避免专家塌缩
- 推理图依赖 token，无法把权重提前合并

LoRA-MoE 把 LoRA 当成专家，自然继承了这些代价：每个 token 只激活部分专家，可合并性也随之消失。

MoSLoRA 完全反过来：**没有路由，所有 $k$ 个子空间一直全开；mixer 是"线性组合器"，不是"离散选择器"**。这个设计的取舍是有意的：

- 你失去了 MoE 那种"想加专家就加几十个"的条件稀疏。
- 你换回了平滑优化、可合并权重，以及和 LoRA 完全一致的部署体验。

LoRA 被广泛使用的场景就是"一个基座 + 少量 adapter，部署成本等于基座"。在这个场景里，MoSLoRA 的取舍正合适。

## 5. mixer 的几种实现选择

mixer $W$ 有几种常见变体。论文默认用最简单那种，但其它形式在后续工作里都很常见，值得一并知道。

### 全局 mixer（默认）

每个被改造的投影对应一个 $W \in \mathbb{R}^{kr\times kr}$，端到端学习，不依赖输入。便宜、稳定、可合并。建议把 $W$ 初始化在单位阵附近——这样 step 0 的 MoSLoRA 行为等价于"$k$ 个独立 LoRA 的和"，非对角的 mixing 项是训练中"长出来"的。

### 输入相关 gate

把 $W$ 换成 $W(x) = g(x) \in \mathbb{R}^{k\times k}$，由一个对 pooled hidden state 的小投影生成。每个输入有自己的组合权重。表达力更强，但代价是：

- 推理无法预合并（$W$ 依赖 $x$）
- 数据少时 gate 容易过拟合

只有在任务分布**真的多模**且数据量足够时才考虑用。

### 按层 / 按投影分组

不同 Transformer 层、不同投影（Q/K/V/O/MLP up/gate/down）用不同的 $W$。代价仍然很小，但归纳偏置更强：浅层可以学一种 mixing 模式，深层学另一种。实践中往往是表达力与稳定性的最佳平衡点。

### 结构化 / 低秩 mixer

当 $k$ 很大时，把稠密的 $W$ 改成低秩或块对角，避免 $(kr)^2$ 暴涨。在 $k\le 8$ 这种常见规模下基本用不到。

## 6. 参数量与算力开销

每个被改造的投影（形状 $d_{out}\times d_{in}$，$k$ 个秩为 $r$ 的子空间）的对比如下：

| 量                          | LoRA                  | MoSLoRA                                       |
|-----------------------------|-----------------------|-----------------------------------------------|
| 可训练参数                  | $r(d_{in}+d_{out})$   | $kr(d_{in}+d_{out}) + (kr)^2$                 |
| 单 token 前向 FLOPs         | $r(d_{in}+d_{out})$   | $kr(d_{in}+d_{out}) + (kr)^2$                 |
| 推理可合并                  | 是                    | 是（全局 mixer）；否（输入相关 gate）         |
| 路由 / 负载均衡             | 无                    | 无                                            |

取一个具体例子：$d_{in}=d_{out}=4096$，$r=8$，$k=4$。LoRA 每个投影约 $65$K 参数；MoSLoRA 约 $263$K 加上 $1024$ 个 mixer 参数。把 8B 模型的全部 attention + MLP 投影都改一遍，整体仍在总参数的 $1\%$ 以下。

mixer 这一项实际上是渐近免费的：只要 $kr \ll d_{in}$（这是你本来就想满足的条件），$(kr)^2$ 就被 $kr(d_{in}+d_{out})$ 完全压住。

## 7. 实证：差距究竟出现在哪里

这篇论文以及后续相关工作里反复出现的现象是：**MoSLoRA 在异质化任务上稳定地比 LoRA 高一小到中等的幅度，并且整体仍在 LoRA 的成本范围内**。

![下游任务表现：MoSLoRA 与 LoRA 在异质化任务上的对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/mixture-of-subspaces-in-low-rank-adaptation-moslora/fig3_downstream_perf.png)

有两个规律值得记住：

1. **任务越异质，差距越大**。单一领域的推理基准上提升较小；指令微调混合数据集和多技能 benchmark 上提升较明显。这和直觉一致：一个方向就够时一个子空间够用，方向多起来才需要多子空间。
2. **当 $r$ 已经到 $8$ 以上时，加 $k$ 比加 $r$ 更划算**。在"参数比 vs 精度"的 Pareto 图上，加子空间能把曲线往上推得更快。

![参数效率：MoSLoRA 把 Pareto 前沿整体上抬](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/mixture-of-subspaces-in-low-rank-adaptation-moslora/fig4_param_efficiency.png)

这背后的几何直观是最干净的解释：

![一个胖子空间 vs 多个细子空间](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/mixture-of-subspaces-in-low-rank-adaptation-moslora/fig5_subspace_visualisation.png)

一个 LoRA 是一条拉长的"主轴"；偏离这条主轴的目标只能被近似投影回去，残差和夹角成正比。MoSLoRA 把若干条细子空间放在不同方向上，再用 mixer 组合，可达集就是一个**结构化的子空间并集**，而不是一条胖主轴。

## 8. MoSLoRA 什么时候值得用，什么时候不值得

**值得用**的几个信号：

- 微调数据跨多个子技能 / 多个领域（指令微调、多领域适配、多模态指令）
- 已经在 LoRA 上把 $r$ 调大、看到精度饱和
- 需要 LoRA 那样的部署性质（可合并、零路由），但又想要超过 vanilla LoRA 的容量、又不想上 LoRA-MoE

**不值得用**的情况：

- 任务窄、同质，LoRA 中等 $r$ 已经能逼近全量微调
- 数据量相对 adapter 规模偏小，mixer 容易过拟合
- 你确实需要最大容量、可以接受真正的 MoE 部署成本——这种场景上 LoRA-MoE 或稀疏 MoE 上限更高

## 9. 实战调参建议

挑几个高信号的 knob：

- **从小起步**：$k=2$ 或 $4$、$r=8$ 是个稳定的默认。$k=8$ 只有在数据非常异质时才有明显额外收益。
- **mixer 初始化在单位阵附近**：$W \leftarrow I + \varepsilon \cdot \text{Gaussian}$，让 step 0 接近"独立 LoRA 之和"，非对角项在训练中按需长出来。Gaussian 直接初始化容易让收敛变慢。
- **优先改 attention，再改 MLP**：Q/K/V/O 是杠杆最大的位置，把这几个先改完，再看是否需要把 MLP up/gate/down 也加上。
- **把 $\alpha/r$ 调小一些**：$k$ 个子空间求和，等效增益放大了 $k$ 倍，把 $\alpha$ 减半是安全起点。
- **盯一下 mixer 的奇异值谱**：如果 $W$ 的奇异值塌缩到一个方向，模型就退化成普通 LoRA 了，往往是正则太重或 mixer 学习率太低。
- **如果不是非要不可，别上输入相关 gate**：全局 mixer 可合并、几乎不会成为瓶颈；动态 gate 训练更难、还破坏可合并性。

## 10. PyTorch 实现

一个最小但完整的实现，可以替换任意 `nn.Linear`：

```python
import torch
import torch.nn as nn

class MoSLoRALinear(nn.Module):
    """y = x W0^T + (alpha / r) * x A^T W^T B^T."""

    def __init__(self, base: nn.Linear, r: int = 8, k: int = 4,
                 alpha: float = 16.0):
        super().__init__()
        self.base = base                # 冻结基座
        for p in self.base.parameters():
            p.requires_grad = False

        d_in  = base.in_features
        d_out = base.out_features
        self.r, self.k = r, k
        self.scale = alpha / r

        # 堆叠的低秩因子: A in (k*r, d_in), B in (d_out, k*r)
        self.A = nn.Parameter(torch.empty(k * r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, k * r))   # B 初始化为 0
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        # mixer: W in (k*r, k*r), 初始化在单位阵附近
        self.W = nn.Parameter(torch.eye(k * r) + 0.01 * torch.randn(k * r, k * r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)              # 基座路径
        h = x @ self.A.T              # (..., k*r)
        h = h @ self.W.T              # (..., k*r) -- mixer 在这里发挥作用
        h = h @ self.B.T              # (..., d_out)
        return y + self.scale * h

    @torch.no_grad()
    def merge(self) -> None:
        """把 MoSLoRA 折叠回基座权重，部署用。"""
        delta = self.scale * (self.B @ self.W @ self.A)   # (d_out, d_in)
        self.base.weight.add_(delta)
        # 清掉 adapter，避免后续前向重复加
        self.B.zero_()
        nn.init.eye_(self.W)
```

三处细节值得强调：

1. `B` 初始化为 $0$，保证微调启动时模型与基座完全一致——和 LoRA 一脉相承。
2. `W` 初始化在单位阵附近：MoSLoRA 起步等价于"$k$ 个独立 LoRA 之和"，非对角的 mixing 是训练中"长出来"的。
3. `merge()` 把整套 MoSLoRA 折叠成一个稠密 `Linear` —— 这一步是 MoSLoRA 在生产环境零推理开销部署的关键。

## 11. 全景对比：LoRA / MoSLoRA / LoRA-MoE / Full FT

| 方法          | 可训练参数比例   | 表达力            | 推理代价                 | 适合场景                                                  |
|---------------|------------------|-------------------|--------------------------|-----------------------------------------------------------|
| **Full FT**   | 100%             | 最高              | 基线（1x）              | 单一同质任务、无部署约束                                  |
| **LoRA**      | $\sim 0.1$–$1\%$ | 中                | $\approx 1$x（可合并）   | 单任务或窄分布任务                                        |
| **MoSLoRA**   | $\sim 0.5$–$3\%$ | 高                | $\approx 1$x（可合并）   | 异质多任务 / 多领域，且要保留 LoRA 风格部署               |
| **LoRA-MoE**  | $\sim 1$–$5\%$   | 高（稀疏激活）    | $> 1$x，不可合并         | 想要更高容量上限，可接受路由 / 不可合并的部署代价         |

核心结论是：MoSLoRA 和 LoRA 在**同一条**部署轴上往容量方向走了一步；LoRA-MoE 则切换到了另一条轴，运维代价完全不同。

## 12. MoSLoRA 最该被使用的场景

- **跨多任务族的指令微调**：代码、数学、推理、写作把权重往不同方向推，单一子空间难以兼顾。
- **多领域适配（金融 + 医疗 + 法律）**：每个领域天然有自己的更新方向；mixer 实际上学到了一种"软的按域组合"，而不需要显式 router。
- **持续 / 增量适配**：可以为新任务追加新的子空间而不动旧的，把 MoSLoRA 当成模块化的容量扩展工具。这一点超出原论文的实验范围，但在后续工作里已经能看到很自然的延伸。

## 总结

把 MoSLoRA 理解成"LoRA 的结构化容量升级"比理解成"轻量化 MoE"更准确：

- LoRA：一个秩为 $r$ 的子空间，权重空间里只有一条方向。
- MoSLoRA：$k$ 个秩为 $r$ 的子空间 + 一个 $kr \times kr$ 的小 mixer，推理时仍可折叠成单一稠密权重。

mixer 才是这套设计能成立的关键：它给了你一个秩为 $kr$ 的**结构化更新流形**，部署足迹和 LoRA 完全一致，并且绕开了 MoE 里所有最难处理的部分——没有 router、没有 top-k、没有负载均衡、不会破坏可合并性。对于跑异质化微调任务、又不愿承担 LoRA-MoE 运维代价的工程团队，目前来看这是 LoRA 家族里"容量–可部署性"权衡最务实的一个选择。

## 参考文献

- Wu, T., Huang, S. and Wei, F., 2024. **Mixture-of-Subspaces in Low-Rank Adaptation**. arXiv:2406.11909. [[paper]](https://arxiv.org/abs/2406.11909) [[code]](https://github.com/wutaiqiang/MoSLoRA)
- Hu, E.J., Shen, Y., Wallis, P., et al., 2022. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [[paper]](https://arxiv.org/abs/2106.09685)
- Fedus, W., Zoph, B. and Shazeer, N., 2022. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. JMLR. [[paper]](https://arxiv.org/abs/2101.03961)
- Zadouri, T., Ustun, A., Ahmadian, A., et al., 2024. **Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning**. arXiv:2309.05444. [[paper]](https://arxiv.org/abs/2309.05444)
- Liu, S.Y., Wang, C.Y., Yin, H., et al., 2024. **DoRA: Weight-Decomposed Low-Rank Adaptation**. ICML 2024. [[paper]](https://arxiv.org/abs/2402.09353)
