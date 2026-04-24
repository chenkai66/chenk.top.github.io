---
title: "Graph Contextualized Self-Attention Network for Session-based Recommendation"
tags:
  - GNN
  - 自注意力
  - 会话推荐
categories: 论文笔记
lang: zh-CN
mathjax: true
---

会话推荐里你能看到的就那么一小段匿名点击序列：没有用户画像、没有历史长期偏好、没有人口统计。所有信号都封装在这几次点击里。**GC-SAN**（IJCAI 2019）的思路很务实：把当时最强的两个想法直接叠起来——SR-GNN 的会话图捕捉局部转移结构，Transformer 的自注意力捕捉长距离意图，最后用一个标量权重把"当前点击"和"全局意图"线性融合。它本身不发明新机制，但作为一个 baseline，至今仍然不容易被同等参数量级的模型超过。

## 你将学到什么

- 为什么会话推荐结构上比传统协同过滤难
- 一段点击序列如何变成一张有向加权图
- GGNN 单元：入/出邻居聚合 + GRU 门控
- 自注意力作为图上下文化嵌入之上的全局编码器
- 融合权重 $w$：最后一击 vs 全局意图
- GC-SAN 适合什么场景，不适合什么场景

## 预备知识

- 图神经网络的消息传递与 GRU 风格更新
- 自注意力（Q/K/V，缩放点积）
- 推荐评估指标 Recall@K、MRR@K

---

## 1. 问题设定与困难来源

记物品全集为 $V = \{v_1, v_2, \dots, v_{|V|}\}$，一次会话是一段按时间排好的点击序列 $s = (v_{s,1}, v_{s,2}, \dots, v_{s,n})$。任务是预测 $v_{s,n+1}$，通常对所有候选打分排序并报告 Recall@K、MRR@K。

为什么这件事比传统协同过滤更棘手：

- **没有长期画像**：没法依赖稳定的用户嵌入或人口统计特征。
- **行为短而嘈杂**：会话里可能夹杂探索性点击、误点、来回跳转。
- **长距离依赖**：早期的"相机"点击，可能直到后面"存储卡"出现时才显出关联。
- **重复转移**：用户经常在几个相关物品之间反复跳，单纯的序列模型容易浪费这种结构信息。

这四种压力相互掣肘。纯序列模型（RNN / Transformer）能处理顺序但把每一步都当成新 token；纯图模型（SR-GNN）能抓回环和重复，但要够到远距离的点击就需要多跳传播。GC-SAN 的设计正是把两者拼起来，避免单独使用任一方法时各自的代价。

## 2. GC-SAN 在已有工作中的位置

在 GC-SAN 之前，常见的 baseline 大致分四类：

- **马尔可夫链**：局部信号强，全局视角缺失。
- **GRU4Rec**：用 GRU 建会话内的顺序依赖，超过几步就开始吃力。
- **NARM、STAMP**：基于注意力的序列模型，长距离依赖处理得更好，但忽略了会话内部天然存在的转移图结构。
- **SR-GNN**：把会话建成有向图，跑门控 GNN，能捕捉丰富的局部结构，但远距离意图依赖多跳传播，层数堆多了又会过平滑。

GC-SAN 的贡献偏**架构**而非算法：保留 SR-GNN 的门控 GNN 作为局部编码器，再在上面叠一层 Transformer 风格的自注意力，让全局依赖不必靠图上多跳来达成。完整流程见图 1。

![GC-SAN 端到端流程：从点击到下一击打分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/graph-contextualized-self-attention-network-for-session-base/fig1_architecture.png)

整个 pipeline 是严格串行的：点击序列 $\to$ 建图 $\to$ GGNN 传播 $\to$ 多层自注意力 $\to$ 融合"最后一击"与"全局意图" $\to$ 对所有物品打分。

## 3. 会话图的构建

对每条会话 $s$ 构造有向图 $G_s = (V_s, E_s)$：

- **节点**：会话中出现过的不同物品（去重）。
- 对相邻点击对 $(v_{s,i}, v_{s,i+1})$ 加一条有向边 $v_{s,i} \to v_{s,i+1}$。
- 同一条转移多次出现就累加权重（或者多重边后再归一化）。

这一步在用紧凑性换信息密度。同一物品在序列里出现两次会被压成一个节点，只有它们之间的**转移**承载重复。回环——比如 A $\to$ B $\to$ A——在图里直接显成环，而单纯的序列模型只会把它当作"A 出现了两次"。

接着行归一化两个邻接矩阵：

$$
A^{out}_{ij} = \frac{w(v_i \to v_j)}{\sum_k w(v_i \to v_k)}, \quad
A^{in}_{ij}  = \frac{w(v_j \to v_i)}{\sum_k w(v_k \to v_i)}.
$$

图 2 走了一个完整例子。

![会话图构建：点击序列到有向加权图再到归一化邻接矩阵](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/graph-contextualized-self-attention-network-for-session-base/fig2_session_graph.png)

注意点击序列 `v1 v2 v3 v2 v4` 收敛成 4 个节点的图，里面有一对 `v2 <-> v3` 回环，以及 `v2` 同时通向 `v3` 和 `v4` 的扇出。原序列长度 5，图只有 4 节点 4 条不同的边——这种压缩正是图视角的价值。

## 4. 局部编码器：会话图上的 GGNN

GC-SAN 直接复用 SR-GNN 的门控 GGNN 单元。每个节点带一个 $d$ 维嵌入 $h_i$。一步传播包含三件事。

**(i) 聚合入/出邻居。** 对节点 $v_i$ 在第 $t$ 步：

$$
a_i^{(t)} \;=\; A^{in}_{i,:} \, H^{(t-1)} W^{in} \;+\; A^{out}_{i,:} \, H^{(t-1)} W^{out} \;+\; b,
$$

其中 $H^{(t-1)} = [h_1^{(t-1)}, \dots, h_{|V_s|}^{(t-1)}]^\top$ 把会话内所有节点嵌入堆起来，$W^{in}, W^{out} \in \mathbb{R}^{d \times d}$ 都是可学习的。这两项分别表示"流向我的证据"和"我流向别人的证据"，对方向敏感的转移很重要。

**(ii) GRU 门控。** 把聚合消息和上一时刻状态结合：

$$
\begin{aligned}
z_i^{(t)} &= \sigma(W_z a_i^{(t)} + U_z h_i^{(t-1)}), \\
r_i^{(t)} &= \sigma(W_r a_i^{(t)} + U_r h_i^{(t-1)}), \\
\tilde h_i^{(t)} &= \tanh\!\bigl(W_h a_i^{(t)} + U_h (r_i^{(t)} \odot h_i^{(t-1)})\bigr), \\
h_i^{(t)} &= (1 - z_i^{(t)}) \odot h_i^{(t-1)} \;+\; z_i^{(t)} \odot \tilde h_i^{(t)}.
\end{aligned}
$$

更新门 $z$ 决定新图信号写进多少；重置门 $r$ 决定形成候选状态时要忘掉多少旧状态。图 3 在单个节点上把这一步可视化。

![GGNN 单元：入/出邻居聚合 + GRU 门控](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/graph-contextualized-self-attention-network-for-session-base/fig3_ggnn_message_passing.png)

跑 $T$ 步传播之后（论文常用 $T=1$，有时 $T=2$），每个节点嵌入都吸收了它在图上的局部邻域。**关键点**：传播是在每条会话各自的图上做的，不是全局物品图，所以嵌入是会话条件化的。

> **实现细节**：会话里物品会重复出现，所以实现里要维护一个 *alias* 映射，把序列位置映射到去重后的节点索引。GGNN 跑完之后，再把节点状态"散布"回序列位置上，才能丢进自注意力。这就是任何标准实现里 `seq_hidden = hidden[alias_inputs]` 在做的事。

## 5. 全局编码器：会话上的自注意力

GGNN 在局部很强，但要触达远端点击需要多跳，而 GGNN 层数堆多了又会**过平滑**——节点嵌入互相塌陷。自注意力直接绕开这个问题：任何位置一步就能注意到任何其他位置。

设 $E^{(0)} \in \mathbb{R}^{n \times d}$ 是把 GGNN 节点状态散布回序列后的位置表示。一层自注意力为：

$$
F = \mathrm{softmax}\!\left(\frac{(E W^Q)(E W^K)^\top}{\sqrt{d}}\right)(E W^V),
$$

后面接一个 point-wise 前馈块加残差连接：

$$
E^{(1)} = \mathrm{ReLU}(F W_1 + b_1) W_2 + b_2 + F.
$$

叠 $k$ 层得到 $E^{(k)}$，也就是**图上下文化**之后的序列表示。图 4 展示了自注意力典型学到的模式，以及为什么 GGNN 单独做不到同样的连接。

![自注意力捕捉 GGNN 局部跳跃难以触及的全局意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/graph-contextualized-self-attention-network-for-session-base/fig4_self_attention.png)

热力图是示意性的，但模式是真实的：第一击"相机"会强烈地注意到会话深处与之主题相关的物品（存储卡、电池、相机包）——这种连接如果靠 GNN 走 4 跳，信号早就被平滑掉了。

## 6. 融合：最后一击 vs 全局意图

会话推荐几乎总是需要显式地混合两种信号：

- **当前兴趣** $h_t$：最后点击物品的嵌入，通常是最强的短期预测信号。
- **全局意图** $a_t$：自注意力栈最后一个位置的输出，整合了整段会话。

GC-SAN 用一个标量权重把它们组合：

$$
s_f \;=\; w \cdot a_t \;+\; (1 - w) \cdot h_t,
$$

然后用 $s_f$ 与物品嵌入表做内积打分、softmax 归一：

$$
\hat y \;=\; \mathrm{softmax}\!\bigl(s_f \, V^\top\bigr).
$$

权重 $w$ 是一个超参（通常的甜区在 $w \in [0.4, 0.6]$）。$w = 0$ 退化为只用最后一击；$w = 1$ 则完全丢掉强短期信号。图 5(b) 的扫描给出典型曲线——中间平坦、两端陡降。

## 7. 训练与评估

大多数会话推荐器用下一击 softmax 上的交叉熵：

$$
\mathcal{L} \;=\; -\sum_{i=1}^{|V|} y_i \log \hat y_i \;+\; \lambda \|\Theta\|^2,
$$

$y$ 是 one-hot 标签，$\Theta$ 是所有可学习参数。当 $|V|$ 很大时，常用 BPR 加负采样：

$$
\mathcal{L}_{\text{BPR}} \;=\; -\sum_{(u, i, j)} \log \sigma(\hat r_{ui} - \hat r_{uj}) + \lambda \|\Theta\|^2,
$$

其中 $i$ 是正样本（被点击），$j$ 是采样的负样本。BPR 适合隐式反馈场景，它直接优化"用户对正样本的偏好高于负样本"这一相对关系，不需要绝对评分。

标准 benchmark 是 **Yoochoose1/64** 和 **Diginetica**，报 **Recall@20** 与 **MRR@20**。图 5 复现了论文里的对比格局。

![GC-SAN 与 SR-GNN 等 baseline 的对比及融合权重消融](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/graph-contextualized-self-attention-network-for-session-base/fig5_perf_vs_baselines.png)

从图里能读出两件事：

1. **GC-SAN 相对 SR-GNN 的提升是稳定的，但不算夸张**（Recall 和 MRR 上大约 +1 到 +2 个点）。在图编码器之上加自注意力的边际价值是真实的，但有上限。
2. **融合权重曲线中间平、两端陡**——这恰恰是一个超参该有的样子：好调、不易翻车。

## 8. 工程实践要点

**alias 映射与批处理。** 会话里有重复物品，要把序列位置映射到去重图上的节点索引。多条会话图一起 batch 是个非平凡的事：要么拼成块对角邻接矩阵，要么用支持批量图操作的库（PyG、RecBole-GNN）。

**复杂度。**
- GGNN：每条会话约 $\mathcal{O}(T \cdot |E_s| \cdot d)$，$T$ 是传播步数，$|E_s|$ 是会话图边数。
- 自注意力：每条会话 $\mathcal{O}(n^2 d + n d^2)$，关于会话长度 $n$ 是二次。会话推荐里 $n$ 通常很短（往往 $< 50$），二次代价完全可接受。

**会改变行为的关键超参。**
- **传播步数 $T$**：太小抓不到多跳转移；太大会过平滑。论文默认 $T = 1$。
- **自注意力层数 / 头数**：层数多了容量大，但在 Diginetica 这种小数据集上容易过拟合。
- **融合权重 $w$**：控制"全局 vs 最后一击"。在验证集上扫一遍即可，最优点通常在 $0.5$ 附近且很平。

**Padding 与 mask。** 自注意力必须 mask 掉 padding 位置，否则梯度会泄漏到无意义的 token 上。这是移植代码时很容易悄无声息退化的一个点。

## 9. 参考实现简版

下面是 RecBole 风格的最简版本。GGNN 单元复用自 SR-GNN，其余的接线才是 GC-SAN 的真正贡献。

```python
import torch
from torch import nn
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole_gnn.model.layers import SRGNNCell


class GCSAN(SequentialRecommender):
    """GGNN 局部编码 + 自注意力全局编码 + 最后一击/全局融合。"""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 超参
        self.n_layers = config["n_layers"]                    # 自注意力层数
        self.n_heads = config["n_heads"]                      # 注意力头数
        self.hidden_size = config["hidden_size"]              # d
        self.inner_size = config["inner_size"]                # FFN 维度
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.step = config["step"]                            # GGNN 传播步数 T
        self.weight = config["weight"]                        # 融合权重 w

        # 模块
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size,
                                           padding_idx=0)
        self.gnncell = SRGNNCell(self.hidden_size)            # SR-GNN 门控 GGNN
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
        )
        self.loss_fct = BPRLoss()

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        # 1. 嵌入会话图中的去重物品
        hidden = self.item_embedding(x)

        # 2. GGNN 传播 T 步（按会话级别的入/出邻接）
        for _ in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        # 3. 把节点状态散布回序列位置
        seq_hidden = hidden[alias_inputs]
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)   # 最后一击 h_t

        # 4. 在图上下文化的序列上跑自注意力栈
        outputs = self.self_attention(seq_hidden, output_all_encoded_layers=True)
        at = self.gather_indexes(outputs[-1], item_seq_len - 1)  # 全局 a_t

        # 5. 线性融合 s_f = w * a_t + (1 - w) * h_t
        return self.weight * at + (1 - self.weight) * ht

    def calculate_loss(self, interaction):
        seq_output = self.forward(
            interaction["x"],
            interaction["edge_index"],
            interaction["alias_inputs"],
            interaction[self.ITEM_SEQ_LEN],
        )
        pos = self.item_embedding(interaction[self.POS_ITEM_ID])
        neg = self.item_embedding(interaction[self.NEG_ITEM_ID])
        return self.loss_fct(
            torch.sum(seq_output * pos, dim=-1),
            torch.sum(seq_output * neg, dim=-1),
        )
```

几个值得注意的点：

- GGNN 单元是**直接 import 复用**而非重写，再次说明 GC-SAN 主要是接线创新。
- `gather_indexes` 在 `item_seq_len - 1` 位置取嵌入，拿到的是真正的最后一击，而不是被 padding 撑出的尾部。
- `self.weight` 是 config 里读的固定标量。一个自然的延伸是让它输入相关（在 $h_t \oplus a_t$ 上接一个小 gating MLP），但论文没有展开。
- 损失可以在 BPR 和全 softmax 交叉熵之间根据 $|V|$ 大小切换。

## 10. 什么时候用 GC-SAN，什么时候不用

**适合：**

- 会话内部有显著的转移结构（回环、重复、相关物品来回跳）。
- 会话长度短到中等，$\mathcal{O}(n^2)$ 自注意力代价可接受。
- 想要一个把图和序列信号都吃到的强 baseline，且不引入额外基础设施。

**局限和风险：**

- 自注意力代价随会话长度二次增长。会话特别长就要换线性注意力变体，或者把会话切块。
- 建图选择（边权、归一化方式）会悄悄影响结果，改动时记得在小验证集上对照。
- 如果物品 metadata 很关键（文本、图像、类目层级），纯 ID 版本的 GC-SAN 没把这些信息吃进去。可以在 GGNN 之前拼接侧信息嵌入，或者切到内容感知变体。
- 相对 SR-GNN 的提升真实但不大。如果上不起一层 self-attention，单独的 SR-GNN 也是个不错的起点。

## 11. 实操总结

GC-SAN 与其说是巧思，不如说是一份冷静的"两个都用"配方：

- **GGNN** 高效抓局部转移和重复——但要够远必须多跳，多跳就过平滑。
- **自注意力** 一步抓长距离依赖——但把输入当扁平序列，看不到图结构。
- 用一个标量**融合权重** $w$ 把"最后一击"和"全局意图"接起来，且甜区很平。

放到 2024 年看，GC-SAN 仍然是一个值得对标的 baseline：它能告诉你新模型是不是真的比"图编码器 + Transformer 块"更善于利用会话结构。如果不是，那就少花点算力，先弄清楚瓶颈在哪。

## 参考文献

- Xu et al., "Graph Contextualized Self-Attention Network for Session-based Recommendation," IJCAI 2019. [论文 PDF](https://www.ijcai.org/proceedings/2019/0547.pdf)
- Wu et al., "Session-based Recommendation with Graph Neural Networks (SR-GNN)," AAAI 2019.
- Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)," ICLR 2016.
- Li et al., "Neural Attentive Session-based Recommendation (NARM)," CIKM 2017.
- Liu et al., "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation," KDD 2018.
