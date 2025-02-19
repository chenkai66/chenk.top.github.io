---
title: Graph Contextualized Self-Attention Network for Session-based Recommendation
tags:
  - GNN
  - Recommend System
categories: Paper
date: 2024-08-11 9:00:00
mathjax: true
---

GC-SAN 是一种新型的会话推荐模型，旨在通过结合图神经网络（GNN）和自注意力机制（Self-Attention），有效捕捉用户当前会话中的物品转移模式。模型首先利用 GNN 处理局部的物品依赖关系，再通过多层自注意力网络捕捉会话中的全局偏好。最终，模型将会话中的局部兴趣和全局兴趣加权融合，生成用户的最终会话表示，以预测下一步的点击物品。实验结果表明，GC-SAN 在多个真实数据集上优于现有的会话推荐方法，尤其在建模长距离物品依赖和捕捉复杂物品关系上表现突出。

<!-- more -->

# 背景介绍

在推荐系统中，**会话推荐（Session-based Recommendation）** 是一种基于用户当前会话行为来预测其下一步点击的任务。不同于传统的推荐系统依赖于用户的长期历史行为和偏好，**会话推荐**更适用于用户匿名或缺乏长期历史信息的场景，如电商平台中的一次性购物或媒体流服务中的即时内容推荐。这类问题的核心在于如何根据用户在单次会话中的物品点击序列，推断出下一步最有可能点击的物品。

传统上，许多研究工作使用**马尔可夫链**（Markov Chain，MC）和**递归神经网络**（Recurrent Neural Networks，RNN）来建模会话中的顺序依赖关系。马尔可夫链假设下一个点击行为仅取决于前一个点击行为，但忽略了全局会话信息的影响。**GRU4Rec** 是基于门控循环单元（GRU）的模型，它通过捕捉用户在当前会话中的短期兴趣，在会话推荐任务中取得了显著成果【Hidasi et al., 2016】。**NARM** 则进一步提升了 GRU4Rec 的表现，通过引入全局和局部兴趣来捕捉用户的主要意图【Li et al., 2017】。

然而，这些基于 RNN 的方法主要关注物品序列的单向依赖，无法有效建模会话中的复杂交互关系。此外，RNN 通常无法捕捉长距离物品间的依赖关系，尤其是在会话较长或点击序列较为复杂的情况下。

为解决上述问题，近年来的研究引入了**图神经网络（GNN）**，例如**SR-GNN**，它将会话中的物品序列建模为一个有向图，通过 GNN 捕捉物品之间的复杂转移模式【Wu et al., 2019】。GNN 在建模局部依赖方面表现出色，能够通过图的结构学习物品之间的多跳关系，进而提升推荐准确性。然而，GNN 也有其局限性，即在处理长距离依赖时存在困难，因为它需要通过多层网络传播信息，导致信息传递效率下降。

在此基础上，**GC-SAN（Graph Contextualized Self-Attention Network）** 结合了**图神经网络**和**自注意力机制（Self-Attention）**，弥补了各自的不足。GNN 用于捕捉会话中的局部依赖关系，自注意力机制则有效地处理全局依赖。通过结合这两者，GC-SAN 能够在保持局部信息学习的同时，更好地捕捉会话中的全局兴趣，提升推荐效果。

这种结合 GNN 和自注意力机制的创新架构使 GC-SAN 能够在建模短期和长期兴趣时更加灵活，从而在实际应用中表现优越，尤其适用于电商和流媒体等场景中不断变化的用户行为分析。

[论文原文链接](https://www.ijcai.org/proceedings/2019/0547.pdf)

# 具体细节

## 问题定义

GC-SAN 旨在解决会话推荐问题。具体来说，在给定用户的当前会话序列的基础上，预测用户可能点击的下一个物品。假设 $V=\{v_1, v_2, \dots, v_{\mid V \mid}\}$ 代表所有物品集合，每个会话序列 $S=\{s_1, s_2, \dots, s_n\}$ 是按时间顺序排列的物品序列。目标是预测序列的下一个点击物品 $s_{t+1}$。

## 动态图结构的构建

对于每个会话序列 $S=\{s_1, s_2, \dots, s_n\}$，我们将物品 $s_i$ 作为图中的节点，点击顺序中的物品对 $(s_{i-1}, s_i)$ 作为有向边，构建一个会话图 $G_s$。图中使用入度矩阵 $M^I$ 和出度矩阵 $M^O$ 来表示会话中的物品关系。我们使用 GNN（图神经网络）在会话图上进行信息传播，生成物品的嵌入向量。

![](https://pic.imgdb.cn/item/66dbdbbad9c307b7e983ae70.png)

## 节点向量的更新

GNN 模型通过以下公式更新每个节点（物品）的向量：

$$
\mathbf{a}_t = \operatorname{Concat} \left( M_t^I \left(\mathbf{s}_1, \dots, \mathbf{s}_n\right) W_a^I + b^I, M_t^O \left(\mathbf{s}_1, \dots, \mathbf{s}_n\right) W_a^O + b^O \right)
$$

- 这里，$\mathbf{s}_1, \dots, \mathbf{s}_n$ 表示会话中所有物品的嵌入向量。
- $M_t^I$ 和 $M_t^O$ 分别是节点 $s_t$ 的入度和出度矩阵的第 $t$ 行，表示物品 $s_t$ 的邻居信息。
- $W_a^I$ 和 $W_a^O$ 是权重矩阵，分别用于入度和出度信息的线性变换。
- $b^I$ 和 $b^O$ 是偏置项。

通过这个公式，模型提取物品之间的局部依赖关系。随后通过门控机制进一步处理这些信息：
$$
\mathbf{z}_t = \sigma \left( W_z \mathbf{a}_t + P_z \mathbf{s}_{t-1} \right)
$$

$$
\mathbf{r}_t = \sigma \left( W_r \mathbf{a}_t + P_r \mathbf{s}_{t-1} \right)
$$

更新门决定了新的信息应该保留多少。$W_z$ 是权重矩阵，$P_z$ 是先前时间步 $t-1$ 的物品嵌入 $\mathbf{s}_{t-1}$ 对当前节点的影响。$\sigma$ 是 sigmoid 函数，用于将结果规范化到 [0, 1] 之间。重置门控制之前的隐藏状态信息在多大程度上被忽略或重置。类似于更新门，$W_r$ 和 $P_r$ 是权重矩阵和历史信息的影响系数。

最终的物品嵌入向量通过以下公式更新：
$$
\tilde{\mathbf{h}}_t = \tanh \left( W_h \mathbf{a}_t + P_h \left( \mathbf{r}_t \odot \mathbf{s}_{t-1} \right) \right)
$$

$\mathbf{r}_t$ 控制历史信息的影响，$\odot$ 表示逐元素相乘操作。$W_h$ 是当前输入的权重矩阵，$P_h$ 是历史状态的权重矩阵。$\tanh$ 函数用于生成非线性的候选隐藏状态 $\tilde{\mathbf{h}}_t$。
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{s}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

当 $\mathbf{z}_t$ 值接近 1 时，模型保留更多的候选状态信息；

当 $\mathbf{z}_t$ 值接近 0 时，模型则更多地依赖于先前的状态 $\mathbf{s}_{t-1}$。

## 自注意力层

在将会话序列传入图神经网络（GNN）之后，模型会获得该会话图中所有节点（物品）的潜在向量，表示为：
$$
\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n]
$$
其中，$\mathbf{h}_i$ 代表会话中第 $i$ 个物品的嵌入向量。

接下来，模型通过**自注意力机制**来捕捉这些物品之间的全局依赖关系。自注意力机制的计算公式为：
$$
\mathbf{F} = \operatorname{softmax} \left( \frac{ (\mathbf{H} \mathbf{W}^Q)(\mathbf{H} \mathbf{W}^K)^{\top} }{\sqrt{d}} \right) (\mathbf{H} \mathbf{W}^V)
$$
- **$\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$** 是三个投影矩阵，将输入嵌入 $\mathbf{H}$ 映射到查询、键和值空间。
- **$\frac{1}{\sqrt{d}}$** 是缩放因子，防止数值过大导致梯度消失。

通过这个公式，模型计算每个物品与其他物品之间的注意力权重，生成一个加权表示 $\mathbf{F}$，捕捉用户当前会话的全局偏好。

## 点乘前馈网络（Point-Wise Feed-Forward Network）

为了增加模型的非线性，GC-SAN 在自注意力操作后使用**点乘前馈网络**。该网络的计算如下：
$$
\mathbf{E} = \operatorname{ReLU}(\mathbf{F} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2 + \mathbf{F}
$$
- **$\mathbf{W}_1, \mathbf{W}_2$** 是两个 $d \times d$ 的权重矩阵，**$\mathbf{b}_1, \mathbf{b}_2$** 是偏置项。
- **ReLU** 激活函数引入非线性，增加模型的表达能力。
- 后面加的 $\mathbf{F}$ 是残差项

此外，模型在训练过程中还使用了 "Dropout" 正则化技术来防止过拟合。

## 多层自注意力

为了捕捉更加复杂的特征，GC-SAN 使用了多层的自注意力机制。第一层自注意力的输出为：
$$
\mathbf{E}^{(1)} = \mathbf{E}
$$
对于第 $k$ 层（$k > 1$），定义如下：
$$
\mathbf{E}^{(k)} = \operatorname{SAN}(\mathbf{E}^{(k-1)})
$$
最终输出 $\mathbf{E}^{(k)} \in \mathbb{R}^{n \times d}$ 表示经过多层自注意力机制后的会话序列表示。

## 预测与模型训练

经过多层自注意力操作后，模型生成了会话的长期自注意力表示 $\mathbf{E}^{(k)}$。为了更好地预测用户的下一个点击物品，GC-SAN 将会话的全局偏好与当前兴趣进行加权组合：

$$
\mathbf{S}_f = \omega \mathbf{E}_n^{(k)} + (1 - \omega) \mathbf{h}_n
$$

其中，$\mathbf{h}_n$ 是最后点击物品的嵌入，$\omega$ 是权重系数。

最终，模型计算每个候选物品的推荐分数，生成概率分布：

$$
\hat{\mathbf{y}}_i = \operatorname{softmax} \left( \mathbf{S}_f^{\top} \mathbf{v}_i \right)
$$

模型通过最小化以下损失函数来训练：

$$
\mathcal{J} = - \sum_{i=1}^{n} \mathbf{y}_i \log \left( \hat{\mathbf{y}}_i \right) + (1 - \mathbf{y}_i) \log \left( 1 - \hat{\mathbf{y}}_i \right) + \lambda \|\theta\|^2
$$

其中，$\mathbf{y}_i$ 是目标物品的 one-hot 编码，$\theta$ 是所有可学习参数的集合。

# 代码实现

下面提供简化版的GCSAN的代码讲解：

```python
import torch
from torch import nn
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole_gnn.model.layers import SRGNNCell  # 引入SRGNN单元用于GNN操作

class GCSAN(SequentialRecommender):
    """
    GCSAN模型结合了图神经网络（GNN）和自注意力机制（Self-Attention）来进行会话推荐。
    GNN用于捕捉局部依赖，自注意力机制用于捕捉全局依赖。
    """

    def __init__(self, config, dataset):
        super(GCSAN, self).__init__(config, dataset)

        # 从config文件加载模型参数
        self.n_layers = config['n_layers']  # 自注意力层的层数
        self.n_heads = config['n_heads']  # 注意力头的数量
        self.hidden_size = config['hidden_size']  # 隐藏层大小
        self.inner_size = config['inner_size']  # Feed-forward层大小
        self.hidden_dropout_prob = config['hidden_dropout_prob']  # 隐藏层dropout
        self.attn_dropout_prob = config['attn_dropout_prob']  # 自注意力dropout
        self.step = config['step']  # GNN传播的步数
        self.weight = config['weight']  # 局部与全局表示的加权参数

        # 定义嵌入层和GNN单元
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.gnncell = SRGNNCell(self.hidden_size)  # GNN单元

        # Transformer自注意力层
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob
        )

        # 损失函数
        self.loss_fct = BPRLoss()  # 使用BPR损失函数（用于推荐任务）

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        """
        前向传播：
        1. 通过GNN捕捉局部依赖。
        2. 使用Transformer自注意力层捕捉全局依赖。
        3. 最终通过加权结合局部和全局的表示来预测下一个点击物品。
        """
        hidden = self.item_embedding(x)  # 获取物品嵌入

        # 通过GNN单元进行局部信息传播
        for i in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        # 获取GNN输出
        seq_hidden = hidden[alias_inputs]
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)  # 获取序列最后一个物品的隐藏状态

        # 自注意力层捕捉全局依赖
        outputs = self.self_attention(seq_hidden, output_all_encoded_layers=True)
        output = outputs[-1]
        at = self.gather_indexes(output, item_seq_len - 1)

        # 将局部和全局信息加权结合
        seq_output = self.weight * at + (1 - self.weight) * ht
        return seq_output

    def calculate_loss(self, interaction):
        """
        计算损失函数，包括BPR损失和正则化损失。
        """
        x = interaction['x']
        edge_index = interaction['edge_index']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(x, edge_index, alias_inputs, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        # 计算BPR损失
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
        loss = self.loss_fct(pos_score, neg_score)

        return loss
```

BPR（Bayesian Personalized Ranking）损失函数是一种广泛用于推荐系统中的排序损失函数，特别是用于处理隐式反馈问题，如点击行为、点赞、浏览等没有明确评分的场景。BPR 损失的目标是最大化用户对正样本（例如用户点击过的物品）和负样本（用户未点击的物品）之间的评分差距。通过比较用户对正样本和负样本的偏好，BPR 函数希望模型学会排序，而不是明确预测每个物品的具体评分。

BPR 的损失函数可以定义为：

$$
\mathcal{L}_{BPR} = -\sum_{(u, i, j)} \ln \, \sigma (\hat{x}_{ui} - \hat{x}_{uj}) + \lambda \, \| \Theta \|^2
$$

其中：
- $ (u, i, j) $ 表示用户 $ u $ 喜欢物品 $ i $，不喜欢物品 $ j $；
- $ \hat{x}_{ui} $ 和 $ \hat{x}_{uj} $ 分别是模型预测用户对物品 $ i $ 和物品 $ j $ 的偏好得分；
- $ \sigma $ 是 Sigmoid 函数，用于将偏好差值转换为概率；
- $ \lambda \, \| \Theta \|^2 $ 是正则化项，用于防止模型过拟合。

BPR 假设用户 $ u $ 对物品 $ i $ 的兴趣比对物品 $ j $ 更高（即用户更喜欢物品 $ i $ 而不是物品 $ j $）。通过最小化负对数似然函数，BPR 让模型学习用户对物品的相对偏好，而不是绝对评分。

BPR 损失函数非常适合处理隐式反馈数据（如购买记录、点击行为等），而不是显式评分。它在个性化推荐中被广泛使用，因为它直接优化推荐排序，而不仅仅是预测分数。







