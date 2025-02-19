---
title: Integrating Large Language Models with Graphical Session-Based Recommendation
tags: 
  - GNN
  - Recommend System
categories: Paper
date: 2024-08-13 9:00:00
mathjax: true
---

在现代推荐系统领域，如何根据用户的短期行为来预测其未来的点击行为是一个重要问题。传统的推荐系统多依赖于用户的长期偏好数据，而在实际应用中，用户的会话（session）行为也往往能很好地反映用户的即时需求。为了解决这一问题，基于会话的推荐系统（Session-based Recommendation, SBR）逐渐兴起。本文介绍的**SR-GNN（Session-based Recurrent Graph Neural Network）** 是一种将图神经网络（GNN）引入到会话推荐中的方法，能够有效捕捉用户在单个会话中的短期行为，从而预测用户的下一个点击行为。SR-GNN 在会话推荐系统中引入了图神经网络，能够有效捕捉物品之间的复杂依赖关系，并结合自注意力机制和门控循环单元（GRU）建模时间顺序。这使得 SR-GNN 能够在短期会话中进行准确的推荐，尤其适用于动态变化的用户行为场景。

<!-- more -->

# 背景介绍

在会话推荐系统中，用户的点击序列是短期行为的体现，并且我们只使用当前会话中的点击数据来预测用户下一个可能点击的物品。在这种背景下，我们无法依赖于用户的长期历史偏好，而是基于当前会话内的物品之间的关系来进行推荐。 会话推荐问题可以表述为：给定一个物品集合 $V = \{v_1, v_2, \ldots, v_m\}$ ，以及一个会话序列 $s = [v_{s,1}, v_{s,2}, \ldots, v_{s,n}]$ ，该序列按照时间顺序排列，其中 $v_{s,i} \in V$ 表示用户在当前会话中的第 $i$ 次点击。我们的目标是预测用户在当前会话中的下一个点击物品 $v_{s,n+1}$。 SR-GNN 的输出是一个物品得分向量 $\hat{\mathbf{y}}$ ，其中每个元素值表示对应物品的推荐分数。得分最高的 $K$ 个物品将作为推荐结果提供给用户。

[论文原文链接](https://arxiv.org/pdf/1811.00855)

# 具体细节

## 会话图的构建

为了更好地捕捉会话内物品之间的复杂关系，SR-GNN 将会话数据转换为图结构。对于每一个会话序列 $s$ ，可以建模为一个有向图 $\mathcal{G}_s = (\mathcal{V}_s, \mathcal{E}_s)$ ，其中：
- **节点**：表示用户在会话中的每一个点击物品；
- **边**：表示物品之间的点击顺序。

例如，用户在会话中依次点击了物品 $v_1 \rightarrow v_2 \rightarrow v_3 \rightarrow v_2 \rightarrow v_4$ ，则我们可以构建如图所示的会话图。对于重复出现的物品（如 $v_2$ ），我们为其构建的边分配一个归一化的权重，权重为该边的出现次数除以该边起始节点的出度。

## 物品嵌入的学习

在构建好会话图之后，SR-GNN 使用**图神经网络（GNN）**来学习物品嵌入。GNN的优势在于能够在图结构上进行信息的传播和聚合，从而捕捉到图中节点（物品）之间的复杂关系。具体来说，每个节点的嵌入 $\mathbf{v}_{s,i}$ 的更新通过以下公式计算：

$$
\mathbf{a}_{s, i}^{t} = \mathbf{A}_{s, i:}\left[\mathbf{v}_{1}^{t-1}, \mathbf{v}_{2}^{t-1}, \ldots, \mathbf{v}_{n}^{t-1}\right]^{\top} \mathbf{H} + \mathbf{b}
$$
该公式表示在图神经网络中的信息传播过程。会话图中的每个节点（物品）不仅仅依赖于自身的信息，它还会从它的邻居节点（即与之有边相连的物品）获取信息，并通过加权求和来更新其自身状态。公式中的关键组成部分包括：

- **$\mathbf{A}_{s, i:}$**：会话图的邻接矩阵。它确定了节点之间的连接方式。对于一个节点 $i$ 来说，这个矩阵决定了它可以从哪些其他节点获取信息。
- **$\left[\mathbf{v}_{1}^{t-1}, \mathbf{v}_{2}^{t-1}, \ldots, \mathbf{v}_{n}^{t-1}\right]$**：表示会话图中所有节点的前一时间步的嵌入向量。
- **$\mathbf{H}$**：一个权重矩阵，控制着如何结合这些信息。
- **$\mathbf{b}$**：偏置项，用于调整输出。

$$
\mathbf{z}_{s, i}^{t} = \sigma\left(\mathbf{W}_{z} \mathbf{a}_{s, i}^{t} + \mathbf{U}_{z} \mathbf{v}_{i}^{t-1}\right)
$$
$$
\mathbf{r}_{s, i}^{t} = \sigma\left(\mathbf{W}_{r} \mathbf{a}_{s, i}^{t} + \mathbf{U}_{r} \mathbf{v}_{i}^{t-1}\right)
$$
$$
\widetilde{\mathbf{v}}_i^t = \tanh \left(\mathbf{W}_o \mathbf{a}_{s, i}^{t} + \mathbf{U}_o \left(\mathbf{r}_{s,i}^t \odot \mathbf{v}_{i}^{t-1}\right)\right)
$$
$$
\mathbf{v}_i^t = \left(1 - \mathbf{z}_{s,i}^t\right) \odot \mathbf{v}_{i}^{t-1} + \mathbf{z}_{s,i}^t \odot \widetilde{\mathbf{v}}_i^t
$$

经过多轮迭代后，节点的最终嵌入 $\mathbf{v}_i^t$ 能够有效捕捉图中物品之间的依赖关系。

## 生成会话表示

在每个会话图中的物品节点嵌入学习完成后，SR-GNN 生成整个会话的表示。这是通过结合**局部嵌入**和**全局嵌入**完成的：

- **局部嵌入**：直接使用最后一个点击物品的嵌入 $\mathbf{v}_n$ 来表示当前用户的短期兴趣。
  $$
  \mathbf{s}_1 = \mathbf{v}_n
  $$
  
- **全局嵌入**：通过自注意力机制将会话中的所有物品嵌入聚合起来，捕捉用户的长时兴趣。
  $$
  \alpha_i = \mathbf{q}^{\top} \sigma\left(\mathbf{W}_1 \mathbf{v}_n + \mathbf{W}_2 \mathbf{v}_i + \mathbf{c}\right)
  $$
  $$
  \mathbf{s}_g = \sum_{i=1}^{n} \alpha_i \mathbf{v}_i
  $$
  
  * **$\mathbf{q}$** 是一个全局向量，它的作用是提供一个权重机制，用来衡量不同物品（节点）的重要性。这个全局向量是通过训练学习到的，它帮助模型对当前会话中的每个物品节点进行一个权重分配。$q$ 的值越大，表明这个物品在整个会话中越重要。
  * **$\mathbf{v}_n$** 表示会话中的**最后一个物品的嵌入**。为什么用最后一个物品呢？因为在很多推荐场景中，用户的最后一个点击行为往往反映了用户最当前的兴趣。最后一个物品的嵌入 $\mathbf{v}_n$ 是一个很重要的信号，它能代表用户对某类物品的偏好。
  * **$\mathbf{v}_i$** 则表示会话中的第 $i$ 个物品的嵌入。这个物品是当前会话中可能存在的某个物品。
  * **$\mathbf{W}_1$ 和 $\mathbf{W}_2$** 是两个权重矩阵，用来将最后一个物品和当前物品的嵌入向量映射到一个新的空间中。通过这两个权重矩阵，模型能够比较当前物品 $v_i$ 与会话中最后一个物品 $v_n$ 之间的相似性。这样可以让模型捕捉到用户在会话中是如何从一个物品逐步过渡到最后一个物品的兴趣变化。
  * 如果某个物品 $v_i$ 和最后一个物品 $v_n$ 的相关性很高，那么它的权重 $\alpha_i$ 会更大，表示这个物品对当前会话的整体偏好影响更大。
  
- **最终嵌入**：通过将局部嵌入 $\mathbf{s}_1$ 和全局嵌入 $\mathbf{s}_g$ 进行线性组合，生成最终的会话表示。
  $$
  \mathbf{s}_h = \mathbf{W}_3 \left[\mathbf{s}_1; \mathbf{s}_g\right]
  $$

## 预测与模型训练

在会话嵌入生成之后，SR-GNN 通过计算每个候选物品的得分 $\hat{\mathbf{z}}_i$ 来进行预测：

$$
\hat{\mathbf{z}}_i = f\left(\mathbf{s}_h, \mathbf{v}_i\right)
$$

模型使用交叉熵损失函数进行训练：

$$
L = -\sum_{i} y_i \log(\hat{y}_i)
$$

其中，$y_i$ 是实际的标签，$\hat{y}_i$ 是模型预测的概率。

在 SR-GNN 模型中，公式 $\hat{\mathbf{z}_{i}}=\mathbf{s}_{\mathrm{h}}^{\top} \mathbf{v}_{i}$ 用于计算每个候选物品 $v_i$ 的推荐得分。它使用会话的混合嵌入表示 $\mathbf{s}_{\mathrm{h}}$ 和候选物品的嵌入向量 $\mathbf{v}_{i}$ 进行内积，得到该物品作为下一个点击物品的分数。

然后，模型将计算得到的分数通过 Softmax 函数转化为概率分布 $\hat{\mathbf{y}}$，表示每个物品成为下一个点击物品的概率：

$$
\hat{\mathbf{y}} = \operatorname{softmax}(\hat{\mathbf{z}})
$$

其中，$\hat{\mathbf{z}} \in \mathbb{R}^{m}$ 是所有候选物品的推荐得分，而 $\hat{\mathbf{y}} \in \mathbb{R}^{m}$ 表示候选物品成为下一个点击物品的概率。

## 损失函数的定义

为了训练模型，使用交叉熵损失函数来衡量模型的预测结果与实际点击物品之间的差异。损失函数的形式如下：

$$
\mathcal{L}(\hat{\mathbf{y}}) = -\sum_{i=1}^{m} \mathbf{y}_{i} \log(\hat{\mathbf{y}}_{i}) + (1 - \mathbf{y}_{i}) \log(1 - \hat{\mathbf{y}}_{i})
$$

其中，$\mathbf{y}_i$ 是真实标签的 one-hot 编码表示，即只有一个元素为 1，其他元素为 0，代表用户实际点击的物品。

## 模型训练

训练过程中，采用反向传播算法（Back-Propagation Through Time, BPTT）来更新模型参数。在会话推荐任务中，大部分会话的长度相对较短，因此建议选择较小的训练步数，以防止过拟合。

这个过程通过不断地调整模型参数，使模型逐步学会捕捉用户的行为模式，从而在新的会话中为用户推荐最有可能点击的物品。

# 代码示例

模型的实现源代码在 https://github.com/CRIPAC-DIG/SR-GNN/tree/master，下面我将提供一个简化版本的代码进行讲解。

## 类定义与初始化

```python
class SimplifiedSRGNN:
    def __init__(self, n_items, hidden_size=100, batch_size=100):
        self.n_items = n_items  # 物品数量，所有会话中涉及到的唯一物品的总数
        self.hidden_size = hidden_size  # 隐藏层的大小，控制嵌入向量的维度
        self.batch_size = batch_size  # 批处理的大小，控制一次处理的数据量

        # 创建物品嵌入矩阵，形状为 [n_items, hidden_size]，表示每个物品的嵌入向量
        self.embedding = tf.get_variable('embedding', shape=[n_items, hidden_size],
                                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
        
        # 定义两个占位符，用于接收会话中的邻接矩阵
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None])  # 入度邻接矩阵
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None])  # 出度邻接矩阵
        self.item = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])  # 物品序列，表示每个会话中点击的物品序列

        # 初始化图神经网络中使用的参数，包括权重矩阵
        self.W_in = tf.get_variable('W_in', shape=[hidden_size, hidden_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.W_out = tf.get_variable('W_out', shape=[hidden_size, hidden_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-0.1, 0.1))
```

- **物品嵌入矩阵**：模型使用 `embedding` 变量表示物品的嵌入向量。每个物品都对应一个向量，大小为 `hidden_size`，这些向量是通过训练来更新的。
- **邻接矩阵**：`adj_in` 和 `adj_out` 是占位符，用于存储会话图的入度和出度邻接矩阵。这些矩阵用于信息传播，帮助模型了解物品之间的点击顺序。
- **权重矩阵**：`W_in` 和 `W_out` 是两个权重矩阵，分别用于对入度和出度邻接矩阵中的物品嵌入进行变换。每个矩阵的大小与 `hidden_size` 相同，用于调整信息传播的权重。

## 图神经网络中的信息传播

```python
def gnn_propagation(self):
        # 获取物品嵌入向量，shape为(batch_size, T, hidden_size)
        item_embeddings = tf.nn.embedding_lookup(self.embedding, self.item)

        # 使用GRU单元，用于捕捉序列中的时间依赖关系
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        # 信息传播步骤（这里简化为仅传播一次）
        for _ in range(1):  # GNN中的传播步数
            item_embeddings_in = tf.matmul(item_embeddings, self.W_in)  # 入度邻接矩阵的变换
            item_embeddings_out = tf.matmul(item_embeddings, self.W_out)  # 出度邻接矩阵的变换

            # 通过邻接矩阵将物品嵌入与邻居节点的嵌入聚合
            in_aggregated = tf.matmul(self.adj_in, item_embeddings_in)
            out_aggregated = tf.matmul(self.adj_out, item_embeddings_out)

            # 将入度和出度信息拼接作为GRU的输入
            aggregated = tf.concat([in_aggregated, out_aggregated], axis=-1)

            # 使用GRU单元更新节点状态
            _, final_state = tf.nn.dynamic_rnn(cell, aggregated, dtype=tf.float32)

        return final_state
```

- **物品嵌入**：通过 `tf.nn.embedding_lookup`，我们从嵌入矩阵中获取当前批次中物品的嵌入向量，形状为 `(batch_size, T, hidden_size)`，其中 `T` 是会话中的物品序列长度。
- **信息传播**：我们通过 `adj_in` 和 `adj_out` 进行入度和出度邻接矩阵的乘法操作，来更新每个节点的嵌入。这里的操作模拟了会话中的物品点击顺序对信息传播的影响。
- **GRU 单元**：`GRU` 是一种循环神经网络单元，用来捕捉序列中的时间依赖性。我们将聚合后的物品嵌入传入 GRU 中，最终得到 `final_state`，它表示了当前批次中物品序列的状态。

## 训练过程

```python
def train(self):
        # GNN传播后计算物品的推荐得分
        logits = tf.matmul(self.gnn_propagation(), self.embedding, transpose_b=True)

        # 目标物品
        targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        # 交叉熵损失函数
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

        # 使用Adam优化器
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        return loss, optimizer
```

- **得分计算**：我们通过 GNN 得到的物品序列的状态 `final_state` 和物品嵌入矩阵做内积运算，得到每个物品的推荐得分 `logits`。
- **交叉熵损失函数**：我们使用 `tf.nn.sparse_softmax_cross_entropy_with_logits` 来计算目标物品的损失，这个损失衡量了模型对下一个物品预测的准确性。
- **优化器**：使用 `Adam` 优化器对模型进行优化，目的是通过最小化损失函数，逐步更新模型参数，提高预测准确性。

## 训练循环

```python
def train_model(n_items, epochs=10):
    model = SimplifiedSRGNN(n_items)

    # 创建会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            # 模拟生成一批数据，包括物品序列和邻接矩阵
            adj_in_batch = np.random.rand(model.batch_size, 10, 10)
            adj_out_batch = np.random.rand(model.batch_size, 10, 10)
            item_batch = np.random.randint(0, n_items, (model.batch_size, 10))
            target_batch = np.random.randint(0, n_items, model.batch_size)

            # 执行训练步骤
            loss, optimizer = model.train()
            feed_dict = {
                model.adj_in: adj_in_batch,
                model.adj_out: adj_out_batch,
                model.item: item_batch,
                model.tar: target_batch
            }
            _, train_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

            print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')
```

- **生成模拟数据**：我们为每个训练批次生成随机的邻接矩阵（`adj_in_batch` 和 `adj_out_batch`）和物品序列（`item_batch`）。
- **执行训练**：在每个训练批次中，我们运行模型的优化器来最小化损失函数，并输出当前批次的损失值。
- **会话管理**：在 TensorFlow 中，通过 `tf.Session()` 来执行计算图，并使用 `sess.run()` 来实际执行模型的计算和优化操作。
