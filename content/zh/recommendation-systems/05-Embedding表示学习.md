---
title: "推荐系统（五）—— Embedding表示学习"
date: 2025-12-13 09:00:00
tags:
  - Recommendation Systems
  - Embedding
  - Representation Learning
categories: 推荐系统
series: recommendation-systems
lang: zh
mathjax: true
description: "推荐系统 Embedding 技术全解：从 Word2Vec、Item2Vec、Node2Vec 到双塔 DSSM 与 YouTube DNN，再到负采样策略与 FAISS/HNSW 近邻检索的工程实践。每节配有可运行的 PyTorch 代码、关键设计权衡与生产经验。"
disableNunjucks: true
series_order: 5
translationKey: "recommendation-systems-5"
---
当 Netflix 向刚看完《蝙蝠侠：黑暗骑士》的用户推荐《盗梦空间》时，这背后的逻辑并不是一条手写的“如果看了诺兰的电影就推荐诺兰”的规则，而是几何学在发挥作用。这两部电影在模型从数十亿次观看行为中学到的 128 维 **嵌入空间** 中距离很近。几何取代了枚举：系统不再通过脆弱的相似性规则逐一比较一部电影与一万五千部其他电影，而是只问一个问题——**这两个向量之间的距离是多少？**

这篇文章拆解了这些向量是如何被学习出来并以生产规模提供服务的。我会从底层直觉出发，依次讲解五大类技术（序列、图、双塔、注意力池化、对比学习），深入负采样的工程细节，并剖析毫秒级延迟下近似最近邻检索的真实性能边界。每一节都配有可以直接运行的代码。

## 你将学到什么

- **Embedding 到底是什么**——为什么“低维”是它的核心价值
- **序列学习方法**：Word2Vec 和 Item2Vec，包括 Skip-gram 的完整推导
- **图学习方法**：Node2Vec 的有偏随机游走机制
- **双塔架构**：DSSM、YouTube DNN，工业界召回阶段的主流选择
- **负采样策略**：均匀采样、按热度采样、Hard Negative、In-batch Negative
- **ANN 在线服务**：FAISS、HNSW、Annoy 的延迟与召回率权衡
- **评估方法**：内在指标（coherence、聚类）和外在指标（HR@K、NDCG）
## 前置知识

- 线性代数基础：向量、点积、矩阵乘法
- 熟悉神经网络和 PyTorch
- 推荐系统基本概念，参考 [第一篇](/zh/recommendation-systems/01-入门与基础概念/)
- 深度学习推荐相关知识有帮助但非必须，参考 [第三篇](/zh/recommendation-systems/03-深度学习基础模型/)

---
## 1. Embedding 基础：是什么，为什么重要

### 1.1 压缩视角

**Embedding 是一个学习到的函数** $f : \mathcal{I} \to \mathbb{R}^{d}$，它将离散对象——用户、电影、SKU 或图中的节点——映射到一个 $d$ 维实数向量。这里的 $d$ 远小于物品总数。

> **类比**：想象一个 Netflix 风格的百万级影片库。最简单的表示方法是为每部电影分配一个百万维的 one-hot 向量。而 Embedding 把这个复杂度降到了 128 个数字——一组模型从数据中挖掘出的潜在属性：比如影片的烧脑程度、暴力指数、视觉暗度，或者是否带有 90 年代风格。

Embedding 的三个特性让它成为现代推荐系统的通用语言：

| 特性 | 价值 |
|---|---|
| **稠密性** | 每一维都携带信息，没有稀疏向量那种浪费空间的问题 |
| **几何性** | "相似"变成了可计算的物理量——内积或余弦值 |
| **可组合性** | 可以求平均、拼接、做注意力机制，还能用于检索 |

### 1.2 为什么稀疏表示行不通

真实的交互矩阵稀疏得惊人。一个拥有 $10^{8}$ 用户和 $10^{7}$ 物品的平台，理论上可能有 $10^{15}$ 个单元格，但实际观测到的非零项通常不到 $10^{11}$——密度低于万分之一。直接存储或分解这种矩阵完全不可行。把每行每列压缩成 $d$ 维稠密向量后，问题就变成了现代硬件擅长处理的稠密矩阵乘法。

![物品 Embedding 空间（t-SNE 投影），按类目聚类，标出 query 物品及其 k-NN 区域](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig1_embedding_space.png)

这张图就是本文的核心直觉。同类物品在向量空间中形成簇，query 物品的近邻正是推荐的目标。**推荐问题被简化为 Embedding 空间中的最近邻搜索**。

### 1.3 学习目标一句话总结

从矩阵分解到 BERT4Rec，几乎所有 Embedding 方法都在优化同一个核心思想：

> **出现在相似上下文中的物品，应该有相似的向量；不相似的则需要被推开。**

这就是从语言学借来的[分布假设](https://en.wikipedia.org/wiki/Distributional_semantics)——"观其同游而知其义"。在推荐场景中，"上下文"可以是：

- 用户会话中的共现（Item2Vec）
- 图中的邻接关系（Node2Vec、GraphSAGE）
- 同一个 query 下的点击（DSSM）
- CTR 日志中的正样本（DeepFM、DLRM）

上下文的定义决定了方法的具体形式。
## 2. 基于序列的 Embedding：Word2Vec 和 Item2Vec

### 2.1 从词到物品

Word2Vec（Mikolov et al., 2013）有两种模式：

- **Skip-gram**：给定中心词，预测周围的上下文词。
- **CBOW（连续词袋模型）**：给定上下文词，预测中心词。

> **类比**：Skip-gram 就像给你一块拼图，让你猜它周围是什么。CBOW 则反过来，给你周围的拼图，让你猜中间那块。

Item2Vec（Barkan & Koenigstein, 2016）的迁移看似简单，但非常强大：**把用户的行为序列当作句子，每个物品当作一个词**。Word2Vec 的整套机制直接复用，毫无改动。

### 2.2 Skip-gram 目标函数推导

给定序列 $S = [i_1, i_2, \dots, i_T]$ 和窗口大小 $c$，Skip-gram 的目标是最大化每个上下文物品在中心物品条件下的对数概率：

$$\mathcal{L} \;=\; \sum_{t=1}^{T} \;\sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p\!\left(i_{t+j}\,\middle|\,i_t\right).$$

最朴素的概率定义是对整个物品集做 softmax：

$$p\!\left(i_{t+j}\,\middle|\,i_t\right) \;=\; \frac{\exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_{t+j}}\right)}{\displaystyle\sum_{k=1}^{|\mathcal{I}|} \exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{k}\right)}.$$

这里 $\mathbf{e}_i$ 是**输入（中心）嵌入**，$\mathbf{e}'_i$ 是**输出（上下文）嵌入**。每个物品有两套向量——第一次见到会觉得奇怪，但这给了模型更多表达自由。

在百万级物品上计算分母显然不现实。**负采样**将其转化为二分类问题：对每个真实的（中心，上下文）正样本对，随机采 $K$ 个"噪声"物品，让模型区分正负样本。目标函数变为：

$$\mathcal{L} \;=\; \sum_{(i_t,\,i_c)} \!\left[\, \log\sigma(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_c})
\;+\; \sum_{k=1}^{K} \mathbb{E}_{i_k \sim P_n}\!\big[\log\sigma(-\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_k})\big]\right],$$

其中 $\sigma$ 是 sigmoid 函数，$P_n$ 是 unigram 频率的 $3/4$ 次幂——这是 Mikolov 的经典经验：纯频率会过度采样热门物品，纯均匀会过度采样长尾物品，0.75 是实证的最佳平衡点。

![Item2Vec Skip-gram 架构：滑动窗口选出中心物品，Embedding 查找得到向量，模型将正样本上下文与按 3/4 次幂分布采样的 K 个负样本对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig2_item2vec_skipgram.png)

### 2.3 完整实现

下面是一个完整的 Item2Vec 实现（Skip-gram + 负采样）。重点关注三处设计：**两套 Embedding 表**、**得分裁剪**、**sigmoid 数值技巧**。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

class Item2Vec(nn.Module):
    """Skip-gram + 负采样

    两套 Embedding：
      - input_embeddings：推理时使用，是物品的"真实"向量
      - output_embeddings：训练辅助，训练完丢弃
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_negatives: int = 5):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_negatives = num_negatives

        # Xavier 初始化让初始内积处于较小范围，
        # 避免第一轮迭代时 sigmoid 已经饱和。
        nn.init.xavier_uniform_(self.input_embeddings.weight)
        nn.init.xavier_uniform_(self.output_embeddings.weight)

    def forward(self, target, context, negatives):
        """
        target    : (B,)          中心物品 id
        context   : (B,)          真实上下文 id
        negatives : (B, K)        每行 K 个随机负样本
        """
        t = self.input_embeddings(target)      # (B, d)
        c = self.output_embeddings(context)    # (B, d)
        n = self.output_embeddings(negatives)  # (B, K, d)

        # 正样本：拉近中心和真实上下文的距离
        pos = torch.clamp((t * c).sum(-1), -10, 10)
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-10)

        # 负样本：推开中心和随机物品的距离
        neg = torch.clamp(torch.bmm(n, t.unsqueeze(-1)).squeeze(-1), -10, 10)
        neg_loss = -torch.log(torch.sigmoid(-neg) + 1e-10).sum(-1)

        return (pos_loss + neg_loss).mean()

    @torch.no_grad()
    def vector(self, item_id: int) -> np.ndarray:
        return self.input_embeddings.weight[item_id].cpu().numpy()

# --- 数据流水线 ----------------------------------------------------------- #

def build_vocab(sequences):
    counter = Counter(item for seq in sequences for item in seq)
    vocab = {item: idx for idx, item in enumerate(counter)}
    return vocab, counter

def make_pairs(sequences, vocab, window=5):
    pairs = []
    for seq in sequences:
        ids = [vocab[i] for i in seq if i in vocab]
        for t, centre in enumerate(ids):
            lo, hi = max(0, t - window), min(len(ids), t + window + 1)
            for j in range(lo, hi):
                if j != t:
                    pairs.append((centre, ids[j]))
    return pairs

def make_neg_sampler(counter, vocab):
    items = list(counter)
    probs = np.array([counter[i] ** 0.75 for i in items], dtype=np.float64)
    probs /= probs.sum()
    idx_lookup = np.array([vocab[i] for i in items])

    def sample(n_rows: int, k: int) -> np.ndarray:
        chosen = np.random.choice(len(items), size=(n_rows, k), p=probs)
        return idx_lookup[chosen]

    return sample

# --- 训练循环 ------------------------------------------------------------- #

if __name__ == "__main__":
    sequences = [
        [1, 2, 3, 4, 5], [2, 3, 4, 6, 7],
        [1, 3, 5, 8, 9], [4, 5, 6, 10, 11],
    ]
    vocab, counter = build_vocab(sequences)
    pairs = make_pairs(sequences, vocab, window=2)
    neg = make_neg_sampler(counter, vocab)

    model = Item2Vec(len(vocab), embedding_dim=64, num_negatives=5)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    BATCH = 32
    for epoch in range(10):
        random.shuffle(pairs)
        running = 0.0
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            t = torch.tensor([p[0] for p in batch], dtype=torch.long)
            c = torch.tensor([p[1] for p in batch], dtype=torch.long)
            ng = torch.tensor(neg(len(batch), 5), dtype=torch.long)

            opt.zero_grad()
            loss = model(t, c, ng)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"epoch {epoch + 1:>2}  loss {running / max(1, len(pairs) // BATCH):.4f}")
```

### 2.4 设计决策与依据

| 决策 | 选择 | 为什么 |
|---|---|---|
| 两套 Embedding 表 | 中心 / 上下文分离 | 让模型更自由地编码"我是什么"和"我常出现在谁旁边"。Word2Vec 标准做法 |
| 负采样数 $K$ | 5 – 20 | 小 $K$ 训练快；大 $K$ 对长尾更好。20 之后边际收益迅速递减 |
| 窗口大小 $c$ | 2 – 5 | 窗口越大噪声越多。会话本身就短，小窗口够用 |
| 负采样分布 | $P_n \propto f^{0.75}$ | 纯频率过度采样爆款；均匀过度采样长尾。3/4 次幂是 Mikolov 实证的最优解 |
| 内积裁剪 `clamp(-10, 10)` | 数值稳定 | 一次溢出的 sigmoid 就能让整个 epoch NaN。便宜的保险 |

### 2.5 实战中的坑

- **冷启动物品**：训练集没出现过的新物品没有 Embedding。三种解法：(1) 用内容特征（文本、图片）训一个旁路网络预测它的 Embedding；(2) 用同类物品的平均向量初始化；(3) 接受随机初始化，让线上早期流量把它"拉"到合适位置。
- **会话长度差异巨大**：截取最近 $N$ 个（一般 50 – 100），过短的填充。超长会话切窗。
- **负样本碰撞**：百万级物品里随机采到正样本的概率不到 $0.1\%$，可以忽略；千级物品要显式排除。
- **重复物品**：忠实用户会把同一首歌循环 50 次。建对前先去除连续重复，否则模型只是学到"这首歌和它自己很像"。

### 2.6 CBOW：对偶变体

CBOW 把窗口内的上下文 Embedding 求平均，用平均向量预测中心物品。训练更快，但对长尾物品略差——Skip-gram 看到每个上下文都是独立的一次梯度，CBOW 把它们平均掉了。

```python
class Item2VecCBOW(nn.Module):
    """从平均的上下文窗口预测中心物品"""

    def __init__(self, vocab_size, embedding_dim, num_negatives=5):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embedding_dim)
        self.out_emb = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, context, target, negatives):
        ctx = self.in_emb(context).mean(dim=1)        # (B, d)
        tgt = self.out_emb(target)                    # (B, d)
        neg = self.out_emb(negatives)                 # (B, K, d)

        pos = torch.clamp((ctx * tgt).sum(-1), -10, 10)
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-10)

        neg = torch.clamp(torch.bmm(neg, ctx.unsqueeze(-1)).squeeze(-1), -10, 10)
        neg_loss = -torch.log(torch.sigmoid(-neg) + 1e-10).sum(-1)
        return (pos_loss + neg_loss).mean()
```

> **推荐场景默认选 Skip-gram**。物品热度长尾分布严重，Skip-gram 给罕见物品分配的梯度比 CBOW 更慷慨。

---
## 3. 基于图的嵌入：Node2Vec

### 3.1 序列模型不够用的时候

Item2Vec 只能捕捉**时间上相邻**的关系。但推荐场景中的数据更像一张图：商品会被共同购买、共同浏览，或者属于同一类目、同一品牌。同一件商品可能同时出现在两个完全不同的"邻居"中——比如一顶帐篷既属于"露营"也属于"骑行装备"。序列模型会把这种结构压平，而图模型则能保留它。

### 3.2 有偏随机游走

Node2Vec（Grover & Leskovec, 2016）通过在图上随机游走生成"句子"，关键在于**怎么走**：通过两个超参数控制游走风格，既可以偏向"在邻域内转"（BFS 风格，捕捉社区结构），也可以偏向"沿着主路远行"（DFS 风格，捕捉结构角色）。

> **类比**：把游走想象成探索一座城市。BFS 风格是"先把这条街所有店逛一遍，再去下一条街"——你会彻底摸清一个区域；DFS 风格是"沿着主路一直开十公里"——你会看清不同区域之间的连接关系。Node2Vec 让你自由调节这两种风格。

具体规则：从 $t$ 走到 $v$ 后，下一步走到邻居 $x$ 的非归一化概率为：

$$\alpha_{p, q}(t, x) \;=\;
\begin{cases}
1/p & \text{若 } d_{t,x} = 0 \;\;\text{（原路返回）} \\
1   & \text{若 } d_{t,x} = 1 \;\;\text{（} x \text{ 也是 } t \text{ 的邻居）} \\
1/q & \text{若 } d_{t,x} = 2 \;\;\text{（} x \text{ 离 } t \text{ 更远一步）}
\end{cases}$$

| 参数设置 | 游走风格 | 捕捉的结构 |
|---|---|---|
| $p$ 小、$q$ 大 | BFS | 社区/聚类结构 |
| $p$ 大、$q$ 小 | DFS | 结构等价性、中枢-辐射模式 |
| $p = q = 1$ | 标准随机游走（DeepWalk） | 两者混合 |

![用户-物品二部图上的有偏随机游走，箭头依次连接 U2 -> I3 -> U4 -> I8 -> I7 -> U3 -> I4，展示一次游走如何同时穿越用户与物品节点](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig4_random_walk.png)

得到一批游走序列后，直接把它们当作"句子"喂给 Skip-gram——算法不关心这些"句子"是来自用户行为还是图游走。

### 3.3 完整实现

```python
import numpy as np
import networkx as nx
import random

class Node2Vec:
    """有偏随机游走 + Skip-gram，输入是 NetworkX 图"""

    def __init__(self, graph, dimensions=64, walk_length=80,
                 num_walks=10, p=1.0, q=1.0, window_size=10):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size

    def _walk(self, start):
        walk = [start]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            nbrs = list(self.graph.neighbors(cur))
            if not nbrs:
                break

            if len(walk) == 1:
                walk.append(random.choice(nbrs))
                continue

            prev = walk[-2]
            weights = []
            for x in nbrs:
                w = self.graph[cur][x].get("weight", 1.0)
                if x == prev:                         # 距离 0
                    weights.append(w / self.p)
                elif self.graph.has_edge(prev, x):    # 距离 1
                    weights.append(w)
                else:                                 # 距离 2
                    weights.append(w / self.q)
            weights = np.array(weights)
            weights /= weights.sum()
            walk.append(np.random.choice(nbrs, p=weights))
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for n in nodes:
                walks.append(self._walk(n))
        return walks

    def fit(self):
        from gensim.models import Word2Vec
        walks = [[str(n) for n in w] for w in self.generate_walks()]
        model = Word2Vec(walks, vector_size=self.dimensions,
                         window=self.window_size, sg=1,
                         negative=5, min_count=0, workers=4)
        return {n: model.wv[str(n)] for n in self.graph.nodes()}
```

### 3.4 图从哪里来

工业场景中，图通常需要从交互日志里**构造**出来。常见做法是用**Jaccard 相似度**对边加权：

```python
from collections import defaultdict

def co_occurrence_graph(interactions, min_jaccard=0.1, max_users=None):
    """从 (user, item) 交互构造物品-物品图。
    边权 = 同时与两件物品发生过交互的用户集合的 Jaccard 相似度。
    """
    item_users = defaultdict(set)
    for u, i in interactions:
        item_users[i].add(u)

    G = nx.Graph()
    items = list(item_users)
    for a in range(len(items)):
        ua = item_users[items[a]]
        if max_users and len(ua) > max_users:
            continue
        G.add_node(items[a])
        for b in range(a + 1, len(items)):
            ub = item_users[items[b]]
            inter = len(ua & ub)
            if inter == 0:
                continue
            j = inter / len(ua | ub)
            if j >= min_jaccard:
                G.add_edge(items[a], items[b], weight=j)
    return G
```

> **工程提醒**：上面的双重循环复杂度是 $O(N^2)$。真实场景中，我会先按用户建倒排索引，按用户遍历，只在共享用户的物品对之间产生候选边，复杂度降到 $O(\sum_u |I_u|^2)$，实战中效率高几个数量级。

---
## 4. 双塔模型：用户和物品分开编码

### 4.1 为什么用双塔，而不是单塔

把用户特征和物品特征拼接起来，送入一个网络，输出分数——为什么不这样做？**因为在线上服务时，每个（用户，候选物品）对都要跑一次网络。** 假设有 1000 万个候选物品，那就意味着每次请求需要做 1000 万次前向传播，完全不可行。

双塔架构将模型拆分为**用户塔** $f_u(\mathbf{x}_u)$ 和**物品塔** $f_i(\mathbf{x}_i)$，然后通过一个相似度函数（通常是余弦相似度）计算两者的输出：

$$\mathbf{e}_u = f_u(\mathbf{x}_u; \theta_u), \qquad
\mathbf{e}_i = f_i(\mathbf{x}_i; \theta_i), \qquad
s(u, i) = \cos(\mathbf{e}_u, \mathbf{e}_i).$$

> **类比**：想象一个相亲 App。一座塔负责生成每个人的“自我介绍”；另一座塔负责生成每个人的“择偶标准”。匹配时只需比较档案，不需要为每对潜在组合重新跑算法。

这种架构的优势非常明显：

1. **离线预计算所有物品向量**：每个物品过一次物品塔，结果存储下来。
2. **线上只跑用户塔**：每次请求只需一次前向传播。
3. **用 ANN 检索 top-K**：毫秒级返回最近邻物品。

这就是现代推荐系统**召回（retrieval / recall）**阶段的标准做法。

![双塔 DSSM：用户特征沿左塔向上，物品特征沿右塔向上，两边都得到 L2 归一化的 d 维向量，顶部余弦相似度给出打分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig3_two_tower.png)

### 4.2 DSSM 实现

DSSM（Huang et al., Microsoft, 2013）是双塔模型的经典之作，最初为搜索引擎设计，如今几乎每个大型推荐系统都能看到它的身影。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_tower(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class DSSM(nn.Module):
    """对称双塔 + 余弦相似度打分"""

    def __init__(self, user_dim, item_dim, embedding_dim=128, hidden=(256, 128)):
        super().__init__()
        self.user_tower = make_tower(user_dim, list(hidden), embedding_dim)
        self.item_tower = make_tower(item_dim, list(hidden), embedding_dim)

    def encode_user(self, x):
        return F.normalize(self.user_tower(x), p=2, dim=-1)

    def encode_item(self, x):
        return F.normalize(self.item_tower(x), p=2, dim=-1)

    def forward(self, user_x, item_x):
        u = self.encode_user(user_x)
        i = self.encode_item(item_x)
        return (u * i).sum(-1)        # 已 L2 归一化，内积即余弦

class SampledSoftmaxLoss(nn.Module):
    """正样本必须在 in-batch 负样本中胜出（带温度的交叉熵）"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, u, pos, neg):
        # u   : (B, d)
        # pos : (B, d)
        # neg : (B, K, d)
        pos_score = (u * pos).sum(-1, keepdim=True) / self.t          # (B, 1)
        neg_score = torch.bmm(u.unsqueeze(1),
                              neg.transpose(1, 2)).squeeze(1) / self.t  # (B, K)
        logits = torch.cat([pos_score, neg_score], dim=1)             # (B, 1+K)
        target = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
        return F.cross_entropy(logits, target)
```

短短 30 行代码里隐藏了几个关键设计：

- **L2 归一化输出**：让余弦相似度等价于内积，可以直接用 FAISS 的 inner-product 索引。
- **温度系数 $\tau$**：取 $\tau \in [0.05, 0.2]$ 让 softmax 更“尖锐”。不加温度时，余弦相似度分布在 $[-1, 1]$ 区间，分布太平坦，学习速度慢。
- **层间 BatchNorm**：在塔的深层保持激活值的合理尺度，尤其是输入特征量纲差异较大时。
- **正样本固定在 0 号位的交叉熵**：天然兼容 in-batch negative，直接把同 batch 内其他正样本当作负样本——见第 5 节。

---
## 5. YouTube DNN：用户历史的池化

### 5.1 背景

YouTube DNN（Covington et al., 2016）是双塔模型的经典变体，专为视频推荐设计。用户的表示不再是简单的特征向量，而是一段**最近观看的视频序列**。用户塔的任务就是**将这段序列压缩成一个向量**。

原始的 YouTube DNN 做法非常直接：对观看视频的嵌入做**平均池化**，然后拼接人口统计特征，再通过一个两层的 MLP。平均池化的延迟极低，在 YouTube 的规模下，每节省一毫秒都能省下真金白银的数据中心成本。

```python
class YouTubeDNN(nn.Module):
    """对观看历史做平均池化，拼接侧信息特征，投影到 d 维"""

    def __init__(self, num_videos, num_categories, d=64,
                 user_hidden=(256, 128), item_hidden=(128, 64)):
        super().__init__()
        self.video = nn.Embedding(num_videos, d)
        self.category = nn.Embedding(num_categories, 16)

        self.user_mlp = make_tower(d + 16, list(user_hidden), d)
        self.item_mlp = make_tower(d + 16, list(item_hidden), d)

    def user_vector(self, history_ids, history_cats):
        h = self.video(history_ids).mean(dim=1)
        c = self.category(history_cats).mean(dim=1)
        return F.normalize(self.user_mlp(torch.cat([h, c], dim=-1)), dim=-1)

    def item_vector(self, item_id, item_cat):
        e = self.video(item_id)
        c = self.category(item_cat)
        return F.normalize(self.item_mlp(torch.cat([e, c], dim=-1)), dim=-1)

    def forward(self, hist_ids, hist_cats, item_id, item_cat):
        u = self.user_vector(hist_ids, hist_cats)
        i = self.item_vector(item_id, item_cat)
        return (u * i).sum(-1)
```

### 5.2 平均池化太粗糙？引入注意力机制

平均池化对每个观看视频一视同仁。但实际情况是，如果一个用户刚看了 10 个美食视频和 1 个 10 秒的汽车广告，他更可能想看更多美食，而不是汽车。Self-Attention 让模型自己学习权重：

```python
class YouTubeDNNWithAttention(nn.Module):
    def __init__(self, num_videos, num_categories, d=64, n_heads=4):
        super().__init__()
        self.video = nn.Embedding(num_videos, d)
        self.category = nn.Embedding(num_categories, 16)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.user_mlp = make_tower(d + 16, [256], d)
        self.item_mlp = make_tower(d + 16, [128], d)

    def user_vector(self, history_ids, history_cats):
        h = self.video(history_ids)                        # (B, T, d)
        h, _ = self.attn(h, h, h)                          # 自注意力
        h = h.mean(dim=1)                                  # 对加权后的序列做池化
        c = self.category(history_cats).mean(dim=1)
        return F.normalize(self.user_mlp(torch.cat([h, c], dim=-1)), dim=-1)

    def item_vector(self, item_id, item_cat):
        e = self.video(item_id)
        c = self.category(item_cat)
        return F.normalize(self.item_mlp(torch.cat([e, c], dim=-1)), dim=-1)
```

> **经典权衡**：平均池化的复杂度是 $O(T)$，速度快到极致；自注意力复杂度是 $O(T^2)$，召回率通常能提升几个百分点。对于长历史的实时召回，针对候选物品的目标注意力（[DIN, Zhou et al., 2018](https://arxiv.org/abs/1706.06978)）往往是更优的工程选择。

---
## 6. 负采样策略

正样本很贵——它们来自真实用户行为。负样本很便宜——随便一抓就是几百万。关键在于**挑对那些便宜的负样本**，因为完全随机的太简单，而难度过高的又会让训练不稳定。

### 6.1 四种策略

| 策略 | 机制 | 适用场景 |
|---|---|---|
| **均匀采样** | 从全量商品池中随机抽取负样本 | 简单基线；适合小规模商品池 |
| **热度感知采样** | 按 $P_n \propto f^{0.75}$ 抽取负样本 | 大规模商品池默认选择；缓解热度偏差 |
| **Hard Negative Mining** | 选取“看起来相关但用户未交互”的商品 | 后期微调；锐化决策边界 |
| **In-batch Negatives** | 将同一批次中其他用户的正样本作为负样本 | 双塔模型大规模训练；几乎零成本 |

### 6.2 为什么 In-batch Negatives 是主力

在一个包含 $B$ 对（用户, 正样本商品）的批次中，其他用户的正样本商品嵌入可以直接用作当前用户的负样本。这样每行就免费获得了 $B - 1$ 个负样本。变体：**添加热度修正项**，抵消 In-batch 采样的偏差——热门商品在批次中出现频率更高，容易被过度惩罚。标准做法（Yi et al., 2019, "Sampling-Bias-Corrected Neural Modeling"）是在 softmax 前从 logit 中减去 $\log p(i)$。

### 6.3 实现代码片段

```python
import numpy as np

def uniform_neg(catalog_size: int, k: int, exclude: set | None = None) -> list[int]:
    out = set()
    while len(out) < k:
        i = int(np.random.randint(catalog_size))
        if exclude is None or i not in exclude:
            out.add(i)
    return list(out)

def popularity_neg(probs: np.ndarray, k: int) -> np.ndarray:
    """probs 是长度为 N 的、已经按 f^0.75 归一化的概率向量"""
    return np.random.choice(len(probs), size=k, replace=False, p=probs)

def in_batch_neg(item_emb: torch.Tensor) -> torch.Tensor:
    """item_emb: (B, d)。返回 (B, B-1, d)，即用批次中其他行作为负样本"""
    B = item_emb.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=item_emb.device)
    idx = (~eye).nonzero(as_tuple=False)[:, 1].view(B, B - 1)
    return item_emb[idx]

def hard_neg(user_emb, item_emb, positives_mask, top_k=100, k=10):
    """Hard negative：与用户最相似但不在正样本中的商品"""
    sims = user_emb @ item_emb.T
    sims[positives_mask] = -1e9
    top = sims.topk(top_k, dim=-1).indices
    pick = torch.randint(0, top_k, (user_emb.size(0), k), device=user_emb.device)
    return top.gather(1, pick)
```

> **课程式训练**：先用均匀采样或热度感知采样训练几个 epoch，让模型稳定下来。之后逐步混入 10% – 30% 的 Hard Negative，锐化决策边界。一开始就纯用 Hard Negative 容易导致模型崩溃——模型还没学会粗略区分，“Hard”对它来说只是噪声。
## 7. 近似最近邻检索（ANN）

### 7.1 在线服务的硬约束

物品库有 100 万条，延迟预算只有 100 毫秒。暴力扫描复杂度是 $O(Nd)$，当 $d=128$ 时，单次查询需要大约 $1.3 \times 10^{8}$ 次乘加运算。单台高性能 CPU 能扛住，但 QPS 上万的集群就顶不住了。**近似最近邻检索**用 1% 到 5% 的召回率换取 10 倍到 100 倍的速度提升。

![ANN 检索：左图展示 IVF 索引在最近的簇质心内搜索；右图对比 Flat、IVF、HNSW、Annoy、IVFPQ 在 100 万物品、d=128 上的延迟与 Recall@10](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig5_ann_search.png)

### 7.2 三大索引家族

| 家族 | 核心思想 | 优势 |
|---|---|---|
| **倒排索引 IVF** | 用 k-means 把向量聚成簇，查询时只扫最近的几个簇 | 速度和召回可调，适合中大型索引 |
| **HNSW**（分层可导航小世界图） | 多层近邻图 + 贪心导航 | 高召回下查询最快，内存占用适中 |
| **乘积量化 PQ** | 把向量切分子向量分别量化，压缩到几个字节 | 内存节省明显，十亿级必备（常与 IVF 结合为 IVFPQ） |

Annoy（Spotify 开源）基于随机投影树，API 简单，适合快速原型开发，但在相同召回率下通常比 HNSW 慢。

### 7.3 FAISS 工程封装

```python
import faiss
import numpy as np

class ANNIndex:
    """对 FAISS 三种常用索引的轻量封装"""

    def __init__(self, dim: int, kind: str = "IVF",
                 nlist: int = 100, hnsw_m: int = 32):
        if kind == "Flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "IVF":
            quantiser = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif kind == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m,
                                             faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(kind)
        self._trained = False

    def build(self, vectors: np.ndarray) -> None:
        x = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(x)             # 归一化后内积等价于余弦相似度
        if hasattr(self.index, "train") and not self._trained:
            self.index.train(x)
            self._trained = True
        self.index.add(x)

    def query(self, vector: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(vector.reshape(1, -1), dtype="float32")
        faiss.normalize_L2(q)
        sims, ids = self.index.search(q, k)
        return sims[0], ids[0]

    def set_nprobe(self, n: int) -> None:
        """IVF 专用：每次查询扫描的簇数量。值越大，越慢但召回越高"""
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = n
```

### 7.4 性能对比一览

| 索引 | 构建时间 | 查询时间 | 内存 | Recall@10 | 适用场景 |
|---|---|---|---|---|---|
| Flat | 秒级 | $O(N)$，慢 | $4Nd$ B | 100% | Ground truth；小规模目录 |
| IVF | 分钟级 | 快 | $4Nd$ B | 95 – 99% | 通用大规模，性能可调 |
| HNSW | 数十分钟 | 极快 | $4Nd$ + 图 | 95 – 99% | 严格延迟要求 |
| IVFPQ | 分钟级 | 快 | $\sim N$ B（8× 压缩） | 90 – 95% | 十亿级，内存受限 |
| Annoy | 分钟级 | 快 | 中 | 90 – 95% | 静态索引，部署简单 |

> **运维经验值**：IVF 起步参数 `nprobe = sqrt(nlist)`，`nlist ≈ sqrt(N)`。HNSW 默认 `M = 32`，`efSearch = 100`。**永远在自己的向量上跑 benchmark**——通用 benchmark 的召回率在不同数据分布下波动很大。
## 8. 评估 Embedding 质量

Embedding 不直接对用户可见，所以需要借助代理指标。我会同时使用 **内在指标**（关注向量本身的性质）和 **外在指标**（关注推荐结果的质量）。

![12 部电影按 4 个类目分组的 Embedding 余弦相似度热力图，块对角线显示同类目相似度高、跨类目相似度接近 0](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding表示学习/fig6_similarity_heatmap.png)

### 8.1 内在指标：几何结构合理吗？

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def neighbour_overlap(emb: dict, gold: dict, k: int = 10) -> float:
    """计算 Embedding top-k 和 ground-truth top-k 的平均 Jaccard 相似度"""
    scores = []
    items = list(emb)
    matrix = np.stack([emb[i] for i in items])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9

    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)

    for idx, item in enumerate(items):
        if item not in gold:
            continue
        top_emb = set(items[j] for j in np.argsort(-sims[idx])[:k])
        top_gold = set(sorted(gold[item], key=gold[item].get, reverse=True)[:k])
        scores.append(len(top_emb & top_gold) / k)
    return float(np.mean(scores)) if scores else 0.0

def cluster_silhouette(emb: dict, n_clusters: int = 10) -> float:
    matrix = np.stack(list(emb.values()))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(matrix)
    return silhouette_score(matrix, labels)
```

### 8.2 外在指标：推荐效果提升了吗？

```python
def hit_rate_ndcg(user_vec: dict, item_vec: dict, test, k: int = 10):
    items = list(item_vec)
    matrix = np.stack([item_vec[i] for i in items])
    item_index = {i: idx for idx, i in enumerate(items)}

    hits = 0
    ndcg = []
    for user_id, true_item in test:
        if user_id not in user_vec or true_item not in item_index:
            continue
        sims = matrix @ user_vec[user_id]
        top = np.argsort(-sims)[:k]
        if item_index[true_item] in top:
            hits += 1
            rank = int(np.where(top == item_index[true_item])[0][0]) + 1
            ndcg.append(1.0 / np.log2(rank + 1))
        else:
            ndcg.append(0.0)
    return {"hit_rate@k": hits / len(test), "ndcg@k": float(np.mean(ndcg))}
```

> **真正的评估体系不止看精度**。覆盖率（推荐系统能覆盖多少比例的物品库）、多样性（推荐列表内部的差异性）、惊喜度（远离用户历史但仍然能转化的推荐）、新鲜度都很重要。一个 HR@10 表现很好却只推头部 1% 物品的模型，本质上只是披着算法外衣的“热门榜单”。
## 9. 常见问题

### Q1. Item2Vec 和矩阵分解（MF），怎么选？

矩阵分解（MF）从完整的用户-物品交互矩阵中学习，捕捉的是**全局**偏好模式。Item2Vec 则从序列数据中学习，捕捉的是**时序共现**关系。如果数据天然有序（比如播放列表、会话记录、观看日志），Item2Vec 通常表现更好；如果有显式评分或长期稳定的偏好数据，MF 更简单且效果也不错。

### Q2. 如何选择嵌入维度？

| 物品规模 | 典型 $d$ |
|---|---|
| < 10K | 32 – 64 |
| 10K – 1M | 64 – 128 |
| > 1M | 128 – 512 |

维度翻倍会让索引体积和查询延迟翻倍，但效果不会成比例提升——收益递减很快。用 HR@K 和 NDCG 验证效果，别盯着训练损失看。

### Q3. 推荐场景下选 Skip-gram 还是 CBOW？

推荐场景几乎总是选 Skip-gram。它会给长尾物品分配更多梯度，而这正是推荐系统需要帮助的地方。

### Q4. Node2Vec 在什么情况下比 Item2Vec 更强？

当物品之间的关系不是天然有序时——比如共购图、社交网络、知识图谱、物品-属性图——任何“邻居”含义超出“时间相邻”的场景，Node2Vec 更适合。

### Q5. 为什么不直接端到端训练一个大模型？

单塔模型同时输入用户和物品特征，无法拆分成独立的编码器，召回阶段复杂度变成 $O(N)$ 每次查询。双塔模型牺牲了一点表达能力，换来预计算和索引的能力——在大规模场景下，这个权衡是必须的。

### Q6. 冷启动怎么处理？

- **新用户**：用户塔用人口统计或上下文特征，不需要用户 ID。
- **新物品**：物品塔用内容特征（文本、图片、类目嵌入）。
- **混合方案**：将学到的 ID 嵌入与内容嵌入相加。ID 部分等数据积累后起作用，内容部分支撑冷启动。

### Q7. 负采样有什么最佳实践？

开始时用均匀采样或基于热度的采样保证稳定性。训几轮后混入 10% – 30% 的难负样本。能用 in-batch 负样本就尽量用——它们几乎是免费的。

### Q8. FAISS、HNSW、Annoy 怎么选？

- **FAISS**：超大规模索引（> 10M 物品）的工业标准，支持 GPU，索引类型最丰富。
- **HNSW**（`hnswlib` 或 FAISS 的 `IndexHNSWFlat`）：高召回率下查询最快，严格延迟要求下的默认选择。
- **Annoy**：适合原型开发或离线重建的静态索引。

### Q9. 没有标签时如何快速验证嵌入质量？

挑几个熟悉的物品，检查它们的最近邻是否合理。用 t-SNE 或 UMAP 可视化，看已知类别是否聚类。用辅助分类标签计算轮廓系数（silhouette score）。

### Q10. 用内积还是余弦相似度？

余弦相似度只看方向，模长被抵消；内积还会奖励大模长——如果你希望热门物品得分更高（训练后它们的嵌入模长通常更大），可以用内积。大多数检索场景下，**对所有向量做 L2 归一化，内积等于余弦相似度**，还能直接用 FAISS 的 inner-product 索引模式。
## 10. 总结

- **Embedding 是密集的、学习得到的向量，把离散身份转化为几何结构**。推荐问题就变成了最近邻搜索。
- **根据数据特点选择方法**：序列数据用 Item2Vec，图数据用 Node2Vec，特征丰富且物品库大时用双塔（DSSM、YouTube DNN）。
- **负采样是模型设计的关键部分**：混合使用均匀采样、热度感知采样、Hard 负例和 In-batch 负例，同时观察课程学习效应。
- **双塔架构是召回阶段的工业标准**，因为它能让物品向量提前计算并建立索引。
- **ANN 搜索让在线服务成为可能**：根据延迟、召回率和内存预算选择 FAISS-IVF、HNSW 或 Annoy，并在自己的数据上做性能测试。
- **评估要兼顾内在和外在指标**：除了 HR@K 和 NDCG，还要关注覆盖率和多样性，否则系统会不知不觉偏向热门物品。
