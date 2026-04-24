---
title: "推荐系统（五）—— Embedding表示学习"
date: 2025-10-31 09:00:00
tags:
  - 推荐系统
  - Embedding
  - 表示学习
categories: 推荐系统
series:
  name: "推荐系统"
  part: 5
  total: 16
lang: zh-CN
mathjax: true
description: "推荐系统 Embedding 技术全解：从 Word2Vec、Item2Vec、Node2Vec 到双塔 DSSM 与 YouTube DNN，再到负采样策略与 FAISS/HNSW 近邻检索的工程实践。每节配有可运行的 PyTorch 代码、关键设计权衡与生产经验。"
disableNunjucks: true
series_order: 5
---

当 Netflix 把《盗梦空间》推给一个刚看完《蝙蝠侠：黑暗骑士》的用户，背后并不是一条手写的"看过 Nolan 就推 Nolan"的规则，而是一个几何问题：这两部电影在一个 128 维的 **Embedding 空间** 里距离很近，而它们和《海底总动员》之间的距离很远。**几何替代了枚举**——系统不再去和数万部电影逐一比对，而是只问一个问题："这两个向量之间有多远？"

本文系统讲解推荐系统中 Embedding 是如何被学出来的、又是如何在毫秒级延迟下被在线服务的。我们会按从概念到工程的顺序走完五大类方法（序列、图、双塔、注意力池化、对比学习），把握负采样的工程艺术，并把近似最近邻检索（ANN）的真实性能边界讲清楚。每一节都配有可直接运行的 PyTorch 代码。

## 你将学到什么

- **Embedding 究竟是什么**——以及"低维稠密"为什么是它的全部价值
- **基于序列的方法**：Word2Vec、Item2Vec 与 Skip-gram 的完整推导
- **基于图的方法**：Node2Vec 的有偏随机游走机制
- **双塔架构**：DSSM、YouTube DNN，工业界召回阶段的事实标准
- **负采样策略**：均匀、按频率、Hard Negative、In-batch Negative
- **ANN 在线服务**：FAISS、HNSW、Annoy 的延迟/召回率权衡
- **评估方法**：内在指标（coherence、聚类）+ 外在指标（HR@K、NDCG）

## 前置知识

- 线性代数基础（向量、内积、矩阵乘法）
- 神经网络与 PyTorch 基础
- 推荐系统基础概念，参考 [第一篇](/zh/recommendation-systems-1-fundamentals/)
- 深度学习推荐有帮助但非必需，参考 [第三篇](/zh/recommendation-systems-3-deep-learning-basics/)

---

## 1. Embedding 基础：是什么，为什么

### 1.1 压缩视角

**Embedding 是一个学到的函数** $f : \mathcal{I} \to \mathbb{R}^{d}$，它把一个离散对象（用户、电影、商品 SKU、图节点）映射到一个 $d$ 维实数向量，其中 $d$ 远小于物品总数 $|\mathcal{I}|$。

> **类比**：把一个百万级商品目录想象成一张超长的清单。最朴素的表示是给每个商品一个百万维的 one-hot 向量。Embedding 则用 128 个数字代替——这是一组**潜在属性的向量化描述**：没有人显式标注"烧脑度""暗黑感""90 年代风"，但模型从数据里把它们抠了出来。

三个性质让 Embedding 成为现代推荐系统的通用语：

| 性质 | 价值 |
|---|---|
| **稠密性** | 每一维都承载信息，没有 one-hot 那种 99% 浪费的稀疏位 |
| **几何性** | "相似"成了可量化的物理量——一个内积或余弦值 |
| **可组合性** | 可以求平均、拼接、做 attention、放进向量索引 |

### 1.2 为什么稀疏表示不行

真实交互矩阵稀疏到离谱。一个有 $10^{8}$ 用户和 $10^{7}$ 物品的平台，矩阵理论上有 $10^{15}$ 个格子，但实际观测的非零项通常少于 $10^{11}$——密度不到万分之一。直接存储或分解这个矩阵不可行；把每行每列压缩成 $d$ 维稠密向量后，问题就变成了现代硬件最擅长的稠密矩阵乘法。

![物品 Embedding 空间（t-SNE 投影），按类目聚色，标出 query 物品及其 k-NN 检索半径](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig1_embedding_space.png)

上图就是本文的中心直觉。同类目的物品在向量空间里聚成簇，query 物品周围的近邻就是我们要推荐的物品。**推荐被还原为 Embedding 空间里的最近邻检索**。

### 1.3 学习目标只有一句话

从矩阵分解到 BERT4Rec，几乎所有 Embedding 方法都在优化同一个想法：

> **出现在相似上下文中的物品，应该有相似的向量；不相似的应该被推开。**

这就是从语言学借来的[分布假设](https://en.wikipedia.org/wiki/Distributional_semantics)——"观其同游而知其义"。在推荐场景里，"上下文"可以是：

- 同一个用户会话中的共现（Item2Vec）
- 图中的邻接关系（Node2Vec、GraphSAGE）
- 同一个 query 下的点击（DSSM）
- CTR 日志中的正样本（DeepFM、DLRM）

**上下文的定义决定了方法的形态。**

---

## 2. 基于序列的 Embedding：Word2Vec 与 Item2Vec

### 2.1 从词到物品

Word2Vec（Mikolov et al., 2013）有两种形式：

- **Skip-gram**：给定中心词，预测窗口内的上下文词
- **CBOW**：给定上下文词，预测中心词

> **类比**：Skip-gram 像是给你一块拼图，让你猜它周围有哪些；CBOW 反过来，给你周围的拼图，让你猜中间那块。

Item2Vec（Barkan & Koenigstein, 2016）的迁移看似平凡却威力巨大：**把用户的行为序列当成句子，把每个物品当成词**，Word2Vec 的整套机制原封不动搬过来。

### 2.2 Skip-gram 目标函数推导

给定序列 $S = [i_1, i_2, \dots, i_T]$ 与窗口大小 $c$，Skip-gram 最大化"由中心词预测上下文"的对数概率：

$$
\mathcal{L} \;=\; \sum_{t=1}^{T} \;\sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p\!\left(i_{t+j}\,\middle|\,i_t\right).
$$

最朴素的概率定义是在整个物品集上做 softmax：

$$
p\!\left(i_{t+j}\,\middle|\,i_t\right) \;=\; \frac{\exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_{t+j}}\right)}{\displaystyle\sum_{k=1}^{|\mathcal{I}|} \exp\!\left(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{k}\right)}.
$$

其中 $\mathbf{e}_i$ 是**输入（中心）Embedding**，$\mathbf{e}'_i$ 是**输出（上下文）Embedding**——每个物品有两套向量，第一次见会觉得别扭，但其实是为了让模型有更多自由度。

显然，分母在百万级物品上的 softmax 计算不可行。**负采样**把它改写成二分类问题：对每个真实的（中心，上下文）正样本对，随机采 $K$ 个"噪声"物品，让模型把正负样本区分开。目标函数变成：

$$
\mathcal{L} \;=\; \sum_{(i_t,\,i_c)} \!\left[\, \log\sigma(\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_c})
\;+\; \sum_{k=1}^{K} \mathbb{E}_{i_k \sim P_n}\!\big[\log\sigma(-\mathbf{e}_{i_t}^{\top}\mathbf{e}'_{i_k})\big]\right],
$$

其中 $\sigma$ 是 sigmoid，$P_n$ 是 unigram 频率的 $3/4$ 次幂——这是 Mikolov 经典经验：纯频率会过度采样热门，纯均匀会过度采样长尾，0.75 是实证的甜蜜点。

![Item2Vec Skip-gram 架构：滑动窗口选出中心物品，Embedding lookup 得到向量，模型把正样本上下文与按 3/4 次幂采样的 K 个负样本对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig2_item2vec_skipgram.png)

### 2.3 完整实现

下面是一个自洽可运行的 Item2Vec（Skip-gram + 负采样）。注意三处关键设计：**两套 Embedding 表**、**得分裁剪**、**sigmoid 数值技巧**。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random


class Item2Vec(nn.Module):
    """Skip-gram + Negative Sampling

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

        # 正样本：把中心和真实上下文的内积推大
        pos = torch.clamp((t * c).sum(-1), -10, 10)
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-10)

        # 负样本：把中心和 K 个随机物品的内积推小
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

## 3. 基于图的 Embedding：Node2Vec

### 3.1 序列不够用的时候

Item2Vec 只看到了**时间上相邻**的关系。但电商或社交数据更像一张图：物品被共同购买、共同浏览、属于同一类目、同一品牌。同一件商品可以同时属于两个完全不同的"邻里"——一顶帐篷既属于"露营"也属于"骑行装备"。序列模型把这种结构压平了，图模型则保留它。

### 3.2 有偏随机游走

Node2Vec（Grover & Leskovec, 2016）从图上"走出"伪句子，关键在于**怎么走**：通过两个超参数让游走者既可能"在邻里里转"（BFS 风格，捕捉社区），也可能"沿着主路远行"（DFS 风格，捕捉结构角色）。

> **类比**：把游走当作探索一座城市。BFS 风格是"先把这条街所有店都走一遍，再去下一条街"——你把一个邻里摸熟了；DFS 风格是"沿着主路走十公里"——你看清了不同邻里之间是怎么连起来的。Node2Vec 让你在两者之间自由切换。

具体规则：从 $t$ 走到 $v$ 之后，下一步走到邻居 $x$ 的非归一化概率为

$$
\alpha_{p, q}(t, x) \;=\;
\begin{cases}
1/p & \text{若 } d_{t,x} = 0 \;\;\text{（即 } x = t\text{，原路返回）} \\
1   & \text{若 } d_{t,x} = 1 \;\;\text{（} x \text{ 也是 } t \text{ 的邻居）} \\
1/q & \text{若 } d_{t,x} = 2 \;\;\text{（} x \text{ 离 } t \text{ 更远一步）}
\end{cases}
$$

| 设置 | 游走风格 | 捕获的结构 |
|---|---|---|
| $p$ 小、$q$ 大 | BFS | 社区/聚类结构 |
| $p$ 大、$q$ 小 | DFS | 结构等价性、中枢-辐射关系 |
| $p = q = 1$ | 标准随机游走（DeepWalk） | 两者混合 |

![用户-物品二部图上的有偏随机游走，箭头依次连接 U2 -> I3 -> U4 -> I8 -> I7 -> U3 -> I4，展示一次游走如何同时穿越用户与物品节点](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig4_random_walk.png)

得到一批游走序列后，把它们当作"句子"喂给 Skip-gram——算法不在乎"句子"是用户行为还是图游走。

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

### 3.4 图本身从哪儿来

工业部署里，图本身往往要从交互日志里**构造**出来。常见做法是用**Jaccard 相似度**对边加权：

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

> **工程提醒**：上面的双重循环是 $O(N^2)$。真实场景要先按用户建倒排索引，按用户遍历，只在共享用户的物品对之间产生候选边，复杂度降到 $O(\sum_u |I_u|^2)$，实战中差几个数量级。

---

## 4. 双塔模型：用户和物品分开编码

### 4.1 为什么要双塔，而不是单塔

把用户特征和物品特征拼起来过一个网络，输出分数——为什么不行？**因为线上要为每一个候选都跑一次网络**。一千万候选 = 一千万次前向传播，根本扛不住。

双塔架构把模型拆成**用户塔** $f_u(\mathbf{x}_u)$ 和**物品塔** $f_i(\mathbf{x}_i)$，再用一个相似度函数（通常是余弦）打分：

$$
\mathbf{e}_u = f_u(\mathbf{x}_u; \theta_u), \qquad
\mathbf{e}_i = f_i(\mathbf{x}_i; \theta_i), \qquad
s(u, i) = \cos(\mathbf{e}_u, \mathbf{e}_i).
$$

> **类比**：像一个相亲 App。一座塔为每个人写一份"我是谁"的档案；另一座塔为每个人写一份"我喜欢什么样的人"的偏好；匹配时只需对比两份档案——不需要重新跑算法。

架构上的好处巨大：

1. **物品向量可离线预计算**：每个物品过一次物品塔，结果存起来。
2. **线上只跑用户塔**：每次请求一次前向。
3. **用 ANN 检索 top-K**：毫秒级返回最近邻物品。

这就是现代推荐系统**召回（retrieval / recall）**阶段的标准配方。

![双塔 DSSM：用户特征沿左塔向上，物品特征沿右塔向上，两边都得到 L2 归一化的 d 维向量，顶部余弦相似度给出打分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig3_two_tower.png)

### 4.2 DSSM 实现

DSSM（Huang et al., Microsoft, 2013）是双塔的鼻祖，最初为搜索而设计，如今几乎每一个大型推荐系统都有它的影子。

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

短短 30 行里藏了几个关键设计：

- **L2 归一化输出**：让余弦等价于内积，可以直接用 FAISS 的 inner-product 索引。
- **温度系数 $\tau$**：取 $\tau \in [0.05, 0.2]$ 让 softmax 更"尖"。不加温度时余弦的 $[-1, 1]$ 区间太平坦，学得慢。
- **层间 BatchNorm**：在塔的深处保持激活值的合理尺度，特别是输入特征量纲悬殊时。
- **正样本固定在 0 号位的交叉熵**：天然兼容 in-batch negative，把同 batch 内其他正样本作为负样本——见第 5 节。

---

## 5. YouTube DNN：把用户历史池化成一个向量

### 5.1 问题设定

YouTube DNN（Covington et al., 2016）是双塔的经典变体，专为视频推荐设计：用户由**最近观看序列**而不是平铺特征描述。用户塔的工作是**把这串序列池化成一个向量**。

原版 YouTube DNN 用最朴素的方法——**对观看视频的 Embedding 求平均**，再拼上人口属性，过两层 MLP。平均池化的延迟无敌，在 YouTube 的 QPS 量级下，每节省一毫秒都是数据中心账单上的真金白银。

```python
class YouTubeDNN(nn.Module):
    """对观看历史做平均池化，拼上侧信息特征，再投影到 d 维"""

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

### 5.2 平均池化太粗：用注意力

平均池化把每个观看一视同仁。但一个用户刚看了 10 个美食视频和 1 个 10 秒的汽车广告，他想要的显然不是更多汽车。Self-Attention 让模型自己学权重：

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
        h = h.mean(dim=1)                                  # 对加权后的序列再池化
        c = self.category(history_cats).mean(dim=1)
        return F.normalize(self.user_mlp(torch.cat([h, c], dim=-1)), dim=-1)

    def item_vector(self, item_id, item_cat):
        e = self.video(item_id)
        c = self.category(item_cat)
        return F.normalize(self.item_mlp(torch.cat([e, c], dim=-1)), dim=-1)
```

> **经典权衡**：平均池化的复杂度 $O(T)$，快到极致；自注意力 $O(T^2)$，召回率通常能涨几个百分点。对长历史的实时召回，针对候选物品的**目标注意力**（[DIN, Zhou et al., 2018](https://arxiv.org/abs/1706.06978)）往往是更平衡的工程选择。

---

## 6. 负采样策略

正样本贵——它来自真实用户行为；负样本便宜——满地都是。艺术在于**挑对便宜的负样本**：完全随机的太简单，逼真到几乎是正样本的又会让训练不稳定。

### 6.1 四大策略

| 策略 | 机制 | 适用 |
|---|---|---|
| **均匀采样** | 从全集中均匀随机采 | 简单基线，小目录 |
| **频率感知** | 按 $P_n \propto f^{0.75}$ 采 | 大目录默认选项；缓解热度偏差 |
| **Hard Negative** | 选"看起来相关但用户没交互"的物品 | 后期微调，磨锐决策边界 |
| **In-batch Negative** | 把同 batch 里别人的正样本当负样本 | 双塔训练神器，几乎免费 |

### 6.2 In-batch Negative 为什么是主力

一个 batch 里有 $B$ 对（用户, 正样本物品），别人那一行的物品 Embedding 就是你这一行的负样本——每行白送 $B-1$ 个负样本。变体：**popularity 修正项**抵消 in-batch 采样的偏差——热门物品在 batch 里出现更频繁，被惩罚过头。标准做法（Yi et al., 2019, "Sampling-Bias-Corrected Neural Modeling"）是在 softmax 前从 logit 里减去 $\log p(i)$。

### 6.3 实现片段

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
    """item_emb: (B, d)。返回 (B, B-1, d)，即用 batch 中其他行作为负样本"""
    B = item_emb.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=item_emb.device)
    idx = (~eye).nonzero(as_tuple=False)[:, 1].view(B, B - 1)
    return item_emb[idx]


def hard_neg(user_emb, item_emb, positives_mask, top_k=100, k=10):
    """Hard negative：与用户最相似但不在正样本里的物品"""
    sims = user_emb @ item_emb.T
    sims[positives_mask] = -1e9
    top = sims.topk(top_k, dim=-1).indices
    pick = torch.randint(0, top_k, (user_emb.size(0), k), device=user_emb.device)
    return top.gather(1, pick)
```

> **课程式训练**：先用均匀/频率采样训几个 epoch 让模型先稳住；之后混入 10 – 30% 的 hard negative 磨锐决策边界。**从一开始就上 hard negative 容易直接崩溃**——模型还没学会粗略区分，"hard"对它而言就是噪声。

---

## 7. 近似最近邻检索（ANN）

### 7.1 在线服务的硬约束

物品库 100 万，延迟预算 100 ms。暴力扫一遍是 $O(Nd)$，$d=128$ 时大约 $1.3 \times 10^{8}$ 次乘加——单 CPU 勉强可以，集群 QPS 上万就崩了。**近似** 最近邻检索用 1 – 5% 的召回率换 10 – 100× 的速度。

![ANN 检索：左图展示 IVF 索引在最近的簇质心内搜索；右图对比 Flat、IVF、HNSW、Annoy、IVFPQ 在 100 万物品、d=128 上的延迟与 Recall@10](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig5_ann_search.png)

### 7.2 三大索引家族

| 家族 | 思想 | 长项 |
|---|---|---|
| **倒排索引 IVF** | k-means 把向量聚成簇，查询时只扫几个最近的簇 | 速度/召回可调；中大型索引的主力 |
| **HNSW**（分层可导航小世界图） | 多层近邻图 + 贪心导航 | 高召回下查询最快；内存适中 |
| **乘积量化 PQ** | 把向量切成子向量分别量化成几个字节 | 压缩率高；十亿级必备（一般 IVFPQ 组合） |

Annoy（Spotify）走随机投影树路线，API 简单，原型化首选；但同等召回率下通常比 HNSW 慢。

### 7.3 FAISS 工程封装

```python
import faiss
import numpy as np


class ANNIndex:
    """对 FAISS 三种常用索引的薄封装"""

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
        faiss.normalize_L2(x)             # 内积 + 归一化 = 余弦
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
        """IVF 专用：每次查询扫多少个簇。值越大，越慢但召回越高"""
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = n
```

### 7.4 性能对比一览

| 索引 | 构建时间 | 查询时间 | 内存 | Recall@10 | 适用场景 |
|---|---|---|---|---|---|
| Flat | 秒级 | $O(N)$，慢 | $4Nd$ B | 100% | Ground truth；小目录 |
| IVF | 分钟级 | 快 | $4Nd$ B | 95 – 99% | 通用大规模，可调 |
| HNSW | 数十分钟 | 极快 | $4Nd$ + 图 | 95 – 99% | 严格延迟预算 |
| IVFPQ | 分钟级 | 快 | $\sim N$ B（8× 压缩） | 90 – 95% | 十亿级，内存受限 |
| Annoy | 分钟级 | 快 | 中 | 90 – 95% | 静态索引，部署简单 |

> **运维经验值**：IVF 起步参数 `nprobe = sqrt(nlist)`，`nlist ≈ sqrt(N)`。HNSW 默认 `M = 32`，`efSearch = 100`。**永远在你自己的向量上 benchmark**——一般 benchmark 上的召回率换到不同分布的数据会大幅波动。

---

## 8. Embedding 质量评估

Embedding 不直接面向用户，所以需要代理指标。**内在指标**（仅看向量本身）+ **外在指标**（看推荐效果）一起用。

![12 部电影按 4 个类目分组的 Embedding 余弦相似度热力图，块对角线高亮显示同类目相似度高、跨类目相似度近 0](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/05-Embedding%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0/fig6_similarity_heatmap.png)

### 8.1 内在指标：几何看起来对吗？

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def neighbour_overlap(emb: dict, gold: dict, k: int = 10) -> float:
    """Embedding top-k 和 ground-truth top-k 的 Jaccard 平均"""
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

### 8.2 外在指标：推荐变好了吗？

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

> **真正的评估套件不止精度**。覆盖率（多少比例的目录有机会被推荐）、多样性（同一推荐列表内的差异度）、惊喜度（远离用户历史但还能转化）、新鲜度都要看。一个 HR@10 赢了但永远只推 1% 头部物品的模型，就是个伪装成 ML 的"热门列表"。

---

## 9. 常见问题

### Q1. Item2Vec vs. 矩阵分解，怎么选？

矩阵分解（MF）从全量交互矩阵学，捕捉**全局**偏好模式；Item2Vec 从序列学，捕捉**时序共现**。如果数据天然有序（歌单、会话、观看日志），Item2Vec 通常更强；如果是显式评分或长期稳定偏好，MF 性价比很高。

### Q2. Embedding 维度怎么定？

| 目录规模 | 典型 $d$ |
|---|---|
| < 10K | 32 – 64 |
| 10K – 1M | 64 – 128 |
| > 1M | 128 – 512 |

维度翻倍 = 索引体积翻倍 + 查询延迟翻倍，但效果不会翻倍。用 HR@K、NDCG 校验，不要盯训练 loss。

### Q3. 推荐场景选 Skip-gram 还是 CBOW？

几乎总选 Skip-gram。物品热度长尾分布严重，Skip-gram 给罕见物品分配更多梯度。

### Q4. Node2Vec 在什么时候比 Item2Vec 强？

当关系不是天然序列时——共购图、社交网络、知识图谱、物品-属性图——任何"邻居"含义超出"时间相邻"的场景。

### Q5. 为什么不端到端训一个大模型？

把用户和物品特征都喂进同一个网络的"单塔"模型无法分解成独立的两个编码器，召回阶段就退化为 $O(N)$ 每查询。双塔用一点点表达力换取**预计算 + 索引**的能力——大规模下，这个交易没得选。

### Q6. 怎么处理冷启动？

- **新用户**：让用户塔吃人口/上下文特征，不依赖 ID。
- **新物品**：让物品塔吃内容（文本、图片、类目向量）。
- **混合**：把学到的 ID Embedding 与内容 Embedding 相加，数据多了 ID 部分占主导，新物品时内容部分撑场。

### Q7. 负采样有最佳配方吗？

起步用均匀或频率感知保稳定。训几个 epoch 后混入 10 – 30% 的 hard negative。能用 in-batch 就用——它几乎免费。

### Q8. FAISS、HNSW、Annoy 怎么选？

- **FAISS**：千万到十亿级的工业标准，支持 GPU，索引种类齐全。
- **HNSW**（`hnswlib` 或 FAISS 的 `IndexHNSWFlat`）：高召回下查询最快，严格延迟预算下的默认选择。
- **Annoy**：原型期或静态索引（每天离线 rebuild）的最佳拍档。

### Q9. 没有标签时怎么粗判 Embedding？

挑几个你熟悉的物品看看最近邻是否合理；t-SNE / UMAP 可视化，看已知类目是否聚成簇；用任何辅助类目算 silhouette score。

### Q10. 内积还是余弦？

余弦只看方向，模长被消掉；内积同时奖励大模长——如果你**希望**热门物品打分高（训练后它们的 Embedding 模长普遍更大），可以用内积。多数检索场景下，**对所有向量做 L2 归一化，用内积 = 用余弦**，同时还能直接走 FAISS 的 inner-product 索引。

---

## 10. 总结

- **Embedding 把离散身份变成几何**，推荐就此变成最近邻检索。
- **按数据形态选方法**：序列 → Item2Vec；图 → Node2Vec；特征丰富 + 大目录 → 双塔（DSSM、YouTube DNN）。
- **负采样是模型的另一半**：均匀 / 频率 / Hard / In-batch 混合用，注意课程效应。
- **双塔是召回阶段的事实标准**，因为它让物品向量可预计算、可索引。
- **ANN 让在线服务可行**：按延迟/召回/内存预算选 FAISS-IVF、HNSW 或 Annoy；**永远在你自己的向量上压测**。
- **评估要内外兼修**：HR@K 与 NDCG 之外，覆盖率、多样性也要盯着，否则会悄悄漂移成"热门列表"。

---

## 系列导航

- **上一篇**：[第四篇 — CTR 预估与点击率建模](/zh/recommendation-systems-4-ctr-prediction/)
- **下一篇**：[第六篇 — 序列推荐与会话建模](/zh/recommendation-systems-6-sequential-recommendation/)
- [回到系列总览（第一篇）](/zh/recommendation-systems-1-fundamentals/)
