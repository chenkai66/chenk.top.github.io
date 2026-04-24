---
title: "推荐系统（四）—— CTR预估与点击率建模"
date: 2024-05-05 09:00:00
tags:
  - 推荐系统
  - CTR预估
  - 深度学习
  - 特征交叉
categories: 推荐系统
series:
  name: "推荐系统"
  part: 4
  total: 16
lang: zh-CN
mathjax: true
description: "CTR预估模型全面解析：从Logistic Regression到FM/FFM，再到DeepFM、xDeepFM、DCN、AutoInt、FiBiNet等深度学习模型，附PyTorch实现与训练策略。"
disableNunjucks: true
---

每次你刷信息流、点开商品推荐或者看下一个推荐视频，背后都有一个 CTR（Click-Through Rate，点击率）预估模型在替你做决定。它要回答一个看似简单的问题：

> **"对这个特定的用户、这个特定的物品、这个特定的时刻，他点击的概率是多少？"**

简单的问题背后是工业界最有经济价值的机器学习问题之一。CTR 提升 1%，在 Google、Amazon、阿里这种规模上意味着数百万美元的额外收入；同样的模型也驱动着信息流、应用商店、新闻 App、视频 App。CTR 预估位于推荐系统的**排序阶段**：召回给出几千个候选，CTR 模型决定最终十几个真正展示出去。

本文带你走一遍过去十年 CTR 模型的演进路径——从一行 Logistic Regression 到基于注意力的架构。我们不会止步于公式，而是对每一个模型问三个问题：

1. **上一代模型的什么缺陷逼出了这个设计？**
2. **它的几何或概率直觉是什么？**
3. **怎么真正实现并把它部署上线？**

读完之后，你应该能够看懂任何一篇现代 CTR 论文，凭记忆画出它的结构图，并且能为自己的系统选对基线。

## 你将学到什么

- CTR 预估的本质，以及它**为什么**比一般的二分类难得多
- **Logistic Regression** 既是基线又是健全性检查——以及它**究竟在哪里失败**
- **Factorization Machines（FM）** 与 **Field-aware FM（FFM）**：在稀疏数据上自动学习二阶交互
- **DeepFM**：工业界最常见的"FM + 深度网络"组合
- **xDeepFM**：通过 CIN 显式建模高阶特征交互
- **DCN**：参数量线性增长的有界阶交叉特征
- **AutoInt**：把自注意力机制用在特征交互上
- **FiBiNet**：用 SENet 学习"哪个特征更重要"，加上更强的双线性交互
- 训练现实：类别不平衡、模型校准、AUC vs Logloss、上线前如何离线评估

## 前置知识

- 熟练使用 Python 和 PyTorch（`nn.Module`、训练循环、Embedding）
- 基本的深度学习概念，以及类别特征的 Embedding 视角（[Part 3](/zh/推荐系统-三-深度学习基础模型/)）
- 二分类、Sigmoid 与交叉熵的基本理解

---

## 理解 CTR 预估问题

### 什么是 CTR 预估

CTR 预估是**带有极端结构的二分类**。给定用户、物品和上下文，估计：

$$P(y = 1 \mid \mathbf{x}), \quad y \in \{0, 1\},\;\; 1 = \text{点击}.$$

特征向量 $\mathbf{x}$ 通常由三类信息拼接：

| 类别 | 例子 |
|---|---|
| 用户特征 | user_id、年龄段、性别、历史行为、国家 |
| 物品特征 | item_id、品牌、类目、价格段、新鲜度 |
| 上下文特征 | 小时、设备、网络、查询词、展示位 |

经验上 $\text{CTR} = \text{点击数} / \text{曝光数}$，模型输出之后被用来 **排序** 候选、**过滤** 低质量物品，并喂给下游的业务目标（广告里 eCPM = CTR x 出价；信息流里则是多目标加权分数）。

### CTR 预估为什么难

下面五个性质让 CTR 预估看起来像普通分类，实际上完全不是：

**1. 极端类别不平衡。** 展示广告 0.1-2%，电商 1-5%，信息流 2-10%。"全预测不点"就能拿 95% 准确率，所以 AUC 和 Logloss 取代了准确率。

**2. 高维超稀疏特征。** One-hot 之后特征维度在 $10^6$ 到 $10^9$ 之间，每个样本只有几十个非零位。每对特征存一个权重根本不可能。

**3. 信号藏在交互里。** "年轻用户"单独是弱信号；"年轻用户 x 动作片 x 晚上"才是金矿。如何**自动地、廉价地**捕捉这些交互，是模型设计的核心问题。

**4. 数据分布持续漂移。** 新物品、爆款、工作日/周末、节日。模型每天甚至每小时重训一次，单看离线 AUC 永远不够。

**5. 极严格的延迟预算。** 排序需要在 100 ms 内（通常 10 ms p99）打分上千个候选。模型大小、Embedding 查表、批处理与架构同等重要。

### CTR 预估的工业 Pipeline

从原始点击日志到排序结果，再回到模型重训，端到端流程是这样的：

![CTR 端到端 pipeline：原始日志、特征工程、Embedding、模型、排序与 A/B、反馈闭环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig6_pipeline.png)

几个值得注意的点：

- **特征工程仍然是真实系统的主战场。** Embedding 能学到的有限，显式交叉特征和统计特征（用户/物品/位置滑窗 CTR）经常是最大 A/B 提升的来源。
- **Embedding 是共享基础设施。** 所有深度 CTR 模型（FM、DeepFM、xDeepFM、DCN、AutoInt、FiBiNet）都共用同一张 Embedding 表，架构主要是定义**Embedding 之间如何交互**。
- **在线反馈闭环。** 昨天的服务日志就是今天的训练数据。模型新鲜度往往比模型复杂度更重要。

带着这张地图，我们沿时间线走一遍架构。

---

## Logistic Regression：基线，也是 FM 存在的理由

哪怕巨型神经网络已经在线上线下处处部署，Logistic Regression（LR）依然不会消亡。它是通用基线、校准锚点，在严格延迟预算的系统里甚至仍在直接打分。

### 模型定义

LR 把点击概率建模为一个线性打分函数过 Sigmoid：

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}.$$

> **大白话：** "把所有特征加权求和，加偏置，再压到 [0,1]。"

通过最小化二元交叉熵训练：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\big].$$

### 为什么 LR 既是宝又是不够用的

几何就把整个故事讲完了。LR 只能学到特征空间里的**一个超平面**。任何"特征 A 只在特征 B 同时存在时才有用"的模式，对它都不可见。最经典的例子就是 XOR 形状的点击行为：

![左图：LR 一条直线无法分开 XOR 形点击数据。右图：加一个交互项立刻恢复结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig1_lr_limitation.png)

左图里"年轻+动作片"和"老年+喜剧"会点击，但"年轻+喜剧"和"老年+动作片"不会。任何一条直线都不行——AUC 在 0.5 附近徘徊。右图加上一个交互项 $x_1 \cdot x_2$，立刻恢复结构。LR 之后所有 CTR 模型，本质都在回答同一个问题：

> **"如何自动地、大规模地发现并表达有用的特征交叉？"**

### 实现

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class LogisticRegression(nn.Module):
    """CTR 预估的 Logistic Regression 基线。"""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_lr(X_train, y_train, X_val, y_val, epochs=100, lr=0.01):
    scaler = StandardScaler()
    X_train_s = torch.FloatTensor(scaler.fit_transform(X_train))
    X_val_s = torch.FloatTensor(scaler.transform(X_val))
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    model = LogisticRegression(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_s), y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_s), y_val_t)
            print(f"Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

    return model, scaler
```

### LR 究竟在哪里失败

1. **没有特征交互。** 所有特征被当作互相独立。
2. **需要人工特征工程。** 想要"年龄 x 类目"必须手工拼出一列；二三阶以上的交叉根本写不完。
3. **决策边界是直线。** 上图已经看到，对 XOR 这类结构无能为力。

这三个失败直接催生了下一个模型：FM。

---

## Factorization Machines（FM）：自动二阶交互

Steffen Rendle 在 2010 年提出的 FM，是第一个让"自动二阶交互"在稀疏数据上既可行又统计有效的模型。

### 核心洞察

朴素的"带交互的 LR"会为每对特征学一个权重 $w_{ij}$。$d$ 个特征意味着 $O(d^2)$ 个参数——而且**大多数特征对在训练集里从未同时出现过**，所以根本学不到。

FM 把每对的权重换成两个低维向量的内积：

$$w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}.$$

> **类比：** 想象有 1000 部电影。要为每对存一个权重需要一百万个数，绝大多数你从未观察过。换个思路：给每部电影一个 $k$ 维"性格向量"，两部电影"合得来"当且仅当向量相似。这样只需要 $1000 \cdot k$ 个数；并且对一对**从未在训练集里共现的电影**，也能预测它们的交互强度——因为每个向量都是从大量其他共现样本里学来的。

最后一条性质——对未见特征对的泛化能力——才是 FM 的真正魔法，也是 FM 在稀疏数据上至今仍然能打的原因。

### 数学形式

$$\hat{y}(\mathbf{x}) = \underbrace{w_0}_{\text{偏置}} + \underbrace{\sum_{i=1}^{d} w_i x_i}_{\text{一阶}} + \underbrace{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j}_{\text{二阶交互}}.$$

二阶项**看起来**是 $O(d^2)$，但有一个漂亮的 $O(k \cdot d)$ 闭式：

$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\left[\Big(\sum_i \mathbf{v}_i x_i\Big)^2 - \sum_i (\mathbf{v}_i x_i)^2\right].$$

> **为什么成立。** 求和的平方包含所有 $i \cdot j$ 项（含 $i=j$）；减去平方求和去掉对角；除以二去掉重复计数。

### 实现

```python
import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """以 field 索引方式表达的 Factorization Machine。"""

    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.linear = nn.Linear(sum(field_dims), 1)
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

    def forward(self, x):
        """x: [batch, num_fields]，每列是一个 field 的类别索引。"""
        # 一阶部分：在 one-hot 视图上做线性
        linear_out = self.linear(self._one_hot(x))

        # 取出每个 field 的 embedding：[batch, num_fields, embed_dim]
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(len(self.field_dims))],
            dim=1,
        )

        # 高效二阶交互（(sum^2 - sum_of_squares)/2 技巧）
        sum_square = torch.sum(embs, dim=1) ** 2
        square_sum = torch.sum(embs ** 2, dim=1)
        interaction = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)

        return torch.sigmoid(linear_out + interaction)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh


model = FactorizationMachine(field_dims=[10, 20, 15], embed_dim=16)
x = torch.LongTensor([[0, 5, 2], [3, 10, 8], [1, 7, 1], [9, 15, 12]])
print(model(x).squeeze())
```

### FM 的优缺点

**优点。** 自动二阶交互；$O(kd)$ 计算；对未见特征对仍有泛化能力。

**缺点。** 只到二阶；并且**一个特征不管和谁交互，用的都是同一个 embedding**——这有时是错的。这一句话就是 FFM 的全部动机。

---

## Field-aware Factorization Machines（FFM）

FFM（2016）只对 FM 做了一处针对性的修改：**每个特征对每个"对方 field"都有一个独立的 embedding。**

### 直觉

在 FM 里，"动作片"无论和"用户年龄"交互还是和"时段"交互，用的是同一个向量。但直觉上"年龄 x 类目"和"时段 x 类目"是两件事。FFM 给每个特征发一组 embedding，每对应一个对方 field 一个。

$$\hat{y}(\mathbf{x}) = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_{i, f_j}, \mathbf{v}_{j, f_i} \rangle x_i x_j.$$

记号 $\mathbf{v}_{i, f_j}$ 读作"特征 $i$ **在与 field $f_j$ 交互时**的 embedding"。

### 实现

```python
class FFM(nn.Module):
    """Field-aware Factorization Machine。"""

    def __init__(self, field_dims, num_fields, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = num_fields
        self.linear = nn.Linear(sum(field_dims), 1)
        # 每个特征对每个对方 field 都有一个 embedding
        self.embeddings = nn.ModuleList([
            nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(num_fields)])
            for dim in field_dims
        ])

    def forward(self, x):
        b = x.size(0)
        linear_out = self.linear(self._one_hot(x))

        interaction = torch.zeros(b, 1, device=x.device)
        for i in range(len(self.field_dims)):
            for j in range(i + 1, len(self.field_dims)):
                v_i_fj = self.embeddings[i][j](x[:, i])  # i 与 field j 交互的 emb
                v_j_fi = self.embeddings[j][i](x[:, j])  # j 与 field i 交互的 emb
                interaction += (v_i_fj * v_j_fi).sum(dim=1, keepdim=True)

        return torch.sigmoid(linear_out + interaction)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

### FFM vs FM 的取舍

| 维度 | FM | FFM |
|---|---|---|
| 参数量 | $O(d \cdot k)$ | $O(d \cdot F \cdot k)$，$F$ 为 field 数 |
| 表达力 | 所有交互共享同一 embedding | Field-aware |
| 领域知识 | 不需要 | 需要 field 划分 |
| 典型用途 | 第一基线 | 早期 Criteo / Avazu Kaggle 冠军方案 |

两者都止步于二阶。要往高阶走，有两条路：**堆非线性**（深度网络），或者**显式构造**（CIN、Cross）。DeepFM 走前者；xDeepFM 和 DCN 走后者。

继续之前，下面这张图是后续文章用到的"交互算子"一览：

![FM、FFM、DeepFM、DCN、AutoInt 各自使用的交互算子并排对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig2_interaction_methods.png)

---

## DeepFM：把 FM 与深度学习接起来

DeepFM（华为，2017）几乎可以毫无疑问地被称为深度 CTR 模型的默认起点。它的想法结构上极简：FM 与深度网络**并联**，共享同一张 Embedding 表。

### 为什么这种组合成立

- **FM 分支**显式捕捉二阶（低阶）交互。
- **Deep 分支**通过堆叠非线性隐式捕捉高阶交互。
- **共享 Embedding**让参数量减半，并强制两条分支对每个特征"是什么意思"达成一致。

> **类比：** 同一个案子上的两位侦探。FM 是规则型选手，擅长简单线索（"这两个特征总是和点击共现"）；深度 MLP 是模式匹配型，能挖出又长又模糊的证据链。两人各打分数，最后求和。

架构图把"并联"这件事讲得很直白：

![DeepFM 架构：共享 Embedding 喂给并联的 FM 与 Deep 分支，求和后过 sigmoid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig3_deepfm_arch.png)

### 数学形式

$$\hat{y}(\mathbf{x}) = \sigma\big(y_{\text{FM}} + y_{\text{Deep}}\big),$$

其中 $y_{\text{FM}}$ 是标准 FM 表达式，$y_{\text{Deep}}$ 是把所有 embedding 拼接后过 MLP：

$$\mathbf{h}_0 = [\mathbf{v}_1; \mathbf{v}_2; \ldots; \mathbf{v}_m], \quad \mathbf{h}_l = \text{ReLU}(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad y_{\text{Deep}} = \mathbf{w}^\top \mathbf{h}_L + b.$$

### 实现

```python
class DeepFM(nn.Module):
    """DeepFM：FM 与深度网络并联，共享 embedding 表。"""

    def __init__(self, field_dims, embed_dim=16, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )

        # FM 分支
        fm_linear = self.linear(self._one_hot(x))
        sum_sq = torch.sum(embs, dim=1) ** 2
        sq_sum = torch.sum(embs ** 2, dim=1)
        fm_interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
        fm_out = fm_linear + fm_interaction

        # Deep 分支（用同一份 embedding）
        deep_out = self.mlp(embs.view(embs.size(0), -1))

        return torch.sigmoid(fm_out + deep_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

DeepFM 是**默认基线**。要起一个新 CTR 系统，从这里开始；分别消融掉 FM 分支和 Deep 分支，只有当任一消融都掉 AUC 时再考虑更复杂的东西。

下两个模型都源于一个诚实的观察：深度 MLP **隐式**学交互，你看不出它学到了什么。这催生了 xDeepFM（CIN）与 DCN（Cross），二者都把高阶结构显式化。

---

## xDeepFM：显式高阶特征交互

xDeepFM（eXtreme Deep Factorization Machine，2018）引入了 **CIN（Compressed Interaction Network）**，按层在 embedding 空间逐层构建高阶交互。

### CIN 怎么工作

把 CIN 想成一座金字塔：

- **第 0 层：** 原始 embedding（一阶）。
- **第 1 层：** 第 0 层每个特征与原始 embedding 逐元素相乘（二阶）。
- **第 2 层：** 第 1 层每个特征再与原始 embedding 逐元素相乘（三阶）。
- ……

每一层的相乘后跟一个可学习的卷积压缩：

$$\mathbf{X}^k_{h, *} = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} W^{k,h}_{i,j}\big(\mathbf{X}^{k-1}_{i,*} \circ \mathbf{X}^0_{j,*}\big),$$

其中 $\circ$ 是 Hadamard（逐元素）积，$W$ 是可学习权重。

> **大白话。** "把上一层每张 feature map 与原始 embedding 逐元素相乘，再用一次 1x1 卷积把所有交叉压回到固定数量的 feature map。逐层堆叠。"

完整的 xDeepFM 是 **Linear + CIN + Deep MLP** 三塔结构，在 sigmoid 前求和。

### 实现

```python
class CIN(nn.Module):
    """xDeepFM 用的 Compressed Interaction Network。"""

    def __init__(self, num_fields, embed_dim, cin_layer_sizes=(100, 100)):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.cin_layers = nn.ModuleList()
        prev_size = num_fields
        for layer_size in cin_layer_sizes:
            self.cin_layers.append(
                nn.Conv1d(prev_size * num_fields, layer_size, kernel_size=1)
            )
            prev_size = layer_size

    def forward(self, embeddings):
        """embeddings: [batch, num_fields, embed_dim]"""
        b = embeddings.size(0)
        X_0 = embeddings
        X_k = X_0
        outputs = []
        for cin_layer in self.cin_layers:
            H = X_k.size(1)
            inter = X_k.unsqueeze(2) * X_0.unsqueeze(1)        # [b, H, m, D]
            inter = inter.view(b, H * self.num_fields, self.embed_dim)
            X_k = torch.relu(cin_layer(inter))                   # [b, layer, D]
            outputs.append(X_k.sum(dim=2))                       # 在 D 维上 sum-pool
        return torch.cat(outputs, dim=1)


class xDeepFM(nn.Module):
    """xDeepFM = Linear + CIN + Deep MLP（基于共享 embedding）。"""

    def __init__(self, field_dims, embed_dim=16,
                 cin_layer_sizes=(100, 100), mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        self.cin = CIN(self.num_fields, embed_dim, cin_layer_sizes)
        self.cin_proj = nn.Linear(sum(cin_layer_sizes), 1)

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        linear_out = self.linear(self._one_hot(x))
        cin_out = self.cin_proj(self.cin(embs))
        deep_out = self.mlp(embs.view(embs.size(0), -1))
        return torch.sigmoid(linear_out + cin_out + deep_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

### xDeepFM vs DeepFM

| 维度 | DeepFM | xDeepFM |
|---|---|---|
| 低阶交互 | 显式（FM） | 显式（FM + CIN） |
| 高阶交互 | 隐式（仅 MLP） | 显式（CIN）+ 隐式（MLP） |
| 可解释性 | 弱 | 较好——可以探查 CIN feature map |
| 推理成本 | 较低 | 较高（CIN 占主要开销） |
| 选用场景 | 默认起点 | DeepFM 已经接近收敛、数据足够丰富时 |

---

## Deep & Cross Network（DCN）：有界阶交叉特征

DCN（Google，2017）走了另一条路。它不是像 CIN 那样堆叠"逐元素相乘 + 可学习卷积"，而是引入一个非常小的模块——**Cross Network**，每多一层就把交互的多项式阶数精确加 1，参数只增加 $O(d)$。

### Cross 层

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \cdot (\mathbf{w}_l^\top \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l.$$

> **大白话。** "对当前状态做一个标量投影，乘到**原始**输入向量上，加偏置，加残差。" 每一步都把 $\mathbf{x}_0$ 再次注入，使交互阶数加 1。

经过 $L$ 层 Cross，模型学到的是原始特征的 $L+1$ 阶多项式——但 Cross 部分总共只用了 $L \cdot d$ 个参数。

下图展示了两件事：每一层 Cross 怎么提阶，以及它相比朴素多项式展开有多便宜。

![左：每一层 Cross 都把 x0 注入一次，使多项式阶数 +1。右：参数代价随阶数变化——DCN 线性增长，朴素多项式爆炸](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig4_dcn_cross.png)

右图是关键。100 维输入下做 6 阶交叉，朴素多项式展开需要 $10^{12}$ 个参数，Cross Network 只要 600 个。

### 实现

```python
class CrossNetwork(nn.Module):
    """Cross Network：有界阶交互，每层参数线性于 d。"""

    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x):
        x_0 = x
        x_l = x
        for layer in self.layers:
            x_l_w = layer(x_l)              # 标量投影 [batch, 1]
            x_l = x_0 * x_l_w + x_l         # 广播乘 + 残差
        return x_l


class DCN(nn.Module):
    """Deep & Cross Network。"""

    def __init__(self, field_dims, embed_dim=16,
                 cross_layers=3, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        input_dim = self.num_fields * embed_dim

        self.cross_net = CrossNetwork(input_dim, cross_layers)

        layers = []
        prev = input_dim
        for dim in mlp_dims:
            layers += [
                nn.Linear(prev, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        self.final = nn.Linear(input_dim + 1, 1)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        flat = embs.view(embs.size(0), -1)

        cross_out = self.cross_net(flat)     # [batch, input_dim]
        deep_out = self.mlp(flat)            # [batch, 1]

        combined = torch.cat([cross_out, deep_out], dim=1)
        return torch.sigmoid(self.final(combined))
```

**DCN 的优点。**

- 显式、**有界**的交互阶数——上线时没有意外。
- Cross 部分相对深度 MLP 极小，延迟接近一个普通 MLP。
- 在 Google 大规模线上验证；后续 v2 提出 DCN-Mix，进一步提升容量。

---

## AutoInt：把注意力当成特征交互引擎

AutoInt（2019）把 Transformer 的核心引擎——**多头自注意力**——搬到了特征交互上。核心主张：不是所有交互都同等重要，注意力可以学到**该把注意力放在哪些特征对上**，多个 head 学多种"相关"的含义。

### 怎么工作

把每个 field 的 embedding 当成一个 token，投影出 query / key / value：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}.$$

> **大白话。** "每个特征问'我应该从谁的 embedding 里读信息？'（Q），公示自己掌握什么（K），并提供可被聚合的内容（V）。Softmax 给出路由权重。"

带 $H$ 个 head 的模型可以并行学到 $H$ 套不同的"相关"概念；堆叠 $L$ 层 AutoInt block，则让信息流动多次，构造更深的组合。

### 实现

```python
import numpy as np


class MultiHeadAttention(nn.Module):
    """AutoInt block 内部使用的多头自注意力。"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: [batch, num_fields, embed_dim]"""
        B, N, D = x.size()
        residual = x
        x = self.norm(x)

        def reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = reshape(self.W_q(x)), reshape(self.W_k(x)), reshape(self.W_v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.dropout(self.W_o(out)) + residual


class AutoInt(nn.Module):
    """AutoInt：在 field embedding 上堆叠多头自注意力。"""

    def __init__(self, field_dims, embed_dim=16, num_attn_layers=3,
                 num_heads=4, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])

        mlp_input = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        h = embs
        for layer in self.attn_layers:
            h = layer(h)

        linear_out = self.linear(self._one_hot(x))
        mlp_out = self.mlp(h.view(h.size(0), -1))
        return torch.sigmoid(linear_out + mlp_out)

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

**AutoInt 的优点。**

- 不需要人工 schema 就能发现重要交互。
- 注意力权重**可被检视**，方便调试与汇报。
- 多头结构天然适合捕捉多种"相关"模式。

---

## FiBiNet：特征重要性 + 双线性交互

FiBiNet（2019）攻击的是其它模型默默假定的两件事：

1. **所有特征同等重要。** 才不是。有的特征是金矿，有的是噪声。FiBiNet 用 **SENet** 学习每个 field 的重要性门控。
2. **逐元素积足够表达交互。** 也未必。FiBiNet 用**双线性**形式替换 Hadamard 积，可以建模不对称、更丰富的交互。

### SENet：学习特征重要性

三步：

- **Squeeze。** 对每个 field 的 embedding 沿 embedding 维取均值——每个 field 一个标量重要性。
- **Excitation。** 一个两层（带瓶颈）MLP 把这些标量映射成每个 field 的门控权重。
- **Reweight。** 把每个 embedding 乘上对应门控。

> **类比：** 一位 DJ 根据当前播放内容动态调节每个轨道的音量推子。

### 双线性交互

把 $\mathbf{v}_i \odot \mathbf{v}_j$ 换成 $\mathbf{v}_i^\top \mathbf{W} \mathbf{v}_j$，其中 $\mathbf{W}$ 是可学习矩阵。变体有：所有 field 对共享 $\mathbf{W}$（Field-All）、每个 field 一个 $\mathbf{W}$（Field-Each）、每对 field 一个 $\mathbf{W}$（Field-Interaction）。

### 实现

```python
class SENet(nn.Module):
    """field 维度上的 Squeeze-and-Excitation 门控。"""

    def __init__(self, num_fields, reduction=4):
        super().__init__()
        reduced = max(1, num_fields // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced),
            nn.ReLU(),
            nn.Linear(reduced, num_fields),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [batch, num_fields, embed_dim]"""
        z = x.mean(dim=2)                            # squeeze
        weights = self.excitation(z).unsqueeze(2)    # excite
        return x * weights                           # reweight


class BilinearInteraction(nn.Module):
    """共享一个 W 的双线性交互。"""

    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x):
        n = x.size(1)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                vi_W = torch.matmul(x[:, i:i + 1, :], self.W)
                out.append((vi_W * x[:, j:j + 1, :]).squeeze(1))
        return torch.stack(out, dim=1)


class FiBiNet(nn.Module):
    """Feature Importance and Bilinear feature Interaction Network。"""

    def __init__(self, field_dims, embed_dim=16, mlp_dims=(128, 64), dropout=0.2):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)

        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        self.linear = nn.Linear(sum(field_dims), 1)
        self.senet = SENet(self.num_fields)
        self.bilinear = BilinearInteraction(embed_dim)

        num_pairs = self.num_fields * (self.num_fields - 1) // 2
        mlp_input = self.num_fields * embed_dim * 2 + num_pairs * embed_dim
        layers = []
        for dim in mlp_dims:
            layers += [
                nn.Linear(mlp_input, dim), nn.BatchNorm1d(dim),
                nn.ReLU(), nn.Dropout(dropout),
            ]
            mlp_input = dim
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(self.num_fields)], dim=1
        )
        linear_out = self.linear(self._one_hot(x))
        senet_embs = self.senet(embs)
        bilinear_out = self.bilinear(embs)

        mlp_in = torch.cat([
            embs.view(embs.size(0), -1),
            senet_embs.view(senet_embs.size(0), -1),
            bilinear_out.view(bilinear_out.size(0), -1),
        ], dim=1)
        return torch.sigmoid(linear_out + self.mlp(mlp_in))

    def _one_hot(self, x):
        b = x.size(0)
        oh = torch.zeros(b, sum(self.field_dims), device=x.device)
        offset = 0
        for i, dim in enumerate(self.field_dims):
            oh.scatter_(1, x[:, i:i + 1] + offset, 1)
            offset += dim
        return oh
```

---

## 模型对比与选择指南

每个工程师真正想问的是：**这些花活儿，AUC 真的提了吗？**

下图汇总了 Criteo 类基准上的典型相对排序。绝对数值是示意——不同数据集、Embedding 维度、训练预算下都会变——但**差距格局**在已发表的报告里相当一致。

![柱状图对比 LR、FM、FFM、DeepFM、xDeepFM、DCN、AutoInt、FiBiNet 在 Criteo 类基准上的 AUC 与 Logloss](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR%E9%A2%84%E4%BC%B0%E4%B8%8E%E7%82%B9%E5%87%BB%E7%8E%87%E5%BB%BA%E6%A8%A1/fig5_auc_logloss.png)

两个观察比绝对数字更重要：

1. **最大单步跳跃发生在 LR -> FM。** 哪怕只是廉价地加上二阶交互，带来的 AUC 提升也比之后任何架构改动都大。
2. **DeepFM 之后的所有模型都挤在 ~0.005 AUC 的窄带里。** 听起来很小。在 Google / Meta 规模上，0.5 milli-AUC 就是真金白银；但在百万用户的初创公司，它在噪声里——特征质量与新鲜度才是决胜场。

### 计算复杂度

| 模型 | 参数量 | 训练速度 | 推理速度 |
|---|---|---|---|
| LR | $O(d)$ | 极快 | 极快 |
| FM | $O(d \cdot k)$ | 快 | 快 |
| FFM | $O(d \cdot F \cdot k)$ | 中 | 中 |
| DeepFM | $O(d \cdot k + \text{MLP})$ | 中 | 中 |
| xDeepFM | $O(d \cdot k + \text{CIN} + \text{MLP})$ | 慢 | 中 |
| DCN | $O(d \cdot k + L \cdot d + \text{MLP})$ | 中 | 中 |
| AutoInt | $O(d \cdot k + L \cdot \text{Attn} + \text{MLP})$ | 中 | 中 |
| FiBiNet | $O(d \cdot k + \text{SE} + \text{Bilinear} + \text{MLP})$ | 慢 | 中 |

### 特征交互能力

| 模型 | 低阶 | 高阶 | 显式 | 隐式 |
|---|---|---|---|---|
| LR | 仅线性 | 否 | 否 | 否 |
| FM | 二阶 | 否 | 是 | 否 |
| FFM | 二阶（field-aware） | 否 | 是 | 否 |
| DeepFM | 二阶 | 是 | 是（FM） | 是（DNN） |
| xDeepFM | 二阶 | 有界 | 是（CIN） | 是（DNN） |
| DCN | 有界阶 | 是 | 是（Cross） | 是（DNN） |
| AutoInt | 任意阶 | 是 | 是（Attention） | 是（DNN） |
| FiBiNet | 双线性二阶 | 是 | 是（Bilinear） | 是（DNN） |

### 一份真正能用的决策清单

- **第一版 / POC。** 用 **LR** 或 **FM**。先把数据 pipeline、评估、上线打通，再加层数。
- **第一版"真"模型。** **DeepFM**。表格里性价比最高的架构。
- **DeepFM 已经平台期、且有 GPU 预算。** 试 **DCN**（更便宜）或 **xDeepFM**（更丰富），不要同时上。
- **field 异质、且想看到交互权重。** **AutoInt**。
- **特征列表又长又乱、且怀疑特征重要性差异大。** **FiBiNet**。
- **极低延迟 / 边缘部署。** 线上 **LR / FM**；离线再用复杂模型做重排或召回 bootstrap。

---

## 训练策略与最佳实践

### 处理类别不平衡

CTR 数据极度不平衡，三个稳妥工具：

**1. 加权 BCE Loss。**

```python
# pos_weight 在损失里上调少数（正）类的权重
pos_weight = torch.tensor([num_negatives / num_positives])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**2. 负采样。** Facebook 这种规模上的标准做法。但要记住：负采样会让预测概率失真，上线前必须重新校准。

```python
import random

def sample_negatives(positives, item_pool, user_history_fn, k=4):
    out = []
    for user_id, pos_item in positives:
        candidates = item_pool - set(user_history_fn(user_id))
        for neg in random.sample(list(candidates), min(k, len(candidates))):
            out.append((user_id, neg, 0))
    return out
```

**3. Focal Loss。** 降低简单样本的权重，让梯度集中在少数难样本上。

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.where(targets == 1, preds, 1 - preds)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

### 正则化

- MLP 层使用 **Dropout 0.2-0.5**；不要直接对 embedding 查表加 dropout。
- 稠密权重用 **L2**（weight decay 1e-5 ~ 1e-6）；embedding 通常需要的正则比稠密权重少。
- 在验证 AUC 上做 **早停**，patience 3-10 个 epoch。

```python
def train_with_early_stopping(model, train_loader, val_loader,
                              epochs=100, patience=10):
    best_loss, wait, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)
        if val_loss < best_loss:
            best_loss, wait, best_state = val_loss, 0, model.state_dict().copy()
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model
```

### 评估指标

**AUC-ROC** 是头号指标。它衡量"随机一个正样本得分高于随机一个负样本"的概率——构造上对类别不平衡免疫。

```python
from sklearn.metrics import roc_auc_score, log_loss


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).squeeze().cpu().numpy())
            labels.extend(y.cpu().numpy())
    return {
        'AUC': roc_auc_score(labels, preds),
        'LogLoss': log_loss(labels, preds),
    }
```

**校准（Calibration）同样重要，对下游竞价场景甚至比 AUC 更关键。** 预测 CTR = 0.05 应该在那个概率桶里对应大约 5% 的真实点击率。用 `sklearn.calibration.calibration_curve` 画可靠性图；系统性高估 / 低估时用 **Platt scaling** 或 **isotonic regression** 校准后再上线。

---

## 常见问题

### Q1：CTR 预估为什么是二分类而不是回归？

目标本身就是一个概率（点击的几率），Bernoulli 是正确的似然。二分类有成熟的指标（AUC、Logloss），优雅地处理不平衡，输出是 $[0, 1]$ 的可解释分数。回归用于点击次数 / 收入 / 观看时长有时是合适的，但单看是否点击，BCE 是标准选择。

### Q2：Embedding 维度怎么选？

从 16 开始。小数据（< 100 万样本）4-8 通常足够；大数据（> 1 亿样本）尝试 16-64。做一个简单消融：维度翻倍若 AUC 涨幅小于 0.001，回到小维度。Embedding 表通常是模型显存与服务成本的大头。

### Q3：FM 与矩阵分解（MF）的区别？

MF 把**单一**的"用户-物品"评分矩阵分解成用户与物品 embedding。FM 严格更一般：它对**任意特征**之间的二阶交互做因子化，可以把人口学、城市、时段等附加信息全部塞进同一个因子化形式里。MF 相当于只有两个 field 的 FM。

### Q4：DeepFM 与 xDeepFM 怎么选？

默认 **DeepFM**。只有当 DeepFM 已经清晰平台期、并且数据丰富到三阶四阶交互真的有意义时，再上 xDeepFM。CIN 部分会让推理成本接近翻倍。

### Q5：物品冷启动怎么办？

通常组合四件事：(1) 用内容特征（文本 / 图像编码器）初始化 embedding；(2) 前几次曝光走热门 fallback；(3) 用上下文 bandit 主动探索，让新物品获得**一些**曝光；(4) 在相关任务上预训练 embedding。准则是：**永远不要让模型只看到新物品的 ID。**

### Q6：特征工程 vs 模型架构，哪个更重要？

几乎总是特征工程，2-3 倍领先。好的交叉特征、合理的分桶、缺失值处理、用户/物品的滑窗统计，通常带 10-30% AUC 提升；在深度 CTR 家族里换架构通常 2-10%。先把特征工程做好，再上更花哨的架构。

### Q7：缺失特征怎么处理？

四个选项：(1) 默认值（0、均值、众数）；(2) 加一个 `is_missing` 二值指示位；(3) 类别特征保留一个特殊"缺失" embedding；(4) 用 KNN 或简单模型插补。选哪一个取决于**缺失本身是不是信息**——比如未登录用户缺人口学信息，这件事本身预测力就很强，应该当成特征。

### Q8：离线评估 vs 在线评估？

**离线：** 时间切分的训练 / 验证 / 测试（**绝对不要**随机切！）。指标：AUC、Logloss。便宜快速，但下游效应（多样性、新鲜度、位置偏置）看不见。**在线：** 真实用户的 A/B 测试。指标：实际 CTR、转化、收入、留存。慢且贵，但是唯一权威信号。永远先离线验证，再上线 A/B。

### Q9：怎么把 CTR 模型部署上线？

四件事：(1) 用 TorchServe / Triton / TF-Serving 提供批量服务；(2) 通过 INT8 量化、Embedding 分片、预取，把 p99 控在 10 ms 内；(3) 监控预测 CTR 的分布漂移——直方图一变就重训或回滚；(4) 模型与**特征 pipeline 一起**版本化——一次 schema 错位就能把 AUC 静默打残。

### Q10：2024-2025 的趋势？

大规模 Transformer 化的交互堆栈、多任务学习（CTR + 转化 + 时长联合建模）、用户-物品图上的 GNN、AutoML 自动搜索 embedding 维度与架构、基于因果与 IPS 的去偏、保护隐私的联邦学习。但基本功——特征质量、交互建模、校准、新鲜度——仍然是不论流行什么趋势都最值得投入的杠杆。

---

## 总结

CTR 预估是现代排序的核心。我们沿时间线走完了从一行线性层到注意力交互发现的演进：

1. **LR** —— 简单、有校准、对特征交互无知。
2. **FM / FFM** —— 在稀疏数据上自动二阶交互；FFM 用更多参数换 field 感知。
3. **DeepFM** —— 工业基座：显式二阶（FM） + 隐式深层（DNN），共享一张 Embedding 表。
4. **xDeepFM** —— 通过 CIN 显式高阶交互。
5. **DCN** —— 参数线性增长的有界阶多项式交叉。
6. **AutoInt** —— 多头自注意力做交互发现与可解释性。
7. **FiBiNet** —— 可学的特征重要性（SENet）+ 双线性交互。

**实战要点。**

- **从简单开始。** LR -> FM -> DeepFM，按这个顺序。AUC 不再涨的那一刻就停。
- **特征第一，架构第二。** 一个新交叉特征往往胜过一个新模型。
- **认真处理不平衡。** 在 pos_weight、负采样+校准、Focal Loss 之间挑一个，坚持到底。
- **诚实地评估。** 时间切分的离线评估，A/B 测试的在线评估，AUC 之外同时看校准。
- **永远迭代。** CTR 系统永远做不完——分布会漂、物品会换、昨天的模型就是今天的基线。

"最好的"模型，是在**你**的延迟预算、**你**的数据上，赢下**你**的 A/B 测试的那一个。先理解问题，再挑解决问题的最小工具。

---

> **系列导航**
>
> 本文是推荐系统系列的**第 4 篇**，共 16 篇。
>
> - [第 1 篇：入门与基础概念](/zh/推荐系统-一-入门与基础概念/)
> - [第 2 篇：协同过滤与矩阵分解](/zh/推荐系统-二-协同过滤与矩阵分解/)
> - [第 3 篇：深度学习基础模型](/zh/推荐系统-三-深度学习基础模型/)
> - **第 4 篇：CTR预估与点击率建模**（当前）
> - [第 5 篇：Embedding表示学习](/zh/推荐系统-五-Embedding表示学习/)
> - [第 6 篇：序列推荐与会话建模](/zh/推荐系统-六-序列推荐与会话建模/)
> - [第 7 篇：图神经网络与社交推荐](/zh/推荐系统-七-图神经网络与社交推荐/)
> - [第 8 篇：知识图谱增强推荐系统](/zh/推荐系统-八-知识图谱增强推荐系统/)
> - 第 9 篇：多任务学习（即将发布）
> - 第 10 篇：深度兴趣网络（即将发布）
> - 第 11 篇：对比学习（即将发布）
> - 第 12 篇：大语言模型推荐（即将发布）
> - 第 13 篇：公平性与可解释性（即将发布）
> - 第 14 篇：跨域推荐与冷启动（即将发布）
> - 第 15 篇：实时推荐与在线学习（即将发布）
> - 第 16 篇：工业实践（即将发布）
