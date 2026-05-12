---
title: "推荐系统（四）—— CTR预估与点击率建模"
date: 2025-12-10 09:00:00
tags:
  - Recommendation Systems
  - CTR Prediction
  - Deep Learning
  - Feature Interactions
categories: 推荐系统
series: recommendation-systems
lang: zh
mathjax: true
description: "CTR预估模型全面解析：从Logistic Regression到FM/FFM，再到DeepFM、xDeepFM、DCN、AutoInt、FiBiNet等深度学习模型，附PyTorch实现与训练策略。"
disableNunjucks: true
series_order: 4
translationKey: "recommendation-systems-4"
---
每次你刷信息流、点击商品推荐或观看推荐视频时，背后都有一个 CTR（点击率）模型在决定给你展示什么——这个模型要回答的问题看似简单：

> **"这个用户此时此刻点击这个物品的概率是多少？"**

这个问题背后是机器学习领域最具经济价值的挑战之一——CTR 提升 1% 在 Google、Amazon、阿里这类规模的平台上可带来数百万美元的额外收益；同一套模型也驱动着信息流、应用商店、新闻 App 和社交 App 等多种推荐场景。 CTR 预估是推荐系统**排序阶段**的核心：召回阶段会生成数千个候选物品， CTR 模型则负责从中筛选出最终展示给用户的十余个。

这篇文章带你回顾过去十年 CTR 模型的演进历程——从简单的 Logistic Regression 到基于注意力机制的复杂架构。我不仅会进行公式推导，还会围绕每个模型聚焦三个关键问题：

1. **上一代模型的哪些缺陷催生了这个设计？**
2. **它的几何或概率直观解释是什么？**
3. **如何实现并将其部署到生产环境？**

读完本文，你将能读懂主流 CTR 论文，默画出核心模型架构，并为实际系统选型提供合理基线。
## 你将学到什么

- CTR 预估问题的本质，以及它**为什么**特别难（不只是标签不平衡的分类问题）
- **Logistic Regression** 作为基线和合理性检查——明确它会在哪里失效
- **Factorization Machines (FM)** 和 **Field-aware FM (FFM)**：自动学习稀疏数据上的二阶特征交互
- **DeepFM**：工业界的主力模型，结合了 FM 和深度网络
- **xDeepFM**：通过 Compressed Interaction Network 显式建模高阶特征交互
- **DCN**：用线性参数复杂度实现有界阶特征交叉
- **AutoInt**：用自注意力机制处理特征交互
- **FiBiNet**：用 SENet 学习哪些特征更重要，并引入更丰富的双线性交互
- 训练中的实际问题：类别不平衡、模型校准、 AUC 和 Logloss 的权衡、如何在 A/B 测试前进行离线评估
## 前置知识

- 熟练掌握 Python 和 PyTorch （`nn.Module`、训练循环、嵌入）
- 了解深度学习基本概念，熟悉类别特征的嵌入视角（[Part 3](/zh/recommendation-systems/03-深度学习基础模型)）
- 清楚二分类、 Sigmoid 和交叉熵的基本原理

---
## 理解 CTR 预估问题

### 什么是 CTR 预估？

CTR 预估是一个具有极端结构特性的二分类问题。给定用户、物品和上下文，目标是估计点击概率：

$$P(y = 1 \mid \mathbf{x}), \quad y \in \{0, 1\},\;\; 1 = \text{点击}.$$
特征向量 $\mathbf{x}$ 是由三类信息拼接而成的：

| 类别 | 示例 |
|---|---|
| 用户特征 | user_id、年龄段、性别、历史行为、国家 |
| 物品特征 | item_id、品牌、类目、价格段、新鲜度 |
| 上下文特征 | 小时、设备、网络、查询词、展示位 |

实际中， CTR 定义为点击数除以曝光数：$\text{CTR} = \text{点击数} / \text{曝光数}$。模型输出用于对候选物品排序、过滤低质物品，并支撑下游业务目标（例如广告中的 eCPM = CTR x 出价，或信息流中的多目标加权分数）。

### 为什么 CTR 预估很难？

CTR 预估表面上是一个标准的分类任务，但实际上完全不同，主要有以下五个原因：

**1. 极端类别不平衡。** 展示广告的点击率通常在 0.1%-2%，电商在 1%-5%，信息流在 2%-10%。一个‘永远预测不点击’的模型准确率可达 95% 以上，却完全不具备实用价值。因此， AUC 和 Logloss 取代了准确率作为评估指标。

**2. 高维稀疏特征。** 经过 one-hot 编码后，特征空间维度达到 $10^6$ 到 $10^9$，而每个样本只激活其中几十个维度。为每对特征存储一个权重是不可能的。

**3. 有效信号往往蕴藏在特征交互之中。** 单独的“年轻用户”是一个弱信号，而“年轻用户 x 动作片 x 晚上”则是强信号。如何自动、高效地建模这些特征交互，是 CTR 建模的核心挑战。

**4. 数据分布持续变化。** 新物品上架、爆款涌现、工作日与周末的周期性变化等，均会导致数据分布发生漂移。模型需每日甚至每小时更新，仅依赖离线 AUC 评估难以反映真实效果。

**5. 严格的延迟约束。** 排序模块需要在不到 100 毫秒（通常 p99 在 10 毫秒以内）内对上千个候选物品打分。模型大小、 Embedding 查找效率和批处理能力与架构设计同等重要。

### CTR 预估的工业 Pipeline

从原始点击日志到排序结果，再到模型重训，整个端到端流程如下：

![CTR 端到端 pipeline：原始日志、特征工程、Embedding、模型、排序与 A/B、反馈闭环](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig6_pipeline.png)

Pipeline 中有几个关键点需要注意：

- **特征工程仍是核心战场。** Embedding 能学到的信息有限，显式交叉特征和统计特征（如用户、物品、位置的滑窗 CTR）往往是带来最大 A/B 提升的关键。
- **Embedding 表是共享的基础设施。** 所有深度 CTR 模型（FM、 DeepFM、 xDeepFM、 DCN、 AutoInt、 FiBiNet）都使用同一张 Embedding 表，不同架构的核心差异，在于如何组织 Embedding 之间的交互。
- **在线反馈形成闭环。** 昨天的服务日志就是今天的训练数据。模型的新鲜度往往比复杂度更重要。

带着这张地图，我沿着时间线梳理一下架构的发展历程。
## Logistic Regression：推荐系统的起点，也是 FM 诞生的原因

尽管生产环境中广泛部署着大型神经网络，Logistic Regression（LR）仍被广泛用作基线模型，既是推荐系统的通用基线和校准锚点，也在延迟敏感场景中承担实际打分任务。

### 它是怎么工作的？

LR 将点击概率建模为一个线性函数通过 Sigmoid 激活后的结果：
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}.$$
> **简单来说：** "对所有特征加权求和，加上偏置，再压缩到 [0,1] 区间。"

训练时，最小化二元交叉熵损失。
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\big].$$
### 为什么 LR 既经典又不够用？

从几何上看， LR 的局限性一目了然。它只能在特征空间中学习一个超平面，无法处理“特征 A 只有在特征 B 同时激活时才有用”的模式。最典型的例子是 XOR 形状的点击行为：

![左图：LR 无法用一条直线分开 XOR 形点击数据。右图：加入交互项后结构得以恢复](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig1_lr_limitation.png)

左图中，“年轻+动作片”和“老年+喜剧”会点击，但“年轻+喜剧”和“老年+动作片”不会。无论参数如何调整，线性决策边界都无法正确划分这些样本，此时 AUC 接近 0.5。右图中，加入一个交互项 $x_1 \cdot x_2$，问题迎刃而解。此后所有 CTR 模型的核心设计目标，均可归结为：

> **“如何自动、高效地发现并表达有用的特征交叉？”**

### 实现代码

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class LogisticRegression(nn.Module):
    """用于 CTR 预估的 Logistic Regression 模型。"""

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

### LR 的具体短板

1. **缺乏特征交互能力。** 所有特征被视为独立，彼此之间没有关联。
2. **依赖人工特征工程。** 如果想捕捉“年龄 x 类目”这样的交互，必须手动构造特征列。超过二阶或三阶的交叉几乎无法实现。
3. **决策边界是线性的。** 上图已经说明， LR 对于 XOR 这类非线性结构完全无能为力。

这三个缺陷直接推动了后续模型的发展，如 FM 的出现。

---
## Factorization Machines （FM）：自动二阶交互

Steffen Rendle 在 2010 年提出的 FM，是第一个让稀疏数据上的“自动二阶交互”既高效又实用的模型。

### 核心洞察

如果用朴素的“带交互的 LR”，需要为每对特征学习一个权重 $w_{ij}$。假设特征数是 $d$，那么参数量就是 $O(d^2)$。但问题在于：绝大多数特征对在训练集中从未共现，导致对应权重无法学习。

FM 的解决办法是，用两个低维向量的内积代替每对特征的权重：
$$w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}.$$
> **类比：** 假设有 1000 部电影。如果为每对电影存一个权重，需要一百万个数，而绝大多数组合从未见过。换个思路：给每部电影分配一个 $k$ 维“性格向量”。两部电影的交互强度由其隐向量的内积决定。这样只需要 $1000 \cdot k$ 个数，而且即使某对电影从未在训练集中共现，也能预测它们的交互强度——因为每个向量是从大量其他共现样本中学到的。

FM 的关键优势在于，能对未在训练数据中共同出现的特征对进行有效泛化。这使其在极端稀疏场景下仍保持有效性，而决策树与线性模型在此类场景中通常表现不佳。

### 数学形式

FM 的预测公式如下：
$$\hat{y}(\mathbf{x}) = \underbrace{w_0}_{\text{偏置}} + \underbrace{\sum_{i=1}^{d} w_i x_i}_{\text{一阶}} + \underbrace{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j}_{\text{二阶交互}}.$$
表面上看，二阶项的复杂度是 $O(d^2)$，但实际上可以通过一个巧妙的公式降到 $O(k \cdot d)$：
$$\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\left[\Big(\sum_i \mathbf{v}_i x_i\Big)^2 - \sum_i (\mathbf{v}_i x_i)^2\right].$$
> **为什么成立？** 求和平方会包含所有 $i \cdot j$ 项（包括 $i=j$ 的情况）；减去平方求和去掉对角线项；最后除以 2 消除重复计数。

### 实现

```python
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    """用于 CTR 预测的 Factorization Machine（基于 field 索引）。"""

    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.linear = nn.Linear(sum(field_dims), 1)
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

    def forward(self, x):
        """x: [batch, num_fields]，每列是一个 field 的类别索引。"""
        # 一阶部分：在 one-hot 视图上做线性变换
        linear_out = self.linear(self._one_hot(x))

        # 取出每个 field 的 embedding：[batch, num_fields, embed_dim]
        embs = torch.stack(
            [self.embedding[i](x[:, i]) for i in range(len(self.field_dims))],
            dim=1,
        )

        # 高效计算二阶交互（(sum^2 - sum_of_squares)/2 技巧）
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

**优点：**  
- 自动捕捉二阶交互  
- 计算复杂度仅为 $O(kd)$  
- 对未见特征对有泛化能力  

**缺点：**  
- 只能捕捉二阶交互  
- 一个特征不管和谁交互，都用同一个 embedding——这有时是错的。这一观察直接催生了 FFM。

---
## Field-aware Factorization Machines (FFM)

FFM （2016）在 FM 基础上的关键改进是：每个特征针对不同 field，维护一组独立的嵌入向量。

### 核心思想

在 FM 中，无论“动作片”是和“用户年龄”交互还是和“时段”交互，用的都是同一个向量。直觉上，“用户年龄”与“物品类目”的交互模式，和“展示时段”与“物品类目”的交互模式并不相同。 FFM 为每个特征分配了一组嵌入向量，每对应一个对方 field 就有一个专属向量。
$$\hat{y}(\mathbf{x}) = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_{i, f_j}, \mathbf{v}_{j, f_i} \rangle x_i x_j.$$
符号 $\mathbf{v}_{i, f_j}$ 表示“特征 $i$ 在与 field $f_j$ 交互时的嵌入向量”。

### 实现

```python
class FFM(nn.Module):
    """Field-aware Factorization Machine。"""

    def __init__(self, field_dims, num_fields, embed_dim=16):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = num_fields
        self.linear = nn.Linear(sum(field_dims), 1)
        # 每个特征对每个对方 field 都有一个独立的嵌入向量
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
                v_i_fj = self.embeddings[i][j](x[:, i])  # 特征 i 对 field j 的嵌入
                v_j_fi = self.embeddings[j][i](x[:, j])  # 特征 j 对 field i 的嵌入
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

### FFM 和 FM 的权衡

| 维度       | FM                  | FFM                          |
|------------|---------------------|------------------------------|
| 参数量     | $O(d \cdot k)$     | $O(d \cdot F \cdot k)$，$F$ 是 field 数 |
| 表达能力   | 所有交互共享嵌入向量 | 每个 field 独立嵌入          |
| 领域知识   | 不需要              | 需要定义 field 结构          |
| 典型用途   | 初步基线模型         | 早期 Criteo / Avazu Kaggle 冠军方案 |

二者均仅建模二阶特征交互。若需建模高阶交互，主流路径有两条：一是通过深度网络隐式学习，二是借助 CIN 或 Cross Network 等模块显式建模。 DeepFM 走的是第一种路线； xDeepFM 和 DCN 走的是第二种路线。

继续之前，先来看一张图，总结了后续内容中涉及的各种“交互算子”：

![FM、FFM、DeepFM、DCN、AutoInt 各自使用的交互算子并排对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig2_interaction_methods.png)

---
## DeepFM：结合 FM 与深度学习

DeepFM （华为， 2017）几乎可以说是深度 CTR 模型的默认起点。它的设计思路非常简洁：并行运行 FM 和深度网络，同时共享嵌入表。

### 为什么这种组合有效

- **FM 分支**显式捕捉二阶（低阶）特征交互。
- **Deep 分支**通过堆叠非线性层隐式捕捉高阶特征交互。
- **共享嵌入**减少了一半参数量，并强制两个分支对每个特征的意义达成一致。

> **类比：** 两位侦探合作破案。 FM 是规则驱动型选手，擅长处理简单线索（"这两个特征总是和点击共现"）。深度 MLP 是模式匹配型选手，能挖掘出复杂且模糊的证据链。两人各自打分，最后加总得出结果。

架构图清晰展示了并行结构：

![DeepFM 架构：共享嵌入喂给并联的 FM 与 Deep 分支，求和后过 sigmoid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig3_deepfm_arch.png)

### 数学公式
$$\hat{y}(\mathbf{x}) = \sigma\big(y_{\text{FM}} + y_{\text{Deep}}\big),$$
其中 $y_{\text{FM}}$ 是标准 FM 表达式，$y_{\text{Deep}}$ 是将所有嵌入拼接后通过 MLP 计算得到：
$$\mathbf{h}_0 = [\mathbf{v}_1; \mathbf{v}_2; \ldots; \mathbf{v}_m], \quad \mathbf{h}_l = \text{ReLU}(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad y_{\text{Deep}} = \mathbf{w}^\top \mathbf{h}_L + b.$$
### 实现

```python
class DeepFM(nn.Module):
    """DeepFM：FM 与深度网络并联，共享嵌入表。"""

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

        # Deep 分支（使用同一份嵌入）
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

DeepFM 是**首选基线模型**。如果我要搭建一个新的 CTR 系统，我会从这里开始，然后分别消融 FM 分支和 Deep 分支。只有当任意一个分支的消融导致 AUC 明显下降时，才会考虑更复杂的模型。

接下来的两个模型源于一个直观的观察：深度 MLP 隐式学习特征交互，但你无法知道它具体学到了什么。这推动了 xDeepFM （CIN）和 DCN （Cross Network）的发展，它们都试图显式建模高阶交互结构。

---
## xDeepFM：显式高阶特征交互

xDeepFM （eXtreme Deep Factorization Machine， 2018）引入了 **CIN （Compressed Interaction Network）**，在 embedding 空间中逐层构建高阶特征交互。

### CIN 的工作原理

可以把 CIN 想象成一个金字塔结构：

- **第 0 层：** 原始 embedding （一阶特征）。
- **第 1 层：** 第 0 层每个特征与原始 embedding 逐元素相乘（二阶特征）。
- **第 2 层：** 第 1 层每个特征再与原始 embedding 逐元素相乘（三阶特征）。
- ……

每一层的交叉操作后会接一个可学习的卷积压缩：
$$\mathbf{X}^k_{h, *} = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} W^{k,h}_{i,j}\big(\mathbf{X}^{k-1}_{i,*} \circ \mathbf{X}^0_{j,*}\big),$$
其中 $\circ$ 表示 Hadamard （逐元素）乘积，$W$ 是可学习权重。

> **简单来说。** "取上一层每张 feature map，与原始 embedding 逐元素相乘，然后用 1x1 卷积压缩所有交叉结果，得到固定数量的 feature map。逐层堆叠。"

完整的 xDeepFM 是 **Linear + CIN + Deep MLP** 的三塔模型，在 sigmoid 前将三部分输出相加。

### 实现

```python
class CIN(nn.Module):
    """xDeepFM 使用的 Compressed Interaction Network。"""

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
            outputs.append(X_k.sum(dim=2))                       # 在 D 维度上 sum-pool
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
| 可解释性 | 较弱 | 更好——可以探查 CIN 的特征图 |
| 推理成本 | 较低 | 较高（CIN 占主导） |
| 适用场景 | 默认起点 | 数据复杂、 DeepFM 效果趋于饱和时 |

---
## Deep & Cross Network （DCN）：有界阶交叉特征

DCN （Google， 2017）另辟蹊径。它没有像 CIN 那样堆叠逐元素相乘和可学习卷积，而是引入了一个极小的模块——**Cross Network**。每增加一层，这个模块都会让交互的多项式阶数精确提升 1，同时每层仅增加 $O(d)$ 参数。

### Cross 层
$$\mathbf{x}_{l+1} = \mathbf{x}_0 \cdot (\mathbf{w}_l^\top \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l.$$
> **简单解释。** "对当前状态做一个标量投影，乘回**原始输入向量**，加上偏置和残差。" 每一步都重新注入 $\mathbf{x}_0$，从而将交互阶数提升 1。

经过 $L$ 层 Cross，模型学到的是原始特征的 $L+1$ 阶多项式——但 Cross 部分总共只用了 $L \cdot d$ 个参数。

下图展示了两个关键点：每一层 Cross 如何提升阶数，以及它相比朴素多项式展开有多高效。

![左：每一层 Cross 都把 x0 注入一次，使多项式阶数 +1。右：参数代价随阶数变化——DCN 线性增长，朴素多项式爆炸](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig4_dcn_cross.png)

右图是重点。在 100 维输入下做 6 阶交叉，朴素多项式展开需要 $10^{12}$ 个参数，而 Cross Network 只需 600 个。

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

**DCN 的优点**

- 显式、**有界**的交互阶数——上线时不会出现意外。
- Cross 部分比深度 MLP 小得多，延迟接近普通 MLP。
- 已在 Google 大规模线上验证；后续 v2 提出 DCN-Mix，进一步提升模型容量。

---
## AutoInt：注意力作为特征交互引擎

AutoInt （2019）将 Transformer 的核心组件——**多头自注意力机制**——引入到特征交互中。核心思想是：并非所有特征交互都同等重要，注意力机制可以自动学习**哪些特征对需要重点关注**，并且通过多个头学习多种“相关性”的定义。

### 工作原理

把每个字段的嵌入向量当作一个 token，分别投影为 query、 key 和 value：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}.$$

> **通俗解释。** “每个特征问‘我该从谁的嵌入向量里读取信息？’（Q），展示自己知道的内容（K），并提供可被聚合的信息（V）。 Softmax 根据相似度计算路由权重。”

使用 $H$ 个头时，模型可以并行学习 $H$ 种不同的特征相关性定义。堆叠 $L$ 层 AutoInt 模块后，信息可以多次流动，从而构建更深层次的特征组合。

### 实现代码

```python
import numpy as np

class MultiHeadAttention(nn.Module):
    """AutoInt 模块内部使用的多头自注意力机制。"""

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
    """AutoInt：在特征嵌入上堆叠多头自注意力模块。"""

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

**AutoInt 的优势**

- 不依赖人工设计 schema，自动发现重要交互。
- 注意力权重**可解释性强**，便于调试和结果汇报。
- 多头结构天然适合捕捉多种特征相关性模式。
## FiBiNet：特征重要性 + 双线性交互

FiBiNet （2019）解决了其他模型中隐含的两个假设问题：

1. **所有特征同等重要。** 实际上并非如此。有些特征信号强，有些则是噪声。 FiBiNet 使用 **SENet** 学习每个字段的重要性权重。
2. **逐元素积能充分表达特征交互。** 但很多时候不够。 FiBiNet 用 **双线性** 形式替代 Hadamard 积，可以建模不对称且更复杂的交互。

### SENet：学习特征重要性

分三步完成：

- **压缩。** 对每个字段的嵌入向量沿嵌入维度取均值，得到一个标量——即该字段的重要性分数。
- **激励。** 用一个两层 MLP （带瓶颈结构）将这些标量映射为每个字段的权重门控。
- **重加权。** 将每个嵌入向量乘以其对应的权重。

> **类比：** DJ 根据当前播放内容动态调整每个音轨的音量推子。

### 双线性交互

把 $\mathbf{v}_i \odot \mathbf{v}_j$ 替换为 $\mathbf{v}_i^\top \mathbf{W} \mathbf{v}_j$，其中 $\mathbf{W}$ 是可学习矩阵。变体包括：所有字段对共享 $\mathbf{W}$（Field-All）、每个字段独享 $\mathbf{W}$（Field-Each）、每对字段独享 $\mathbf{W}$（Field-Interaction）。

### 实现

```python
class SENet(nn.Module):
    """字段维度上的 Squeeze-and-Excitation 门控。"""

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
        z = x.mean(dim=2)                            # 压缩
        weights = self.excitation(z).unsqueeze(2)    # 激励
        return x * weights                           # 重加权

class BilinearInteraction(nn.Module):
    """共享单个 W 的双线性交互。"""

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
    """特征重要性与双线性交互网络。"""

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
## 模型对比与选择指南

每个工程师最关心的问题是：**这些方法真的能提升 AUC 吗？**

下图总结了 Criteo 类基准测试中模型的典型相对排序。绝对数值仅供参考——不同数据集、 Embedding 维度和训练预算会导致变化——但**差距模式**在已发表的报告中高度一致。

![柱状图对比 LR、FM、FFM、DeepFM、xDeepFM、DCN、AutoInt、FiBiNet 在 Criteo 类基准上的 AUC 与 Logloss](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/recommendation-systems/04-CTR预估与点击率建模/fig5_auc_logloss.png)

有两个观察比绝对数值更重要：

1. **从 LR 到 FM 是最大的单步提升。** 即使简单地加入二阶交互，带来的 AUC 提升也超过后续任何架构改进。
2. **DeepFM 及其后续模型的 AUC 差距在 ~0.005 范围内。** 听起来很小，但在 Google 或 Meta 的规模下， 0.5 milli-AUC 就意味着真金白银；而在百万用户量级的初创公司，这点差距可能淹没在噪声中——特征质量和新鲜度才是关键。

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

### 实用决策清单

- **第一版系统 / POC。** 用 **LR** 或 **FM**。先确保数据 pipeline、评估和上线流程跑通，再考虑加层数。
- **第一版“真”模型。** 选 **DeepFM**。表格里性价比最高的模型。
- **DeepFM 达到瓶颈且有 GPU 预算。** 试 **DCN**（更轻量）或 **xDeepFM**（更强大），不要同时尝试两者。
- **异质字段且需要解释交互权重。** 用 **AutoInt**。
- **特征列表长且杂乱，怀疑特征重要性差异大。** 选 **FiBiNet**。
- **超低延迟 / 边缘部署。** 线上用 **LR / FM**，离线用复杂模型做重排或召回初始化。

---
## 训练策略与最佳实践

### 处理类别不平衡

CTR 数据极度不均衡，我推荐三种可靠的工具：

**1. 加权 BCE Loss**

```python
# pos_weight 提升少数（正）类在损失中的权重
pos_weight = torch.tensor([num_negatives / num_positives])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**2. 负采样**  
这是 Facebook 级别推荐系统中的标准做法。需要注意的是，负采样会导致预测概率失真，上线前必须重新校准。

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

**3. Focal Loss**  
Focal Loss 会降低简单样本的权重，让梯度集中在少数难样本上。

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

- MLP 层使用 **Dropout 0.2-0.5**，但不要直接对 embedding 查表加 dropout。
- 稠密权重用 **L2 正则**（weight decay 1e-5 到 1e-6）， embedding 的正则通常比稠密权重少。
- 在验证 AUC 上实现 **早停**， patience 设置为 3-10 个 epoch。

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

**AUC-ROC** 是核心指标。它衡量随机一个正样本得分高于随机一个负样本的概率，天然对类别不平衡免疫。

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

**校准同样重要，甚至比 AUC 更关键**，尤其是在下游竞价场景中。预测 CTR 为 0.05 时，应该在该概率桶内对应大约 5% 的真实点击率。可以用 `sklearn.calibration.calibration_curve` 绘制可靠性图。如果发现系统性高估或低估，上线前用 **Platt scaling** 或 **isotonic regression** 进行校准。

---
## 常见问题

### 为什么 CTR 预估是二分类，而不是回归？

CTR 的目标是一个概率值（点击的可能性），而 Bernoulli 分布正好对应这个目标。二分类有成熟的评估指标（AUC、 Logloss），能优雅处理样本不平衡问题，输出的分数也在 0 到 1 之间，易于解释。回归有时用于预测点击次数、收入或观看时长，但针对点击行为本身， BCE 是标准选择。

### 如何选择 Embedding 维度？

从 16 开始。小数据集（< 1M 样本）用 4-8 维就够了；大数据集（> 100M 样本）可以尝试 16-64 维。快速验证：如果维度翻倍后 AUC 提升不到 0.001，就回到较小维度。 Embedding 表通常是模型内存和服务成本的主要消耗。

### FM 和矩阵分解（MF）有什么区别？

MF 只对**单一**的用户-物品评分矩阵做分解，得到用户和物品的隐向量。 FM 更通用：它对**任意特征**之间的二阶交互进行因子化，还能把年龄、城市、时段等辅助信息融入同一个因子化框架。 MF 其实是 FM 的特例，只有两个字段。

### DeepFM 和 xDeepFM 怎么选？

优先用 **DeepFM**。只有在 DeepFM 明显遇到瓶颈，并且数据足够丰富到三阶、四阶交互确实有意义时，才考虑 xDeepFM。 CIN 模块会让推理成本几乎翻倍。

### 如何处理冷启动物品？

通常结合四种方法：(1) 用内容特征（文本/图像编码器）初始化隐向量；(2) 新物品前几次曝光回退到热度排序；(3) 用上下文 Bandit 主动探索，确保新物品获得**一定量**曝光；(4) 在相关任务上预训练隐向量。记住一条原则：**永远不要让模型只看到新物品的 ID。**

### 特征工程和模型架构，哪个更重要？

几乎总是特征工程，重要性高出 2-3 倍。好的交叉特征、合理的分桶策略、缺失值处理、用户/物品的滑窗统计，通常能带来 10%-30% 的 AUC 提升；而在深度 CTR 模型家族中切换架构，提升通常只有 2%-10%。先做好特征工程，再考虑更复杂的架构。

### 缺失特征怎么处理？

四个方法：(1) 使用默认值（0、均值、众数）；(2) 添加一个 `is_missing` 二值特征；(3) 为类别特征保留一个特殊的“缺失”嵌入；(4) 用 KNN 或简单模型插补。选择依据是**缺失本身是否携带信息**——比如未登录用户缺少人口学特征，这件事本身就很有预测力，应该作为特征。

### 离线评估和在线评估的区别？

**离线评估：** 按时间切分训练/验证/测试集（**绝对不能随机切分！**）。指标： AUC、 Logloss。速度快、成本低，但无法反映下游效应（多样性、新鲜度、位置偏差）。**在线评估：** 用真实用户做 A/B 测试。指标：实际 CTR、转化率、收入、留存率。虽然慢且昂贵，但这是唯一权威的信号。永远先离线验证，再上线测试。

### 如何将 CTR 模型部署到生产环境？

四步走：(1) 用 TorchServe / Triton / TF-Serving 提供批量服务；(2) 通过 INT8 量化、 Embedding 分片、预取等手段，将 p99 延迟控制在 10ms 内；(3) 监控预测 CTR 的分布漂移——如果直方图发生变化，及时重训或回滚；(4) 模型和**特征流水线一起版本化**——一次 schema 不匹配就能静默摧毁 AUC。

### 2024-2025 年的趋势是什么？

大规模 Transformer 化的交互建模、多任务学习（联合预测 CTR + 转化率 + 观看时长）、用户-物品图上的 GNN、 AutoML 自动搜索 Embedding 维度与架构、基于因果推断和 IPS 的去偏、保护隐私的联邦学习。不过，无论趋势如何变化，基本功——特征质量、交互建模、校准、新鲜度——始终是最值得投入的核心杠杆。
## 总结

CTR 预估是现代排序系统的核心。从单层线性模型到基于注意力的交互发现，推荐系统的架构演进经历了以下关键阶段：

1. **LR** —— 简单、校准性强，但无法捕捉特征交互。
2. **FM / FFM** —— 自动建模稀疏数据上的二阶交互； FFM 通过增加参数引入了 field 感知能力。
3. **DeepFM** —— 工业界的主力模型：显式二阶交互（FM）+ 隐式深度学习（DNN），共享同一张嵌入表。
4. **xDeepFM** —— 通过 CIN 实现显式的高阶特征交互。
5. **DCN** —— 参数线性增长的有界阶多项式交叉。
6. **AutoInt** —— 使用多头自注意力机制挖掘和解释特征交互。
7. **FiBiNet** —— 可学习的特征重要性（SENet）与双线性交互结合。

**实战经验总结**

- **从简单开始**。按 LR -> FM -> DeepFM 的顺序尝试。 AUC 不再提升时就停止。
- **优先优化特征，再考虑架构**。新增一个交叉特征通常比换一个新模型更有效。
- **认真处理样本不平衡**。选择 pos_weight、负采样+校准或 Focal Loss 中的一种，并坚持使用。
- **评估要真实可信**。离线用时间切分验证，在线用 A/B 测试，同时关注 AUC 和校准效果。
- **持续迭代**。 CTR 系统永远不会结束。数据分布会漂移，物品会更新，昨天的模型就是今天的基线。

“最好的”模型，是在你的延迟预算内、在你的数据上，通过你的 A/B 测试胜出的那个。先理解问题，再选择能解决问题的最简单工具。


---

> 
>
> 本文是推荐系统系列的**第 4 篇**，共 16 篇。
>
> - [第 1 篇：入门与基础概念](/zh/recommendation-systems/01-入门与基础概念)
> - [第 2 篇：协同过滤与矩阵分解](/zh/recommendation-systems/02-协同过滤与矩阵分解/)
> - [第 3 篇：深度学习基础模型](/zh/recommendation-systems/03-深度学习基础模型)
> - **第 4 篇： CTR 预估与点击率建模**（当前）
> - [第 5 篇：Embedding表示学习](/zh/recommendation-systems/05-embedding表示学习/)
> - [第 6 篇：序列推荐与会话建模](/zh/recommendation-systems/06-序列推荐与会话建模/)
> - [第 7 篇：图神经网络与社交推荐](/zh/recommendation-systems/07-图神经网络与社交推荐/)
> - [第 8 篇：知识图谱增强推荐系统](/zh/recommendation-systems/08-知识图谱增强推荐系统/)
> - [第 9 篇：多任务学习](/zh/recommendation-systems/09-多任务学习与多目标优化/)
> - [第 10 篇：深度兴趣网络](/zh/recommendation-systems/10-深度兴趣网络与注意力机制/)
> - [第 11 篇：对比学习](/zh/recommendation-systems/11-对比学习与自监督学习/)
> - [第 12 篇：大语言模型推荐](/zh/recommendation-systems/12-大语言模型与推荐系统/)
> - [第 13 篇：公平性与可解释性](/zh/recommendation-systems/13-公平性-去偏与可解释性/)
> - [第 14 篇：跨域推荐与冷启动](/zh/recommendation-systems/14-跨域推荐与冷启动解决方案/)
> - [第 15 篇：实时推荐与在线学习](/zh/recommendation-systems/15-实时推荐与在线学习/)
> - [第 16 篇：工业实践](/zh/recommendation-systems/16-工业级架构与最佳实践/)
