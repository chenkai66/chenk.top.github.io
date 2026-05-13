---
title: "迁移学习（四）：小样本学习"
date: 2025-05-19 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Few-Shot Learning
  - Meta-Learning
  - MAML
  - Prototypical Networks
  - Metric Learning
categories: 迁移学习
series: transfer-learning
lang: zh
mathjax: true
description: "从极少样本中学会新概念：N-way K-shot 评测协议、度量学习（Siamese、Prototypical、Matching、Relation 网络）、元学习（MAML、Reptile）、Episode 训练范式，以及一份可直接运行的 Prototypical 网络实现。"
disableNunjucks: true
series_order: 4
translationKey: "transfer-learning-4"
---
给小孩看一张穿山甲的照片，他这辈子都能认出穿山甲；而给深度学习模型看一张照片，它的回答基本是随机瞎猜。小样本学习旨在填补这一差距，使分类器在每类只有 1 到 10 个标注样本的情况下也能正常工作。

关键不在于更努力地死记硬背每个类别，而是学会**如何从极少的样本中学习**，并将这种能力迁移到测试时从未见过的新类别上。本文将介绍当前主导该领域的两大方法家族：**度量学习**（学习一个优良的距离函数）和**元学习**（学习一个优良的初始化）。

## 你将学到什么

- N-way K-shot 评测协议，以及为何标准训练在此设定下失效
- 度量学习：Siamese、Prototypical、Matching 和 Relation 网络
- 元学习：MAML 及其一阶变体（FOMAML、Reptile）
- 情节式训练：让训练时的难度与测试时对齐
- 一个简洁、端到端的 Prototypical 网络 PyTorch 实现

**前置知识**：本系列第 1–2 篇；熟悉 PyTorch 和基础优化。

---

## 小样本学习的挑战

![5-way 1-shot 评测：左侧支持集、右侧查询集](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig1_nway_kshot.png)

### 问题设定：N-way K-shot

为了确保论文之间可比，社区采用统一的评估协议：

- **N-way**：模型需在 $N$ 个类别中进行分类。
- **K-shot**：每个类别仅有 $K$ 个带标签样本。

例如，“5-way 1-shot”任务意味着：给你 5 个从未见过的类别，每类仅提供 1 张带标签图像，然后要求你对一批新的查询图像进行分类。

每次评估（称为一个 episode）包含：

- **支持集** $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{NK}$：共 $N \times K$ 个带标签样本；
- **查询集** $\mathcal{Q} = \{(x_j, y_j)\}_{j=1}^{NQ}$：待分类的无标签图像（真实标签仅用于计算准确率）。

最终报告的性能是数百至数千个 episode 的平均准确率，并附带 95% 置信区间——因为单个 episode 的方差极大，不加置信区间的数字几乎无法横向比较。

### 为何标准训练会失败

普通分类器在此场景下面临三重困境：

- **数据稀缺**：当 $K = 1$ 时，根本无法估计类内方差；即使 $K = 5$，也仅能勉强估算。
- **过拟合**：高容量网络倾向于直接记忆支持样本，而非学习具有泛化能力的判别规则。
- **类间相似性**：来自同一领域的新类别（如两种狗）往往仅在细微特征上存在差异，而随机初始化的分类器没有理由关注这些细节。

仅靠经验风险最小化加上权重衰减远远不够：正则化虽能防止参数爆炸，却无法注入从单张图像泛化所需的归纳偏置。

### 核心洞见

要从小样本中有效学习，必须依赖**先验知识**。小样本学习通过在大量*基类*（base classes，每类有充足样本）上训练，然后在互不相交的*新类*（novel classes，每类仅有少量样本）上评估，来获取这种先验。主流路径有两条：

1. **度量学习**：训练一个骨干网络，使其嵌入空间天然具备类间分离性，从而可用少量支持样本的位置刻画新类，并通过距离对查询样本分类。
2. **元学习**：在大量模拟的小样本任务上训练，使网络“学会快速适应”——即仅需几步梯度更新即可适配新任务。这里，“快速适应”本身成为优化目标。

两者共享相同的数据划分（基类 vs. 新类），但先验知识的注入方式不同：度量学习将其编码进嵌入空间，元学习则将其编码进优化的初始状态。

---

## 度量学习：用距离做分类

![原型网络的嵌入空间：每类样本聚集在原型周围，决策区域按最近原型规则划分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig2_prototypical.png)

度量学习的核心思想一句话即可概括：学习一个嵌入函数 $f_\theta$，使得同类样本在嵌入空间中聚集，异类样本彼此远离；随后，通过查询样本与支持样本的距离进行分类。

### Siamese 网络

这是该家族的早期代表。两个权重共享的编码器 $f_\theta$ 分别嵌入一对输入，其距离定义为：
$$d(x_1, x_2) = \|f_\theta(x_1) - f_\theta(x_2)\|_2.$$
训练采用**对比损失**（contrastive loss）：
$$\mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2,$$
其中 $y = 1$ 表示同类对（拉近），$y = 0$ 表示异类对（推开，直至距离超过间隔 $m$）。测试时，查询样本被赋予其最近邻支持样本的标签。

### Prototypical 网络（原型网络）

原型网络改进了两两比较的方式，将每个类的支持样本压缩为一个单一的“原型点”。

#### 计算原型

对于类别 $c$ 的支持样本 $\{x_1^c, \ldots, x_K^c\}$，其原型为嵌入均值：
$$\mathbf{c}_c = \frac{1}{K} \sum_{k=1}^{K} f_\theta(x_k^c).$$
几何上，这相当于该类在嵌入空间中的质心。

#### 分类

对查询样本 $x_q$，以负平方欧氏距离作为 logit，再经 softmax 得到概率：
$$P(y = c \mid x_q) = \frac{\exp\bigl(-d(f_\theta(x_q), \mathbf{c}_c)\bigr)}{\sum_{c'} \exp\bigl(-d(f_\theta(x_q), \mathbf{c}_{c'})\bigr)}, \qquad d(u, v) = \|u - v\|_2^2.$$
训练时对每个 episode 的查询预测使用交叉熵损失，端到端优化。

#### 为何原型方法合理？

若假设类条件嵌入服从共享各向同性协方差的高斯分布 $P(x \mid y = c) = \mathcal{N}(\mu_c, \sigma^2 I)$，则最大似然分类器恰好选择最近的均值。因此，原型网络可视为该（虽强但实用）假设下贝叶斯最优分类器的深度学习实现——这也是其在实践中表现优异的根本原因。

另一个更简洁的观察是：在平方欧氏距离下，任意两类间的决策边界都是嵌入空间中的超平面。因此，原型网络等价于在所学空间中使用*线性分类器*，只不过其权重由原型几何隐式决定。

### Matching 网络（匹配网络）

匹配网络摒弃了“硬性最近邻”的规则，转而对整个支持集施加软注意力。

![Matching 网络：余弦相似度经过 softmax，得到对支持样本的注意力权重](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig3_matching.png)

预测结果是对标签的加权求和：
$$P(y \mid x_q, \mathcal{S}) = \sum_{i=1}^{NK} a(x_q, x_i) \cdot y_i, \qquad a(x_q, x_i) = \mathrm{softmax}_i\bigl(\cos(f(x_q), g(x_i))\bigr).$$
其中 $y_i$ 为 one-hot 标签向量，因此预测是若干 one-hot 向量的凸组合。

该论文另一贡献是**全上下文嵌入**（full context embeddings）：通过双向 LSTM 遍历整个支持集，使每个支持嵌入都能感知其他所有支持样本。其直觉在于，判别性特征取决于你试图区分的其他类别——而 LSTM 能让网络表达这种依赖关系。

### Relation 网络（关系网络）

关系网络更进一步：不再使用固定度量（如欧氏或余弦距离），而是**学习一个度量函数**。一个小网络 $g_\phi$ 接收拼接后的查询嵌入与类原型，输出标量相似度：
$$r_{q, c} = g_\phi\bigl(\mathrm{concat}(f_\theta(x_q),\, \mathbf{c}_c)\bigr) \in [0, 1].$$
![Relation 网络：共享嵌入模块 + 学习出来的关系模块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig5_relation.png)

训练目标为 $r_{q, c} = \mathbb{1}\{y_q = c\}$，使用均方误差损失，两个模块联合训练。为何要这样做？固定度量隐含假设嵌入空间各向同性——即每个维度同等重要。而学习度量允许网络自动降低对当前任务无信息量的维度的权重。

---

## 元学习：学会学习

度量学习将先验知识融入嵌入空间，而元学习则将其直接融入优化过程本身。模型在大量任务上训练，使得对新任务的适应仅需几步梯度更新。

### MAML：模型无关的元学习

MAML 的思想简单却惊人有效：寻找一个初始化参数 $\theta$，使得对任意新任务的支持集执行一两次梯度更新后，即可获得高性能模型。

![MAML：一个元初始化通过少量内环梯度步适配到多个任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig4_maml.png)

#### 算法

对每个采样任务 $\mathcal{T}_i$（含支持集与查询集）：

1. **内环**（任务自适应）：在支持集损失上执行一步（或几步）梯度更新：
   $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{support}}(\theta).$$
2. **外环**（元更新）：用适配后的参数 $\theta_i'$ 在查询集上评估损失，并更新初始化：
   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta_i').$$
外环梯度需穿过内环更新，涉及支持集损失对 $\theta$ 的二阶导数——即 Hessian-向量乘积。

#### 一阶近似（FOMAML）

精确的二阶 MAML 在参数维度 $d$ 上内存开销为 $O(d^2)$，且实现复杂。FOMAML 直接忽略二阶项，近似为：
$$
abla_\theta \mathcal{L}(\theta_i') \approx \nabla_{\theta_i'} \mathcal{L}(\theta_i'),
$$
即直接使用适配点处的梯度，假装 $\theta_i'$ 与 $\theta$ 无关。此举将开销降至 $O(d)$，而准确率几乎不变。

#### 几何直觉

MAML 将 $\theta$ 推向损失景观中一个**适合快速适应的平坦区域**：从此点出发，沿任意任务方向走几步即可抵达低损失区。可将 $\theta$ 视为通用发射台，而非通用好模型。

### Reptile：更简单的方案

Reptile 完全省去内环求导。采样一个任务，在其上运行 $k$ 步普通 SGD 得到 $\tilde{\theta}$，然后将元参数向该结果微调：
$$\theta \leftarrow \theta + \epsilon \,(\tilde{\theta} - \theta).$$
算法仅此而已。尽管简单，其效果却几乎与 MAML 相当——因为在大量任务上反复将元参数推向任务特定解，最终会使其落在所有任务解的公共“甜点”附近。

| 方法    | 梯度阶数 | 单步代价       | 实现难度 | miniImageNet (5w-5s)* |
|---------|----------|----------------|----------|------------------------|
| MAML    | 二阶     | 高（Hessian）  | 困难     | ~63%                   |
| FOMAML  | 一阶     | 中等           | 简单     | ~62%                   |
| Reptile | 一阶     | 低             | 极简     | ~66%                   |

*数据来自原始论文；不同实现间可能存在差异。

---

## 情节式训练

标准监督训练将整个基类数据集一次性喂给网络进行分类。而情节式训练则重构整个训练循环，使其与测试过程一致。

![分幕训练：每一步都是一个全新的 N-way K-shot 任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig7_episodic.png)

### 一“幕”如何构建

每次迭代执行以下步骤：

1. 从基类池中随机采样 $N$ 个类别；
2. 每类抽取 $K$ 个样本组成**支持集**；
3. 每类再抽取 $Q$ 个样本组成**查询集**；
4. 仅基于该支持集，训练模型对查询集进行分类。

```python
for epoch in range(num_epochs):
    for episode in range(episodes_per_epoch):
        classes = sample(base_classes, N)
        support = sample_from_classes(classes, K)
        query   = sample_from_classes(classes, Q)

        prototypes = compute_prototypes(support)
        logits     = -distance(query, prototypes)
        loss       = cross_entropy(logits, query_labels)

        loss.backward()
        optimizer.step()
```

### 为何这很重要

模型在训练中永远无法看到完整的基类数据集。每次梯度更新都在模拟一个小样本任务，因此网络习得的归纳偏置恰好匹配测试所需。这本质上是一种课程学习，而课程内容就是测试时的真实条件。

一个有力的验证实验是：关闭情节式训练，直接训练一个 $|C_{\text{base}}|$ 路的普通分类器，然后在冻结的特征上附加线性分类头。若骨干网络足够强（如深层 ResNet 配合强数据增强），这种“Baseline++”方案在标准基准上可与度量学习和元学习方法媲美。Chen 等人（ICLR 2019）借此指出，情节式训练的重要性可能被高估，而骨干网络的容量与预训练质量更为关键。

---

## 这些方法到底效果如何？

![miniImageNet 5-way 评测：代表性方法的准确率](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/04-few-shot-learning/fig6_mini_imagenet.png)

上述数据源自原始论文（后续工作通过更大骨干网络和预训练技巧已进一步提升性能）。两点关键结论：

- **1-shot 与 5-shot 的差距巨大**：从 1 个样本增至 5 个，通常带来 10–20 个百分点的提升——这提醒我们，哪怕极少量的数据，其价值也远超精巧的架构设计。
- **方法性能高度收敛**：一旦固定骨干网络，Prototypical、Matching、Relation 和 MAML 家族的性能通常相差仅几个百分点。选型应基于工程考量（如实现简易性、计算预算、工具链支持），而非追逐最后一点精度。

---

## 完整实现：Prototypical 网络

```python
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool，miniImageNet 的标配模块。"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)

class ProtoNetEncoder(nn.Module):
    """4 层 CNN 编码器，将 84x84 RGB 图像映射到 1600 维向量。"""
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),   # 84 -> 42
            ConvBlock(hidden_dim, hidden_dim),    # 42 -> 21
            ConvBlock(hidden_dim, hidden_dim),    # 21 -> 10
            ConvBlock(hidden_dim, hidden_dim),    # 10 ->  5
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0), -1)

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def compute_prototypes(self, support_emb, support_labels, n_way):
        """对每个类别的支持嵌入取均值，生成原型。"""
        prototypes = [
            support_emb[support_labels == c].mean(dim=0)
            for c in range(n_way)
        ]
        return torch.stack(prototypes)  # (n_way, embed_dim)

    def forward(self, support_imgs, support_lbls, query_imgs, n_way):
        support_emb = self.encoder(support_imgs)
        query_emb   = self.encoder(query_imgs)
        prototypes  = self.compute_prototypes(support_emb, support_lbls, n_way)
        # 负欧氏距离作为 logits
        dists  = torch.cdist(query_emb, prototypes, p=2)
        return -dists

class EpisodeSampler:
    """从扁平化的 (data, labels) 中生成 N-way K-shot 的 episode。"""
    def __init__(self, data, labels, n_way, n_support, n_query, n_episodes):
        self.data, self.labels = data, labels
        self.n_way, self.n_support = n_way, n_support
        self.n_query, self.n_episodes = n_query, n_episodes
        self.classes = np.unique(labels)
        self.class_indices = {c: np.where(labels == c)[0] for c in self.classes}

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._sample()

    def _sample(self):
        chosen = np.random.choice(self.classes, self.n_way, replace=False)
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []

        for new_label, c in enumerate(chosen):
            idxs = np.random.choice(
                self.class_indices[c],
                self.n_support + self.n_query,
                replace=False,
            )
            for idx in idxs[:self.n_support]:
                s_imgs.append(self.data[idx]); s_lbls.append(new_label)
            for idx in idxs[self.n_support:]:
                q_imgs.append(self.data[idx]); q_lbls.append(new_label)

        return (
            torch.stack([torch.FloatTensor(x) for x in s_imgs]),
            torch.LongTensor(s_lbls),
            torch.stack([torch.FloatTensor(x) for x in q_imgs]),
            torch.LongTensor(q_lbls),
        )

def train(model, train_data, train_lbls, val_data, val_lbls,
          n_way=5, n_support=5, n_query=15, n_episodes=100,
          num_epochs=50, lr=1e-3, device='cpu'):
    """Episode 训练循环，带周期性验证。"""
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(num_epochs):
        # ---- 训练阶段 ----
        model.train()
        sampler = EpisodeSampler(train_data, train_lbls,
                                 n_way, n_support, n_query, n_episodes)
        loss_sum, acc_sum = 0.0, 0.0
        for s_img, s_lbl, q_img, q_lbl in tqdm(sampler, desc=f'Epoch {epoch+1}'):
            s_img, s_lbl = s_img.to(device), s_lbl.to(device)
            q_img, q_lbl = q_img.to(device), q_lbl.to(device)

            logits = model(s_img, s_lbl, q_img, n_way)
            loss = crit(logits, q_lbl)

            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item()
            acc_sum  += (logits.argmax(1) == q_lbl).float().mean().item()

        # ---- 验证阶段 ----
        model.eval()
        val_sampler = EpisodeSampler(val_data, val_lbls,
                                     n_way, n_support, n_query, n_episodes)
        val_acc = 0.0
        with torch.no_grad():
            for s_img, s_lbl, q_img, q_lbl in val_sampler:
                s_img, s_lbl = s_img.to(device), s_lbl.to(device)
                q_img, q_lbl = q_img.to(device), q_lbl.to(device)
                logits = model(s_img, s_lbl, q_img, n_way)
                val_acc += (logits.argmax(1) == q_lbl).float().mean().item()
        val_acc /= n_episodes
        print(f"  train_loss={loss_sum/n_episodes:.4f}  "
              f"train_acc={acc_sum/n_episodes:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), 'best_protonet.pt')

# ---- 用随机数据跑一遍，确认流程通了 ----
if __name__ == '__main__':
    num_classes, samples_per_class, img_size = 64, 600, 84
    all_data = np.random.randn(num_classes * samples_per_class,
                               3, img_size, img_size).astype(np.float32)
    all_labels = np.repeat(np.arange(num_classes), samples_per_class)

    train_classes = int(num_classes * 0.8)
    train_mask = all_labels < train_classes
    val_mask   = all_labels >= train_classes

    encoder = ProtoNetEncoder()
    model = PrototypicalNetwork(encoder)
    train(model, all_data[train_mask], all_labels[train_mask],
          all_data[val_mask], all_labels[val_mask],
          n_way=5, n_support=5, n_query=15, num_epochs=10)
```

### 代码要点

| 组件                  | 功能                                         |
|-----------------------|----------------------------------------------|
| `ProtoNetEncoder`     | 4 层 CNN，miniImageNet 实验的标准骨干网络     |
| `compute_prototypes`  | 对每类支持嵌入取均值                         |
| `forward`             | 返回负欧氏距离作为 logits                    |
| `EpisodeSampler`      | 每次迭代生成一个 N-way K-shot episode        |
| `train`               | 情节式训练循环，含周期性验证                 |

两个实现细节值得强调：

- **`torch.cdist(..., p=2)` 返回的是欧氏距离（非平方）**。将其取负作为 logits 在 argmax 上可行，但严格来说不符“高斯均值即贝叶斯最优”的推导。实践中影响甚微；若需严格对应，可手动平方。
- **采样器内部必须将支持类别重映射为 $0, \ldots, N-1$**，以确保交叉熵目标的形状正确。

---

## 常见问题

**小样本学习与普通迁移学习有何区别？**  
它是迁移学习的极限情形。普通迁移学习通常假设有数百个目标标签，微调分类头即可胜任；而小样本学习仅有 1–10 个标签。这一差距如此之大，以至于仅靠下游训练技巧已不够，必须在训练阶段引入专门机制——如情节采样、度量或元学习目标。

**为何原型网络使用均值作为原型？**  
在共享各向同性协方差的高斯类条件假设下，类均值即为贝叶斯最优分类器。即便该假设不成立，均值仍具足够鲁棒性——尤其当 $K \ge 5$ 时效果更佳。

**MAML 与原型网络，该如何选择？**  
默认选用原型网络：更简单、更快、原型可解释，且在标准图像基准上通常持平甚至优于 MAML。仅在以下情况考虑 MAML：(a) 任务差异显著，彼此外观迥异；(b) 数据非图像，且缺乏优质预训练嵌入；(c) 需要整个网络参与适应，而非仅更新最终分类头。

**需要多少基类？**  
越多越好。标准基准中，miniImageNet 使用 64 个基类，Omniglot 超过 1200 个。若基类少于约 30 个，模型易对基类过拟合，导致新类准确率骤降。

**是否适用于非图像数据？**  
适用。原型网络可用于任何可嵌入的数据——文本（Transformer 编码器）、图（GNN）、音频（频谱 CNN）。MAML 与 Reptile 本身模型无关，情节协议亦不依赖模态。

**为何必须报告置信区间？**  
单 episode 准确率方差极大，一个困难 episode 可导致 10–20 个百分点的波动。仅通过数百 episode 的均值加 95% 置信区间，才能实现跨论文的可靠比较。

---

## 总结

小样本学习直击深度学习的最大实践瓶颈：长尾场景下的数据稀缺。

- **度量学习**（Siamese、Prototypical、Matching、Relation Networks）构建一个“距离即不相似度”的嵌入空间，方法简单、高效、可解释，其中 Prototypical Networks 是默认首选。
- **元学习**（MAML、FOMAML、Reptile）寻找一个初始化点，使其经几步梯度更新即可适配任意新任务，灵活性更高但计算成本更大、可解释性较弱。
- **情节式训练**是统一范式：每次迭代模拟一个全新小样本任务，确保训练与测试难度一致。

跨方法比较揭示一个常被忽视的事实：一旦骨干网络固定，各类方法性能迅速收敛——这提醒我们，骨干网络的容量与预训练质量，至少与顶层的小样本算法同等重要。

下一篇：[第 5 章——知识蒸馏](/zh/transfer-learning/05-知识蒸馏/)，我们将探讨如何将大型教师模型压缩为轻量学生模型，同时保持性能接近。

---

## 参考文献

1. Snell et al. (2017). *Prototypical Networks for Few-shot Learning.* NeurIPS. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
2. Finn et al. (2017). *Model-Agnostic Meta-Learning (MAML).* ICML. [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)
3. Vinyals et al. (2016). *Matching Networks for One Shot Learning.* NeurIPS. [arXiv:1606.04080](https://arxiv.org/abs/1606.04080)
4. Sung et al. (2018). *Learning to Compare: Relation Network for Few-Shot Learning.* CVPR. [arXiv:1711.06025](https://arxiv.org/abs/1711.06025)
5. Nichol et al. (2018). *On First-Order Meta-Learning Algorithms (Reptile).* [arXiv:1803.02999](https://arxiv.org/abs/1803.02999)
6. Koch et al. (2015). *Siamese Neural Networks for One-shot Image Recognition.* ICML Deep Learning Workshop.
7. Chen et al. (2019). *A Closer Look at Few-shot Classification.* ICLR. [arXiv:1904.04232](https://arxiv.org/abs/1904.04232)
8. Wang et al. (2020). *Generalizing from a Few Examples: A Survey on Few-Shot Learning.* ACM Computing Surveys. [arXiv:1904.05046](https://arxiv.org/abs/1904.05046)
