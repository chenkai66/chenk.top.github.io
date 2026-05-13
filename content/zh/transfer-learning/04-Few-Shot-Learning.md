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

关键在于学会如何从极少的样本中快速学习，并将这种能力应用到测试时完全没见过的新类别上，而不是死记硬背每个类别。这篇文章介绍的是当前最主流的两类方法：**度量学习**，即学习一个好用的距离函数；**元学习**，即学习一个优秀的初始化。
## 你将学到什么
- N-way K-shot 评测协议及标准训练方法为何失效
- 度量学习：Siamese、Prototypical、Matching 和 Relation 网络
- 元学习：MAML 及其一阶变体（FOMAML、Reptile）
- 情节式训练：使训练难度与测试难度相匹配
- 一个简洁、端到端的 Prototypical 网络 PyTorch 实现

**前置知识：** 本系列第 1、2 篇；熟悉 PyTorch 和基础优化方法。

---
## 小样本学习的挑战

![5-way 1-shot 评测：左侧支持集、右侧查询集](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig1_nway_kshot.png)

### 问题设定： N-way K-shot

为了方便论文之间的比较，社区统一了一套评估标准：

- **N-way**：模型需要在 $N$ 个类别中进行分类。
- **K-shot**：每个类别只有 $K$ 个带标签的样本。

例如，"5-way 1-shot" 任务是：给你 5 个从未见过的类别，每类只提供 1 张带标签的图片，然后让你对一批新的查询图片进行分类。

每次评估包含两个部分：

- **支持集** $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{NK}$：$N \times K$ 个带标签的样本。
- **查询集** $\mathcal{Q} = \{(x_j, y_j)\}_{j=1}^{NQ}$：待分类的无标签图片，标签仅用于计算准确率。

最终报告的数字是几百到几千次评估的平均准确率，并附上 95% 的置信区间。由于单次评估的方差很大，没有置信区间的数字很难横向对比。

### 标准训练为何失效

普通分类器面临三大难题：

- **数据太少**：当 $K=1$ 时，根本无法估计类内方差；即使 $K=5$，也只能勉强应付。
- **过拟合严重**：高容量网络会直接记住支持集样本，而不是学习一个能泛化的判别规则。
- **类间相似**：新类别如果来自同一领域（比如两种狗），差异往往只体现在细微特征上，随机初始化的分类器完全没有理由关注这些特征。

经验风险最小化加上权重衰减并不够用：正则化可以防止参数爆炸，但无法注入从少量样本泛化所需的归纳偏置。

### 核心思路
要从小样本中学习，必须依赖 **先验知识**。小样本学习的做法是在大量 *基类* 上训练（每个基类有很多样本），然后在不相交的 *新类* 上进行小样本评估。引入先验知识主要有两条路径：

1. **度量学习**：训练一个嵌入空间，让同类样本聚集在一起，异类样本互相远离。新类可以用支持样本在这个空间中的位置来表征，查询样本通过距离进行分类。
2. **元学习**：在大量模拟的小样本任务上训练，让网络学会通过几次梯度更新快速适应。把“快速适应”本身当作优化目标。

两者使用相同的数据划分（基类 vs 新类），但先验知识的注入方式不同：度量学习将其嵌入到表示空间，元学习将其嵌入到优化的初始状态。
## 度量学习：用距离来做分类

![原型网络的嵌入空间：每类样本聚集在原型周围，决策区域按最近原型规则划分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig2_prototypical.png)

度量学习的核心思想很简单：训练一个嵌入函数 $f_\theta$，让同类样本靠得近、异类样本离得远，然后根据查询样本和支持样本的距离来分类。

### Siamese 网络

这是度量学习家族的开山鼻祖。两个共享权重的编码器 $f_\theta$ 把一对输入映射到嵌入空间，距离定义为：

$$d(x_1, x_2) = \|f_\theta(x_1) - f_\theta(x_2)\|_2.$$
训练时使用 **对比损失（contrastive loss）**：
$$\mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2,$$
其中 $y=1$ 表示同类对，目标是拉近距离；$y=0$ 表示异类对，目标是把距离推到大于间隔 $m$ 后停止优化。测试时，直接计算查询样本和每个支持样本的距离，取最近的那个标签。

### Prototypical 网络（原型网络）

原型网络在两两比较的基础上更进一步，把每个类的支持样本压缩成一个点。

#### 计算原型

对于类别 $c$ 的支持样本 $\{x_1^c, \ldots, x_K^c\}$，原型就是这些样本嵌入的均值：
$$\mathbf{c}_c = \frac{1}{K} \sum_{k=1}^{K} f_\theta(x_k^c).$$
从几何上看，这就是该类在嵌入空间中的“质心”。

#### 分类

对查询样本 $x_q$，用负的平方欧氏距离作为 logit，再通过 softmax 得到概率分布：
$$P(y = c \mid x_q) = \frac{\exp\bigl(-d(f_\theta(x_q), \mathbf{c}_c)\bigr)}{\sum_{c'} \exp\bigl(-d(f_\theta(x_q), \mathbf{c}_{c'})\bigr)}, \qquad d(u, v) = \|u - v\|_2^2.$$
训练时用查询集上的交叉熵损失，端到端反向传播。

#### 为什么原型方法合理？

假设类条件嵌入服从共享各向同性协方差的高斯分布 $P(x \mid y = c) = \mathcal{N}(\mu_c, \sigma^2 I)$，那么最大似然分类规则就是“选最近的均值”。原型网络可以看作这个贝叶斯最优分类器在深度学习中的实现——这也是它实际效果好的根本原因。

还有一个直观解释：在平方欧氏距离下，任意两类之间的决策面都是嵌入空间中的超平面。因此，原型网络等价于 *学到的嵌入空间中的线性分类器*，只不过线性权重被原型几何隐式约束了。

### Matching 网络（匹配网络）

匹配网络改进了“取最近原型”的硬规则，改为对整个支持集的软注意力。

![Matching 网络：余弦相似度经过 softmax，得到对支持样本的注意力权重](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig3_matching.png)

预测是对标签的加权求和：
$$P(y \mid x_q, \mathcal{S}) = \sum_{i=1}^{NK} a(x_q, x_i) \cdot y_i, \qquad a(x_q, x_i) = \mathrm{softmax}_i\bigl(\cos(f(x_q), g(x_i))\bigr).$$
其中 $y_i$ 是 one-hot 标签，预测结果是若干 one-hot 向量的凸组合。

论文的另一个亮点是 **Full Context Embeddings**：用双向 LSTM 对整个支持集进行编码，让每个支持嵌入都能“看到”其他所有支持样本。直觉上，判别性特征取决于你要区分哪些类别——LSTM 让网络能表达这种语义。

### Relation 网络（关系网络）

关系网络更进一步：不再用固定的度量（如欧氏距离或余弦相似度），而是 **学一个度量函数**。一个小的神经网络 $g_\phi$ 接受拼接后的嵌入，输出一个标量相似度：
$$r_{q, c} = g_\phi\bigl(\mathrm{concat}(f_\theta(x_q),\, \mathbf{c}_c)\bigr) \in [0, 1].$$
![Relation 网络：共享嵌入模块 + 学习出来的关系模块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig5_relation.png)

训练目标是 $r_{q, c} = \mathbb{1}\{y_q = c\}$，用 MSE 损失，两个模块联合训练。为什么要这么做？固定度量隐含了一个假设：嵌入空间是各向同性的，每个维度同等重要。但实际情况并非如此，让网络自己学度量，可以自动降低那些对当前任务无用的维度的权重。
## 元学习：学会学习

度量学习把先验知识融入嵌入空间，而元学习则直接将先验融入优化过程。模型在大量任务上训练，使得它只需几步梯度更新就能快速适应新任务。

### MAML：模型无关的元学习

MAML 的思路简单却非常有效：找到一个初始化参数 $\theta$，使得对任意新任务的支持集做一两次梯度更新后，就能得到一个性能不错的模型。

![MAML：一个元初始化通过少量内环梯度步适配到多个任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig4_maml.png)

#### 算法

对于每个采样的任务 $\mathcal{T}_i$（包含支持集和查询集）：

1. **内环（任务自适应）**：在支持集损失上做一步（或几步）梯度更新：
   $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{support}}(\theta).$$
2. **外环（元更新）**：用适配后的参数 $\theta_i'$ 在查询集上评估损失，并更新初始化参数：
   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta_i').$$

外环梯度需要穿过内环更新，涉及支持集损失对 $\theta$ 的二阶导数——即 Hessian-向量乘积。

#### 一阶近似（FOMAML）

精确的二阶 MAML 在参数维度 $d$ 上的内存开销是 $O(d^2)$，实现起来也复杂。 FOMAML 直接去掉二阶项，用以下公式近似：
$$\nabla_\theta \mathcal{L}(\theta_i') \approx \nabla_{\theta_i'} \mathcal{L}(\theta_i'),$$
也就是直接用适配点上的梯度当作元梯度，假装 $\theta_i'$ 不依赖于 $\theta$。这样内存开销降到 $O(d)$，性能几乎没有变化。

#### 几何直觉

MAML 把 $\theta$ 推向损失曲面上一个适合快速适应的区域：从这个点出发，沿任何任务方向走几步都能到达低损失区域。可以把 $\theta$ 看作一个通用起点，而不是一个通用好模型。

### Reptile：更简单的方案

Reptile 完全省掉了内环求导。随机采样一个任务，在其上运行 $k$ 步普通 SGD 得到 $\tilde{\theta}$，然后将元参数朝这个结果挪动一点：
$$\theta \leftarrow \theta + \epsilon \,(\tilde{\theta} - \theta).$$

整个算法就这么简单。尽管看起来简陋，但效果几乎和 MAML 持平。因为反复将元参数推向不同任务的解，最终它会靠近所有任务解的共同甜点。

| 方法    | 梯度阶 | 单步代价   | 实现难度 | miniImageNet (5w-5s)\* |
|---------|--------|------------|----------|------------------------|
| MAML    | 二阶   | 高（Hessian） | 难     | ~63%                   |
| FOMAML  | 一阶   | 中等       | 简单     | ~62%                   |
| Reptile | 一阶   | 低         | 极简     | ~66%                   |

\*数据来自原始论文，不同实现之间可能略有差异。

---
## 分幕训练

普通的监督学习把整个基类数据集丢给网络，让它学会分类。分幕训练则完全不同，它把训练过程改造成和测试过程几乎一样的形式。

![分幕训练：每一步都是一个全新的 N-way K-shot 任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig7_episodic.png)

### 一“幕”是如何构建的

每次迭代都按以下步骤进行：

1. 从基类池中随机抽取 $N$ 个类别。
2. 每个类别选 $K$ 个样本，组成 **支持集**。
3. 再为每个类别选 $Q$ 个样本，组成 **查询集**。
4. 让模型只根据支持集的信息，对查询集中的样本进行分类。

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

### 为什么这很重要

模型在训练时永远看不到完整的基类数据集。每次梯度更新都在模拟一个小样本任务，因此网络学到的归纳偏置正好是测试时需要的那种。这种训练方式本质上是一种课程学习，而课程的内容就是测试时的实际条件。

一个有趣的实验可以验证这一点：关掉分幕训练，直接训练一个 $|C_{\text{base}}|$ 类的普通分类器，然后在冻结的特征上加一个线性分类头。如果 backbone 足够强（比如深 ResNet 或大量数据增强），这种简单的 "Baseline++" 方法完全可以媲美各种度量学习和元学习方法。 Chen 等人在 ICLR 2019 的研究就用这个结果表明，分幕训练并没有大家想象的那么重要，真正关键的是 backbone 的容量和预训练的质量。
## 这些方法到底效果如何？

![miniImageNet 5-way 评测：代表性方法的准确率](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-few-shot-learning/fig6_mini_imagenet.png)

上面的数据来自原始论文（后来的研究通过更大的 backbone 和更复杂的预训练技巧，把这些数字刷得更高）。我总结了两点：

- **1-shot 和 5-shot 的差距非常大。** 从一个样本增加到五个样本，通常能提升 10% 到 20% 的准确率——这再次说明，即使是很小的数据量，也比精巧的架构设计更重要。
- **方法之间的结果很接近。** 只要 backbone 固定，原型网络、匹配网络、关系网络和 MAML 系列的结果基本都在几个百分点之内。选型时根据工程需求（比如实现简单性、计算预算、工具链支持）来决定就好，没必要为了最后一点精度纠结。

---
## 完整实现： Prototypical 网络

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
| `ProtoNetEncoder`     | 4 块 CNN， miniImageNet 实验的标准 backbone   |
| `compute_prototypes`  | 对每类的支持嵌入取均值                       |
| `forward`             | 返回负欧氏距离作为 logits                    |
| `EpisodeSampler`      | 每次迭代生成一个 N-way K-shot episode        |
| `train`               | Episode 训练循环，带验证                     |

两个实现细节需要注意：

- **`torch.cdist(..., p=2)` 返回的是欧氏距离，不是平方欧氏距离。** 取负作为 logits 在 argmax 上没问题，但严格来说和“高斯均值是贝叶斯最优”的推导不完全一致。实践中影响不大；如果要严格对应，可以平方一下。
- **采样器内部必须把支持类别重新编号为 $0, \ldots, N-1$**，否则交叉熵的标签维度会出错。

---
## 深度问答

**小样本学习和普通迁移学习有什么区别？**  
小样本学习是迁移学习的极限场景。普通迁移学习通常假设有几百个目标标签，微调一个分类头就能搞定大部分工作。而小样本学习只有 1 到 10 个标签。这个差距太大了，光靠下游训练技巧解决不了问题，必须在训练阶段引入专门的设计：比如 episode 采样、度量学习或元学习目标。

**为什么原型网络用均值作为原型？**  
在共享各向同性协方差的高斯类条件假设下，类均值就是贝叶斯最优分类器。即使这个假设不成立，均值的鲁棒性依然很强——尤其是当 $K \ge 5$ 时，效果更明显。

**MAML 和原型网络，我该选哪个？**  
优先选择原型网络：实现简单、速度快、原型直观可解释，而且在标准图像任务上表现不输 MAML，甚至更好。以下三种情况可以考虑 MAML：(a) 任务差异大，彼此看起来完全不同；(b) 数据不是图像，也没有高质量的预训练嵌入；(c) 需要整个网络都参与适应，而不仅仅是最后的分类头。

**基类数量需要多少？**  
越多越好。标准基准数据集里， miniImageNet 用了 64 个基类， Omniglot 更是超过 1200 个。如果基类少于 30 个，模型很容易对基类本身过拟合，新类的准确率会大幅下降。

**这些方法适用于非图像数据吗？**  
适用。原型网络对任何有合理嵌入的数据都有效——文本可以用 Transformer 编码器，图结构数据可以用 GNN，音频可以用频谱 CNN。 MAML 和 Reptile 本身就是模型无关的算法。 Episode 协议也不挑数据模态。

**为什么一定要报告置信区间？**  
单个 episode 的准确率波动很大，一个特别难的 episode 可能会让结果下降 10 到 20 个百分点。只有通过几百个 episode 的均值加上 95% 置信区间，才能让不同论文的结果具有可比性。
## 小结

小样本学习直击深度学习最大的实际痛点：长尾数据稀缺。

- **度量学习**（Siamese、 Prototypical、 Matching、 Relation Networks）构建一个嵌入空间，让距离直接反映不相似性。方法简单、速度快、结果直观，其中原型网络（Prototypical Networks）是最常用的默认选择。
- **元学习**（MAML、 FOMAML、 Reptile）寻找一个初始化点，从这里出发只需几步梯度更新就能适应新任务。灵活性更高，但计算成本也更大，解释性稍弱。
- **Episode 训练** 是统一的训练框架：每次迭代都模拟一个全新的小样本任务，确保训练难度和测试难度一致。

横向对比发现一个常被忽视的事实：一旦固定 backbone，不同方法的性能差距迅速缩小。这提醒我， backbone 的容量和预训练质量，至少和小样本算法本身一样重要。

下一篇 [第 5 章——知识蒸馏](/zh/transfer-learning/05-知识蒸馏/)，我会换个方向，看看如何把一个庞大的教师模型压缩成一个轻量级的学生模型，同时保持性能接近。 

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
