---
title: "迁移学习（四）：小样本学习"
date: 2025-05-19 09:00:00
tags:
  - 度量学习
  - 元学习
  - 小样本学习
  - MAML
  - 原型网络
  - 迁移学习
categories:
  - 迁移学习
series:
  name: "迁移学习"
  part: 4
  total: 12
lang: zh-CN
mathjax: true
description: "从极少样本中学会新概念：N-way K-shot 评测协议、度量学习（Siamese、Prototypical、Matching、Relation 网络）、元学习（MAML、Reptile）、Episode 训练范式，以及一份可直接运行的 Prototypical 网络实现。"
disableNunjucks: true
series_order: 4
---

给一个孩子看一张穿山甲的照片，他这辈子都能认出穿山甲。给深度学习模型看一张，它给你的回答和瞎猜没什么两样。**小样本学习（Few-Shot Learning）** 要做的，就是把这条鸿沟填上——让分类器在每类只有 1 到 10 个标注样本的情况下也能工作。

关键不在于"把每一类学得更牢"，而是要让模型 **学会如何从极少样本中学习**，然后把这种能力迁移到训练时从未见过的新类别上。本文围绕当下的两大主流路线展开：**度量学习** 学一个好的距离函数，**元学习** 学一个好的初始化。

## 你将学到

- N-way K-shot 评测协议，以及标准训练为什么在这上面会崩
- 度量学习：Siamese、Prototypical、Matching、Relation 四种网络
- 元学习：MAML 与它的一阶变体（FOMAML、Reptile）
- Episode 训练范式：让训练时的难度和测试时一致
- 一份完整、可运行的 Prototypical 网络 PyTorch 实现

**前置知识：** 本系列第 1、2 两篇；熟悉 PyTorch 与基础优化方法。

---

## 小样本学习的挑战

![5-way 1-shot 评测：左侧支持集、右侧查询集](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig1_nway_kshot.png)

### 问题设定：N-way K-shot

社区使用一套统一的评测协议，方便不同论文之间互相比较：

- **N-way**：模型要在 $N$ 个类别中做分类。
- **K-shot**：每个类别只有 $K$ 个带标签样本可用。

所以 "5-way 1-shot" 的任务就是：给你 5 个从未见过的类别，每类只配 1 张带标签的照片；现在请你对一批查询图片做分类。

每个评测 episode 由两部分组成：

- **支持集** $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{NK}$：$N \times K$ 个带标签样本；
- **查询集** $\mathcal{Q} = \{(x_j, y_j)\}_{j=1}^{NQ}$：未带标签的待分类样本，标签只用于打分。

最终的报告数字是几百到几千个 episode 的平均准确率，并附带 95% 置信区间——单个 episode 的方差很大，没有置信区间的数字基本没法横向比较。

### 标准训练为什么会崩

三股力量同时压在普通分类器身上：

- **样本太少**：$K=1$ 时连类内方差都估不出来，$K=5$ 也只是勉强够用。
- **过拟合严重**：高容量网络会把支持样本背下来，而不是去学一个能泛化的判别规则。
- **类间过近**：新类别如果来自同一个领域（比如两种狗），区别往往只在很细微的局部特征上，随机初始化的分类器根本没动力去关注它们。

经验风险最小化加上权重衰减是不够的：正则化能阻止参数爆炸，但提供不了"从一张图泛化"所需要的归纳偏置。

### 核心思路

要从少量样本里学到东西，必须有 **先验知识**。小样本学习获取先验的方式是：在数量充足的 *基类（base classes）* 上训练，然后在不相交的 *新类（novel classes）* 上做小样本评测。引入先验的两条主路线如下：

1. **度量学习**：训一个能让同类样本聚成一团、异类样本互相远离的嵌入空间。新类只需要用支持样本在这个空间里的位置来表征，查询样本按距离判类。
2. **元学习**：在大量"模拟出来的小样本任务"上训练，让网络 *学会被几步梯度更新所适配*。把"快速适配"本身当作要优化的目标。

两者用的是同一份数据划分（基类 vs 新类），区别在于把先验知识藏在哪里：度量学习把它塞进嵌入，元学习把它塞进优化的初始化。

---

## 度量学习：用距离来分类

![原型网络的嵌入空间：每类样本聚集在原型周围，决策区域按最近原型规则划分](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig2_prototypical.png)

度量学习的整套思路一句话就能说清：学一个嵌入 $f_\theta$，让同类样本距离近、异类样本距离远，然后用查询样本到支持样本的距离来判类。

### Siamese 网络

度量学习里最早的成员。两个权重共享的编码器 $f_\theta$ 把一对输入映射成嵌入向量，距离定义为

$$d(x_1, x_2) = \|f_\theta(x_1) - f_\theta(x_2)\|_2.$$

训练用 **对比损失（contrastive loss）**：

$$\mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2,$$

其中 $y=1$ 表示同类对（目标是把距离压小），$y=0$ 表示异类对（目标是把距离推到大于间隔 $m$ 之后再不去管）。测试时把查询样本和每个支持样本算距离，选最近的那个的标签。

### Prototypical 网络（原型网络）

原型网络在 Siamese 的两两比较之上更进一步：每个类只用一个点来代表。

#### 计算原型

对类别 $c$ 的支持样本 $\{x_1^c, \ldots, x_K^c\}$，原型就是嵌入的均值：

$$\mathbf{c}_c = \frac{1}{K} \sum_{k=1}^{K} f_\theta(x_k^c).$$

几何上就是该类在嵌入空间里的"质心"。

#### 分类

对查询样本 $x_q$，用负的平方欧氏距离作 logit，再过一个 softmax：

$$P(y = c \mid x_q) = \frac{\exp\bigl(-d(f_\theta(x_q), \mathbf{c}_c)\bigr)}{\sum_{c'} \exp\bigl(-d(f_\theta(x_q), \mathbf{c}_{c'})\bigr)}, \qquad d(u, v) = \|u - v\|_2^2.$$

训练用查询集上的交叉熵损失，端到端反传。

#### 为什么"取均值"是有道理的

如果假设类条件嵌入服从共享各向同性协方差的高斯分布 $P(x \mid y = c) = \mathcal{N}(\mu_c, \sigma^2 I)$，那么最大似然分类规则恰好就是"选最近的均值"。原型网络可以看作这个贝叶斯最优分类器在深度模型上的实现——这也是它在实践中长期保持竞争力的根本原因。

还有一个更干净的观察：在平方欧氏距离下，任意两类之间的决策面都是嵌入空间里的一个超平面。所以原型网络等价于 *学到的嵌入空间里的线性分类器*，只不过线性权重被原型几何隐式约束住了。

### Matching 网络（匹配网络）

匹配网络把"取最近原型"这种硬规则换成了对整个支持集的软注意力。

![Matching 网络：把余弦相似度过 softmax，得到对支持样本的注意力权重](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig3_matching.png)

预测就是标签的加权求和：

$$P(y \mid x_q, \mathcal{S}) = \sum_{i=1}^{NK} a(x_q, x_i) \cdot y_i, \qquad a(x_q, x_i) = \mathrm{softmax}_i\bigl(\cos(f(x_q), g(x_i))\bigr).$$

其中 $y_i$ 是 one-hot 标签，所以预测是若干 one-hot 向量的凸组合。

论文的另一个贡献是 **Full Context Embeddings**：用一个双向 LSTM 把整个支持集编码一遍，让每个支持嵌入都"看到"其他所有支持样本。直觉是：什么算判别性特征，取决于你要和哪些类别区分开——LSTM 让网络能把这层语义表达出来。

### Relation 网络（关系网络）

关系网络再向前一步：不再用固定的距离度量（欧氏、余弦），而是 **学一个度量函数**。一个小的神经网络 $g_\phi$ 接受拼接后的嵌入，输出一个标量相似度：

$$r_{q, c} = g_\phi\bigl(\mathrm{concat}(f_\theta(x_q),\, \mathbf{c}_c)\bigr) \in [0, 1].$$

![Relation 网络：共享嵌入模块 + 学习出来的关系模块](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig5_relation.png)

训练目标是 $r_{q, c} = \mathbb{1}\{y_q = c\}$，用 MSE 损失，两个模块联合训练。为什么要这么做？固定度量隐式假设嵌入空间是各向同性的——每个维度同等重要。但实际并非如此，让网络自己去学度量，可以自动降低那些对当前任务无用的维度的权重。

---

## 元学习：学会学习

度量学习把先验塞进嵌入，元学习则把先验塞进优化过程本身。模型在大量任务上训练，使得 *把它适配到一个新任务* 只需要几步梯度。

### MAML：模型无关的元学习

MAML 的想法朴素得近乎暴力：找一个初始化 $\theta$，从它出发，对任意新任务的支持集做一两步梯度，就能得到一个不错的模型。

![MAML：一个元初始化，通过内环少量梯度步迅速适配到不同任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig4_maml.png)

#### 算法

对采样到的每个任务 $\mathcal{T}_i$（带各自的支持/查询集）：

1. **内环（任务自适应）**：在支持损失上做一步（或几步）梯度下降：
   $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{support}}(\theta).$$
2. **外环（元更新）**：用 *适配后的参数* 在查询集上算损失，再回过头更新初始化：
   $$\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}^{\text{query}}(\theta_i').$$

外环梯度需要 *穿过* 内环这一步更新求导，会涉及支持损失对 $\theta$ 的二阶导数——也就是一个 Hessian-向量 乘积。

#### 一阶近似（FOMAML）

精确的二阶 MAML 在参数维度 $d$ 上的内存代价是 $O(d^2)$，实现起来也很别扭。FOMAML 直接把二阶项扔掉，用

$$\nabla_\theta \mathcal{L}(\theta_i') \approx \nabla_{\theta_i'} \mathcal{L}(\theta_i'),$$

也就是把"在适配点上的梯度"直接当作元梯度，假装 $\theta_i'$ 不依赖 $\theta$。代价降到 $O(d)$，准确率几乎不变。

#### 几何直觉

MAML 把 $\theta$ 推到损失曲面上一个 **对快速适配友好** 的区域：从这一点出发，沿任意任务的方向走几步都能到达低损失。可以把 $\theta$ 看成一个"通用发射台"，而不是一个"通用好模型"。

### Reptile：再简化一层

Reptile 把内环里的求导也省掉了。采样一个任务，在它上面跑 $k$ 步普通 SGD 得到 $\tilde{\theta}$，然后把元参数往这个结果上挪一点：

$$\theta \leftarrow \theta + \epsilon \,(\tilde{\theta} - \theta).$$

整个算法就这么一行。简单到让人怀疑，但效果几乎和 MAML 持平：把元参数反复朝各种任务的解推一推，最终它会落在所有任务解的某个共同甜点附近。

| 方法    | 梯度阶 | 单步代价   | 实现难度 | miniImageNet (5w-5s)\* |
|---------|--------|------------|----------|------------------------|
| MAML    | 二阶   | 高（Hessian） | 难     | ~63%                   |
| FOMAML  | 一阶   | 中等       | 简单     | ~62%                   |
| Reptile | 一阶   | 低         | 极简     | ~66%                   |

\*数字来自原始论文，不同实现之间会有出入。

---

## Episode 训练

普通监督训练把整个基类数据集喂给网络，让它做分类。Episode 训练把整个训练循环重写成"长得像测试循环"的样子。

![Episode 训练：每一步都是一个全新的 N-way K-shot 任务](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig7_episodic.png)

### 一个 episode 是怎么搭出来的

每次迭代：

1. 从基类池里采样 $N$ 个类别；
2. 每类取 $K$ 个样本，组成 **支持集**；
3. 每类再取 $Q$ 个样本，组成 **查询集**；
4. 让模型在仅看到支持集的前提下，预测查询集的标签。

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

### 这件事的意义

模型从来没机会"一次性看到全部基类"，每一次梯度更新都在模拟一个小样本任务。于是网络学到的归纳偏置，恰好就是测试时所需要的那种偏置。这是把"测试条件"当成 *课程* 的课程学习。

值得做一次对照实验：把 episode 训练关掉，直接训一个 $|C_{\text{base}}|$-类的普通分类器，再在冻结的特征上挂一个线性头。配上一个强 backbone（深 ResNet、大量数据增强），这种 "Baseline++" 配方足以与各种度量、元学习方法五五开——Chen 等人在 ICLR 2019 用这个结果说明：episode 训练并没有想象中那么关键，**backbone 的容量和预训练质量更值钱**。

---

## 这些方法到底做得多好？

![miniImageNet 5-way 评测：代表性方法的报告准确率](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/transfer-learning/04-Few-Shot-Learning/fig6_mini_imagenet.png)

上图的数字都来自原始论文（后续工作通过更大的 backbone 和更花哨的预训练把这些数字进一步刷高）。可以读出两件事：

- **从 1-shot 到 5-shot 的跨度极大。** 多 4 个样本一般就能涨 10–20 个点——再次提醒：哪怕是一点点数据增量，都足以压过精巧的架构选择。
- **方法在结果上扎堆。** 一旦把 backbone 固定下来，原型网络、匹配网络、关系网络、MAML 系列的数字都会落在几个百分点的范围里。选型时按工程口味来（实现简洁度、算力预算、生态）就好，没必要为了最后一个百分点纠结。

---

## 完整实现：Prototypical 网络

```python
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool，miniImageNet 标配的基本块。"""
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
    """4 层 CNN 编码器：把 84x84 的 RGB 图映射成 1600 维向量。"""
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
        """对每个类的支持嵌入取均值得到原型。"""
        prototypes = [
            support_emb[support_labels == c].mean(dim=0)
            for c in range(n_way)
        ]
        return torch.stack(prototypes)  # (n_way, embed_dim)

    def forward(self, support_imgs, support_lbls, query_imgs, n_way):
        support_emb = self.encoder(support_imgs)
        query_emb   = self.encoder(query_imgs)
        prototypes  = self.compute_prototypes(support_emb, support_lbls, n_way)
        # 负欧氏距离作为 logit
        dists  = torch.cdist(query_emb, prototypes, p=2)
        return -dists


class EpisodeSampler:
    """从扁平化的 (data, labels) 中产出 N-way K-shot episode。"""
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
    """Episode 训练循环 + 周期性验证。"""
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(num_epochs):
        # ---- 训练 ----
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

        # ---- 验证 ----
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


# ---- 用随机数据跑一遍，确认管道通了 ----
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

| 组件                  | 作用                                         |
|-----------------------|----------------------------------------------|
| `ProtoNetEncoder`     | 4 块 CNN，miniImageNet 实验的标准 backbone   |
| `compute_prototypes`  | 对每类的支持嵌入取均值                       |
| `forward`             | 返回负欧氏距离作为 logit                     |
| `EpisodeSampler`      | 每次迭代构建一个 N-way K-shot episode        |
| `train`               | Episode 训练循环 + 验证                      |

两个实现细节值得拎出来：

- **`torch.cdist(..., p=2)` 返回的是欧氏距离，不是平方欧氏距离。** 取负作为 logit 走 argmax 不会有问题，但严格意义上和"高斯均值是贝叶斯最优"那个推导不完全对应。实践中影响很小；要严格对应的话把它平方一下。
- **采样器内部一定要把支持类别重新编号成 $0, \ldots, N-1$**，不然交叉熵的标签维度会对不上。

---

## 深度 Q&A

**小样本学习和普通的迁移学习有什么不同？**
小样本是迁移学习的极限情况。普通迁移学习通常假设目标任务至少有几百个标签，光靠微调一个分类头就能解决大部分问题。小样本只有 1–10 个标签——这点差距大到必须在训练阶段就动手术：要么 episode 采样，要么换上度量/元学习目标，光靠下游训练技巧救不回来。

**为什么原型网络用均值做原型？**
在共享各向同性协方差的高斯类条件假设下，类均值就是贝叶斯最优分类器。即使这个假设并不严格成立，均值的鲁棒性也足够好——尤其在 $K \ge 5$ 的时候。

**MAML 和原型网络，到底用哪个？**
默认选原型网络：实现更简单、跑得更快、原型本身可以可视化、在标准图像评测上还能打平甚至超过 MAML。下面三种情况再考虑 MAML：(a) 任务彼此差异很大、长得不像；(b) 数据不是图像、又没有现成的好嵌入；(c) 你确实需要"整个网络都被适配"，而不只是末端的分类头。

**基类数量大概要多少？**
越多越好。标准评测里 miniImageNet 用 64 个基类，Omniglot 多达 1200+。少于 30 个基类的时候，模型会反过来在基类自身上严重过拟合，新类准确率直接坍缩。

**这套东西对非图像数据管用吗？**
管用。原型网络对任何"有意义嵌入"都成立——文本用 Transformer，图用 GNN，音频用频谱 CNN。MAML 和 Reptile 设计上就是模型无关的。Episode 协议也不在意模态。

**为什么报告时一定要带置信区间？**
单 episode 的方差非常大——一个特别难的 episode 能让准确率掉 10 到 20 个点。只有报告几百个 episode 的均值加 95% CI，论文之间的数字才有可比性。

---

## 总结

小样本学习正面攻击深度学习最棘手的实际瓶颈：长尾里的数据稀缺。

- **度量学习**（Siamese、Prototypical、Matching、Relation Networks）学一个"距离即不相似度"的嵌入空间。简单、快、可解释，原型网络是默认主力。
- **元学习**（MAML、FOMAML、Reptile）学一个"几步梯度就能到达任意任务最优解"的初始化。更灵活，但代价更大、可解释性更弱。
- **Episode 训练** 是把这两条路串起来的关键训练范式：每次迭代都是一个全新的小样本任务，让训练时的难度直接对齐测试时的难度。

横向对比下来还有一个常被忽略的事实：一旦 backbone 固定，各家方法的数字会迅速收拢——**backbone 的容量和预训练质量，至少和小样本算法本身一样重要。**

下一篇 [第 5 章——知识蒸馏](/zh/迁移学习-五-知识蒸馏/)，我们换个方向，看看怎么把一个庞大的教师模型压缩成一个轻巧、行为相似的学生模型。

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

---

## 系列导航

| 部分 | 主题 |
|------|------|
| [1](/zh/迁移学习-一-基础与核心概念/) | 基础与核心概念 |
| [2](/zh/迁移学习-二-预训练与微调技术/) | 预训练与微调 |
| [3](/zh/迁移学习-三-域适应方法/) | 域适应 |
| **4** | **小样本学习（本文）** |
| [5](/zh/迁移学习-五-知识蒸馏/) | 知识蒸馏 |
| [6](/zh/迁移学习-六-多任务学习/) | 多任务学习 |
| [7](/zh/迁移学习-七-零样本学习/) | 零样本学习 |
| [8](/zh/迁移学习-八-多模态迁移/) | 多模态迁移 |
| [9](/zh/迁移学习-九-参数高效微调/) | 参数高效微调 |
| [10](/zh/迁移学习-十-持续学习/) | 持续学习 |
| [11](/zh/迁移学习-十一-跨语言迁移/) | 跨语言迁移 |
| [12](/zh/迁移学习-十二-工业应用与最佳实践/) | 工业应用与最佳实践 |
