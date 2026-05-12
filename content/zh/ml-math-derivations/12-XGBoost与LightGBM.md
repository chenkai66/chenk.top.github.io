---
title: "机器学习数学推导（十二）：XGBoost 与 LightGBM"
date: 2026-01-31 09:00:00
tags:
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Regularization
  - Histogram Algorithm
  - Mathematical Derivations
  - Machine Learning
categories: 机器学习
series: ml-math-derivations
lang: zh
mathjax: true
description: "从 XGBoost 的二阶泰勒展开到 LightGBM 的直方图加速，本文系统推导两大工业级梯度提升框架——正则化目标函数、分裂增益闭式解、GOSS 单边采样与 EFB 互斥特征绑定的数学原理。"
disableNunjucks: true
series_order: 12
translationKey: "ml-math-derivations-12"
---
XGBoost 和 LightGBM 是表格数据领域的两大利器——从 Kaggle 排行榜到风控系统、广告排序和用户流失预测，背后几乎都有它们的身影。两者都基于梯度提升树（Gradient-Boosted Trees，见第 11 篇），但在工程设计上选择了截然不同的方向。

- **XGBoost** 主攻*数学优化*：把损失函数的二阶导数引入目标函数，对树结构本身做正则化，把分裂点选择变成一个闭式解公式。
- **LightGBM** 主攻*系统效率*：将特征离散化为小直方图，按叶子节点逐层生长，丢弃信息量低的样本（GOSS），并合并互斥的稀疏特征（EFB）。

从 API 看，两者似乎可以无缝替换，但当数据量 $N$ 或特征维度 $d$ 增大时，它们的表现差异会变得非常明显。我会在本文中推导这些设计背后的每一个公式，让你看完调参指南后，能清楚知道每个参数存在的原因。

![机器学习数学推导（十二）：XGBoost 与 LightGBM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/illustration_1.png)
## 你将学到什么

- XGBoost 的二阶泰勒展开如何直接计算出最优叶权重的闭式解，以及任意树结构的"结构分数"。
- 分裂增益公式中为什么自带一个剪枝惩罚项 $\gamma$。
- LightGBM 的直方图算法怎样把单特征分裂的复杂度从 $O(N)$ 降到 $O(K)$，并实现"直方图减法"技巧。
- GOSS 的统计学原理是什么， EFB 的具体构造方法又是怎样的。
- 按层生长和按叶子生长分别在什么情况下表现更好。
## 前置知识

- GBDT 基础（本系列第 11 篇）。
- 一阶和二阶泰勒展开。
- 使用不纯度和增益进行决策树分裂。

---
## 一张图看懂 Boosting

先别管公式，直接看梯度提升到底在干什么。我从一个常数预测开始（也就是 $y$ 的均值），然后让每棵新树去拟合当前的残差。前面的树抓住大趋势，后面的树专注于修正局部细节。

![梯度提升：迭代拟合残差](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig1_boosting_iterations.png)

第一棵树拟合完后，结果是个粗糙的阶梯函数，残差非常大。等累积到一百棵小步长树（$\eta = 0.1$）后，整体已经捕捉到了底层的 $\sin(1.5x) + 0.35x$ 趋势，残差基本变成了噪声。 XGBoost 和 LightGBM 的区别只在于它们拟合每棵树的具体方法——但上面这个迭代框架是完全一样的。

---
## XGBoost：把数学做到极致

### 正则化目标函数

普通 GBDT 只关心最小化经验损失。 XGBoost 不一样，它加了一个树复杂度项，让优化器知道什么样的树才算好：

$$\mathcal{L}^{(t)} \;=\; \sum_{i=1}^N L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr) \;+\; \Omega(f_t),$$

其中 $\hat y_i^{(t-1)}$ 是前 $t-1$ 轮的预测结果，$f_t$ 是本轮要拟合的新树，正则项定义为：

$$\Omega(f_t) \;=\; \gamma\, T \;+\; \tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2.$$

- $T$ 是叶子节点的数量，$\gamma T$ 是每片叶子的成本，相当于一个软剪枝阈值。
- $w_j$ 是叶子 $j$ 的预测值，$\tfrac{1}{2}\lambda \sum w_j^2$ 是对叶子权重的 L2 收缩。

### 二阶泰勒展开

把单样本损失在当前预测值 $\hat y_i^{(t-1)}$ 处做二阶展开：

$$L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr)
\;\approx\; L\bigl(y_i,\; \hat y_i^{(t-1)}\bigr) + g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2,$$

梯度和海森矩阵分别是：

$$g_i \;=\; \partial L / \partial \hat y_i^{(t-1)}, \qquad h_i \;=\; \partial^2 L / \partial (\hat y_i^{(t-1)})^2.$$

去掉与 $f_t$ 无关的零阶常数项，第 $t$ 轮的代理目标函数对 $f_t$ 是纯二次的：

$$\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{i=1}^N \Bigl[g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2\Bigr] + \Omega(f_t).$$

这是 XGBoost 和传统 GBDT 的核心区别：二阶导数 $h_i$ 进入了目标函数，相当于免费拿到了牛顿法的曲率信息。

### 最优叶权重与结构分数

用两部分表示一棵树：叶分配函数 $q : \mathbb{R}^d \to \{1,\ldots,T\}$ 和权重向量 $\mathbf{w}$。按叶子分组样本，记 $I_j = \{ i : q(\mathbf{x}_i) = j \}$，并定义：

$$G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i.$$

目标函数在叶子之间完全解耦：

$$\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{j=1}^T \Bigl[G_j\, w_j + \tfrac{1}{2}(H_j + \lambda)\, w_j^2\Bigr] + \gamma T.$$

每个叶子的目标函数都是关于 $w_j$ 的独立二次函数。令 $\partial / \partial w_j = 0$，得到最优叶权重：

$$\boxed{\,w_j^{*} \;=\; -\frac{G_j}{H_j + \lambda}\,}$$

代回后可以得到**结构分数**——这棵树形能达到的最优损失：

$$\widetilde{\mathcal{L}}^{*}(q) \;=\; -\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j + \lambda} \;+\; \gamma T.$$

分数越低越好。注意，结构分数只依赖于树结构 $q$，权重已经被解析地优化掉了。这样一来，树学习就变成了结构搜索问题。

### 分裂增益

考虑是否要把某个叶子拆成左 ($I_L$)、右 ($I_R$) 两个子节点，结构分数的变化是：

$$\boxed{\;\text{Gain} \;=\; \frac{1}{2}\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma\;}$$

这里有两点值得注意：

1. 中括号里的部分是结构分数和的下降量，在减去 $\gamma$ 之前总是非负的（柯西-施瓦茨/方差分解）。
2. 常数 $\gamma$ 直接充当分裂的**阈值**：如果结构改善不足以超过 $\gamma$，分裂就会被拒绝。不需要额外的后剪枝步骤，剪枝已经内嵌在增益公式里。

### 常见损失函数的梯度

| 损失 | $g_i$ | $h_i$ |
|---|---|---|
| 平方损失 $\tfrac{1}{2}(y-\hat y)^2$ | $\hat y_i - y_i$ | $1$ |
| Logistic （$p = \sigma(\hat y)$） | $p_i - y_i$ | $p_i(1-p_i)$ |
| Softmax （类别 $c$） | $p_{ic} - \mathbb{1}[y_i = c]$ | $p_{ic}(1-p_{ic})$ |

平方损失的 $h_i \equiv 1$，二阶目标退化为普通的残差拟合， XGBoost 就回到了 GBDT。 Logistic 和 Softmax 的海森矩阵携带了真实信息（饱和区 $p(1-p)$ 很小），牛顿步的优势明显。

### 分裂查找算法

有了增益公式，剩下的问题就是如何枚举候选分裂点。把每个特征都预排序就是精确算法，单特征单节点复杂度是 $O(N)$；近似算法只挑少量候选分位点（用 $h_i$ 加权，因为 $h_i$ 在二次目标中代表样本重要性）。稀疏感知变体在每次分裂时学习一个"缺失值默认走向"，遍历时只扫非缺失项——在稀疏 one-hot 数据上效果显著。

![XGBoost 精确预排序 vs LightGBM 直方图分裂查找](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig2_split_finding.png)

左图扫描所有不同取值（共 $N-1$ 个候选）。右图把同一份数据分成 $K = 32$ 个桶，只评估 $K-1$ 个候选切点——却几乎选到了同一个阈值、同一个增益。这正是 LightGBM 的核心思路：$N$ 维度的分辨率大部分都是浪费。
## LightGBM：极致的工程优化

![机器学习数学推导（十二）：XGBoost 与 LightGBM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/illustration_2.png)

### 直方图算法

LightGBM 在训练前会一次性将每个特征离散化为 $K$ 个整数桶，默认值是 255。对每个叶子节点和每个特征，它都会构建一个 $(G_b, H_b)$ 的直方图：

$$G_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} g_i, \qquad H_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} h_i.$$

单特征的复杂度从 $O(N)$ 降到 $O(K)$，内存占用和分裂查找都受益：

| 维度 | 精确（XGBoost） | 直方图（LightGBM） |
|---|---|---|
| 单特征内存 | $O(N)$ | $O(K)$ |
| 单特征分裂查找 | $O(N)$ | $O(K)$ |
| 缓存友好度 | 差（随机访问预排序数据） | 好（连续存储的桶） |

还有一个**直方图减法**技巧：如果已知一个子节点的直方图，另一个子节点的直方图可以直接用*父节点减去已知子节点*得到。这样，构建两个子节点的代价就等于构建一个子节点的代价。在深树中，这种方法可以将每层的工作量减半。

### 按叶子生长 vs 按层生长

XGBoost 默认使用**按层生长**（BFS）：先分裂当前深度的所有节点，再进入下一层。而 LightGBM 使用**按叶子生长**（best-first）：始终选择增益最大的叶子节点进行分裂。

![Leaf-wise 与 Level-wise 树生长对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig5_growth.png)

在相同的节点预算下，按层生长会生成一棵深度为 $\log_2 T$ 的平衡树；而按叶子生长可能形成一条细长的链，深入输入空间的某个区域。同样数量的叶子节点，按叶子生长通常能获得更低的训练损失——但如果不限制 `max_depth` 和 `min_data_in_leaf`，很容易过拟合。

### GOSS：基于梯度的单边采样

海森加权后的梯度 $g_i$ 能直接反映样本 $i$ 对下一个分裂的贡献。在一棵已经拟合得不错的树中，大多数样本的 $|g_i| \approx 0$（已经被很好地拟合了）。丢弃这些样本几乎不会损失信息——只要重新调整权重，确保 $G$ 和 $H$ 的统计量保持无偏。

GOSS 通过 $a, b \in (0, 1)$ 分三步完成：

1. 按 $|g_i|$ 降序排列，保留前 $a\cdot N$ 个样本，记为 $A$。
2. 从剩下的 $(1-a) N$ 个样本中随机抽取 $b \cdot N$ 个，记为 $B$。
3. 计算 $G_L, H_L, G_R, H_R$ 时，将 $B$ 中的样本统一乘以 $\dfrac{1-a}{b}$ 来抵消子采样的影响。

最终的增益估计公式变为：

$$\widetilde{\text{Gain}} \;=\; \frac{1}{2}\!\left[\frac{(G_L^A + \tfrac{1-a}{b}G_L^B)^2}{H_L^A + \tfrac{1-a}{b}H_L^B + \lambda} + \frac{(G_R^A + \tfrac{1-a}{b}G_R^B)^2}{H_R^A + \tfrac{1-a}{b}H_R^B + \lambda} - \cdots \right].$$

LightGBM 论文证明，这种子采样引入的方差是 $O(1/\sqrt{n_l})$，比叶子内部本身的采样噪声衰减得更快。

![GOSS 保留信息丰富的尾部，对小梯度样本重加权](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig3_goss.png)

右图是关键：取 $a = 0.20, b = 0.10$，每轮只处理 **30% 的样本**，但重加权后的 $G$ 总和仍与全数据 $G$ 在期望上相等。

### EFB：互斥特征绑定

高维稀疏特征——如 one-hot 类别、词袋模型、点击标志——很少同时非零：一行有 `country = JP` 就不会有 `country = US`。 EFB 利用这种**互斥性**来压缩特征轴。

具体做法是构建一张*冲突图*：节点是特征，边权是两特征同时非零的行数。将彼此无边的特征装进同一个组，这是图着色问题。 EFB 用贪心算法解决：按度数降序遍历特征，将每个特征放入第一个“没有冲突邻居”的现有组。

选定一个组 $\{f_{i_1}, f_{i_2}, \ldots\}$ 后，用整数偏移将它们合并成一列 $\tilde f$，使各自的桶段不重叠：

$$\tilde f_n \;=\; \begin{cases} \text{bin}(x_{n, i_k}) + o_k & \text{若 } x_{n, i_k} \neq 0 \\ 0 & \text{否则} \end{cases}, \qquad o_k = \sum_{r < k} K_{i_r}.$$

![EFB 把多列稀疏互斥特征合并为一列](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig4_efb.png)

A 面板是一段近乎互斥的稀疏块（想象 6 个 one-hot 哑变量）。 B 面板的红边标出违反互斥的特征对，节点颜色是贪心着色给出的组号。 C 面板是合并后的列：每个原特征占据互不重叠的桶段，因此 $\tilde f$ 的直方图能精确还原各原特征的统计量——但列数减少了。

在高维稀疏问题中， EFB 经常将有效 $d$ 减少 5 到 10 倍，进一步放大了直方图构建的加速效果。
## 两种“重要性”定义：同一个模型，两种视角

XGBoost 和 LightGBM 都提供了特征重要性的计算方法，但它们回答的问题并不相同。

![特征重要性：XGBoost gain 与 LightGBM split 计数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig6_feature_importance.png)

- **XGBoost (gain)** 每次用某个特征进行分裂时，会累加增益值。它更倾向于奖励那些*能一次性大幅降低损失*的特征。
- **LightGBM (split count)** 统计每个特征被选为分裂点的次数。它更关注那些*频繁被选中*的特征，即使每次分裂的效果并不显著。

两种排名经常不一致——但都没错，只是侧重点不同。如果我想知道“哪些特征对损失下降贡献最大”，我会看 gain；如果我想了解“模型在迭代中依赖了哪些特征”，我会看 split count。如果需要一个与模型无关的解释方法， SHAP 值可以衡量每个特征对预测结果的边际贡献。
## XGBoost 和 LightGBM 对比速览

| 维度 | XGBoost | LightGBM |
|---|---|---|
| 分裂策略 | 按层（BFS） | 按叶子（best-first） |
| 基础算法 | 预排序精确 / 分位 sketch | 直方图，$K$ 桶 |
| 单特征内存 | $O(N)$ | $O(K)$ |
| 样本效率 | 行列子采样 | GOSS |
| 稀疏/类别特征 | 稀疏感知默认方向 | EFB + 原生类别支持 |
| 失败模式 | 数据量大时变慢 | 不设 `max_depth` 易过拟合 |
| 适用场景 | 小/中等数据量，调参精细 | 大数据量，高维度，稀疏特征 |

### 实战中的训练成本

LightGBM 的直方图、 GOSS 和 EFB 技术栈，换来的是更高的吞吐量：

![训练时间随样本数变化与 Pareto 视图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost与LightGBM/fig7_time_vs_accuracy.png)

当 $N = 10^4$ 时，三种工具的训练时间相差不过几秒，选哪个都行。但当 $N = 10^6$ 时， LightGBM 的速度比 XGBoost 快了大约 5 倍，测试精度却几乎没差别。 CatBoost 的速度介于两者之间，但在类别特征密集的场景下，靠 ordered boosting 技术独占优势。
## 极简 NumPy 版 XGBoost

这段代码实现了二阶目标、闭式叶权重计算、基于增益的分裂以及 $\gamma$ 剪枝功能：

```python
import numpy as np

class XGBoostTree:
    """单棵 XGBoost 树：闭式叶权重 + 基于增益的分裂"""

    def __init__(self, max_depth=6, min_child_weight=1, gamma=0, lambda_=1):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tree = None

    def fit(self, X, g, h):
        self.tree = self._build(X, g, h, depth=0)
        return self

    def _gain(self, GL, HL, GR, HR, G, H):
        return 0.5 * (
            GL**2 / (HL + self.lambda_)
            + GR**2 / (HR + self.lambda_)
            - G**2 / (H + self.lambda_)
        ) - self.gamma

    def _best_split(self, X, g, h):
        N, d = X.shape
        G, H = g.sum(), h.sum()
        best_gain, best = 0.0, None
        for j in range(d):
            order = np.argsort(X[:, j])
            xs, gs, hs = X[order, j], g[order], h[order]
            GL, HL = 0.0, 0.0
            for i in range(N - 1):
                GL += gs[i]; HL += hs[i]
                GR, HR = G - GL, H - HL
                if xs[i] == xs[i + 1]:
                    continue
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                gain = self._gain(GL, HL, GR, HR, G, H)
                if gain > best_gain:
                    best_gain = gain
                    best = (j, 0.5 * (xs[i] + xs[i + 1]))
        return best, best_gain

    def _build(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        leaf_weight = -G / (H + self.lambda_)  # 闭式解
        if depth >= self.max_depth or len(X) < 2:
            return {"w": leaf_weight}
        split, gain = self._best_split(X, g, h)
        if split is None or gain <= 0:  # 这里实现 $\gamma$ 剪枝
            return {"w": leaf_weight}
        j, thr = split
        m = X[:, j] <= thr
        return {
            "f": j, "t": thr,
            "L": self._build(X[m], g[m], h[m], depth + 1),
            "R": self._build(X[~m], g[~m], h[~m], depth + 1),
        }

    def predict(self, X):
        return np.array([self._pred(x, self.tree) for x in X])

    def _pred(self, x, n):
        if "w" in n:
            return n["w"]
        return self._pred(x, n["L"] if x[n["f"]] <= n["t"] else n["R"])

class XGBoost:
    """基于二阶梯度和海森矩阵的加法集成模型"""

    def __init__(self, n_estimators=100, lr=0.1, max_depth=6,
                 gamma=0, lambda_=1, objective="mse"):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.objective = objective
        self.trees, self.base = [], None

    def _gh(self, y, p):
        if self.objective == "mse":
            return p - y, np.ones_like(y)
        s = 1.0 / (1.0 + np.exp(-p))  # logistic 函数
        return s - y, s * (1.0 - s)

    def fit(self, X, y):
        self.base = y.mean() if self.objective == "mse" else 0.0
        pred = np.full(len(y), self.base)
        for _ in range(self.n_estimators):
            g, h = self._gh(y, pred)
            tree = XGBoostTree(self.max_depth, gamma=self.gamma,
                               lambda_=self.lambda_).fit(X, g, h)
            self.trees.append(tree)
            pred += self.lr * tree.predict(X)
        return self

    def predict(self, X):
        p = np.full(len(X), self.base)
        for t in self.trees:
            p += self.lr * t.predict(X)
        return p
```

整套二阶机制能在 100 行内实现，原因在于有了增益公式后，算法的核心就是不断寻找最优分裂点、递归构建树结构并重复这一过程。

---
## Q&A 精选

### 为什么需要二阶信息？

一阶信息告诉你方向，二阶信息告诉你曲率，也就是步子能迈多大才安全。牛顿法在接近最优解时之所以能二次收敛，就是因为考虑了曲率。对每个叶子节点，二阶目标直接给出闭式最优权重 $-G/(H+\lambda)$，而不是靠学习率去试探。

### 为什么 XGBoost 用 $\gamma$ 剪枝，而不是事后剪？

因为结构分数本质上就是损失减去 $\gamma T$。如果某个分裂的增益达不到 $\gamma$，那它实际上会让正则化损失变大。拒绝这种分裂不是凭经验，而是目标函数下的正确选择。

### 按叶子生长什么时候会出问题？

小数据集上容易出问题，因为最高增益的叶子可能只是在追逐噪声。解决办法是设置 `max_depth` 和 `min_data_in_leaf`（小数据集上可以设到 100--1000）。大数据集上，按叶子生长基本不会有问题。

### 调参顺序是什么？

1. 先把学习率设为 `0.05`--`0.1`，用早停法确定 `n_estimators`。
2. 调整树的形状：`max_depth`（LightGBM 用 `num_leaves`）、`min_child_weight` 或 `min_data_in_leaf`。
3. 调整正则化参数：`gamma`、`lambda`，再加上行采样和列采样（`subsample`、`colsample_bytree`）。
4. 如果还是欠拟合，就降低学习率，同时增加 `n_estimators`。

### GBDT 和深度学习怎么选？

表格数据：几乎总是 GBDT 更强。非结构化数据（图像、文本、音频）：深度学习完胜。分界点大概在"是否已经学到了稠密表征"——一旦特征变得稠密且连续，深度网络就能追上来。

---
## 练习题

**练习 1 —— 二阶梯度**
平方损失 $L = \tfrac{1}{2}(y - \hat y)^2$，其中 $y = 5$、$\hat y = 3$。计算得 $g = \hat y - y = -2$，$h = 1$。当 $\lambda = 0$ 时，单叶子的牛顿步长 $w^* = -g/h = 2$，刚好等于残差。

**练习 2 —— 分裂增益**
某叶节点的 $G = -2$、$H = 10$。候选分裂将 $G_L = -1.5, H_L = 6$ 分到左侧，$G_R = -0.5, H_R = 4$ 分到右侧。取 $\lambda = 1, \gamma = 0.5$：

$$\text{Gain} = \tfrac{1}{2}\!\left[\tfrac{2.25}{7} + \tfrac{0.25}{5} - \tfrac{4}{11}\right] - 0.5 \approx -0.50.$$

结构改进不足以抵消 $\gamma$，因此放弃这次分裂。增大 $\gamma$ 会让阈值更严格，减小 $\lambda$ 则会放宽阈值。

**练习 3 —— GOSS 采样预算**
假设 $N = 1000$、$a = 0.2$、$b = 0.1$：保留 $200$ 个大梯度样本，再加 $0.1 \cdot 800 = 80$ 个小梯度样本。有效样本数为 $280$（占 $28\%$）。小样本集的重加权因子为 $(1 - a)/b = 8$。

---
## 参考文献

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.
- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics.
