---
title: "机器学习数学推导（十二）：XGBoost 与 LightGBM"
date: 2026-02-28 09:00:00
tags:
  - 机器学习
  - 梯度提升
  - 正则化
  - XGBoost
  - LightGBM
  - 直方图算法
  - 数学推导
categories: 机器学习
series:
  name: "机器学习数学推导"
  part: 12
  total: 20
lang: zh-CN
mathjax: true
description: "从 XGBoost 的二阶泰勒展开到 LightGBM 的直方图加速，本文系统推导两大工业级梯度提升框架——正则化目标函数、分裂增益闭式解、GOSS 单边采样与 EFB 互斥特征绑定的数学原理。"
disableNunjucks: true
series_order: 12
---

XGBoost 与 LightGBM 是当下表格数据领域最常用的两套库——Kaggle 榜单、风控流水线、广告排序、流失预测，背后多半都是它们。两者共享同一个骨架（梯度提升树，见第十一篇），但在工程取舍上走了完全不同的路：

- **XGBoost** 在*数学*上做文章：把损失的二阶导数引入目标，对树本身做正则化，把分裂选择转化为闭式打分。
- **LightGBM** 在*系统*上做文章：把特征离散为小直方图，按叶子（而非按层）生长，丢掉信息量低的样本（GOSS），把互斥稀疏特征打包合并（EFB）。

API 上两者看起来几乎可以互换，可一旦 $N$ 或 $d$ 变大，行为差异就被无情放大。本文把这些工程选择背后的公式逐一推完，让你读调参指南时能说出每个旋钮*为什么*存在。

## 你将学到

- XGBoost 的二阶泰勒展开如何给出闭式最优叶权重和任意树结构的"结构分数"。
- 分裂增益公式为什么天然带一个剪枝阈值 $\gamma$。
- LightGBM 的直方图算法如何把单特征分裂代价从 $O(N)$ 降到 $O(K)$，以及由此衍生的"直方图减法"技巧。
- GOSS 的精确统计含义，以及 EFB 的构造步骤。
- 什么场景按层生长更好，什么场景按叶子生长更好。

## 前置知识

- GBDT 基础（本系列第十一篇）。
- 一阶/二阶泰勒展开。
- 决策树的不纯度与增益分裂准则。

---

## 先看一张图：boosting 在干什么

公式之前，先看梯度提升真正做的事情。我们从一个常数预测出发（$y$ 的均值），让每棵新树去拟合*当前残差*——前几棵粗略勾出主结构，后几棵不断修正局部误差。

![梯度提升：迭代拟合残差的过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig1_boosting_iterations.png)

第一棵树之后，拟合还只是粗糙的阶梯函数，残差很大；积累一百棵小步长（$\eta = 0.1$）的树之后，整体已经吃下了底层的 $\sin(1.5x) + 0.35x$ 趋势，残差基本只剩噪声。XGBoost 与 LightGBM 的区别只在于*怎么*拟合每一棵树——上面这个迭代骨架两者完全一致。

---

## XGBoost：把数学做到极致

### 正则化目标函数

普通 GBDT 只最小化经验损失。XGBoost 把"什么样的树才算好"也写进了目标：

$$
\mathcal{L}^{(t)} \;=\; \sum_{i=1}^N L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr) \;+\; \Omega(f_t),
$$

其中 $\hat y_i^{(t-1)}$ 是前 $t-1$ 轮的累计预测，$f_t$ 是本轮要新加的树，正则项

$$
\Omega(f_t) \;=\; \gamma\, T \;+\; \tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2.
$$

- $T$ 是叶子数量，$\gamma T$ 是**每叶代价**，相当于一道软剪枝阈值。
- $w_j$ 是叶节点 $j$ 存的预测值，$\tfrac{1}{2}\lambda \sum w_j^2$ 是**叶权重的 L2 收缩**。

### 二阶泰勒展开

把单样本损失在当前预测 $\hat y_i^{(t-1)}$ 处做二阶展开：

$$
L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr)
\;\approx\; L\bigl(y_i,\; \hat y_i^{(t-1)}\bigr) + g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2,
$$

其中梯度与海森：

$$
g_i \;=\; \partial L / \partial \hat y_i^{(t-1)}, \qquad h_i \;=\; \partial^2 L / \partial (\hat y_i^{(t-1)})^2.
$$

略去与 $f_t$ 无关的零阶常数，本轮的代理目标对 $f_t$ 是纯二次的：

$$
\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{i=1}^N \Bigl[g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2\Bigr] + \Omega(f_t).
$$

这是 XGBoost 与传统 GBDT 之间唯一的设计分水岭：二阶导数 $h_i$ 进入目标，相当于*免费*拿到了牛顿法的曲率信息。

### 闭式叶权重与结构分数

把树写成两部分：叶分配函数 $q : \mathbb{R}^d \to \{1,\ldots,T\}$ 与权重向量 $\mathbf{w}$。按叶子分组样本，记 $I_j = \{ i : q(\mathbf{x}_i) = j \}$，并定义

$$
G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i.
$$

目标在叶之间彻底解耦：

$$
\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{j=1}^T \Bigl[G_j\, w_j + \tfrac{1}{2}(H_j + \lambda)\, w_j^2\Bigr] + \gamma T.
$$

每个叶子都成了 $w_j$ 的独立二次函数。令 $\partial / \partial w_j = 0$：

$$
\boxed{\,w_j^{*} \;=\; -\frac{G_j}{H_j + \lambda}\,}
$$

代回得到**结构分数**——这棵树形所能取到的最优损失：

$$
\widetilde{\mathcal{L}}^{*}(q) \;=\; -\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j + \lambda} \;+\; \gamma T.
$$

值越小越好。注意结构分数只依赖于树结构 $q$，权重已经被解析地"消掉"了——树学习因此被转化为结构搜索。

### 分裂增益

考虑把某叶节点拆成左 ($I_L$)、右 ($I_R$) 两个子节点，结构分数的变化是

$$
\boxed{\;\text{Gain} \;=\; \frac{1}{2}\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma\;}
$$

有两点值得停下来体会：

1. 中括号里那一项是**结构分数和的下降量**，未减 $\gamma$ 之前总是 $\ge 0$（柯西-施瓦茨/方差分解）。
2. 常数 $\gamma$ 在这里直接当**阈值**用：如果结构改善撑不到 $\gamma$，分裂被否决——根本不需要后剪枝步骤，剪枝已经写进了增益公式。

### 常见损失函数的梯度

| 损失 | $g_i$ | $h_i$ |
|---|---|---|
| 平方损失 $\tfrac{1}{2}(y-\hat y)^2$ | $\hat y_i - y_i$ | $1$ |
| Logistic（$p = \sigma(\hat y)$） | $p_i - y_i$ | $p_i(1-p_i)$ |
| Softmax（类别 $c$） | $p_{ic} - \mathbb{1}[y_i = c]$ | $p_{ic}(1-p_{ic})$ |

平方损失的 $h_i \equiv 1$，二阶目标退化为普通残差拟合，XGBoost 就回到了 GBDT。Logistic 和 Softmax 的海森携带真实信息（饱和区 $p(1-p)$ 很小），牛顿步明显比一阶下降更值。

### 分裂查找算法

有了增益公式，剩下的只是*怎么*枚举候选分裂。把每个特征都预排序就是精确算法，单特征单节点 $O(N)$；*近似*算法只挑少量候选分位点（用 $h_i$ 加权，因为 $h_i$ 在二次目标中扮演样本重要性）。**稀疏感知**变体在每次分裂时学习一个"缺失值默认走向"，遍历时只扫非缺失项——在稀疏 one-hot 数据上是默默无闻的大胜利。

![XGBoost 精确预排序 vs LightGBM 直方图分裂查找](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig2_split_finding.png)

左图扫遍所有不同取值（共 $N-1$ 个候选）。右图把同一份数据分进 $K = 32$ 个桶，只评估 $K-1$ 个候选切点——却几乎挑到了同一个阈值、同一个增益。这正是 LightGBM 的核心赌注：$N$ 维度的分辨率绝大部分都是浪费。

---

## LightGBM：把工程做到极致

### 直方图算法

LightGBM 在训练前一次性把每个特征离散为 $K$ 个整数桶（默认 255）。每个叶子、每个特征都建一份 $(G_b, H_b)$ 直方图：

$$
G_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} g_i, \qquad H_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} h_i.
$$

单特征复杂度从 $O(N)$ 跌到 $O(K)$，存储和计算同时受益：

| 维度 | 精确（XGBoost） | 直方图（LightGBM） |
|---|---|---|
| 单特征内存 | $O(N)$ | $O(K)$ |
| 单特征分裂查找 | $O(N)$ | $O(K)$ |
| 缓存友好度 | 差（预排序后随机访问） | 好（桶在内存中连续） |

还有一个**直方图减法**技巧：兄弟节点的直方图可以由*父节点直方图减去已构建子节点*得到。换句话说，构建两个子节点的代价就是构建一个的代价。在深树里这把每层的开销都砍了一半。

### 按叶子生长 vs 按层生长

XGBoost 默认**按层生长**（BFS）：当前深度的所有节点都分裂完，才进入下一层。LightGBM 默认**按叶子生长**（best-first）：始终去分裂当前增益最大的那一个叶子。

![Leaf-wise 与 Level-wise 树生长对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig5_growth.png)

在相同节点预算下，按层生长会得到一棵深度 $\log_2 T$ 的平衡树；按叶子生长则可能形成一条又细又长的链，深深扎进输入空间的某一区域。同样数量的叶子，按叶子生长几乎总能取得更低的训练损失——但如果不卡 `max_depth` 和 `min_data_in_leaf`，过拟合也来得格外猛。

### GOSS：基于梯度的单边采样

海森加权后的梯度 $g_i$ 直接告诉你样本 $i$ 对下一个分裂能贡献多少。一棵已经拟合得不错的树里，绝大多数样本的 $|g_i| \approx 0$（早就被拟合好了）。把它们扔掉几乎不损失信息——*前提是*重新调权，让 $G$ 和 $H$ 的统计量保持无偏。

GOSS 用 $a, b \in (0, 1)$ 三步搞定：

1. 按 $|g_i|$ 降序，保留前 $a\cdot N$ 个样本，记为 $A$。
2. 从剩下的 $(1-a) N$ 中均匀随机抽 $b \cdot N$ 个，记为 $B$。
3. 计算 $G_L, H_L, G_R, H_R$ 时，$B$ 中样本统一乘上 $\dfrac{1-a}{b}$ 来抵消子采样。

最终的增益估计变成

$$
\widetilde{\text{Gain}} \;=\; \frac{1}{2}\!\left[\frac{(G_L^A + \tfrac{1-a}{b}G_L^B)^2}{H_L^A + \tfrac{1-a}{b}H_L^B + \lambda} + \frac{(G_R^A + \tfrac{1-a}{b}G_R^B)^2}{H_R^A + \tfrac{1-a}{b}H_R^B + \lambda} - \cdots \right].
$$

LightGBM 论文证明：这种子采样引入的方差是 $O(1/\sqrt{n_l})$，比叶子内部本身的采样噪声衰减得还快。

![GOSS 保留信息丰富的尾部，对小梯度样本重加权](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig3_goss.png)

右图是关键：取 $a = 0.20, b = 0.10$，每轮只接触 **30% 的样本**，重加权后的 $G$ 总和仍与全数据 $G$ 在期望上相等。

### EFB：互斥特征绑定

高维稀疏特征——one-hot 类目、词袋、点击标志——很少同时取非零：一行有 `country = JP` 就不会同时有 `country = US`。EFB 利用这种**互斥性**压缩特征轴。

具体做法是建一张*冲突图*：节点是特征，边权是两特征*同时*非零的行数。把彼此无边的特征装进同一个组，是图着色问题。EFB 用贪心解：按度数降序遍历特征，每个特征放进第一个"和它没冲突邻居"的现有桶。

选定一个组 $\{f_{i_1}, f_{i_2}, \ldots\}$ 之后，把它们用整数偏移合并成一列 $\tilde f$，使各自的桶段不重叠：

$$
\tilde f_n \;=\; \begin{cases} \text{bin}(x_{n, i_k}) + o_k & \text{若 } x_{n, i_k} \neq 0 \\ 0 & \text{否则} \end{cases}, \qquad o_k = \sum_{r < k} K_{i_r}.
$$

![EFB 把多列稀疏互斥特征合并为一列](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig4_efb.png)

A 面板是一段近乎互斥的稀疏块（想象 6 个 one-hot 哑变量）。B 面板的红边标出违反互斥的特征对，节点颜色是贪心着色给出的组号。C 面板是合并后的列：每个原特征占据互不重叠的桶段，所以 $\tilde f$ 的直方图能精确还原各原特征的统计——但*列数减少了*。

在高维稀疏问题上，EFB 经常把有效 $d$ 砍掉 5--10 倍，直方图构建的加速由此被进一步放大。

---

## 两种"重要性"：同一个模型，两个答案

XGBoost 与 LightGBM 都暴露了特征重要性，但它们回答的问题不同。

![特征重要性：XGBoost gain 与 LightGBM split 计数](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig6_feature_importance.png)

- **XGBoost (gain)** 把每次用某特征分裂时的增益累加起来。它奖励那些*一次大裂变*就把损失劈开的特征。
- **LightGBM (split count)** 数一个特征被选作分裂的次数。它奖励那些*被频繁选用*的特征，哪怕每次贡献都不大。

两个排名常常不一致——但都没错，只是问题不同。想知道"哪些特征推动了损失下降"，看 gain；想知道"模型在迭代过程中依赖哪些特征"，看 split。要更模型无关的解释，可以用 SHAP，它度量的是每个特征对每个预测的边际贡献。

---

## XGBoost vs LightGBM 速查表

| 维度 | XGBoost | LightGBM |
|---|---|---|
| 分裂策略 | 按层（BFS） | 按叶子（best-first） |
| 基础算法 | 预排序精确 / 分位 sketch | 直方图，$K$ 桶 |
| 单特征内存 | $O(N)$ | $O(K)$ |
| 样本效率 | 行/列子采样 | GOSS |
| 稀疏 / 类别特征 | 稀疏感知默认方向 | EFB + 原生类别支持 |
| 失败模式 | 大 $N$ 时偏慢 | 不限 `max_depth` 易过拟合 |
| 适用场景 | 小/中 $N$、需稳健调参 | 大 $N$、大 $d$、多稀疏特征 |

### 实战训练成本

直方图 + GOSS + EFB 这一套，换来的是 LightGBM 的吞吐量优势：

![训练时间随样本数变化与 Pareto 视图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/12-XGBoost%E4%B8%8ELightGBM/fig7_time_vs_accuracy.png)

$N = 10^4$ 时三者只差几秒，选哪个都无所谓。$N = 10^6$ 时，LightGBM 比 XGBoost 快约 5 倍，而测试精度几乎不可分辨。CatBoost 在速度上夹在中间，但靠 ordered boosting 在类别特征密集的场景独占一席。

---

## 极简 NumPy 版 XGBoost

下面这段代码包含了二阶目标、闭式叶权重、增益分裂以及 $\gamma$ 剪枝：

```python
import numpy as np


class XGBoostTree:
    """单棵 XGBoost 树：闭式叶权重 + 增益分裂"""

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
        leaf_weight = -G / (H + self.lambda_)            # 闭式解
        if depth >= self.max_depth or len(X) < 2:
            return {"w": leaf_weight}
        split, gain = self._best_split(X, g, h)
        if split is None or gain <= 0:                    # 此处即 gamma 剪枝
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
    """二阶梯度+海森的可加集成"""

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
        s = 1.0 / (1.0 + np.exp(-p))                     # logistic
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

整套二阶机制能在 100 行内写完，是因为有了增益公式之后，算法剩下的部分就只是"找最优分裂、递归、重复"。

---

## Q&A 精选

### 为什么一定要用二阶信息？

一阶给方向，二阶给曲率——也就是步长能跨多远。牛顿法在最优点附近二次收敛，正是因为它把曲率算了进去。在每个叶子上，二阶目标直接给出*闭式*最优权重 $-G/(H+\lambda)$，不再需要靠学习率去试。

### 为什么 XGBoost 通过 $\gamma$ 剪枝，而不是事后剪？

因为结构分数本身就是减去了 $\gamma T$ 的损失。一个增益不超过 $\gamma$ 的分裂，会让正则化损失*上升*。拒绝它不是启发式，而是目标函数下的正确决策。

### 按叶子生长什么时候反而吃亏？

小数据。最高增益的那个叶子可能在追逐噪声。补救方法是 `max_depth` + `min_data_in_leaf`（小数据上有时要设到 100--1000）。大数据上按叶子生长几乎是免费午餐。

### 调参顺序？

1. 学习率定在 `0.05`--`0.1`，用早停定 `n_estimators`。
2. 调树形：`max_depth`（LightGBM 用 `num_leaves`）、`min_child_weight` / `min_data_in_leaf`。
3. 调正则：`gamma`、`lambda`，加上行/列采样。
4. 还欠拟合就再降学习率、加 `n_estimators`。

### GBDT 和深度学习怎么选？

表格数据：几乎永远 GBDT 更好。非结构化（图像、文本、音频）：深度学习碾压。分界点大致在"是否已经有一份学好的稠密表征"——一旦特征是稠密连续的，深度网才追得上。

---

## 练习题

**练习 1 ——二阶梯度。**
平方损失 $L = \tfrac{1}{2}(y - \hat y)^2$，$y = 5$、$\hat y = 3$：$g = \hat y - y = -2$，$h = 1$。在 $\lambda = 0$ 的单叶子上，牛顿步 $w^* = -g/h = 2$，正好等于残差。

**练习 2 ——分裂增益。**
某叶节点 $G = -2, H = 10$。候选分裂 $G_L = -1.5, H_L = 6$、$G_R = -0.5, H_R = 4$。取 $\lambda = 1, \gamma = 0.5$：

$$
\text{Gain} = \tfrac{1}{2}\!\left[\tfrac{2.25}{7} + \tfrac{0.25}{5} - \tfrac{4}{11}\right] - 0.5 \approx -0.50.
$$

结构改善撑不到 $\gamma$，分裂被拒。$\gamma$ 越大，阈值越严；$\lambda$ 越小，阈值越松。

**练习 3 ——GOSS 采样预算。**
$N = 1000$、$a = 0.2$、$b = 0.1$：保留 $200$ 个大梯度样本，加上 $0.1 \cdot 800 = 80$ 个小梯度样本。有效样本 $280$（$28\%$）。小样本的重加权因子 $(1 - a)/b = 8$。

---

## 参考文献

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., Meng, Q., Finley, T., 等 (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova, L., Gusev, G., Vorobev, A., 等 (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.
- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics.

---

## 系列导航

- 上一篇：[第十一篇 -- 集成学习](/zh/机器学习数学推导-十一-集成学习/)
- 下一篇：[第十三篇 -- EM 算法与 GMM](/zh/机器学习数学推导-十三-EM算法与GMM/)
- [查看本系列全部 20 篇文章](/tags/机器学习/)
