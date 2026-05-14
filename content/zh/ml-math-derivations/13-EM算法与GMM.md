---
title: "机器学习数学推导（十三）：EM 算法与 GMM"
date: 2026-02-01 09:00:00
tags:
  - EM Algorithm
  - Expectation Maximization
  - Gaussian Mixture Model
  - GMM
  - Latent Variables
  - Mathematical Derivations
  - Machine Learning
categories: 机器学习
series: ml-math-derivations
lang: zh
mathjax: true
description: "从 Jensen 不等式与 ELBO 出发推导 EM 算法，证明其单调上升性，并完整给出高斯混合模型（GMM）的 E 步、M 步更新公式、模型选择以及与 K-means 的关系。"
disableNunjucks: true
series_order: 13
translationKey: "ml-math-derivations-13"
---
数据中常隐含难以观测的结构：样本所属的簇未知，某些特征的真实值缺失，文本背后的潜在主题也不明确。这些隐变量让最大似然估计变得棘手——似然函数变成“对数里面套求和”的形式，既没有闭式解，梯度方法也容易被隐变量困住。**EM 算法**用一个看似简单的思路巧妙绕开这一难题：交替进行两步操作——先基于当前参数下的隐变量后验分布计算期望（E 步），再将这些期望当作真实值来更新模型参数（M 步）。每次迭代都严格保证对数似然值不减。本文将从第一性原理出发推导 EM 算法，利用 Jensen 不等式证明其单调上升性质，并将其应用于最经典的场景——**高斯混合模型（GMM）**，即 K-means 的软化、椭球化推广。

![机器学习数学推导（十三）：EM算法与GMM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/illustration_1.png)

---

## 你将学到什么

- 隐变量为何使 MLE 困难（“log 套 sum”问题）
- 如何借助 Jensen 不等式构建**证据下界（ELBO）**
- EM 算法本质上是在 $(q, \boldsymbol{\theta})$ 上对 ELBO 进行**交替最大化**
- 严谨证明 $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$
- GMM 的完整 E 步与 M 步公式（支持全协方差）
- K-means 是 GMM 在硬分配、球形协方差下的极限情形
- 使用 **BIC / AIC** 选择分量数 $K$

## 前置知识

- 最大似然估计
- 多元高斯分布密度
- Jensen 不等式与 KL 散度
- K-means 聚类

---
## 隐变量与不完全数据似然

### 设定

我们用联合分布 $p(\mathbf{x}, z \mid \boldsymbol{\theta})$ 对观测数据 $\mathbf{x}_1,\dots,\mathbf{x}_N$ 和隐变量 $z_1,\dots,z_N$ 建模。实际中只能观测到 $\mathbf{X}$，而 $\mathbf{Z}$ 始终不可见。**不完全数据对数似然**定义为：
$$
\ell(\boldsymbol{\theta}) \;=\; \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \boldsymbol{\theta})
\;=\; \sum_{i=1}^{N} \log \sum_{z} p(\mathbf{x}_i, z \mid \boldsymbol{\theta}).
$$
问题根源在于对数内部的求和。由于该求和项存在，对数无法分解为各成分的独立项，导致梯度无法拆解，也无法获得闭式最大化解。

### 混合模型的例子

以包含 $K$ 个成分的高斯混合模型为例：
$$p(\mathbf{x}\mid \boldsymbol{\theta}) \;=\; \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).$$
若已知每个样本来自哪个成分，参数估计就退化为 $K$ 个独立的加权高斯 MLE——轻而易举。但现实中我们并不知道成分归属，而这正是 EM 算法要解决的核心问题。

![GMM 三个高斯成分及其协方差椭圆](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig1_gmm_clusters.png)

上图展示了 `sklearn.mixture.GaussianMixture` 在二维数据上拟合出的三个高斯成分。每个叉号代表均值 $\boldsymbol{\mu}_k$，内侧椭圆是 1 倍标准差等高线，外侧椭圆是 2 倍标准差，$\pi_k$ 是对应的混合权重。

---
## ELBO 与 Jensen 不等式

### 引入辅助分布 $q$

任选一个定义在隐变量上的分布 $q(z)$，对原式做乘除变换：
$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
= \log \sum_{z} q(z)\, \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}.
$$
由于 $\log$ 是凹函数，应用 **Jensen 不等式**可得：
$$
\boxed{\;
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;\geq\;
\sum_{z} q(z)\, \log \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}
\;\equiv\;
\mathcal{L}(q,\boldsymbol{\theta}).
\;}
$$
该下界 $\mathcal{L}$ 称为 **证据下界（ELBO, Evidence Lower Bound）**，它同时依赖于变分分布 $q$ 和模型参数 $\boldsymbol{\theta}$。

### 精确分解

通过直接代数变换（无需不等式），可得到如下恒等式：
$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;=\;
\mathcal{L}(q,\boldsymbol{\theta})
\;+\;
\mathrm{KL}\bigl[q(z)\,\Vert\, p(z\mid \mathbf{x},\boldsymbol{\theta})\bigr].
$$
由此立即得出两个关键结论：

1. ELBO 始终不超过真实对数似然，即 $\mathcal{L} \leq \log p(\mathbf{x}\mid\boldsymbol{\theta})$，因为 KL 散度非负；
2. 当且仅当 $q(z) = p(z \mid \mathbf{x}, \boldsymbol{\theta})$（即后验分布）时，ELBO 与真实似然相等，下界达到紧致。

这一恒等式构成了 EM 算法的全部理论基础。

---
## EM 是 ELBO 上的坐标上升

![机器学习数学推导（十三）：EM算法与GMM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/illustration_2.png)

EM 算法通过交替优化 $\mathcal{L}$ 关于 $q$ 和 $\boldsymbol{\theta}$ 的两个变量，逐步提升下界。

### 两步走

**E 步**  
固定当前参数 $\boldsymbol{\theta}^{(t)}$，最大化 $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$ 关于 $q$。最优解即为后验分布：
$$q^{(t)}(z) \;=\; p\bigl(z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}\bigr).$$
此步完成后，下界变为紧致：$\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) = \log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})$。

**M 步**  
固定 $q^{(t)}$，最大化 $\mathcal{L}(q^{(t)}, \boldsymbol{\theta})$ 关于 $\boldsymbol{\theta}$。由于 $q^{(t)}$ 的熵与 $\boldsymbol{\theta}$ 无关，该步骤等价于最大化 **Q 函数**：
$$
Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})
\;=\;
\mathbb{E}_{z\sim q^{(t)}}\!\bigl[\log p(\mathbf{x}, z\mid \boldsymbol{\theta})\bigr].
$$
直观上，Q 函数即为在当前后验下完全数据对数似然的期望。

### 单调上升的证明

将以下三步不等式串联：
$$
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})
\;\overset{(a)}{=}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)})
\;\overset{(b)}{\leq}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)})
\;\overset{(c)}{\leq}\;
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t+1)}).
$$
其中，(a) 成立是因为 E 步使下界紧致；(b) 由 M 步的定义保证（$\boldsymbol{\theta}^{(t+1)}$ 是 $\mathcal{L}(q^{(t)}, \cdot)$ 的最大化点）；(c) 则源于 ELBO 恒小于等于真实对数似然。因此，
$$\boxed{\;\ell(\boldsymbol{\theta}^{(t+1)}) \;\geq\; \ell(\boldsymbol{\theta}^{(t)})\;}$$
在每次迭代中均成立，仅在不动点处取等号。EM 收敛至 $\ell$ 的驻点——通常是局部极大值，偶尔为鞍点。**它不保证收敛到全局最优**，因此实践中需采用多次随机或 K-means 初始化，并保留最优结果。

### 两条曲线的视角

从 ELBO 视角看，EM 的动态过程非常清晰：每次 E 步将 KL 差距压缩至零；随后 M 步同步抬升对数似然与 ELBO，差距重新打开，直至下一次 E 步。

![ELBO 与对数似然的轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig7_elbo_vs_loglik.png)

图中琥珀色带表示 KL 散度 $\mathrm{KL}\bigl[q\,\Vert\, p(z\mid\mathbf{x},\boldsymbol{\theta})\bigr]$。绿色圆点标记 E 步完成后的时刻，此时 KL 散度按构造为零。

---
## EM 算法与高斯混合模型

### 模型

单个观测值的生成过程如下：

1. 从分类分布中采样成分标签 $z_i \sim \mathrm{Categorical}(\pi_1,\dots,\pi_K)$；
2. 给定 $z_i = k$，从高斯分布中采样 $\mathbf{x}_i \mid z_i = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。

模型参数为 $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$，满足 $\sum_k \pi_k = 1$，且每个 $\boldsymbol{\Sigma}_k \succ 0$。

### E 步：计算责任度

由于隐变量离散，后验概率可直接由贝叶斯公式得出。定义成分 $k$ 对样本 $i$ 的**责任度（responsibility）**：
$$
\boxed{\;
\gamma_{ik}
\;=\;
p\bigl(z_i = k \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}\bigr)
\;=\;
\frac{\pi_k\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
\;}
$$
每行 $(\gamma_{i1},\dots,\gamma_{iK})$ 之和为 1，表示样本对各成分的软分配概率。

![E 步：软分配](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig2_e_step.png)

左图中，每个网格点的颜色由三个成分的责任度加权混合而成：单一成分主导区域颜色纯净，边界处则呈现混合色。右图展示了 12 个样本点的责任度矩阵 $\gamma_{ik}$，每行和为 1。

### M 步：加权最大似然估计

将高斯密度代入 Q 函数并引入拉格朗日乘子处理 $\sum_k \pi_k = 1$ 约束，可得闭式更新公式。令 $N_k = \sum_{i=1}^{N} \gamma_{ik}$ 表示成分 $k$ 的有效样本数：
$$
\boxed{\;
\pi_k = \frac{N_k}{N},\qquad
\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,\mathbf{x}_i,\qquad
\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\!\top}.
\;}
$$
这些公式与标准高斯 MLE 形式一致，只是每个样本按其责任度加权。

![一次 M 步：更新前 vs 更新后](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig3_m_step.png)

即使从一个故意设置得很差的初始值开始，仅一次 M 步就能将均值（红色箭头所示）拉向数据，并调整协方差椭圆以匹配观测分布。经过少数几次 E-M 迭代，拟合结果已基本准确。

### 实战中的收敛性

对 EM 进行多次随机重启并观察对数似然变化：

![对数似然单调不减](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig4_loglik_monotone.png)

每条曲线均单调非减——这是算法的理论保证。不同重启可能收敛至不同局部最优；虚线是 `sklearn.mixture.GaussianMixture` 在 `n_init=20` 下找到的最佳值。**实践中建议采用多次随机或 K-means 初始化，并保留效果最好的结果。**

---
## K-means 是 GMM 的硬分配、球形极限

设所有 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$，并令 $\epsilon \to 0$。此时高斯密度趋于无限尖锐，离样本最近的均值对应的责任度趋近于 1，其余趋近于 0。E 步退化为**硬分配**，M 步退化为对分配点求均值——这正是 K-means 的更新规则。

![各向异性数据上 K-means 与 GMM 对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig6_kmeans_vs_gmm.png)

在各向异性数据上，两者差异显著：K-means（左）强制使用球形 Voronoi 单元，将拉长的簇切割得十分生硬；GMM（右）则用椭圆沿数据主轴方向贴合分布。**只要簇非各向同性，或需要软隶属概率，GMM 就应作为默认选择。**

## 如何选择成分数 $K$

似然值随 $K$ 增大而单调递增（模型更灵活），故仅凭 $\ell$ 无法确定 $K$。需采用带复杂度惩罚的准则：
$$
\mathrm{BIC}(K) = -2\,\hat{\ell}(K) + p_K\,\log N,
\qquad
\mathrm{AIC}(K) = -2\,\hat{\ell}(K) + 2\,p_K,
$$
其中 $p_K$ 为参数总数。对于 $d$ 维全协方差 GMM，有 $p_K = (K-1) + Kd + K\frac{d(d+1)}{2}$。

![BIC 和 AIC 随成分数 K 的变化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig5_bic_aic.png)

从 $K=1$ 到真实值 $K=3$，两条曲线迅速下降，之后趋于平缓甚至回升。BIC 因含额外 $\log N$ 因子，对复杂度惩罚更重，通常偏好更小的 $K$。但在此例中，两者均给出了正确答案。

---
## 参考实现

此处提供一个极简的 NumPy 实现，严格对应前述公式。本文所有实验与图表均通过 `sklearn.mixture.GaussianMixture` 验证。

```python
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    """全协方差高斯混合模型，使用 EM 算法训练。"""

    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg=1e-6):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    # ----- E 步：计算责任度 -----
    def _e_step(self, X):
        comp = np.column_stack([
            self.weights[k] * multivariate_normal.pdf(
                X, self.means[k], self.covs[k])
            for k in range(self.K)
        ])
        ll = np.log(comp.sum(axis=1) + 1e-300).sum()
        return comp / (comp.sum(axis=1, keepdims=True) + 1e-300), ll

    # ----- M 步：更新参数 -----
    def _m_step(self, X, gamma):
        N_k = gamma.sum(axis=0)
        self.weights = N_k / X.shape[0]
        self.means = (gamma.T @ X) / N_k[:, None]
        d = X.shape[1]
        for k in range(self.K):
            diff = X - self.means[k]
            self.covs[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k]
            self.covs[k] += self.reg * np.eye(d)  # 加入正则化避免奇异矩阵

    def fit(self, X):
        N, d = X.shape
        rng = np.random.default_rng(0)
        idx = rng.choice(N, self.K, replace=False)
        self.means = X[idx].copy()
        self.weights = np.full(self.K, 1.0 / self.K)
        self.covs = np.array([np.cov(X.T) + self.reg * np.eye(d)] * self.K)

        prev = -np.inf
        for it in range(self.max_iter):
            gamma, ll = self._e_step(X)
            self._m_step(X, gamma)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict(self, X):
        return np.argmax(self._e_step(X)[0], axis=1)
```
## 数值实现的几个坑

上述推导虽数学优雅，但工程实现中潜藏诸多陷阱。我在生产环境中反复遭遇以下三类问题。

**责任度下溢。** 高维高斯的对数密度通常为很大的负数。当维度 $D \gtrsim 50$ 时，直接计算 $\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ 的指数在 float32 下会下溢为零，导致责任度出现 $0/0$。解决方案是采用 log-sum-exp 技巧：
$$\log \gamma_{ik} = \log\pi_k + \log\mathcal{N}_k(\mathbf{x}_i) - \mathrm{logsumexp}_j\big(\log\pi_j + \log\mathcal{N}_j(\mathbf{x}_i)\big),$$
再通过 $\gamma_{ik} = \exp(\log \gamma_{ik})$ 恢复原始值。**务必在归一化前全程保持对数运算。**

**协方差矩阵退化。** 当某成分仅覆盖单个点时，$\boldsymbol{\Sigma}_k$ 趋向秩零矩阵，行列式趋于零，似然值爆炸至 $+\infty$。这虽是 MLE 的正确解，但无实用价值。两种常用缓解措施：(1) 添加岭项 $\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \lambda \mathbf{I}$，其中 $\lambda \approx 10^{-6} \cdot \mathrm{tr}(\boldsymbol{\Sigma}_k)/D$；(2) 若某成分的有效计数 $N_k = \sum_i \gamma_{ik}$ 低于阈值（如 $N_k < 1$），则重新初始化该成分。

**$\boldsymbol{\Sigma}_k^{-1}$ 的条件数问题。** 计算马氏距离 $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ 时，切勿显式求逆。正确做法是对 $\boldsymbol{\Sigma}$ 做 Cholesky 分解 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$，再通过前代法解 $\mathbf{L}\mathbf{y} = \mathbf{x} - \boldsymbol{\mu}$。此时距离平方为 $\Vert \mathbf{y}\Vert^2$，对数行列式为 $2\sum_d \log L_{dd}$。该方法将复杂度从 $O(D^3)$ 降至 $O(D^2)$，且在 $\boldsymbol{\Sigma}$ 条件数较大时数值更稳定。

迭代过程中可设一监控指标：ELBO 必须单调非减。若下降幅度超过 $10^{-6}$，应优先排查数值实现错误，而非质疑数学推导。

## scikit-learn 中的实现

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',   # 'diag' 在高维数据上快得多
    reg_covar=1e-6,           # 岭正则化项
    init_params='k-means++',  # 用 K-means 初始化；纯随机初始化容易发散
    n_init=5,                 # 运行 5 次，保留 ELBO 最优的结果
    max_iter=200,
    tol=1e-4,
)
gmm.fit(X)
log_resp = gmm.predict_proba(X)         # γ_{ik}
bic = gmm.bic(X)                        # 用于选择 K 值
```

有两个参数值得特别关注。`covariance_type='diag'` 会舍弃 $\boldsymbol{\Sigma}_k$ 的非对角元素，将参数量从 $K D(D+1)/2$ 减至 $KD$，单次迭代复杂度也从 $O(NKD^2)$ 降至 $O(NKD)$。对 $D=128$ 的嵌入向量，这能将运行时间从小时级缩短至分钟级。`init_params='k-means++'` 会先运行 K-means 初始化均值；否则当分量数超过 4 时，EM 常收敛至退化解（如某成分吸收全部数据）。

BIC 分数 $\mathrm{BIC} = -2\log\hat L + p\log N$（$p$ 为参数量）是选择 $K$ 最简便且有理论依据的方法。遍历 $K \in \{1, \dots, K_{\max}\}$ 并选取拐点即可。**切勿依赖单次拟合结果——务必设置 `n_init >= 5`，因 EM 仅能收敛至局部最优。**

---
## 常见问题

### EM 能找到全局最优解吗？

不能。单调上升仅保证收敛至驻点（通常为局部极大值，偶为鞍点）。建议采用多次重启（随机或 K-means 初始化），并保留最终对数似然最高的结果。

### GMM 和 K-means 的区别在哪？何时需特别注意？

以下情况应选用 GMM：  
(i) 簇呈明显椭圆或各向异性；  
(ii) 下游任务需软隶属概率用于校准；  
(iii) 需要生成式密度模型进行采样或异常检测。  

若簇大致球形且分离良好，K-means 更快且足够。

### 协方差矩阵奇异或成分坍塌怎么办？

可添加岭正则项 $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$（参考实现已采用）。也可共享协方差、限制为对角形式，或在检测到坍塌时重启成分。

### 为何 E 步能使下界变紧？

因 $\log p = \mathcal{L}(q,\boldsymbol{\theta}) + \mathrm{KL}[q\Vert p(z\mid\mathbf{x},\boldsymbol{\theta})]$，而 KL 散度在 $q$ 等于后验时恰为零。

### 广义 EM 是什么？

广义 EM 将 M 步的“完全最大化”放宽为“任何能提升 $Q$ 的更新”（如单步梯度上升）。单调上升性质依然成立。

### EM 与变分推断有何关联？

变分 EM 将 E 步的真实后验放宽至可处理分布族 $q \in \mathcal{Q}$。分解式 $\log p = \mathcal{L} + \mathrm{KL}$ 不变，算法交替执行：在 $\mathcal{Q}$ 内最小化 KL（E 步），以及对 $\boldsymbol{\theta}$ 最大化 $\mathcal{L}$（M 步）。详见第 14 篇。

---
## 练习题
**E1 （E 步计算）.** 考虑一维 GMM，$K=2$，先验相等 $\pi_1 = \pi_2 = 1/2$，$\mu_1 = 0,\, \mu_2 = 3,\, \sigma^2 = 1$。求 $x_i = 1.5$ 时的 $\gamma_{i1}$。

*解.* 由对称性，两分量密度在 $x=1.5$ 处相等，故 $\gamma_{i1} = 1/2$。

**E2 （M 步更新）.** 样本 $x_1 = 1,\; x_2 = 4$ 的责任值为 $\gamma_{11} = 0.8,\; \gamma_{21} = 0.3$。求 M 步更新后的 $\mu_1$。

*解.* $N_1 = 0.8 + 0.3 = 1.1$，故 $\mu_1 = (0.8 \cdot 1 + 0.3 \cdot 4) / 1.1 = 2.0 / 1.1 \approx 1.82$。

**E3 （单调性分析）.** 在链式关系 $\log p(\boldsymbol{\theta}^{(t)}) = \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) \leq \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)}) \leq \log p(\boldsymbol{\theta}^{(t+1)})$ 中，M 步最优性体现在哪？ELBO 不等式又体现在哪？

*解.* 中间 $\leq$ 由 M 步定义（$\boldsymbol{\theta}^{(t+1)}$ 最大化 $\mathcal{L}(q^{(t)}, \cdot)$）；右侧 $\leq$ 是 ELBO 不等式在新参数下的应用。

**E4 （奇异极限退化为 K-means）.** 证明：若固定 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$ 并令 $\epsilon \to 0^+$，EM 的均值更新将退化为 K-means 更新。

*提示.* 写出 $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \epsilon\mathbf{I}) \propto \exp(-\Vert \mathbf{x} - \boldsymbol{\mu}_k\Vert^2 / (2\epsilon))$。当 $\epsilon \to 0$ 时，对 $-\Vert\mathbf{x}-\boldsymbol{\mu}_k\Vert^2$ 的 softmax 退化为硬 argmin，$\gamma_{ik}\in\{0,1\}$，M 步均值更新即 K-means 的簇均值更新。

**E5 （缺失数据处理）.** 设样本 $\mathbf{x}_i$ 的第 $j$ 维缺失，设计 EM 方法将缺失值视为隐变量，并联合更新参数与缺失值。

*提示.* 将缺失值 $x_{ij}$ 加入隐变量 $z$。E 步计算 $\mathbb{E}[x_{ij} \mid \text{已观测}, \boldsymbol{\theta}^{(t)}]$（含二阶矩）。M 步将这些期望作为插补值代入加权 MLE 公式。

---
## 参考文献

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. **Chapter 9**.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. **Chapter 11**.
- McLachlan, G., & Krishnan, T. (2007). *The EM Algorithm and Extensions* (2nd ed.). Wiley.
- Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 89, 355-368.
