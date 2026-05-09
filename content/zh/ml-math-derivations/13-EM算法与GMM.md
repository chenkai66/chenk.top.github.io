---
title: "机器学习数学推导（十三）：EM算法与GMM"
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
数据里经常藏着看不见的结构——样本属于哪个簇不知道，某个特征的真实值缺失了，一段文本背后的主题也摸不着。这些隐变量让最大似然估计变得很难处理：似然函数变成“对数里面套求和”的形式，既没有闭式解，梯度方法也容易卡在隐变量上。**EM 算法**用一个看似简单的思路绕开了这个问题：先根据隐变量的后验分布“猜”一次（E 步），再把参数当成已知数据来“拟合”一次（M 步），交替进行。每次迭代都能保证似然值不会下降。我从第一性原理推导 EM 算法，用 Jensen 不等式证明它的单调上升性质，最后把它应用到最经典的场景——**高斯混合模型（GMM）**，也就是 K-means 的软化、椭球化版本。

![机器学习数学推导（十三）：EM算法与GMM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/illustration_1.png)
## 你将学到

- 隐变量为什么让 MLE 变得困难（"log 套 sum"问题）
- 如何用 Jensen 不等式构建**证据下界（ELBO）**
- EM 算法就是在 $(q, \boldsymbol{\theta})$ 上对 ELBO 进行**交替优化**
- 严谨证明 $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$
- GMM 的完整 E 步和 M 步公式（支持任意协方差）
- K-means 是 GMM 在硬分配、球形协方差下的特例
- 使用 **BIC / AIC** 确定分量数 $K$
## 前置知识

- 最大似然估计
- 多元高斯分布密度
- Jensen 不等式与 KL 散度
- K-means 聚类

---
## 1. 隐变量与不完全数据似然

### 1.1 设定

我用联合分布 $p(\mathbf{x}, z \mid \boldsymbol{\theta})$ 来建模观测值 $\mathbf{x}_1,\dots,\mathbf{x}_N$ 和隐变量 $z_1,\dots,z_N$。实际中，我只能看到 $\mathbf{X}$，看不到 $\mathbf{Z}$。**不完全数据对数似然**的公式是：

$$\ell(\boldsymbol{\theta}) \;=\; \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \boldsymbol{\theta})
\;=\; \sum_{i=1}^{N} \log \sum_{z} p(\mathbf{x}_i, z \mid \boldsymbol{\theta}).$$

问题出在 log 里面的求和符号。因为这个求和，log 不再能分解到各个成分上，梯度也没法分项计算，更找不到闭式解。

### 1.2 混合模型的例子

对于一个包含 $K$ 个成分的高斯混合模型，概率密度函数是这样的：

$$p(\mathbf{x}\mid \boldsymbol{\theta}) \;=\; \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).$$

如果我知道每个点来自哪个成分，问题就简单了：直接拆成 $K$ 个独立的加权高斯最大似然估计（MLE），轻松搞定。但问题是我不知道，而这正是 EM 算法要解决的。

![GMM 三个高斯成分及其协方差椭圆](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig1_gmm_clusters.png)

上图展示了 `sklearn.mixture.GaussianMixture` 在二维数据上拟合出的三个高斯成分。每个叉号代表均值 $\boldsymbol{\mu}_k$，内侧椭圆是 1 倍标准差等高线，外侧椭圆是 2 倍标准差，$\pi_k$ 是对应的混合权重。

---
## 2. ELBO 与 Jensen 不等式

### 2.1 引入辅助分布 $q$

随便选一个隐变量的分布 $q(z)$，然后做个乘除操作：

$$\log p(\mathbf{x}\mid \boldsymbol{\theta})
= \log \sum_{z} q(z)\, \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}.$$

因为 $\log$ 是凹函数，根据 **Jensen 不等式**，可以得到

$$\boxed{\;
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;\geq\;
\sum_{z} q(z)\, \log \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}
\;\equiv\;
\mathcal{L}(q,\boldsymbol{\theta}).
\;}$$

这里的 $\mathcal{L}$ 就是 **证据下界（ELBO，Evidence Lower Bound）**，它既依赖变分分布 $q$，也依赖参数 $\boldsymbol{\theta}$。

### 2.2 精确分解

直接推导一下，不需要用不等式，就能得出一个**等式**：

$$\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;=\;
\mathcal{L}(q,\boldsymbol{\theta})
\;+\;
\mathrm{KL}\bigl[q(z)\,\Vert\, p(z\mid \mathbf{x},\boldsymbol{\theta})\bigr].$$

从这个等式能直接看出两点：

1. ELBO 永远不会超过 $\log p(\mathbf{x}\mid\boldsymbol{\theta})$，因为 KL 散度 $\mathrm{KL}\geq 0$。
2. 只有当 $q(z) = p(z\mid \mathbf{x}, \boldsymbol{\theta})$（也就是真实后验）时，ELBO 才会等于 $\log p$，达到紧界。

这个等式就是 EM 算法的核心引擎。

---
## 3. EM 是 ELBO 上的坐标上升

![机器学习数学推导（十三）：EM算法与GMM — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/illustration_2.png)

EM 算法通过交替优化两个参数，不断抬高 $\mathcal{L}$。

### 3.1 两步走

**E 步**  
固定 $\boldsymbol{\theta}^{(t)}$，对 $q$ 最大化 $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$。最优解是后验分布：

$$q^{(t)}(z) \;=\; p\bigl(z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}\bigr).$$

执行完 E 步后，下界变紧：$\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) = \log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})$。

**M 步**  
固定 $q^{(t)}$，对 $\boldsymbol{\theta}$ 最大化 $\mathcal{L}(q^{(t)}, \boldsymbol{\theta})$。去掉与 $\boldsymbol{\theta}$ 无关的 $q^{(t)}$ 的熵，等价于最大化 **Q 函数**：

$$Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})
\;=\;
\mathbb{E}_{z\sim q^{(t)}}\!\bigl[\log p(\mathbf{x}, z\mid \boldsymbol{\theta})\bigr].$$

直观来看，Q 函数就是当前后验下完全数据对数似然的期望。

### 3.2 单调上升的证明

将三个不等式串联起来：

$$\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})
\;\overset{(a)}{=}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)})
\;\overset{(b)}{\leq}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)})
\;\overset{(c)}{\leq}\;
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t+1)}).$$

(a) 成立是因为 E 步让下界变紧；(b) 是 M 步的定义；(c) 是因为 ELBO 始终小于等于 $\log p$。因此：

$$\boxed{\;\ell(\boldsymbol{\theta}^{(t+1)}) \;\geq\; \ell(\boldsymbol{\theta}^{(t)})\;}$$

每轮迭代都成立，只有在固定点时取等号。EM 收敛到 $\ell$ 的稳定点——通常是局部最大值，偶尔是鞍点。它无法保证找到全局最大值，所以多次随机重启很重要。

### 3.3 两条曲线的视角

从 ELBO 的角度看，EM 的动力学非常直观。每次 E 步把 KL 差距压到零，M 步同时抬升对数似然和 ELBO，差距重新打开，直到下一次 E 步。

![ELBO 与对数似然的轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig7_elbo_vs_loglik.png)

中间琥珀色带表示 KL 散度 $\mathrm{KL}\bigl[q\,\Vert\, p(z\mid\mathbf{x},\boldsymbol{\theta})\bigr]$。绿色圆点标记 E 步完成后的时刻，此时 KL 散度为零。

---
## 4. EM 算法与高斯混合模型

### 4.1 模型

单个观测值的生成过程如下：

1. 从分类分布中抽取成分标签 $z_i \sim \mathrm{Categorical}(\pi_1,\dots,\pi_K)$。
2. 根据 $z_i = k$，从高斯分布中抽取 $\mathbf{x}_i \mid z_i = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。

模型参数为 $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$，满足 $\sum_k \pi_k = 1$，且每个 $\boldsymbol{\Sigma}_k \succ 0$。

### 4.2 E 步：计算责任度

隐变量是离散的，后验概率直接用贝叶斯公式计算。定义成分 $k$ 对样本 $i$ 的**责任度（responsibility）**：

$$\boxed{\;
\gamma_{ik}
\;=\;
p\bigl(z_i = k \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}\bigr)
\;=\;
\frac{\pi_k\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
\;}$$

每行 $(\gamma_{i1},\dots,\gamma_{iK})$ 的和为 1，表示样本对各成分的软分配。

![E 步：软分配](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig2_e_step.png)

左图中，每个网格点的颜色由三个成分的责任度混合而成：单一成分主导时颜色纯，边界处颜色混合。右图展示了 12 个采样点的责任度矩阵 $\gamma_{ik}$，每行的和为 1。

### 4.3 M 步：加权最大似然估计

将高斯密度代入 $Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})$，并加入拉格朗日乘子约束 $\sum_k \pi_k = 1$，可以得到闭式更新公式。令 $N_k = \sum_{i=1}^{N} \gamma_{ik}$ 表示成分 $k$ 的有效样本数：

$$\boxed{\;
\pi_k = \frac{N_k}{N},\qquad
\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,\mathbf{x}_i,\qquad
\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\!\top}.
\;}$$

这些公式和标准高斯最大似然估计的形式完全一致，只是每个样本按其责任度进行了加权。

![一次 M 步：更新前 vs 更新后](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig3_m_step.png)

即使从一个故意设置得很差的初始值开始，仅一次 M 步就能让均值（红色箭头所示）向数据靠拢，协方差椭圆也调整到匹配观测分布。再迭代几次，拟合结果基本就正确了。

### 4.4 实战中的收敛性

多次随机重启 EM 算法，观察对数似然的变化：

![对数似然单调不减](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig4_loglik_monotone.png)

每条曲线都单调非递减——这是 EM 算法的理论保证。不同重启可能收敛到不同的局部最优；虚线是 `sklearn.mixture.GaussianMixture` 在 `n_init=20` 下找到的最佳值。**实践中建议多次随机或 K-means 初始化，保留效果最好的结果。**

---
## 5. K-means 是 GMM 的硬分配、球形极限

假设所有 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$，然后让 $\epsilon \to 0$。高斯密度会变得无限尖锐，离数据点最近的均值的责任趋于 1，其他均值的责任趋于 0。E 步骤退化为**硬分配**，M 步骤退化为对分配到的点求平均——这正好就是 K-means。

![各向异性数据上 K-means 与 GMM 对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig6_kmeans_vs_gmm.png)

在各向异性数据上的差异非常明显：K-means（左）强制划分出球形 Voronoi 单元，把拉长的簇切得非常生硬；GMM（右）则用椭圆沿主轴方向贴合数据分布。只要簇不是各向同性，或者我需要软分配概率，GMM 就应该是首选方法。
## 6. 如何选择成分数 $K$

似然值会随着 $K$ 增大而单调递增（因为模型更灵活），所以单靠 $\ell$ 无法确定 $K$。我需要用复杂度惩罚准则来选择：

$$\mathrm{BIC}(K) = -2\,\hat{\ell}(K) + p_K\,\log N,
\qquad
\mathrm{AIC}(K) = -2\,\hat{\ell}(K) + 2\,p_K,$$

其中 $p_K$ 是参数总数。对于 $d$ 维全协方差 GMM，参数总数为 $p_K = (K-1) + Kd + K\frac{d(d+1)}{2}$。

![BIC 和 AIC 随成分数 K 的变化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM算法与GMM/fig5_bic_aic.png)

从 $K=1$ 到真实值 $K=3$，两条曲线迅速下降，之后趋于平缓甚至回升。BIC 对复杂度的惩罚更重（多了一个 $\log N$ 因子），因此通常会选择更小的 $K$。不过在这个例子中，两种方法都给出了正确答案。

---
## 7. 参考实现

这里是一个极简的 NumPy 实现，直接对应了前面提到的公式。为了验证结果，本文所有实验和图表都用 `sklearn.mixture.GaussianMixture` 做了对比。

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
## 8. 数值实现的几个坑

上面的推导数学上很优雅，但工程实现中却埋了不少雷。我在生产环境中被以下三种问题反复折磨。

**责任度下溢。** 高维 Gaussian 的对数密度通常是一个很大的负数。当维度 $D \gtrsim 50$ 时，直接对 $\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ 取指数运算，在 float32 精度下结果会变成 0，导致责任度计算出现 $0/0$ 的情况。解决办法是用 log-sum-exp 技巧：

$$\log \gamma_{ik} = \log\pi_k + \log\mathcal{N}_k(\mathbf{x}_i) - \mathrm{logsumexp}_j\big(\log\pi_j + \log\mathcal{N}_j(\mathbf{x}_i)\big),$$

然后通过 $\gamma_{ik} = \exp(\log \gamma_{ik})$ 恢复原始值。在最后一步归一化之前，所有计算都保持在对数域中。

**协方差矩阵退化。** 当某个 component 只捕捉到一个点时，$\boldsymbol{\Sigma}_k$ 会退化为接近零秩的矩阵，行列式趋近于 0，似然值则爆炸到 $+\infty$。这并不是 bug，而是 MLE 的正确解，但实际中毫无意义。我常用的两种应对方法是：（1）加一个小的岭项，令 $\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \lambda \mathbf{I}$，其中 $\lambda \approx 10^{-6} \cdot \mathrm{tr}(\boldsymbol{\Sigma}_k)/D$；（2）如果某个 component 的有效计数 $N_k = \sum_i \gamma_{ik}$ 低于阈值（我一般取 $N_k < 1$），就重新初始化这个 component。

**$\boldsymbol{\Sigma}_k^{-1}$ 的条件数问题。** 计算 Mahalanobis 距离 $(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ 时，千万不要显式求逆。正确的做法是先对 $\boldsymbol{\Sigma}$ 做 Cholesky 分解，得到 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$，然后用前代法解 $\mathbf{L}\mathbf{y} = \mathbf{x} - \boldsymbol{\mu}$。距离平方就是 $\Vert \mathbf{y}\Vert^2$，而 log-determinant 则是 $2\sum_d \log L_{dd}$。这种方法的复杂度从 $O(D^3)$ 降到 $O(D^2)$，并且在 $\boldsymbol{\Sigma}$ 条件数较大时数值稳定性更好。

迭代过程中，给自己设一个监控指标：ELBO 必须单调不降。如果发现 ELBO 下降幅度超过 $10^{-6}$，先检查数值实现，别怀疑数学推导。
## 9. scikit-learn 中的实现

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

有两个参数特别重要。`covariance_type='diag'` 会去掉 $\boldsymbol{\Sigma}_k$ 的非对角元素，参数量从 $K D(D+1)/2$ 减少到 $KD$，每次迭代的计算复杂度也从 $O(NKD^2)$ 降到 $O(NKD)$。对于 $D=128$ 的嵌入向量来说，这能将运行时间从小时级缩短到分钟级。`init_params='k-means++'` 会先跑一遍 K-means 来初始化均值，否则在分量数超过 4 时，EM 算法很容易收敛到一个退化解，比如某个分量占据所有数据。

BIC 分数公式为 $\mathrm{BIC} = -2\log\hat L + p\log N$（其中 $p$ 是参数数量），这是选择 $K$ 值最简单且有理论依据的方法。遍历 $K \in \{1, \dots, K_{\max}\}$，找到拐点即可。单次拟合的结果不可靠，务必设置 `n_init >= 5`，因为 EM 算法只能找到局部最优解。

---
## Q&A

### EM 能找到全局最优解吗？

不能。单调递增只能保证收敛到一个稳定点，通常是局部最大值，有时是鞍点。为了应对这个问题，我会用多次重启的方法（随机初始化或用 K-means 初始化），然后保留最终对数似然值最高的结果。

### GMM 和 K-means 的区别在哪？什么时候需要特别注意？

以下情况我会选择 GMM：  
1. 簇的形状明显是椭圆形或者各向异性；  
2. 下游任务需要软隶属概率来进行校准；  
3. 需要生成式密度模型来采样或计算异常分数。  

如果簇大致是球形且分得比较开，K-means 更快，也完全够用。

### 协方差矩阵奇异或者成分坍塌怎么办？

可以加一个岭正则项 $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$（参考实现中就是这样做的）。也可以让不同成分共享协方差矩阵，或者限制为对角协方差。如果检测到成分坍塌，直接重启也是一种办法。

### 为什么 E 步能让下界变紧？

因为 $\log p = \mathcal{L}(q,\boldsymbol{\theta}) + \mathrm{KL}[q\Vert p(z\mid\mathbf{x},\boldsymbol{\theta})]$，而 KL 散度恰好在 $q$ 等于真实后验时为零。

### 广义 EM 是什么？

广义 EM 把 M 步的“完全最大化”放宽为“任何能让 $Q$ 增加的更新”，比如做一次梯度上升。即使这样，单调递增的性质依然成立。

### EM 和变分推断有什么关系？

变分 EM 把 E 步的真实后验放宽到一组可处理的分布族 $q \in \mathcal{Q}$。分解公式 $\log p = \mathcal{L} + \mathrm{KL}$ 不变，算法现在交替进行两步：在 $\mathcal{Q}$ 内最小化 KL（E 步），以及对 $\boldsymbol{\theta}$ 最大化 $\mathcal{L}$（M 步）。详见第 14 篇。

### EM 还能解决哪些问题？

EM 是隐马尔可夫模型训练（Baum-Welch 算法）、缺失数据插补、混合专家模型、主题模型 LDA 的变分推断等问题的首选框架。只要问题中有隐变量，并且完全数据下的极大似然估计容易求解，EM 就能派上用场。
## 练习题与解答

**E1（E 步计算）.** 考虑一个一维 GMM，$K=2$，先验相等 $\pi_1 = \pi_2 = 1/2$，$\mu_1 = 0,\, \mu_2 = 3,\, \sigma^2 = 1$。求 $x_i = 1.5$ 时的 $\gamma_{i1}$。

*解.* 因为对称性，两个分量密度在 $x=1.5$ 处相等，所以 $\gamma_{i1} = 1/2$。

**E2（M 步更新）.** 有两个样本 $x_1 = 1,\; x_2 = 4$，对应的责任值是 $\gamma_{11} = 0.8,\; \gamma_{21} = 0.3$。求 M 步更新后的 $\mu_1$。

*解.* $N_1 = 0.8 + 0.3 = 1.1$，于是 $\mu_1 = (0.8 \cdot 1 + 0.3 \cdot 4) / 1.1 = 2.0 / 1.1 \approx 1.82$。

**E3（单调性分析）.** 在链式关系 $\log p(\boldsymbol{\theta}^{(t)}) = \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) \leq \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)}) \leq \log p(\boldsymbol{\theta}^{(t+1)})$ 中，M 步的最优性体现在哪一步？ELBO 不等式又体现在哪一步？

*解.* 中间的 $\leq$ 是 M 步定义 $\boldsymbol{\theta}^{(t+1)}$ 的结果；右边的 $\leq$ 是 ELBO 不等式在新参数下的应用。

**E4（奇异极限退化为 K-means）.** 证明：如果固定所有 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$ 并令 $\epsilon \to 0^+$，EM 更新均值的过程会退化为 K-means 的均值更新。

*提示.* 写出 $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \epsilon\mathbf{I}) \propto \exp(-\Vert \mathbf{x} - \boldsymbol{\mu}_k\Vert^2 / (2\epsilon))$。当 $\epsilon \to 0$ 时，对 $-\Vert\mathbf{x}-\boldsymbol{\mu}_k\Vert^2$ 的 softmax 操作退化为硬 argmin，$\gamma_{ik}\in\{0,1\}$，M 步均值更新变为 K-means 的簇均值更新。

**E5（缺失数据处理）.** 假设样本 $\mathbf{x}_i$ 的第 $j$ 维缺失，设计一个 EM 方法，将缺失值视为额外的隐变量，并联合更新参数和缺失值。

*提示.* 把缺失值 $x_{ij}$ 加入隐变量 $z$。E 步计算 $\mathbb{E}[x_{ij} \mid \text{已观测}, \boldsymbol{\theta}^{(t)}]$（包括二阶矩）。M 步用这些期望值作为插值，代入加权 MLE 更新公式中。

---
## 参考文献

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. **Chapter 9**.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. **Chapter 11**.
- McLachlan, G., & Krishnan, T. (2007). *The EM Algorithm and Extensions* (2nd ed.). Wiley.
- Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 89, 355-368.
