---
title: "机器学习数学推导（十三）：EM算法与GMM"
date: 2024-03-13 09:00:00
tags:
  - 机器学习
  - 期望最大化
  - 高斯混合模型
  - 隐变量
  - EM算法
  - GMM
  - 数学推导
categories: 机器学习
series:
  name: "机器学习数学推导"
  part: 13
  total: 20
lang: zh-CN
mathjax: true
description: "从 Jensen 不等式与 ELBO 出发推导 EM 算法，证明其单调上升性，并完整给出高斯混合模型（GMM）的 E 步、M 步更新公式、模型选择以及与 K-means 的关系。"
---

数据里常常藏着看不见的结构——你不知道每个样本属于哪个簇、不知道某个特征的真实取值、不知道一段文本背后是哪些主题在驱动。这些**隐变量**让最大似然估计变得棘手：似然函数变成"对数里套求和"的形式，既无闭式解，梯度法也容易绕进死胡同。**EM 算法**用一招看似朴素的拆解破解了这种困境：在隐变量的后验下"猜"一次（E 步），再把参数当成完全数据来"拟合"一次（M 步），交替进行。每一轮迭代都被数学保证不会让似然下降。本文从第一性原理推导 EM，借 Jensen 不等式证明其单调上升性，并把它落到最经典的应用——**高斯混合模型（GMM）**——上：K-means 的软化、椭球化版本。

## 你将学到

- 隐变量为什么让 MLE 变难（"log 套 sum"问题）
- 怎样用 Jensen 不等式构造**证据下界（ELBO）**
- EM 算法等价于在 $(q, \boldsymbol{\theta})$ 上对 ELBO 做**坐标上升**
- 严谨证明 $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$
- GMM 的完整 E 步 / M 步更新公式（任意协方差）
- K-means 是 GMM 的硬分配、球形协方差极限
- 用 **BIC / AIC** 选择成分数 $K$

## 先修

- 极大似然估计
- 多元高斯密度
- Jensen 不等式与 KL 散度
- K-means 聚类

---

## 1. 隐变量与不完全数据似然

### 1.1 设定

观测 $\mathbf{x}_1,\dots,\mathbf{x}_N$ 与隐变量 $z_1,\dots,z_N$ 由联合分布 $p(\mathbf{x}, z \mid \boldsymbol{\theta})$ 生成。我们只能看到 $\mathbf{X}$，看不到 $\mathbf{Z}$。**不完全数据对数似然**为

$$
\ell(\boldsymbol{\theta}) \;=\; \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \boldsymbol{\theta})
\;=\; \sum_{i=1}^{N} \log \sum_{z} p(\mathbf{x}_i, z \mid \boldsymbol{\theta}).
$$

**问题就出在 log 里面那个求和**：log 不再分配到各成分上，梯度无法分项写出，闭式最大化也走不通。

### 1.2 混合模型的例子

对于含 $K$ 个成分的高斯混合，

$$
p(\mathbf{x}\mid \boldsymbol{\theta}) \;=\; \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).
$$

如果**已知**每个点来自哪个成分，问题立刻退化成 $K$ 个独立的加权高斯 MLE——平凡。我们恰恰不知道，而 EM 正是在补上这一刀。

![GMM 三个高斯成分及其协方差椭圆](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig1_gmm_clusters.png)

上图是 `sklearn.mixture.GaussianMixture` 在二维数据上拟合出的三个高斯成分：每个叉号是均值 $\boldsymbol{\mu}_k$，内椭圆是 1 倍标准差等高线，外椭圆是 2 倍标准差，$\pi_k$ 是各自的混合权重。

---

## 2. ELBO 与 Jensen 不等式

### 2.1 引入辅助分布 $q$

任取一个隐变量上的分布 $q(z)$，乘除一下：

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
= \log \sum_{z} q(z)\, \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}.
$$

由于 $\log$ 是凹函数，**Jensen 不等式**给出

$$
\boxed{\;
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;\geq\;
\sum_{z} q(z)\, \log \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}
\;\equiv\;
\mathcal{L}(q,\boldsymbol{\theta}).
\;}
$$

这个 $\mathcal{L}$ 就是 **证据下界（ELBO，Evidence Lower Bound）**，它同时依赖 $q$ 和 $\boldsymbol{\theta}$。

### 2.2 精确分解

直接代换、不需要不等式，可以得到一个**等式**：

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;=\;
\mathcal{L}(q,\boldsymbol{\theta})
\;+\;
\mathrm{KL}\bigl[q(z)\,\Vert\, p(z\mid \mathbf{x},\boldsymbol{\theta})\bigr].
$$

由此立得两条结论：

1. ELBO **永远** $\leq \log p(\mathbf{x}\mid\boldsymbol{\theta})$，因为 $\mathrm{KL}\geq 0$；
2. **当且仅当** $q(z) = p(z\mid \mathbf{x}, \boldsymbol{\theta})$（即真实后验）时，下界**取紧**。

这一等式就是整个 EM 的发动机。

---

## 3. EM 是 ELBO 上的坐标上升

EM 反复抬高 $\mathcal{L}$，每次只动其中一个参数。

### 3.1 两步走

**E 步.** 固定 $\boldsymbol{\theta}^{(t)}$，对 $q$ 最大化 $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$。最优解就是后验：

$$
q^{(t)}(z) \;=\; p\bigl(z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}\bigr).
$$

执行完 E 步，下界**变紧**：$\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) = \log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})$。

**M 步.** 固定 $q^{(t)}$，对 $\boldsymbol{\theta}$ 最大化 $\mathcal{L}(q^{(t)}, \boldsymbol{\theta})$。把与 $\boldsymbol{\theta}$ 无关的 $q^{(t)}$ 的熵丢掉，等价于最大化 **Q 函数**：

$$
Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})
\;=\;
\mathbb{E}_{z\sim q^{(t)}}\!\bigl[\log p(\mathbf{x}, z\mid \boldsymbol{\theta})\bigr].
$$

直观理解：Q 函数就是"在当前后验下完全数据对数似然的期望"。

### 3.2 单调上升的证明

把三步串起来：

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})
\;\overset{(a)}{=}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)})
\;\overset{(b)}{\leq}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)})
\;\overset{(c)}{\leq}\;
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t+1)}).
$$

(a) 由 E 步使下界取紧得到；(b) 由 M 步的定义；(c) 是 ELBO 不等式。从而

$$
\boxed{\;\ell(\boldsymbol{\theta}^{(t+1)}) \;\geq\; \ell(\boldsymbol{\theta}^{(t)})\;}
$$

每轮都成立。EM 收敛到 $\ell$ 的稳定点——通常是局部极大，偶尔是鞍点。**它不保证全局最优**，所以多次随机重启很重要。

### 3.3 两条曲线的视角

把 ELBO 与 $\log p$ 同时画出来，动力学就一目了然：每次 E 步把 KL 间隙压到零，M 步同时抬升两条曲线，间隙再度打开。

![ELBO 与对数似然的轨迹](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig7_elbo_vs_loglik.png)

中间填色的琥珀色带就是 KL 散度 $\mathrm{KL}\bigl[q\,\Vert\, p(z\mid\mathbf{x},\boldsymbol{\theta})\bigr]$。绿色圆点标出 E 步执行后那一刻——按构造此时 KL 等于零。

---

## 4. EM for 高斯混合模型

### 4.1 模型

单个观测的生成过程：

1. 抽成分标号 $z_i \sim \mathrm{Categorical}(\pi_1,\dots,\pi_K)$；
2. 抽 $\mathbf{x}_i \mid z_i = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。

参数 $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$，约束 $\sum_k \pi_k = 1$、各 $\boldsymbol{\Sigma}_k \succ 0$。

### 4.2 E 步：责任度

隐变量是离散的，后验直接用贝叶斯公式。定义成分 $k$ 对样本 $i$ 的**责任度（responsibility）**：

$$
\boxed{\;
\gamma_{ik}
\;=\;
p\bigl(z_i = k \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}\bigr)
\;=\;
\frac{\pi_k\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
\;}
$$

每行 $(\gamma_{i1},\dots,\gamma_{iK})$ 之和为 1，对应该样本"软"的成分隶属概率。

![E 步：软分配](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig2_e_step.png)

左图把每个网格点按三个成分的责任度做颜色混合：单一成分主导处颜色纯，边界处颜色互渗。右图是 12 个采样点的责任度矩阵 $\gamma_{ik}$，每行加起来为 1。

### 4.3 M 步：加权最大似然

把高斯密度代入 $Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})$，对 $\pi_k$ 用拉格朗日乘子处理 $\sum_k \pi_k = 1$，可得闭式更新。记 $N_k = \sum_{i=1}^{N} \gamma_{ik}$ 为成分 $k$ 的**有效样本数**：

$$
\boxed{\;
\pi_k = \frac{N_k}{N},\qquad
\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,\mathbf{x}_i,\qquad
\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\!\top}.
\;}
$$

形式上和单个高斯的 MLE 公式一模一样，只是把每个样本按责任度做了加权。

![一次 M 步：更新前 vs 更新后](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig3_m_step.png)

刻意从糟糕的初值出发，仅一次 M 步，红色箭头标出三个均值被"拉"向真实数据的位置，协方差椭圆也撑开匹配观测散布。再迭代几轮，整个拟合基本就到位了。

### 4.4 实战中的收敛

跑几次随机重启，把对数似然画出来：

![对数似然单调不减](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig4_loglik_monotone.png)

每条曲线都是非递减的——这就是 EM 的算法保证。不同的重启会停在不同的盆地里；虚线是 `sklearn.mixture.GaussianMixture` 用 `n_init=20` 拿到的最好值。**实践中务必多次随机或 K-means 初始化，留下最好的那次。**

---

## 5. K-means：GMM 的硬分配、球形极限

令所有 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$，再让 $\epsilon \to 0$。高斯密度变得无限尖锐，最近的均值得到的责任度趋于 1，其它趋于 0。E 步退化为**硬分配**，M 步退化为对所属点取平均——这恰好就是 K-means。

![各向异性数据上 K-means 与 GMM 对比](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig6_kmeans_vs_gmm.png)

在各向异性数据上对比格外清楚：K-means（左）强行划出球形 Voronoi 单元，把拉长的簇切得很别扭；GMM（右）用沿主轴方向的椭圆贴合数据。**只要簇不是各向同性，或者你需要软分配概率，GMM 都应该是默认选择。**

---

## 6. 怎么选成分数 $K$

似然总是随 $K$ 单调增大（自由度更高），所以单看 $\ell$ 没法选 $K$。要用复杂度惩罚准则：

$$
\mathrm{BIC}(K) = -2\,\hat{\ell}(K) + p_K\,\log N,
\qquad
\mathrm{AIC}(K) = -2\,\hat{\ell}(K) + 2\,p_K,
$$

其中 $p_K$ 是参数总数：$d$ 维全协方差 GMM 有 $p_K = (K-1) + Kd + K\frac{d(d+1)}{2}$ 个自由参数。

![BIC/AIC 随 K 变化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/ml-math-derivations/13-EM%E7%AE%97%E6%B3%95%E4%B8%8EGMM/fig5_bic_aic.png)

两条曲线从 $K=1$ 到真实 $K=3$ 急剧下降，之后趋平或回升。BIC 的复杂度惩罚比 AIC 重（多了 $\log N$ 因子），通常更倾向小 $K$。这里两者都给出了正确答案。

---

## 7. 参考实现

下面是直接对照公式写的极简 NumPy 实现。本文所有实验和图都用 `sklearn.mixture.GaussianMixture` 做了交叉验证。

```python
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    """全协方差高斯混合模型，EM 训练。"""

    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg=1e-6):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    # ----- E 步：责任度 -----
    def _e_step(self, X):
        comp = np.column_stack([
            self.weights[k] * multivariate_normal.pdf(
                X, self.means[k], self.covs[k])
            for k in range(self.K)
        ])
        ll = np.log(comp.sum(axis=1) + 1e-300).sum()
        return comp / (comp.sum(axis=1, keepdims=True) + 1e-300), ll

    # ----- M 步：加权 MLE -----
    def _m_step(self, X, gamma):
        N_k = gamma.sum(axis=0)
        self.weights = N_k / X.shape[0]
        self.means = (gamma.T @ X) / N_k[:, None]
        d = X.shape[1]
        for k in range(self.K):
            diff = X - self.means[k]
            self.covs[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k]
            self.covs[k] += self.reg * np.eye(d)  # 正则化避免奇异

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

---

## Q&A

**Q1：EM 会收敛到全局最优吗？** 不会。单调上升只保证收敛到稳定点，通常是局部极大，偶尔是鞍点。实战中务必多次重启（随机或 K-means 初始化），留下最终对数似然最高的那次。

**Q2：GMM 与 K-means 何时差距明显？** 出现以下任何一种情况都该用 GMM：（i）簇形状明显椭圆/各向异性；（ii）下游需要**软**隶属概率做校准；（iii）需要生成式密度模型用于采样或异常打分。如果簇大致是球形、分得开，K-means 更快也够用。

**Q3：协方差奇异 / 成分坍塌怎么办？** 加岭正则 $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$（参考实现就这么做的）；或者跨成分共享协方差；或限制为对角；又或者检测到坍塌就重启。

**Q4：为什么 E 步能让"下界取紧"？** 因为 $\log p = \mathcal{L}(q,\boldsymbol{\theta}) + \mathrm{KL}[q\Vert p(z\mid\mathbf{x},\boldsymbol{\theta})]$，而 KL 恰好在 $q$ 等于真实后验时为零。

**Q5：广义 EM 是什么？** 把 M 步从"完全最大化"放宽为"任何能让 $Q$ 上升的更新"（比如做几步梯度上升）。前面那条单调上升论证依然成立。

**Q6：EM 与变分推断的关系？** 变分 EM 把 E 步的真实后验放宽到一族可处理分布 $\mathcal{Q}$。$\log p = \mathcal{L} + \mathrm{KL}$ 的分解不变，算法变成在 $\mathcal{Q}$ 内最小化 KL（E 步）、再对 $\boldsymbol{\theta}$ 最大化 $\mathcal{L}$（M 步）。详见第 14 篇。

**Q7：EM 还能解决什么问题？** 隐马尔可夫模型的训练（Baum-Welch 算法）、缺失数据插补、混合专家模型、主题模型 LDA 的变分 EM——只要存在隐变量、且完全数据下的 MLE 容易做，EM 都是首选框架。

---

## 练习题与解答

**E1（E 步计算）.** 一维 GMM，$K=2$，等先验 $\pi_1 = \pi_2 = 1/2$，$\mu_1 = 0,\, \mu_2 = 3,\, \sigma^2 = 1$。计算 $x_i = 1.5$ 处的 $\gamma_{i1}$。

*解.* 由对称性，两成分密度在 $x=1.5$ 处相等，故 $\gamma_{i1} = 1/2$。

**E2（M 步更新）.** 两个样本 $x_1 = 1,\; x_2 = 4$，责任度 $\gamma_{11} = 0.8,\; \gamma_{21} = 0.3$。求 M 步后的 $\mu_1$。

*解.* $N_1 = 0.8 + 0.3 = 1.1$，$\mu_1 = (0.8 \times 1 + 0.3 \times 4) / 1.1 = 2.0 / 1.1 \approx 1.82$。

**E3（单调上升的证明溯源）.** 在链 $\log p(\boldsymbol{\theta}^{(t)}) = \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) \leq \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)}) \leq \log p(\boldsymbol{\theta}^{(t+1)})$ 中，M 步的最优性出现在哪一步？ELBO 不等式又出现在哪一步？

*解.* 中间的 $\leq$ 来自 M 步对 $\boldsymbol{\theta}^{(t+1)}$ 的最优性定义；右边的 $\leq$ 是 ELBO 不等式，套到新参数上。

**E4（球形极限退化为 K-means）.** 证明：若固定所有 $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$ 并令 $\epsilon \to 0^+$，EM 的均值更新会退化为 K-means 的均值更新。

*提示.* $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \epsilon\mathbf{I}) \propto \exp(-\Vert \mathbf{x} - \boldsymbol{\mu}_k\Vert^2 / (2\epsilon))$。当 $\epsilon \to 0$，对 $-\Vert\mathbf{x}-\boldsymbol{\mu}_k\Vert^2$ 的 softmax 退化为 hard argmin，$\gamma_{ik}\in\{0,1\}$，M 步均值即为簇均值。

**E5（缺失数据）.** 若样本 $\mathbf{x}_i$ 的第 $j$ 维缺失，怎样把缺失值并入隐变量、用 EM 同时估计参数与缺失值？

*提示.* 把缺失项 $x_{ij}$ 加进 $z$。E 步计算 $\mathbb{E}[x_{ij} \mid \text{已观测}, \boldsymbol{\theta}^{(t)}]$（连同二阶矩）；M 步把这些期望当作"插值"代入加权 MLE 更新。

---

## 参考文献

- **Dempster, A. P., Laird, N. M., & Rubin, D. B.** (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.
- **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. **第 9 章**。
- **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. **第 11 章**。
- **McLachlan, G., & Krishnan, T.** (2007). *The EM Algorithm and Extensions* (2nd ed.). Wiley.
- **Neal, R. M., & Hinton, G. E.** (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 89, 355-368.

---

## 系列导航

- 上一篇：[第十二篇 -- XGBoost与LightGBM](/zh/机器学习数学推导-十二-XGBoost与LightGBM/)
- 下一篇：[第十四篇 -- 变分推断与变分EM](/zh/机器学习数学推导-十四-变分推断与变分EM/)
- [查看本系列全部20篇文章](/tags/机器学习/)
