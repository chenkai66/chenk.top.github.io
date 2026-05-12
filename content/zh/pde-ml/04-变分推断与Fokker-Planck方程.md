---
title: "偏微分方程与机器学习（四）：变分推断与Fokker-Planck方程"
date: 2024-06-15 09:00:00
tags:
  - PDE
  - Variational Inference
  - Fokker-Planck
  - ELBO
  - Langevin Dynamics
categories: PDE与机器学习
series: pde-ml
lang: zh
mathjax: true
description: "变分推断与 Langevin MCMC 在连续时间下是同一个 Fokker-Planck PDE：从 SDE 推导密度演化、KL 散度作为 Wasserstein 梯度流、SVGD 粒子方法、对数 Sobolev 不等式给出指数收敛、贝叶斯神经网络应用。"
disableNunjucks: true
series_order: 4
translationKey: "pde-ml-4"
---
![偏微分方程与机器学习（四）：变分推断与Fokker-Planck方程 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-Variational-Inference/illustration_1.png)

## 本文的七个维度

1. **动机**： VI 和 MCMC 尽管看似不同，实际上都在解同一个 PDE。
2. **理论**：从 SDE 推导出 Fokker-Planck 方程
3. **几何**： KL 散度是 Wasserstein 空间的梯度流
4. **算法**： Langevin Monte Carlo、平均场 VI 和 SVGD
5. **收敛**：对数 Sobolev 不等式保证指数 KL 收敛
6. **数值实验**： 7 张图，附完整代码脚本
7. **应用**：用 Langevin 采样近似贝叶斯神经网络的后验
## 你将学到

- 任意 It&ocirc; SDE 的概率密度满足 Fokker-Planck 方程
- Langevin 动力学是一种实用的采样算法，但存在离散化误差
- 在 Wasserstein 空间最小化 $\mathrm{KL}(q\|p^\star)$ 就是 Fokker-Planck PDE。
- 在连续时间下，变分推断与 Langevin MCMC 完全等价
- Stein 变分梯度下降（SVGD）：用确定性粒子连接两种方法
- 贝叶斯神经网络的后验推断实践
## 前置知识

- 概率论基础：贝叶斯定理、KL 散度和期望
- 第 3 篇提到的 Wasserstein 梯度流
- 随机分析基本概念：布朗运动和 It&ocirc; 积分
- 实验需要 Python 和 PyTorch
## 1. 推断问题

贝叶斯推断需要计算后验分布。

$$p(\theta \mid x) \;=\; \frac{p(x \mid \theta)\,p(\theta)}{\int p(x \mid \theta')\,p(\theta')\,d\theta'},$$

但分母（边际似然）通常无法直接求解，因此提出了两类近似方法：

- **变分推断（VI）**：选择可处理的分布族 $\{q_\phi\}$，优化目标为最小化

  $$\mathrm{KL}(q_\phi \,\|\, p(\cdot \mid x)) = \mathbb{E}_{q_\phi}\!\left[\log \tfrac{q_\phi(\theta)}{p(\theta \mid x)}\right],$$
  等价于最大化 **证据下界（ELBO）** $\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(x\mid\theta)] - \mathrm{KL}(q_\phi \| p(\theta))$。

- **马尔可夫链蒙特卡洛（MCMC）**：构造平稳分布为 $p(\cdot \mid x)$ 的马氏链。**Langevin 动力学**是基于梯度的典型实例。

两者看似不同：VI 是有限维参数优化，MCMC 是无限时间随机过程；但从 PDE 视角来看，它们只是同一种概率测度演化的不同采样方式。
## 2. 从 SDE 到 Fokker-Planck

考虑 It&ocirc; SDE
$$dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t.$$
任取光滑测试函数 $f$，用 It&ocirc; 引理求 $\mathbb{E} f(X_t)$ 的导数，再分部积分（假设无穷远处 $p$ 及其导数为零），得到 **Fokker-Planck 方程**：
$$
\boxed{\;\partial_t p \;=\; -\nabla\!\cdot\!(\mu\, p) \;+\; \tfrac{1}{2}\,\nabla\!\cdot\!\nabla\!\cdot\!(D\, p),\quad D = \sigma\sigma^\top.\;}
$$
第一项是漂移（输运），第二项是扩散（弥散）。设 $\mu = -\nabla V$、$\sigma = \sqrt{2\tau} I$，对应 **过阻尼 Langevin SDE**：
$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p \nabla V\bigr) + \tau\,\Delta p,$$
稳态解唯一（弱条件），即 **Gibbs 分布** $p_\infty \propto e^{-V/\tau}$。取 $V = -\log p^\star$、$\tau = 1$，稳态分布就是目标 $p^\star$。

![双势阱中 Fokker-Planck 方程的密度演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig1_fokker_planck_evolution.png)
*图 1. 用有限差分求解 1D Fokker-Planck 方程。初始为左阱中的窄高斯，密度逐步扩散、跨越势垒，最终收敛到对称的 Gibbs 平稳分布 $p_\infty \propto e^{-V/D}$（右图）。*
## 3. Langevin 动力学：采样看作 PDE

**过阻尼 Langevin 方程** 用于从 $p^\star \propto e^{-V}$ 采样：
$$dX_t = -\nabla V(X_t)\,dt + \sqrt{2\tau}\,dW_t.$$
离散化后得到 **未校正 Langevin 算法（ULA）**，即 Euler-Maruyama 格式：
$$X_{k+1} = X_k - \eta\,\nabla V(X_k) + \sqrt{2\eta\tau}\,\xi_k,\qquad \xi_k \sim \mathcal{N}(0, I).$$
```python
import torch, numpy as np

def langevin_sample(grad_log_p, x0, step=0.01, n_steps=10_000, tau=1.0):
    """过阻尼 Langevin 采样器（即 ULA）。

    grad_log_p : 返回 grad(log p*(x)) 的函数
    x0         : (n_particles, dim) 初始位置
    """
    x = x0.clone()
    for _ in range(n_steps):
        x = x + step * grad_log_p(x) + np.sqrt(2 * step * tau) * torch.randn_like(x)
    return x
```

ULA 的偏差是 $O(\eta)$。**MALA**（Metropolis 校正版 Langevin）通过 accept-reject 步骤实现无偏。**HMC**（哈密顿蒙特卡洛）是带动量的欠阻尼版本。

![Langevin SDE 轨迹与经验密度。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig2_langevin_sde_to_density.png)
*图 2. 左：双势阱中 25 条粒子轨迹，部分粒子有限时间内不跨越势垒。右： 400 个粒子直方图随时间收敛到 Gibbs 目标 -- 这对应图 1 的 PDE 随机实现。*
## 4. KL 散度是 Wasserstein 梯度流

将 KL 散度对 $p^\star \propto e^{-V}$ 分解：
$$\mathcal{F}[p] = \mathrm{KL}(p\,\|\,p^\star) = \underbrace{\int p\log p\,dx}_{\text{负熵 }\mathcal{H}[p]} + \underbrace{\int p\, V\,dx}_{\text{势能}} + \text{常数}.$$
这是第 3 篇提到的 **自由能泛函**。 Jordan-Kinderlehrer-Otto （JKO， 1998）定理指出，其 **2-Wasserstein 梯度流** 为
$$\partial_t p = \nabla\!\cdot\!\bigl(p\nabla V\bigr) + \Delta p,$$
正是 $\tau = 1$ 时 Langevin 的 Fokker-Planck 方程。

> **等价性**。在 Wasserstein 空间最小化 $\mathrm{KL}(\cdot \| p^\star)$ 和运行目标为 $p^\star$ 的 Langevin 动力学，本质是同一个 PDE。 VI 和 Langevin MCMC 是同一连续时间梯度流的两种离散化方法。

| 视角 | 变分推断 | Langevin MCMC |
|---|---|---|
| 目标 | 最小化 $\mathrm{KL}(q_\phi \| p^\star)$ | 从 $p^\star$ 采样 |
| 状态 | 参数 $\phi$ | 粒子 $\{X^{(i)}\}$ |
| 步骤 | ELBO 上做梯度步 | SDE 的 Euler-Maruyama |
| 连续极限 | KL 的 Wasserstein 梯度流 | Fokker-Planck 方程 |
| 稳态 | 若分布族足够表达，$q^\star = p^\star$ | $p_\infty = p^\star$ |
| 偏差 | 受限分布族 + Adam 噪声 | 离散化 $O(\eta)$ |

![KL 散度作为 Wasserstein 梯度流。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig3_kl_gradient_flow.png)
*图 3. 两个初值密度（集中型与发散型）在 FP 方程驱动下收敛到双峰目标。右图显示 KL 散度单调递减 -- 梯度流保证的实际效果。*
## 5. VI 与 MCMC 实际差异

连续极限下相同，但有限时间行为差别很大。

- **VI （最小化反向 KL）是模式寻求型**：$q$ 限制在简单分布族时，最优解塌缩到单一模式，低估不确定性。
- **MCMC 是质量覆盖型**：链足够长时按比例访问每个模式，但跨越势垒可能指数慢。

![VI 与 MCMC 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig4_vi_vs_mcmc.png)
*图 4. 左：双峰后验下，最优平均场高斯（反向 KL 意义）只能覆盖一个峰。右： 4000 个 Langevin 样本同时覆盖两个峰 -- 前提是 1D 例子中势垒较低。*
## 6. Stein 变分梯度下降

SVGD （Liu & Wang, 2016）是介于 VI 和 MCMC 的确定性粒子方法。维护粒子 $\{x_i\}_{i=1}^n$，更新公式如下：
$$x_i \leftarrow x_i + \eta\,\hat\phi^*(x_i),\quad \hat\phi^*(x) = \tfrac{1}{n}\sum_{j=1}^n \Bigl[\,k(x_j,x)\,\nabla_{x_j}\log p^\star(x_j) + \nabla_{x_j} k(x_j,x)\Bigr],$$
核函数 $k(x,y) = \exp(-\|x-y\|^2 / 2h^2)$ 是 RBF 核，带宽 $h$ 用中位数启发式确定。两项作用相反：

- **漂移项** $k\,\nabla\log p^\star$：推动粒子向高概率区域移动。
- **排斥项** $\nabla k$：推开粒子，避免塌缩到单一模式。

```python
import numpy as np
from scipy.spatial.distance import cdist

def svgd_step(x, score, eta=0.05):
    n = x.shape[0]
    sq = cdist(x, x) ** 2
    h  = np.sqrt(0.5 * np.median(sq) / np.log(n + 1)) + 1e-6
    K  = np.exp(-sq / (2 * h**2))
    grad_K = -(x[:, None] - x[None, :]) / h**2 * K[..., None]
    phi = (K @ score - grad_K.sum(axis=0)) / n
    return x + eta * phi
```

无穷粒子极限下， SVGD 满足 PDE：
$$\partial_t p = -\nabla\!\cdot\!\bigl(p\, v[p]\bigr),\quad v[p](x) = \mathbb{E}_{y\sim p}\bigl[k(y,x)\nabla\log p^\star(y) + \nabla_y k(y,x)\bigr],$$
当带宽 $h \to 0$，退化为标准 Fokker-Planck 方程。 SVGD 是核平滑的 FP 求解器。

![SVGD 在双峰目标上的粒子演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig5_svgd_particles.png)
*图 5. 左： 80 个粒子从原点出发，几百步内分裂覆盖两个峰。右：完整轨迹显示核排斥防止塌缩，漂移项稳定粒子在 $\pm 2$ 附近。*
## 7. 收敛理论

**定义（LSI）。** 若 $p^\star$ 满足常数 $\lambda > 0$ 的 **对数 Sobolev 不等式**，则对所有 $p \ll p^\star$：
$$\mathrm{KL}(p \,\|\, p^\star) \leq \frac{1}{2\lambda}\, I(p \,\|\, p^\star),\quad I(p\|p^\star) = \int p\,\bigl\|\nabla\log\tfrac{p}{p^\star}\bigr\|^2 dx.$$
右侧是 **Fisher 信息**，也是 FP 方程中 KL 的耗散率：
$$\frac{d}{dt}\mathrm{KL}(p_t\,\|\,p^\star) = -\,I(p_t\,\|\,p^\star).$$
结合 LSI，$\frac{d}{dt}\mathrm{KL} \leq -2\lambda\,\mathrm{KL}$。由 Gr&ouml;nwall 不等式得：
$$
\boxed{\;\mathrm{KL}(p_t\,\|\,p^\star) \leq e^{-2\lambda t}\,\mathrm{KL}(p_0\,\|\,p^\star).\;}
$$
强对数凹目标（$\nabla^2 V \succeq mI$）自动满足 LSI，且 $\lambda \geq m$ [Bakry-&Eacute;mery]。多峰目标的 $\lambda$ 很小，解释了实践中指数级慢混合现象。

![收敛速率分析。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig6_convergence_analysis.png)
*图 6. 左：不同 $\lambda$ 下理论 KL 衰减曲线 $e^{-2\lambda t}$。右： VI、 Langevin MCMC、 SVGD 在光滑高斯目标上的实测 KL 轨迹 -- 三者收敛速率与噪声特性各异。*
## 8. 应用：贝叶斯神经网络

贝叶斯神经网络对权重加先验 $p(w)$，求后验 $p(w \mid \mathcal{D}) \propto p(\mathcal{D}\mid w)\,p(w)$。小网络的后验也难解析，但 Langevin 动力学只需
$$\nabla_w \log p(w \mid \mathcal{D}) = \nabla_w \log p(\mathcal{D}\mid w) + \nabla_w \log p(w),$$
即反向传播算的梯度，再加高斯先验项。**SGLD**（Welling & Teh, 2011）用 mini-batch 梯度代替全批梯度，适配现代规模。

下图用 24 个随机傅里叶特征构建 "贝叶斯 NN"，使权重后验明确，并用全批 Langevin 采样。

![贝叶斯神经网络后验带。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig7_bayesian_nn.png)
*图 7. 左：训练数据有缺口的回归问题， 90% Langevin 后验带在数据缺失处加宽。右：预测标准差在缺口处峰值 -- 这是点估计网络缺失的 **认知不确定性**。*
## 总结

- 任意 It&ocirc; SDE 都对应一个 Fokker-Planck PDE，描述其密度演化。
- Langevin 动力学采样自 $p^\star \propto e^{-V}$； ULA/MALA/HMC 是离散实现。
- $\mathrm{KL}(\cdot \,\|\, p^\star)$ 是 Wasserstein 梯度流能量泛函；其流方程就是 Langevin FP 方程。 VI 和 MCMC 在连续时间下等价。
- SVGD 是核平滑的确定性粒子近似，避免了 MCMC 的随机游走低效问题。
- 收敛速率为 $2\lambda$，$\lambda$ 是 $p^\star$ 的 log-Sobolev 常数；高势垒混合是实际瓶颈。
- 贝叶斯神经网络后验采样归结为在损失景观上运行 Langevin 或 SVGD。

**系列结语**  
四篇文章用 PDE 统一了科学计算与机器学习：从神经网络求解 PDE （PINNs），到学习算子（FNO/DeepONet），再到训练作为梯度流，最后到概率推断作为 Fokker-Planck 动力学。核心主题：机器学习中的离散算法常可理解为连续 PDE 的时间离散化， PDE 理论是证明收敛性的关键语言。
## 10. 数值实现：能跑起来的 SDE 模拟

连续 Langevin SDE $dX = -\nabla U(X)\,dt + \sqrt{2}\,dW$ 离散化为
$$ X_{k+1} = X_k - \eta\,\nabla U(X_k) + \sqrt{2\eta}\,\xi_k,\quad \xi_k \sim \mathcal{N}(0, I). $$
这就是 **Euler-Maruyama**，算法全貌。 Python：

```python
import numpy as np
def langevin(grad_U, x0, eta=1e-3, n_steps=10000):
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    for _ in range(n_steps):
        x = x - eta*grad_U(x) + np.sqrt(2*eta)*np.random.randn(*x.shape)
        samples.append(x.copy())
    return np.array(samples)
```

实际运行会踩三个坑：

1. **步长偏置。** EM 的稳态分布和真 SDE 不一样，偏差量级 $O(\eta)$。要么 $\eta \to 0$，牺牲混合速度；要么套 Metropolis-Hastings 接受/拒绝——变成 MALA，无偏但每步多一次 log-density 计算。
2. **重尾分布炸梯度。** 若 $U$ 增长慢于二次， EM 在尾部发散。换 Milstein 高阶方法，或截断梯度。神经网络 log-density 必须处理。
3. **多模态目标卡住。** 朴素 Langevin 进了山谷就出不来。 Replica exchange 开 $K$ 条链，温度 $T_1 < \dots < T_K$，定期交换样本。代价是 $K$ 倍计算，但双峰后验混合速度提升数量级。

论文写"用 Langevin 采样"，背后至少踩一个坑。这些细节通常不提。
## 11. SVGD 的实现：理论藏着三个坑

梯度流公式
$$ \dot x_i = \frac{1}{n}\sum_j \bigl[k(x_j, x_i)\nabla\log p(x_j) + \nabla_{x_j}k(x_j, x_i)\bigr] $$
看着优雅，写对很难。

**Bug 1：带宽选错全崩。** RBF 核 $k(x, y) = \exp(-\|x-y\|^2/h)$，$h$ 不对就完蛋。常用中位数法：$h = \text{med}(\{\|x_i - x_j\|^2\}) / \log n$。高维 ($d > 20$) 下核值几乎为零，斥力消失，粒子全塌到模式点。

**Bug 2：梯度符号易错。** $\nabla_{x_j}k(x_j, x_i) = -\frac{2}{h}(x_j - x_i)k(x_j, x_i)$。负号搞反，斥力变引力，粒子聚成团，分布覆盖不了。半夜推导时特别容易出错。

**Bug 3：高维下 $n$ 太小。** SVGD 需要 $n \gtrsim d$ 才能撑起空间。原论文 [Liu & Zhu 2018] 用 $n = 50$ 跑 $d = 200$ 的贝叶斯神经网络，后验协方差秩最多 50，离真实差太远。后来的工作 [Chen & Ghattas 2020] 提出随机投影或矩阵值核改进。

记住一点：**定期检查平均 kernel 值**。低于 0.01，斥力失效，结果不可信。
## 12. 扩散模型：同一个 Fokker-Planck，时间反向

扩散模型训练网络近似每个噪声等级 $t$ 的 score $\nabla \log p_t(x)$。采样时跑**时间反向**的 SDE：
$$ dX = \bigl[-\nabla U(X) - 2\nabla\log p_t(X)\bigr]\,dt + \sqrt{2}\,d\bar W. $$

整个流程是 Fokker-Planck 的故事：

- **正向**：纯加噪。密度从数据 $p_0$ 演化到高斯 $p_T$。标准 FP 方程，$T$ 足够大。
- **Score Matching**：训练 $s_\theta(x, t) \approx \nabla\log p_t(x)$。用 Denoising Score Matching (Vincent, 2011)。关键技巧是 $\nabla_x \log p_t(x) = \mathbb{E}[\nabla_x \log q(x|x_0)\,|\,x]$，条件高斯 $q(x|x_0)$ 的 score 可直接计算。
- **反向**：用 Anderson (1982) 的时间反演 SDE 和学到的 score。每步是带学习漂移修正的 Langevin。

没人明说的事：**扩散模型 = SVGD 把 kernel 换成学到的 score 场**。 SVGD 手动平衡"斥力 vs 吸引力"，扩散从数据中学。两者都属于"密度上的梯度流"，第 4 节的 Wasserstein 几何正是描述它们的语言。

PDE-ML 第七章单独展开扩散模型细节，这里只点透 Fokker-Planck 的关系。
## 参考文献

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savar&eacute;. *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkh&auml;user, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.
