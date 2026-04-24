---
title: "PDE与机器学习（四）：变分推断与Fokker-Planck方程"
date: 2024-09-04 09:00:00
tags:
  - 变分推断
  - Fokker-Planck
  - PDE
  - 概率
  - 扩散过程
categories:
  - PDE与机器学习
series:
  name: "PDE与机器学习"
  part: 4
  total: 8
lang: zh-CN
mathjax: true
description: "变分推断与 Langevin MCMC 在连续时间下是同一个 Fokker-Planck PDE：从 SDE 推导密度演化、KL 散度作为 Wasserstein 梯度流、SVGD 粒子方法、对数 Sobolev 不等式给出指数收敛、贝叶斯神经网络应用。"
---

> **系列**：PDE 与机器学习 -- 第 4 篇 / 共 4 篇
> [<-- 上一篇：变分原理与优化](/zh/PDE与机器学习-三-变分原理与优化/)

## 本文的七个维度

1. **动机**：为什么 VI 与 MCMC 看似不同，却在解同一个 PDE。
2. **理论**：从随机微分方程严格推导 Fokker-Planck 方程。
3. **几何**：KL 散度作为 Wasserstein 空间中的梯度流。
4. **算法**：Langevin Monte Carlo、平均场 VI、SVGD。
5. **收敛**：对数 Sobolev 不等式与指数收敛速率。
6. **数值实验**：7 张可复现图，附完整脚本。
7. **应用**：用 Langevin 采样近似贝叶斯神经网络后验。

## 你将学到

- 任意 It&ocirc; SDE 的概率密度满足 Fokker-Planck 方程。
- Langevin 动力学作为采样算法的实用性，及其离散化误差。
- 在 Wasserstein 空间中最小化 $\mathrm{KL}(q\|p^\star)$ **本身就是** Fokker-Planck PDE。
- 变分推断与 Langevin MCMC 在连续时间下完全等价。
- Stein 变分梯度下降（SVGD）：用确定性粒子求解变分推断。
- 用上述工具做贝叶斯神经网络的后验推断。

## 前置知识

- 概率论（贝叶斯定理、KL 散度、期望）。
- 第 3 篇的 Wasserstein 梯度流。
- 一点点随机分析直觉（布朗运动、It&ocirc; 积分）。
- Python / PyTorch 用于实验。

---

## 1. 推断问题

贝叶斯推断要求后验

$$
p(\theta \mid x) \;=\; \frac{p(x \mid \theta)\,p(\theta)}{\int p(x \mid \theta')\,p(\theta')\,d\theta'},
$$

但分母（边际似然）几乎总是无法解析计算。两类近似方法应运而生：

- **变分推断（VI）**：选定可处理的分布族 $\{q_\phi\}$，最小化

  $$\mathrm{KL}(q_\phi \,\|\, p(\cdot \mid x)) = \mathbb{E}_{q_\phi}\!\left[\log \tfrac{q_\phi(\theta)}{p(\theta \mid x)}\right],$$

  等价于最大化 **证据下界（ELBO）** $\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(x\mid\theta)] - \mathrm{KL}(q_\phi \| p(\theta))$。

- **马尔可夫链蒙特卡洛（MCMC）**：构造平稳分布恰为 $p(\cdot \mid x)$ 的马氏链。**Langevin 动力学**是其中最自然的梯度型实例。

两者表面上完全不同：VI 是 $\phi$ 上的有限维优化，MCMC 是无限时间的随机过程。**PDE 视角告诉我们：它们是同一个概率测度演化的两种采样方式。**

## 2. 从 SDE 到 Fokker-Planck

考虑 It&ocirc; SDE

$$dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t.$$

任取光滑测试函数 $f$，由 It&ocirc; 引理求 $\mathbb{E} f(X_t)$ 的导数，再做分部积分（设无穷远处 $p$ 及其导数为零），即可得到 **Fokker-Planck 方程**：

$$
\boxed{\;\partial_t p \;=\; -\nabla\!\cdot\!(\mu\, p) \;+\; \tfrac{1}{2}\,\nabla\!\cdot\!\nabla\!\cdot\!(D\, p),\quad D = \sigma\sigma^\top.\;}
$$

第一项是**漂移**（输运），第二项是**扩散**（弥散）。当 $\mu = -\nabla V$、$\sigma = \sqrt{2\tau} I$，对应 **过阻尼 Langevin SDE**：

$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p \nabla V\bigr) + \tau\,\Delta p,$$

唯一稳态解（在弱条件下）即 **Gibbs 分布** $p_\infty \propto e^{-V/\tau}$。取 $V = -\log p^\star$、$\tau = 1$，则稳态分布恰为目标 $p^\star$。

![双势阱中 Fokker-Planck 方程的密度演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig1_fokker_planck_evolution.png)
*图 1. 用有限差分求解 1D Fokker-Planck 方程。初始为左阱中的窄高斯，密度逐步扩散、跨越势垒，最终收敛到对称的 Gibbs 平稳分布 $p_\infty \propto e^{-V/D}$（右图）。*

## 3. Langevin 动力学：把采样变成 PDE

**过阻尼 Langevin 方程** 用于从 $p^\star \propto e^{-V}$ 采样：

$$dX_t = -\nabla V(X_t)\,dt + \sqrt{2\tau}\,dW_t.$$

离散化得到 **未校正 Langevin 算法（ULA）**，即 Euler-Maruyama 格式：

$$X_{k+1} = X_k - \eta\,\nabla V(X_k) + \sqrt{2\eta\tau}\,\xi_k,\qquad \xi_k \sim \mathcal{N}(0, I).$$

```python
import torch, numpy as np

def langevin_sample(grad_log_p, x0, step=0.01, n_steps=10_000, tau=1.0):
    """过阻尼 Langevin 采样器（即 ULA）。

    grad_log_p : 返回 grad(log p*(x)) 的可调用对象
    x0         : (n_particles, dim) 初始位置
    """
    x = x0.clone()
    for _ in range(n_steps):
        x = x + step * grad_log_p(x) + np.sqrt(2 * step * tau) * torch.randn_like(x)
    return x
```

ULA 的偏差是 $O(\eta)$；**MALA**（Metropolis 校正版 Langevin）通过 accept-reject 步骤恢复严格无偏；**HMC**（哈密顿蒙特卡洛）则是带动量的欠阻尼版本。

![Langevin SDE 轨迹与其经验密度。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig2_langevin_sde_to_density.png)
*图 2. 左：双势阱中的 25 条粒子轨迹，部分粒子在有限时间内不会跨越势垒。右：400 个粒子的直方图随时间逐步收敛到 Gibbs 目标 -- 这就是图 1 那个 PDE 的随机实现。*

## 4. KL 散度是 Wasserstein 梯度流

把 KL 散度相对 $p^\star \propto e^{-V}$ 分解：

$$\mathcal{F}[p] = \mathrm{KL}(p\,\|\,p^\star) = \underbrace{\int p\log p\,dx}_{\text{负熵 }\mathcal{H}[p]} + \underbrace{\int p\, V\,dx}_{\text{势能}} + \text{常数}.$$

这就是第 3 篇出现过的 **自由能泛函**。Jordan-Kinderlehrer-Otto（JKO，1998）定理告诉我们，它的 **2-Wasserstein 梯度流** 为

$$\partial_t p = \nabla\!\cdot\!\bigl(p\nabla V\bigr) + \Delta p,$$

正是 Langevin 在 $\tau = 1$ 时的 Fokker-Planck 方程。于是：

> **等价性**。在 Wasserstein 空间最小化 $\mathrm{KL}(\cdot \| p^\star)$ 与运行目标为 $p^\star$ 的 Langevin 动力学，**是同一个 PDE**。VI 与 Langevin MCMC 是同一个连续时间梯度流的两种算法离散化。

| 视角 | 变分推断 | Langevin MCMC |
|---|---|---|
| 目标 | 最小化 $\mathrm{KL}(q_\phi \| p^\star)$ | 从 $p^\star$ 采样 |
| 状态 | 参数 $\phi$ | 粒子 $\{X^{(i)}\}$ |
| 步骤 | ELBO 上的梯度步 | SDE 的 Euler-Maruyama |
| 连续极限 | KL 的 Wasserstein 梯度流 | Fokker-Planck 方程 |
| 稳态 | 若分布族足够表达，$q^\star = p^\star$ | $p_\infty = p^\star$ |
| 偏差 | 受限分布族 + Adam 噪声 | 离散化 $O(\eta)$ |

![KL 散度作为 Wasserstein 梯度流。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig3_kl_gradient_flow.png)
*图 3. 两个不同初值（集中型与发散型）在 FP 方程驱动下都收敛到双峰目标。右图显示 KL 散度对 $p^\star$ 单调递减 -- 这就是梯度流保证的体现。*

## 5. VI 与 MCMC 在实践中的差异

虽然连续极限相同，但有限时间行为差距巨大。

- **VI（最小化反向 KL）是模式寻求型**：当 $q$ 限制在简单分布族内，最优解会塌缩到单一模式，从而**低估不确定性**。
- **MCMC 是质量覆盖型**：足够长的链按比例访问每个模式，但跨越势垒的混合时间可能指数慢。

![VI 与 MCMC 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig4_vi_vs_mcmc.png)
*图 4. 左：双峰后验下，最优平均场高斯（反向 KL 意义）只能覆盖一个峰。右：4000 个 Langevin 样本同时覆盖两个峰 -- 但前提是势垒在这个 1D 例子中较低。*

## 6. Stein 变分梯度下降

SVGD（Liu & Wang, 2016）是介于 VI 与 MCMC 之间的 **确定性粒子方法**。维护粒子 $\{x_i\}_{i=1}^n$，按下式更新：

$$x_i \leftarrow x_i + \eta\,\hat\phi^*(x_i),\quad \hat\phi^*(x) = \tfrac{1}{n}\sum_{j=1}^n \Bigl[\,k(x_j,x)\,\nabla_{x_j}\log p^\star(x_j) + \nabla_{x_j} k(x_j,x)\Bigr],$$

其中 $k(x,y) = \exp(-\|x-y\|^2 / 2h^2)$ 是 RBF 核（带宽 $h$ 用中位数启发式确定）。两项含义截然相反：

- **漂移项** $k\,\nabla\log p^\star$：把粒子推向高概率区。
- **排斥项** $\nabla k$：把粒子相互推开，避免塌缩到单一模式。

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

无穷粒子极限下，SVGD 满足

$$\partial_t p = -\nabla\!\cdot\!\bigl(p\, v[p]\bigr),\quad v[p](x) = \mathbb{E}_{y\sim p}\bigl[k(y,x)\nabla\log p^\star(y) + \nabla_y k(y,x)\bigr],$$

当带宽 $h \to 0$ 时回到标准 Fokker-Planck 方程。**SVGD 是核平滑后的 FP 求解器。**

![SVGD 在双峰目标上的粒子演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig5_svgd_particles.png)
*图 5. 左：80 个粒子从原点出发，几百步内分裂覆盖两个峰。右：完整粒子轨迹显示核排斥力防止塌缩，漂移项把粒子稳定在 $\pm 2$ 附近。*

## 7. 收敛理论

**定义（LSI）。** 称 $p^\star$ 满足常数为 $\lambda > 0$ 的 **对数 Sobolev 不等式**，若对所有 $p \ll p^\star$：

$$\mathrm{KL}(p \,\|\, p^\star) \leq \frac{1}{2\lambda}\, I(p \,\|\, p^\star),\quad I(p\|p^\star) = \int p\,\bigl\|\nabla\log\tfrac{p}{p^\star}\bigr\|^2 dx.$$

右侧 **Fisher 信息** 恰是 FP 方程下 KL 的耗散率：

$$\frac{d}{dt}\mathrm{KL}(p_t\,\|\,p^\star) = -\,I(p_t\,\|\,p^\star).$$

由 LSI 可得 $\frac{d}{dt}\mathrm{KL} \leq -2\lambda\,\mathrm{KL}$，由 Gr&ouml;nwall 不等式：

$$
\boxed{\;\mathrm{KL}(p_t\,\|\,p^\star) \leq e^{-2\lambda t}\,\mathrm{KL}(p_0\,\|\,p^\star).\;}
$$

强对数凹目标（$\nabla^2 V \succeq mI$）由 Bakry-&Eacute;mery 自动满足 LSI 且 $\lambda \geq m$。多峰目标的 $\lambda$ 通常很小，正好解释了实践中观察到的指数级慢混合。

![收敛速率分析。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig6_convergence_analysis.png)
*图 6. 左：三个不同 $\lambda$ 下的理论 KL 衰减曲线 $e^{-2\lambda t}$。右：VI、Langevin MCMC、SVGD 在光滑高斯目标上的实测 KL 轨迹 -- 三者都收敛，但速率与噪声特性差异明显。*

## 8. 应用：贝叶斯神经网络

贝叶斯神经网络在权重上加先验 $p(w)$ 求后验 $p(w \mid \mathcal{D}) \propto p(\mathcal{D}\mid w)\,p(w)$。即使是小网络，后验也无法解析；但 Langevin 只需要

$$\nabla_w \log p(w \mid \mathcal{D}) = \nabla_w \log p(\mathcal{D}\mid w) + \nabla_w \log p(w),$$

也就是反向传播本来就在算的梯度，加一个高斯先验项。**SGLD**（Stochastic Gradient Langevin Dynamics，Welling & Teh, 2011）用 mini-batch 梯度替代全批梯度，让该方法在现代规模上可行。

下图用 24 个随机傅里叶特征作为可控的 "贝叶斯 NN"，使权重后验良好定义，并用全批 Langevin 采样。

![贝叶斯神经网络后验带。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD%E4%B8%8EFokker-Planck%E6%96%B9%E7%A8%8B/fig7_bayesian_nn.png)
*图 7. 左：训练数据有缺口的回归问题，90% Langevin 后验带在数据缺失处明显加宽。右：预测标准差在数据缺口处达到峰值 -- 这正是普通点估计网络所缺失的 **认知不确定性**。*

## 9. 总结

- 任意 It&ocirc; SDE 都对应一个描述其密度演化的 Fokker-Planck PDE。
- Langevin 动力学从 $p^\star \propto e^{-V}$ 采样；ULA / MALA / HMC 是其离散实现。
- $\mathrm{KL}(\cdot \,\|\, p^\star)$ 是 Wasserstein 梯度流的能量泛函；其流方程**就是** Langevin 的 FP 方程。VI 与 MCMC 在连续时间下等价。
- SVGD 是同一流的核平滑确定性粒子近似，避免了 MCMC 的随机游走低效。
- 收敛速率为 $2\lambda$（$\lambda$ 为 LSI 常数）；高势垒下的混合是实际瓶颈。
- 贝叶斯神经网络的后验采样退化为在损失景观上跑 Langevin（或 SVGD）。

**系列结语**。四篇文章用 PDE 把科学计算与机器学习串在一起 -- 从用神经网络求解 PDE（PINNs），到学习算子（FNO/DeepONet），到把训练理解为梯度流，再到把概率推断理解为 Fokker-Planck 动力学。**机器学习中的离散算法，常常最适合理解为某个连续 PDE 的时间离散化；而 PDE 理论是证明收敛性的语言。**

## 参考文献

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savar&eacute;. *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkh&auml;user, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.

---

## 系列导航

| 部分 | 主题 |
|------|------|
| [1](/zh/PDE与机器学习-一-物理信息神经网络/) | 物理信息神经网络 |
| [2](/zh/PDE与机器学习-二-神经算子理论/) | 神经算子理论 |
| [3](/zh/PDE与机器学习-三-变分原理与优化/) | 变分原理与优化 |
| **4** | **变分推断与 Fokker-Planck 方程（本文）** |
| [5](/zh/PDE与机器学习-五-辛几何与保结构网络/) | 辛几何与保结构网络 |
| [6](/zh/PDE与机器学习-六-连续归一化流与Neural-ODE/) | 连续归一化流与 Neural ODE |
| [7](/zh/PDE与机器学习-七-扩散模型与Score-Matching/) | 扩散模型与 Score Matching |
| [8](/zh/PDE与机器学习-八-反应扩散系统与GNN/) | 反应扩散系统与 GNN |
