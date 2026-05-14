---
title: "偏微分方程与机器学习（四）：变分推断与 Fokker-Planck 方程"
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
![偏微分方程与机器学习（四）：变分推断与Fokker-Planck方程 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/illustration_1.png)


---

## 本文的七个维度

1. **动机**：为什么 VI 和 MCMC 看似迥异，实则求解同一个 PDE。
2. **理论**：从 SDE 推导出 Fokker-Planck 方程。
3. **几何**：KL 散度作为 Wasserstein 梯度流。
4. **算法**：Langevin Monte Carlo、平均场 VI 与 SVGD。
5. **收敛**：对数 Sobolev 不等式与 KL 的指数衰减。
6. **数值实验**：7 张可复现的图表，附完整代码。
7. **应用**：通过后验采样实现贝叶斯神经网络。

## 你将学到什么
- 任意 Itô SDE 所诱导的概率密度演化均受 Fokker-Planck 方程支配。
- Langevin 动力学是一种实用的采样算法，但其离散化会引入误差。
- 在 Wasserstein 空间中最小化 $\mathrm{KL}(q \| p^\star)$，本质上就是在求解 Fokker-Planck PDE。
- 在连续时间极限下，变分推断与 Langevin MCMC 具有深刻的等价性。
- Stein 变分梯度下降（SVGD）：一种确定性的粒子方法，巧妙地融合了两种范式。
- 贝叶斯神经网络中的实际后验推断。

## 前置知识

- 概率论基础（贝叶斯定理、KL 散度、期望）。
- 第三篇中介绍的 Wasserstein 梯度流。
- 对随机微积分的初步直觉（布朗运动、Itô 积分）。
- 实验部分需熟悉 Python / PyTorch。


---

## 推断问题

贝叶斯推断的目标是计算后验分布
$$p(\theta \mid x) \;=\; \frac{p(x \mid \theta)\, p(\theta)}{\int p(x \mid \theta')\, p(\theta')\, d\theta'},$$
但分母中的边际似然在非平凡模型下通常不可积。为此，两大类近似算法应运而生：

- **变分推断（VI）**：选取一个易处理的分布族 $\{q_\phi\}$，并最小化
  $$\mathrm{KL}\bigl(q_\phi \,\|\, p(\cdot\mid x)\bigr) \;=\; \mathbb{E}_{q_\phi}\!\left[\log \tfrac{q_\phi(\theta)}{p(\theta\mid x)}\right],$$
  等价于最大化 **证据下界（ELBO）** $\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(x\mid\theta)] - \mathrm{KL}(q_\phi \| p(\theta))$。

- **马尔可夫链蒙特卡洛（MCMC）**：构造一个平稳分布恰好为 $p(\cdot\mid x)$ 的马尔可夫链。**Langevin 动力学** 是其中基于梯度的经典代表。

二者表面上截然不同：VI 是对有限维参数 $\phi$ 的优化，而 MCMC 是一个无限时间的随机过程。然而，从 PDE 的视角看，它们实则是**同一概率测度演化过程**的不同实现方式。

## 从 SDE 到 Fokker-Planck

考虑一个 Itô SDE：
$$dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t.$$
对任意光滑测试函数 $f$，结合 Itô 引理与分部积分（假设密度及其导数在无穷远处趋于零），可得 **Fokker-Planck（FP）方程**：
$$
\boxed{\;\partial_t p \;=\; -\nabla\!\cdot\!(\mu\, p) \;+\; \tfrac{1}{2}\,\nabla\!\cdot\!\nabla\!\cdot\!(D\, p),\qquad D = \sigma\sigma^\top.\;}
$$
第一项代表**漂移**（输运），第二项代表**扩散**（弥散）。对于 **过阻尼 Langevin SDE**，取 $\mu = -\nabla V$、$\sigma = \sqrt{2\tau} I$，则 FP 方程变为：
$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p\,\nabla V\bigr) + \tau\, \Delta p,$$
在温和正则性条件下，其唯一稳态解即为 **Gibbs 分布** $p_\infty \propto e^{-V/\tau}$。若令 $V = -\log p^\star$ 且 $\tau = 1$，则稳态分布恰好等于目标分布 $p^\star$。

![双势阱中 Fokker-Planck 方程的密度演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig1_fokker_planck_evolution.png)
*图 1. 通过有限差分法求解 FP 方程。初始密度为左势阱中的窄高斯分布，随时间演化逐渐扩散、跨越势垒，最终收敛至对称的 Gibbs 密度 $p_\infty \propto e^{-V/D}$（右图）。*

## Langevin 动力学：采样即 PDE

用于从 $p^\star \propto e^{-V}$ 采样的 **过阻尼 Langevin 方程** 为：
$$dX_t = -\nabla V(X_t)\, dt + \sqrt{2\tau}\, dW_t.$$
其离散形式——**未校正 Langevin 算法（ULA）**——采用 Euler-Maruyama 格式：
$$X_{k+1} \;=\; X_k - \eta\, \nabla V(X_k) + \sqrt{2\eta\tau}\, \xi_k, \qquad \xi_k \sim \mathcal{N}(0, I).$$
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

ULA 存在 $O(\eta)$ 的偏差；**MALA**（Metropolis 校正 Langevin）通过接受-拒绝步骤恢复无偏性。**HMC**（哈密顿蒙特卡洛）则是其自然的欠阻尼动量版本。

![Langevin SDE 轨迹与经验密度。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig2_langevin_sde_to_density.png)
*图 2. 左：25 条代表性粒子在双势阱中运动，许多在有限时间内无法跨越势垒。右：400 个粒子的直方图随时间增长逐渐收敛至 Gibbs 目标分布——这正是图 1 中 FP 方程的随机实现。*

## KL 散度是 Wasserstein 梯度流

将相对于 $p^\star \propto e^{-V}$ 的 KL 散度分解为：
$$\mathcal{F}[p] \;=\; \mathrm{KL}(p\,\|\,p^\star) \;=\; \underbrace{\int p\log p\,dx}_{\text{负熵 }\mathcal{H}[p]} \;+\; \underbrace{\int p\, V\,dx}_{\text{势能}} \;+\; \text{常数}.$$
这正是第三篇中提到的 **自由能泛函**。Jordan-Kinderlehrer-Otto（JKO）定理（1998）指出，其 **Wasserstein-2 梯度流** 为：
$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p \nabla V\bigr) + \Delta p,$$
恰好对应 $\tau = 1$ 时 Langevin 的 FP 方程。因此：

> **等价性**。在 Wasserstein 空间中最小化 $\mathrm{KL}(\cdot \| p^\star)$ 与运行目标为 $p^\star$ 的 Langevin 动力学，本质上是**同一个 PDE**。VI 与 Langevin MCMC 只是这一连续时间梯度流的两种算法离散化方式。

| 维度 | 变分推断 | Langevin MCMC |
|---|---|---|
| 目标 | 最小化 $\mathrm{KL}(q_\phi \| p^\star)$ | 从 $p^\star$ 采样 |
| 状态 | 参数 $\phi$ | 粒子 $\{X^{(i)}\}$ |
| 更新步 | ELBO 上的梯度步 | SDE 的 Euler-Maruyama 步 |
| 连续极限 | KL 的 Wasserstein 梯度流 | Fokker-Planck 方程 |
| 稳态 | 若分布族足够表达，则 $q^\star = p^\star$ | $p_\infty = p^\star$ |
| 偏差来源 | 分布族受限 + 优化器噪声 | 离散化误差 $O(\eta)$ |

![KL 散度作为 Wasserstein 梯度流。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig3_kl_gradient_flow.png)
*图 3. 两种初始密度（集中型与弥散型）在 FP 方程驱动下向双峰目标演化。右图显示它们到 $p^\star$ 的 KL 散度单调递减——这正是梯度流理论所保证的行为。*

## VI 与 MCMC 的实践差异

尽管在连续极限下等价，二者在有限时间内的表现却大相径庭：

- **最小化反向 KL 的 VI 是“模式寻求”型**：当 $q$ 被限制在简单分布族（如高斯）时，最优解倾向于塌缩到单一众数，从而严重低估不确定性。
- **MCMC 是“质量覆盖”型**：只要链足够长，它会按真实质量比例访问所有众数，但在高势垒之间混合可能呈指数级缓慢。

![VI 与 MCMC 对比。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig4_vi_vs_mcmc.png)
*图 4. 左：在双峰后验下，最优的平均场高斯分布（以反向 KL 衡量）仅能拟合其中一个峰。右：4000 个 Langevin 样本正确覆盖了两个峰——但这仅因该一维示例中的势垒较低。*

## Stein 变分梯度下降

SVGD（Liu and Wang, 2016）是一种**确定性**粒子方法，巧妙地居于 VI 与 MCMC 之间。它维护一组粒子 $\{x_i\}_{i=1}^n$，并按如下规则更新：
$$x_i \;\leftarrow\; x_i + \eta\, \hat\phi^*(x_i),\qquad \hat\phi^*(x) = \tfrac{1}{n}\sum_{j=1}^n \Bigl[\,k(x_j, x)\,\nabla_{x_j}\log p^\star(x_j) \;+\; \nabla_{x_j} k(x_j, x)\,\Bigr],$$
其中 RBF 核 $k(x, y) = \exp(-\|x-y\|^2 / 2h^2)$ 的带宽 $h$ 通常用中位数启发式选取。该更新包含两个作用相反的项：

- **漂移项** $k\,\nabla\log p^\star$ 将粒子推向高概率区域；
- **排斥项** $\nabla k$ 则推开粒子，防止其塌缩至单一众数。

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

在粒子数趋于无穷的极限下，SVGD 满足 PDE：
$$\partial_t p \;=\; -\nabla\!\cdot\!\bigl(p\, v[p]\bigr),\qquad v[p](x) = \mathbb{E}_{y \sim p}\!\bigl[k(y,x)\nabla\log p^\star(y) + \nabla_y k(y,x)\bigr],$$
且当带宽 $h \to 0$ 时，该 PDE 退化为标准的 Fokker-Planck 方程。因此，SVGD 本质上是一种**核平滑的 FP 方程求解器**。

![SVGD 在双峰目标上的粒子演化。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig5_svgd_particles.png)
*图 5. 左：80 个 SVGD 粒子从原点出发，在数百次迭代内分裂并填充两个众数。右：完整轨迹显示，核排斥力有效防止了塌缩，而漂移项则将粒子稳定在 $\pm 2$ 附近。*

## 收敛理论

**定义（LSI）**。若 $p^\star$ 满足常数 $\lambda > 0$ 的 **对数 Sobolev 不等式（LSI）**，则对所有光滑且绝对连续于 $p^\star$ 的密度 $p$，有：
$$\mathrm{KL}(p \,\|\, p^\star) \;\leq\; \frac{1}{2\lambda}\, I(p \,\|\, p^\star),\qquad I(p\|p^\star) = \int p\, \bigl\|\nabla \log \tfrac{p}{p^\star}\bigr\|^2 dx.$$
右侧即为 **Fisher 信息**，同时也是 FP 方程中 KL 散度的耗散率：
$$\frac{d}{dt}\,\mathrm{KL}(p_t\,\|\,p^\star) \;=\; -\, I(p_t\,\|\,p^\star).$$
结合 LSI 可得 $\frac{d}{dt}\mathrm{KL}(p_t \| p^\star) \leq -2\lambda\, \mathrm{KL}(p_t \| p^\star)$，再由 Grönwall 不等式即得：
$$
\boxed{\;\mathrm{KL}(p_t\,\|\,p^\star) \;\leq\; e^{-2\lambda t}\, \mathrm{KL}(p_0\,\|\,p^\star).\;}
$$
强对数凹目标（即 $\nabla^2 V \succeq mI$）自动满足 LSI，且 $\lambda \geq m$（Bakry-Émery 理论）。而多峰目标通常具有极小的 $\lambda$，这解释了实践中观察到的指数级慢混合现象。

![收敛速率分析。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig6_convergence_analysis.png)
*图 6. 左：三种不同 log-Sobolev 常数下的理论 KL 衰减曲线 $e^{-2\lambda t}$。右：在光滑高斯目标上，VI、Langevin MCMC 与 SVGD 的实测 KL 轨迹——三者均收敛，但速率与噪声特性各异。*

## 应用：贝叶斯神经网络

贝叶斯神经网络对权重 $w$ 设定先验 $p(w)$，并寻求后验 $p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w)\, p(w)$。即使对于小型网络，后验也难以解析，但 Langevin 动力学仅需计算：
$$
\nabla_w \log p(w \mid \mathcal{D}) \;=\; \nabla_w \log p(\mathcal{D} \mid w) + \nabla_w \log p(w),
$$
这恰好就是反向传播中计算的梯度，再加上高斯先验项。**随机梯度 Langevin 动力学（SGLD）**（Welling and Teh, 2011）进一步用小批量梯度估计替代全数据梯度，使其适用于现代大规模场景。

下图使用 24 个随机傅里叶特征构建一个可处理的“贝叶斯 NN”，使得权重后验明确定义，并通过全批量 Langevin 进行采样。

![贝叶斯神经网络后验带。](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/fig7_bayesian_nn.png)
*图 7. 左：在训练数据存在缺口的回归任务中，90% 的 Langevin 后验预测带在缺失区域显著加宽。右：预测标准差在数据缺口处达到峰值——这正是点估计网络所缺乏的 **认知不确定性**。*

## 总结

- 任意 Itô SDE 都对应一个 Fokker-Planck PDE，描述其概率密度的演化。
- Langevin 动力学用于从 $p^\star \propto e^{-V}$ 采样；ULA / MALA / HMC 是其实用的离散实现。
- $\mathrm{KL}(\cdot \,\|\, p^\star)$ 是 Wasserstein 梯度流的能量泛函；其流方程正是 Langevin 的 FP 方程。因此，VI 与 MCMC 在连续时间下完全等价。
- SVGD 是一种核平滑的、确定性的粒子近似方法，避免了 MCMC 的随机游走低效问题。
- 收敛速率呈指数级，速率为 $2\lambda$，其中 $\lambda$ 是 $p^\star$ 的 log-Sobolev 常数；实践中，穿越高势垒的混合速度是主要瓶颈。
- 贝叶斯神经网络的后验采样可归结为在损失景观上运行 Langevin 或 SVGD。

**系列结语**。在四篇文章中，我们借助 PDE 统一了科学计算与机器学习：从用神经网络求解 PDE（PINNs），到学习解算子（FNO/DeepONet），再到将训练视为梯度流，最后将概率推断理解为 Fokker-Planck 动力学。贯穿始终的主题是：**机器学习中的离散算法，通常最好被理解为某个连续 PDE 的时间离散化**，而 PDE 理论正是证明其收敛性的通用语言。

## 数值实现：真正能跑起来的 SDE 模拟

连续 Langevin SDE $dX = -\nabla U(X)\,dt + \sqrt{2}\,dW$ 的离散形式为：
$$ X_{k+1} = X_k - \eta\,\nabla U(X_k) + \sqrt{2\eta}\,\xi_k,\quad \xi_k \sim \mathcal{N}(0, I). $$
这就是 **Euler-Maruyama（EM）** 方法，也是整个算法的核心。Python 实现如下：

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

实践中会遇到三个典型问题：

1. **步长偏差**。EM 采样的稳态分布与真实 SDE 略有不同，偏差为 $O(\eta)$。要么让 $\eta \to 0$（牺牲混合速度），要么在其外层包裹 Metropolis-Hastings 接受-拒绝步骤——这就得到了 MALA（Metropolis 校正 Langevin），虽无偏但每步需额外一次对数密度计算。
2. **重尾分布导致发散**。若势能 $U$ 的增长慢于二次，EM 在尾部会爆炸。此时应改用高阶格式（如 Milstein）或对梯度进行裁剪。对于神经网络的对数密度，这一步通常是必需的。
3. **多模态目标下的困局**。朴素 Langevin 一旦陷入某个势阱就难以逃逸。**副本交换（Replica Exchange）**（又称并行退火）通过同时运行 $K$ 条温度为 $T_1 < \dots < T_K$ 的链并定期交换状态来缓解此问题。虽然计算成本增加 $K$ 倍，但在双峰后验等场景下，混合速度可提升一个数量级。

凡是你看到“我们用 Langevin 采样”的地方，背后几乎必然涉及上述三个问题之一。可惜论文通常对此避而不谈。

## SVGD 的实践：理论背后藏着三个陷阱

梯度流公式
$$ \dot x_i = \frac{1}{n}\sum_j \bigl[k(x_j, x_i)\nabla\log p(x_j) + \nabla_{x_j}k(x_j, x_i)\bigr] $$
看似优雅，但正确实现并不容易。

**Bug 1：带宽选择不当**。RBF 核 $k(x, y) = \exp(-\|x-y\|^2/h)$ 对带宽 $h$ 极其敏感。标准做法是中位数技巧：$h = \text{med}(\{\|x_i-x_j\|^2\})/\log n$。若不这么做，在 $d > 20$ 的高维空间中，核值对几乎所有粒子对都接近零，排斥力随之消失，粒子最终全部塌缩至众数。

**Bug 2：梯度符号错误**。$\nabla_{x_j}k(x_j, x_i) = -\frac{2}{h}(x_j - x_i)k(x_j, x_i)$。负号一旦弄反，排斥力就变成了吸引力，导致粒子聚集而非覆盖。这种错误在深夜推导时极易发生。

**Bug 3：高维下粒子数不足**。SVGD 需要粒子数 $n \gtrsim d$ 才能有效张成空间。例如，原始论文（Liu & Zhu, 2018）在 $d = 200$ 的贝叶斯神经网络上仅用 $n = 50$ 个粒子，导致所恢复后验的协方差矩阵秩至多为 50，远非真实情况。后续工作（如 Chen & Ghattas, 2020）通过随机投影或矩阵值核来缓解此问题。

若只记住一点：**定期监测平均成对核值**。一旦低于 0.01，就意味着排斥相互作用已失效，结果不可信。

## 基于 Score 的扩散模型：同一个 Fokker-Planck，时间反转

扩散模型训练一个网络来近似每个噪声等级 $t$ 下的 score $\nabla \log p_t(x)$。采样时则运行一个**逆时间 SDE**：
$$ dX = \bigl[-\nabla U(X) - 2\nabla\log p_t(X)\bigr]\,dt + \sqrt{2}\,d\bar W. $$
整个流程本质上仍是 Fokker-Planck 的故事：

- **前向过程**：纯加噪。密度从数据分布 $p_0$ 演化至高斯分布 $p_T$，遵循标准 FP 方程，其中 $\sigma$ 的选择确保 $T$ 足够大。
- **Score 匹配**：通过去噪 score 匹配（Vincent, 2011）训练 $s_\theta(x, t) \approx \nabla\log p_t(x)$。其核心技巧在于：对于条件高斯 $q(x|x_0)$，有 $\nabla_x \log p_t(x) = \mathbb{E}[\nabla_x \log q(x|x_0)\,|\,x]$，可直接计算。
- **逆向过程**：利用 Anderson（1982）提出的逆时间 SDE 和学到的 score 进行采样。每一步本质上是一次带有学习漂移修正的 Langevin 更新。

鲜少有人明确指出：**扩散模型其实就是 SVGD，只不过将核函数替换为了从数据中学到的 score 场**。SVGD 手动平衡“排斥 vs 吸引”，而扩散模型则从数据中学习这一平衡。正因如此，二者同属“密度上的梯度流”这一框架，而第四节所述的 Wasserstein 几何正是描述它们的恰当语言。

PDE-ML 系列第七章将专门深入探讨扩散模型；此处仅旨在点明其与 Fokker-Planck 方程的深刻联系。

## 参考文献

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savaré. *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkhäuser, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.
