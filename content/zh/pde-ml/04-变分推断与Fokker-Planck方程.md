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
![偏微分方程与机器学习（四）：变分推断与Fokker-Planck方程 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-Variational-Inference/illustration_1.png)


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

### 哈密顿蒙特卡洛：动量突破随机游走的瓶颈

ULA（无调整朗之万算法）依靠随机扩散进行探索，在高维空间或跨越能量势垒时，其收敛速度极其缓慢。**哈密顿蒙特卡洛（HMC）** 引入辅助动量变量 $v$，并在联合能量函数 $H(\theta, v) = V(\theta) + \frac{1}{2}\|v\|^2$ 上模拟哈密顿动力学。动量使采样器能够像“滚动”一样滑过低概率密度的谷底，而无需被动等待噪声将其“推”过势垒。

其中，**蛙跳积分器（leapfrog integrator）** 具备体积守恒性（辛结构）和时间可逆性，结合后续的 Metropolis 接受-拒绝校正，即可严格满足细致平衡条件：

```python
import numpy as np

def hmc_sample(V, grad_V, x0, step=0.02, L=20, n_samples=2000):
    # 使用蛙跳积分的哈密顿蒙特卡洛采样器
    # grad_V: 势能 V(x) = -log p*(x) 的梯度函数
    # L: 每次提议所执行的蛙跳步数
    d = x0.shape[0]
    x = x0.copy()
    samples = [x.copy()]
    accepted = 0

    for _ in range(n_samples):
        v = np.random.randn(d)  # 从标准正态分布采样动量
        x_prop, v_prop = x.copy(), v.copy()

        # 蛙跳积分（leapfrog integration）
        v_prop = v_prop - 0.5 * step * grad_V(x_prop)
        for l_step in range(L - 1):
            x_prop = x_prop + step * v_prop
            v_prop = v_prop - step * grad_V(x_prop)
        x_prop = x_prop + step * v_prop
        v_prop = v_prop - 0.5 * step * grad_V(x_prop)

        # Metropolis 接受/拒绝步骤
        H_current = 0.5 * np.dot(v, v) + V(x)      # 当前哈密顿量
        H_proposed = 0.5 * np.dot(v_prop, v_prop) + V(x_prop)
        log_alpha = H_current - H_proposed

        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
            accepted += 1

        samples.append(x.copy())

    print(f"HMC 接受率: {accepted / n_samples:.2%}")
    return np.array(samples)
```

为什么 HMC 显著优于 ULA？考虑一个具有高度为 $B$ 势垒的双阱势函数：ULA 穿越该势垒所需的步数约为 $O(e^B / \eta)$；而 HMC 只需初始动量提供足够动能——由于 $v$ 每次独立采样，获得足够能量的概率约为 $\sim e^{-B}$。随后，蛙跳轨迹将以弹道式（ballistic）运动在 $L$ 步内直接越过势垒。这使得混合时间从指数级依赖 $B$（扩散主导）大幅降低为多项式级（弹道输运主导）。

从偏微分方程（PDE）视角看：HMC 对应于**欠阻尼朗之万方程**（即含动能项的动力学版本）：
$$
d\theta = v\,dt,\quad dv = -\nabla V(\theta)\,dt - \gamma v\,dt + \sqrt{2\gamma}\,dW,
$$
其对应的 Fokker–Planck 方程具有二阶结构，收敛速度明显快于过阻尼（纯位置更新）情形。

## Langevin 动力学：采样即 PDE

![Langevin动力学: 200个粒子在双势阱中采样并收敛至Gibbs平衡态](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/pde-ml/04-变分推断与Fokker-Planck方程/anim_langevin_sampling.gif)

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

### 随机梯度朗之万动力学（SGLD）

全批量朗之万采样要求每一步都基于整个数据集计算梯度 $\nabla V(\theta) = -\sum_{i=1}^N \nabla \log p(x_i \mid \theta) - \nabla \log p(\theta)$。当数据量 $N = 10^6$ 时，这种计算开销显然难以承受。**随机梯度朗之万动力学（SGLD）**（Welling 和 Teh，2011）用小批量梯度估计替代全梯度，并巧妙地让注入的噪声身兼二职：既作为随机微分方程（SDE）中的扩散项，又起到正则化作用：

$$\theta_{k+1} = \theta_k + \frac{\eta_k}{2}\left(\nabla \log p(\theta_k) + \frac{N}{B}\sum_{i \in \mathcal{B}_k} \nabla \log p(x_i \mid \theta_k)\right) + \sqrt{\eta_k}\,\xi_k$$

其中 $\mathcal{B}_k$ 是大小为 $B$ 的随机小批量，$\xi_k \sim \mathcal{N}(0, I)$。核心洞见在于：若步长 $\eta_k$ 按照衰减策略趋于零，则小批量引入的梯度噪声（量级为 $O(\eta_k)$）将逐渐主导人为注入的噪声（量级为 $O(\sqrt{\eta_k})$），从而使算法平滑地从 SGD（以优化为目标）过渡到朗之万采样（以近似后验分布为目标）。

```python
import numpy as np

def sgld(grad_log_prior, grad_log_likelihood, x0, data,
         batch_size=64, n_steps=50000, eta0=1e-3, decay=0.9999):
    # 随机梯度朗之万动力学（SGLD）
    # grad_log_prior: log p(theta) 关于 theta 的梯度
    # grad_log_likelihood: 单个数据点 x_i 对应的 log p(x_i | theta) 关于 theta 的梯度
    N = len(data)
    d = x0.shape[0]
    x = x0.copy()
    samples = []

    for k in range(n_steps):
        eta = eta0 * (decay ** k)
        # 小批量梯度估计
        batch_idx = np.random.choice(N, batch_size, replace=False)
        grad_lik = np.zeros(d)
        for i in batch_idx:
            grad_lik += grad_log_likelihood(x, data[i])
        grad_lik *= (N / batch_size)  # 缩放至全量数据集对应的梯度尺度

        grad_total = grad_log_prior(x) + grad_lik
        noise = np.sqrt(eta) * np.random.randn(d)
        x = x + 0.5 * eta * grad_total + noise

        if k % 100 == 0:
            samples.append(x.copy())

    return np.array(samples)
```

在实践中，SGLD 已成为“大规模贝叶斯深度学习”的核心引擎——它复用了与标准 SGD 完全相同的底层小批量训练基础设施。每一步的计算开销与常规训练完全一致；唯一新增的操作就是注入噪声项 $\sqrt{\eta_k}\,\xi_k$。当然，这也带来权衡：有限的步长会引入渐近偏差；而判断采样是否收敛，则需持续监控梯度估计器中“噪声”与“信号”的比值。

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

### 采样器对比：ULA vs MALA vs HMC vs SGLD

下表总结了我们此前讨论的四种基于梯度的 MCMC 算法。所有算法均以目标分布 $p^\star \propto e^{-V}$ 为采样目标，该分布在 $d$ 维空间中具有条件数 $\kappa = L/m$（即光滑性常数 $L$ 与强凸性常数 $m$ 的比值）。

| 算法 | 每步计算开销 | 偏差 | 收敛速度（总变差距离降至 $\varepsilon$ 所需步数） | 最适用场景 |
|------|----------------|------|---------------------------------------------------|-------------|
| **ULA**（无调整 Langevin 算法） | 1 次梯度计算 | 渐近偏差为 $O(\eta d)$ | $\tilde{O}(\kappa^2 d / \varepsilon^2)$ 步 | 低维（$d$ 较小）、梯度计算极快、可容忍一定偏差 |
| **MALA**（Metropolis 调整版 Langevin 算法） | 1 次梯度计算 + 1 次密度函数求值 | 无偏差（精确抽样） | $\tilde{O}(\kappa d^{1/3} / \varepsilon^{2/3})$ 步 | 中等维度、需要无偏样本 |
| **HMC**（Hamiltonian Monte Carlo） | $L$ 次梯度计算（$L$ 为 Leapfrog 步数） | 无偏差（精确抽样） | $\tilde{O}(\kappa^{1/2} d^{1/4})$ 步 | 高维（$d$ 较大）、目标函数光滑、主流工具如 Stan / PyMC 默认采用 |
| **SGLD**（随机梯度 Langevin 动力学） | 1 个 mini-batch 的梯度计算 | $O(\eta + \sigma^2_B \eta)$（$\sigma^2_B$ 为 mini-batch 梯度方差） | 无简洁收敛界（过程非平稳） | 样本量 $N$ 极大时的贝叶斯深度学习等场景 |

关键观察：

- **HMC 是中等维度（$d < 10^4$）下的黄金标准**：当全梯度计算可行时，其 $d^{1/4}$ 的维度依赖性远优于 ULA 的线性依赖 $d$。
- **SGLD 在墙钟时间（wall-clock time）上更具优势**：当数据集规模 $N$ 极大时，每步仅需 $O(B)$ 计算（$B$ 为 mini-batch 大小），而非 $O(N)$；但若步长 $\eta$ 固定，其渐近偏差无法完全消除。
- **MALA 是对 ULA 偏差的自然修正方案**，但在高维情形下，其性能提升远不如 HMC 显著。
- 这四种算法本质上都对应同一 Fokker-Planck 动力学流，差异仅在于离散化策略（如 Euler 或 Leapfrog）以及是否引入动量变量。

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

### 自适应带宽与非高斯后验分布

中位数启发式法（median heuristic）$h = \text{med}(\|x_i - x_j\|^2) / \log n$ 实现简单，但鲁棒性较差。它在以下两种常见场景中会失效：

1. **尺度不均的多峰目标分布**：若一个峰紧致、另一个峰弥散，则单一全局带宽无法同时满足——在紧致簇内提供足够的粒子排斥力，又在两峰之间的间隙中维持必要的吸引力。
2. **香蕉形（强相关）后验分布**：此时成对距离主要由长轴方向主导，导致计算出的 $h$ 在窄方向上过大；粒子只能沿香蕉曲线滑动，却始终无法填满其横向宽度。

一种实用的改进方案是 **逐粒子自适应带宽（per-particle adaptive bandwidth）**：为每个粒子 $i$ 单独计算其局部带宽 $h_i$，依据其 $k$ 近邻距离确定。由此生成的空间变核（spatially-varying kernel）可自然适配局部几何结构：

```python
import numpy as np
from scipy.spatial.distance import cdist

def svgd_adaptive(x, score_fn, eta=0.05, k_neighbors=5):
    # 使用逐粒子自适应带宽的 SVGD
    n, d = x.shape
    score = score_fn(x)  # 形状为 (n, d)
    dists = cdist(x, x)  # 形状为 (n, n)

    # 基于 k 近邻距离计算每个粒子的局部带宽
    sorted_dists = np.sort(dists, axis=1)
    h_local = sorted_dists[:, k_neighbors]  # 到第 k 个近邻的距离
    h_local = np.maximum(h_local, 1e-6)     # 防止带宽过小导致数值不稳定

    # 使用带宽的几何平均构造核矩阵
    h_matrix = np.sqrt(np.outer(h_local, h_local))  # 形状为 (n, n)
    K = np.exp(-dists**2 / (2 * h_matrix**2))

    # 计算核函数关于第二变量的梯度：d/dx_j k(x_j, x_i)
    diff = x[:, None, :] - x[None, :, :]  # 形状为 (n, n, d)
    grad_K = -diff / (h_matrix[:, :, None]**2) * K[:, :, None]

    # SVGD 更新步
    phi = (K @ score + grad_K.sum(axis=0)) / n
    return x + eta * phi
```

**香蕉形后验分布示例**：考虑二维分布  
$$p^\star(x_1, x_2) \propto \exp\bigl(-\frac{1}{2}(x_1^2/s_1^2 + (x_2 - x_1^2)^2/s_2^2)\bigr)$$  
其中 $s_1 = 2,\, s_2 = 0.5$。该分布形成一条狭窄而弯曲的脊线，全局固定带宽的 SVGD 很难充分覆盖其整个结构。采用自适应带宽后，仅需 500 次迭代，粒子即可沿整条“香蕉”均匀铺开；而基于中位数启发式的 SVGD 则迅速坍缩至原点附近的弯曲处。

这一现象具有普适意义：在真实的贝叶斯后验分布中（它们极少是高斯分布），局部几何自适应并非锦上添花——而是决定算法能否实现**全局覆盖**还是陷入**模式坍缩（mode collapse）** 的关键分水岭。更进一步，矩阵值核（matrix-valued kernels，Wang 等，2019）通过为每个粒子配备完整的 $d \times d$ 度量张量，将这种几何感知能力提升到了更高维度。

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

### 完整实验：贝叶斯神经网络在 1D 回归任务上的不确定性建模

为了让贝叶斯不确定性变得直观可感，我们使用一组带人为空缺（gap）的合成数据训练一个小型神经网络，并通过随机梯度朗之万动力学（SGLD）对后验分布进行采样。理想情况下，预测置信带应在数据缺失区域（即空缺处）显著变宽。

```python
import numpy as np

# 生成训练数据：在 x=1 到 x=3 之间刻意留出空缺
np.random.seed(42)
x_left = np.random.uniform(-2, 1, 40)   # 左侧数据点（x ∈ [-2, 1)）
x_right = np.random.uniform(3, 6, 40)   # 右侧数据点（x ∈ (3, 6]）
x_train = np.concatenate([x_left, x_right])
y_train = np.sin(x_train) + 0.1 * np.random.randn(len(x_train))  # 添加高斯噪声

# 简单的单隐层网络：输入 → 20 个神经元 → 输出
# 总参数量：20×1（W₁） + 20（b₁） + 1×20（W₂） + 1（b₂） = 61
def init_weights():
    W1 = np.random.randn(20, 1) * 0.5   # 输入到隐层权重（20×1）
    b1 = np.zeros(20)                    # 隐层偏置（20 维）
    W2 = np.random.randn(1, 20) * 0.5    # 隐层到输出权重（1×20）
    b2 = np.zeros(1)                     # 输出偏置（1 维）
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

def forward(params, x):
    W1 = params[:20].reshape(20, 1)
    b1 = params[20:40]
    W2 = params[40:60].reshape(1, 20)
    b2 = params[60:61]
    h = np.tanh(x.reshape(-1, 1) @ W1.T + b1)  # 隐层激活：(N, 20)
    return (h @ W2.T + b2).ravel()  # 输出：(N,)

def grad_log_posterior(params, x_batch, y_batch, N, sigma_y=0.1, sigma_w=1.0):
    # 使用有限差分法计算 log p(params | data) 的梯度
    # 实际应用中应使用自动微分；此处为清晰起见采用显式实现
    B = len(x_batch)
    pred = forward(params, x_batch)
    residual = y_batch - pred

    # 对数似然梯度（假设观测噪声服从高斯分布）
    eps = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        pred_plus = forward(params_plus, x_batch)
        dll_di = np.sum(residual * (pred_plus - pred) / eps) / sigma_y**2
        grad[i] = dll_di

    # 将小批量梯度缩放到全量数据尺度，并加入高斯先验项
    grad = (N / B) * grad - params / sigma_w**2
    return grad

# SGLD 后验采样
def sgld_bnn(x_train, y_train, n_steps=20000, batch_size=16, eta=1e-4):
    N = len(x_train)
    params = init_weights()
    posterior_samples = []

    for k in range(n_steps):
        idx = np.random.choice(N, batch_size, replace=False)
        grad = grad_log_posterior(params, x_train[idx], y_train[idx], N)
        noise = np.sqrt(eta) * np.random.randn(len(params))
        params = params + 0.5 * eta * grad + noise

        # 烧入期（burn-in）后开始收集样本，每 50 步保存一次
        if k > 10000 and k % 50 == 0:
            posterior_samples.append(params.copy())

    return np.array(posterior_samples)

# 执行采样并计算预测统计量
samples = sgld_bnn(x_train, y_train)
x_test = np.linspace(-3, 7, 200)
predictions = np.array([forward(s, x_test) for s in samples])

mean_pred = predictions.mean(axis=0)      # 预测均值
std_pred = predictions.std(axis=0)        # 预测标准差（即后验预测不确定性）
# std_pred 在区间 [1, 3] 内达到峰值 —— 这正是数据空缺区域
# 该不确定性属于认知不确定性（epistemic uncertainty）：模型清楚地知道自己“不知道什么”
```

该结果展现了贝叶斯推断的核心优势：**校准良好的不确定性估计**。在空缺区域 $x \in [1, 3]$ 中，后验预测标准差比数据密集区高出 3–5 倍。相比之下，使用标准 SGD 训练的点估计网络（point-estimate network）会自信地、但毫无依据地在空缺处进行插值，且完全无法提示用户其预测结果并不可靠。

这绝非仅具教学意义的玩具性质特性——它正是以下关键方向的理论基石：  
- **主动学习**（Active Learning）：优先查询不确定性最高的样本；  
- **安全强化学习**（Safe Reinforcement Learning）：规避认知不确定性过高的状态；  
- **模型选择**（Model Selection）：在验证集上更倾向选择预测置信带更紧凑的模型。

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

鲜少有人明确指出：**扩散模型其实就是 SVGD，只不过将核函数替换为了从数据中学到的 score 场**。SVGD 手动平衡“排斥 vs 吸引”，而扩散模型则从数据中学习这一平衡。正因如此，二者同属“密度上的梯度流”这一框架，而[第四节](#kl-散度是-wasserstein-梯度流)所述的 Wasserstein 几何正是描述它们的恰当语言。

PDE-ML 系列[第七章](/zh/pde-ml/07-扩散模型与score-matching/)将专门深入探讨扩散模型；此处仅旨在点明其与 Fokker-Planck 方程的深刻联系。

## 参考文献

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savaré. *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkhäuser, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.
