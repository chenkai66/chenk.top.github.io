---
title: '优化理论（二）：光滑性、强凸性与 Nesterov 加速'
date: 2022-09-15 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
lang: zh
mathjax: true
description: "用三个核心概念理清优化直觉：Lipschitz 光滑性决定步长上限，强凸性决定收敛速度与解的唯一性，Nesterov 加速在不牺牲稳定性的前提下加速到达。含关键定理证明与最小二乘实验对比。"
disableNunjucks: true
translationKey: "optim-02"
series: optimization-theory
series_order: 2
aliases:
  - /zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/
---
大量关于优化器的“民间智慧”其实可以归结为三个核心概念：

- **梯度有多陡？** Lipschitz 光滑性（$L$-smoothness）限制了步长上限。
- **底部有多尖锐？** $\mu$-强凸性决定了收敛速率，并保证最小值点唯一。
- **能否在不牺牲稳定性的情况下更快到达？** Nesterov 加速和自适应重启将每条件数的代价从 $\kappa$ 降至 $\sqrt{\kappa}$。

本文将这三个概念串成一条主线：用最少的不等式建立几何直觉，证明关键定理，最后通过一个最小二乘实验，让 GD、Heavy Ball 和 Nesterov 正面交锋。目标不是堆砌公式——而是让你面对新问题时，能立刻回答：“该用多大步长？收敛速率是多少？加速是否值得？”


---

## 你将学到什么

- Lipschitz 连续性的几何意义：每个点都位于一个包含函数图像的斜率锥内。
- $L$-光滑性的等价刻画：函数被一族二次函数从上方夹住（下降引理）。
- 强凸性作为二次 *下界*，并自然导出最小值点的存在性与唯一性。
- 为何条件数 $\kappa = L/\mu$ 控制 GD 的迭代复杂度，以及 Nesterov 如何将其替换为 $\sqrt{\kappa}$。
- 为何即使在强凸情形下仍需自适应重启，并通过最小二乘实验加以验证。

## 前置知识

- 多元微积分（梯度、Hessian、泰勒展开）。
- 基础凸分析（凸集、一阶条件）。

---

## Lipschitz 连续性与梯度光滑性

### Lipschitz 连续性的几何图像

**定义（Lipschitz 连续）**。函数 $f:\mathbb{R}^n\to\mathbb{R}$ 是 $L$-Lipschitz 的，若存在 $L\ge 0$，使得对所有 $x, y$，
$$|f(y) - f(x)| \le L\,\|y - x\|.$$
从几何上看，这是一个 **双侧锥约束**：在任意锚点 $(x_0, f(x_0))$ 处，斜率为 $\pm L$ 的双锥必须完全包含整个函数图像。一旦函数出现近似垂直的切线（例如 $\sqrt{|x|}$ 在原点附近），就不存在有限的 $L$ 能将其包含，此时 $f$ 不再是 Lipschitz 的。

![Lipschitz cone contains the graph; functions with vertical tangents cannot fit inside any finite-L cone](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lipschitz-continuity-strong-convexity-nesterov/fig1_lipschitz_geometry.png)

左图：$\sin x$ 是 1-Lipschitz；三个采样锚点处的橙色锥体完全包含图像。右图：$\mathrm{sign}(x)\sqrt{|x|}$ 在原点处斜率发散；红色区域表示候选 $L=2$ 的锥体被穿透的位置。

**两个会反复出现的推论：**

1. **一致连续性**：取 $\delta = \varepsilon / L$。
2. **封闭性**：Lipschitz 函数在加法、标量乘法和复合下封闭（常数相乘）。这使我们能将复杂模型分解为具有已知常数的 Lipschitz 构件。

### 梯度-Lipschitz = $L$-光滑性

实践中，我们更关心 $\nabla f$ 的 Lipschitz 性质，而非 $f$ 本身，因为它直接限制步长：
$$
\|
\nabla f(y) - 
\nabla f(x)\| \le L\,\|y - x\|.
$$
满足此条件的可微函数称为 **$L$-光滑**。它有一个更实用的等价表述，即 **下降引理**：
$$
\boxed{\,f(y) \le f(x) + \langle
\nabla f(x), y - x\rangle + \frac{L}{2}\,\|y - x\|^2.\,}
$$
也就是说：**$f$ 被一族曲率为 $L$ 的向上抛物线从上方夹住**。切线是最坏情况下的下包络；抛物线则是实际的上包络。

![An L-smooth function is bounded above by a family of quadratic upper models](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lipschitz-continuity-strong-convexity-nesterov/fig2_l_smooth_quadratic.png)

蓝色曲线为 $f(x) = \tfrac{1}{2}\sin(2x) + \tfrac{1}{2}x^2$。在三个锚点处，虚线二次函数 $f(x_0) + f'(x_0)(x-x_0) + \tfrac{L}{2}(x-x_0)^2$ 全局位于 $f$ 上方，而点线切线仅在 $f$ 局部凸的区域下方。

**为何这个不等式如此重要？** 将 $y = x - \eta\nabla f(x)$ 代入：
$$
f(y) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|
\nabla f(x)\|^2.
$$
只要 $\eta \le 1/L$，括号内 $\ge 1/2$，因此 **每一步都严格减小 $f$**，且减小量至少为常数乘以 $\|\nabla f\|^2$。这才是“步长至多为 $1/L$”的真正来源——并非传言，而是下降引理的直接推论。

![GD on a 1D quadratic at three step sizes: contracting at eta=0.8/L, exact at eta=1/L, divergent at eta=2.2/L](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig7_stepsize.png)

同一二次函数 $f(x) = \tfrac{L}{2}x^2$（此处 $L = 4$）上的三种情形。左：小步长（$\eta = 0.8/L$）单调缓慢下降。中：标准步长 $\eta = 1/L$ 从 $x_0 = 1$ 一步直达最优解——无超调下的最快方式。右：超过 $2/L$ 上限后，迭代发散，下降引理失效。区间 $[1/L,\, 2/L]$ 正是收敛最快 *且* 安全的紧致带；低于此则浪费迭代，高于此则发散。

### 三个实例详解

| Function | Gradient | Spectral norm of Hessian | $L$ |
|---|---|---|---|
| $\tfrac{1}{2}\|x\|^2$ | $x$ | $1$ | $1$ |
| $\tfrac{1}{2}\|Ax-b\|^2$ | $A^\top(Ax-b)$ | $\lambda_{\max}(A^\top A)$ | $\lambda_{\max}(A^\top A)$ |
| Logistic $\log(1 + e^{-y\,\theta^\top x})$ (one sample) | $-y\,\sigma(-y\theta^\top x)\,x$ | $\sigma(\cdot)\sigma(-\cdot)\,xx^\top$ | $\tfrac{1}{4}\|x\|^2$ |

第三行给出了逻辑回归的标准 $L = \tfrac{1}{4n}\sum_i\|x_i\|^2$ —— 关键在于 $\sigma(\cdot)\sigma(-\cdot)\le 1/4$。

### Hessian 谱范数蕴含梯度 Lipschitz

**定理 1（Hessian 谱范数 $\Rightarrow$ Lipschitz 梯度）**。若 $f$ 二阶可微且 $\sup_x \|\nabla^2 f(x)\|_2 \le L$，则 $\nabla f$ 是 $L$-Lipschitz 的。

**证明**。由 Newton-Leibniz 公式，
$$
\nabla f(y) - 
\nabla f(x) = \int_0^1 
\nabla^2 f(x + t(y-x))\,(y-x)\,\mathrm dt.
$$
取 2-范数：
$$
\|
\nabla f(y) - 
\nabla f(x)\| \le \int_0^1 \|
\nabla^2 f(\cdot)\|_2\,\|y-x\|\,\mathrm dt \le L\,\|y-x\|.\quad\blacksquare
$$
对于二次函数 $f(x) = \tfrac{1}{2}x^\top H x$，这恰好给出 $L = \lambda_{\max}(H)$ —— 与下降引理使用的 $L$ 相同，因此步长规则 $\eta \le 1/\lambda_{\max}(H)$ 是紧致的。

---

## 强凸性：存在性、唯一性与二次增长

### 三种等价定义

**定义（$\mu$-强凸）**。可微函数 $f$ 是 $\mu$-强凸的（$\mu>0$），若对所有 $x, y$，
$$
f(y) \ge f(x) + \langle
\nabla f(x), y - x\rangle + \frac{\mu}{2}\,\|y - x\|^2.
$$
它有三种等价形式，各自适用于不同场景：

1. **二次下界**（上述不等式）：$f$ 位于一族曲率为 $\mu$ 的向上抛物线上方。
2. **$f - \tfrac{\mu}{2}\|x\|^2$ 是凸的**：减去“$\mu$-曲率质量”后剩下凸函数。
3. **二阶条件**（若 $f$ 是 $C^2$）：$\nabla^2 f(x) \succeq \mu I$。

![A strongly convex function is bounded BELOW by a family of mu-quadratics; convex but not strongly convex (mu = 0) for contrast](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lipschitz-continuity-strong-convexity-nesterov/fig3_strong_convexity.png)

左：一个强凸函数及其三个二次下界模型（每个锚点一个）。右：$f(x) = 0.05\,x^4$ 是凸的但 **非强凸** —— 在原点过于平坦，无法容纳任何 $\mu>0$ 的抛物线下界。这正是此类目标无法获得线性收敛的原因。

### 存在性与唯一性

**定理 2（存在性）**。若 $f$ 下半连续，且某个次水平集 $\{x : f(x)\le\alpha\}$ 非空且有界，则 $f$ 在该集合上取得最小值。

这只是 Weierstrass 定理：闭且有界 = 紧致，下半连续函数在紧致集上取得下确界。

**定理 3（强凸性蕴含唯一性）**。$\mathbb{R}^n$ 上的 $\mu$-强凸函数（$\mu > 0$）至多有一个全局最小值点。

**证明**。设 $x^\star, y^\star$ 均为最小值点，则 $\nabla f(x^\star) = 0$，代入下界不等式（$x = x^\star, y = y^\star$）得
$$f(y^\star) \ge f(x^\star) + 0 + \frac{\mu}{2}\|y^\star - x^\star\|^2.$$
但 $f(y^\star) = f(x^\star)$，故 $\|y^\star - x^\star\| = 0$。$\blacksquare$

**推论（PL / 二次增长）**。在下界中令 $x = x^\star$，
$$f(y) - f^\star \ge \frac{\mu}{2}\|y - x^\star\|^2,$$
即 **代价至少以二次速率远离最小值点增长**。这是所有后续收敛证明中将“小函数值”转化为“小到最优点距离”的关键杠杆。

### $L$-光滑与 $\mu$-强凸结合：条件数

若 $f$ 同时满足
$$
\mu I \preceq
\nabla^2 f(x) \preceq L I,
$$
则所有方向曲率被压缩至 $[\mu, L]$。其比值
$$\boxed{\,\kappa := \frac{L}{\mu} \ge 1\,}$$
称为 **条件数**，它控制后续所有收敛速率。大的 $\kappa$ 意味着函数“又长又窄”——最陡方向濒临不稳定，而最平缓方向几乎不提供信号。这正是优化困难的结构性根源。

---

## 加速梯度下降：从 $\kappa$ 到 $\sqrt{\kappa}$

### 普通 GD 的两个上界

**定理 4（GD 在凸 + $L$-光滑情形，次线性速率）**。取 $\eta = 1/L$：
$$f(x_t) - f^\star \le \frac{L\,\|x_0 - x^\star\|^2}{2t} = \mathcal O(1/t).$$
**定理 5（GD 在 $\mu$-强凸 + $L$-光滑情形，线性速率）**。取 $\eta = 1/L$：
$$\|x_t - x^\star\|^2 \le \Big(1 - \frac{1}{\kappa}\Big)^t \|x_0 - x^\star\|^2.$$
达到误差 $\varepsilon$ 需要 $t = \mathcal O(\kappa\log(1/\varepsilon))$。**条件数线性进入**：在 $\kappa = 10^4$ 的最小二乘问题中，精度每提高 10 倍，大约额外需要 $2\times 10^4$ 次迭代。

### Nesterov：前瞻将 $\kappa$ 替换为 $\sqrt{\kappa}$

经典 Polyak Heavy Ball 形式为
$$
x_{t+1} = x_t - \alpha
\nabla f(x_t) + \beta(x_t - x_{t-1}).
$$
它在严格凸 *二次函数* 上确实能达到 $\sqrt{\kappa}$ 速率，但 **无法在一般强凸函数上保证加速** —— 反例会导致发散。Nesterov 的关键改进是 **在前瞻点处计算梯度**，而非在 $x_t$ 处：
$$
\begin{aligned}
y_t &= x_t + \beta_t (x_t - x_{t-1}), \\
x_{t+1} &= y_t - \eta\,
\nabla f(y_t).
\end{aligned}
$$
直觉：先用动量外推预测落点，再用该点梯度校正。这点预见性足以在所有 $L$-光滑凸函数上保持加速。

**定理 6（Nesterov，凸情形，$\mathcal O(1/t^2)$）**。对 $L$-光滑凸函数 $f$，取 $\eta = 1/L$ 和经典权重 $\beta_t = (t-1)/(t+2)$，
$$f(x_t) - f^\star \le \frac{2L\,\|x_0 - x^\star\|^2}{(t+1)^2}.$$
**定理 7（Nesterov，强凸情形，$\mathcal O((1 - 1/\sqrt{\kappa})^t)$）**。对 $\mu$-强凸 + $L$-光滑函数 $f$，取 $\eta = 1/L$ 和恒定动量
$$\beta = \frac{1 - \sqrt{1/\kappa}}{1 + \sqrt{1/\kappa}},$$
可得 $f(x_t) - f^\star \le \big(1 - \sqrt{1/\kappa}\big)^t (f(x_0) - f^\star)$，故 $t = \mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$。

![Trajectories on a rotated ill-conditioned quadratic: GD zigzags across the steep direction while Nesterov rolls smoothly along the valley](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig6_trajectories.png)

左：在 $\kappa = 30$ 的旋转二次函数上运行 80 次迭代。GD（蓝色，$\eta = 1.9/L$ 以放大效果）在陡峭方向超调，在狭窄山谷中来回振荡，沿平坦轴净进展缓慢。Nesterov（紫色，$\eta = 1/L$）沿山谷积累动量，几乎不激发陡峭模式。右：到 $x^\star$ 距离的对数尺度图 —— 斜率差异正是定理 5 和 7 所预测的 $\kappa$ 与 $\sqrt{\kappa}$ 速率差距的几何体现。

![GD vs Heavy Ball vs Nesterov on a strongly convex quadratic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lipschitz-continuity-strong-convexity-nesterov/fig4_convergence_rates.png)

左（$\kappa=100$）：对数坐标下三种算法均线性下降，但 Nesterov（紫色）和 Heavy Ball（橙色）斜率比 GD（蓝色）更陡 —— 这正是 $\sqrt{\kappa}$ 与 $\kappa$ 差距的可视化。虚线为理论包络；与实证轨迹吻合。右：聚焦前 80 次迭代，Nesterov **非单调** —— 在下包络内振荡。这是加速的代价。

### 自适应重启修复加速的副作用

加速的缺点是 **非单调性** —— 函数值周期性上升。这在两种情况下尤为痛苦：

- **$\mu$ 未知**：定理 7 需要 $\sqrt{1/\kappa}$ 作为动量；估计错误会破坏速率。
- **局部强凸**：$f$ 全局仅凸但在 $x^\star$ 附近强凸；使用凸速率的 Nesterov 表现不佳。

**自适应重启（O'Donoghue & Candès, 2015）**：每当 **梯度方向反转**（$\langle\nabla f(y_t), x_{t+1}-x_t\rangle > 0$）或 **函数值上升** 时，重置动量并将迭代计数器归零 $t \leftarrow 1$。

**定理 8（重启 Nesterov 在未知 $\mu$ 下仍达最优速率）**。对 $\mu$-强凸 + $L$-光滑函数 $f$，重启 Nesterov 仍实现 $\mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$ 次迭代，**无需知道 $\mu$**。

证明概要：两次重启之间运行凸速率 Nesterov，间隙以 $\mathcal O(1/k^2)$ 下降。结合二次增长推论，可得“每次重启至少将间隙减半”。几何减半 $\log(1/\varepsilon)$ 次即可达到 $\varepsilon$。每段长度约 $\sim \sqrt{\kappa}$，故总迭代次数为 $\sim \sqrt{\kappa}\log(1/\varepsilon)$。

### $\kappa$ 实际影响有多大

![Condition number kappa controls iterations: GD grows linearly, Nesterov as sqrt(kappa)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lipschitz-continuity-strong-convexity-nesterov/fig5_condition_number.png)

左（双对数坐标）：在合成最小二乘问题上，达到 $10^{-6}$ 相对间隙所需的迭代次数，GD 严格随 $\kappa$ 增长，Nesterov 则随 $\sqrt{\kappa}$ 增长。右：加速比 $T_{\text{GD}}/T_{\text{AGD}}$ 几乎精确跟踪 $\sqrt{\kappa}$。当 $\kappa = 10^4$ 时加速比约 100 倍；$\kappa = 100$ 时约 10 倍。**条件越差，加速收益越大。**

### 决策表

| Situation | Recommendation | Why |
|---|---|---|
| Small $\kappa$ ($\le 50$), keep code simple | GD | 加速的复杂性收益很小 |
| Large $\kappa$, $\mu$ known | Constant-momentum Nesterov (Thm 7) | 有闭式最优 $\beta$ |
| Large $\kappa$, $\mu$ unknown / locally s.c. | Adaptive-restart Nesterov | 自适应 $\mu$ 并保持最优速率 |
| Strictly convex quadratic (least squares) | Conjugate Gradient first | 最多 $n$ 步有限终止，优于 Nesterov |
| Non-convex but locally near-convex (deep nets) | Momentum / Adam with warmup + cosine | 理论速率不可证；工程经验主导 |

---

## 最小二乘实验

### 问题与精确常数

考虑
$$\min_{x\in\mathbb{R}^n} f(x) = \frac{1}{2}\|Ax - b\|^2, \qquad A\in\mathbb{R}^{m\times n},\; b\in\mathbb{R}^m.$$
求导：
$$
\nabla f(x) = A^\top(Ax - b), \qquad 
\nabla^2 f(x) = A^\top A,
$$
故
$$L = \lambda_{\max}(A^\top A), \qquad \mu = \lambda_{\min}(A^\top A), \qquad \kappa = \kappa(A^\top A) = \kappa(A)^2.$$
**关键提醒**。条件数是 $A$ 条件数的 *平方*。因此一个看似正常的 $A$（$\kappa(A) = 100$）会变成最小二乘中的困难实例（$\kappa = 10^4$）—— 对正规方程梯度而言尤其如此。

### 实现

```python
import numpy as np

def gd(A, b, n_iter, eta=None):
    """Gradient descent with step 1/L."""
    L = np.linalg.eigvalsh(A.T @ A).max()
    eta = eta or 1.0 / L
    x = np.zeros(A.shape[1])
    hist = []
    for _ in range(n_iter):
        g = A.T @ (A @ x - b)
        hist.append(np.linalg.norm(g))
        x -= eta * g
    return x, hist

def nesterov_strongconvex(A, b, n_iter):
    """Nesterov AGD using known mu, L (constant-momentum form)."""
    eigs = np.linalg.eigvalsh(A.T @ A)
    L, mu = eigs.max(), max(eigs.min(), 1e-12)
    eta = 1.0 / L
    sk = np.sqrt(mu / L)
    beta = (1 - sk) / (1 + sk)
    x_prev = x = np.zeros(A.shape[1])
    hist = []
    for _ in range(n_iter):
        y = x + beta * (x - x_prev)
        g = A.T @ (A @ y - b)
        hist.append(np.linalg.norm(A.T @ (A @ x - b)))
        x_prev, x = x, y - eta * g
    return x, hist

def nesterov_restart(A, b, n_iter):
    """Adaptive-restart Nesterov: reset momentum on function-value uptick."""
    L = np.linalg.eigvalsh(A.T @ A).max()
    eta = 1.0 / L
    x_prev = x = np.zeros(A.shape[1])
    t = 1
    f_prev = 0.5 * np.linalg.norm(A @ x - b) ** 2
    hist = []
    for _ in range(n_iter):
        beta = (t - 1) / (t + 2)
        y = x + beta * (x - x_prev)
        g = A.T @ (A @ y - b)
        x_new = y - eta * g
        f_new = 0.5 * np.linalg.norm(A @ x_new - b) ** 2
        if f_new > f_prev:               # restart trigger
            t = 1
            x_new = x - eta * (A.T @ (A @ x - b))
            f_new = 0.5 * np.linalg.norm(A @ x_new - b) ** 2
        else:
            t += 1
        hist.append(np.linalg.norm(A.T @ (A @ x_new - b)))
        x_prev, x, f_prev = x, x_new, f_new
    return x, hist
```

### 观察结果

在合成实例上（$m = 200, n = 100$，$\kappa(A) \approx 100$，故 $\kappa(A^\top A) \approx 10^4$）：

- **GD**：需约 $4\times 10^4$ 次迭代达到 $10^{-6}$ 相对梯度范数；平滑但缓慢。
- **Nesterov（恒定动量）**：约 400 次迭代；轨迹如图 4（右）所示在谷底振荡。
- **重启 Nesterov**：约 500 次迭代且几乎完全单调 —— 三者中最稳健。

实证加速比与图 5 的 $\sqrt{\kappa}$ 预测一致（$10^4 \to 100\times$）。

---

## 总结与后续方向

### 速查表

| Assumption | Algorithm | Rate | Step size |
|---|---|---|---|
| $L$-smooth, convex | GD | $\mathcal O(1/t)$ | $\eta = 1/L$ |
| $L$-smooth, $\mu$-strongly convex | GD | $\big(1 - 1/\kappa\big)^t$ | $\eta = 1/L$ |
| $L$-smooth, convex | Nesterov | $\mathcal O(1/t^2)$ | $\eta = 1/L,\ \beta_t = (t-1)/(t+2)$ |
| $L$-smooth, $\mu$-strongly convex | Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | $\eta = 1/L,\ \beta = (1-\sqrt{1/\kappa})/(1+\sqrt{1/\kappa})$ |
| Same, $\mu$ unknown | Restart Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | adaptive |

### 面对新问题的三个反射

当一个新目标函数摆在面前时，走以下循环：

1. **有多陡？** 估计 $L$（最大特征值或回溯线搜索）。它立即给出步长上限。
2. **底部有多尖锐？** 估计 $\mu$（最小特征值或 PL 常数）。它告诉你是否可能线性收敛。
3. **加速是否值得？** 看 $\kappa = L/\mu$。$\kappa < 50$：GD 足够。$\kappa \in [50, 10^4]$：切换到 Nesterov。
$\kappa > 10^4$：考虑预条件或二阶方法。

### 故事的延续

- **非凸 + PL 条件**。放弃强凸性；若 $\tfrac{1}{2}\|\nabla f\|^2 \ge \mu(f - f^\star)$ 成立，GD 仍线性收敛。这是“为何过参数化深度网络能线性训练”的理论种子（Karimi et al., 2016）。
- **含噪声的加速**。原始 Nesterov 对随机梯度不鲁棒。SAG / SVRG / Katyusha 通过方差缩减将强凸随机速率拉回 $\sqrt{\kappa}$ 区间。
- **二阶加速**。Sophia 和 Shampoo 通过（块）对角 Hessian 预条件，有效重写条件数 —— 这是 2024 年大规模预训练的活跃前沿。

**References**

1. Nesterov, Y. (1983). *A method of solving a convex programming problem with convergence rate $\mathcal O(1/k^2)$.* Soviet Mathematics Doklady, 27(2), 372–376.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
3. Bubeck, S. (2015). *Convex Optimization: Algorithms and Complexity.* Foundations and Trends in Machine Learning, 8(3-4), 231–357.
4. O'Donoghue, B., & Candès, E. (2015). *Adaptive restart for accelerated gradient schemes.* Foundations of Computational Mathematics, 15(3), 715–732.
5. Karimi, H., Nutini, J., & Schmidt, M. (2016). *Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition.* ECML-PKDD.