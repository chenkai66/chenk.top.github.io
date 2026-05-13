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
许多优化中的所谓“玄学”现象，其实都可以归结为三个核心概念：

- **梯度有多陡？** Lipschitz 光滑性（$L$-smoothness）限定了步长上限。
- **谷底有多尖？** $\mu$-强凸性决定了收敛速率，并保证极小值点唯一。
- **能否在不牺牲稳定性的情况下更快抵达？** Nesterov 加速配合自适应重启，将依赖条件数 $\kappa$ 的复杂度从 $\kappa$ 降至 $\sqrt{\kappa}$。
\n本文将这三个概念串成一条清晰的逻辑线：先用最少的不等式建立几何直觉，再证明关键定理，最后通过最小二乘实验对比 GD、Heavy Ball 和 Nesterov 的实际表现。目标不是堆砌公式，而是让你面对新问题时，能立刻判断：“该用多大步长？预期什么收敛速率？加速是否值得？”

## 你将学到

- Lipschitz 连续性的几何意义：函数图像始终被一个以任意点为顶点、斜率为 $\pm L$ 的双锥所包裹。
- $L$-光滑性的等价刻画：函数被一族开口向上的二次函数从上方包住（下降引理）。
- 强凸性作为二次**下界**，其自然推论是极小值点的存在性与唯一性。
- 为何条件数 $\kappa = L/\mu$ 控制 GD 的迭代复杂度，以及 Nesterov 如何将其替换为 $\sqrt{\kappa}$。
- 为何即使在强凸情形下仍需自适应重启，并通过最小二乘实验加以验证。

## 前置知识

- 多元微积分（梯度、Hessian、Taylor 展开）。
- 凸分析基础（凸集、一阶条件）。

---

## 1. Lipschitz 连续性与梯度光滑性

## 1.1 Lipschitz 连续性的几何图像

**定义（Lipschitz 连续）**：函数 $f:\mathbb{R}^n\to\mathbb{R}$ 是 $L$-Lipschitz 的，若存在 $L\ge 0$，使得对所有 $x, y$，

$$|f(y) - f(x)| \le L\,\|y - x\|.$$
\n几何上，这构成一个**双向锥约束**：以任意锚点 $(x_0, f(x_0))$ 为顶点、斜率为 $\pm L$ 的双锥必须完全包含函数图像。一旦函数在某处出现近乎垂直的切线（例如 $\sqrt{|x|}$ 在原点附近），就不存在有限的 $L$ 能满足该条件，函数也就不再是 Lipschitz 的。

![Lipschitz 锥包住整张函数图；含 vertical-tangent 的函数无法被任何有限 L 容纳](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/fig1_lipschitz_geometry.png)
\n左图：$\sin x$ 是 1-Lipschitz 的，三个样本锚点处的橙色锥体完整包裹了函数图像。右图：$\mathrm{sign}(x)\sqrt{|x|}$ 在原点处导数发散，红色区域显示当尝试用 $L=2$ 构造锥体时，函数图像会刺穿该锥体。

**两条反复出现的推论**：

1. **一致连续性**：取 $\delta = \varepsilon / L$ 即可。
2. **封闭性**：Lipschitz 函数在加法、数乘和复合下封闭（常数相乘），这使我们能将复杂模型分解为具有已知 Lipschitz 常数的基本模块。

## 1.2 梯度 Lipschitz = $L$-光滑性
\n实践中，我们更关心梯度 $\nabla f$ 的 Lipschitz 性，因为它直接限制步长：

$$\|\nabla f(y) - \nabla f(x)\| \le L\,\|y - x\|.$$
\n满足此条件的可微函数称为 **$L$-光滑**。它有一个更实用的等价表述——**下降引理（descent lemma）**：

$$\boxed{\,f(y) \le f(x) + \langle \nabla f(x), y - x\rangle + \frac{L}{2}\,\|y - x\|^2.\,}$$
\n即：**$f$ 被一族曲率为 $L$、开口向上的抛物线从上方包住**。切线只是局部最差情况下的下界，而抛物线才是全局有效的上界。

![L-smooth 函数被开口为 L 的二次抛物面从上方包住](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/fig2_l_smooth_quadratic.png)
\n蓝色曲线为 $f(x) = \tfrac{1}{2}\sin(2x) + \tfrac{1}{2}x^2$。在三个锚点处，虚线所示的二次函数 $f(x_0) + f'(x_0)(x-x_0) + \tfrac{L}{2}(x-x_0)^2$ 始终位于 $f$ 上方，而点线表示的切线仅在 $f$ 局部凸的区域才位于其下方。

**为何这个不等式如此重要？** 将 $y = x - \eta\nabla f(x)$ 代入得：

$$f(y) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|\nabla f(x)\|^2.$$
\n只要 $\eta \le 1/L$，括号内项 $\ge 1/2$，因此**每一步都严格下降**，且下降量至少为常数倍的 $\|\nabla f\|^2$。这正是“步长不超过 $1/L$”的真实来源——并非经验之谈，而是下降引理的直接推论。

![一维二次函数上 GD 在三种步长下的行为：eta=0.8/L 收敛、eta=1/L 一步到位、eta=2.2/L 发散](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig7_stepsize.png)
\n对同一二次函数 $f(x) = \tfrac{L}{2}x^2$（此处 $L = 4$）的三种情形。左：小步长（$\eta = 0.8/L$）单调下降；中：经典步长 $\eta = 1/L$ 从 $x_0 = 1$ 一步直达最优解——这是不过冲前提下的最快方式；右：一旦步长超过 $2/L$，迭代就会发散，下降引理失效。区间 $[1/L,\, 2/L]$ 正是收敛最快且仍安全的临界带；低于此区间浪费迭代，高于则发散。

## 1.3 三个典型例子

| 函数 | 梯度 | Hessian 谱范数 | $L$ |
|---|---|---|---|
| $\tfrac{1}{2}\|x\|^2$ | $x$ | $1$ | $1$ |
| $\tfrac{1}{2}\|Ax-b\|^2$ | $A^\top(Ax-b)$ | $\lambda_{\max}(A^\top A)$ | $\lambda_{\max}(A^\top A)$ |
| Logistic $\log(1 + e^{-y\,\theta^\top x})$（单样本） | $-y\,\sigma(-y\theta^\top x)\,x$ | $\sigma(\cdot)\sigma(-\cdot)\,xx^\top$ | $\tfrac{1}{4}\|x\|^2$ |
\n第三行给出了 logistic 回归的标准 $L = \tfrac{1}{4n}\sum_i\|x_i\|^2$——其中 $\sigma(\cdot)\sigma(-\cdot)\le 1/4$ 是关键。

## 1.4 Hessian 谱界蕴含梯度 Lipschitz

**定理 1（Hessian 谱范数 $\Rightarrow$ Lipschitz 梯度）**：若 $f$ 二阶可微且 $\sup_x \|\nabla^2 f(x)\|_2 \le L$，则 $\nabla f$ 是 $L$-Lipschitz 的。

**证明**：由 Newton-Leibniz 公式，

$$\nabla f(y) - \nabla f(x) = \int_0^1 \nabla^2 f(x + t(y-x))\,(y-x)\,\mathrm dt.$$
\n取 2-范数：

$$\|\nabla f(y) - \nabla f(x)\| \le \int_0^1 \|\nabla^2 f(\cdot)\|_2\,\|y-x\|\,\mathrm dt \le L\,\|y-x\|.\quad\blacksquare$$
\n对二次函数 $f(x) = \tfrac{1}{2}x^\top H x$，该结论给出精确的 $L = \lambda_{\max}(H)$——与下降引理中的 $L$ 一致，因此步长规则 $\eta \le 1/\lambda_{\max}(H)$ 是紧的。

---

## 2. 强凸性：存在性、唯一性与二次增长

## 2.1 三种等价定义

**定义（$\mu$-强凸）**：可微函数 $f$ 是 $\mu$-强凸的（$\mu>0$），若对所有 $x, y$，

$$f(y) \ge f(x) + \langle \nabla f(x), y - x\rangle + \frac{\mu}{2}\,\|y - x\|^2.$$
\n它有三种等价形式，各自适用于不同场景：

1. **二次下界**（上述不等式）：$f$ 位于一族曲率为 $\mu$、开口向上的抛物线上方。
2. **$f - \tfrac{\mu}{2}\|x\|^2$ 是凸函数**：减去“$\mu$-曲率质量”后剩余部分仍是凸的。
3. **二阶条件**（若 $f \in C^2$）：$\nabla^2 f(x) \succeq \mu I$。

![强凸函数被开口为 mu 的二次抛物面从下方托住；凸但非强凸的情形（mu=0）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/fig3_strong_convexity.png)
\n左图：一个强凸函数及其在三个锚点处的二次下界模型。右图：$f(x) = 0.05\,x^4$ 是凸的但**非强凸**——在原点附近过于平坦，无法被任何 $\mu>0$ 的抛物线从下方支撑。这正是此类目标无法获得线性收敛的原因。

## 2.2 存在性与唯一性

**定理 2（存在性）**：若 $f$ 下半连续，且某个子水平集 $\{x : f(x)\le\alpha\}$ 非空且有界，则 $f$ 在该集合上达到最小值。
\n这本质上是 Weierstrass 定理：闭且有界集是紧集，下半连续函数在紧集上必达下确界。

**定理 3（强凸性蕴含唯一性）**：$\mu$-强凸函数（$\mu > 0$）在 $\mathbb{R}^n$ 上至多有一个全局极小值点。

**证明**：设 $x^\star, y^\star$ 均为极小值点，则 $\nabla f(x^\star) = 0$。将 $x = x^\star, y = y^\star$ 代入下界不等式：

$$f(y^\star) \ge f(x^\star) + 0 + \frac{\mu}{2}\|y^\star - x^\star\|^2.$$
\n但 $f(y^\star) = f(x^\star)$，故 $\|y^\star - x^\star\| = 0$。$\blacksquare$

**推论（PL / 二次增长）**：令 $x = x^\star$，得

$$f(y) - f^\star \ge \frac{\mu}{2}\|y - x^\star\|^2,$$
\n即**函数值偏离最优值的程度至少随距离平方增长**。这一性质在后续所有收敛性证明中，都将“小函数值”转化为“接近最优解”。

## 2.3 同时满足 $L$-光滑与 $\mu$-强凸：条件数登场
\n若 $f$ 同时满足

$$\mu I \preceq \nabla^2 f(x) \preceq L I,$$
\n则所有方向的曲率被压缩在 $[\mu, L]$ 内。比值

$$\boxed{\,\kappa := \frac{L}{\mu} \ge 1\,}$$
\n称为**条件数**，它控制着所有后续收敛速率。大的 $\kappa$ 意味着函数“又长又窄”——最陡方向濒临不稳定，而最平缓方向几乎无信号。这正是优化困难的结构性根源。

---

## 3. 加速梯度下降：从 $\kappa$ 到 $\sqrt{\kappa}$

## 3.1 普通 GD 的两个上界

**定理 4（GD 在凸 + $L$-光滑下的次线性速率）**：取 $\eta = 1/L$，

$$f(x_t) - f^\star \le \frac{L\,\|x_0 - x^\star\|^2}{2t} = \mathcal O(1/t).$$

**定理 5（GD 在 $\mu$-强凸 + $L$-光滑下的线性速率）**：取 $\eta = 1/L$，

$$\|x_t - x^\star\|^2 \le \Big(1 - \frac{1}{\kappa}\Big)^t \|x_0 - x^\star\|^2.$$
\n达到误差 $\varepsilon$ 需 $t = \mathcal O(\kappa\log(1/\varepsilon))$。**条件数线性进入复杂度**：在 $\kappa = 10^4$ 的最小二乘问题中，精度每提升十倍约需额外 $2\times 10^4$ 次迭代。

## 3.2 Nesterov：用前瞻换取 $\sqrt{\kappa}$
\n经典 Polyak Heavy Ball 更新为

$$x_{t+1} = x_t - \alpha\nabla f(x_t) + \beta(x_t - x_{t-1}).$$
\n它在严格凸**二次型**上确实达到 $\sqrt{\kappa}$ 速率，但**无法保证在一般强凸函数上加速**——存在反例使其发散。Nesterov 的关键改进是**在前瞻点计算梯度**，而非在 $x_t$：

$$\begin{aligned}\ny_t &= x_t + \beta_t (x_t - x_{t-1}), \\\nx_{t+1} &= y_t - \eta\,\nabla f(y_t).
\end{aligned}$$
\n直觉是：先按动量外推到即将到达的位置 $y_t$，再用该点的梯度进行修正。这点前瞻性足以在所有 $L$-光滑凸函数上保持加速。

**定理 6（Nesterov，凸情形，$\mathcal O(1/t^2)$）**：对 $L$-光滑凸函数，取 $\eta = 1/L$ 及经典权重 $\beta_t = (t-1)/(t+2)$，

$$f(x_t) - f^\star \le \frac{2L\,\|x_0 - x^\star\|^2}{(t+1)^2}.$$

**定理 7（Nesterov，强凸情形，$\mathcal O((1 - 1/\sqrt{\kappa})^t)$）**：对 $\mu$-强凸 + $L$-光滑函数，取 $\eta = 1/L$ 及常数动量

$$\beta = \frac{1 - \sqrt{1/\kappa}}{1 + \sqrt{1/\kappa}},$$
\n则 $f(x_t) - f^\star \le \big(1 - \sqrt{1/\kappa}\big)^t (f(x_0) - f^\star)$，故 $t = \mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$。

![旋转后的病态二次型上的迭代轨迹：GD 沿陡方向来回震荡，Nesterov 沿低谷顺势滑行](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/optimization-theory/02-smoothness-strong-convexity-nesterov/fig6_trajectories.png)
\n左图：在 $\kappa = 30$ 的旋转二次型上运行 80 次迭代。GD（蓝色，为放大效果取 $\eta = 1.9/L$）在陡峭方向反复过冲，在狭窄山谷中曲折前进，沿平坦轴进展缓慢；Nesterov（紫色，$\eta = 1/L$）则沿山谷积累动量，几乎不激发陡峭模式。右图：到 $x^\star$ 距离的对数图——斜率差异正是定理 5 与 7 所预测的 $\kappa$ 与 $\sqrt{\kappa}$ 速率差距的几何体现。

![GD vs Heavy Ball vs Nesterov 在强凸二次型上的收敛曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/fig4_convergence_rates.png)
\n左图（$\kappa=100$）：在对数坐标下，三种算法均呈线性下降，但 Nesterov（紫色）和 Heavy Ball（橙色）的斜率明显陡于 GD（蓝色）——这正是 $\sqrt{\kappa}$ 与 $\kappa$ 的差距。虚线为理论包络，与实测轨迹吻合。右图：放大前 80 次迭代，可见 Nesterov **非单调**——它在理论下界内震荡，这是加速的代价。

## 3.3 自适应重启解决加速的副作用
\n加速的主要缺点是**非单调性**——函数值会周期性上升。这在两种情况下尤为棘手：

- **$\mu$ 未知**：定理 7 需要 $\sqrt{1/\kappa}$ 来设定动量；估计错误会破坏速率。
- **局部强凸**：函数全局仅凸，但在 $x^\star$ 附近强凸；此时使用凸情形的 Nesterov 会表现不佳。

**自适应重启（O'Donoghue & Candès, 2015）**：每当**梯度方向反转**（$\langle\nabla f(y_t), x_{t+1}-x_t\rangle > 0$）或**函数值上升**时，重置动量并将迭代计数器归零（$t \leftarrow 1$）。

**定理 8（重启 Nesterov 在未知 $\mu$ 下仍达最优速率）**：对 $\mu$-强凸 + $L$-光滑函数，重启 Nesterov 仍实现 $\mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$ 迭代复杂度，**无需预先知道 $\mu$**。
\n证明概要：两次重启之间运行凸情形 Nesterov，误差以 $\mathcal O(1/k^2)$ 下降。结合二次增长推论，可得“每次重启至少将误差减半”。经 $\log(1/\varepsilon)$ 次几何减半后达到 $\varepsilon$。每段长度约 $\sim \sqrt{\kappa}$，总迭代数约为 $\sim \sqrt{\kappa}\log(1/\varepsilon)$。

## 3.4 条件数究竟影响多大

![条件数 kappa 对迭代次数的影响：GD 线性增长，Nesterov 平方根增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法/fig5_condition_number.png)
\n左图（双对数坐标）：在合成最小二乘问题上，GD 达到 $10^{-6}$ 相对误差所需的迭代数严格沿 $\kappa$ 增长，而 Nesterov 沿 $\sqrt{\kappa}$ 增长。右图：加速比 $T_{\text{GD}}/T_{\text{AGD}}$ 几乎精确跟踪 $\sqrt{\kappa}$。当 $\kappa = 10^4$ 时加速约 100 倍；$\kappa = 100$ 时约 10 倍。**条件数越差，加速收益越大**。

## 3.5 决策参考表

| 场景 | 推荐 | 理由 |
|---|---|---|
| 小 $\kappa$（$\le 50$），追求代码简洁 | GD | 加速带来的复杂度收益有限 |
| 大 $\kappa$，且 $\mu$ 已知 | 定理 7 的常数动量 Nesterov | 有闭式最优 $\beta$ |
| 大 $\kappa$，$\mu$ 未知或仅局部强凸 | 自适应重启 Nesterov | 自适应 $\mu$ 并保持最优速率 |
| 严格凸二次型（如最小二乘） | 优先使用共轭梯度（CG） | 在 $\le n$ 步内有限终止，优于 Nesterov |
| 非凸但局部近凸（如深度网络） | 带预热和余弦衰减的 Momentum / Adam | 理论速率不可证，工程经验主导 |

---

## 4. 最小二乘实验

## 4.1 问题与精确常数
\n考虑

$$\min_{x\in\mathbb{R}^n} f(x) = \frac{1}{2}\|Ax - b\|^2, \qquad A\in\mathbb{R}^{m\times n},\; b\in\mathbb{R}^m.$$
\n求导得：

$$\nabla f(x) = A^\top(Ax - b), \qquad \nabla^2 f(x) = A^\top A,$$
\n因此

$$L = \lambda_{\max}(A^\top A), \qquad \mu = \lambda_{\min}(A^\top A), \qquad \kappa = \kappa(A^\top A) = \kappa(A)^2.$$

**关键提醒**：条件数是矩阵 $A$ 条件数的**平方**。因此一个看似正常的 $A$（$\kappa(A) = 100$）在最小二乘中会变成 $\kappa = 10^4$ 的困难实例——尤其对正规方程的梯度而言。

## 4.2 实现细节

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
    """Nesterov AGD using known mu, L (constant momentum form)."""
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

## 4.3 观察结果
\n在 $m = 200, n = 100$、$\kappa(A) \approx 100$（即 $\kappa(A^\top A) \approx 10^4$）的合成实例上：

- **GD**：需约 $4\times 10^4$ 次迭代才能达到 $10^{-6}$ 相对梯度范数；过程平滑但缓慢。
- **Nesterov（常数动量）**：约 400 次迭代；轨迹如图 4（右）所示，在谷底震荡。
- **重启 Nesterov**：约 500 次迭代，且几乎完全单调——三者中最稳健。
\n实测加速比与图 5 的 $\sqrt{\kappa}$ 预测一致（$10^4 \to 100\times$）。

---

## 5. 总结与后续方向

## 5.1 速查表

| 假设 | 算法 | 速率 | 步长 |
|---|---|---|---|
| $L$-光滑，凸 | GD | $\mathcal O(1/t)$ | $\eta = 1/L$ |
| $L$-光滑，$\mu$-强凸 | GD | $\big(1 - 1/\kappa\big)^t$ | $\eta = 1/L$ |
| $L$-光滑，凸 | Nesterov | $\mathcal O(1/t^2)$ | $\eta = 1/L,\ \beta_t = (t-1)/(t+2)$ |
| $L$-光滑，$\mu$-强凸 | Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | $\eta = 1/L,\ \beta = (1-\sqrt{1/\kappa})/(1+\sqrt{1/\kappa})$ |
| 同上，$\mu$ 未知 | 重启 Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | 自适应 |

## 5.2 面对新问题的三个直觉
\n当一个新目标函数摆在面前时，按此流程思考：

1. **梯度有多陡？** 估计 $L$（最大特征值或回溯线搜索），它直接给出步长上限。
2. **谷底有多尖？** 估计 $\mu$（最小特征值或 PL 常数），它决定能否实现线性收敛。
3. **加速是否值得？** 看 $\kappa = L/\mu$。若 $\kappa < 50$，GD 足够；若 $\kappa \in [50, 10^4]$，切换至 Nesterov；若 $\kappa > 10^4$，考虑预条件或二阶方法。

## 5.3 后续方向

- **非凸 + PL 条件**：放弃强凸性；若 $\tfrac{1}{2}\|\nabla f\|^2 \ge \mu(f - f^\star)$ 成立，GD 仍线性收敛。这是“过参数化深度网络为何线性收敛”的理论起点（Karimi et al., 2016）。
- **含噪声的加速**：标准 Nesterov 对随机梯度不鲁棒。SAG / SVRG / Katyusha 等方差缩减方法通过降低噪声，将强凸随机优化的速率重新拉回 $\sqrt{\kappa}$ 阶。
- **二阶加速**：Sophia 和 Shampoo 使用（块）对角 Hessian 进行预条件，实质上是重写条件数——这是 2024 年大规模预训练的前沿方向。

**参考文献**

1. Nesterov, Y. (1983). *A method of solving a convex programming problem with convergence rate $\mathcal O(1/k^2)$.* Soviet Mathematics Doklady, 27(2), 372–376.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
3. Bubeck, S. (2015). *Convex Optimization: Algorithms and Complexity.* Foundations and Trends in Machine Learning, 8(3-4), 231–357.
4. O'Donoghue, B., & Candès, E. (2015). *Adaptive restart for accelerated gradient schemes.* Foundations of Computational Mathematics, 15(3), 715–732.
5. Karimi, H., Nutini, J., & Schmidt, M. (2016). *Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition.* ECML-PKDD.
