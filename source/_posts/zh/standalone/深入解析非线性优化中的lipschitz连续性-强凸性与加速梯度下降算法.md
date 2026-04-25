---
title: "Lipschitz 连续性、强凸性与加速梯度下降"
date: 2023-02-11 09:00:00
tags:
  - ML
  - Optimization
  - Convex Analysis
categories: Algorithm
lang: zh-CN
mathjax: true
description: "用三个核心概念理清优化直觉：Lipschitz 光滑性决定步长上限，强凸性决定收敛速度与解的唯一性，Nesterov 加速在不牺牲稳定性的前提下加速到达。含关键定理证明与最小二乘实验对比。"
disableNunjucks: true
---

很多优化"玄学"其实都能被三个概念讲清楚：

- **梯度有多陡** —— Lipschitz 光滑性（$L$-smoothness）决定了步长的上限。
- **谷底有多硬** —— 强凸性（$\mu$-strong convexity）决定了收敛能有多快、解是否唯一。
- **能不能更快到达谷底** —— Nesterov 加速与重启策略，在不牺牲稳定性的前提下把每代价 $\kappa$ 的对数收敛压成 $\sqrt{\kappa}$。

本文把它们放在同一条逻辑链上：先用最小必要的定义和不等式把直觉钉牢，再给出关键定理与证明，最后用最小二乘实验对比 GD、Heavy Ball 与 Nesterov 的收敛行为。目标不是堆公式，而是让你在面对一个新问题时，能用这三件事快速判断"该用多大步长、预期什么收敛速度、加速是否值得"。

## 你将学到

- Lipschitz 连续性的几何含义：每一点周围都有一个"斜率锥"约束着函数图像。
- $L$-smoothness 的等价刻画：函数被一族二次曲面从上方包住（descent lemma）。
- 强凸性的等价刻画：函数被一族二次曲面从下方托住，唯一极小存在性自然出现。
- 条件数 $\kappa = L/\mu$ 如何决定 GD 的迭代复杂度，以及为什么 Nesterov 把它换成了 $\sqrt{\kappa}$。
- 重启策略为何对强凸函数仍然必要，以及在最小二乘问题上的实测对比。

## 前置知识

- 多元微积分（梯度、Hessian、Taylor 展开）。
- 凸函数的基本定义与一阶条件。

---

# 1. Lipschitz 连续性与梯度光滑性

## 1.1 Lipschitz 连续性的几何直觉

**定义（Lipschitz 连续）**：函数 $f:\mathbb{R}^n\to\mathbb{R}$ 是 $L$-Lipschitz 连续的，如果存在常数 $L\ge 0$，使得对任意 $x, y$，

$$
|f(y) - f(x)| \le L\,\|y - x\|.
$$

几何上看，这是一个**双向锥约束**：以任一点 $(x_0, f(x_0))$ 为顶点，斜率为 $\pm L$ 的双向锥必须包住整张函数图。一旦在某一点出现"近乎竖直"的切线（如 $\sqrt{|x|}$ 在 $x=0$ 附近），就找不到任何有限的 $L$ 能容纳函数，函数就不再 Lipschitz。

![Lipschitz 锥包住整张函数图；含 vertical-tangent 的函数无法被任何有限 L 容纳](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84lipschitz%E8%BF%9E%E7%BB%AD%E6%80%A7-%E5%BC%BA%E5%87%B8%E6%80%A7%E4%B8%8E%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/fig1_lipschitz_geometry.png)

左图：$\sin x$ 是 1-Lipschitz 的，每个橙色锥体都不被函数图穿出。右图：$\mathrm{sign}(x)\sqrt{|x|}$ 在原点附近导数发散，红色区域显示候选斜率 $L=2$ 的锥体在 $x=0$ 附近被函数破坏。

**两条立刻能用的性质**：

1. **一致连续**：Lipschitz $\Rightarrow$ 一致连续。证明只需令 $\delta = \varepsilon/L$。
2. **闭合性**：Lipschitz 函数族对加法、数乘、复合（Lipschitz 常数相乘）封闭，方便我们把复杂模型拆成已知 Lipschitz 块。

## 1.2 梯度 Lipschitz = $L$-smoothness

工程中我们更关心**梯度的 Lipschitz 性**——它直接决定步长上限：

$$
\|\nabla f(y) - \nabla f(x)\| \le L\,\|y - x\|.
$$

满足这条的可微函数称为 **$L$-光滑（$L$-smooth）**。它有一个比定义本身更好用的等价表述，称为**下降引理（descent lemma）**：

$$
\boxed{\,f(y) \le f(x) + \langle \nabla f(x), y - x\rangle + \frac{L}{2}\,\|y - x\|^2.\,}
$$

也就是：**$f$ 永远被一族开口向上、曲率为 $L$ 的二次曲面从上方包住**——切线只是最差的下界，二次曲面才是真正的上界。

![L-smooth 函数被开口为 L 的二次抛物面从上方包住](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84lipschitz%E8%BF%9E%E7%BB%AD%E6%80%A7-%E5%BC%BA%E5%87%B8%E6%80%A7%E4%B8%8E%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/fig2_l_smooth_quadratic.png)

图中蓝色曲线是 $f(x) = \tfrac{1}{2}\sin(2x) + \tfrac{1}{2}x^2$，三个锚点处的虚线是各自的二次上界 $f(x_0) + f'(x_0)(x-x_0) + \tfrac{L}{2}(x-x_0)^2$，而点线则是普通切线。可以清楚看到，二次上界确实从上方包住了 $f$，而切线只在凸的部分才托住函数。

**为什么这条不等式对优化特别重要？** 在 $y = x - \eta\nabla f(x)$ 上代入得

$$
f(y) \le f(x) - \eta\Big(1 - \frac{L\eta}{2}\Big)\|\nabla f(x)\|^2.
$$

只要 $\eta \le 1/L$，括号项 $\ge 1/2$，**每一步都严格下降**，且下降量被梯度范数平方控制。这就是"步长不要超过 $1/L$"的硬来源——不是经验之谈，而是 descent lemma 的直接推论。

## 1.3 三个例子算一遍

| 函数 | 梯度 | Hessian 谱范数 | $L$ |
|---|---|---|---|
| $\tfrac{1}{2}\|x\|^2$ | $x$ | $1$ | $1$ |
| $\tfrac{1}{2}\|Ax-b\|^2$ | $A^\top(Ax-b)$ | $\lambda_{\max}(A^\top A)$ | $\lambda_{\max}(A^\top A)$ |
| Logistic $\log(1 + e^{-y\,\theta^\top x})$（单样本） | $-y\,\sigma(-y\theta^\top x)\,x$ | $\sigma(\cdot)\sigma(-\cdot)\,xx^\top$ | $\tfrac{1}{4}\|x\|^2$ |

第三行给出了 logistic 回归常用的 $L = \tfrac{1}{4}\sum_i\|x_i\|^2 / n$，其中 $\sigma(\cdot)\sigma(-\cdot)\le 1/4$ 是关键。

## 1.4 判别准则：通过 Hessian 给 L

**定理 1（Hessian 谱范数 $\Rightarrow$ 梯度 Lipschitz）**：若 $f$ 二阶可微且 $\sup_x \|\nabla^2 f(x)\|_2 \le L$，则 $\nabla f$ 是 $L$-Lipschitz 的。

**证明**：对任意 $x, y$，由 Newton-Leibniz 与链式法则，

$$
\nabla f(y) - \nabla f(x) = \int_0^1 \nabla^2 f(x + t(y-x))\,(y-x)\,\mathrm dt.
$$

两边取 2-范数：

$$
\|\nabla f(y) - \nabla f(x)\| \le \int_0^1 \|\nabla^2 f(\cdot)\|_2\,\|y-x\|\,\mathrm dt \le L\,\|y-x\|.\quad\blacksquare
$$

**实用推论**：对二次函数 $f(x) = \tfrac{1}{2}x^\top H x$，最优步长直接来自 $H$ 的最大特征值——这条结论对任何 $L$-光滑函数都给出"局部二次近似"的步长上限。

---

# 2. 强凸性：极小存在、唯一、且远离原点的代价二次增长

## 2.1 定义与三种等价刻画

**定义（$\mu$-强凸）**：可微函数 $f$ 是 $\mu$-强凸的（$\mu>0$），如果对任意 $x, y$，

$$
f(y) \ge f(x) + \langle \nabla f(x), y - x\rangle + \frac{\mu}{2}\,\|y - x\|^2.
$$

它有三个完全等价的描述，每一个都在不同的场合更顺手：

1. **二次下界**（上式本身）：函数被一族开口向上、曲率为 $\mu$ 的二次曲面**从下方托住**。
2. **$f - \tfrac{\mu}{2}\|x\|^2$ 是凸函数**：把"超出 $\mu$-曲率"的部分扣掉之后还是凸的。
3. **二阶条件**（若 $f$ 二次可微）：$\nabla^2 f(x) \succeq \mu I$。

![强凸函数被开口为 mu 的二次抛物面从下方托住；凸但非强凸的情形（mu=0）](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84lipschitz%E8%BF%9E%E7%BB%AD%E6%80%A7-%E5%BC%BA%E5%87%B8%E6%80%A7%E4%B8%8E%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/fig3_strong_convexity.png)

左图：强凸函数在每一点处都被一条曲率为 $\mu$ 的二次曲线托住。右图：$f(x) = 0.05\,x^4$ 是凸的但**不是强凸的**——在 $x=0$ 附近 $f$ 太平，没有任何 $\mu>0$ 的抛物线能贴在下面，这是为什么在这种函数上不可能得到线性收敛速率。

## 2.2 极小存在性与唯一性

**定理 2（极小存在性）**：若 $f$ 下半连续，且存在某个 $\alpha$ 使子水平集 $\{x : f(x)\le\alpha\}$ 非空且有界，则 $f$ 在该子水平集上达到极小。

证明就是 Weierstrass：有界 + 闭 = 紧，下半连续函数在紧集上必有极小。

**定理 3（强凸 $\Rightarrow$ 极小唯一）**：$\mu$-强凸函数（$\mu > 0$）在 $\mathbb{R}^n$ 上至多有一个全局极小点。

**证明**：设 $x^\star, y^\star$ 都是全局极小，则 $\nabla f(x^\star) = 0$。在二次下界中代入 $x = x^\star, y = y^\star$：

$$
f(y^\star) \ge f(x^\star) + 0 + \frac{\mu}{2}\|y^\star - x^\star\|^2.
$$

由于 $f(y^\star) = f(x^\star)$，必有 $\|y^\star - x^\star\| = 0$，即 $x^\star = y^\star$。$\blacksquare$

**推论（PL / 二次增长）**：把上式以 $x = x^\star$ 重写一遍，得

$$
f(y) - f^\star \ge \frac{\mu}{2}\|y - x^\star\|^2,
$$

也就是**离最优点越远，代价至少二次增长**。这是后面收敛性证明里"用次优值反过来界定迭代点距离"的关键。

## 2.3 同时 $L$-smooth 且 $\mu$-strongly convex：条件数登场

如果 $f$ 同时满足

$$
\mu I \preceq \nabla^2 f(x) \preceq L I,
$$

那么所有方向上的曲率都被夹在 $[\mu, L]$ 之间。比值

$$
\boxed{\,\kappa := \frac{L}{\mu} \ge 1\,}
$$

称为**条件数**，是后续所有收敛速率的核心量。$\kappa$ 越大，函数"长且窄"，最陡方向已经接近发散、最缓方向却几乎没有信号——这是优化困难的本质来源。

---

# 3. 加速梯度下降：从 $\kappa$ 到 $\sqrt{\kappa}$

## 3.1 普通梯度下降的两条上界

**定理 4（GD 在凸 + $L$-smooth 下的次线性率）**：取 $\eta = 1/L$，则

$$
f(x_t) - f^\star \le \frac{L\,\|x_0 - x^\star\|^2}{2t} = \mathcal O(1/t).
$$

**定理 5（GD 在 $\mu$-强凸 + $L$-smooth 下的线性率）**：取 $\eta = 1/L$，则

$$
\|x_t - x^\star\|^2 \le \Big(1 - \frac{1}{\kappa}\Big)^t \|x_0 - x^\star\|^2.
$$

要把误差缩到 $\varepsilon$，迭代次数 $t = \mathcal O(\kappa\log(1/\varepsilon))$。**条件数线性进入复杂度**——在 $\kappa = 10^4$ 的最小二乘上意味着每次精度提升十倍要再跑约 $2\times 10^4$ 步。

## 3.2 Nesterov 加速：用一个"前瞻点"换 $\sqrt{\kappa}$

经典动量（Polyak Heavy Ball）的更新是

$$
x_{t+1} = x_t - \alpha\nabla f(x_t) + \beta(x_t - x_{t-1}),
$$

它在严格凸二次型上确实达到 $\sqrt{\kappa}$ 的速率，但**对一般强凸函数无法保证全局加速**——几个反例就能让它发散。Nesterov 的关键改动是**把梯度评估搬到动量更新之后的"前瞻点"**：

$$
\begin{aligned}
y_t &= x_t + \beta_t (x_t - x_{t-1}), \\
x_{t+1} &= y_t - \eta\,\nabla f(y_t).
\end{aligned}
$$

直觉上：先按动量"瞄一眼"未来位置 $y_t$，再用那里的梯度修正方向。这一点点前瞻信息让它在所有 $L$-smooth 凸函数上都能保住加速率。

**定理 6（Nesterov 凸情形 $\mathcal O(1/t^2)$）**：对 $L$-smooth 凸函数，取 $\eta = 1/L$ 与经典权重序列 $\beta_t = (t-1)/(t+2)$，

$$
f(x_t) - f^\star \le \frac{2L\,\|x_0 - x^\star\|^2}{(t+1)^2}.
$$

**定理 7（Nesterov 强凸情形 $\mathcal O((1 - 1/\sqrt{\kappa})^t)$）**：对 $\mu$-强凸 + $L$-smooth 函数，取 $\eta = 1/L$ 与常数动量

$$
\beta = \frac{1 - \sqrt{1/\kappa}}{1 + \sqrt{1/\kappa}},
$$

则 $f(x_t) - f^\star \le \big(1 - \sqrt{1/\kappa}\big)^t (f(x_0) - f^\star)$，对应 $t = \mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$。

![GD vs Heavy Ball vs Nesterov 在强凸二次型上的收敛曲线](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84lipschitz%E8%BF%9E%E7%BB%AD%E6%80%A7-%E5%BC%BA%E5%87%B8%E6%80%A7%E4%B8%8E%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/fig4_convergence_rates.png)

左图（$\kappa=100$）：在严格 log 坐标下，三条算法都呈线性下降，但 Nesterov（紫色）和 Heavy Ball（橙色）的斜率明显比 GD（蓝色）陡——这正是 $\sqrt{\kappa}$ vs $\kappa$ 的差距。虚线是理论速率包络，与实测拟合得很好。右图：把前 80 代放大，可见 Nesterov **不严格单调**——它会沿"低谷"震荡，但震荡始终被理论下界包住，这是加速法的典型行为。

## 3.3 重启策略：解决"加速副作用"

加速法的代价是**非单调**——某些迭代步函数值会回升。这在两种情形下尤其讨厌：

- **强凸参数 $\mu$ 未知**：定理 7 的最优 $\beta$ 需要 $\sqrt{1/\kappa}$，估错就退化。
- **局部强凸**：函数全局只是凸，但在最优点附近强凸；用凸版 Nesterov 收敛会变慢。

**自适应重启（O'Donoghue & Candès, 2015）**：每当观察到**梯度方向反转**（$\langle\nabla f(y_t), x_{t+1}-x_t\rangle > 0$）或**函数值上升**，就把动量清零、$t$ 归零，重新开始。

**定理 8（重启 Nesterov 在强凸下达到最优率）**：对 $\mu$-强凸 + $L$-smooth 函数，重启 Nesterov 仍达到 $\mathcal O(\sqrt{\kappa}\log(1/\varepsilon))$ 复杂度，且**不需要预先知道 $\mu$**。

证明思路：在两次重启之间，跑的是凸版 Nesterov，$\mathcal O(1/k^2)$ 的误差结合二次增长 $f - f^\star \ge \tfrac{\mu}{2}\|x - x^\star\|^2$ 得到"每次重启误差至少减半"，几何减半 $\log(1/\varepsilon)$ 次后达成目标。每段长度 $\sim \sqrt{\kappa}$，总迭代数 $\sim \sqrt{\kappa}\log(1/\varepsilon)$。

## 3.4 条件数到底带来多少差距

![条件数 kappa 对迭代次数的影响：GD 线性增长，Nesterov 平方根增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/%E6%B7%B1%E5%85%A5%E8%A7%A3%E6%9E%90%E9%9D%9E%E7%BA%BF%E6%80%A7%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84lipschitz%E8%BF%9E%E7%BB%AD%E6%80%A7-%E5%BC%BA%E5%87%B8%E6%80%A7%E4%B8%8E%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95/fig5_condition_number.png)

左图（双对数）：在最小二乘合成数据上，GD 达到 $10^{-6}$ 相对精度所需迭代数严格沿 $\kappa$ 直线增长，Nesterov 沿 $\sqrt{\kappa}$ 增长。右图：加速比 = $T_{\text{GD}}/T_{\text{AGD}}$ 与 $\sqrt{\kappa}$ 几乎重合：$\kappa = 10^4$ 时加速 100 倍，$\kappa = 100$ 时加速 10 倍——条件数越差，加速法越值。

## 3.5 选择决策表

| 场景 | 推荐 | 理由 |
|---|---|---|
| $\kappa$ 小（$\le 50$），代码要简单 | GD | 加速带来的复杂度收益不显著 |
| $\kappa$ 大且 $\mu$ 已知 | 常数动量 Nesterov（定理 7） | 闭式最优 $\beta$ |
| $\kappa$ 大但 $\mu$ 未知/局部强凸 | 自适应重启 Nesterov | 自适应 $\mu$ 同时获得最优率 |
| 严格凸二次型（如最小二乘） | 共轭梯度 (CG) 优先 | 在 $\le n$ 步内有限终止，比 Nesterov 还快 |
| 非凸但局部接近凸（深度网络） | 带预热和余弦衰减的动量 / Adam | 理论速率不再可证，工程经验主导 |

---

# 4. 最小二乘实验

## 4.1 问题与数学性质

考虑最小二乘问题

$$
\min_{x\in\mathbb{R}^n} f(x) = \frac{1}{2}\|Ax - b\|^2, \qquad A\in\mathbb{R}^{m\times n},\; b\in\mathbb{R}^m.
$$

直接求导：

$$
\nabla f(x) = A^\top(Ax - b), \qquad \nabla^2 f(x) = A^\top A.
$$

于是

$$
L = \lambda_{\max}(A^\top A), \qquad \mu = \lambda_{\min}(A^\top A), \qquad \kappa = \kappa(A^\top A) = \kappa(A)^2.
$$

**注意**：条件数是 $A$ 的条件数的**平方**——这就是为什么"看起来还行"的 $A$（$\kappa(A) = 100$）在最小二乘里会变成 $\kappa = 10^4$ 的硬骨头。

## 4.2 算法实现要点

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

## 4.3 实测结论

在 $m = 200, n = 100$、$\kappa(A) \approx 100$ 即 $\kappa(A^\top A) \approx 10^4$ 的合成实例上：

- **GD**：$10^{-6}$ 相对梯度范数需 $\sim 4\times 10^4$ 步；曲线平滑但慢。
- **Nesterov（常数动量）**：$\sim 400$ 步；曲线沿"低谷"周期震荡，与图 4 右图一致。
- **重启 Nesterov**：$\sim 500$ 步，几乎完全单调，鲁棒性最好。

加速比与图 5 的 $\sqrt{\kappa}$ 预测吻合（$10^4 \to 100\times$）。

---

# 5. 总结与延伸

## 5.1 核心定理一图流

| 假设 | 算法 | 收敛速率 | 步长 |
|---|---|---|---|
| $L$-smooth, 凸 | GD | $\mathcal O(1/t)$ | $\eta = 1/L$ |
| $L$-smooth, $\mu$-strongly convex | GD | $\big(1 - 1/\kappa\big)^t$ | $\eta = 1/L$ |
| $L$-smooth, 凸 | Nesterov | $\mathcal O(1/t^2)$ | $\eta = 1/L,\ \beta_t = (t-1)/(t+2)$ |
| $L$-smooth, $\mu$-strongly convex | Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | $\eta = 1/L,\ \beta = (1-\sqrt{1/\kappa})/(1+\sqrt{1/\kappa})$ |
| 同上，$\mu$ 未知 | 重启 Nesterov | $\big(1 - 1/\sqrt{\kappa}\big)^t$ | 自适应 |

## 5.2 三个判断 reflex

新问题摆上桌时，按这个顺序回答：

1. **梯度有多陡？** 估 $L$（最大特征值或 backtracking line search），它直接给出步长上限。
2. **谷底有多硬？** 估 $\mu$（最小特征值或 PL 常数），它决定能否拿到线性收敛。
3. **加速值不值？** 看 $\kappa = L/\mu$：$\kappa < 50$ 用 GD，$\kappa \in [50, 10^4]$ 用 Nesterov，$\kappa > 10^4$ 考虑预条件 / 二阶法。

## 5.3 延伸阅读

- **非凸 + PL 条件**：放弃强凸，只要 $\tfrac{1}{2}\|\nabla f\|^2 \ge \mu(f - f^\star)$，GD 仍然线性收敛——这是深度学习中"过参数化为何能线性收敛"的理论起点（Karimi et al., 2016）。
- **加速 + 噪声**：Nesterov 对随机梯度并不天然鲁棒。SAG / SVRG / Katyusha 等方差缩减方法把强凸下的随机收敛重新拉回 $\sqrt{\kappa}$ 级别。
- **二阶加速**：Sophia、Shampoo 用对角或块对角 Hessian 预条件，等价于把有效条件数 $\kappa$ 直接改写——这是 2024 年大模型预训练的活跃方向。

**参考文献**：

1. Nesterov, Y. (1983). *A method of solving a convex programming problem with convergence rate $\mathcal O(1/k^2)$.* Soviet Mathematics Doklady, 27(2), 372–376.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
3. Bubeck, S. (2015). *Convex Optimization: Algorithms and Complexity.* Foundations and Trends in Machine Learning, 8(3-4), 231–357.
4. O'Donoghue, B., & Candès, E. (2015). *Adaptive restart for accelerated gradient schemes.* Foundations of Computational Mathematics, 15(3), 715–732.
5. Karimi, H., Nutini, J., & Schmidt, M. (2016). *Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition.* ECML-PKDD.
