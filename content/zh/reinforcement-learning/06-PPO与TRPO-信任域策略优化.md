---
title: "强化学习（六）：PPO与TRPO：信任域策略优化"
date: 2025-08-26 09:00:00
tags:
  - Reinforcement Learning
  - PPO
  - TRPO
  - Policy Optimization
  - Trust Region
  - RLHF
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "PPO和TRPO的完整推导：从策略优化的不稳定性到信任域约束，PPO的裁剪技巧，以及PPO在RLHF中的关键角色。"
disableNunjucks: true
series_order: 6
translationKey: "reinforcement-learning-6"
---
策略梯度（第三部分）直接优化策略，绕开了离散的 `argmax` 操作，还能自然处理随机策略。但它存在一个致命缺陷：一次错误的更新就可能导致策略性能急剧恶化。而且采样分布和策略是绑定的，想恢复过来几乎不可能。

**信任域方法**把这个问题说得很清楚：每次更新限制的是行为变化，而不是参数变化。 TRPO 通过硬 KL 约束和二阶优化器实现了这一点，而 PPO 则用一行带 clip 的简单算式达到了类似效果。简单有效的方法胜出了——PPO 不仅训练了 OpenAI Five，还支撑了 ChatGPT 的 RLHF 阶段，几乎所有现代机器人策略都依赖它，至今仍是深度强化学习领域的主力算法。
![强化学习（六）：PPO与TRPO：信任域策略优化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-ppo-and-trpo/illustration_1.png)

## 你将学到什么

- 为什么朴素策略梯度会彻底崩溃，我将用一个实际例子来说明
- **重要性采样**：复用离策略数据的关键方法
- **TRPO**：单调改进的理论下界、自然梯度、共轭梯度法和线搜索技术
- **PPO-Clip** 和 **PPO-Penalty**：用一阶优化逼近信任域的两种实现
- **PPO 在 RLHF 中的应用**：如何对齐 ChatGPT 级别的模型， DPO/IPO/KTO 又有哪些局限
- 一份实用的超参数调优与调试指南，风格接近 `cleanrl` 或 Stable Baselines 3

**前置知识**：[第三篇](/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法)（REINFORCE、 Actor-Critic、优势函数）。了解 KL 散度和 Fisher 信息会有帮助，但不是必须的，我会在文中适时讲解。

---
## 策略更新为什么会不稳定

### 一个极端的例子

假设当前策略在状态 $s$ 下，$\pi(a_1|s) = 0.9$，$\pi(a_2|s) = 0.1$。运气不好，我采样到了 $a_2$，而环境恰好返回了奖励 $+100$。 REINFORCE 估计器会把 $\nabla_\theta \log \pi(a_2|s)$ 乘上 $+100$，然后做一次梯度更新。结果，$a_2$ 的概率可能从 $0.1$ 直接跳到 $0.7$。但实际上，$a_2$ 的期望回报很可能远不如 $a_1$。

这一过程暴露了三个根本性问题：

1. **方差爆炸**： score 函数里有 $1/\pi$ 这一项，动作越罕见，梯度越大。这就是离策略 REINFORCE 不稳定的原因。
2. **分布漂移**：下一批轨迹是用更新后的*新策略*采样的。如果新策略变差了，采到的数据只会强化这个糟糕的信号，形成恶性循环。
3. **不可逆性**：在监督学习中，单次错误更新仅导致损失函数值暂时上升；而在强化学习中，一次不当更新会显著改变策略分布，导致后续采样数据失真——要修正这种偏差，往往需要数量级更高的额外样本。

### 参数空间会骗人，策略空间不会

考虑两个高斯策略 $\pi_1 = \mathcal{N}(0, 0.01)$ 和 $\pi_2 = \mathcal{N}(0, 10)$。它们的参数（均值与对数标准差）在欧氏距离上非常接近，但行为完全不同：$\pi_1$ 几乎总是输出 0，而 $\pi_2$ 把动作均匀地撒在整个值域上。

关键启示在于：在参数空间中采用固定步长的更新，可能在策略行为层面引发剧烈甚至失控的变化。任何“安全”保证都必须建立在*分布空间*中，而 KL 散度 $D_{KL}(\pi_{\text{old}} \| \pi_\theta)$ 是最自然的度量。

![信任域：许多个 KL 受限的小步保持安全；一次大的参数步会跌下悬崖](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig1_trust_region.png)

这张图把几何关系讲得很清楚：绿到红的曲面是一个假想的 $J(\theta)$ 地形，山脊狭窄、悬崖陡峭。朴素策略梯度（左图）沿梯度方向迈了一大步，直接踩到悬崖上； TRPO （右图）每一步都被约束在当前策略的 KL 球内，轨迹紧贴山脊，最终稳稳收敛到附近的最优解。
## 重要性采样：连接离策略数据的桥梁

在策略方法每次梯度更新后都会丢弃数据，因为数据分布发生了变化。**重要性采样**通过重新加权让我们能够复用这些数据：

$$\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}\!\left[\tfrac{q(x)}{p(x)}\, f(x)\right]$$
将旧策略和新策略代入策略梯度目标函数，就得到了**代理目标**：
$$L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right]$$
概率比 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$ 是本文所有算法的核心。

记住两点：

- 当 $\theta = \theta_{\text{old}}$ 时，$L^{\text{IS}}$ 的值和梯度与真实策略梯度目标 $J(\theta)$ 完全一致——因此它在局部是可靠的。
- 远离 $\theta_{\text{old}}$ 时， IS 估计器的方差大约按 $\exp(2\,D_{KL})$ 增长（见下图）。 KL 距离小，复用可靠； KL 距离大，估计结果就会充满噪声。

![重要性采样比的分布以及方差随 KL 的增长](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig5_importance_sampling.png)

左图展示了不同策略距离下的 $r_t$ 分布：当 $D_{KL}\!\le\!0.02$ 时，分布紧密集中在 PPO 的绿色裁剪区间 $[0.8, 1.2]$ 内；而当 $D_{KL}\!=\!0.05$ 时，右尾显著拉长。右图以对数刻度绘制了 IS 方差随 KL 的变化，橙色区域表示裁剪“节省”下来的方差。这就是为什么要把每次更新限制在小信任域内的定量依据。
## TRPO：信任域策略优化

### 单调改进的下界

Schulman 等人在 2015 年的研究中，基于 Kakade 和 Langford 在 2002 年的工作，证明了新策略的真实回报可以用代理目标加上一个 KL 惩罚项来下界：
$$J(\pi_{\text{new}}) \;\geq\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) \;-\; C \cdot D_{KL}^{\max}\!\left(\pi_{\text{old}} \,\|\, \pi_{\text{new}}\right)$$
其中 $C = 4\varepsilon\gamma/(1-\gamma)^2$，$\varepsilon$ 是优势函数的最大幅度，$\gamma$ 是折扣因子。结论很直观：**只要在提升代理目标的同时控制住 $D_{KL}^{\max}$ 的大小，就能保证策略单调改进**。

不过，实际应用中常数 $C$ 太保守，直接用并不现实。 TRPO 把惩罚项换成硬约束，并通过实验调整参数。

### 带约束的优化问题
$$\max_\theta \;\; \mathbb{E}\!\left[\tfrac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\,\hat{A}(s,a)\right] \quad\text{s.t.}\quad \bar{D}_{KL}\!\left(\pi_{\text{old}} \,\|\, \pi_\theta\right) \leq \delta$$
通常取 $\delta \approx 0.01$。这里用的是平均 KL 而不是最大 KL，因为从样本估计平均 KL 更高效。

### 自然梯度：如何定义“小”

普通 SGD 在欧氏球 $\|\Delta\theta\|^2 \le c$ 内寻找使线性化目标最大的方向；而**自然梯度**换了一种度量方式，把欧氏球换成“KL 球”。在参数空间局部， KL 散度可以近似为一个二次型，其 Hessian 就是**Fisher 信息矩阵**：
$$D_{KL}(\pi_\theta \,\|\, \pi_{\theta+\Delta\theta}) \;\approx\; \tfrac{1}{2}\,\Delta\theta^\top F\,\Delta\theta, \qquad F = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta\,\nabla_\theta \log \pi_\theta^\top\right]$$
求解这个带约束的优化问题后，得到自然梯度更新公式 $\Delta\theta \propto F^{-1}\nabla J$。它指向的是**策略空间**中的最陡上升方向，而不是参数空间中的方向——这正是我想要的。

### 实现：共轭梯度 + 线搜索

对于百万级参数的网络，$F$ 有 $\sim 10^{12}$ 个元素，根本存不下。 TRPO 用了两个技巧绕过这个问题：

1. **共轭梯度法**：求解 $Fx = g$ 时，全程只需要计算 Fisher-向量乘积 $Fv$（通过两次 `autograd` 反向传播即可完成），无需显式构造 $F$。
2. **回溯线搜索**：沿自然梯度方向逐步缩小步长，直到满足 KL 约束且代理目标确实改善。这一步确保了即使二阶近似不够准确，单调性依然能近似成立。

```python
def conjugate_gradient(fisher_vector_product, b, n_steps=10, tol=1e-10):
    """求解 F x = b，无需显式构造 F；代价为 n_steps 次 (Fv) 操作。"""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = r.dot(r)
    for _ in range(n_steps):
        Ap = fisher_vector_product(p)
        alpha = rdotr / (p.dot(Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

# 共轭梯度结束后，由 KL 预算 delta 唯一确定步长
step = torch.sqrt(2 * delta / (x.dot(fisher_vector_product(x)) + 1e-8)) * x
# 然后做回溯线搜索，确保 KL <= delta 且代理目标改善
```

**优点**：理论上有单调改进的保证；在 Humanoid、 Ant 这类复杂控制任务上非常稳定，而朴素的策略梯度方法容易发散。

**缺点**：核心代码约 300 行，每次更新需要计算 10–20 次 Hessian-向量乘积；多轮复用同一批数据收益有限（共轭梯度已经充分利用了曲率信息）；分布式训练不友好（共轭梯度每步都需要同步）。
## PPO：用两成复杂度，拿到九成收益

![强化学习（六）：PPO与TRPO：信任域策略优化 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-ppo-and-trpo/illustration_2.png)

2017 年， Schulman 和同事问了一个问题：*能不能只用一阶优化方法，达到类似 TRPO 的稳定性？* 答案就是 PPO。从那以后， PPO 成了强化学习领域的主流算法。

### PPO-Clip：核心技巧

定义裁剪后的代理目标：
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\Big(r_t(\theta)\,\hat{A}_t,\;\; \mathrm{clip}\!\left(r_t(\theta),\,1\!-\!\varepsilon,\,1\!+\!\varepsilon\right)\hat{A}_t\Big)\right]$$
通常 $\varepsilon \approx 0.2$。这个设计是故意不对称的：`min` 让目标变得**保守**，总是取裁剪和未裁剪值中较小的那个。

![按优势符号拆分的 PPO 裁剪代理目标](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig2_ppo_clipping.png)

两种情况，简单总结如下：

- **$\hat{A}>0$（好动作）**：未裁剪的目标会随着 $r_t \to \infty$ 持续增大；裁剪版在 $r_t > 1+\varepsilon$ 时给奖励封顶。`min` 会选择封顶值，因此超过封顶后梯度降为零。意思是：*别因为一个 minibatch 的好运，就把动作概率推到 $(1+\varepsilon)\pi_{\text{old}}$ 以上。*
- **$\hat{A}<0$（坏动作）**：对称地，裁剪版限制了 $r_t < 1-\varepsilon$ 时损失能负到多深。新策略一旦把概率压低到一定程度，梯度就停了。意思是：*别因为一次倒霉，就把某个动作彻底压死。*

更深层的问题是：*为什么用 `min` 而不是 `max`？* 如果总是取较大的那个，优化器会奖励大比率，直接踩破信任域。`min` 把信任域的思想写进了目标里，全程不需要算 KL。

### PPO-Penalty：自适应版本

第二种变体不常见，但在某些场景下很有用：直接加 KL 罚项，并自适应调整系数：
$$L^{\text{KL}}(\theta) = \mathbb{E}\!\left[r_t(\theta)\,\hat{A}_t\right] \;-\; \beta\,\mathbb{E}\!\left[D_{KL}(\pi_{\text{old}}\,\|\,\pi_\theta)\right]$$
每次迭代后调整 $\beta$：如果观测到的 KL 高于 $1.5\,\delta_{\text{target}}$，就翻倍；低于 $\delta_{\text{target}}/1.5$，就减半。

![自适应 KL 罚项：目标形状与 beta 的调度过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig3_kl_penalty.png)

左图展示了 $\beta$ 如何塑造目标：$\beta=0$ 退化为无约束代理目标（不稳定）；$\beta$ 很大时，最优解被拉回 $\theta_{\text{old}}$。右图是真实的自适应过程——为了把观测 KL 维持在目标值附近，$\beta$ 在对数刻度上跨数量级变动。早期的 InstructGPT 内部用的就是 KL 自适应版，后来被 clip 版取代。

### 代理目标地形：为什么裁剪更稳

![代理目标 vs 真实目标：未裁剪会误导优化器，裁剪能保持诚实](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig6_surrogate_landscape.png)

这张图是视觉上的核心结论。黑线是沿一维切片的*真实*回报 $J(\theta)$，先有一个峰，紧接着是一个掉进低谷的悬崖。橙色虚线是未裁剪的 IS 代理目标，它一路上升，会把 SGD 引到悬崖里。蓝线是 PPO 裁剪后的目标：信任域内（绿色带）紧贴真实曲线；信任域外它被“压平”了，擦掉了那段会把我们带下悬崖的梯度。

### 一个完整的 PPO 实现

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    """共享底座，独立的 actor / critic 头。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state):
        x = self.shared(state)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, eps_clip=0.2, k_epochs=10, c_vf=0.5, c_ent=0.01):
        self.gamma, self.lam = gamma, lam
        self.eps_clip, self.k_epochs = eps_clip, k_epochs
        self.c_vf, self.c_ent = c_vf, c_ent
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()

    def compute_gae(self, rewards, values, dones, next_value):
        """广义优势估计 GAE（Schulman 等，2016）。"""

        advantages = []
        gae = 0.0
        values = list(values) + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages)

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        with torch.no_grad():
            values = self.policy(states)[1].squeeze().numpy()
            next_val = self.policy(torch.FloatTensor(next_state).unsqueeze(0))[1].item()

        advantages = self.compute_gae(rewards, values, dones, next_val)
        returns = advantages + torch.FloatTensor(values)
        # 按批归一化优势：成本极低，对稳定性至关重要
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            probs, vals = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            vals = vals.squeeze()

            # 在对数空间算比率，数值更稳
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(vals, returns)
            loss = policy_loss + self.c_vf * value_loss - self.c_ent * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
```

普通 actor-critic 的基础上只多了几行关键代码，就能跑出一个百回合解决 CartPole、千万步学会 Humanoid 行走的算法。
## TRPO 与 PPO 正面对比

| 维度 | TRPO | PPO-Clip |
|------|------|----------|
| 优化器 | 共轭梯度 + 线搜索（二阶） | Adam （一阶） |
| KL 约束 | 硬约束，严格 $\le \delta$ | 软约束，逐样本 clip |
| 每批数据更新次数 | 1 | 3–10 epochs |
| Hessian-向量乘积 | $\sim$10–20 | 0 |
| 核心代码量 | $\sim$300 行 | $\sim$100 行 |
| 理论保证 | 单调改进（在下界范围内） | 形式上没有 |
| GPU / 分布式友好度 | 差（CG 每步要同步） | 优秀 |
| 超参敏感性 | $\delta$ 必须调 | $\varepsilon = 0.2$ 几乎万用 |

PPO 的实战优势不是靠 clip 单独撑起来的，而是多个因素叠加的结果：

- **每批数据跑多轮更新**——同一批轨迹能榨出更多学习信号。
- **熵奖励**让探索保持活跃，省去了设计噪声调度的麻烦。
- **每步没有反向曲率开销**——一个 epoch 只需一次 Adam 更新。
- **天生适合并行化**——可以同时跑 64 个 actor 收集 rollout。

![PPO 在远低于 TRPO 的成本下，效果持平甚至更好](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig4_benchmark.png)

左边是 MuJoCo 的训练曲线，右边是 Atari 上的人类归一化得分。这些图来自原始 PPO 论文，后来被 Engstrom 等人（2020）复现： PPO-Clip 全面碾压 A2C，在大多数任务上也略胜 TRPO，而且每秒挂钟时间能多跑约 10 倍的步数。

---
## PPO 在 RLHF 中的角色

PPO 最重要的应用是 **RLHF （基于人类反馈的强化学习）**——正是这个算法把 GPT-3 变成了 ChatGPT。

### 三阶段流程

1. **监督微调（SFT）**  
   用高质量的人类示范数据微调预训练的大语言模型，得到参考模型 $\pi_{\text{ref}}$。

2. **奖励模型训练**  
   收集人类偏好数据，比如“回答 A 比回答 B 更好”。用 Bradley-Terry 成对损失函数训练奖励模型 $R_\phi(x, y)$。

3. **PPO 微调**  
   把大语言模型当作策略，提示词 $x$ 当作状态，生成的回答 $y$ 当作动作，奖励模型 $R_\phi(x, y)$ 提供奖励信号。然后运行 PPO 算法。

### RLHF 的目标函数
$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\,y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x, y) - \beta\, D_{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)\right]$$

这里有两个信任域在起作用：

- **PPO 的裁剪机制**  
  防止每次更新时策略崩溃，和经典强化学习中的作用完全一致。

- **KL 散度惩罚项**  
  $\beta D_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ 防止策略长期偏离 SFT 模型。如果没有这一项， PPO 会找到奖励模型的漏洞——生成的回答虽然得分高，但可能是胡言乱语、攻击性内容或谄媚式回应。这项 KL 惩罚就是所谓的“对齐税”，确保生成的内容仍然像人类写的。

### 为什么 RLHF 比 CartPole 难

- 一个“回合”就是一次生成，单条样本的计算开销巨大（70B 参数模型的前向 + 反向传播）。
- 奖励模型本身也是学出来的，噪声较大。因此裁剪范围通常较小， KL 系数较大（$\beta \in [0.01, 0.2]$）。
- 动作空间是整个词汇表（约 5 万 token），序列长度可达数百步。优势估计必须方差极低——这就是为什么大家更喜欢用带 GAE 和大批量的 PPO，而不是朴素的策略梯度方法。

### DPO、 IPO、 KTO 的定位

在大规模场景下运行 PPO 太复杂了——需要维护四个模型副本（policy、 ref、 reward、 value），还要多卡同步。于是出现了一批**直接偏好学习**方法：

- **DPO**（Rafailov 等， 2023）  
  将最优 RLHF 策略重参数化为闭式解，用监督对比损失训练，不需要奖励模型，也不需要做 rollout。

- **IPO**  
  修复了 DPO 在高置信度偏好对上的过拟合问题。

- **KTO**  
  使用单回答的符号反馈（“好”/“坏”），不再需要成对比较。

直接方法的优势：基础设施简单，方差更低。 PPO 依然胜出的场景：当你需要真正的奖励信号时（在线学习、使用工具的智能体、代码/数学环境中有验证器提供奖励）， PPO 仍然是唯一的通用解决方案。

---
## 调参与排错指南

### 超参速查表

| 参数 | 常见范围 | 备注 |
|------|---------|------|
| 学习率 | $1\text{e-}4$ 到 $3\text{e-}4$ | 训练后期常线性退火到 0 |
| 裁剪 $\varepsilon$ | 0.1 – 0.3 | 0.2 在 90% 的场景中直接能用 |
| GAE $\lambda$ | 0.9 – 0.99 | 默认值为 0.95；价值函数不可靠时调高 |
| 折扣 $\gamma$ | 0.99 – 0.999 | 长时间任务需要更大的 $\gamma$ |
| Rollout 批量 | 2048 – 8192 （单环境时拉长） | 批量越大，优势估计方差越小 |
| PPO epochs | 3 – 10 | 超过 15 几乎必然过拟合 |
| Mini-batch 大小 | 64 – 256 | 与 rollout 批量无关 |
| 熵系数 | 0.0 – 0.01 | Atari 需要更大（约 0.01）； MuJoCo 常取 0 |
| 价值损失系数 | 0.5 – 1.0 | critic 难学时调高 |
| 梯度裁剪范数 | 0.5 – 1.0 | 防止异常 batch 的“安全带” |

![超参敏感度：裁剪范围、学习率、epoch×batch 网格](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/06-PPO与TRPO-信任域策略优化/fig7_hyperparameter.png)

这张图展示了三种典型模式。裁剪 $\varepsilon$ 的最优区间**又宽又平**——基本不用调；学习率的最优区间**窄得多**——偏离 3 倍就会损失 5%-10% 的回报； epoch × batch 的热力图揭示了一个交互效应：小批量上多 epoch 很容易过拟合当前数据，导致 KL 预算被浪费。

### 排错清单

- **新旧策略的近似 KL**：每次更新应在 0.01–0.02 之间。如果大于 0.05，说明 clip 没起作用，应该降低学习率或缩小 mini-batch。（注意常见的“单样本”估计 $\frac{1}{2}(\log r)^2$ 有偏；推荐使用 Schulman 的无偏估计 $r - 1 - \log r$。）
- **clip 触发率**：被裁剪到边界的样本比例。 10%-30% 是健康的； 0% 说明没用信任域；超过 50% 说明一直在边界上训练。
- **熵**：应平滑下降。如果几十步内骤降到 0，那是**过早收敛**，需要调高熵系数。
- **价值函数解释方差**：公式为 $1 - \mathrm{Var}(R - V)/\mathrm{Var}(R)$，几百次更新内应升到 0.5 以上；如果一直接近 0， critic 可能有问题。
- **优势归一化**：要在每个 minibatch 内做，而不是每个 epoch。漏掉这一步是 PPO 实现中最常见的 bug。

### 常见故障

| 症状 | 可能原因 | 修复 |
|------|---------|------|
| 奖励先涨后崩 | KL 步子太大， clip 没拉住 | 降低学习率；减少每批 epoch 数 |
| 奖励一直不动 | 优势在 batch 内归一化为 0 | 检查奖励是否为常数；扩大探索 |
| 50 步内熵塌成 0 | 没加熵奖励，从一开始就贪心 | 加 `c_ent` $\ge 0.005$，提高温度 |
| critic 预测爆炸 | 没归一化奖励；奖励无界 | 用滑动统计裁剪或归一化奖励 |
| CartPole 能跑，连续控制不行 | Gaussian 策略用了离散网络结构 | 使用 tanh 压缩，学习可调的 log-std |
## 小结

信任域方法彻底解决了策略梯度最尴尬的失败模式——一步走错，半小时白训。两条核心思想撑起了整个领域：

- **TRPO** 把 Kakade-Langford 的策略改进下界变成了算法：优化代理目标，硬约束 KL 散度，用自然梯度求解。理论很美，但工程实现复杂。
- **PPO** 放弃了硬约束，改用裁剪后的代理目标加一阶优化。虽然丢掉了单调改进的理论保证，但换来了实际中真正重要的东西：代码简单、支持多 epoch 更新、可以并行采样 rollout，而且默认参数非常稳健。

更深层的工程启示是普适的：**一个简单且局部诚实的近似，往往胜过复杂但全局正确的方案**——尤其是当“正确”的方法单步代价高出两个数量级时。 PPO 对 TRPO 的胜利，和 Adam 在深度学习中对二阶优化器的胜利如出一辙。

PPO 的影响力早已超出经典 RL 的范围。过去三年训练的所有主流对齐 LLM——ChatGPT、 Claude、 Gemini、 Llama-2 chat——都在后训练阶段跑了 PPO。某种意义上，**正是这个 clip，让现代 AI 助手不至于在奖励模型面前失控。**

---
## 参考文献

- Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). Trust Region Policy Optimization. *ICML*. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Kakade, S., & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. *ICML*.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. *ICLR*. [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Huang, S., Dossa, R., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. (2022). The 37 Implementation Details of PPO. *ICLR Blog Post*. [iclr-blog-track.github.io](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
