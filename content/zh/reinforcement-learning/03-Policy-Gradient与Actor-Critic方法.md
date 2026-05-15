---
title: "强化学习（三）：Policy Gradient 与 Actor-Critic 方法"
date: 2025-08-11 09:00:00
tags:
  - Reinforcement Learning
  - Policy Gradient
  - REINFORCE
  - Actor-Critic
  - A3C
  - DDPG
  - TD3
  - SAC
  - PPO
categories: 强化学习
series: reinforcement-learning
lang: zh
mathjax: true
description: "从 REINFORCE 到 SAC——策略梯度方法如何直接优化策略，自然处理连续动作，驱动 PPO、TD3 和 SAC 等现代算法。"
disableNunjucks: true
series_order: 3
series_total: 12
translationKey: "reinforcement-learning-3"
---
DQN 证明了深度强化学习能够成功解决 Atari 游戏，但其能力存在明显局限：仅适用于**离散动作空间**。若用于控制具有七个连续关节角度的机械臂，则会完全失效——因为每一步动作选择都需要额外求解一个内部优化问题。

**策略梯度方法**换了一条路子。它不学习价值函数，也不依赖价值函数间接导出策略，而是**直接对策略参数进行优化**。仅此一项改变，便使强化学习得以处理连续动作、随机策略，并能自然应对最优策略本身即具随机性的问题（例如石头剪刀布）。
![强化学习（三）：Policy Gradient与Actor-Critic方法 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/illustration_1.png)


---

## 你将学到什么

- 策略梯度为什么重要，以及**策略梯度定理**到底讲了什么
- **REINFORCE**：最简单的策略梯度算法，以及它高方差的问题有多棘手
- **Actor-Critic** 架构，以及优势函数 $A = Q - V$ 是如何降低方差的
- **GAE($\lambda$)**：用一个参数搞定偏差-方差权衡
- 连续控制的经典算法：**DDPG / TD3 / SAC**
- 一份基于工业实践的**算法选型指南**

**前置知识**：[第 1 篇](/zh/reinforcement-learning/01-基础与核心概念/)（MDP、价值函数、TD 学习）和 [第 2 篇](/zh/reinforcement-learning/02-q-learning与深度q网络)（DQN、目标网络、经验回放）。

---
## 为什么选择策略梯度？

DQN 学习的是 $Q(s,a)$，然后通过贪心策略选择动作：$\pi(s) = \arg\max_a Q(s,a)$。这种间接方法带来了四个问题：

1. **只支持离散动作**。在连续空间中计算 $\arg\max$ 本身就是一个复杂的优化问题，而且每一步环境交互都要重复一次。
2. **无法表达随机策略**。贪心策略是确定性的，但在类似“匹配硬币”这样的博弈中，**最优策略本质上是随机的**。
3. **误差被放大**。$Q$ 值的逼近误差会被 $\max$ 操作符放大——这就是我们在第 2 篇文章中用 Double DQN 部分解决的高估偏差问题。
4. **探索方式生硬**。$\epsilon$-greedy 是一种临时补丁：它注入噪声的方式完全没有理论依据。

策略梯度方法绕过了这些问题，直接将策略参数化为 $\pi_\theta(a|s)$：

- 对于**离散动作**：网络最后一层加一个 softmax，输出一个分类分布。
- 对于**连续动作**：网络输出高斯分布的参数（均值 $\mu_\theta(s)$ 和对数标准差 $\log\sigma_\theta(s)$），通常再用 $\tanh$ 将动作限制在合法范围内。

![离散分类分布与连续 tanh-Gaussian 策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig6_action_policies.png)

无论是哪种情况，训练机制都是一样的：从 $\pi_\theta$ 中采样动作，然后调整参数，让好动作更有可能被选中。

### 策略梯度定理

设 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$ 是策略 $\pi_\theta$ 生成轨迹的期望回报。我们需要计算 $\nabla_\theta J(\theta)$ 来进行梯度上升。

**策略梯度定理**（Sutton 等，2000）给出了一个简洁且可采样的形式：
$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\Big[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\;\cdot\;Q^{\pi_\theta}(s_t, a_t)\Big]
$$
这里有三点需要注意：

- $\nabla_\theta \log \pi_\theta(a|s)$ 被称为**得分函数**。单独看它，就是让动作 $a$ 的概率稍微增加一点的方向。
- $Q^\pi(s,a)$ 是**标量权重**：好动作会放大得分函数，坏动作会反转它。
- **环境动态 $P(s'|s,a)$ 从公式中消失了**。我们不需要知道环境动态，采样到的轨迹就足够了。这正是该方法**无模型（model-free）**的原因。

直观来看，该定理的含义是：**将策略的概率质量向高回报动作方向移动**。

![策略梯度作为 score function 对动作分布的更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig1_policy_gradient_theorem.png)

左图展示了更新前后的 $\pi_\theta(a|s)$：概率质量向奖励峰值移动。右图展示了更新方向本身：score 函数 × 奖励 × 当前密度。乘积为正的地方，提升 $\pi(a)$；乘积为负的地方，降低它。

### 高方差问题与基线技巧

上面的估计量是**无偏的**，但方差非常大。$Q^\pi$ 的值可能达到几百甚至上千，一次幸运的回合就能让 $\theta$ 飞到天上去。

一个简单的恒等式帮了忙。对于**任意只依赖状态、不依赖动作**的函数 $b(s)$：
$$\mathbb{E}_{a \sim \pi_\theta}\!\big[\,\nabla_\theta \log \pi_\theta(a|s)\,\cdot\,b(s)\,\big] \;=\; 0$$
因此，从 $Q$ 中减去 $b(s)$，梯度的期望保持不变，但方差可以大幅降低。最优的选择是状态价值函数 $V^\pi(s)$，减去后得到的就是**优势函数**：
$$A^\pi(s,a) \;=\; Q^\pi(s,a) - V^\pi(s)$$
“优势”的含义很直观：动作 $a$ 比当前状态下“平均动作”好多少？将信号居中到零附近，梯度就不会剧烈波动了。

![Q(s,a)、价值基线 V(s) 与优势 A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig4_advantage_decomposition.png)

右图中绿色柱子表示值得加强的动作，红色柱子表示需要抑制的动作。DQN 的“挑最大 $Q$”离散视角，被替换成了一个连续、带符号的更新：“根据每个动作的相对优势，按比例抬升或压低策略。”

## REINFORCE：蒙特卡洛策略梯度

**REINFORCE**（Williams，1992）是教科书级别的起点。它直接用实际的折扣回报 $G_t$ 来估计 $Q^\pi(s_t, a_t)$。

### 算法

1. 在策略 $\pi_\theta$ 下跑出一条完整轨迹 $\tau = (s_0, a_0, r_0, \ldots, s_T)$。
2. 计算每一步的折扣回报：$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$。
3. 估计梯度：$\hat g = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,(G_t - b(s_t))$。
4. 更新参数：$\theta \leftarrow \theta + \alpha\,\hat g$。

这就是整个算法。简单到极致，但正是这种朴素让它有价值。

### 带 Baseline 的 REINFORCE 在 CartPole 上的应用

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def reinforce_with_baseline(env_name="CartPole-v1", episodes=1000,
                            gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    value = ValueNetwork(state_dim)
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = optim.Adam(value.parameters(), lr=1e-3)

    history = []

    for ep in range(episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []

        while True:
            st = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(st)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value(st))

            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            if done:
                break

        # 折扣回报 G_t（Q 的蒙特卡洛估计）
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        log_probs = torch.stack(log_probs)
        values = torch.cat(values)

        # 优势函数 = G_t - V_phi(s_t)（减去 baseline）
        advantages = returns - values.detach()

        # 策略优化：对 log pi * advantage 做梯度上升 -> 取负后最小化
        policy_loss = -(log_probs * advantages).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # 价值优化：让 V_phi 回归经验回报
        value_loss = F.mse_loss(values, returns)
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        history.append(sum(rewards))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1:4d}  avg100 = "
                  f"{np.mean(history[-100:]):6.1f}")

    return policy, history
```

CartPole 通常在 100 到 200 个回合内就能解决。但任务一旦变难，REINFORCE 的短板就暴露无遗。

**优点**：简单、无偏、不挑动作空间。  
**缺点**：梯度方差大，轨迹只能用一次（无法 off-policy 复用），更新只能等到回合结束。

## Actor-Critic：用 TD 估计替代回报

![强化学习（三）：Policy Gradient与Actor-Critic方法 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/reinforcement-learning/03-policy-gradient-and-actor-critic/illustration_2.png)

REINFORCE 必须等到回合结束才能计算 $G_t$，但这个回报包含了未来所有状态、动作和奖励的噪声。能不能做得更好？

**Actor-Critic** 的思路是：训练一个额外的网络——**critic** $V_\phi(s)$——用它来引导梯度信号。

- **Actor** $\pi_\theta(a|s)$：决定采取什么动作。
- **Critic** $V_\phi(s)$：评估当前状态的好坏。

核心改动是引入了**TD 误差优势**：
$$\hat A_t \;=\; r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t)$$
该估计仅依赖单步转移的随机性，而非整条轨迹后续所有随机因素。虽然不完美的 critic 会引入偏差，但方差显著降低，整体收益远大于代价。

两个网络通常共享主干结构：

![Actor-Critic 共享底座架构与更新流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig3_actor_critic_architecture.png)

TD 误差 $\delta_t$ 身兼两职：在 actor 中作为**优势权重**调整梯度，在 critic 中作为**回归目标**优化价值函数。

### 方差减少的效果有多大？

在同一组轨迹上对比：橙线使用原始蒙特卡洛回报，蓝线使用带学习基线的优势。

![REINFORCE 原始回报 vs A2C 优势的逐步梯度信号](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig2_variance_reduction.png)

在相同轨迹下，两种估计量具有相同的期望梯度；右图显示实际效果：方差降低约一个数量级。这正是 A2C/PPO 的训练曲线比 REINFORCE 平滑得多的原因。

### A2C 的代码实现

```python
class ActorCritic(nn.Module):
    """共享底座的 Actor-Critic。"""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)   # 策略头
        self.critic = nn.Linear(hidden, 1)            # 价值头

    def forward(self, state):
        h = self.trunk(state)
        return F.softmax(self.actor(h), dim=-1), self.critic(h).squeeze(-1)
```

A3C（Mnih 等，2016）将这套方法扩展到多个异步 worker 上运行。现代实践中更常用的是同步版本 **A2C**：从 $N$ 个并行环境中同步采集数据，然后合并梯度。思路相同，但对 GPU 更友好。

### GAE：在 TD 和蒙特卡洛之间找到平衡

单步 TD 偏差大但方差小，蒙特卡洛则相反。**广义优势估计（GAE）**（Schulman 等，2016）通过一个超参数 $\lambda \in [0, 1]$ 在两者之间插值：
$$
\hat A_t^{\text{GAE}(\lambda)} \;=\; \sum_{k=0}^{\infty} (\gamma\lambda)^k\,\delta_{t+k},
\quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$\lambda = 0$ 对应单步 TD 优势，$\lambda = 1$ 对应完整蒙特卡洛回报（减去基线）。实际应用中，$\lambda$ 通常取 $[0.9, 0.97]$ 之间的值。

![GAE 偏差-方差权衡曲线与多步回报权重](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig5_gae_lambda_sweep.png)

左图展示的是定性的权衡曲线，右图揭示了权重分配的实际效果：$\lambda$ 越大，信用分配会分散到更多未来的 TD 误差上；$\lambda$ 越小，则更倾向于信任最近一步的误差。PPO 默认的 $\lambda = 0.95$、$\gamma = 0.99$ 几乎正好落在最佳区域。

## 连续控制：DDPG 和 TD3

像机器人关节力矩这样的连续动作，策略通常用高斯分布建模：$a \sim \mathcal{N}(\mu_\theta(s),\,\sigma_\theta(s))$。但采样会引入噪声，影响精细控制。**确定性策略** $a = \mu_\theta(s)$ 直接输出动作，避免了噪声干扰，同时梯度形式非常简洁。

### DDPG：深度确定性策略梯度

DDPG（Lillicrap 等，2016）把 DQN 的稳定性技巧融入 Actor-Critic 框架：

- **经验回放 + 目标网络**（继承自 DQN）。
- **确定性策略梯度**（Silver 等，2014）：
$$
\nabla_\theta J \;=\; \mathbb{E}_{s \sim \rho^\beta}\!\Big[\,\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;\nabla_\theta \mu_\theta(s)\,\Big]
$$
从右往左看：调整 $\theta$ 让 $\mu_\theta(s)$ 移动，权重是 $Q$ 在该点对 $a$ 的变化率。这就是链式法则的直接应用。

探索通过在动作上加噪声实现：$a_t = \mu_\theta(s_t) + \mathcal{N}(0,\sigma)$。

### TD3：三个让 DDPG 更稳定的改进

DDPG 沿用了 DQN 的高估偏差问题，且对超参数和实现细节高度敏感。**TD3**（Fujimoto 等，2018）用三个独立且有效的改进解决了这个问题：

1. **截断双 Q 学习**。训练两个 critic，目标值取两者预测中的较小值：$y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde a')$。
2. **延迟策略更新**。critic 更新 $d$ 次（通常 $d=2$），actor 才更新一次。给 critic 时间先稳定下来。
3. **目标策略平滑**。为目标动作加一段截断噪声：$\tilde a' = \mu_{\theta'}(s') + \mathrm{clip}(\epsilon, -c, c)$，$\epsilon \sim \mathcal{N}(0, \sigma)$。这迫使 critic 对 $a$ 平滑，避免 actor 利用 Q 函数的尖峰。

```python
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.total_it = 0

    def train(self, replay_buffer, batch_size=256,
              gamma=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_delay=2):
        self.total_it += 1
        state, action, reward, next_state, done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # (3) 目标策略平滑
            noise = (torch.randn_like(action) * policy_noise) \
                .clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise) \
                .clamp(-self.max_action, self.max_action)

            # (1) 截断双 Q 目标
            target_q = torch.min(
                self.critic_1_target(next_state, next_action),
                self.critic_2_target(next_state, next_action),
            )
            target_q = reward + (1 - done) * gamma * target_q

        critic_loss = (
            F.mse_loss(self.critic_1(state, action), target_q)
            + F.mse_loss(self.critic_2(state, action), target_q)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # (2) 延迟更新 actor 和目标网络
        if self.total_it % policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            # ... 两个 critic 也做同样的 Polyak 平均
```

这三个改进让 DDPG 从“调参后偶尔能用”变成了一个稳定、可复现的强基线。

## SAC：最大熵强化学习

TD3 也有问题：策略可能会坍缩成一个狭窄的分布，导致探索停止。**Soft Actor-Critic**（Haarnoja 等，2018）直接从根源下手——**改了目标函数**：
$$J(\pi) \;=\; \mathbb{E}\!\Big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal H[\pi(\cdot|s_t)]\big)\Big]$$
熵项 $\mathcal H[\pi]$ 鼓励智能体保持不确定性，温度参数 $\alpha$ 调节这个权衡。Bellman 备份也做了调整：“软” Q 目标加了一个对数策略期望。

SAC 的实用性强，主要得益于三项关键工程设计：

- **自动调节温度**。$\alpha$ 通过梯度下降学习，目标是满足一个预设的熵约束，省得我们瞎猜。
- **squashed Gaussian 随机策略**。actor 输出 $(\mu, \log\sigma)$，采样后经过 $\tanh$，再用雅可比行列式修正 log-prob。
- **双 critic + 截断双 Q + 离策略回放 + 软目标更新**——熟悉的 TD3 套路。

实际用起来，SAC 在 MuJoCo 基准上能持平甚至超过 TD3，而且对超参数没那么敏感。在真实硬件上做连续控制时，很多实验室都把 SAC 当作默认起点。

## 为什么这东西能奏效？在 $\theta$ 空间里爬一座带噪声的山

退一步看，有必要把视角拉远。本文提到的所有算法，其实都是同一个核心思想的不同实现：**在策略参数空间中对 $J(\theta)$ 做随机梯度上升**。这里的“随机”二字分量很重——梯度估计本身充满噪声，有时甚至大得离谱，而且损失曲面还是非凸的。

![策略回报曲面上的随机梯度上升](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient与Actor-Critic方法/fig7_policy_optimization_landscape.png)

这张图清楚地说明了几件事：

- 路径是**锯齿状**的，一点都不平滑。方差缩减技术（baseline、advantage、GAE）的作用，就是让路径少抖一点，但最终目的地不变。
- **局部平台确实存在**。SAC 的熵正则化和随机策略设计，部分目的就是为了避免智能体卡在这些平台上。
- 曲面本身**会随着 $\theta$ 的变化而变化**——因为数据分布是 on-policy 的。这就是为什么 off-policy 方法（DDPG、TD3、SAC）需要小心修正，也是为什么 PPO 在下一节中要对更新步长做 clip 操作的根本原因。

---
## 算法选型指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 离散动作，快速验证 | **PPO** | 稳定、简单，所有主流库都支持得很好 |
| 连续动作，采样成本高 | **SAC** 或 **TD3** | 离策略算法，样本效率更高 |
| 连续动作，仿真资源充足 | **PPO** | 训练曲线平滑，容易扩展到多个环境 |
| 需要随机策略 | **SAC** | 最大熵框架天然支持随机策略 |
| 稀疏奖励 | **SAC** | 熵正则化让策略持续探索，直到找到奖励 |
| 学习入门，建立直觉 | **REINFORCE -> A2C -> PPO** | 每一步只引入一个新概念 |

**截至 2026 年的实际应用情况：**

- OpenAI（Dota 2、ChatGPT 的 RLHF）：**PPO**
- DeepMind（连续控制研究）：**SAC** 及其变种
- 伯克利机器人组（真实世界操作）：**SAC**
- TD3 仍是连续控制基准中最常用的参考基线

---
## 总结

策略梯度方法让强化学习走出了离散动作的局限：

- **REINFORCE** 展示了直接通过梯度上升优化策略的可能性——思路简洁，但实际用起来噪声很大。
- **Actor-Critic + 优势函数** 用少量偏差换来了方差的大幅降低，训练终于变得可行。
- **GAE($\lambda$)** 把偏差和方差的权衡简化成了一个可调参数。
- **DDPG / TD3** 把离策略效率引入连续控制，靠的是确定性策略和 DQN 风格的稳定性技巧。
- **SAC** 加入熵正则化后，迅速成为连续控制的首选方法。
- **PPO**（[第 6 篇](/zh/reinforcement-learning/06-ppo与trpo-信任域策略优化) 的主角）把信任域思想简化成 clipped surrogate，成为业界标配。

这些方法全是 **model-free** 的：它们从交互中学习，完全不依赖显式的环境建模。这种通用性是它们的优势，但代价是要消耗数百万样本。

**下一篇：** [第 4 篇](/zh/reinforcement-learning/04-探索策略与好奇心驱动学习/) 聚焦 **探索问题**——当环境几乎没有任何反馈时，智能体如何找到第一个奖励？

---
## 参考文献

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. *ICML*.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)*.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *ICML*.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: off-policy maximum entropy deep RL. *ICML*.
