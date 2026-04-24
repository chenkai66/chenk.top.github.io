---
title: "强化学习（三）：Policy Gradient与Actor-Critic方法"
date: 2025-06-23 09:00:00
tags:
  - 强化学习
  - 策略梯度
  - REINFORCE
  - Actor-Critic
  - SAC
categories: 强化学习
series:
  name: "强化学习"
  part: 3
  total: 12
lang: zh-CN
mathjax: true
description: "从REINFORCE到SAC——策略梯度方法如何直接优化策略，自然处理连续动作，驱动PPO、TD3和SAC等现代算法。"
disableNunjucks: true
series_order: 3
---

DQN 证明了深度强化学习能玩转 Atari，但它有一个硬上限：**只能处理离散动作空间**。让它去控制一只七自由度机械臂的关节角度，立刻就垮了——你得在每一步动作之前先解一个内层优化问题。

**策略梯度方法**走的是完全不同的路线。它不再去学价值函数、再从中"推导"出策略，而是**直接对策略本身做优化**。仅这一处改动，就把强化学习的应用版图扩展到了连续动作、随机策略，乃至那些最优解本身就带有随机性的问题（想想石头剪刀布）。

## 你将学到什么

- 为什么需要策略梯度，**策略梯度定理**到底在说什么
- **REINFORCE**：最朴素的策略梯度算法，以及它高方差的代价
- **Actor-Critic** 架构，以及优势函数 $A = Q - V$ 是如何把方差按下去的
- **GAE($\lambda$)**：把偏差-方差权衡浓缩成一个旋钮
- 连续控制三件套：**DDPG / TD3 / SAC**
- 一份贴合工业现状的**算法选型指南**

**前置知识**：[第 1 篇](/zh/强化学习-一-基础与核心概念/)（MDP、价值函数、TD 学习）和 [第 2 篇](/zh/强化学习-二-q-learning与深度q网络/)（DQN、目标网络、经验回放）。

---

## 1. 为什么需要策略梯度？

DQN 学的是 $Q(s,a)$，再贪心地选动作 $\pi(s) = \arg\max_a Q(s,a)$。这种"绕一道弯"的做法带来四个问题：

1. **只能处理离散动作。** 在连续空间上做 $\arg\max$ 本身就是一个非平凡的优化问题，每一步都要重新解一遍。
2. **不能表达随机策略。** 贪心策略天生是确定性的，但在很多博弈场景里（比如石头剪刀布）**最优策略本身就是随机的**。
3. **误差被放大。** $Q$ 的逼近误差会被 $\max$ 算子放大——也就是第 2 篇里 Double DQN 部分修补过的高估偏差。
4. **探索是临时拼凑的。** $\epsilon$-greedy 是个补丁：注入噪声的方式毫无原则可言。

策略梯度方法换了个思路：**直接把策略参数化**为 $\pi_\theta(a|s)$。

- **离散动作**：网络末端接一层 softmax，输出一个分类分布。
- **连续动作**：网络输出高斯分布的参数（均值 $\mu_\theta(s)$ 与对数标准差 $\log\sigma_\theta(s)$），通常再用 $\tanh$ 把动作压缩到合法范围内。

![离散 softmax 策略与连续 tanh-Gaussian 策略](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig6_action_policies.png)

不管是哪一种，训练机器都是同一套：从 $\pi_\theta$ 中采样动作，然后把参数推向"让好动作更可能发生"的方向。

### 1.1 策略梯度定理

设 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$ 是策略 $\pi_\theta$ 产生的轨迹的期望回报，我们想要 $\nabla_\theta J(\theta)$ 来做梯度上升。

**策略梯度定理**（Sutton 等，2000）给出了一个干净、可采样的形式：

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\Big[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\;\cdot\;Q^{\pi_\theta}(s_t, a_t)\Big]
$$

有三件事值得停下来想清楚：

- $\nabla_\theta \log \pi_\theta(a|s)$ 叫**得分函数（score function）**，单看它，就是把动作 $a$ 的概率往上抬一点点的方向。
- $Q^\pi(s,a)$ 是**标量权重**——好动作让得分函数被放大，坏动作把它翻个号。
- **环境动态 $P(s'|s,a)$ 在公式里消失了**。我们根本不需要知道它，采样到的轨迹就够用——这正是该方法**model-free** 的根源。

直观上，定理说的就是"把概率质量挪向那些实际回报高的动作"：

![策略梯度作为 score function 对动作分布的更新](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig1_policy_gradient_theorem.png)

左图把更新前后的 $\pi_\theta(a|s)$ 叠在一起：质量在向 reward 峰移动。右图画的是更新方向本身——score 函数 × reward × 当前密度。乘积为正的地方，把 $\pi(a)$ 抬上去；为负的地方，把它压下去。

### 1.2 高方差问题与 baseline 技巧

上面那个估计量是**无偏**的，但方差大得离谱。$Q^\pi$ 的取值动辄上百上千，一次走运的回合就能把 $\theta$ 推到天上去。

一个简单的恒等式救了我们。对**任意只跟状态有关、不依赖动作**的函数 $b(s)$：

$$
\mathbb{E}_{a \sim \pi_\theta}\!\big[\,\nabla_\theta \log \pi_\theta(a|s)\,\cdot\,b(s)\,\big] \;=\; 0
$$

所以从 $Q$ 里扣掉 $b(s)$，期望意义下梯度纹丝不动，但方差可以被大幅压缩。最优的 $b$ 是状态价值函数 $V^\pi(s)$，扣完之后剩下的就是**优势函数**：

$$
A^\pi(s,a) \;=\; Q^\pi(s,a) - V^\pi(s)
$$

"优势"的字面意思就是它本身：动作 $a$ 比当前状态下的"平均动作"好多少？把信号居中到零附近，梯度才不会左右乱晃。

![Q(s,a)、价值基线 V(s) 与优势 A](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig4_advantage_decomposition.png)

右图里绿色柱子是值得加强的动作，红色柱子是该被压下去的。DQN 那种"挑最大的 Q"的离散视角，被替换成了一个连续、有正负号的更新："让策略按每个动作的相对优势成比例地抬升或压低。"

---

## 2. REINFORCE：蒙特卡洛策略梯度

**REINFORCE**（Williams，1992）是教科书的起点。它直接拿一整条轨迹的折扣回报 $G_t$ 来做 $Q^\pi(s_t, a_t)$ 的蒙特卡洛估计。

### 2.1 算法

1. 用 $\pi_\theta$ 跑出一条完整轨迹 $\tau = (s_0, a_0, r_0, \ldots, s_T)$。
2. 算每一步的折扣回报：$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$。
3. 估计梯度：$\hat g = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,(G_t - b(s_t))$。
4. 更新：$\theta \leftarrow \theta + \alpha\,\hat g$。

整个算法就这四步——朴素得近乎天真，但这恰恰是它的价值。

### 2.2 在 CartPole 上跑带 baseline 的 REINFORCE

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

        # 优势 = G_t - V_phi(s_t)（baseline 扣减）
        advantages = returns - values.detach()

        # 策略：对 log pi * advantage 做梯度上升 -> 取负后做最小化
        policy_loss = -(log_probs * advantages).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # 价值：让 V_phi 回归到经验回报
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

CartPole 通常 100--200 个回合就解了。但任务一难起来，REINFORCE 的短板就立刻显形。

**优点**：简单、无偏、对动作空间无要求。
**缺点**：梯度方差极大，每条轨迹只用一次（无法 off-policy 复用），而且只能在回合结束时才更新。

---

## 3. Actor-Critic：用 TD 估计替代回报

REINFORCE 要等到回合结束才能算 $G_t$，而这个回报里夹带着未来**所有**状态、动作、奖励的随机性。能不能再快一点？

**Actor-Critic** 的回答是：再训练一个网络——**critic** $V_\phi(s)$——拿它来 bootstrap 梯度信号。

- **Actor** $\pi_\theta(a|s)$：决定怎么动。
- **Critic** $V_\phi(s)$：评估当前状态有多好。

关键替换是**TD 误差形式的优势**：

$$
\hat A_t \;=\; r_t + \gamma\,V_\phi(s_{t+1}) - V_\phi(s_t)
$$

它只依赖**一步**的随机性，而不是整条尾巴。方差骤降，代价是不完美的 critic 会引入一点偏差——这是个非常划算的交易。

两个网络通常共享 backbone：

![Actor-Critic 共享底座架构与更新流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig3_actor_critic_architecture.png)

TD 误差 $\delta_t$ 一身二职：在 actor 这边充当**优势权重**，在 critic 那边充当**回归目标**。

### 3.1 这降的方差有多夸张？

把 REINFORCE 的原始回报和 A2C 的优势放在同一批轨迹上对比：

![REINFORCE 原始回报 vs A2C 优势的逐步梯度信号](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig2_variance_reduction.png)

同一批轨迹、同一个期望梯度。右图给出实打实的收益：方差缩小了一个数量级。这正是为什么 A2C/PPO 的训练曲线远比 REINFORCE 平滑。

### 3.2 A2C 的实现

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

A3C（Mnih 等，2016）把这套结构铺到多个异步 worker 上跑。如今实践中更常见的是它的同步版本 **A2C**：用 $N$ 个并行环境齐步采集数据，再做一次合并梯度。思路相同，但对 GPU 更友好。

### 3.3 GAE：在 TD 与蒙特卡洛之间架一个旋钮

一步 TD 偏差大方差小，蒙特卡洛刚好相反。**广义优势估计（GAE）**（Schulman 等，2016）用一个超参数 $\lambda \in [0, 1]$ 在两端之间插值：

$$
\hat A_t^{\text{GAE}(\lambda)} \;=\; \sum_{k=0}^{\infty} (\gamma\lambda)^k\,\delta_{t+k},
\quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$\lambda = 0$ 退化为单步 TD 优势，$\lambda = 1$ 退化为完整蒙特卡洛回报（减掉 baseline）。实战中常取 $[0.9, 0.97]$ 之间。

![GAE 偏差-方差权衡曲线与多步回报权重](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig5_gae_lambda_sweep.png)

左图给出的是定性的权衡曲线，右图揭示了权重到底是怎么分配的：$\lambda$ 越大，credit 被摊到越多的未来 TD 误差上；$\lambda$ 越小，越只信一步。PPO 默认的 $\lambda = 0.95$、$\gamma = 0.99$ 几乎正落在甜蜜区里。

---

## 4. 连续控制：DDPG 与 TD3

机器人关节力矩这类连续动作，最自然的策略形式就是高斯：$a \sim \mathcal{N}(\mu_\theta(s),\,\sigma_\theta(s))$。但采样会引入噪声，对精细控制是负担。**确定性策略** $a = \mu_\theta(s)$ 直接输出动作，没有这层噪声——并且天然带有一种特别干净的梯度形式。

### 4.1 DDPG：深度确定性策略梯度

DDPG（Lillicrap 等，2016）把 DQN 的稳定性手段嫁接到 Actor-Critic 框架上：

- **经验回放 + 目标网络**（继承自 DQN）。
- **确定性策略梯度**（Silver 等，2014）：

$$
\nabla_\theta J \;=\; \mathbb{E}_{s \sim \rho^\beta}\!\Big[\,\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;\nabla_\theta \mu_\theta(s)\,\Big]
$$

从右往左读：把 $\theta$ 沿着 $\mu_\theta(s)$ 移动的方向推，权重是 $Q$ 在这一点上对 $a$ 的上升斜率。本质上就是一条链式法则。

探索靠在动作上加噪声实现：$a_t = \mu_\theta(s_t) + \mathcal{N}(0,\sigma)$。

### 4.2 TD3：三个把 DDPG 救回来的 trick

DDPG 继承了 DQN 的高估偏差，并以脆弱出名。**TD3**（Fujimoto 等，2018）用三个**互相独立、各自有效**的 trick 一举把它救回来：

1. **截断双 Q 学习。** 训两个 critic，在算目标时取它们俩中较小的那个：$y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde a')$。
2. **延迟策略更新。** critic 每更新 $d$ 次（通常 $d=2$），actor 才更新一次——给 critic 留出"先稳下来"的时间。
3. **目标策略平滑。** 给目标动作加一段被截断的噪声：$\tilde a' = \mu_{\theta'}(s') + \mathrm{clip}(\epsilon, -c, c)$。这样 critic 必须对 $a$ 平滑，actor 也就没法去钻 Q 函数里那些尖刺。

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

        # (2) 延迟更新 actor 与目标网络
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

这三招让 DDPG 从"调好参数偶尔能跑"变成了一个能稳定复现的强基线。

---

## 5. SAC：最大熵强化学习

即便是 TD3，也仍可能遇到一种失败模式：策略坍缩成一个非常窄的分布，从此放弃探索。**Soft Actor-Critic**（Haarnoja 等，2018）从根上动手——**改了优化目标**：

$$
J(\pi) \;=\; \mathbb{E}\!\Big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal H[\pi(\cdot|s_t)]\big)\Big]
$$

熵项 $\mathcal H[\pi]$ 给"保持不确定"这件事开工资，温度 $\alpha$ 控制权衡。Bellman 备份相应被改写成"软"形式，目标里多了一个对数策略的期望。

让 SAC 成为业界主力的，是三个工程上的细节：

- **温度自动调节。** $\alpha$ 自己也是被梯度下降学出来的，对应一个目标熵约束——你不用瞎猜它该取多少。
- **squashed Gaussian 随机策略。** actor 输出 $(\mu, \log\sigma)$，采样后再过 $\tanh$，并用换元的雅可比修正 log-prob。
- **双 critic + 截断双 Q + 离策略回放 + 软目标更新**——典型的 TD3 风格栈。

实战中 SAC 在 MuJoCo 这类基准上对标甚至超越 TD3，并且对超参数明显更不挑。在真实硬件上做连续控制，许多实验室都是从 SAC 起步。

---

## 6. 退一步看：在 $\theta$ 空间里爬一座有噪声的山

有必要把镜头拉远。本文里所有算法，都是同一件事的不同变种：**对 $J(\theta)$ 在策略参数空间里做随机梯度上升**。"随机"两个字承担了很多重量——梯度估计本身有噪声（有时大得离谱），损失曲面也并非凸的。

![策略回报曲面上的随机梯度上升](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/reinforcement-learning/03-Policy-Gradient%E4%B8%8EActor-Critic%E6%96%B9%E6%B3%95/fig7_policy_optimization_landscape.png)

这张图把几件事说得很直白：

- 路径是**锯齿状**而不是光滑的。所谓方差缩减（baseline、advantage、GAE），买到的是"路径少抖一点"，目的地不变。
- **局部平台真实存在。** SAC 的熵正则与随机策略，部分作用就是不让你卡在这里。
- 曲面本身**会随着 $\theta$ 改变**——数据分布是 on-policy 的。这正是 off-policy 算法（DDPG、TD3、SAC）需要小心校正、PPO 要在下一章里 clip 掉过大更新步的根本原因。

---

## 7. 算法选型指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 离散动作，快速原型 | **PPO** | 稳定、简单、所有库都支持得很好 |
| 连续动作，采样昂贵 | **SAC** 或 **TD3** | 离策略，样本效率高 |
| 连续动作，仿真便宜 | **PPO** | 训练曲线更平滑，方便横向并行 |
| 必须用随机策略 | **SAC** | 最大熵框架自带这件事 |
| 稀疏奖励 | **SAC** | 熵正则让策略保持探索，等到第一个奖励出现 |
| 入门、构建直觉 | **REINFORCE -> A2C -> PPO** | 每一步只增加一个新概念 |

**截至 2026 年的工业应用现状：**

- OpenAI（Dota 2、ChatGPT 的 RLHF）：**PPO**
- DeepMind（连续控制研究）：**SAC** 及其变种
- 伯克利机器人组（真实世界操控）：**SAC**
- TD3 仍然是连续控制基准里最常被引用的参考基线

---

## 8. 小结

策略梯度方法把强化学习推到了离散动作之外的世界：

- **REINFORCE** 证明了"直接对策略做梯度上升"是可行的——概念上极干净，实践中极嘈杂。
- **Actor-Critic + 优势** 用一点偏差换来一个数量级的方差降幅，让训练真正变得可行。
- **GAE($\lambda$)** 把偏差-方差的取舍浓缩成一个可调旋钮。
- **DDPG / TD3** 把离策略效率带进了连续控制，靠确定性策略 + DQN 风格的稳定性手段。
- **SAC** 加了熵正则化，成了连续控制的默认选项。
- **PPO**（[第 6 篇](/zh/强化学习-六-ppo与trpo-信任域策略优化/)主角）把信任域思想简化成一个 clipped surrogate，成为业界主力。

这些方法全部都是 **model-free** 的：从交互中学，从不显式对环境建模。这种通用性是它们的优势，也意味着它们要消耗成百上千万的样本——这是它们的代价。

**下一篇：**[第 4 篇](/zh/强化学习-四-探索策略与好奇心驱动学习/) 直击**探索问题**——当环境几乎给不出任何反馈时，智能体到底是怎么找到第一个奖励的？

---

## 参考文献

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. *ICML*.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. *ICML*.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: off-policy maximum entropy deep RL. *ICML*.

---

## 系列导航

| 部分 | 主题 |
|------|------|
| 1 | [基础与核心概念](/zh/强化学习-一-基础与核心概念/) |
| 2 | [Q-Learning 与深度 Q 网络](/zh/强化学习-二-q-learning与深度q网络/) |
| **3** | **Policy Gradient 与 Actor-Critic 方法（本文）** |
| 4 | [探索策略与好奇心驱动学习](/zh/强化学习-四-探索策略与好奇心驱动学习/) |
| 5 | [Model-Based 强化学习与世界模型](/zh/强化学习-五-model-based强化学习与世界模型/) |
| 6 | [PPO 与 TRPO：信任域策略优化](/zh/强化学习-六-ppo与trpo-信任域策略优化/) |
